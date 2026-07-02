"""Online mount calibration — direct port of openpilot's
`selfdrive/locationd/calibrationd.py` algorithm, adapted for our
game-screen-capture pipeline.

We track three quantities:
  - device-frame pitch (mount tilt, +down)
  - device-frame yaw (mount yaw, +left)
  - camera height above the road (m)

Algorithm matches openpilot:

  1. Per-sample gates (must all pass):
       v_ego        > MIN_SPEED_FILTER (15 mph = 6.7 m/s)
       |yaw_rate|   < MAX_YAW_RATE_FILTER (2 deg/s)
       model vx     > MIN_SPEED_FILTER (catches stationary model output
                                        that disagrees with telemetry)
       lat-vel-std  / vx < MAX_VEL_ANGLE_STD (0.25 deg) — model is
                                              uncertain about lateral
                                              motion → reject
       height-std   < MAX_HEIGHT_STD (30 mm)

  2. Compute observed_rpy from raw pose via openpilot's formula
     (`calibrationd.py:172-175`):
         observed_pitch = -atan2(pose.vz, pose.vx)
         observed_yaw   =  atan2(pose.vy, pose.vx)
     This is the RESIDUAL after the warp — if `calib.pitch_deg` /
     `calib.yaw_deg` are already correct, observation is ~0.

  3. Compose onto the current smoothed rpy via rotation composition
     (not Euler-EMA):
         R_new = R(current_smooth) @ R(observed)
         new_rpy = euler_from_rot(R_new)
     Matches `calibrationd.py:176` exactly.

  4. Block-aggregate: each accepted sample feeds an in-flight block.
     After BLOCK_SIZE = 100 samples the block average is committed to
     the rpy/height history (ringbuffer of INPUTS_WANTED = 50).

  5. The smoothed estimate exposed to callers is the mean over the
     committed history. With INPUTS_NEEDED (5) blocks landed we report
     calStatus = CALIBRATED; below that, UNCALIBRATED with progress.
     If the smoothed rpy or height drifts outside the validation
     bounds, calStatus = INVALID.

  6. Only commit to `calib.pitch_deg`, `calib.yaw_deg`, and
     `calib.height_m` once calStatus == CALIBRATED. Below that the
     warp + overlay continue with the manually-tuned (or
     previously-persisted) values. Height matters for the overlay's
     vertical placement (it sets the road-plane z_FRD inside
     `Calibration.road_to_image`), so writing it back closes the loop
     on full hands-off mount calibration.

`pose_std` and `road_transform_std` come from `postprocess.decode`
(linear stddevs after exponentiating the log-space model output).
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from simsteer.paths import state_path
from simsteer.core.warp import euler_from_rot, rot_from_euler

if TYPE_CHECKING:
    from simsteer.core.calibration import Calibration


WRITE_CALIBRATION = True

# ----- openpilot's calibrationd constants (loosened for game use) -----
# Openpilot uses 5 blocks × 100 samples = 500 ACCEPTED samples for
# CALIBRATED status. That's calibrated for slow real-car driving where
# samples are easy to come by. For game capture with the user actively
# steering through normal corrections, every gate fires often and the
# 500-sample bar takes 15+ minutes. Cut INPUTS_NEEDED to 3 blocks =
# 300 accepted samples while keeping BLOCK_SIZE=100 so each committed
# block still has comma's smoothing quality.
INPUTS_NEEDED = 3     # min blocks for CALIBRATED status (was 5)
INPUTS_WANTED = 50    # ringbuffer size — extras improve stability
BLOCK_SIZE = 100      # samples per block average

MIN_SPEED_FILTER_MPS = 15 * 0.44704              # 15 mph → 6.7 m/s
MAX_YAW_RATE_FILTER = math.radians(2.0)          # 2 deg/s
MAX_VEL_ANGLE_STD = math.radians(0.5)            # 0.5 deg (was 0.25)
MAX_HEIGHT_STD = math.exp(-3.5)                  # ≈ 30 mm

# Validation window — wider than openpilot's comma-3 (-5.2°…+9.7°)
# because game mounts (truck cab, AC dash) can sit further off the
# road plane than a windshield-mounted phone.
PITCH_MIN_DEG, PITCH_MAX_DEG = -10.0, 20.0
YAW_MIN_DEG, YAW_MAX_DEG = -8.0, 8.0
HEIGHT_MIN_M, HEIGHT_MAX_M = 0.5, 4.0

# Lateral-acceleration gate — yaw_rate × v_ego, an upper bound on how
# hard the vehicle is turning. Loosened from 1.0 to 1.5 m/s² so
# normal highway lane-keeping corrections pass; the `|yaw_rate| < 2°/s`
# sample-local gate still catches harder turns.
MAX_LAT_ACC_MPS2 = 1.5

# Human-input gate — detect that the user is actively manually
# steering (telemetry `game_steer` is changing fast) and suppress
# samples for a cooldown window. Without this, a single hard wheel
# yank corrupts the in-flight block instantly — vy spikes, observed
# yaw spikes with it, and the whole batch drags off-axis.
#
# Original values (0.05 / 30 frames) were way too aggressive: any 5%
# steering wobble was a yank, and the 1.5 s cooldown blocked >75%
# of normal driving frames. The fix targets ACTUAL yanks: ≥0.15 axis
# delta in one frame, ~0.5 s cooldown so a recovery wobble doesn't
# extend the window indefinitely.
HUMAN_STEER_DELTA = 0.15            # axis units per frame (was 0.05)
HUMAN_SUPPRESS_FRAMES = 10          # ~0.5 s at 20 Hz (was 30)

# Block-mean watchdog — compare each newly-committed block's mean to
# the running mean of prior blocks. A block whose pitch or yaw is
# more than this many degrees off the trend is presumed to be poisoned
# (transient maneuver that slipped the per-sample gates, model glitch)
# and discarded. Without the watchdog, a single bad block can drag the
# smoothed estimate by ~1° and take many more good blocks to recover.
BLOCK_JUMP_THRESHOLD_DEG = 3.0

# Auto-recovery: how many consecutive committed blocks must land
# INVALID before we abandon the run and reset. Without this, a
# calibration that diverges (model bias, post-warp residual that
# doesn't average to zero, etc.) stays stuck in INVALID forever and
# the user has to delete the state file manually. 3 blocks at ~12 s
# each is ~36 s of bad samples before we give up.
AUTO_RESET_INVALID_BLOCKS = 3


class CalStatus(IntEnum):
    UNCALIBRATED = 0
    CALIBRATED = 1
    INVALID = 2


def _wrap_rotation_composition(current_rpy_rad: tuple[float, float, float],
                               observed_rpy_rad: tuple[float, float, float]
                               ) -> tuple[float, float, float]:
    """openpilot's `calibrationd.py:176` operation."""
    R = rot_from_euler(current_rpy_rad) @ rot_from_euler(observed_rpy_rad)
    return euler_from_rot(R)


class LiveCalib:
    def __init__(self, game: str | None = None) -> None:
        # In-flight block accumulator. Sum + count, committed as a mean
        # every BLOCK_SIZE samples. Pre-allocated as ndarrays so the
        # composition math stays in radians.
        self._block_rpy_sum = np.zeros(3, dtype=np.float64)
        self._block_height_sum = 0.0
        self._block_n = 0

        # Committed history (one entry per completed block). Capped at
        # INPUTS_WANTED so we drop the oldest as new blocks land.
        self._rpys: deque[np.ndarray] = deque(maxlen=INPUTS_WANTED)
        self._heights: deque[float] = deque(maxlen=INPUTS_WANTED)

        # Wide-camera mount — informational, tracked separately so the
        # HUD can sanity-check the model's two camera estimates.
        self._wide_sum = np.zeros(3, dtype=np.float64)
        self._wide_n = 0
        self.wide_from_device_estimate: np.ndarray | None = None

        # Per-frame counters for the HUD.
        self.samples = 0       # accepted samples (across all blocks)
        self.rejected = 0
        self.rej_speed = 0
        self.rej_turning = 0
        self.rej_vel_std = 0
        self.rej_height_std = 0
        self.rej_pitch = 0
        self.rej_yaw = 0
        self.rej_lat_acc = 0
        self.rej_human = 0
        self.rej_block_jump = 0       # block-level (not sample-level)

        # Human-input gate state. Frame countdown so the suppress
        # actually elapses (a sample-count target wouldn't, since
        # rejected samples don't increment `self.samples`).
        self._last_game_steer: float | None = None
        self._suppress_frames_remaining = 0

        # Auto-recovery state.
        self._consecutive_invalid_blocks = 0
        # Sticky message exposed for the HUD when we auto-reset, so the
        # user knows why the bar suddenly jumped back to 0%.
        self.last_reset_reason: str | None = None

        # Acceptance-rate tracking: timestamps of recently-accepted
        # samples, used to compute samples/sec for HUD ETA. Bounded
        # to ~last 10s of samples (=~200 at 20 Hz).
        self._accept_timestamps: deque[float] = deque(maxlen=200)

        # Last-sample telemetry for HUD display.
        self.last_raw_pitch = 0.0
        self.last_raw_yaw = 0.0
        self.last_observed_height = 0.0

        # State.
        self.cal_status = CalStatus.UNCALIBRATED
        self._wrote_to_calib = False
        # Shadow mode switch: when False the learner keeps estimating
        # (blocks land, HUD shows what it WOULD apply) but never
        # touches `calib` — the A/B toggle for "is livecalib helping
        # or hurting". Not persisted; the app owns the choice.
        self.apply = True
        # Persist-to-disk counter (every N committed blocks).
        self._blocks_since_save = 0

        # Per-game persistence. None = no save/load. Restoring block
        # history on launch is what lets the first-drive wizard survive
        # a quit/relaunch mid-calibration.
        self._game = game
        if game is not None:
            self.load_state(game)

    # ----- public read-only accessors (HUD / consumers) -----

    @property
    def pitch_estimate(self) -> float | None:
        """Smoothed pitch in degrees, or None if no block has landed."""
        if not self._rpys:
            return None
        return math.degrees(float(np.mean(np.stack(self._rpys, axis=0)[:, 1])))

    @property
    def yaw_estimate(self) -> float | None:
        if not self._rpys:
            return None
        return math.degrees(float(np.mean(np.stack(self._rpys, axis=0)[:, 2])))

    @property
    def height_estimate(self) -> float | None:
        if not self._heights:
            return None
        return float(np.mean(self._heights))

    @property
    def blocks(self) -> int:
        """Number of committed blocks (the openpilot `valid_blocks` count)."""
        return len(self._rpys)

    @property
    def block_progress(self) -> float:
        """Fraction of the current in-flight block that's filled.
        Lets the HUD show a 'next block lands in N samples' progress."""
        return self._block_n / BLOCK_SIZE

    @property
    def writes_enabled(self) -> bool:
        """True once enough blocks have landed AND the smoothed rpy is
        within validation bounds. Below that we don't touch `calib`.
        `self.apply = False` (shadow mode) blocks writes regardless."""
        return (WRITE_CALIBRATION
                and self.apply
                and self.cal_status == CalStatus.CALIBRATED)

    # ----- ingest -----

    def update(self, calib: "Calibration",
               pose: np.ndarray | None,
               road_transform: np.ndarray | None,
               actual_yaw_rate: float | None,
               v_ego: float | None,
               pose_std: np.ndarray | None = None,
               road_transform_std: np.ndarray | None = None,
               wide_from_device_euler: np.ndarray | None = None,
               game_steer: float | None = None) -> None:
        # Human-input gate: tracked even when we'd reject for other
        # reasons, so the cooldown starts the instant a yank happens
        # rather than the next time we'd otherwise pass.
        if game_steer is not None:
            if self._last_game_steer is not None:
                if abs(game_steer - self._last_game_steer) > HUMAN_STEER_DELTA:
                    self._suppress_frames_remaining = HUMAN_SUPPRESS_FRAMES
            self._last_game_steer = float(game_steer)
        if self._suppress_frames_remaining > 0:
            self._suppress_frames_remaining -= 1

        # --- gate set 1: telemetry ---
        if pose is None or v_ego is None:
            return
        if v_ego < MIN_SPEED_FILTER_MPS:
            self.rej_speed += 1
            self.rejected += 1
            return
        if (actual_yaw_rate is not None
                and abs(actual_yaw_rate) > MAX_YAW_RATE_FILTER):
            self.rej_turning += 1
            self.rejected += 1
            return
        if (actual_yaw_rate is not None
                and abs(actual_yaw_rate) * v_ego > MAX_LAT_ACC_MPS2):
            self.rej_lat_acc += 1
            self.rejected += 1
            return
        if self._suppress_frames_remaining > 0:
            self.rej_human += 1
            self.rejected += 1
            return

        vx, vy, vz = float(pose[0]), float(pose[1]), float(pose[2])

        # --- gate set 2: model self-consistency ---
        if vx < MIN_SPEED_FILTER_MPS:
            self.rej_speed += 1
            self.rejected += 1
            return

        # Lateral-velocity angle stddev gate — when the model is
        # uncertain about its lateral velocity, the implied yaw is
        # noisy and updates will poison the calibration. openpilot
        # uses `atan2(transStd[1], trans[0]) < MAX_VEL_ANGLE_STD`.
        if pose_std is not None and len(pose_std) >= 2:
            vel_angle_std = math.atan2(float(pose_std[1]), max(vx, 1e-3))
            if vel_angle_std > MAX_VEL_ANGLE_STD:
                self.rej_vel_std += 1
                self.rejected += 1
                return

        # Road-height stddev gate — the model's `road_transform_std[2]`
        # is uncertainty on the camera-above-road height. Reject if it
        # exceeds 30 mm (openpilot's `MAX_HEIGHT_STD = exp(-3.5)`).
        observed_height = None
        if road_transform is not None and len(road_transform) >= 3:
            observed_height = float(road_transform[2])
            self.last_observed_height = observed_height
            if road_transform_std is not None and len(road_transform_std) >= 3:
                if float(road_transform_std[2]) > MAX_HEIGHT_STD:
                    self.rej_height_std += 1
                    self.rejected += 1
                    return

        # --- residual rpy from pose (openpilot calibrationd.py:172-175) ---
        observed_pitch = -math.atan2(vz, vx)
        observed_yaw = math.atan2(vy, vx)
        self.last_raw_pitch = math.degrees(observed_pitch)
        self.last_raw_yaw = math.degrees(observed_yaw)

        # --- rotation composition onto current calib (not EMA) ---
        current_rpy = (0.0,
                       math.radians(calib.pitch_deg),
                       math.radians(calib.yaw_deg))
        new_rpy = _wrap_rotation_composition(
            current_rpy, (0.0, observed_pitch, observed_yaw))

        # Sanity-bound the composed value before letting it land.
        new_pitch_deg = math.degrees(new_rpy[1])
        new_yaw_deg = math.degrees(new_rpy[2])
        if not (PITCH_MIN_DEG <= new_pitch_deg <= PITCH_MAX_DEG):
            self.rej_pitch += 1
            self.rejected += 1
            return
        if not (YAW_MIN_DEG <= new_yaw_deg <= YAW_MAX_DEG):
            self.rej_yaw += 1
            self.rejected += 1
            return
        if (observed_height is not None
                and not (HEIGHT_MIN_M <= observed_height <= HEIGHT_MAX_M)):
            self.rejected += 1
            return

        # --- accept: feed the in-flight block ---
        self._block_rpy_sum += np.asarray(new_rpy, dtype=np.float64)
        if observed_height is not None:
            self._block_height_sum += observed_height
        self._block_n += 1
        self.samples += 1
        self._accept_timestamps.append(time.monotonic())

        # Wide-camera mount estimate (slice 99:105) — averaged separately
        # for HUD; openpilot publishes this on liveCalibration but doesn't
        # loop it back into the warp.
        if (wide_from_device_euler is not None
                and len(wide_from_device_euler) >= 3):
            self._wide_sum += np.asarray(
                wide_from_device_euler[:3], dtype=np.float64)
            self._wide_n += 1

        if self._block_n >= BLOCK_SIZE:
            self._commit_block(calib)
        elif (self._game is not None
                and self._block_n % 20 == 0
                and self._block_n > 0):
            # Periodic in-flight save (~once per second at 20 Hz). Caps
            # worst-case progress loss on a Quit to ~1 s.
            self.save_state(self._game)

    def _commit_block(self, calib: "Calibration") -> None:
        """Block is full — fold the running averages into history.

        The block-mean watchdog (`BLOCK_JUMP_THRESHOLD_DEG`) compares
        this block's mean rpy to the running mean of prior blocks.
        Blocks that jump too far are dropped — a single bad block
        otherwise drags the smoothed estimate by ~1° and takes many
        more good blocks to recover.
        """
        block_rpy = self._block_rpy_sum / max(self._block_n, 1)
        block_height = (self._block_height_sum / self._block_n
                        if (self._block_n > 0
                            and self._block_height_sum != 0.0)
                        else None)

        # Watchdog: only enforce once we have prior blocks to compare to.
        # Single committed block has no trend; first samples may legitimately
        # carry large residuals if the user just nudged the calib.
        watchdog_rejected = False
        if self._rpys:
            mean_arr = np.mean(np.stack(self._rpys, axis=0), axis=0)
            d_pitch_deg = math.degrees(abs(block_rpy[1] - mean_arr[1]))
            d_yaw_deg = math.degrees(abs(block_rpy[2] - mean_arr[2]))
            if (d_pitch_deg > BLOCK_JUMP_THRESHOLD_DEG
                    or d_yaw_deg > BLOCK_JUMP_THRESHOLD_DEG):
                watchdog_rejected = True
                self.rej_block_jump += 1

        if not watchdog_rejected:
            self._rpys.append(block_rpy)
            if block_height is not None:
                self._heights.append(block_height)
            if self._wide_n > 0:
                self.wide_from_device_estimate = self._wide_sum / self._wide_n

        # Reset in-flight block (whether we kept it or dropped it).
        self._block_rpy_sum[:] = 0.0
        self._block_height_sum = 0.0
        self._block_n = 0
        self._wide_sum[:] = 0.0
        self._wide_n = 0

        # Recompute calStatus and possibly commit to calib.
        self._refresh_cal_status()

        # Auto-recovery: track consecutive INVALID blocks. Once we cross
        # the threshold, reset everything — the warp baseline is so far
        # off that further updates won't pull it back, and the user is
        # better off restarting calibration than staring at INVALID.
        if self.cal_status == CalStatus.INVALID:
            self._consecutive_invalid_blocks += 1
            if self._consecutive_invalid_blocks >= AUTO_RESET_INVALID_BLOCKS:
                self.reset(reason=(
                    f"auto-reset after {self._consecutive_invalid_blocks} "
                    f"consecutive INVALID blocks — drift unrecoverable"))
                return
        else:
            self._consecutive_invalid_blocks = 0

        if self.writes_enabled:
            pitch = self.pitch_estimate
            yaw = self.yaw_estimate
            height = self.height_estimate
            if pitch is not None:
                calib.pitch_deg = float(pitch)
            if yaw is not None:
                calib.yaw_deg = float(yaw)
            # Height directly drives `road_to_image`'s placement of the
            # road plane (z_FRD = +height_m), so a wrong manual value
            # vertically misaligns the overlay. The per-sample
            # `MAX_HEIGHT_STD` gate (~30 mm) already drops noisy samples
            # before they reach the block average, so by the time we
            # commit here the value has been triple-filtered.
            if height is not None and HEIGHT_MIN_M <= height <= HEIGHT_MAX_M:
                calib.height_m = float(height)
            self._wrote_to_calib = True
            self._blocks_since_save += 1
            if self._blocks_since_save >= 2:
                calib.save(game=self._game)
                self._blocks_since_save = 0

        # Persist our own block history every commit (~once per ~12s of
        # accepted highway samples). This is what lets the first-drive
        # wizard survive a Quit/relaunch — without it, _rpys resets to
        # empty on every launch and the user has to start over.
        if self._game is not None:
            self.save_state(self._game)

    def _refresh_cal_status(self) -> None:
        n = len(self._rpys)
        if n == 0:
            self.cal_status = CalStatus.UNCALIBRATED
            return
        pitch = self.pitch_estimate
        yaw = self.yaw_estimate
        height = self.height_estimate
        if pitch is None or yaw is None:
            self.cal_status = CalStatus.UNCALIBRATED
            return
        if not (PITCH_MIN_DEG <= pitch <= PITCH_MAX_DEG):
            self.cal_status = CalStatus.INVALID
            return
        if not (YAW_MIN_DEG <= yaw <= YAW_MAX_DEG):
            self.cal_status = CalStatus.INVALID
            return
        if height is not None and not (HEIGHT_MIN_M <= height <= HEIGHT_MAX_M):
            self.cal_status = CalStatus.INVALID
            return
        if n >= INPUTS_NEEDED:
            self.cal_status = CalStatus.CALIBRATED
        else:
            self.cal_status = CalStatus.UNCALIBRATED

    # ----- diagnostics summary for HUD -----

    @property
    def status_label(self) -> str:
        return {
            CalStatus.UNCALIBRATED: "UNCALIBRATED",
            CalStatus.CALIBRATED: "CALIBRATED",
            CalStatus.INVALID: "INVALID",
        }[self.cal_status]

    # ----- back-compat attributes the HUD already reads -----

    @property
    def MIN_SAMPLES_TO_WRITE(self) -> int:
        """Total samples needed before the first block lands × required
        blocks for CALIBRATED status. Used by the HUD to show progress."""
        return INPUTS_NEEDED * BLOCK_SIZE

    # ----- recovery -----

    def reset(self, reason: str | None = None) -> None:
        """Wipe all calibration state and remove the on-disk snapshot.

        Called from three places:
          - auto-reset after sustained INVALID (drift unrecoverable),
          - the overlay `R` hotkey (user-triggered),
          - the `--reset-calib` CLI flag at startup.

        Leaves `last_reset_reason` set so the HUD can surface why the
        bar suddenly returned to 0%."""
        self._block_rpy_sum[:] = 0.0
        self._block_height_sum = 0.0
        self._block_n = 0
        self._rpys.clear()
        self._heights.clear()
        self._wide_sum[:] = 0.0
        self._wide_n = 0
        self.wide_from_device_estimate = None
        self.samples = 0
        self.rejected = 0
        self.rej_speed = 0
        self.rej_turning = 0
        self.rej_vel_std = 0
        self.rej_height_std = 0
        self.rej_pitch = 0
        self.rej_yaw = 0
        self.rej_lat_acc = 0
        self.rej_human = 0
        self.rej_block_jump = 0
        self.cal_status = CalStatus.UNCALIBRATED
        self._consecutive_invalid_blocks = 0
        self._wrote_to_calib = False
        self._blocks_since_save = 0
        self._accept_timestamps.clear()
        self.last_reset_reason = reason
        if self._game is not None:
            try:
                p = state_path("livecalib_state", self._game)
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    # ----- diagnostics -----

    @property
    def acceptance_rate_hz(self) -> float:
        """Recent acceptance rate (samples/sec). 0 if we haven't seen
        any accepted samples in the last 10s. Used by the wizard to
        compute ETA-to-CALIBRATED."""
        if len(self._accept_timestamps) < 2:
            return 0.0
        now = time.monotonic()
        window_start = self._accept_timestamps[0]
        # Drop samples older than 10s for ETA freshness.
        if now - window_start > 10.0:
            window_start = now - 10.0
        # Count samples within the window.
        recent = sum(1 for t in self._accept_timestamps if t >= window_start)
        span = max(0.5, now - window_start)
        return recent / span

    def eta_to_calibrated_s(self) -> float | None:
        """Estimated seconds remaining until CALIBRATED, based on the
        current acceptance rate. None if we have no rate signal yet or
        we're already done."""
        if self.cal_status == CalStatus.CALIBRATED:
            return 0.0
        rate = self.acceptance_rate_hz
        if rate < 0.1:
            return None
        remaining_samples = (INPUTS_NEEDED * BLOCK_SIZE) - (
            self.blocks * BLOCK_SIZE + self._block_n)
        if remaining_samples <= 0:
            return 0.0
        return remaining_samples / rate

    # ----- persistence -----

    def save_state(self, game: str) -> None:
        """Serialize the committed block history (and the in-flight block
        accumulator) so a Quit/relaunch resumes calibration where it left
        off. Called on every block commit; failures are swallowed."""
        try:
            payload = {
                "rpys": [[float(x) for x in r] for r in self._rpys],
                "heights": [float(h) for h in self._heights],
                "wide_sum": [float(x) for x in self._wide_sum],
                "wide_n": int(self._wide_n),
                "wide_from_device_estimate": (
                    [float(x) for x in self.wide_from_device_estimate]
                    if self.wide_from_device_estimate is not None else None),
                "block_rpy_sum": [float(x) for x in self._block_rpy_sum],
                "block_height_sum": float(self._block_height_sum),
                "block_n": int(self._block_n),
                "samples": int(self.samples),
                "rejected": int(self.rejected),
                "cal_status": int(self.cal_status),
            }
            state_path("livecalib_state", game).write_text(
                json.dumps(payload, indent=2))
        except OSError:
            pass

    def load_state(self, game: str) -> None:
        """Restore the block history saved by `save_state`. No-op if no
        state file exists (fresh user / fresh game)."""
        p = state_path("livecalib_state", game)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return
        try:
            for r in data.get("rpys", []):
                arr = np.asarray(r, dtype=np.float64)
                if arr.shape == (3,):
                    self._rpys.append(arr)
            for h in data.get("heights", []):
                self._heights.append(float(h))
            ws = data.get("wide_sum")
            if ws is not None and len(ws) == 3:
                self._wide_sum = np.asarray(ws, dtype=np.float64)
            self._wide_n = int(data.get("wide_n", 0))
            wfde = data.get("wide_from_device_estimate")
            if wfde is not None and len(wfde) == 3:
                self.wide_from_device_estimate = np.asarray(
                    wfde, dtype=np.float64)
            brs = data.get("block_rpy_sum")
            if brs is not None and len(brs) == 3:
                self._block_rpy_sum = np.asarray(brs, dtype=np.float64)
            self._block_height_sum = float(data.get("block_height_sum", 0.0))
            self._block_n = int(data.get("block_n", 0))
            self.samples = int(data.get("samples", 0))
            self.rejected = int(data.get("rejected", 0))
            # Refresh status from the loaded history rather than trusting
            # the saved enum — bounds may have changed between versions.
            self._refresh_cal_status()
        except (ValueError, TypeError):
            # Bad shape / corrupt file → start fresh, leave _rpys etc as init.
            self._rpys.clear()
            self._heights.clear()
            self._block_n = 0
            self._block_rpy_sum[:] = 0.0
            self._block_height_sum = 0.0
