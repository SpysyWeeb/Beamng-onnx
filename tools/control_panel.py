#!/usr/bin/env python3
"""M3 control panel — closed-loop driving with the viewer inside.

One window: the live overlay view on top, a button strip + HUD at the
bottom. The model steers and (optionally) drives throttle/brake via
simsteer's controllers; buttons/keys send desires (lane changes, turns)
straight into the model's desire input — the same mechanism openpilot's
blinker uses.

Pipeline per 20 Hz frame:
    camera stream_raw -> FrameQueue warp -> model (desire vec) -> decode
    -> LateralController (curvature -> axis via LiveParams learned rack)
    -> LongitudinalController (plan accel + corner braking + lead ACC)
    -> vehicle.control()

LiveParams (game="beamng") learns axis->wheel online. Wheel angle is
synthesized from the MODEL's ego yaw rate (pose[5]) via the bicycle
model — BeamNG doesn't expose road-wheel angle, and this matches the
Forza/AC profile upstream. Before first engage, run CAL (button or 'r'):
a scripted slalom that feeds the learner until trusted (~100 samples).

Keys:
    e engage/disengage      f force-engage (skip trust gate)
    a/d lane change L/R     z/c turn L/R
    l toggle longitudinal   g toggle BeamNG AI driving (calibration aid)
    r run auto-calibration  -/= max speed down/up
    ESC quit (sends neutral control)

Usage (BeamNG running via launch_beamng.sh):
    .venv/bin/python3 tools/control_panel.py [--split] [--scale 0.8]
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

np.seterr(divide="ignore", invalid="ignore")

from simsteer.core.calibration import Calibration
from simsteer.core.constants import DESIRE_LEN
from simsteer.core.control import ControllerConfig, LateralController, LongitudinalController
from simsteer.core.learners.livecalib import LiveCalib
from simsteer.core.learners.liveparams import LiveParams
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue
from simsteer.core.supercombo import SupercomboModel
from simsteer.ui.overlay import draw_overlay

from beamng.world import (BeamNGOnnxWorld, CAM_W, CAM_H, CAM_FOV_H_DEG,
                          CAM_HEIGHT_M, CAM_LATERAL_SIGN, SPAWN_POS,
                          SPAWN_ROT_QUAT)

# ASCII only: cv2's Qt backend fails setMouseCallback ("NULL window
# handler") when the window name contains non-ASCII (e.g. an em-dash).
WINDOW = "Beamng-onnx control panel"
WHEELBASE_M = 2.9          # bastion-ish; constant error folds into LiveParams
PANEL_H = 96               # button strip + HUD height (px)
DESIRE_HOLD_S = {1: 3.0, 2: 3.0, 3: 2.5, 4: 2.5}   # turn L/R, lane L/R
DESIRE_NAME = {1: "TURN L", 2: "TURN R", 3: "LANE L", 4: "LANE R"}
WARMUP_FRAMES = 100        # model context fill before engage allowed
# Calibration v3: deterministic system-ID. Teleport to the straight
# highway spawn, drive straight to CAL_V, apply short alternating
# steering pulses, measure the yaw response through the model's pose,
# and solve the rack ratio by least squares (through the origin —
# FILTER_DIRECT is linear with b~0, measured). No BeamNG AI (drives
# mid-road with tiny inputs -> poor excitation, fit differed run to
# run) and no user. Alternating pulses keep net heading ~zero so the
# car snakes around the lane instead of leaving the road.
CAL_V = 8.0                      # m/s during pulses (drift ~ v^2 k)
CAL_PULSES = [+0.05, -0.05, +0.09, -0.09, +0.13, -0.13]
CAL_PULSE_S = 1.4                # per pulse
CAL_PULSE_SETTLE_S = 0.5         # ignore samples while yaw settles
# Longitudinal system-ID: pedal steps on the straight, measure achieved
# accel (wheelspeed slope) and fit the real pedal->accel gains so a
# commanded 2 m/s^2 actually produces 2 m/s^2. (kind, pedal, duration).
# "rec" = recover speed between brake steps (advances early at 18 m/s).
# "slow" drops back to ~9 m/s between throttle steps so each measures
# from the same speed (at 20+ m/s the engine is power-limited and the
# full-throttle point reads LOW, corrupting the fit). "coast" measures
# pedals-free drag+engine-braking so the brake fit is net of it —
# conflating drag into the brake intercept made the coast gap swallow
# the whole ISO command range and the service brakes never fired.
CAL_LONG_SCRIPT = [
    ("thr", 0.35, 1.6), ("slow", 0.0, 5.0),
    ("thr", 0.70, 1.6), ("slow", 0.0, 5.0),
    ("thr", 1.00, 1.6),
    ("rec", 0.0, 6.0),
    ("coast", 0.0, 1.8),
    ("brk", 0.30, 1.2), ("rec", 0.0, 6.0),
    ("brk", 0.60, 1.2), ("rec", 0.0, 6.0),
    ("brk", 1.00, 2.5),
]
CAL_LONG_SETTLE_S = 0.4          # pedal/weight-transfer settling per step
CAL_DURATION_S = (4.0 + 15.0 + len(CAL_PULSES) * CAL_PULSE_S
                  + sum(s[2] for s in CAL_LONG_SCRIPT) + 2.0)


class Telemetry(threading.Thread):
    """Polls world telemetry (~10 Hz TCP) off the model loop."""

    def __init__(self, world: BeamNGOnnxWorld):
        super().__init__(daemon=True, name="telemetry")
        self.world = world
        self.lock = threading.Lock()
        self.data = {"v_ego": 0.0, "steering_deg": 0.0,
                     "steering_input": 0.0, "heading_rad": 0.0,
                     "pos": (0.0, 0.0, 0.0)}
        self.ok = False
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                d = self.world.poll_telemetry()
                with self.lock:
                    self.data = d
                    self.ok = True
            except Exception:
                self.ok = False
            time.sleep(0.1)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.data)

    def stop(self) -> None:
        self._stop.set()


MOD_PORT = 64257
MOD_CMDS = {"engage", "lane_l", "lane_r", "turn_l", "turn_r", "long",
            "cal", "ai", "spd_dn", "spd_up"}


class ModBridge(threading.Thread):
    """UDP link to the in-game imgui panel (beamng_mod/onnx-panel).

    The mod sends button clicks as single-token datagrams; we relay
    them into App.action(). We push a short status string back to the
    last-seen client so the in-game window shows engagement state."""

    def __init__(self, app: "App"):
        super().__init__(daemon=True, name="mod-bridge")
        self.app = app
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", MOD_PORT))
        self.sock.settimeout(0.3)
        self.client = None
        self._stop = threading.Event()

    def status_line(self) -> str:
        app = self.app
        v = app.tel.snapshot()["v_ego"] * 2.237
        cap = app.cfg.max_speed_mps * 2.237
        s = ("ENGAGED" if app.engaged else
             ("CAL" if app.cal_active else "manual"))
        line = (f"{s} | {v:.0f} mph | cap {cap:.0f} | "
                f"long {app.long_mode}")
        if app.desire_idx is not None:
            line += f" | {DESIRE_NAME[app.desire_idx]}"
        return line

    def run(self) -> None:
        last_push = 0.0
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(256)
                self.client = addr
                cmd = data.decode("ascii", "ignore").strip()
                if cmd in MOD_CMDS:
                    self.app.action(cmd)
                elif cmd == "hello" and time.monotonic() - last_push > 5:
                    print("[panel] in-game panel connected", flush=True)
            except socket.timeout:
                pass
            except OSError:
                break
            now = time.monotonic()
            if self.client and now - last_push > 0.4:
                last_push = now
                try:
                    self.sock.sendto(self.status_line().encode("ascii"),
                                     self.client)
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


class Panel:
    """Buttons + HUD strip below the camera view."""

    DESIRE_KEYS = ("lane_l", "lane_r", "turn_l", "turn_r")

    def __init__(self, state: "App"):
        self.app = state
        self.buttons: list[tuple[tuple[int, int, int, int], str, str]] = []
        # desire button currently held down (mouse) — main loop repeats
        # the action each frame so the desire stays alive while held
        self.held_desire: str | None = None

    def layout(self, y0: int) -> None:
        labels = [("engage", ""), ("lane_l", "< LANE"), ("lane_r", "LANE >"),
                  ("turn_l", "< TURN"), ("turn_r", "TURN >"),
                  ("long", ""), ("cal", "CAL"), ("ai", "AI"),
                  ("spd_dn", "SPD-"), ("spd_up", "SPD+")]
        n = len(labels)
        w = CAM_W // n
        self.buttons = [((i * w, y0, w - 4, 40), key, lab)
                        for i, (key, lab) in enumerate(labels)]

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONUP:
            self.held_desire = None
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (bx, by, bw, bh), key, _ in self.buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.app.action(key)
                if key in self.DESIRE_KEYS:
                    self.held_desire = key
                return

    def draw(self, canvas: np.ndarray) -> None:
        app = self.app
        for (bx, by, bw, bh), key, lab in self.buttons:
            if key == "engage":
                lab = "DISENGAGE" if app.engaged else "ENGAGE"
                col = (60, 60, 200) if app.engaged else (60, 160, 60)
            elif key == "long":
                lab = f"LONG: {app.long_mode.upper()}"
                col = {"exp": (60, 120, 160), "chill": (90, 120, 60),
                       "off": (70, 70, 70)}[app.long_mode]
            elif key == "ai":
                lab = f"AI {'ON' if app.ai_on else 'off'}"
                col = (140, 90, 40) if app.ai_on else (70, 70, 70)
            elif key == "cal":
                col = (40, 110, 140) if app.cal_active else (70, 70, 70)
                if app.cal_active:
                    lab = f"CAL {int(app.cal_left_s)}s"
            else:
                col = (70, 70, 70)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, -1)
            cv2.putText(canvas, lab, (bx + 8, by + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)


class App:
    def __init__(self, args):
        self.calib = Calibration(image_w=CAM_W, image_h=CAM_H,
                                 fov_h_deg=CAM_FOV_H_DEG,
                                 height_m=CAM_HEIGHT_M,
                                 lateral_sign=CAM_LATERAL_SIGN)
        # World FIRST, GPU session second (RDNA4 level-load VRAM rule).
        self.world = BeamNGOnnxWorld()
        if args.split:
            self.model = DrivingModel(
                providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
                policy_providers=["CPUExecutionProvider"], intra_op_threads=3)
            self._decode = lambda i, b, d: decode(*self.model.step(i, b, desire=d))
            name = f"split ({self.model.active_provider})"
        else:
            self.model = SupercomboModel(
                providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
                intra_op_threads=3)
            self._decode = lambda i, b, d: self.model.decode(
                self.model.step(i, b, desire=d))
            name = f"supercombo ({self.model.active_provider})"
        print(f"[panel] model: {name}", flush=True)

        self.queue = FrameQueue()
        self.tel = Telemetry(self.world)
        self.tel.start()

        # In-game imgui panel (beamng_mod/): load the extension if the
        # mod is mounted; harmless no-op inside pcall when it isn't.
        try:
            self.world.bng.control.queue_lua_command(
                "pcall(function() extensions.load('onnxPanel') end)")
        except Exception as exc:
            print(f"[panel] in-game panel load skipped: {exc}", flush=True)

        # per-game config: CAL persists the measured lookahead here
        cfg = ControllerConfig.load(game="beamng")
        cfg.wheelbase_m = WHEELBASE_M
        cfg.max_speed_mps = 55.0 / 2.237   # start on the 5-mph grid
        self.cfg = cfg
        self.lp = LiveParams(game="beamng")
        # Online camera-pose calibration (openpilot's calibrationd):
        # learns effective pitch/yaw/height while driving, commits into
        # calib (warp + overlay) once CALIBRATED. Persists per-game, so
        # like openpilot it fully calibrates once and refines forever.
        self.lc = LiveCalib(game="beamng")
        self.lat = LateralController(cfg, live_params=self.lp)
        self.long = LongitudinalController(cfg)

        self.engaged = False
        # exp: model's e2e accel; chill: cruise at cap + leads/corners;
        # off: user owns the pedals, model steers only.
        self.long_mode = "exp"
        self.ai_on = False
        self.frame_idx = 0
        self.desire_idx: int | None = None
        self.desire_until = 0.0
        self.cal_active = False
        self.cal_until = 0.0
        self._cal_speed = 0.0
        self.banner = ""
        self.banner_until = 0.0
        self.last_steer = 0.0
        self.last_thr = 0.0
        self.last_brk = 0.0
        self.hz = 0.0
        self._signal_state: str | None = None
        self._off_ema = 0.0
        # flight recorder: capture what the model saw when its plan
        # velocity collapses mid-drive (the "stops on open highway"
        # mystery) — main loop writes the frame+state on request
        self._incident_t = 0.0
        self.incident_note: str | None = None

    # ---- UI actions ----

    @property
    def cal_left_s(self) -> float:
        return max(0.0, self.cal_until - time.monotonic())

    def set_banner(self, text: str, secs: float = 2.5) -> None:
        self.banner = text
        self.banner_until = time.monotonic() + secs
        print(f"[panel] {text}", flush=True)

    def engage_allowed(self) -> tuple[bool, str]:
        if self.frame_idx < WARMUP_FRAMES:
            return False, f"model warming up ({self.frame_idx}/{WARMUP_FRAMES})"
        if not self.tel.ok:
            return False, "no telemetry"
        if not self.lp.trusted():
            return False, (f"steering not calibrated "
                           f"({max(self.lp.samples, self.lp.session_samples)}"
                           f"/100 samples — run CAL or drive manually)")
        return True, ""

    def action(self, key: str) -> None:
        now = time.monotonic()
        if key == "engage":
            if self.engaged:
                self.disengage("button")
            else:
                ok, why = self.engage_allowed()
                if ok:
                    self.engage()
                else:
                    self.set_banner(f"engage blocked: {why}")
        elif key == "force":
            self.engage(forced=True)
        elif key in ("lane_l", "lane_r", "turn_l", "turn_r"):
            idx = {"turn_l": 1, "turn_r": 2, "lane_l": 3, "lane_r": 4}[key]
            if self.desire_idx == idx and now < self.desire_until:
                # repeat while held (widget resends every 300 ms, cv2
                # key autorepeat, desktop mouse-hold): keep the desire
                # alive ~0.8 s past the last repeat — openpilot's
                # blinker-held-desire-active behavior. A single tap
                # still gets the full DESIRE_HOLD_S pulse via max().
                self.desire_until = max(self.desire_until, now + 0.8)
            else:
                self.desire_idx = idx
                self.desire_until = now + DESIRE_HOLD_S[idx]
                self.set_banner(f"desire: {DESIRE_NAME[idx]}")
            self._update_signal()
        elif key == "long":
            order = ["exp", "chill", "off"]
            self.long_mode = order[(order.index(self.long_mode) + 1) % 3]
            desc = {"exp": "EXPERIMENTAL (model drives the pedals)",
                    "chill": "CHILL (cruise at cap, brake for leads)",
                    "off": "OFF (your pedals, model steers)"}
            self.set_banner(f"long: {desc[self.long_mode]}", 3.0)
        elif key == "ai":
            self.ai_on = not self.ai_on
            if self.ai_on:
                if self.engaged:
                    self.disengage("AI takeover")
                self.world.ai_drive(20.0)
            else:
                self.world.ai_disable()
            self.set_banner(f"BeamNG AI {'ON' if self.ai_on else 'OFF'}")
        elif key == "cal":
            self.start_cal()
        elif key in ("spd_dn", "spd_up"):
            # step in whole-5-mph notches (55, 60, 65 ...) like a real
            # cruise stalk; snap first in case the cap started off-grid
            step = -5.0 if key == "spd_dn" else 5.0
            mph = round(self.cfg.max_speed_mps * 2.237 / 5.0) * 5.0 + step
            mph = min(100.0, max(10.0, mph))
            self.cfg.max_speed_mps = mph / 2.237
            self.set_banner(f"max speed {mph:.0f} mph")

    def _update_signal(self) -> None:
        """Blinker follows the active desire (left for turn/lane L,
        right for R, off when it expires). Sends only on transitions."""
        want = None
        if self.desire_idx in (1, 3):
            want = "left"
        elif self.desire_idx in (2, 4):
            want = "right"
        if want != self._signal_state:
            self._signal_state = want
            try:
                self.world.set_signal(want)
            except Exception:
                pass

    def engage(self, forced: bool = False) -> None:
        if self.ai_on:
            self.world.ai_disable()
            self.ai_on = False
        self.world.realistic_gearbox()   # arcade + held brake = REVERSE
        self.world.ensure_drive()        # post-reset the box sits in N/P
        self.cal_active = False
        self.lat.reset()
        self.long.reset()
        self.engaged = True
        self.set_banner("ENGAGED" + (" (forced)" if forced else ""))

    def disengage(self, reason: str) -> None:
        self.engaged = False
        self.lat.reset()
        self.long.reset()
        self.world.apply(0.0, 0.0, 0.0)
        self.set_banner(f"DISENGAGED ({reason})")

    def start_cal(self) -> None:
        if self.engaged:
            self.disengage("calibration")
        if self.ai_on:
            self.world.ai_disable()
            self.ai_on = False
        self.world.realistic_gearbox()
        self.world.ensure_drive()
        self.world.vehicle.teleport(SPAWN_POS, rot_quat=SPAWN_ROT_QUAT)
        self.cal_active = True
        self.cal_until = time.monotonic() + CAL_DURATION_S
        self._cal_phase = "settle"
        self._cal_t0 = time.monotonic()
        self._cal_samples: list[tuple[float, float]] = []
        self._cal_series: list[tuple[float, float]] = []   # (axis, wheel)/frame
        self._cal_long: list[tuple[str, float, float]] = []  # (kind, pedal, accel)
        self._cal_long_i = 0
        self._cal_vt: list[tuple[float, float]] = []       # (t, v) per step
        self.set_banner("CAL: scripted system-ID at spawn — hands off", 4.0)

    def _cal_finish(self) -> None:
        self.cal_active = False
        self.world.apply(0.0, 0.0, 0.6)
        n = len(self._cal_samples)
        sw2 = sum(w * w for _, w in self._cal_samples)
        if n < 40 or sw2 < 1e-5:
            self.set_banner(f"CAL FAILED (n={n}) — fit unchanged, retry", 5.0)
            return
        # least squares through the origin: axis = a * wheel
        a = sum(ax * w for ax, w in self._cal_samples) / sw2
        if not (0.5 <= abs(a) <= 50.0):
            self.set_banner(f"CAL FAILED (a={a:.2f} implausible) — retry", 5.0)
            return
        # Direct measurement outranks RLS: install the fit and mark it
        # mature (freeze-level sample count) so online updates can't
        # wander it the way the old AI-watching calibration could.
        # Install as a strong PRIOR, not a frozen constant: RLS keeps
        # learning while driving (openpilot's torqued behavior — auto-
        # freeze is disabled in this fork of LiveParams by design).
        # Tight covariance so a session's first noisy samples can't
        # yank the measured seed; the fit stays adaptive to damage,
        # vehicle changes, etc.
        self.lp.x[:] = [a, 0.0, 0.0]
        self.lp.P = np.diag((0.25, 1e-6, 1e-4))
        self.lp.samples = 1000
        self.lp.session_samples = 1000
        self.lp.save()

        # Loop lag by cross-correlation over the pulse train (command
        # vs model-pose wheel response) — measured ~300 ms, but that
        # CONFLATES perception (render + camera + inference, ~200 ms)
        # with actuation. The model's plan is already expressed in
        # camera-time, so only the ACTUATION side belongs in the
        # steerActuatorDelay analog: direct-input t90 (~0.08 s) + our
        # EPS-emulation filter. Using the full 300 ms made the car turn
        # in early AND hard ("takes curves sharper than it needs to").
        lag_s = 0.0
        if len(self._cal_series) > 40:
            ax = np.array([p[0] for p in self._cal_series])
            wh = np.array([p[1] for p in self._cal_series])
            best_c = -1e9
            for k in range(0, 13):                      # 0..0.6 s @ 20 Hz
                a_s = ax[:len(ax) - k] if k else ax
                w_s = wh[k:]
                sd = float(np.std(a_s) * np.std(w_s))
                if sd < 1e-9:
                    continue
                c = float(np.mean((a_s - a_s.mean()) * (w_s - w_s.mean())) / sd)
                if c > best_c:
                    best_c, lag_s = c, k * 0.05
        self.cfg.lookahead_s = float(np.clip(
            0.10 + self.cfg.steer_smooth_s, 0.10, 0.40))

        # Pedal-map fit: achieved accel vs pedal position, slope with
        # intercept (the intercept absorbs drag/rolling resistance —
        # the speed-I trim covers that online). Sets the scales so a
        # commanded m/s^2 produces that m/s^2 in the game.
        long_msg = ""
        print(f"[cal] long steps (kind, pedal, dv/dt): "
              f"{[(k, p, round(s, 2)) for k, p, s in self._cal_long]}",
              flush=True)
        thr = [(p, s) for k, p, s in self._cal_long if k == "thr"]
        brk = [(p, -s) for k, p, s in self._cal_long if k == "brk"]
        coast = [-s for k, _, s in self._cal_long if k == "coast"]
        # pedals-free deceleration (drag + engine braking) — the brake
        # fit must be net of this or its intercept swallows the whole
        # ISO command range as "coast gap"
        d_coast = float(np.clip(np.mean(coast), 0.3, 3.0)) if coast else 1.5
        # Interpolation tables (measured points, monotonic-filtered):
        # the brake response saturates, so no parametric fit — the
        # curve IS the calibration.
        if len(thr) >= 2:
            pts = [(-d_coast, 0.0)] + sorted(
                (float(s), float(p)) for p, s in thr)
            table = [pts[0]]
            for a_m, p_m in pts[1:]:
                if a_m > table[-1][0] + 0.05:
                    table.append((a_m, p_m))
            if len(table) >= 3:
                self.cfg.pedal_thr_map = [list(x) for x in table]
                self.cfg.max_accel_mps2 = table[-1][0]   # HUD/legacy
                long_msg += f"  thr[{table[-1][0]:.1f}max]"
        if len(brk) >= 2:
            pts = [(d_coast, 0.0)] + sorted(
                (float(d), float(p)) for p, d in brk)
            table = [pts[0]]
            for d_m, p_m in pts[1:]:
                if d_m > table[-1][0] + 0.05:
                    table.append((d_m, p_m))
            if len(table) >= 3:
                # terminal point: anything beyond the strongest
                # measured decel gets full pedal (AEB lands here)
                table.append((table[-1][0] + 3.0, 1.0))
                self.cfg.pedal_brk_map = [list(x) for x in table]
                self.cfg.max_decel_mps2 = table[-2][0]
                long_msg += (f"  brk[{table[-2][0]:.1f}max"
                             f" coast {d_coast:.1f}]")
        # Execute the plan at an openpilot-like horizon. The old 1.0 s
        # anticipation applied the deceleration planned for 1.3 s in
        # the future NOW — the whole stop profile ran early and the car
        # halted metres before the line. Corner braking keeps its own
        # 6 s scan horizon; this only times plan-following.
        self.cfg.long_anticipation_s = 0.5
        self.cfg.save(game="beamng")

        self.set_banner(f"CAL done: a={a:+.2f} ({n} samples, frozen)  "
                        f"lag={lag_s*1000:.0f}ms lookahead "
                        f"{self.cfg.lookahead_s:.2f}s{long_msg}", 6.0)

    # ---- per-frame control ----

    def control_tick(self, decoded, v_ego: float, dt: float) -> None:
        now = time.monotonic()

        # Synthesized wheel angle from the model's ego yaw rate.
        yaw_rate = float(decoded.pose[5])
        k_meas = yaw_rate / max(v_ego, 1.0)
        wheel = math.atan(k_meas * WHEELBASE_M) if v_ego > 1.0 else None

        desire_active = (self.desire_idx is not None
                         and now < self.desire_until)
        if not desire_active:
            self.desire_idx = None
        self._update_signal()

        steer, thr, brk = 0.0, 0.0, 0.0
        commanded = None
        if self.cal_active:
            phase_t = now - self._cal_t0
            if self._cal_phase == "settle":
                self.world.apply(0.0, 0.0, 0.2)   # post-teleport physics
                if phase_t > 2.0:
                    # AFTER the teleport: teleporting resets the vehicle
                    # and drops the box back to neutral — shifting in
                    # start_cal() was undone and CAL revved in N
                    self.world.realistic_gearbox()
                    self.world.ensure_drive()
                    self._cal_phase, self._cal_t0 = "accel", now
            elif self._cal_phase == "accel":
                self.world.apply(0.0, 0.45, 0.0)
                if v_ego >= CAL_V or phase_t > 15.0:
                    self._cal_phase, self._cal_t0 = "pulse", now
            elif self._cal_phase == "pulse":
                i = int(phase_t // CAL_PULSE_S)
                if i >= len(CAL_PULSES):
                    self._cal_phase, self._cal_t0 = "long", now
                    self._cal_long_i = 0
                    self._cal_vt = []
                else:
                    axis = CAL_PULSES[i]
                    thr_c = 0.25 if v_ego < CAL_V else 0.05
                    self.world.apply(axis, thr_c, 0.0)
                    in_pulse = phase_t - i * CAL_PULSE_S
                    if wheel is not None and v_ego > 4.0:
                        # full series (transients included) for the lag
                        # cross-correlation
                        self._cal_series.append((axis, wheel))
                        if in_pulse > CAL_PULSE_SETTLE_S:
                            # settled samples only for the gain fit
                            self._cal_samples.append((axis, wheel))
            elif self._cal_phase == "long":
                if self._cal_long_i >= len(CAL_LONG_SCRIPT):
                    self._cal_finish()
                else:
                    kind, pedal, dur = CAL_LONG_SCRIPT[self._cal_long_i]
                    if kind == "thr":
                        self.world.apply(0.0, pedal, 0.0)
                    elif kind == "brk":
                        self.world.apply(0.0, 0.0, pedal)
                    elif kind == "coast":
                        self.world.apply(0.0, 0.0, 0.0)
                    elif kind == "slow":   # back to thr-step start speed
                        self.world.apply(0.0, 0.0, 0.4)
                    else:   # rec: recover speed between brake steps
                        self.world.apply(0.0, 0.7, 0.0)
                    measured = kind in ("thr", "brk", "coast")
                    if measured and phase_t > CAL_LONG_SETTLE_S:
                        self._cal_vt.append((now, v_ego))
                    done = (phase_t > dur
                            or (kind == "rec" and v_ego >= 18.0)
                            or (kind == "slow" and v_ego <= 9.0))
                    if done:
                        if measured and len(self._cal_vt) >= 6:
                            ts = np.array([p[0] for p in self._cal_vt])
                            vs = np.array([p[1] for p in self._cal_vt])
                            slope = float(np.polyfit(ts - ts[0], vs, 1)[0])
                            self._cal_long.append((kind, pedal, slope))
                        self._cal_long_i += 1
                        self._cal_t0 = now
                        self._cal_vt = []
        elif self.engaged:
            lane_change_cmd = desire_active and self.desire_idx in (3, 4)
            steer = self.lat.compute(decoded, v_ego,
                                     actual_wheel_angle=wheel,
                                     lane_change_command_active=lane_change_cmd,
                                     dt=dt)
            if self.long_mode == "off":
                # steering only — no throttle/brake API calls, so the
                # player's own pedal inputs pass through untouched
                self.world.apply(steer, None, None)
            else:
                thr, brk = self.long.compute(decoded, v_ego,
                                             mode=self.long_mode)
                self.world.apply(steer, thr, brk)
            commanded = steer

        # Feed the learners: our command while we drive, the game's own
        # steering input while a human / BeamNG AI drives (passive
        # fit). Skipped during CAL — the system-ID computes its own
        # fit and installs it wholesale.
        tel = self.tel.snapshot()
        game_steer = (None if self.cal_active else
                      (commanded if commanded is not None
                       else tel["steering_input"]))
        self.lp.update(game_steer, v_ego, wheel, commanded_axis=game_steer)

        # Camera-pose learner runs on every frame (its own gates handle
        # speed/turning/uncertainty); commits into self.calib — which
        # the warp and overlay read — once CALIBRATED.
        if not self.cal_active:
            self.lc.update(self.calib, decoded.pose, decoded.road_transform,
                           yaw_rate, v_ego,
                           pose_std=decoded.pose_std,
                           road_transform_std=decoded.road_transform_std,
                           wide_from_device_euler=decoded.wide_from_device_euler,
                           game_steer=tel["steering_input"])

        self.last_steer, self.last_thr, self.last_brk = steer, thr, brk

        # flight recorder trigger: engaged, moving, and the plan's
        # velocity target collapsed well below current speed with no
        # lead engaged — capture the scene (throttled to 1 per 4 s)
        if (self.engaged and v_ego > 8.0
                and self.long.last_v_target < 0.5 * v_ego
                and self.long.last_lead_x > 400
                and now - self._incident_t > 4.0):
            self._incident_t = now
            self.incident_note = (
                f"v={v_ego:.1f} vT={self.long.last_v_target:.1f} "
                f"aT={self.long.last_a_target:+.2f} "
                f"lanes={np.round(decoded.lane_lines_prob, 2).tolist()} "
                f"leadP={float(decoded.lead_prob[0]):.2f} "
                f"desire_state={np.round(decoded.desire_state, 2).tolist()}")

    # ---- HUD ----

    def hud_lines(self, decoded, v_ego: float) -> list[str]:
        lp = self.lp
        probs = np.round(decoded.lane_lines_prob, 2)
        lead_p = float(decoded.lead_prob[0]) if decoded.lead_prob.size else 0.0
        l1 = (f"{'ENGAGED' if self.engaged else 'manual '}  "
              f"{v_ego*2.237:4.0f} mph (cap {self.cfg.max_speed_mps*2.237:.0f})  "
              f"steer {self.last_steer:+.2f}  thr {self.last_thr:.2f}  "
              f"brk {self.last_brk:.2f}  {self.hz:4.1f} Hz")
        l2 = (f"lanes {probs}  lead {lead_p:.2f}"
              + (f" @{self.long.last_lead_x:.0f}m" if self.long.last_lead_x < 500 else "")
              + f"  curv {self.lat.last_curvature:+.4f}"
              + f"  vT {min(self.long.last_v_target, 99)*2.237:3.0f}"
              + f"  aT {self.long.last_a_target:+.1f}"
              + (f"  corner@{self.long.last_corner_t:.0f}s"
                 if self.long.last_v_safe_corner < self.long.last_v_target + 1
                 else "")
              + ("  AEB!" if self.long.last_aeb else ""))
        # slow lane-center offset EMA (~10 s): steady nonzero sign at
        # highway = controls/mount bias (watch trim fight it); sign
        # flipping with scene/curves = model behavior
        off_now = float(np.mean(decoded.lane_lines[1:3, 0, 0]))
        if abs(off_now) < 3.0:
            self._off_ema += 0.005 * (off_now - self._off_ema)
        trust = "OK" if lp.trusted() else "COLD"
        l3 = (f"rack a={lp.a_linear:+.2f} b={lp.b_quad:+.4f} "
              f"n={max(lp.samples, lp.session_samples)} [{trust}]"
              f"  off~{self._off_ema:+.2f}m"
              f"  trim {self.lat.axis_trim_state:+.3f}"
              f"({self.lat.last_trim_frozen_reason or 'run'})"
              f"  cam[{self.lc.blocks}b"
              f"{'*' if self.lc.writes_enabled else ''}]"
              f"  aFB {self.long.last_a_fb:+.2f}")
        if self.desire_idx is not None:
            l3 += f"   >>> {DESIRE_NAME[self.desire_idx]} " \
                  f"({self.desire_until - time.monotonic():.1f}s)"
        return [l1, l2, l3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true",
                    help="use the split 0.11.1 pair instead of supercombo")
    ap.add_argument("--scale", type=float, default=0.8)
    args = ap.parse_args()

    app = App(args)
    panel = Panel(app)
    panel.layout(CAM_H + 4)
    bridge = ModBridge(app)
    bridge.start()

    canvas = np.zeros((CAM_H + PANEL_H, CAM_W, 3), dtype=np.uint8)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(CAM_W * args.scale),
                     int((CAM_H + PANEL_H) * args.scale))
    # Qt backend realizes the window asynchronously — setMouseCallback
    # throws "NULL window handler" until the event loop has actually
    # created it. Pump the loop and retry; fall back to keyboard-only.
    mouse_ok = False
    for _ in range(20):
        cv2.imshow(WINDOW, canvas)
        cv2.waitKey(20)
        try:
            cv2.setMouseCallback(WINDOW, panel.on_mouse)
            mouse_ok = True
            break
        except cv2.error:
            time.sleep(0.05)
    if not mouse_ok:
        print("[panel] WARNING: mouse buttons unavailable in this cv2 "
              "build — use the keyboard bindings (see --help)", flush=True)
    last = time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            bgr = app.world.poll_bgr()
            if bgr is None:
                if (cv2.waitKey(10) & 0xFF) == 27:
                    break
                continue
            app.frame_idx += 1
            dt = max(1e-3, t0 - last)
            last = t0

            desire_vec = None
            if app.desire_idx is not None:
                desire_vec = np.zeros(DESIRE_LEN, dtype=np.float32)
                desire_vec[app.desire_idx] = 1.0

            img, big = app.queue.push(bgr, app.calib)
            d = app._decode(img, big, desire_vec)

            tel = app.tel.snapshot()
            v_ego = tel["v_ego"]
            app.control_tick(d, v_ego, dt)
            if panel.held_desire:
                app.action(panel.held_desire)   # extend while held

            # auto-disengage on loop-rate collapse
            if app.engaged and app.hz and app.hz < 8.0 \
                    and app.frame_idx > WARMUP_FRAMES:
                app.disengage(f"loop rate {app.hz:.0f} Hz")

            canvas[:CAM_H] = draw_overlay(bgr, d, app.calib)
            canvas[CAM_H:] = 24
            panel.draw(canvas)
            for i, line in enumerate(app.hud_lines(d, v_ego)):
                cv2.putText(canvas, line, (10, CAM_H + 58 + 16 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (230, 230, 230), 1, cv2.LINE_AA)
            if time.monotonic() < app.banner_until:
                cv2.putText(canvas, app.banner, (CAM_W // 2 - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(canvas, app.banner, (CAM_W // 2 - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 255), 2,
                            cv2.LINE_AA)
            if app.incident_note:
                inc_dir = os.path.join(ROOT, "debug_out", "incidents")
                os.makedirs(inc_dir, exist_ok=True)
                stamp = time.strftime("%H%M%S")
                cv2.imwrite(os.path.join(inc_dir, f"stop_{stamp}.png"), canvas)
                with open(os.path.join(inc_dir, f"stop_{stamp}.txt"), "w") as f:
                    f.write(app.incident_note + "\n")
                print(f"[panel] INCIDENT captured: {app.incident_note}",
                      flush=True)
                app.incident_note = None

            cv2.imshow(WINDOW, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord("e"):
                app.action("engage")
            elif key == ord("f"):
                app.action("force")
            elif key == ord("a"):
                app.action("lane_l")
            elif key == ord("d"):
                app.action("lane_r")
            elif key == ord("z"):
                app.action("turn_l")
            elif key == ord("c"):
                app.action("turn_r")
            elif key == ord("l"):
                app.action("long")
            elif key == ord("g"):
                app.action("ai")
            elif key == ord("r"):
                app.action("cal")
            elif key == ord("-"):
                app.action("spd_dn")
            elif key == ord("="):
                app.action("spd_up")

            el = time.monotonic() - t0
            app.hz = 1.0 / el if el > 0 else 0.0
            time.sleep(max(0.0, 0.05 - el))
        return 0
    finally:
        try:
            if app.lp.session_samples > 0:
                app.lp.save()
        except Exception:
            pass
        app.world.apply(0.0, 0.0, 0.0)
        bridge.stop()
        app.tel.stop()
        app.world.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
