"""Plan -> gamepad axes (steering + throttle/brake).

LATERAL — feed-forward + slow steady-state trim:

    desired_k    = desired_curvature_from_plan(v_ego, lookahead_s)
    target_wheel = atan(desired_k * wheelbase * authority)   # bicycle model
    axis_ff      = LiveParams.axis_for_wheel_angle(target_wheel, v_ego)
    axis_trim    = leaky_integrator(LPF(target_wheel - actual_wheel))
    axis         = clip(axis_ff + axis_trim + axis_bias, ±steer_max)

The model's plan trajectory is trained to go through lane center; the
controller just follows it. No explicit lane-keep — if the truck hugs
a wall, the fix is upstream (calibration / camera mount / model input
quality), not a band-aid in this controller. `steer_authority` is a
flat multiplier on the commanded curvature; openpilot doesn't have
this knob.

`LiveParams` learns the inverse rack mapping `axis = a*wheel +
b*wheel*v² + c` from telemetry (with a speed-stiffness term so the
same axis produces a different wheel angle at parking vs highway,
matching ETS2's variable-ratio rack). `axis_for_wheel_angle(target,
v_ego)` returns the FF axis directly. `lateral_sign` is NOT applied
here — LiveParams absorbs whatever sign exists between gamepad axis
and the truck's actual wheel response.

The `axis_trim` term is a leaky integrator on the LPF'd wheel-angle
error. It corrects steady-state offsets that the FF stack alone
can't see (LiveParams bias, residual rack-fit error, model plan
offset). An earlier PID was removed for fighting LiveParams during
transients — this trim is built to specifically *not* re-create
that failure mode: ~10 s integrator τ + 60 s leak τ (40× slower
than the wheel actuator), frozen during every transient regime
(corners, lane changes, saturation, intervention, cold RLS, low
speed), and hard-clipped to ±wheel_trim_clip so it can only nudge.

LONGITUDINAL — accel feed-forward + light P on speed error, mapped to
trigger pedals:

    v_target      = plan velocity at long_lookahead_s
    a_target      = plan accel at long_lookahead_s
    a_cmd         = a_target + speed_p_gain * (v_target - v_ego)
    throttle/brake = a_cmd mapped to [0,1] via max_accel/decel

Lookahead is longer than the steering one because braking input takes
seconds to bleed off speed — we need to react to upcoming curves before
we're in them. The model's plan already encodes "slow down for the bend"
in its velocity prediction; we just follow it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from simsteer.core.constants import T_IDXS
from simsteer.core.learners.liveparams import LiveParams
from simsteer.paths import load_with_fallback, state_path
from simsteer.core.postprocess import Decoded, desired_curvature_lag_adjusted

CONFIG_PATH = state_path("controller")


@dataclass
class ControllerConfig:
    # Cap on the gamepad axis after the inversion. ETS2 saturates the
    # wheel well before axis=1 anyway; lower this if the truck still
    # over-steers (e.g. 0.5 to halve the response).
    steer_max: float = 1.0
    # Static steer-actuator delay (s). Equivalent to openpilot's per-car
    # STEER_ACTUATOR_DELAY — the controller reads the model's plan this
    # far ahead so the commanded curvature lines up with the wheel
    # response by the time it actually arrives. Default 0.3 s is in
    # the openpilot range (most cars 0.10-0.40 s). Tune from the
    # Lateral section; openpilot ships per-car constants, not online
    # estimation.
    lookahead_s: float = 0.3
    # Used when SCS telemetry is unavailable (no plugin). Doesn't drive
    # the truck, just keeps the curvature math sane.
    default_speed: float = 22.0
    # Below this speed the curvature math blows up (k = yaw / v). Command
    # zero instead of garbage.
    min_speed: float = 1.0
    # Truck wheelbase (m), used by the bicycle-model conversion. EU
    # tractors are ~3.5-4.2; default 4.0. A constant error here just
    # rolls into LiveParams.scale, so tuning it precisely doesn't help.
    wheelbase_m: float = 4.0
    # Constant axis bias added to the controller's output. Use to
    # cancel persistent lateral drift the AI can't account for itself
    # (model training-data bias, residual mounting offset, AC alignment
    # quirks, etc). Positive nudges right, negative nudges left.
    # Typical useful range: ±0.1.
    axis_bias: float = 0.0
    # Flat multiplier on the desired curvature (plan + lane-keep) before
    # the bicycle-model conversion to wheel angle. openpilot does NOT
    # have this knob — they take the model's planned curvature as-is
    # and close the loop with a PID/Torque on lateral-accel error. We
    # keep a flat multiplier as a project-specific knob. Default 1.0
    # (comma-faithful) works best here — higher just over-steers. Tune
    # it live in the tuner's Lateral section if a game ever needs more.
    steer_authority: float = 1.0

    # Anticipation buffer added to `lookahead_s` when computing
    # `desired_curvature_lag_adjusted`. Reads the plan's heading at
    # `lookahead_s + curvature_anticipation_s` into the future.
    # openpilot uses +0.2 s here ("for other delays"), but for the
    # sim's gamepad->game path that anticipates corners too early
    # (the truck turns in before the bend), so we default to 0.0 —
    # turn-in happens later. Lower still (negative) reacts even
    # closer to the present; the loop's k/l keys nudge this live.
    # Negative subtracts from lookahead — the function floors total
    # delay at 1 ms so negative values can't break the math.
    curvature_anticipation_s: float = 0.0

    # ISO lateral limits (openpilot drive_helpers.clip_curvature,
    # EU-guideline constants). When lat_jerk_max_mps3 is set the
    # curvature rate limit becomes jerk/v^2 — fast wheel at city
    # speeds, gentle at highway speeds — replacing the legacy
    # speed-interpolated table that starved sharp low-speed turns.
    # lat_accel_max_mps2 clamps commanded lateral g (the "what the
    # vehicle is capable of" bound comma assigns per car). Set
    # lat_jerk_max_mps3 to None to fall back to the legacy table.
    # openpilot's road values are 5.0 / 3.0; those are tuned for a
    # real EPS and passenger comfort, and 3.0 lateral g is exactly why
    # real openpilot balks at sharp city corners. The sim car grips
    # ~8+ m/s^2, so run "strong car" bounds: 2x wheel rate, 4.5 g cap.
    # jerk 10 keeps the fast city wheel; the accel clamp comes back
    # DOWN to 3.5 — at 4.5 the e2e plan's natural inside-line bias
    # executed at full strength (geometry audit exonerated the inputs:
    # vx ratio 0.995, yaw 1.028), cutting curves hard enough to cross
    # lines and trigger unprompted lane commits. Real openpilot's 3.0
    # is partly what suppresses curve-cutting.
    lat_jerk_max_mps3: float | None = 10.0
    lat_accel_max_mps2: float = 3.5
    # First-order smoothing on the final steering axis — emulated EPS
    # actuator dynamics. A real steering motor is a mechanical low-pass
    # (openpilot leans on it: modeld's LAT_SMOOTH_SECONDS is 0 because
    # the EPS does the smoothing); our FILTER_DIRECT input has none, so
    # the plan's frame-to-frame flicker reaches the wheel raw and the
    # model watches its own twitch through the camera. tau 0.1 s ~
    # t90 0.23 s, comparable to a quick EPS. 0 disables.
    steer_smooth_s: float = 0.1

    # Speed-dependent steering response (variable-ratio rack in ETS2
    # and most games) is handled inside LiveParams now — it fits a
    # `b · wheel · v²` stiffness term, so `axis_for_wheel_angle(...,
    # v_ego)` already compensates. No knob here.

    # Closed-loop trim on wheel-angle error. Re-added after the
    # earlier PID was removed for fighting LiveParams during
    # transients — this version operates ~40× slower than the
    # wheel actuator (~10 s integrator τ + 60 s leak τ) and freezes
    # during every transient regime (corners, lane changes,
    # saturation, intervention, cold RLS, low speed). Output is
    # hard-clipped to ±wheel_trim_clip so even if it goes wrong it
    # can only nudge — not swing the axis the wrong way like the
    # old PID could.
    #
    # The pipeline is now:
    #     axis = clip(axis_ff + axis_trim + axis_bias, ±steer_max)
    # where axis_trim is the dynamic correction and axis_bias is the
    # static user knob — both serve their own role.
    wheel_trim_enabled: bool = True
    # Integrator gain. Output (axis) per (rad·s) of LPF'd error. The
    # original 0.005 was sized for 0.1-rad-scale errors; a REAL steady
    # bias (road crown / axis offset) is ~0.002 rad of wheel angle —
    # the field chart showed the desired-vs-actual curvature traces
    # riding a constant step apart (want +0.5, act -0.1 x1000) that
    # 0.005 would have needed an hour to cancel. 0.5 closes it in
    # ~5-10 s and is still slow against the wheel: loop crossover
    # ~0.04 Hz (gain x dwheel/daxis ~ 0.5/rack_a), ~70 deg phase
    # margin with the 1 s error LPF.
    wheel_trim_gain: float = 0.5
    # Hard clip on the integrator output. ±0.05 axis is enough to
    # cancel a typical steady-state offset without ever dominating
    # the FF axis.
    wheel_trim_clip: float = 0.05
    # Leak time constant — the integrator forgets stale trim over
    # this many seconds when the gate is open. 60 s means a stale
    # value decays by ~63% per minute. Prevents the trim from
    # latching onto a value that was valid in one stretch of road
    # but stops being so after a long pause.
    wheel_trim_leak_s: float = 60.0
    # LPF α on the wheel-angle error before it feeds the integrator.
    # α=0.05 at 20 Hz ≈ 20-frame averaging, equivalent τ ≈ 1 s. The
    # LPF (combined with the slow integrator gain) is what stops
    # transient swings from translating into integrator action.
    wheel_trim_error_lpf_alpha: float = 0.05

    # Confidence threshold on the model's desire_state for lane-change
    # indices (3 = lane-L, 4 = lane-R) above which we consider the
    # controller to be mid-lane-change. Used to enable the steering
    # boost during the maneuver.
    lane_change_desire_threshold: float = 0.3
    # How long to hold the desire pulse active after the user presses
    # A/D. openpilot's DesireHelper holds the desire active across
    # PRE_LANE_CHANGE + LANE_CHANGE_STARTING — a few seconds. A
    # single-frame pulse gets washed out of our 5 s context buffer
    # before the model commits. Default 2.5 s gives the model enough
    # sustained input to see the maneuver through; bump to 4-5 s if
    # lane changes still feel half-hearted.
    lane_change_hold_s: float = 2.5

    # ----- LONGITUDINAL -----
    # Extra anticipation (seconds) on top of `lookahead_s` for the
    # longitudinal controller. Pedal commands take much longer to act
    # on speed than steering does on heading — we have to see the bend
    # coming before we're in it. Effective long-action-time =
    # lookahead_s + long_anticipation_s. lookahead_s is the static
    # steerActuatorDelay tunable from the tuner.
    long_anticipation_s: float = 0.5
    # m/s^2 of *commanded* acceleration that maps to a full-pressed
    # throttle (axis=1.0). This is a PEDAL SCALE, not a limit — it
    # should reflect what the vehicle actually does at full pedal, or
    # a modest command saturates the pedal and the car lunges (a 2
    # m/s^2 request with scale 1.5 was full throttle = 4-6 m/s^2 in a
    # BeamNG sedan). The LIMITS live in accel_cmd_max/min below.
    max_accel_mps2: float = 4.0
    # Same on the brake side. Full brake in a game car is ~8-10 m/s^2.
    max_decel_mps2: float = 8.0
    # Calibrated pedal maps (CAL's longitudinal system-ID): measured
    # (accel, pedal) points, inverted by interpolation. Tables, not a
    # parametric fit, because the measured brake response SATURATES
    # (0.3 pedal ~ 5 m/s^2, 0.6 ~ 8, 1.0 no more) — a line either
    # over-brakes 2x or its intercept swallows the ISO command range.
    #   pedal_thr_map: ascending [[accel_mps2, throttle], ...]; first
    #     point is coast (zero pedal, negative accel from drag).
    #   pedal_brk_map: ascending [[decel_mps2, brake], ...]; first
    #     point is coast decel at zero pedal.
    # None -> legacy scale-only mapping above.
    pedal_thr_map: list | None = None
    pedal_brk_map: list | None = None
    # The tables' zero-pedal anchors (coast decel / drag) were measured
    # at ~18-25 m/s; drag and engine braking shrink with speed, so the
    # anchors are scaled by v/18 (floored) before inversion — otherwise
    # gentle stop-phase commands (-0.5..-1.2) fall into a phantom coast
    # gap at low speed and the brake never fires at the end of stops.
    pedal_cal_v: float = 18.0
    # Stopping state (openpilot LongControl 'stopping'): once the plan
    # wants a stop and speed drops below stop_hold_speed, RAMP the
    # brake up to stop_hold_brake over stop_brake_ramp_s and hold —
    # an automatic in D creeps through the stop line at zero pedal
    # otherwise. Entry speed must be ABOVE creep speed (~1 m/s) or a
    # weakly-braked car settles into a permanent 2 mph creep just
    # outside the gate (field report 2026-07-01).
    stop_hold_speed: float = 1.5
    stop_hold_brake: float = 0.3
    stop_brake_ramp_s: float = 1.0
    # Closed loop on MEASURED acceleration — openpilot's actual long
    # mechanism (controlsd runs a PID on a_target vs a_ego; it doesn't
    # "learn" gains, feedback adapts to the vehicle in real time).
    # Proportional on the LPF'd error between what we commanded ~0.3 s
    # ago and what the car actually did (dv/dt). Corrects pedal-map
    # residuals, load, slopes, damage — live. 0 disables.
    # This loop carries ~1 s of total delay (setpoint history + LPFs +
    # pedal response). P-dominant designs oscillated here TWICE (0.4
    # and even 0.12 cycled gas/brake in the field) — so the loop is
    # INTEGRAL-dominant: accel_fb_i (1/s) integrates the LPF'd error,
    # which rejects the steady want-vs-act accel gap the charts showed
    # (+0.8 wanted, +0.1 delivered: pedal maps measured at one speed/
    # gear under-deliver in others) while staying blind to jitter and
    # phase-safe against the delay. accel_fb_p stays available but
    # defaults to 0.
    accel_fb_p: float = 0.0
    accel_fb_i: float = 0.4
    accel_fb_clip: float = 0.8
    # Hard clamp on the commanded acceleration — openpilot's ISO
    # comfort limits (ACCEL_MAX/ACCEL_MIN). AEB is exempt.
    accel_cmd_max_mps2: float = 2.0
    accel_cmd_min_mps2: float = -3.5
    # Longitudinal jerk limits (m/s^3): rate-limit the accel command so
    # pedal transitions are deliberate, not stabs — this is what gives
    # openpilot its "weighted" long feel. Asymmetric: releasing toward
    # positive (brake off / throttle build) may move faster than
    # ramping the brake in, so post-AEB recovery isn't sluggish.
    accel_jerk_down_mps3: float = 2.5
    accel_jerk_up_mps3: float = 4.0
    # First-order smoothing on the accel command — the exact mechanism
    # openpilot uses (modeld LONG_SMOOTH_SECONDS = 0.3): the plan's
    # frame-to-frame accel fluctuations get filtered before actuation,
    # so the brake pedal is heavy and deliberate instead of twitchy.
    # 0 disables. AEB bypasses it.
    long_smooth_s: float = 0.3
    # P term on velocity error (m/s^2 commanded per m/s of error). Keeps
    # the loop tracking the plan's velocity when the FF accel alone
    # under/overshoots.
    speed_p_gain: float = 0.5
    # Integral trim on velocity error. A P-only loop droops: holding
    # speed against drag needs constant throttle, which P can only
    # produce from a standing error — measured ~4 mph short of the cap
    # at highway speed. The integrator supplies the drag feedforward.
    # Gated (no windup while saturated / braking / far from target)
    # and leaky, mirroring the wheel-trim design.
    speed_i_gain: float = 0.1
    speed_i_clip: float = 0.8       # max m/s^2 the integrator may add
    speed_i_leak_s: float = 30.0
    # Symmetric deadband around 0 m/s^2 — avoids the actuator hunting
    # between throttle and brake when the desired accel is near zero
    # (cruise on flat road). Below this the controller commands 0/0.
    accel_deadband_mps2: float = 0.10
    # Hard cap on ego speed (m/s). The plan's velocity target is clamped
    # to this before computing accel, and any positive accel command is
    # zeroed once we're at/above the cap. Set high (e.g. 50) to
    # effectively disable. Default 25 m/s ≈ 90 km/h ≈ 56 mph.
    max_speed_mps: float = 25.0
    # Maximum lateral acceleration (m/s^2) allowed through upcoming
    # corners. We scan the plan's curvature over the next few seconds,
    # compute the minimum speed that keeps lateral g below this, and
    # clamp v_target to it. This is what makes the truck actually
    # brake hard for tight bends instead of waiting for the model's
    # (often conservative) planned velocity to drop. 2-3 m/s^2 feels
    # comfortable, 4-5 is firm, 6+ is aggressive. Set 0 to disable.
    # MUST NOT exceed lat_accel_max_mps2 (the ISO clamp on commanded
    # curvature): if long carries more speed into a bend than lateral
    # is allowed to use, the car understeers off the curve. Matched.
    max_lat_accel_mps2: float = 3.5
    # How far ahead in the plan to scan for the tightest upcoming
    # corner. Should be ≥ long_anticipation_s. Default 6 s covers
    # ~150 m at 25 m/s and ~200 m at 33 m/s — long enough to register
    # a hard bend at race speed in time to brake into it. The plan
    # itself spans 10 s (T_IDXS max), so anything up to 10 s is
    # readable; longer than that just clamps. Bumped from 3 s after
    # the hood-cam-on-AC case where the model registered curves but
    # the scanner truncated before reaching them.
    corner_scan_horizon_s: float = 6.0

    # ----- ACC / lead following -----
    # The model emits 3 lead-vehicle hypotheses with (x, y, v, a) at
    # 6 future timesteps each, plus a per-lead probability. When the
    # most-confident lead's probability exceeds `lead_min_prob`, we
    # constrain v_target with a time-headway controller (this matches
    # openpilot's `longitudinal_planner.py` + `longitudinal_mpc_lib`
    # in concept, though we use a simpler closed-form law instead of
    # an MPC).
    #
    # The constraint:
    #     desired_dist = TR * v_ego + min_gap
    #     gap_err      = lead_x - desired_dist          (+ = too far)
    #     v_lead_set   = lead_v + lead_gap_p_gain·gap_err
    #     v_target     = min(v_target, v_lead_set)
    #
    # So at the desired headway we match the lead's speed; closer than
    # that we slow further (v_set drops below v_lead); farther we let
    # the upstream v_target stand. The gain is small on purpose — the
    # rest of the longitudinal loop (FF accel from a_target + speed P)
    # closes out the residual error.
    #
    # AEB-ish: if time-to-collision drops below `lead_ttc_brake_s`
    # we override a_cmd directly to a hard decel, regardless of the
    # plan. Below this TTC the closed-form follower can't keep up.
    lead_follow_enabled: bool = True
    # Confident leads only. 0.5 let phantom leads (shadows, signs,
    # overpasses) trigger ACC/AEB braking on an empty road; 0.7 ignores
    # those. Lower it in the tuner if it ever misses a real lead.
    lead_min_prob: float = 0.7
    lead_time_headway_s: float = 1.5      # TR — comfort default
    lead_min_gap_m: float = 5.0           # bumper-to-bumper minimum
    lead_gap_p_gain: float = 0.3          # closure-rate response
    lead_ttc_brake_s: float = 2.0         # below this TTC, override to max decel

    @classmethod
    def load(cls, game: str | None = None,
             path: Path | None = None) -> "ControllerConfig":
        """Load this game's controller config. Falls back to the legacy
        single-file `controller.json` if a per-game file doesn't exist
        yet (smooth migration). Pass `path` to override entirely."""
        if path is None:
            path = load_with_fallback("controller", game)
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            cfg = cls(**{k: v for k, v in data.items()
                         if k in cls.__dataclass_fields__})
        else:
            cfg = cls()
        # No actuator-delay floor for sim use: the game applies steering
        # almost instantly, so lookahead_s can sit near 0 ("snap to now"),
        # which is what the user wants here. Only guard against a negative
        # value, which would break the lag-curvature math.
        if cfg.lookahead_s < 0.0:
            cfg.lookahead_s = 0.0
        return cfg

    def save(self, game: str | None = None,
             path: Path | None = None) -> None:
        if path is None:
            path = state_path("controller", game)
        path.write_text(json.dumps(asdict(self), indent=2))


class LateralController:
    def __init__(self, cfg: ControllerConfig | None = None,
                 live_params: LiveParams | None = None) -> None:
        self.cfg = cfg or ControllerConfig()
        self.live_params = live_params or LiveParams()
        self.last_curvature = 0.0
        self.last_target_wheel = 0.0
        self.last_axis = 0.0
        self.last_axis_target = 0.0
        self.last_in_lane_change = False
        self.last_authority = 1.0
        # Previous-frame desired curvature (pre-authority, pre-scale).
        # Comma's rate-limit clamps THIS frame's request to a small
        # delta from this value — that's what stops a sudden plan jump
        # from translating to a sudden steer-in. Carried across frames.
        self.last_desired_k_raw = 0.0
        # Closed-loop trim state. `axis_trim_state` is the integrator
        # output (clipped to ±cfg.wheel_trim_clip); `lpf_wheel_error`
        # is the heavy LPF on the wheel-angle error that feeds it.
        # Both reset to zero on engage/disengage transitions.
        self.axis_trim_state = 0.0
        self.lpf_wheel_error = 0.0
        # Last-frame trim freeze reason — for HUD diagnostics. Empty
        # string when the integrator was running.
        self.last_trim_frozen_reason = ""
        # Emulated-EPS smoothing state (see cfg.steer_smooth_s).
        self._axis_smooth = 0.0

    def reset(self) -> None:
        """Drop derived state. Call on disengage/re-engage transitions
        so nothing carries across handoffs."""
        self.last_curvature = 0.0
        self.last_target_wheel = 0.0
        self.last_axis = 0.0
        self.last_axis_target = 0.0
        self.last_in_lane_change = False
        self.last_authority = 1.0
        self.last_desired_k_raw = 0.0
        self.axis_trim_state = 0.0
        self.lpf_wheel_error = 0.0
        self.last_trim_frozen_reason = ""
        self._axis_smooth = 0.0

    def compute(self, decoded: Decoded, v_ego: float,
                actual_wheel_angle: float | None = None,
                lane_change_command_active: bool = False,
                dt: float = 0.05) -> float:
        """Plan -> gamepad axis. `actual_wheel_angle` (rad) feeds the
        closed-loop trim integrator; pass None to disable feedback
        for this frame. `dt` (s) is used by the integrator + leak —
        defaults to 0.05 s (20 Hz) for callers that don't measure it."""
        cfg = self.cfg
        if v_ego < cfg.min_speed:
            self.last_curvature = 0.0
            self.last_target_wheel = 0.0
            self.last_axis = 0.0
            self.last_axis_target = 0.0
            self.last_in_lane_change = False
            self.last_authority = 0.0
            self._axis_smooth = 0.0
            return 0.0

        # Plan-following via openpilot's `get_lag_adjusted_curvature`:
        # lag-corrected pure-pursuit + rate-limited so a step jump in
        # the model's plan can't translate into a step jump in steer
        # output. The rate-limit is what prevents "the AI steers in
        # too early" — without it, the moment the model first sees
        # an upcoming corner the commanded curvature snaps to its
        # target value; with it, the command ramps in over hundreds
        # of ms even if the plan jumps. Comma's rate is speed-
        # interpolated and gives ~850 ms turn-in at highway speed,
        # which is what feels natural in their own car.
        k_raw = desired_curvature_lag_adjusted(
            decoded.plan, v_ego,
            steer_actuator_delay=cfg.lookahead_s,
            last_desired_curvature=self.last_desired_k_raw,
            extra_buffer_s=cfg.curvature_anticipation_s,
            lat_jerk_max_mps3=cfg.lat_jerk_max_mps3,
            lat_accel_max_mps2=cfg.lat_accel_max_mps2,
        )
        self.last_desired_k_raw = k_raw
        k_total = k_raw

        # Detect lane-change state for the steer-authority boost. Reads
        # the model's own desire_state plus the user-command flag —
        # `lane_change_command_active` catches the initial frames
        # before the model's output reflects the user-pressed A/D.
        lc_prob = float(decoded.desire_state[3]) + float(decoded.desire_state[4])
        in_lane_change = (lc_prob > cfg.lane_change_desire_threshold
                          or lane_change_command_active)
        self.last_in_lane_change = in_lane_change

        # Authority — flat multiplier on the commanded curvature.
        # openpilot has no such multiplier; we keep a flat one as a
        # project-specific knob. Lane changes use the model's own
        # desire pulse + sustained desire input; no separate gain.
        authority = cfg.steer_authority
        k_total *= authority

        self.last_curvature = k_total
        self.last_authority = authority

        target_wheel = math.atan(k_total * cfg.wheelbase_m)
        self.last_target_wheel = target_wheel

        # FF axis from LiveParams inversion. v_ego enters the speed-
        # stiffness term so the inverse demands more axis at speed.
        axis_ff = self.live_params.axis_for_wheel_angle(
            target_wheel, v_ego=v_ego)

        # Closed-loop steady-state trim. Corrects drift the FF stack
        # can't see (rack-fit residual, model plan offset). The LPF
        # runs unconditionally when telemetry is available so the
        # filter is fresh when the gate opens; the integrator only
        # runs while every transient regime is clear (see docstring
        # in the ControllerConfig fields above for the rationale).
        if cfg.wheel_trim_enabled and actual_wheel_angle is not None:
            wheel_error = target_wheel - float(actual_wheel_angle)
            self.lpf_wheel_error = (
                (1.0 - cfg.wheel_trim_error_lpf_alpha) * self.lpf_wheel_error
                + cfg.wheel_trim_error_lpf_alpha * wheel_error)

            # Gate freeze reasons (first match wins; "" = running).
            # Each maps to a transient regime where wheel-error is
            # dominated by lag, not by the steady-state offset the
            # trim is meant to correct.
            intervened_recently = (
                time.monotonic() - self.live_params.last_intervened_ts < 1.0)
            if v_ego < 8.0:
                freeze_reason = "slow"
            elif abs(target_wheel) > 0.05:
                freeze_reason = "corner"
            elif self.last_in_lane_change:
                freeze_reason = "lc"
            elif abs(axis_ff) > 0.85:
                freeze_reason = "sat"
            elif self.live_params._consecutive_bad_fit > 10:
                freeze_reason = "rls-bad"
            elif not self.live_params.trusted():
                freeze_reason = "cold"
            elif intervened_recently:
                freeze_reason = "user"
            else:
                freeze_reason = ""

            if not freeze_reason:
                self.axis_trim_state += (
                    dt * cfg.wheel_trim_gain * self.lpf_wheel_error)
                # Exponential leak — forgets stale trim over
                # wheel_trim_leak_s if the gate stays open.
                self.axis_trim_state *= math.exp(-dt / cfg.wheel_trim_leak_s)
                self.axis_trim_state = max(
                    -cfg.wheel_trim_clip,
                    min(cfg.wheel_trim_clip, self.axis_trim_state))
            self.last_trim_frozen_reason = freeze_reason
        elif not cfg.wheel_trim_enabled:
            # Decay trim to zero quickly when disabled so the tuner
            # toggle is responsive — doesn't leave a stuck trim
            # behind. Same form as the leak path.
            self.axis_trim_state *= math.exp(-dt / 1.0)
            self.last_trim_frozen_reason = "off"
            self.lpf_wheel_error = 0.0
        else:
            self.last_trim_frozen_reason = "no-wheel"

        # Sum FF + trim + manual bias, then clip. Trim and axis_bias
        # serve different roles — bias is a static user knob, trim is
        # the dynamic auto-knob. Both stack additively on top of FF.
        axis = axis_ff + self.axis_trim_state + cfg.axis_bias
        axis = max(-cfg.steer_max, min(cfg.steer_max, axis))
        self.last_axis_target = axis   # pre-EPS-filter (telemetry TGT %)
        # Emulated EPS: first-order low-pass on the final axis. The
        # direct game input has no actuator dynamics, so without this
        # every 20 Hz plan flicker reaches the wheel raw.
        if cfg.steer_smooth_s > 0:
            alpha = 1.0 - math.exp(-dt / cfg.steer_smooth_s)
            self._axis_smooth += alpha * (axis - self._axis_smooth)
            axis = self._axis_smooth
        else:
            self._axis_smooth = axis
        self.last_axis = axis
        return axis


class LongitudinalController:
    """Plan -> (throttle, brake) trigger axes.

    Feed-forward on the model's planned acceleration plus a small P term
    on the velocity gap at `long_lookahead_s` into the future. Commands
    in m/s^2 map linearly to throttle / brake via `max_accel_mps2` and
    `max_decel_mps2`. Negative commanded accel becomes brake; positive
    becomes throttle. A symmetric deadband prevents pedal hunting near
    zero.

    The plan tensor's columns 3 and 6 are vel_x and accel_x respectively
    (see `PlanField` in pilot/constants.py). At index 0 they're the
    current-frame estimate; later indices are the model's predicted
    trajectory.
    """

    _T_IDXS = np.asarray(T_IDXS, dtype=np.float32)

    def __init__(self, cfg: ControllerConfig | None = None) -> None:
        self.cfg = cfg or ControllerConfig()
        self.last_v_target = 0.0
        self.last_a_target = 0.0
        self.last_a_cmd = 0.0
        self._a_cmd_smooth = 0.0
        self._v_err_i = 0.0
        self._was_moving = False
        self._stop_brake = 0.0
        # accel-feedback state: measured accel LPF, previous v, and a
        # short history of commanded accel (setpoint ~0.3 s ago)
        self._v_prev: float | None = None
        self._a_meas = 0.0
        self._a_err_lpf = 0.0
        self._a_fb_i = 0.0
        from collections import deque as _dq
        self._a_hist = _dq(maxlen=6)
        self.last_a_fb = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0
        # Corner-anticipation diagnostics for the HUD.
        self.last_v_safe_corner = float("inf")  # inf = no corner near
        self.last_corner_t = 0.0
        self.last_corner_k = 0.0
        # Lead-following diagnostics for the HUD.
        self.last_lead_prob = 0.0
        self.last_lead_x = float("inf")        # inf = no lead engaged
        self.last_lead_v = 0.0
        self.last_lead_v_target = float("inf")  # v_target imposed by lead
        self.last_lead_ttc = float("inf")
        self.last_aeb = False

    def reset(self) -> None:
        self.last_v_target = 0.0
        self.last_a_target = 0.0
        self.last_a_cmd = 0.0
        self._a_cmd_smooth = 0.0
        self._v_err_i = 0.0
        self._was_moving = False
        self._stop_brake = 0.0
        # accel-feedback state: measured accel LPF, previous v, and a
        # short history of commanded accel (setpoint ~0.3 s ago)
        self._v_prev: float | None = None
        self._a_meas = 0.0
        self._a_err_lpf = 0.0
        self._a_fb_i = 0.0
        from collections import deque as _dq
        self._a_hist = _dq(maxlen=6)
        self.last_a_fb = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_v_safe_corner = float("inf")
        self.last_corner_t = 0.0
        self.last_corner_k = 0.0
        self.last_lead_prob = 0.0
        self.last_lead_x = float("inf")
        self.last_lead_v = 0.0
        self.last_lead_v_target = float("inf")
        self.last_lead_ttc = float("inf")
        self.last_aeb = False

    def compute(self, decoded: Decoded, v_ego: float,
                mode: str = "exp") -> tuple[float, float]:
        """mode='exp': end-to-end — follow the model's own planned
        velocity/acceleration (slows for intersections, scene-dependent
        stops). mode='chill': classic ACC — cruise at max_speed_mps;
        the model's plan is used only for corner anticipation, and the
        vision leads for following/AEB."""
        cfg = self.cfg

        v_plan = decoded.plan[:, 3]   # vel_x over the 33 plan timesteps
        a_plan = decoded.plan[:, 6]   # accel_x
        yaw_rate_plan = decoded.plan[:, 14]  # yaw rate (rad/s) along plan

        # Effective lookahead = actuator delay (cfg.lookahead_s, the
        # static steerActuatorDelay) + the anticipation buffer for
        # seeing corners coming.
        t = max(0.0, float(cfg.lookahead_s) + float(cfg.long_anticipation_s))
        if mode == "chill":
            v_target = float(cfg.max_speed_mps)
            a_target = 0.0
        else:
            v_target = float(np.interp(t, self._T_IDXS, v_plan))
            a_target = float(np.interp(t, self._T_IDXS, a_plan))

        # Corner anticipation: find the tightest upcoming planned
        # curvature within `corner_scan_horizon_s` and override v_target
        # if its lateral accel would exceed `max_lat_accel_mps2`. This
        # is what makes us brake hard for bends — without it we just
        # follow the model's planned velocity, which is often too
        # conservative on its braking to make a sharp game corner.
        v_safe_corner = float("inf")
        corner_t = 0.0
        corner_k = 0.0
        if cfg.max_lat_accel_mps2 > 0.0:
            scan_horizon = max(t, float(cfg.corner_scan_horizon_s))
            mask = (self._T_IDXS >= 0.0) & (self._T_IDXS <= scan_horizon)
            ts_scan = self._T_IDXS[mask]
            vs_scan = np.maximum(v_plan[mask], 0.5)         # avoid /0
            ks_scan = np.abs(yaw_rate_plan[mask]) / vs_scan  # |κ| = |ω|/v
            # Safe speed at each timestep that keeps |a_lat| = v^2·|κ|
            # ≤ max_lat_accel. sqrt(a_max/|κ|), clipped where |κ| is
            # near zero (straight road → no limit).
            ks_safe = np.maximum(ks_scan, 1e-6)
            v_safe_at_t = np.sqrt(cfg.max_lat_accel_mps2 / ks_safe)
            if v_safe_at_t.size > 0:
                idx_min = int(np.argmin(v_safe_at_t))
                v_safe_corner = float(v_safe_at_t[idx_min])
                corner_t = float(ts_scan[idx_min])
                corner_k = float(ks_scan[idx_min])
                if v_safe_corner < v_target:
                    v_target = v_safe_corner

        # ACC / lead following. The most-confident lead (index 0 in
        # the decoded.leads tensor — already sorted by prob in MHP
        # selection order) constrains v_target via a time-headway
        # rule. Below `lead_ttc_brake_s` time-to-collision we engage
        # AEB and override a_cmd directly to max decel.
        lead_v_target = float("inf")
        lead_ttc = float("inf")
        aeb = False
        lead_prob = float(decoded.lead_prob[0]) if decoded.lead_prob.size > 0 else 0.0
        lead_x = float("inf")
        lead_v = 0.0
        if (cfg.lead_follow_enabled
                and decoded.leads.size > 0
                and lead_prob >= cfg.lead_min_prob):
            # leads[i, t, k] — k = 0:x, 1:y, 2:v, 3:a. Use the t=0
            # (current-frame) sample for the most-confident lead.
            lead_x = float(decoded.leads[0, 0, 0])
            lead_v = float(decoded.leads[0, 0, 2])
            # Closing rate (positive = closing). v_ego - lead_v.
            closure = v_ego - lead_v
            # Time-to-collision: only meaningful when closing.
            if closure > 0.1 and lead_x > 0.0:
                lead_ttc = lead_x / closure
            # Desired bumper-to-bumper distance.
            desired_dist = (cfg.lead_time_headway_s * max(v_ego, 0.0)
                            + cfg.lead_min_gap_m)
            gap_err = lead_x - desired_dist
            # Setpoint: at the desired gap, match lead speed. Inside
            # it, slow further; outside it, the controller doesn't
            # clamp at all (returns inf, never wins the min below).
            lead_v_target = lead_v + cfg.lead_gap_p_gain * gap_err
            # Never command negative speed via the lead law — the
            # follower can stop but not reverse.
            lead_v_target = max(0.0, lead_v_target)
            if lead_v_target < v_target:
                v_target = lead_v_target
            # AEB trigger.
            if lead_ttc < cfg.lead_ttc_brake_s and closure > 0.1:
                aeb = True
        self.last_lead_prob = lead_prob
        self.last_lead_x = lead_x
        self.last_lead_v = lead_v
        self.last_lead_v_target = lead_v_target
        self.last_lead_ttc = lead_ttc
        self.last_aeb = aeb

        # Cap the velocity target. If the model wants to go faster than
        # `max_speed_mps`, we don't follow — the P term then naturally
        # brakes (or eases off throttle) as v_ego approaches the cap.
        v_target = min(v_target, cfg.max_speed_mps)
        v_err = v_target - v_ego
        a_cmd = a_target + cfg.speed_p_gain * v_err

        # Integral trim (drag feedforward). Integrate only in the
        # steady regime: rolling, close to target, command not
        # saturated, no AEB — otherwise it winds up during launches
        # and braking and dumps it later.
        dt = 0.05
        if (v_ego > 3.0 and abs(v_err) < 5.0 and not aeb
                and cfg.accel_cmd_min_mps2 < a_cmd < cfg.accel_cmd_max_mps2):
            self._v_err_i += cfg.speed_i_gain * v_err * dt
            self._v_err_i *= math.exp(-dt / max(cfg.speed_i_leak_s, 1e-3))
            self._v_err_i = float(np.clip(self._v_err_i,
                                          -cfg.speed_i_clip, cfg.speed_i_clip))
        a_cmd += self._v_err_i
        # And belt-and-braces: never command positive accel once we're
        # at/above the cap, even if the plan's a_target was high.
        if v_ego >= cfg.max_speed_mps and a_cmd > 0:
            a_cmd = 0.0

        # ISO comfort clamp + smoothing + jerk rate-limit (openpilot-
        # style). The e2e plan can request violent accelerations; clamp
        # the command, low-pass it, and ramp it so pedals move
        # deliberately. dt is the 20 Hz frame (DT_MDL) — callers don't
        # measure it for us.
        a_cmd = float(np.clip(a_cmd, cfg.accel_cmd_min_mps2,
                              cfg.accel_cmd_max_mps2))
        if cfg.long_smooth_s > 0:
            alpha = 1.0 - math.exp(-dt / cfg.long_smooth_s)
            self._a_cmd_smooth += alpha * (a_cmd - self._a_cmd_smooth)
            a_cmd = self._a_cmd_smooth
        else:
            self._a_cmd_smooth = a_cmd
        a_cmd = float(np.clip(
            a_cmd,
            self.last_a_cmd - cfg.accel_jerk_down_mps3 * dt,
            self.last_a_cmd + cfg.accel_jerk_up_mps3 * dt))

        # AEB override — full brake, exempt from the comfort limits.
        if aeb:
            a_cmd = -cfg.max_decel_mps2
            self._a_cmd_smooth = a_cmd

        # --- measured-accel feedback (openpilot's a_ego loop) ---
        # Compare what we commanded ~0.3 s ago against what the car
        # actually did (LPF'd dv/dt) and correct the pedal command.
        # This is how openpilot adapts long to any car in real time —
        # feedback, not learned constants. Setpoint (pre-correction)
        # is what feeds the jerk limiter and history, so the loop
        # can't chase its own corrections.
        if self._v_prev is not None:
            a_raw = (v_ego - self._v_prev) / dt
            self._a_meas += 0.15 * (float(np.clip(a_raw, -12, 12))
                                    - self._a_meas)
        self._v_prev = v_ego
        setpoint = a_cmd
        self._a_hist.append(setpoint)
        gate = (not aeb and v_ego > 2.0 and self._stop_brake == 0.0
                and cfg.accel_cmd_min_mps2 < setpoint < cfg.accel_cmd_max_mps2
                and len(self._a_hist) == self._a_hist.maxlen)
        if gate:
            err = float(self._a_hist[0]) - self._a_meas
            self._a_err_lpf += 0.15 * (err - self._a_err_lpf)
            if cfg.accel_fb_i > 0:
                # INTEGRAL-dominant by design: an integrator only
                # accumulates PERSISTENT error (the steady want/act gap
                # in the charts), is blind to frame-to-frame jitter,
                # and at ki=0.4 crosses over at ~0.06 Hz — stable with
                # ~67 deg phase margin against this loop's ~1 s delay.
                # The P design oscillated here twice; don't raise
                # accel_fb_p above ~0.1 without run-log evidence.
                self._a_fb_i += cfg.accel_fb_i * self._a_err_lpf * dt
                self._a_fb_i *= math.exp(-dt / 60.0)     # forget slowly
                self._a_fb_i = float(np.clip(
                    self._a_fb_i, -cfg.accel_fb_clip, cfg.accel_fb_clip))
        else:
            self._a_err_lpf *= 0.9
        fb = float(np.clip(
            cfg.accel_fb_p * self._a_err_lpf + self._a_fb_i,
            -cfg.accel_fb_clip, cfg.accel_fb_clip))
        self.last_a_fb = fb
        a_cmd = setpoint + fb

        # Pedal mapping. Calibrated affine inversion when CAL has
        # measured the car (note the coast gap: zero pedal already
        # gives -pedal_thr_off from drag, the lightest brake touch
        # gives -pedal_brk_off from bite; commands between coast).
        # Legacy scale-only mapping otherwise, with the anti-hunt
        # deadband on the BRAKE side only — cruising needs small
        # sustained throttle, zeroing it caused a droop/limit-cycle.
        if cfg.pedal_thr_map and cfg.pedal_brk_map:
            ta = [p[0] for p in cfg.pedal_thr_map]
            tp = [p[1] for p in cfg.pedal_thr_map]
            bd = [p[0] for p in cfg.pedal_brk_map]
            bp = [p[1] for p in cfg.pedal_brk_map]
            # Split each table into (pedal-component, pedal) with the
            # coast anchor speed-scaled: the engine/brake torque
            # components transfer across speeds, the drag part doesn't.
            vfrac = float(np.clip(v_ego / max(cfg.pedal_cal_v, 1.0),
                                  0.15, 1.0))
            drag_eff = -ta[0] * vfrac        # ta[0] is coast accel (<0)
            tx = [x - ta[0] for x in ta]     # engine component, >= 0
            bx = [x - bd[0] for x in bd]     # brake component, >= 0
            coast_decel_eff = bd[0] * vfrac
            self._was_moving = self._was_moving or v_ego > 2.0
            stopping = (self._was_moving and v_ego < cfg.stop_hold_speed
                        and a_cmd < 0.2 and v_target < 1.0)
            if not stopping:
                self._stop_brake = 0.0
            if stopping:
                # Came to a stop while driving: ramp the brake up and
                # hold against automatic creep (openpilot LongControl
                # 'stopping' ramps to stopAccel the same way). Releases
                # the moment the plan wants speed again (v_target).
                # NOT armed before the first movement — holding at
                # engage-from-standstill deadlocks: the model sees a
                # parked scene and plans zero forever (creep is what
                # seeds a standstill launch).
                ramp = cfg.stop_hold_brake * dt / max(cfg.stop_brake_ramp_s,
                                                      1e-3)
                self._stop_brake = min(cfg.stop_hold_brake,
                                       self._stop_brake + ramp)
                throttle, brake = 0.0, self._stop_brake
            elif a_cmd + drag_eff >= 0.0:
                throttle = float(np.clip(
                    np.interp(a_cmd + drag_eff, tx, tp), 0.0, 1.0))
                brake = 0.0
            elif -a_cmd <= coast_decel_eff + cfg.accel_deadband_mps2:
                throttle, brake = 0.0, 0.0   # coasting already does it
            else:
                throttle = 0.0
                brake = float(np.clip(
                    np.interp(-a_cmd - coast_decel_eff, bx, bp), 0.0, 1.0))
        elif a_cmd > 0:
            throttle = min(1.0, a_cmd / max(cfg.max_accel_mps2, 1e-3))
            brake = 0.0
        elif a_cmd > -cfg.accel_deadband_mps2:
            throttle, brake = 0.0, 0.0
        else:
            throttle = 0.0
            brake = min(1.0, -a_cmd / max(cfg.max_decel_mps2, 1e-3))

        self.last_v_target = v_target
        self.last_a_target = a_target
        self.last_a_cmd = setpoint   # pre-feedback, keeps jerk/history sane
        self.last_throttle = throttle
        self.last_brake = brake
        self.last_v_safe_corner = v_safe_corner
        self.last_corner_t = corner_t
        self.last_corner_k = corner_k
        return throttle, brake
