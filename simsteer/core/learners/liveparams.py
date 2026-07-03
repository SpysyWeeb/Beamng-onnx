"""Online identification of the truck's steering response, with the
speed-stiffness correction that openpilot's `paramsd` uses for real
vehicles.

The model is comma's bicycle model with `stiffness_factor`, inverted
for our purpose (we have wheel angle from telemetry — what we need is
the gamepad axis that produces a target wheel angle). For a vehicle
where wheel angle vs rack input falls off with speed (variable-ratio
steering, ETS2's stability assist, real-car understeer), the inverse
relationship is:

    axis  ≈  a · wheel  +  b · wheel · v²  +  c

  - a (axis per rad of wheel) is the low-speed linear coefficient.
  - b (axis per rad·m²/s²) is the speed-stiffness coefficient — it's
    what makes the controller automatically command MORE axis at
    highway speed for the same plan curvature. Analogous to comma's
    `stiffnessFactor` in `selfdrive/locationd/paramsd.py`.
  - c is the axis offset bias.

The form is linear in (a, b, c), so standard RLS still fits it. We
observe `gameSteer_lagged` (ETS2's post-smoothing rack input, which
is what actually drives the wheels) and regress against
`H = [wheel, wheel·v², 1]`.

Why fit the INVERSE rather than the forward `wheel = scale · axis`:

  - At runtime we need axis given target_wheel + v_ego — that's what
    the controller commands. Fitting the inverse means no inversion
    at runtime: plug target_wheel + v² straight in.
  - With our `wheel` and `gameSteer` both coming from low-noise SCS
    telemetry, the noise-model asymmetry between the two directions
    is negligible.

The old `scale`/`offset` are still exposed as derived properties for
HUD readability (`scale ≈ 1/a` at zero speed). Existing
`liveparams_*.json` files written in the old 2-parameter format are
auto-migrated on load.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simsteer.paths import load_with_fallback, state_path

CONFIG_PATH = state_path("liveparams")

# How many frames the truck's wheel takes to respond to an input change
# (ETS2 input filter + steering rack). ~5 frames at 20 Hz = 250 ms.
LAG_FRAMES_DEFAULT = 5

# Forgetting factor — closer to 1 = slower adaptation. See the
# adaptive_forget() docstring.
FORGET = 0.999
FORGET_FLOOR = 0.99
FORGET_RESPONSIVENESS = 0.009

# Watchdog: if abs(innovation) stays above this for `BAD_FIT_PATIENCE`
# consecutive accepted samples, the RLS state is persistently failing
# to predict — covariance gets reset to its initial values so the next
# few samples carry outsized weight.
#
# Units: axis (since the observation we regress is `gameSteer`). 0.10
# means predictions are wrong by >10% of full deflection.
BAD_FIT_INNOV_THRESHOLD = 0.10
BAD_FIT_PATIENCE = 30            # ~1.5 s @ 20 Hz

# Trouble detector: speed collapse implies crash / off-road; skip
# updates until speed recovers.
TROUBLE_SPEED_DROP_FRAC = 0.5
TROUBLE_MIN_PEAK_MPS = 3.0
TROUBLE_WINDOW_S = 5.0

# Intervention threshold: when the observed gameSteer disagrees with
# what we commanded by more than this, the human is adding (or
# subtracting) input on top of the AI. Used to be a hard rejection
# gate; now it routes the sample into the intervention-learning path
# (see `_intervention_learn`) instead of the normal RLS update.
USER_ASSIST_DELTA = 0.15

# Intervention learning rate. When the user is intervening, we don't
# run RLS (the noise / lag structure of mid-correction samples breaks
# the regression's assumptions). Instead we compute the `a` that
# would have made `axis = a·wheel + b·wheel·v² + c` equal the
# OBSERVED gameSteer at this frame's wheel, and blend toward it at
# this rate. The blend is small — 0.02 over many frames lets a
# sustained help/fight reshape `a` over a few seconds, while a single
# spurious sample shifts it by <2%.
INTERVENTION_LEARN_RATE = 0.02
# Minimum wheel magnitude required to invert the model into an implied
# `a`. Below this the division `axis/wheel` is dominated by noise.
INTERVENTION_MIN_WHEEL = 0.005

# Saturation gate: when the controller is commanding near the rail,
# the observation reflects mechanical/slip dynamics, not the linear
# inverse-rack mapping we're fitting. Skip those samples — they are
# the well-known under-steer feedback-loop trigger.
SATURATION_AXIS = 0.9

# Per-game gate profiles. ETS2 was working well with the original
# strict values; Forza / AC need looser ones because:
#   - they synthesize wheel angle from yaw rate via bicycle model
#     (instead of reading it directly), so the (axis, wheel) pair has
#     less independent variation;
#   - Forza's signed-byte steer field quantizes to ~0.008 per step;
#   - the player keeps the stick swinging more in racy games — there's
#     no ETS2-style rack smoothing forcing input quiescence.
#
# `_gates_for(game)` below picks the right profile per LiveParams
# instance based on `self.game`.

# ETS2 defaults — the values that worked. KEEP THESE; do not touch
# without per-game opt-in. `lag_frames` is also per-game because the
# rack delay differs: ETS2 smooths input ~250 ms (5 frames @ 20 Hz),
# Forza / AC respond ~50 ms (1 frame), and pairing a too-old axis
# sample with the current wheel reading turns most "steering bursts"
# into a string of rejected-because-axis_lagged-is-near-zero events.
_GATES_DEFAULT = {
    "min_speed_mps": 5.0,
    "min_axis": 0.005,
    "min_wheel": 0.0,         # ETS2 already filters via min_axis
    "max_axis_spread_for_fit": 0.10,
    "max_innovation_cold": 0.60,
    "c_abs_max": 0.10,
    "lag_frames": 5,
    # Initial state-covariance diagonal — bigger = faster initial
    # convergence at the cost of more sample-to-sample swing. ETS2
    # was tuned with these and worked; do not touch without need.
    "P_init": (0.1, 1e-4, 1e-3),
    # Process-noise covariance added each step (Kalman Q). Matches
    # paramsd's role: lets the state slowly drift even when nothing
    # new is being observed, AND keeps the covariance from shrinking
    # to zero (which would freeze the fit). Small positive values
    # only — too big and a glitchy sample yanks the state.
    "Q_diag": (1e-7, 1e-10, 1e-6),
}

# Forza / AC overrides. We can't use min_axis to filter straight-
# driving samples (the signed-byte / atan path produces sub-noise
# axis values that we want — they contain real micro-corrections),
# so instead we filter on the synthesized WHEEL: if the wheel
# angle is below ~0.3° (0.005 rad), the sample carries no info
# about the rack response regardless of axis, and letting it
# through to RLS develops covariance off-diagonals that wash out
# the good cornering data within seconds of straight driving.
#
# `P_init` is much bigger so cold-start converges quickly from the
# default a=10 toward Forza's true a (~3). `Q_diag` is bigger too
# so the filter doesn't lose responsiveness over long sessions.
_GATES_SYNTH_WHEEL = {
    "min_speed_mps": 3.0,
    "min_axis": 0.0,
    "min_wheel": 0.005,       # ~0.3° — reject straight-driving samples
    "max_axis_spread_for_fit": 0.30,
    "max_innovation_cold": 2.0,
    "c_abs_max": 0.02,
    "lag_frames": 1,
    "P_init": (10.0, 1e-3, 1e-2),
    "Q_diag": (1e-5, 1e-8, 1e-5),
}


def _gates_for(game: str | None) -> dict:
    """Return the gate profile for a game. ETS2 (and unknown) → strict;
    Forza / AC / BeamNG (synthesized wheel angles) → loose."""
    if game in ("forza", "ac", "beamng"):
        return dict(_GATES_SYNTH_WHEEL)
    return dict(_GATES_DEFAULT)


# Back-compat shims — old code reads these as module constants. They
# now reflect the DEFAULT (ETS2) profile. The per-instance values are
# read off `self._gates` inside `update()`.
MIN_SPEED_MPS = _GATES_DEFAULT["min_speed_mps"]
MIN_AXIS = _GATES_DEFAULT["min_axis"]
MAX_AXIS_SPREAD_FOR_FIT = _GATES_DEFAULT["max_axis_spread_for_fit"]

# Outlier gate: reject samples where the wheel angle is past plausible
# road driving. Spin-outs / wall-hits produce huge implied wheel
# angles that drag the fit if accepted.
#
# 0.3 rad ≈ 17°. Real on-road wheel angle (front wheel, not steering
# wheel) almost never exceeds this — most highway is <0.05 rad.
MAX_WHEEL_FOR_FIT = 0.3

# Innovation gates. Warm value is shared across games (it's a glitch
# filter once the fit is trusted). Cold value is per-game — ETS2's
# strict 0.60 worked there, Forza/AC need 2.0 because the initial
# `a=10` is far from their true rack ratio (~3) and 0.60 rejected
# every cold sample. See `_gates_for`.
MAX_INNOVATION_WARM = 0.30
MAX_INNOVATION_COLD = _GATES_DEFAULT["max_innovation_cold"]

# Bounds on the fitted parameters. The SIGN of `a` is whatever the
# game's axis-to-wheel relationship requires — some setups have a
# positive correlation (axis_right → wheel_right) and some negative
# (axis_right → wheel_left). LiveParams absorbs that sign so the
# controller and `lateral_sign` can remain on a fixed convention.
# Magnitude constraints only: |a| ∈ [A_ABS_MIN, A_ABS_MAX], etc.
#
# `b` shares the sign of `a` — the speed-stiffness term is always in
# the same direction as the base linear response. We enforce that
# post-RLS so the regression remains unconstrained.
# Lower floor on the EFFECTIVE |a| used at inversion time. Without
# this, RLS can pin a near zero (degenerate but mathematically valid)
# and `axis_for_wheel_angle` then outputs ~0 for any target wheel —
# i.e. the controller decides "no steering is best." The internal
# state is allowed to roam in [-A_ABS_MAX, +A_ABS_MAX] (so RLS can
# track sign flips and unusual cars), but at INVERSION TIME we clamp
# the magnitude so the controller always commands at least a usable
# axis. 0.5 ≈ scale of 2 rad/axis, comfortably small but non-degenerate.
A_ABS_MAX = 50.0
A_ABS_MIN_RUNTIME = 0.5
B_ABS_MAX = 0.10
# `c` is a small TRIM bias. Per-game profile decides the clamp —
# ETS2's 0.10 worked, Forza/AC need 0.02 (otherwise c absorbs the
# rack-gain mismatch instead of pushing it into `a`). See
# `_gates_for`. Module-level shim points at the ETS2 value.
C_ABS_MAX = _GATES_DEFAULT["c_abs_max"]

# Initial guess. a=10 ≈ 1 rad of wheel needs 10 units of axis at low
# speed (so effective scale ≈ 0.1 wheel/axis). b=0 lets RLS discover
# the speed term. c=0. Used as final fallback when no per-(game, device)
# seed exists in _INITIAL_SEEDS below.
A_INIT = 10.0
B_INIT = 0.0
C_INIT = 0.0

# Per-(game, device) cold-start seeds. Applied only when no persisted
# liveparams_*.json file exists yet — first launch ever for that
# (game, device) combo. The universal a=10 fallback is far from real
# game racks (typically 3-5), so the cold filter spends thousands of
# samples migrating before useful, AND a/c are collinear in near-
# straight driving so it can converge to a wrong local minimum. Per-
# game seeds get the filter close enough that first-session driving
# is already reasonable.
#
# Numbers below are best-guess starting points based on the rack
# model `axis = a·wheel + b·wheel·v² + c`:
#  - ETS2 gamepad: a≈3.5 (axis per rad at parking), b≈0.0015 — at
#    22 m/s gives effective ≈ 4.2 axis/rad (variable-ratio assist).
#  - ETS2 wheel (vJoy): a≈4.0, b≈0 — vJoy bypasses ETS2's gamepad
#    rack assist so the relationship is linear in speed.
#  - Forza / AC: a≈3.0, b≈0.0010 — synthesized wheel angle from yaw
#    rate, looser overall but similar magnitude.
# Adjust if your fits converge somewhere very different.
DEFAULT_SEED: tuple[float, float, float] = (A_INIT, B_INIT, C_INIT)
_INITIAL_SEEDS: dict[tuple[str | None, str | None], tuple[float, float, float]] = {
    ("ets2", "gamepad"): (3.5, 0.0015, 0.0),
    ("ets2", "wheel"):   (4.0, 0.0,    0.0),
    ("forza", None):     (3.0, 0.0010, 0.0),
    ("ac", None):        (3.0, 0.0010, 0.0),
    # BeamNG via input.event FILTER_DIRECT (see beamng/world.py):
    # steering=+1 -> RIGHT turn; in the model's native frame (y
    # right-positive) that's +wheel, so a is POSITIVE. Direct input is
    # linear and bypasses the speed limiter: 0.25 axis = 124 deg
    # steering wheel (495 deg lock) ~ 0.13 rad road wheel -> a ~ 2.0,
    # b ~ 0 (measured, tools/m3_sign_check.py + step probe).
    ("beamng", None):    (2.0,  0.0,   0.0),
    # No-tech.key screen mode drives a virtual G29 (simsteer/io/
    # vwheel.py). The G29's ~900 deg soft-lock maps to the car's
    # steering differently from FILTER_DIRECT, so this is a starting
    # guess only — screen mode learns the rack online while engaged
    # and the trim absorbs the residual.
    ("screen", None):    (2.0,  0.0,   0.0),
}


def _seed_for(game: str | None,
              device_kind: str | None) -> tuple[float, float, float]:
    """Resolve the cold-start seed for (game, device_kind). Falls back
    (game, device_kind) → (game, None) → DEFAULT_SEED so a game with no
    device-specific seed still gets a per-game one, and an unknown game
    falls all the way back to the universal default."""
    if (game, device_kind) in _INITIAL_SEEDS:
        return _INITIAL_SEEDS[(game, device_kind)]
    if (game, None) in _INITIAL_SEEDS:
        return _INITIAL_SEEDS[(game, None)]
    return DEFAULT_SEED


# Trust gates — either the lifetime persisted samples or the fresh
# session samples crossing the threshold is sufficient. Halved from
# 200 because LiveCalib's gates were rejecting heavily and the user
# was waiting 15+ min to engage; 100 samples is enough for a useable
# fit on highway driving with the active probe injecting variance.
TRUSTED_MIN_SAMPLES = 100

# Auto-freeze: once we've accumulated this many samples the fit is
# mature enough that further RLS updates are more likely to fit
# transient AI-confusion dynamics than the actual rack response.
FREEZE_AFTER_SAMPLES = 2000


@dataclass
class _Persisted:
    a: float = A_INIT
    b: float = B_INIT
    c: float = C_INIT
    samples: int = 0


class LiveParams:
    def __init__(self, lag_frames: int | None = None,
                 game: str | None = None,
                 device_kind: str | None = None) -> None:
        self.game = game
        # Per-device persistence: the rack response (and especially
        # the AXIS->WHEEL sign) differs between the ViGEm gamepad and
        # the vJoy wheel in the same game — ETS2 treats them as
        # separate input categories. We persist them separately so
        # switching devices reloads the correct fit instead of using
        # a value that was right for the OTHER device.
        self.device_kind = device_kind
        # Per-game gate profile. ETS2 uses strict defaults (the values
        # that worked in testing); Forza/AC use loosened gates because
        # they synthesize wheel from yaw rate and the rack response
        # has different signal characteristics. `lag_frames` is also
        # part of the profile — Forza/AC have a fast rack and need
        # `lag_frames=1`, ETS2 needs the full 5 (= 250 ms smoothing).
        self._gates = _gates_for(game)
        if lag_frames is None:
            lag_frames = int(self._gates["lag_frames"])
        # Detect cold start (no persisted file) so the startup print
        # below can announce that a/b/c came from the seed table, not
        # a learned fit. Tuner / HUD read this flag to dim or annotate
        # the values.
        resolved = self._resolve_path(game=game, device_kind=device_kind)
        self._was_seeded = (resolved is None)
        loaded = self._load(game=game, device_kind=device_kind,
                            path=resolved)
        if self._was_seeded:
            print(f"liveparams [{device_kind or 'no device'}]: "
                  f"cold start — seeded for game={game!r}: "
                  f"a={loaded.a:.3f} b={loaded.b:.4f} c={loaded.c:+.4f}")
        # State vector x = [a, b, c]: axis = a·wheel + b·wheel·v² + c.
        self.x = np.array([loaded.a, loaded.b, loaded.c], dtype=np.float64)
        # Initial state-covariance diagonal (P0). Per-game — Forza/AC
        # use bigger values to converge from defaults faster, ETS2
        # uses the tighter values that worked there. The b prior is
        # always smaller than a's because b's regressor is wheel·v²
        # which can be ~50 at highway speed, so its natural Kalman
        # gain is correspondingly smaller.
        p_init = self._gates["P_init"]
        self.P = np.diag(p_init)
        # MATURE restore: a fit with 500+ samples behind it (CAL
        # system-ID installs 1000) is a measurement, not a guess.
        # Restoring it with the cold-start P told the filter the value
        # was nearly worthless, and one stretch of degenerate straight-
        # highway samples wandered a=2.0 down to 0.84 within minutes
        # of a relaunch (field: understeer, act-curvature compressed
        # vs want). Same tight P as the CAL install — Q keeps it
        # adapting, just at prior-respecting speed.
        if loaded.samples >= 500:
            self.P = np.diag((0.25, 1e-6, 1e-4))
        # Process noise (Kalman Q) — added each RLS step to keep the
        # covariance from collapsing to zero (which would freeze the
        # state) and to let the state slowly drift between samples,
        # exactly like openpilot's `paramsd` does. Values are small;
        # tuned so a single step of process noise can't yank the
        # state but cumulative drift over hours stays responsive.
        self._Q = np.diag(self._gates["Q_diag"])
        self.samples = loaded.samples
        self.session_samples = 0
        self.lag_frames = lag_frames
        # Buffers — lag_frames+1 deep so `[0]` is the value lag_frames
        # samples in the past once full.
        self._axis_buf: deque[float] = deque(maxlen=lag_frames + 1)
        self._commanded_buf: deque[float] = deque(maxlen=lag_frames + 1)
        self._v_buf: deque[float] = deque(maxlen=lag_frames + 1)
        self.last_innovation = 0.0
        self.last_predicted = 0.0
        self.last_actual = 0.0
        self._innov_ema = 0.0
        self._actual_ema = 0.0
        self._dirty = 0
        # Rejection counters for HUD diagnostics.
        self.rej_wheel = 0
        self.rej_innovation = 0
        self.rej_trouble = 0
        self.rej_assist = 0
        self.rej_saturated = 0
        self.rej_transient = 0       # mid-ramp — quiescence gate fired
        # Previously-silent gates that just `return`. Counting them
        # makes diagnosing "why isn't liveparams updating?" tractable
        # — especially for Forza/AC where wheel is synthesized and a
        # subtle gate can swallow every sample.
        self.rej_slow = 0            # v_ego below MIN_SPEED_MPS
        self.rej_no_wheel = 0        # wheel_angle_rad is None this frame
        self.rej_small_axis = 0      # |axis_lagged| below MIN_AXIS
        self.rej_small_wheel = 0     # |wheel| below profile min_wheel (Forza/AC)
        self.watchdog_resets = 0
        self.frozen = False
        self._speed_history: deque[float] = deque(
            maxlen=int(TROUBLE_WINDOW_S * 20))
        self._consecutive_bad_fit = 0
        # Trajectory ring buffer (time, a, b, c) — appended on every
        # accepted RLS update. Used by a future tuner plot tab; the
        # HUD itself reads `last_update_ts` and the trust_level
        # property for now.
        self.history: deque[tuple[float, float, float, float]] = deque(
            maxlen=600)
        self.last_update_ts: float = 0.0
        # Set whenever an intervention-learning sample is processed.
        # Decays back to 0.0 over ~1 s — read by the controller's trim
        # integrator to freeze itself while the human is fighting/helping
        # the wheel (the gate is `last_intervened_ts` within 1 s of
        # now). Lighter than a boolean flag because it lets the gate
        # auto-clear without a separate tick.
        self.last_intervened_ts: float = 0.0

    # ----- raw parameters -----

    @property
    def a_linear(self) -> float:
        """Linear coefficient: axis per radian of wheel at zero speed."""
        return float(self.x[0])

    @a_linear.setter
    def a_linear(self, value: float) -> None:
        v = float(value)
        sign = -1.0 if v < 0 else 1.0
        self.x[0] = sign * min(A_ABS_MAX, abs(v))

    @property
    def b_quad(self) -> float:
        """Speed-quadratic coefficient: extra axis per (rad·v²)."""
        return float(self.x[1])

    @b_quad.setter
    def b_quad(self, value: float) -> None:
        v = float(value)
        sign = -1.0 if v < 0 else 1.0
        self.x[1] = sign * min(B_ABS_MAX, abs(v))

    @property
    def c_bias(self) -> float:
        """Axis offset bias."""
        return float(self.x[2])

    @c_bias.setter
    def c_bias(self, value: float) -> None:
        c_max = self._gates["c_abs_max"]
        self.x[2] = max(-c_max, min(c_max, float(value)))

    # ----- HUD-friendly derived views -----

    @property
    def scale(self) -> float:
        """Wheel-per-axis at zero speed = 1/a. HUD-compatible alias —
        small values (e.g. ≈0.1) mean "needs lots of axis per radian
        of wheel," which is what users intuited from the old fit."""
        a = self.x[0]
        return float(1.0 / a) if abs(a) > 1e-6 else 0.0

    @property
    def offset(self) -> float:
        """Equivalent additive wheel offset = -c/a. HUD-compatible."""
        a = self.x[0]
        return float(-self.x[2] / a) if abs(a) > 1e-6 else 0.0

    def effective_scale_at(self, v_ego: float) -> float:
        """Wheel-per-axis at a given speed: 1 / (a + b·v²).
        Drops with speed when b > 0 — the whole point of fitting b."""
        denom = self.x[0] + self.x[1] * v_ego * v_ego
        return float(1.0 / denom) if abs(denom) > 1e-6 else 0.0

    @property
    def scale_std(self) -> float:
        """Std of `a` (the linear coefficient). Relative confidence
        read; not directly comparable across fits."""
        return float(np.sqrt(self.P[0, 0]))

    @property
    def innov_ema(self) -> float:
        """EMA of |innovation| over accepted RLS updates. Same units as
        the regression target (axis). HUD reads this as a quick health
        gauge — should stabilize near zero on a healthy fit."""
        return float(self._innov_ema)

    @property
    def innov_ratio(self) -> float:
        if self._actual_ema < 1e-4:
            return 0.0
        return self._innov_ema / self._actual_ema

    @property
    def auto_frozen(self) -> bool:
        # Sample-count auto-freeze is disabled — the rate-limit and
        # rejection gates handle the corruption-from-bad-samples
        # problem better than a hard sample-count cutoff did.
        return False

    @property
    def trust_level(self) -> str:
        """Coarse one-word status. The HUD colors itself by this:
            LOCKED      — user has manually frozen the fit
            RECOVERING  — watchdog about to fire (>10 consecutive bad)
            WARMING     — fewer than TRUSTED_MIN_SAMPLES accepted
            UNSTABLE    — innov_ratio > 0.30 — predictions consistently off
            HEALTHY     — innov_ratio < 0.05 — predictions essentially right
            SETTLED     — in between, fit is mature but not exceptional
        Cheap to compute, no allocations beyond the f-string."""
        if self.locked:
            return "LOCKED"
        if self._consecutive_bad_fit > 10:
            return "RECOVERING"
        if not self.trusted():
            return f"WARMING ({self.samples}/{TRUSTED_MIN_SAMPLES})"
        if self.innov_ratio > 0.30:
            return "UNSTABLE"
        if self.innov_ratio < 0.05:
            return "HEALTHY"
        return "SETTLED"

    @property
    def locked(self) -> bool:
        """True iff the user has manually frozen the fit. When True,
        BOTH the RLS update AND the intervention-learning path are
        skipped — the state vector x = [a, b, c] stays exactly where
        the user left it. Toggle via the 'Lock' checkbox in the
        tuner's Lateral tab → LiveParams section. Frozen state isn't
        persisted across launches by design — every session starts
        unlocked unless the user re-locks."""
        return bool(self.frozen)

    def adaptive_forget(self) -> float:
        # Adaptive (lower) forgetting is for cold convergence — when the
        # filter is wrong and we need to bias toward new samples to
        # escape. Once trusted the fit is reliable for the regime it
        # was learned in, and high-innovation samples are usually
        # off-regime transients (highway-trained fit going through a
        # tight off-highway turn at low v²). Speeding up adaptation
        # there corrupts the original fit instead of refining it —
        # observed runaway from a≈4 to a≈19 after one off-highway
        # detour. Trusted fits stick to FORGET (slowest).
        if self.trusted():
            return FORGET
        ratio = min(self.innov_ratio, 1.0)
        return max(FORGET_FLOOR, FORGET - FORGET_RESPONSIVENESS * ratio)

    def _in_trouble(self, v_ego: float) -> bool:
        # Disabled. The detector latched on pause/unpause cycles —
        # while paused, telemetry stalls so the speed history stays
        # full of pre-pause highs, and on resume the low current
        # speed looks like a 50% collapse. Other gates (quiescence,
        # saturation, user-assist, MIN_SPEED_MPS, MAX_WHEEL_FOR_FIT)
        # already filter genuine crash / off-road samples.
        return False

    def trusted(self) -> bool:
        return max(self.samples, self.session_samples) >= TRUSTED_MIN_SAMPLES

    def axis_for_wheel_angle(self, target_wheel_rad: float,
                             v_ego: float = 0.0) -> float:
        """Invert the inverse-rack model to get the axis that produces
        `target_wheel_rad` at speed `v_ego`:
            axis = a · w + b · w · v² + c
        Always uses the current state — so manually-set sliders take
        effect immediately, before RLS has had a chance to converge.
        Defaults at construction are (A_INIT, B_INIT, C_INIT) which is
        the same fallback we used to have; loaded files override them.

        |a| is floored at A_ABS_MIN_RUNTIME so the controller never
        decides 'no steering is best' just because RLS pinned a near
        zero on degenerate data (e.g. lots of straight-line samples
        with no rack excitation). Sign of a is preserved — RLS still
        owns the sign discovery."""
        a, b, c = float(self.x[0]), float(self.x[1]), float(self.x[2])
        if 0.0 <= a < A_ABS_MIN_RUNTIME:
            a = A_ABS_MIN_RUNTIME
        elif -A_ABS_MIN_RUNTIME < a < 0.0:
            a = -A_ABS_MIN_RUNTIME
        v2 = v_ego * v_ego
        axis = a * target_wheel_rad + b * target_wheel_rad * v2 + c
        return float(axis)

    @property
    def a_below_floor(self) -> bool:
        """True when RLS's internal |a| is below the runtime clamp,
        i.e. the fit has degenerated and the controller is currently
        running on the floor value. Surfaced in the HUD so the user
        knows their fit is unhealthy even when the truck appears to
        steer normally (because the floor is doing the work)."""
        return abs(float(self.x[0])) < A_ABS_MIN_RUNTIME

    def _intervention_learn(self, target_axis: float, wheel: float,
                            v2: float) -> None:
        """User is intervening — blend `a` toward the value that would
        have made our AI command exactly the observed gameSteer.

        Solve  target_axis = a·wheel + b·wheel·v² + c  for a, with
        b and c held at their current values, then EMA toward it.
        Skipped when wheel is too small for the inversion to be
        numerically meaningful.

        Sets `last_intervened_ts` so the controller's trim integrator
        can freeze itself while the human is on the wheel — the trim
        and the intervention path both nudge axis toward the human's
        intent, so we let intervention own that correction."""
        # Always mark as intervened, even if we skip the math below,
        # so a streak of small-wheel intervention frames still freezes
        # the trim integrator.
        self.last_intervened_ts = time.monotonic()
        if abs(wheel) < INTERVENTION_MIN_WHEEL:
            return
        implied_a = (target_axis
                     - self.x[2]
                     - self.x[1] * wheel * v2) / wheel
        alpha = INTERVENTION_LEARN_RATE
        new_a = (1.0 - alpha) * self.x[0] + alpha * implied_a
        # Magnitude clamp (keep sign).
        sign_a = -1.0 if new_a < 0 else 1.0
        self.x[0] = sign_a * min(A_ABS_MAX, abs(new_a))

    def update(self, game_steer: float | None, v_ego: float | None,
               wheel_angle_rad: float | None,
               commanded_axis: float | None = None,
               gate_override: dict | None = None) -> None:
        """Feed one observation. Regression target is `game_steer`;
        regressors are `[wheel, wheel·v², 1]`.

        `gate_override`, when provided, shadows `self._gates` for this
        single call — used by the guided-calibration routine to widen
        gates without mutating the per-game profile. Only the keys
        present override; missing keys fall through to `self._gates`.
        ETS2's strict module-level defaults stay untouched."""
        if game_steer is None:
            return
        # Per-call gate dict. Without an override this is just the
        # instance's stored profile; with one we merge so the routine
        # can loosen e.g. innovation gate + Q without touching the
        # baseline. `Q` is recomputed only when the override changes
        # Q_diag, otherwise the cached self._Q is used.
        gates = (self._gates if gate_override is None
                 else {**self._gates, **gate_override})
        Q = (self._Q if gate_override is None or "Q_diag" not in gate_override
             else np.diag(gates["Q_diag"]))
        self._axis_buf.append(float(game_steer))
        self._commanded_buf.append(
            float(commanded_axis) if commanded_axis is not None else 0.0)
        self._v_buf.append(float(v_ego) if v_ego is not None else 0.0)
        if wheel_angle_rad is None or v_ego is None:
            self.rej_no_wheel += 1
            return
        if self._in_trouble(v_ego):
            self.rej_trouble += 1
            return
        if v_ego < gates["min_speed_mps"]:
            self.rej_slow += 1
            return
        if len(self._axis_buf) <= self.lag_frames:
            return

        axis_lagged = float(self._axis_buf[0])
        v_lagged = float(self._v_buf[0])
        if abs(axis_lagged) < gates["min_axis"]:
            self.rej_small_axis += 1
            return
        if abs(float(wheel_angle_rad)) > MAX_WHEEL_FOR_FIT:
            self.rej_wheel += 1
            return
        # Small-wheel gate (per-game). For Forza/AC where the wheel
        # is synthesized from yaw rate, a near-zero wheel means the
        # vehicle isn't turning enough to extract any info about the
        # rack — and admitting these samples lets the bias term `c`
        # bleed into `a` and `b` through covariance off-diagonals.
        # ETS2's profile keeps this at 0 because its min_axis already
        # filters that case.
        if abs(float(wheel_angle_rad)) < gates["min_wheel"]:
            self.rej_small_wheel += 1
            return

        # Quiescence gate. Reject samples where the input has been
        # changing fast across the lag window — `axis_lagged` and the
        # current wheel observation aren't actually a matched pair in
        # that case, and the resulting innovation is a lag artifact.
        if len(self._axis_buf) >= 2:
            spread = max(self._axis_buf) - min(self._axis_buf)
            if spread > gates["max_axis_spread_for_fit"]:
                self.rej_transient += 1
                return

        # Intervention learning + saturation gate. When the observed
        # gameSteer differs from what we commanded, the human is
        # intervening — adding (helping) or subtracting (fighting)
        # input. Instead of dropping these samples we compute the
        # `a` that would have made our AI command match the OBSERVED
        # gameSteer at this frame's wheel angle, and blend toward
        # that value. Sustained help → a grows → next AI command
        # steers more. Sustained fight → a shrinks → next AI command
        # steers less. Saturation is still rejected outright (mech
        # limit, not the linear rack).
        wheel = float(wheel_angle_rad)
        v2 = v_lagged * v_lagged
        if commanded_axis is not None and len(self._commanded_buf) > self.lag_frames:
            commanded_lagged = float(self._commanded_buf[0])
            if abs(commanded_lagged) >= SATURATION_AXIS:
                self.rej_saturated += 1
                return
            if abs(axis_lagged - commanded_lagged) > USER_ASSIST_DELTA:
                # Honor the user's lock — intervention still counts as
                # "modifying the fit," so a locked LiveParams stays put
                # even if the user nudges the wheel mid-drive.
                if not self.locked:
                    self._intervention_learn(axis_lagged, wheel, v2)
                self.rej_assist += 1   # counts as "intervention" for HUD
                return

        # Regression: axis = a·wheel + b·wheel·v² + c.
        # (`wheel` and `v2` were already computed above for the
        # intervention check.)
        H = np.array([wheel, wheel * v2, 1.0], dtype=np.float64)
        predicted = float(H @ self.x)
        actual = axis_lagged
        innovation = actual - predicted
        innov_gate = (MAX_INNOVATION_WARM if self.trusted()
                      else gates["max_innovation_cold"])
        if abs(innovation) > innov_gate:
            self.rej_innovation += 1
            self._innov_ema = 0.95 * self._innov_ema + 0.05 * abs(innovation)
            self._actual_ema = 0.95 * self._actual_ema + 0.05 * abs(actual)
            self.last_innovation = innovation
            self.last_predicted = predicted
            self.last_actual = actual
            return

        if self.locked:
            self._innov_ema = 0.95 * self._innov_ema + 0.05 * abs(innovation)
            self._actual_ema = 0.95 * self._actual_ema + 0.05 * abs(actual)
            self.last_innovation = innovation
            self.last_predicted = predicted
            self.last_actual = actual
            return

        forget = self.adaptive_forget()
        denom = forget + float(H @ self.P @ H)
        K = (self.P @ H) / denom
        self.x = self.x + K * innovation
        # Standard RLS covariance update + Kalman process noise.
        # Without the `+ Q` term P shrinks monotonically and the
        # filter eventually stops responding to new information
        # (sample-rich, low-uncertainty regime). Adding Q each step
        # keeps it alive — what openpilot's `paramsd` does via its
        # EKF process-noise matrix. `Q` here is the override-merged
        # version when a gate_override was passed in.
        self.P = (self.P - np.outer(K, H @ self.P)) / forget + Q

        # Bounds-clamp each parameter. `a` keeps its sign; `b` is
        # sign-locked to `a` (the stiffness term is always in the same
        # direction as the base linear response, so the rack doesn't
        # flip direction with speed). `c` is a free bias.
        sign_a = -1.0 if self.x[0] < 0 else 1.0
        self.x[0] = sign_a * min(A_ABS_MAX, abs(self.x[0]))
        self.x[1] = sign_a * min(B_ABS_MAX, max(0.0, sign_a * self.x[1]))
        c_max = gates["c_abs_max"]
        self.x[2] = max(-c_max, min(c_max, self.x[2]))

        self.samples += 1
        self.session_samples += 1
        self.last_innovation = innovation
        self.last_predicted = predicted
        self.last_actual = actual
        self._innov_ema = 0.95 * self._innov_ema + 0.05 * abs(innovation)
        self._actual_ema = 0.95 * self._actual_ema + 0.05 * abs(actual)
        # Trajectory log + last-update timestamp for HUD diagnostics.
        # Appended only on accepted updates so the buffer reflects what
        # the filter actually saw, not every frame.
        self.last_update_ts = time.monotonic()
        self.history.append((self.last_update_ts,
                             float(self.x[0]), float(self.x[1]),
                             float(self.x[2])))

        if abs(innovation) > BAD_FIT_INNOV_THRESHOLD:
            self._consecutive_bad_fit += 1
            # Watchdog inflates Kalman gain to unstick a cold fit. It
            # MUST NOT fire when trusted — observed runaway: highway-
            # learned fit goes off-highway, low-v² samples produce
            # large innovation, watchdog resets P, the next off-regime
            # samples carry huge weight, `a` drifts from ~4 to ~19
            # and the fit doesn't recover even back on highway
            # (closed-loop reinforcement). Trusted fits just keep
            # their slow RLS adaptation; off-regime errors accumulate
            # softly and wash out when the regime returns.
            if (not self.trusted()
                    and self._consecutive_bad_fit >= BAD_FIT_PATIENCE):
                self.P = np.diag(gates["P_init"])
                self.watchdog_resets += 1
                self._consecutive_bad_fit = 0
        else:
            self._consecutive_bad_fit = 0

        self._dirty += 1
        if self._dirty >= 200:
            self.save()
            self._dirty = 0

    def reset(self, wipe_disk: bool = False) -> None:
        """Wipe in-memory RLS state to the per-(game, device) seed.
        With `wipe_disk=True`, also remove the persisted JSON so the
        next session starts fresh — used by the overlay `R` hotkey
        and the `--reset-calib` CLI flag. Without `wipe_disk`, the
        on-disk fit is preserved until the next `save()` overwrites
        it (useful for tuner-triggered resets that want a known-good
        backup on disk).

        Uses the per-(game, device) seed for x — the same starting
        point a fresh cold start would use, not the universal
        A_INIT/B_INIT/C_INIT."""
        a, b, c = _seed_for(self.game, self.device_kind)
        self.x = np.array([a, b, c], dtype=np.float64)
        self.P = np.diag(self._gates["P_init"])
        self.samples = 0
        self.session_samples = 0
        self._axis_buf.clear()
        self._commanded_buf.clear()
        self._v_buf.clear()
        self._speed_history.clear()
        self._consecutive_bad_fit = 0
        self.rej_wheel = 0
        self.rej_innovation = 0
        self.rej_trouble = 0
        self.rej_assist = 0
        self.rej_saturated = 0
        self.rej_transient = 0
        self.rej_slow = 0
        self.rej_no_wheel = 0
        self.rej_small_axis = 0
        self.rej_small_wheel = 0
        self.watchdog_resets = 0
        self.history.clear()
        self.last_update_ts = 0.0
        if wipe_disk:
            key = self.game
            if self.device_kind and self.game:
                key = f"{self.game}_{self.device_kind}"
            try:
                p = state_path("liveparams", key)
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    def reset_covariance_only(self) -> None:
        """Reset P (and consecutive-bad-fit counter) to cold-start
        values without touching the state vector x = [a, b, c].

        Used by the guided-calibration routine between steps so each
        new step's samples carry full Kalman gain instead of being
        diluted by the previous step's accumulated certainty. Doesn't
        clear sample counts or rejection counters — those still reflect
        lifetime activity."""
        self.P = np.diag(self._gates["P_init"])
        self._consecutive_bad_fit = 0

    # ----- persistence -----

    @classmethod
    def _resolve_path(cls, game: str | None,
                      device_kind: str | None) -> Path | None:
        """Per-(game, device_kind) → per-game → legacy fallback chain.

        Preferred: `liveparams_<game>_<device>.json`.
        Fallback 1: `liveparams_<game>.json` (pre-device versions).
        Fallback 2: `liveparams.json` (legacy single-file).
        Returns the first existing path, or None if none exist."""
        if device_kind and game:
            key = f"{game}_{device_kind}"
            game_dev_path = state_path("liveparams", key)
            if game_dev_path.exists():
                return game_dev_path
        # Per-game (pre-device-suffix) file.
        if game:
            game_path = state_path("liveparams", game)
            if game_path.exists():
                return game_path
        # Legacy single-file.
        legacy = state_path("liveparams", None)
        if legacy.exists():
            return legacy
        return None

    @classmethod
    def _load(cls, game: str | None = None,
              device_kind: str | None = None,
              path: Path | None = None) -> _Persisted:
        # The "no good data" return value is the per-(game, device)
        # seed, not the universal A_INIT/B_INIT/C_INIT — cold start
        # converges much faster from a per-game ballpark than from a=10.
        def _seeded() -> _Persisted:
            a, b, c = _seed_for(game, device_kind)
            return _Persisted(a=a, b=b, c=c, samples=0)

        if path is None:
            path = cls._resolve_path(game, device_kind)
        if path is None or not path.exists():
            return _seeded()
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return _seeded()
        # New format: {"a", "b", "c", "samples"}.
        if "a" in data:
            try:
                a = float(data.get("a", A_INIT))
                b = float(data.get("b", B_INIT))
                c = float(data.get("c", C_INIT))
                samples = int(data.get("samples", 0))
            except (ValueError, TypeError):
                return _seeded()
        else:
            # Legacy format: {"scale", "offset", "samples"}. Migrate:
            #   wheel = scale·axis + offset
            # ⇒ axis = (1/scale)·wheel + (-offset/scale)
            # ⇒ a = 1/scale, b = 0, c = -offset/scale.
            #
            # The SIGN of `scale` is preserved — it captures the
            # axis-to-wheel correlation direction in the user's game,
            # which `lateral_sign` is independent of. (We previously
            # abs()'d this and broke users whose legacy fit was
            # negative.)
            #
            # Samples are ALWAYS reset to 0 on migration: the old
            # 2-param fit's sample count doesn't validate the new
            # 3-param model. Preserving it would auto-freeze the new
            # fit on first load and prevent `b` (speed-stiffness)
            # from ever learning.
            try:
                scale = float(data.get("scale", 0.1))
                offset = float(data.get("offset", 0.0))
            except (ValueError, TypeError):
                return _seeded()
            if abs(scale) < 1e-6:
                return _seeded()
            a = 1.0 / scale
            b = B_INIT
            c = -offset / scale
            samples = 0
        # Bounds check (magnitudes only — signs are user-data). Out-
        # of-range loads were probably written by an older code
        # version against a different model; reset to the per-game
        # seed and samples=0 so the user gets a fresh, well-seeded fit.
        # Use the LARGER (ETS2) c-bound here as the "obviously garbage"
        # threshold — accept files that came from any profile, runtime
        # gate clamps them tighter on the next RLS step if needed.
        in_bounds = (abs(a) <= A_ABS_MAX
                     and abs(b) <= B_ABS_MAX
                     and abs(c) <= _GATES_DEFAULT["c_abs_max"])
        if not in_bounds:
            return _seeded()
        return _Persisted(a=a, b=b, c=c, samples=samples)

    def save(self, path: Path | None = None) -> None:
        if path is None:
            # Always write to the per-(game, device) path so the next
            # load picks up THIS device's fit, not the other one's.
            key = self.game
            if self.device_kind and self.game:
                key = f"{self.game}_{self.device_kind}"
            path = state_path("liveparams", key)
        try:
            path.write_text(json.dumps({
                "a": self.a_linear,
                "b": self.b_quad,
                "c": self.c_bias,
                "samples": self.samples,
            }, indent=2))
        except OSError:
            pass

    def switch_device(self, new_device_kind: str | None) -> None:
        """Save the current fit (under the OLD device's path) and
        reload from the NEW device's path. Called by the tuner when
        the user toggles between gamepad and wheel — each device has
        a different rack response sign / magnitude, so we keep one
        fit per device-kind instead of trying to share one.

        State that's NOT persisted (session_samples, rejection
        counters, RLS covariance) gets reset, matching what would
        happen on a fresh launch with the new device."""
        if new_device_kind == self.device_kind:
            return
        # Persist current state to the OLD device's path.
        self.save()
        # Switch identity and load fresh.
        self.device_kind = new_device_kind
        # If the new device has no persisted file, _load returns the
        # seed — mark the instance as seeded again so HUD/print know.
        resolved = self._resolve_path(game=self.game,
                                      device_kind=new_device_kind)
        self._was_seeded = (resolved is None)
        loaded = self._load(game=self.game, device_kind=new_device_kind,
                            path=resolved)
        if self._was_seeded:
            print(f"liveparams [{new_device_kind}]: cold start — "
                  f"seeded for game={self.game!r}: "
                  f"a={loaded.a:.3f} b={loaded.b:.4f} c={loaded.c:+.4f}")
        self.x = np.array([loaded.a, loaded.b, loaded.c], dtype=np.float64)
        self.P = np.diag(self._gates["P_init"])
        self.samples = loaded.samples
        self.session_samples = 0
        self._axis_buf.clear()
        self._commanded_buf.clear()
        self._v_buf.clear()
        self._speed_history.clear()
        self._consecutive_bad_fit = 0
        self._innov_ema = 0.0
        self._actual_ema = 0.0
        # Reset diagnostics so the HUD shows the new device's fresh
        # session, not the previous device's history.
        self.rej_wheel = 0
        self.rej_innovation = 0
        self.rej_trouble = 0
        self.rej_assist = 0
        self.rej_saturated = 0
        self.rej_transient = 0
        self.rej_slow = 0
        self.rej_no_wheel = 0
        self.rej_small_axis = 0
        self.rej_small_wheel = 0
        self.watchdog_resets = 0
        self.history.clear()
        self.last_update_ts = 0.0
        self._dirty = 0
