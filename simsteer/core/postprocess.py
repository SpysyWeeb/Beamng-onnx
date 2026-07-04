"""Decode the flat outputs from vision and policy into structured fields.

For each MDN-ish output the model emits `mu` first then `std` of the
same length, so we just take the first half. The single-hypothesis plan
in the current model is shape (33, 15) for mu (and the same for std).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from simsteer.core.constants import (
    IDX_N,
    NUM_LANE_LINES,
    NUM_ROAD_EDGES,
    PLAN_WIDTH,
    POLICY_SLICES,
    T_IDXS,
    VISION_SLICES,
    X_IDXS,
    PlanField,
)


def _slice(flat: np.ndarray, name: str, table: dict[str, tuple[int, int]]) -> np.ndarray:
    a, b = table[name]
    return flat[a:b]


def _split_mu_std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    half = arr.shape[-1] // 2
    return arr[..., :half], arr[..., half:]


# Lead vehicle layout (matches openpilot's ModelConstants):
#   - 3 selected hypotheses
#   - 6 timesteps per lead
#   - 4 metrics per timestep (x, y, v, a)
# 144 lead floats = 3 * (6*4 mu + 6*4 std).
LEAD_MHP_SELECTION = 3
LEAD_TRAJ_LEN = 6
LEAD_WIDTH = 4
LEAD_T_IDXS = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]  # seconds — openpilot's grid


@dataclass
class Decoded:
    plan: np.ndarray              # (33, 15) — desired trajectory
    plan_std: np.ndarray
    lane_lines: np.ndarray        # (4, 33, 2)
    lane_lines_prob: np.ndarray   # (4,) — sigmoid'd
    road_edges: np.ndarray        # (2, 33, 2)
    pose: np.ndarray              # (6,) — vehicle ego-motion (m/s, rad/s) in device frame
    pose_std: np.ndarray          # (6,) — model's per-axis std for the above; gates LiveCalib
    road_transform: np.ndarray    # (6,) — road frame in device coords (trans + euler)
    road_transform_std: np.ndarray  # (6,) — std on the same; height-std gate uses [2]
    wide_from_device_euler: np.ndarray  # (3,) — model's wide-cam mount estimate (rad)
    desire_state: np.ndarray      # (8,) — softmax over desires
    lead_prob: np.ndarray         # (3,)
    leads: np.ndarray             # (3, 6, 4) — mu(x, y, v, a) per lead/timestep
    leads_std: np.ndarray         # (3, 6, 4) — same shape, std deviations
    # Direct action head (big_driving_supercombo): [lat_action, accel]
    # at the model's requested action_t horizon. None on models whose
    # action slot is padding (CD210) — those derive from the plan.
    # master get_action_from_model: desiredCurvature = action[0] /
    # max(1, v_ego)^2, desiredAcceleration = action[1].
    action: np.ndarray | None = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def decode(vision_flat: np.ndarray, policy_flat: np.ndarray) -> Decoded:
    # --- Vision ---
    pose_raw = _slice(vision_flat, "pose", VISION_SLICES)         # 12 = mu(6) + std(6)
    pose_mu, pose_std = _split_mu_std(pose_raw)
    # Model emits std in log-space (matches openpilot's convention) —
    # exponentiate so consumers see linear stddevs they can threshold
    # against physical units (e.g. 30 mm for height stddev).
    pose_std = np.exp(pose_std)

    road_tf_raw = _slice(vision_flat, "road_transform", VISION_SLICES)  # 12 = mu(6) + std(6)
    road_tf_mu, road_tf_std = _split_mu_std(road_tf_raw)
    road_tf_std = np.exp(road_tf_std)

    wide_raw = _slice(vision_flat, "wide_from_device_euler", VISION_SLICES)  # 6 = mu(3) + std(3)
    wide_mu, _ = _split_mu_std(wide_raw)

    ll_raw = _slice(vision_flat, "lane_lines", VISION_SLICES)     # 528 = (4*33*2)*2
    ll_per_side = NUM_LANE_LINES * IDX_N * 2
    ll_mu = ll_raw[:ll_per_side].reshape(NUM_LANE_LINES, IDX_N, 2)

    re_raw = _slice(vision_flat, "road_edges", VISION_SLICES)     # 264 = (2*33*2)*2
    re_per = NUM_ROAD_EDGES * IDX_N * 2
    re_mu = re_raw[:re_per].reshape(NUM_ROAD_EDGES, IDX_N, 2)

    ll_prob_raw = _slice(vision_flat, "lane_lines_prob", VISION_SLICES)  # 8 = 4 mu + 4 std
    ll_prob = _sigmoid(ll_prob_raw[:NUM_LANE_LINES])

    lead_prob_raw = _slice(vision_flat, "lead_prob", VISION_SLICES)
    lead_prob = _sigmoid(lead_prob_raw)

    # Lead trajectory predictions. Slice is 3*6*4*2 = 144 floats.
    # Per-lead block: 24 mu values then 24 std values, each laid out
    # as (timestep, metric) in row-major order. Order of leads is the
    # MHP selection order (highest-confidence first by convention).
    lead_raw = _slice(vision_flat, "lead", VISION_SLICES)
    leads_mu = np.empty((LEAD_MHP_SELECTION, LEAD_TRAJ_LEN, LEAD_WIDTH),
                        dtype=np.float32)
    leads_std = np.empty_like(leads_mu)
    n_per_lead = LEAD_TRAJ_LEN * LEAD_WIDTH  # 24
    for i in range(LEAD_MHP_SELECTION):
        base = i * (2 * n_per_lead)
        leads_mu[i] = lead_raw[base:base + n_per_lead].reshape(
            LEAD_TRAJ_LEN, LEAD_WIDTH)
        leads_std[i] = lead_raw[base + n_per_lead:base + 2 * n_per_lead].reshape(
            LEAD_TRAJ_LEN, LEAD_WIDTH)

    # --- Policy ---
    plan_raw = _slice(policy_flat, "plan", POLICY_SLICES)         # 990 = 33*15*2
    plan = plan_raw[: IDX_N * PLAN_WIDTH].reshape(IDX_N, PLAN_WIDTH)
    plan_std = plan_raw[IDX_N * PLAN_WIDTH:].reshape(IDX_N, PLAN_WIDTH)

    desire_raw = _slice(policy_flat, "desire_state", POLICY_SLICES)
    desire_state = _softmax(desire_raw)

    return Decoded(
        plan=plan, plan_std=plan_std,
        lane_lines=ll_mu, lane_lines_prob=ll_prob,
        road_edges=re_mu,
        pose=pose_mu, pose_std=pose_std,
        road_transform=road_tf_mu, road_transform_std=road_tf_std,
        wide_from_device_euler=wide_mu,
        desire_state=desire_state,
        lead_prob=lead_prob,
        leads=leads_mu, leads_std=leads_std,
    )


MIN_SPEED = 1.0

# DT_MDL = the model's tick period (20 Hz → 50 ms). Comma uses this as
# the "one frame" duration in the curvature rate-limit.
DT_MDL = 0.05

# Speed-interpolated max curvature rate (1/m per second). Verbatim
# from openpilot's `MAX_CURVATURE_RATES` constants in
# `selfdrive/controls/lib/drive_helpers.py`. At 2 m/s the controller
# can swing through a 100-m-radius turn (κ=0.01) in 0.27 s; at 29 m/s
# the same swing takes 2.9 s — proportional to the time-to-cover the
# corner.
# Unwind-vs-wind rate asymmetry (Hyundai carcontroller 7/3).
_UNWIND_RATE_RATIO = 7.0 / 3.0
_MAX_CURV_RATE_LO_V = 2.0
_MAX_CURV_RATE_HI_V = 29.0
_MAX_CURV_RATE_LO = 0.03762194918267951
_MAX_CURV_RATE_HI = 0.003441203371932992

# Comma's `get_lag_adjusted_curvature` adds 0.2 s to the actuator
# delay before reading the plan, with the comment "extra .2 s for
# other delays we don't model" (CAN / queue / inter-process). We
# adopt it because the user's testing showed the controller felt
# laggy without it — the model's predicted heading at exactly
# lookahead_s ahead is "right now" relative to actuator response,
# leaving no margin for the rest of the pipeline. Putting it back.
EXTRA_LAG_BUFFER_S = 0.2

# Master drive_helpers MIN_STABLE_DELAY: below this action horizon the
# psi/(v*t) form is numerically unstable, so the heading target is read
# at the stable point and scaled down linearly.
_MIN_STABLE_DELAY = 0.3


def desired_curvature_lag_adjusted(
    plan: np.ndarray, v_ego: float, steer_actuator_delay: float,
    last_desired_curvature: float = 0.0,
    extra_buffer_s: float | None = None,
    lat_jerk_max_mps3: float | None = None,
    lat_accel_max_mps2: float | None = None,
    override_desired_k: float | None = None,
) -> float:
    """Lag-adjusted desired curvature, ported from openpilot's
    `selfdrive/controls/lib/drive_helpers.py:get_lag_adjusted_curvature`.

    Computes the curvature to command NOW so that after the actuator
    delay we're tracking the plan's heading:

        delay = actuator_delay + extra_buffer_s
        psi_at_delay     = plan_yaw interpolated at `delay`
        avg_k_desired    = psi_at_delay / (v_ego * delay)
        cur_k_desired    = plan_yaw_rate[0] / max(plan_v[0], MIN_SPEED)
        desired_curvature = 2 * avg_k_desired - cur_k_desired

    `extra_buffer_s` is the user-tunable anticipation buffer (default
    falls back to `EXTRA_LAG_BUFFER_S = 0.2 s`, matching openpilot).
    Higher = read further into the plan → AI starts the steering motion
    earlier; lower = react closer to the present → starts later. A
    slider in the tuner's Lateral tab drives this.

    Then RATE-LIMIT relative to the previous-frame value so a step
    change in the model's plan doesn't translate into a step change
    in steering output. The rate cap is speed-interpolated:
        max_rate(v) = lerp(v, [2, 29], [0.0376, 0.0034]) 1/m/s
    Per frame at 50 ms: max change ~ 0.0019 (at 2 m/s) → 0.00017 (at
    29 m/s). At 20 m/s a 0-to-100-m-radius (κ=0.01) ramp takes ~17
    frames ≈ 0.85 s.

    `last_desired_curvature` is the value commanded on the previous
    frame; the caller carries this across frames.
    """
    v = max(v_ego, MIN_SPEED)
    buf = EXTRA_LAG_BUFFER_S if extra_buffer_s is None else float(extra_buffer_s)
    delay = max(float(steer_actuator_delay) + buf, 1e-3)

    if override_desired_k is not None:
        # Direct action head (big model): the network already computed
        # the curvature for t+action_t, so the plan psi math is
        # skipped — but the rate limiter and clamps below still apply.
        desired_k = float(override_desired_k)
        return _shape_desired_k(desired_k, v, last_desired_curvature,
                                lat_jerk_max_mps3, lat_accel_max_mps2)

    yaw_col = plan[:, PlanField.EULER.start + 2]              # column 11
    yaw_rate_col = plan[:, PlanField.ORIENTATION_RATE.start + 2]  # column 14
    t_idxs = np.asarray(T_IDXS, dtype=np.float32)

    # Master drive_helpers get_curvature_from_plan: below
    # MIN_STABLE_DELAY the heading target is read at the stable point
    # and scaled linearly (psi/(v*delay) would otherwise blow up as
    # delay -> 0), and the current-curvature term divides by EGO
    # speed, not plan v[0].
    if delay < _MIN_STABLE_DELAY:
        psi_at_delay = (delay / _MIN_STABLE_DELAY) * float(
            np.interp(_MIN_STABLE_DELAY, t_idxs, yaw_col))
    else:
        psi_at_delay = float(np.interp(delay, t_idxs, yaw_col))
    avg_k = psi_at_delay / (v * delay)
    cur_k = float(yaw_rate_col[0]) / v

    desired_k = 2.0 * avg_k - cur_k

    # The 2*avg - cur extrapolation can overshoot PAST the tightest
    # curvature the plan actually reaches — worst on a fast-tightening
    # hairpin, where it commanded ~36% more curvature than the model's
    # own plan and steered the car to the INSIDE (into oncoming / the
    # inside dirt; 2026-07-03 forensics, frame 17105). Cap the
    # anticipated magnitude to the plan's own peak curvature within the
    # lookahead window so we lead the turn-in TIMING without ever
    # commanding more turn than the model plans.
    horizon = delay * 1.5
    kwin = yaw_rate_col[t_idxs <= horizon] / v
    if kwin.size:
        k_cap = float(np.max(np.abs(kwin)))
        desired_k = float(np.clip(desired_k, -k_cap, k_cap))

    # Rate-limit against the previous commanded curvature. This is the
    # mechanism that prevents "steers in too early": a step jump in
    # plan curvature gets ramped over many frames.
    #
    # Two forms:
    return _shape_desired_k(desired_k, v, last_desired_curvature,
                            lat_jerk_max_mps3, lat_accel_max_mps2)




def _shape_desired_k(desired_k: float, v: float,
                     last_desired_curvature: float,
                     lat_jerk_max_mps3: float | None,
                     lat_accel_max_mps2: float | None) -> float:
    """Shared shaping for a desired curvature, whatever produced it
    (plan psi math or a direct model action head): the ISO-jerk
    rate limiter with unwind asymmetry, then the optional lat-accel
    window and speed-aware ceiling."""
    #  - ISO lateral-jerk (current openpilot, drive_helpers.clip_curvature):
    #    max_rate = MAX_LATERAL_JERK / v^2. Speed-aware the right way
    #    around — a slow car may swing the wheel fast (city corners,
    #    offramps), a fast car may not (no highway wobble). Enabled by
    #    passing lat_jerk_max_mps3 (openpilot uses 5.0).
    #  - Legacy speed-interpolated table (older openpilot) otherwise.
    #    At 10 m/s it allows ~5x LESS wheel rate than ISO, which made
    #    the model "freak out and stop" in city turns: the plan asked
    #    for curvature the ramp couldn't reach in time.
    if lat_jerk_max_mps3 is not None:
        max_rate = float(lat_jerk_max_mps3) / (v * v)
    else:
        max_rate = float(np.interp(
            v, [_MAX_CURV_RATE_LO_V, _MAX_CURV_RATE_HI_V],
            [_MAX_CURV_RATE_LO, _MAX_CURV_RATE_HI],
        ))
    max_delta = max_rate * DT_MDL
    # openpilot lets the wheel UNWIND (curvature moving toward zero)
    # faster than it winds in — the carcontroller torque-rate limits
    # are asymmetric (e.g. Hyundai STEER_DELTA_UP 3 vs _DOWN 7 per
    # frame). Apply the same ~2.3x allowance to deltas that shrink
    # the command's magnitude.
    unwind = _UNWIND_RATE_RATIO
    lo = max_delta * (unwind if last_desired_curvature > 0 else 1.0)
    hi = max_delta * (unwind if last_desired_curvature < 0 else 1.0)
    safe_desired_k = float(np.clip(
        desired_k,
        last_desired_curvature - lo,
        last_desired_curvature + hi,
    ))
    # openpilot's clip_curvature also clamps the magnitude: lateral
    # accel (v^2 * k) to +/-3.0 m/s^2, and curvature to +/-0.2. But the
    # flat 0.2 is a highway-hardware constant — a real 90-degree city
    # corner is radius 3-4 m (k 0.25-0.3), which 0.2 forbids at ANY
    # speed. That's why openpilot runs wide on sharp turns, and logged
    # turns here pinned at 0.2 while lat accel was only ~3 (turn
    # forensics 2026-07-02). Open the ceiling to 0.35 at crawl speed,
    # tapering back to the ISO 0.2 wherever v^2*0.35 would exceed the
    # comfort bound — the lat-accel clamp stays the binding limit.
    if lat_accel_max_mps2 is not None:
        m = float(lat_accel_max_mps2)
        safe_desired_k = float(np.clip(
            safe_desired_k, -m / (v * v), m / (v * v)))
        k_cap = float(np.clip(m / (v * v), 0.2, 0.35))
        safe_desired_k = float(np.clip(safe_desired_k, -k_cap, k_cap))
    return safe_desired_k


# Back-compat alias for callers that haven't switched over yet.
def desired_curvature(plan: np.ndarray, v_ego: float,
                      lookahead_s: float = 0.3) -> float:
    """Deprecated: use `desired_curvature_lag_adjusted` instead.
    Kept so older call sites still work; behaves like comma's
    `get_lag_adjusted_curvature` with the rate-limit disabled
    (last=0, no clamp visible to the caller)."""
    return desired_curvature_lag_adjusted(
        plan, v_ego, lookahead_s, last_desired_curvature=0.0)


def apply_yaw_calibration(plan: np.ndarray, lane_lines: np.ndarray,
                          road_edges: np.ndarray,
                          yaw_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate model outputs around the vertical axis by `yaw_rad`.

    The model's outputs are nominally in device frame, but only if the
    camera is mounted at the same orientation it was trained on. With a
    real mounting yaw offset (LiveCalib measures this), the outputs end
    up in a rotated frame: positions sit slightly to one side of where
    they should, plan_yaw_rate has a constant bias, and the controller
    drifts laterally as a result.

    `yaw_rad` is the *correction* — typically `-live_calib.yaw_estimate`
    (we rotate the outputs back by the negative of the detected mounting
    angle).

    Position rotation:
        x' =  x*cos - y*sin
        y' =  x*sin + y*cos
    Orientation shift: yaw component (Euler index 2) gets +yaw_rad
    (rotating the whole plan by yaw_rad means each pose's heading is
    yaw_rad larger).
    """
    if abs(yaw_rad) < 1e-4:
        return plan, lane_lines, road_edges
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)

    plan = plan.copy()
    # plan columns: pos(0:3), vel(3:6), accel(6:9), euler(9:12), orient_rate(12:15)
    for col_x, col_y in ((0, 1), (3, 4), (6, 7)):
        x, y = plan[:, col_x].copy(), plan[:, col_y].copy()
        plan[:, col_x] = x * c - y * s
        plan[:, col_y] = x * s + y * c
    # Euler yaw (index 11) and orient_rate yaw (index 14): shift by yaw_rad.
    # Rate is invariant to a constant rotation, but we still bump the
    # orientation entry so anything reading absolute heading is consistent.
    plan[:, 11] += yaw_rad

    # Lane lines: shape (4, 33, 2). Index [..., 0] = y_offset, [..., 1] = z.
    # We rotate each polyline in (x_idxs, y_offset) space — the x_idxs come
    # from the constants, not from the lane_lines tensor itself.
    xs = np.asarray(X_IDXS, dtype=lane_lines.dtype)
    new_lane_lines = lane_lines.copy()
    for i in range(lane_lines.shape[0]):
        y = lane_lines[i, :, 0]
        new_lane_lines[i, :, 0] = xs * s + y * c
    new_road_edges = road_edges.copy()
    for i in range(road_edges.shape[0]):
        y = road_edges[i, :, 0]
        new_road_edges[i, :, 0] = xs * s + y * c

    return plan, new_lane_lines, new_road_edges


