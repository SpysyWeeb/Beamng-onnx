"""Core regression checks for the BeamNG control stack.

Replaces upstream's tools/test_* suite (which tested the deleted
`pilot/` package). Everything here runs against simsteer.core with
synthetic Decoded frames — no game, no model files.

    .venv/bin/python3 tools/test_core_regression.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simsteer.core.control import ControllerConfig, LongitudinalController
from simsteer.core.postprocess import Decoded, desired_curvature_lag_adjusted

FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def mk(v_plan: float, yaw_rate_fn=lambda t: 0.0, turn_p: float = 0.0):
    plan = np.zeros((33, 15), dtype=np.float32)
    plan[:, 3] = v_plan
    t_idxs = LongitudinalController(ControllerConfig())._T_IDXS
    plan[:, 14] = [yaw_rate_fn(t) for t in t_idxs]
    d = np.zeros(8, dtype=np.float32)
    d[0], d[2] = 1.0 - turn_p, turn_p
    z = np.zeros
    return Decoded(plan=plan, plan_std=plan, lane_lines=z((4, 33, 2)),
                   lane_lines_prob=np.full(4, .9), road_edges=z((2, 33, 2)),
                   pose=z(6), pose_std=np.ones(6), road_transform=z(6),
                   road_transform_std=np.ones(6), wide_from_device_euler=z(3),
                   desire_state=d, lead_prob=z(3), leads=z((3, 6, 4)),
                   leads_std=np.ones((3, 6, 4)))


def test_turn_governor():
    cfg = ControllerConfig()
    lc = LongitudinalController(cfg)
    lc.compute(mk(9.0), 9.0, mode="exp")
    check("straight street untouched", lc.last_v_target > 8.5)
    lc.compute(mk(8.0, lambda t: 0.5 if 2 <= t <= 5 else 0.0), 8.0, mode="exp")
    check("90-deg turn 3 s out clamps to turn speed",
          abs(lc.last_v_target - cfg.turn_speed_mps) < 0.01,
          f"vT={lc.last_v_target:.2f}")
    lc.compute(mk(25.0, lambda t: 0.125), 25.0, mode="exp")
    check("highway sweeper (k=0.005) NOT clamped to turn speed",
          lc.last_v_target > 15.0, f"vT={lc.last_v_target:.2f}")
    lc.compute(mk(20.0, lambda t: 0.3), 20.0, mode="exp")
    check("canyon bend governed by corner scanner (~sqrt(3/k))",
          13.0 < lc.last_v_target < 15.0, f"vT={lc.last_v_target:.2f}")
    lc.compute(mk(9.0, turn_p=0.6), 9.0, mode="exp")
    check("desire-commanded turn clamps",
          abs(lc.last_v_target - cfg.turn_speed_mps) < 0.01)


def test_fb_integrator_gate():
    cfg = ControllerConfig()
    lc = LongitudinalController(cfg)
    for _ in range(400):
        lc.compute(mk(20.0), 19.6, mode="exp")   # steady, small droop
    fb_a = lc._a_fb_i
    for i in range(600):                          # setpoint swings, ego far
        lc.compute(mk(20.0 if (i // 60) % 2 == 0 else 13.0), 16.5,
                   mode="exp")
    drift = abs(lc._a_fb_i - fb_a)
    check("integrator learns at steady cruise", abs(fb_a) > 0.02,
          f"fb={fb_a:+.3f}")
    check("integrator frozen during setpoint swings", drift < 0.05,
          f"drift={drift:.3f}")


def test_no_vision_intervention():
    # The lost-road stop was removed by design (2026-07-02): like real
    # openpilot, low lane confidence must NOT touch longitudinal — the
    # controller has no such state at all.
    lc = LongitudinalController(ControllerConfig())
    check("controller has no road_lost intervention",
          not hasattr(lc, "road_lost"))
    lc.compute(mk(15.0), 15.0, mode="exp")
    check("plan followed regardless of vision confidence",
          lc.last_v_target > 14.0)


def test_curvature_clip():
    # Default policy (2026-07-02, user decision): execution clamps OFF
    # — only the rate limiter shapes steering.
    cfg = ControllerConfig()
    check("execution clamps disabled by default",
          cfg.lat_accel_max_mps2 <= 0)
    plan = mk(4.0, lambda t: 1.2).plan     # hard-turning plan
    plan[:, 11] = [2.0 * t for t in
                   LongitudinalController(ControllerConfig())._T_IDXS]
    k_free = desired_curvature_lag_adjusted(
        plan, 4.0, 0.25, last_desired_curvature=0.30,
        lat_jerk_max_mps3=10.0, lat_accel_max_mps2=None)
    check("unclamped: curvature free past every old ceiling",
          abs(k_free) > 0.30, f"k={k_free:.3f}")
    # the clamp MECHANISM stays available when explicitly enabled:
    k_flat = desired_curvature_lag_adjusted(
        plan, 4.0, 0.25, last_desired_curvature=0.30,
        lat_jerk_max_mps3=10.0, lat_accel_max_mps2=3.5)
    check("re-enabled clamp: low-speed ceiling 0.2-0.35 works",
          0.2 < abs(k_flat) <= 0.35, f"k={k_flat:.3f}")
    v = 14.3
    plan = mk(v, lambda t: 0.4).plan
    t_idxs = LongitudinalController(ControllerConfig())._T_IDXS
    plan[:, 11] = [0.5 * t for t in t_idxs]   # heading ramp: demand > clamp
    args = dict(last_desired_curvature=0.016, lat_jerk_max_mps3=10.0,
                lat_accel_max_mps2=3.5)
    k0 = desired_curvature_lag_adjusted(plan, v, 0.25, **args)
    k_bank = desired_curvature_lag_adjusted(plan, v, 0.25, roll_glat=-0.98,
                                            **args)
    check("right-hand bank allows more right curvature", k_bank > k0,
          f"flat={k0:.4f} banked={k_bank:.4f}")
    k_adverse = desired_curvature_lag_adjusted(plan, v, 0.25, roll_glat=0.98,
                                               **args)
    check("adverse bank allows less right curvature", k_adverse < k0,
          f"adverse={k_adverse:.4f}")


def test_min_stable_delay():
    # below 0.3 s the heading target must be the scaled stable-point
    # read (master drive_helpers), not a raw interp at a tiny delay
    plan = mk(10.0, lambda t: 0.2).plan
    plan[:, 11] = [0.2 * t for t in
                   LongitudinalController(ControllerConfig())._T_IDXS]
    k_small = desired_curvature_lag_adjusted(plan, 10.0, 0.05,
                                             extra_buffer_s=0.0)
    k_stable = desired_curvature_lag_adjusted(plan, 10.0, 0.30,
                                              extra_buffer_s=0.0)
    check("sub-stable delay does not blow up curvature",
          abs(k_small) <= abs(k_stable) * 1.05 + 1e-6,
          f"k(0.05s)={k_small:.4f} k(0.3s)={k_stable:.4f}")


if __name__ == "__main__":
    test_turn_governor()
    test_fb_integrator_gate()
    test_no_vision_intervention()
    test_curvature_clip()
    test_min_stable_delay()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(1 if FAILS else 0)
