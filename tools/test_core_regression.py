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


def test_corner_scanner():
    # The turn governor was removed; the corner scanner (comfort-based
    # brake-for-the-bend) stays and follows the plan's own curvature.
    cfg = ControllerConfig()
    lc = LongitudinalController(cfg)
    lc.compute(mk(9.0), 9.0, mode="exp")
    check("straight street untouched", lc.last_v_target > 8.5)
    lc.compute(mk(25.0, lambda t: 0.125), 25.0, mode="exp")
    check("highway sweeper (k=0.005) barely slowed",
          lc.last_v_target > 15.0, f"vT={lc.last_v_target:.2f}")
    lc.compute(mk(20.0, lambda t: 0.3), 20.0, mode="exp")
    check("canyon bend scanned to ~sqrt(max_lat/k)",
          13.0 < lc.last_v_target < 15.0, f"vT={lc.last_v_target:.2f}")


def test_speed_scale_persistence():
    # the sim-scale speed correction is a per-vehicle calibration value:
    # warm-started from disk, survives engage/disengage, saved clamped.
    from simsteer.core.control import load_speed_scale, save_speed_scale
    from simsteer.paths import data_dir
    key = "beamng-regressioncar"
    save_speed_scale(key, 1.137)
    check("speed scale round-trips", abs(load_speed_scale(key) - 1.137) < 1e-3)
    save_speed_scale(key, 5.0)
    check("insane scale clamped on save", load_speed_scale(key) <= 1.4)
    check("unknown vehicle warm-starts at 1.0",
          load_speed_scale("beamng-nosuchcar") == 1.0)
    lc = LongitudinalController(ControllerConfig(), speed_scale=1.12)
    check("controller warm-starts from persisted scale",
          abs(lc.speed_scale - 1.12) < 1e-6)
    lc.reset()
    check("speed scale NOT wiped on engage/disengage",
          abs(lc.speed_scale - 1.12) < 1e-6)
    os.remove(os.path.join(str(data_dir()), f"speed_scale_{key}.json"))


def test_brake_ramp_responsive():
    # A sudden hard-stop demand must build braking quickly, not crawl in
    # over a second (the red-light-overshoot bug). From a coast, a -3.5
    # a_target should drive a_cmd past -2.0 within ~0.5 s (10 frames).
    cfg = ControllerConfig()
    lc = LongitudinalController(cfg)
    lc.compute(mk(20.0), 20.0, mode="exp")          # coasting steady
    # model now plans a hard stop (plan accel col 6 = -3.5); ego hasn't
    # slowed yet, so a_cmd must ramp in fast
    d = mk(20.0)
    d.plan[:, 6] = -3.5
    frames_to_2 = None
    for i in range(30):
        lc.compute(d, 20.0, mode="exp")
        if frames_to_2 is None and lc.last_a_cmd <= -2.0:
            frames_to_2 = i + 1
    check("hard-stop braking builds within ~0.5s (not 1.4s)",
          frames_to_2 is not None and frames_to_2 <= 12,
          f"reached -2.0 m/s^2 in {frames_to_2} frames "
          f"({(frames_to_2 or 99) * 50}ms)")
    check("decel floor lets the model's ~-4 m/s^2 stops through",
          cfg.accel_cmd_min_mps2 <= -4.0, f"floor={cfg.accel_cmd_min_mps2}")


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


def test_understeer_ff_gate_and_learn():
    # FF is speed-GATED (no boost at city-turn speed, where it would
    # over-steer) and its magnitude is learned CLOSED-LOOP from delivery.
    from simsteer.core.control import LateralController
    cfg = ControllerConfig()
    lc = LateralController(cfg)
    d = mk(8.0)
    d.plan[:, 11] = [0.5 * t for t in
                     LongitudinalController(ControllerConfig())._T_IDXS]
    lc.compute(d, 8.0)
    check("understeer FF ~1.0 at city-turn speed (8 m/s)",
          abs(lc.last_ff - 1.0) < 0.02, f"ff={lc.last_ff:.3f}")
    lc.reset()
    lc.compute(d, 25.0)
    check("understeer FF boosts at highway speed (25 m/s)",
          lc.last_ff > 1.2, f"ff={lc.last_ff:.3f}")

    # under-delivery (achieved < commanded) must RAISE the learned FF
    lc = LateralController(cfg, understeer_ff=1.30)
    d = mk(25.0)
    d.pose[5] = 0.8 * 0.01 * 25.0        # k_meas = 0.8 * k_cmd
    lc._k_recent.extend([0.01] * 10)     # steady high-speed curve
    for _ in range(200):
        lc._learn_understeer(d, 25.0, False)
    check("under-delivery raises learned FF", lc.understeer_ff > 1.32,
          f"ff={lc.understeer_ff:.3f}")

    # over-delivery must LOWER it
    lc = LateralController(cfg, understeer_ff=1.50)
    d = mk(25.0)
    d.pose[5] = 1.2 * 0.01 * 25.0        # k_meas = 1.2 * k_cmd
    lc._k_recent.extend([0.01] * 10)
    for _ in range(200):
        lc._learn_understeer(d, 25.0, False)
    check("over-delivery lowers learned FF", lc.understeer_ff < 1.48,
          f"ff={lc.understeer_ff:.3f}")

    # gate blocks learning at city-turn speed
    lc = LateralController(cfg, understeer_ff=1.30)
    d = mk(8.0)
    d.pose[5] = 0.8 * 0.01 * 8.0
    lc._k_recent.extend([0.01] * 10)
    for _ in range(200):
        lc._learn_understeer(d, 8.0, False)
    check("no understeer learning below the gate",
          abs(lc.understeer_ff - 1.30) < 1e-6, f"ff={lc.understeer_ff:.3f}")


def test_action_head_override():
    # big-model direct action head: lateral must track action[0]/v^2
    # (rate-limited), ignoring the plan psi math
    from simsteer.core.control import LateralController
    cfg = ControllerConfig()
    lc = LateralController(cfg)
    d = mk(20.0)                       # plan says dead straight
    d.action = np.array([0.004 * 400.0, 0.0], dtype=np.float32)  # k=0.004
    for _ in range(40):                # let the rate limiter converge
        lc.compute(d, 20.0)
    check("action head drives curvature despite straight plan",
          abs(lc.last_curvature - 0.004) < 5e-4,
          f"k={lc.last_curvature:.4f} (want 0.004)")
    lc.reset()
    d2 = mk(20.0)                      # action=None -> plan path: ~0
    lc.compute(d2, 20.0)
    check("no action head -> plan path unaffected",
          abs(lc.last_curvature) < 5e-4)


if __name__ == "__main__":
    test_corner_scanner()
    test_speed_scale_persistence()
    test_brake_ramp_responsive()
    test_fb_integrator_gate()
    test_no_vision_intervention()
    test_curvature_clip()
    test_min_stable_delay()
    test_understeer_ff_gate_and_learn()
    test_action_head_override()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES: {FAILS}'}")
    sys.exit(1 if FAILS else 0)
