#!/usr/bin/env python3
"""M3 unattended validation — calibrate, engage, drive, score.

Runs the control panel's App headless (no window):
  1. Auto-calibration slalom until LiveParams is trusted.
  2. Engage (lateral + longitudinal), speed cap from --cap.
  3. Drive for --seconds, sampling stability metrics.
  4. Report: lane-center offset, inner-lane confidence, speed, steering
     activity. FAILs loudly on speed collapse (crash) or lost lanes.

Usage: .venv/bin/python3 tools/m3_auto_test.py [--seconds 120] [--cap 20]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

np.seterr(divide="ignore", invalid="ignore")

from simsteer.core.constants import DESIRE_LEN  # noqa: E402
from tools.control_panel import App, WARMUP_FRAMES  # noqa: E402


def step_once(app: App) -> tuple[object, float] | None:
    t0 = time.monotonic()
    bgr = app.world.poll_bgr()
    if bgr is None:
        time.sleep(0.02)
        return None
    app.frame_idx += 1
    img, big = app.queue.push(bgr, app.calib)
    d = app._decode(img, big, None)
    v = app.tel.snapshot()["v_ego"]
    app.control_tick(d, v, 0.05)
    el = time.monotonic() - t0
    app.hz = 1.0 / el if el > 0 else 0.0
    time.sleep(max(0.0, 0.05 - el))
    return d, v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--cap", type=float, default=20.0, help="speed cap m/s")
    args_ns = types.SimpleNamespace(split=False, scale=1.0)
    args = ap.parse_args()

    app = App(args_ns)
    app.cfg.max_speed_mps = args.cap
    try:
        print("[test] warmup ...", flush=True)
        while app.frame_idx < WARMUP_FRAMES:
            step_once(app)

        print("[test] calibration slalom ...", flush=True)
        app.start_cal()
        while app.cal_active:
            step_once(app)
        n = max(app.lp.samples, app.lp.session_samples)
        print(f"[test] cal result: a={app.lp.a_linear:+.3f} "
              f"b={app.lp.b_quad:+.5f} c={app.lp.x[2]:+.4f} "
              f"samples={n} trusted={app.lp.trusted()}", flush=True)
        if not app.lp.trusted():
            print("[test] FAIL: learner not trusted after calibration")
            return 1

        # AI is still driving post-cal; engage takes over from motion.
        print("[test] ENGAGING (lat+long) ...", flush=True)
        app.engage()
        deadline = time.monotonic() + args.seconds
        offs, probs, vs, steers = [], [], [], []
        peak_v = 0.0
        t_report = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            r = step_once(app)
            if r is None:
                continue
            d, v = r
            peak_v = max(peak_v, v)
            vs.append(v)
            steers.append(app.last_steer)
            # lane-center offset: midpoint of the inner lane pair at x=0.
            # lane_lines[:,0,0] = lateral y at the first x index.
            inner = d.lane_lines[1:3, 0, 0]
            offs.append(float(np.mean(inner)))
            probs.append(d.lane_lines_prob[1:3].copy())
            if peak_v > 8.0 and v < 2.0:
                print(f"[test] FAIL: speed collapsed ({v:.1f} m/s after "
                      f"peaking {peak_v:.1f}) — crash/stuck", flush=True)
                break
            if time.monotonic() > t_report:
                t_report += 10.0
                print(f"  t={args.seconds - (deadline - time.monotonic()):5.0f}s"
                      f"  v={v:5.1f} m/s  steer={app.last_steer:+.2f}"
                      f"  off={offs[-1]:+.2f} m"
                      f"  lanes={np.round(d.lane_lines_prob[1:3], 2)}",
                      flush=True)
        app.disengage("test end")

        offs_a = np.abs(np.array(offs[20:]))   # skip launch
        probs_a = np.stack(probs[20:])
        vs_a = np.array(vs[20:])
        print("\n[test] REPORT")
        print(f"  frames engaged: {len(offs)}   mean v: {vs_a.mean():.1f} m/s "
              f"(peak {peak_v:.1f})")
        print(f"  lane-center offset |mean|: {offs_a.mean():.2f} m   "
              f"p95: {np.percentile(offs_a, 95):.2f} m   "
              f"max: {offs_a.max():.2f} m")
        print(f"  inner-lane prob mean: {probs_a.mean():.2f}")
        print(f"  steering |mean|: {np.abs(np.array(steers[20:])).mean():.3f}   "
              f"max: {np.abs(np.array(steers[20:])).max():.2f}")
        ok = offs_a.mean() < 1.0 and probs_a.mean() > 0.25 and vs_a.mean() > 5.0
        print(f"  VERDICT: {'PASS' if ok else 'WEAK — inspect'}")
        return 0 if ok else 2
    finally:
        app.world.apply(0.0, 0.0, 0.0)
        app.tel.stop()
        app.world.close()


if __name__ == "__main__":
    sys.exit(main())
