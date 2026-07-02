#!/usr/bin/env python3
"""Metric-scale audit: the model's perceived motion vs ground truth.

Upstream simsteer calls wrong FOV "the #1 cause of the plan veering off
the road" and checks it via the ratio of the MODEL's perceived forward
speed (pose vx) to true speed. We never ran that on BeamNG — and the
same scale error would corrupt everything downstream at once: the rack
fit (wheel is synthesized from model yaw), commanded curvature (curves
cut too sharp), and the e2e plan's speed (running slow / stopping).

BeamNG AI drives; we compare per frame:
  - model pose[0] (vx, m/s)      vs telemetry wheelspeed
  - model pose[5] (yaw rate)     vs d(heading)/dt from vehicle state
  - model lane width             vs the ~3.7 m US standard

Healthy: vx ratio 1.00 +/- 0.05 (upstream: >1.05 = declared FOV too
high, <0.95 = too low).

Usage: .venv/bin/python3 tools/geometry_check.py [--seconds 75]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

np.seterr(divide="ignore", invalid="ignore")

from simsteer.core.calibration import Calibration
from simsteer.core.preprocess import FrameQueue
from simsteer.core.supercombo import SupercomboModel
from beamng.world import (BeamNGOnnxWorld, CAM_W, CAM_H, CAM_FOV_H_DEG,
                          CAM_HEIGHT_M, CAM_LATERAL_SIGN)


def unwrap(d: float) -> float:
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=75.0)
    ap.add_argument("--stock-calib", action="store_true",
                    help="ignore livecalib-learned pitch/yaw (A/B test)")
    args = ap.parse_args()

    if args.stock_calib:
        calib = Calibration(image_w=CAM_W, image_h=CAM_H,
                            fov_h_deg=CAM_FOV_H_DEG, height_m=CAM_HEIGHT_M,
                            lateral_sign=CAM_LATERAL_SIGN)
    else:
        calib = Calibration.load(game="beamng") if hasattr(Calibration, "load") \
            else Calibration(image_w=CAM_W, image_h=CAM_H,
                             fov_h_deg=CAM_FOV_H_DEG, height_m=CAM_HEIGHT_M,
                             lateral_sign=CAM_LATERAL_SIGN)

    world = BeamNGOnnxWorld()
    model = SupercomboModel(
        providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
        intra_op_threads=3)
    queue = FrameQueue()
    world.ai_drive(15.0)
    print("[geo] AI driving; warmup ...", flush=True)
    t_end = time.monotonic() + 8.0
    while time.monotonic() < t_end:
        bgr = world.poll_bgr()
        if bgr is not None:
            img, big = queue.push(bgr, calib)
            model.step(img, big)
        time.sleep(0.02)

    vx_pairs, yaw_pairs, gaps = [], [], []
    prev_heading, prev_t = None, None
    gt_yaw_lpf = 0.0
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        bgr = world.poll_bgr()
        if bgr is None:
            time.sleep(0.02)
            continue
        img, big = queue.push(bgr, calib)
        d = model.decode(model.step(img, big))
        tel = world.poll_telemetry()
        v = tel["v_ego"]

        if prev_heading is not None:
            dt_h = t0 - prev_t
            if dt_h > 1e-3:
                gt = unwrap(tel["heading_rad"] - prev_heading) / dt_h
                gt_yaw_lpf += 0.2 * (gt - gt_yaw_lpf)
        prev_heading, prev_t = tel["heading_rad"], t0

        if v > 8.0:
            vx_pairs.append((float(d.pose[0]), v))
            # model frame: y right-positive, heading CCW-positive ->
            # ground-truth right turn is NEGATIVE heading rate
            yaw_pairs.append((float(d.pose[5]), -gt_yaw_lpf))
            ys = np.sort(d.lane_lines[1:3, 0, 0])
            if d.lane_lines_prob[1] > 0.5 and d.lane_lines_prob[2] > 0.5:
                gaps.append(float(ys[1] - ys[0]))
        el = time.monotonic() - t0
        time.sleep(max(0.0, 0.05 - el))

    world.ai_disable()
    world.apply(0.0, 0.0, 0.5)
    world.close()

    vx = np.array(vx_pairs)
    print(f"\n[geo] frames used: {len(vx)}   calib pitch={calib.pitch_deg:.2f} "
          f"yaw={calib.yaw_deg:.2f} h={calib.height_m:.2f} "
          f"fov={calib.fov_h_deg:.1f}")
    ratios = vx[:, 0] / vx[:, 1]
    print(f"[geo] SPEED  vx_model/v_true: mean {np.mean(ratios):.3f}  "
          f"median {np.median(ratios):.3f}  p5 {np.percentile(ratios,5):.3f} "
          f"p95 {np.percentile(ratios,95):.3f}")
    yw = np.array([p for p in yaw_pairs if abs(p[1]) > 0.008])
    if len(yw) > 20:
        k = float(np.sum(yw[:, 0] * yw[:, 1]) / np.sum(yw[:, 1] ** 2))
        print(f"[geo] YAW    model/true slope: {k:.3f}   (n={len(yw)})")
    else:
        print(f"[geo] YAW    too few turning samples (n={len(yw)})")
    if gaps:
        print(f"[geo] LANE   width mean {np.mean(gaps):.2f} m  "
              f"(std {np.std(gaps):.2f}; US standard 3.7)")
    print("[geo] healthy: speed ratio 1.00+/-0.05; >1.05 = declared FOV "
          "too high, <0.95 = too low (upstream fov doc)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
