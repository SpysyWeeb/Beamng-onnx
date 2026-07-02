#!/usr/bin/env python3
"""M1 — live BeamNG frames through SimSteer's warp + model, no control.

Feeds ~N seconds of camera frames through the untouched SimSteer core
(FrameQueue warp/YUV/stacking -> DrivingModel ONNX -> decode) and reports
what the model sees. Saves debug images so the warp can be eyeballed:

    debug_out/raw.png     — the camera frame as captured
    debug_out/narrow.png  — the medmodel view the model actually gets
    debug_out/wide.png    — the sbigmodel view

Usage (inside the distrobox, BeamNG running via launch_beamng.sh):
    .venv/bin/python3 tools/m1_beamng_frame.py [--seconds 6] [--ai]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from simsteer.core.calibration import Calibration
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode, desired_curvature
from simsteer.core.preprocess import FrameQueue, yuv6_to_bgr

from beamng.world import BeamNGOnnxWorld, CAM_W, CAM_H, CAM_FOV_H_DEG, CAM_HEIGHT_M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--ai", action="store_true",
                    help="let BeamNG's AI drive during capture (gives the model motion)")
    args = ap.parse_args()

    calib = Calibration(image_w=CAM_W, image_h=CAM_H,
                        fov_h_deg=CAM_FOV_H_DEG, height_m=CAM_HEIGHT_M)

    print("[m1] loading model (CPU) ...", flush=True)
    model = DrivingModel(providers=["CPUExecutionProvider"],
                         policy_providers=["CPUExecutionProvider"])
    queue = FrameQueue()

    world = BeamNGOnnxWorld()
    try:
        if args.ai:
            world.ai_drive(25.0)
            print("[m1] AI driving on", flush=True)

        deadline = time.monotonic() + args.seconds
        frames = 0
        last_dec = None
        lane_prob_hist: list[np.ndarray] = []
        t_step = []
        raw = None
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            bgr = world.poll_bgr()
            if bgr is None:
                time.sleep(0.02)
                continue
            raw = bgr
            img, big_img = queue.push(bgr, calib)
            vis, pol = model.step(img, big_img)
            last_dec = decode(vis, pol)
            lane_prob_hist.append(last_dec.lane_lines_prob.copy())
            frames += 1
            t_step.append(time.monotonic() - t0)
            # pace to ~20 Hz
            time.sleep(max(0.0, 0.05 - (time.monotonic() - t0)))

        if last_dec is None:
            print("[m1] no frames received from BeamNG", flush=True)
            return 1

        probs = np.stack(lane_prob_hist)
        tel = world.poll_telemetry()
        d = last_dec
        plan_xy = d.plan[:, :2]                      # (33, [x fwd, y left+])
        k = desired_curvature(d.plan, v_ego=max(tel["v_ego"], 1.0))

        print(f"\n[m1] frames={frames}  mean step={np.mean(t_step)*1000:.1f} ms "
              f"({1/np.mean(t_step):.1f} Hz capable)")
        print(f"[m1] v_ego={tel['v_ego']:.1f} m/s  steering={tel['steering_deg']:.1f} deg")
        print(f"[m1] lane_lines_prob last={np.round(d.lane_lines_prob, 2)}  "
              f"mean={np.round(probs.mean(axis=0), 2)}  max={np.round(probs.max(axis=0), 2)}")
        # lane_lines: (4, 33, [y lateral, z height]) sampled along X_IDXS
        print(f"[m1] lane_lines y@0m={np.round(d.lane_lines[:, 0, 0], 2)}")
        print(f"[m1] plan: x reach={plan_xy[-1, 0]:.1f} m  |y| max={np.abs(plan_xy[:,1]).max():.2f} m")
        print(f"[m1] desired_curvature={k:+.5f} (1/m)")
        print(f"[m1] lead prob={float(d.lead_prob[0]) if np.ndim(d.lead_prob) else float(d.lead_prob):.2f}")

        out = os.path.join(ROOT, "debug_out")
        os.makedirs(out, exist_ok=True)
        cv2.imwrite(os.path.join(out, "raw.png"), raw)
        if queue.last_yuv_narrow is not None:
            cv2.imwrite(os.path.join(out, "narrow.png"), yuv6_to_bgr(queue.last_yuv_narrow))
        if queue.last_yuv_wide is not None:
            cv2.imwrite(os.path.join(out, "wide.png"), yuv6_to_bgr(queue.last_yuv_wide))
        print(f"[m1] debug images -> {out}/raw.png, narrow.png, wide.png", flush=True)
        return 0
    finally:
        world.close()


if __name__ == "__main__":
    sys.exit(main())
