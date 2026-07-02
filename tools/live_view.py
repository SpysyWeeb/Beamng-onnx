#!/usr/bin/env python3
"""Live viewer — watch the model plan on BeamNG in real time.

Runs the M1 pipeline continuously and shows SimSteer's own overlay
(green plan line + accel-colored path wedge, lane lines, road edges,
horizon, timestep markers) in a window, with a small status line for
lane confidence / curvature / reach / rate.

Drive the car yourself in BeamNG, or pass --ai to let BeamNG's AI drive.

Usage (inside the distrobox, BeamNG running via launch_beamng.sh):
    .venv/bin/python3 tools/live_view.py [--ai] [--scale 0.8]

Keys in the window:  q or ESC — quit
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

# upstream calibration projections divide by depth and mask the result with
# np.where(valid, ...) — the transient inf/nan warnings are expected noise
np.seterr(divide="ignore", invalid="ignore")

from simsteer.core.calibration import Calibration
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode, desired_curvature
from simsteer.core.preprocess import FrameQueue
from simsteer.ui.overlay import draw_overlay

from beamng.world import BeamNGOnnxWorld, CAM_W, CAM_H, CAM_FOV_H_DEG, CAM_HEIGHT_M, CAM_LATERAL_SIGN

WINDOW = "Beamng-onnx — model plan"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", action="store_true", help="BeamNG AI drives")
    ap.add_argument("--scale", type=float, default=0.8, help="window scale")
    args = ap.parse_args()

    calib = Calibration(image_w=CAM_W, image_h=CAM_H,
                        fov_h_deg=CAM_FOV_H_DEG, height_m=CAM_HEIGHT_M,
                        lateral_sign=CAM_LATERAL_SIGN)

    print("[view] loading model (CPU) ...", flush=True)
    model = DrivingModel(providers=["CPUExecutionProvider"],
                         policy_providers=["CPUExecutionProvider"])
    queue = FrameQueue()

    world = BeamNGOnnxWorld()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(CAM_W * args.scale), int(CAM_H * args.scale))

    try:
        if args.ai:
            world.ai_drive(25.0)
            print("[view] AI driving on", flush=True)
        else:
            print("[view] manual mode — drive the car in BeamNG", flush=True)

        hz = 0.0
        tel_t = 0.0
        v_ego = 0.0
        while True:
            t0 = time.monotonic()
            bgr = world.poll_bgr()
            if bgr is None:
                time.sleep(0.02)
                continue

            img, big_img = queue.push(bgr, calib)
            d = decode(*model.step(img, big_img))

            # telemetry at ~2 Hz is plenty for the HUD
            if t0 - tel_t > 0.5:
                v_ego = world.poll_telemetry()["v_ego"]
                tel_t = t0

            out = draw_overlay(bgr, d, calib)
            k = desired_curvature(d.plan, v_ego=max(v_ego, 1.0))
            reach = d.plan[-1, 0]
            probs = np.round(d.lane_lines_prob, 2)
            status = (f"{v_ego*2.237:4.0f} mph   lanes {probs}   "
                      f"curv {k:+.4f}   reach {reach:5.1f} m   {hz:4.1f} Hz")
            cv2.putText(out, status, (12, CAM_H - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, status, (12, CAM_H - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            dt = time.monotonic() - t0
            hz = 1.0 / dt if dt > 0 else 0.0
            time.sleep(max(0.0, 0.05 - dt))
        return 0
    finally:
        world.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
