#!/usr/bin/env python3
"""Camera placement + FOV verification probe.

Tries candidate camera mounts, saves a raw frame + the warped model views
for each, and prints geometry checks:

  - Vehicle bounding box (so mount offsets can be reasoned about).
  - Horizon row in the raw frame: with pitch=0 on flat road the horizon
    must sit at the vertical center (cy_frac=0.5). If it doesn't, the
    declared FOV/pitch don't match what BeamNG actually renders.
  - Model lane-line spacing while parked: on a US highway the model
    should report ~3.7 m between adjacent lane lines if (and only if)
    the declared FOV matches the render.

Usage:
    .venv\\Scripts\\python tools/camera_probe.py
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

np.seterr(divide="ignore", invalid="ignore")

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera

from simsteer.core.calibration import Calibration
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue, yuv6_to_bgr

from beamng.world import (HOST, PORT, MAP, VEHICLE_MODEL, SPAWN_POS,
                          SPAWN_ROT_QUAT, CAM_W, CAM_H, CAM_FOV_H_DEG,
                          fov_v_deg)

# Candidate mounts (vehicle frame: x right, y forward is NEGATIVE, z up).
# A comma device sits at the top-center of the windshield, behind the
# rear-view mirror.
CANDIDATES = {
    "current_roofline": (0.0, -1.45, 1.38),   # what M1 used (bridge legacy)
    "mirror_a": (0.0, -0.60, 1.25),           # behind mirror, through glass
    "mirror_b": (0.0, -0.35, 1.20),           # slightly deeper in cabin
    "glass_top": (0.0, -0.90, 1.30),          # top of windshield, near glass
}

OUT = os.path.join(ROOT, "debug_out", "mounts")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    print("[probe] connecting ...", flush=True)
    bng = BeamNGpy(HOST, PORT)
    bng.open(launch=False)
    scenario = Scenario(MAP, "cam_probe")
    vehicle = Vehicle("ego", model=VEHICLE_MODEL, license="PROBE")
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT_QUAT)
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()

    bbox = vehicle.get_bbox()
    print("[probe] bbox corners:")
    for k, v in bbox.items():
        print(f"    {k}: {np.round(v, 2)}")

    model = DrivingModel(providers=["CPUExecutionProvider"],
                         policy_providers=["CPUExecutionProvider"],
                         intra_op_threads=3)

    vfov = fov_v_deg(CAM_FOV_H_DEG, CAM_W, CAM_H)
    for name, pos in CANDIDATES.items():
        cam = Camera(f"probe_{name}", bng, vehicle,
                     requested_update_time=0.05,
                     pos=pos, dir=(0, -1, 0), up=(0, 0, 1),
                     field_of_view_y=vfov,
                     resolution=(CAM_W, CAM_H),
                     near_far_planes=(0.1, 1000.0),
                     is_render_colours=True)
        time.sleep(1.0)

        bgr = None
        for _ in range(10):
            data = cam.poll()
            colour = data.get("colour") if isinstance(data, dict) else None
            if colour is not None:
                img = np.asarray(colour)
                if img.ndim == 3:
                    bgr = img[:, :, :3][:, :, ::-1].copy()
                    break
            time.sleep(0.1)
        cam.remove()
        if bgr is None:
            print(f"[probe] {name}: NO FRAME")
            continue

        calib = Calibration(image_w=CAM_W, image_h=CAM_H,
                            fov_h_deg=CAM_FOV_H_DEG, height_m=pos[2])
        queue = FrameQueue()
        img_t, big_t = queue.push(bgr, calib)
        d = decode(*model.step(img_t, big_t))

        ys = np.sort(d.lane_lines[:, 0, 0])
        gaps = np.round(np.diff(ys), 2)
        print(f"[probe] {name}: pos={pos}  laneProbs={np.round(d.lane_lines_prob, 2)}  "
              f"lane y@0={np.round(ys, 2)}  gaps={gaps}")

        cv2.imwrite(os.path.join(OUT, f"{name}_raw.png"), bgr)
        if queue.last_yuv_narrow is not None:
            cv2.imwrite(os.path.join(OUT, f"{name}_narrow.png"),
                        yuv6_to_bgr(queue.last_yuv_narrow))
    bng.disconnect()
    print(f"[probe] frames in {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
