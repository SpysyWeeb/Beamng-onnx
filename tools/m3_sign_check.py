#!/usr/bin/env python3
"""M3 pre-flight: empirically verify BeamNG's control-sign conventions.

Before the first closed-loop run we need to KNOW (not assume):
  1. Does vehicle.control(steering=+x) turn the car LEFT or RIGHT?
  2. What sign does electrics `steering` / `steering_input` report for it?
  3. Roughly what road-wheel angle does full-ish axis produce
     (checks the LiveParams seed a≈1.6)?

openpilot's convention: positive curvature = LEFT turn (z-up, CCW+).
BeamNG world is z-up with heading atan2(dir_y, dir_x), so heading
INCREASING = CCW = LEFT there too.

Usage: .venv\\Scripts\\python tools/m3_sign_check.py
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from beamng.world import BeamNGOnnxWorld


def unwrap(prev: float, cur: float) -> float:
    d = cur - prev
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def run_pulse(world: BeamNGOnnxWorld, axis: float, secs: float) -> dict:
    """Hold `axis` steering with gentle throttle; integrate heading."""
    tel0 = world.poll_telemetry()
    heading = tel0["heading_rad"]
    total_dyaw = 0.0
    v_samples, si_samples, sd_samples = [], [], []
    t_end = time.monotonic() + secs
    while time.monotonic() < t_end:
        world.apply(steering=axis, throttle=0.25, brake=0.0)
        time.sleep(0.1)
        tel = world.poll_telemetry()
        total_dyaw += unwrap(heading, tel["heading_rad"])
        heading = tel["heading_rad"]
        v_samples.append(tel["v_ego"])
        si_samples.append(tel["steering_input"])
        sd_samples.append(tel["steering_deg"])
    v = float(np.mean(v_samples))
    yaw_rate = total_dyaw / secs
    # curvature k = yaw_rate / v; road-wheel angle ~ atan(k * wheelbase)
    k = yaw_rate / max(v, 0.5)
    wheel = math.atan(k * 2.9)  # ~sedan wheelbase
    return {
        "axis": axis, "dyaw_deg": math.degrees(total_dyaw),
        "yaw_rate": yaw_rate, "v": v, "curvature": k, "wheel_rad": wheel,
        "steering_input": float(np.mean(si_samples)),
        "steering_deg": float(np.mean(sd_samples)),
    }


def main() -> int:
    world = BeamNGOnnxWorld()
    try:
        print("[sign] rolling up to speed ...", flush=True)
        t_end = time.monotonic() + 5.0
        while time.monotonic() < t_end:
            world.apply(steering=0.0, throttle=0.4, brake=0.0)
            time.sleep(0.1)

        results = []
        for axis in (+0.3, -0.3):
            print(f"[sign] pulse steering={axis:+.1f} ...", flush=True)
            r = run_pulse(world, axis, 2.5)
            results.append(r)
            direction = "LEFT (CCW)" if r["dyaw_deg"] > 0 else "RIGHT (CW)"
            print(f"  axis {r['axis']:+.2f}: turned {direction} "
                  f"({r['dyaw_deg']:+.1f} deg over pulse), v={r['v']:.1f} m/s")
            print(f"    curvature={r['curvature']:+.4f} 1/m  "
                  f"wheel~{r['wheel_rad']:+.3f} rad")
            print(f"    electrics: steering_input={r['steering_input']:+.3f}  "
                  f"steering_deg={r['steering_deg']:+.1f}")
            # straighten + settle between pulses
            t_end = time.monotonic() + 2.0
            while time.monotonic() < t_end:
                world.apply(steering=0.0, throttle=0.2, brake=0.0)
                time.sleep(0.1)

        world.apply(0.0, 0.0, 1.0)
        print("\n[sign] VERDICT:")
        r = results[0]
        turn_left_on_pos = r["dyaw_deg"] > 0
        print(f"  control(steering=+1) turns "
              f"{'LEFT' if turn_left_on_pos else 'RIGHT'}")
        print(f"  openpilot +curvature = LEFT -> LiveParams 'a' sign should "
              f"be {'POSITIVE' if turn_left_on_pos else 'NEGATIVE'}")
        # implied a magnitude: axis / wheel (at low speed)
        wheels = [abs(x["wheel_rad"]) for x in results]
        if min(wheels) > 1e-3:
            imp_a = np.mean([abs(x["axis"]) / abs(x["wheel_rad"])
                             for x in results])
            sign = 1.0 if turn_left_on_pos else -1.0
            print(f"  implied |a| ~ {imp_a:.2f} -> seed a = {sign*imp_a:.2f}")
        si_sign_matches = (results[0]["steering_input"] > 0)
        print(f"  electrics steering_input sign "
              f"{'MATCHES' if si_sign_matches else 'OPPOSES'} commanded axis")
        return 0
    finally:
        world.apply(0.0, 0.0, 1.0)
        world.close()


if __name__ == "__main__":
    sys.exit(main())
