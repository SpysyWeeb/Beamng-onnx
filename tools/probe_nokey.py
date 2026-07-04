#!/usr/bin/env python3
"""Settle what BeamNG.drive exposes to beamngpy WITHOUT a tech.key.

We already know (START's handshake) the socket opens. This probe walks
the rest and ends on the architectural fork:
  1-4  classical features — connect, find car, speed (Electrics/State),
       control (vehicle.control). These are license-free IF the socket
       opens, which it does.
  5    the automated Camera sensor — THE fork:
         * renders  -> full tech mode works unlicensed, no screen capture
         * refused  -> camera is tech-gated, no-key path must screen-grab

Steps 1-4 are non-destructive; step 5 briefly creates and removes a
small camera on the player's car.

Usage (BeamNG.drive, no tech.key):
  # you already have the game running:
  .venv\\Scripts\\python tools/probe_nokey.py
  # or let the probe launch it for you:
  .venv\\Scripts\\python tools/probe_nokey.py --launch
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launcher_core import (BIN_RELS, TECH_PORT, detect_beamng,  # noqa: E402
                           find_binary, launch_game_process, port_open)


def _ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def _no(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def launch_game(install: str) -> bool:
    """Boot BeamNG the same way launcher_core does, then wait for the
    tech port. Works for BeamNG.drive.x64 too (find_binary accepts it)."""
    exe = find_binary(install)
    if exe is None:
        _no(f"no {'/'.join(BIN_RELS)} binary under {install!r}")
        return False
    print(f"launching {os.path.basename(exe)} (tech server on "
          f"{TECH_PORT}) ...")
    launch_game_process(exe, install)
    t_end = time.monotonic() + 240
    while time.monotonic() < t_end and not port_open():
        time.sleep(2.0)
    if not port_open():
        _no("gave up waiting for the tech port — did the game boot?")
        return False
    _ok(f"tech port {TECH_PORT} is open")
    return True


def probe() -> int:
    """Return 0 if the license-free hybrid is viable, 1 otherwise."""
    from beamngpy import BeamNGpy
    from beamngpy.sensors import Electrics, State

    print("\n1) socket handshake (the load-bearing question) ...")
    try:
        bng = BeamNGpy("127.0.0.1", TECH_PORT)
        bng.open(launch=False)
    except Exception as exc:
        _no(f"beamngpy could not connect: {exc}")
        print("\nVERDICT: the beamngpy socket is license-gated on this "
              "build.\n         Screen mode must stay beamngpy-free "
              "(virtual wheel, no\n         beamngpy speed). The hybrid "
              "is NOT possible here.")
        return 1
    _ok("connected — beamngpy talks to the game without a tech.key")

    try:
        print("\n2) find a car the player is sitting in ...")
        current = bng.vehicles.get_current(include_config=False)
        if not current:
            _no("no vehicle found — get in a car in-game, then re-run")
            bng.disconnect()
            return 1
        vid = next(iter(current))
        veh = current[vid]
        _ok(f"found vehicle {vid!r} (model={getattr(veh, 'model', '?')})")

        print("\n3) classical telemetry (speed for the model) ...")
        veh.connect(bng)
        veh.sensors.attach("electrics", Electrics())
        veh.sensors.attach("state", State())
        veh.sensors.poll()
        el = veh.sensors["electrics"]
        st = veh.state or {}
        wheelspeed = el.get("wheelspeed")
        vel = st.get("vel")
        if wheelspeed is None and vel is None:
            _no("no speed field came back")
            bng.disconnect()
            return 1
        _ok(f"Electrics.wheelspeed = {wheelspeed} m/s")
        _ok(f"State.vel = {vel} (m/s vector)")
        _ok(f"steering_input={el.get('steering_input')} "
            f"throttle={el.get('throttle_input')} "
            f"brake={el.get('brake_input')}")

        print("\n4) control authority (a brief throttle blip) ...")
        veh.control(throttle=0.3)
        time.sleep(0.6)
        veh.control(throttle=0.0, brake=1.0)
        time.sleep(0.4)
        veh.control(brake=0.0)
        _ok("vehicle.control() accepted throttle + brake")
    except Exception as exc:
        _no(f"a classical feature failed: {exc}")
        print("\nVERDICT: the socket opened but a classical feature was "
              "refused —\n         unexpected; capture the traceback "
              "above.")
        try:
            bng.disconnect()
        except Exception:
            pass
        return 1

    # 5) THE architectural fork: does the automated Camera sensor work
    #    without a tech.key? If yes, full tech mode works unlicensed and
    #    we never need screen capture. If it raises, the camera is gated
    #    and the no-key path must screen-capture the camera.
    print("\n5) automated Camera sensor (the fork) ...")
    camera_ok = False
    try:
        from beamngpy.sensors import Camera
        cam = Camera(
            "probecam", bng, veh,
            pos=(0.0, -0.5, 1.3), dir=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0),
            resolution=(320, 160), field_of_view_y=50,
            near_far_planes=(0.1, 1000.0),
            is_render_colours=True, is_render_annotations=False,
            is_render_depth=False,
            is_streaming=False, is_using_shared_memory=False)
        data = cam.poll()
        colour = data.get("colour") if isinstance(data, dict) else None
        cam.remove()
        if colour is None:
            _no("Camera created but returned no colour frame")
        else:
            camera_ok = True
            _ok(f"Camera rendered a frame ({getattr(colour, 'size', '?')} "
                "px) WITHOUT a tech.key")
    except Exception as exc:
        _no(f"Camera refused: {exc}")

    bng.disconnect()
    if camera_ok:
        print("\nVERDICT: FULL TECH MODE works WITHOUT a tech.key.")
        print("  Socket + control + speed + CAMERA all render unlicensed.")
        print("  No screen capture needed — the tech.key toggle can just")
        print("  gate whether we require the key, not what features we use.")
    else:
        print("\nVERDICT: HYBRID path (camera is tech-gated).")
        print("  Socket + control + speed work unlicensed, but the Camera")
        print("  is refused. No-key path: beamngpy scenario/control/speed +")
        print("  screen-captured camera (driver cam -> hood cam).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--launch", action="store_true",
                    help="boot BeamNG first (use with BeamNG.drive, no key)")
    ap.add_argument("--path", default=detect_beamng(),
                    help="BeamNG install dir (default: auto-detect)")
    args = ap.parse_args()

    if args.launch:
        if not args.path or not os.path.isdir(args.path):
            _no(f"BeamNG install not found: {args.path!r}")
            return 1
        if not port_open() and not launch_game(args.path):
            return 1
    elif not port_open():
        _no(f"nothing on port {TECH_PORT} — start BeamNG with "
            f"-tcom -tport {TECH_PORT}, or pass --launch")
        return 1

    return probe()


if __name__ == "__main__":
    raise SystemExit(main())
