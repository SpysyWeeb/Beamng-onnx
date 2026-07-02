"""Minimal BeamNG world: connect to a running BeamNG.tech instance, spawn a
scenario with one vehicle and ONE wide camera, and poll frames/telemetry.

Unlike a real comma device (two physical cameras) or the openpilot sim
(two rendered sensors), SimSteer's preprocess derives both model views
(narrow medmodel + wide sbigmodel) from a single source frame — so one
camera render is all we need. The source must cover at least the wide
view's ~59 deg HFOV; we render 100 deg so the narrow ~31 deg crop still
maps ~516 source px onto the model's 512-wide input at 1664x832.

BeamNG must already be running with the tech server:
    bash launch_beamng.sh     (from this repo)
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Electrics

HOST = "127.0.0.1"
PORT = 64256

MAP = "west_coast_usa"
VEHICLE_MODEL = "bastion"
# Highway spawn, same map/pose the author has used for months of testing.
SPAWN_POS = (-829.5, -499.0, 106.8)
SPAWN_ROT_QUAT = (0.0, 0.0, -0.9272, 0.3746)

# Camera mount relative to the vehicle (BeamNG vehicles face -Y): top-center
# of the windshield glass, where a comma device is installed. Chosen from a
# 4-mount probe (tools/camera_probe.py, 2026-07-02): shows the hood dome at
# the frame bottom like real comma footage, with none of the glass-streak /
# dashboard / hood-LOD artifacts the deeper in-cabin mounts render, and the
# tightest model lane-width consistency (3.46-3.49 m gaps).
CAM_POS = (0.0, -0.90, 1.30)
CAM_DIR = (0.0, -1.0, 0.0)
CAM_HEIGHT_M = 1.30  # approx height above road; LiveCalib refines online

# SimSteer's Calibration defaults lateral_sign=-1 (right for ETS2/AC/Forza's
# screen-capture + gamepad conventions). Our beamngpy render is NOT mirrored
# (verified: warped model input matches raw orientation) and the model's
# native y is right-positive, so the overlay projection needs +1 — with -1
# the drawn plan/lanes mirror and "curve the wrong way".
CAM_LATERAL_SIGN = 1.0

CAM_W, CAM_H = 1664, 832
CAM_FOV_H_DEG = 100.0


def fov_v_deg(fov_h_deg: float, w: int, h: int) -> float:
    """Square pixels: tan(v/2) = tan(h/2) * H/W. beamngpy wants vertical."""
    return math.degrees(2 * math.atan(
        math.tan(math.radians(fov_h_deg) / 2) * h / w))


class BeamNGOnnxWorld:
    def __init__(self, host: str = HOST, port: int = PORT):
        print(f"[world] connecting to BeamNG at {host}:{port} ...", flush=True)
        import threading
        self._ctl_lock = threading.Lock()
        self.bng = BeamNGpy(host, port)
        self.bng.open(launch=False)

        print(f"[world] loading scenario {MAP} ...", flush=True)
        scenario = Scenario(MAP, "beamng_onnx")
        self.vehicle = Vehicle("ego", model=VEHICLE_MODEL, license="ONNX")
        scenario.add_vehicle(self.vehicle, pos=SPAWN_POS, rot_quat=SPAWN_ROT_QUAT)
        scenario.make(self.bng)
        self.bng.scenario.load(scenario)
        self.bng.scenario.start()

        self.vehicle.sensors.attach("electrics", Electrics())

        self.camera = Camera(
            "onnxcam", self.bng, self.vehicle,
            requested_update_time=0.05,  # 20 Hz — never render more than we consume
            pos=CAM_POS, dir=CAM_DIR, up=(0, 0, 1),
            field_of_view_y=fov_v_deg(CAM_FOV_H_DEG, CAM_W, CAM_H),
            resolution=(CAM_W, CAM_H),
            near_far_planes=(0.1, 1000.0),
            is_render_colours=True,
            is_render_annotations=False,
            is_render_depth=False,
            # M2: BeamNG writes frames into a shared-memory buffer we read
            # with stream() — no per-frame TCP round-trip like poll().
            is_streaming=True,
            is_using_shared_memory=True,
        )
        print("[world] scenario live, camera attached (shmem streaming).",
              flush=True)

    # ---- frames ----

    def poll_bgr(self) -> np.ndarray | None:
        """Latest camera frame as HxWx3 BGR uint8, or None if not ready.

        stream_raw() reads the shmem bytes with no PIL decode — 0.13 ms vs
        12 ms for stream()/poll(). cvtColor both drops alpha and swaps
        RGB->BGR (simsteer's pipeline is cv2 land) in one SIMD pass.
        """
        raw = self.camera.stream_raw()
        buf = raw.get("colour") if isinstance(raw, dict) else None
        if buf is None or len(buf) == 0:
            return None
        arr = np.frombuffer(buf, dtype=np.uint8)
        ch = arr.size // (CAM_W * CAM_H)
        if ch not in (3, 4) or arr.size != CAM_W * CAM_H * ch:
            return None
        img = arr.reshape(CAM_H, CAM_W, ch)
        code = cv2.COLOR_RGBA2BGR if ch == 4 else cv2.COLOR_RGB2BGR
        return cv2.cvtColor(img, code)

    # ---- telemetry ----

    def poll_telemetry(self) -> dict:
        """v_ego (m/s), steering wheel angle (deg), normalized steering
        input (-1..1), and world heading (rad, CCW+) from Electrics +
        vehicle state. One TCP round-trip (~few ms) — poll from a side
        thread, not the 20 Hz model loop."""
        with self._ctl_lock:
            self.vehicle.sensors.poll()
        el = self.vehicle.sensors["electrics"]
        st = self.vehicle.state or {}
        d = st.get("dir", (0.0, -1.0, 0.0))
        return {
            "v_ego": float(el.get("wheelspeed", 0.0)),
            "steering_deg": float(el.get("steering", 0.0)),
            "steering_input": float(el.get("steering_input", 0.0)),
            "heading_rad": math.atan2(float(d[1]), float(d[0])),
            "pos": tuple(st.get("pos", (0.0, 0.0, 0.0))),
        }

    # ---- control (M3) ----

    def apply(self, steering: float, throttle: float | None,
              brake: float | None) -> None:
        """Steering goes through input.event FILTER_DIRECT (2): measured
        t90 0.08 s vs 0.28 s via vehicle.control(), which also applies
        speed-sensitive limiting. Direct is linear and instant — the
        rack LiveParams fits is then actually the rack, not the input
        smoother. Throttle/brake keep the normal control() path.

        Pass throttle=None and brake=None for steering-only: no
        control() call at all, so the player's own pedal inputs are
        NOT overwritten (sending 0/0 at 20 Hz stomps them).

        _ctl_lock serialises with poll_telemetry(): both use the same
        per-vehicle TCP connection (bridge lesson — races hang it)."""
        s = float(np.clip(steering, -1, 1))
        with self._ctl_lock:
            self.vehicle.queue_lua_command(
                f"input.event('steering', {s:.4f}, 2)")
            if throttle is not None or brake is not None:
                self.vehicle.control(
                    throttle=float(np.clip(throttle or 0.0, 0, 1)),
                    brake=float(np.clip(brake or 0.0, 0, 1)))

    # ---- misc ----

    def ai_drive(self, speed_ms: float = 25.0) -> None:
        """Let BeamNG's AI drive (useful for feeding the model motion
        without closing the loop)."""
        self.vehicle.ai.set_mode("span")
        self.vehicle.ai.set_speed(speed_ms, mode="limit")

    def ai_disable(self) -> None:
        """Return input authority to us / the player."""
        self.vehicle.ai.set_mode("disabled")

    def realistic_gearbox(self) -> None:
        """Required before the model drives: in arcade mode, holding
        the brake at low speed shifts into REVERSE — the controller
        'holds the brakes' and the car backs up. Realistic automatic
        keeps D engaged; brake is just brake."""
        with self._ctl_lock:
            self.vehicle.set_shift_mode("realistic_automatic")

    def ensure_drive(self) -> None:
        """Shift the automatic into D. After a spawn or vehicle reset
        it sits in N/P and an engage just revs in neutral. Index table
        measured on the bastion automatic: 0=N, 1=P (!), 2=D, 3=S,
        -1=R — Drive is index 2. No-op if already in D."""
        with self._ctl_lock:
            self.vehicle.queue_lua_command(
                "if controller.mainController then "
                "controller.mainController.shiftToGearIndex(2) end")

    def set_signal(self, direction: str | None) -> None:
        """Turn signals: 'left', 'right', or None for off.

        Writing electrics.values directly is a no-op (electrics
        recomputes them every frame — measured); the working API is the
        toggle functions, so read the latched left_signal/right_signal
        state and toggle only on mismatch (idempotent)."""
        want_l = "true" if direction == "left" else "false"
        want_r = "true" if direction == "right" else "false"
        lua = (
            "local e = electrics.values "
            "local l = (e.left_signal ~= nil and e.left_signal ~= 0 "
            "and e.left_signal ~= false) "
            "local r = (e.right_signal ~= nil and e.right_signal ~= 0 "
            "and e.right_signal ~= false) "
            f"if l ~= {want_l} then electrics.toggle_left_signal() end "
            f"if r ~= {want_r} then electrics.toggle_right_signal() end")
        with self._ctl_lock:
            self.vehicle.queue_lua_command(lua)

    def close(self) -> None:
        try:
            self.camera.remove()
        except Exception as exc:
            # A failed remove LEAKS a full-res streaming camera that keeps
            # rendering forever (bridge project hit this as a GPU-load
            # runaway) — make it loud.
            print(f"[world] WARNING: camera remove FAILED ({exc}) — "
                  f"old camera may still be rendering!", flush=True)
        try:
            self.bng.disconnect()
        except Exception:
            pass
