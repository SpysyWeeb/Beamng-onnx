"""HybridWorld — the no-tech.key backend that keeps beamngpy for
everything it's allowed to do (scenario load/start, vehicle spawn,
freeroam attach, direct-input steering, vehicle.control, real Electrics
telemetry, AI, traffic, gearbox, blinkers) and swaps ONLY the one
tech-gated piece — the automated Camera sensor — for a capture of the
game window.

Probed live 2026-07-04 with the tech.key removed from the install dir:
the beamngpy socket, Scenario.make/load/start, Vehicle spawn, Electrics
/State and vehicle.control ALL work without a license; only the Camera
throws `BNGValueError: This feature requires a BeamNG.tech license.` So
we subclass the full BeamNGOnnxWorld and override just the camera — no
beamngpy Camera is ever constructed (so the license error can't fire),
and poll_bgr() reads the game window instead.

Camera geometry: the player drives the HOOD cam; frame size comes from
the captured window and the horizontal FOV from config (match the
in-game FOV). LiveCalib refines pitch/height on top like a fresh comma
install. Everything else — real wheelspeed for v_ego, FILTER_DIRECT
steering, CAL system-ID, per-vehicle rack fit — is byte-identical to
tech mode because it all rides the same beamngpy connection.
"""
from __future__ import annotations

import time

import numpy as np  # noqa: F401  (kept for parity with the base module)

from beamng.world import BeamNGOnnxWorld
from simsteer.io.screen_capture import ScreenCapture


class HybridWorld(BeamNGOnnxWorld):
    # We DO have telemetry here (this replaced the retired pure-capture
    # + virtual-wheel path): v_ego comes from real Electrics.wheelspeed.
    no_telemetry = False

    def __init__(self, *args, window_hint: str = "BeamNG",
                 fov_h_deg: float = 90.0, **kwargs):
        # Full beamngpy bring-up: scenario / attach / defer, telemetry,
        # control. _make_camera() is overridden below to return None, so
        # the base ctor's `self.camera = self._make_camera()` never
        # touches the licensed sensor.
        super().__init__(*args, **kwargs)
        self.capture = ScreenCapture(window_hint)
        self.frame_w, self.frame_h = self.capture.size
        self.fov_h_deg = float(fov_h_deg)
        self._last_frame = None
        self._stale_since: float | None = None
        print(f"[hybrid-world] beamngpy control+telemetry live; camera "
              f"from screen capture {self.frame_w}x{self.frame_h} @ "
              f"{self.fov_h_deg:.0f} deg hFOV (set the driver cam to the "
              f"HOOD cam)", flush=True)

    # ---- camera: the ONE tech-gated feature, replaced by capture ----

    def _make_camera(self):
        # No beamngpy Camera — returning None keeps self.camera None, so
        # the base class's relink() camera swap and close() become
        # harmless (guarded) no-ops. poll_bgr() is overridden below.
        return None

    def poll_bgr(self):
        frame = self.capture.grab()
        now = time.monotonic()
        if frame is None:
            if self._stale_since is None:
                self._stale_since = now
            if now - self._stale_since > 8.0:
                raise RuntimeError(
                    "game window lost for 8 s — did the game close?")
            return self._last_frame
        self._stale_since = None
        if frame.shape[1] != self.frame_w or frame.shape[0] != self.frame_h:
            # window resized: the panel rebuilds its warp off frame_w/h
            self.frame_w, self.frame_h = frame.shape[1], frame.shape[0]
            print(f"[hybrid-world] window resized to "
                  f"{self.frame_w}x{self.frame_h}", flush=True)
        self._last_frame = frame
        return frame

    # ---- teardown: no beamngpy camera to remove, just the capture ----

    def close(self) -> None:
        try:
            self.capture.close()
        finally:
            try:
                self.bng.disconnect()
            except Exception:
                pass
