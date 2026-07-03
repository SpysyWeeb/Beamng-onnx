"""ScreenWorld — the no-tech.key backend: same interface the control
panel expects from BeamNGOnnxWorld, built from a game-window capture
and a virtual G29 wheel. Zero beamngpy.

What can't exist here and how it's handled:
  - telemetry: there is none. poll_telemetry() returns zeros and sets
    `no_telemetry`; the panel substitutes the MODEL's own ego speed
    (decoded pose vx — validated 0.5% accurate against ground truth
    in tech mode) for v_ego.
  - gearbox / AI / traffic / blinkers / teleports: player's job or
    unavailable; calls are safe no-ops that log once.
  - camera geometry: the player drives the HOOD cam; frame size comes
    from the captured window and the horizontal FOV from config
    (match the in-game FOV setting). LiveCalib learns pitch/height on
    top, exactly like a fresh comma install.
"""
from __future__ import annotations

import time

import numpy as np

from simsteer.io.screen_capture import ScreenCapture
from simsteer.io.vwheel import VirtualWheel


class ScreenWorld:
    no_telemetry = True
    attached = False
    bng = None
    vehicle = None

    def __init__(self, window_hint: str = "BeamNG",
                 fov_h_deg: float = 90.0):
        self.capture = ScreenCapture(window_hint)
        self.wheel = VirtualWheel()
        self.frame_w, self.frame_h = self.capture.size
        self.fov_h_deg = float(fov_h_deg)
        self._last_frame: np.ndarray | None = None
        self._stale_since: float | None = None
        self._warned: set[str] = set()
        print(f"[screen-world] capture {self.frame_w}x{self.frame_h} @ "
              f"{self.fov_h_deg:.0f} deg hFOV, virtual wheel up",
              flush=True)

    # ---- frames ----

    def poll_bgr(self) -> np.ndarray | None:
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
            # window resized: report once; the panel rebuilds its warp
            self.frame_w, self.frame_h = frame.shape[1], frame.shape[0]
            self._warn("resize", f"window resized to "
                                 f"{self.frame_w}x{self.frame_h}")
        self._last_frame = frame
        return frame

    # ---- telemetry (none) ----

    def poll_telemetry(self) -> dict:
        return {"v_ego": 0.0, "steering_deg": 0.0, "steering_input": 0.0,
                "throttle_input": 0.0, "brake_input": 0.0,
                "heading_rad": 0.0, "pos": (0.0, 0.0, 0.0),
                "roll_glat": 0.0}

    # ---- control ----

    def apply(self, steering: float, throttle: float | None,
              brake: float | None) -> None:
        self.wheel.set(float(steering), throttle, brake)

    # ---- tech-mode-only surface: safe no-ops ----

    def _warn(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(f"[screen-world] {msg}", flush=True)

    def realistic_gearbox(self, *a, **k) -> None:
        self._warn("gear", "no gearbox control in screen mode — put the "
                           "car in D (automatic) yourself")

    def ensure_drive(self, *a, **k) -> None:
        self.realistic_gearbox()

    def ai_drive(self, *a, **k) -> None:
        self._warn("ai", "BeamNG AI toggle needs tech mode")

    def ai_disable(self, *a, **k) -> None:
        pass

    def set_signal(self, *a, **k) -> None:
        pass

    def spawn_traffic(self, *a, **k) -> None:
        self._warn("traffic", "traffic spawning needs tech mode")

    def close(self) -> None:
        try:
            self.wheel.close()
        finally:
            self.capture.close()
