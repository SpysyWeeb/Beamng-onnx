"""Game-window capture for the no-tech.key screen mode.

BeamNG's Linux build is SDL-on-X11, so under a Wayland desktop it is
an XWayland window — and XWayland windows carry real backing pixmaps
that XGetImage can read per-window. That matters because on rootless
XWayland the classic "grab the root screen" path (mss & friends) hits
X protocol errors: the root isn't composited. Per-window drawable
capture is the one X11 path that works here, validated on this
machine against a known-color probe window (pixel-exact).

No beamngpy, no portals, no game cooperation. The player uses the
HOOD camera in-game so the captured view approximates a windshield
mount. The window is re-located every couple of seconds so moves and
resizes are followed; fullscreen is the easy case (no decorations).

Test (game running):
    DISPLAY=:0 .venv/bin/python3 -m simsteer.io.screen_capture
"""
from __future__ import annotations

import time

import numpy as np

try:
    from Xlib import X, display
except ImportError as exc:                       # pragma: no cover
    raise ImportError(
        "screen mode needs python-xlib: uv pip install python-xlib"
    ) from exc

_RELOCATE_S = 2.0


def _wm_name(w) -> str:
    try:
        n = w.get_wm_name()
        if isinstance(n, bytes):
            n = n.decode("utf-8", "ignore")
        return str(n or "")
    except Exception:
        return ""


# Our own UI windows (start panel, control panel) all carry "onnx" in
# their title/class. The default hint "beamng" matches them too, so we
# must never capture ourselves — that produces the infinite-mirror
# feedback (game window recreates on focus change -> stale handle ->
# re-search -> our own window is the only other "beamng" match).
_SELF_MARK = "onnx"


def _find_window(dsp, hint: str):
    """Largest viewable window whose title or WM_CLASS contains `hint`
    (case-insensitive), is at least video-sized, and is NOT one of our
    own UI windows. Largest-area wins: the game's render surface is the
    big one; decoration wrappers and stray tiny windows lose."""
    hint = hint.lower()
    best, best_area = None, -1

    def visit(w):
        nonlocal best, best_area
        try:
            children = w.query_tree().children
        except Exception:
            children = []
        for c in children:
            visit(c)
        try:
            if w.get_attributes().map_state != X.IsViewable:
                return
            cls = w.get_wm_class() or ("", "")
            hay = f"{_wm_name(w)} {cls[0]} {cls[1]}".lower()
            if hint not in hay or _SELF_MARK in hay:
                return
            geo = w.get_geometry()
            if geo.width < 640 or geo.height < 360:
                return
            area = geo.width * geo.height
            if area > best_area:
                best, best_area = w, area
        except Exception:
            pass

    visit(dsp.screen().root)
    return best


class ScreenCapture:
    """BGR frames of the game window via per-window XGetImage."""

    def __init__(self, window_hint: str = "BeamNG",
                 timeout_s: float = 30.0):
        self._dsp = display.Display()
        self._hint = window_hint
        self._win = None
        self._w = self._h = 0
        self._located_at = 0.0
        t0 = time.monotonic()
        while self._win is None:
            self._win = _find_window(self._dsp, window_hint)
            if self._win is None:
                if time.monotonic() - t0 > timeout_s:
                    raise RuntimeError(
                        f"no visible window matching {window_hint!r} — "
                        "is the game running on this display?")
                time.sleep(1.0)
        self._locate()
        print(f"[capture] window {_wm_name(self._win)!r} "
              f"{self._w}x{self._h}", flush=True)

    def _locate(self) -> None:
        geo = self._win.get_geometry()
        self._w, self._h = geo.width, geo.height
        self._located_at = time.monotonic()

    @property
    def size(self) -> tuple[int, int]:
        return self._w, self._h

    def grab(self) -> np.ndarray | None:
        """One BGR frame, or None if the window is gone (caller may
        retry / re-init)."""
        now = time.monotonic()
        if now - self._located_at > _RELOCATE_S:
            try:
                self._locate()
            except Exception:
                self._win = _find_window(self._dsp, self._hint)
                if self._win is None:
                    return None
                self._locate()
        try:
            img = self._win.get_image(0, 0, self._w, self._h,
                                      X.ZPixmap, 0xFFFFFFFF)
        except Exception:
            self._located_at = 0.0     # force relocate next call
            return None
        frame = np.frombuffer(img.data, dtype=np.uint8).reshape(
            self._h, self._w, 4)
        return np.ascontiguousarray(frame[:, :, :3])   # BGRX -> BGR

    def close(self) -> None:
        try:
            self._dsp.close()
        except Exception:
            pass


if __name__ == "__main__":                       # pragma: no cover
    import sys
    import cv2
    hint = sys.argv[1] if len(sys.argv) > 1 else "BeamNG"
    cap = ScreenCapture(hint)
    n, t0 = 0, time.monotonic()
    frame = cap.grab()
    while time.monotonic() - t0 < 3.0:
        f = cap.grab()
        if f is not None:
            frame, n = f, n + 1
    print(f"{n / (time.monotonic() - t0):.0f} fps  frame "
          f"{None if frame is None else frame.shape}")
    if frame is not None:
        cv2.imwrite("debug_out/screen_capture_test.png", frame)
        print("wrote debug_out/screen_capture_test.png")
    cap.close()
