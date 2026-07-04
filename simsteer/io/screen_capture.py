"""Game-window capture for the no-tech.key screen mode — platform front.

The interface is one class, `ScreenCapture` (find the game window by a
title hint, `grab()` BGR frames, follow moves/resizes), with a backend
per platform:

  - Windows: PrintWindow(PW_RENDERFULLCONTENT) per-window capture,
    raw user32/gdi32 via ctypes — `screen_capture_win`. Run the game
    windowed or borderless (exclusive fullscreen bypasses DWM).
  - Linux:   per-window XGetImage on X11/XWayland (python-xlib) —
    `screen_capture_x11`.

Test (game running):
    python -m simsteer.io.screen_capture [window-hint]
"""
from __future__ import annotations

import sys

if sys.platform == "win32":
    from simsteer.io.screen_capture_win import ScreenCapture
else:
    from simsteer.io.screen_capture_x11 import ScreenCapture

__all__ = ["ScreenCapture"]


if __name__ == "__main__":                       # pragma: no cover
    import time

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
        import os
        os.makedirs("debug_out", exist_ok=True)
        cv2.imwrite("debug_out/screen_capture_test.png", frame)
        print("wrote debug_out/screen_capture_test.png")
    cap.close()
