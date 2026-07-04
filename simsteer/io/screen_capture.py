"""Game-window capture for the no-tech.key screen mode.

Per-window capture via PrintWindow(PW_CLIENTONLY | PW_RENDERFULLCONTENT):
plain BitBlt from a window DC comes back black for hardware-accelerated
(DirectX) windows, but PW_RENDERFULLCONTENT (Win 8.1+) asks DWM for the
composited client area, which works for BeamNG's D3D swapchain even when
the window is partly occluded. The one mode it cannot see is *exclusive*
fullscreen (which bypasses DWM entirely) — run the game windowed or
borderless-fullscreen for hybrid mode.

Why not the alternatives: Desktop Duplication (dxcam & friends) grabs a
whole monitor, not a window — anything on top of the game (including our
own panels) lands in the model's view; Windows.Graphics.Capture needs a
WinRT package + a capture-session dance for the same pixels. PrintWindow
is per-window, occlusion-proof, dependency-free (raw user32/gdi32
through ctypes), and comfortably beats the 20 Hz the model needs.

No beamngpy, no game cooperation. The player uses the HOOD camera
in-game so the captured view approximates a windshield mount. The window
is re-located every couple of seconds so moves and resizes are followed.

Test (game running):
    .venv\\Scripts\\python -m simsteer.io.screen_capture [window-hint]
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import numpy as np

_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32

# Without DPI awareness Windows lies about client rects on scaled
# displays (returns virtualized coordinates), and the capture comes out
# resampled. Per-process, idempotent, harmless if dearpygui already set it.
try:
    _user32.SetProcessDPIAware()
except Exception:                                # pragma: no cover
    pass

_RELOCATE_S = 2.0
_PW_CLIENTONLY = 0x1
_PW_RENDERFULLCONTENT = 0x2
_BI_RGB = 0

# Our own UI windows (start panel, control panel) all carry "onnx" in
# their title/class. The default hint "beamng" matches them too, so we
# must never capture ourselves — that produces the infinite-mirror
# feedback (game window recreates on focus change -> stale handle ->
# re-search -> our own window is the only other "beamng" match).
_SELF_MARK = "onnx"


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD),
                ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER),
                ("bmiColors", wintypes.DWORD * 3)]


def _window_title(hwnd) -> str:
    n = _user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    _user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _window_class(hwnd) -> str:
    buf = ctypes.create_unicode_buffer(256)
    _user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _client_size(hwnd) -> tuple[int, int]:
    r = wintypes.RECT()
    if not _user32.GetClientRect(hwnd, ctypes.byref(r)):
        return 0, 0
    return r.right - r.left, r.bottom - r.top


def _find_window(hint: str):
    """Largest visible top-level window whose title or class contains
    `hint` (case-insensitive), is at least video-sized, and is NOT one
    of our own UI windows. Largest-area wins: the game's render surface
    is the big one; stray tiny windows lose."""
    hint = hint.lower()
    best: dict = {"hwnd": None, "area": -1}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd) or _user32.IsIconic(hwnd):
            return True
        hay = f"{_window_title(hwnd)} {_window_class(hwnd)}".lower()
        if hint not in hay or _SELF_MARK in hay:
            return True
        w, h = _client_size(hwnd)
        if w < 640 or h < 360:
            return True
        if w * h > best["area"]:
            best["hwnd"], best["area"] = hwnd, w * h
        return True

    _user32.EnumWindows(visit, 0)
    return best["hwnd"]


class ScreenCapture:
    """BGR frames of the game window via PrintWindow."""

    def __init__(self, window_hint: str = "BeamNG",
                 timeout_s: float = 30.0):
        self._hint = window_hint
        self._hwnd = None
        self._w = self._h = 0
        self._located_at = 0.0
        t0 = time.monotonic()
        while self._hwnd is None:
            self._hwnd = _find_window(window_hint)
            if self._hwnd is None:
                if time.monotonic() - t0 > timeout_s:
                    raise RuntimeError(
                        f"no visible window matching {window_hint!r} — "
                        "is the game running? (exclusive fullscreen "
                        "can't be captured; use borderless/windowed)")
                time.sleep(1.0)
        self._locate()
        print(f"[capture] window {_window_title(self._hwnd)!r} "
              f"{self._w}x{self._h}", flush=True)

    def _locate(self) -> None:
        if not _user32.IsWindow(self._hwnd):
            raise RuntimeError("window gone")
        w, h = _client_size(self._hwnd)
        if w <= 0 or h <= 0:
            raise RuntimeError("window has no client area (minimized?)")
        self._w, self._h = w, h
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
                self._hwnd = _find_window(self._hint)
                if self._hwnd is None:
                    return None
                try:
                    self._locate()
                except Exception:
                    return None
        w, h = self._w, self._h
        sdc = _user32.GetDC(None)
        mdc = _gdi32.CreateCompatibleDC(sdc)
        bmp = _gdi32.CreateCompatibleBitmap(sdc, w, h)
        try:
            _gdi32.SelectObject(mdc, bmp)
            if not _user32.PrintWindow(
                    self._hwnd, mdc,
                    _PW_CLIENTONLY | _PW_RENDERFULLCONTENT):
                self._located_at = 0.0   # force relocate next call
                return None
            bmi = _BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = w
            bmi.bmiHeader.biHeight = -h          # top-down rows
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 32
            bmi.bmiHeader.biCompression = _BI_RGB
            frame = np.empty((h, w, 4), dtype=np.uint8)
            got = _gdi32.GetDIBits(
                mdc, bmp, 0, h,
                frame.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(bmi), 0)
            if got != h:
                self._located_at = 0.0
                return None
            return np.ascontiguousarray(frame[:, :, :3])   # BGRX -> BGR
        finally:
            _gdi32.DeleteObject(bmp)
            _gdi32.DeleteDC(mdc)
            _user32.ReleaseDC(None, sdc)

    def close(self) -> None:
        self._hwnd = None


if __name__ == "__main__":                       # pragma: no cover
    import os
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
        os.makedirs("debug_out", exist_ok=True)
        cv2.imwrite("debug_out/screen_capture_test.png", frame)
        print("wrote debug_out/screen_capture_test.png")
    cap.close()
