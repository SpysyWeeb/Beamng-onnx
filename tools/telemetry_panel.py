"""High-quality dearpygui port of the op-replay-clipper telemetry
panel (the rotating wheel + arc markers + readouts + confidence bar).

Same layout as control_panel.draw_telemetry, but drawn with dpg's
anti-aliased primitives and TTF text instead of cv2 — crisp at any
size. One TelemetryPanel owns a drawlist; call update() each frame to
redraw it from App state.
"""
from __future__ import annotations

import math
import time

import numpy as np
import dearpygui.dearpygui as dpg

from control_panel import _build_wheel_icon

# palette (RGB; the cv2 panel stored these BGR)
BG     = (12, 16, 22)
LABEL  = (140, 150, 165)
WHITE  = (240, 240, 235)
BLUE   = (74, 163, 232)      # desired
GREEN  = (126, 209, 126)     # actual
YELLOW = (232, 192, 74)      # target %
ORANGE = (232, 130, 74)      # applied %
DIM    = (46, 56, 68)
RED    = (220, 70, 70)

W, H = 520, 712
CX, CY, R = 188, 206, 90     # wheel center + radius


class TelemetryPanel:
    def __init__(self):
        self._tex = None

    def build(self, parent) -> None:
        # wheel as an RGBA texture (white where the icon mask is set),
        # drawn rotated each frame via draw_image_quad
        icon = _build_wheel_icon(2 * R)
        s = icon.shape[0]
        rgba = np.zeros((s, s, 4), dtype=np.float32)
        m = icon > 127
        rgba[m, :3] = np.array(WHITE) / 255.0
        rgba[m, 3] = 1.0
        with dpg.texture_registry():
            self._tex = dpg.add_static_texture(
                s, s, rgba.flatten(), tag="wheel_tex")
        self._dl = dpg.add_drawlist(width=W, height=H, parent=parent)

    # ---- primitives ----
    def _txt(self, x, y, s, size, col):
        dpg.draw_text((x, y), s, size=size, color=col, parent=self._dl)

    def _row(self, y, label, value, vcol):
        self._txt(30, y, label, 15, LABEL)
        self._txt(178, y - 6, value, 26, vcol)

    def _arc_pt(self, axis, radius):
        th = math.radians(float(np.clip(axis, -1, 1)) * 120.0)
        return (CX + radius * math.sin(th), CY - radius * math.cos(th))

    def update(self, app, tel, v_ego) -> None:
        dpg.delete_item(self._dl, children_only=True)
        d = self._dl
        dpg.draw_rectangle((0, 0), (W, H), fill=BG, color=BG, parent=d)

        self._txt(24, 22, "TELEMETRY", 24, LABEL)
        mode = app.long_mode.upper()
        mcol = {"EXP": BLUE, "CHILL": GREEN, "OFF": LABEL}[mode]
        self._txt(W - 150, 24, mode, 18, mcol)

        # arc rings (top, spanning the marker range)
        for rr in (R + 14, R + 22, R + 30):
            pts = [self._arc_pt(a / 100.0, rr)
                   for a in range(-105, 106, 7)]
            dpg.draw_polyline(pts, color=DIM, thickness=3, parent=d)

        # steering wheel, rotated by actual wheel degrees (CCW+ = left)
        a = -math.radians(float(tel.get("steering_deg", 0.0)))
        ca, sa = math.cos(a), math.sin(a)
        def rot(dx, dy):
            return (CX + dx * ca - dy * sa, CY + dx * sa + dy * ca)
        dpg.draw_image_quad(
            "wheel_tex", rot(-R, -R), rot(R, -R), rot(R, R), rot(-R, R),
            parent=d)

        # arc markers: blue = target, orange = applied cmd, white = car
        if app.engaged:
            dpg.draw_circle(self._arc_pt(app.lat.last_axis_target, R + 22),
                            5, fill=BLUE, color=BLUE, parent=d)
            dpg.draw_circle(self._arc_pt(app.lat.last_axis, R + 22),
                            5, fill=ORANGE, color=ORANGE, parent=d)
        dpg.draw_circle(self._arc_pt(tel.get("steering_input", 0.0), R + 30),
                        7, fill=WHITE, color=WHITE, parent=d)

        # turn arrows
        lch = app.desire_idx in (1, 3)
        rch = app.desire_idx in (2, 4)
        self._txt(CX - R - 60, CY - 34, "<", 44, WHITE if lch else DIM)
        self._txt(CX + R + 32, CY - 34, ">", 44, WHITE if rch else DIM)

        # confidence bar (right edge)
        conf = float(np.clip(app.conf_now, 0.0, 1.0))
        tx, t_top, t_bot = W - 52, 100, H - 150
        self._txt(tx - 26, 78, "CONF", 15, LABEL)
        dpg.draw_line((tx, t_top), (tx, t_bot), color=DIM, thickness=3,
                      parent=d)
        ccol = (GREEN if conf > 0.5 else BLUE if conf > 0.25 else RED)
        by = t_bot - conf * (t_bot - t_top)
        dpg.draw_circle((tx, by), 13, fill=(10, 10, 10), color=(10, 10, 10),
                        parent=d)
        dpg.draw_circle((tx, by), 10, fill=ccol, color=ccol, parent=d)

        # readout rows
        ry = CY + R + 46
        v2 = max(v_ego, 1.0) ** 2
        self._row(ry, "DES LAT", f"{v2 * app.lat.last_curvature:+.2f} m/s2", BLUE)
        self._row(ry + 38, "ACT LAT", f"{v2 * app.k_meas_now:+.2f} m/s2", GREEN)
        self._row(ry + 76, "TGT %", f"{app.lat.last_axis_target * 100:+.0f}%",
                  YELLOW)
        self._row(ry + 114, "APP %", f"{app.lat.last_axis * 100:+.0f}%", ORANGE)
        self._row(ry + 152, "ACTUAL", f"{tel.get('steering_deg', 0):+.1f} deg",
                  WHITE)
        hands = (time.monotonic() - app.lp.last_intervened_ts) < 1.0
        self._row(ry + 190, "HANDS", "ON WHEEL" if hands else "OFF WHEEL",
                  WHITE)

        # pedals: DRIVER | ONNX
        py = ry + 228
        self._txt(30, py, "DRIVER", 14, LABEL)
        self._txt(300, py, "ONNX", 14, LABEL)
        user_pedals = app.long_mode == "off"
        for i, (name, dval, oval, col) in enumerate((
                ("GAS", tel.get("throttle_input", 0.0), app.last_thr, GREEN),
                ("BRAKE", tel.get("brake_input", 0.0), app.last_brk, RED))):
            yy = py + 28 + i * 42
            self._txt(30, yy, name, 15, LABEL)
            dtxt = f"{dval*100:.0f}%" if user_pedals and dval > 0.02 else "OFF"
            self._txt(112, yy, dtxt, 15, WHITE)
            bx, bw = 300, 170
            dpg.draw_rectangle((bx, yy), (bx + bw, yy + 12),
                               fill=(46, 40, 32), color=(46, 40, 32), parent=d)
            if oval > 0.005:
                dpg.draw_rectangle((bx, yy), (bx + int(bw * min(oval, 1.0)),
                                   yy + 12), fill=col, color=col, parent=d)
            self._txt(bx + bw + 8, yy, f"{oval*100:.0f}%", 15, WHITE)

        # accel strip
        ay = py + 120
        for name, val, col, x in (
                ("A EGO", app._a_meas_app, GREEN, 30),
                ("A TARGET", app.long.last_a_target, BLUE, 160),
                ("CMD", app.long.last_a_cmd, YELLOW, 300),
                ("OUT", app.long.last_a_cmd + app.long.last_a_fb, WHITE, 410)):
            self._txt(x, ay, name, 13, LABEL)
            self._txt(x, ay + 24, f"{val:+.2f}", 24, col)
