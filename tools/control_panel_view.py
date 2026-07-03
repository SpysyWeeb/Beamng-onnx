"""dearpygui view for the control panel — crisp fonts, a live camera
texture, real plots, and legible readouts, replacing the hand-drawn
cv2 canvas.

Split of concerns:
  - the ENGINE (App + the loop below) is unchanged logic: poll frame,
    run the model, control_tick, drive the sender. It runs on a
    background thread and publishes the overlaid camera frame + reads
    of App state.
  - this VIEW renders that state at up to 30 Hz on the main thread
    (dearpygui must own the main thread).

Reuses App, the overlay compositor, and all constants from
control_panel.py — only the presentation changed.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import dearpygui.dearpygui as dpg

import control_panel as cp
from control_panel import App, DESIRE_LEN, WARMUP_FRAMES, ROOT
from simsteer.ui.overlay import draw_overlay
from telemetry_panel import TelemetryPanel, W

CAM_DW = 900                    # camera display width (px); height per aspect
PLOT_H = 150


class Engine(threading.Thread):
    """The 20 Hz model+control loop, headless. Publishes the latest
    overlaid RGBA frame for the texture; all other display data the
    view reads straight off App (simple float/str reads)."""

    def __init__(self, app: App):
        super().__init__(daemon=True, name="engine")
        self.app = app
        self.rgba: np.ndarray | None = None
        self.cam_w = self.cam_h = 0
        self._stop = threading.Event()
        self._first = threading.Event()
        self.perf = {"gap": 0.0, "model": 0.0, "ctl": 0.0, "n": 0}

    def run(self) -> None:
        app = self.app
        last = time.monotonic()
        while not self._stop.is_set():
            t0 = time.monotonic()
            bgr = app.world.poll_bgr()
            if bgr is None:
                time.sleep(0.005)
                continue
            app.frame_idx += 1
            dt = max(1e-3, t0 - last)
            last = t0
            self.perf["gap"] += dt

            desire_vec = None
            if app.desire_idx is not None:
                desire_vec = np.zeros(DESIRE_LEN, dtype=np.float32)
                desire_vec[app.desire_idx] = 1.0

            tm = time.monotonic()
            img, big = app.queue.push(bgr, app.calib)
            d = app._decode(img, big, desire_vec)
            self.perf["model"] += time.monotonic() - tm

            tc = time.monotonic()
            tel = app.tel.snapshot()
            v_ego = tel["v_ego"]
            if app.screen_mode:
                app._vision_v += 0.25 * (
                    max(0.0, float(d.pose[0])) - app._vision_v)
                v_ego = app._vision_v
            app.control_tick(d, v_ego, dt)
            self.perf["ctl"] += time.monotonic() - tc
            app._last_tel = tel
            app._last_v = v_ego
            app._last_decoded = d

            # loop-rate health (same policy as the cv2 loop)
            if app.engaged and app.hz and app.hz < 8.0 \
                    and app.frame_idx > WARMUP_FRAMES:
                app.disengage(f"loop rate {app.hz:.0f} Hz")
            if app.hz:
                app.hz_ema += 0.05 * (app.hz - app.hz_ema)
            if (app.engaged and app.frame_idx > WARMUP_FRAMES
                    and app.hz_ema < 16.0
                    and time.monotonic() > app._rate_warned_until):
                app._rate_warned_until = time.monotonic() + 10.0
                app.set_banner(
                    f"loop {app.hz_ema:.0f} Hz < 20: model time-warped - "
                    "lower game graphics or use a smaller model", 6.0)

            # composite overlay -> RGBA for the texture
            over = draw_overlay(bgr, d, app.calib)
            if self.cam_w == 0:
                h, w = over.shape[:2]
                self.cam_w, self.cam_h = CAM_DW, max(1, CAM_DW * h // w)
            small = cp.cv2.resize(over, (self.cam_w, self.cam_h),
                                  interpolation=cp.cv2.INTER_AREA)
            rgba = np.empty((self.cam_h, self.cam_w, 4), dtype=np.float32)
            rgba[:, :, 0] = small[:, :, 2] / 255.0     # R <- B
            rgba[:, :, 1] = small[:, :, 1] / 255.0
            rgba[:, :, 2] = small[:, :, 0] / 255.0     # B <- R
            rgba[:, :, 3] = 1.0
            self.rgba = rgba
            self._first.set()

            if app.incident_note:
                self._save_incident(over, app.incident_note)
                app.incident_note = None

            self.perf["n"] += 1
            if self.perf["n"] >= 100:
                self._flush_perf()

            el = time.monotonic() - t0
            app.hz = 1.0 / el if el > 0 else 0.0
            time.sleep(max(0.0, 0.05 - el))

    def _save_incident(self, over, note) -> None:
        inc = os.path.join(ROOT, "debug_out", "incidents")
        os.makedirs(inc, exist_ok=True)
        stamp = time.strftime("%H%M%S")
        cp.cv2.imwrite(os.path.join(inc, f"stop_{stamp}.png"), over)
        with open(os.path.join(inc, f"stop_{stamp}.txt"), "w") as f:
            f.write(note + "\n")
        print(f"[panel] INCIDENT captured: {note}", flush=True)

    def _flush_perf(self) -> None:
        p, n = self.perf, self.perf["n"]
        line = (f"gap {p['gap']/n*1000:5.1f} ms/frame "
                f"({n/max(p['gap'],1e-3):4.1f} Hz)  |  "
                f"model {p['model']/n*1000:5.1f}  ctl {p['ctl']/n*1000:4.1f} ms")
        print(f"[perf] {line}", flush=True)
        os.makedirs(os.path.join(ROOT, "debug_out"), exist_ok=True)
        with open(os.path.join(ROOT, "debug_out", "perf_last.txt"), "w") as f:
            f.write(line + "\n")
        for k in p:
            p[k] = 0
        p["n"] = 0

    def wait_first(self, timeout=60.0) -> bool:
        return self._first.wait(timeout)

    def stop(self) -> None:
        self._stop.set()


# ---- view helpers ----

def _ff(deq):
    """Forward-fill NaNs so the want-lines stay continuous on the plot
    (the deques carry NaN when a series isn't live)."""
    out, last = [], 0.0
    for v in deq:
        if v == v:                     # not NaN
            last = float(v)
        out.append(last)
    return out


class ControlView:
    C_WANT = (255, 170, 0)             # orange
    C_ACT = (90, 220, 120)             # green

    def __init__(self, app: App):
        self.app = app
        self.eng = Engine(app)
        self.tpanel = TelemetryPanel()

    # button/key -> the exact same App.action the cv2 panel used
    def _act(self, cmd):
        return lambda *_: self.app.action(cmd)

    def _theme(self):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvAll):
                c = dpg.add_theme_color
                c(dpg.mvThemeCol_WindowBg, (16, 18, 22))
                c(dpg.mvThemeCol_ChildBg, (22, 25, 30))
                c(dpg.mvThemeCol_FrameBg, (34, 38, 45))
                c(dpg.mvThemeCol_Button, (44, 88, 140))
                c(dpg.mvThemeCol_ButtonHovered, (58, 112, 176))
                c(dpg.mvThemeCol_Text, (223, 229, 236))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
        return t

    def _keymap(self):
        keys = {dpg.mvKey_E: "engage", dpg.mvKey_F: "force",
                dpg.mvKey_A: "lane_l", dpg.mvKey_D: "lane_r",
                dpg.mvKey_Z: "turn_l", dpg.mvKey_C: "turn_r",
                dpg.mvKey_L: "long", dpg.mvKey_G: "ai",
                dpg.mvKey_R: "cal", dpg.mvKey_V: "cam"}
        with dpg.handler_registry():
            for k, cmd in keys.items():
                dpg.add_key_press_handler(
                    k, callback=(lambda s, a, c=cmd: self.app.action(c)))
            dpg.add_key_press_handler(
                dpg.mvKey_B, callback=lambda *_: setattr(
                    self.app, "charts_on", not self.app.charts_on))
            # NB: dpg 2.3.1 ships broken legacy key constants with tiny
            # ASCII values (mvKey_Plus=61) that fall outside the valid
            # keycode range (520-630) and match EVERY frame — binding
            # one runs the handler continuously (this is what pinned the
            # speed cap at 100). Use only in-range codes. The main-row
            # '=' has no named constant here; its real code is 602
            # (contiguous after Slash=600, Semicolon=601), sitting right
            # next to the numpad '+' (mvKey_Add=626).
            _EQUAL = getattr(dpg, "mvKey_Equal", 602)
            for kd in (dpg.mvKey_Minus, dpg.mvKey_Subtract):
                dpg.add_key_press_handler(kd, callback=self._act("spd_dn"))
            for ku in (_EQUAL, dpg.mvKey_Add):
                dpg.add_key_press_handler(ku, callback=self._act("spd_up"))

    def build(self):
        app = self.app
        self.eng.start()
        if not self.eng.wait_first():
            raise RuntimeError("no camera frames after 60 s")
        cw, ch = self.eng.cam_w, self.eng.cam_h

        dpg.create_context()
        for cand in ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                     "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
            if os.path.isfile(cand):
                with dpg.font_registry():
                    dpg.bind_font(dpg.add_font(cand, 18))
                break

        with dpg.texture_registry():
            dpg.add_raw_texture(
                cw, ch, self.eng.rgba.flatten(), format=dpg.mvFormat_Float_rgba,
                tag="cam_tex")

        with dpg.window(tag="root", no_title_bar=True, no_move=True,
                        no_resize=True):
            dpg.add_text("", tag="banner", color=(90, 220, 255))
            dpg.add_image("cam_tex", width=cw, height=ch)

            # telemetry panel (left) beside the plots + text (right)
            with dpg.group(horizontal=True):
                with dpg.group() as tp_parent:
                    self.tpanel.build(tp_parent)
                with dpg.group():
                    for tag, title, ylim in (
                            ("p_spd", "mph", (0, 60)),
                            ("p_acc", "m/s2", (-4, 4)),
                            ("p_crv", "curv x1000", (-15, 15))):
                        with dpg.plot(label=title, height=PLOT_H,
                                      width=cw - W - 24, no_menus=True,
                                      no_box_select=True):
                            dpg.add_plot_axis(dpg.mvXAxis,
                                              no_tick_labels=True,
                                              tag=f"{tag}_x")
                            yax = dpg.add_plot_axis(dpg.mvYAxis,
                                                    tag=f"{tag}_y")
                            dpg.set_axis_limits(yax, *ylim)
                            dpg.add_line_series([], [], parent=yax,
                                                tag=f"{tag}_want")
                            dpg.add_line_series([], [], parent=yax,
                                                tag=f"{tag}_act")
                    dpg.add_text("", tag="t_rack")
                    dpg.add_text("", tag="t_cam")
                    dpg.add_text("", tag="t_lead")

            # speed cap slider (drag to set; no repeat-spam)
            with dpg.group(horizontal=True):
                dpg.add_text("SPEED CAP")
                dpg.add_slider_int(
                    tag="spd_slider", width=cw - 140, min_value=10,
                    max_value=100, format="%d mph",
                    default_value=int(round(app.cfg.max_speed_mps * 2.237)),
                    callback=lambda s, v: setattr(
                        app.cfg, "max_speed_mps", v / 2.237))

            # buttons
            with dpg.group(horizontal=True):
                dpg.add_button(label="ENGAGE", width=150, height=40,
                               callback=self._act("engage"))
                dpg.add_button(label="LONG", width=110, height=40,
                               callback=self._act("long"))
                dpg.add_button(label="CAL", width=90, height=40,
                               callback=self._act("cal"))
                dpg.add_button(label="CAM A/B", width=110, height=40,
                               callback=self._act("cam"))
                dpg.add_button(label="AI", width=80, height=40,
                               callback=self._act("ai"))
            with dpg.group(horizontal=True):
                for lab, cmd in (("< LANE", "lane_l"), ("LANE >", "lane_r"),
                                 ("< TURN", "turn_l"), ("TURN >", "turn_r"),
                                 ("SPD -", "spd_dn"), ("SPD +", "spd_up")):
                    dpg.add_button(label=lab, width=105, height=34,
                                   callback=self._act(cmd))

        self._keymap()
        dpg.bind_theme(self._theme())
        dpg.create_viewport(title="Beamng-onnx control panel",
                            width=cw + 24, height=ch + 430)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("root", True)

    def _update(self):
        app = self.app
        if self.eng.rgba is not None:
            dpg.set_value("cam_tex", self.eng.rgba.flatten())
        # banner
        dpg.set_value("banner",
                      app.banner if time.monotonic() < app.banner_until else "")
        # plots
        ch = app.chart
        x = list(range(len(ch["v"])))
        mph = 2.237
        dpg.set_value("p_spd_want", [x, [v * mph for v in _ff(ch["vT"])]])
        dpg.set_value("p_spd_act", [x, [v * mph for v in _ff(ch["v"])]])
        dpg.set_value("p_acc_want", [x, _ff(ch["a"])])
        dpg.set_value("p_acc_act", [x, _ff(ch["aM"])])
        dpg.set_value("p_crv_want", [x, [v * 1000 for v in _ff(ch["k"])]])
        dpg.set_value("p_crv_act", [x, [v * 1000 for v in _ff(ch["kM"])]])
        for tag in ("p_spd", "p_acc", "p_crv"):
            dpg.set_axis_limits(f"{tag}_x", max(0, len(x) - 300), max(1, len(x)))
        # readouts
        tel = getattr(app, "_last_tel", {}) or {}
        v = getattr(app, "_last_v", 0.0)
        lo, la, lp = app.long, app.lat, app.lp
        # the rich telemetry panel (wheel + arcs + readouts + conf)
        self.tpanel.update(app, tel, v)
        # keep the slider in sync when SPD keys/buttons move the cap
        if not dpg.is_item_active("spd_slider"):
            dpg.set_value("spd_slider",
                          int(round(app.cfg.max_speed_mps * 2.237)))
        dpg.set_value("t_rack", f"rack   a={lp.a_linear:+.2f} "
                                f"n={max(lp.samples, lp.session_samples)} "
                                f"{'OK' if lp.trusted() else 'COLD'}  "
                                f"trim {la.axis_trim_state:+.3f}")
        pit = app.lc.pitch_estimate
        dpg.set_value("t_cam", f"cam    {app.lc.blocks}blk "
                               f"{'LIVE' if app.calib_live else 'shadow'}"
                               + (f" p{pit:+.2f}" if pit is not None else ""))
        lx = lo.last_lead_x
        dpg.set_value("t_lead",
                      f"lead   {('%.0fm' % lx) if lx < 500 else 'none'}"
                      + ("  AEB!" if lo.last_aeb else ""))

    def run(self):
        self.build()
        try:
            while dpg.is_dearpygui_running():
                self._update()
                dpg.render_dearpygui_frame()
        finally:
            self.eng.stop()
            app = self.app
            try:
                app.sender.stop()
            except Exception:
                pass
            try:
                if app.lp.session_samples > 0:
                    app.lp.save()
            except Exception:
                pass
            try:
                app.world.apply(0.0, 0.0, 0.0)
                app.world.close()
            except Exception:
                pass
            dpg.destroy_context()


def run_view(app: App) -> int:
    ControlView(app).run()
    return 0
