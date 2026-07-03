#!/usr/bin/env python3
"""Start panel — the front door, rebuilt in dearpygui (crisp fonts,
GPU widgets) from the hand-drawn cv2 version.

The tech.key checkbox switches the whole form:
  checked  — tech mode: BeamNG install path, map, vehicle, traffic
             (beamngpy drives everything).
  unchecked — screen mode: no map/vehicle/traffic (the player sets
             those in-game); just the hood-cam FOV. We capture the
             game window and drive a virtual wheel.

All state and the launch flow live in tools/launcher_core.py; this
file is only the view. Run: .venv/bin/python3 tools/start_panel.py
"""
from __future__ import annotations

import os
import time

import dearpygui.dearpygui as dpg

from launcher_core import (LauncherCore, ROOT, scan_models, scan_split,
                           detect_tech_key, scan_levels)

BROWSE = "browse..."


def _names(paths):
    return [os.path.basename(p) for p in paths]


def _theme():
    """Dark, high-contrast theme — the readable replacement for the
    hand-drawn palette."""
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvAll):
            c = dpg.add_theme_color
            c(dpg.mvThemeCol_WindowBg, (18, 20, 24))
            c(dpg.mvThemeCol_FrameBg, (34, 38, 45))
            c(dpg.mvThemeCol_FrameBgHovered, (46, 52, 62))
            c(dpg.mvThemeCol_Button, (44, 88, 140))
            c(dpg.mvThemeCol_ButtonHovered, (58, 112, 176))
            c(dpg.mvThemeCol_ButtonActive, (72, 136, 210))
            c(dpg.mvThemeCol_Header, (44, 88, 140))
            c(dpg.mvThemeCol_Text, (222, 228, 235))
            c(dpg.mvThemeCol_CheckMark, (120, 200, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 9)
    return t


class StartView:
    def __init__(self):
        self.c = LauncherCore()

    # ---- callbacks ----
    def _set(self, field):
        def cb(sender, value):
            setattr(self.c, field, value)
            self._refresh_visibility()
        return cb

    def _pick_model(self, which):
        def cb(sender, value):
            if value == BROWSE:
                return
            lst = {"model": self.c.models, "vision": self.c.vision_models,
                   "policy": self.c.policy_models}[which]
            for p in lst:
                if os.path.basename(p) == value:
                    setattr(self.c, f"{which}_model" if which != "model"
                            else "model", p)
        return cb

    def _tech_toggle(self, sender, value):
        self.c.tech_key = value
        self._refresh_visibility()

    def _arch_toggle(self, sender, value):
        self.c.arch = "split" if value else "supercombo"
        self._refresh_visibility()

    def _start(self):
        self.c.start()

    def _refresh_visibility(self):
        tech = self.c.tech_key
        split = self.c.arch == "split"
        for tag in ("grp_path", "grp_map", "grp_vehicle", "grp_traffic"):
            dpg.configure_item(tag, show=tech)
        dpg.configure_item("grp_fov", show=not tech)
        dpg.configure_item("grp_super", show=not split)
        dpg.configure_item("grp_split", show=split)
        dpg.set_value("mode_caption",
                      "TECH MODE — beamngpy picks the map & car"
                      if tech else
                      "SCREEN MODE — captures the game window, you drive "
                      "the menus & hood cam")

    # ---- build ----
    def build(self):
        dpg.create_context()
        with dpg.font_registry():
            # a real TTF for crisp text at size; fall back to default
            fpath = None
            for cand in ("/usr/share/fonts/dejavu-sans-fonts/"
                         "DejaVuSans.ttf",
                         "/usr/share/fonts/truetype/dejavu/"
                         "DejaVuSans.ttf",
                         "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
                if os.path.isfile(cand):
                    fpath = cand
                    break
            if fpath:
                big = dpg.add_font(fpath, 20)
                dpg.bind_font(big)

        with dpg.window(tag="root", no_title_bar=True, no_move=True,
                        no_resize=True):
            dpg.add_text("BEAMNG-ONNX", color=(120, 200, 255))
            dpg.add_text("drive BeamNG with comma.ai's model",
                         color=(140, 150, 160))
            dpg.add_text("", tag="mode_caption", color=(200, 180, 120))
            dpg.add_separator()

            dpg.add_checkbox(label="I have a BeamNG.tech tech.key",
                             default_value=self.c.tech_key,
                             callback=self._tech_toggle)

            # tech-only: install path
            with dpg.group(tag="grp_path"):
                dpg.add_text("BeamNG install")
                dpg.add_input_text(default_value=self.c.path, width=-1,
                                   callback=self._set("path"))
            # tech-only: map
            with dpg.group(tag="grp_map"):
                dpg.add_text("Map")
                dpg.add_combo(self.c.levels, default_value=self.c.map,
                              width=-1, callback=self._set("map"))
            # tech-only: vehicle
            with dpg.group(tag="grp_vehicle"):
                dpg.add_text("Vehicle")
                dpg.add_combo(self.c.vehicles, default_value=self.c.vehicle,
                              width=-1, callback=self._set("vehicle"))
            # tech-only: traffic
            with dpg.group(tag="grp_traffic"):
                dpg.add_text("Traffic (AI cars)")
                dpg.add_slider_int(default_value=self.c.traffic, min_value=0,
                                   max_value=12, width=-1,
                                   callback=self._set("traffic"))
            # screen-only: hood-cam FOV
            with dpg.group(tag="grp_fov"):
                dpg.add_text("Hood-cam horizontal FOV (match in-game)")
                dpg.add_slider_float(default_value=self.c.screen_fov,
                                     min_value=60.0, max_value=120.0,
                                     format="%.0f deg", width=-1,
                                     callback=self._set("screen_fov"))

            dpg.add_separator()
            dpg.add_checkbox(label="split vision + policy (advanced)",
                             default_value=(self.c.arch == "split"),
                             callback=self._arch_toggle)
            with dpg.group(tag="grp_super"):
                dpg.add_text("Model")
                dpg.add_combo(_names(self.c.models),
                              default_value=os.path.basename(self.c.model),
                              width=-1, callback=self._pick_model("model"))
            with dpg.group(tag="grp_split"):
                with dpg.group(horizontal=True):
                    with dpg.group():
                        dpg.add_text("Vision")
                        dpg.add_combo(
                            _names(self.c.vision_models),
                            default_value=os.path.basename(
                                self.c.vision_model) if self.c.vision_model
                            else "", width=230,
                            callback=self._pick_model("vision"))
                    with dpg.group():
                        dpg.add_text("Policy")
                        dpg.add_combo(
                            _names(self.c.policy_models),
                            default_value=os.path.basename(
                                self.c.policy_model) if self.c.policy_model
                            else "", width=230,
                            callback=self._pick_model("policy"))

            dpg.add_separator()
            dpg.add_button(label="START", width=-1, height=48,
                           callback=self._start)
            dpg.add_text("Status", color=(140, 150, 160))
            dpg.add_text("", tag="status", wrap=560)

        dpg.bind_theme(_theme())
        dpg.create_viewport(title="Beamng-onnx", width=620, height=760)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("root", True)
        self._refresh_visibility()

    def run(self):
        self.build()
        while dpg.is_dearpygui_running():
            dpg.set_value("status", "\n".join(self.c.status[-10:]))
            # only close AFTER the control panel has actually been
            # spawned (launched is set at the very end of the flow) —
            # never during the game-boot wait.
            if self.c.launched and time.monotonic() - self.c.done_at > 3.0:
                break
            dpg.render_dearpygui_frame()
        dpg.destroy_context()


def main() -> int:
    StartView().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
