#!/usr/bin/env python3
"""Start panel — the front door, rebuilt in dearpygui (crisp fonts,
GPU widgets) from the hand-drawn cv2 version.

The tech.key checkbox only changes the CAMERA source (the one
tech-gated feature); map/vehicle/traffic/freeroam work either way:
  checked  — tech mode: beamngpy Camera sensor renders the view.
  unchecked — hybrid mode: same beamngpy scenario + control + real
             speed, camera from a screen capture (shows the hood-cam
             FOV field so the warp matches the in-game view).

All state and the launch flow live in tools/launcher_core.py; this
file is only the view. Run: .venv\\Scripts\\python tools/start_panel.py
"""
from __future__ import annotations

import os
import time

import dearpygui.dearpygui as dpg

from launcher_core import (LauncherCore, ROOT, scan_models, scan_split,
                           detect_tech_key, scan_levels, model_name,
                           VEHICLE_LABELS)

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

    def _set_vehicle(self, sender, value):
        # the combo shows display labels ("D-Series (truck)"); store the
        # beamngpy model-string key ("pickup") that everything spawns on
        for key, label in VEHICLE_LABELS.items():
            if label == value:
                self.c.vehicle = key
                return
        self.c.vehicle = value   # already a key (unknown label)

    def _pick_model(self, which):
        def cb(sender, value):
            if value == BROWSE:
                return
            lst = {"model": self.c.models, "vision": self.c.vision_models,
                   "policy": self.c.policy_models}[which]
            path = next((p for p in lst
                         if os.path.basename(p) == value), None)
            if path is None:
                return
            setattr(self.c, "model" if which == "model"
                    else f"{which}_model", path)
            # keep the split pair matched (POP vision -> POP policy)
            if which in ("vision", "policy") and self.c.link_split:
                self._sync_split(source=which)
        return cb

    def _sync_split(self, source: str) -> None:
        """Point the OTHER split dropdown at the same-named model as the
        one the user just changed. No-op if there's no matching name."""
        other = "policy" if source == "vision" else "vision"
        want = model_name(getattr(self.c, f"{source}_model"))
        pool = (self.c.policy_models if other == "policy"
                else self.c.vision_models)
        match = next((p for p in pool if model_name(p) == want), None)
        if match:
            setattr(self.c, f"{other}_model", match)
            dpg.set_value(f"combo_{other}", os.path.basename(match))

    def _toggle_link(self, sender, value):
        self.c.link_split = bool(value)
        if self.c.link_split:
            # re-linking: snap policy onto the current vision's name
            self._sync_split(source="vision")

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
        fr = self.c.freeroam
        # Install path matters in BOTH modes now: no-key mode also boots
        # the game (with -tcom) so beamngpy can route to it for control +
        # speed, so keep the path input visible regardless of the key.
        dpg.configure_item("grp_path", show=True)
        # Scenario options work in BOTH modes now: no-key (hybrid) drives
        # the same beamngpy scenario, only the camera is screen-captured.
        dpg.configure_item("grp_freeroam", show=True)
        # map/vehicle/traffic don't matter in freeroam — you load the map
        # and spawn/LINK your own car in-game.
        for tag in ("grp_map", "grp_vehicle", "grp_traffic"):
            dpg.configure_item(tag, show=not fr)
        # Hood-cam FOV only matters for the screen-captured camera (no-key).
        dpg.configure_item("grp_fov", show=not tech)
        dpg.configure_item("grp_super", show=not split)
        dpg.configure_item("grp_split", show=split)
        dpg.set_value("mode_caption",
                      "TECH MODE — beamngpy renders the camera too"
                      if tech else
                      "NO-KEY MODE — beamngpy scenario + control + speed, "
                      "camera from screen capture (set the hood cam)")

    # ---- build ----
    def build(self):
        dpg.create_context()
        with dpg.font_registry():
            # a real TTF for crisp text at size; fall back to default
            fonts = os.path.join(
                os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
            fpath = None
            for cand in (os.path.join(fonts, "segoeui.ttf"),
                         os.path.join(fonts, "calibri.ttf"),
                         os.path.join(fonts, "arial.ttf")):
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
            # tech-only: freeroam toggle (hides map/vehicle when on)
            with dpg.group(tag="grp_freeroam"):
                dpg.add_checkbox(
                    label="Freeroam — spawn your own car, LINK on START "
                          "(keeps Tab/reset; drive a lead car)",
                    default_value=self.c.freeroam,
                    callback=self._set("freeroam"))
            # tech-only, non-freeroam: map
            with dpg.group(tag="grp_map"):
                dpg.add_text("Map")
                dpg.add_combo(self.c.levels, default_value=self.c.map,
                              width=-1, callback=self._set("map"))
            # tech-only: vehicle (labels shown, model-string keys stored)
            with dpg.group(tag="grp_vehicle"):
                dpg.add_text("Vehicle")
                dpg.add_combo(
                    [VEHICLE_LABELS.get(v, v) for v in self.c.vehicles],
                    default_value=VEHICLE_LABELS.get(self.c.vehicle,
                                                     self.c.vehicle),
                    width=-1, callback=self._set_vehicle)
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
                            _names(self.c.vision_models), tag="combo_vision",
                            default_value=os.path.basename(
                                self.c.vision_model) if self.c.vision_model
                            else "", width=210,
                            callback=self._pick_model("vision"))
                    with dpg.group():
                        # link toggle: keeps vision & policy on the same
                        # model name. Checked = matched (the pairs are
                        # trained together; mixing usually won't work).
                        dpg.add_text("Link")
                        dpg.add_checkbox(
                            tag="chk_link_split",
                            default_value=self.c.link_split,
                            callback=self._toggle_link)
                    with dpg.group():
                        dpg.add_text("Policy")
                        dpg.add_combo(
                            _names(self.c.policy_models), tag="combo_policy",
                            default_value=os.path.basename(
                                self.c.policy_model) if self.c.policy_model
                            else "", width=210,
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
        # ensure the split pair starts matched when linked
        if self.c.link_split and self.c.vision_models and self.c.policy_models:
            self._sync_split(source="vision")

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
