#!/usr/bin/env python3
"""Start panel — the front door for trying this project.

Collects what we need to run, then launches everything:
  - BeamNG install location (auto-detected from common Steam paths)
  - tech.key checkbox: with a BeamNG.tech license we drive through
    beamngpy (camera sensor + direct control). Without one the plan
    is player-view capture + an emulated controller — NOT BUILT YET,
    so Start is blocked with a note.
  - Map dropdown (scanned from the install's content/levels):
    west_coast_usa runs our scripted scenario with the calibrated
    spawn. Any other map goes through FREEROAM INTERCEPTION — the
    user loads the map in-game, enters a car, presses START and the
    model hooks the current vehicle (control_panel --attach).
  - Vehicle dropdown (bastion only for now — CAL's gear table and
    wheelbase are measured for it).

Usage: .venv/bin/python3 tools/start_panel.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WINDOW = "Beamng-onnx start panel"      # ASCII only (cv2-Qt quirk)
# Same footprint as the control panel window.
UI_W, UI_H = 760, 1252
FONT = cv2.FONT_HERSHEY_SIMPLEX
C_BG = (26, 18, 10)
C_LABEL = (150, 138, 118)
C_WHITE = (235, 240, 240)
C_BLUE = (232, 163, 74)
C_GREEN = (110, 200, 110)
C_DIM = (60, 50, 38)
C_FIELD = (44, 34, 22)
C_RED = (70, 70, 210)

CONFIG_PATH = os.path.join(ROOT, "launcher_beamng.json")
TECH_PORT = 64256

STEAM_CANDIDATES = [
    "~/.local/share/Steam/steamapps/common/BeamNG.drive",
    "~/.steam/steam/steamapps/common/BeamNG.drive",
    "~/.local/share/Steam/steamapps/common/BeamNG.tech",
]


def detect_beamng() -> str:
    for p in STEAM_CANDIDATES:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            return p
    return ""


def detect_tech_key(install: str) -> bool:
    cands = [os.path.join(install, "tech.key")] if install else []
    cands += [os.path.expanduser(f"~/.local/share/BeamNG/{d}/tech.key")
              for d in ("BeamNG.tech", "BeamNG.drive")]
    cands += [os.path.expanduser(
        f"~/.local/share/BeamNG/{d}/current/tech.key")
        for d in ("BeamNG.tech", "BeamNG.drive")]
    return any(os.path.isfile(c) for c in cands)


def scan_levels(install: str) -> list[str]:
    """Level names from the install's content/levels zips.
    west_coast_usa first (it's the fully-scripted one)."""
    utility = {"template", "smallgrid", "autotest", "garage_v2",
               "showroom_v2"}    # not drivable roads
    levels = []
    d = os.path.join(install, "content", "levels")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d), key=str.lower):
            if f.endswith(".zip") and not f.startswith("_") \
                    and f[:-4].lower() not in utility:
                levels.append(f[:-4])
    if not levels:
        levels = ["west_coast_usa", "east_coast_usa", "italy",
                  "jungle_rock_island", "hirochi_raceway"]
    if "west_coast_usa" in levels:
        levels.remove("west_coast_usa")
    return ["west_coast_usa"] + levels


DEFAULT_MODEL = os.path.join(ROOT, "models", "driving_supercombo.onnx")
BROWSE = "browse for an .onnx file ..."


def scan_models() -> list[str]:
    """Supercombo-compatible .onnx files under models/ (full paths).
    Split vision/policy halves are excluded — they're not runnable as
    a supercombo; the SPLIT POLICY mode scans them separately."""
    d = os.path.join(ROOT, "models")
    out = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d), key=str.lower):
            if (f.endswith(".onnx")
                    and not f.startswith(("driving_vision",
                                          "driving_policy"))):
                out.append(os.path.join(d, f))
    return out or [DEFAULT_MODEL]


def scan_split(kind: str) -> list[str]:
    """driving_<kind>*.onnx halves under models/ (kind: vision|policy)."""
    d = os.path.join(ROOT, "models")
    out = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d), key=str.lower):
            if f.endswith(".onnx") and f.startswith(f"driving_{kind}"):
                out.append(os.path.join(d, f))
    return out


def browse_onnx() -> str | None:
    """Native file picker (tkinter ships with the venv python)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Pick a supercombo-compatible .onnx",
            initialdir=os.path.join(ROOT, "models"),
            filetypes=[("ONNX model", "*.onnx"), ("all files", "*")])
        root.destroy()
        return path or None
    except Exception as exc:
        print(f"[start] file dialog unavailable ({exc}) — run the "
              f"control panel with --model <path> instead", flush=True)
        return None


def port_open(port: int = TECH_PORT) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def find_binary(install: str) -> str | None:
    for rel in ("BinLinux/BeamNG.drive.x64", "BinLinux/BeamNG.tech.x64"):
        p = os.path.join(install, rel)
        if os.path.isfile(p):
            return p
    return None


class StartPanel:
    def __init__(self) -> None:
        cfg = {}
        if os.path.isfile(CONFIG_PATH):
            try:
                cfg = json.load(open(CONFIG_PATH))
            except Exception:
                cfg = {}
        self.path = cfg.get("beamng_path") or detect_beamng()
        self.tech_key = bool(cfg.get("tech_key",
                                     detect_tech_key(self.path)))
        self.levels = scan_levels(self.path)
        self.map = cfg.get("map", "west_coast_usa")
        if self.map not in self.levels:
            self.map = self.levels[0]
        # bastion = the measured-and-tuned sedan; pickup = Gavril
        # D-Series crew-cab 4x4 auto (2020 Sierra AT4 recreation,
        # camera at the real user's 1.66 m comma mount height)
        self.vehicles = ["bastion", "pickup"]
        self.vehicle = cfg.get("vehicle", "bastion")
        if self.vehicle not in self.vehicles:
            self.vehicle = "bastion"
        self.models = scan_models()
        self.model = cfg.get("model") or DEFAULT_MODEL
        if not os.path.isfile(self.model):
            self.model = self.models[0]
        self.traffic = max(0, min(12, int(cfg.get("traffic", 0))))
        # model architecture: one supercombo vs split vision+policy
        self.arch = cfg.get("arch", "supercombo")
        if self.arch not in ("supercombo", "split"):
            self.arch = "supercombo"
        self.vision_models = scan_split("vision")
        self.policy_models = scan_split("policy")
        self.vision_model = cfg.get("vision_model") or (
            self.vision_models[0] if self.vision_models else "")
        self.policy_model = cfg.get("policy_model") or (
            self.policy_models[0] if self.policy_models else "")
        if not os.path.isfile(self.vision_model) and self.vision_models:
            self.vision_model = self.vision_models[0]
        if not os.path.isfile(self.policy_model) and self.policy_models:
            self.policy_model = self.policy_models[0]
        self._cfg_prefix = cfg.get("panel_cmd_prefix")

        self.focus: str | None = None      # "path" while typing
        self.open_dropdown: str | None = None
        self.status: list[str] = ["ready"]
        self.busy = False
        self.launched = False
        self.close_at = float("inf")   # auto-close after a good START
        self._rects: dict[str, tuple[int, int, int, int]] = {}
        self._drop_rects: list[tuple[tuple[int, int, int, int], str]] = []

    # ---- persistence ----

    def save(self) -> None:
        data = {"beamng_path": self.path, "tech_key": self.tech_key,
                "map": self.map, "vehicle": self.vehicle,
                "model": self.model, "traffic": self.traffic,
                "arch": self.arch, "vision_model": self.vision_model,
                "policy_model": self.policy_model}
        if self._cfg_prefix:
            data["panel_cmd_prefix"] = self._cfg_prefix
        json.dump(data, open(CONFIG_PATH, "w"), indent=2)

    def log(self, msg: str) -> None:
        self.status.append(msg)
        self.status = self.status[-8:]
        print(f"[start] {msg}", flush=True)

    # ---- the START flow (worker thread — UI keeps drawing) ----

    def start(self) -> None:
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._start_flow, daemon=True).start()

    def _start_flow(self) -> None:
        try:
            self.save()
            if not self.tech_key:
                self.log("no tech.key: player-view + virtual-gamepad "
                         "mode is not built yet.")
                self.log("get a (free for personal use) BeamNG.tech "
                         "license, or watch this repo.")
                return
            if not os.path.isdir(self.path):
                self.log(f"BeamNG location not found: {self.path!r}")
                return
            if not port_open():
                exe = find_binary(self.path)
                if exe is None:
                    self.log("no BinLinux binary under that location "
                             "- is this the BeamNG install dir?")
                    return
                self.log("launching BeamNG (tech server on "
                         f"port {TECH_PORT}) ...")
                env = dict(os.environ)
                env["RADV_DEBUG"] = (env.get("RADV_DEBUG", "")
                                     + ",nocompute").lstrip(",")
                # start_new_session: detach the game from our terminal's
                # process group. Since the launcher auto-closes after
                # START, users close the (now idle-looking) terminal —
                # without setsid that terminal's hangup signal reached
                # the game and shut it down cleanly mid-drive.
                subprocess.Popen(
                    ["nice", "-n", "10", exe, "-nosteam",
                     "-tcom", "-tport", str(TECH_PORT)],
                    cwd=self.path, env=env, start_new_session=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.log("waiting for the tech port (game boot takes "
                         "a minute or two) ...")
                t_end = time.monotonic() + 240
                while time.monotonic() < t_end and not port_open():
                    time.sleep(2.0)
                if not port_open():
                    self.log("gave up waiting for the tech port.")
                    return
                self.log("BeamNG is up.")

            attach = self.map != "west_coast_usa"
            if attach:
                # Freeroam interception: the player must be sitting in
                # a car on their chosen map before we hook it.
                self.log(f"checking for a drivable car ({self.map}) ...")
                try:
                    from beamngpy import BeamNGpy
                    bng = BeamNGpy("127.0.0.1", TECH_PORT)
                    bng.open(launch=False)
                    vehicles = bng.vehicles.get_current(
                        include_config=False)
                    bng.disconnect()
                except Exception as exc:
                    self.log(f"probe failed: {exc}")
                    return
                if not vehicles:
                    self.log(f"in the game: Freeroam -> {self.map}, "
                             "enter a car, then press START again.")
                    return
                self.log(f"found vehicle(s): {list(vehicles)} - "
                         "hooking the current car.")
            args = ["--map", self.map, "--vehicle", self.vehicle]
            if attach:
                args.append("--attach")
            if self.arch == "split":
                for p, nm in ((self.vision_model, "vision"),
                              (self.policy_model, "policy")):
                    if not os.path.isfile(p):
                        self.log(f"{nm} model missing: {p or '(none)'}")
                        return
                args += ["--split", "--vision", self.vision_model,
                         "--policy", self.policy_model]
            elif os.path.realpath(self.model) != os.path.realpath(
                    DEFAULT_MODEL):
                if not os.path.isfile(self.model):
                    self.log(f"model file missing: {self.model}")
                    return
                args += ["--model", self.model]
            if self.traffic > 0:
                args += ["--traffic", str(self.traffic)]
            # `panel_cmd_prefix` in launcher_beamng.json reroutes the
            # control-panel spawn (e.g. into a distrobox where the
            # ROCm runtime lives); our args are appended verbatim.
            # Default: this same python, this repo.
            prefix = self._cfg_prefix or [
                sys.executable,
                os.path.join(ROOT, "tools", "control_panel.py")]
            env = dict(os.environ)
            env.setdefault("GLIBC_TUNABLES", "glibc.rtld.execstack=2")
            # detached for the same reason as the game above: the
            # control panel must survive the terminal that ran start.sh
            subprocess.Popen(prefix + args, cwd=ROOT, env=env,
                             start_new_session=True)
            self.log("control panel launching - its window appears "
                     "once the scenario loads.")
            self.log("closing this launcher ...")
            self.launched = True
            self.close_at = time.monotonic() + 3.0
        finally:
            self.busy = False

    # ---- UI ----

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.open_dropdown:
            for (bx, by, bw, bh), val in self._drop_rects:
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    if self.open_dropdown == "map":
                        self.map = val
                    elif self.open_dropdown == "model":
                        if val == BROWSE:
                            p = browse_onnx()
                            if p:
                                self.model = p
                                if p not in self.models:
                                    self.models.append(p)
                        else:
                            for m in self.models:
                                if os.path.basename(m) == val:
                                    self.model = m
                    elif self.open_dropdown in ("vision", "policy"):
                        lst = (self.vision_models
                               if self.open_dropdown == "vision"
                               else self.policy_models)
                        if val == BROWSE:
                            p = browse_onnx()
                            if p:
                                if p not in lst:
                                    lst.append(p)
                                if self.open_dropdown == "vision":
                                    self.vision_model = p
                                else:
                                    self.policy_model = p
                        else:
                            for m in lst:
                                if os.path.basename(m) == val:
                                    if self.open_dropdown == "vision":
                                        self.vision_model = m
                                    else:
                                        self.policy_model = m
                    else:
                        self.vehicle = val
            self.open_dropdown = None
            return
        self.focus = None
        for key, (bx, by, bw, bh) in self._rects.items():
            if not (bx <= x <= bx + bw and by <= y <= by + bh):
                continue
            if key == "path":
                self.focus = "path"
            elif key == "tech":
                self.tech_key = not self.tech_key
            elif key in ("map", "vehicle", "model", "vision", "policy"):
                self.open_dropdown = key
            elif key == "arch":
                self.arch = ("split" if self.arch == "supercombo"
                             else "supercombo")
            elif key == "traffic_dn":
                self.traffic = max(0, self.traffic - 1)
            elif key == "traffic_up":
                self.traffic = min(12, self.traffic + 1)
            elif key == "start":
                self.start()
            return

    def on_key(self, key: int) -> None:
        if self.focus != "path":
            return
        if key in (8, 127):                      # backspace / delete
            self.path = self.path[:-1]
        elif key == 13:                          # enter
            self.focus = None
            self.levels = scan_levels(self.path)
            self.tech_key = detect_tech_key(self.path)
        elif 32 <= key <= 126:
            self.path += chr(key)

    def draw(self) -> np.ndarray:
        c = np.zeros((UI_H, UI_W, 3), dtype=np.uint8)
        c[:] = C_BG
        self._rects.clear()

        cv2.putText(c, "BEAMNG-ONNX", (36, 78), FONT, 1.4, C_WHITE, 3,
                    cv2.LINE_AA)
        cv2.putText(c, "openpilot driving models inside BeamNG",
                    (38, 116), FONT, 0.55, C_LABEL, 1, cv2.LINE_AA)

        def field(y: int, key: str, text: str, focused: bool,
                  h: int = 44) -> None:
            box = (36, y, UI_W - 72, h)
            self._rects[key] = box
            cv2.rectangle(c, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), C_FIELD, -1)
            cv2.rectangle(c, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]),
                          C_BLUE if focused else C_DIM, 1)
            shown = text if len(text) < 62 else "..." + text[-59:]
            cv2.putText(c, shown + ("_" if focused else ""),
                        (box[0] + 12, box[1] + 29), FONT, 0.52, C_WHITE,
                        1, cv2.LINE_AA)

        y = 170
        cv2.putText(c, "BEAMNG LOCATION", (36, y), FONT, 0.55, C_LABEL,
                    1, cv2.LINE_AA)
        field(y + 14, "path", self.path or "(click and type the "
              "install path)", self.focus == "path")
        ok = os.path.isdir(self.path)
        cv2.putText(c, "found" if ok else "not found",
                    (36, y + 84), FONT, 0.5,
                    C_GREEN if ok else C_RED, 1, cv2.LINE_AA)

        y = 300
        box = (36, y, 30, 30)
        self._rects["tech"] = (36, y, 560, 34)
        cv2.rectangle(c, (box[0], box[1]),
                      (box[0] + box[2], box[1] + box[3]), C_FIELD, -1)
        cv2.rectangle(c, (box[0], box[1]),
                      (box[0] + box[2], box[1] + box[3]), C_DIM, 1)
        if self.tech_key:
            cv2.putText(c, "x", (box[0] + 7, box[1] + 24), FONT, 0.8,
                        C_GREEN, 2, cv2.LINE_AA)
        cv2.putText(c, "I have a tech.key (BeamNG.tech license)",
                    (84, y + 23), FONT, 0.55, C_WHITE, 1, cv2.LINE_AA)
        if not self.tech_key:
            cv2.putText(c, "without one: player-view capture + virtual"
                        " gamepad - coming later, START is disabled",
                        (36, y + 58), FONT, 0.45, C_RED, 1, cv2.LINE_AA)

        def dropdown(y: int, key: str, label: str, value: str,
                     note: str, x0: int = 36,
                     width: int = UI_W - 72) -> None:
            if label:
                cv2.putText(c, label, (x0, y), FONT, 0.55, C_LABEL, 1,
                            cv2.LINE_AA)
            box = (x0, y + 14, width, 44)
            self._rects[key] = box
            cv2.rectangle(c, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), C_FIELD, -1)
            cv2.rectangle(c, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), C_DIM, 1)
            val = value
            max_chars = max(6, (width - 44) // 11)
            if len(val) > max_chars:
                val = val[:max_chars - 2] + ".."
            cv2.putText(c, val, (box[0] + 12, box[1] + 29), FONT,
                        0.58, C_WHITE, 1, cv2.LINE_AA)
            cv2.putText(c, "v", (box[0] + box[2] - 28, box[1] + 29),
                        FONT, 0.6, C_LABEL, 1, cv2.LINE_AA)
            if note:
                cv2.putText(c, note, (x0, y + 84), FONT, 0.45, C_LABEL,
                            1, cv2.LINE_AA)

        map_note = ("scripted scenario at the calibrated highway spawn"
                    if self.map == "west_coast_usa" else
                    "freeroam mode: load the map in-game, enter a car, "
                    "then press START - the model hooks that car")
        dropdown(430, "map", "MAP", self.map, map_note)
        dropdown(560, "vehicle", "VEHICLE", self.vehicle,
                 "CAL gear table + wheelbase are measured for the "
                 "bastion")
        # architecture toggle: the title line is the switch
        arch_label = ("SUPERCOMBO" if self.arch == "supercombo"
                      else "SPLIT POLICY")
        arch_hint = ("[click for split vision+policy]"
                     if self.arch == "supercombo"
                     else "[click for single supercombo]")
        cv2.putText(c, arch_label, (36, 682), FONT, 0.55,
                    (120, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(c, arch_hint, (230, 682), FONT, 0.42, C_DIM, 1,
                    cv2.LINE_AA)
        self._rects["arch"] = (36, 664, 480, 24)
        if self.arch == "supercombo":
            dropdown(690, "model", "", os.path.basename(self.model),
                     "supercombo-compatible .onnx; pick 'browse' in "
                     "the list to point anywhere on disk")
        else:
            half = (UI_W - 72 - 16) // 2
            dropdown(692, "vision", "VISION",
                     os.path.basename(self.vision_model) or "(none)",
                     "", x0=36, width=half)
            dropdown(692, "policy", "POLICY",
                     os.path.basename(self.policy_model) or "(none)",
                     "", x0=36 + half + 16, width=half)

        # traffic scroller: [-] N [+]
        y = 788
        cv2.putText(c, "TRAFFIC", (36, y + 24), FONT, 0.55, C_LABEL,
                    1, cv2.LINE_AA)
        for key, bx, sym in (("traffic_dn", 170, "-"),
                             ("traffic_up", 292, "+")):
            box = (bx, y, 44, 38)
            self._rects[key] = box
            cv2.rectangle(c, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), C_FIELD, -1)
            cv2.rectangle(c, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), C_DIM, 1)
            cv2.putText(c, sym, (box[0] + 15, box[1] + 27), FONT, 0.8,
                        C_WHITE, 2, cv2.LINE_AA)
        cv2.putText(c, f"{self.traffic:2d}", (232, y + 27), FONT, 0.75,
                    C_WHITE, 2, cv2.LINE_AA)
        cv2.putText(c, "AI cars spawned around you (0 = empty roads)",
                    (356, y + 25), FONT, 0.45, C_LABEL, 1, cv2.LINE_AA)

        y = 842
        box = (36, y, UI_W - 72, 64)
        self._rects["start"] = box
        col = (60, 160, 60) if (self.tech_key and not self.busy) \
            else (60, 70, 60)
        cv2.rectangle(c, (box[0], box[1]),
                      (box[0] + box[2], box[1] + box[3]), col, -1)
        label = "WORKING ..." if self.busy else "START"
        cv2.putText(c, label, (UI_W // 2 - 60, y + 42), FONT, 0.95,
                    C_WHITE, 2, cv2.LINE_AA)

        y = 950
        cv2.putText(c, "STATUS", (36, y), FONT, 0.55, C_LABEL, 1,
                    cv2.LINE_AA)
        for i, line in enumerate(self.status[-8:]):
            cv2.putText(c, line, (36, y + 30 + 26 * i), FONT, 0.48,
                        C_WHITE, 1, cv2.LINE_AA)

        cv2.putText(c, "github.com/SpysyWeeb/Beamng-onnx  -  branch "
                    "linux", (36, UI_H - 24), FONT, 0.45, C_DIM, 1,
                    cv2.LINE_AA)

        # dropdown overlay drawn last so it sits on top
        self._drop_rects.clear()
        if self.open_dropdown:
            if self.open_dropdown == "map":
                opts = self.levels
            elif self.open_dropdown == "model":
                opts = [os.path.basename(m) for m in self.models] \
                    + [BROWSE]
            elif self.open_dropdown == "vision":
                opts = [os.path.basename(m) for m in self.vision_models] \
                    + [BROWSE]
            elif self.open_dropdown == "policy":
                opts = [os.path.basename(m) for m in self.policy_models] \
                    + [BROWSE]
            else:
                opts = self.vehicles
            bx, by, bw, _ = self._rects[self.open_dropdown]
            oy = by + 46
            for opt in opts[:16]:
                r = (bx, oy, bw, 34)
                self._drop_rects.append((r, opt))
                cur = opt == {
                    "map": self.map,
                    "model": os.path.basename(self.model),
                    "vision": os.path.basename(self.vision_model),
                    "policy": os.path.basename(self.policy_model),
                    "vehicle": self.vehicle}[self.open_dropdown]
                cv2.rectangle(c, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]),
                              (58, 46, 30) if cur else (38, 30, 20), -1)
                cv2.rectangle(c, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]),
                              C_DIM, 1)
                cv2.putText(c, opt, (r[0] + 12, r[1] + 24), FONT, 0.52,
                            C_WHITE, 1, cv2.LINE_AA)
                oy += 34
        return c


def main() -> int:
    panel = StartPanel()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(UI_W * 0.8), int(UI_H * 0.8))
    mouse_ok = False
    for _ in range(20):
        cv2.imshow(WINDOW, panel.draw())
        cv2.waitKey(20)
        try:
            cv2.setMouseCallback(WINDOW, panel.on_mouse)
            mouse_ok = True
            break
        except cv2.error:
            time.sleep(0.05)
    if not mouse_ok:
        print("[start] mouse unavailable in this cv2 build", flush=True)
    while True:
        cv2.imshow(WINDOW, panel.draw())
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        if time.monotonic() > panel.close_at:
            break                     # control panel is on its way
        if key != 255:
            panel.on_key(key)
    panel.save()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
