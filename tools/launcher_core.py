"""Headless launcher logic shared by the start panel — all the state,
detection, persistence and the START flow, with no UI. The view
(dearpygui) only reads/writes these fields and calls start().

Two run modes:
  tech   — beamngpy: map + vehicle + traffic, our camera sensor and
           direct control. Needs a BeamNG.tech tech.key.
  screen — no tech.key: capture the game window (player uses the hood
           cam) and drive a virtual wheel. No map/vehicle/traffic
           choices (the player sets those in-game).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "launcher_beamng.json")
TECH_PORT = 64256
DEFAULT_MODEL = os.path.join(ROOT, "models", "driving_supercombo.onnx")

STEAM_CANDIDATES = [
    "~/.local/share/Steam/steamapps/common/BeamNG.drive",
    "~/.steam/steam/steamapps/common/BeamNG.drive",
    "~/.local/share/Steam/steamapps/common/BeamNG.tech",
]
UTILITY_LEVELS = {"template", "smallgrid", "autotest", "garage_v2",
                  "showroom_v2"}
VEHICLES = ["bastion", "pickup"]


def detect_beamng() -> str:
    for p in STEAM_CANDIDATES:
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            return p
    return ""


def detect_tech_key(install: str) -> bool:
    cands = [os.path.join(install, "tech.key")] if install else []
    for d in ("BeamNG.tech", "BeamNG.drive"):
        cands.append(os.path.expanduser(f"~/.local/share/BeamNG/{d}/tech.key"))
        cands.append(os.path.expanduser(
            f"~/.local/share/BeamNG/{d}/current/tech.key"))
    return any(os.path.isfile(c) for c in cands)


def scan_levels(install: str) -> list[str]:
    levels = []
    d = os.path.join(install, "content", "levels")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d), key=str.lower):
            if (f.endswith(".zip") and not f.startswith("_")
                    and f[:-4].lower() not in UTILITY_LEVELS):
                levels.append(f[:-4])
    if not levels:
        levels = ["west_coast_usa", "east_coast_usa", "italy",
                  "jungle_rock_island", "hirochi_raceway"]
    if "west_coast_usa" in levels:
        levels.remove("west_coast_usa")
    return ["west_coast_usa"] + levels


def _scan(pred) -> list[str]:
    d = os.path.join(ROOT, "models")
    out = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d), key=str.lower):
            if f.endswith(".onnx") and pred(f):
                out.append(os.path.join(d, f))
    return out


def scan_models() -> list[str]:
    out = _scan(lambda f: not f.startswith(("driving_vision",
                                            "driving_policy")))
    return out or [DEFAULT_MODEL]


def scan_split(kind: str) -> list[str]:
    return _scan(lambda f: f.startswith(f"driving_{kind}"))


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


class LauncherCore:
    def __init__(self):
        cfg = {}
        if os.path.isfile(CONFIG_PATH):
            try:
                cfg = json.load(open(CONFIG_PATH))
            except Exception:
                cfg = {}
        self.path = cfg.get("beamng_path") or detect_beamng()
        self.tech_key = bool(cfg.get("tech_key", detect_tech_key(self.path)))
        self.levels = scan_levels(self.path)
        self.map = cfg.get("map", "west_coast_usa")
        if self.map not in self.levels:
            self.map = self.levels[0]
        self.vehicles = VEHICLES
        self.vehicle = cfg.get("vehicle", "bastion")
        if self.vehicle not in self.vehicles:
            self.vehicle = "bastion"
        self.models = scan_models()
        self.model = cfg.get("model") or self.models[0]
        if not os.path.isfile(self.model):
            self.model = self.models[0]
        self.traffic = max(0, min(12, int(cfg.get("traffic", 0))))
        # Freeroam: attach to a car YOU spawn (native controls kept — Tab,
        # reset, spawning a lead car all work) instead of the scripted
        # scenario. Forces attach mode even for west_coast_usa.
        self.freeroam = bool(cfg.get("freeroam", False))
        self.arch = cfg.get("arch", "supercombo")
        if self.arch not in ("supercombo", "split"):
            self.arch = "supercombo"
        self.vision_models = scan_split("vision")
        self.policy_models = scan_split("policy")
        self.vision_model = cfg.get("vision_model") or (
            self.vision_models[0] if self.vision_models else "")
        self.policy_model = cfg.get("policy_model") or (
            self.policy_models[0] if self.policy_models else "")
        # screen mode: in-game hood-cam horizontal FOV
        self.screen_fov = float(cfg.get("screen_fov", 90.0))
        self._cfg_prefix = cfg.get("panel_cmd_prefix")

        self.status: list[str] = ["ready"]
        self.busy = False
        self.launched = False
        self.done_at = 0.0     # monotonic when a good START finished

    # ---- persistence ----

    def save(self) -> None:
        data = {"beamng_path": self.path, "tech_key": self.tech_key,
                "map": self.map, "vehicle": self.vehicle,
                "model": self.model, "traffic": self.traffic,
                "freeroam": self.freeroam,
                "arch": self.arch, "vision_model": self.vision_model,
                "policy_model": self.policy_model,
                "screen_fov": self.screen_fov}
        if self._cfg_prefix:
            data["panel_cmd_prefix"] = self._cfg_prefix
        json.dump(data, open(CONFIG_PATH, "w"), indent=2)

    def log(self, msg: str) -> None:
        self.status.append(msg)
        self.status = self.status[-10:]
        print(f"[start] {msg}", flush=True)

    # ---- START ----

    def start(self) -> None:
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._flow, daemon=True).start()

    def _panel_cmd(self, args: list[str]) -> list[str]:
        prefix = self._cfg_prefix or [
            sys.executable, os.path.join(ROOT, "tools", "control_panel.py")]
        return prefix + args

    def _spawn_panel(self, args: list[str]) -> None:
        env = dict(os.environ)
        env.setdefault("GLIBC_TUNABLES", "glibc.rtld.execstack=2")
        # capture the control panel's stdout/stderr — it's spawned
        # detached, so without this any crash is invisible (window
        # pops up and vanishes). Read debug_out/control_panel_last.log.
        os.makedirs(os.path.join(ROOT, "debug_out"), exist_ok=True)
        logf = open(os.path.join(ROOT, "debug_out",
                                 "control_panel_last.log"), "w")
        subprocess.Popen(self._panel_cmd(args), cwd=ROOT, env=env,
                         start_new_session=True, stdout=logf,
                         stderr=subprocess.STDOUT)
        self.log("control panel log -> debug_out/control_panel_last.log")
        self.launched = True
        self.done_at = time.monotonic()

    def _flow(self) -> None:
        try:
            self.save()
            if self.tech_key:
                self._flow_tech()
            else:
                self._flow_screen()
        finally:
            self.busy = False

    # -- screen mode: no beamngpy, capture the running game --
    def _flow_screen(self) -> None:
        self.log("screen mode: make sure BeamNG is running, you are in "
                 "a car, and the HOOD camera is active.")
        args = ["--screen", "--fov", str(int(self.screen_fov))]
        if not self._model_args(args):
            return
        self._spawn_panel(args)
        self.log("control panel launching — it captures the game window "
                 "and drives a virtual wheel.")
        self.log("bind the virtual wheel once in the game's controls if "
                 "steering/pedals don't respond.")

    # -- tech mode: beamngpy scenario / freeroam interception --
    def _flow_tech(self) -> None:
        if not os.path.isdir(self.path):
            self.log(f"BeamNG location not found: {self.path!r}")
            return
        if not port_open():
            exe = find_binary(self.path)
            if exe is None:
                self.log("no BinLinux binary under that location — is "
                         "this the BeamNG install dir?")
                return
            self.log(f"launching BeamNG (tech server on {TECH_PORT}) ...")
            env = dict(os.environ)
            env["RADV_DEBUG"] = (env.get("RADV_DEBUG", "")
                                 + ",nocompute").lstrip(",")
            subprocess.Popen(
                ["nice", "-n", "10", exe, "-nosteam", "-tcom",
                 "-tport", str(TECH_PORT)],
                cwd=self.path, env=env, start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.log("waiting for the tech port (game boot takes a "
                     "minute or two) ...")
            t_end = time.monotonic() + 240
            while time.monotonic() < t_end and not port_open():
                time.sleep(2.0)
            if not port_open():
                self.log("gave up waiting for the tech port.")
                return

        # The tport socket opens EARLY in boot — before the game can
        # actually handle beamngpy commands. Spawning the panel now
        # makes its bng.open() hit the game mid-init, throw, and crash
        # (game left at the menu, no scene). Wait for a REAL handshake.
        self.log("BeamNG port up — waiting for it to be ready ...")
        if not self._wait_ready():
            self.log("BeamNG never became ready for beamngpy.")
            return
        self.log("BeamNG is ready.")

        attach = self.freeroam or self.map != "west_coast_usa"
        if attach and not self._probe_attach():
            return
        args = ["--map", self.map, "--vehicle", self.vehicle]
        if attach:
            args.append("--attach")
        if not self._model_args(args):
            return
        if self.traffic > 0:
            args += ["--traffic", str(self.traffic)]
        self._spawn_panel(args)
        self.log("control panel launching — its window appears once the "
                 "scenario loads.")

    def _wait_ready(self, timeout_s: float = 180.0) -> bool:
        """Poll a real beamngpy handshake until it succeeds — the port
        opening isn't enough, the game must accept a connection."""
        from beamngpy import BeamNGpy
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            try:
                bng = BeamNGpy("127.0.0.1", TECH_PORT)
                bng.open(launch=False)
                bng.disconnect()
                return True
            except Exception:
                time.sleep(3.0)
        return False

    def _probe_attach(self) -> bool:
        self.log(f"checking for a drivable car ({self.map}) ...")
        try:
            from beamngpy import BeamNGpy
            bng = BeamNGpy("127.0.0.1", TECH_PORT)
            bng.open(launch=False)
            vehicles = bng.vehicles.get_current(include_config=False)
            bng.disconnect()
        except Exception as exc:
            self.log(f"probe failed: {exc}")
            return False
        if not vehicles:
            self.log(f"in the game: Freeroam -> {self.map}, enter a car, "
                     "then press START again.")
            return False
        self.log(f"found {list(vehicles)} — hooking the current car.")
        return True

    def _model_args(self, args: list[str]) -> bool:
        """Append model flags; False (and logs) if a file is missing."""
        if self.arch == "split":
            for p, nm in ((self.vision_model, "vision"),
                          (self.policy_model, "policy")):
                if not os.path.isfile(p):
                    self.log(f"{nm} model missing: {p or '(none)'}")
                    return False
            args += ["--split", "--vision", self.vision_model,
                     "--policy", self.policy_model]
        elif os.path.realpath(self.model) != os.path.realpath(DEFAULT_MODEL):
            if not os.path.isfile(self.model):
                self.log(f"model file missing: {self.model}")
                return False
            args += ["--model", self.model]
        return True
