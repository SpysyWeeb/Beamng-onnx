#!/usr/bin/env python3
"""M3 control panel — closed-loop driving with the viewer inside.

One window: the live overlay view on top, a button strip + HUD at the
bottom. The model steers and (optionally) drives throttle/brake via
simsteer's controllers; buttons/keys send desires (lane changes, turns)
straight into the model's desire input — the same mechanism openpilot's
blinker uses.

Pipeline per 20 Hz frame:
    camera stream_raw -> FrameQueue warp -> model (desire vec) -> decode
    -> LateralController (curvature -> axis via LiveParams learned rack)
    -> LongitudinalController (plan accel + corner braking + lead ACC)
    -> vehicle.control()

LiveParams (game="beamng") learns axis->wheel online. Wheel angle is
synthesized from the MODEL's ego yaw rate (pose[5]) via the bicycle
model — BeamNG doesn't expose road-wheel angle, and this matches the
Forza/AC profile upstream. Before first engage, run CAL (button or 'r'):
a scripted slalom that feeds the learner until trusted (~100 samples).

Keys:
    e engage/disengage      f force-engage (skip trust gate)
    a/d lane change L/R     z/c turn L/R
    l toggle longitudinal   g toggle BeamNG AI driving (calibration aid)
    r run auto-calibration  -/= max speed down/up
    ESC quit (sends neutral control)

Usage (BeamNG running via launch_beamng.sh):
    .venv/bin/python3 tools/control_panel.py [--split] [--scale 0.8]
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

np.seterr(divide="ignore", invalid="ignore")

from simsteer.core.calibration import Calibration
from simsteer.core.constants import DESIRE_LEN
from simsteer.core.control import ControllerConfig, LateralController, LongitudinalController
from simsteer.core.learners.livecalib import LiveCalib
from simsteer.core.learners.liveparams import LiveParams
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue
from simsteer.core.supercombo import SupercomboModel
from simsteer.ui.overlay import draw_overlay

from beamng.world import (BeamNGOnnxWorld, CAM_W, CAM_H, CAM_FOV_H_DEG,
                          CAM_HEIGHT_M, CAM_LATERAL_SIGN, SPAWN_POS,
                          VEHICLE_SPECS,
                          SPAWN_ROT_QUAT)

# ASCII only: cv2's Qt backend fails setMouseCallback ("NULL window
# handler") when the window name contains non-ASCII (e.g. an em-dash).
WINDOW = "Beamng-onnx control panel"
# wheelbase now comes from beamng.world.VEHICLE_SPECS per vehicle
# Vertical-rectangle layout (op-replay-clipper style): model view on
# top, telemetry panel below, buttons + HUD at the bottom.
UI_W = 760
CAM_VIEW_H = UI_W * CAM_H // CAM_W      # camera aspect preserved
TELEM_H = 704
BTN_ROW_H = 42
HUD_H = 84
TOTAL_H = CAM_VIEW_H + TELEM_H + 2 * BTN_ROW_H + HUD_H
FONT = cv2.FONT_HERSHEY_SIMPLEX
# telemetry palette (BGR) — dark navy like the reference UI
C_BG = (26, 18, 10)
C_LABEL = (150, 138, 118)
C_WHITE = (235, 240, 240)
C_BLUE = (232, 163, 74)     # desired (lat accel)
C_GREEN = (126, 209, 126)   # actual / measured
C_YELLOW = (74, 192, 232)   # target steer %
C_ORANGE = (74, 130, 232)   # applied steer %
C_DIM = (60, 50, 38)
DESIRE_HOLD_S = {1: 3.0, 2: 3.0, 3: 2.5, 4: 2.5}   # turn L/R, lane L/R
DESIRE_NAME = {1: "TURN L", 2: "TURN R", 3: "LANE L", 4: "LANE R"}
WARMUP_FRAMES = 100        # model context fill before engage allowed
# Calibration v3: deterministic system-ID. Teleport to the straight
# highway spawn, drive straight to CAL_V, apply short alternating
# steering pulses, measure the yaw response through the model's pose,
# and solve the rack ratio by least squares (through the origin —
# FILTER_DIRECT is linear with b~0, measured). No BeamNG AI (drives
# mid-road with tiny inputs -> poor excitation, fit differed run to
# run) and no user. Alternating pulses keep net heading ~zero so the
# car snakes around the lane instead of leaving the road.
CAL_V = 8.0                      # m/s during pulses (drift ~ v^2 k)
CAL_PULSES = [+0.05, -0.05, +0.09, -0.09, +0.13, -0.13]
CAL_PULSE_S = 1.4                # per pulse
CAL_PULSE_SETTLE_S = 0.5         # ignore samples while yaw settles
# Longitudinal system-ID: pedal steps on the straight, measure achieved
# accel (wheelspeed slope) and fit the real pedal->accel gains so a
# commanded 2 m/s^2 actually produces 2 m/s^2. (kind, pedal, duration).
# "rec" = recover speed between brake steps (advances early at 18 m/s).
# "slow" drops back to ~9 m/s between throttle steps so each measures
# from the same speed (at 20+ m/s the engine is power-limited and the
# full-throttle point reads LOW, corrupting the fit). "coast" measures
# pedals-free drag+engine-braking so the brake fit is net of it —
# conflating drag into the brake intercept made the coast gap swallow
# the whole ISO command range and the service brakes never fired.
CAL_LONG_SCRIPT = [
    ("thr", 0.35, 1.6), ("slow", 0.0, 5.0),
    ("thr", 0.70, 1.6), ("slow", 0.0, 5.0),
    ("thr", 1.00, 1.6),
    ("rec", 0.0, 6.0),
    ("coast", 0.0, 1.8),
    ("brk", 0.30, 1.2), ("rec", 0.0, 6.0),
    ("brk", 0.60, 1.2), ("rec", 0.0, 6.0),
    ("brk", 1.00, 2.5),
]
CAL_LONG_SETTLE_S = 0.4          # pedal/weight-transfer settling per step
CAL_DURATION_S = (4.0 + 15.0 + len(CAL_PULSES) * CAL_PULSE_S
                  + sum(s[2] for s in CAL_LONG_SCRIPT) + 2.0)


class Telemetry(threading.Thread):
    """Polls world telemetry (~10 Hz TCP) off the model loop."""

    def __init__(self, world: BeamNGOnnxWorld):
        super().__init__(daemon=True, name="telemetry")
        self.world = world
        self.lock = threading.Lock()
        self.data = {"v_ego": 0.0, "steering_deg": 0.0,
                     "steering_input": 0.0, "heading_rad": 0.0,
                     "throttle_input": 0.0, "brake_input": 0.0,
                     "pos": (0.0, 0.0, 0.0), "roll_glat": 0.0}
        self.ok = False
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                d = self.world.poll_telemetry()
                with self.lock:
                    self.data = d
                    self.ok = True
            except Exception:
                self.ok = False
            time.sleep(0.1)

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.data)

    def stop(self) -> None:
        self._stop.set()


MOD_PORT = 64257
MOD_CMDS = {"engage", "lane_l", "lane_r", "turn_l", "turn_r", "long",
            "cal", "ai", "spd_dn", "spd_up", "cam", "snap"}


class ControlSender(threading.Thread):
    """~100 Hz steering executor (openpilot's split: modeld plans at
    20 Hz, controlsd actuates at 100). The main loop submits the
    rate-limited steering TARGET at 20 Hz; this thread continuously
    first-order-smooths the applied axis toward it (the emulated-EPS
    filter used to run inside the 20 Hz loop, which staircased the
    wheel in 50 ms holds and stacked a full tick of lag on top of the
    filter's own). world.apply() is a ~8 ms TCP round-trip, so the
    real rate self-paces to whatever the vehicle socket sustains.

    Goes idle 0.25 s after the last submit — manual driving, CAL's
    direct applies, and the disengaged state are never fought."""

    def __init__(self, world: BeamNGOnnxWorld, steer_tau: float = 0.05):
        super().__init__(daemon=True, name="ctl-sender")
        self.world = world
        self.steer_tau = steer_tau
        self._steer_now = 0.0
        self._last_submit = 0.0
        self._latest: tuple | None = None
        self._ev = threading.Event()
        self._stop = threading.Event()

    def submit(self, steer: float, thr: float | None,
               brk: float | None) -> None:
        self._latest = (steer, thr, brk)
        self._last_submit = time.monotonic()
        self._ev.set()

    def run(self) -> None:
        last_t = time.monotonic()
        while not self._stop.is_set():
            cmd = self._latest
            now = time.monotonic()
            if cmd is None or now - self._last_submit > 0.25:
                self._latest = None
                if self._ev.wait(timeout=0.5):
                    self._ev.clear()
                    # wake from idle: snap to the fresh target so the
                    # wheel doesn't sweep in from a stale value
                    if self._latest is not None:
                        self._steer_now = self._latest[0]
                    last_t = time.monotonic()
                continue
            dt = min(now - last_t, 0.05)
            last_t = now
            target, thr, brk = cmd
            if self.steer_tau > 1e-4:
                alpha = 1.0 - math.exp(-dt / self.steer_tau)
                self._steer_now += alpha * (target - self._steer_now)
            else:
                self._steer_now = target
            try:
                self.world.apply(self._steer_now, thr, brk)
            except Exception:
                pass
            time.sleep(max(0.0, 0.01 - (time.monotonic() - now)))

    def stop(self) -> None:
        self._stop.set()
        self._ev.set()


class ModBridge(threading.Thread):
    """UDP link to the in-game imgui panel (beamng_mod/onnx-panel).

    The mod sends button clicks as single-token datagrams; we relay
    them into App.action(). We push a short status string back to the
    last-seen client so the in-game window shows engagement state."""

    def __init__(self, app: "App"):
        super().__init__(daemon=True, name="mod-bridge")
        self.app = app
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", MOD_PORT))
        self.sock.settimeout(0.3)
        self.client = None
        self._stop = threading.Event()

    def status_line(self) -> str:
        app = self.app
        v = app.tel.snapshot()["v_ego"] * 2.237
        cap = app.cfg.max_speed_mps * 2.237
        s = ("ENGAGED" if app.engaged else
             ("CAL" if app.cal_active else "manual"))
        line = (f"{s} | {v:.0f} mph | cap {cap:.0f} | "
                f"long {app.long_mode}")
        if app.desire_idx is not None:
            line += f" | {DESIRE_NAME[app.desire_idx]}"
        return line

    def run(self) -> None:
        last_push = 0.0
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(256)
                self.client = addr
                cmd = data.decode("ascii", "ignore").strip()
                if cmd in MOD_CMDS:
                    self.app.action(cmd)
                elif cmd == "hello" and time.monotonic() - last_push > 5:
                    print("[panel] in-game panel connected", flush=True)
            except socket.timeout:
                pass
            except OSError:
                break
            now = time.monotonic()
            if self.client and now - last_push > 0.4:
                last_push = now
                try:
                    self.sock.sendto(self.status_line().encode("ascii"),
                                     self.client)
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass


class Panel:
    """Buttons + HUD strip below the camera view."""

    DESIRE_KEYS = ("lane_l", "lane_r", "turn_l", "turn_r")

    def __init__(self, state: "App"):
        self.app = state
        self.buttons: list[tuple[tuple[int, int, int, int], str, str]] = []
        # desire button currently held down (mouse) — main loop repeats
        # the action each frame so the desire stays alive while held
        self.held_desire: str | None = None

    def layout(self, y0: int) -> None:
        rows = [
            [("engage", ""), ("long", ""), ("cal", "CAL"),
             ("cam", ""), ("ai", "AI")],
            [("lane_l", "< LANE"), ("lane_r", "LANE >"),
             ("turn_l", "< TURN"), ("turn_r", "TURN >"),
             ("spd_dn", "SPD-"), ("spd_up", "SPD+")],
        ]
        self.buttons = []
        for r, row in enumerate(rows):
            w = UI_W // len(row)
            for i, (key, lab) in enumerate(row):
                self.buttons.append(
                    ((i * w + 2, y0 + r * BTN_ROW_H, w - 4, BTN_ROW_H - 6),
                     key, lab))

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONUP:
            self.held_desire = None
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (bx, by, bw, bh), key, _ in self.buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.app.action(key)
                if key in self.DESIRE_KEYS:
                    self.held_desire = key
                return

    def draw(self, canvas: np.ndarray) -> None:
        app = self.app
        for (bx, by, bw, bh), key, lab in self.buttons:
            if key == "engage":
                lab = "DISENGAGE" if app.engaged else "ENGAGE"
                col = (60, 60, 200) if app.engaged else (60, 160, 60)
            elif key == "long":
                lab = f"LONG: {app.long_mode.upper()}"
                col = {"exp": (60, 120, 160), "chill": (90, 120, 60),
                       "off": (70, 70, 70)}[app.long_mode]
            elif key == "ai":
                lab = f"AI {'ON' if app.ai_on else 'off'}"
                col = (140, 90, 40) if app.ai_on else (70, 70, 70)
            elif key == "cal":
                col = (40, 110, 140) if app.cal_active else (70, 70, 70)
                if app.cal_active:
                    lab = f"CAL {int(app.cal_left_s)}s"
            elif key == "cam":
                lab = f"CAM {'LIVE' if app.calib_live else 'shdw'}"
                col = (150, 120, 40) if app.calib_live else (70, 70, 70)
            else:
                col = (70, 70, 70)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, -1)
            cv2.putText(canvas, lab, (bx + 7, by + 25), FONT, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)


class App:
    def __init__(self, args):
        vehicle = getattr(args, "vehicle", None) or "bastion"
        spec = VEHICLE_SPECS.get(vehicle, VEHICLE_SPECS["bastion"])
        self._cam_height = spec["cam_height_m"]
        self.wheelbase = spec["wheelbase_m"]
        # per-vehicle state files: the bastion's CAL rack fit / pedal
        # maps / camera pose must not bleed into other vehicles
        self._game_key = ("beamng" if vehicle == "bastion"
                          else f"beamng-{vehicle}")
        self.screen_mode = bool(getattr(args, "screen", False))
        self._vision_v = 0.0
        if self.screen_mode:
            # No-tech.key mode: game-window capture + virtual wheel,
            # zero beamngpy. Camera geometry comes from the captured
            # window + the in-game hood-cam FOV; LiveCalib learns
            # pitch/height on top like a fresh comma install. State
            # files live under their own "screen" key.
            from simsteer.io.screen_world import ScreenWorld
            self.world = ScreenWorld(
                window_hint=getattr(args, "window", None) or "BeamNG",
                fov_h_deg=float(getattr(args, "fov", None) or 90.0))
            self._game_key = "screen"
            self._cam_height = 1.2      # unknown; LiveCalib refines
            self.calib = Calibration(image_w=self.world.frame_w,
                                     image_h=self.world.frame_h,
                                     fov_h_deg=self.world.fov_h_deg,
                                     height_m=self._cam_height,
                                     lateral_sign=CAM_LATERAL_SIGN)
        else:
            self.calib = Calibration(image_w=CAM_W, image_h=CAM_H,
                                     fov_h_deg=CAM_FOV_H_DEG,
                                     height_m=self._cam_height,
                                     lateral_sign=CAM_LATERAL_SIGN)
            # World FIRST, GPU session second (RDNA4 level-load rule).
            self.world = BeamNGOnnxWorld(
                map_name=getattr(args, "map", None) or "west_coast_usa",
                vehicle_model=getattr(args, "vehicle", None) or "bastion",
                attach=bool(getattr(args, "attach", False)))
            n_traffic = int(getattr(args, "traffic", 0) or 0)
            if n_traffic > 0:
                self.world.spawn_traffic(n_traffic)

        # per-game config: CAL persists the measured lookahead here.
        # Loaded BEFORE the model so the action_t horizons fed to the
        # network match our real lat/long action timing.
        cfg = ControllerConfig.load(game=self._game_key)
        cfg.wheelbase_m = self.wheelbase
        cfg.max_speed_mps = 55.0 / 2.237   # start on the 5-mph grid
        self.cfg = cfg

        if args.split:
            self.model = DrivingModel(
                providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
                policy_providers=["CPUExecutionProvider"], intra_op_threads=3,
                vision_path=getattr(args, "vision", None) or None,
                policy_path=getattr(args, "policy", None) or None)
            self._decode = lambda i, b, d: decode(*self.model.step(i, b, desire=d))
            vp = getattr(args, "vision", None)
            name = ("split ({}){}".format(
                self.model.active_provider,
                f" [{os.path.basename(vp)}]" if vp else ""))
        else:
            model_path = getattr(args, "model", None) or None
            self.model = SupercomboModel(
                providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
                intra_op_threads=3, model_path=model_path,
                lat_action_t=cfg.lookahead_s + cfg.curvature_anticipation_s,
                long_action_t=cfg.lookahead_s + cfg.long_anticipation_s)
            self._decode = lambda i, b, d: self.model.decode(
                self.model.step(i, b, desire=d))
            has_action = isinstance(
                self.model.output_slices.get("action"), slice)
            name = (f"supercombo ({self.model.active_provider})"
                    + (f" [{os.path.basename(model_path)}]"
                       if model_path else "")
                    + (" +action-head" if has_action else ""))
        print(f"[panel] model: {name}", flush=True)

        self.queue = FrameQueue()
        self.tel = Telemetry(self.world)
        self.tel.start()

        # In-game imgui panel (beamng_mod/): load the extension if the
        # mod is mounted; harmless no-op inside pcall when it isn't.
        try:
            self.world.bng.control.queue_lua_command(
                "pcall(function() extensions.load('onnxPanel') end)")
        except Exception as exc:
            print(f"[panel] in-game panel load skipped: {exc}", flush=True)

        # after cfg: the 100 Hz steering executor takes its EPS time
        # constant from it (constructing this earlier crashed on boot)
        self.sender = ControlSender(self.world,
                                    steer_tau=cfg.steer_smooth_s)
        self.sender.start()
        self.lp = LiveParams(game=self._game_key)
        # Online camera-pose calibration (openpilot's calibrationd):
        # learns effective pitch/yaw/height while driving, commits into
        # calib (warp + overlay) once CALIBRATED. Persists per-game, so
        # like openpilot it fully calibrates once and refines forever.
        self.lc = LiveCalib(game=self._game_key)
        # SHADOW by default (field report 2026-07-01: stops-short, lane
        # drift, and long hunting all appeared together after livecalib
        # started applying — and it was rejecting 73% of its samples).
        # The learner keeps estimating and the HUD shows what it WOULD
        # apply; the CAM button ('v') A/Bs application live.
        self.calib_live = False
        self.lc.apply = False
        self.lat = LateralController(cfg, live_params=self.lp)
        self.long = LongitudinalController(cfg)

        self.engaged = False
        # exp: model's e2e accel; chill: cruise at cap + leads/corners;
        # off: user owns the pedals, model steers only.
        self.long_mode = "exp"
        self.ai_on = False
        self.frame_idx = 0
        self.desire_idx: int | None = None
        self.desire_until = 0.0
        self.cal_active = False
        self.cal_until = 0.0
        self._cal_speed = 0.0
        self.banner = ""
        self.banner_until = 0.0
        self.last_steer = 0.0
        self.last_thr = 0.0
        self.last_brk = 0.0
        self.hz = 0.0
        self.hz_ema = 20.0
        self._rate_warned_until = 0.0
        self._signal_state: str | None = None
        self._off_ema = 0.0
        # flight recorder: capture what the model saw when its plan
        # velocity collapses mid-drive (the "stops on open highway"
        # mystery) — main loop writes the frame+state on request
        self._incident_t = 0.0
        self.incident_note: str | None = None
        self._snap_req = False
        self._conf_low_t = 0.0
        self._conf_warned = False
        self._lc_committed = False
        # Model-wants vs car-does instrumentation: ~15 s sparkline ring
        # buffers drawn on the viewer, plus a per-session CSV run log
        # (every control tick) for offline plots — tools/plot_run.py.
        self.charts_on = True
        self._last_spd_t = 0.0
        self.chart = {k: deque(maxlen=300)
                      for k in ("v", "vT", "a", "aM", "k", "kM")}
        self._a_meas_app = 0.0
        self._chart_v_prev: float | None = None
        self.lane_off_now = 0.0
        # live values for the telemetry panel (op-replay-clipper style)
        self.k_meas_now = 0.0
        self.conf_now = 0.0
        self._log_f = None
        self.log_path = os.path.join(
            ROOT, "debug_out",
            f"run_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    # ---- UI actions ----

    @property
    def cal_left_s(self) -> float:
        return max(0.0, self.cal_until - time.monotonic())

    def set_banner(self, text: str, secs: float = 2.5) -> None:
        self.banner = text
        self.banner_until = time.monotonic() + secs
        print(f"[panel] {text}", flush=True)

    def engage_allowed(self) -> tuple[bool, str]:
        if self.frame_idx < WARMUP_FRAMES:
            return False, f"model warming up ({self.frame_idx}/{WARMUP_FRAMES})"
        if self.screen_mode:
            # No telemetry and no CAL here by design: the rack starts
            # from a seed and learns online WHILE engaged (the only
            # time we command the wheel and can observe the response).
            # Gating engage on a trusted rack would deadlock — you can
            # never earn the samples. Best-effort start; the wheel-trim
            # integrator bounds the seed error and the driver is the
            # fallback, exactly like a fresh comma calibration drive.
            return True, ""
        if not self.tel.ok:
            return False, "no telemetry"
        if not self.lp.trusted():
            return False, (f"steering not calibrated "
                           f"({max(self.lp.samples, self.lp.session_samples)}"
                           f"/100 samples — run CAL or drive manually)")
        return True, ""

    def action(self, key: str) -> None:
        now = time.monotonic()
        if key == "engage":
            if self.engaged:
                self.disengage("button")
            else:
                ok, why = self.engage_allowed()
                if ok:
                    self.engage()
                else:
                    self.set_banner(f"engage blocked: {why}")
        elif key == "force":
            self.engage(forced=True)
        elif key in ("lane_l", "lane_r", "turn_l", "turn_r"):
            idx = {"turn_l": 1, "turn_r": 2, "lane_l": 3, "lane_r": 4}[key]
            if self.desire_idx == idx and now < self.desire_until:
                # repeat while held (widget resends every 300 ms, cv2
                # key autorepeat, desktop mouse-hold): keep the desire
                # alive ~0.8 s past the last repeat — openpilot's
                # blinker-held-desire-active behavior. A single tap
                # still gets the full DESIRE_HOLD_S pulse via max().
                self.desire_until = max(self.desire_until, now + 0.8)
            else:
                self.desire_idx = idx
                self.desire_until = now + DESIRE_HOLD_S[idx]
                self.set_banner(f"desire: {DESIRE_NAME[idx]}")
            self._update_signal()
        elif key == "long":
            order = ["exp", "chill", "off"]
            self.long_mode = order[(order.index(self.long_mode) + 1) % 3]
            desc = {"exp": "EXPERIMENTAL (model drives the pedals)",
                    "chill": "CHILL (cruise at cap, brake for leads)",
                    "off": "OFF (your pedals, model steers)"}
            self.set_banner(f"long: {desc[self.long_mode]}", 3.0)
        elif key == "ai":
            self.ai_on = not self.ai_on
            if self.ai_on:
                if self.engaged:
                    self.disengage("AI takeover")
                self.world.ai_drive(20.0)
            else:
                self.world.ai_disable()
            self.set_banner(f"BeamNG AI {'ON' if self.ai_on else 'OFF'}")
        elif key == "cal":
            self.start_cal()
        elif key == "cam":
            self.set_calib_live(not self.calib_live)
        elif key == "snap":
            # remote flight-recorder shutter: capture the full canvas
            # + model-belief note on the next tick, no trigger gates
            self._snap_req = True
        elif key in ("spd_dn", "spd_up"):
            # step in whole-5-mph notches (55, 60, 65 ...) like a real
            # cruise stalk; snap first in case the cap started off-grid.
            # Debounced: dpg key/button handlers repeat at the OS
            # key-repeat rate, which otherwise ran the cap to its
            # ceiling in one press.
            if now - self._last_spd_t < 0.25:
                return
            self._last_spd_t = now
            step = -5.0 if key == "spd_dn" else 5.0
            mph = round(self.cfg.max_speed_mps * 2.237 / 5.0) * 5.0 + step
            mph = min(100.0, max(10.0, mph))
            self.cfg.max_speed_mps = mph / 2.237
            self.set_banner(f"max speed {mph:.0f} mph")

    def set_calib_live(self, on: bool) -> None:
        """A/B the camera-pose learner's APPLICATION (not its learning).
        ON: commit the learned pitch/yaw/height into the warp now and
        let future blocks keep refining it. OFF (shadow): the learner
        keeps estimating but the warp returns to the stock mount —
        the model sees exactly what it saw before livecalib existed."""
        self.calib_live = on
        self.lc.apply = on
        if on:
            if self.lc.pitch_estimate is not None:
                self.calib.pitch_deg = float(self.lc.pitch_estimate)
                self.calib.yaw_deg = float(self.lc.yaw_estimate or 0.0)
            if self.lc.height_estimate is not None:
                self.calib.height_m = float(self.lc.height_estimate)
            self.set_banner(
                f"camera calib LIVE: pitch {self.calib.pitch_deg:+.2f} "
                f"yaw {self.calib.yaw_deg:+.2f} h {self.calib.height_m:.2f}",
                3.5)
        else:
            self.calib.pitch_deg = 0.0
            self.calib.yaw_deg = 0.0
            self.calib.height_m = self._cam_height
            self.set_banner("camera calib SHADOW: stock mount pose "
                            "(learner keeps estimating)", 3.5)

    def _update_signal(self) -> None:
        """Blinker follows the active desire (left for turn/lane L,
        right for R, off when it expires). Sends only on transitions."""
        want = None
        if self.desire_idx in (1, 3):
            want = "left"
        elif self.desire_idx in (2, 4):
            want = "right"
        if want != self._signal_state:
            self._signal_state = want
            try:
                self.world.set_signal(want)
            except Exception:
                pass

    def engage(self, forced: bool = False) -> None:
        if self.ai_on:
            self.world.ai_disable()
            self.ai_on = False
        self.world.realistic_gearbox()   # arcade + held brake = REVERSE
        self.world.ensure_drive()        # post-reset the box sits in N/P
        self.cal_active = False
        self.lat.reset()
        self.long.reset()
        self.engaged = True
        if self.screen_mode and not self.lp.trusted():
            self.set_banner("ENGAGED - steering rack is learning; expect "
                            "a wobble for the first ~1-2 min", 5.0)
        else:
            self.set_banner("ENGAGED" + (" (forced)" if forced else ""))

    def disengage(self, reason: str) -> None:
        self.engaged = False
        self.lat.reset()
        self.long.reset()
        self.world.apply(0.0, 0.0, 0.0)
        self.set_banner(f"DISENGAGED ({reason})")

    def start_cal(self) -> None:
        if getattr(self.world, "no_telemetry", False):
            self.set_banner("CAL needs tech mode (telemetry + teleport); "
                            "screen mode learns the rack online instead",
                            5.0)
            return
        if getattr(self.world, "attached", False):
            self.set_banner("CAL teleports to the west_coast_usa spawn — "
                            "unavailable in attach/freeroam mode", 5.0)
            return
        if self.engaged:
            self.disengage("calibration")
        if self.ai_on:
            self.world.ai_disable()
            self.ai_on = False
        self.world.realistic_gearbox()
        self.world.ensure_drive()
        self.world.vehicle.teleport(SPAWN_POS, rot_quat=SPAWN_ROT_QUAT)
        self.cal_active = True
        self.cal_until = time.monotonic() + CAL_DURATION_S
        self._cal_phase = "settle"
        self._cal_t0 = time.monotonic()
        self._cal_samples: list[tuple[float, float]] = []
        self._cal_series: list[tuple[float, float]] = []   # (axis, wheel)/frame
        self._cal_long: list[tuple[str, float, float]] = []  # (kind, pedal, accel)
        self._cal_long_i = 0
        self._cal_vt: list[tuple[float, float]] = []       # (t, v) per step
        self.set_banner("CAL: scripted system-ID at spawn — hands off", 4.0)

    def _cal_finish(self) -> None:
        self.cal_active = False
        self.world.apply(0.0, 0.0, 0.6)
        n = len(self._cal_samples)
        sw2 = sum(w * w for _, w in self._cal_samples)
        if n < 40 or sw2 < 1e-5:
            self.set_banner(f"CAL FAILED (n={n}) — fit unchanged, retry", 5.0)
            return
        # least squares through the origin: axis = a * wheel
        a = sum(ax * w for ax, w in self._cal_samples) / sw2
        if not (0.5 <= abs(a) <= 50.0):
            self.set_banner(f"CAL FAILED (a={a:.2f} implausible) — retry", 5.0)
            return
        # Direct measurement outranks RLS: install the fit and mark it
        # mature (freeze-level sample count) so online updates can't
        # wander it the way the old AI-watching calibration could.
        # Install as a strong PRIOR, not a frozen constant: RLS keeps
        # learning while driving (openpilot's torqued behavior — auto-
        # freeze is disabled in this fork of LiveParams by design).
        # Tight covariance so a session's first noisy samples can't
        # yank the measured seed; the fit stays adaptive to damage,
        # vehicle changes, etc.
        self.lp.x[:] = [a, 0.0, 0.0]
        self.lp.P = np.diag((0.25, 1e-6, 1e-4))
        self.lp.samples = 1000
        self.lp.session_samples = 1000
        self.lp.save()

        # Loop lag by cross-correlation over the pulse train (command
        # vs model-pose wheel response) — measured ~300 ms, but that
        # CONFLATES perception (render + camera + inference, ~200 ms)
        # with actuation. The model's plan is already expressed in
        # camera-time, so only the ACTUATION side belongs in the
        # steerActuatorDelay analog: direct-input t90 (~0.08 s) + our
        # EPS-emulation filter. Using the full 300 ms made the car turn
        # in early AND hard ("takes curves sharper than it needs to").
        lag_s = 0.0
        if len(self._cal_series) > 40:
            ax = np.array([p[0] for p in self._cal_series])
            wh = np.array([p[1] for p in self._cal_series])
            best_c = -1e9
            for k in range(0, 13):                      # 0..0.6 s @ 20 Hz
                a_s = ax[:len(ax) - k] if k else ax
                w_s = wh[k:]
                sd = float(np.std(a_s) * np.std(w_s))
                if sd < 1e-9:
                    continue
                c = float(np.mean((a_s - a_s.mean()) * (w_s - w_s.mean())) / sd)
                if c > best_c:
                    best_c, lag_s = c, k * 0.05
        # Lead the plan by the measured command->response delay:
        # k_des->k_meas cross-correlation reads 0.30-0.35 s on canyon
        # sweepers at 25-28 m/s (rate limiter + EPS tau + vehicle yaw
        # response). Under-leading self-inflates: the late response
        # leaves the car wide mid-curve, the model asks for extra
        # curvature, and the want/act chart splits ~25% on every ramp.
        # 0.30 covers the vehicle side (upper end of the measured
        # 0.30-0.35); the EPS constant rides on top. The deliberate
        # early-turn-in preference lives in curvature_anticipation_s,
        # NOT here — this number is the physics.
        self.cfg.lookahead_s = float(np.clip(
            0.30 + self.cfg.steer_smooth_s, 0.10, 0.40))

        # Pedal-map fit: achieved accel vs pedal position, slope with
        # intercept (the intercept absorbs drag/rolling resistance —
        # the speed-I trim covers that online). Sets the scales so a
        # commanded m/s^2 produces that m/s^2 in the game.
        long_msg = ""
        print(f"[cal] long steps (kind, pedal, dv/dt): "
              f"{[(k, p, round(s, 2)) for k, p, s in self._cal_long]}",
              flush=True)
        thr = [(p, s) for k, p, s in self._cal_long if k == "thr"]
        brk = [(p, -s) for k, p, s in self._cal_long if k == "brk"]
        coast = [-s for k, _, s in self._cal_long if k == "coast"]
        # pedals-free deceleration (drag + engine braking) — the brake
        # fit must be net of this or its intercept swallows the whole
        # ISO command range as "coast gap"
        d_coast = float(np.clip(np.mean(coast), 0.3, 3.0)) if coast else 1.5
        # Interpolation tables (measured points, monotonic-filtered):
        # the brake response saturates, so no parametric fit — the
        # curve IS the calibration.
        if len(thr) >= 2:
            pts = [(-d_coast, 0.0)] + sorted(
                (float(s), float(p)) for p, s in thr)
            table = [pts[0]]
            for a_m, p_m in pts[1:]:
                if a_m > table[-1][0] + 0.05:
                    table.append((a_m, p_m))
            if len(table) >= 3:
                self.cfg.pedal_thr_map = [list(x) for x in table]
                self.cfg.max_accel_mps2 = table[-1][0]   # HUD/legacy
                long_msg += f"  thr[{table[-1][0]:.1f}max]"
        if len(brk) >= 2:
            pts = [(d_coast, 0.0)] + sorted(
                (float(d), float(p)) for p, d in brk)
            table = [pts[0]]
            for d_m, p_m in pts[1:]:
                if d_m > table[-1][0] + 0.05:
                    table.append((d_m, p_m))
            if len(table) >= 3:
                # Wheel lockup can make FULL pedal measure LESS decel
                # than a partial press — the inverse map must still be
                # monotonic in pedal or interp commands LESS brake for
                # MORE demand (shipped once: [.., 7.2->1.0, 7.9->0.6]).
                for i in range(1, len(table)):
                    table[i] = (table[i][0],
                                max(table[i][1], table[i - 1][1]))
                # terminal point: anything beyond the strongest
                # measured decel gets full pedal (AEB lands here)
                table.append((table[-1][0] + 3.0, 1.0))
                self.cfg.pedal_brk_map = [list(x) for x in table]
                self.cfg.max_decel_mps2 = table[-2][0]
                long_msg += (f"  brk[{table[-2][0]:.1f}max"
                             f" coast {d_coast:.1f}]")
        # Execute the plan at an openpilot-like horizon. The old 1.0 s
        # anticipation applied the deceleration planned for 1.3 s in
        # the future NOW — the whole stop profile ran early and the car
        # halted metres before the line; 0.5 still landed red-light
        # stops early (lane confidence fades at crawl speed, so the
        # model's re-planning can't claw the shift back). Corner
        # braking keeps its own 6 s scan horizon; this only times
        # plan-following.
        self.cfg.long_anticipation_s = 0.3
        self.cfg.save(game=self._game_key)

        self.set_banner(f"CAL done: a={a:+.2f} ({n} samples, frozen)  "
                        f"lag={lag_s*1000:.0f}ms lookahead "
                        f"{self.cfg.lookahead_s:.2f}s{long_msg}", 6.0)

    # ---- per-frame control ----

    def control_tick(self, decoded, v_ego: float, dt: float) -> None:
        now = time.monotonic()

        # Synthesized wheel angle from the model's ego yaw rate.
        yaw_rate = float(decoded.pose[5])
        k_meas = yaw_rate / max(v_ego, 1.0)
        wheel = math.atan(k_meas * self.wheelbase) if v_ego > 1.0 else None

        desire_active = (self.desire_idx is not None
                         and now < self.desire_until)
        # DesireHelper-style completion cut: one lane change per
        # command. The model's desire pulse lives ~5 s in its input
        # buffer while the maneuver takes ~3.5 s; holding the desire
        # past completion let the leftover tail wind up a SECOND
        # change (logged: clean 1-lane change at t+3.7 s, second
        # right-commit at t+5.3 s -> "cuts across several lanes").
        # openpilot's DesireHelper drops the desire the moment the
        # model's own lane-change belief collapses after being high;
        # mirror that here instead of running the timer out.
        if desire_active and self.desire_idx in (3, 4):
            lc_p = (float(decoded.desire_state[3])
                    + float(decoded.desire_state[4]))
            if lc_p > 0.3:
                self._lc_committed = True
            elif self._lc_committed and lc_p < 0.1:
                self.desire_until = 0.0
                desire_active = False
                self._lc_committed = False
                self.set_banner("lane change complete")
        else:
            self._lc_committed = False
        if not desire_active:
            self.desire_idx = None
        self._update_signal()

        steer, thr, brk = 0.0, 0.0, 0.0
        commanded = None
        if self.cal_active:
            phase_t = now - self._cal_t0
            if self._cal_phase == "settle":
                self.world.apply(0.0, 0.0, 0.2)   # post-teleport physics
                if phase_t > 2.0:
                    # AFTER the teleport: teleporting resets the vehicle
                    # and drops the box back to neutral — shifting in
                    # start_cal() was undone and CAL revved in N
                    self.world.realistic_gearbox()
                    self.world.ensure_drive()
                    self._cal_phase, self._cal_t0 = "accel", now
            elif self._cal_phase == "accel":
                self.world.apply(0.0, 0.45, 0.0)
                if v_ego >= CAL_V or phase_t > 15.0:
                    self._cal_phase, self._cal_t0 = "pulse", now
            elif self._cal_phase == "pulse":
                i = int(phase_t // CAL_PULSE_S)
                if i >= len(CAL_PULSES):
                    self._cal_phase, self._cal_t0 = "long", now
                    self._cal_long_i = 0
                    self._cal_vt = []
                else:
                    axis = CAL_PULSES[i]
                    thr_c = 0.25 if v_ego < CAL_V else 0.05
                    self.world.apply(axis, thr_c, 0.0)
                    in_pulse = phase_t - i * CAL_PULSE_S
                    if wheel is not None and v_ego > 4.0:
                        # full series (transients included) for the lag
                        # cross-correlation
                        self._cal_series.append((axis, wheel))
                        if in_pulse > CAL_PULSE_SETTLE_S:
                            # settled samples only for the gain fit
                            self._cal_samples.append((axis, wheel))
            elif self._cal_phase == "long":
                if self._cal_long_i >= len(CAL_LONG_SCRIPT):
                    self._cal_finish()
                else:
                    kind, pedal, dur = CAL_LONG_SCRIPT[self._cal_long_i]
                    if kind == "thr":
                        self.world.apply(0.0, pedal, 0.0)
                    elif kind == "brk":
                        self.world.apply(0.0, 0.0, pedal)
                    elif kind == "coast":
                        self.world.apply(0.0, 0.0, 0.0)
                    elif kind == "slow":   # back to thr-step start speed
                        self.world.apply(0.0, 0.0, 0.4)
                    else:   # rec: recover speed between brake steps
                        self.world.apply(0.0, 0.7, 0.0)
                    measured = kind in ("thr", "brk", "coast")
                    if measured and phase_t > CAL_LONG_SETTLE_S:
                        self._cal_vt.append((now, v_ego))
                    done = (phase_t > dur
                            or (kind == "rec" and v_ego >= 18.0)
                            or (kind == "slow" and v_ego <= 9.0))
                    if done:
                        if measured and len(self._cal_vt) >= 6:
                            ts = np.array([p[0] for p in self._cal_vt])
                            vs = np.array([p[1] for p in self._cal_vt])
                            slope = float(np.polyfit(ts - ts[0], vs, 1)[0])
                            self._cal_long.append((kind, pedal, slope))
                        self._cal_long_i += 1
                        self._cal_t0 = now
                        self._cal_vt = []
        elif self.engaged:
            lane_change_cmd = desire_active and self.desire_idx in (3, 4)
            steer = self.lat.compute(decoded, v_ego,
                                     actual_wheel_angle=wheel,
                                     lane_change_command_active=lane_change_cmd,
                                     dt=dt,
                                     roll_glat=self.tel.snapshot().get(
                                         "roll_glat", 0.0))
            if self.screen_mode and wheel is not None:
                # screen mode has no CAL and no telemetry: learn the
                # axis->wheel rack online from our own command vs the
                # model-observed yaw — upstream simsteer's gamepad-mode
                # design. (The tech-mode ban on online rack learning
                # is about CAL being the sole writer; here CAL cannot
                # exist.)
                self.lp.update(steer, v_ego, wheel,
                               commanded_axis=steer)
            if self.long_mode == "off":
                # steering only — no throttle/brake API calls, so the
                # player's own pedal inputs pass through untouched
                self.sender.submit(steer, None, None)
            else:
                thr, brk = self.long.compute(decoded, v_ego,
                                             mode=self.long_mode)
                self.sender.submit(steer, thr, brk)
            commanded = steer

        # The rack fit is a MEASURED CONSTANT, not a live learner
        # (user decision 2026-07-02, after the online RLS dragged the
        # gain 2.0→0.71/0.84/1.27 across three sessions): the CAL
        # system-ID is the only writer — like openpilot's per-car
        # steering constants. No engaged-time learning (circular:
        # our command × model-derived wheel through variable lag) and
        # no manual-drive learning either. The trim integrator still
        # nudges residuals while engaged, bounded and leaky.
        tel = self.tel.snapshot()

        # Camera-pose learner runs on every frame (its own gates handle
        # speed/turning/uncertainty); commits into self.calib — which
        # the warp and overlay read — once CALIBRATED.
        if not self.cal_active:
            self.lc.update(self.calib, decoded.pose, decoded.road_transform,
                           yaw_rate, v_ego,
                           pose_std=decoded.pose_std,
                           road_transform_std=decoded.road_transform_std,
                           wide_from_device_euler=decoded.wide_from_device_euler,
                           game_steer=tel["steering_input"])

        self.last_steer, self.last_thr, self.last_brk = steer, thr, brk

        # ---- model-wants vs car-does instrumentation ----
        # Own measured-accel LPF (the long controller's only updates
        # while it runs; we want the trace during manual driving too).
        if self._chart_v_prev is not None and dt > 1e-3:
            a_raw = (v_ego - self._chart_v_prev) / dt
            self._a_meas_app += 0.15 * (
                float(np.clip(a_raw, -12.0, 12.0)) - self._a_meas_app)
        self._chart_v_prev = v_ego
        self.lane_off_now = float(np.mean(decoded.lane_lines[1:3, 0, 0]))
        self.k_meas_now = k_meas if v_ego > 1.0 else 0.0
        self.conf_now = float(np.mean(decoded.lane_lines_prob[1:3]))

        # Low-lane-vision ALERT — advisory only, like openpilot's
        # low-confidence driver alerts. The old "road lost" guard that
        # stopped the car here is gone by design (2026-07-02): real
        # openpilot never intervenes on vision confidence, it drives
        # best-effort on whatever the model outputs and the driver is
        # the fallback. Same deal here — the banner tells the driver,
        # the pedals stay with the model / lead / AEB logic.
        if self.engaged and self.conf_now < 0.15 and v_ego > 3.0:
            self._conf_low_t += dt
        elif self.conf_now > 0.40 or not self.engaged:
            self._conf_low_t = 0.0
            self._conf_warned = False
        if self._conf_low_t > 2.5 and not self._conf_warned:
            self._conf_warned = True
            self.set_banner("lane vision low - best effort", 4.0)

        long_live = self.engaged and self.long_mode != "off" \
            and not self.cal_active
        lat_live = self.engaged and not self.cal_active
        nan = float("nan")
        self.chart["v"].append(v_ego)
        self.chart["vT"].append(
            min(self.long.last_v_target, 99.0) if long_live else nan)
        self.chart["a"].append(self.long.last_a_cmd if long_live else nan)
        self.chart["aM"].append(self._a_meas_app)
        self.chart["k"].append(
            self.lat.last_curvature if lat_live else nan)
        yaw_k = yaw_rate / max(v_ego, 1.0) if v_ego > 1.0 else 0.0
        self.chart["kM"].append(yaw_k)
        if self._log_f is None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._log_f = open(self.log_path, "w", buffering=1)
            self._log_f.write(
                "t,eng,mode,cal,v_ego,v_target,a_target,a_cmd,a_fb,"
                "a_meas,thr,brk,steer,k_des,k_meas,lane_off,lead_x,"
                "lead_p,pitch_applied,pitch_learned,"
                "lp0,lp1,lp2,lp3,des_i,des_p,v_end,trim,roll\n")
            print(f"[panel] run log: {self.log_path}", flush=True)
        lead_p = (float(decoded.lead_prob[0])
                  if decoded.lead_prob.size else 0.0)
        self._log_f.write(
            f"{now:.3f},{int(self.engaged)},{self.long_mode},"
            f"{int(self.cal_active)},{v_ego:.3f},"
            f"{min(self.long.last_v_target, 99.0):.3f},"
            f"{self.long.last_a_target:.3f},{self.long.last_a_cmd:.3f},"
            f"{self.long.last_a_fb:.3f},{self._a_meas_app:.3f},"
            f"{thr:.3f},{brk:.3f},{steer:.4f},"
            f"{self.lat.last_curvature:.5f},{yaw_k:.5f},"
            f"{self.lane_off_now:.3f},"
            f"{min(self.long.last_lead_x, 999.0):.1f},{lead_p:.2f},"
            f"{self.calib.pitch_deg:.3f},"
            f"{self.lc.pitch_estimate if self.lc.pitch_estimate is not None else 0.0:.3f},"
            f"{decoded.lane_lines_prob[0]:.2f},{decoded.lane_lines_prob[1]:.2f},"
            f"{decoded.lane_lines_prob[2]:.2f},{decoded.lane_lines_prob[3]:.2f},"
            f"{int(np.argmax(decoded.desire_state))},"
            f"{float(np.max(decoded.desire_state)):.2f},"
            f"{float(decoded.plan[-1, 3]):.2f},"
            f"{self.lat.axis_trim_state:+.4f},"
            f"{tel.get('roll_glat', 0.0):+.2f}\n")

        # flight recorder trigger: engaged, moving, and the plan's
        # velocity target collapsed well below current speed with no
        # lead engaged — capture the scene (throttled to 1 per 4 s)
        if (self.engaged and v_ego > 8.0
                and self.long.last_v_target < 0.5 * v_ego
                and self.long.last_lead_x > 400
                and now - self._incident_t > 4.0):
            self._incident_t = now
            self.incident_note = (
                f"v={v_ego:.1f} vT={self.long.last_v_target:.1f} "
                f"aT={self.long.last_a_target:+.2f} "
                f"lanes={np.round(decoded.lane_lines_prob, 2).tolist()} "
                f"leadP={float(decoded.lead_prob[0]):.2f} "
                f"desire_state={np.round(decoded.desire_state, 2).tolist()}")
        if self._snap_req:
            self._snap_req = False
            self.incident_note = (
                f"SNAP v={v_ego:.1f} vT={self.long.last_v_target:.1f} "
                f"aT={self.long.last_a_target:+.2f} "
                f"lanes={np.round(decoded.lane_lines_prob, 2).tolist()} "
                f"leadP={float(decoded.lead_prob[0]):.2f} "
                f"desire_state={np.round(decoded.desire_state, 2).tolist()}")

    # ---- HUD ----

    def hud_lines(self, decoded, v_ego: float) -> list[str]:
        lp = self.lp
        probs = np.round(decoded.lane_lines_prob, 2)
        lead_p = float(decoded.lead_prob[0]) if decoded.lead_prob.size else 0.0
        l1 = (f"{'ENGAGED' if self.engaged else 'manual '}  "
              f"{v_ego*2.237:4.0f} mph (cap {self.cfg.max_speed_mps*2.237:.0f})  "
              f"steer {self.last_steer:+.2f}  thr {self.last_thr:.2f}  "
              f"brk {self.last_brk:.2f}  work {self.hz:3.0f}Hz")
        l2 = (f"lanes {probs}  lead {lead_p:.2f}"
              + (f" @{self.long.last_lead_x:.0f}m" if self.long.last_lead_x < 500 else "")
              + f"  vT {min(self.long.last_v_target, 99)*2.237:3.0f}"
              + f"  aT {self.long.last_a_target:+.1f}"
              + (f"  corner@{self.long.last_corner_t:.0f}s"
                 if self.long.last_v_safe_corner < self.long.last_v_target + 1
                 else "")
              + ("  AEB!" if self.long.last_aeb else ""))
        # slow lane-center offset EMA (~10 s): steady nonzero sign at
        # highway = controls/mount bias (watch trim fight it); sign
        # flipping with scene/curves = model behavior
        off_now = float(np.mean(decoded.lane_lines[1:3, 0, 0]))
        if abs(off_now) < 3.0:
            self._off_ema += 0.005 * (off_now - self._off_ema)
        trust = "OK" if lp.trusted() else "COLD"
        lc_p = self.lc.pitch_estimate
        l3 = (f"rack a={lp.a_linear:+.2f} b={lp.b_quad:+.4f} "
              f"n={max(lp.samples, lp.session_samples)} [{trust}]"
              f"  off~{self._off_ema:+.2f}m"
              f"  trim {self.lat.axis_trim_state:+.3f}"
              f"({self.lat.last_trim_frozen_reason or 'run'})")
        l4 = (f"cam[{self.lc.blocks}b "
              f"{'LIVE' if self.calib_live else 'shdw'}"
              + (f" p{lc_p:+.2f}" if lc_p is not None else "")
              + f"]  aFB {self.long.last_a_fb:+.2f}")
        if self.desire_idx is not None:
            l4 += f"   >>> {DESIRE_NAME[self.desire_idx]} " \
                  f"({self.desire_until - time.monotonic():.1f}s)"
        return [l1, l2, l3, l4]


def draw_charts(canvas: np.ndarray, app: App) -> None:
    """Model-wants (orange) vs car-does (green) sparklines, last ~15 s,
    stacked top-right over the camera view. The 'is it the model or is
    it the controls' question in one glance: traces apart = execution
    error (controls); traces together but misbehaving = the model
    asked for it."""
    panels = [
        ("mph", "vT", "v", 2.237, 4.0),
        ("m/s^2", "a", "aM", 1.0, 1.0),
        ("curv x1000", "k", "kM", 1000.0, 2.0),
    ]
    W, H, M = 300, 80, 8
    x1 = canvas.shape[1] - W - M
    for i, (label, want_key, does_key, scale, min_span) in enumerate(panels):
        y0 = M + i * (H + M)
        want = np.array(app.chart[want_key], dtype=np.float64) * scale
        does = np.array(app.chart[does_key], dtype=np.float64) * scale
        roi = canvas[y0:y0 + H, x1:x1 + W]
        roi[:] = (roi * 0.25).astype(np.uint8)
        cv2.rectangle(canvas, (x1, y0), (x1 + W - 1, y0 + H - 1),
                      (90, 90, 90), 1)
        both = np.concatenate([want[~np.isnan(want)],
                               does[~np.isnan(does)]])
        if both.size < 2:
            continue
        lo, hi = float(both.min()), float(both.max())
        mid, span = (lo + hi) / 2.0, max(hi - lo, min_span)
        lo, hi = mid - span / 2.0, mid + span / 2.0

        def ypix(val: float) -> int:
            return y0 + 4 + int((H - 22) * (1.0 - (val - lo) / (hi - lo)))

        if lo < 0.0 < hi:
            yz = ypix(0.0)
            cv2.line(canvas, (x1, yz), (x1 + W - 1, yz), (70, 70, 70), 1)
        n = max(len(want), len(does))
        for series, col in ((does, (90, 220, 120)), (want, (0, 170, 255))):
            prev = None
            for j, val in enumerate(series):
                if np.isnan(val):
                    prev = None
                    continue
                pt = (x1 + int(W * j / max(n - 1, 1)), ypix(float(val)))
                if prev is not None:
                    cv2.line(canvas, prev, pt, col, 1, cv2.LINE_AA)
                prev = pt
        wv = want[~np.isnan(want)]
        dv = does[~np.isnan(does)]
        # label in the series colors: want = orange, act = green
        fnt, sc = cv2.FONT_HERSHEY_SIMPLEX, 0.42
        parts = ((f"{label}  ", (200, 200, 200)),
                 ("want " + (f"{wv[-1]:+.1f}" if wv.size else "--"),
                  (0, 170, 255)),
                 ("  act " + (f"{dv[-1]:+.1f}" if dv.size else "--"),
                  (90, 220, 120)))
        tx = x1 + 6
        for s, col in parts:
            cv2.putText(canvas, s, (tx, y0 + H - 6), fnt, sc, col, 1,
                        cv2.LINE_AA)
            tx += cv2.getTextSize(s, fnt, sc, 1)[0][0]


def _build_wheel_icon(size: int = 186) -> np.ndarray:
    """Grayscale steering-wheel mask: rim ring + three spokes + hub.
    Rotated per frame by the ACTUAL steering angle, so the on-screen
    wheel turns with the car's."""
    m = np.zeros((size, size), np.uint8)
    c = size // 2
    r = c - 4
    cv2.circle(m, (c, c), r, 255, -1)
    cv2.circle(m, (c, c), r - 16, 0, -1)          # rim ring
    for dx, dy in ((-1.0, 0.22), (1.0, 0.22), (0.0, 1.0)):   # 3 spokes
        n = math.hypot(dx, dy)
        x2 = int(c + (r - 6) * dx / n)
        y2 = int(c + (r - 6) * dy / n)
        cv2.line(m, (c, c), (x2, y2), 255, 18)
    cv2.circle(m, (c, c), 26, 255, -1)            # hub
    return m


_WHEEL_ICON = _build_wheel_icon()


def draw_telemetry(canvas: np.ndarray, app: App, tel: dict,
                   v_ego: float) -> None:
    """op-replay-clipper-style live telemetry: what the model wants vs
    what the controller applied vs what the car is doing."""
    y0 = CAM_VIEW_H
    canvas[y0:y0 + TELEM_H] = C_BG

    cv2.putText(canvas, "TELEMETRY", (28, y0 + 40), FONT, 0.85,
                C_LABEL, 2, cv2.LINE_AA)
    mode = app.long_mode.upper()
    mcol = {"EXP": C_BLUE, "CHILL": C_GREEN, "OFF": C_LABEL}[mode]
    cv2.putText(canvas, mode, (UI_W - 150, y0 + 38), FONT, 0.6, mcol, 1,
                cv2.LINE_AA)

    # ---- steering wheel + arcs ----
    size = _WHEEL_ICON.shape[0]
    cx, cy = 250, y0 + 66 + size // 2
    for rr in (size // 2 + 12, size // 2 + 20, size // 2 + 28):
        cv2.ellipse(canvas, (cx, cy), (rr, rr), 0, -210, 30, C_DIM, 3,
                    cv2.LINE_AA)
    # BeamNG electrics 'steering' is wheel degrees, positive = left
    # (CCW): warpAffine positive angle is also CCW, so pass through.
    rot = cv2.warpAffine(
        _WHEEL_ICON,
        cv2.getRotationMatrix2D((size / 2, size / 2),
                                float(tel["steering_deg"]), 1.0),
        (size, size))
    x1, ytop = cx - size // 2, cy - size // 2
    roi = canvas[ytop:ytop + size, x1:x1 + size]
    roi[rot > 127] = C_WHITE

    # arc markers, axis units (-1..1) mapped to +/-120 deg from top:
    # white = the car's wheel, orange = applied cmd, blue = pre-EPS tgt
    def arc_pt(axis_val: float, radius: int) -> tuple[int, int]:
        th = math.radians(float(np.clip(axis_val, -1, 1)) * 120.0)
        return (int(cx + radius * math.sin(th)),
                int(cy - radius * math.cos(th)))

    if app.engaged:
        cv2.circle(canvas, arc_pt(app.lat.last_axis_target,
                                  size // 2 + 20), 5, C_BLUE, -1,
                   cv2.LINE_AA)
        cv2.circle(canvas, arc_pt(app.lat.last_axis, size // 2 + 20), 5,
                   C_ORANGE, -1, cv2.LINE_AA)
    cv2.circle(canvas, arc_pt(tel["steering_input"], size // 2 + 28), 7,
               C_WHITE, -1, cv2.LINE_AA)
    lch = app.desire_idx in (1, 3)
    rch = app.desire_idx in (2, 4)
    cv2.putText(canvas, "<", (cx - size // 2 - 74, cy - 20), FONT, 1.5,
                C_WHITE if lch else C_DIM, 3, cv2.LINE_AA)
    cv2.putText(canvas, ">", (cx + size // 2 + 44, cy - 20), FONT, 1.5,
                C_WHITE if rch else C_DIM, 3, cv2.LINE_AA)

    # ---- confidence pill (right edge) ----
    conf = float(np.clip(app.conf_now, 0.0, 1.0))
    tx = UI_W - 68
    t_top, t_bot = y0 + 96, y0 + TELEM_H - 130
    cv2.putText(canvas, "CONF", (tx - 24, y0 + 78), FONT, 0.5, C_LABEL,
                1, cv2.LINE_AA)
    cv2.line(canvas, (tx, t_top), (tx, t_bot), C_DIM, 3, cv2.LINE_AA)
    ccol = ((80, 220, 120) if conf > 0.5 else
            (74, 192, 232) if conf > 0.25 else (60, 60, 220))
    by = int(t_bot - conf * (t_bot - t_top))
    cv2.circle(canvas, (tx, by), 14, (10, 10, 10), -1, cv2.LINE_AA)
    cv2.circle(canvas, (tx, by), 11, ccol, -1, cv2.LINE_AA)

    # ---- value rows ----
    def row(y: int, label: str, value: str, vcol) -> None:
        cv2.putText(canvas, label, (36, y), FONT, 0.55, C_LABEL, 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, value, (200, y), FONT, 0.8, vcol, 2,
                    cv2.LINE_AA)

    ry = cy + size // 2 + 52
    v2 = max(v_ego, 1.0) ** 2
    row(ry, "DES LAT", f"{v2 * app.lat.last_curvature:+.2f} m/s2", C_BLUE)
    row(ry + 38, "ACT LAT", f"{v2 * app.k_meas_now:+.2f} m/s2", C_GREEN)
    row(ry + 76, "TGT %", f"{app.lat.last_axis_target * 100:+.0f}%",
        C_YELLOW)
    row(ry + 114, "APP %", f"{app.lat.last_axis * 100:+.0f}%", C_ORANGE)
    row(ry + 152, "ACTUAL", f"{tel['steering_deg']:+.1f} deg", C_WHITE)
    hands_on = (time.monotonic() - app.lp.last_intervened_ts) < 1.0
    row(ry + 190, "HANDS", "ON WHEEL" if hands_on else "OFF WHEEL",
        C_WHITE)

    # ---- pedals: DRIVER column | OPENPILOT column ----
    py = ry + 226
    cv2.putText(canvas, "DRIVER", (36, py), FONT, 0.5, C_LABEL, 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "ONNX", (300, py), FONT, 0.5, C_LABEL, 1,
                cv2.LINE_AA)

    def bar(x: int, y: int, frac: float, col) -> None:
        w, h = 240, 12
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (50, 42, 32), -1)
        if frac > 0.005:
            cv2.rectangle(canvas, (x, y),
                          (x + int(w * min(frac, 1.0)), y + h), col, -1)

    # driver pedals only mean anything when the user owns them
    user_pedals = app.long_mode == "off"
    thr_in, brk_in = tel["throttle_input"], tel["brake_input"]
    for i, (name, dval, oval, col) in enumerate((
            ("GAS", thr_in, app.last_thr, C_GREEN),
            ("BRAKE", brk_in, app.last_brk, (60, 60, 220)))):
        yy = py + 34 + i * 46
        cv2.putText(canvas, name, (36, yy), FONT, 0.55, C_LABEL, 1,
                    cv2.LINE_AA)
        dtxt = f"{dval * 100:.0f}%" if user_pedals and dval > 0.02 \
            else "OFF"
        cv2.putText(canvas, dtxt, (120, yy), FONT, 0.6, C_WHITE, 1,
                    cv2.LINE_AA)
        bar(300, yy - 11, oval, col)
        cv2.putText(canvas, f"{oval * 100:.0f}%", (552, yy), FONT, 0.55,
                    C_WHITE, 1, cv2.LINE_AA)

    # ---- accel strip: A EGO | A TARGET | CMD | OUT ----
    ay = py + 128
    cols = [("A EGO", app._a_meas_app, C_GREEN, 36),
            ("A TARGET", app.long.last_a_target, C_BLUE, 210),
            ("CMD", app.long.last_a_cmd, C_YELLOW, 400),
            ("OUT", app.long.last_a_cmd + app.long.last_a_fb, C_WHITE,
             540)]
    for name, val, col, x in cols:
        cv2.putText(canvas, name, (x, ay), FONT, 0.5, C_LABEL, 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, f"{val:+.2f}", (x, ay + 34), FONT, 0.8, col,
                    2, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true",
                    help="use the split 0.11.1 pair instead of supercombo")
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--map", default="west_coast_usa",
                    help="map for the scripted scenario (attach mode "
                         "ignores this)")
    ap.add_argument("--vehicle", default="bastion",
                    help="vehicle model to spawn / prefer when attaching")
    ap.add_argument("--attach", action="store_true",
                    help="hook the vehicle already loaded in-game "
                         "(Freeroam interception) instead of spawning "
                         "our scenario")
    ap.add_argument("--model", default=None,
                    help="path to a supercombo-compatible .onnx "
                         "(default: models/driving_supercombo.onnx)")
    ap.add_argument("--traffic", type=int, default=0,
                    help="spawn N game-managed AI traffic vehicles")
    ap.add_argument("--vision", default=None,
                    help="with --split: path to a driving_vision .onnx")
    ap.add_argument("--policy", default=None,
                    help="with --split: path to a driving_policy .onnx")
    ap.add_argument("--screen", action="store_true",
                    help="no-tech.key mode: capture the game window "
                         "(hood cam) and drive via a virtual wheel — "
                         "no beamngpy")
    ap.add_argument("--window", default="BeamNG",
                    help="with --screen: window title/class substring")
    ap.add_argument("--fov", type=float, default=90.0,
                    help="with --screen: horizontal FOV (deg) of the "
                         "in-game hood cam")
    ap.add_argument("--classic", action="store_true",
                    help="use the legacy hand-drawn cv2 UI instead of "
                         "the dearpygui one")
    args = ap.parse_args()

    app = App(args)
    bridge = ModBridge(app)
    bridge.start()
    if not args.classic:
        # default: the crisp dearpygui view (engine runs on a thread).
        from control_panel_view import run_view
        return run_view(app)

    panel = Panel(app)
    btn_y0 = CAM_VIEW_H + TELEM_H + 4
    panel.layout(btn_y0)

    canvas = np.zeros((TOTAL_H, UI_W, 3), dtype=np.uint8)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(UI_W * args.scale),
                     int(TOTAL_H * args.scale))
    # Qt backend realizes the window asynchronously — setMouseCallback
    # throws "NULL window handler" until the event loop has actually
    # created it. Pump the loop and retry; fall back to keyboard-only.
    mouse_ok = False
    for _ in range(20):
        cv2.imshow(WINDOW, canvas)
        cv2.waitKey(20)
        try:
            cv2.setMouseCallback(WINDOW, panel.on_mouse)
            mouse_ok = True
            break
        except cv2.error:
            time.sleep(0.05)
    if not mouse_ok:
        print("[panel] WARNING: mouse buttons unavailable in this cv2 "
              "build — use the keyboard bindings (see --help)", flush=True)
    last = time.monotonic()
    # per-stage timing: where does the tick go? gap = frame-to-frame
    # (includes waiting on the game's camera), model = warp+inference,
    # draw = all UI. If gap >> work, the GAME's render rate is the
    # bottleneck (camera frames simply aren't arriving at 20 Hz).
    perf = {"gap": 0.0, "model": 0.0, "ctl": 0.0, "draw": 0.0, "n": 0}
    try:
        while True:
            t0 = time.monotonic()
            bgr = app.world.poll_bgr()
            if bgr is None:
                if (cv2.waitKey(10) & 0xFF) == 27:
                    break
                continue
            app.frame_idx += 1
            dt = max(1e-3, t0 - last)
            last = t0
            perf["gap"] += dt

            desire_vec = None
            if app.desire_idx is not None:
                desire_vec = np.zeros(DESIRE_LEN, dtype=np.float32)
                desire_vec[app.desire_idx] = 1.0

            t_m = time.monotonic()
            img, big = app.queue.push(bgr, app.calib)
            d = app._decode(img, big, desire_vec)
            perf["model"] += time.monotonic() - t_m

            t_c = time.monotonic()
            tel = app.tel.snapshot()
            v_ego = tel["v_ego"]
            if app.screen_mode:
                # no telemetry: the model's own ego-speed estimate
                # (pose vx, 0.5%-accurate vs ground truth in tech
                # mode) is the speed source, lightly filtered
                app._vision_v += 0.25 * (
                    max(0.0, float(d.pose[0])) - app._vision_v)
                v_ego = app._vision_v
            app.control_tick(d, v_ego, dt)
            perf["ctl"] += time.monotonic() - t_c
            if panel.held_desire:
                app.action(panel.held_desire)   # extend while held

            # auto-disengage on loop-rate collapse
            if app.engaged and app.hz and app.hz < 8.0 \
                    and app.frame_idx > WARMUP_FRAMES:
                app.disengage(f"loop rate {app.hz:.0f} Hz")
            # sustained under-rate warning: the model's temporal
            # buffers assume 20 Hz — at ~12 Hz it sees a slow-motion
            # world and its plans thrash (big-model field report:
            # 'spazzing', steer activity 20x). Warn loudly instead of
            # driving badly until the 8 Hz hard stop.
            if app.hz:
                app.hz_ema += 0.05 * (app.hz - app.hz_ema)
            if (app.engaged and app.frame_idx > WARMUP_FRAMES
                    and app.hz_ema < 16.0
                    and time.monotonic() > app._rate_warned_until):
                app._rate_warned_until = time.monotonic() + 10.0
                app.set_banner(
                    f"loop {app.hz_ema:.0f} Hz < 20: model time-warped"
                    " - lower game graphics or use a smaller model", 6.0)

            # UI at the full 20 Hz (draw is ~4 ms — affordable on the
            # small model). Auto-degrades to alternate-frame 10 Hz
            # when the tick loses headroom (work-rate under 30 Hz
            # means 33+ ms of work in the 50 ms budget — big-model
            # contention). Incident capture always forces a draw.
            t_d = time.monotonic()
            lite = bool(app.hz) and app.hz < 30.0
            do_draw = (bool(app.incident_note) or not lite
                       or (app.frame_idx & 1) == 0)
            if do_draw:
                canvas[:CAM_VIEW_H] = cv2.resize(
                    draw_overlay(bgr, d, app.calib), (UI_W, CAM_VIEW_H),
                    interpolation=cv2.INTER_AREA)
                if app.charts_on:
                    draw_charts(canvas, app)
                draw_telemetry(canvas, app, tel, v_ego)
                canvas[btn_y0 - 4:] = 24
                panel.draw(canvas)
                hud_y = btn_y0 + 2 * BTN_ROW_H + 14
                for i, line in enumerate(app.hud_lines(d, v_ego)):
                    cv2.putText(canvas, line, (10, hud_y + 17 * i),
                                FONT, 0.4, (230, 230, 230), 1,
                                cv2.LINE_AA)
                if time.monotonic() < app.banner_until:
                    cv2.putText(canvas, app.banner, (16, 34), FONT,
                                0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(canvas, app.banner, (16, 34), FONT,
                                0.7, (80, 220, 255), 2, cv2.LINE_AA)
            if app.incident_note:
                inc_dir = os.path.join(ROOT, "debug_out", "incidents")
                os.makedirs(inc_dir, exist_ok=True)
                stamp = time.strftime("%H%M%S")
                cv2.imwrite(os.path.join(inc_dir, f"stop_{stamp}.png"), canvas)
                with open(os.path.join(inc_dir, f"stop_{stamp}.txt"), "w") as f:
                    f.write(app.incident_note + "\n")
                print(f"[panel] INCIDENT captured: {app.incident_note}",
                      flush=True)
                app.incident_note = None

            if do_draw:
                cv2.imshow(WINDOW, canvas)
            perf["draw"] += time.monotonic() - t_d
            perf["n"] += 1
            if perf["n"] >= 100:
                n = perf["n"]
                line = (f"gap {perf['gap']/n*1000:5.1f} ms/frame "
                        f"({n/max(perf['gap'],1e-3):4.1f} Hz)  |  "
                        f"model {perf['model']/n*1000:5.1f}  "
                        f"ctl {perf['ctl']/n*1000:4.1f}  "
                        f"draw {perf['draw']/n*1000:4.1f} ms")
                print(f"[perf] {line}", flush=True)
                os.makedirs(os.path.join(ROOT, "debug_out"),
                            exist_ok=True)
                with open(os.path.join(ROOT, "debug_out",
                                       "perf_last.txt"), "w") as f:
                    f.write(line + "\n")
                for k in perf:
                    perf[k] = 0.0
                perf["n"] = 0

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord("e"):
                app.action("engage")
            elif key == ord("f"):
                app.action("force")
            elif key == ord("a"):
                app.action("lane_l")
            elif key == ord("d"):
                app.action("lane_r")
            elif key == ord("z"):
                app.action("turn_l")
            elif key == ord("c"):
                app.action("turn_r")
            elif key == ord("l"):
                app.action("long")
            elif key == ord("g"):
                app.action("ai")
            elif key == ord("r"):
                app.action("cal")
            elif key == ord("v"):
                app.action("cam")
            elif key == ord("b"):
                app.charts_on = not app.charts_on
            elif key == ord("-"):
                app.action("spd_dn")
            elif key == ord("="):
                app.action("spd_up")

            el = time.monotonic() - t0
            app.hz = 1.0 / el if el > 0 else 0.0
            time.sleep(max(0.0, 0.05 - el))
        return 0
    finally:
        try:
            if app.lp.session_samples > 0:
                app.lp.save()
        except Exception:
            pass
        try:
            if app._log_f is not None:
                app._log_f.close()
        except Exception:
            pass
        app.sender.stop()
        app.world.apply(0.0, 0.0, 0.0)
        bridge.stop()
        app.tel.stop()
        app.world.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
