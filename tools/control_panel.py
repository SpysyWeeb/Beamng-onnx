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
import sys
import threading
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

np.seterr(divide="ignore", invalid="ignore")

from simsteer.core.calibration import Calibration
from simsteer.core.constants import DESIRE_LEN
from simsteer.core.control import ControllerConfig, LateralController, LongitudinalController
from simsteer.core.learners.liveparams import LiveParams
from simsteer.core.model import DrivingModel
from simsteer.core.postprocess import decode
from simsteer.core.preprocess import FrameQueue
from simsteer.core.supercombo import SupercomboModel
from simsteer.ui.overlay import draw_overlay

from beamng.world import (BeamNGOnnxWorld, CAM_W, CAM_H, CAM_FOV_H_DEG,
                          CAM_HEIGHT_M, CAM_LATERAL_SIGN)

# ASCII only: cv2's Qt backend fails setMouseCallback ("NULL window
# handler") when the window name contains non-ASCII (e.g. an em-dash).
WINDOW = "Beamng-onnx control panel"
WHEELBASE_M = 2.9          # bastion-ish; constant error folds into LiveParams
PANEL_H = 96               # button strip + HUD height (px)
DESIRE_HOLD_S = {1: 3.0, 2: 3.0, 3: 2.5, 4: 2.5}   # turn L/R, lane L/R
DESIRE_NAME = {1: "TURN L", 2: "TURN R", 3: "LANE L", 4: "LANE R"}
WARMUP_FRAMES = 100        # model context fill before engage allowed
# Calibration = BeamNG's AI drives while LiveParams fits passively from
# electrics steering_input (verified to reflect AI input). The AI stays
# on the road — an open-loop scripted slalom did not (it fed the learner
# rail-scraping garbage and produced wild fits). Two speeds, because at
# a single speed the rack model's `a` and `b·v²` terms are collinear
# and the split extrapolates wrong. (target_v m/s, duration s):
CAL_PHASES = [(12.0, 40.0), (18.0, 45.0)]
CAL_DURATION_S = sum(p[1] for p in CAL_PHASES)


class Telemetry(threading.Thread):
    """Polls world telemetry (~10 Hz TCP) off the model loop."""

    def __init__(self, world: BeamNGOnnxWorld):
        super().__init__(daemon=True, name="telemetry")
        self.world = world
        self.lock = threading.Lock()
        self.data = {"v_ego": 0.0, "steering_deg": 0.0,
                     "steering_input": 0.0, "heading_rad": 0.0,
                     "pos": (0.0, 0.0, 0.0)}
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


class Panel:
    """Buttons + HUD strip below the camera view."""

    def __init__(self, state: "App"):
        self.app = state
        self.buttons: list[tuple[tuple[int, int, int, int], str, str]] = []

    def layout(self, y0: int) -> None:
        labels = [("engage", ""), ("lane_l", "< LANE"), ("lane_r", "LANE >"),
                  ("turn_l", "< TURN"), ("turn_r", "TURN >"),
                  ("long", ""), ("cal", "CAL"), ("ai", "AI"),
                  ("spd_dn", "SPD-"), ("spd_up", "SPD+")]
        n = len(labels)
        w = CAM_W // n
        self.buttons = [((i * w, y0, w - 4, 40), key, lab)
                        for i, (key, lab) in enumerate(labels)]

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (bx, by, bw, bh), key, _ in self.buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.app.action(key)
                return

    def draw(self, canvas: np.ndarray) -> None:
        app = self.app
        for (bx, by, bw, bh), key, lab in self.buttons:
            if key == "engage":
                lab = "DISENGAGE" if app.engaged else "ENGAGE"
                col = (60, 60, 200) if app.engaged else (60, 160, 60)
            elif key == "long":
                lab = f"LONG {'ON' if app.long_enabled else 'off'}"
                col = (90, 120, 60) if app.long_enabled else (70, 70, 70)
            elif key == "ai":
                lab = f"AI {'ON' if app.ai_on else 'off'}"
                col = (140, 90, 40) if app.ai_on else (70, 70, 70)
            elif key == "cal":
                col = (40, 110, 140) if app.cal_active else (70, 70, 70)
                if app.cal_active:
                    lab = f"CAL {int(app.cal_left_s)}s"
            else:
                col = (70, 70, 70)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, -1)
            cv2.putText(canvas, lab, (bx + 8, by + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)


class App:
    def __init__(self, args):
        self.calib = Calibration(image_w=CAM_W, image_h=CAM_H,
                                 fov_h_deg=CAM_FOV_H_DEG,
                                 height_m=CAM_HEIGHT_M,
                                 lateral_sign=CAM_LATERAL_SIGN)
        # World FIRST, GPU session second (RDNA4 level-load VRAM rule).
        self.world = BeamNGOnnxWorld()
        if args.split:
            self.model = DrivingModel(
                providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
                policy_providers=["CPUExecutionProvider"], intra_op_threads=3)
            self._decode = lambda i, b, d: decode(*self.model.step(i, b, desire=d))
            name = f"split ({self.model.active_provider})"
        else:
            self.model = SupercomboModel(
                providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
                intra_op_threads=3)
            self._decode = lambda i, b, d: self.model.decode(
                self.model.step(i, b, desire=d))
            name = f"supercombo ({self.model.active_provider})"
        print(f"[panel] model: {name}", flush=True)

        self.queue = FrameQueue()
        self.tel = Telemetry(self.world)
        self.tel.start()

        cfg = ControllerConfig()
        cfg.wheelbase_m = WHEELBASE_M
        cfg.max_speed_mps = 25.0
        self.cfg = cfg
        self.lp = LiveParams(game="beamng")
        self.lat = LateralController(cfg, live_params=self.lp)
        self.long = LongitudinalController(cfg)

        self.engaged = False
        self.long_enabled = True
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
            self.desire_idx = idx
            self.desire_until = now + DESIRE_HOLD_S[idx]
            self.set_banner(f"desire: {DESIRE_NAME[idx]}")
        elif key == "long":
            self.long_enabled = not self.long_enabled
            self.set_banner(f"longitudinal {'ON' if self.long_enabled else 'OFF'}")
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
        elif key == "spd_dn":
            self.cfg.max_speed_mps = max(5.0, self.cfg.max_speed_mps - 2.5)
            self.set_banner(f"max speed {self.cfg.max_speed_mps*2.237:.0f} mph")
        elif key == "spd_up":
            self.cfg.max_speed_mps = min(45.0, self.cfg.max_speed_mps + 2.5)
            self.set_banner(f"max speed {self.cfg.max_speed_mps*2.237:.0f} mph")

    def engage(self, forced: bool = False) -> None:
        if self.ai_on:
            self.world.ai_disable()
            self.ai_on = False
        self.world.realistic_gearbox()   # arcade + held brake = REVERSE
        self.cal_active = False
        self.lat.reset()
        self.long.reset()
        self.engaged = True
        self.set_banner("ENGAGED" + (" (forced)" if forced else ""))

    def disengage(self, reason: str) -> None:
        self.engaged = False
        self.lat.reset()
        self.long.reset()
        self.world.apply(0.0, 0.0, 0.0)
        self.set_banner(f"DISENGAGED ({reason})")

    def start_cal(self) -> None:
        if self.engaged:
            self.disengage("calibration")
        self.cal_active = True
        self.cal_until = time.monotonic() + CAL_DURATION_S
        self._cal_speed = CAL_PHASES[0][0]
        self.world.ai_drive(self._cal_speed)
        self.ai_on = True
        self.set_banner("CAL: BeamNG AI drives, learner watches — hands off",
                        4.0)

    # ---- per-frame control ----

    def control_tick(self, decoded, v_ego: float, dt: float) -> None:
        now = time.monotonic()

        # Synthesized wheel angle from the model's ego yaw rate.
        yaw_rate = float(decoded.pose[5])
        k_meas = yaw_rate / max(v_ego, 1.0)
        wheel = math.atan(k_meas * WHEELBASE_M) if v_ego > 1.0 else None

        desire_active = (self.desire_idx is not None
                         and now < self.desire_until)
        if not desire_active:
            self.desire_idx = None

        steer, thr, brk = 0.0, 0.0, 0.0
        commanded = None
        if self.cal_active:
            if now >= self.cal_until:
                self.cal_active = False
                self.lp.save()
                # AI keeps driving after cal (self.ai_on stays True) —
                # engage() will take over from motion, no dead stop.
                self.set_banner(
                    f"CAL done: a={self.lp.a_linear:.2f} "
                    f"b={self.lp.b_quad:.4f} "
                    f"samples={self.lp.session_samples} "
                    f"trusted={self.lp.trusted()}", 5.0)
            else:
                t = CAL_DURATION_S - self.cal_left_s
                target_v, acc = CAL_PHASES[-1][0], 0.0
                for pv, pd in CAL_PHASES:
                    acc += pd
                    if t < acc:
                        target_v = pv
                        break
                if target_v != self._cal_speed:
                    self._cal_speed = target_v
                    self.world.vehicle.ai.set_speed(target_v, mode="limit")
        elif self.engaged:
            lane_change_cmd = desire_active and self.desire_idx in (3, 4)
            steer = self.lat.compute(decoded, v_ego,
                                     actual_wheel_angle=wheel,
                                     lane_change_command_active=lane_change_cmd,
                                     dt=dt)
            if self.long_enabled:
                thr, brk = self.long.compute(decoded, v_ego)
            self.world.apply(steer, thr, brk)
            commanded = steer

        # Feed the learner: our command while we drive, the game's own
        # steering input while a human / BeamNG AI drives (passive fit).
        tel = self.tel.snapshot()
        game_steer = commanded if commanded is not None else tel["steering_input"]
        self.lp.update(game_steer, v_ego, wheel, commanded_axis=game_steer)

        self.last_steer, self.last_thr, self.last_brk = steer, thr, brk

    # ---- HUD ----

    def hud_lines(self, decoded, v_ego: float) -> list[str]:
        lp = self.lp
        probs = np.round(decoded.lane_lines_prob, 2)
        lead_p = float(decoded.lead_prob[0]) if decoded.lead_prob.size else 0.0
        l1 = (f"{'ENGAGED' if self.engaged else 'manual '}  "
              f"{v_ego*2.237:4.0f} mph (cap {self.cfg.max_speed_mps*2.237:.0f})  "
              f"steer {self.last_steer:+.2f}  thr {self.last_thr:.2f}  "
              f"brk {self.last_brk:.2f}  {self.hz:4.1f} Hz")
        l2 = (f"lanes {probs}  lead {lead_p:.2f}"
              + (f" @{self.long.last_lead_x:.0f}m" if self.long.last_lead_x < 500 else "")
              + f"  curv {self.lat.last_curvature:+.4f}"
              + (f"  AEB!" if self.long.last_aeb else ""))
        trust = "OK" if lp.trusted() else "COLD"
        l3 = (f"rack a={lp.a_linear:+.2f} b={lp.b_quad:+.4f} "
              f"n={max(lp.samples, lp.session_samples)} [{trust}]")
        if self.desire_idx is not None:
            l3 += f"   >>> {DESIRE_NAME[self.desire_idx]} " \
                  f"({self.desire_until - time.monotonic():.1f}s)"
        return [l1, l2, l3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true",
                    help="use the split 0.11.1 pair instead of supercombo")
    ap.add_argument("--scale", type=float, default=0.8)
    args = ap.parse_args()

    app = App(args)
    panel = Panel(app)
    panel.layout(CAM_H + 4)

    canvas = np.zeros((CAM_H + PANEL_H, CAM_W, 3), dtype=np.uint8)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, int(CAM_W * args.scale),
                     int((CAM_H + PANEL_H) * args.scale))
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

            desire_vec = None
            if app.desire_idx is not None:
                desire_vec = np.zeros(DESIRE_LEN, dtype=np.float32)
                desire_vec[app.desire_idx] = 1.0

            img, big = app.queue.push(bgr, app.calib)
            d = app._decode(img, big, desire_vec)

            tel = app.tel.snapshot()
            v_ego = tel["v_ego"]
            app.control_tick(d, v_ego, dt)

            # auto-disengage on loop-rate collapse
            if app.engaged and app.hz and app.hz < 8.0 \
                    and app.frame_idx > WARMUP_FRAMES:
                app.disengage(f"loop rate {app.hz:.0f} Hz")

            canvas[:CAM_H] = draw_overlay(bgr, d, app.calib)
            canvas[CAM_H:] = 24
            panel.draw(canvas)
            for i, line in enumerate(app.hud_lines(d, v_ego)):
                cv2.putText(canvas, line, (10, CAM_H + 58 + 16 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (230, 230, 230), 1, cv2.LINE_AA)
            if time.monotonic() < app.banner_until:
                cv2.putText(canvas, app.banner, (CAM_W // 2 - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(canvas, app.banner, (CAM_W // 2 - 200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 255), 2,
                            cv2.LINE_AA)
            cv2.imshow(WINDOW, canvas)

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
        app.world.apply(0.0, 0.0, 0.0)
        app.tel.stop()
        app.world.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
