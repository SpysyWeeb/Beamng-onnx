"""Runner for openpilot's re-unified `driving_supercombo.onnx` (2026 master).

Comma merged driving_vision + driving_policy back into one graph. The image
inputs are unchanged (two stacked YUV6 frames, t and t-4, per camera view),
so FrameQueue output feeds it directly. What differs from DrivingModel:

  - `features_buffer` (1, 24, 512) holds PAST hidden states only — the model
    appends the current frame's feature internally. modeld keeps a 96-deep
    ringbuffer (hidden_state of steps t-96..t-1) and samples every
    FRAME_SKIP-th entry, so the newest sampled feature is t-4.
  - `desire_pulse` is (1, 25, 8): a 100-deep ringbuffer max-pooled over each
    FRAME_SKIP window. modeld also zeroes desire[0] and only passes rising
    edges, so a held desire doesn't re-trigger; replicated here.
  - New `action_t` input: [lat_action_t, long_action_t] — how far ahead (s)
    the model's action outputs should aim. modeld computes these from
    actuator delays; we take fixed values good for a sim.
  - One flat output (2576). The slice map is embedded in the ONNX metadata
    (`output_slices`, base64 pickle) rather than hardcoded, and there is no
    separate desired_curvature output — derive it from the plan (helpers at
    the bottom, ported from openpilot's drive_helpers).

decode() maps the flat output into simsteer's Decoded so the overlay,
curvature, and learner code run unchanged.
"""

from __future__ import annotations

import codecs
import pickle

import numpy as np

from simsteer.core.constants import (
    DESIRE_LEN,
    FEATURE_LEN,
    FRAME_SKIP,
    IDX_N,
    NUM_LANE_LINES,
    NUM_ROAD_EDGES,
    PLAN_WIDTH,
)
from simsteer.core.model import _make_session
from simsteer.core.postprocess import (
    Decoded,
    LEAD_MHP_SELECTION,
    LEAD_TRAJ_LEN,
    LEAD_WIDTH,
    _sigmoid,
    _softmax,
)
from simsteer.paths import model_path

SUPERCOMBO_PATH = model_path("driving_supercombo.onnx")

# Stds come out in log-space; clip before exp like openpilot's safe_exp
# (above ~11 the float16-trained values are garbage anyway).
_EXP_CLIP = 11.0


def _safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, None, _EXP_CLIP))


class SupercomboModel:
    """Stateful runner: holds the feature/desire ringbuffers between steps.

    step() returns the flat float32 output; decode() turns one into a
    Decoded. Kept separate so callers can stash raw outputs cheaply.
    """

    def __init__(self, providers: list[str] | None = None,
                 intra_op_threads: int | None = None,
                 lat_action_t: float = 0.3,
                 long_action_t: float = 0.5,
                 model_path: str | None = None) -> None:
        if providers is None:
            providers = ["CPUExecutionProvider"]
        from pathlib import Path
        self.session = _make_session(
            Path(model_path) if model_path else SUPERCOMBO_PATH,
            providers, intra_op_threads)
        self.active_provider = self.session.get_providers()[0]

        meta = self.session.get_modelmeta().custom_metadata_map
        self.output_slices: dict[str, slice] = pickle.loads(
            codecs.decode(meta["output_slices"].encode(), "base64"))
        self.checkpoint: str = meta.get("model_checkpoint", "?")

        shapes = {i.name: i.shape for i in self.session.get_inputs()}
        n_feats = shapes["features_buffer"][1]    # 24
        self.n_desire = shapes["desire_pulse"][1]  # 25
        self.feat_buf_len = FRAME_SKIP * n_feats          # 96
        self.desire_buf_len = FRAME_SKIP * self.n_desire  # 100

        self._feat_buf = np.zeros((self.feat_buf_len, FEATURE_LEN), dtype=np.float32)
        self._desire_buf = np.zeros((self.desire_buf_len, DESIRE_LEN), dtype=np.float32)
        self._prev_hidden = np.zeros(FEATURE_LEN, dtype=np.float32)
        self._prev_desire = np.zeros(DESIRE_LEN, dtype=np.float32)
        self._action_t = np.asarray([[lat_action_t, long_action_t]], dtype=np.float16)

    def step(self, img: np.ndarray, big_img: np.ndarray | None = None,
             desire: np.ndarray | None = None,
             traffic_convention: tuple[float, float] = (0.0, 1.0),
             ) -> np.ndarray:
        """Run one 20 Hz step. Returns the flat (2576,) float32 output."""
        if big_img is None:
            big_img = img
        if desire is None:
            desire = np.zeros(DESIRE_LEN, dtype=np.float32)

        # Rising-edge pulse, matching modeld: index 0 ("none") is forced to
        # zero, and a desire only enters the buffer on the frame it turns on.
        desire = desire.astype(np.float32, copy=True)
        desire[0] = 0.0
        pulse = np.where(desire - self._prev_desire > 0.99, desire, 0.0)
        self._prev_desire = desire

        # Shift ringbuffers (oldest first). The feature pushed is LAST step's
        # hidden state — the model computes the current one internally.
        self._feat_buf = np.roll(self._feat_buf, -1, axis=0)
        self._feat_buf[-1] = self._prev_hidden
        self._desire_buf = np.roll(self._desire_buf, -1, axis=0)
        self._desire_buf[-1] = pulse

        features_buffer = self._feat_buf[::FRAME_SKIP][np.newaxis].astype(np.float16)
        desire_pulse = self._desire_buf.reshape(
            self.n_desire, FRAME_SKIP, DESIRE_LEN).max(axis=1)[np.newaxis].astype(np.float16)
        tc = np.asarray([traffic_convention], dtype=np.float16)

        outs = self.session.run(None, {
            "img": img.astype(np.uint8, copy=False),
            "big_img": big_img.astype(np.uint8, copy=False),
            "features_buffer": features_buffer,
            "desire_pulse": desire_pulse,
            "traffic_convention": tc,
            "action_t": self._action_t,
        })
        flat = outs[0].astype(np.float32).reshape(-1)
        self._prev_hidden = flat[self.output_slices["hidden_state"]].copy()
        return flat

    # ---- decoding ----

    def _mu_std(self, flat: np.ndarray, name: str,
                shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        raw = flat[self.output_slices[name]]
        n = raw.size // 2
        return raw[:n].reshape(shape), _safe_exp(raw[n:]).reshape(shape)

    def decode(self, flat: np.ndarray) -> Decoded:
        plan, plan_std = self._mu_std(flat, "plan", (IDX_N, PLAN_WIDTH))
        lane_lines, _ = self._mu_std(flat, "lane_lines",
                                     (NUM_LANE_LINES, IDX_N, 2))
        road_edges, _ = self._mu_std(flat, "road_edges",
                                     (NUM_ROAD_EDGES, IDX_N, 2))
        pose, pose_std = self._mu_std(flat, "pose", (6,))
        road_tf, road_tf_std = self._mu_std(flat, "road_transform", (6,))
        wide_euler, _ = self._mu_std(flat, "wide_from_device_euler", (3,))
        leads, leads_std = self._mu_std(
            flat, "lead", (LEAD_MHP_SELECTION, LEAD_TRAJ_LEN, LEAD_WIDTH))

        # 8 raw values interleave (other, prob) per line — openpilot's
        # fill_model_msg reads sigmoid(x)[1::2] for the 4 line probs.
        ll_prob = _sigmoid(flat[self.output_slices["lane_lines_prob"]])[1::2]
        lead_prob = _sigmoid(flat[self.output_slices["lead_prob"]])
        desire_state = _softmax(flat[self.output_slices["desire_state"]])

        return Decoded(
            plan=plan, plan_std=plan_std,
            lane_lines=lane_lines, lane_lines_prob=ll_prob,
            road_edges=road_edges,
            pose=pose, pose_std=pose_std,
            road_transform=road_tf, road_transform_std=road_tf_std,
            wide_from_device_euler=wide_euler,
            desire_state=desire_state,
            lead_prob=lead_prob,
            leads=leads, leads_std=leads_std,
        )


# ---- plan -> action (ported from openpilot drive_helpers, for M3) ----
# This checkpoint has no direct `action` output (its last 2 floats are pad),
# so like modeld's fallback branch we derive the commands from the plan.

MIN_SPEED = 0.3
MIN_STABLE_DELAY = 0.1


def get_curvature_from_plan(plan: np.ndarray, t_idxs: np.ndarray,
                            v_ego: float, action_t: float) -> float:
    """Curvature to command now so heading matches the plan at t=action_t."""
    yaws = plan[:, 11]        # euler yaw
    yaw_rates = plan[:, 14]   # orientation_rate yaw
    if action_t < MIN_STABLE_DELAY:
        psi_target = (action_t / MIN_STABLE_DELAY) * np.interp(
            MIN_STABLE_DELAY, t_idxs, yaws)
    else:
        psi_target = np.interp(action_t, t_idxs, yaws)
    v = max(v_ego, MIN_SPEED)
    curv_from_psi = psi_target / (v * action_t)
    return float(2 * curv_from_psi - yaw_rates[0] / v)


def get_accel_from_plan(plan: np.ndarray, t_idxs: np.ndarray,
                        action_t: float,
                        v_ego_stopping: float = 0.3) -> tuple[float, bool]:
    """Acceleration to command now to hit the plan's speed at t=action_t.
    Returns (accel, should_stop)."""
    speeds = plan[:, 3]   # velocity x
    accels = plan[:, 6]   # acceleration x
    v_now, a_now = speeds[0], accels[0]
    if action_t < MIN_STABLE_DELAY:
        v_target = v_now + (action_t / MIN_STABLE_DELAY) * (
            np.interp(MIN_STABLE_DELAY, t_idxs, speeds) - v_now)
    else:
        v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / action_t - a_now
    should_stop = bool(v_now < v_ego_stopping and a_target < 0.1)
    return float(a_target), should_stop
