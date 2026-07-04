"""Two-model driving inference: vision -> policy.

Each step:
  1. Run vision on (img, big_img) — both (1, 12, 128, 256) uint8.
  2. Slice `hidden_state` from the vision output; push it onto the
     features ringbuffer; sample 25 timesteps as `features_buffer`.
  3. Push the desire pulse onto the desire ringbuffer; max-pool over
     each FRAME_SKIP window to get 25 timesteps as `desire_pulse`.
  4. Run policy on (desire_pulse, traffic_convention, features_buffer).
  5. Return the flat outputs from both models (as float32 numpy arrays);
     postprocess.py decodes them.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

from simsteer.core.constants import (
    DESIRE_BUFFER_LEN,
    DESIRE_LEN,
    FEATURE_BUFFER_LEN,
    FEATURE_LEN,
    FRAME_SKIP,
    VISION_SLICES,
)
from simsteer.paths import model_path

VISION_PATH = model_path("driving_vision.onnx")
POLICY_PATH = model_path("driving_policy.onnx")


def _make_session(path: Path, providers: list[str],
                  intra_op_threads: int | None = None) -> ort.InferenceSession:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python tools/fetch_model.py`")
    so = ort.SessionOptions()
    if intra_op_threads is not None:
        # ORT's CPU EP defaults to one thread per core, which saturates the
        # whole machine at 20 Hz even though a few threads already beat the
        # 50 ms frame budget. Cap it when running alongside a game.
        so.intra_op_num_threads = intra_op_threads
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


class DrivingModel:
    """Stateful: holds the feature/desire ringbuffers between steps."""

    FEAT_BUF_LEN = FRAME_SKIP * (FEATURE_BUFFER_LEN - 1) + 1   # 97
    DESIRE_BUF_LEN = FRAME_SKIP * DESIRE_BUFFER_LEN            # 100

    def __init__(self, providers: list[str] | None = None,
                 policy_providers: list[str] | None = None,
                 intra_op_threads: int | None = None,
                 vision_path: str | None = None,
                 policy_path: str | None = None) -> None:
        # DirectML accelerates the vision encoder (the big one). The policy
        # head currently fails to initialize on DML (opset-20 op the DML EP
        # rejects with E_INVALIDARG), and at ~3 ms on CPU it isn't the
        # bottleneck anyway. Caller can override either list.
        if providers is None:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        if policy_providers is None:
            policy_providers = ["CPUExecutionProvider"]
        from pathlib import Path
        self.vision = _make_session(
            Path(vision_path) if vision_path else VISION_PATH,
            providers, intra_op_threads)
        self.policy = _make_session(
            Path(policy_path) if policy_path else POLICY_PATH,
            policy_providers, intra_op_threads)
        self.active_provider = self.vision.get_providers()[0]
        self.policy_provider = self.policy.get_providers()[0]

        # Ringbuffers (float32 internally; cast to float16 only when feeding policy).
        self._feat_q: deque[np.ndarray] = deque(
            [np.zeros(FEATURE_LEN, dtype=np.float32) for _ in range(self.FEAT_BUF_LEN)],
            maxlen=self.FEAT_BUF_LEN,
        )
        self._desire_q: deque[np.ndarray] = deque(
            [np.zeros(DESIRE_LEN, dtype=np.float32) for _ in range(self.DESIRE_BUF_LEN)],
            maxlen=self.DESIRE_BUF_LEN,
        )
        # Previous frame's raw desire, for the rising-edge pulse (below).
        self._prev_desire = np.zeros(DESIRE_LEN, dtype=np.float32)

    def step(self, img: np.ndarray, big_img: np.ndarray | None = None,
             desire: np.ndarray | None = None,
             traffic_convention: tuple[float, float] = (1.0, 0.0),
             ) -> tuple[np.ndarray, np.ndarray]:
        # traffic_convention: openpilot's modeld sets index `int(is_rhd)`
        # to 1.0 (is_rhd = driver sits on the right = left-side traffic).
        #   is_rhd False -> index 0 -> (1, 0) = RIGHT-side traffic (US, EU)
        #   is_rhd True  -> index 1 -> (0, 1) = LEFT-side traffic  (UK, JP, AU)
        # An earlier comment here had this table inverted, and the (0, 1)
        # default it justified told the model we drive on the LEFT — on US
        # maps that read as "permanently in the oncoming lane": left lane
        # bias, defensive phantom stops, launch hesitance.
        """Run one inference step. Returns (vision_out_flat, policy_out_flat)
        as float32 numpy arrays of shapes (1576,) and (1000,)."""
        if big_img is None:
            big_img = img

        # --- Vision ---
        vision_outputs = self.vision.run(
            None, {"img": img.astype(np.uint8, copy=False),
                   "big_img": big_img.astype(np.uint8, copy=False)})
        vision_flat = vision_outputs[0].astype(np.float32).reshape(-1)

        # Latch hidden_state into the feature ringbuffer.
        h_start, h_end = VISION_SLICES["hidden_state"]
        hidden = vision_flat[h_start:h_end]
        self._feat_q.append(hidden)

        # Push desire pulse into its ringbuffer. openpilot's modeld feeds
        # the desire as a rising-EDGE pulse, not a held signal: only the
        # frame a desire first goes high is nonzero, then it's zero even
        # while the command is still held. The model latches the maneuver
        # from that single pulse; feeding a HELD one-hot instead fills
        # the whole buffer with 1.0 (out of distribution) and makes the
        # policy throw the wheel to the commanded side while the plan
        # collapses to a straight line. Callers may pass a held one-hot;
        # convert it here.  new = where(desire - prev > 0.99, desire, 0)
        if desire is None:
            desire = np.zeros(DESIRE_LEN, dtype=np.float32)
        desire = desire.astype(np.float32, copy=False)
        pulse = np.where(desire - self._prev_desire > 0.99, desire, 0.0
                         ).astype(np.float32)
        self._prev_desire = desire
        self._desire_q.append(pulse)

        # Build policy inputs by sampling the queues.
        feat_arr = np.stack(list(self._feat_q), axis=0)        # (97, 512)
        feat_sampled = feat_arr[::FRAME_SKIP]                  # (25, 512)
        features_buffer = feat_sampled[np.newaxis, ...].astype(np.float16)

        des_arr = np.stack(list(self._desire_q), axis=0)        # (100, 8)
        des_pooled = des_arr.reshape(DESIRE_BUFFER_LEN, FRAME_SKIP, DESIRE_LEN).max(axis=1)
        desire_pulse = des_pooled[np.newaxis, ...].astype(np.float16)  # (1, 25, 8)

        tc = np.asarray([traffic_convention], dtype=np.float16)  # (1, 2)

        # --- Policy ---
        policy_outputs = self.policy.run(None, {
            "desire_pulse": desire_pulse,
            "traffic_convention": tc,
            "features_buffer": features_buffer,
        })
        policy_flat = policy_outputs[0].astype(np.float32).reshape(-1)

        return vision_flat, policy_flat
