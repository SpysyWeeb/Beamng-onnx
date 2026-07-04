"""Print input/output names + shapes for the driving models, plus the
embedded `output_slices` metadata that tells us where each named output
(plan, lane_lines, hidden_state, etc.) lives inside the flat tensor.

Run after fetch_model.py. Use this any time we suspect the model version
changed — the rest of the code makes assumptions about these shapes.

    python tools/inspect_model.py
"""

from __future__ import annotations

import base64
import pickle
import sys
from pathlib import Path

import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
MODELS = [
    ROOT / "models" / "split" / "driving_vision.onnx",
    ROOT / "models" / "split" / "driving_policy.onnx",
]


def describe(path: Path) -> None:
    if not path.exists():
        print(f"  (missing — run tools/fetch_model.py)")
        return
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"  size: {path.stat().st_size / 1e6:.1f} MB")
    print(f"  INPUTS:")
    for inp in sess.get_inputs():
        print(f"    {inp.name:30s} {str(inp.shape):30s} {inp.type}")
    print(f"  OUTPUTS:")
    for out in sess.get_outputs():
        print(f"    {out.name:30s} {str(out.shape):30s} {out.type}")

    meta = sess.get_modelmeta().custom_metadata_map
    raw = meta.get("output_slices")
    if raw is not None:
        slices = pickle.loads(base64.b64decode(raw))
        print(f"  output_slices ({len(slices)}):")
        for name, sl in slices.items():
            start = sl.start or 0
            stop = "end" if sl.stop is None else sl.stop
            n = "?" if sl.stop is None else sl.stop - start
            print(f"    {name:30s} [{start:5d}:{stop!s:>5}]  ({n} floats)")


def main() -> int:
    for m in MODELS:
        print(f"\n=== {m.name} ===")
        describe(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
