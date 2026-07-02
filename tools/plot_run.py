#!/usr/bin/env python3
"""Plot a control-panel run log: what the model WANTED vs what the car
actually DID, over the whole session.

The panel writes debug_out/run_YYYYMMDD_HHMMSS.csv every control tick
(20 Hz). This renders it as stacked time-series so misbehavior can be
attributed: target/measured traces apart = execution problem (pedal
maps, brake ramp, steering gain); traces together but wrong = the model
asked for it (scene/model judgment).

Usage:
    .venv/bin/python3 tools/plot_run.py            # newest run log
    .venv/bin/python3 tools/plot_run.py <log.csv>  # specific log
Writes <log>.png next to the CSV.
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MPH = 2.237


def main() -> int:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        logs = sorted(glob.glob(os.path.join(ROOT, "debug_out", "run_*.csv")))
        if not logs:
            print("no run logs in debug_out/ — drive with the panel open")
            return 1
        path = logs[-1]
    data = np.genfromtxt(path, delimiter=",", names=True,
                         dtype=None, encoding="utf-8")
    if data.size < 10:
        print(f"{path}: too few rows ({data.size})")
        return 1
    t = data["t"] - data["t"][0]
    eng = data["eng"].astype(float)
    # longitudinal targets are meaningless while disengaged / long-off
    long_on = (eng > 0.5) & (np.char.not_equal(
        data["mode"].astype(str), "off")) & (data["cal"] < 0.5)
    vt = np.where(long_on, data["v_target"], np.nan)
    at = np.where(long_on, data["a_target"], np.nan)
    ac = np.where(long_on, data["a_cmd"], np.nan)
    kd = np.where((eng > 0.5) & (data["cal"] < 0.5), data["k_des"], np.nan)

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(os.path.basename(path), fontsize=11)

    ax = axes[0]
    ax.plot(t, data["v_ego"] * MPH, color="tab:green", lw=1.0, label="v ego")
    ax.plot(t, vt * MPH, color="tab:orange", lw=1.0, label="v target")
    ax.set_ylabel("mph")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.plot(t, data["a_meas"], color="tab:green", lw=0.9, label="a measured")
    ax.plot(t, ac, color="tab:orange", lw=0.9, label="a cmd (setpoint)")
    ax.plot(t, at, color="tab:red", lw=0.7, alpha=0.6, label="a plan target")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("m/s$^2$")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.plot(t, data["thr"], color="tab:green", lw=0.9, label="throttle")
    ax.plot(t, -data["brk"], color="tab:red", lw=0.9, label="-brake")
    ax.plot(t, data["a_fb"], color="tab:purple", lw=0.7, label="a feedback")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("pedals")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[3]
    ax.plot(t, data["k_meas"] * 1000, color="tab:green", lw=0.9,
            label="curvature measured")
    ax.plot(t, kd * 1000, color="tab:orange", lw=0.9,
            label="curvature desired")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("curv x1000 (1/m)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[4]
    ax.plot(t, data["lane_off"], color="tab:blue", lw=0.9,
            label="lane-center offset")
    ax.plot(t, data["pitch_applied"], color="tab:gray", lw=0.8,
            label="calib pitch applied (deg)")
    ax.plot(t, data["pitch_learned"], color="tab:brown", lw=0.8, ls="--",
            label="calib pitch learned (deg)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylabel("m / deg")
    ax.set_xlabel("t (s)")
    ax.legend(loc="upper left", fontsize=8)

    # shade engaged spans on every panel
    d = np.diff(np.concatenate([[0.0], eng, [0.0]]))
    starts, ends = np.where(d > 0)[0], np.where(d < 0)[0] - 1
    for ax in axes:
        for s, e in zip(starts, ends):
            s_i, e_i = min(s, len(t) - 1), min(e, len(t) - 1)
            ax.axvspan(t[s_i], t[e_i], color="tab:green", alpha=0.06)

    out = os.path.splitext(path)[0] + ".png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  ({data.size} rows, {t[-1]:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
