#!/usr/bin/env python3
"""Figures for Experiment 4b (Environment C, CPU only).

s1 -> media/hackathon/cosmos_fd_axis_sweep.png: slot 8 (robomind-franka encoding) vs the
      untrained libero slot, frame k=16 for each commanded axis; arrows = displacement
      predicted from the fitted action frame, dots = tracked gripper.
s3 -> media/hackathon/cosmos_fd_slot8_replay.png: real vs main / reversed / zero on real replays.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e5_analyze import REPLAY, S_START, TRACKERS, jobs, proj, vid  # noqa: E402


def s1(ep="can_s1_e0020_200000", out=ROOT / "media/hackathon/cosmos_fd_axis_sweep.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads((ROOT / "results/hackathon/explore/e5_probe.json").read_text())
    M = np.asarray(res["S1"]["by_encoding"]["R_q"]["trackers"]["farn"]["best_M"])
    js = {(j["stage"], j["unit"], j.get("enc"), j["arm"]): j for j in jobs()}
    d = np.load(REPLAY / f"{ep}.npz", allow_pickle=True)
    T, P, Rs, p0 = d["cam_T"], d["eef_pos"][S_START], d["eef_mat"][S_START], d["eef_px_raw"][S_START].astype(float)
    arms = ["hold", "+x", "-x", "+y", "-y", "+z", "-z"]
    rows = [("S1", "R_q", "trained slot 8 (robomind-franka)", 0.16, lambda i: Rs @ M[:, i]),
            ("S2", "libero", "untrained slot 5 (libero)", 0.32, lambda i: np.eye(3)[i])]
    fig, axes = plt.subplots(len(rows), len(arms) + 1, figsize=(2.4 * (len(arms) + 1), 2.7 * len(rows)), squeeze=False)
    for r, (st, enc, label, dist, frame) in enumerate(rows):
        ax = axes[r, 0]
        ax.imshow(d["frames"][S_START]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(label, fontsize=9)
        if r == 0:
            ax.set_title("input frame (k=0)", fontsize=9)
        for c, arm in enumerate(arms):
            j = js.get((st, ep, enc, arm))
            ax = axes[r, c + 1]
            ax.set_xticks([]); ax.set_yticks([])
            if j is None:
                continue
            v = vid(j)
            ax.imshow(v[16])
            tr = TRACKERS["farn"](v, p0)
            ax.plot(tr[:, 1], tr[:, 0], "-", color="#ffd400", lw=1.5)
            ax.plot([tr[16, 1]], [tr[16, 0]], "o", ms=5, color="#ffd400")
            if arm != "hold":
                i = "xyz".index(arm[1])
                sgn = 1 if arm[0] == "+" else -1
                q = proj(T, P + sgn * dist * frame(i)) - proj(T, P)
                ax.annotate("", xy=(p0[1] + q[1], p0[0] + q[0]), xytext=(p0[1], p0[0]),
                            arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=2))
            if r == 0:
                ax.set_title(f"command {arm}  (k=16)", fontsize=9)
    fig.suptitle("Experiment 4b: does the generated gripper go where it is told? Cyan = commanded direction "
                 "(fitted action frame), yellow = tracked gripper", x=0.01, ha="left", weight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def s3(units=None, out=ROOT / "media/hackathon/cosmos_fd_slot8_replay.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(ROOT / "results/hackathon/explore/e5_probe_s3_units.csv")
    q = df[(df.tracker == "farn") & (df.K == 16) & (df.real_px >= 10)].sort_values("real_px", ascending=False)
    js = {(j["unit"], j["arm"]): j for j in jobs() if j["stage"] == "S3"}
    if units is None:  # two successes ranked by true motion, plus the worst Square failure (not cherry-picked)
        u = q.unit.drop_duplicates()
        worst = q[q.unit.str.startswith("square")].sort_values("cos_main_minus_rev").unit.iloc[0]
        units = [u.iloc[0], u.iloc[len(u) // 3], worst]
    cols = ["real", "main", "reversed", "zero"]
    fig, axes = plt.subplots(len(units), len(cols) * 2, figsize=(2.3 * len(cols) * 2, 2.4 * len(units)), squeeze=False)
    for i, unit in enumerate(units):
        ep, s = unit.split("|")
        d = np.load(REPLAY / f"{ep}.npz", allow_pickle=True)
        idx = np.asarray(js[(unit, "main")]["real_idx"])
        for c, arm in enumerate(cols):
            v = d["frames"][idx] if arm == "real" else vid(js[(unit, arm)])
            for kk, k in enumerate((8, 16)):
                ax = axes[i, c * 2 + kk]
                ax.imshow(v[k]); ax.set_xticks([]); ax.set_yticks([])
                r_, c_ = d["eef_px_raw"][idx[k]]
                ax.plot([c_], [r_], "o", ms=5, mfc="none", mec="#00ff00", mew=1.5)
                if i == 0:
                    ax.set_title(f"{arm}  k={k}", fontsize=9)
        axes[i, 0].set_ylabel(f"{ep}\nstart {s}", fontsize=8)
    fig.suptitle("Experiment 4b: Cosmos3-Edge forward dynamics (trained Franka slot) on recorded simulated robosuite policy "
                 "rollouts; rows 1-2: typical Can clips, row 3: worst Square clip (green ring = true end-effector position)",
                 x=0.01, ha="left", weight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["s1"]):
        print({"s1": s1, "s3": s3}[name]())
