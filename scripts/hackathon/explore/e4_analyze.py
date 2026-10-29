#!/usr/bin/env python3
"""Experiment 4 (exploratory), step 3 — fidelity of Cosmos3-Edge action-conditioned
forward dynamics against the REAL replayed robosuite frames (Environment B or C).

Per job (episode, chunk start s, condition) and future frame k = 1..16:
  psnr_full     PSNR(generated_k, real_{s+k}) on the whole 256x256 frame
  psnr_grip     PSNR in a 40x40 window centred on the TRUE projected EEF pixel at s+k
  motion_corr   Pearson correlation between |gen_k - gen_0| and |real_k - real_0|
                (does the generated motion happen where the real motion happens?)
Conditions are compared PAIRED on the same (episode, start) units; CIs come from
a bootstrap that resamples episodes (clusters).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data/outputs/hackathon_explore/e4_cosmos_fd"
OUT = ROOT / "results/hackathon/explore"
W = 20  # half window


def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(m, 1e-9))


def crop(img, rc):
    r, c = int(np.clip(rc[0], W, img.shape[0] - W)), int(np.clip(rc[1], W, img.shape[1] - W))
    return img[r - W:r + W, c - W:c + W]


def motion_corr(gen, real, k):
    g = np.abs(gen[k].astype(np.float64) - gen[0].astype(np.float64)).mean(-1).ravel()
    r = np.abs(real[k].astype(np.float64) - real[0].astype(np.float64)).mean(-1).ravel()
    if g.std() < 1e-9 or r.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(g, r)[0, 1])


def main():
    jobs = [json.loads(l) for l in (BASE / "gen/jobs.jsonl").read_text().splitlines()]
    replays = {}
    rows = []
    for j in jobs:
        ep, s, cond = j["episode"], j["start"], j["condition"]
        if ep not in replays:
            replays[ep] = dict(np.load(BASE / "replay" / f"{ep}.npz", allow_pickle=True))
        d = replays[ep]
        gen = np.load(BASE / "gen" / f"{ep}__{s:03d}__{cond}.npz")["video"]
        real = d["frames"][s:s + gen.shape[0]]
        n = min(len(gen), len(real))
        pf, pg, mc = [], [], []
        for k in range(1, n):
            pf.append(psnr(gen[k], real[k]))
            rc = d["eef_px_raw"][s + k]
            pg.append(psnr(crop(gen[k], rc), crop(real[k], rc)))
            mc.append(motion_corr(gen, real, k))
        disp = float(np.linalg.norm(d["eef_pos"][s + n - 1] - d["eef_pos"][s]))
        rows.append({"episode": ep, "task": j["task"], "start": s, "condition": cond,
                     "psnr_full": float(np.mean(pf)), "psnr_grip": float(np.mean(pg)),
                     "psnr_grip_last": float(pg[-1]), "motion_corr": float(np.nanmean(mc)),
                     "eef_displacement_m": disp, "gen_seconds": j["seconds"]})
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e4_cosmos_fd_jobs.csv", index=False)
    conds = sorted(df.condition.unique())
    summary = df.groupby("condition")[["psnr_full", "psnr_grip", "psnr_grip_last", "motion_corr"]].mean().round(3)
    rng = np.random.default_rng(0)
    paired = []
    piv = {m: df.pivot_table(index=["episode", "start"], columns="condition", values=m) for m in ("psnr_full", "psnr_grip", "motion_corr")}
    for m, p in piv.items():
        for c in conds:
            if c == "main" or c not in p:
                continue
            diff = (p["main"] - p[c]).dropna()
            eps = diff.index.get_level_values(0).unique().tolist()
            boots = []
            for _ in range(4000):
                pick = rng.choice(eps, size=len(eps), replace=True)
                boots.append(np.mean(np.concatenate([diff.loc[e].values for e in pick])))
            paired.append({"metric": m, "main_minus": c, "mean_diff": float(diff.mean()),
                           "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                           "main_wins": int((diff > 0).sum()), "units": int(len(diff)), "episodes": len(eps)})
    # moving-chunk subset (real EEF moved >= 5 cm in the chunk): where action conditioning should matter
    mov = df[df.eef_displacement_m >= 0.05]
    summary_moving = mov.groupby("condition")[["psnr_grip", "motion_corr"]].mean().round(3)
    res = {"label": "EXPLORATORY (post-hoc design; adapter follows the Cosmos framework LIBERO pipeline)",
           "n_jobs": len(df), "episodes": int(df.episode.nunique()), "chunk_starts": sorted(df.start.unique().tolist()),
           "per_condition_mean": summary.reset_index().to_dict("records"),
           "per_condition_mean_moving_chunks": summary_moving.reset_index().to_dict("records"),
           "n_moving_units": int(mov[mov.condition == "main"].shape[0]),
           "paired_vs_main": paired, "mean_gen_seconds": float(df.gen_seconds.mean())}
    (OUT / "e4_cosmos_fd.json").write_text(json.dumps(res, indent=2))
    print(summary.to_string())
    print("moving chunks:\n", summary_moving.to_string())
    for p in paired:
        print(f"{p['metric']:12s} main - {p['main_minus']:9s} = {p['mean_diff']:+.3f}  CI {np.round(p['ci95'], 3)}  wins {p['main_wins']}/{p['units']}")


if __name__ == "__main__":
    main()


def figure(units=None, out=ROOT / "media/hackathon/cosmos_forward_dynamics.png"):
    """Real vs generated frames (k = 0, 8, 16) for a few moving chunks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(OUT / "e4_cosmos_fd_jobs.csv")
    mov = df[(df.condition == "main") & (df.eef_displacement_m >= 0.08)].sort_values(["episode", "start"])
    if units is None:
        units = [tuple(x) for x in mov[["episode", "start"]].drop_duplicates().iloc[[0, len(mov) // 2, -1]].values]
    rows = ["real", "main", "reversed", "zero"]
    fig, axes = plt.subplots(len(units), len(rows) * 3, figsize=(24, 2.3 * len(units)), squeeze=False)
    for i, (ep, s) in enumerate(units):
        d = np.load(BASE / "replay" / f"{ep}.npz", allow_pickle=True)
        for j, cond in enumerate(rows):
            vid = d["frames"][s:s + 17] if cond == "real" else np.load(BASE / "gen" / f"{ep}__{int(s):03d}__{cond}.npz")["video"]
            for kk, k in enumerate((0, 8, 16)):
                ax = axes[i, j * 3 + kk]
                ax.imshow(vid[min(k, len(vid) - 1)]); ax.set_xticks([]); ax.set_yticks([])
                r, c = d["eef_px_raw"][s + k]
                ax.plot([c], [r], "o", ms=4, mfc="none", mec="#00ff00")
                if i == 0:
                    ax.set_title(f"{cond}  k={k}", fontsize=9)
            if j == 0:
                axes[i, 0].set_ylabel(f"{ep}\nstart {s}", fontsize=8)
    fig.suptitle("Experiment 4 - Cosmos3-Edge action-conditioned forward dynamics vs the real robosuite rollout "
                 "(green ring = true end-effector position)", x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "figure":
    print(figure())
