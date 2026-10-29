#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- y1 verification, part 2: is the fixed 25/25 seed split typical?

Outcome-only descriptive checks (no selection rule is fitted here):
  * success rate of pool half (100000-100024) vs held-out half (100025-100049)
  * the y1 reference (perfect judge, no prior, all 25 pool episodes -> pool-best pick) regret on
    the original split vs 5000 random 25/25 partitions (percentile of the original split)
  * zero-rollout latest-epoch pick vs that pool-best pick over 2000 random partitions (paired)
  * within-run Spearman of each PLV metric with epoch (outcome-free) and rank-partial
    correlation with 50-episode success given epoch (descriptive only)

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y1_plv_prior_budget_verify_split.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/hackathon/explore/y1_plv_prior_budget_verification_split.json"
COLS = ["val_action_l1", "open_loop_smoothness", "plv1_smoothness", "replan_overlap_h16", "boundary_jump_h16",
        "val_loss_logged"]


def load():
    df = pd.read_csv(ROOT / "results/hackathon/baseline_metrics.csv")
    runs = []
    for r, g in df.groupby("run_id"):
        g = g.sort_values("epoch")
        rd = ROOT / Path(g.checkpoint_path.iloc[0]).parents[1]
        rows = {}
        for line in (rd / "logs.json.txt").read_text().splitlines():
            try:
                x = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in x:
                rows.setdefault(int(x["epoch"]), {}).update(x)
        Y = np.array([[float(rows[e][f"test/sim_max_reward_{s}"]) > 0.5 for s in range(100000, 100050)]
                      for e in g.epoch.astype(int)], float)
        runs.append((r, g.epoch.values, Y, g))
    return runs


def regrets(runs, pc, hc):
    last, pb = [], []
    for _, ep, Y, _ in runs:
        H, P = Y[:, hc].mean(1), Y[:, pc].mean(1)
        last.append(H.max() - H[-1])
        pb.append(H.max() - H[np.flatnonzero(P == P.max())].mean())
    return np.array(last), np.array(pb)


def main():
    runs = load()
    out = {"name": "y1_plv_prior_budget_verification_split", "exploratory": True}
    out["half_success"] = {}
    for r, ep, Y, _ in runs:
        late = ep >= np.median(ep)
        out["half_success"][r] = {"pool_half_all": round(Y[:, :25].mean(), 4), "heldout_half_all": round(Y[:, 25:].mean(), 4),
                                  "pool_half_late": round(Y[late, :25].mean(), 4), "heldout_half_late": round(Y[late, 25:].mean(), 4)}
    o_last, o_pb = regrets(runs, np.arange(25), np.arange(25, 50))
    rng = np.random.default_rng(5)
    ref = []
    for _ in range(5000):
        pm = rng.permutation(50)
        ref.append(regrets(runs, pm[:25], pm[25:])[1].mean())
    ref = np.array(ref)
    sw = regrets(runs, np.arange(25, 50), np.arange(25))[1].mean()
    out["reference_pool_best_25N"] = {"original": round(float(o_pb.mean()), 4), "original_per_run": o_pb.round(4).tolist(),
                                      "resplit_mean": round(float(ref.mean()), 4), "resplit_median": round(float(np.median(ref)), 4),
                                      "resplit_p2.5_p97.5": np.percentile(ref, [2.5, 97.5]).round(4).tolist(),
                                      "percentile_of_original": round(float((ref <= o_pb.mean() + 1e-12).mean()), 4),
                                      "swapped_halves": round(float(sw), 4),
                                      "note": "pool-best with ties averaged = expected regret of y1's perfect/no-prior/uniform 25N pick"}
    rng = np.random.default_rng(11)
    L, P = [], []
    for _ in range(2000):
        pm = rng.permutation(50)
        a, b = regrets(runs, pm[:25], pm[25:])
        L.append(a)
        P.append(b)
    L, P = np.array(L), np.array(P)
    out["zero_rollout_latest_epoch_vs_pool_best"] = {
        "original_last_per_run": o_last.round(4).tolist(), "original_last_mean": round(float(o_last.mean()), 4),
        "resplit_last_mean": round(float(L.mean()), 4), "resplit_last_p2.5_p97.5": np.percentile(L.mean(1), [2.5, 97.5]).round(4).tolist(),
        "resplit_last_per_run_mean": L.mean(0).round(4).tolist(),
        "resplit_last_per_run_share_zero": (L == 0).mean(0).round(3).tolist(),
        "resplit_poolbest_mean": round(float(P.mean()), 4),
        "resplit_diff_last_minus_poolbest_mean": round(float((L - P).mean(1).mean()), 4),
        "resplit_diff_p2.5_p97.5": np.percentile((L - P).mean(1), [2.5, 97.5]).round(4).tolist(),
        "share_last_better": round(float(((L - P).mean(1) < 0).mean()), 4),
        "share_last_worse": round(float(((L - P).mean(1) > 0).mean()), 4),
        "runs": [r for r, *_ in runs]}
    sp = {}
    for r, ep, Y, g in runs:
        re_, rs = rankdata(g.epoch), rankdata(g.success)
        d = {}
        for c in COLS:
            rho = float(spearmanr(-g[c].values, g.epoch.values)[0])
            rm = rankdata(-g[c].values)
            if abs(rho) > 0.999:
                part = None  # rank-identical to epoch: partial correlation undefined
            else:
                res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)  # noqa: E731
                part = round(float(np.corrcoef(res(rm, re_), res(rs, re_))[0, 1]), 3)
            d[c] = {"spearman_oriented_vs_epoch": round(rho, 3), "rank_partial_vs_success_given_epoch": part}
        sp[r] = d
    out["plv_vs_epoch"] = sp
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
