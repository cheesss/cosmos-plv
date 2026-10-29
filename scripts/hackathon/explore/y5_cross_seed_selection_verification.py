#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- adversarial verification of y5_cross_seed_selection.

Independent re-derivation (own parsers, own statistics) of the y5 claims, plus
sensitivity checks the original did not run:
  * pair table / counts from raw logs.json.txt (not the y5 loader)
  * pooled Spearman under both tie conventions (expected-over-random-tie-breaks,
    as in y5, and the midrank convention of cosmos_plv.ranking / Experiment 1)
  * pairwise decisions recomputed; Holm over 12 columns and over 10 distinct
    metrics; the replan "worse than chance" claim under Holm
  * VLM decisions at the two eval-epoch primary pairs (ties?)
  * judge-file agreement on Can (v6 vs v7, v2 vs pad201)
  * random-split (not just HA/HB) split-half reliability distribution
  * env seed 100003 analysis restricted to common10 (removes the x5-epoch
    over-representation of seed1), eval-seed direction, and failure clustering
  * edge effect of the non-circular moving-block bootstrap + key Delta-vs-epoch
    CIs recomputed with (a) circular blocks and (b) episode-only resampling
  * paired judge-vs-instrumented K=10 Spearman CI (is "K, not the judge" supported?)
  * h10 == h16 root cause from the raw chunk.csv files
CPU only.  Writes results/hackathon/explore/y5_cross_seed_selection_verification.json

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y5_cross_seed_selection_verification.py
"""
from __future__ import annotations

import glob
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/hackathon/explore/y5_cross_seed_selection_verification.json"
RUNS = {
    "can_seed1": ROOT / "data/outputs/hackathon_pool/paper_dp_can_abs_ph_seed1_e200_s200_r5_inf100",
    "can_seed2": ROOT / "data/outputs/hackathon_pool/paper_dp_can_abs_ph_seed2_e200_s200_r10_inf100",
}
TS = list(range(100000, 100050))
ES = list(range(200000, 200010))
JUDGES = {
    "v6": "data/outputs/hackathon_cosmos/eval_local_v6.jsonl",
    "v7": "data/outputs/hackathon_cosmos/eval_local_v7.jsonl",
    "v2": "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl",
    "pad201": "data/outputs/hackathon_cosmos/eval_nebius_v2_pad201.jsonl",
}
CHEAP = ["val_action_l1", "open_loop_smoothness", "plv1_smoothness", "replan_overlap_h16", "replan_overlap_h10",
         "boundary_jump_h16", "boundary_jump_h10", "val_loss", "train_loss", "support_drift",
         "dynamics_uncertainty", "teacher_forced_latent_mse"]
DISTINCT = [c for c in CHEAP if not c.endswith("_h10")]


# ------------------------------------------------------------------ independent loaders
def parse_logs(run_dir):
    by = {}
    with open(run_dir / "logs.json.txt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = f"test/sim_max_reward_{TS[0]}"
            if k in r and "epoch" in r:
                by.setdefault(int(r["epoch"]), {}).update(r)
    return {e: np.array([float(r[f"test/sim_max_reward_{s}"]) > 0.5 for s in TS], float) for e, r in by.items()}


def load():
    base = pd.read_csv(ROOT / "results/hackathon/baseline_metrics.csv")
    ck = base[base.run_id.isin(RUNS)].sort_values(["run_id", "epoch"]).reset_index(drop=True)
    Y = np.zeros((len(ck), 50))
    logs = {r: parse_logs(d) for r, d in RUNS.items()}
    for i, row in ck.iterrows():
        Y[i] = logs[row.run_id][int(row.epoch)]
    assert np.abs(Y.mean(1) - ck.success.values).max() < 1e-9
    E = np.full((len(ck), 10), np.nan)
    for i, row in ck.iterrows():
        p = ROOT / "data/outputs/hackathon_rollouts_eval" / row.checkpoint_id / "summary.json"
        if p.is_file():
            for ep in json.loads(p.read_text())["episodes"]:
                E[i, ES.index(int(ep["env_seed"]))] = float(bool(ep["success"]))
    pos = {c: i for i, c in enumerate(ck.checkpoint_id)}
    V = {}
    for j, f in JUDGES.items():
        V[j] = np.full((len(ck), 10), np.nan)
        for line in open(ROOT / f):
            r = json.loads(line)
            if r["checkpoint_id"] in pos:
                V[j][pos[r["checkpoint_id"]], ES.index(int(r["env_seed"]))] = float(bool(r["cosmos_task_success"]))
    return ck, Y, E, V


# ------------------------------------------------------------------ statistics
def spearman_mid(x, y):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def spearman_exp(x, y):
    """Expected Spearman over uniform random tie-breaks of x (y midranked)."""
    n = len(x)
    rx, ry = stats.rankdata(x) - (n + 1) / 2, stats.rankdata(y) - (n + 1) / 2
    dy = (ry * ry).sum()
    return float((rx * ry).sum() / math.sqrt(n * (n * n - 1) / 12 * dy)) if dy > 0 else np.nan


def holm(pvals: dict):
    order = sorted(pvals, key=lambda m: pvals[m])
    out, run = {}, 0.0
    for r, m in enumerate(order):
        run = max(run, min(1.0, (len(order) - r) * pvals[m]))
        out[m] = run
    return out


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [np.nan, np.nan]


def main():
    t0 = time.time()
    ck, Y, E, V = load()
    res = {"label": "EXPLORATORY / post-hoc adversarial verification of y5_cross_seed_selection"}
    s1 = ck[ck.run_id == "can_seed1"].set_index("epoch")
    s2 = ck[ck.run_id == "can_seed2"].set_index("epoch")
    common = sorted(set(s1.index) & set(s2.index))
    idx1 = {e: int(ck.index[(ck.run_id == "can_seed1") & (ck.epoch == e)][0]) for e in s1.index}
    idx2 = {e: int(ck.index[(ck.run_id == "can_seed2") & (ck.epoch == e)][0]) for e in s2.index}
    A, B = np.arange(25), np.arange(25, 50)

    # ---- 1. pair table
    pairs = []
    for e in common:
        i1, i2 = idx1[e], idx2[e]
        pairs.append({"epoch": int(e), "S1": Y[i1].mean(), "S2": Y[i2].mean(), "d": Y[i2].mean() - Y[i1].mean(),
                      "dA": Y[i2][A].mean() - Y[i1][A].mean(), "dB": Y[i2][B].mean() - Y[i1][B].mean()})
    big = [p["epoch"] for p in pairs if abs(p["d"]) >= 0.06 - 1e-9]
    anyd = [p["epoch"] for p in pairs if abs(p["d"]) >= 0.02 - 1e-9]
    res["pairs"] = {"n_common": len(pairs), "primary_epochs": big, "any_difference_epochs": anyd,
                    "max_abs_d_from_e60": float(max(abs(p["d"]) for p in pairs if p["epoch"] >= 60)),
                    "n_seed2_better_any": int(sum(p["d"] > 0 for p in pairs if p["epoch"] in anyd)),
                    "final": {"seed1": float(Y[idx1[200]].mean()), "seed2": float(Y[idx2[200]].mean())}}

    # ---- 2. pairwise decisions (own implementation)
    def decide(c, e):
        a, b = float(ck.loc[idx1[e], c]), float(ck.loc[idx2[e], c])
        return 0.5 if a == b else float(b < a)   # 1 = picks seed2

    pw = {}
    for name, eps in (("primary", big), ("any", anyd)):
        truth = {e: float(next(p["d"] for p in pairs if p["epoch"] == e) > 0) for e in eps}
        pm = {}
        for c in CHEAP:
            dec = np.array([decide(c, e) for e in eps])
            tr = np.array([truth[e] for e in eps])
            k = int(((dec == tr) & (dec != 0.5)).sum())
            n = int((dec != 0.5).sum())
            pm[c] = {"correct": k, "n": n, "binom_p": float(stats.binomtest(k, n, 0.5).pvalue) if n else np.nan,
                     "picks": {str(e): ("s2" if d == 1 else "s1" if d == 0 else "tie") for e, d in zip(eps, dec)}}
        pw[name] = pm
    pw["holm_12_columns_any"] = holm({c: pw["any"][c]["binom_p"] for c in CHEAP})
    pw["holm_10_distinct_any"] = holm({c: pw["any"][c]["binom_p"] for c in DISTINCT})
    # one-sided "worse than chance" for replan
    k = pw["any"]["replan_overlap_h16"]["correct"]
    pw["replan_one_sided_p_worse"] = float(stats.binomtest(k, 14, 0.5, alternative="less").pvalue)
    # TF latent split early/late
    tf = pw["any"]["teacher_forced_latent_mse"]["picks"]
    tr_any = {str(p["epoch"]): ("s2" if p["d"] > 0 else "s1") for p in pairs if p["epoch"] in anyd}
    pw["tf_latent_any_early_le60"] = {"correct": int(sum(tf[e] == tr_any[e] for e in tf if int(e) <= 60)),
                                      "n": int(sum(1 for e in tf if int(e) <= 60))}
    pw["tf_latent_any_late_ge80"] = {"correct": int(sum(tf[e] == tr_any[e] for e in tf if int(e) >= 80)),
                                     "n": int(sum(1 for e in tf if int(e) >= 80)),
                                     "always_seed2_correct": int(sum(tr_any[e] == "s2" for e in tf if int(e) >= 80))}
    rp = pw["any"]["replan_overlap_h16"]["picks"]
    pw["replan_any_late_ge80"] = {"correct": int(sum(rp[e] == tr_any[e] for e in rp if int(e) >= 80)),
                                  "n": int(sum(1 for e in rp if int(e) >= 80)),
                                  "picks_seed1_late": int(sum(rp[e] == "s1" for e in rp if int(e) >= 80))}
    # K=10 / VLM at eval-epoch primary pairs
    k10 = {}
    for e in [x for x in big if x % 20 == 0]:
        i1, i2 = idx1[e], idx2[e]
        k10[str(e)] = {"S50": [float(Y[i1].mean()), float(Y[i2].mean())],
                       "instrumented": [float(E[i1].mean()), float(E[i2].mean())],
                       **{j: [float(V[j][i1].mean()), float(V[j][i2].mean())] for j in JUDGES}}
    pw["k10_primary_eval_pairs"] = k10
    res["pairwise"] = pw

    # ---- 3. judge agreement on Can
    ev = np.isfinite(E).all(1)
    res["judges_can"] = {
        "n_episodes": int(ev.sum() * 10),
        "v6_vs_v7_n_differ": int((V["v6"][ev] != V["v7"][ev]).sum()),
        "v2_vs_pad201_n_differ": int((V["v2"][ev] != V["pad201"][ev]).sum()),
        "acc_vs_instrumented": {j: float((V[j][ev] == E[ev]).mean()) for j in JUDGES}}

    # ---- 4. pooled Spearman under both tie conventions
    def pool(sel):
        return np.nonzero(sel)[0]
    ep = ck.epoch.values
    pools = {"common10": pool(ep % 10 == 0), "eval20": pool(ep % 20 == 0), "late_common10": pool((ep % 10 == 0) & (ep >= 100))}
    conv = {}
    for pn, ix in pools.items():
        y = Y[ix].mean(1)
        sc = {"epoch": ep[ix].astype(float)}
        for c in ["val_action_l1", "open_loop_smoothness", "replan_overlap_h16", "plv1_smoothness", "teacher_forced_latent_mse"]:
            sc[c] = -ck[c].values[ix]
        if pn == "eval20":
            sc["k10_instrumented"] = E[ix].mean(1)
            for j in JUDGES:
                sc[f"k10_{j}"] = V[j][ix].mean(1)
        conv[pn] = {m: {"expected_random_tiebreak": spearman_exp(s, y), "midrank": spearman_mid(s, y)} for m, s in sc.items()}
    res["pooled_spearman_tie_conventions"] = conv

    # ---- 5. split-half reliability: fixed HA/HB vs random 25/25 splits
    rng = np.random.default_rng(11)
    rel = {}
    for pn, ix in pools.items():
        fixed = spearman_mid(Y[ix][:, A].mean(1), Y[ix][:, B].mean(1))
        rs = []
        for _ in range(2000):
            perm = rng.permutation(50)
            rs.append(spearman_mid(Y[ix][:, perm[:25]].mean(1), Y[ix][:, perm[25:]].mean(1)))
        rel[pn] = {"fixed_HA_HB": fixed, "random_split_mean": float(np.nanmean(rs)),
                   "random_split_2.5_97.5": ci(rs), "frac_random_splits_below_fixed": float(np.nanmean(np.array(rs) < fixed))}
    # late pool without env seed 100003
    ix = pools["late_common10"]
    keep = np.array([i for i in range(50) if TS[i] != 100003])
    rs = []
    for _ in range(2000):
        perm = rng.permutation(keep)
        rs.append(spearman_mid(Y[ix][:, perm[:24]].mean(1), Y[ix][:, perm[24:]].mean(1)))
    rel["late_common10_without_100003_random_split_mean"] = float(np.nanmean(rs))
    res["split_half_reliability"] = rel

    # ---- 6. env seed 100003
    j3 = TS.index(100003)
    out3 = {}
    for name, sel in (("all_late_ge100", ep >= 100), ("common10_late_ge100", (ep >= 100) & (ep % 10 == 0))):
        r1 = np.nonzero(sel & (ck.run_id == "can_seed1").values)[0]
        r2 = np.nonzero(sel & (ck.run_id == "can_seed2").values)[0]
        k1, n1 = int((1 - Y[r1, j3]).sum()), len(r1)
        k2, n2 = int((1 - Y[r2, j3]).sum()), len(r2)
        tot1, tot2 = int((1 - Y[r1]).sum()), int((1 - Y[r2]).sum())
        out3[name] = {"seed1": f"{k1}/{n1}", "seed2": f"{k2}/{n2}",
                      "fisher_p": float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]),
                      "seed1_fail_sequence_by_epoch": {int(ep[i]): int(1 - Y[i, j3]) for i in r1},
                      "seed2_fail_sequence_by_epoch": {int(ep[i]): int(1 - Y[i, j3]) for i in r2},
                      "late_failures_total": {"seed1": tot1, "seed2": tot2},
                      "late_failures_excluding_100003": {"seed1": tot1 - k1, "seed2": tot2 - k2},
                      "n_episodes_per_seed": {"seed1": n1 * 50, "seed2": n2 * 50}}
    # contiguous failure runs (dependence): number of maximal runs of consecutive failing checkpoints for seed1
    seq = list(out3["all_late_ge100"]["seed1_fail_sequence_by_epoch"].values())
    out3["seed1_all_late_n_failure_runs"] = int(sum(1 for i, v in enumerate(seq) if v == 1 and (i == 0 or seq[i - 1] == 0)))
    # eval seeds (disjoint start states): late failures
    late_ev = (ep >= 100) & ev
    out3["eval_seeds_late_failures"] = {r: f"{int((1 - E[late_ev & (ck.run_id == r).values]).sum())}/"
                                           f"{int(late_ev[(ck.run_id == r).values].sum() * 10)}" for r in RUNS}
    out3["bonferroni_50_seeds_all_late"] = min(1.0, 50 * out3["all_late_ge100"]["fisher_p"])
    res["env_seed_100003"] = out3

    # ---- 7. plateau numbers quoted in the report
    late = ck[(ck.epoch >= 100) & (ck.epoch % 10 == 0)]
    res["plateaus"] = {
        "replan_h16_e200": {r: float(ck.loc[(ck.run_id == r) & (ck.epoch == 200), "replan_overlap_h16"].iloc[0]) for r in RUNS},
        "tf_latent_late_mean": late.groupby("run_id").teacher_forced_latent_mse.mean().to_dict(),
        "late_mean_cost_lower_for_seed1": {c: bool(late.groupby("run_id")[c].mean()["can_seed1"] <
                                                   late.groupby("run_id")[c].mean()["can_seed2"]) for c in CHEAP},
    }

    # ---- 8. h10 == h16 root cause
    hz = {}
    for f in sorted(glob.glob(str(ROOT / "data/outputs/hackathon_metrics/*/chunk.csv"))):
        ch = pd.read_csv(f)
        a = ch[ch.chunk_action_horizon == 10].sort_values(["checkpoint_path", "chunk_sampling_seed"])
        b = ch[ch.chunk_action_horizon == 16].sort_values(["checkpoint_path", "chunk_sampling_seed"])
        hz[Path(f).parent.name] = {
            "execution_stride": sorted(set(ch.chunk_execution_stride)),
            "policy_calls_by_H": {int(h): sorted(set(g.chunk_policy_calls)) for h, g in ch.groupby("chunk_action_horizon")},
            "rowwise_equal_boundary_jump": float((a.boundary_jump_l1_norm.values == b.boundary_jump_l1_norm.values).mean()),
            "rowwise_equal_replan_overlap": float((a.replan_overlap_l1_norm.values == b.replan_overlap_l1_norm.values).mean()),
            "rowwise_equal_support_drift": float((a.chunk_support_drift.values == b.chunk_support_drift.values).mean()),
            "rowwise_equal_curvature": float((a.chunk_curvature_norm.values == b.chunk_curvature_norm.values).mean())}
    res["h10_h16"] = hz

    # ---- 9. bootstrap sensitivity (re-uses the y5 evaluate_pool for identical method definitions)
    spec = importlib.util.spec_from_file_location("y5", ROOT / "scripts/hackathon/explore/y5_cross_seed_selection.py")
    y5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(y5)
    ck5, Y5, E5, V5, _, _ = y5.load_data()
    assert np.array_equal(Y5, Y) and np.allclose(np.nan_to_num(E5, nan=-1), np.nan_to_num(E, nan=-1))
    # bin inclusion probabilities of the y5 (non-circular) moving-block scheme
    rngb = np.random.default_rng(5)
    ix = pools["common10"]
    bins = (ep[ix] // 10).astype(int)
    cnt = np.zeros(21)
    for _ in range(20000):
        sub = y5.block_bins(ep[ix], 10, 4, rngb)
        u, c = np.unique(bins[sub], return_counts=True)
        cnt[u] += c / 2
    res["bootstrap_bin_mean_copies_y5_scheme"] = {str(int(b * 10)): float(cnt[b] / 20000) for b in range(21)}

    def circ_bins(epochs, bin_w, block, rng_):
        bn = np.floor(epochs / bin_w).astype(int)
        ub = np.unique(bn)
        nb = len(ub)
        starts = rng_.integers(0, nb, size=int(math.ceil(nb / block)))
        drawn = np.concatenate([(np.arange(s, s + block) % nb) for s in starts])[:nb]
        mem = {b: np.nonzero(bn == ub[b])[0] for b in range(nb)}
        return np.concatenate([mem[b] for b in drawn])

    keys = {("common10", "HB"): ["k25_rollouts_fit_half", "val_action_l1", "rankavg_epoch+val_action_l1",
                                 "fit_half_best_cheap"],
            ("common10", "HA"): ["k25_rollouts_fit_half", "val_action_l1"],
            ("common10", "S50"): ["val_action_l1", "rankavg_epoch+val_action_l1", "lex_epoch>val_action_l1",
                                  "open_loop_smoothness"],
            ("eval20", "S50"): ["k10_sim", "vlm_nebius_v2", "vlm_local_v6", "val_action_l1"]}
    nb = 1000
    sens = {}
    for scheme in ("circular_block", "episode_only_fixed_pool"):
        acc = {}
        rr = np.random.default_rng(99)
        for b in range(nb):
            tr = {"A": rr.integers(0, 25, 25), "B": rr.integers(0, 25, 25)}
            evx = rr.integers(0, 10, 10)
            for pn in ("common10", "eval20"):
                ixp = pools[pn]
                if scheme == "circular_block":
                    cfg = y5.POOLS[pn]
                    sub = ixp[circ_bins(ep[ixp], cfg["bin_w"], cfg["block"], rr)]
                else:
                    sub = ixp
                r_, _ = y5.evaluate_pool(ck5, Y5, E5, V5, sub, tr, evx, pn == "eval20")
                for (p_, t_), ms in keys.items():
                    if p_ != pn:
                        continue
                    for m in ms + (["vlm_nebius_v2", "vlm_local_v6"] if pn == "eval20" else []):
                        if m not in r_[t_]:
                            continue
                        acc.setdefault(f"{p_}|{t_}|{m}", []).append(r_[t_][m]["spearman"] - r_[t_]["epoch"]["spearman"])
                    if pn == "eval20":
                        acc.setdefault("eval20|S50|nemotron_minus_instrumented", []).append(
                            r_["S50"]["vlm_nebius_v2"]["spearman"] - r_["S50"]["k10_sim"]["spearman"])
                        acc.setdefault("eval20|S50|cosmos_minus_instrumented", []).append(
                            r_["S50"]["vlm_local_v6"]["spearman"] - r_["S50"]["k10_sim"]["spearman"])
        sens[scheme] = {k: {"ci95": ci(v), "p_le_0": float(np.mean(np.array(v) <= 0))} for k, v in acc.items()}
    res["bootstrap_sensitivity_delta_spearman_vs_epoch"] = sens
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
