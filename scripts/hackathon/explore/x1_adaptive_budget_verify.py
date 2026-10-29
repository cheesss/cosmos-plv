#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- adversarial checks of x1_adaptive_budget.

Reuses x1's own loader and simulator (imported, unmodified) and adds:
  A. judge confusion recount (current vs .orig labels), cascade identity, checkpoint clustering
     of false positives (cluster bootstrap of FPR), Nebius-call share of the cascade;
  B. zero-rollout baselines (pick the latest epoch; pick by each prior alone) on the primary split
     and on the SAME 30 random 25/25 re-splits x1 used (same split RNG);
  C. threshold / grid sensitivity of the "budget to excess <= 0.02" claims, from x1's JSON;
  D. per-run (leave-one-run-out) budget-to-threshold;
  E. data-level uncertainty: seed bootstrap (pool seeds resampled within the pool half, held-out
     seeds within the held-out half) re-running x1's simulate_split with a perfect judge;
  F. decomposition of the 25N "noise floor" under a binomial model (held-out noise vs pool noise);
  G. cross-checkpoint seed-outcome correlation (x1 quoted 0.05-0.09 without a file);
  H. val_loss argmin epoch per run.
Writes results/hackathon/explore/x1_adaptive_budget_verification.json.

  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/x1_adaptive_budget_verify.py [--boot 100] [--reps 300]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x1_adaptive_budget as m  # noqa: E402

SCR = ROOT / "data/outputs/hackathon_explore/x1_adaptive_budget_verification"
X1_JSON = ROOT / "results/hackathon/explore/x1_adaptive_budget.json"
OUT = ROOT / "results/hackathon/explore/x1_adaptive_budget_verification.json"
COS = ROOT / "data/outputs/hackathon_cosmos"


def wilson(k, n):
    return list(m.wilson(k, n))


# ------------------------------------------------------------------ A. judges
def judge_audit():
    out = {}
    for tag, suf in (("current", ".jsonl"), ("orig_at_x1_run", ".orig.jsonl")):
        rows = {}
        for name, stem in (("local", "eval_local_v6"), ("nebius", "eval_nebius_v2")):
            d = {}
            for line in (COS / (stem + suf)).read_text().splitlines():
                r = json.loads(line)
                if r.get("scored") and r.get("parse_status") == "ok" and isinstance(r.get("cosmos_task_success"), bool):
                    d[(r["checkpoint_id"], int(r["env_seed"]))] = r
            rows[name] = d
        L, N = rows["local"], rows["nebius"]
        keys = sorted(set(L) & set(N))
        res = {"n_local": len(L), "n_nebius": len(N), "n_both": len(keys)}
        judges = {"local": lambda k: L[k]["cosmos_task_success"],
                  "nebius": lambda k: N[k]["cosmos_task_success"],
                  "cascade": lambda k: L[k]["cosmos_task_success"] and N[k]["cosmos_task_success"]}
        for jn, f in judges.items():
            fails = [k for k in keys if not L[k]["truth_success"]]
            succ = [k for k in keys if L[k]["truth_success"]]
            fp = [k for k in fails if f(k)]
            fn = [k for k in succ if not f(k)]
            dropped = [k for k in fails if L[k]["truth_failure_type"] == "dropped"]
            other = [k for k in fails if L[k]["truth_failure_type"] != "dropped"]
            # checkpoint-cluster bootstrap of FPR (failures are clustered in 15 checkpoints)
            ck = sorted({k[0] for k in fails})
            per = {c: (sum(1 for k in fails if k[0] == c), sum(1 for k in fp if k[0] == c)) for c in ck}
            rng = np.random.default_rng(1)
            bs = []
            for _ in range(4000):
                s = rng.choice(ck, len(ck))
                nf = sum(per[c][0] for c in s)
                bs.append(sum(per[c][1] for c in s) / nf if nf else np.nan)
            res[jn] = {"fp": len(fp), "n_fail": len(fails), "fn": len(fn), "n_succ": len(succ),
                       "fpr_wilson95": wilson(len(fp), len(fails)),
                       "fpr_checkpoint_cluster_boot95": [float(x) for x in np.nanpercentile(bs, [2.5, 97.5])],
                       "fp_by_checkpoint": {c: f"{v[1]}/{v[0]}" for c, v in per.items() if v[1]},
                       "fp_dropped": f"{sum(1 for k in dropped if f(k))}/{len(dropped)}",
                       "fp_not_dropped": f"{sum(1 for k in other if f(k))}/{len(other)}",
                       "fn_keys": [list(k) for k in fn]}
        res["cascade_nebius_calls"] = sum(1 for k in keys if L[k]["cosmos_task_success"])
        res["local_fn_subset_of_nebius_fn"] = set(map(tuple, res["local"]["fn_keys"])) <= set(map(tuple, res["nebius"]["fn_keys"]))
        res["cascade_fp_equals_nebius_fp"] = res["cascade"]["fp"] == res["nebius"]["fp"]
        out[tag] = res
    return out


def effective_fpr(bf, ds):
    q = ds["hi"]["dropped"] / ds["hi"]["failures"]
    fd = bf["dropped"]["fp"] / bf["dropped"]["n_fail"]
    fo = bf["not_dropped"]["fp"] / bf["not_dropped"]["n_fail"]
    return q * fd + (1 - q) * fo


# ------------------------------------------------------------------ B. zero-rollout baselines
def zero_rollout(runs, n_splits=30, seed=20261029):
    n_seeds = len(runs[0]["seeds"])
    half = n_seeds // 2
    res = {}
    def regrets(pi, hi):
        out = {}
        for r in runs:
            H = r["Y"][:, hi].mean(1)
            Pm = r["Y"][:, pi].mean(1)
            latest = int(np.argmax(r["epochs"]))
            d = {"latest_epoch_pick": float(H.max() - H[latest]),
                 "full_pool_uniform_25N": float(H.max() - H[Pm == Pm.max()].mean()),
                 "latest_3_pool_best_25N_each": None}
            last3 = np.argsort(-r["epochs"])[:3]
            sub = Pm[last3]
            d["latest_3_pool_best_25N_each"] = float(H.max() - H[last3[sub == sub.max()]].mean())
            for p in ("val_loss", "train_action_mse"):
                col, lower = m.PRIORS[p]
                v = np.asarray(r["metrics"][col], float)
                key = np.where(np.isfinite(v), v if lower else -v, np.inf)
                i = sorted(range(len(v)), key=lambda i: (key[i], -r["epochs"][i]))[0]
                d[f"{p}_only_pick"] = float(H.max() - H[i])
            out[r["run"]] = d
        mean = {k: float(np.mean([out[r][k] for r in out])) for k in next(iter(out.values()))}
        return out, mean
    per, mean = regrets(np.arange(half), np.arange(half, n_seeds))
    res["primary"] = {"per_run": per, "mean_over_runs": mean}
    split_rng = np.random.default_rng(seed + 99)  # identical to x1's re-split RNG
    acc = []
    for _ in range(n_splits):
        perm = split_rng.permutation(n_seeds)
        _, mm = regrets(np.sort(perm[:half]), np.sort(perm[half:]))
        acc.append(mm)
    res["random_splits"] = {}
    rng = np.random.default_rng(5)
    for k in acc[0]:
        v = np.array([a[k] for a in acc])
        bm = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        res["random_splits"][k] = {"mean": float(v.mean()), "boot95": [float(x) for x in np.percentile(bm, [2.5, 97.5])]}
    return res


# ------------------------------------------------------------------ C/D. threshold sensitivity
def crossing(rows, judge, st, key, thr, per_run=None):
    rs = sorted([x for x in rows if x["judge"] == judge and x["strategy"] == st], key=lambda x: x["budget_mult"])
    prev = None
    for x in rs:
        if per_run is None:
            v = x[key]
        else:
            p = [q for q in x["per_run"] if q["run"] == per_run][0]
            v = p["excess_vs_full_pool"] if key == "mean_excess_vs_full_pool" else p["mean_regret"]
        if v <= thr + 1e-12:
            if prev is None:
                return x["budget_mult"], x["budget_mult"]
            (m0, v0) = prev
            # log-linear interpolation in budget between grid points
            f = (v0 - thr) / (v0 - v) if v0 != v else 1.0
            return x["budget_mult"], float(math.exp(math.log(m0) + f * (math.log(x["budget_mult"]) - math.log(m0))))
        prev = (x["budget_mult"], v)
    return None, None


def threshold_sensitivity(d):
    rows = d["primary"]
    out = {}
    strats = ["uniform", "successive_halving", "ucb1", "thompson", "top_two_thompson"]
    for j in ["perfect", "nemotron_nebius_v2", "cosmos3_edge_local_v6", "cosmos3_edge_local_v6_ftype", "nemotron_nebius_v2_ftype"]:
        out[j] = {}
        for thr in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03):
            rec = {}
            for st in strats:
                g, itp = crossing(rows, j, st, "mean_excess_vs_full_pool", thr)
                rec[st] = {"grid": g, "interp": itp}
            u, s = rec["uniform"]["interp"], rec["successive_halving"]["interp"]
            t = rec["top_two_thompson"]["interp"]
            rec["ratio_uniform_over_SH"] = None if not (u and s) else u / s
            rec["ratio_uniform_over_TTTS"] = None if not (u and t) else u / t
            out[j][str(thr)] = rec
    per_run = {}
    for r in [c["run"] for c in d["context"]]:
        per_run[r] = {st: crossing(rows, "perfect", st, "mean_excess_vs_full_pool", 0.02, per_run=r)[0] for st in strats}
    # leave-one-run-out mean excess crossing
    lo = {}
    runs = [c["run"] for c in d["context"]]
    for leave in runs:
        keep = [r for r in runs if r != leave]
        rec = {}
        for st in strats:
            rs = sorted([x for x in rows if x["judge"] == "perfect" and x["strategy"] == st], key=lambda x: x["budget_mult"])
            hit = None
            for x in rs:
                v = np.mean([p["excess_vs_full_pool"] for p in x["per_run"] if p["run"] in keep])
                if v <= 0.02 + 1e-12:
                    hit = x["budget_mult"]
                    break
            rec[st] = hit
        lo[f"without_{leave[9:]}"] = rec
    # raw regret values at a few budgets (perfect) for context
    val = {}
    for st in strats + ["late_epoch_top30+top_two_thompson", "late_epoch_top30+successive_halving", "late_epoch_top30+uniform"]:
        val[st] = {str(x["budget_mult"]): round(x["mean_regret"], 4) for x in rows
                   if x["judge"] == "perfect" and x["strategy"] == st and x["budget_mult"] in (0.25, 0.5, 1, 3, 4, 6, 8, 12, 25)}
    return {"excess_crossing_by_threshold": out, "per_run_excess_le_0.02_perfect": per_run,
            "leave_one_run_out_excess_le_0.02_perfect": lo, "perfect_mean_regret_values": val}


# ------------------------------------------------------------------ E. seed bootstrap
def seed_bootstrap(runs, B, reps, seed=7):
    strategies = ["uniform", "successive_halving", "thompson", "top_two_thompson",
                  "late_epoch_top30+top_two_thompson", "random_top30+top_two_thompson"]
    mults = (1, 2, 3, 4, 6, 8, 12, 16, 25)
    rng = np.random.default_rng(seed)
    half = len(runs[0]["seeds"]) // 2
    curves = {st: [] for st in strategies}
    floors = []
    for b in range(B):
        pool_cols = rng.integers(0, half, half)
        ho_cols = half + rng.integers(0, half, half)
        per_st = {st: [] for st in strategies}
        fl = []
        for ri, r in enumerate(runs):
            rr = dict(r)
            rr["Y"] = r["Y"][:, np.concatenate([pool_cols, ho_cols])]
            res, _, _, H = m.simulate_split(rr, np.arange(half), np.arange(half, 2 * half), ["perfect"], None, reps,
                                            seed=10_000 + 31 * b + ri, strategies=strategies)
            Pm = rr["Y"][:, :half].mean(1)
            f = float(H.max() - H[Pm == Pm.max()].mean())
            fl.append(f)
            N = len(r["epochs"])
            for st in strategies:
                per_st[st].append([float(res[("perfect", st)][max(1, int(round(mm * N)))].mean()) for mm in mults])
        floors.append(float(np.mean(fl)))
        for st in strategies:
            curves[st].append(np.mean(per_st[st], axis=0))
    floors = np.array(floors)
    out = {"B": B, "reps": reps, "mults": list(mults),
           "floor_mean": float(floors.mean()), "floor_2.5_97.5": [float(x) for x in np.percentile(floors, [2.5, 97.5])]}
    cross = {}
    for st in strategies:
        c = np.array(curves[st])  # [B, len(mults)]
        out[st] = {"mean": c.mean(0).round(4).tolist(),
                   "p2.5": np.percentile(c, 2.5, axis=0).round(4).tolist(),
                   "p97.5": np.percentile(c, 97.5, axis=0).round(4).tolist()}
        ex = c - floors[:, None]
        hit = []
        for bi in range(B):
            k = np.flatnonzero(ex[bi] <= 0.02 + 1e-12)
            hit.append(mults[k[0]] if len(k) else 50.0)  # 50 = not reached within 25N
        cross[st] = np.array(hit, float)
    for st in strategies:
        h = cross[st]
        out[st]["excess_le_0.02_crossing_mult"] = {
            "median": float(np.median(h)), "p10": float(np.percentile(h, 10)), "p90": float(np.percentile(h, 90)),
            "share_le_4N": float((h <= 4).mean()), "share_ge_12N": float((h >= 12).mean())}
    ratio = cross["uniform"] / cross["successive_halving"]
    ratio_t = cross["uniform"] / cross["top_two_thompson"]
    out["ratio_uniform_over_SH_crossing"] = {"median": float(np.median(ratio)), "p10": float(np.percentile(ratio, 10)),
                                             "p90": float(np.percentile(ratio, 90))}
    out["ratio_uniform_over_TTTS_crossing"] = {"median": float(np.median(ratio_t)), "p10": float(np.percentile(ratio_t, 10)),
                                               "p90": float(np.percentile(ratio_t, 90))}
    u = np.array(curves["uniform"])
    for other in ("successive_halving", "top_two_thompson", "late_epoch_top30+top_two_thompson"):
        dlt = u - np.array(curves[other])
        out[f"uniform_minus_{other}"] = {"mean": dlt.mean(0).round(4).tolist(),
                                         "p2.5": np.percentile(dlt, 2.5, axis=0).round(4).tolist(),
                                         "p97.5": np.percentile(dlt, 97.5, axis=0).round(4).tolist()}
    lt = np.array(curves["late_epoch_top30+top_two_thompson"])
    out["late_epoch_TTTS_le_0.02_share_by_mult"] = (lt <= 0.02 + 1e-12).mean(0).round(3).tolist()
    return out


# ------------------------------------------------------------------ F. floor decomposition
def floor_decomposition(runs, sims=20000, seed=3):
    rng = np.random.default_rng(seed)
    out = {}
    for model in ("p=50ep_mean", "p=posterior_mean_(k+1)/(n+2)"):
        per = {}
        for r in runs:
            k = r["Y"].sum(1)
            n = r["Y"].shape[1]
            p = k / n if model.startswith("p=50") else (k + 1) / (n + 2)
            Hs = rng.binomial(25, p[None, :], size=(sims, len(p))) / 25
            Ps = rng.binomial(25, p[None, :], size=(sims, len(p))) / 25
            best = np.flatnonzero(p == p.max())
            # oracle knows true p (pick uniformly among true-best ties)
            ob = rng.choice(best, sims)
            reg_oracle = Hs.max(1) - Hs[np.arange(sims), ob]
            # pool argmax (25 eps), random tie-break
            jit = rng.random(Ps.shape) * 1e-9
            pk = (Ps + jit).argmax(1)
            reg_pool = Hs.max(1) - Hs[np.arange(sims), pk]
            true_reg = p.max() - p[pk]
            per[r["run"]] = {"heldout_measured_regret_oracle_pick": float(reg_oracle.mean()),
                             "heldout_measured_regret_pool25_pick": float(reg_pool.mean()),
                             "true_regret_pool25_pick": float(true_reg.mean())}
        per["mean_over_runs"] = {kk: float(np.mean([per[r["run"]][kk] for r in runs])) for kk in next(iter(per.values()))}
        out[model] = per
    return out


# ------------------------------------------------------------------ G. seed correlation
def seed_correlation(runs, half=25):
    out = {}
    for r in runs:
        Y = r["Y"][:, :half]
        keep = Y.std(1) > 0
        C = np.corrcoef(Y[keep])
        iu = np.triu_indices(C.shape[0], 1)
        # residual seed effect: corr of per-seed outcome after removing checkpoint means
        out[r["run"]] = {"n_nonconstant_ckpts": int(keep.sum()),
                         "mean_pairwise_corr_pool_seeds": float(np.nanmean(C[iu]))}
        Y2 = r["Y"]
        keep2 = Y2.std(1) > 0
        C2 = np.corrcoef(Y2[keep2])
        iu2 = np.triu_indices(C2.shape[0], 1)
        out[r["run"]]["mean_pairwise_corr_all50"] = float(np.nanmean(C2[iu2]))
    return out


def val_loss_shape(runs):
    out = {}
    for r in runs:
        v = np.asarray(r["metrics"]["val_loss"], float)
        i = int(np.nanargmin(v))
        out[r["run"]] = {"argmin_epoch": int(r["epochs"][i]), "min": float(v[i]), "last_epoch": int(r["epochs"][-1]),
                         "last": float(v[-1])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    d = json.loads(X1_JSON.read_text())
    frozen = m.load_runs(SCR / "frozen_pool")
    current = m.load_runs(ROOT / "data/outputs/hackathon_pool")
    res = {"name": "x1_adaptive_budget_verification", "exploratory": True,
           "command": " ".join([sys.executable, *sys.argv]),
           "snapshots": {"frozen_N": {r["run"]: len(r["epochs"]) for r in frozen},
                         "current_N": {r["run"]: len(r["epochs"]) for r in current},
                         "current_max_epoch": {r["run"]: int(r["epochs"][-1]) for r in current}}}
    res["A_judges"] = judge_audit()
    bf_cur = {}
    for tag in ("current", "orig_at_x1_run"):
        for jn in ("local", "nebius"):
            fd = res["A_judges"][tag][jn]["fp_dropped"].split("/")
            fo = res["A_judges"][tag][jn]["fp_not_dropped"].split("/")
            bf_cur[(tag, jn)] = {"dropped": {"fp": int(fd[0]), "n_fail": int(fd[1])},
                                 "not_dropped": {"fp": int(fo[0]), "n_fail": int(fo[1])}}
    ds_orig = d["dropped_share_eval_rollouts"]
    ds_cur = m.dropped_share_from_eval(ROOT / "data/outputs/hackathon_rollouts_eval")
    res["A_effective_fpr_hi_checkpoints"] = {
        "orig_labels": {jn: effective_fpr(bf_cur[("orig_at_x1_run", jn)], ds_orig) for jn in ("local", "nebius")},
        "current_labels": {jn: effective_fpr(bf_cur[("current", jn)], ds_cur) for jn in ("local", "nebius")},
        "dropped_share_orig": ds_orig, "dropped_share_current": ds_cur}
    print(f"[{time.time() - t0:.0f}s] A done", flush=True)
    res["B_zero_rollout_frozen"] = zero_rollout(frozen)
    res["B_zero_rollout_current"] = zero_rollout(current)
    print(f"[{time.time() - t0:.0f}s] B done", flush=True)
    res["C_D_thresholds_from_x1_json"] = threshold_sensitivity(d)
    print(f"[{time.time() - t0:.0f}s] C/D done", flush=True)
    res["F_floor_decomposition_frozen"] = floor_decomposition(frozen)
    res["G_seed_correlation_frozen"] = seed_correlation(frozen)
    res["H_val_loss_shape_frozen"] = val_loss_shape(frozen)
    res["H_val_loss_shape_current"] = val_loss_shape(current)
    print(f"[{time.time() - t0:.0f}s] F/G/H done", flush=True)
    res["E_seed_bootstrap_frozen"] = seed_bootstrap(frozen, a.boot, a.reps)
    print(f"[{time.time() - t0:.0f}s] E done", flush=True)
    res["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
