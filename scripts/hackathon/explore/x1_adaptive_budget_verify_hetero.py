#!/usr/bin/env python3
"""EXPLORATORY verification of x1_adaptive_budget: what if VLM-judge false positives are
correlated within a checkpoint (not iid per episode, as x1 assumes)?

1. Fit a beta-binomial (task-specific mean, shared concentration kappa) to the per-checkpoint
   false-positive counts among true failures in the eval JSONL (15 checkpoints x 10 episodes).
2. Re-run x1's simulate_split (imported, unmodified) on the frozen x1 snapshot with an extra judge
   family '<judge>_het<kappa>': per repetition and checkpoint, FPR_c ~ Beta(mu*kappa, (1-mu)*kappa),
   mu drawn from the same Beta posterior x1 uses.  Same FNR model as x1.
Writes results/hackathon/explore/x1_adaptive_budget_verification_hetero.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x1_adaptive_budget as m  # noqa: E402

SCR = ROOT / "data/outputs/hackathon_explore/x1_adaptive_budget_verification"
OUT = ROOT / "results/hackathon/explore/x1_adaptive_budget_verification_hetero.json"
COS = ROOT / "data/outputs/hackathon_cosmos"
MULTS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 25)


def per_checkpoint(stem):
    d = {}
    for line in (COS / f"{stem}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not (r.get("scored") and r.get("parse_status") == "ok") or r["truth_success"]:
            continue
        c = d.setdefault(r["checkpoint_id"], [r["task"], 0, 0])
        c[1] += 1
        c[2] += int(bool(r["cosmos_task_success"]))
    return d


def fit_bb(d):
    tasks = sorted({v[0] for v in d.values()})
    data = [(tasks.index(v[0]), v[1], v[2]) for v in d.values()]

    def nll(th):
        mus = 1 / (1 + np.exp(-th[:-1]))
        k = math.exp(th[-1])
        s = 0.0
        for t, n, x in data:
            a, b = mus[t] * k, (1 - mus[t]) * k
            s -= betaln(x + a, n - x + b) - betaln(a, b)
        return s
    best = None
    for k0 in (0.0, 1.0, 2.0, 3.0, 5.0):
        r = minimize(nll, np.array([-1.5] * len(tasks) + [k0]), method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or r.fun < best.fun:
            best = r
    mus = 1 / (1 + np.exp(-best.x[:-1]))
    kappa = math.exp(best.x[-1])
    # likelihood-ratio vs kappa -> infinity (binomial)
    nll_inf = nll(np.concatenate([best.x[:-1], [math.log(1e6)]]))
    prof = {}
    for kk in (0.5, 1, 2, 5, 10, 20, 50, 1e6):
        r = minimize(lambda th: nll(np.concatenate([th, [math.log(kk)]])), best.x[:-1], method="Nelder-Mead")
        prof[str(kk)] = float(r.fun)
    return {"tasks": tasks, "mu": dict(zip(tasks, mus.tolist())), "kappa_mle": kappa, "nll": float(best.fun),
            "nll_binomial_limit": float(nll_inf), "lr_stat_vs_binomial": float(2 * (nll_inf - best.fun)),
            "profile_nll_by_kappa": prof,
            "per_checkpoint_fp_over_fail": {k: f"{v[2]}/{v[1]}" for k, v in sorted(d.items())}}


_orig_draws = m.judge_rate_draws


def patched(spec, counts, task, R, rng, pool_success=None, dshare=None):
    if "_het" in spec:
        base, kap = spec.split("_het")
        kap = float(kap)
        c = counts[base].get(task) or counts[base]["all"]
        if c["n_fail"] < 5 or c["n_succ"] < 5:
            c = counts[base]["all"]
        N = len(pool_success)
        mu = rng.beta(1 + c["fp"], 1 + c["n_fail"] - c["fp"], size=(R, 1))
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        fpr = rng.beta(mu * kap, (1 - mu) * kap, size=(R, N))
        fnr = rng.beta(1 + c["fn"], 1 + c["n_succ"] - c["fn"], size=R)[:, None]
        return fpr, fnr
    return _orig_draws(spec, counts, task, R, rng, pool_success=pool_success, dshare=dshare)


def main():
    fits = {"cosmos3_edge_local_v6": fit_bb(per_checkpoint("eval_local_v6")),
            "nemotron_nebius_v2": fit_bb(per_checkpoint("eval_nebius_v2"))}
    m.judge_rate_draws = patched
    counts = m.judge_counts()
    runs = m.load_runs(SCR / "frozen_pool")
    judges = ["perfect", "cosmos3_edge_local_v6", "nemotron_nebius_v2"]
    for jn, f in fits.items():
        kf = max(0.5, min(f["kappa_mle"], 1e4))
        judges += [f"{jn}_het{kf:.3g}", f"{jn}_het2", f"{jn}_het5"]
    strategies = ["uniform", "successive_halving", "top_two_thompson"]
    R = 1000
    half = len(runs[0]["seeds"]) // 2
    curves = {}
    floors = []
    for ri, r in enumerate(runs):
        res, _, _, H = m.simulate_split(r, np.arange(half), np.arange(half, 2 * half), judges, counts, R,
                                        seed=777 + ri, strategies=strategies)
        Pm = r["Y"][:, :half].mean(1)
        floors.append(float(H.max() - H[Pm == Pm.max()].mean()))
        N = len(r["epochs"])
        for key, dd in res.items():
            curves.setdefault(key, []).append([float(dd[max(1, int(round(mm * N)))].mean()) for mm in MULTS])
    floor = float(np.mean(floors))
    table = {}
    for (j, st), v in curves.items():
        mean = np.mean(v, axis=0)
        k = np.flatnonzero(mean - floor <= 0.02 + 1e-12)
        table.setdefault(j, {})[st] = {"mean_regret_by_mult": dict(zip(map(str, MULTS), mean.round(4).tolist())),
                                       "excess_le_0.02_mult": MULTS[k[0]] if len(k) else None}
    out = {"name": "x1_adaptive_budget_verification_hetero", "exploratory": True, "reps": R,
           "snapshot": "frozen x1 snapshot (N=22/18/16), primary split", "perfect_floor": floor,
           "beta_binomial_fits": fits, "results": table}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(fits, indent=1))
    for j in table:
        print(j, {st: (table[j][st]["excess_le_0.02_mult"], table[j][st]["mean_regret_by_mult"]["4"],
                       table[j][st]["mean_regret_by_mult"]["25"]) for st in table[j]})


if __name__ == "__main__":
    main()
