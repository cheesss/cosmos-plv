#!/usr/bin/env python3
"""EXPLORATORY / post-hoc -- adversarial verification of z1_tier_triage.

Independent checks (CPU only, no Nebius / GPU; z1_tier_triage.py is imported for its data
loaders and judge fit only, never modified):

  1. data facts behind the headline (share near the 0.9 boundary, exact-boundary mass)
  2. K = 0 prior-only accuracies: are they driven by floating-point tie-breaking?
  3. EXACT (no Monte-Carlo) world-B plug-in accuracy curves (binomial sums) for the perfect
     detector and for the simulated judges (average over judge-parameter draws), incl.
     the boundary rule (GOOD: k/K > 0.9 vs >= 0.9), asymptotic ceiling, non-monotonicity
  4. EXACT world-A (random 25/25 split, hypergeometric) plug-in curve and sequential rule
     for the perfect detector
  5. EXACT sequential stopping (dynamic programme over (K, k)) in world B with paired,
     run-stratified checkpoint bootstrap CIs: flat vs fixed-K, prog_epoch vs tierflat
  6. judge-model robustness: leave-one-RUN-out judge fits, covariate = 50-ep training
     rate instead of 10-ep eval rate; effect on world-B plug-in accuracy / minK
  7. clear-cut stratum: point estimates and sensitivity to the stratum definition
  8. world R: which checkpoints drive real - sim; are the 'near-plateau FPs' shared
     across the two judges (episode-level ambiguity / label noise)?
Writes results/hackathon/explore/z1_tier_triage_verification.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import binom, hypergeom

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import z1_tier_triage as z1  # noqa: E402

OUT = ROOT / "results/hackathon/explore/z1_tier_triage_verification.json"
rng0 = np.random.default_rng(424242)


def tier(k, K, good_ge=False):
    k = np.asarray(k)
    K = np.asarray(K)
    good = (10 * k >= 9 * K) if good_ge else (10 * k > 9 * K)
    return np.where(2 * k < K, 0, np.where(good, 2, 1))


def strat_boot_w(run_idx, B, rng):
    C = len(run_idx)
    W = np.zeros((B, C))
    for b in range(B):
        for ri in np.unique(run_idx):
            ii = np.where(run_idx == ri)[0]
            W[b, ii] = np.bincount(rng.choice(len(ii), len(ii)), minlength=len(ii))
    return W


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def minK(curve, target=0.9):
    hit = np.where(np.asarray(curve)[1:] >= target)[0]
    return int(hit[0] + 1) if len(hit) else None


def minK_stay(curve, target=0.9):
    """smallest K such that the curve stays >= target for every K' >= K (within range)."""
    c = np.asarray(curve)
    for K in range(1, len(c)):
        if np.all(c[K:] >= target):
            return K
    return None


# ----------------------------------------------------------------------------- setup
pool = z1.load_pool()
Y = pool["Y"].astype(int)
C = Y.shape[0]
y50 = Y.sum(1)
p50 = y50 / 50
tasks = pool["task"]
run_idx = pool["run_idx"]
truth50 = tier(y50, 50)
evs = {k: z1.load_eval(v) for k, v in z1.JUDGE_FILES.items()}
fits = {j: z1.fit_judge(evs[j]) for j in ("nemotron", "cosmos")}
priors = z1.build_priors(pool)
R = {"name": "z1_tier_triage_verification", "exploratory": True}

# ----------------------------------------------------------------------------- 1 data facts
near = (p50 >= 0.80) & (p50 <= 0.95)
R["data_facts"] = {
    "n_ckpt": int(C),
    "tier50_share_BAD_OK_GOOD": (np.bincount(truth50, minlength=3) / C).round(4).tolist(),
    "share_p50_in_[0.80,0.95]": float(near.mean()), "n_p50_in_[0.80,0.95]": int(near.sum()),
    "n_p50_exactly_0.90": int((y50 == 45).sum()), "n_p50_exactly_0.50": int((y50 == 25).sum()),
    "n_p50_0.48_or_0.52": int(np.isin(y50, (24, 26)).sum()),
    "n_p50_0.88_or_0.92": int(np.isin(y50, (44, 46)).sum()),
    "ckpts_at_0.90": [pool["ids"][c] for c in np.where(y50 == 45)[0]],
    "GOOD_by_task": {t: int(((truth50 == 2) & (tasks == t)).sum()) for t in ("can", "square")},
}

# ----------------------------------------------------------------------------- 2 K=0 ties
k0 = {}
for pn in priors:
    P0 = priors[pn] @ z1.TMASK
    srt = np.sort(P0, 1)
    tie = np.abs(srt[:, 2] - srt[:, 1]) < 1e-9
    dec = P0.argmax(1)
    top = P0.max(1, keepdims=True)
    cand = np.abs(P0 - top) < 1e-9
    dec_okfirst = np.where(cand[:, 1], 1, np.where(cand[:, 0], 0, 2))
    dec_goodfirst = np.where(cand[:, 2], 2, np.where(cand[:, 1], 1, 0))
    k0[pn] = {"n_ties_top2_within_1e-9": int(tie.sum()),
              "acc_vs_tier50_argmax_as_coded": float((dec == truth50).mean()),
              "acc_ties_to_OK": float((dec_okfirst == truth50).mean()),
              "acc_ties_to_GOOD": float((dec_goodfirst == truth50).mean()),
              "share_called_GOOD_as_coded": float((dec == 2).mean())}
R["K0_prior_tie_check"] = k0


# ----------------------------------------------------------------------------- 3 exact world B plug-in
def exact_B_plugin(q, Kmax=50, good_ge=False, truth=truth50):
    """q [C] judged-success prob per ckpt; returns per-ckpt acc [C, Kmax+1] (index 0 unused)."""
    out = np.zeros((len(q), Kmax + 1))
    for K in range(1, Kmax + 1):
        k = np.arange(K + 1)
        pm = binom.pmf(k[None], K, q[:, None])
        out[:, K] = (pm * (tier(k, K, good_ge)[None] == truth[:, None])).sum(1)
    return out


def judge_q_draws(fit, D, rng, mode="q", s=None):
    """[D, C] judged-success probabilities in world B; FPR covariate s (default p50)."""
    s = p50 if s is None else s
    Q = np.zeros((D, C))
    for d in range(D):
        dj = z1.draw_judge(fit, rng)
        fnr = np.array([dj[t]["fnr"] for t in tasks])
        if mode == "q":
            fpr = np.array([float(z1.fpr_q(fit, t, ss, dj["theta"])) for t, ss in zip(tasks, s)])
        else:
            fpr = np.array([dj[t]["fpr_iid"] for t in tasks])
        Q[d] = p50 * (1 - fnr) + (1 - p50) * fpr
    return Q


def curve_summary(per_ckpt, Kmax=50, W=None):
    pt = per_ckpt.mean(0)
    d = {"acc_at": {str(K): round(float(pt[K]), 4) for K in (1, 5, 10, 20, 25, 30, 35, 36, 40, 45, 50, 100, 200) if K <= Kmax},
         "minK90_first_crossing": minK(pt), "minK90_stays_above": minK_stay(pt), "minK80": minK(pt, 0.8),
         "K_below_0.9_after_first_crossing": ([int(K) for K in range(minK(pt) or Kmax + 1, Kmax + 1) if pt[K] < 0.9]
                                               if minK(pt) else None)}
    if W is not None:
        bs = W @ per_ckpt / W.sum(1, keepdims=True)
        d["acc_ci95_at"] = {str(K): ci(bs[:, K]) for K in (10, 25, 50) if K <= Kmax}
        mk = np.array([minK(x) or Kmax + 1 for x in bs])
        d["minK90_boot"] = {"median": float(np.median(mk)), "p2.5": float(np.percentile(mk, 2.5)),
                            "p97.5": float(np.percentile(mk, 97.5)), "frac_reached": float(np.mean(mk <= Kmax))}
    return d, pt


WB = strat_boot_w(run_idx, 2000, np.random.default_rng(7))
exB = {}
perf_B = exact_B_plugin(p50.astype(float), 200)
exB["perfect_gt"], perf_B_pt = curve_summary(perf_B, 200, WB)
exB["perfect_ge_decision_rule"], _ = curve_summary(exact_B_plugin(p50.astype(float), 200, good_ge=True), 200, WB)
# asymptotic ceiling: ckpts exactly on a boundary are 50/50 forever
lim = np.where(y50 == 45, 0.5, np.where(y50 == 25, 0.5, 1.0))  # k/K -> 0.9 / 0.5 : P(correct) -> 1/2 (CLT)
exB["perfect_asymptotic_ceiling_Kinf"] = float(lim.mean())
for j in ("nemotron", "cosmos"):
    Q = judge_q_draws(fits[j], 300, np.random.default_rng([11, len(j)]))
    pc = np.mean([exact_B_plugin(Q[d], 50) for d in range(0, 300, 3)], 0)  # 100 draws
    exB[f"{j}_qworld"], _ = curve_summary(pc, 50, WB)
    Qi = judge_q_draws(fits[j], 100, np.random.default_rng([12, len(j)]), mode="iid")
    pci = np.mean([exact_B_plugin(Qi[d], 50) for d in range(100)], 0)
    exB[f"{j}_iidworld"], _ = curve_summary(pci, 50, WB)
    # point-fit world (no parameter uncertainty)
    fnr = np.array([fits[j][t]["fnr"] for t in tasks])
    fpr = np.array([float(z1.fpr_q(fits[j], t, s)) for t, s in zip(tasks, p50)])
    exB[f"{j}_qworld_pointfit"], _ = curve_summary(exact_B_plugin(p50 * (1 - fnr) + (1 - p50) * fpr, 50), 50)
R["exact_world_B_plugin"] = exB

# ----------------------------------------------------------------------------- 4 exact world A (perfect)
def exact_A_plugin(good_ge=False, Kmax=25):
    acc = np.zeros((C, Kmax + 1))
    for c in range(C):
        y = y50[c]
        m = np.arange(0, 26)
        pm = hypergeom.pmf(m, 50, y, 25)  # pool successes
        truth = tier(y - m, 25)
        for K in range(1, Kmax + 1):
            k = np.arange(K + 1)
            pk = hypergeom.pmf(k[None], 25, m[:, None], K)  # [m, k]
            pk = np.nan_to_num(pk)
            corr = tier(k, K, good_ge)[None] == truth[:, None]
            acc[c, K] = (pm[:, None] * pk * corr).sum()
    return acc


accA = exact_A_plugin()
exA, accA_pt = curve_summary(accA, 25, WB)
exA_ge, _ = curve_summary(exact_A_plugin(good_ge=True), 25, WB)
# ceiling: probability two independent halves agree is NOT a ceiling for a rule that is allowed to be
# biased; report the half-vs-half agreement over random splits (expected) for reference
agree = np.zeros(C)
for c in range(C):
    m = np.arange(26)
    pm = hypergeom.pmf(m, 50, y50[c], 25)
    agree[c] = (pm * (tier(m, 25) == tier(y50[c] - m, 25))).sum()
R["exact_world_A_perfect"] = {"plugin_gt": exA, "plugin_ge_decision_rule": exA_ge,
                              "expected_half_vs_half_tier_agreement_random_split": float(agree.mean())}


# ----------------------------------------------------------------------------- 5 exact sequential
def post_tables(prior_row, Kmax, L=None):
    """tier posterior [K+1, k+1, 3] for one prior row on the grid; L = likelihood table or perfect."""
    if L is None:
        L = z1.lik_table(z1.PGRID.copy(), Kmax)
    num = np.einsum("Kkg,g,gt->Kkt", L, prior_row, z1.TMASK, optimize=True)
    return num / np.where(num.sum(-1, keepdims=True) > 0, num.sum(-1, keepdims=True), 1)


def seq_dp_world_B(Pt, q, truth, Kmax, thr=0.9):
    """exact sequential stopping: Pt [K+1,k+1,3] tier posterior used by the rule, q judged-success prob.
    returns (P(correct), E[K])."""
    alive = np.zeros(Kmax + 1)
    alive[0] = 1.0
    acc = 0.0
    EK = 0.0
    for K in range(1, Kmax + 1):
        new = np.zeros(Kmax + 1)
        new[1:K + 1] += alive[:K] * q
        new[:K] += alive[:K] * (1 - q)
        mx = Pt[K, :K + 1].max(-1)
        dec = Pt[K, :K + 1].argmax(-1)
        stop = (mx >= thr) if K < Kmax else np.ones(K + 1, bool)
        mass = new[:K + 1] * stop
        acc += (mass * (dec == truth)).sum()
        EK += mass.sum() * K
        alive = np.zeros(Kmax + 1)
        alive[:K + 1] = new[:K + 1] * (~stop)
    return acc, EK


def seq_dp_world_A_perfect(prior_row, y, thr=0.9, Kmax=25):
    Pt = post_tables(prior_row, Kmax)
    m_all = np.arange(26)
    pm = hypergeom.pmf(m_all, 50, y, 25)
    acc = 0.0
    EK = 0.0
    for m, w in zip(m_all, pm):
        if w < 1e-15:
            continue
        truth = int(tier(y - m, 25))
        alive = np.zeros(Kmax + 1)
        alive[0] = 1.0
        for K in range(1, Kmax + 1):
            new = np.zeros(Kmax + 1)
            kk = np.arange(K)  # successes so far among K-1 draws
            ps = np.clip((m - kk) / (25 - (K - 1)), 0, 1)
            new[1:K + 1] += alive[:K] * ps
            new[:K] += alive[:K] * (1 - ps)
            mx = Pt[K, :K + 1].max(-1)
            dec = Pt[K, :K + 1].argmax(-1)
            stop = (mx >= thr) if K < Kmax else np.ones(K + 1, bool)
            mass = new[:K + 1] * stop
            acc += w * (mass * (dec == truth)).sum()
            EK += w * mass.sum() * K
            alive = np.zeros(Kmax + 1)
            alive[:K + 1] = new[:K + 1] * (~stop)
    return acc, EK


seq = {}
LB = z1.lik_table(z1.PGRID.copy(), 50)
per = {}
for pn in ("flat", "tierflat", "prog_epoch", "prog_replan_overlap_h16", "prog_val_action_l1"):
    a = np.zeros(C)
    e = np.zeros(C)
    for c in range(C):
        a[c], e[c] = seq_dp_world_B(post_tables(priors[pn][c], 50, LB), p50[c], truth50[c], 50)
    per[pn] = (a, e)
    bsA = WB @ a / WB.sum(1)
    bsE = WB @ e / WB.sum(1)
    seq[f"B_perfect_{pn}"] = {"acc": float(a.mean()), "acc_ci95": ci(bsA), "meanK": float(e.mean()), "meanK_ci95": ci(bsE),
                              "meanK_by_tier50": {z1.TIERS[t]: float(e[truth50 == t].mean()) for t in range(3)},
                              "acc_by_tier50": {z1.TIERS[t]: float(a[truth50 == t].mean()) for t in range(3)},
                              "fixed_K_plugin_interp_at_meanK": float(np.interp(e.mean(), np.arange(201), perf_B_pt)),
                              "best_fixed_K_plugin_K<=ceil(meanK)": float(perf_B_pt[1:int(np.ceil(e.mean())) + 1].max())}


def paired(a1, a2):
    d = a1 - a2
    bs = WB @ d / WB.sum(1)
    return {"point": float(d.mean()), "ci95": ci(bs)}


# paired: flat sequential vs fixed K plug-in at round(meanK)
Kf = int(round(per["flat"][1].mean()))
seq["paired_B_flat_seq_minus_fixedK_plugin"] = {"fixed_K": Kf, "acc_diff": paired(per["flat"][0], perf_B[:, Kf]),
                                                 "fixed_K_plus1": Kf + 1, "acc_diff_Kplus1": paired(per["flat"][0], perf_B[:, Kf + 1])}
seq["paired_B_prog_epoch_minus_tierflat"] = {"acc_diff": paired(per["prog_epoch"][0], per["tierflat"][0]),
                                              "meanK_diff": paired(per["prog_epoch"][1], per["tierflat"][1]),
                                              "rel_meanK_change": float(per["prog_epoch"][1].mean() / per["tierflat"][1].mean() - 1)}
seq["paired_B_prog_epoch_minus_flat"] = {"acc_diff": paired(per["prog_epoch"][0], per["flat"][0]),
                                          "meanK_diff": paired(per["prog_epoch"][1], per["flat"][1])}
# world A sequential (perfect, exact)
for pn in ("flat", "tierflat", "prog_epoch"):
    a = np.zeros(C)
    e = np.zeros(C)
    for c in range(C):
        a[c], e[c] = seq_dp_world_A_perfect(priors[pn][c], y50[c])
    per[f"A_{pn}"] = (a, e)
    seq[f"A_perfect_{pn}"] = {"acc": float(a.mean()), "acc_ci95": ci(WB @ a / WB.sum(1)), "meanK": float(e.mean()),
                              "fixed_K_plugin_interp_at_meanK": float(np.interp(e.mean(), np.arange(26), accA_pt))}
seq["paired_A_prog_epoch_minus_tierflat"] = {"acc_diff": paired(per["A_prog_epoch"][0], per["A_tierflat"][0]),
                                              "meanK_diff": paired(per["A_prog_epoch"][1], per["A_tierflat"][1]),
                                              "rel_meanK_change": float(per["A_prog_epoch"][1].mean() / per["A_tierflat"][1].mean() - 1)}
KfA = int(round(per["A_flat"][1].mean()))
seq["paired_A_flat_seq_minus_fixedK_plugin"] = {"fixed_K": KfA, "acc_diff": paired(per["A_flat"][0], accA[:, KfA])}


# judges, sequential with marginalised likelihood + flat prior, world B q-world (exact DP, averaged over draws)
for j in ("nemotron", "cosmos"):
    Lm = {t: z1.lik_table_marg(fits[j], t, 50, seed=20261029) for t in ("can", "square")}
    Pt_flat = {t: post_tables(np.full(z1.G, 1 / z1.G), 50, Lm[t]) for t in ("can", "square")}
    Q = judge_q_draws(fits[j], 60, np.random.default_rng([21, len(j)]))
    a = np.zeros(C)
    e = np.zeros(C)
    for d in range(Q.shape[0]):
        for c in range(C):
            aa, ee = seq_dp_world_B(Pt_flat[tasks[c]], Q[d, c], truth50[c], 50)
            a[c] += aa / Q.shape[0]
            e[c] += ee / Q.shape[0]
    seq[f"B_{j}_qmarg_flat"] = {"acc": float(a.mean()), "acc_ci95_ckpt_boot_only": ci(WB @ a / WB.sum(1)), "meanK": float(e.mean()),
                                "note": "60 judge-parameter draws; CI excludes parameter uncertainty"}
R["exact_sequential"] = seq

# ----------------------------------------------------------------------------- 6 judge robustness
rob = {}
ckpt_run = {c: int(run_idx[i]) for i, c in enumerate(pool["ids"])}
p50d = dict(zip(pool["ids"], p50))
for j in ("nemotron", "cosmos"):
    ev = evs[j]
    rows = {}
    fits_v = {"full": fits[j]}
    for ri, rn in enumerate(pool["run_names"]):
        excl = tuple(c for c in ev if ckpt_run[c] == ri)
        if rn.find("square") >= 0:
            continue  # square has one run: cannot drop it and still fit a_square
        fits_v[f"drop_run{ri}"] = z1.fit_judge(ev, exclude=excl)
    # covariate = 50-ep training rate instead of eval 10-ep rate
    ev_p = {c: {**v, "s_eval": float(p50d[c])} for c, v in ev.items()}
    fits_v["cov_p50"] = z1.fit_judge(ev_p)
    for name, f in fits_v.items():
        fnr = np.array([f[t]["fnr"] for t in tasks])
        fpr = np.array([float(z1.fpr_q(f, t, s)) for t, s in zip(tasks, p50)])
        pc = exact_B_plugin(p50 * (1 - fnr) + (1 - p50) * fpr, 50)
        pt = pc.mean(0)
        rows[name] = {"theta_a_can_a_square_b": np.round(f["theta"], 3).tolist(), "se": np.round(np.sqrt(np.diag(f["cov"])), 3).tolist(),
                      "fpr_can_s0.9": float(z1.fpr_q(f, "can", 0.9)), "fpr_square_s0.85": float(z1.fpr_q(f, "square", 0.85)),
                      "pointfit_B_plugin_acc_K25": round(float(pt[25]), 4), "pointfit_B_plugin_acc_K50": round(float(pt[50]), 4),
                      "pointfit_B_minK90": minK(pt)}
        # OK->GOOD rate at K=25 (exact)
        K = 25
        k = np.arange(K + 1)
        q = p50 * (1 - fnr) + (1 - p50) * fpr
        pm = binom.pmf(k[None], K, q[:, None])
        rows[name]["pointfit_B_conf_OK_to_GOOD_K25"] = float((pm[:, tier(k, K) == 2].sum(1))[truth50 == 1].mean())
    rob[j] = rows
R["judge_fit_robustness"] = rob

# ----------------------------------------------------------------------------- 7 clear-cut stratum
cc = {}
defs = {"z1_def_outside[0.45,0.55]&[0.85,0.95]": ~(((p50 >= 0.45) & (p50 <= 0.55)) | ((p50 >= 0.85) & (p50 <= 0.95))),
        "outside[0.40,0.60]&[0.80,1.00)": ~(((p50 >= 0.40) & (p50 <= 0.60)) | ((p50 >= 0.80) & (p50 < 1.0))),
        "outside[0.48,0.52]&[0.88,0.92]": ~(((p50 >= 0.48) & (p50 <= 0.52)) | ((p50 >= 0.88) & (p50 <= 0.92)))}
for dn, msk in defs.items():
    cc[dn] = {"n": int(msk.sum()), "A_perfect_point_minK90": minK(accA[msk].mean(0)),
              "A_perfect_acc_K14": float(accA[msk, 14].mean()), "A_perfect_acc_K25": float(accA[msk, 25].mean()),
              "B_perfect_point_minK90": minK(perf_B[msk].mean(0)), "B_perfect_acc_K15": float(perf_B[msk, 15].mean())}
cc["note"] = "strata are defined on the 50-ep outcome (the same episodes that make the truth and the K draws); a conditional statement, not a usable a-priori triage rule"
R["clear_cut_stratum_check"] = cc

# ----------------------------------------------------------------------------- 8 world R anatomy
wr = {}
raw = {}
for j, stem in (("nemotron", "eval_nebius_v2.jsonl"), ("cosmos", "eval_local_v6.jsonl")):
    raw[j] = {}
    for line in (ROOT / "data/outputs/hackathon_cosmos" / stem).read_text().splitlines():
        r = json.loads(line)
        raw[j][(r["checkpoint_id"], int(r["env_seed"]))] = r
idx = {c: i for i, c in enumerate(pool["ids"])}
cids = sorted(evs["nemotron"])
anat = []
for c in cids:
    i = idx[c]
    tT = int(truth50[i])
    row = {"ckpt": c, "p50": float(p50[i]), "tier50": z1.TIERS[tT]}
    for j in ("nemotron", "cosmos"):
        v = evs[j][c]
        row["eval_truth_succ"] = v["n_succ"]
        row[f"{j}_judged"] = v["J"]
        row[f"{j}_fp"] = v["fp"]
        row[f"{j}_fn"] = v["fn"]
        row[f"{j}_K10_correct"] = bool(int(tier(v["J"], 10)) == tT)
    row["perfect_K10_correct"] = bool(int(tier(row["eval_truth_succ"], 10)) == tT)
    anat.append(row)
wr["flip_rows_real_vs_perfect_K10"] = [r for r in anat if r["nemotron_K10_correct"] != r["perfect_K10_correct"]
                                       or r["cosmos_K10_correct"] != r["perfect_K10_correct"]]
wr["counts_K10_correct"] = {k: int(sum(r[k] for r in anat)) for k in ("perfect_K10_correct", "nemotron_K10_correct", "cosmos_K10_correct")}
# FP episodes in high-eval-quality checkpoints and cross-judge overlap
fail_eps = [(k, r) for k, r in raw["nemotron"].items() if r.get("truth_success") is False]
both = sum(1 for k, r in fail_eps if r.get("cosmos_task_success") and raw["cosmos"].get(k, {}).get("cosmos_task_success"))
nfp = sum(1 for k, r in fail_eps if r.get("cosmos_task_success"))
cfp = sum(1 for k, r in fail_eps if raw["cosmos"].get(k, {}).get("cosmos_task_success"))
nF = len(fail_eps)
wr["fp_cross_judge"] = {"n_truth_fail_eps": nF, "nemotron_fp": nfp, "cosmos_fp": cfp, "both_fp": both,
                        "expected_both_if_independent": nfp * cfp / nF,
                        "P(nem_fp|cos_fp)": both / cfp if cfp else None, "P(nem_fp)": nfp / nF}
hq = []
for (c, s), r in sorted(raw["nemotron"].items()):
    v = evs["nemotron"][c]
    if r.get("truth_success") is False and v["n_succ"] >= 8:
        rc = raw["cosmos"].get((c, s), {})
        hq.append({"ckpt": c, "seed": s, "eval_truth_succ": v["n_succ"], "p50": float(p50d[c]),
                   "truth_failure_type": r.get("truth_failure_type"), "truth_lifted": r.get("truth_lifted"),
                   "truth_grasped": r.get("truth_grasped"), "truth_dropped": r.get("truth_dropped"),
                   "nemotron_says_success": r.get("cosmos_task_success"), "cosmos_says_success": rc.get("cosmos_task_success"),
                   "nemotron_expl": (r.get("cosmos_explanation") or "")[:220], "cosmos_expl": (rc.get("cosmos_explanation") or "")[:220]})
wr["truth_fail_episodes_in_ckpts_with_eval_succ>=8"] = hq
R["world_R_anatomy"] = wr

OUT.write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (int(o) if isinstance(o, np.integer) else float(o))))
print("wrote", OUT)


# ----------------------------------------------------------------------------- 9 extra checks (appended)
ext = {}
# 9a prog prior on a FLAT base (BAD mass learned from epoch / PLV percentile, non-BAD mass in flat proportions)
flatT = z1.TMASK / z1.TMASK.sum(0, keepdims=True)
for m in ("epoch", "replan_overlap_h16", "val_loss_logged"):
    pb = (priors[f"prog_{m}"] @ z1.TMASK)[:, 0]
    pf = np.stack([flatT @ np.array([pb[c], 0.8 * (1 - pb[c]), 0.2 * (1 - pb[c])]) for c in range(C)])
    pf = pf / pf.sum(1, keepdims=True)
    aB = np.zeros(C)
    eB = np.zeros(C)
    aA = np.zeros(C)
    eA = np.zeros(C)
    for c in range(C):
        aB[c], eB[c] = seq_dp_world_B(post_tables(pf[c], 50, LB), p50[c], truth50[c], 50)
        aA[c], eA[c] = seq_dp_world_A_perfect(pf[c], y50[c])
    ext[f"prog_flatbase_{m}"] = {
        "B": {"acc": float(aB.mean()), "meanK": float(eB.mean()), "acc_minus_flat": paired(aB, per["flat"][0]),
              "meanK_minus_flat": paired(eB, per["flat"][1]), "rel_meanK_change": float(eB.mean() / per["flat"][1].mean() - 1),
              "meanK_by_tier50": {z1.TIERS[t]: float(eB[truth50 == t].mean()) for t in range(3)}},
        "A": {"acc": float(aA.mean()), "meanK": float(eA.mean()), "acc_minus_flat": paired(aA, per["A_flat"][0]),
              "meanK_minus_flat": paired(eA, per["A_flat"][1]), "rel_meanK_change": float(eA.mean() / per["A_flat"][1].mean() - 1)}}
# 9b prog_epoch (tierflat base) sequential vs fixed-K plug-in at its own mean budget
Kp = int(round(per["prog_epoch"][1].mean()))
ext["B_prog_epoch_seq_minus_plugin_fixedK"] = {"fixed_K": Kp, "acc_diff": paired(per["prog_epoch"][0], perf_B[:, Kp])}
ext["B_tierflat_seq_minus_plugin_fixedK"] = {"fixed_K": int(round(per["tierflat"][1].mean())),
                                             "acc_diff": paired(per["tierflat"][0], perf_B[:, int(round(per["tierflat"][1].mean()))])}
# 9c Nemotron / Cosmos world-B plug-in accuracy distribution over many judge-parameter draws (exact per draw)
for j in ("nemotron", "cosmos"):
    Q = judge_q_draws(fits[j], 1500, np.random.default_rng([31, len(j)]))
    acc50 = np.zeros(len(Q))
    acc40 = np.zeros(len(Q))
    acc25 = np.zeros(len(Q))
    for d in range(len(Q)):
        for K, arr in ((50, acc50), (40, acc40), (25, acc25)):
            k = np.arange(K + 1)
            pm = binom.pmf(k[None], K, Q[d][:, None])
            arr[d] = (pm * (tier(k, K)[None] == truth50[:, None])).sum(1).mean()
    ext[f"{j}_B_plugin_over_1500_param_draws"] = {
        "acc25_mean": float(acc25.mean()), "acc40_mean": float(acc40.mean()), "acc50_mean": float(acc50.mean()),
        "acc50_median": float(np.median(acc50)), "acc50_p5_p95": [float(np.percentile(acc50, 5)), float(np.percentile(acc50, 95))],
        "frac_draws_acc50_ge_0.9": float((acc50 >= 0.9).mean())}
# 9d leave-one-RUN-out predictive check of the judge models (can runs only; square has a single run)
from scipy.stats import binom as _b  # noqa: E402
lo = {}
for j in ("nemotron", "cosmos"):
    ev = evs[j]
    res = {"q": 0.0, "iid": 0.0, "q_abs_err": [], "iid_abs_err": []}
    for ri in (0, 1):
        excl = tuple(c for c in ev if ckpt_run[c] == ri)
        f = z1.fit_judge(ev, exclude=excl)
        rr = np.random.default_rng([41, ri, len(j)])
        th = rr.multivariate_normal(f["theta"], f["cov"], size=4000)
        cc_ = f["can"]
        fnr = rr.beta(1 + cc_["fn"], 1 + cc_["n_succ"] - cc_["fn"], size=4000)
        fpri = rr.beta(1 + cc_["fp"], 1 + cc_["n_fail"] - cc_["fp"], size=4000)
        for c in excl:
            v = ev[c]
            for mode, fpr in (("q", z1.fpr_q(f, "can", v["s_eval"], th)), ("iid", fpri)):
                pmf = z1.judged_pmf(v["n_succ"], v["n_fail"], fnr, fpr)
                res[mode] += float(np.log(max(pmf[v["J"]], 1e-12)))
                res[f"{mode}_abs_err"].append(abs(float((np.arange(len(pmf)) * pmf).sum()) - v["J"]) / 10)
    lo[j] = {"loro_logscore_sum_q": res["q"], "loro_logscore_sum_iid": res["iid"],
             "loro_mae_q": float(np.mean(res["q_abs_err"])), "loro_mae_iid": float(np.mean(res["iid_abs_err"])), "n_ckpt": 22}
ext["judge_loro_predictive_can_runs"] = lo
R["extra_checks"] = ext
OUT.write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (int(o) if isinstance(o, np.integer) else float(o))))
print("wrote (with extra checks)", OUT)

# ----------------------------------------------------------------------------- 10 verdict (static summary, see the md)
R["verdict"] = json.loads('{"status": "partially_confirmed", "reran": true, "rerun_reproduction": "z1_tier_triage.py --reps 250 --orders 8 and the iid sensitivity run reproduce results/hackathon/explore/z1_tier_triage.json and z1_tier_triage_iidworld.json with 0 numeric differences (outputs in data/outputs/hackathon_explore/z1_tier_triage_verification/)", "confirmed": ["saved VLM outputs: all listed files exist with the stated row counts; results/hackathon/raw_model_outputs/*.jsonl are byte-identical to data/outputs/hackathon_cosmos/*.jsonl; raw_text 330/330; request_id 330/330 for Nebius rows (none for local Cosmos, as expected)", "exact (no-MC) world-B perfect plug-in accuracy 0.813/0.876/0.925 at K=10/25/50 matches z1 0.814/0.876/0.925", "exact world-A perfect plug-in 0.757/0.840 at K=10/25 matches z1; K=25 accuracy equals the expected half-vs-half agreement 0.840 by construction", "exact sequential flat-prior world B: 0.9155 at 25.07 rollouts (BAD/OK/GOOD 9.0/25.1/28.8); paired gain vs fixed K=25 plug-in +3.9pt [2.9, 4.9]", "Nemotron / Cosmos qmarg-flat sequential 0.876 @34.9 / 0.883 @42.9 (z1 0.873 @35.0 / 0.881 @42.9)", "quality-dependent FPR beats constant FPR also under leave-one-RUN-out on the two can runs (log score -10.2 vs -14.6 Nemotron, -11.3 vs -16.9 Cosmos)", "world-R numbers (0.939 / 0.909 / 0.758 at K=10) and the four can GOOD checkpoints with 1 failure judged success by both VLMs", "Cosmos never reaches 0.9 by K=50 under any judge-model variant tried (0.82-0.88)"], "issues": "see md"}')
OUT.write_text(json.dumps(R, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (int(o) if isinstance(o, np.integer) else float(o))))
