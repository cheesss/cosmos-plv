#!/usr/bin/env python3
"""EXPLORATORY -- print summary tables of results/hackathon/explore/z1_tier_triage.json (markdown)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/hackathon/explore/z1_tier_triage.json"
r = json.loads(path.read_text())


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def cis(c):
    return f"[{f(c[0], 2)}, {f(c[1], 2)}]"


for w in ("world_A_split", "world_B_parametric"):
    if w not in r:
        continue
    W = r[w]
    Kmax = W["Kmax"]
    Ks = [K for K in (1, 3, 5, 10, 15, 20, 25, 50) if K <= Kmax]
    print(f"\n### {w}  (reps {W['reps']}, orders {W['orders']}, truth tier share BAD/OK/GOOD {W['truth_tier_share']}, world judge {W['world_judge']})\n")
    print("| rule | " + " | ".join(f"K={K}" for K in Ks) + " | minK 90% (point; boot median [2.5,97.5], reached) | minK 80% | seq 0.9: acc / mean K | seq 0.95: acc / mean K |")
    print("|---" * (len(Ks) + 5) + "|")
    for name, v in W["rules"].items():
        a = v["acc_point"]
        c = v["acc_ci95"]
        cells = [f"{a[K]:.3f} {cis(c[K])}" for K in Ks]
        mb = v["minK_90_bootstrap"]
        mk = f"{v['minK_90_point'] or '>' + str(Kmax)}; {mb['median']:.0f} [{mb['p2.5']:.0f}, {mb['p97.5']:.0f}], {mb['frac_reached']:.2f}"
        sq = v.get("sequential", {})
        s9 = sq.get("0.9")
        s95 = sq.get("0.95")
        s9s = f"{s9['acc_point']:.3f} {cis(s9['acc_ci95'])} / {s9['mean_rollouts_point']:.1f} {cis(s9['mean_rollouts_ci95'])}" if s9 else "-"
        s95s = f"{s95['acc_point']:.3f} / {s95['mean_rollouts_point']:.1f}" if s95 else "-"
        print(f"| {name} | " + " | ".join(cells) + f" | {mk} | {v['minK_80_point'] or '>' + str(Kmax)} | {s9s} | {s95s} |")
    if "fixed_split_acc" in next(iter(W["rules"].values())):
        print("\nfixed split (pool 100000-24 / held-out 100025-49):")
        for name in ("perfect|naive", "perfect|bayes_tierflat", "nemotron|naive", "nemotron|bayes_tierflat", "cosmos|naive", "cosmos|bayes_tierflat"):
            v = W["rules"][name]
            print(f"  {name}: " + ", ".join(f"K{K}={v['fixed_split_acc'][K]:.3f}" for K in Ks) + f"  minK90={v['fixed_split_minK_90']}")
    print("\nclear-cut stratum", W["clear_cut_stratum"])
    for name in ("perfect|naive", "perfect|bayes_tierflat", "nemotron|naive", "nemotron|bayes_tierflat", "cosmos|naive", "cosmos|bayes_tierflat", "perfect|bayes_prog_epoch", "cosmos|bayes_prog_epoch"):
        v = W["rules"][name]
        print(f"  {name}: clear-cut acc " + ", ".join(f"K{K}={cis(v['clear_cut_acc_ci95_at'][str(K)])}" for K in Ks) + f"; clear-cut minK90 boot median {v['clear_cut_minK_90_bootstrap_median']}")
    print("\nby task (point) K=10 / K=Kmax:")
    for name in ("perfect|naive", "perfect|bayes_tierflat", "nemotron|naive", "nemotron|bayes_tierflat", "cosmos|naive", "cosmos|bayes_tierflat"):
        v = W["rules"][name]["acc_by_task_point"]
        print(f"  {name}: can {v['can'][10]:.3f}/{v['can'][Kmax]:.3f}  square {v['square'][10]:.3f}/{v['square'][Kmax]:.3f}")
    print("\nconfusion (rows truth BAD/OK/GOOD, cols call) at K=10 and Kmax:")
    for name in ("perfect|bayes_tierflat", "nemotron|naive", "nemotron|bayes_tierflat", "cosmos|naive", "cosmos|bayes_tierflat", "cosmos|bayes_iid_tierflat"):
        cm = W["rules"][name]["confusion_rows_truth_cols_call"]
        print(f"  {name}: K10 {cm['10']}  K{Kmax} {cm[str(Kmax)]}")
    print("\nrecall by truth tier at K=10:")
    for name in ("perfect|naive", "perfect|bayes_tierflat", "nemotron|naive", "nemotron|bayes_tierflat", "cosmos|naive", "cosmos|bayes_tierflat"):
        rc = W["rules"][name]["recall_point"]
        print(f"  {name}: " + ", ".join(f"{t}={rc[t][10]:.3f}" for t in rc))
    print("\nprior-only (K=0) accuracy:", {k: round(v["point"], 3) for k, v in W["prior_only_K0_acc"].items()})
    print("\naccuracy at K=10 by 50-ep band:", json.dumps(W["acc_K10_by_p50_band"]))
    print("\nworst checkpoints at K=10 (bayes_tierflat):")
    for j, rows in W["worst_checkpoints_K10"].items():
        print(f"  {j}: " + "; ".join(f"{x['checkpoint_id']} p50={x['p50']:.2f} acc10={x['acc_K10']:.2f} accKmax={x['acc_Kmax']:.2f}" for x in rows[:6]))
print("\n### judge validation (33 eval checkpoints, leave-one-checkpoint-out)\n")
for j, v in r["judge_validation"].items():
    print(j, json.dumps(v["summary"]), "implied FPR", json.dumps(v["implied_fpr"]))
print("\n### world R (real recorded verdicts)\n")
for j, v in r["world_R_real_tier_check"].items():
    print(j, "truth tier share (50-ep)", v["truth_tier_share_p50"])
    for k, x in v["rules"].items():
        print(f"  {k}: " + ", ".join(f"K{K}={x['acc_point'][K]:.3f} {cis(x['acc_ci95'][K])}" for K in (1, 3, 5, 10)))
    for rule in ("naive", "bayes_tierflat"):
        d = v[f"real_minus_sim_{rule}"]
        print(f"  real-sim {rule}: " + ", ".join(f"K{K}={d['point'][K]:+.3f} {cis(d['ci95'][K])}" for K in (1, 3, 5, 10)) + f" mean|d| per ckpt K10 {d['mean_abs_per_ckpt_K10']:.3f}")
print("\nprior diagnostics:", json.dumps(r["prior_diagnostics"]))
