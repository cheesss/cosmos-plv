#!/usr/bin/env python3
"""EXPLORATORY -- print markdown tables from results/hackathon/explore/y1_plv_prior_budget*.json.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y1_plv_prior_budget_tables.py [json ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MULTS = (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20, 25)
SHOW = (0.5, 1, 2, 4, 8)
JUDGES_MAIN = ["perfect", "nemotron_iid", "nemotron_bb", "nemotron_bbq", "cosmos_iid", "cosmos_bb", "cosmos_bbq"]


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def ci(c, d=3):
    return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def row(L, j, p, st):
    return [x for x in L["rows"] if x["judge"] == j and x["prior"] == p and x["strategy"] == st][0]


def pair(L, j, p, st):
    return [x for x in L["paired"] if x["judge"] == j and x["prior"] == p and x["strategy"] == st][0]


def cross_str(c):
    pt = ">25" if c["point"] is None else f"{c['point']:g}"
    med = ">25" if c["boot_median"] >= 50 else f"{c['boot_median']:g}"
    p10 = ">25" if c["boot_p10"] >= 50 else f"{c['boot_p10']:g}"
    p90 = ">25" if c["boot_p90"] >= 50 else f"{c['boot_p90']:g}"
    return f"{pt} ({med}; {p10}-{p90}; {c['share_reached_le_25N']:.2f})"


def main():
    paths = sys.argv[1:] or [str(ROOT / "results/hackathon/explore/y1_plv_prior_budget.json")]
    for path in paths:
        d = json.loads(Path(path).read_text())
        print(f"# {path}\n")
        priors = d["protocol"]["priors"]
        # judge models
        print("## Judge error models (final eval JSONLs)\n")
        print("| judge | task | FP / failures | FN / successes | bb kappa MLE (LR vs binomial) | P(kappa<=10) bb | bbq slope b (SE) | bbq P(kappa<=10) | bbq FPR at s=0.5 / 0.9 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for fam, src in (("nemotron", "nemotron_nebius_v2"), ("cosmos", "cosmos3_edge_local_v6")):
            c = d["judge_counts_final_eval"][src]
            fb = d["judge_fits"][fam]
            for t in ("can", "square"):
                b = fb["bbq"]
                print(f"| {fam} | {t} | {c[t]['fp']}/{c[t]['n_fail']} | {c[t]['fn']}/{c[t]['n_succ']} | "
                      f"{fb['bb']['kappa_mle_grid']:.2f} ({fb['bb']['lr_stat_vs_binomial_limit']:.2f}) | {fb['bb']['p_kappa_le_10']:.2f} | "
                      f"{b['theta_mle'][2]:.2f} ({b['slope_se']:.2f}) | {b['p_kappa_le_10']:.2f} | "
                      f"{b['implied_fpr_at_s'][t]['0.5']:.2f} / {b['implied_fpr_at_s'][t]['0.9']:.2f} |")
        print()
        for lvl, L in d["levels"].items():
            print(f"## Level {lvl}\n")
            print("problems: " + "; ".join(f"{p['run']} E<={p['E']} N={p['n_checkpoints']} k30={p['k30']}" for p in L["problems"]))
            ref = L["reference_perfect_none_uniform_25N"]
            print(f"\nreference (perfect, no prior, uniform 25N): {ref['point']:.4f} {ci(ref['ci95'], 4)}; per run {ref['per_run_point']}\n")
            # zero-rollout
            print("### Zero-rollout: top-30% floor regret and top-1 pick regret (mean over runs)\n")
            print("| prior | top-30% floor | floor - epoch | floor - random | top-1 pick | per-run floor | per-run top-1 |")
            print("|---|---|---|---|---|---|---|")
            for p in priors[1:]:
                z = L["zero_rollout"][p]
                fe = z.get("floor_minus_epoch")
                fr = z.get("floor_minus_random")
                print(f"| {p} | {f(z['top30_floor_regret']['point'])} {ci(z['top30_floor_regret']['ci95'])} | "
                      f"{'' if fe is None else f(fe['point']) + ' ' + ci(fe['ci95'])} | {'' if fr is None else f(fr['point']) + ' ' + ci(fr['ci95'])} | "
                      f"{f(z['top1_pick_regret']['point'])} {ci(z['top1_pick_regret']['ci95'])} | "
                      + ", ".join(f"{v:.2f}" for v in z["per_run_point_floor"].values()) + " | "
                      + ", ".join(f"{v:.2f}" for v in z["per_run_point_top1"].values()) + " |")
            print()
            # subset composition
            print("### Top-30% subsets (epochs) and Spearman vs held-out success (original split)\n")
            for run, cc in L["subset_composition"].items():
                print(f"* {run}: held-out best {cc['heldout_best']:.2f} at epochs {cc['best_epochs']}")
                for p in priors[2:]:
                    print(f"  * {p}: rho={cc[p]['spearman_vs_heldout']:.3f} top30={cc[p]['top30_epochs']}")
            print()
            judges = JUDGES_MAIN if lvl == "full" else ["perfect", "nemotron_bb", "cosmos_bb"]
            for st in ("top_two_thompson", "successive_halving", "uniform", "thompson"):
                for j in judges:
                    if st != "top_two_thompson" and j not in ("perfect", "nemotron_bb"):
                        continue
                    if lvl != "full" and st not in ("top_two_thompson", "uniform"):
                        continue
                    print(f"### {lvl} | judge={j} | strategy={st}: mean regret (95% seed-bootstrap CI)\n")
                    print("| prior | " + " | ".join(f"{m}N" for m in SHOW) + " | avg 0.5N-8N | 25N | crossing excess<=0.02 (point; boot median; p10-p90; share reached) | d avg vs random | d avg vs epoch |")
                    print("|---|" + "---|" * (len(SHOW) + 5))
                    for p in priors:
                        x = row(L, j, p, st)
                        cells = [f"{x['mean_regret'][MULTS.index(m)]:.3f} [{x['ci95_lo'][MULTS.index(m)]:.3f}, {x['ci95_hi'][MULTS.index(m)]:.3f}]" for m in SHOW]
                        a = x["avg_0.5N_8N"]
                        dr = de = ""
                        if p != "none":
                            pp = pair(L, j, p, st)
                            if "vs_random" in pp:
                                q = pp["vs_random"]["avg_0.5N_8N"]
                                dr = f"{q['point']:+.3f} {ci(q['ci95'])}"
                            if "vs_epoch" in pp:
                                q = pp["vs_epoch"]["avg_0.5N_8N"]
                                de = f"{q['point']:+.3f} {ci(q['ci95'])}"
                        print(f"| {p} | " + " | ".join(cells) + f" | {a['point']:.3f} {ci(a['ci95'])} | {x['mean_regret'][-1]:.3f} | "
                              f"{cross_str(x['crossing_excess_le_thr_vs_ref'])} | {dr} | {de} |")
                    print()
            # budget-to-threshold tables (needs the multi-threshold fields)
            if "crossing_by_threshold_vs_ref" in L["rows"][0]:
                for st in ("top_two_thompson", "uniform", "successive_halving"):
                    print(f"### {lvl} | {st}: budget multiple (B/N) to reach excess regret <= thr vs the perfect no-prior uniform 25N reference\n")
                    print("cell = point (bootstrap median; p10-p90); '>25' = not reached within 25N\n")
                    jl = ["perfect", "nemotron_iid", "nemotron_bb", "nemotron_bbq", "cosmos_iid", "cosmos_bb", "cosmos_bbq"]
                    print("| prior | thr | " + " | ".join(jl) + " |")
                    print("|---|---|" + "---|" * len(jl))
                    for p in priors:
                        for t in ("0.0", "0.01", "0.02"):
                            cells = []
                            for j in jl:
                                c = row(L, j, p, st)["crossing_by_threshold_vs_ref"][t]
                                pt = ">25" if c["point"] is None else f"{c['point']:g}"
                                md = ">25" if c["boot_median"] >= 50 else f"{c['boot_median']:g}"
                                p10 = ">25" if c["boot_p10"] >= 50 else f"{c['boot_p10']:g}"
                                p90 = ">25" if c["boot_p90"] >= 50 else f"{c['boot_p90']:g}"
                                cells.append(f"{pt} ({md}; {p10}-{p90})")
                            print(f"| {p} | {t} | " + " | ".join(cells) + " |")
                    print()
                    print(f"paired crossing (thr 0.01), {st}: ratio control/prior median [p10,p90]; share prior earlier/later\n")
                    print("| prior | judge | vs none | vs random | vs epoch |")
                    print("|---|---|---|---|---|")
                    for p in priors[1:]:
                        for j in ("perfect", "nemotron_bb", "cosmos_bb"):
                            cp = pair(L, j, p, st)["crossing_paired"]["0.01"]
                            cells = []
                            for ctrl in ("none", "random", "epoch"):
                                if f"vs_{ctrl}" not in cp:
                                    cells.append("")
                                    continue
                                q = cp[f"vs_{ctrl}"]
                                cells.append(f"{q['ratio_ctrl_over_prior_median']:.2g} [{q['ratio_p10_p90'][0]:.2g}, {q['ratio_p10_p90'][1]:.2g}]; "
                                             f"{q['share_prior_earlier']:.2f}/{q['share_prior_later']:.2f}")
                            print(f"| {p} | {j} | " + " | ".join(cells) + " |")
                    print()
            # beats-both summary
            print(f"### {lvl}: PLV priors whose paired CI beats BOTH random and epoch\n")
            hits_avg, hits_any, n = [], [], 0
            for x in L["paired"]:
                if "beats_both_avg_ci" not in x:
                    continue
                n += 1
                if x["beats_both_avg_ci"]:
                    hits_avg.append((x["judge"], x["strategy"], x["prior"]))
                if x["beats_both_any_mult_ci"]:
                    hits_any.append((x["judge"], x["strategy"], x["prior"], x["beats_both_any_mult_ci"]))
            print(f"cells tested (judge x strategy x PLV prior): {n}; beats both on avg 0.5N-8N: {len(hits_avg)}; "
                  f"beats both at >=1 of 0.5/1/2/4/8N: {len(hits_any)}")
            for h_ in hits_avg:
                print("  avg:", h_)
            for h_ in hits_any:
                print("  any:", h_)
            # best PLV vs epoch per judge (TTTS)
            print(f"\n{lvl}: TTTS, per judge, PLV prior with the lowest point avg regret vs epoch:")
            for j in judges:
                cand = sorted(d["protocol"]["plv_priors"], key=lambda p: row(L, j, p, "top_two_thompson")["avg_0.5N_8N"]["point"])
                best = cand[0]
                pp = pair(L, j, best, "top_two_thompson")
                print(f"  {j}: {best} avg={row(L, j, best, 'top_two_thompson')['avg_0.5N_8N']['point']:.4f} "
                      f"vs epoch {pp['vs_epoch']['avg_0.5N_8N']} vs random {pp['vs_random']['avg_0.5N_8N']}")
            print()




def headline(path, levels=("full", "trunc_1", "trunc_2", "trunc_3"), mult=1):
    """Compact pre-specified headline tables used in docs/hackathon/explore/y1_plv_prior_budget.md."""
    d = json.loads(Path(path).read_text())
    mi = MULTS.index(mult)
    pri = ["none", "random", "epoch", "replan_overlap_h16", "val_action_l1", "epoch+replan_overlap_h16", "val_loss_logged"]
    for lvl in levels:
        L = d["levels"][lvl]
        print(f"\n#### {lvl}: TTTS regret at {mult}N (mean over runs, 95% CI) and avg 0.5N-8N\n")
        print("| judge | " + " | ".join(pri) + " |")
        print("|---|" + "---|" * len(pri))
        for j in JUDGES_MAIN:
            cells = []
            for p in pri:
                x = row(L, j, p, "top_two_thompson")
                cells.append(f"{x['mean_regret'][mi]:.3f} [{x['ci95_lo'][mi]:.3f}, {x['ci95_hi'][mi]:.3f}] / {x['avg_0.5N_8N']['point']:.3f}")
            print(f"| {j} | " + " | ".join(cells) + " |")
        print(f"\n#### {lvl}: paired TTTS differences at {mult}N (prior - control; negative = prior better)\n")
        print("| judge | prior | vs none | vs random | vs epoch |")
        print("|---|---|---|---|---|")
        for j in ("perfect", "nemotron_bb", "nemotron_bbq", "cosmos_bb", "cosmos_bbq"):
            for p in ("epoch", "replan_overlap_h16", "val_action_l1", "epoch+replan_overlap_h16"):
                pp = pair(L, j, p, "top_two_thompson")
                cells = []
                for ctrl in ("none", "random", "epoch"):
                    q = pp.get(f"vs_{ctrl}")
                    cells.append("" if q is None else f"{q['by_mult'][str(mult)]['point']:+.3f} {ci(q['by_mult'][str(mult)]['ci95'])}")
                print(f"| {j} | {p} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--headline":
        for pth in sys.argv[2:] or [str(ROOT / "results/hackathon/explore/y1_plv_prior_budget.json")]:
            headline(pth)
    else:
        main()
