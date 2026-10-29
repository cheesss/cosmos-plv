#!/usr/bin/env python3
"""EXPLORATORY -- z1 verification, part 2: accuracy / mean-rollout FRONTIER of exact sequential rules
(perfect detector, worlds A and B) for flat vs epoch-progress priors over several stop thresholds.
A prior 'helps' only if its frontier lies above the flat-prior frontier at equal mean rollouts.
Writes results/hackathon/explore/z1_tier_triage_verification_frontier.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import z1_tier_triage_verify as V  # noqa: E402  (re-runs part 1 on import; cheap)

z1 = V.z1
C, priors, p50, y50, truth50 = V.C, V.priors, V.p50, V.y50, V.truth50
flatT = z1.TMASK / z1.TMASK.sum(0, keepdims=True)
pb = (priors["prog_epoch"] @ z1.TMASK)[:, 0]
cands = {"flat": priors["flat"], "tierflat": priors["tierflat"], "prog_epoch(tierflat base, z1)": priors["prog_epoch"],
         "prog_epoch(flat base, verifier)": np.stack([flatT @ np.array([pb[c], 0.8 * (1 - pb[c]), 0.2 * (1 - pb[c])]) for c in range(C)])}
THR = (0.7, 0.8, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975)
out = {}
for name, P in cands.items():
    P = P / P.sum(1, keepdims=True)
    rows = []
    for thr in THR:
        aB = np.zeros(C); eB = np.zeros(C); aA = np.zeros(C); eA = np.zeros(C)
        for c in range(C):
            Pt = V.post_tables(P[c], 50, V.LB)
            aB[c], eB[c] = V.seq_dp_world_B(Pt, p50[c], truth50[c], 50, thr)
            aA[c], eA[c] = V.seq_dp_world_A_perfect(P[c], y50[c], thr)
        rows.append({"thr": thr, "B_acc": float(aB.mean()), "B_meanK": float(eB.mean()), "A_acc": float(aA.mean()), "A_meanK": float(eA.mean())})
    out[name] = rows
# finer low thresholds for the flat reference frontier (low-budget region)
ex = []
for thr in (0.72, 0.74, 0.76, 0.78, 0.82):
    aB = np.zeros(C); eB = np.zeros(C); aA = np.zeros(C); eA = np.zeros(C)
    for c in range(C):
        Pt = V.post_tables(priors["flat"][c], 50, V.LB)
        aB[c], eB[c] = V.seq_dp_world_B(Pt, p50[c], truth50[c], 50, thr)
        aA[c], eA[c] = V.seq_dp_world_A_perfect(priors["flat"][c], y50[c], thr)
    ex.append({"thr": thr, "B_acc": float(aB.mean()), "B_meanK": float(eB.mean()), "A_acc": float(aA.mean()), "A_meanK": float(eA.mean())})
out["flat"] = sorted(out["flat"] + ex, key=lambda r: r["thr"])
# flat frontier interpolated at each other rule's mean rollouts
fl = out["flat"]
for w in ("A", "B"):
    xs = np.array([r[f"{w}_meanK"] for r in fl]); ys = np.array([r[f"{w}_acc"] for r in fl]); o = np.argsort(xs)
    for name, rows in out.items():
        for r in rows:
            r[f"{w}_flat_frontier_acc_at_same_meanK"] = float(np.interp(r[f"{w}_meanK"], xs[o], ys[o]))
            r[f"{w}_gain_vs_flat_frontier"] = r[f"{w}_acc"] - r[f"{w}_flat_frontier_acc_at_same_meanK"]
# fixed-K plug-in curves for reference
out["plugin_fixedK_B"] = {str(K): float(V.perf_B_pt[K]) for K in (10, 15, 18, 19, 20, 21, 22, 25, 30)}
out["plugin_fixedK_A"] = {str(K): float(V.accA_pt[K]) for K in (10, 12, 14, 15, 16, 17, 19, 20, 25)}
p = V.ROOT / "results/hackathon/explore/z1_tier_triage_verification_frontier.json"
p.write_text(json.dumps(out, indent=1))
for name, rows in out.items():
    if isinstance(rows, list):
        print(name)
        for r in rows:
            print(f"  thr {r['thr']:.3f}  B acc {r['B_acc']:.3f} K {r['B_meanK']:.1f} (gain vs flat frontier {r['B_gain_vs_flat_frontier']:+.3f})   A acc {r['A_acc']:.3f} K {r['A_meanK']:.1f} ({r['A_gain_vs_flat_frontier']:+.3f})")
print(out["plugin_fixedK_B"]); print(out["plugin_fixedK_A"])
print("wrote", p)
