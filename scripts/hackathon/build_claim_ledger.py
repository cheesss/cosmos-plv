#!/usr/bin/env python3
"""Cosmos-PLV claim ledger: every quantitative submission claim, traced to a result file.

Run in Environment B (pandas, numpy, scipy; CPU only, no network):

    source scripts/hackathon/envb.sh
    python scripts/hackathon/build_claim_ledger.py

What it does
* Recomputes each ledger value from the result files where the raw rows are available
  (baseline_metrics.csv, cosmos_diagnostics.csv, hybrid_screening.csv, the r1 events CSV, the
  c1 episodes CSV, the s2 eval summaries, the Nebius call logs, ...). Where only a summary JSON
  exists (bootstrap CIs, verifier outputs) the value is read from that JSON.
* Compares every recomputed value with the number stored in the result JSON/CSV and prints any
  difference ("recompute mismatches").
* Locates the quoted wording of each claim in the documents by snippet search, so the file:line
  references stay current while other people edit the docs.
* Re-checks a list of known document inconsistencies and prints which are still present.
* Writes submission/claim_ledger.csv (nothing else).

Nothing here calls Nebius or any other API, and nothing runs on the GPU.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/hackathon"
EXP = RES / "explore"
DATA = ROOT / "data/outputs"
OUT = ROOT / "submission/claim_ledger.csv"
ME = "scripts/hackathon/build_claim_ledger.py"

COLUMNS = ["claim_id", "exact_wording", "experiment", "n", "baseline", "metric_definition", "value", "ci95",
           "status", "preregistered", "independently_verified", "source_file", "computing_script", "limitations",
           "approved_short_wording"]

# Same catalog list prices as scripts/hackathon/make_figures.py (USD per 1M input / output tokens).
CATALOG_USD_PER_1M = {"nvidia/Nemotron-3.5-Super-VL": (0.30, 0.90), "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24)}

ROWS: list[dict] = []
MISMATCH: list[str] = []
ERRORS: list[str] = []
_DOC_CACHE: dict[str, list[str]] = {}


# ----------------------------------------------------------------------------------------- helpers
def jload(p: Path):
    return json.loads(Path(p).read_text())


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(ROOT)) if Path(p).is_absolute() else str(p)


def f3(x, d=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.{d}f}"


def ci(lo, hi, d=3) -> str:
    return f"[{f3(lo, d)}, {f3(hi, d)}]"


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def auroc(score, y) -> float:
    """Mann-Whitney AUROC, ties count 1/2; NaN scores dropped."""
    s, y = np.asarray(score, float), np.asarray(y, bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def tobool(series: pd.Series) -> pd.Series:
    return series.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))


def regret_at_q(score, success, q, keys):
    """S_best - max S over the ceil(q*N) best-ranked (ties broken by key), PLV-paper convention."""
    score, success = np.asarray(score, float), np.asarray(success, float)
    n = len(score)
    k = max(1, int(math.ceil(q * n)))
    order = sorted(range(n), key=lambda j: (-score[j], str(keys[j])))
    return float(success.max() - success[order[:k]].max())


def expect(label: str, got, want, tol: float = 6e-4):
    """Record a difference between a recomputed value and the value stored in a result file."""
    try:
        if isinstance(want, (list, tuple)):
            bad = any(abs(float(g) - float(w)) > tol for g, w in zip(got, want))
        elif isinstance(want, (int, float)):
            bad = abs(float(got) - float(want)) > tol
        else:
            bad = got != want
    except Exception:  # noqa: BLE001
        bad = True
    if bad:
        MISMATCH.append(f"{label}: recomputed {got!r} vs result file {want!r}")


def doc_lines(path: str) -> list[str]:
    if path not in _DOC_CACHE:
        p = ROOT / path
        _DOC_CACHE[path] = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    return _DOC_CACHE[path]


def _norm(s: str) -> str:
    return (s.replace("−", "-").replace("–", "-").replace("—", "-").replace("’", "'")
            .replace("**", "").replace("`", ""))


def loc(path: str, snippet: str) -> str:
    lines = doc_lines(path)
    for i, line in enumerate(lines, 1):
        if snippet in line:
            return f"{path}:{i}"
    ns = _norm(snippet)
    for i, line in enumerate(lines, 1):
        if ns in _norm(line):
            return f"{path}:{i}"
    return f"{path}:? (wording not found; the doc may have changed)"


def quote(refs) -> str:
    return " | ".join(f'"{snip}" [{loc(path, snip)}]' for path, snip in refs)


def add(claim_id, refs, experiment, n, baseline, metric_definition, value, ci95, status, preregistered,
        independently_verified, source_file, limitations, approved_short_wording, computing_script=None):
    ROWS.append({"claim_id": claim_id, "exact_wording": quote(refs) if refs else "(not quoted in the audited docs)",
                 "experiment": experiment, "n": str(n), "baseline": baseline, "metric_definition": metric_definition,
                 "value": value, "ci95": ci95 if ci95 else "none", "status": status, "preregistered": preregistered,
                 "independently_verified": independently_verified, "source_file": source_file,
                 "computing_script": computing_script or ME, "limitations": limitations,
                 "approved_short_wording": approved_short_wording})


def block(name):
    """Run one claim block; a missing input becomes an ERROR row instead of a crash."""
    def deco(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            ERRORS.append(f"{name}: {type(e).__name__}: {e}")
            add(f"{name}-ERROR", None, name, "", "", "", f"NOT COMPUTED: {type(e).__name__}: {e}", "", "exploratory",
                "", "no", "", "input missing or changed; rerun after fixing", "")
        return fn
    return deco


README, D05, D07, D08, D09, D10, D11 = ("README_HACKATHON.md", "docs/hackathon/05_nebius_execution.md",
                                        "docs/hackathon/07_experiment_results.md",
                                        "docs/hackathon/08_original_vs_new_work.md",
                                        "docs/hackathon/09_devpost_submission.md", "docs/hackathon/10_final_status.md",
                                        "docs/hackathon/11_explorations.md")
DEMO, FIGS, S2FIG = "configs/hackathon/demo_plan.json", "scripts/hackathon/make_figures.py", \
    "scripts/hackathon/explore/s2_figure.py"
AUDITED_DOCS = [README, D05, D07, D09, D10, D11, DEMO, FIGS, S2FIG]

# Values shared between blocks and the doc checks.
V: dict = {}


# ===================================================================================== historical
@block("HIST")
def _hist():
    tex = (ROOT / "docs/plv_arxiv/main.tex").read_text(encoding="utf-8")
    m = re.search(r"([\d,]+) low-dimensional robomimic checkpoints from (\d+) training runs, with (\d+) simulator", tex)
    n_ck, n_runs, n_roll = m.group(1), int(m.group(2)), int(m.group(3))
    m2 = re.search(r"Across (\d+) Diffusion Policy runs, PLV-1 improves on Action L1 in correlation \(\$([\d.]+)\$ "
                   r"versus \$([\d.]+)\$\) and Regret@30 \(\$([\d.]+)\$ versus \$([\d.]+)\$\)", tex)
    dp_runs, plv_rho, l1_rho, plv_reg, l1_reg = m2.groups()
    m3 = re.search(r"especially open-loop smoothness \(\$([\d.]+)\$\)", tex)
    ol_reg = m3.group(1)
    V["hist_dp_runs"] = dp_runs
    src = "docs/plv_arxiv/main.tex (abstract l.43; Results l.217, l.219); not a results/hackathon file"
    add("HIST-1", [(D08, "1,365"), (D07, "1,365-checkpoint"), (D09, "study of 1,365 checkpoints")],
        "Historical PLV paper (prior work)", f"{n_ck} checkpoints / {n_runs} runs / {n_roll} rollouts each",
        "-", "count of checkpoints and training runs in the paper's main retrospective cohort",
        f"{n_ck} checkpoints, {n_runs} runs (BC-RNN + Diffusion Policy), {n_roll} rollouts per checkpoint", "",
        "historical", "no (paper: 'retrospective, not preregistered')", "no (not re-run; paper checkpoints deleted)",
        src, "Prior work frozen before the hackathon; cannot be reproduced here.",
        "Prior study (not re-run): 1,365 checkpoints from 45 runs, 50 rollouts each.", computing_script=f"{ME} (regex on main.tex)")
    add("HIST-2", [(D07, "PLV-1 smoothness: Spearman 0.556, Regret@30 0.006"), (D08, "Spearman 0.556 and Regret@30 0.006")],
        "Historical PLV paper (prior work)", f"{dp_runs} Diffusion Policy runs (subset of the {n_runs})",
        "Action L1", "Spearman(success, -metric) and Regret@30 on the paper's Diffusion Policy runs",
        f"PLV-1 rho {plv_rho} vs Action L1 {l1_rho}; Regret@30 {plv_reg} vs {l1_reg}", "",
        "historical", "no", "no (not re-run)", src,
        f"Diffusion Policy cohort only = {dp_runs} runs, not 45; BC-RNN favours cheaper metrics.",
        f"Prior study, Diffusion Policy runs ({dp_runs}): PLV-1 rho 0.556 vs Action L1 0.462.",
        computing_script=f"{ME} (regex on main.tex)")
    add("HIST-3", [(D07, "Pooled open-loop smoothness: Regret@30 0.019")], "Historical PLV paper (prior work)",
        f"{n_runs} runs pooled", "random retention 0.046", "pooled Regret@30 over all runs (BC-RNN + DP)",
        f"open-loop smoothness Regret@30 {ol_reg}", "", "historical", "no", "no (not re-run)", src,
        "Pooled over both policy families, not the Diffusion Policy cohort.",
        "Prior study, all 45 runs pooled: open-loop smoothness Regret@30 0.019.",
        computing_script=f"{ME} (regex on main.tex)")


# ===================================================================================== Experiment 1
E1_METRICS = [("epoch", False, "Training epoch (free baseline)"), ("replan_overlap_h16", True, "Replan overlap (L3)"),
              ("open_loop_smoothness", True, "Open-loop smoothness (L2)"), ("boundary_jump_h16", True, "Boundary jump (L3)"),
              ("val_action_l1", True, "Action L1 (L1)"), ("plv1_smoothness", True, "PLV-1 smoothness (L2)"),
              ("val_loss_logged", True, "Validation loss")]
E1_REFS = {"epoch": [(README, "| **0.815** | 0.801 | 0.788 | 0.782 | 0.772 | 0.759 | −0.481 |"),
                     (D09, "baseline (0.815)")],
           "replan_overlap_h16": [(D07, "| Replan overlap (model feedback, L3) |")],
           "open_loop_smoothness": [(D07, "| Open-loop smoothness (L2) |")],
           "boundary_jump_h16": [(D07, "| Boundary jump (L3) |")],
           "val_action_l1": [(D07, "| Action L1 (L1) |")],
           "plv1_smoothness": [(D07, "| PLV-1 smoothness (L2) |")],
           "val_loss_logged": [(D09, "anti-correlated with success (−0.48)"), (D07, "| Validation loss |")]}


@block("E1")
def _e1():
    bm = pd.read_csv(RES / "baseline_metrics.csv")
    br = pd.read_csv(RES / "baseline_ranking.csv")
    runs = {r: len(g) for r, g in bm.groupby("run_id")}
    n_roll = sorted(bm.n_train_rollouts.unique().tolist())
    total = int(bm.n_train_rollouts.sum())
    V["e1_n"], V["e1_runs"] = len(bm), runs
    add("E1-POOL", [(README, "93 retrained Diffusion Policy checkpoints"), (D09, "93 Diffusion Policy checkpoints with 4,650 simulator rollouts"),
                    (README, "each with 50 training-time simulator rollouts (4,650 rollouts)")],
        "Experiment 1 (baseline reproduction)", f"{len(bm)} checkpoints", "-",
        "checkpoints in baseline_metrics.csv; target = training-time success over n_train_rollouts env seeds 100000-100049",
        f"{len(bm)} checkpoints in {len(runs)} runs ({', '.join(f'{k} {v}' for k, v in runs.items())}); "
        f"{n_roll} rollouts each = {total:,} rollouts", "", "primary", "with amendments (06_experiment_protocol.md)",
        "no external verifier; recomputed here", "results/hackathon/baseline_metrics.csv",
        "Short runs (200-300 epochs); 3 runs only; training-time success is itself a 50-rollout estimate.",
        "93 retrained checkpoints in 3 runs, 50 training-time rollouts each (4,650).")
    rows_e1 = {}
    for col, lib, label in E1_METRICS:
        per = []
        for run, g in bm.groupby("run_id", sort=True):
            g = g[np.isfinite(g[col].astype(float))]
            s = (-g[col] if lib else g[col]).values.astype(float)
            rho = float(spearmanr(s, g.success.values).statistic)
            reg = regret_at_q(s, g.success.values, 0.3, g.checkpoint_path.values)
            r = br[(br.metric == col) & (br.run_id == run) & (br.subset == "all")].iloc[0]
            expect(f"E1 {col} {run} rho", rho, float(r.spearman))
            expect(f"E1 {col} {run} regret@30", reg, float(r["regret@30"]))
            per.append((run, len(g), rho, reg, float(r.spearman_ci_lo), float(r.spearman_ci_hi)))
        m_rho, m_reg = float(np.mean([p[2] for p in per])), float(np.mean([p[3] for p in per]))
        mr = br[(br.metric == col) & (br.run_id == "mean_over_runs") & (br.subset == "all")].iloc[0]
        expect(f"E1 {col} mean rho", m_rho, float(mr.spearman))
        expect(f"E1 {col} mean regret@30", m_reg, float(mr["regret@30"]))
        rows_e1[col] = (m_rho, m_reg)
        late = br[(br.metric == col) & (br.run_id == "mean_over_runs") & (br.subset == "late_half")].iloc[0]
        add(f"E1-RHO-{col}", E1_REFS.get(col), "Experiment 1 (baseline reproduction)",
            f"{sum(p[1] for p in per)} checkpoints / 3 runs", "training epoch (0.815)" if col != "epoch" else "-",
            "oriented Spearman(score, 50-rollout training-time success) within each run, unweighted mean over 3 runs",
            f"{label}: mean rho {f3(m_rho)}; per run " + ", ".join(f"{p[0]} {f3(p[2], 2)}" for p in per)
            + f"; late half (epoch >= run median, {int(late.n_checkpoints)} ckpts) {f3(float(late.spearman), 2)}",
            "per-run bootstrap (2,000 over checkpoints): " + ", ".join(f"{p[0]} {ci(p[4], p[5], 2)}" for p in per)
            + "; no CI for the 3-run mean", "primary", "with amendments (06_experiment_protocol.md)",
            "no external verifier; recomputed here from baseline_metrics.csv (matches baseline_ranking.csv)",
            "results/hackathon/baseline_metrics.csv; results/hackathon/baseline_ranking.csv",
            "3 runs only; mean has no CI; most late-half per-run CIs include 0; target is a 50-rollout estimate.",
            {"epoch": "Free baseline: later epoch is better, mean within-run rho 0.815.",
             "val_loss_logged": "Validation loss is anti-correlated with success (mean rho -0.48)."}.get(
                col, f"{label}: mean within-run rho {f3(m_rho)} (below the epoch baseline 0.815)."))
    reg_vals = {c: v[1] for c, v in rows_e1.items()}
    per_sq = []
    for run, g in bm.groupby("run_id", sort=True):
        per_sq.append((run, regret_at_q(g.epoch.values.astype(float), g.success.values, 0.3, g.checkpoint_path.values)))
    add("E1-REGRET30", [(README, "Every PLV-family metric keeps a near-best checkpoint in its top 30% (Regret@30 0.007)"),
                        (D07, "Each keeps a near-best checkpoint in its top 30% (Regret@30 0.007)")],
        "Experiment 1 (baseline reproduction)", "93 checkpoints / 3 runs (k = ceil(0.3 N) = 13 / 7 / 10)",
        "epoch Regret@30 0.0067",
        "Regret@30 = best success in run - best success among the top ceil(0.3N) by the metric; mean over runs; units = success fraction",
        "; ".join(f"{c} {f3(v, 4)}" for c, v in reg_vals.items()) + "; epoch per run "
        + ", ".join(f"{r} {f3(x, 2)}" for r, x in per_sq), "none", "primary",
        "with amendments (06_experiment_protocol.md)", "no external verifier; recomputed here",
        "results/hackathon/baseline_metrics.csv; baseline_ranking.csv",
        "0.0067 = (0 + 0 + 0.02)/3: only square_seed1 has non-zero regret (one 50-rollout step = 0.02).",
        "Every PLV metric and the epoch keep a near-best checkpoint in the top 30% (Regret@30 0.007; val. loss 0.047).")
    rr = []
    for run, g in bm.groupby("run_id", sort=True):
        rr.append((run, float(spearmanr(-g.replan_overlap_h16, g.epoch).statistic),
                   float(spearmanr(-g.boundary_jump_h16, g.epoch).statistic),
                   bool(np.allclose(g.replan_overlap_h10, g.replan_overlap_h16))))
    add("E1-REPLAN-EPOCH", [(README, "(ρ with epoch 0.96–1.00)"), (D07, "(ρ with epoch 0.96–1.00)")],
        "Experiment 1 (baseline reproduction)", "93 checkpoints / 3 runs", "-",
        "Spearman(-replan disagreement, epoch) within run",
        "replan overlap " + ", ".join(f"{r[0]} {f3(r[1])}" for r in rr) + "; boundary jump "
        + ", ".join(f"{r[0]} {f3(r[2])}" for r in rr) + f"; H10 == H16 in all runs: {all(r[3] for r in rr)}",
        "", "secondary", "no (descriptive)", "no; recomputed here", "results/hackathon/baseline_metrics.csv",
        "Descriptive; explains why replanning tracks training progress in these short runs.",
        "Replanning disagreement falls almost monotonically with training (rho with epoch 0.96-1.00).")


# ===================================================================================== Experiment 2
def _confusion(g):
    t, p = tobool(g.truth_success), tobool(g.cosmos_task_success)
    tp, tn = int((t & p).sum()), int((~t & ~p).sum())
    fp, fn = int((~t & p).sum()), int((t & ~p).sum())
    n = len(g)
    return dict(n=n, tp=tp, tn=tn, fp=fp, fn=fn, acc=(tp + tn) / n, bal=0.5 * (tp / (tp + fn) + tn / (tn + fp)),
                ci=wilson(tp + tn, n), n_succ=tp + fn, n_fail=tn + fp)


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


@block("E2")
def _e2():
    cd = pd.read_csv(RES / "cosmos_diagnostics.csv")
    s2 = {(r["backend"]): r for r in jload(RES / "experiment2_summary.json")["by_backend_variant"]}
    loc_ = cd[cd.backend == "local"]
    neb = cd[cd.backend == "nebius"]
    L, N = _confusion(loc_), _confusion(neb)
    for name, c in (("local", L), ("nebius", N)):
        r = s2[name]
        expect(f"E2 {name} acc", c["acc"], r["success_acc"], 6e-4)
        expect(f"E2 {name} CI", c["ci"], r["success_acc_ci95"], 6e-4)
        expect(f"E2 {name} balanced", c["bal"], r["balanced_acc"], 6e-4)
        expect(f"E2 {name} fp/fn", (c["fp"], c["fn"]), (r["fp"], r["fn"]), 0)
    n_ck = cd.checkpoint_id.nunique()
    seeds = sorted(cd.env_seed.unique().tolist())
    base = N["n_succ"] / N["n"]
    V.update(e2_N=N, e2_L=L)
    add("E2-N", [(D09, "On 330 real rollout videos (33 checkpoints x 10 seeds)"),
                 (D07, "263 succeeded and 67 failed"), (D09, "| always answer \"success\" (base rate) | 0.797 | 0.500 | 67 of 67 |")],
        "Experiment 2 (VLM success judging)", f"{N['n']} episodes", "always answer 'success'",
        "eval episodes scored by both judges; base rate = share of true successes (privileged robosuite check)",
        f"{N['n']} episodes = {n_ck} checkpoints x {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}); {N['n_succ']} successes, "
        f"{N['n_fail']} failures; base-rate accuracy {f3(base)} (balanced 0.500)",
        f"{ci(*wilson(N['n_succ'], N['n']))} (Wilson, base rate)", "primary", "with amendments (06_experiment_protocol.md)",
        "no external verifier for Exp 2 itself; recomputed here from cosmos_diagnostics.csv",
        "results/hackathon/cosmos_diagnostics.csv", "One simulator, Franka Panda, 2 tasks, agentview camera; zero-shot.",
        "330 eval rollouts (33 checkpoints x 10 seeds); 80% succeed, so 'always success' scores 0.797.")
    for cid, c, name, refs, lat in (
            ("E2-NEMOTRON", N, "Nemotron-3.5-Super-VL (Nebius, v2_evidence_first, video)",
             [(README, "Nemotron judges rollout success with **96.4%** accuracy on 330 recorded rollouts (base rate 79.7%; balanced 92.7%)"),
              (D09, "| NVIDIA Nemotron-3.5-Super-VL via Nebius Token Factory | **0.964** [0.938, 0.979] | 0.927 | 9 of 67 (13%) |"),
              (D07, "| **Nemotron-3.5-Super-VL** (Nebius, video v2) | 0.964 [0.938, 0.979] | 0.927 | 9 | 3 |"),
              (DEMO, "Nemotron 96.4% accuracy on 330 recorded rollouts (base rate 79.7%)")], neb),
            ("E2-COSMOS", L, "Cosmos3-Edge (local, v6_perception_last, last frame)",
             [(README, "Cosmos3-Edge scores 91.8%"),
              (D09, "| NVIDIA Cosmos3-Edge, locally | **0.918** [0.884, 0.943] | 0.832 | 21 of 67 (31%) |"),
              (D07, "| **Cosmos3-Edge** (local, last-frame perception v6) | 0.918 [0.884, 0.943] | 0.832 | 21 | 6 |"),
              (DEMO, "Cosmos3-Edge 91.8%")], loc_)):
        by_task = {t: _confusion(g)["acc"] for t, g in lat.groupby("task")}
        add(cid, refs, "Experiment 2 (VLM success judging)", f"{c['n']} episodes ({c['n_fail']} failures, {c['n_succ']} successes)",
            f"base rate {f3(base)} (balanced 0.500)",
            "binary success accuracy vs privileged simulator label; balanced = mean(TPR, TNR); FP = failure called success",
            f"{name}: accuracy {f3(c['acc'])} ({c['tp'] + c['tn']}/{c['n']}); balanced {f3(c['bal'])}; FP {c['fp']}/{c['n_fail']} "
            f"({100 * c['fp'] / c['n_fail']:.0f}%); FN {c['fn']}/{c['n_succ']}; Can {f3(by_task['can'])}, Square {f3(by_task['square'])}; "
            f"mean latency {f3(float(lat.latency_s.mean()), 2)} s",
            f"{ci(*c['ci'])} (Wilson, episodes treated as independent)", "primary", "with amendments (06_experiment_protocol.md)",
            "no external verifier; recomputed here from cosmos_diagnostics.csv (matches experiment2_summary.json)",
            "results/hackathon/cosmos_diagnostics.csv; experiment2_summary.json",
            "Wilson CI ignores clustering by checkpoint (10 episodes each); successful clips stop 0.8 s after success "
            "(see E2-LENGTH); zero-shot; prompt chosen on 36 dev episodes.",
            {"E2-NEMOTRON": "Nemotron on Nebius: 0.964 [0.938, 0.979] success accuracy on 330 rollouts (9 of 67 failures called success).",
             "E2-COSMOS": "Cosmos3-Edge locally: 0.918 [0.884, 0.943] (21 of 67 failures called success)."}[cid])
    # length-neutral re-run, recomputed from the raw jsonl
    pad = pd.DataFrame(_jsonl(DATA / "hackathon_cosmos/eval_nebius_v2_pad201.jsonl"))
    org = pd.DataFrame(_jsonl(DATA / "hackathon_cosmos/eval_nebius_v2.jsonl"))
    P = _confusion(pad)
    j = jload(RES / "experiment2_backend_comparison_pad201.json")["overall"]["nebius"][0]
    expect("E2 pad acc", P["acc"], j["success_acc"])
    expect("E2 pad fp/fn", (P["fp"], P["fn"]), (j["fp"], j["fn"]), 0)
    m = org.merge(pad, on=["checkpoint_id", "env_seed"], suffixes=("_o", "_p"))
    flips = int((m.cosmos_task_success_o.astype(str) != m.cosmos_task_success_p.astype(str)).sum())
    first = pad.sort_values("started_utc").head(250)[["checkpoint_id", "env_seed"]]
    o250 = _confusion(org.merge(first, on=["checkpoint_id", "env_seed"]))
    p250 = _confusion(pad.merge(first, on=["checkpoint_id", "env_seed"]))
    m250 = m.merge(first, on=["checkpoint_id", "env_seed"])
    flips250 = int((m250.cosmos_task_success_o.astype(str) != m250.cosmos_task_success_p.astype(str)).sum())
    V.update(e2_P=P, e2_flips=flips, e2_o250=o250, e2_p250=p250, e2_flips250=flips250)
    add("E2-LENGTH", [(README, "| same, length-neutral clips (robustness) | 0.961 [0.934, 0.977] | 0.909 | 3.6 s |"),
                      (D09, "Accuracy is unchanged when clip length is neutralised"),
                      (D07, "4 of 250 judgements (first 250 episodes)"),
                      (D11, "4/250 judgements changed between the two conditions")],
        "Experiment 2 (robustness, added by amendment 05:55 KST)", f"{P['n']} episodes",
        "original Nemotron v2 (0.964, FP 9, FN 3)",
        "same prompt with every clip padded to the 20.1 s timeout (final frame held) and a fixed stated duration",
        f"length-neutral accuracy {f3(P['acc'])} ({P['tp'] + P['tn']}/{P['n']}); balanced {f3(P['bal'])}; FP {P['fp']}, FN {P['fn']}; "
        f"{flips}/{P['n']} verdicts flipped vs original. First-250 snapshot: {o250['tp'] + o250['tn']}/250 -> "
        f"{p250['tp'] + p250['tn']}/250 ({f3(o250['acc'])} both), FP {o250['fp']}->{p250['fp']}, FN {o250['fn']}->{p250['fn']}, {flips250} flips",
        f"{ci(*P['ci'])} (Wilson)", "secondary", "with amendments (robustness check added 2026-10-29 05:55 KST, after x4)",
        "partly: the clip-length leak was found by the x4 verifier; numbers recomputed here from the raw jsonl",
        "data/outputs/hackathon_cosmos/eval_nebius_v2_pad201.jsonl; results/hackathon/experiment2_backend_comparison_pad201.json",
        "Accuracy is stable but FP rises 9 -> 12 and balanced accuracy falls 0.927 -> 0.909; 'unchanged' overstates it.",
        "With length-neutral clips Nemotron scores 0.961 [0.934, 0.977] vs 0.964 (5 of 330 verdicts change).")
    # failure types
    f = neb[~tobool(neb.truth_success)]
    counts = f.truth_failure_type.value_counts()
    maj_type, maj_n = counts.index[0], int(counts.iloc[0])
    nem_ft = float((f.truth_failure_type == f.cosmos_failure_type).mean())
    v7 = pd.DataFrame(_jsonl(DATA / "hackathon_cosmos/eval_local_v7.jsonl"))
    v7f = v7[~v7.truth_success.astype(bool)]
    cos_ft = float((v7f.truth_failure_type == v7f.cosmos_failure_type).mean())
    padf = pad[~pad.truth_success.astype(bool)]
    pad_ft = float((padf.truth_failure_type == padf.cosmos_failure_type).mean())
    bc = jload(RES / "experiment2_backend_comparison.json")["overall"]
    expect("E2 nemotron ft acc on failures", nem_ft, bc["nebius"][0]["failure_type_acc_on_failures"])
    expect("E2 cosmos v7 ft acc on failures", cos_ft, bc["local_failure"][0]["failure_type_acc_on_failures"])
    expect("E2 pad ft acc on failures", pad_ft, j["failure_type_acc_on_failures"])
    conf = pd.crosstab(f.truth_failure_type, f.cosmos_failure_type.fillna("none"))
    ng = conf.loc["no_grasp"].to_dict() if "no_grasp" in conf.index else {}
    dr = conf.loc["dropped"].to_dict() if "dropped" in conf.index else {}
    V.update(e2_ng=ng, e2_dr=dr, e2_counts=counts.to_dict())
    add("E2-FAILTYPE", [(README, "score below a majority-class baseline"),
                        (D07, "Accuracy on failed episodes is 0.31 for Nemotron, 0.36 with length-neutral"),
                        (D07, "Nemotron calls most missed grasps and drops \"not_placed\""),
                        (DEMO, "Failure TYPES are not solved.")],
        "Experiment 2 (failure taxonomy)", f"{len(f)} failed episodes ({', '.join(f'{k} {v}' for k, v in counts.items())})",
        f"constant '{maj_type}' guess {f3(maj_n / len(f))} ({maj_n}/{len(f)})",
        "failure-type accuracy on truly failed episodes (labels after the 05:20 KST square-label fix)",
        f"Nemotron v2 {f3(nem_ft)}; Nemotron length-neutral {f3(pad_ft)}; Cosmos v7 8-keyframe track {f3(cos_ft)}; "
        f"majority baseline {f3(maj_n / len(f))}. Nemotron on missed grasps: {ng}; on drops: {dr}",
        f"Nemotron {ci(*wilson(int(round(nem_ft * len(f))), len(f)))}; majority {ci(*wilson(maj_n, len(f)))} (Wilson)",
        "primary", "with amendments (failure labels re-annotated 2026-10-29 05:20 KST, after eval diagnostics)",
        "no external verifier; recomputed here", "results/hackathon/cosmos_diagnostics.csv; data/outputs/hackathon_cosmos/eval_local_v7.jsonl",
        f"Nemotron labels {ng.get('not_placed', 0)}/{counts.get('no_grasp', 0)} missed grasps 'not_placed' "
        f"(and {ng.get('no_grasp', 0)} correctly), so 'most missed grasps' is wrong; {dr.get('not_placed', 0)}/{counts.get('dropped', 0)} drops.",
        "Failure types are not solved: every judge is below a constant 'no_grasp' guess (0.31 / 0.36 / 0.21 vs 0.57).")
    # agreement
    mm = loc_.merge(neb, on=["checkpoint_id", "env_seed"], suffixes=("_l", "_n"))
    pl, pn, t = tobool(mm.cosmos_task_success_l), tobool(mm.cosmos_task_success_n), tobool(mm.truth_success_l)
    agree = pl == pn
    a_ok = int((pl[agree] == t[agree]).sum())
    dis_n = int((pn[~agree] == t[~agree]).sum())
    ag = jload(RES / "experiment2_backend_comparison.json")["agreement"]
    expect("E2 agree", int(agree.sum()), ag["agree"], 0)
    expect("E2 acc when agree", a_ok / int(agree.sum()), ag["accuracy_when_agree"])
    add("E2-AGREE", [(README, "The two judges agree on 94% of episodes, and are 97% accurate when they agree"),
                     (D09, "When the two judges agree (94% of episodes), they are right 97% of the time"),
                     (D07, "the two judges agree on 93.6% of episodes; when they agree they")],
        "Experiment 2 (VLM success judging)", f"{len(mm)} episodes", "-",
        "share of episodes with the same binary verdict; accuracy on those episodes",
        f"agree {int(agree.sum())}/{len(mm)} = {f3(agree.mean())}; right when agreeing {a_ok}/{int(agree.sum())} = "
        f"{f3(a_ok / agree.sum())}; disagreements {int((~agree).sum())}, Nemotron right in {dis_n}",
        f"accuracy when agree {ci(*wilson(a_ok, int(agree.sum())))} (Wilson)", "secondary", "no (descriptive)",
        "no; recomputed here", "results/hackathon/cosmos_diagnostics.csv; experiment2_backend_comparison.json",
        "Agreement is not independent evidence: both judges share the wrong-compartment blind spot (c1).",
        "The judges agree on 94% of episodes and are right on 97% of those.")
    x0 = jload(EXP / "x0_cascade.json")["dev"]
    add("E2-DEV", [(D07, "Nemotron 0.861). Eval has more successes.")],
        "Experiment 2 (dev split)", f"{x0['local_only']['n']} dev episodes", "-",
        "success accuracy on the disjoint dev split used for prompt selection",
        f"Cosmos v6 {f3(x0['local_only']['accuracy'])}, Nemotron v2 {f3(x0['nebius_only']['accuracy'])}",
        f"Cosmos {ci(*x0['local_only']['ci95'])}, Nemotron {ci(*x0['nebius_only']['ci95'])}", "secondary",
        "with amendments", "no", "results/hackathon/explore/x0_cascade.json",
        "36 episodes from 3 checkpoints; prompt variants were selected on these episodes.",
        "On the 36-episode dev split: Cosmos 0.806, Nemotron 0.861.")
    dev = pd.DataFrame(_jsonl(DATA / "hackathon_cosmos/dev_prompt_selection.jsonl"))
    v1 = dev[(dev.variant == "v1_task_first") & (dev.backend == "local")]
    said = v1.apply(lambda r: bool(r.cosmos_task_success) if r.parse_status == "ok"
                    else bool(re.search(r'"task_success"\s*:\s*true', str(r.raw_text), re.I)), axis=1)
    add("E2-NARRATION", [(D09, "It called 12 of 12 dev episodes successful when only 3 were"),
                         (D07, "Giving Cosmos3-Edge the task goal plus the video gave \"success\" for 12/12 dev")],
        "Experiment 2 (prompt development, dev only)", f"{len(v1)} dev episodes (dev_can_s2_e0010)", "-",
        "episodes the goal-prompted Cosmos variant v1_task_first called successful (parsed JSON, or the success field in unparsable text)",
        f"{int(said.sum())}/{len(v1)} called success ({int((v1.parse_status == 'ok').sum())} parsed + "
        f"{int((v1.parse_status != 'ok').sum())} unparsable JSON whose text also says success); true successes {int(v1.truth_success.astype(bool).sum())}",
        "", "exploratory", "no (dev screening)", "no; recounted here",
        "data/outputs/hackathon_cosmos/dev_prompt_selection.jsonl", "12 episodes of one checkpoint; reasoning mode on.",
        "Given the task goal, Cosmos3-Edge called 12 of 12 dev episodes successful (3 were).")


# ===================================================================================== Experiment 3
@block("E3")
def _e3():
    hs = pd.read_csv(RES / "hybrid_screening.csv")
    hr = pd.read_csv(RES / "hybrid_ranking.csv")

    def pick_regret(col, lower_better=False):
        out = {}
        for run, g in hs.groupby("run_id", sort=True):
            s = (-g[col] if lower_better else g[col]).values.astype(float)
            out[run] = regret_at_q(s, g.success.values, 1e-9, g.checkpoint_path.values)
        return out

    def twostage(vlm):
        out = {}
        for run, g in hs.groupby("run_id", sort=True):
            k = max(1, math.ceil(0.3 * len(g)))
            top = g.sort_values(["val_action_l1", "checkpoint_path"]).head(k)
            sc = (top[vlm] - 1e-6 * top.val_action_l1).values
            order = sorted(range(len(top)), key=lambda j: (-sc[j], str(top.checkpoint_path.values[j])))
            out[run] = float(g.success.max() - top.success.values[order[0]])
        return out

    res = {"epoch": pick_regret("epoch"), "k_sim_success": pick_regret("k_sim_success"),
           "vlm_success_nebius": pick_regret("vlm_success_nebius"), "vlm_success_local": pick_regret("vlm_success_local"),
           "val_action_l1": pick_regret("val_action_l1", True),
           "hybrid_twostage_l1_nebius": twostage("vlm_success_nebius"), "hybrid_twostage_l1_local": twostage("vlm_success_local")}
    for col, d in res.items():
        stored = hr[(hr.metric == col) & hr["pick_regret@1"].notna()].set_index("run_id")["pick_regret@1"].to_dict()
        for run, v in d.items():
            expect(f"E3 pick regret {col} {run}", v, stored[run])
    rho = hr[hr.run_id == "mean_over_runs"].set_index("metric")
    mean = {c: float(np.mean(list(d.values()))) for c, d in res.items()}
    V["e3"] = mean
    n_ck, n_runs = len(hs), hs.run_id.nunique()
    add("E3-TWOSTAGE", [(README, "beat the epoch pick (Experiment 3), so the app says so"),
                        (D09, "pick, and the app says so"),
                        (D07, "| Two-stage: top-30% Action L1 → VLM-judged | — | 0.007 | 0.040 |")],
        "Experiment 3 (hybrid screening)", f"{n_ck} eval checkpoints / {n_runs} runs (11 each)",
        f"latest-epoch pick, regret {f3(mean['epoch'], 4)}",
        "regret of the single final pick = best 50-rollout success in run - success of the pick; mean over 3 runs",
        f"two-stage (top-30% Action L1 -> Nemotron-judged K=10): {f3(mean['hybrid_twostage_l1_nebius'], 3)} "
        f"(per run {', '.join(f'{r} {f3(v, 2)}' for r, v in res['hybrid_twostage_l1_nebius'].items())}); "
        f"same with Cosmos {f3(mean['hybrid_twostage_l1_local'], 3)}; epoch pick {f3(mean['epoch'], 4)} "
        f"({', '.join(f'{r} {f3(v, 2)}' for r, v in res['epoch'].items())}); Regret@30 of both 0.0067",
        "none (3 runs)", "primary", "with amendments (06_experiment_protocol.md)",
        "no external verifier; recomputed here from hybrid_screening.csv",
        "results/hackathon/hybrid_screening.csv; hybrid_ranking.csv",
        "3 runs, 11 checkpoints each; target is a 50-rollout estimate (0.02 granularity); no CI.",
        "The two-stage rule does not beat the free epoch pick: pick regret 0.040 vs 0.007.")
    add("E3-K10", [(README, "(pick regret 0.06–0.09 vs 0.007)"),
                   (D07, "| Nemotron-judged success, 10 rollouts | 0.743 | 0.013 | 0.060 |")],
        "Experiment 3 (hybrid screening)", f"{n_ck} eval checkpoints x 10 eval rollouts",
        f"epoch: rho {f3(float(rho.loc['epoch', 'spearman']))}, pick regret {f3(mean['epoch'], 4)}",
        "mean within-run Spearman and pick regret of scores built from 10 eval rollouts",
        f"Nemotron-judged rho {f3(float(rho.loc['vlm_success_nebius', 'spearman']))} / pick {f3(mean['vlm_success_nebius'])}; "
        f"Cosmos-judged {f3(float(rho.loc['vlm_success_local', 'spearman']))} / {f3(mean['vlm_success_local'])}; "
        f"privileged sim {f3(float(rho.loc['k_sim_success', 'spearman']))} / {f3(mean['k_sim_success'])}; "
        f"Action L1 {f3(float(rho.loc['val_action_l1', 'spearman']))} / {f3(mean['val_action_l1'])}",
        "none", "primary", "with amendments (06_experiment_protocol.md)", "no; recomputed here",
        "results/hackathon/hybrid_screening.csv; hybrid_ranking.csv",
        "K=10 is far below the 50-rollout target resolution; 3 runs.",
        "Ten extra rollouts rank near-equal checkpoints worse than free cheap scores; VLM-judged is no worse than privileged labels.")


# ===================================================================================== exploratory efficiency
@block("X")
def _x():
    xv = jload(EXP / "x1_adaptive_budget_verification.json")
    eb = xv["E_seed_bootstrap_frozen"]
    snap = xv["snapshots"]["frozen_N"]
    x1 = jload(EXP / "x1_adaptive_budget.json")
    thr = {(t["judge"], t["strategy"]): t["mult_excess_vs_mean_full_pool_floor_le_thr"] for t in x1["random_splits"]["thresholds"]}
    r = eb["ratio_uniform_over_TTTS_crossing"]
    V["x1"] = dict(ratio=r, snap=snap, u=eb["uniform"]["excess_le_0.02_crossing_mult"],
                   t=eb["top_two_thompson"]["excess_le_0.02_crossing_mult"])
    add("X1-ADAPTIVE", [(README, "adaptive rollout allocation needs about 2–3x fewer rollouts than uniform on this pool"),
                        (D09, "adaptive rollout allocation needs about 2–3x fewer rollouts than uniform"),
                        (D11, "Adaptive allocation needs about 2–3x fewer rollouts than uniform allocation for the same excess regret"),
                        (D07, "cuts rollouts about 2–3x versus uniform allocation"),
                        (README, "About **3x fewer rollouts** with adaptive allocation"), (DEMO, "about 3x fewer rollouts with adaptive allocation")],
        "x1 adaptive allocation (exploratory, post-hoc simulation)",
        f"frozen pool snapshot {sum(snap.values())} checkpoints ({', '.join(str(v) for v in snap.values())} per run), "
        f"B={eb['B']} seed bootstraps x {eb['reps']} orderings",
        "uniform (round-robin) allocation, no prior",
        "budget (total rollouts / N checkpoints, i.e. mean rollouts per checkpoint) at which excess regret over the "
        "full-pool (25 rollouts each) floor first falls to <= 0.02; regret measured on 25 held-out training-time seeds; "
        "perfect (simulator) success labels",
        f"uniform/top-two-Thompson budget ratio: median {r['median']:.0f}, p10 {r['p10']:.0f}, p90 {r['p90']:.0f} "
        f"(uniform crosses at median {eb['uniform']['excess_le_0.02_crossing_mult']['median']:.0f}N, TTTS at "
        f"{eb['top_two_thompson']['excess_le_0.02_crossing_mult']['median']:.0f}N; successive halving ratio median "
        f"{eb['ratio_uniform_over_SH_crossing']['median']:.0f}). Random-split crossings: perfect uniform "
        f"{thr[('perfect', 'uniform')]}N vs TTTS {thr[('perfect', 'top_two_thompson')]}N; Nemotron-judged uniform "
        f"{thr[('nemotron_nebius_v2', 'uniform')]}N vs TTTS {thr[('nemotron_nebius_v2', 'top_two_thompson')]}N",
        f"10-90% of the seed-bootstrap ratio: [{r['p10']:.0f}, {r['p90']:.0f}]", "exploratory", "no (post-hoc)",
        "yes: adversarial verifier (x1_adaptive_budget_verification.json); verdict 'partially confirmed'",
        "results/hackathon/explore/x1_adaptive_budget_verification.json (E_seed_bootstrap_frozen); x1_adaptive_budget.json",
        "Simulated by resampling logged training-time rollouts, not new rollouts; 56-checkpoint snapshot of the pool; "
        "perfect labels (VLM noise raises the budget); absolute budgets unstable as the pool grows. No PLV prior is involved; "
        "with the free latest-epoch top-30% prior TTTS reaches the target at the smallest budget tested (1N), and y1 found "
        "no PLV prior better than the epoch prior.",
        "In a post-hoc simulation, adaptive allocation (top-two Thompson) reached the same excess regret with about "
        "2-3x fewer rollouts than uniform allocation (median 3x).")
    loco = jload(EXP / "x2_early_stop_verification_loco.json")
    oc = jload(EXP / "x2_early_stop_verification.json")["oracle_and_post_success"]
    t_, s_ = loco["loco/time_only"], loco["loco/state_combo"]
    V["x2"] = dict(t=t_, s=s_, oracle=oc)
    add("X2-EARLYABORT", [(README, "a time-horizon early abort saves 16–20% of robot time"),
                          (D09, "a time-horizon early abort saves 16–20% of robot time"),
                          (D11, "Held-out abort rules save 16–20% of total robot time at 2/141 false aborts"),
                          (D07, "cuts robot time by 16–20%"), (README, "**16–20% less robot time** with early abort"), (DEMO, "16-20% robot time saved by early abort")],
        "x2 early abort (exploratory, post-hoc)",
        f"{t_['n']} eval episodes from {t_['n_checkpoints']} checkpoints (snapshot); {t_['n_successes']} successes",
        "running every episode to its recorded end (denominator = total recorded eval robot steps)",
        "fraction of total robot time saved by an abort rule refitted leave-one-checkpoint-out (max 0 false aborts on the "
        "other 19 checkpoints); false abort = aborting an episode that would have succeeded",
        f"time-only horizon cut (can 300 steps = 15 s, square 260 steps = 13 s): {f3(t_['saved_frac'])} saved, "
        f"{t_['false_aborts']}/{t_['n_successes']} false aborts; time + privileged-state rule (square: no lift by 10 s, drop): "
        f"{f3(s_['saved_frac'])} saved, {s_['false_aborts']}/{s_['n_successes']}; ranking unchanged (Spearman 1.0). "
        f"Oracle ceiling (all failing time) {f3(oc['oracle_frac'])}",
        f"time-only {ci(*t_['saved_frac_ci95_cluster_boot'], 2)}, state-combo {ci(*s_['saved_frac_ci95_cluster_boot'], 2)} "
        "(checkpoint-cluster bootstrap)", "exploratory", "no (post-hoc)",
        "yes: adversarial verifier (x2_early_stop_verification*.json); verdict 'partially confirmed'",
        "results/hackathon/explore/x2_early_stop_verification_loco.json; x2_early_stop_verification.json",
        "The 20% end needs privileged simulator state (no real-robot equivalent); the time-only rule is 16%. "
        "200-episode, 20-checkpoint snapshot; LOCO refit still uses eval outcomes of sibling checkpoints. No PLV or "
        "epoch prior is involved. Stopping on the success signal alone saves another 5.2% (post-success hold).",
        "A fixed time horizon fitted on other checkpoints saves 16% of robot time (20% with privileged state) at 2/141 "
        "false aborts, without changing the ranking.")
    x0 = jload(EXP / "x0_cascade.json")["eval"]
    cd = pd.read_csv(RES / "cosmos_diagnostics.csv")
    L = cd[cd.backend == "local"].set_index(["checkpoint_id", "env_seed"])
    N = cd[cd.backend == "nebius"].set_index(["checkpoint_id", "env_seed"]).reindex(L.index)
    pl, pn, t = tobool(L.cosmos_task_success), tobool(N.cosmos_task_success), tobool(L.truth_success)
    casc = pl & pn
    calls = int(pl.sum())
    acc = float((casc == t).mean())
    fn, fp = int((t & ~casc).sum()), int((~t & casc).sum())
    expect("X0 calls", calls, x0["cascade"]["nebius_calls"], 0)
    expect("X0 acc", acc, x0["cascade"]["accuracy"])
    expect("X0 fn/fp", (fn, fp), (x0["cascade"]["fn"], x0["cascade"]["fp"]), 0)
    V["x0"] = dict(calls=calls, n=len(L), acc=acc, fn=fn, fp=fp)
    add("X0-CASCADE", [(README, "**16% fewer Nebius calls** with a local-first cascade"), (README, "uses 16% fewer Nebius calls"), (D09, "a local-Cosmos-first cascade uses 16% fewer Nebius calls"),
                       (D11, "using 84% of the Nebius calls (16% fewer); false negatives rise from 3 to 8"),
                       (D07, "16% fewer Nebius calls (84% of calls) at accuracy 0.952 [0.923, 0.970]")],
        "x0 local-first cascade (exploratory, post-hoc, deterministic)", f"{len(L)} eval episodes",
        "Nemotron on every episode: 330 calls, accuracy 0.964, FN 3, FP 9",
        "rule: local Cosmos 'failure' is accepted; local 'success' is sent to Nemotron. Unit = Nebius calls per episode",
        f"{calls}/{len(L)} calls = {f3(calls / len(L))} ({100 * (1 - calls / len(L)):.1f}% fewer); accuracy {f3(acc)}; "
        f"FN {fn} (vs 3), FP {fp} (vs 9); mean latency {x0['cascade']['mean_latency_s']} s vs {x0['nebius_only']['mean_latency_s']} s",
        f"{ci(*x0['cascade']['ci95'])} (Wilson)", "exploratory", "no (post-hoc)",
        "no independent verifier (deterministic rule); recomputed here from cosmos_diagnostics.csv",
        "results/hackathon/explore/x0_cascade.json; results/hackathon/cosmos_diagnostics.csv",
        "Saves calls only because 20% of episodes fail; the saving shrinks for strong checkpoints; latency per episode rises "
        "(sequential). An earlier n=150 snapshot overstated the saving (71% of calls). No PLV/epoch prior involved.",
        "A local-Cosmos-first cascade uses 16% fewer Nebius calls (278 of 330) at accuracy 0.952 vs 0.964; missed successes 3 -> 8.")
    y1 = jload(EXP / "y1_plv_prior_budget_verification.json")
    cc = y1["claim_checks_on_y1_json"]["bootstrap"]["full"]
    ov = y1["subset_overlap_with_epoch_top30"]["full"]
    plv = {k: v["min_per_run"] for k, v in ov.items() if "+" not in k and k != "val_loss_logged"}
    add("Y1-PRIOR", [(D07, "they never beat the epoch prior (0/336 comparisons)"),
                     (D11, "In 0 of 336 judge × strategy × prior cells per history level did a PLV prior beat epoch")],
        "y1 PLV-metric prior for rollout allocation (exploratory, post-hoc)", "93 checkpoints; 336 judge x strategy x prior cells",
        "latest-epoch top-30% prior",
        "cells in which a PLV-metric top-30% prior beats the epoch prior on mean regret (bootstrap)",
        f"{cc['beats_epoch_alone_avg_or_any_mult']}/{cc['n_cells']} cells; min per-run overlap of PLV top-30% with epoch top-30%: "
        f"{min(plv.values()):.3f}-{max(plv.values()):.3f}", "", "exploratory", "no (post-hoc)",
        "yes: adversarial verifier (y1_plv_prior_budget_verification.json)",
        "results/hackathon/explore/y1_plv_prior_budget_verification.json",
        "Short monotone runs: the PLV top-30% sets are mostly the epoch top-30% sets.",
        "PLV priors never beat the free epoch prior (0 of 336 cells).")
    y4 = jload(EXP / "y4_grounded_reports_verification.json")
    ra = y4["random_audit_manual"]
    add("Y4-REPORTS", [(D07, "not reliable (5/16 contained a false statement)"), (D11, "Cost: 118 calls, about $0.03")],
        "y4 grounded LLM reports (exploratory)", f"{ra['n_reports']} randomly audited reports", "-",
        "reports with a clear false statement in a manual random audit",
        f"{ra['reports_with_clear_error']}/{ra['n_reports']} (auto-checker caught {ra['clear_errors_flagged_by_y4_verifier']})",
        f"{ci(*ra['wilson95'])} (Wilson)", "exploratory", "no", "yes: adversarial verifier (manual audit)",
        "results/hackathon/explore/y4_grounded_reports_verification.json",
        "16 reports; the Lightning log holds 131 calls (118 report calls + 13 verification calls).",
        "LLM-written reports were not reliable (5 of 16 contained a false statement), so the app shows only measured values.")


@block("Y3")
def _y3():
    d = jload(EXP / "y3_visual_robustness_verification.json")
    rc = d["recomputed_conditions"]
    deg = ("down224", "down112", "jpeg20", "blur", "dim")
    dd = [x["d_acc"] for x in d["paired"] if x["condition"] in deg]
    add("Y3-CAMERA", [(README, "camera placement matters, image quality does not"), (D07, "**Camera choice:** matters more than image quality")],
        "y3 visual robustness (exploratory, post-hoc)",
        f"{d['truth_and_clusters']['n_success'] + d['truth_and_clusters']['n_fail']} replayed episodes per condition "
        f"({d['truth_and_clusters']['n_checkpoints']} checkpoints; bit-exact MuJoCo replay)", "original agentview video",
        "balanced success accuracy per camera; raw accuracy change per image degradation (paired, checkpoint-cluster bootstrap)",
        "wrist camera balanced acc Cosmos " + f3(rc["local"]["cam_robot0_eye_in_hand"]["bal_acc"]) + ", Nemotron "
        + f3(rc["nebius"]["cam_robot0_eye_in_hand"]["bal_acc"]) + " (agentview " + f3(rc["local"]["cam_agentview"]["bal_acc"]) + " / "
        + f3(rc["nebius"]["cam_agentview"]["bal_acc"]) + "); raw accuracy change under degradations " + f"{min(dd):+.3f} to {max(dd):+.3f}",
        "per-comparison cluster CIs in y3_visual_robustness_verification.json (Bonferroni over 18 comparisons)", "exploratory",
        "no (post-hoc)", "yes: adversarial verifier", "results/hackathon/explore/y3_visual_robustness_verification.json",
        "60 episodes; the wrist effect is confounded by the prompt saying 'fixed third-person camera'; the docs' |delta| <= 0.03 "
        "is after re-weighting to the eval mix (raw changes are larger, none survives correction).",
        "Camera placement matters (both judges at chance on the wrist camera); image degradation has no detectable effect.")


# ===================================================================================== E4 / E4b
@block("E4")
def _e4():
    dw = jload(EXP / "e5_domain_weights.json")
    slots = {s["slot"]: s for s in dw["slots"]}
    fd = jload(EXP / "e4_cosmos_fd.json")
    conds = [c["condition"] for c in fd["per_condition_mean"]]
    add("E4-RETRACTED", [(D07, "E4 sent policy actions through Cosmos3-Edge's `libero` action slot (385 generations,"),
                         (D09, "10 slots on the model card are trained")],
        "E4 Cosmos forward dynamics via libero slot (retracted)", f"{fd['n_jobs']} generations, {len(conds)} conditions", "-",
        "CPU read of Cosmos3-Edge@344d602b action-slot weights (zero biases + weights within init bound = untrained)",
        f"libero slot {5}: trained={slots[5]['trained']}; trained slots {dw['trained_slots']} ({len(dw['trained_slots'])}); "
        f"robocasa slot 30 trained={slots[30]['trained']}; conditions {conds}",
        "", "exploratory", "no", "yes: 4 verifier lenses (07 §C3)",
        "results/hackathon/explore/e5_domain_weights.json; e4_cosmos_fd.json",
        "E4 is invalid as a test of action conditioning (it tested a random projection).",
        "E4 is retracted: it used Cosmos3-Edge's untrained libero action slot.")
    pr = jload(EXP / "e5_probe.json")
    s3 = pr["S3"]
    tr = ("farn", "dis", "tmpl")
    k16 = {t: s3["summary"][f"{t}_K16"] for t in tr}
    fam = s3["family_level"]
    psnr = {r["arm"]: r for r in s3["psnr"]}
    pbs = {(r["start"], r["arm"]): r for r in s3["psnr_by_start"]}
    V["e4b"] = dict(k16=k16, fam=fam)
    add("E4B-S3", [(README, "motion points along the true motion (cos 0.78–0.85)"),
                   (D09, "cos 0.78–0.85, 15 clips, 5 initial states"), (D11, "cos 0.78–0.85")],
        "E4b Cosmos3-Edge trained Franka slot 8 (exploratory)",
        f"{k16['farn']['moving_units']} moving clips of {k16['farn']['units']} (K=16), {len(s3['families'])} initial-state families",
        "time-reversed actions (drift cancelled: main - reversed)",
        "cosine between generated (main - reversed) gripper motion and true gripper motion; 3 trackers (farneback / DIS / template)",
        "cos " + " / ".join(f3(k16[t]["cos_main_minus_rev"]) for t in tr)
        + "; family-level mean of 5 family means " + " / ".join(f3(fam[t]["mean_of_family_means"]) for t in tr)
        + "; t(4) lower bounds " + " / ".join(f3(fam[t]["t_ci_lower"]) for t in tr)
        + "; exact sign-flip p " + " / ".join(f3(fam[t]["sign_flip_p_one_sided"], 4) for t in tr),
        "clip bootstrap " + ", ".join(ci(*k16[t]["cos_main_minus_rev_ci"], 2) for t in tr),
        "exploratory", "with amendments (S3 criteria written ~20 s after launch, before outputs were read)",
        "yes: 4 verifier lenses (07 §C3)", "results/hackathon/explore/e5_probe.json (S3.summary *_K16, S3.family_level)",
        "Only 5 initial-state families (smallest possible sign-flip p 1/32); clip CIs treat clips as independent; "
        "the '0.10-0.19 for mismatched clips' control quoted in 07 is not in any result file.",
        "Zero-shot, Cosmos3-Edge's trained Franka slot moves the generated gripper roughly along the recorded motion "
        "(cos 0.78-0.85 on 15 clips from 5 initial states).")
    add("E4B-WEAK", [(README, "Motion is compressed to about 0.4x"), (D07, "Generated motion is about 0.4x the true motion"),
                     (D07, "(19.9 vs 20.4 dB; +0.7 dB at start 16, −1.8 dB at start 48)"), (D07, "**Square is weak** (cos 0.47–0.63)")],
        "E4b Cosmos3-Edge trained Franka slot 8 (exploratory)", "22 clips (15 moving, K=16)",
        "zero actions / frozen first frame",
        "gain = |generated (main - reversed)| / |true| motion; end-point error in px; full-frame PSNR vs repeating frame 0",
        "gain " + " / ".join(f3(k16[t]["gain_main_minus_rev"]) for t in tr)
        + "; Can cos " + " / ".join(f3(fam[t]["by_task"]["can"]) for t in tr)
        + ", Square " + " / ".join(f3(fam[t]["by_task"]["square"]) for t in tr)
        + "; end error Can main " + " / ".join(str(fam[t]["end_err_by_task"]["end_err_main"]["can"]) for t in tr)
        + " vs zero " + " / ".join(str(fam[t]["end_err_by_task"]["end_err_zero"]["can"]) for t in tr)
        + "; Square main " + " / ".join(str(fam[t]["end_err_by_task"]["end_err_main"]["square"]) for t in tr)
        + " vs zero " + " / ".join(str(fam[t]["end_err_by_task"]["end_err_zero"]["square"]) for t in tr)
        + f"; PSNR main {psnr['main']['psnr_full']} vs frozen {psnr['main']['psnr_frozen']} dB "
        f"(start 16 {pbs[(16, 'main')]['psnr_full'] - pbs[(16, 'main')]['psnr_frozen']:+.2f}, start 48 "
        f"{pbs[(48, 'main')]['psnr_full'] - pbs[(48, 'main')]['psnr_frozen']:+.2f}); matched-start action effect "
        + " / ".join(f3(fam[t]["matched_s16_main_minus_rev_px"], 1) for t in tr) + " px vs seed floor "
        + " / ".join(f3(fam[t]["matched_s16_seed_floor_px"], 1) for t in tr) + " px",
        "gain clip bootstrap " + ", ".join(ci(*k16[t]["gain_main_minus_rev_ci"], 2) for t in tr),
        "exploratory", "with amendments", "yes: 4 verifier lenses", "results/hackathon/explore/e5_probe.json",
        "Not a faithful rollout; cannot replace rollouts; not used for any ranking claim.",
        "Generated motion is about 0.4x the true motion, Square is weak, and pixels are no better than a frozen frame.")
    s1 = pr["S1"]["by_encoding"]["R_q"]["trackers"]
    s0 = pr["S0"]
    add("E4B-S0S1", [(D07, "mean cos 0.87–0.94 over 12 probes, 3 trackers; next-best proper rotation 0.43–0.53"),
                     (D07, "reproduces the card's output (30.8 dB vs 26–27 dB between seeds)")],
        "E4b positive control and axis sweep (exploratory)", "S0: one scene; S1: 4 frames x 3 axes x 3 trackers", "seed change / next-best rotation",
        "S0 PSNR vs model-card output; S1 best-rotation cosine between commanded axis and generated gripper motion",
        f"S0 PSNR vs card {f3(s0['psnr_main0_vs_card_output'], 1)} dB vs between seeds {f3(s0['psnr_main0_vs_main1'], 1)} dB; "
        f"action effect {', '.join(f'{k} {f3(v, 1)}' for k, v in s0['action_effect_mad'].items())} vs seed {f3(s0['seed_noise_mad'], 1)}; "
        "S1 cos " + " / ".join(f3(s1[t]["best_cos"]) for t in ("farn", "dis", "tmpl"))
        + ", next-best " + " / ".join(f3(s1[t]["second_cos"]) for t in ("farn", "dis", "tmpl"))
        + ", motion " + " / ".join(f3(s1[t]["median_abs_r_px"], 1) for t in ("farn", "dis", "tmpl")) + " px",
        "", "exploratory", "yes (S0-S2 criteria fixed before running)", "yes: 4 verifier lenses",
        "results/hackathon/explore/e5_probe.json (S0, S1)",
        "S0 has one between-seed PSNR pair (26.2 dB), not a 26-27 range; reflection ties, so the 3-D frame is not identified.",
        "Through the trained Franka slot, +-x/y/z commands move the generated gripper consistently per axis.")


# ===================================================================================== r1
@block("R1")
def _r1():
    ev = pd.read_csv(EXP / "r1_counterfactual_replan_eval_events.csv")
    js = jload(EXP / "r1_counterfactual_replan_eval.json")
    H = ev[ev.kind == "HIGH"]
    C = ev[ev.kind == "CONTROL"]
    helpful = H.benefit > 0
    a_j1b, a_d = auroc(H.j1b_rule, helpful), auroc(H.d_all, helpful)
    ph = js["primary_HIGH"]
    expect("R1 J1b", a_j1b, ph["j1b_rule"]["auroc"])
    expect("R1 d", a_d, ph["d_all"]["auroc"])
    Hn = H[~H.determined.astype(bool)]
    a_j1b_n, a_d_n = auroc(Hn.j1b_rule, Hn.benefit > 0), auroc(Hn.d_all, Hn.benefit > 0)
    ex = js["post_hoc_after_verification"]["HIGH_excluding_determined"]
    expect("R1 J1b excl", a_j1b_n, ex["j1b_rule"]["auroc"])
    expect("R1 d excl", a_d_n, ex["d_all"]["auroc"])
    add("R1-PRIMARY", [(D09, "The pre-registered primary was negative"),
                       (D07, "| Nemotron J1b perception rule (**primary**) | 0.404 [0.340, 0.477] |"),
                       (D11, "Nemotron J1b 0.40 vs d 0.79")],
        "r1 counterfactual replanning (exploratory, pre-registered)",
        f"{len(H)} HIGH boundaries (max-d per episode, {H.checkpoint_id.nunique()} checkpoints); {int(helpful.sum())} helpful",
        "PLV replan disagreement d (all dims)",
        "AUROC for 'helpful' (benefit = p_NEW - p_OLD > 0, K=4 continuations per branch) vs rest",
        f"J1b {f3(a_j1b)}; d {f3(a_d)}; J1b - d {f3(ph['j1b_rule_minus_d_all']['diff'])}; J1 {f3(ph['j1_necessity']['auroc'])}; "
        f"Cosmos rule {f3(ph['cosmos_rule']['auroc'])}; privileged {f3(ph['privileged']['auroc'])} / raw {f3(ph['privileged_raw']['auroc'])}",
        f"J1b {ci(*ph['j1b_rule']['ci'])}; d {ci(*ph['d_all']['ci'])}; diff {ci(*ph['j1b_rule_minus_d_all']['ci'])} (bootstrap)",
        "exploratory", "with amendments (A1.1-A1.4 before eval calls; interim look at 309/400 events disclosed)",
        "yes: 4 verifier lenses; AUROCs recomputed here from the events CSV",
        "results/hackathon/explore/r1_counterfactual_replan_eval.json; r1_counterfactual_replan_eval_events.csv",
        "Primary is negative. 77/300 boundaries were decided by construction (selection defect); K=4 labels are "
        "consistent with rollout noise.", "Zero-shot judges could not tell when a replan helped (J1b 0.40 vs d 0.79).")
    add("R1-EXCL-DECIDED", [(README, "Judges at chance (about 0.49 on undecided boundaries)"),
                            (D07, "Excluding them, J1b is 0.486 [0.42, 0.55], d is 0.745")],
        "r1 (post hoc, after verification)", f"{len(Hn)} HIGH boundaries not decided by construction "
        f"({int(H.determined.astype(bool).sum())} decided excluded)", "d", "same AUROC on non-decided HIGH events",
        f"J1b {f3(a_j1b_n)}; d {f3(a_d_n)}; J1b - d {f3(ex['j1b_rule_minus_d_all']['diff'])}; Cosmos {f3(ex['cosmos_rule']['auroc'])}; "
        f"J1 {f3(ex['j1_necessity']['auroc'])}",
        f"J1b {ci(*ex['j1b_rule']['ci'])}; d {ci(*ex['d_all']['ci'])}; diff {ci(*ex['j1b_rule_minus_d_all']['ci'])}",
        "exploratory", "no (post hoc, after verification)", "yes; recomputed here",
        "results/hackathon/explore/r1_counterfactual_replan_eval.json (post_hoc_after_verification)",
        "Post hoc; 57 of J1b's 92 HIGH flags were on decided events.",
        "On the 223 non-decided boundaries the judges are at chance (J1b 0.49, Cosmos 0.54).")
    cons = lambda df: df.benefit != 0  # noqa: E731
    paired = C.cluster.unique()
    Hp = H[H.cluster.isin(paired)].set_index("cluster")
    Cp = C.set_index("cluster")
    hc, cc = cons(Hp.reindex(paired)), cons(Cp.reindex(paired))
    b_, c_ = int((hc & ~cc).sum()), int((~hc & cc).sum())
    mc_p = binomtest(b_, b_ + c_).pvalue if b_ + c_ else 1.0
    fp_ = fisher_exact([[int(cons(H).sum()), len(H) - int(cons(H).sum())], [int(cons(C).sum()), len(C) - int(cons(C).sum())]]).pvalue
    hv = js["post_hoc_after_verification"]["HIGH_vs_CONTROL_consequential"]
    expect("R1 paired H/C", (int(hc.sum()), int(cc.sum())), (hv["paired_HIGH"], hv["paired_CONTROL"]), 0)
    expect("R1 fisher", fp_, hv["fisher_p"], 2e-3)
    perm = js["post_hoc_after_verification"]["permutation_null_HIGH"]
    add("R1-LOCALIZE", [(README, "(19 vs 18 of 100)"), (D11, "(19 vs 18 of 100)"), (D11, "(permutation P=0.26)")],
        "r1 (post hoc, after verification)", f"{len(paired)} paired episodes (HIGH + low-d CONTROL)",
        "same-episode below-median-d CONTROL boundary",
        "consequential = benefit != 0; McNemar exact on discordant pairs; permutation null reshuffles the 8 outcomes",
        f"paired: HIGH {int(hc.sum())} vs CONTROL {int(cc.sum())} of {len(paired)} consequential (discordant {b_}/{c_}, McNemar p {f3(mc_p, 2)}); "
        f"all: HIGH {int(cons(H).sum())}/{len(H)} vs CONTROL {int(cons(C).sum())}/{len(C)} (Fisher p {f3(fp_, 2)}); "
        f"permutation: consequential {perm['consequential']['observed']} vs null {perm['consequential']['null_mean']:.1f} "
        f"(P {perm['consequential']['p_ge_observed']}), helpful {perm['helpful']['observed']} vs {perm['helpful']['null_mean']:.1f} "
        f"(P {perm['helpful']['p_ge_observed']}); d predicts recorded episode failure AUROC "
        f"{f3(js['post_hoc_after_verification']['d_all_predicts_recorded_episode_failure_HIGH'])}",
        "", "exploratory", "no (post hoc)", "yes; paired counts and Fisher p recomputed here",
        "results/hackathon/explore/r1_counterfactual_replan_eval_events.csv; r1_counterfactual_replan_eval.json",
        "K=4 continuations cannot separate helpful from harmful replans.",
        "PLV's d flags fragile episodes, not the replan that mattered: a low-d boundary in the same episode was "
        "consequential as often (19 vs 18 of 100).")
    nd = H[~H.determined.astype(bool)].copy()
    sn = nd.succ_new.map(ast.literal_eval)
    so = nd.succ_old.map(ast.literal_eval)
    allok = sn.map(all) & so.map(all)
    q = nd[allok]
    nf, of = int((q.steps_new < q.steps_old).sum()), int((q.steps_new > q.steps_old).sum())
    p_sign = binomtest(nf, nf + of).pvalue
    st = js["secondary_outcomes_HIGH_not_determined"]["steps_to_success_all_8_succeed"]
    expect("R1 steps new/old faster", (nf, of), (st["new_faster"], st["old_faster"]), 0)
    expect("R1 sign p", p_sign, st["sign_test_p"], 1e-3)
    add("R1-STEPS", [(README, "the new plan reaches success sooner (80 vs 51, p = 0.014)"),
                     (D09, "replanning reaches success about 3.5 steps sooner")],
        "r1 pre-registered secondary (reported after verification)", f"{len(q)} non-decided HIGH events where all 8 continuations succeed",
        "OLD plan tail", "steps to success, NEW vs OLD branch; two-sided sign test",
        f"NEW faster {nf}, OLD faster {of}, ties {len(q) - nf - of}; sign p {f3(p_sign, 3)}; mean steps saved "
        f"{f3(float((q.steps_old - q.steps_new).mean()), 2)}; d predicts it at AUROC {f3(st['auroc_d_all_new_faster'], 2)}",
        "", "exploratory", "yes (pre-registered secondary; missing from the first output, added after verification)",
        "yes; recomputed here from the events CSV", "results/hackathon/explore/r1_counterfactual_replan_eval_events.csv",
        "Speed only; d does not predict which replans are faster.",
        "The one robust effect: replanning reaches success sooner (80 vs 51 events, p = 0.014).")
    j2 = js["post_hoc_after_verification"]["j2_combined_decided_consequential_HIGH"]
    jc = js["j2_combined_on_consequential_HIGH"]
    add("R1-J2", [(D09, "Choosing between the two drawn paths did worse than always replanning"),
                  (D07, "abstained on 29 of 52 consequential events")],
        "r1 (post hoc)", f"{jc['n']} consequential HIGH events; {j2['coverage']} decided by J2", "always choose the NEW plan",
        "accuracy of Nemotron's two-path choice J2 on events where it did not abstain",
        f"abstained {jc['choice_counts']['tie']}/{jc['n']}; accuracy on decided {f3(j2['accuracy'], 3)} vs always-new "
        f"{f3(j2['always_new_accuracy_same_events'], 3)}; label '1' share {j2['label_1_share_all_events']}",
        "", "exploratory", "no (post hoc)", "yes (verifier)", "results/hackathon/explore/r1_counterfactual_replan_eval.json",
        "23 decided events only.", "Choosing between the two drawn paths did worse than always replanning (0.61 vs 0.70).")
    cnt = js["counts"]
    add("R1-COUNTS", [(D07, "HIGH: 30 helpful, 22 harmful, 248 neutral.")], "r1", "400 events (300 HIGH + 100 CONTROL)", "-",
        "helpful / harmful / neutral = benefit > 0 / < 0 / = 0",
        f"HIGH {int((H.benefit > 0).sum())}/{int((H.benefit < 0).sum())}/{int((H.benefit == 0).sum())}; CONTROL "
        f"{int((C.benefit > 0).sum())}/{int((C.benefit < 0).sum())}/{int((C.benefit == 0).sum())}; HIGH checkpoints "
        f"{H.checkpoint_id.nunique()}; JSON n_events {js['n_events']}",
        "", "exploratory", "with amendments", "yes; recounted here", "results/hackathon/explore/r1_counterfactual_replan_eval_events.csv",
        "", "400 same-state branch pairs: 300 max-disagreement boundaries plus 100 low-disagreement controls.")
    expect("R1 HIGH counts", (int((H.benefit > 0).sum()), int((H.benefit < 0).sum())), (cnt["HIGH"]["helpful"], cnt["HIGH"]["harmful"]), 0)


# ===================================================================================== c1
@block("C1")
def _c1():
    ep = pd.read_csv(EXP / "c1_coverage_shift_episodes.csv")
    js = jload(EXP / "c1_coverage_shift.json")
    succ = ep.success.astype(str).str.lower().eq("true")
    f = ep[~succ].copy()
    fs = f.nemotron_says_success.fillna(0).astype(float) > 0.5
    by = {s: (int(fs[f.stratum == s].sum()), int((f.stratum == s).sum())) for s in ("M", "U", "T")}
    cs = f.cosmos_says_success.fillna(0).astype(float) > 0.5
    cby = {s: (int(cs[f.stratum == s].sum()), int((f.stratum == s).sum())) for s in ("M", "U", "T")}
    a = auroc(f.q, fs)
    n = js["nemotron"]
    expect("C1 auroc", a, n["auroc_q"])
    for s in by:
        expect(f"C1 FS {s}", by[s], (n["by_stratum"][s]["false_successes"], n["by_stratum"][s]["failures"]), 0)
    V["c1"] = dict(by=by, a=a)
    add("C1-PRIMARY", [(README, "false successes rose 6/25, 14/27, 13/22"),
                       (D09, "rose from 6/25 to 14/27 to 13/22"), (D11, "6/25, 14/27, 13/22 (M/U/T)"),
                       (D07, "AUROC of q for false success is 0.673 [0.556, 0.750]")],
        "c1 coverage shift (exploratory, pre-registered)",
        f"{len(ep)} Can episodes from {ep.checkpoint_id.nunique()} checkpoints "
        f"({', '.join(f'{s} {ep[ep.stratum == s].checkpoint_id.nunique()} ckpts/{int((ep.stratum == s).sum())} eps' for s in ('M', 'U', 'T'))}); {len(f)} failures",
        "stratum M (q 0.38-0.70)", "false success = failure the judge calls success; AUROC of checkpoint quality q for false success among failures",
        f"Nemotron FS {', '.join(f'{s} {k}/{m}' for s, (k, m) in by.items())}; AUROC {f3(a)}; Fisher T vs M p {f3(n['fisher_T_vs_M_p'], 3)}; "
        f"Cosmos FS {', '.join(f'{s} {k}/{m}' for s, (k, m) in cby.items())}",
        f"checkpoint-cluster {ci(*n['ci_checkpoint'])}; seed-cluster {ci(*n['ci_seed'])}; two-way "
        f"{ci(*js['post_hoc_after_verification']['two_way_bootstrap_nemotron'])} (post hoc)",
        "exploratory", "with amendments (A1 before any judge call; erratum to A1 after verification)",
        "yes: 4 verifier lenses; counts and AUROC recomputed here from the episodes CSV",
        "results/hackathon/explore/c1_coverage_shift.json; c1_coverage_shift_episodes.csv",
        "Positive only with checkpoint clusters; seed-cluster and two-way CIs include 0.5; two initial states (600002, 600011) "
        "give 23/33 false successes; leaving out 600011 gives 0.589 [0.439, 0.694].",
        "The share of failures Nemotron accepts rose with checkpoint quality (6/25, 14/27, 13/22); positive but fragile.")
    fine = pd.crosstab(f.final_place_fine, fs)
    wc = f[f.final_place == "goal_bin_other_compartment"]
    v2c = int((wc.nemotron_v2c_says_success.fillna(0).astype(float) > 0.5).sum())
    nog = f[f.failure_type == "no_grasp"]
    nog_rej_n = int((nog.nemotron_says_success.fillna(0).astype(float) < 0.5).sum())
    nog_rej_c = int((nog.cosmos_says_success.fillna(0).astype(float) < 0.5).sum())
    V["c1"].update(n_fs=int(fs.sum()), wc_acc=int(fs[f.final_place == "goal_bin_other_compartment"].sum()), n_wc=len(wc), v2c=v2c)
    add("C1-MECHANISM", [(README, "Nemotron called **32 of 33** \"can in the wrong compartment\" failures a success, and 24 of 33 even when told the compartment"),
                         (D09, "successes ended there, 32 of them with the can in the wrong compartment"),
                         (D09, "(24/33 still accepted)"),
                         (DEMO, "Nemotron called 32 of 33 wrong-compartment failures 'success' - and 24/33 even when the prompt stated the compartment"),
                         (D09, "The judges never miss a failed grasp (21/21 caught)")],
        "c1 (secondary 7/8 from amendment A1; place split post hoc)", f"{len(f)} failures; {len(wc)} wrong-compartment failures", "-",
        "where the failed can came to rest (privileged final object position); v2c = compartment-explicit prompt",
        f"Nemotron FS {int(fs.sum())}: " + ", ".join(f"{p} {int(fine.loc[p, True]) if True in fine.columns else 0}/{int(fine.loc[p].sum())}"
                                                      for p in fine.index)
        + f"; v2c still accepts {v2c}/{len(wc)} wrong-compartment cans; no-grasp failures rejected: Nemotron {nog_rej_n}/{len(nog)}, "
        f"Cosmos {nog_rej_c}/{len(nog)}; dropped-only AUROC {f3(n['dropped_only']['auroc_q'], 2)}",
        "", "exploratory", "with amendments (A1)", "yes: 4 verifier lenses; recounted here",
        "results/hackathon/explore/c1_coverage_shift_episodes.csv; c1_coverage_shift.json",
        "'All 33 at the goal bin' = 32 wrong compartment + 1 on a rim/divider. Two different '32 of 33' statements are both "
        "correct: 32 of the 33 wrong-compartment failures were accepted, and 32 of the 33 false successes were wrong-compartment "
        "cans. Only 2 v2c rejections cite the compartment (verifier note, not in a result file).",
        "Judges accept almost every failure that ends at the goal bin (32 of 33 wrong-compartment cans), even when told the compartment (24 of 33).")
    pre = {s: (int(((ep.stratum == s) & succ & (ep.nemotron_says_success > 0.5)).sum()),
               int(((ep.stratum == s) & (ep.nemotron_says_success > 0.5)).sum())) for s in ("M", "U", "T")}
    for s, (k, m) in pre.items():
        expect(f"C1 precision {s}", k / m, n["precision_of_claimed_success"][s])
    blk = ep[ep.black_video.astype(str).str.lower().eq("true")]
    sens = js["post_hoc_after_verification"]["sensitivity_valid_videos"]
    add("C1-PRECISION-BLACK", [(D11, "Precision of judged successes rises with quality (0.85/0.90/0.97)"), (D07, "| Nemotron precision of claimed success | 0.854 | 0.898 | 0.965 |"),
                               (D07, "16 T-stratum success videos at seed 600003 are black after frame 0")],
        "c1", f"{len(ep)} episodes", "-", "precision = true successes / episodes Nemotron calls success, per stratum",
        "precision " + ", ".join(f"{s} {f3(k / m)} ({k}/{m})" for s, (k, m) in pre.items())
        + f"; black videos {len(blk)} (strata {sorted(blk.stratum.unique().tolist())}, seeds {sorted(blk.env_seed.unique().tolist())}); "
        f"Nemotron called {int((blk.nemotron_says_success > 0.5).sum())} of them success, Cosmos {int((blk.cosmos_says_success > 0.5).sum())}; "
        f"Nemotron sensitivity on valid videos {f3(sens['nemotron'])}",
        "", "exploratory", "with amendments", "yes; recounted here", "results/hackathon/explore/c1_coverage_shift_episodes.csv",
        "16 black (EGL artefact) success videos; precision is on all judged-success episodes.",
        "Precision of judged successes rises with quality (0.85, 0.90, 0.97).")


# ===================================================================================== s1 / s2
def _eval_success(d: Path):
    e = jload(d / "summary.json")["episodes"]
    return {int(x["env_seed"]): bool(x["success"]) for x in e}, e


@block("S1")
def _s1():
    js = jload(EXP / "s1_self_improve.json")
    sr = js["success_rate"]
    ev = DATA / "hackathon_explore/s1_selfimprove/eval"
    rec = {}
    if ev.exists():
        for arm in sr:
            if (ev / arm / "summary.json").exists():
                s, _ = _eval_success(ev / arm)
                rec[arm] = float(np.mean(list(s.values())))
                expect(f"S1 {arm}", rec[arm], sr[arm])
    p = js["paired"]
    arms = {k: v for k, v in sr.items() if k != "base"}
    add("S1-NULL", [(README, "Can judged self-successes lift a plateaued policy (0.885)?"),
                    (D09, "All arms stayed at 0.875–0.89 on 200 paired seeds: Nemotron −0.5 points"),
                    (D11, "Every arm scored 0.875–0.89 on 200 paired seeds")],
        "s1 self-training (exploratory, pre-registered)", "200 paired eval seeds per arm", "continue (fine-tune on the original 36 demos)",
        "closed-loop success on 200 fresh paired seeds; paired difference vs continue, seed bootstrap",
        f"base {sr['base']}; arms {', '.join(f'{k} {v}' for k, v in arms.items())}; Nemotron - continue {p['primary_nemotron_minus_continue']['diff']:+.3f}; "
        f"oracle - continue {p['oracle_minus_continue']['diff']:+.3f}; all - continue {p['all_minus_continue']['diff']:+.3f}",
        f"Nemotron {ci(*p['primary_nemotron_minus_continue']['ci95'])}; oracle {ci(*p['oracle_minus_continue']['ci95'])}",
        "exploratory", "yes (no amendments; deviations disclosed)", "yes: 4 verifier lenses, no bug",
        "results/hackathon/explore/s1_self_improve.json" + ("; eval summaries recomputed" if rec else ""),
        "One run per arm; base is a plateaued policy (fresh eval 0.885; its filename says 0.860 = training-time 50 rollouts); "
        "the recipe did not respond to data content, so it cannot rank label sources.",
        "Self-training a plateaued 0.885 policy on judged successes changed nothing (all arms 0.875-0.89).")


@block("S2")
def _s2():
    js = jload(EXP / "s2_self_improve.json")
    sr = js["success_rate"]
    root = DATA / "hackathon_explore/s2_selfimprove"
    succ = {}
    for arm in sr:
        s, e = _eval_success(root / "eval" / arm)
        succ[arm] = (s, e)
        expect(f"S2 {arm}", float(np.mean(list(s.values()))), sr[arm])
    seeds = sorted(succ["continue"][0])

    def paired(a, b):
        x = np.array([succ[a][0][s] for s in seeds], float)
        y = np.array([succ[b][0][s] for s in seeds], float)
        return float((x - y).mean()), int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())

    p = js["paired"]
    out = {}
    for key, (a, b) in {"human_minus_continue": ("human", "continue"), "oracle_minus_continue": ("oracle", "continue"),
                        "primary_nemotron_minus_continue": ("nemotron", "continue"), "nemotron_minus_oracle": ("nemotron", "oracle")}.items():
        d, ao, bo = paired(a, b)
        expect(f"S2 {key} diff", d, p[key]["diff"])
        expect(f"S2 {key} discordant", (ao, bo), tuple(p[key]["discordant"]), 0)
        out[key] = (d, ao, bo, p[key]["ci95"])
    add("S2-ARMS", [(README, "lowered policy success by **10.4 points** vs continued training (95% CI −14.8 to −6.0). 36 human demos added +8.2."),
                    (README, "Human demos **+8.2** [+5.2, +11.4]. Own successes with simulator labels −3.2 [−7.0, +0.6]. Nemotron-filtered **−10.4** [−14.8, −6.0]"),
                    (D09, "The episodes Nemotron called successes: 0.752, −10.4 [−14.8, −6.0]"),
                    (D11, "continue 0.856; human +8.2 [+5.2, +11.4]; oracle −3.2 [−7.0, +0.6]; Nemotron −10.4 [−14.8, −6.0]"),
                    (DEMO, "success fell 10.4 points vs continued training, while 36 human demos added +8.2")],
        "s2 self-training redesign (exploratory, pre-registered)", f"{len(seeds)} paired eval seeds ({seeds[0]}-{seeds[-1]}) per arm",
        f"continue (fine-tune on the original 36 demos), {sr['continue']}",
        "closed-loop success on 500 fresh paired seeds; paired difference vs continue; seed bootstrap 10,000; exact McNemar",
        f"base {sr['base']}, continue {sr['continue']}, human {sr['human']} ({out['human_minus_continue'][0]:+.3f}), oracle {sr['oracle']} "
        f"({out['oracle_minus_continue'][0]:+.3f}), nemotron {sr['nemotron']} ({out['primary_nemotron_minus_continue'][0]:+.3f}); "
        f"post hoc nemotron - oracle {out['nemotron_minus_oracle'][0]:+.3f}; discordant h/c {out['human_minus_continue'][1]}/{out['human_minus_continue'][2]}, "
        f"n/c {out['primary_nemotron_minus_continue'][1]}/{out['primary_nemotron_minus_continue'][2]}",
        f"human {ci(*out['human_minus_continue'][3])}; oracle {ci(*out['oracle_minus_continue'][3])}; nemotron "
        f"{ci(*out['primary_nemotron_minus_continue'][3])}; nemotron - oracle {ci(*out['nemotron_minus_oracle'][3])} (post hoc)",
        "exploratory", "yes (no amendments during the run)", "yes: 1 independent verifier; arm rates and paired diffs recomputed here",
        "results/hackathon/explore/s2_self_improve.json; data/outputs/hackathon_explore/s2_selfimprove/eval/*/summary.json",
        "One training run per arm (CIs cover eval seeds only); fixed 8,000-step budget dilutes the original demos "
        "(10.5% of samples in oracle); judge clip length follows the simulator's success signal; H2 not interpretable "
        "(oracle did not beat continue).",
        "Filtering self-training data with Nemotron made the policy 10 points worse than continued training "
        "(0.752 vs 0.856); 36 human demos added 8 points.")
    nl = js["nemotron_labels"]
    td = js["training_data"]["nemotron"]
    # post-hoc goal-bin failure counts on the eval seeds (same final-place geometry as explore/c1_analyze.py)
    bin2, size = np.array([0.1, 0.28, 0.8]), np.array([0.39, 0.49])

    def over_bin(npz):
        pz = np.load(ROOT / npz)["object_pos"][-1]
        return bool(abs(pz[0] - bin2[0]) < size[0] / 2 and abs(pz[1] - bin2[1]) < size[1] / 2), \
            bool(abs(pz[0] - bin2[0]) < size[0] / 2 and abs(pz[1] - bin2[1]) < size[1] / 2 and bin2[2] < pz[2] < bin2[2] + 0.1)

    gb, wcmp = {}, {}
    for arm in ("nemotron", "oracle"):
        gb[arm], wcmp[arm] = {}, {}
        for x in succ[arm][1]:
            if not x["success"]:
                o, w = over_bin(x["episode_npz"])
                gb[arm][int(x["env_seed"])], wcmp[arm][int(x["env_seed"])] = o, w
    g_n, g_o = sum(gb["nemotron"].values()), sum(gb["oracle"].values())
    disc_n = sum(1 for s, v in gb["nemotron"].items() if v and not gb["oracle"].get(s, False))
    disc_o = sum(1 for s, v in gb["oracle"].items() if v and not gb["nemotron"].get(s, False))
    V["s2"] = dict(g_n=g_n, g_o=g_o)
    add("S2-LABELS", [(README, "Nemotron accepted 58/149 failures, 44 of them in the wrong compartment"),
                      (D09, "Nemotron accepted 58 of 149 failures, 44 of them with the can in the wrong"),
                      (S2FIG, "Nemotron accepted 58/149 failures (44 cans in the wrong compartment)"),
                      (D11, "mostly through extra goal-bin misplacements (46 vs 19)"), (DEMO, "Nemotron let 58 failures into self-training data")],
        "s2 (secondary 5 + post hoc)", f"{nl['failures']} failures among 400 collected episodes", "simulator labels (oracle mask)",
        "judge-accepted failures; their share of the Nemotron arm's self-collected frames; goal-bin eval failures (post hoc)",
        f"accepted {nl['false_successes']}/{nl['failures']} failures (precision {f3(nl['precision'])}, recall {f3(nl['recall'])}); "
        f"resting place {nl['false_success_final_place']}; {td['failure_frames']:,}/{td['self_frames']:,} = {f3(nl['contamination_frames'])} "
        f"of self frames; post hoc eval failures resting over the goal bin: nemotron {g_n} vs oracle {g_o} (discordant {disc_n}/{disc_o}; "
        f"strictly inside a wrong compartment {sum(wcmp['nemotron'].values())} vs {sum(wcmp['oracle'].values())})",
        "", "exploratory", "yes for the label audit; the 46 vs 19 split is post hoc", "yes (verifier); goal-bin counts recomputed here from episode npz",
        "results/hackathon/explore/s2_self_improve.json; data/outputs/hackathon_explore/s2_selfimprove/build_info.json, eval/*/episodes/*.npz",
        "'46 vs 19' counts every failure resting over the goal-bin footprint (wrong compartment + rim/divider/held above); "
        "strict wrong-compartment counts are 31 vs 12.",
        "Nemotron accepted 58 of 149 failed episodes (44 cans in the wrong compartment); kept whole, they were 40% of the self-collected frames.")
    # base-number resolution
    ck = succ["base"][1][0]["checkpoint_path"]
    mfn = re.search(r"epoch=(\d+)-test_mean_score=([\d.]+)\.ckpt", ck)
    run_dir = ROOT / Path(ck).parents[1]
    logs = [json.loads(l) for l in (run_dir / "logs.json.txt").read_text().splitlines() if l.strip()]
    lr = [r for r in logs if r.get("epoch") == int(mfn.group(1)) and "test/mean_score" in r][0]
    n_test = len([k for k in lr if k.startswith("test/sim_max_reward_")])
    k_test = int(sum(v for k, v in lr.items() if k.startswith("test/sim_max_reward_")))
    shards = []
    for sh in sorted((root / "collect").glob("*/summary.json")):
        d = jload(sh)
        e = d["episodes"]
        shards.append((sh.parent.name, len(e), sum(bool(x["success"]) for x in e), min(x["env_seed"] for x in e), max(x["env_seed"] for x in e)))
    tot, k_tot = sum(s[1] for s in shards), sum(s[2] for s in shards)
    expect("S2 collection rate", k_tot / tot, js["collection_success_rate"])
    be = succ["base"][0]
    V["s2_base"] = dict(train=float(mfn.group(2)), k_test=k_test, n_test=n_test, coll=k_tot / tot, shards=shards,
                        eval=float(np.mean(list(be.values()))), n_eval=len(be))
    add("S2-BASE", [(README, "Same question, redesigned: weaker base"),
                    (D09, "weaker base (0.656)"), (D11, "weaker base (0.656)"), (S2FIG, "base policy\\n(36 demos, epoch 125)")],
        "s2 base checkpoint B2 (definitions of 'base success')",
        f"three different samples: {n_test} training-time rollouts / {tot} collection episodes / {len(be)} fresh eval seeds",
        "-", "success rate of the same checkpoint under three protocols (see value)",
        f"(1) training-time success in the filename/train log: {mfn.group(2)} = {k_test}/{n_test} rollouts on seeds 100000-100049 "
        f"(epoch {int(mfn.group(1))}; this is the 'about 0.58'); (2) collection success: {k_tot}/{tot} = {f3(k_tot / tot, 4)} over the 400 "
        "collected episodes (" + ", ".join(f"{s[0]} {s[2]}/{s[1]} = {f3(s[2] / s[1])} on seeds {s[3]}-{s[4]}" for s in shards)
        + f"); (3) fresh paired eval: {sum(be.values())}/{len(be)} = {f3(float(np.mean(list(be.values()))))} on seeds "
        f"{min(be)}-{max(be)} (the docs' 0.656, the only one comparable with the arms)",
        f"eval {ci(*wilson(sum(be.values()), len(be)))}; training-time {ci(*wilson(k_test, n_test))}; collection {ci(*wilson(k_tot, tot))} (Wilson)",
        "exploratory", "yes (protocol names the file and 'training-time success 0.58')", "yes; recomputed here from the train log and summaries",
        f"{rel(run_dir / 'logs.json.txt')}; data/outputs/hackathon_explore/s2_selfimprove/collect/*/summary.json, eval/base/summary.json",
        "Shard a (0.655) is numerically close to the eval rate (0.656) but is a different sample; quote 0.656 only as the "
        "500-seed eval success.",
        "Base B2 (epoch 125, 36 demos): 0.656 on the 500 paired eval seeds (0.58 at training time on 50 rollouts).")


# ===================================================================================== Nebius totals
@block("NEB")
def _neb():
    logs = {"main": RES / "nebius_calls.jsonl", "explore": EXP / "nebius_calls_explore.jsonl"}
    per_log, per_model = {}, {}
    last, first = "", "9999"
    lat = {}
    for name, p in logs.items():
        rows = _jsonl(p)
        per_log[name] = dict(n=len(rows), failed=sum(bool(r.get("error")) for r in rows),
                             last=max(r["logged_utc"] for r in rows), first=min(r["started_utc"] for r in rows),
                             prompt=sum(r.get("prompt_tokens") or 0 for r in rows),
                             completion=sum(r.get("completion_tokens") or 0 for r in rows))
        lat[name] = np.array([r["latency_s"] for r in rows if r.get("latency_s") is not None], float)
        last, first = max(last, per_log[name]["last"]), min(first, per_log[name]["first"])
        for r in rows:
            m = r.get("served_model") or r["model_id"]
            t = per_model.setdefault(m, {"calls": 0, "prompt": 0, "completion": 0, "unpriced": 0})
            t["calls"] += 1
            if r.get("prompt_tokens") is None or r.get("completion_tokens") is None:
                t["unpriced"] += 1
                continue
            t["prompt"] += r["prompt_tokens"]
            t["completion"] += r["completion_tokens"]
    for m, t in per_model.items():
        pin, pout = CATALOG_USD_PER_1M[m]
        t["usd"] = t["prompt"] / 1e6 * pin + t["completion"] / 1e6 * pout
    total = sum(v["n"] for v in per_log.values())
    usd = sum(t["usd"] for t in per_model.values())
    usd_all_supervl = sum((t["prompt"] * 0.30 + t["completion"] * 0.90) / 1e6 for t in per_model.values())
    # cross-check with make_figures._nebius_total (same files, same prices)
    try:
        sys.path.insert(0, str(ROOT / "scripts/hackathon"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("mf", ROOT / "scripts/hackathon/make_figures.py")
        mf = importlib.util.module_from_spec(spec)
        import matplotlib
        matplotlib.use("Agg")
        spec.loader.exec_module(mf)
        nt = mf._nebius_total()
        expect("NEB total (make_figures)", sum(nt["calls"].values()), total, 0)
        expect("NEB usd (make_figures)", sum(t["usd"] for t in nt["models"].values()), usd, 1e-6)
        mf_note = f"make_figures._nebius_total(): {sum(nt['calls'].values())} calls, ${sum(t['usd'] for t in nt['models'].values()):.4f}"
    except Exception as e:  # noqa: BLE001
        mf_note = f"make_figures._nebius_total() not importable here ({type(e).__name__})"
    V["neb"] = dict(per_log=per_log, per_model=per_model, total=total, usd=usd, last=last, first=first,
                    lat={k: dict(mean=float(v.mean()), p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)),
                                 max=float(v.max())) for k, v in lat.items()})
    add("NEB-TOTAL", [(D09, "All Nebius calls in the study (7,003, 0 failed) used about 14.6 M input tokens and 0.78 M output tokens"),
                      (D10, "7,003 logged calls, 0 failures"), (DEMO, "logged Nebius calls cost about $5 at the public list price"),
                      (D05, "\"calls_total\": 772"), (D05, "\"calls_total\": 2380")],
        "Nebius Token Factory usage (all logged calls)", f"{total:,} logged calls", "-",
        "rows in results/hackathon/nebius_calls.jsonl + results/hackathon/explore/nebius_calls_explore.jsonl; failed = non-null "
        "'error'; cost = logged tokens x public catalog list price per served model (make_figures.CATALOG_USD_PER_1M)",
        f"{total:,} calls = {per_log['main']['n']} main + {per_log['explore']['n']:,} exploratory; failed {sum(v['failed'] for v in per_log.values())}; "
        + "; ".join(f"{m}: {t['calls']:,} calls, {t['prompt']:,} in / {t['completion']:,} out = ${t['usd']:.4f}" for m, t in per_model.items())
        + f"; TOTAL {sum(t['prompt'] for t in per_model.values()):,} in / {sum(t['completion'] for t in per_model.values()):,} out = "
        f"${usd:.2f} (all at Super-VL price: ${usd_all_supervl:.2f}); first call {first}; log cutoff (last logged) {last}; {mf_note}",
        "none", "secondary", "no (usage accounting, not a study)", "yes: recomputed here and cross-checked with make_figures._nebius_total()",
        "results/hackathon/nebius_calls.jsonl; results/hackathon/explore/nebius_calls_explore.jsonl; scripts/hackathon/make_figures.py",
        "Catalog-price estimate from logged tokens, not billing; cached-input discounts not applied; counts are as of the "
        "log cutoff and grow with every new dashboard call. The Lightning calls are already priced at the Lightning rate.",
        f"{total:,} logged Nebius calls, 0 failed, about ${usd:.2f} at catalog list price (estimate, not billing; log cutoff {last[:16]} UTC).")


# ===================================================================================== demo video
def _ffprobe(path: Path):
    """Container duration and frame size of a local video (read-only; None if ffprobe/the file is unavailable)."""
    import shutil
    import subprocess
    if not path.exists() or not shutil.which("ffprobe"):
        return None
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration:stream=width,height",
                        "-of", "json", str(path)], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    j = json.loads(r.stdout)
    st = (j.get("streams") or [{}])[0]
    return dict(duration=float(j["format"]["duration"]), w=st.get("width"), h=st.get("height"))


@block("DEMO")
def _demo():
    plan = jload(ROOT / DEMO)
    dur, real = 0.0, 0.0
    for it in plan:
        d = (it["end"] - it["start"]) / it.get("speed", 1.0) if it["kind"] == "clip" else float(it["seconds"])
        dur += d
        if it["kind"] in ("clip", "sidebyside"):
            real += d
    vids = {p: _ffprobe(ROOT / p) for p in ("media/hackathon/demo.mp4", "submission/demo_final.mp4")}
    V["demo"] = dict(dur=dur, real=real, vids=vids)
    mm = lambda x: f"{int(x // 60)}:{int(round(x % 60)):02d}"  # noqa: E731
    vtxt = "; ".join(f"{p}: {mm(v['duration'])} ({v['duration']:.1f} s, {v['w']}x{v['h']})" if v else f"{p}: unreadable or absent"
                     for p, v in vids.items())
    add("DEMO-LENGTH", [(D10, "(2:23, 1600x992, English captions)"), (D10, "About 84 s of the 143 s video"),
                        (README, "(2 min 23 s;"), (README, "2:25, 1080p, real screen recordings")],
        "Demo video", f"{len(plan)} planned segments", "-",
        "ffprobe container duration of the rendered video; plan total = sum of segment durations in demo_plan.json (clip length / speed-up)",
        f"{vtxt}; current demo_plan.json sums to {dur:.1f} s ({mm(dur)}), of which real dashboard + rollout footage {real:.1f} s",
        "none", "secondary", "no (not a study)", "no; measured here with ffprobe",
        "configs/hackathon/demo_plan.json; media/hackathon/demo.mp4; submission/demo_final.mp4",
        "The plan and the rendered file can diverge while the video is being re-cut; quote the ffprobe duration of the "
        "file that is uploaded.", "Demo video under 3 minutes; quote the final file's measured length.")


# ===================================================================================== doc consistency checks
def doc_checks():
    nb, e2 = V.get("neb"), V
    checks = []

    def chk(cid, path, snippet, correct, kind="stale", rx=False):
        lines = doc_lines(path)
        if rx:
            hits = [i for i, l in enumerate(lines, 1) if re.search(snippet, l)]
        else:
            hits = [i for i, l in enumerate(lines, 1) if snippet in l or _norm(snippet) in _norm(l)]
        checks.append((cid, path, hits, snippet, correct, kind))

    if nb:
        pm, pe = nb["per_log"]["main"], nb["per_log"]["explore"]
        usd_main = pm["prompt"] * 0.3e-6 + pm["completion"] * 0.9e-6
        sv = nb["per_model"]["nvidia/Nemotron-3.5-Super-VL"]["calls"]
        lt = nb["per_model"].get("nvidia/Nemotron-3_5-Lightning", {}).get("calls", 0)
        chk("NEB-05-main-calls", D05, '"calls_total": 772', f"main log now {pm['n']} calls (cutoff {pm['last']})")
        chk("NEB-05-explore-calls", D05, '"calls_total": 2380', f"explore log now {pe['n']:,} calls (cutoff {pe['last']})")
        chk("NEB-05-snapshot", D05, "## Measured usage (generated 2026-10-29T02:25:19Z)", f"regenerate; last logged call {nb['last']}")
        chk("NEB-05-main-last", D05, '"last_call_utc": "2026-10-29T02:15:23+00:00"', f"main last logged {pm['last']}")
        chk("NEB-05-explore-last", D05, '"last_call_utc": "2026-10-28T21:56:47+00:00"', f"explore last logged {pe['last']}")
        chk("NEB-05-main-tokens", D05, '"prompt": 2598343', f"main prompt tokens {pm['prompt']:,}, completion {pm['completion']:,}")
        chk("NEB-05-explore-tokens", D05, '"prompt": 6172302', f"explore prompt {pe['prompt']:,}, completion {pe['completion']:,}")
        chk("NEB-05-main-usd", D05, '"usd_estimate_at_catalog_price": 0.8662', f"main ${usd_main:.4f}")
        chk("NEB-05-explore-usd", D05, '"usd_estimate_at_catalog_price": 2.1825', f"explore ${nb['usd'] - usd_main:.4f} at per-model prices")
        chk("NEB-05-tags", D05, '"explore": 2380', f"explore tags now include r1_*, s1_base, s2_base, ... ({pe['n']:,} rows)")
        chk("NEB-05-onelog", D05, "Every call appends one sanitized record to `results/hackathon/nebius_calls.jsonl`",
            "exploratory calls go to results/hackathon/explore/nebius_calls_explore.jsonl", "incomplete")
        for old in ("3,152", "5,392", "$3.60", "10.6 M input"):
            for p in AUDITED_DOCS:
                chk(f"NEB-old-{old}", p, old, f"{nb['total']:,} calls / ${nb['usd']:.2f}")
        for p in AUDITED_DOCS:  # every "N,NNN (logged) (Nebius) calls" that disagrees with the live logs
            for i, line in enumerate(doc_lines(p), 1):
                for m in re.finditer(r"(\d,\d{3})\s+(?:logged\s+)?(?:Nebius\s+)?calls|\((\d,\d{3}), 0 failed\)", line):
                    num = int((m.group(1) or m.group(2)).replace(",", ""))
                    if num not in (nb["total"], pe["n"], sv):
                        checks.append(("NEB-count", p, [i], m.group(0), f"logs hold {nb['total']:,} calls at cutoff {nb['last']}", "stale count"))
        chk("NEB-09-upperbound", D09, "this is an upper bound, since some exploratory calls went to the cheaper Lightning model",
            "Lightning calls are already priced at the Lightning rate; it is an upper bound only because cached-input "
            "discounts are not applied", "wrong reasoning")
        chk("NEB-10-served", D10, r"logged calls, 0 failures, served model `nvidia/Nemotron-3\.5-Super-VL`",
            f"{sv:,} Super-VL + {lt} Lightning", "incomplete", rx=True)
        chk("NEB-figE-snapshot", FIGS, "Main pipeline only (results/hackathon/cost_latency.json, snapshot",
            f"cost_latency.json snapshot 2026-10-29T02:09:57Z (2,593,864 / 96,224 tokens, $0.86); main log now "
            f"{pm['n']} calls, {pm['prompt']:,} / {pm['completion']:,} = ${usd_main:.2f}", "labelled stale snapshot")
    dm = V.get("demo")
    if dm:
        v = dm["vids"].get("media/hackathon/demo.mp4")
        meas = f"media/hackathon/demo.mp4 = {v['duration']:.1f} s {v['w']}x{v['h']}" if v else "media/hackathon/demo.mp4 not measurable"
        for p in AUDITED_DOCS:
            if p == DEMO:
                continue
            for i, line in enumerate(doc_lines(p), 1):
                for m in re.finditer(r"(?<![\d:T.])(\d):(\d\d)(?![\d:])|(\d) min (\d+) s|(\d{3}) s video|1080p", line):
                    checks.append(("DEMO-length", p, [i], m.group(0),
                                   f"{meas}; current demo_plan.json sums to {dm['dur']:.1f} s ({dm['real']:.1f} s real footage)", "verify"))
    if "e2_P" in e2:
        chk("E2-len-09", D09, "Accuracy is unchanged when clip length is neutralised",
            f"0.961 vs 0.964 accuracy; balanced 0.909 vs 0.927; FP 12 vs 9 ({e2['e2_flips']}/330 flips)", "overstated")
        chk("E2-len-07", D07, "4 of 250 judgements (first 250 episodes)",
            f"full split: {e2['e2_flips']} of 330 changed (4 of the first 250)", "stale snapshot")
        chk("E2-len-11a", D11, "The frozen Nemotron v2 prompt was re-run on the 250 eval episodes",
            f"pad201 now covers all 330: original 0.964 [0.938, 0.979] FP 9 FN 3; length-neutral 0.961 [0.934, 0.977] FP 12 FN 1; "
            f"{e2['e2_flips']}/330 flips", "stale snapshot")
        chk("E2-ftype-07", D07, "Nemotron calls most missed grasps and drops \"not_placed\"",
            f"missed grasps {e2['e2_ng']}; drops {e2['e2_dr']}", "wrong")
    chk("ACC-11-090", D11, "0.90–0.96", "eval-split accuracy is 0.918 (Cosmos) to 0.964 (Nemotron): 0.92-0.96", "imprecise")
    chk("HIST-07-45", D07, "These come from the primary Diffusion Policy cohort of the paper's 1,365-checkpoint",
        f"DP cohort = {V.get('hist_dp_runs', '24')} runs; 45 runs = whole study; the 0.019 bullet is pooled over all 45", "scope")
    chk("E4B-07-mismatch", D07, "against 0.10–0.19 for mismatched clips", "no mismatched-clip control in e5_probe.json or any result file", "untraceable")
    chk("E4B-07-S0", D07, "30.8 dB vs 26–27 dB between seeds", "e5_probe.json S0 has one between-seed pair: 26.2 dB", "imprecise")
    chk("PREREG-09", D09, "The exploratory studies are post-hoc, except r1 and c1, which were pre-registered",
        "r1, c1, s1 and s2 were pre-registered (09 itself labels s1/s2 pre-registered)", "contradiction")
    for p in (README, D09, D07, D11, DEMO):
        chk("X2-scope", p, r"16[–-]20\s?% (?:of )?(?:total )?robot time|16[–-]20% less robot time|cuts robot time by 16[–-]20%",
            "time-only horizon (no privileged state): 16.2% at 2/141 false aborts; 20.1% needs privileged state; "
            "200-episode / 20-checkpoint snapshot", "missing qualifiers", rx=True)
        chk("X1-scope", p, r"(?:2[–-]3x|3x) fewer rollouts|rollouts about 2[–-]3x",
            "say: post-hoc simulation, budget to reach excess regret <= 0.02 vs uniform, perfect labels, 56-checkpoint "
            "snapshot; median ratio 3 (p10-p90 2-4)", "missing qualifiers", rx=True)
    chk("SRC-07-every", D07, "Every number below is read from a file in `results/hackathon/`",
        "§A comes from docs/plv_arxiv/main.tex; 12/12 narration and s2 46 vs 19 come from data/outputs; some E4b/r1 details only from verifier reports",
        "overstated")
    chk("VERIF-11", D11, "Each study was implemented by one agent and checked by an\nindependent"[:40],
        "x0 and w4 were not independently verified (11 itself says so)", "overstated")
    return checks


# ===================================================================================== main
def main():
    checks = doc_checks()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in ROWS:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    print(f"wrote {rel(OUT)}: {len(ROWS)} claims")
    nb = V.get("neb")
    if nb:
        print(f"Nebius: {nb['total']:,} calls, ${nb['usd']:.2f} at catalog price, log cutoff {nb['last']}")
    print(f"recompute mismatches vs result files: {len(MISMATCH)}")
    for m in MISMATCH:
        print("  MISMATCH", m)
    for e in ERRORS:
        print("  ERROR", e)
    present = [c for c in checks if c[2]]
    print(f"document checks: {len(present)} still present (of {len(checks)} checked)")
    for cid, path, hits, snip, correct, kind in present:
        print(f"  [{kind}] {cid}: {path}:{','.join(map(str, hits))}  '{snip[:70]}'  -> {correct}")
    missing_quotes = sorted({r["claim_id"] for r in ROWS if "wording not found" in r["exact_wording"]})
    if missing_quotes:
        print("claims whose quoted wording was not found (docs edited since the ledger was written):", ", ".join(missing_quotes))
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
