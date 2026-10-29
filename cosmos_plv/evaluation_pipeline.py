"""Stage F: combine cheap screening, recorded rollouts and VLM diagnostics into a
recommendation and a per-checkpoint report.

The recommendation answers: "Which checkpoint should I evaluate next, and why?"
It never reads the target success ``S_i`` (column ``success``); that column is
only shown afterwards as verification when it exists.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

OUTCOME_COLUMNS = ("success",)  # training-time 50-rollout success: never used to choose
CHEAP_LABELS = {"val_action_l1": "Action L1", "open_loop_smoothness": "open-loop smoothness",
                "plv1_smoothness": "PLV-1 smoothness"}


def recommend(screen: pd.DataFrame, run_id: str, q: float = 0.3, cheap_col: str = "val_action_l1",
              vlm_col: Optional[str] = None, failures: Optional[pd.DataFrame] = None) -> Dict[str, object]:
    """Two-stage rule: keep top-q by a cheap cost, then order by VLM-judged success.

    Ties in VLM-judged success are broken by the cheap cost (lower first); the
    kept checkpoints that tie with the pick are returned as ``vlm_tie``.
    ``failures``: per-episode diagnostics with columns checkpoint_id and
    cosmos_failure_type (predicted), used only to explain the choice.  Predicted
    failure types are below a majority-class guess in Experiment 2, so they are
    labelled unreliable and returned separately as ``predicted_failure_modes``.
    """
    g = screen[screen.run_id == run_id].copy()
    if g.empty:
        raise ValueError(f"no checkpoints for run {run_id!r}")
    g = g.drop(columns=[c for c in OUTCOME_COLUMNS if c in g], errors="ignore")
    k = max(1, math.ceil(q * len(g)))
    g = g.sort_values([cheap_col, "checkpoint_path"])
    g["cheap_rank"] = range(1, len(g) + 1)
    kept = g.head(k).copy()
    tied: List[str] = []
    if vlm_col and vlm_col in kept and kept[vlm_col].notna().any():
        kept = kept.sort_values([vlm_col, cheap_col], ascending=[False, True])
        basis = "cheap screen, then VLM-judged success on recorded rollouts"
        tied = kept.checkpoint_id[kept[vlm_col] == kept[vlm_col].iloc[0]].tolist()
    else:
        basis = "cheap screen only (no VLM diagnostics available)"
    pick = kept.iloc[0]
    reasons: List[str] = [
        f"{cheap_col} rank {int(pick.cheap_rank)}/{len(g)} (kept top {k} = {int(q * 100)}%)",
    ]
    if vlm_col and vlm_col in pick and pd.notna(pick[vlm_col]):
        reasons.append(f"VLM judged {pick[vlm_col]:.0%} of its recorded attempts successful")
    if len(tied) > 1:
        reasons.append(f"VLM-judged success ties at {pick[vlm_col]:.0%} for {len(tied)} of the {k} kept checkpoints; "
                       f"the tie was broken by {CHEAP_LABELS.get(cheap_col, cheap_col)} (lower wins)")
    modes: Dict[str, int] = {}
    if failures is not None and len(failures):
        f = failures[(failures.checkpoint_id == pick.checkpoint_id) & (failures.cosmos_failure_type.notna())]
        modes = dict(Counter(x for x in f.cosmos_failure_type if x != "none").most_common())
        if modes:
            reasons.append("predicted failure modes (unreliable: failure-type accuracy is below a majority-class "
                           "guess): " + ", ".join(f"{k} x{v}" for k, v in modes.items()))
    return {
        "run_id": run_id,
        "recommended_checkpoint": pick.checkpoint_id,
        "basis": basis,
        "reasons": reasons,
        "retained": kept.checkpoint_id.tolist(),
        "k": k,
        "n": len(g),
        "vlm_tie": tied,
        "predicted_failure_modes": modes,
    }


def latest_epoch_pick(screen: pd.DataFrame, run_id: str) -> Dict[str, object]:
    """Free baseline: the most-trained checkpoint of the run (no metric, no rollouts, no outcomes)."""
    g = screen[screen.run_id == run_id]
    if g.empty:
        raise ValueError(f"no checkpoints for run {run_id!r}")
    pick = g.sort_values(["epoch", "checkpoint_path"], ascending=[False, True]).iloc[0]
    return {"run_id": run_id, "recommended_checkpoint": pick.checkpoint_id, "epoch": int(pick.epoch),
            "basis": "latest epoch (free baseline)"}


def rule_regrets(ranking: pd.DataFrame) -> pd.DataFrame:
    """Measured regret of each rule's single pick, from hybrid_ranking.csv (Experiment 3).

    One row per rule (``metric``, ``label``), one column per run with that run's
    ``pick_regret@1`` and ``mean_pick_regret`` = mean over runs, as printed by
    analyze_hybrid.py.  Sorted best (lowest) first.
    """
    if ranking is None or not len(ranking) or "pick_regret@1" not in ranking:
        return pd.DataFrame()
    p = ranking[ranking["pick_regret@1"].notna()]
    wide = p.pivot_table(index=["metric", "label"], columns="run_id", values="pick_regret@1", aggfunc="first")
    out = wide.reset_index()
    out.columns.name = None
    out.insert(2, "mean_pick_regret", wide.mean(axis=1).values)
    out.insert(3, "n_runs", wide.notna().sum(axis=1).values)
    return out.sort_values(["mean_pick_regret", "metric"]).reset_index(drop=True)


def failure_type_reliability(exp2: Optional[dict]) -> Dict[str, Dict[str, object]]:
    """Failure-type accuracy on failed episodes vs. always guessing the most common true type.

    ``exp2``: experiment2_backend_comparison.json.  Returns, per judge key that
    predicts failure types, ``acc`` (accuracy on failures), ``majority_type``,
    ``majority_acc`` and ``n_failures`` (both from ``failure_type_confusion``).
    """
    out: Dict[str, Dict[str, object]] = {}
    if not exp2:
        return out
    for key, rows in (exp2.get("overall") or {}).items():
        acc = rows[0].get("failure_type_acc_on_failures") if rows else None
        conf = (exp2.get("failure_type_confusion") or {}).get(key)
        if acc is None or not conf:
            continue
        truth = {t: sum(pred.values()) for t, pred in conf.items() if t != "none"}
        n = sum(truth.values())
        if not n:
            continue
        top = max(truth, key=truth.get)
        out[key] = {"acc": float(acc), "majority_type": top, "majority_acc": truth[top] / n, "n_failures": n}
    return out


# a grasp / pick-up / hold claim, unless its clause is negated or only anticipates it
_GRASP = re.compile(r"\b(?:pick(?:s|ed|ing)?\b(?:\s+\w+){0,3}?\s+up|grasp(?:s|ed|ing)?|grabb?(?:s|ed|ing)?|"
                    r"lift(?:s|ed|ing)?|hold(?:s|ing)?|held|carr(?:y|ies|ied|ying)|"
                    r"clos(?:e|es|ed|ing)\b(?:\s+\w+){0,3}?\s+(?:around|on))\b", re.I)
_NEGATED = re.compile(r"\b(?:never|not|no|nor|without|fail(?:s|ed|ing)?|unable|cannot|attempt(?:s|ed|ing)?|"
                      r"tr(?:y|ies|ied|ying)|ready to|about to)\b|n't\b", re.I)
_CLAUSE = re.compile(r"[.;:,]|\b(?:but|while|though|although|however|whereas)\b", re.I)


def claims_grasp(text: Optional[str]) -> bool:
    """True when a free-text model explanation asserts that the object was grasped, lifted or held.

    Conservative: a clause containing a negation ("never closed around it or
    lifted it") or only an intention ("ready to be grasped") is not a claim.
    """
    if not isinstance(text, str):
        return False
    return any(_GRASP.search(c) and not _NEGATED.search(c) for c in _CLAUSE.split(text))


def checkpoint_report(row: pd.Series, episodes: pd.DataFrame) -> str:
    """Markdown report for one checkpoint (baseline scores, rollouts, Cosmos output, verified outcome)."""
    lines = [f"### {row.checkpoint_id}", "", f"* checkpoint: `{row.checkpoint_path}`",
             f"* run: {row.run_id}, epoch {int(row.epoch)}"]
    for col, label in (("val_action_l1", "Action L1"), ("open_loop_smoothness", "Open-loop smoothness"),
                       ("plv1_smoothness", "PLV-1 smoothness"), ("replan_overlap_h16", "Replan overlap (H=16)")):
        if col in row and pd.notna(row[col]):
            lines.append(f"* {label}: {row[col]:.4g}")
    if "success" in row and pd.notna(row["success"]):
        lines.append(f"* verified training-time success (50 rollouts, seeds 100000-100049): {row['success']:.2f}")
    if len(episodes):
        lines += ["", "| seed | sim outcome | VLM success | VLM failure type | backend |", "|---|---|---|---|---|"]
        for _, e in episodes.iterrows():
            lines.append(f"| {e.env_seed} | {e.truth_failure_type} | {e.cosmos_task_success} | "
                         f"{e.cosmos_failure_type} | {e.backend} |")
    return "\n".join(lines)
