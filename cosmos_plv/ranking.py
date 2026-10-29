"""Checkpoint-ranking statistics, following the PLV paper conventions.

* Oriented Spearman: rho(-cost, success) for cost metrics (lower is better),
  rho(score, success) for score metrics (higher is better).
* Regret@q = S_best - max S over the ceil(q*N) best-ranked checkpoints
  (ties broken by checkpoint path, as in the paper).
* Statistics are computed WITHIN a training run (task/seed), then averaged.
* Uncertainty: percentile bootstrap over checkpoints within each run.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / math.sqrt((rx * rx).sum() * (ry * ry).sum()))


def oriented_score(values: Sequence[float], lower_is_better: bool) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return -v if lower_is_better else v


def regret_at_q(score: Sequence[float], success: Sequence[float], q: float,
                tiebreak: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """score: higher = better ranked.  Returns regret, k, N."""
    score = np.asarray(score, dtype=float)
    success = np.asarray(success, dtype=float)
    m = np.isfinite(score) & np.isfinite(success)
    idx = np.nonzero(m)[0]
    n = len(idx)
    if n == 0:
        return {"regret": float("nan"), "k": 0, "n": 0}
    k = max(1, int(math.ceil(q * n)))
    keys = [str(tiebreak[i]) if tiebreak is not None else f"{i:08d}" for i in idx]
    order = sorted(range(n), key=lambda j: (-score[idx[j]], keys[j]))
    kept = [idx[j] for j in order[:k]]
    best = float(success[idx].max())
    return {"regret": best - float(success[kept].max()), "k": k, "n": n,
            "best_success": best, "kept_best_success": float(success[kept].max())}


def bootstrap_ci(fn, n: int, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.integers(0, n, size=n)
        v = fn(sample)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def evaluate_metric(df, metric: str, lower_is_better: bool, success_col: str = "success",
                    group_col: str = "run_id", qs: Iterable[float] = (0.3,),
                    n_boot: int = 2000) -> List[Dict[str, object]]:
    """Per-run rows plus a 'mean' row (unweighted mean across runs)."""
    rows: List[Dict[str, object]] = []
    qs = list(qs)
    for run, g in df.groupby(group_col, sort=True):
        g = g[np.isfinite(g[metric].astype(float)) & np.isfinite(g[success_col].astype(float))]
        if len(g) < 3:
            continue
        s = oriented_score(g[metric].values, lower_is_better)
        y = g[success_col].values.astype(float)
        rho = spearman(s, y)
        ci = bootstrap_ci(lambda ix: spearman(s[ix], y[ix]), len(g), n_boot=n_boot)
        row: Dict[str, object] = {"metric": metric, "run_id": run, "n_checkpoints": len(g),
                                  "spearman": rho, "spearman_ci_lo": ci[0], "spearman_ci_hi": ci[1]}
        tb = g["checkpoint_path"].values if "checkpoint_path" in g else None
        for q in qs:
            r = regret_at_q(s, y, q, tb)
            row[f"regret@{int(round(q * 100))}"] = r["regret"]
            row[f"k@{int(round(q * 100))}"] = r["k"]
        rows.append(row)
    if rows:
        mean = {"metric": metric, "run_id": "mean_over_runs", "n_checkpoints": int(sum(r["n_checkpoints"] for r in rows))}
        for key in rows[0]:
            if key.startswith(("spearman", "regret@")) and not key.endswith(("ci_lo", "ci_hi")):
                mean[key] = float(np.nanmean([r[key] for r in rows]))
        rows.append(mean)
    return rows
