if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


METRIC_DIRECTIONS = {
    "train_loss": "min",
    "val_loss_logged": "min",
    "val_action_mse": "min",
    "val_action_l1": "min",
    "val_action_l2": "min",
    "teacher_forced_latent_mse": "min",
    "teacher_forced_latent_l2": "min",
    "teacher_forced_latent_uncertainty": "min",
    "plv_lite_score_raw": "min",
    "plv_lite_score_norm": "min",
    "plv_final_score_raw": "min",
    "plv_final_score_norm": "min",
    "plv_final_no_progress_norm": "min",
    "plv_risk_score_raw": "min",
    "plv_risk_score_norm": "min",
    "plv_risk_no_progress_score_raw": "min",
    "plv_risk_no_progress_norm": "min",
    "plv_stability_score_norm": "min",
    "plv_uncertainty_score_norm": "min",
    "plv_smoothness_score_norm": "min",
    "plv_support_smooth_score_norm": "min",
    "plv_no_support_norm": "min",
    "plv_no_progress_norm": "min",
    "plv_no_uncertainty_norm": "min",
    "plv_no_smoothness_norm": "min",
    "support_drift": "min",
    "support_drift_final": "min",
    "support_drift_max": "min",
    "dynamics_uncertainty": "min",
    "dynamics_uncertainty_final": "min",
    "dynamics_uncertainty_max": "min",
    "smoothness": "min",
    "open_loop_smoothness": "min",
    "open_loop_action_delta": "min",
    "open_loop_action_l2": "min",
    "progress_delta": "max",
    "progress_delta_final": "max",
    "progress_delta_min": "max",
}


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def collect_xy(rows: List[Dict[str, str]], metric: str, target: str) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        x = parse_float(row.get(metric))
        y = parse_float(row.get(target))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def safe_corr(fn, x, y):
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return "", ""
    result = fn(x, y)
    statistic = getattr(result, "statistic", None)
    pvalue = getattr(result, "pvalue", None)
    if statistic is None:
        statistic = getattr(result, "correlation", result[0])
    if pvalue is None:
        pvalue = result[1]
    return float(statistic), float(pvalue)


def regret_at_k(rows: List[Dict[str, str]], metric: str, target: str, direction: str, k: int):
    usable = []
    for row in rows:
        x = parse_float(row.get(metric))
        y = parse_float(row.get(target))
        if x is None or y is None:
            continue
        usable.append((x, y))
    if len(usable) == 0:
        return ""
    best = max(y for _, y in usable)
    reverse = direction == "max"
    top = sorted(usable, key=lambda item: item[0], reverse=reverse)[: min(k, len(usable))]
    selected = max(y for _, y in top)
    return float(best - selected)


def analyze(rows: List[Dict[str, str]], group_keys: List[str], target: str, ks: List[int]):
    grouped = {("all",): rows}
    if group_keys:
        grouped = {}
        for row in rows:
            key = tuple(row.get(k, "") for k in group_keys)
            grouped.setdefault(key, []).append(row)

    out_rows = []
    for group, group_rows in grouped.items():
        for metric, direction in METRIC_DIRECTIONS.items():
            x, y = collect_xy(group_rows, metric, target)
            x_oriented = -x if direction == "min" else x
            spearman, spearman_p = safe_corr(spearmanr, x_oriented, y)
            pearson, pearson_p = safe_corr(pearsonr, x_oriented, y)
            raw_spearman, raw_spearman_p = safe_corr(spearmanr, x, y)
            raw_pearson, raw_pearson_p = safe_corr(pearsonr, x, y)
            result = {
                "group": "/".join(group),
                "metric": metric,
                "direction": direction,
                "n": int(x.size),
                "target": target,
                "spearman": spearman,
                "spearman_p": spearman_p,
                "pearson": pearson,
                "pearson_p": pearson_p,
                "raw_spearman": raw_spearman,
                "raw_spearman_p": raw_spearman_p,
                "raw_pearson": raw_pearson,
                "raw_pearson_p": raw_pearson_p,
            }
            for k in ks:
                result[f"regret_at_{k}"] = regret_at_k(group_rows, metric, target, direction, k)
            out_rows.append(result)
    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="test_mean_score")
    parser.add_argument("--group-keys", nargs="*", default=[])
    parser.add_argument("--ks", nargs="*", type=int, default=[1, 3, 5])
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = list(csv.DictReader(input_path.open()))
    out_rows = analyze(rows, group_keys=args.group_keys, target=args.target, ks=args.ks)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else [
        "group", "metric", "direction", "n", "target", "spearman", "spearman_p", "pearson", "pearson_p"
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
