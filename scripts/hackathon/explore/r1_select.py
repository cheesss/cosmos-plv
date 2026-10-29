#!/usr/bin/env python3
"""r1 counterfactual replanning, step 1 — select replanning events (CPU, any environment).

Rules are fixed in docs/hackathon/explore/r1_counterfactual_replan_protocol.md.
No branch outcome is used. Output: results/hackathon/explore/r1_events_{dev,eval}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cosmos_plv.cheap_metrics import replan_alignment  # noqa: E402

SPLITS = {"eval": ROOT / "data/outputs/hackathon_rollouts_eval", "dev": ROOT / "data/outputs/hackathon_rollouts_dev"}
DIM_GROUPS = {"pos": slice(0, 3), "rot": slice(3, 9), "grip": slice(9, 10), "all": slice(0, 10)}


def load(split):
    eps = []
    for summ in sorted(SPLITS[split].glob("*/summary.json")):
        s = json.loads(summ.read_text())
        for ep in s["episodes"]:
            npz = ROOT / ep["episode_npz"]
            if npz.is_file():
                d = np.load(npz)
                eps.append({"meta": ep, "action_pred": d["action_pred"], "plan_t": d["plan_t"], "n_actions": len(d["action"])})
    return eps


def task_scales():
    """Per-dim std of the predicted actions pooled over all eval episodes of a task (outcome-free; as x3)."""
    pooled = {}
    for e in load("eval"):
        pooled.setdefault(e["meta"]["task"], []).append(e["action_pred"].reshape(-1, 10))
    return {t: np.concatenate(v).std(axis=0) + 1e-6 for t, v in pooled.items()}


def boundaries(e, scale):
    m = e["meta"]
    al = replan_alignment(m["horizon"], m["n_obs_steps"], m["n_action_steps"])
    ap, pt = e["action_pred"].astype(np.float64), e["plan_t"]
    rows = []
    for c in range(len(ap) - 1):
        T = int(pt[c + 1])
        if T - int(pt[c]) != m["n_action_steps"] or T < 16 or T > e["n_actions"] - 8:
            continue
        old = ap[c, al["old_slice"][0]:al["old_slice"][1]] / scale
        new = ap[c + 1, al["new_slice"][0]:al["new_slice"][1]] / scale
        r = {"c": c, "T": T}
        for g, sl in DIM_GROUPS.items():
            r[f"d_{g}"] = float(np.abs(old[:, sl] - new[:, sl]).mean())
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    a = ap.parse_args()
    scales = task_scales()
    rng = np.random.default_rng(0)
    eps = [e for e in load(a.split) if not e["meta"]["checkpoint_id"].endswith("_e0000")]
    ctrl_eps = set(rng.choice(len(eps), size=len(eps) // 3, replace=False).tolist())
    events = []
    for i, e in enumerate(eps):
        m = e["meta"]
        rows = boundaries(e, scales[m["task"]])
        if not rows:
            continue
        base = {"checkpoint_id": m["checkpoint_id"], "checkpoint_path": m["checkpoint_path"], "task": m["task"],
                "env_seed": int(m["env_seed"]), "episode_npz": m["episode_npz"], "video": m["video"],
                "video_fps": m["video_fps"], "episode_success": bool(m["success"]), "episode_failure_type": m["failure_type"]}
        hi = max(rows, key=lambda r: r["d_all"])
        events.append(base | hi | {"kind": "HIGH"})
        if i in ctrl_eps:
            med = np.median([r["d_all"] for r in rows])
            low = [r for r in rows if r["d_all"] <= med and r["c"] != hi["c"]]
            if low:
                events.append(base | low[int(rng.integers(len(low)))] | {"kind": "CONTROL"})
    rng2 = np.random.default_rng(0)
    for ev in events:  # J2 label order: is the NEW plan shown as "1"?
        ev["new_is_1"] = bool(rng2.integers(2))
    out = ROOT / f"results/hackathon/explore/r1_events_{a.split}.json"
    out.write_text(json.dumps({"protocol": "docs/hackathon/explore/r1_counterfactual_replan_protocol.md",
                               "split": a.split, "n_episodes": len(eps), "events": events}, indent=1))
    k = [e["kind"] for e in events]
    print(f"{a.split}: {len(eps)} episodes, {k.count('HIGH')} HIGH, {k.count('CONTROL')} CONTROL -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
