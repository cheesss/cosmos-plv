#!/usr/bin/env python3
"""List every eval episode where a backend's success judgement disagrees with the
simulator, with the object's final position, for qualitative error analysis.
Writes results/hackathon/experiment2_error_analysis.json (Environment A or B)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILES = {"nebius_v2": "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl",
         "local_v6": "data/outputs/hackathon_cosmos/eval_local_v6.jsonl"}
out = []
for name, f in FILES.items():
    p = ROOT / f
    if not p.exists():
        continue
    for line in open(p):
        r = json.loads(line)
        if not r.get("scored") or bool(r["cosmos_task_success"]) == bool(r["truth_success"]):
            continue
        summ = json.loads((ROOT / f"data/outputs/hackathon_rollouts_eval/{r['checkpoint_id']}/summary.json").read_text())
        e = [x for x in summ["episodes"] if x["env_seed"] == r["env_seed"]][0]
        d = np.load(ROOT / e["episode_npz"])
        out.append({"backend": name, "checkpoint_id": r["checkpoint_id"], "env_seed": r["env_seed"], "task": r["task"],
                    "truth_success": r["truth_success"], "truth_failure_type": r["truth_failure_type"],
                    "pred_success": r["cosmos_task_success"], "pred_failure_type": r.get("cosmos_failure_type"),
                    "object_start_xyz": np.round(d["object_pos"][0], 3).tolist(),
                    "object_end_xyz": np.round(d["object_pos"][-1], 3).tolist(),
                    "first_success_step": e["first_success_step"], "n_steps": e["n_steps"],
                    "explanation": r.get("cosmos_explanation"), "video": r["video"]})
(ROOT / "results/hackathon/experiment2_error_analysis.json").write_text(json.dumps(out, indent=2))
print(len(out), "disagreements written")
