#!/usr/bin/env python3
"""Verification add-on for x2_early_stop (EXPLORATORY): score Nemotron B_evidence evidence fields
(object_lifted, gripper_closed_on_object) against privileged state at the cut step, from saved outputs
only (no new calls), and print the trace of the single high-confidence false abort.  CPU only.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x2_early_stop_verify_evidence.py
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
import x2_early_stop_verify as V  # noqa: E402

ev = {(e["checkpoint_id"], e["env_seed"]): e for e in V.frozen_load_split("eval")}
rows = [json.loads(l) for l in (ROOT / "data/outputs/hackathon_explore/x2_early_stop/nebius_prefix_calls.jsonl").read_text().splitlines()]
rows = [r for r in rows if r["split"] == "eval" and r["variant"] == "B_evidence"]
checks = {
    "object_lifted_vs_h_at_cut_ge_5cm": ("object_lifted", lambda e, c: bool(e["h"][c] >= 0.05)),
    "object_lifted_vs_ever_lifted_by_cut": ("object_lifted", lambda e, c: bool((e["h"][10:c + 1] >= 0.05).any())),
    "gripper_closed_vs_finger_contact_last_3_steps": ("gripper_closed_on_object", lambda e, c: bool(e["grasped"][max(0, c - 2):c + 1].any())),
}
res = {}
for name, (fld, fn) in checks.items():
    C = Counter()
    for r in rows:
        e = ev[(r["checkpoint_id"], r["env_seed"])]
        C[(fn(e, r["cut_step"]), r["parsed"].get(fld))] += 1
    res[name] = {f"truth_{a}_pred_{b}": v for (a, b), v in sorted(C.items(), key=str)}
e = ev[("can_s1_e0020", 200004)]
res["high_conf_false_abort_trace_can_s1_e0020_seed200004"] = {
    "first_success_step": e["fss"], "first_contact_step": int(np.argmax(e["grasped"])),
    "max_h_before_cut_m": round(float(e["h"][10:81].max()), 3), "h_at_cut80": round(float(e["h"][80]), 3),
    "contact_at_cut80": bool(e["grasped"][80]), "h_at_160": round(float(e["h"][160]), 3), "contact_at_160": bool(e["grasped"][160]),
    "calls": [{"cut": r["cut_step"], "will_succeed": r["will_succeed"], "confidence": r["confidence"], "reason": r["parsed"].get("reason")}
              for r in rows if r["checkpoint_id"] == "can_s1_e0020" and r["env_seed"] == 200004]}
p = ROOT / "results/hackathon/explore/x2_early_stop_verification_evidence.json"
p.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
