#!/usr/bin/env python3
"""Re-annotate recorded episodes from their saved state traces with the CURRENT
failure_annotator, and refresh the truth fields of every diagnostic JSONL.

Idempotent. The first run keeps the original files as summary.orig.json and
<name>.orig.jsonl. Success labels come from robosuite's own check and do not
change; only the derived failure taxonomy (grasped/lifted/dropped/failure_type) can.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cosmos_plv.failure_annotator import annotate_episode  # noqa: E402

KEYS = ("success", "first_success_step", "reached", "grasped", "lifted", "dropped",
        "max_object_height_m", "object_xy_displacement_m", "failure_type", "n_steps")


def reannotate(root: Path):
    truth, changes = {}, Counter()
    for summ in sorted(root.glob("*/summary.json")):
        orig = summ.with_name("summary.orig.json")
        if not orig.exists():
            shutil.copy2(summ, orig)
        js = json.loads(summ.read_text())
        for ep in js["episodes"]:
            d = np.load(ROOT / ep["episode_npz"])
            ann = annotate_episode({k: d[k] for k in ("eef_pos", "object_pos", "grasped", "success")})
            if ann["failure_type"] != ep["failure_type"]:
                changes[(ep["task"], ep["failure_type"], ann["failure_type"])] += 1
            assert ann["success"] == ep["success"], "success label must not change"
            ep.update({k: ann[k] for k in KEYS})
            truth[(ep["checkpoint_id"], ep["env_seed"])] = ep
        js["annotator"] = "failure_annotator settled-height reference (SETTLE_STEPS=10)"
        summ.write_text(json.dumps(js, indent=2, sort_keys=True, default=float))
    return truth, changes


def refresh_jsonl(path: Path, truth):
    orig = path.with_name(path.stem + ".orig.jsonl")
    if not orig.exists():
        shutil.copy2(path, orig)
    rows, n = [], 0
    for line in path.read_text().splitlines():
        r = json.loads(line)
        t = truth.get((r["checkpoint_id"], r["env_seed"]))
        if t is not None:
            r.update({"truth_success": t["success"], "truth_failure_type": t["failure_type"],
                      "truth_grasped": t["grasped"], "truth_lifted": t["lifted"], "truth_dropped": t["dropped"]})
            if r.get("scored"):
                if r.get("cosmos_failure_type") is not None:
                    r["failure_type_correct"] = r["cosmos_failure_type"] == t["failure_type"]
                for f, tk in (("lifted_correct", "lifted"), ("dropped_correct", "dropped"), ("contact_correct", "grasped")):
                    src = {"lifted_correct": "cosmos_object_lifted", "dropped_correct": "cosmos_object_dropped",
                           "contact_correct": "cosmos_contact_observed"}[f]
                    if r.get(src) is not None:
                        r[f] = bool(r[src]) == bool(t[tk])
            n += 1
        rows.append(r)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return n


if __name__ == "__main__":
    truth, changes = {}, Counter()
    for d in ("data/outputs/hackathon_rollouts_eval", "data/outputs/hackathon_rollouts_dev"):
        t, c = reannotate(ROOT / d)
        truth.update(t); changes.update(c)
    print("label changes (task, old, new):", dict(changes))
    for p in sorted((ROOT / "data/outputs/hackathon_cosmos").glob("*.jsonl")):
        if p.name.endswith(".orig.jsonl") or "infra_failures" in p.name:
            continue
        print(p.name, "rows refreshed:", refresh_jsonl(p, truth))
