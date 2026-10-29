#!/usr/bin/env python3
"""Verification add-on for x2_early_stop side finding (EXPLORATORY): re-annotate failure types with the
settled object height (median of steps 5..10) instead of step 0, by monkeypatching a COPY of the trace
(the annotator file is not modified), and recompute Nebius v2 / Cosmos v6 failure-type accuracy.  CPU only.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x2_early_stop_verify_relabel.py
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cosmos_plv.failure_annotator import annotate_episode  # noqa: E402

lab = {}
for p in sorted((ROOT / "data/outputs/hackathon_rollouts_eval").glob("*/summary.json")):
    for ep in json.loads(p.read_text())["episodes"]:
        z = np.load(ROOT / ep["episode_npz"])
        tr = {k: z[k] for k in ("eef_pos", "object_pos", "grasped", "success")}
        a0 = annotate_episode(tr)
        obj = tr["object_pos"].astype(float).copy()
        zs = float(np.median(obj[5:11, 2]))
        obj[0, 2] = zs  # annotator reads its reference height from row 0 only
        tr2 = dict(tr, object_pos=obj)
        a1 = annotate_episode(tr2)
        lab[(ep["checkpoint_id"], int(ep["env_seed"]))] = (ep["task"], ep["failure_type"], a0["failure_type"], a1["failure_type"])
print("summary label == annotator(step0):", sum(v[1] == v[2] for v in lab.values()), "/", len(lab))
print("square relabel transitions:", dict(Counter((v[1], v[3]) for v in lab.values() if v[0] == "square" and v[1] != "none")))
print("can relabel transitions:", dict(Counter((v[1], v[3]) for v in lab.values() if v[0] == "can" and v[1] != "none")))
out = {}
for fn in ("eval_nebius_v2.jsonl", "eval_local_v6.jsonl"):
    rows = [json.loads(l) for l in (ROOT / "data/outputs/hackathon_cosmos" / fn).read_text().splitlines()]
    rows = [r for r in rows if r.get("scored", True) and not r["truth_success"]]
    res = {}
    for t in ("all", "square"):
        rr = [r for r in rows if t == "all" or r["task"] == t]
        rr = [r for r in rr if (r["checkpoint_id"], int(r["env_seed"])) in lab]
        old = sum(r.get("cosmos_failure_type") == r["truth_failure_type"] for r in rr)
        new = sum(r.get("cosmos_failure_type") == lab[(r["checkpoint_id"], int(r["env_seed"]))][3] for r in rr)
        res[t] = {"n_failures": len(rr), "acc_old_labels": round(old / len(rr), 4) if rr else None,
                  "acc_settled_labels": round(new / len(rr), 4) if rr else None,
                  "pred_counts": dict(Counter(r.get("cosmos_failure_type") for r in rr))}
    out[fn] = res
    print(fn, json.dumps(res))
(ROOT / "results/hackathon/explore/x2_early_stop_verification_relabel.json").write_text(json.dumps(
    {"relabel_transitions_square": {f"{a}->{b}": c for (a, b), c in Counter((v[1], v[3]) for v in lab.values() if v[0] == "square" and v[1] != "none").items()},
     "relabel_transitions_can": {f"{a}->{b}": c for (a, b), c in Counter((v[1], v[3]) for v in lab.values() if v[0] == "can" and v[1] != "none").items()},
     "failure_type_accuracy": out}, indent=1))
