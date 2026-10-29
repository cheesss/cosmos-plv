#!/usr/bin/env python3
"""Verification add-on for x2_early_stop (EXPLORATORY): leave-one-checkpoint-out (LOCO) and
leave-one-run-out (LORO, can only) fitting of the Part-1 abort rules, on the frozen 20-checkpoint
eval snapshot. Seed cross-fit in x2 fits on other episodes of the SAME checkpoints; this tests
transfer to an unseen checkpoint / run.  CPU only.

  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x2_early_stop_verify_loco.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
import x2_early_stop_analyze as A  # noqa: E402
import x2_early_stop_verify as V  # noqa: E402

ev = V.frozen_load_split("eval")
out = {}
for fam in ("time_only", "state_combo"):
    for scheme in ("loco", "loro_can"):
        ab = [None] * len(ev)
        chosen = []
        for task in ("can", "square"):
            E = [e for e in ev if e["task"] == task]
            if scheme == "loco":
                groups = sorted({e["checkpoint_id"] for e in E})
                key = lambda e: e["checkpoint_id"]  # noqa: E731
            else:
                if task != "can":
                    continue
                groups = sorted({e["run"] for e in E})
                key = lambda e: e["run"]  # noqa: E731
            for g in groups:
                fit = [e for e in E if key(e) != g]
                params, rule = A.select(fit, fam, 0)
                chosen.append({"task": task, "held_out": g, "params": params})
                for i, e in enumerate(ev):
                    if e["task"] == task and key(e) == g:
                        ab[i] = rule(e)
        idx = [i for i, e in enumerate(ev) if (scheme == "loco" or e["task"] == "can")]
        E2 = [ev[i] for i in idx]
        ab2 = [ab[i] for i in idx]
        st = V.policy_stats(E2, ab2)
        st["by_task"] = {t: V.policy_stats([e for e in E2 if e["task"] == t], [a for e, a in zip(E2, ab2) if e["task"] == t])
                         for t in ("can", "square") if any(e["task"] == t for e in E2)}
        st["ranking"] = A.ranking_effect(E2, ab2)["per_run"]
        st["chosen"] = chosen
        out[f"{scheme}/{fam}"] = st
p = ROOT / "results/hackathon/explore/x2_early_stop_verification_loco.json"
p.write_text(json.dumps(A.strip(out), indent=1, default=str))
for k, s in out.items():
    print(k, s["n"], s["saved_frac"], s["saved_frac_ci95_cluster_boot"], f"FA {s['false_aborts']}/{s['n_successes']}",
          s["false_abort_rate_ci95_cluster_boot"], s["false_aborts_by_checkpoint"],
          {t: (x["saved_frac"], x["false_aborts"], x["n_successes"]) for t, x in s["by_task"].items()}, s["ranking"])
    print("   ", [(c["held_out"], c["params"]) for c in s["chosen"]])
