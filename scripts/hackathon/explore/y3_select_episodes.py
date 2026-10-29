#!/usr/bin/env python3
"""y3_visual_robustness: fixed-rule stratified episode subset (EXPLORATORY, post-hoc).

Rule (no VLM output is read):
  * universe = eval episodes (seeds 200000-200009) of the 25 checkpoints that the
    frozen eval diagnostics cover (checkpoint ids in eval_local_v6.jsonl);
  * strata = (task, verified failure_type) with failure_type in {none, no_grasp, dropped};
  * quotas: can none 12 / no_grasp 10 / dropped 8, square none 12 / no_grasp 9 / dropped 9 (60);
  * order inside a stratum: round-robin over checkpoints (checkpoint order and episode
    order both by sha256("y3|<checkpoint_id>|<env_seed>")), so several checkpoints appear.
  If a stratum has fewer episodes than its quota, all are taken.

  python scripts/hackathon/explore/y3_select_episodes.py
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QUOTA = {("can", "none"): 12, ("can", "no_grasp"): 10, ("can", "dropped"): 8,
         ("square", "none"): 12, ("square", "no_grasp"): 9, ("square", "dropped"): 9}
OUT = ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json"


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    diag_ckpts = sorted({json.loads(l)["checkpoint_id"] for l in
                         (ROOT / "data/outputs/hackathon_cosmos/eval_local_v6.jsonl").read_text().splitlines()})
    strata = defaultdict(list)
    for ck in diag_ckpts:
        summ = json.loads((ROOT / "data/outputs/hackathon_rollouts_eval" / ck / "summary.json").read_text())
        for ep in summ["episodes"]:
            if not ep.get("video"):
                continue
            strata[(ep["task"], ep["failure_type"])].append(ep)
    chosen = []
    counts = {}
    for key, quota in QUOTA.items():
        eps = strata.get(key, [])
        by_ck = defaultdict(list)
        for ep in eps:
            by_ck[ep["checkpoint_id"]].append(ep)
        for ck in by_ck:
            by_ck[ck].sort(key=lambda e: h(f"y3|{e['checkpoint_id']}|{e['env_seed']}"))
        order = sorted(by_ck, key=lambda c: h(f"y3|{c}"))
        picked, r = [], 0
        while len(picked) < quota and any(len(by_ck[c]) > r for c in order):
            for c in order:
                if len(picked) >= quota:
                    break
                if len(by_ck[c]) > r:
                    picked.append(by_ck[c][r])
            r += 1
        counts[f"{key[0]}/{key[1]}"] = {"available": len(eps), "picked": len(picked)}
        for ep in picked:
            chosen.append({k: ep[k] for k in ("checkpoint_id", "checkpoint_path", "task", "env_seed", "success",
                                              "failure_type", "video", "video_frames", "video_fps",
                                              "episode_npz", "n_steps", "render_hw", "camera")})
    out = {
        "name": "y3_visual_robustness",
        "rule": __doc__.split("Rule (no VLM output is read):")[1].split("python scripts")[0].strip(),
        "universe_checkpoints": diag_ckpts,
        "strata": counts,
        "n": len(chosen),
        "n_checkpoints": len({e["checkpoint_id"] for e in chosen}),
        "episodes": chosen,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"n": out["n"], "n_checkpoints": out["n_checkpoints"], "strata": counts}, indent=1))


if __name__ == "__main__":
    main()
