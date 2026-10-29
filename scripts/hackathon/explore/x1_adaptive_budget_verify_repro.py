#!/usr/bin/env python3
"""EXPLORATORY verification of x1_adaptive_budget -- exact reproduction on a frozen snapshot.

Rebuilds the data snapshot x1 saw at its final run (2026-10-28T20:04:47Z):
  * pool logs truncated to the epochs it reported (can_s1 <= 105, can_s2 <= 170, square_s1 <= 150),
    with zero-byte placeholder checkpoint names (the loader only globs file names; no checkpoint
    is read or copied);
  * eval judge JSONL = the *.orig.jsonl backups (labels before the 05:15 KST relabel);
  * eval rollout summaries = summary.orig.json of the 20 eval checkpoints that existed at that time.
Then calls x1's own main() unchanged, writing only into the verification scratch dir.

Usage (env B, repo root):
  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/x1_adaptive_budget_verify_repro.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import x1_adaptive_budget as m  # noqa: E402

SCR = ROOT / "data/outputs/hackathon_explore/x1_adaptive_budget_verification"
CAPS = {"paper_dp_can_abs_ph_seed1_e200_s200_r5_inf100": 105,
        "paper_dp_can_abs_ph_seed2_e200_s200_r10_inf100": 170,
        "paper_dp_square_abs_ph_seed1_e300_s200_r10_inf100": 150}
EVAL_EXCLUDE = {"can_s2_e0180", "can_s1_e0120"}  # created after 20:04:47Z


def build_frozen():
    pool = SCR / "frozen_pool"
    for run, cap in CAPS.items():
        src = ROOT / "data/outputs/hackathon_pool" / run
        dst = pool / run
        (dst / ".hydra").mkdir(parents=True, exist_ok=True)
        (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / ".hydra/config.yaml", dst / ".hydra/config.yaml")
        keep = []
        for line in (src / "logs.json.txt").read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in r and int(r["epoch"]) <= cap:
                keep.append(line)
        (dst / "logs.json.txt").write_text("\n".join(keep) + "\n")
        for f in (src / "checkpoints").glob("epoch=*.ckpt"):
            mm = m.CKPT_RE.search(f.name)
            if mm and int(mm.group("epoch")) <= cap:
                (dst / "checkpoints" / f.name).touch()
    ev = SCR / "frozen_eval"
    for d in sorted((ROOT / "data/outputs/hackathon_rollouts_eval").iterdir()):
        if d.name in EVAL_EXCLUDE or not (d / "summary.orig.json").is_file():
            continue
        (ev / d.name).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(d / "summary.orig.json", ev / d.name / "summary.json")
    return pool, ev


if __name__ == "__main__":
    pool, ev = build_frozen()
    m.JUDGE_FILES = {"cosmos3_edge_local_v6": m.COSMOS / "eval_local_v6.orig.jsonl",
                     "nemotron_nebius_v2": m.COSMOS / "eval_nebius_v2.orig.jsonl"}
    sys.argv = [sys.argv[0], "--pool", str(pool), "--eval-rollouts", str(ev),
                "--out", str(SCR / "repro_frozen.json"), "--fig", str(SCR / "repro_frozen.png")]
    m.main()
