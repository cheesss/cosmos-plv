#!/usr/bin/env python3
"""Checkpoint discovery + validation manifest for the hackathon pool.

Writes results/hackathon/checkpoint_manifest.csv and checkpoint_manifest.json:
one row per saved checkpoint with run, seed, epoch, logged validation loss, train
loss, training-time success (50 rollouts), sha256, size and wall-clock times.
Validation: every checkpoint filename must parse, its epoch must have a logged
rollout, n_test must be 50, and the executable chunking must match the paper
(horizon 16, n_obs 2, n_action 8).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cosmos_plv.checkpoint_adapter import discover_checkpoints  # noqa: E402
from cosmos_plv.provenance import git_commit, utc_now, write_json  # noqa: E402


def mtime(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(timespec="seconds")


def main():
    pool = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/outputs/hackathon_pool")
    recs = discover_checkpoints(pool, hash_files=True)
    rows, problems = [], []
    runs = {}
    for r in recs:
        run_dir = ROOT / r.run_dir
        if r.run_dir not in runs:
            start = (run_dir / ".hydra" / "config.yaml").stat().st_mtime
            done = run_dir / ".paper_run_done"
            log = (run_dir / "train.log").read_text(errors="ignore") if (run_dir / "train.log").exists() else ""
            m = re.search(r"number of parameters: ([0-9.e+]+)", log)
            runs[r.run_dir] = {
                "run_name": r.run_name, "run_dir": r.run_dir, "task": r.task, "seed": r.seed,
                "started_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(timespec="seconds"),
                "finished_utc": mtime(done) if done.exists() else None,
                "wall_hours": round((done.stat().st_mtime - start) / 3600, 3) if done.exists() else None,
                "n_parameters": float(m.group(1)) if m else None,
                "config": r.config,
                "_start": start,
            }
        ck = ROOT / r.checkpoint_path
        row = r.to_dict()
        row.pop("config")
        row["saved_utc"] = mtime(ck)
        row["hours_since_run_start"] = round((ck.stat().st_mtime - runs[r.run_dir]["_start"]) / 3600, 3)
        row["architecture"] = "ConditionalUnet1D (DiffusionUnetLowdimPolicy), EMA weights"
        rows.append(row)
        c = r.config
        if (c.get("horizon"), c.get("n_obs_steps"), c.get("n_action_steps")) != (16, 2, 8):
            problems.append(f"{r.checkpoint_id}: chunking {c.get('horizon')}/{c.get('n_obs_steps')}/{c.get('n_action_steps')}")
        if r.n_train_rollouts != 50:
            problems.append(f"{r.checkpoint_id}: n_test={r.n_train_rollouts}")
        if r.val_loss is None:
            problems.append(f"{r.checkpoint_id}: no logged val_loss for epoch {r.epoch}")
    df = pd.DataFrame(rows).sort_values(["run_name", "epoch"])
    out = ROOT / "results/hackathon"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "checkpoint_manifest.csv", index=False)
    write_json(out / "checkpoint_manifest.json", {
        "generated_utc": utc_now(), "git_commit": git_commit(), "pool": str(pool.relative_to(ROOT)),
        "n_checkpoints": len(df), "validation_problems": problems,
        "runs": [{k: v for k, v in x.items() if not k.startswith("_")} for x in runs.values()],
        "gpu_utilization": "not recorded per run; 1x RTX 3090 shared by 3 concurrent runs and rollout jobs",
    })
    print(df.groupby("run_name").agg(n=("epoch", "size"), last=("epoch", "max"),
                                     max_success=("train_rollout_success", "max")).to_string())
    print("validation problems:", problems or "none")


if __name__ == "__main__":
    main()
