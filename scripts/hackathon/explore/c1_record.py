#!/usr/bin/env python3
"""c1 coverage shift, step 1 — record the pre-registered Can rollouts (Environment A, GPU).
Protocol: docs/hackathon/explore/c1_coverage_shift_protocol.md

One checkpoint per call (fresh process; avoids the EGL render-context growth noted in record_eval_loop.sh).
Recorder settings are those of the Experiment 2 eval set (record_eval_set.py), only the seeds differ.
Output: data/outputs/hackathon_explore/c1_coverage/<checkpoint_id>/{summary.json,episodes/,videos/}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from cosmos_plv.checkpoint_adapter import discover_checkpoints  # noqa: E402
from cosmos_plv.provenance import environment_info, git_commit, utc_now, write_json  # noqa: E402
from cosmos_plv.rollout_recorder import RecorderConfig, record_checkpoint  # noqa: E402

OUT = ROOT / "data/outputs/hackathon_explore/c1_coverage"
STRATA = {
    "M": (["can_s1_e0015", "can_s2_e0020", "can_s1_e0020", "can_s1_e0025", "can_s1_e0030"], 12),
    "U": (["can_s2_e0040", "can_s2_e0030", "can_s2_e0050", "can_s1_e0040", "can_s1_e0045", "can_s1_e0095",
           "can_s2_e0060", "can_s1_e0035", "can_s1_e0050", "can_s1_e0055"], 16),
    "T": ([f"can_s1_e{e:04d}" for e in (60, 65, 70, 75, 80, 85, 105, 130, 150, 185)]
          + [f"can_s2_e{e:04d}" for e in (70, 80, 90, 110, 130, 150)], 25),
}
SEED0 = 600000


def plan():
    return [(ck, s, n) for s, (cks, n) in STRATA.items() for ck in cks]


def wait_for_gpu(need_mib):
    """Serialise start-up across workers: wait for enough free GPU memory (about 350 MiB per batched env with its
    EGL context, measured on the first workers), and keep the lock for 90 s while this process allocates."""
    import fcntl
    import subprocess
    import threading
    import time

    OUT.mkdir(parents=True, exist_ok=True)
    fd = open(OUT / ".gpu_start.lock", "w")
    fcntl.flock(fd, fcntl.LOCK_EX)
    while True:
        free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
        if free >= need_mib:
            break
        time.sleep(20)
    threading.Timer(90, lambda: fcntl.flock(fd, fcntl.LOCK_UN)).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="record this checkpoint id; omit to list the plan")
    a = ap.parse_args()
    if not a.checkpoint:
        for ck, s, n in plan():
            print(ck, s, n, "done" if (OUT / ck / "summary.json").exists() else "todo")
        return
    stratum, n = next((s, n) for ck, s, n in plan() if ck == a.checkpoint)
    out = OUT / a.checkpoint
    if (out / "summary.json").exists():
        print("already recorded", a.checkpoint)
        return
    wait_for_gpu(need_mib=1000 + 350 * n)
    rec = next(r for r in discover_checkpoints(ROOT / "data/outputs/hackathon_pool") if r.checkpoint_id == a.checkpoint)
    seeds = list(range(SEED0, SEED0 + n))
    cfg = RecorderConfig(batch_size=len(seeds))
    rows = record_checkpoint(ROOT / rec.checkpoint_path, out, seeds, seeds, cfg, {}, checkpoint_id=rec.checkpoint_id)
    write_json(out / "summary.json", {
        "study": "c1_coverage_shift", "stratum": stratum,
        "checkpoint": rec.to_dict(), "checkpoint_id": rec.checkpoint_id, "seeds": seeds, "video_seeds": seeds,
        "recorder": cfg.__dict__, "git_commit": git_commit(), "env": environment_info(),
        "finished_utc": utc_now(), "episodes": rows,
        "success_rate": sum(x["success"] for x in rows) / len(rows),
    })
    print(json.dumps({"checkpoint_id": rec.checkpoint_id, "stratum": stratum, "q": rec.train_rollout_success,
                      "success": sum(x["success"] for x in rows) / len(rows)}), flush=True)


if __name__ == "__main__":
    main()
