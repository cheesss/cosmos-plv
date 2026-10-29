#!/usr/bin/env python3
"""Record the frozen evaluation rollout set (docs/hackathon/06_experiment_protocol.md).

Skips checkpoints already recorded; safe to re-run while training continues.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from cosmos_plv.checkpoint_adapter import discover_checkpoints  # noqa: E402
from cosmos_plv.provenance import environment_info, git_commit, utc_now, write_json  # noqa: E402
from cosmos_plv.rollout_recorder import RecorderConfig, record_checkpoint  # noqa: E402

EVAL_EPOCH_STEP = {"can": 20, "square": 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/outputs/hackathon_pool")
    ap.add_argument("--out", default="data/outputs/hackathon_rollouts_eval")
    ap.add_argument("--seeds", default="200000:200010")
    ap.add_argument("--max-checkpoints", type=int, default=0)
    ap.add_argument("--wait-for-pool", action="store_true",
                    help="keep polling until every expected eval checkpoint is recorded")
    ap.add_argument("--expected", default="can_s1:200,can_s2:200,square_s1:300",
                    help="run:last_epoch pairs used with --wait-for-pool")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split(":"))
    seeds = list(range(a, b))
    expected = set()
    for item in args.expected.split(","):
        run, last = item.split(":")
        step = EVAL_EPOCH_STEP[run.split("_")[0]]
        expected |= {f"{run}_e{e:04d}" for e in range(0, int(last) + 1, step)}
    cfg = RecorderConfig(batch_size=len(seeds))
    cache: dict = {}
    n = 0
    while True:
        did = record_pass(args, seeds, cfg, cache)
        n += did
        done = {p.parent.name for p in Path(args.out).glob("*/summary.json")}
        if not args.wait_for_pool or expected <= done:
            break
        if did == 0:
            import time
            time.sleep(120)
    print(f"recorded {n} checkpoints", flush=True)


def record_pass(args, seeds, cfg, cache) -> int:
    import time

    recs = [r for r in discover_checkpoints(Path(args.pool)) if r.epoch % EVAL_EPOCH_STEP[r.task] == 0
            and time.time() - (ROOT / r.checkpoint_path).stat().st_mtime > 60]  # skip files still being written
    n = 0
    ordered = sorted(recs, key=lambda r: (r.epoch, r.task, r.seed))
    for i, r in enumerate(ordered):
        if i % args.num_shards != args.shard:
            continue
        out = Path(args.out) / r.checkpoint_id
        if (out / "summary.json").exists():
            continue
        lock = Path(args.out) / f".{r.checkpoint_id}.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            try:
                os.kill(int(lock.read_text() or 0), 0)
                continue  # another live recorder owns it
            except (ProcessLookupError, ValueError):
                lock.write_text(str(os.getpid()))
        if args.max_checkpoints and n >= args.max_checkpoints:
            break
        rows = record_checkpoint(ROOT / r.checkpoint_path, out, seeds, seeds, cfg, cache, checkpoint_id=r.checkpoint_id)
        write_json(out / "summary.json", {
            "checkpoint": r.to_dict(), "checkpoint_id": r.checkpoint_id, "seeds": seeds, "video_seeds": seeds,
            "recorder": cfg.__dict__, "git_commit": git_commit(), "env": environment_info(),
            "finished_utc": utc_now(), "episodes": rows,
            "success_rate": sum(x["success"] for x in rows) / len(rows),
        })
        lock.unlink(missing_ok=True)
        n += 1
        print(json.dumps({"checkpoint_id": r.checkpoint_id, "train_rollout_success": r.train_rollout_success,
                          "eval_success": sum(x["success"] for x in rows) / len(rows)}), flush=True)
    return n


if __name__ == "__main__":
    main()
