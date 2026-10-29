#!/usr/bin/env python3
"""Record real robosuite rollouts (video + state traces) for one or more checkpoints.

Example:
  python scripts/hackathon/record_rollouts.py \
      --checkpoint data/outputs/hackathon_pool/<run>/checkpoints/epoch=0050-test_mean_score=0.800.ckpt \
      --out data/outputs/hackathon_rollouts --seeds 200000:200020 --video-seeds 200000:200020
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

from cosmos_plv.rollout_recorder import RecorderConfig, record_checkpoint  # noqa: E402
from cosmos_plv.provenance import environment_info, git_commit, utc_now, write_json  # noqa: E402


def parse_range(text: str):
    if ":" in text:
        a, b = text.split(":")
        return list(range(int(a), int(b)))
    return [int(x) for x in text.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True)
    ap.add_argument("--checkpoint-id", action="append", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="200000:200020")
    ap.add_argument("--video-seeds", default=None, help="default: all seeds")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--sampling-seed", type=int, default=0)
    ap.add_argument("--render", default="448x448")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    seeds = parse_range(args.seeds)
    video = parse_range(args.video_seeds) if args.video_seeds else seeds
    h, w = (int(x) for x in args.render.split("x"))
    cfg = RecorderConfig(render_hw=(h, w), batch_size=args.batch_size,
                         sampling_seed=args.sampling_seed, device=args.device)
    ids = args.checkpoint_id or [None] * len(args.checkpoint)
    cache: dict = {}
    for ckpt, cid in zip(args.checkpoint, ids):
        cid = cid or Path(ckpt).parent.parent.name + "__" + Path(ckpt).stem
        out = Path(args.out) / cid
        rows = record_checkpoint(Path(ckpt), out, seeds, video, cfg, cache, checkpoint_id=cid)
        write_json(out / "summary.json", {
            "checkpoint_id": cid, "checkpoint": ckpt, "seeds": seeds, "video_seeds": video,
            "recorder": cfg.__dict__, "git_commit": git_commit(), "env": environment_info(),
            "finished_utc": utc_now(), "episodes": rows,
            "success_rate": sum(r["success"] for r in rows) / max(1, len(rows)),
        })
        print(json.dumps({"checkpoint_id": cid, "n": len(rows),
                          "success_rate": sum(r["success"] for r in rows) / max(1, len(rows)),
                          "failures": {f: sum(r["failure_type"] == f for r in rows) for f in
                                       sorted({r["failure_type"] for r in rows})}}))


if __name__ == "__main__":
    main()
