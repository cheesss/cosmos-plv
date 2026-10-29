#!/usr/bin/env python3
"""Run NVIDIA Cosmos (or the Nebius-hosted NVIDIA VLM) on recorded rollout videos.

Environment B (py3.11, transformers>=5.15).  For every episode with a video in
``<rollouts>/*/summary.json`` it stores the raw generation, the parsed and
validated diagnostic, the verified simulator labels, latency and provenance.

  python scripts/hackathon/run_cosmos_diagnostics.py \
      --rollouts data/outputs/hackathon_rollouts_dev --variant v2_evidence_first \
      --out data/outputs/hackathon_cosmos/dev_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from cosmos_plv.provenance import git_commit, repo_relative, utc_now  # noqa: E402
from cosmos_plv.visual_diagnostics import (  # noqa: E402
    agreement_row,
    build_endstate_prompt,
    build_keyframe_prompt,
    build_perception_prompt,
    build_prompt,
    parse_diagnostic,
    parse_endstate,
    parse_perception,
    aggregate_track,
)


def last_frame_png(video: Path, out: Path) -> Path:
    return frames_png(video, out.parent, out.stem, [-1])[0]


def frames_png(video: Path, out_dir: Path, stem: str, idx):
    import imageio.v2 as imageio

    frames = [f for f in imageio.get_reader(str(video))]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in idx:
        j = i % len(frames)
        path = out_dir / f"{stem}_f{j:03d}.png"
        imageio.imwrite(path, frames[j])
        paths.append(path)
    return paths


def iter_episodes(roots):
    for root in roots:
        for summ in sorted(Path(root).glob("*/summary.json")):
            data = json.loads(summ.read_text())
            for ep in data["episodes"]:
                if ep.get("video"):
                    yield summ.parent, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="v2_evidence_first")
    ap.add_argument("--backend", choices=("local", "nebius"), default="local")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--no-reasoning", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keyframes", type=int, default=6)
    ap.add_argument("--pad-to-frames", type=int, default=0,
                    help="length-neutral videos: pad with the final frame to this many frames (nebius only)")
    ap.add_argument("--checkpoint-filter", default="")
    ap.add_argument("--seeds", default="", help="comma list or a:b range of env seeds to include")
    ap.add_argument("--episodes", default="", help="JSON file with a list of [checkpoint_id, env_seed] pairs to include")
    ap.add_argument("--evidence", default="results/hackathon/nebius_calls.jsonl",
                    help="Nebius call log (exploratory studies use results/hackathon/explore/nebius_calls_explore.jsonl)")
    args = ap.parse_args()

    seed_filter = None
    if args.seeds:
        if ":" in args.seeds:
            a, b = args.seeds.split(":")
            seed_filter = set(range(int(a), int(b)))
        else:
            seed_filter = {int(x) for x in args.seeds.split(",")}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["checkpoint_id"], r["env_seed"], r["variant"], r["backend"], bool(r["reasoning"])))

    episode_filter = ({(c, int(e)) for c, e in json.loads(Path(args.episodes).read_text())}
                      if args.episodes else None)
    reasoning = not args.no_reasoning
    if args.backend == "local":
        from cosmos_plv.cosmos_backend import CosmosReasonerLocal

        model = CosmosReasonerLocal()
        print(f"loaded {model.model_id}@{model.revision} in {model.load_s:.1f}s", flush=True)
    else:
        from cosmos_plv.nebius_backend import NebiusVLMClient

        model = NebiusVLMClient(evidence_path=str(ROOT / args.evidence))

    n = 0
    for ep_dir, ep in iter_episodes(args.rollouts):
        if args.checkpoint_filter and args.checkpoint_filter not in ep["checkpoint_id"]:
            continue
        if seed_filter is not None and int(ep["env_seed"]) not in seed_filter:
            continue
        if episode_filter is not None and (ep["checkpoint_id"], int(ep["env_seed"])) not in episode_filter:
            continue
        key = (ep["checkpoint_id"], ep["env_seed"], args.variant + (f"_pad{args.pad_to_frames}" if args.pad_to_frames else ""),
               args.backend, not args.no_reasoning)
        if key in done:
            continue
        if args.limit and n >= args.limit:
            break
        video = ROOT / ep["video"]
        duration = ep["video_frames"] / float(ep["video_fps"])
        if args.pad_to_frames:
            # length-neutral condition: every clip is padded (final frame held) to the
            # timeout length and the prompt states that fixed length
            duration = args.pad_to_frames / float(ep["video_fps"])
        images = None
        use_video = True
        stem = f"seed{ep['env_seed']}"
        if args.variant == "v4_keyframes":
            n_f = ep["video_frames"]
            idx = [int(round(x)) for x in __import__("numpy").linspace(0, n_f - 1, args.keyframes)]
            images = [str(p) for p in frames_png(video, ep_dir / "keyframes", stem, idx)]
            prompt = build_keyframe_prompt(ep["task"], [i / float(ep["video_fps"]) for i in idx], reasoning)
            use_video = False
        elif args.variant == "v6_perception_last":
            images = [str(p) for p in frames_png(video, ep_dir / "keyframes", stem, [-1])]
            prompt = build_perception_prompt(ep["task"], reasoning)
            use_video = False
        elif args.variant == "v5_endstate":
            images = [str(p) for p in frames_png(video, ep_dir / "keyframes", stem, [0, -1])]
            prompt = build_endstate_prompt(ep["task"], reasoning)
            use_video = False
        else:
            prompt = build_prompt(ep["task"], args.fps, duration, reasoning, args.variant)
            if args.variant == "v3_evidence_lastframe":
                images = [str(last_frame_png(video, ep_dir / "last_frames" / f"{stem}.png"))]
        vid = str(video) if use_video else None
        track = None
        if args.variant == "v7_perception_track":
            n_f = ep["video_frames"]
            idx = [int(round(x)) for x in __import__("numpy").linspace(0, n_f - 1, args.keyframes)]
            paths = [str(p) for p in frames_png(video, ep_dir / "keyframes", stem, idx)]
            times = [i / float(ep["video_fps"]) for i in idx]
            prompt = build_perception_prompt(ep["task"], reasoning)
            calls, parsed_frames = [], []
            for path in paths:
                if args.backend == "local":
                    r1 = model.generate(prompt, images=[path], max_new_tokens=args.max_new_tokens,
                                        reasoning=reasoning, seed=args.seed)
                else:
                    r1 = model.generate(prompt, images=[path], max_tokens=args.max_new_tokens,
                                        reasoning=reasoning, tag=f"{ep['checkpoint_id']}/seed{ep['env_seed']}")
                calls.append(r1)
                parsed_frames.append(parse_perception(r1.raw_text, ep["task"]))
            images = paths
            track = {
                "times_s": times,
                "frames": [{"image": repo_relative(pth), "raw_text": c.raw_text, "latency_s": c.latency_s,
                            "parse_status": pf.parse_status, **{f"p_{k}": v for k, v in pf.fields.items()}}
                           for pth, c, pf in zip(paths, calls, parsed_frames)],
            }
            res = calls[-1]
            res.latency_s = round(sum(c.latency_s for c in calls), 3)
            res.prompt_tokens = sum(c.prompt_tokens or 0 for c in calls)
            res.completion_tokens = sum(c.completion_tokens or 0 for c in calls)
            res.raw_text = json.dumps([c.raw_text for c in calls])
            res.error = "; ".join(c.error for c in calls if c.error) or None
        elif args.backend == "local":
            res = model.generate(prompt, video=vid, images=images, fps=args.fps,
                                 max_new_tokens=args.max_new_tokens, reasoning=reasoning, seed=args.seed)
        else:
            res = model.generate(prompt, video=vid, images=images, max_tokens=args.max_new_tokens,
                                 reasoning=reasoning, tag=f"{ep['checkpoint_id']}/seed{ep['env_seed']}",
                                 pad_video_to_s=(args.pad_to_frames / float(ep["video_fps"])) if args.pad_to_frames else None)
        if track is not None:
            parsed = aggregate_track(ep["task"], parsed_frames, track["times_s"])
        elif args.variant == "v5_endstate":
            parsed = parse_endstate(res.raw_text)
        elif args.variant == "v6_perception_last":
            parsed = parse_perception(res.raw_text, ep["task"])
        else:
            parsed = parse_diagnostic(res.raw_text)
        truth = {k: ep[k] for k in ("success", "failure_type", "grasped", "lifted", "dropped")}
        row = {
            "checkpoint_id": ep["checkpoint_id"],
            "checkpoint_path": ep["checkpoint_path"],
            "task": ep["task"],
            "env_seed": ep["env_seed"],
            "sampling_seed": ep["sampling_seed"],
            "video": ep["video"],
            "video_frames": ep["video_frames"],
            "video_fps": ep["video_fps"],
            "variant": args.variant + (f"_pad{args.pad_to_frames}" if args.pad_to_frames else ""),
            "backend": args.backend,
            "reasoning": reasoning,
            "input_fps": args.fps if use_video else None,
            "input_images": [repo_relative(p) for p in images] if images else None,
            "generation_seed": args.seed,
            "truth_success": ep["success"],
            "truth_failure_type": ep["failure_type"],
            "truth_grasped": ep["grasped"],
            "truth_lifted": ep["lifted"],
            "truth_dropped": ep["dropped"],
            **parsed.to_row(),
            **agreement_row(parsed, truth),
            "raw_text": res.raw_text,
            "track": track,
            "reasoning_text": parsed.reasoning,
            "model_id": res.model_id,
            "model_revision": res.model_revision,
            "served_model": res.served_model,
            "request_id": res.request_id,
            "latency_s": res.latency_s,
            "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "backend_error": res.error,
            "backend_params": res.params,
            "started_utc": res.started_utc,
            "git_commit": git_commit(),
            "logged_utc": utc_now(),
        }
        with open(out, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        n += 1
        print(json.dumps({k: row.get(k) for k in ("checkpoint_id", "env_seed", "truth_success",
              "truth_failure_type", "cosmos_task_success", "cosmos_failure_type", "parse_status",
              "latency_s", "backend_error")}), flush=True)


if __name__ == "__main__":
    main()
