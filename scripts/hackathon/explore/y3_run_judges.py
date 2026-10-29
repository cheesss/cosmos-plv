#!/usr/bin/env python3
"""y3_visual_robustness Step 4 (Environment B): frozen judges on each visual condition.

  local  : Cosmos3-Edge v6_perception_last (last frame only, reasoning off, max_new_tokens
           300, generation seed 0) -- identical settings to eval_local_v6.jsonl.
  nebius : Nemotron-3.5-Super-VL v2_evidence_first on the video, padded to 20.1 s
           (prompt = build_prompt(task, 4, 20.1, False, 'v2_evidence_first'),
           generate(..., max_tokens=700, temperature 0.2, pad_video_to_s=20.1)) --
           identical to eval_nebius_v2_pad201.jsonl.
The prompts are NOT changed for other cameras (robustness of the frozen prompt).

Conditions: orig (recorded agentview), cam_<camera> (replayed + re-rendered;
cam_agentview is the re-render control), and the degradations of y3_degrade.py.
Resumable: rows already in the output jsonl are skipped.

  y3_gpu_run.sh envB scripts/hackathon/explore/y3_run_judges.py --backend local
  PYTHONNOUSERSITE=1 ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/y3_run_judges.py --backend nebius
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from cosmos_plv.visual_diagnostics import (  # noqa: E402
    build_perception_prompt, build_prompt, parse_diagnostic, parse_perception)

SCR = ROOT / "data/outputs/hackathon_explore/y3_visual_robustness"
EVIDENCE = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
CAMS = ("agentview", "frontview", "birdview", "robot0_eye_in_hand")


def build_manifest():
    eps = json.loads((ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json").read_text())["episodes"]
    meta = {(e["checkpoint_id"], e["env_seed"]): e for e in eps}
    rows = []
    for r in json.loads((SCR / "degrade/manifest.json").read_text()):
        rows.append(r)
    log = SCR / "render/replay_log_full.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if not (r["success_match"] and r["max_abs_obj_err_m"] == 0.0 and r["max_abs_eef_err_m"] == 0.0):
                continue  # only deterministic replays are judged
            for c, s in r["saved"].items():
                rows.append({"condition": f"cam_{c}", "checkpoint_id": r["checkpoint_id"], "env_seed": r["env_seed"],
                             "mp4": s["mp4"], "last_png": s["last_png"]})
    for r in rows:
        e = meta[(r["checkpoint_id"], r["env_seed"])]
        r.update(task=e["task"], truth_success=bool(e["success"]), truth_failure_type=e["failure_type"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("local", "nebius"), required=True)
    ap.add_argument("--conditions", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=700)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = SCR / f"judge_{args.backend}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if not r.get("infra_error"):
                done.add((r["condition"], r["checkpoint_id"], r["env_seed"]))
    todo = [r for r in build_manifest() if (r["condition"], r["checkpoint_id"], r["env_seed"]) not in done]
    if args.conditions:
        keep = set(args.conditions.split(","))
        todo = [r for r in todo if r["condition"] in keep]
    # interleave conditions so a partial run still covers every condition
    todo.sort(key=lambda r: (r["checkpoint_id"], r["env_seed"], r["condition"]))
    if args.limit:
        todo = todo[: args.limit]
    print(f"[y3] {args.backend}: {len(todo)} items to run", flush=True)
    lock = threading.Lock()

    def write(row):
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    if args.backend == "local":
        from cosmos_plv.cosmos_backend import CosmosReasonerLocal

        model = CosmosReasonerLocal()
        print(f"[y3] loaded {model.model_id}@{model.revision} in {model.load_s:.1f}s", flush=True)
        for i, r in enumerate(todo):
            prompt = build_perception_prompt(r["task"], False)
            res = model.generate(prompt, images=[str(ROOT / r["last_png"])], max_new_tokens=300, reasoning=False, seed=0)
            parsed = parse_perception(res.raw_text, r["task"])
            oom = bool(res.error and "out of memory" in res.error.lower())
            row = {**r, "backend": "local", "variant": "v6_perception_last", "parse_status": parsed.parse_status,
                   "pred_success": parsed.fields.get("task_success"), "object_location": parsed.fields.get("object_location"),
                   "held_by_gripper": parsed.fields.get("held_by_gripper"), "raw_text": res.raw_text,
                   "latency_s": res.latency_s, "backend_error": res.error, "infra_error": oom,
                   "model_id": res.model_id, "model_revision": res.model_revision, "params": res.params}
            write(row)
            if i % 20 == 0:
                print(f"[y3] {i}/{len(todo)} {r['condition']} {r['checkpoint_id']} {r['env_seed']} "
                      f"{parsed.parse_status} pred={row['pred_success']} truth={r['truth_success']} {res.latency_s}s",
                      flush=True)
        return

    from cosmos_plv.nebius_backend import NebiusVLMClient

    n_prev = 0
    if EVIDENCE.exists():
        n_prev = sum(1 for l in EVIDENCE.read_text().splitlines() if '"explore/y3_visual_robustness/' in l)
    budget = args.max_calls - n_prev
    print(f"[y3] previous y3 nebius calls {n_prev}; budget left {budget}", flush=True)
    todo = todo[: max(0, budget)]
    counter = {"n": 0}
    local = threading.local()

    def run(r):
        if not hasattr(local, "client"):
            local.client = NebiusVLMClient(evidence_path=str(EVIDENCE))
        prompt = build_prompt(r["task"], 4, 20.1, False, "v2_evidence_first")
        tag = f"explore/y3_visual_robustness/{r['condition']}/{r['checkpoint_id']}/seed{r['env_seed']}"
        res = local.client.generate(prompt, video=str(ROOT / r["mp4"]), max_tokens=700, reasoning=False,
                                    tag=tag, pad_video_to_s=20.1)
        parsed = parse_diagnostic(res.raw_text)
        row = {**r, "backend": "nebius", "variant": "v2_evidence_first_pad201", "parse_status": parsed.parse_status,
               "pred_success": parsed.fields.get("task_success"), "pred_failure_type": parsed.fields.get("failure_type"),
               "explanation": parsed.fields.get("explanation"), "raw_text": res.raw_text, "latency_s": res.latency_s,
               "backend_error": res.error, "infra_error": bool(res.error), "model_id": res.model_id,
               "served_model": res.served_model, "request_id": res.request_id, "params": res.params, "tag": tag}
        write(row)
        with lock:
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                print(f"[y3] nebius {counter['n']}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, todo))
    print("[y3] nebius done", counter["n"], flush=True)


if __name__ == "__main__":
    main()
