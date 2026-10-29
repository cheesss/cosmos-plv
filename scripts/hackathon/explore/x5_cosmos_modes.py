#!/usr/bin/env python3
"""x5_cosmos_modes (EXPLORATORY / post-hoc): alternative ways to apply local Cosmos3-Edge.

ONE GPU process: load nvidia/Cosmos3-Edge once, then
  1. dev split (36 episodes, prompt development only): run 4 variants per episode
       a_roi_track8   ROI crop (fixed box per task) + 2x bicubic upsample of the 8 v7 keyframes,
                      v6 perception prompt, v7 aggregation rule
       b_track15      15 evenly spaced keyframes (= v7's 8 + the 7 midpoints; shared frames reuse the
                      existing v7 per-frame outputs), v6 prompt, v7 rule
       c_pairwise8    7 pairwise questions on consecutive v7 keyframes (motion up/down/stayed,
                      held in image 1/2, location in image 2), pre-registered pair rule
       d_reason_last  v6 last-frame perception with reasoning enabled
  2. pre-registered selection on dev only (see x5_common: SUCCESS_CANDIDATES / FTYPE_CANDIDATES)
  3. eval split: apply the chosen 1-2 variants unchanged, in seed-major interleaved order, until the
     eval time cap; every processed episode is recorded.
All per-call outputs are cached in data/outputs/hackathon_explore/x5_cosmos_modes/calls.jsonl, so a
re-run resumes and extends to new eval episodes.

Run (Env B, under the GPU lock):
  flock /tmp/claude-1000/gpu_explore.lock env COSMOS_MAX_GPU_FRACTION=0.25 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
    ~/.venvs/cosmos-plv/bin/python scripts/hackathon/explore/x5_cosmos_modes.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x5_common as C  # noqa: E402

from cosmos_plv.provenance import utc_now  # noqa: E402
from cosmos_plv.visual_diagnostics import build_perception_prompt, parse_perception  # noqa: E402

CALLS = C.SCRATCH / "calls.jsonl"
STATE = C.SCRATCH / "run_state.json"


def gpu_free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0])


class Runner:
    def __init__(self, model):
        self.model = model
        self.cache = {}
        if CALLS.exists():
            for line in CALLS.read_text().splitlines():
                r = json.loads(line)
                self.cache[(r["variant"], r["checkpoint_id"], r["env_seed"], r["unit"])] = r
        self.n_new = 0
        self.oom = 0

    def call(self, variant, ep, unit, prompt, images, reasoning, max_new_tokens, parser):
        key = (variant, ep["checkpoint_id"], ep["env_seed"], unit)
        if key in self.cache:
            return self.cache[key]
        for attempt in range(2):
            res = self.model.generate(prompt, images=[str(p) for p in images], max_new_tokens=max_new_tokens,
                                      reasoning=reasoning, seed=0)
            if res.error and "out of memory" in res.error.lower():
                self.oom += 1
                import torch

                torch.cuda.empty_cache()
                print(f"CUDA OOM (attempt {attempt + 1}) {key}", flush=True)
                if attempt == 1:
                    raise RuntimeError("CUDA OOM twice: infrastructure failure")
                continue
            break
        parsed = parser(res.raw_text)
        row = {
            "variant": variant, "checkpoint_id": ep["checkpoint_id"], "env_seed": ep["env_seed"],
            "task": ep["task"], "unit": unit, "images": [os.path.relpath(str(p), str(C.ROOT)) for p in images],
            "prompt": prompt, "raw_text": res.raw_text, "parse_status": parsed.parse_status,
            "parse_errors": parsed.errors, "fields": parsed.fields, "reasoning_text": parsed.reasoning,
            "latency_s": res.latency_s, "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens, "backend_error": res.error, "params": res.params,
            "model_id": res.model_id, "model_revision": res.model_revision, "started_utc": res.started_utc,
            "oom_retries": attempt,
        }
        with open(CALLS, "a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self.cache[key] = row
        self.n_new += 1
        return row

    # ---------------------------------------------------------------- variants
    def run_d(self, ep):
        last = ep["video_frames"] - 1
        img = C.frame_png(ep, last)
        self.call("d_reason_last", ep, f"f{last:03d}", build_perception_prompt(ep["task"], reasoning=True),
                  [img], True, 1024, lambda raw: parse_perception(raw, ep["task"]))

    def run_a(self, ep):
        idx = C.keyframe_indices(ep["video_frames"], 8)
        prompt = build_perception_prompt(ep["task"], reasoning=False)
        for j in reversed(idx):  # last frame first: success result exists even if interrupted
            self.call("a_roi_track8", ep, f"f{j:03d}", prompt, [C.frame_png(ep, j, crop=True)], False, 300,
                      lambda raw: parse_perception(raw, ep["task"]))

    def run_b(self, ep, v7_frames):
        idx = C.keyframe_indices(ep["video_frames"], 15)
        prompt = build_perception_prompt(ep["task"], reasoning=False)
        have = v7_frames.get((ep["checkpoint_id"], ep["env_seed"]), {})
        for j in idx:
            if j in have:
                continue  # shared with v7: reuse the existing per-frame output (same prompt/image/seed)
            self.call("b_track15", ep, f"f{j:03d}", prompt, [C.frame_png(ep, j)], False, 300,
                      lambda raw: parse_perception(raw, ep["task"]))

    def run_c(self, ep):
        idx = C.keyframe_indices(ep["video_frames"], 8)
        fps = float(ep["video_fps"])
        for i in range(len(idx) - 1):
            j0, j1 = idx[i], idx[i + 1]
            self.call("c_pairwise8", ep, f"p{j0:03d}_{j1:03d}", C.build_pair_prompt(ep["task"], (j1 - j0) / fps),
                      [C.frame_png(ep, j0), C.frame_png(ep, j1)], False, 200,
                      lambda raw: C.parse_pair(raw, ep["task"]))


def load_v7_frames():
    """(checkpoint_id, env_seed) -> {frame_index: per-frame record} from the existing v7 runs."""
    out = {}
    for path in (C.COSMOS_DIR / "dev_prompt_selection.jsonl", C.COSMOS_DIR / "eval_local_v7.jsonl"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("variant") != "v7_perception_track" or r.get("backend") != "local" or not r.get("track"):
                continue
            d = {}
            for fr in r["track"]["frames"]:
                j = int(Path(fr["image"]).stem.split("_f")[-1])
                d[j] = fr
            out[(r["checkpoint_id"], int(r["env_seed"]))] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-minutes", type=float, default=32.0)
    ap.add_argument("--eval-minutes", type=float, default=16.0)
    ap.add_argument("--wait-minutes", type=float, default=20.0)
    ap.add_argument("--min-free-mib", type=int, default=9000)
    ap.add_argument("--skip-dev", action="store_true", help="re-run: reuse cached dev calls and selection")
    a = ap.parse_args()
    C.SCRATCH.mkdir(parents=True, exist_ok=True)
    state = {"started_utc": utc_now(), "argv": sys.argv, "env": {k: os.environ.get(k) for k in (
        "COSMOS_MAX_GPU_FRACTION", "PYTORCH_CUDA_ALLOC_CONF", "HF_HUB_OFFLINE")}}

    t_wait = time.time()
    free = gpu_free_mib()
    state["gpu_free_mib_checks"] = [free]
    while free < a.min_free_mib and time.time() - t_wait < a.wait_minutes * 60:
        print(f"GPU free {free} MiB < {a.min_free_mib}; waiting 60 s", flush=True)
        time.sleep(60)
        free = gpu_free_mib()
        state["gpu_free_mib_checks"].append(free)
    if free < a.min_free_mib:
        state["aborted"] = f"GPU free memory stayed below {a.min_free_mib} MiB"
        STATE.write_text(json.dumps(state, indent=2))
        print(state["aborted"], flush=True)
        sys.exit(3)

    from cosmos_plv.cosmos_backend import CosmosReasonerLocal

    model = CosmosReasonerLocal()
    state["model"] = {"id": model.model_id, "revision": model.revision, "load_s": round(model.load_s, 1)}
    print(f"loaded {model.model_id}@{model.revision} in {model.load_s:.1f}s; free now {gpu_free_mib()} MiB",
          flush=True)
    R = Runner(model)
    v7 = load_v7_frames()

    # ------------------------------------------------------------ dev
    dev = C.interleave(list(C.iter_episodes(C.DEV_ROOT)))
    t0 = time.time()
    dev_done = []
    for ep in dev:
        if not a.skip_dev and time.time() - t0 > a.dev_minutes * 60:
            break
        cached = all(
            any(k[0] == v and k[1] == ep["checkpoint_id"] and k[2] == ep["env_seed"] for k in R.cache)
            for v in ("d_reason_last", "a_roi_track8", "c_pairwise8"))
        if a.skip_dev and not cached:
            continue
        te = time.time()
        R.run_d(ep)
        R.run_a(ep)
        R.run_c(ep)
        R.run_b(ep, v7)
        dev_done.append([ep["checkpoint_id"], ep["env_seed"]])
        print(f"dev {len(dev_done)}/{len(dev)} {ep['checkpoint_id']} seed{ep['env_seed']} "
              f"{time.time() - te:.0f}s new_calls={R.n_new}", flush=True)
    state["dev_episodes_done"] = dev_done
    state["dev_wall_s"] = round(time.time() - t0, 1)

    # ------------------------------------------------------------ pre-registered selection (dev only)
    import x5_analyze as A

    dev_rows = A.variant_rows(R.cache, v7, [tuple(x) for x in dev_done], C.DEV_ROOT)
    sel = A.select(dev_rows)
    state["selection"] = sel
    STATE.write_text(json.dumps(state, indent=2))
    print("selection:", json.dumps(sel), flush=True)
    chosen = sorted(set(sel["chosen"]))

    # ------------------------------------------------------------ eval
    ev = C.interleave(list(C.iter_episodes(C.EVAL_ROOT)))
    state["eval_episodes_available"] = len(ev)
    t1 = time.time()
    eval_done = []
    for ep in ev:
        if time.time() - t1 > a.eval_minutes * 60:
            break
        te = time.time()
        for v in chosen:
            {"a_roi_track8": R.run_a, "c_pairwise8": R.run_c, "d_reason_last": R.run_d,
             "b_track15": lambda e: R.run_b(e, v7)}[v](ep)
        eval_done.append([ep["checkpoint_id"], ep["env_seed"]])
        print(f"eval {len(eval_done)}/{len(ev)} {ep['checkpoint_id']} seed{ep['env_seed']} "
              f"{time.time() - te:.0f}s new_calls={R.n_new}", flush=True)
    state["eval_episodes_done"] = eval_done
    state["eval_wall_s"] = round(time.time() - t1, 1)
    state["new_calls"] = R.n_new
    state["oom_events"] = R.oom
    state["finished_utc"] = utc_now()
    STATE.write_text(json.dumps(state, indent=2))
    print("done", json.dumps({k: state[k] for k in ("new_calls", "oom_events", "dev_wall_s", "eval_wall_s")}),
          flush=True)


if __name__ == "__main__":
    main()
