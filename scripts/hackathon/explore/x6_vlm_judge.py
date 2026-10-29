#!/usr/bin/env python3
"""x6_visual_plan_critic, stage 2 (Environment B, EXPLORATORY).

Ask an NVIDIA VLM a task-free question about the rendered policy plan:
  "will the planned motion bring the gripper close to the <object>?"
on the overlays written by x6_plan_render.py.

Variants (image files):  plan1 = first policy call only; plan3 = 3 real chunks;
raw = the same initial frame WITHOUT dots (control: what does the model say
when there is no plan to judge?).

Backends: nebius (Nemotron-3.5-Super-VL, Token Factory) or local (Cosmos3-Edge).
Re-runnable: rows already in the output JSONL are skipped.

  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/x6_vlm_judge.py --backend nebius --variants plan1 --max-calls 260
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

from cosmos_plv.visual_diagnostics import _first_json_object, split_reasoning  # noqa: E402

NAME = "x6_visual_plan_critic"
PLANS = ROOT / f"data/outputs/hackathon_explore/{NAME}/plans"
# overlays redrawn on validated reference frames (see x6_redraw.py); raw control frames are the references
CLEAN = ROOT / f"data/outputs/hackathon_explore/{NAME}/plans_clean"
REF = {"can": "can_s1_e0000", "square": "square_s1_e0000"}
EVIDENCE = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
OBJECT = {
    "can": "the red can",
    "square": "the square nut (the flat square metal ring with a handle lying on the table)",
}
PROMPT_PLAN = (
    "The image shows a robot arm above a work surface. The coloured dots connected by a line show where "
    "the robot gripper plans to move next (from dark to light: dark purple = now, light yellow = last "
    "planned point). Will this planned motion bring the gripper close to {obj}?\n"
    "Answer with ONE JSON object only, no other text:\n"
    '{{"heads_to_object": true|false, "reaches_object": true|false, "explanation": "<one short sentence>"}}\n'
    '"heads_to_object" = the planned path moves towards {obj}; "reaches_object" = the last planned points '
    "end at or right next to {obj}."
)
PROMPT_RAW = (
    "The image shows a robot arm above a work surface. Is the robot gripper currently close to {obj}?\n"
    "Answer with ONE JSON object only, no other text:\n"
    '{{"heads_to_object": true|false, "reaches_object": true|false, "explanation": "<one short sentence>"}}\n'
    '"heads_to_object" = the gripper is closer to {obj} than to the opposite side of the scene; '
    '"reaches_object" = the gripper is at or right next to {obj}.'
)
KEYS = ("heads_to_object", "reaches_object")


def parse(raw: str):
    _r, ans = split_reasoning(raw or "")
    blob = _first_json_object(ans)
    if blob is None:
        return {"parse_status": "no_json"}
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return {"parse_status": "bad_json"}
    out = {"parse_status": "ok"}
    for k in KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.lower() in ("true", "false"):
            v = v.lower() == "true"
        if not isinstance(v, bool):
            return {"parse_status": f"bad_field_{k}"}
        out[k] = v
    out["explanation"] = str(obj.get("explanation", ""))[:400]
    return out


def jobs(variants, seeds_filter):
    for cdir in sorted(p for p in PLANS.iterdir() if p.is_dir()):
        cid = cdir.name
        task = cid.split("_")[0]
        for npz in sorted(cdir.glob("seed*.npz")):
            seed = int(npz.stem[4:])
            if seeds_filter and seed not in seeds_filter:
                continue
            for v in variants:
                # "<variant>_rep" = test-retest repeat of the same image (independent call)
                vimg = v[:-4] if v.endswith("_rep") else v
                img = (PLANS / REF[task] if vimg == "raw" else CLEAN / cid) / f"seed{seed}_{vimg}.png"
                if img.is_file():
                    yield {"checkpoint_id": cid, "task": task, "env_seed": seed, "variant": v, "image": str(img)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("nebius", "local"), default="nebius")
    ap.add_argument("--variants", default="plan1")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-calls", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--filter", default="")
    ap.add_argument("--ids", default="", help="comma list of checkpoint_ids (exact)")
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out or ROOT / f"data/outputs/hackathon_explore/{NAME}/vlm_{args.backend}_clean.jsonl")
    seeds_filter = {int(s) for s in args.seeds.split(",") if s}
    done = set()
    if out.is_file():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("error"):
                done.add((r["checkpoint_id"], r["env_seed"], r["variant"]))
    todo = [j for j in jobs(args.variants.split(","), seeds_filter)
            if (j["checkpoint_id"], j["env_seed"], j["variant"]) not in done and args.filter in j["checkpoint_id"]
            and (not args.ids or j["checkpoint_id"] in args.ids.split(","))]
    # global Nebius budget guard (800 for all of x6; we stop at 400)
    if args.backend == "nebius" and EVIDENCE.is_file():
        used = sum(1 for line in EVIDENCE.read_text().splitlines() if f'"explore/{NAME}' in line)
        room = max(0, 400 - used)
        print(f"[x6-vlm] nebius calls already used by {NAME}: {used}; room {room}", flush=True)
        todo = todo[:room]
    todo = todo[: args.max_calls]
    print(f"[x6-vlm] {len(todo)} calls to make -> {out}", flush=True)
    if not todo:
        return
    lock = threading.Lock()

    def prompt_for(j):
        tmpl = PROMPT_RAW if j["variant"].startswith("raw") else PROMPT_PLAN
        return tmpl.format(obj=OBJECT[j["task"]])

    def write(row):
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")

    if args.backend == "nebius":
        from cosmos_plv.nebius_backend import NebiusVLMClient

        local = threading.local()

        def run(j):
            if not hasattr(local, "client"):
                local.client = NebiusVLMClient(evidence_path=str(EVIDENCE))
            res = local.client.generate(prompt_for(j), images=[j["image"]], max_tokens=args.max_tokens,
                                        reasoning=False,
                                        tag=f"explore/{NAME}/{j['variant']}/{j['checkpoint_id']}/seed{j['env_seed']}")
            row = {**j, "backend": "nebius", "model_id": res.model_id, "raw_text": res.raw_text,
                   "latency_s": res.latency_s, "prompt_tokens": res.prompt_tokens,
                   "completion_tokens": res.completion_tokens, "error": res.error, **parse(res.raw_text)}
            write(row)
            return row

        with ThreadPoolExecutor(args.workers) as ex:
            for i, row in enumerate(ex.map(run, todo)):
                if i % 25 == 0:
                    print(f"[x6-vlm] {i + 1}/{len(todo)} {row['checkpoint_id']} {row['parse_status']} "
                          f"{row.get('heads_to_object')} {row.get('reaches_object')}", flush=True)
    else:
        from cosmos_plv.cosmos_backend import CosmosReasonerLocal

        model = CosmosReasonerLocal()
        print(f"[x6-vlm] cosmos loaded in {model.load_s:.1f}s", flush=True)
        for i, j in enumerate(todo):
            res = model.generate(prompt_for(j), images=[j["image"]], max_new_tokens=args.max_tokens,
                                 reasoning=False, seed=0)
            row = {**j, "backend": "local", "model_id": res.model_id, "model_revision": res.model_revision,
                   "raw_text": res.raw_text, "latency_s": res.latency_s, "prompt_tokens": res.prompt_tokens,
                   "completion_tokens": res.completion_tokens, "error": res.error,
                   "load_s": round(model.load_s, 2), **parse(res.raw_text)}
            write(row)
            if res.error and "out of memory" in res.error.lower():
                print("[x6-vlm] CUDA OOM -> stop (infrastructure failure)", flush=True)
                break
            if i % 25 == 0:
                print(f"[x6-vlm] {i + 1}/{len(todo)} {row['checkpoint_id']} {row['parse_status']} "
                      f"{row.get('heads_to_object')} {row.get('reaches_object')} {res.latency_s}s", flush=True)


if __name__ == "__main__":
    main()
