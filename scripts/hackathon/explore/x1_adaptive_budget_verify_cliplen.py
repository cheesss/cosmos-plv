#!/usr/bin/env python3
"""EXPLORATORY verification probe for x1_adaptive_budget: is Nemotron v2's measured error rate
helped by the clip-length cue?

In the eval set every failure clip is 20.1 s (episode runs to 400 steps) while successes stop
16 steps after success (5.3-15.0 s), and the v2 prompt states "The clip lasts about X seconds".
x1 feeds the measured Nemotron FPR/FNR into its budget simulation as if they were the judge's
intrinsic error rates (and motivates VLM judges for settings without a success detector, where
clips cannot be cut at success).  This probe removes the cue and re-asks Nemotron:
  * all eval failures of type 'dropped' (the FP-prone near-misses x1's _ftype model is built on):
    same video, prompt WITHOUT the duration sentence;
  * 13 eval successes drawn with a fixed seed: video padded to 201 frames (20.1 s) by repeating
    the last frame, prompt WITHOUT the duration sentence.
Selection rule and n were fixed before any call.  Hard cap: 30 Nebius calls (resumable; calls
already logged in the scratch JSONL are not repeated).

  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/x1_adaptive_budget_verify_cliplen.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cosmos_plv.nebius_backend import NebiusVLMClient, sha256_bytes  # noqa: E402
from cosmos_plv.visual_diagnostics import build_prompt, parse_diagnostic  # noqa: E402

NAME = "x1_adaptive_budget_verification"
SCR = ROOT / "data/outputs/hackathon_explore" / NAME / "cliplen"
LOG = SCR / "calls.jsonl"
OUT = ROOT / "results/hackathon/explore" / f"{NAME}_cliplen.json"
EVID = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
MAX_CALLS = 30
N_SUCC = 13


def pad_video(src: Path, dst: Path, n_frames: int = 201, fps: int = 10):
    import imageio.v2 as imageio
    r = imageio.get_reader(str(src))
    frames = [np.asarray(f) for f in r]
    r.close()
    frames += [frames[-1]] * max(0, n_frames - len(frames))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(dst), fps=fps, codec="libx264", quality=8, macro_block_size=16,
                            pixelformat="yuv420p") as w:
        for f in frames:
            w.append_data(f)
    return len(frames)


def main():
    SCR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (ROOT / "data/outputs/hackathon_cosmos/eval_nebius_v2.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("scored") and r.get("parse_status") == "ok"]
    # sanity: the original prompt is reproduced byte-for-byte (with duration)
    r0 = rows[0]
    p0 = build_prompt(r0["task"], 4.0, r0["video_frames"] / float(r0["video_fps"]), False, "v2_evidence_first")
    prompt_repro_ok = sha256_bytes(p0.encode()) == r0["backend_params"]["prompt_sha256"]
    fails = sorted([r for r in rows if not r["truth_success"] and r["truth_failure_type"] == "dropped"],
                   key=lambda r: (r["checkpoint_id"], r["env_seed"]))
    succ = sorted([r for r in rows if r["truth_success"]], key=lambda r: (r["checkpoint_id"], r["env_seed"]))
    rng = np.random.default_rng(20261029)
    succ = [succ[i] for i in sorted(rng.choice(len(succ), N_SUCC, replace=False))]
    plan = [("dropped_failure_no_duration", r) for r in fails] + [("success_padded_no_duration", r) for r in succ]
    plan = plan[:MAX_CALLS]
    done = {}
    if LOG.is_file():
        for line in LOG.read_text().splitlines():
            x = json.loads(line)
            done[(x["checkpoint_id"], x["env_seed"])] = x
    client = None
    n_calls = 0
    for cond, r in plan:
        key = (r["checkpoint_id"], int(r["env_seed"]))
        if key in done:
            continue
        if len(done) >= MAX_CALLS:  # done = logged records (previous + this invocation)
            break
        video = ROOT / r["video"]
        n_frames = r["video_frames"]
        if cond.startswith("success"):
            dst = SCR / "padded" / f"{r['checkpoint_id']}_seed{r['env_seed']}.mp4"
            n_frames = pad_video(video, dst)
            video = dst
        prompt = build_prompt(r["task"], 4.0, None, False, "v2_evidence_first")
        if client is None:
            client = NebiusVLMClient(evidence_path=str(EVID))
        res = client.generate(prompt, video=str(video), max_tokens=700, reasoning=False,
                              tag=f"explore/{NAME}/cliplen/{r['checkpoint_id']}/seed{r['env_seed']}")
        n_calls += 1
        parsed = parse_diagnostic(res.raw_text)
        rec = {"condition": cond, "checkpoint_id": r["checkpoint_id"], "env_seed": int(r["env_seed"]), "task": r["task"],
               "truth_success": r["truth_success"], "truth_failure_type": r["truth_failure_type"],
               "orig_video_frames": r["video_frames"], "probe_video_frames": n_frames,
               "orig_judge_success": r["cosmos_task_success"],
               "probe_parse_status": parsed.parse_status, "probe_judge_success": parsed.fields.get("task_success"),
               "probe_failure_type": parsed.fields.get("failure_type"),
               "error": res.error, "latency_s": res.latency_s, "raw_text": res.raw_text}
        with LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        done[key] = rec
        print(cond, key, "orig", r["cosmos_task_success"], "probe", rec["probe_judge_success"], rec["probe_parse_status"], flush=True)
    recs = list(done.values())
    summ = {}
    for cond in ("dropped_failure_no_duration", "success_padded_no_duration"):
        xs = [x for x in recs if x["condition"] == cond and x["probe_parse_status"] == "ok"
              and isinstance(x["probe_judge_success"], bool)]
        judged_success_orig = sum(1 for x in xs if x["orig_judge_success"])
        judged_success_probe = sum(1 for x in xs if x["probe_judge_success"])
        flips = [(x["checkpoint_id"], x["env_seed"], x["orig_judge_success"], x["probe_judge_success"]) for x in xs
                 if x["orig_judge_success"] != x["probe_judge_success"]]
        summ[cond] = {"n_planned": sum(1 for c, _ in plan if c == cond),
                      "n_scored": len(xs), "n_unparsed_or_error": sum(1 for x in recs if x["condition"] == cond) - len(xs),
                      "judged_success_orig_prompt_with_duration": judged_success_orig,
                      "judged_success_probe_no_duration": judged_success_probe, "flips": flips}
    out = {"name": f"{NAME}_cliplen", "exploratory": True, "prompt_reproduction_sha_ok": prompt_repro_ok,
           "n_new_nebius_calls_this_invocation": n_calls, "n_records": len(recs), "max_calls": MAX_CALLS,
           "selection": "all eval failures with truth_failure_type=='dropped' (current labels) + 13 successes "
                        "drawn with default_rng(20261029) from the sorted success list",
           "summary": summ}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
