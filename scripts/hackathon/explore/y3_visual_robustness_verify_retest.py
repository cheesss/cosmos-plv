#!/usr/bin/env python3
"""Verifier for y3_visual_robustness: Nemotron test-retest of the birdview discordant pairs.

The only non-wrist camera effect whose 95% CI excludes 0 is Nemotron cam_birdview vs
cam_agentview (-0.150; 12 vs 3 discordant pairs).  Nemotron runs at temperature 0.2 and
flips ~7% of predictions between runs (3 of 4 orig-vs-cam_agentview flips were on
byte-identical inputs).  This script re-runs BOTH conditions on the 15 discordant episodes
(30 calls, hard cap) with the identical frozen prompt / params of y3_run_judges.py, to see
whether the discordance persists.  Selection is on the y3 discordance (by design), so a
noise-only effect would shrink toward 0 on retest (regression to the mean); a persistent
effect should mostly reproduce.

  PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 ~/.venvs/cosmos-plv/bin/python \
      scripts/hackathon/explore/y3_visual_robustness_verify_retest.py
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from cosmos_plv.nebius_backend import NebiusVLMClient  # noqa: E402
from cosmos_plv.visual_diagnostics import build_prompt, parse_diagnostic  # noqa: E402

SCR = ROOT / "data/outputs/hackathon_explore/y3_visual_robustness"
OUT = ROOT / "data/outputs/hackathon_explore/y3_visual_robustness_verification/retest_nebius.jsonl"
EVID = ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl"
MAX_CALLS = 30


def main():
    rows = {}
    for line in (SCR / "judge_nebius.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[(r["condition"], r["checkpoint_id"], r["env_seed"])] = r
    eps = json.loads((ROOT / "results/hackathon/explore/y3_visual_robustness_episodes.json").read_text())["episodes"]

    def ok(r):
        return r["parse_status"] in ("ok", "ok_repaired") and r["pred_success"] is not None and \
            bool(r["pred_success"]) == bool(r["truth_success"])

    disc = []
    for e in eps:
        k = (e["checkpoint_id"], e["env_seed"])
        a, b = rows[("cam_agentview",) + k], rows[("cam_birdview",) + k]
        if ok(a) != ok(b):
            disc.append((k, "ctrl_right_cond_wrong" if ok(a) else "cond_right_ctrl_wrong"))
    todo = [(k, kind, c) for k, kind in disc for c in ("cam_agentview", "cam_birdview")]
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["condition"], r["checkpoint_id"], r["env_seed"]))
    n_prev = sum(1 for l in EVID.read_text().splitlines() if '"explore/y3_visual_robustness_verification/' in l) if EVID.exists() else 0
    todo = [t for t in todo if (t[2],) + t[0] not in done][: max(0, MAX_CALLS - n_prev)]
    print(f"discordant episodes {len(disc)}; calls to make {len(todo)} (previous verifier calls {n_prev})", flush=True)
    lock = threading.Lock()
    local = threading.local()

    def run(t):
        k, kind, cond = t
        src = rows[(cond,) + k]
        if not hasattr(local, "client"):
            local.client = NebiusVLMClient(evidence_path=str(EVID))
        prompt = build_prompt(src["task"], 4, 20.1, False, "v2_evidence_first")
        tag = f"explore/y3_visual_robustness_verification/retest/{cond}/{k[0]}/seed{k[1]}"
        res = local.client.generate(prompt, video=str(ROOT / src["mp4"]), max_tokens=700, reasoning=False,
                                    tag=tag, pad_video_to_s=20.1)
        p = parse_diagnostic(res.raw_text)
        row = {"condition": cond, "checkpoint_id": k[0], "env_seed": k[1], "task": src["task"],
               "truth_success": src["truth_success"], "y3_discordance": kind, "y3_pred_success": src["pred_success"],
               "parse_status": p.parse_status, "pred_success": p.fields.get("task_success"),
               "pred_failure_type": p.fields.get("failure_type"), "explanation": p.fields.get("explanation"),
               "backend_error": res.error, "request_id": res.request_id, "latency_s": res.latency_s, "tag": tag}
        with lock:
            with open(OUT, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(run, todo))

    rt = {}
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        rt[(r["condition"], r["checkpoint_id"], r["env_seed"])] = r

    def ok2(r):
        return r is not None and r["parse_status"] in ("ok", "ok_repaired") and r["pred_success"] is not None and \
            bool(r["pred_success"]) == bool(r["truth_success"])

    summ = {"ctrl_right_cond_wrong": {"n": 0, "persist": 0, "agentview_still_right": 0, "birdview_still_wrong": 0},
            "cond_right_ctrl_wrong": {"n": 0, "persist": 0, "agentview_still_wrong": 0, "birdview_still_right": 0}}
    for k, kind in disc:
        a, b = rt.get(("cam_agentview",) + k), rt.get(("cam_birdview",) + k)
        if a is None or b is None:
            continue
        s = summ[kind]
        s["n"] += 1
        if kind == "ctrl_right_cond_wrong":
            s["agentview_still_right"] += ok2(a)
            s["birdview_still_wrong"] += not ok2(b)
            s["persist"] += ok2(a) and not ok2(b)
        else:
            s["agentview_still_wrong"] += not ok2(a)
            s["birdview_still_right"] += ok2(b)
            s["persist"] += (not ok2(a)) and ok2(b)
    print(json.dumps(summ, indent=1))
    print("errors", sum(1 for r in rt.values() if r["backend_error"]), "unparsed",
          sum(1 for r in rt.values() if r["parse_status"] not in ("ok", "ok_repaired")))


if __name__ == "__main__":
    main()
