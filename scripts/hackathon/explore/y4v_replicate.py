#!/usr/bin/env python
"""Verification of y4_grounded_reports (EXPLORATORY): small fresh-sample replication with the
FROZEN v2 prompts (prompt text built by y4_grounded_reports.py, unchanged), at most 13 Nebius calls.

  A) 10 calls: one new sample for each of the 10 test checkpoints where RULE_CKPT says "yes"
     -> does the LLM still answer spend_more_rollouts = "no"?
  B) 3 calls: run-level can_s1 (max_tokens 1500) -> does the template-echo failure recur?
Outputs go to data/outputs/hackathon_explore/y4_grounded_reports_verification/replicate.jsonl;
evidence via NebiusVLMClient to results/hackathon/explore/nebius_calls_explore.jsonl with tag
explore/y4_grounded_reports_verification/...
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
import y4_grounded_reports as G  # noqa: E402
OUT = ROOT / "data/outputs/hackathon_explore/y4_grounded_reports_verification"
TAG = "explore/y4_grounded_reports_verification"
MAX_CALLS = 13


def main():
    from cosmos_plv.nebius_backend import NebiusVLMClient
    facts, runs, truth = G.load()
    test = [c for c in sorted(facts) if c not in G.DEV_CKPTS and facts[c]["nemotron"]]
    yes = [c for c in test if truth[c]["rule_ckpt"] == "yes"]
    jobs = [("ckpt", c, 1200) for c in yes] + [("run", "can_s1", 1500)] * 3
    assert len(jobs) <= MAX_CALLS, len(jobs)
    cl = NebiusVLMClient(model_id=G.MODEL, evidence_path=G.EVIDENCE, timeout_s=180.0)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "replicate.jsonl", "a") as f:
        for i, (kind, key, mt) in enumerate(jobs):
            prompt = G.ckpt_prompt(facts[key], "v2") if kind == "ckpt" else G.run_prompt(runs[key], "v2")
            tag = f"{TAG}/{kind}/v2/{key}/r{i}"
            r = cl.generate(prompt, max_tokens=mt, reasoning=False, temperature=0.2, system=G.SYSTEM, tag=tag)
            rec = {"kind": kind, "key": key, "tag": tag, "latency_s": r.latency_s, "error": r.error,
                   "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                   "request_id": r.request_id, "raw_text": r.raw_text, "logged_unix": time.time()}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(i, kind, key, "err" if r.error else "ok", r.latency_s, flush=True)


if __name__ == "__main__":
    main()
