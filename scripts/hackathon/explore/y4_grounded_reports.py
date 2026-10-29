#!/usr/bin/env python
"""y4_grounded_reports, step 2: LLM-written per-checkpoint / per-run reports + automatic grounding check.

EXPLORATORY / post-hoc. Text LLM: nvidia/Nemotron-3_5-Lightning on Nebius Token Factory
(confirmed via NebiusVLMClient.list_models()). Inputs are the fact sheets produced by
y4_build_facts.py (measured facts only; no target S_i).

Dev/test split (fixed rule, decided before any report was generated):
  dev checkpoints  = first and last eval checkpoint of can_s2 (e0000, e0200) and the two middle
                     positions of the square_s1 checkpoints with diagnostics (e0090, e0120)
  dev run          = can_s2;  test runs = can_s1, square_s1
  test checkpoints = every other checkpoint with Nemotron diagnostics; checkpoints without
                     diagnostics form a separate "missing data" stress set.
Prompt iteration: at most 3 versions, evaluated on dev only; frozen version = lowest dev grounding
error count (unsupported numbers + miscited numbers + wrong failure claims + wrong dominant mode),
ties -> fewer completion tokens.

Usage:
  y4_grounded_reports.py gen --phase dev --variant v1 --samples 2
  y4_grounded_reports.py gen --phase test --variant v2 --samples 3
  y4_grounded_reports.py gen --phase run_dev --variant v1 --samples 2
  y4_grounded_reports.py gen --phase run_test --variant v2 --samples 3
  y4_grounded_reports.py verify --variant v1 --phase dev     (print verifier output)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/hackathon/explore"))
import y4_verify as V  # noqa: E402

OUT = ROOT / "data/outputs/hackathon_explore/y4_grounded_reports"
GEN = OUT / "generations.jsonl"
EVIDENCE = str(ROOT / "results/hackathon/explore/nebius_calls_explore.jsonl")
MODEL = "nvidia/Nemotron-3_5-Lightning"
TAG = "explore/y4_grounded_reports"
MAX_CALLS = 700
DEV_CKPTS = ["can_s2_e0000", "can_s2_e0200", "square_s1_e0090", "square_s1_e0120"]
DEV_RUNS = ["can_s2"]
TEST_RUNS = ["can_s1", "square_s1"]

SYSTEM = ("You write short, strictly factual evaluation reports about robot-learning checkpoints. "
          "You never state a number that is not in the provided fact sheet.")

SHEET_GUIDE = """The FACT SHEET (JSON) describes one Diffusion Policy checkpoint trained on the robomimic "{task}" task. It holds measured facts only:
- sim: outcomes of the 10 evaluation rollouts judged from privileged simulator state (GROUND TRUTH). failure_types counts failed episodes per type: no_grasp = never grasped the object; dropped = grasped/lifted it, then lost it; not_placed = held it but did not place it in the target; other.
- nemotron: the same 10 rollouts judged from video by the vision-language model Nemotron-3.5-Super-VL (agree_with_sim = episodes where its success/failure verdict matches the simulator; judged_failure_types = its own failure-type guesses; failure_type_matches_sim = failed episodes where its type equals the simulator type). nemotron_padded: same judge on clips padded to equal length. cosmos: Cosmos3-Edge judging the last frame. A null value means not measured.
- cheap_metrics: offline metrics computed without rollouts (lower = better). run_context: position within the training run (rank 1 = best; ties share a rank).
- episodes: per-episode records keyed by env seed; nemotron_note is the VLM's one-sentence description (it can be wrong; the simulator label is ground truth)."""

CKPT_FORMAT = """Return ONLY a JSON object with exactly these keys:
{{"checkpoint_id": "{cid}",
 "physical_summary": "1-3 sentences: what this checkpoint does well and badly, physically",
 "dominant_failure_mode": "one of no_grasp | dropped | not_placed | other | none (by simulator labels)",
 "failure_mode_evidence": "1-2 sentences",
 "spend_more_rollouts": "yes or no",
 "recommendation_reason": "1-2 sentences: should more evaluation rollouts be spent on this checkpoint, and why"}}"""

GROUNDING_V1 = """Grounding rules:
1. Every number you write must be copied from the FACT SHEET and followed immediately by its key in square brackets, e.g. "6/10 [sim.successes]", "0.313 [sim.success_rate_wilson95_lo]", "9.4 s [episodes.200000.sim_success_time_s]".
2. Do not compute new numbers (no differences, sums, averages or percentages that are not in the sheet).
3. Treat the simulator as ground truth; say so when the VLM judges disagree with it.
4. If a field is null or missing, say it is unavailable; do not guess."""

# v2/v3 written after inspecting the v1 dev outputs (dev checkpoints only). v1 dev problems:
# citations malformed ("[a=10, b=0]", key lists), counts attributed to the wrong judge ("Nemotron
# judges all 10 as no_grasp" when the sheet says 7), absent failure types mentioned, and the
# recommendation logic "10/10 -> no more rollouts", which is the opposite of the screening goal.
OBJECTIVE = """Purpose of the recommendation: the researcher wants to find the best checkpoint of this training run while spending as few extra rollouts as possible. 10 rollouts cannot separate checkpoints whose success counts are close, so extra rollouts are worth spending on checkpoints that are contenders for the best in the run (judged success near the top of the run, see run_context; good cheap-metric ranks when available) and are wasted on checkpoints that are clearly worse than others in the run. A perfect 10/10 does not by itself make more rollouts unnecessary if other checkpoints of the run also reach a similar count."""

GROUNDING_V2 = GROUNDING_V1 + """
5. Cite exactly one key per bracket, written exactly as in the sheet, with nothing else inside the bracket (no "=value", no lists). Write counts as "k/10 [key]"; do not convert them to percentages.
6. The simulator, Nemotron and Cosmos are different sources and often disagree. Every count must name its source in the same clause ("the simulator recorded ...", "Nemotron judged ...") and cite that source's key.
7. Name failure types only with the exact tokens no_grasp, dropped, not_placed, other, and only types whose count in sim.failure_types or nemotron.judged_failure_types is above 0. Do not mention types that did not occur.

""" + OBJECTIVE

GROUNDING_V3 = GROUNDING_V2 + """

Before answering, re-check every bracketed key: it must exist in the FACT SHEET exactly as written, and the number in front of it must equal that key's value. Remove any sentence you cannot support."""

GROUNDING = {"v1": GROUNDING_V1, "v2": GROUNDING_V2, "v3": GROUNDING_V3}

RUN_GUIDE = """The RUN FACT SHEET (JSON) summarizes the evaluation checkpoints of one Diffusion Policy training run on the robomimic "{task}" task. Per checkpoint (keyed by checkpoint_id): epoch; sim_successes = successes in 10 evaluation rollouts judged from privileged simulator state (GROUND TRUTH); sim_failure_types = simulator failure-type counts (no_grasp, dropped, not_placed, other); nemotron_judged_successes / nemotron_agree_with_sim = the vision-language judge Nemotron-3.5-Super-VL on the same videos; cosmos_judged_successes = Cosmos3-Edge on the last frame; cheap offline metrics (lower = better) and *_rank_in_run (1 = best). A null value means not measured."""

RUN_FORMAT = """A researcher can afford more evaluation rollouts for only one checkpoint of this run next. Return ONLY a JSON object with exactly these keys:
{{"run": "{run}",
 "run_summary": "2-4 sentences: how physical performance and failure modes change over training",
 "evaluate_next": "one checkpoint_id from the sheet",
 "evaluate_next_reason": "1-3 sentences",
 "alternatives": ["0-2 other checkpoint_ids"],
 "caveats": "1 sentence"}}"""

RUN_GROUNDING_EXTRA = {
    "v1": """Grounding rules:
1. Every number you write must be copied from the RUN FACT SHEET and followed immediately by its key in square brackets, e.g. "9/10 [checkpoints.{ex}.sim_successes]".
2. Do not compute new numbers (no differences, sums, averages or percentages that are not in the sheet).
3. Treat the simulator as ground truth.
4. Use only checkpoint_ids that appear in the sheet.""",
}
RUN_OBJECTIVE = """Purpose: the researcher wants to find the best checkpoint of this run with as few extra rollouts as possible; extra rollouts should go to the checkpoint whose current evidence makes it the strongest contender for the best (10 rollouts cannot separate close checkpoints)."""
RUN_GROUNDING_EXTRA["v2"] = RUN_GROUNDING_EXTRA["v1"] + """
5. Cite exactly one key per bracket, written exactly as in the sheet, nothing else inside the bracket. Write counts as "k/10 [key]"; do not convert them to percentages.
6. The simulator, Nemotron and Cosmos are different sources; every count must name its source in the same clause and cite that source's key.
7. Name failure types only with the exact tokens no_grasp, dropped, not_placed, other; put the checkpoint_id in the same sentence; never say a type "disappeared" or "never occurred" unless every later checkpoint has a count of 0 for it.
8. Describe a trend in a metric only if you cite the values at both ends.

""" + RUN_OBJECTIVE
RUN_GROUNDING_EXTRA["v3"] = RUN_GROUNDING_EXTRA["v2"] + """

Before answering, re-check every bracketed key: it must exist in the sheet exactly as written, and the number in front of it must equal its value. Remove any sentence you cannot support."""


def ckpt_prompt(facts: dict, variant: str) -> str:
    return "\n\n".join([
        SHEET_GUIDE.format(task=facts["task"]),
        CKPT_FORMAT.format(cid=facts["checkpoint_id"]),
        GROUNDING[variant],
        "FACT SHEET:\n" + json.dumps(facts, separators=(",", ":")),
    ])


def run_prompt(rf: dict, variant: str) -> str:
    ex = next(iter(rf["checkpoints"]))
    return "\n\n".join([
        RUN_GUIDE.format(task=rf["task"]),
        RUN_FORMAT.format(run=rf["run"]),
        RUN_GROUNDING_EXTRA[variant].format(ex=ex),
        "RUN FACT SHEET:\n" + json.dumps(rf, separators=(",", ":")),
    ])


def load():
    facts = json.loads((OUT / "fact_sheets_ckpt.json").read_text())
    runs = json.loads((OUT / "fact_sheets_run.json").read_text())
    truth = json.loads((OUT / "truth_rules.json").read_text())
    return facts, runs, truth


def phase_items(phase, facts, runs):
    if phase == "dev":
        return [("ckpt", c) for c in DEV_CKPTS]
    if phase == "test":
        return [("ckpt", c) for c in sorted(facts) if c not in DEV_CKPTS and facts[c]["nemotron"]]
    if phase == "missing":
        return [("ckpt", c) for c in sorted(facts) if not facts[c]["nemotron"]]
    if phase == "run_dev":
        return [("run", r) for r in DEV_RUNS]
    if phase == "run_test":
        return [("run", r) for r in TEST_RUNS]
    raise SystemExit(f"unknown phase {phase}")


def n_calls_so_far():
    if not Path(EVIDENCE).is_file():
        return 0
    n = 0
    for line in open(EVIDENCE):
        if f'"{TAG}/' in line:
            n += 1
    return n


_lock = threading.Lock()


def gen(args):
    from cosmos_plv.nebius_backend import NebiusVLMClient
    facts, runs, truth = load()
    items = phase_items(args.phase, facts, runs)
    jobs = [(kind, key, s) for kind, key in items for s in range(args.samples)]
    used = n_calls_so_far()
    if used + len(jobs) > MAX_CALLS:
        raise SystemExit(f"call budget: used {used}, requested {len(jobs)}, max {MAX_CALLS}")
    print(f"calls used so far {used}; launching {len(jobs)}", flush=True)
    local = threading.local()

    def client():
        if not hasattr(local, "c"):
            local.c = NebiusVLMClient(model_id=MODEL, evidence_path=EVIDENCE, timeout_s=180.0)
        return local.c

    def one(job):
        kind, key, s = job
        prompt = ckpt_prompt(facts[key], args.variant) if kind == "ckpt" else run_prompt(runs[key], args.variant)
        tag = f"{TAG}/{args.phase}/{args.variant}/{key}/s{s}"
        r = client().generate(prompt, max_tokens=args.max_tokens, reasoning=args.reasoning,
                              temperature=args.temperature, system=SYSTEM, tag=tag)
        rec = {"phase": args.phase, "variant": args.variant, "kind": kind, "key": key, "sample": s,
               "tag": tag, "model_id": MODEL, "served_model": r.served_model, "request_id": r.request_id,
               "temperature": args.temperature, "reasoning": args.reasoning, "max_tokens": args.max_tokens,
               "latency_s": r.latency_s, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
               "error": r.error, "raw_text": r.raw_text, "prompt_chars": len(prompt),
               "logged_unix": time.time()}
        with _lock:
            with open(GEN, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(one, jobs):
            print(rec["key"], rec["sample"], "err" if rec["error"] else "ok", rec["latency_s"],
                  rec["prompt_tokens"], rec["completion_tokens"], flush=True)


def load_generations(phase=None, variant=None):
    out = []
    if not GEN.is_file():
        return out
    for line in GEN.read_text().splitlines():
        d = json.loads(line)
        if phase and d["phase"] != phase:
            continue
        if variant and d["variant"] != variant:
            continue
        out.append(d)
    return out


def verify_rec(rec, facts, runs, truth):
    rep, status = V.parse_json_obj(rec["raw_text"])
    out = {"parse_status": status if not rec.get("error") else "api_error"}
    if rep is None or not isinstance(rep, dict):
        return rep, out
    if rec["kind"] == "ckpt":
        out.update(V.verify_ckpt_report(rep, facts[rec["key"]], truth[rec["key"]], prompt_constants=[10, 0.313, 9.4, 6]))
    else:
        out.update(V.verify_run_report(rep, runs[rec["key"]], truth[f"RUN::{rec['key']}"], facts))
    return rep, out


def verify_cmd(args):
    facts, runs, truth = load()
    for rec in load_generations(args.phase, args.variant):
        rep, v = verify_rec(rec, facts, runs, truth)
        print("=" * 100)
        print(rec["key"], rec["sample"], rec["variant"], v.get("parse_status"), "lat", rec["latency_s"],
              "tok", rec["prompt_tokens"], rec["completion_tokens"])
        if args.show:
            print(json.dumps(rep, indent=1) if rep else rec["raw_text"][:1500])
        keys = ["n_numbers", "n_unsupported", "n_cited_numbers", "n_miscited", "n_bad_key", "n_uncited_numbers",
                "n_failure_mentions", "n_wrong_failure_claims", "n_count_claims", "n_misattributed_counts", "dominant_reported", "dominant_correct",
                "recommend_reported", "rule", "rule_consistent", "evaluate_next", "rule_accepted", "invented_checkpoint_ids"]
        print({k: v[k] for k in keys if k in v})
        for u in v.get("unsupported", []):
            print("  UNSUPPORTED", u)
        for u in v.get("miscited", []):
            print("  MISCITED", u)
        for u in v.get("wrong_failure_claims", []):
            print("  WRONG-FT", {k: u[k] for k in ("type", "span", "negated", "count", "reason", "sentence")})
        for u in v.get("misattributed_counts", []):
            print("  MISATTR", u)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--phase", required=True)
    g.add_argument("--variant", required=True)
    g.add_argument("--samples", type=int, default=1)
    g.add_argument("--temperature", type=float, default=0.2)
    g.add_argument("--max_tokens", type=int, default=1200)
    g.add_argument("--reasoning", action="store_true")
    g.add_argument("--workers", type=int, default=4)
    v = sub.add_parser("verify")
    v.add_argument("--phase")
    v.add_argument("--variant")
    v.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.cmd == "gen":
        gen(a)
    else:
        verify_cmd(a)


if __name__ == "__main__":
    main()
