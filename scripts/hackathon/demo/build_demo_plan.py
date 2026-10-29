#!/usr/bin/env python3
"""Write configs/hackathon/demo_plan.json from the dashboard recording's timeline (record_dashboard.py), so clip
boundaries follow the real session instead of hand-typed seconds. Captions quote only numbers that are in the results
files (see submission/claim_ledger.csv); figures come from scripts/hackathon/make_submission_figures.py."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S = "data/outputs/hackathon_demo/session.webm"
F = "submission/figures"
t = json.loads((ROOT / "data/outputs/hackathon_demo/timeline.json").read_text())
e2 = {r["backend"]: r for r in json.loads((ROOT / "results/hackathon/experiment2_summary.json").read_text())["by_backend_variant"]}
s2 = json.loads((ROOT / "results/hackathon/explore/s2_self_improve.json").read_text())
c1 = json.loads((ROOT / "results/hackathon/explore/c1_coverage_shift.json").read_text())
nem, cos = e2["nebius"], e2["local"]
base = (nem["tp"] + nem["fn"]) / nem["episodes"]
wc = c1["nemotron"]["false_success_by_final_place"]["goal_bin_other_compartment"]
v2c = c1["nemotron_v2c_compartment"]["false_success_by_final_place"]["goal_bin_other_compartment"]
p, h = s2["paired"]["primary_nemotron_minus_continue"], s2["paired"]["human_minus_continue"]
nl = s2["nemotron_labels"]
calls = sum(len((ROOT / f).read_text().splitlines())
            for f in ("results/hackathon/nebius_calls.jsonl", "results/hackathon/explore/nebius_calls_explore.jsonl"))

nar = {d["line"]: d for d in json.loads((ROOT / "submission/narration/durations.json").read_text())["lines"]}
seg_line = {sg: d for d in nar.values() for sg in d["segments"]}


def secs(index, minimum=4.0):
    """still segments last as long as their narration line (+ lead-in and a short pause)"""
    d = seg_line.get(index)
    return round(max(minimum, (d["duration_s"] + 0.7) if d else minimum), 2)


plan = [
    {"kind": "sidebyside", "paths": ["data/outputs/hackathon_rollouts_eval/can_s1_e0000/videos/seed200000.mp4",
                                     "data/outputs/hackathon_rollouts_eval/can_s2_e0100/videos/seed200008.mp4"],
     "labels": ["epoch 0: never grasps (simulator: failure)", "epoch 100: picks and places (simulator: success)"],
     "seconds": secs(0), "tag": "REAL robosuite / MuJoCo rollouts recorded by the pipeline"},
    {"kind": "card", "seconds": secs(1), "title": "Cosmos-PLV",
     "body": "Which robot checkpoint should I evaluate next - and can I trust the AI judge?\n\n"
             "For robot-learning researchers who train many imitation-learning checkpoints.\n"
             "Nebius x NVIDIA Global AI Hackathon 2026 - Physical AI track\n\n"
             "Dashboard clips are real screen recordings of the running application;\n"
             "speed-ups are labelled in the top-left corner."},
    {"kind": "image", "path": f"{F}/system_architecture.png", "seconds": secs(2), "tag": "system architecture"},
    {"kind": "clip", "path": S, "start": t["selected"], "end": t["recommend_done"] + 0.3, "speed": 1.0,
     "tag": "LIVE screen recording - 1x"},
    {"kind": "clip", "path": S, "start": t["cheap_start"], "end": t["cheap_done"] + 2, "speed": 2.4,
     "tag": "LIVE - original PLV metric script - 2.4x"},
    {"kind": "clip", "path": S, "start": t["roll_tab"], "end": t["cosmos_start"], "speed": 1.9,
     "tag": "LIVE screen recording - recorded rollout video - 1.9x"},
    # The recorder's "done" marks fire about 3.5 s before the live answer is drawn (Streamlit re-renders replace the
    # spinner node); the offsets below were checked frame by frame on this session (submission/video_validation.md).
    {"kind": "clip", "path": S, "start": t["cosmos_start"], "end": t["cosmos_done"] + 3.6, "speed": 3.0,
     "tag": "LIVE - NVIDIA Cosmos3-Edge on the local RTX 3090 - 3x while waiting"},
    {"kind": "clip", "path": S, "start": t["cosmos_done"] + 3.6, "end": t["cosmos_done"] + 7.8, "speed": 1.0,
     "tag": "LIVE - Cosmos3-Edge answer - 1x"},
    {"kind": "clip", "path": S, "start": t["nebius_start"] - 1, "end": t["nebius_done"] + 3.6, "speed": 2.0,
     "tag": "LIVE - request to Nebius Token Factory - 2x while waiting"},
    {"kind": "clip", "path": S, "start": t["nebius_done"] + 3.6, "end": t["nebius_done"] + 9.0, "speed": 1.0,
     "tag": "LIVE - Nemotron-3.5-Super-VL verdict from Nebius (request id, latency, tokens) - 1x"},
    {"kind": "clip", "path": S, "start": t["results"], "end": t["results_done"], "speed": 1.7,
     "tag": "LIVE screen recording - 1.7x"},
    {"kind": "image", "path": f"{F}/judge_accuracy.png", "seconds": secs(11), "tag": f"measured: {nem['episodes']} recorded rollouts"},
    {"kind": "image", "path": f"{F}/evaluation_efficiency.png", "seconds": secs(12), "tag": "EXPLORATORY, post-hoc"},
    {"kind": "clip", "path": S, "start": t["audits"], "end": t["audits_done"], "speed": 1.15,
     "tag": "LIVE screen recording - pre-registered audits - 1.15x"},
    {"kind": "image", "path": f"{F}/real_case_wrong_compartment.png", "seconds": secs(14),
     "tag": "real c1 episode and Nemotron's recorded answer"},
    {"kind": "image", "path": f"{F}/accuracy_vs_blind_spot.png", "seconds": secs(15), "tag": "pre-registered s2 study"},
    {"kind": "image", "path": "media/hackathon/cost_latency.png", "seconds": secs(16), "tag": "measured from the call logs"},
    {"kind": "card", "seconds": secs(17) + 0.8, "title": "Audit the judge before you train on it",
     "body": f"+ Zero-shot NVIDIA judges replace a privileged success detector ({100 * cos['success_acc']:.1f}-"
             f"{100 * nem['success_acc']:.1f}% accuracy).\n"
             "+ Adaptive evaluation cuts rollouts (exploratory); the free 'latest epoch' pick\n"
             "   was not beaten by PLV metrics in these runs.\n"
             "- Judges miss near-goal failures; self-training on their labels cost 10 points.\n\n"
             "github.com/cheesss/cosmos-plv\n"
             "NVIDIA Cosmos3-Edge (local GPU) - NVIDIA Nemotron-3.5-Super-VL on Nebius Token Factory"},
]
total = sum(s.get("seconds", 0) for s in plan) + sum((s["end"] - s["start"]) / s["speed"] for s in plan if s["kind"] == "clip")
live = sum((s["end"] - s["start"]) / s["speed"] for s in plan if s["kind"] == "clip") + plan[0]["seconds"]
for k, sgm in enumerate(plan):
    sgm.setdefault("caption", "")
(ROOT / "configs/hackathon/demo_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
print(f"plan written: {len(plan)} segments, {total:.1f} s (real application + simulator footage {live:.1f} s)")
