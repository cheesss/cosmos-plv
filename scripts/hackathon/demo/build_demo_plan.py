#!/usr/bin/env python3
"""Write configs/hackathon/demo_plan.json from the dashboard recording's timeline (record_dashboard.py), so clip
boundaries follow the real session instead of hand-typed seconds."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
S = "data/outputs/hackathon_demo/session.webm"
t = json.loads((ROOT / "data/outputs/hackathon_demo/timeline.json").read_text())
calls = sum(len((ROOT / p).read_text().splitlines())
            for p in ("results/hackathon/nebius_calls.jsonl", "results/hackathon/explore/nebius_calls_explore.jsonl"))

plan = [
    {"kind": "card", "seconds": 7, "title": "Cosmos-PLV",
     "body": "Which robot-learning checkpoint should I evaluate next, and why?\n"
             "Nebius x NVIDIA Global AI Hackathon 2026 - Physical AI track\n\n"
             "Dashboard clips are real screen recordings of the running application\n"
             "(live metric script, live Cosmos3-Edge, live Nebius call); speed-ups are labelled."},
    {"kind": "image", "path": "media/hackathon/checkpoint_comparison.png", "seconds": 8,
     "caption": "Problem: offline losses do not tell checkpoints apart, and every closed-loop rollout costs robot time. "
                "93 retrained Diffusion Policy checkpoints, 4,650 simulator rollouts."},
    {"kind": "image", "path": "media/hackathon/architecture.png", "seconds": 8,
     "caption": "Pipeline: cheap PLV screening -> real robosuite rollouts -> NVIDIA Cosmos3-Edge (local GPU) + "
                "NVIDIA Nemotron-3.5-Super-VL served by Nebius Token Factory."},
    {"kind": "clip", "path": S, "start": t["selected"], "end": t["recommend_done"], "speed": 1.0,
     "caption": "Live dashboard: the pick, the free latest-epoch baseline, each rule's measured regret, and a "
                "judge-trust warning for strong checkpoints (from the pre-registered c1 audit)."},
    {"kind": "clip", "path": S, "start": t["cheap_start"], "end": t["cheap_done"] + 2, "speed": 2.5,
     "caption": "Cheap screening runs the ORIGINAL PLV script on this checkpoint (Action L1, open-loop smoothness) - live, 2.5x speed."},
    {"kind": "sidebyside", "paths": ["data/outputs/hackathon_rollouts_eval/can_s1_e0000/videos/seed200000.mp4",
                                     "data/outputs/hackathon_rollouts_eval/can_s2_e0100/videos/seed200008.mp4"],
     "labels": ["epoch 0: never grasps (sim label: no_grasp)", "epoch 100: picks and places (sim label: success)"],
     "seconds": 8, "caption": "Real simulator rollouts recorded by the pipeline (robosuite / MuJoCo, 10 fps, 2x speed); "
                              "privileged labels are used only as ground truth."},
    {"kind": "clip", "path": S, "start": t["roll_tab"], "end": t["cosmos_start"], "speed": 1.5,
     "caption": "Rollout viewer: real episode video, verified simulator outcome, and Cosmos3-Edge's per-keyframe answers "
                "to the task-free question 'where is the can?' (1.5x)."},
    {"kind": "clip", "path": S, "start": t["cosmos_start"], "end": t["cosmos_done"] + 1.5, "speed": 2.0,
     "caption": "Live: NVIDIA Cosmos3-Edge on the local RTX 3090 (load + 8 keyframes, 2x speed). A fixed rule turns "
                "visual states into grasp / drop / success."},
    {"kind": "clip", "path": S, "start": t["nebius_start"] - 1, "end": t["nebius_done"] + 6, "speed": 1.5,
     "caption": "Live: the same episode goes to NVIDIA Nemotron-3.5-Super-VL on Nebius Token Factory; request id, "
                "latency and tokens are logged as evidence."},
    {"kind": "clip", "path": S, "start": t["results"], "end": t["results_done"], "speed": 1.4,
     "caption": "Results on 330 real rollouts: Nemotron judges success with 0.964 accuracy, Cosmos3-Edge 0.918. "
                "Failure TYPES are not solved (below a majority baseline)."},
    {"kind": "clip", "path": S, "start": t["audits"], "end": t["audits_done"], "speed": 1.0,
     "caption": "When can you trust the judge? Pre-registered audits: judges cannot tell when a replan helps (r1), and "
                "they accept failures that end at the goal bin, so strong checkpoints look better than they are (c1)."},
    {"kind": "image", "path": "media/hackathon/c1_wrong_compartment.png", "seconds": 7,
     "caption": "The blind spot: a can in the wrong compartment (left) is a simulator failure, but Nemotron calls it a "
                "success - and still accepts 24 of 33 such cans when the prompt states the compartment."},
    {"kind": "image", "path": "media/hackathon/self_training_s2.png", "seconds": 9,
     "caption": "Using the verdicts to self-train (s2, pre-registered): 36 human demos add +8 points; Nemotron-filtered "
                "self-data costs 10 points, because the blind spot let 58 failures into the training set."},
    {"kind": "image", "path": "media/hackathon/cost_latency.png", "seconds": 6,
     "caption": f"Measured cost: about 2-3 s per episode per judge. All {calls:,} logged Nebius calls of the project "
                "cost about $5 at the catalog price."},
    {"kind": "card", "seconds": 14, "title": "What we learned (all measured)",
     "body": "+ Zero-shot NVIDIA judges replace a privileged success detector: 0.92-0.96 accuracy from video.\n"
             "+ Adaptive allocation needs 2-3x fewer rollouts; early abort saves 16-20% robot time.\n"
             "- No PLV metric beats the free 'latest epoch' pick in these runs.\n"
             "- Judges miss near-goal failures, so strong checkpoints look better than they are.\n"
             "- Self-training on their verdicts cost 10 points: audit the judge before you train on it.\n"
             "Models: nvidia/Cosmos3-Edge (local), nvidia/Nemotron-3.5-Super-VL (Nebius Token Factory)."},
]
total = sum(s.get("seconds", 0) for s in plan) + sum((s["end"] - s["start"]) / s["speed"] for s in plan if s["kind"] == "clip")
(ROOT / "configs/hackathon/demo_plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
print(f"plan written: {len(plan)} segments, {total:.1f} s")
