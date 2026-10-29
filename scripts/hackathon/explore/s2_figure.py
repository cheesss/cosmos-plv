#!/usr/bin/env python3
"""s2 figure: paired eval success per arm with 95% seed-bootstrap CIs -> media/hackathon/self_training_s2.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E = ROOT / "data/outputs/hackathon_explore/s2_selfimprove/eval"
ARMS = [("base", "base policy\n(36 demos, epoch 125)", "#9aa0a6"), ("continue", "continue training\non the 36 demos", "#2F4B7C"),
        ("oracle", "+ own successes\n(simulator labels)", "#55A868"), ("nemotron", "+ own episodes judged\nsuccessful by Nemotron", "#8E44AD"),
        ("human", "+ 36 human demos\n(positive control)", "#f5a623")]
rng = np.random.default_rng(0)
vals, los, his = [], [], []
for arm, _, _ in ARMS:
    x = np.array([e["success"] for e in json.loads((E / arm / "summary.json").read_text())["episodes"]], float)
    b = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    vals.append(x.mean()); los.append(x.mean() - np.percentile(b, 2.5)); his.append(np.percentile(b, 97.5) - x.mean())
fig, ax = plt.subplots(figsize=(12, 5.6))
ax.bar(range(len(ARMS)), vals, yerr=[los, his], capsize=6, color=[c for *_, c in ARMS])
for i, v in enumerate(vals):
    ax.text(i, v + his[i] + 0.012, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(range(len(ARMS)), [l for _, l, _ in ARMS], fontsize=10.5)
ax.set_ylim(0.5, 1.02)
ax.set_ylabel("closed-loop success (500 paired eval seeds)")
ax.set_title("s2 (pre-registered): self-training on Nemotron-judged successes made the policy WORSE\n"
             "Nemotron accepted 58/149 failures (44 cans in the wrong compartment); human demos help, own successes do not",
             fontsize=11.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = ROOT / "media/hackathon/self_training_s2.png"
fig.savefig(out, dpi=130)
print(out)
