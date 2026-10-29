#!/usr/bin/env python3
"""CPU-only check of which Cosmos3-Edge action domains have trained action projections.

Reads only the four DomainAwareLinear tensors (action_proj_in/out .fc / .bias) from the
transformer safetensors; no model is built. A slot whose biases are exactly zero and whose
weights match the initialization statistics of the never-used slots was never trained.
Output: results/hackathon/explore/e5_domain_weights.json
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
SNAP = Path.home() / ".cache/huggingface/hub/models--nvidia--Cosmos3-Edge/snapshots/344d602b128d1bbdacb43b08d0a3626f46343e29/transformer"
NAMES = {0: "no_action", 1: "av", 2: "camera_pose", 3: "hand_pose", 4: "pusht", 5: "libero", 6: "umi", 7: "bridge_orig_lerobot",
         8: "droid_lerobot / robomind-franka", 9: "galbot", 12: "robomind-franka-dual", 13: "robomind-ur", 15: "agibotworld",
         20: "fractal", 30: "robocasa (framework only)"}


def main():
    wmap = json.loads((SNAP / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    f = safe_open(SNAP / wmap["action_proj_in.bias.weight"], "pt", device="cpu")
    bi, bo = f.get_tensor("action_proj_in.bias.weight"), f.get_tensor("action_proj_out.bias.weight")
    win = f.get_tensor("action_proj_in.fc.weight").float().view(32, 64, 2048)
    wout = f.get_tensor("action_proj_out.fc.weight").float().view(32, 2048, 64)
    rows = []
    for d in range(32):
        cn = wout[d].norm(dim=0)
        rows.append({"slot": d, "name": NAMES.get(d, ""), "bias_in_nonzero": int((bi[d] != 0).sum()),
                     "bias_out_nonzero": int((bo[d] != 0).sum()), "out_cols_norm_gt_0.8": int((cn > 0.8).sum()),
                     "max_abs_w_in": float(win[d].abs().max()), "max_abs_w_out": float(wout[d].abs().max())})
        rows[-1]["trained"] = rows[-1]["bias_in_nonzero"] > 0 or rows[-1]["bias_out_nonzero"] > 0
    res = {"model": "nvidia/Cosmos3-Edge@344d602b", "trained_slots": [r["slot"] for r in rows if r["trained"]], "slots": rows}
    out = ROOT / "results/hackathon/explore/e5_domain_weights.json"
    out.write_text(json.dumps(res, indent=2))
    for r in rows:
        if r["name"]:
            print(f"{r['slot']:2d} {r['name']:32s} trained={r['trained']!s:5s} bias_in {r['bias_in_nonzero']:4d} bias_out {r['bias_out_nonzero']:2d}"
                  f" |w_in|max {r['max_abs_w_in']:.4f} |w_out|max {r['max_abs_w_out']:.4f}")
    print("trained:", res["trained_slots"])


if __name__ == "__main__":
    main()
