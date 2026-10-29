#!/usr/bin/env python3
"""y3_visual_robustness debug (Env A, run under the GPU lock): first rendered frame of 3 episodes
WITHOUT closing envs between episodes, vs the recorded video (PSNR printed by a follow-up check)."""
import sys, json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
from cosmos_plv.rollout_recorder import _make_env, _render
from cosmos_plv.video_recorder import read_mp4
D=ROOT/'data/outputs/hackathon_explore/y3_visual_robustness/debug'
eps=json.loads((ROOT/'results/hackathon/explore/y3_visual_robustness_episodes.json').read_text())['episodes'][:3]
OBS=["object","robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos"]
for i,ep in enumerate(eps):
    env=_make_env(ROOT/f"data/robomimic/datasets/{ep['task']}/ph/low_dim_abs.hdf5",OBS,True,True,(448,448),'agentview')
    env.seed(int(ep['env_seed'])); env.reset()
    a=_render(env).copy()
    orig=read_mp4(ROOT/ep['video'])[0]
    imageio.imwrite(D/f'ep{i}_replay0.png', a); imageio.imwrite(D/f'ep{i}_orig0.png', orig)
    print(i, ep['checkpoint_id'], ep['env_seed'], a.shape, a.mean(), orig.mean(), flush=True)
    # do NOT close: test whether closing caused the mismatch
