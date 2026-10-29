"""Stage C: real closed-loop robosuite rollouts with synchronized video + state.

Each episode records, per low-level environment step (20 Hz):
    executed absolute action (7-D: pos, axis-angle, gripper), end-effector
    position, object position, finger-pad grasp contact, task success;
and per policy call (every n_action_steps steps):
    the full predicted action chunk ``action_pred`` (horizon x 10, rotation_6d),
    which supports execution-matched replanning diagnostics on REAL feedback;
and, for video episodes, an agentview RGB frame every ``steps_per_frame`` steps.

Seeds are explicit; diffusion sampling noise is seeded per batch.  The rollout
loop mirrors diffusion_policy/env_runner/robomimic_lowdim_runner.py
(obs history padded with the earliest obs, success = any step success).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .failure_annotator import annotate_episode
from .provenance import REPO_ROOT, repo_relative, utc_now
from .sim_probe import make_probe
from .video_recorder import write_mp4


@dataclass
class RecorderConfig:
    render_hw: tuple = (448, 448)
    camera: str = "agentview"
    steps_per_frame: int = 2  # 20 Hz sim -> 10 fps video
    hold_after_success: int = 16  # keep recording briefly after success
    batch_size: int = 20
    sampling_seed: int = 0
    device: str = "cuda:0"


def _make_env(
    dataset_path: Path,
    obs_keys: List[str],
    abs_action: bool,
    render: bool,
    render_hw: tuple = (448, 448),
    camera: str = "agentview",
):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    from diffusion_policy.common.robomimic_util import set_absolute_controller_config
    from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
    from diffusion_policy.env_runner.robomimic_lowdim_runner import (
        make_robosuite_metadata_compatible,
    )

    env_meta = FileUtils.get_env_metadata_from_dataset(str(dataset_path))
    env_meta = make_robosuite_metadata_compatible(env_meta)
    if abs_action:
        set_absolute_controller_config(env_meta["env_kwargs"]["controller_configs"])
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=render, use_image_obs=False
    )
    return RobomimicLowdimWrapper(
        env=env, obs_keys=obs_keys, init_state=None, render_hw=tuple(render_hw), render_camera_name=camera
    )


def _render(env) -> np.ndarray:
    """Render one env; several offscreen EGL contexts share this process, so
    make this env's context current first (robosuite does not do it per call)."""
    ctx = getattr(env.env.env.sim, "_render_context_offscreen", None)
    if ctx is not None:
        ctx.gl_ctx.make_current()
    return env.render(mode="rgb_array")


def _stack_last_n(history: deque, n: int) -> np.ndarray:
    items = list(history)[-n:]
    if len(items) < n:
        items = [items[0]] * (n - len(items)) + items
    return np.stack(items, axis=0)


def record_checkpoint(
    ckpt_path: Path,
    out_dir: Path,
    seeds: Sequence[int],
    video_seeds: Optional[Sequence[int]] = None,
    cfg: Optional[RecorderConfig] = None,
    dataset_cache: Optional[dict] = None,
    checkpoint_id: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Run ``len(seeds)`` episodes of one checkpoint; returns episode summaries."""
    import torch

    from diffusion_policy.model.common.rotation_transformer import RotationTransformer
    from diffusion_policy.scripts.plv_chunk_aware_checkpoint_metrics import set_sampling_seed
    from diffusion_policy.scripts.plv_policy_induced_sim_calibration import undo_abs_action

    from .checkpoint_adapter import load_policy

    cfg = cfg or RecorderConfig()
    out_dir = Path(out_dir)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    video_seeds = set(seeds if video_seeds is None else video_seeds)

    t_load = time.time()
    task_cfg, policy, _dataset, ex = load_policy(ckpt_path, cfg.device, dataset_cache)
    load_s = time.time() - t_load
    task = str(task_cfg.task.task_name)
    dataset_path = REPO_ROOT / ex["dataset_path"]
    transformer = RotationTransformer("axis_angle", "rotation_6d") if ex["abs_action"] else None
    n_obs, n_act, max_steps = ex["n_obs_steps"], ex["n_action_steps"], ex["max_steps"]
    a_off = ex["action_offset"]

    summaries: List[Dict[str, object]] = []
    seeds = list(seeds)
    for b0 in range(0, len(seeds), cfg.batch_size):
        batch = seeds[b0 : b0 + cfg.batch_size]
        envs = [
            _make_env(dataset_path, ex["obs_keys"], ex["abs_action"], s in video_seeds, cfg.render_hw, cfg.camera)
            for s in batch
        ]
        probes, hist, traces, frames, plans, done, hold = [], [], [], [], [], [], []
        for env, seed in zip(envs, batch):
            env.seed(seed)
            obs = env.reset()
            probes.append(make_probe(task, env.env.env))
            hist.append(deque([obs], maxlen=n_obs + 1))
            s0 = probes[-1].sample()
            traces.append({k: [v] for k, v in s0.items()} | {"action": [], "t": [0]})
            frames.append([_render(env)] if seed in video_seeds else [])
            plans.append({"t": [], "action_pred": []})
            done.append(False)
            hold.append(-1)
        if hasattr(policy, "reset"):
            policy.reset()
        set_sampling_seed(cfg.sampling_seed + 7919 * (b0 // cfg.batch_size))

        t0 = time.time()
        t = 0
        n_calls = 0
        while t < max_steps and not all(done):
            obs_hist = np.stack([_stack_last_n(h, n_obs) for h in hist]).astype(np.float32)
            with torch.no_grad():
                out = policy.predict_action({"obs": torch.from_numpy(obs_hist).to(cfg.device)})
            n_calls += 1
            action = out["action"].detach().cpu().numpy()[:, ex["n_latency_steps"] :]
            action_pred = out["action_pred"].detach().cpu().numpy()
            if not np.all(np.isfinite(action)):
                raise RuntimeError("NaN/Inf action from policy")
            env_action = undo_abs_action(action, transformer)
            for i in range(len(envs)):
                if not done[i]:
                    plans[i]["t"].append(t)
                    plans[i]["action_pred"].append(action_pred[i])
            for k in range(env_action.shape[1]):
                if t >= max_steps:
                    break
                t += 1
                for i, env in enumerate(envs):
                    if done[i]:
                        continue
                    obs, _r, _d, _info = env.step(env_action[i, k])
                    hist[i].append(obs)
                    s = probes[i].sample()
                    for key, val in s.items():
                        traces[i][key].append(val)
                    traces[i]["action"].append(env_action[i, k].astype(np.float32))
                    traces[i]["t"].append(t)
                    if batch[i] in video_seeds and t % cfg.steps_per_frame == 0:
                        frames[i].append(_render(env))
                    if s["success"] and hold[i] < 0:
                        hold[i] = t + cfg.hold_after_success
                    if hold[i] >= 0 and t >= hold[i]:
                        done[i] = True
        wall = time.time() - t0

        for i, seed in enumerate(batch):
            tr = {k: np.asarray(v) for k, v in traces[i].items()}
            ann = annotate_episode(tr)
            ep_path = out_dir / "episodes" / f"seed{seed}.npz"
            np.savez_compressed(
                ep_path,
                plan_t=np.asarray(plans[i]["t"], dtype=np.int32),
                action_pred=np.asarray(plans[i]["action_pred"], dtype=np.float32),
                **{k: v for k, v in tr.items()},
            )
            video_path = None
            if frames[i]:
                video_path = out_dir / "videos" / f"seed{seed}.mp4"
                write_mp4(video_path, frames[i], fps=20 // cfg.steps_per_frame)
            summaries.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_path": repo_relative(ckpt_path),
                    "task": task,
                    "env_seed": int(seed),
                    "sampling_seed": int(cfg.sampling_seed + 7919 * (b0 // cfg.batch_size)),
                    "batch_index": int(b0 // cfg.batch_size),
                    "episode_npz": repo_relative(ep_path),
                    "video": repo_relative(video_path) if video_path else None,
                    "video_fps": 20 // cfg.steps_per_frame if video_path else None,
                    "video_frames": len(frames[i]) if video_path else 0,
                    "camera": cfg.camera,
                    "render_hw": list(cfg.render_hw),
                    "batch_wall_s": round(wall, 2),
                    "policy_calls": n_calls,
                    "load_s": round(load_s, 2),
                    "recorded_utc": utc_now(),
                    "horizon": ex["horizon"],
                    "n_obs_steps": n_obs,
                    "n_action_steps": n_act,
                    "action_offset": a_off,
                    "num_inference_steps": ex["num_inference_steps"],
                    **ann,
                }
            )
        for env in envs:
            try:
                env.env.env.close()
            except Exception:
                pass
    return summaries
