if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from diffusion_policy.common.pytorch_util import dict_apply


OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass
class DynamicsEnsemble:
    weights: List[np.ndarray]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    tree: cKDTree
    support_progress: np.ndarray
    task_name: str = ""
    goal_pos: Optional[np.ndarray] = None
    goal_height: Optional[float] = None
    progress_mode: str = "semantic"
    episode_split: str = "train"
    n_support_episodes: int = 0
    n_support_steps: int = 0
    n_dynamics_transitions: int = 0

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def predict(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.concatenate([
            self.normalize_obs(obs),
            self.normalize_action(action),
            np.ones((obs.shape[0], 1), dtype=np.float32),
        ], axis=-1)
        preds = np.stack([x @ w for w in self.weights], axis=0)
        mean_norm = preds.mean(axis=0)
        var = preds.var(axis=0).mean(axis=-1)
        mean = mean_norm * self.obs_std + self.obs_mean
        return mean.astype(np.float32), var.astype(np.float32)

    def predict_uncertainty_components(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
        _, var = self.predict(obs, action)
        zeros = np.zeros_like(var, dtype=np.float32)
        return {
            "epi": var.astype(np.float32),
            "alea": zeros,
            "total": var.astype(np.float32),
        }

    def nearest_support(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist, idx = self.tree.query(self.normalize_obs(obs), k=1)
        if self.progress_mode == "support_time":
            progress = self.support_progress[idx].astype(np.float32)
        else:
            progress = compute_progress_score(
                obs=obs,
                task_name=self.task_name,
                goal_pos=self.goal_pos,
                goal_height=self.goal_height,
            )
        return dist.astype(np.float32), progress.astype(np.float32)


class MLPDynamics(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ProbabilisticMLPDynamics(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.mean_head = torch.nn.Linear(hidden_dim, output_dim)
        self.logvar_head = torch.nn.Linear(hidden_dim, output_dim)
        self.max_logvar = torch.nn.Parameter(torch.full((1, output_dim), 0.5))
        self.min_logvar = torch.nn.Parameter(torch.full((1, output_dim), -10.0))

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        raw_logvar = self.logvar_head(h)
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar


class GRUDynamics(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.cell = torch.nn.GRUCell(input_dim, hidden_dim)
        self.head = torch.nn.Linear(hidden_dim, output_dim)

    def forward_step(self, obs_action: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.cell(obs_action, hidden)
        return self.head(hidden), hidden


@dataclass
class MLPDynamicsEnsemble:
    models: List[MLPDynamics]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    tree: cKDTree
    support_progress: np.ndarray
    device: torch.device
    task_name: str = ""
    goal_pos: Optional[np.ndarray] = None
    goal_height: Optional[float] = None
    progress_mode: str = "semantic"
    episode_split: str = "train"
    n_support_episodes: int = 0
    n_support_steps: int = 0
    n_dynamics_transitions: int = 0

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def predict(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_np = np.concatenate([
            self.normalize_obs(obs),
            self.normalize_action(action),
        ], axis=-1).astype(np.float32)
        x = torch.from_numpy(x_np).to(self.device)
        preds = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                preds.append(model(x).detach().cpu().numpy())
        pred_arr = np.stack(preds, axis=0)
        mean_norm = pred_arr.mean(axis=0)
        var = pred_arr.var(axis=0).mean(axis=-1)
        mean = mean_norm * self.obs_std + self.obs_mean
        return mean.astype(np.float32), var.astype(np.float32)

    def predict_uncertainty_components(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
        _, var = self.predict(obs, action)
        zeros = np.zeros_like(var, dtype=np.float32)
        return {
            "epi": var.astype(np.float32),
            "alea": zeros,
            "total": var.astype(np.float32),
        }

    def nearest_support(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist, idx = self.tree.query(self.normalize_obs(obs), k=1)
        if self.progress_mode == "support_time":
            progress = self.support_progress[idx].astype(np.float32)
        else:
            progress = compute_progress_score(
                obs=obs,
                task_name=self.task_name,
                goal_pos=self.goal_pos,
                goal_height=self.goal_height,
            )
        return dist.astype(np.float32), progress.astype(np.float32)


@dataclass
class ProbabilisticMLPDynamicsEnsemble:
    models: List[ProbabilisticMLPDynamics]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    tree: cKDTree
    support_progress: np.ndarray
    device: torch.device
    task_name: str = ""
    goal_pos: Optional[np.ndarray] = None
    goal_height: Optional[float] = None
    progress_mode: str = "semantic"
    episode_split: str = "train"
    n_support_episodes: int = 0
    n_support_steps: int = 0
    n_dynamics_transitions: int = 0

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def _predict_norm_components(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        x_np = np.concatenate([
            self.normalize_obs(obs),
            self.normalize_action(action),
        ], axis=-1).astype(np.float32)
        x = torch.from_numpy(x_np).to(self.device)
        means = []
        variances = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                mean, logvar = model(x)
                means.append(mean.detach().cpu().numpy())
                variances.append(torch.exp(logvar).clamp_min(1e-6).detach().cpu().numpy())
        mean_arr = np.stack(means, axis=0)
        var_arr = np.stack(variances, axis=0)
        mean_norm = mean_arr.mean(axis=0)
        epi = mean_arr.var(axis=0).mean(axis=-1)
        alea = var_arr.mean(axis=0).mean(axis=-1)
        total = epi + alea
        return mean_norm, {
            "epi": epi.astype(np.float32),
            "alea": alea.astype(np.float32),
            "total": total.astype(np.float32),
        }

    def predict(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_norm, comps = self._predict_norm_components(obs, action)
        mean = mean_norm * self.obs_std + self.obs_mean
        return mean.astype(np.float32), comps["total"].astype(np.float32)

    def predict_uncertainty_components(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
        _, comps = self._predict_norm_components(obs, action)
        return comps

    def nearest_support(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist, idx = self.tree.query(self.normalize_obs(obs), k=1)
        if self.progress_mode == "support_time":
            progress = self.support_progress[idx].astype(np.float32)
        else:
            progress = compute_progress_score(
                obs=obs,
                task_name=self.task_name,
                goal_pos=self.goal_pos,
                goal_height=self.goal_height,
            )
        return dist.astype(np.float32), progress.astype(np.float32)


@dataclass
class GRUDynamicsEnsemble:
    models: List[GRUDynamics]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    tree: cKDTree
    support_progress: np.ndarray
    device: torch.device
    hidden_dim: int
    task_name: str = ""
    goal_pos: Optional[np.ndarray] = None
    goal_height: Optional[float] = None
    progress_mode: str = "semantic"
    episode_split: str = "train"
    n_support_episodes: int = 0
    n_support_steps: int = 0
    n_dynamics_transitions: int = 0
    _hidden: Optional[List[torch.Tensor]] = None

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def reset_state(self, batch_size: Optional[int] = None) -> None:
        if batch_size is None:
            self._hidden = None
            return
        self._hidden = [
            torch.zeros((batch_size, self.hidden_dim), device=self.device)
            for _ in self.models
        ]

    def prime_context(self, obs_hist: torch.Tensor) -> None:
        batch = obs_hist.shape[0]
        if obs_hist.shape[1] <= 1:
            self.reset_state(batch)
            return
        self.reset_state(batch)
        zero_action = np.zeros((batch, self.action_mean.shape[-1]), dtype=np.float32)
        for t in range(obs_hist.shape[1] - 1):
            obs = obs_hist[:, t].detach().cpu().numpy().astype(np.float32)
            self.predict(obs, zero_action)

    def _ensure_hidden(self, batch_size: int) -> None:
        if self._hidden is None or len(self._hidden) != len(self.models) or self._hidden[0].shape[0] != batch_size:
            self.reset_state(batch_size)

    def _predict_norm_components(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        batch_size = obs.shape[0]
        self._ensure_hidden(batch_size)
        x_np = np.concatenate([
            self.normalize_obs(obs),
            self.normalize_action(action),
        ], axis=-1).astype(np.float32)
        x = torch.from_numpy(x_np).to(self.device)
        preds = []
        next_hidden = []
        with torch.no_grad():
            for model, hidden in zip(self.models, self._hidden):
                model.eval()
                pred, hidden_next = model.forward_step(x, hidden)
                preds.append(pred.detach().cpu().numpy())
                next_hidden.append(hidden_next.detach())
        self._hidden = next_hidden
        pred_arr = np.stack(preds, axis=0)
        mean_norm = pred_arr.mean(axis=0)
        epi = pred_arr.var(axis=0).mean(axis=-1).astype(np.float32)
        zeros = np.zeros_like(epi, dtype=np.float32)
        return mean_norm, {"epi": epi, "alea": zeros, "total": epi}

    def predict(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_norm, comps = self._predict_norm_components(obs, action)
        mean = mean_norm * self.obs_std + self.obs_mean
        return mean.astype(np.float32), comps["epi"].astype(np.float32)

    def predict_uncertainty_components(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
        # This method advances the recurrent state. Call predict() and reuse its default
        # uncertainty when exact state synchronization matters.
        _, comps = self._predict_norm_components(obs, action)
        return comps

    def nearest_support(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist, idx = self.tree.query(self.normalize_obs(obs), k=1)
        if self.progress_mode == "support_time":
            progress = self.support_progress[idx].astype(np.float32)
        else:
            progress = compute_progress_score(
                obs=obs,
                task_name=self.task_name,
                goal_pos=self.goal_pos,
                goal_height=self.goal_height,
            )
        return dist.astype(np.float32), progress.astype(np.float32)


def read_json_lines(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def summarize_logs(log_path: Path) -> Tuple[Dict, Dict[int, Dict]]:
    rolling = {}
    by_epoch = {}
    for record in read_json_lines(log_path):
        rolling.update(record)
        if "epoch" in record:
            try:
                by_epoch[int(record["epoch"])] = dict(rolling)
            except Exception:
                pass
    return rolling, by_epoch


def infer_epoch_from_checkpoint(path: Path, fallback: Optional[int]) -> Optional[int]:
    match = re.search(r"epoch=(\d+)", path.name)
    if match:
        return int(match.group(1))
    return fallback


def discover_checkpoints(roots: Iterable[Path], skip_latest: bool = False) -> List[Path]:
    paths = []
    for root in roots:
        if root.is_file() and root.suffix == ".ckpt":
            paths.append(root)
        elif root.exists():
            paths.extend(root.glob("**/checkpoints/*.ckpt"))
    if skip_latest:
        paths = [path for path in paths if path.name != "latest.ckpt"]
    return sorted(set(paths))


def stack_samples(dataset, n_samples: int) -> Dict[str, torch.Tensor]:
    n = min(n_samples, len(dataset))
    samples = [dataset[i] for i in range(n)]
    return {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in samples[0].keys()
    }


def get_policy_action_start(cfg, policy_name: str) -> int:
    if "RobomimicLowdimPolicy" in policy_name:
        return 0
    n_obs_steps = int(cfg.n_obs_steps)
    oa_step_convention = bool(OmegaConf.select(cfg, "policy.oa_step_convention", default=False))
    return n_obs_steps - 1 if oa_step_convention else n_obs_steps


def load_policy_and_dataset(ckpt_path: Path, device: torch.device, use_ema: bool):
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    state_key = "ema_model" if use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key])
    policy.to(device)
    policy.eval()
    return cfg, policy, dataset, state_key


def evaluate_action_metrics(policy, cfg, dataset, device: torch.device, max_samples: int) -> Dict[str, float]:
    val_dataset = dataset.get_validation_dataset()
    if len(val_dataset) == 0:
        val_dataset = dataset
    batch = stack_samples(val_dataset, max_samples)
    batch = dict_apply(batch, lambda x: x.to(device))
    if hasattr(policy, "reset"):
        policy.reset()
    obs_input = batch["obs"][:, : int(cfg.n_obs_steps)]
    with torch.no_grad():
        result = policy.predict_action({"obs": obs_input})
    pred = result["action"]
    start = get_policy_action_start(cfg, cfg.policy._target_)
    end = min(start + pred.shape[1], batch["action"].shape[1])
    pred = pred[:, : end - start]
    target = batch["action"][:, start:end]
    err = pred - target
    return {
        "val_action_mse": float(torch.mean(err ** 2).item()),
        "val_action_l1": float(torch.mean(torch.abs(err)).item()),
        "val_action_l2": float(torch.linalg.norm(err, dim=-1).mean().item()),
        "val_eval_samples": int(target.shape[0]),
    }


def evaluate_teacher_forced_latent(policy, cfg, dataset, dynamics, device: torch.device,
        max_samples: int) -> Dict[str, float]:
    val_dataset = dataset.get_validation_dataset()
    if len(val_dataset) == 0:
        val_dataset = dataset
    batch = stack_samples(val_dataset, max_samples)
    obs_seq = batch["obs"].to(device)
    n_obs_steps = int(cfg.n_obs_steps)
    horizon = obs_seq.shape[1]
    if horizon <= n_obs_steps:
        return {
            "teacher_forced_latent_mse": "",
            "teacher_forced_latent_l2": "",
            "teacher_forced_latent_uncertainty": "",
            "teacher_forced_latent_steps": 0,
        }

    sq_errors = []
    l2_errors = []
    uncertainties = []
    if hasattr(policy, "reset"):
        policy.reset()
    with torch.no_grad():
        for t in range(n_obs_steps - 1, horizon - 1):
            obs_hist = obs_seq[:, t - n_obs_steps + 1:t + 1]
            result = policy.predict_action({"obs": obs_hist})
            action = result["action"][:, 0].detach().cpu().numpy().astype(np.float32)
            current_obs = obs_seq[:, t].detach().cpu().numpy().astype(np.float32)
            target_next = obs_seq[:, t + 1].detach().cpu().numpy().astype(np.float32)
            next_obs, var = dynamics.predict(current_obs, action)
            err = next_obs - target_next
            sq_errors.append(np.mean(err ** 2, axis=-1))
            l2_errors.append(np.linalg.norm(err, axis=-1))
            uncertainties.append(var)
    sq_arr = np.concatenate(sq_errors, axis=0)
    l2_arr = np.concatenate(l2_errors, axis=0)
    unc_arr = np.concatenate(uncertainties, axis=0)
    return {
        "teacher_forced_latent_mse": float(sq_arr.mean()),
        "teacher_forced_latent_l2": float(l2_arr.mean()),
        "teacher_forced_latent_uncertainty": float(unc_arr.mean()),
        "teacher_forced_latent_steps": int(sq_arr.shape[0]),
    }


def episode_slices(episode_ends: np.ndarray) -> Iterable[Tuple[int, int]]:
    start = 0
    for end in episode_ends:
        yield start, int(end)
        start = int(end)


def get_object_progress_slices(task_name: str, obs_dim: int) -> Tuple[Optional[slice], Optional[slice]]:
    task_name = task_name.lower()
    if task_name == "lift":
        return slice(7, 10), slice(0, 3)
    if task_name in {"can", "square"}:
        return slice(0, 3), slice(7, 10)
    if obs_dim >= 14:
        return slice(0, 3), slice(7, 10)
    return None, None


def compute_progress_reference(obs: np.ndarray, episode_ends: np.ndarray,
        task_name: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
    _, obj_slice = get_object_progress_slices(task_name, obs.shape[-1])
    if obj_slice is None:
        return None, None
    final_obs = []
    for start, end in episode_slices(episode_ends):
        if end > start:
            final_obs.append(obs[end - 1])
    if not final_obs:
        return None, None
    final_obs = np.stack(final_obs, axis=0)
    final_obj_pos = final_obs[:, obj_slice]
    goal_pos = np.median(final_obj_pos, axis=0, keepdims=True).astype(np.float32)
    goal_height = float(np.median(final_obj_pos[:, 2]))
    return goal_pos, goal_height


def compute_progress_score(obs: np.ndarray, task_name: str,
        goal_pos: Optional[np.ndarray], goal_height: Optional[float]) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    rel_slice, obj_slice = get_object_progress_slices(task_name, obs.shape[-1])
    if obj_slice is None:
        return np.zeros(obs.shape[0], dtype=np.float32)

    obj_pos = obs[:, obj_slice]
    task_name = task_name.lower()
    if task_name == "lift":
        reach = np.linalg.norm(obs[:, rel_slice], axis=-1) if rel_slice is not None else 0.0
        if goal_height is None:
            return obj_pos[:, 2].astype(np.float32)
        height = np.minimum(obj_pos[:, 2], goal_height)
        return (height - 0.10 * reach).astype(np.float32)

    if goal_pos is None:
        goal_pos = np.zeros((1, 3), dtype=np.float32)
    goal_dist = np.linalg.norm(obj_pos - goal_pos, axis=-1)
    reach = np.linalg.norm(obs[:, rel_slice], axis=-1) if rel_slice is not None else 0.0
    return (-goal_dist - 0.25 * reach).astype(np.float32)


def resolve_episode_mask(dataset, episode_split: str) -> np.ndarray:
    episode_ends = dataset.replay_buffer.episode_ends[:]
    if episode_split == "all":
        return np.ones(episode_ends.shape, dtype=bool)
    if episode_split == "train":
        mask = getattr(dataset, "train_mask", None)
    elif episode_split in {"val", "valid", "validation"}:
        mask = getattr(dataset, "val_mask", None)
        if mask is None:
            train_mask = getattr(dataset, "train_mask", None)
            mask = None if train_mask is None else ~train_mask
    else:
        raise ValueError(f"Unsupported episode_split={episode_split!r}")
    if mask is None:
        return np.ones(episode_ends.shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != episode_ends.shape:
        raise ValueError(
            f"episode mask shape {mask.shape} does not match episode ends {episode_ends.shape}"
        )
    if not np.any(mask):
        raise ValueError(f"episode_split={episode_split!r} selected zero episodes")
    return mask


def collect_transition_arrays(dataset, max_transitions: int, seed: int,
        task_name: str, progress_mode: str, episode_split: str):
    rb = dataset.replay_buffer
    all_obs = rb["obs"].astype(np.float32)
    all_action = rb["action"].astype(np.float32)
    episode_ends = rb.episode_ends[:]
    episode_mask = resolve_episode_mask(dataset, episode_split)
    x_obs = []
    x_action = []
    y_obs = []
    support_progress_parts = []
    obs_parts = []
    action_parts = []
    selected_episode_ends = []
    selected_steps = 0
    selected_episodes = 0
    for episode_idx, (start, end) in enumerate(episode_slices(episode_ends)):
        if not episode_mask[episode_idx]:
            continue
        ep_obs = all_obs[start:end]
        ep_action = all_action[start:end]
        if ep_obs.shape[0] == 0:
            continue
        obs_parts.append(ep_obs)
        action_parts.append(ep_action)
        selected_steps += ep_obs.shape[0]
        selected_episode_ends.append(selected_steps)
        selected_episodes += 1
        if end - start < 2:
            support_progress_parts.append(
                np.linspace(0.0, 1.0, end - start, dtype=np.float32)
            )
            continue
        progress = np.linspace(0.0, 1.0, ep_obs.shape[0], dtype=np.float32)
        support_progress_parts.append(progress)
        x_obs.append(ep_obs[:-1])
        x_action.append(ep_action[:-1])
        y_obs.append(ep_obs[1:])

    if not obs_parts or not x_obs:
        raise ValueError(
            f"Not enough transitions for dynamics with episode_split={episode_split!r}"
        )

    obs = np.concatenate(obs_parts, axis=0)
    action = np.concatenate(action_parts, axis=0)
    selected_episode_ends = np.asarray(selected_episode_ends, dtype=episode_ends.dtype)
    support_progress = np.concatenate(support_progress_parts, axis=0)
    goal_pos, goal_height = compute_progress_reference(
        obs=obs,
        episode_ends=selected_episode_ends,
        task_name=task_name,
    )
    if progress_mode != "support_time":
        support_progress = compute_progress_score(
            obs=obs,
            task_name=task_name,
            goal_pos=goal_pos,
            goal_height=goal_height,
        )

    x_obs = np.concatenate(x_obs, axis=0)
    x_action = np.concatenate(x_action, axis=0)
    y_obs = np.concatenate(y_obs, axis=0)
    rng = np.random.default_rng(seed)
    if x_obs.shape[0] > max_transitions:
        idx = rng.choice(x_obs.shape[0], size=max_transitions, replace=False)
        x_obs = x_obs[idx]
        x_action = x_action[idx]
        y_obs = y_obs[idx]
    return (
        obs,
        action,
        x_obs,
        x_action,
        y_obs,
        support_progress,
        goal_pos,
        goal_height,
        selected_episodes,
        selected_steps,
    )


def build_dynamics_ensemble(dataset, n_models: int, max_transitions: int,
        ridge: float, seed: int, task_name: str, progress_mode: str,
        episode_split: str) -> DynamicsEnsemble:
    (
        obs,
        action,
        x_obs,
        x_action,
        y_obs,
        support_progress,
        goal_pos,
        goal_height,
        selected_episodes,
        selected_steps,
    ) = collect_transition_arrays(
        dataset=dataset,
        max_transitions=max_transitions,
        seed=seed,
        task_name=task_name,
        progress_mode=progress_mode,
        episode_split=episode_split,
    )
    obs_mean = obs.mean(axis=0, keepdims=True)
    obs_std = obs.std(axis=0, keepdims=True) + 1e-6
    action_mean = action.mean(axis=0, keepdims=True)
    action_std = action.std(axis=0, keepdims=True) + 1e-6
    y_norm = (y_obs - obs_mean) / obs_std
    base_x = np.concatenate([
        (x_obs - obs_mean) / obs_std,
        (x_action - action_mean) / action_std,
        np.ones((x_obs.shape[0], 1), dtype=np.float32),
    ], axis=-1)

    weights = []
    rng = np.random.default_rng(seed)
    eye = np.eye(base_x.shape[1], dtype=np.float32)
    eye[-1, -1] = 0.0
    for _ in range(n_models):
        idx = rng.choice(base_x.shape[0], size=base_x.shape[0], replace=True)
        bx = base_x[idx]
        by = y_norm[idx]
        lhs = bx.T @ bx + ridge * eye
        rhs = bx.T @ by
        weights.append(np.linalg.solve(lhs, rhs).astype(np.float32))

    tree = cKDTree(((obs - obs_mean) / obs_std).astype(np.float32))
    return DynamicsEnsemble(
        weights=weights,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
        tree=tree,
        support_progress=support_progress,
        task_name=task_name,
        goal_pos=goal_pos,
        goal_height=goal_height,
        progress_mode=progress_mode,
        episode_split=episode_split,
        n_support_episodes=int(selected_episodes),
        n_support_steps=int(selected_steps),
        n_dynamics_transitions=int(x_obs.shape[0]),
    )


def build_mlp_dynamics_ensemble(dataset, n_models: int, max_transitions: int, seed: int,
        device: torch.device, hidden_dim: int, epochs: int, batch_size: int, lr: float,
        task_name: str, progress_mode: str, episode_split: str) -> MLPDynamicsEnsemble:
    (
        obs,
        action,
        x_obs,
        x_action,
        y_obs,
        support_progress,
        goal_pos,
        goal_height,
        selected_episodes,
        selected_steps,
    ) = collect_transition_arrays(
        dataset=dataset,
        max_transitions=max_transitions,
        seed=seed,
        task_name=task_name,
        progress_mode=progress_mode,
        episode_split=episode_split,
    )
    obs_mean = obs.mean(axis=0, keepdims=True)
    obs_std = obs.std(axis=0, keepdims=True) + 1e-6
    action_mean = action.mean(axis=0, keepdims=True)
    action_std = action.std(axis=0, keepdims=True) + 1e-6
    x_np = np.concatenate([
        (x_obs - obs_mean) / obs_std,
        (x_action - action_mean) / action_std,
    ], axis=-1).astype(np.float32)
    y_np = ((y_obs - obs_mean) / obs_std).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    n = x.shape[0]
    input_dim = x.shape[1]
    output_dim = y.shape[1]
    steps_per_epoch = max(1, math.ceil(n / batch_size))
    models = []
    for model_idx in range(n_models):
        torch.manual_seed(seed + model_idx)
        model = MLPDynamics(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 1009 * (model_idx + 1))
        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                idx = torch.randint(0, n, (min(batch_size, n),), generator=generator, device=device)
                pred = model(x[idx])
                loss = torch.mean((pred - y[idx]) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        models.append(model)
    tree = cKDTree(((obs - obs_mean) / obs_std).astype(np.float32))
    return MLPDynamicsEnsemble(
        models=models,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
        tree=tree,
        support_progress=support_progress,
        device=device,
        task_name=task_name,
        goal_pos=goal_pos,
        goal_height=goal_height,
        progress_mode=progress_mode,
        episode_split=episode_split,
        n_support_episodes=int(selected_episodes),
        n_support_steps=int(selected_steps),
        n_dynamics_transitions=int(x_np.shape[0]),
    )


def build_pe_mlp_dynamics_ensemble(dataset, n_models: int, max_transitions: int, seed: int,
        device: torch.device, hidden_dim: int, epochs: int, batch_size: int, lr: float,
        task_name: str, progress_mode: str, episode_split: str) -> ProbabilisticMLPDynamicsEnsemble:
    (
        obs,
        action,
        x_obs,
        x_action,
        y_obs,
        support_progress,
        goal_pos,
        goal_height,
        selected_episodes,
        selected_steps,
    ) = collect_transition_arrays(
        dataset=dataset,
        max_transitions=max_transitions,
        seed=seed,
        task_name=task_name,
        progress_mode=progress_mode,
        episode_split=episode_split,
    )
    obs_mean = obs.mean(axis=0, keepdims=True)
    obs_std = obs.std(axis=0, keepdims=True) + 1e-6
    action_mean = action.mean(axis=0, keepdims=True)
    action_std = action.std(axis=0, keepdims=True) + 1e-6
    x_np = np.concatenate([
        (x_obs - obs_mean) / obs_std,
        (x_action - action_mean) / action_std,
    ], axis=-1).astype(np.float32)
    y_np = ((y_obs - obs_mean) / obs_std).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    n = x.shape[0]
    input_dim = x.shape[1]
    output_dim = y.shape[1]
    steps_per_epoch = max(1, math.ceil(n / batch_size))
    models = []
    for model_idx in range(n_models):
        torch.manual_seed(seed + model_idx)
        model = ProbabilisticMLPDynamics(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        model.train()
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 1009 * (model_idx + 1))
        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                idx = torch.randint(0, n, (min(batch_size, n),), generator=generator, device=device)
                mean, logvar = model(x[idx])
                inv_var = torch.exp(-logvar)
                nll = 0.5 * torch.mean((y[idx] - mean) ** 2 * inv_var + logvar)
                logvar_reg = 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
                loss = nll + logvar_reg
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        models.append(model)
    tree = cKDTree(((obs - obs_mean) / obs_std).astype(np.float32))
    return ProbabilisticMLPDynamicsEnsemble(
        models=models,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
        tree=tree,
        support_progress=support_progress,
        device=device,
        task_name=task_name,
        goal_pos=goal_pos,
        goal_height=goal_height,
        progress_mode=progress_mode,
        episode_split=episode_split,
        n_support_episodes=int(selected_episodes),
        n_support_steps=int(selected_steps),
        n_dynamics_transitions=int(x_np.shape[0]),
    )


def collect_sequence_arrays(dataset, max_sequences: int, seed: int, task_name: str,
        progress_mode: str, episode_split: str, sequence_len: int):
    rb = dataset.replay_buffer
    all_obs = rb["obs"].astype(np.float32)
    all_action = rb["action"].astype(np.float32)
    episode_ends = rb.episode_ends[:]
    episode_mask = resolve_episode_mask(dataset, episode_split)
    obs_parts = []
    action_parts = []
    support_progress_parts = []
    selected_episode_ends = []
    selected_steps = 0
    selected_episodes = 0
    seq_obs = []
    seq_action = []
    for episode_idx, (start, end) in enumerate(episode_slices(episode_ends)):
        if not episode_mask[episode_idx]:
            continue
        ep_obs = all_obs[start:end]
        ep_action = all_action[start:end]
        if ep_obs.shape[0] == 0:
            continue
        obs_parts.append(ep_obs)
        action_parts.append(ep_action)
        selected_steps += ep_obs.shape[0]
        selected_episode_ends.append(selected_steps)
        selected_episodes += 1
        support_progress_parts.append(np.linspace(0.0, 1.0, ep_obs.shape[0], dtype=np.float32))
        if ep_obs.shape[0] <= sequence_len:
            continue
        for local_start in range(0, ep_obs.shape[0] - sequence_len):
            seq_obs.append(ep_obs[local_start:local_start + sequence_len + 1])
            seq_action.append(ep_action[local_start:local_start + sequence_len])

    if not obs_parts or not seq_obs:
        raise ValueError(
            f"Not enough sequences for GRU dynamics with episode_split={episode_split!r}"
        )
    obs = np.concatenate(obs_parts, axis=0)
    action = np.concatenate(action_parts, axis=0)
    selected_episode_ends = np.asarray(selected_episode_ends, dtype=episode_ends.dtype)
    support_progress = np.concatenate(support_progress_parts, axis=0)
    goal_pos, goal_height = compute_progress_reference(
        obs=obs,
        episode_ends=selected_episode_ends,
        task_name=task_name,
    )
    if progress_mode != "support_time":
        support_progress = compute_progress_score(
            obs=obs,
            task_name=task_name,
            goal_pos=goal_pos,
            goal_height=goal_height,
        )

    seq_obs = np.stack(seq_obs, axis=0)
    seq_action = np.stack(seq_action, axis=0)
    rng = np.random.default_rng(seed)
    if seq_obs.shape[0] > max_sequences:
        idx = rng.choice(seq_obs.shape[0], size=max_sequences, replace=False)
        seq_obs = seq_obs[idx]
        seq_action = seq_action[idx]
    return (
        obs,
        action,
        seq_obs,
        seq_action,
        support_progress,
        goal_pos,
        goal_height,
        selected_episodes,
        selected_steps,
    )


def build_gru_dynamics_ensemble(dataset, n_models: int, max_sequences: int, seed: int,
        device: torch.device, hidden_dim: int, epochs: int, batch_size: int, lr: float,
        task_name: str, progress_mode: str, episode_split: str, context_len: int,
        sequence_len: int = 10, scheduled_sampling_prob: float = 0.5) -> GRUDynamicsEnsemble:
    (
        obs,
        action,
        seq_obs,
        seq_action,
        support_progress,
        goal_pos,
        goal_height,
        selected_episodes,
        selected_steps,
    ) = collect_sequence_arrays(
        dataset=dataset,
        max_sequences=max_sequences,
        seed=seed,
        task_name=task_name,
        progress_mode=progress_mode,
        episode_split=episode_split,
        sequence_len=sequence_len,
    )
    obs_mean = obs.mean(axis=0, keepdims=True)
    obs_std = obs.std(axis=0, keepdims=True) + 1e-6
    action_mean = action.mean(axis=0, keepdims=True)
    action_std = action.std(axis=0, keepdims=True) + 1e-6
    obs_np = ((seq_obs - obs_mean) / obs_std).astype(np.float32)
    action_np = ((seq_action - action_mean) / action_std).astype(np.float32)
    obs_t = torch.from_numpy(obs_np).to(device)
    action_t = torch.from_numpy(action_np).to(device)
    n = obs_t.shape[0]
    steps_per_epoch = max(1, math.ceil(n / batch_size))
    input_dim = obs_t.shape[-1] + action_t.shape[-1]
    output_dim = obs_t.shape[-1]
    models = []
    for model_idx in range(n_models):
        torch.manual_seed(seed + model_idx)
        model = GRUDynamics(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + 1009 * (model_idx + 1))
        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                idx = torch.randint(0, n, (min(batch_size, n),), generator=generator, device=device)
                obs_batch = obs_t[idx]
                action_batch = action_t[idx]
                hidden = torch.zeros((obs_batch.shape[0], hidden_dim), device=device)
                current_obs = obs_batch[:, 0]
                losses = []
                for t in range(sequence_len):
                    if t < int(context_len):
                        current_obs = obs_batch[:, t]
                    elif scheduled_sampling_prob > 0:
                        sample_mask = (
                            torch.rand((obs_batch.shape[0], 1), generator=generator, device=device)
                            < scheduled_sampling_prob
                        )
                        current_obs = torch.where(sample_mask, current_obs.detach(), obs_batch[:, t])
                    x = torch.cat([current_obs, action_batch[:, t]], dim=-1)
                    pred, hidden = model.forward_step(x, hidden)
                    target = obs_batch[:, t + 1]
                    losses.append(torch.mean((pred - target) ** 2))
                    current_obs = pred
                loss = torch.stack(losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        models.append(model)
    tree = cKDTree(((obs - obs_mean) / obs_std).astype(np.float32))
    return GRUDynamicsEnsemble(
        models=models,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
        tree=tree,
        support_progress=support_progress,
        device=device,
        hidden_dim=hidden_dim,
        task_name=task_name,
        goal_pos=goal_pos,
        goal_height=goal_height,
        progress_mode=progress_mode,
        episode_split=episode_split,
        n_support_episodes=int(selected_episodes),
        n_support_steps=int(selected_steps),
        n_dynamics_transitions=int(seq_obs.shape[0]),
    )


def predict_with_uncertainty_components(dynamics, obs: np.ndarray, action: np.ndarray):
    next_obs, default_unc = dynamics.predict(obs, action)
    if isinstance(dynamics, GRUDynamicsEnsemble):
        zeros = np.zeros_like(default_unc, dtype=np.float32)
        comps = {
            "epi": default_unc.astype(np.float32),
            "alea": zeros,
            "total": default_unc.astype(np.float32),
        }
    elif hasattr(dynamics, "predict_uncertainty_components"):
        comps = dynamics.predict_uncertainty_components(obs, action)
    else:
        zeros = np.zeros_like(default_unc, dtype=np.float32)
        comps = {
            "epi": default_unc.astype(np.float32),
            "alea": zeros,
            "total": default_unc.astype(np.float32),
        }
    return next_obs, default_unc, comps


def rollout_plv_lite(policy, cfg, dataset, dynamics: DynamicsEnsemble, device: torch.device,
        n_starts: int, horizon: int) -> Dict[str, float]:
    val_dataset = dataset.get_validation_dataset()
    if len(val_dataset) == 0:
        val_dataset = dataset
    batch = stack_samples(val_dataset, n_starts)
    obs_hist = batch["obs"][:, : int(cfg.n_obs_steps)].to(device)
    actions = []
    support_dists = []
    progresses = []
    uncertainties = []
    uncertainties_epi = []
    uncertainties_alea = []
    uncertainties_total = []

    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(dynamics, "prime_context"):
        dynamics.prime_context(obs_hist)
    elif hasattr(dynamics, "reset_state"):
        dynamics.reset_state(obs_hist.shape[0])

    with torch.no_grad():
        for _ in range(horizon):
            result = policy.predict_action({"obs": obs_hist})
            action = result["action"][:, 0].detach().cpu().numpy().astype(np.float32)
            current_obs = obs_hist[:, -1].detach().cpu().numpy().astype(np.float32)
            next_obs, var, comps = predict_with_uncertainty_components(dynamics, current_obs, action)
            dist, progress = dynamics.nearest_support(next_obs)
            actions.append(action)
            support_dists.append(dist)
            progresses.append(progress)
            uncertainties.append(var)
            uncertainties_epi.append(comps["epi"])
            uncertainties_alea.append(comps["alea"])
            uncertainties_total.append(comps["total"])
            next_obs_t = torch.from_numpy(next_obs).to(device=device, dtype=obs_hist.dtype)[:, None]
            obs_hist = torch.cat([obs_hist[:, 1:], next_obs_t], dim=1)

    action_arr = np.stack(actions, axis=1)
    support_arr = np.stack(support_dists, axis=1)
    progress_arr = np.stack(progresses, axis=1)
    uncertainty_arr = np.stack(uncertainties, axis=1)
    uncertainty_epi_arr = np.stack(uncertainties_epi, axis=1)
    uncertainty_alea_arr = np.stack(uncertainties_alea, axis=1)
    uncertainty_total_arr = np.stack(uncertainties_total, axis=1)
    init_dist, init_progress = dynamics.nearest_support(batch["obs"][:, int(cfg.n_obs_steps) - 1].numpy())
    if action_arr.shape[1] >= 3:
        jerk = action_arr[:, 2:] - 2 * action_arr[:, 1:-1] + action_arr[:, :-2]
        smoothness = np.mean(jerk ** 2)
    else:
        smoothness = 0.0

    progress_delta = progress_arr.mean(axis=1) - init_progress
    support_drift = support_arr.mean()
    uncertainty = uncertainty_arr.mean()
    support_drift_final = support_arr[:, -1].mean()
    support_drift_max = support_arr.max(axis=1).mean()
    progress_delta_final = progress_arr[:, -1] - init_progress
    progress_delta_min = progress_arr.min(axis=1) - init_progress
    uncertainty_final = uncertainty_arr[:, -1].mean()
    uncertainty_max = uncertainty_arr.max(axis=1).mean()
    uncertainty_epi = uncertainty_epi_arr.mean()
    uncertainty_epi_final = uncertainty_epi_arr[:, -1].mean()
    uncertainty_epi_max = uncertainty_epi_arr.max(axis=1).mean()
    uncertainty_alea = uncertainty_alea_arr.mean()
    uncertainty_alea_final = uncertainty_alea_arr[:, -1].mean()
    uncertainty_alea_max = uncertainty_alea_arr.max(axis=1).mean()
    uncertainty_total = uncertainty_total_arr.mean()
    uncertainty_total_final = uncertainty_total_arr[:, -1].mean()
    uncertainty_total_max = uncertainty_total_arr.max(axis=1).mean()
    # Lower is better. This raw combination is normalized later across checkpoints.
    raw_score = support_drift + uncertainty + smoothness - progress_delta.mean()
    final_score = support_drift_final + uncertainty_final + smoothness - progress_delta_final.mean()
    risk_score = support_drift_max + uncertainty_max + smoothness - progress_delta_final.mean()
    risk_no_progress_score = support_drift_max + uncertainty_max + smoothness
    return {
        "plv_lite_score_raw": float(raw_score),
        "plv_final_score_raw": float(final_score),
        "plv_risk_score_raw": float(risk_score),
        "plv_risk_no_progress_score_raw": float(risk_no_progress_score),
        "support_drift": float(support_drift),
        "support_drift_initial": float(init_dist.mean()),
        "support_drift_final": float(support_drift_final),
        "support_drift_max": float(support_drift_max),
        "progress_delta": float(progress_delta.mean()),
        "progress_delta_final": float(progress_delta_final.mean()),
        "progress_delta_min": float(progress_delta_min.mean()),
        "dynamics_uncertainty": float(uncertainty),
        "dynamics_uncertainty_final": float(uncertainty_final),
        "dynamics_uncertainty_max": float(uncertainty_max),
        "dynamics_uncertainty_epi": float(uncertainty_epi),
        "dynamics_uncertainty_epi_final": float(uncertainty_epi_final),
        "dynamics_uncertainty_epi_max": float(uncertainty_epi_max),
        "dynamics_uncertainty_alea": float(uncertainty_alea),
        "dynamics_uncertainty_alea_final": float(uncertainty_alea_final),
        "dynamics_uncertainty_alea_max": float(uncertainty_alea_max),
        "dynamics_uncertainty_total": float(uncertainty_total),
        "dynamics_uncertainty_total_final": float(uncertainty_total_final),
        "dynamics_uncertainty_total_max": float(uncertainty_total_max),
        "smoothness": float(smoothness),
        "plv_rollout_starts": int(action_arr.shape[0]),
        "plv_horizon": int(horizon),
    }


def checkpoint_metadata(ckpt_path: Path, cfg, logs_by_epoch: Dict[int, Dict]) -> Dict[str, object]:
    run_dir = ckpt_path.parent.parent
    fallback_epoch = max(logs_by_epoch.keys(), default=None)
    epoch = infer_epoch_from_checkpoint(ckpt_path, fallback_epoch)
    record = logs_by_epoch.get(epoch, {}) if epoch is not None else {}
    policy_target = str(cfg.policy._target_)
    if "RobomimicLowdimPolicy" in policy_target:
        policy = "bc_rnn"
    elif "DiffusionUnetLowdimPolicy" in policy_target:
        policy = "diffusion_policy"
    else:
        policy = policy_target.split(".")[-1]
    return {
        "checkpoint_path": str(ckpt_path),
        "run_dir": str(run_dir),
        "policy": policy,
        "task": str(cfg.task.task_name),
        "dataset_type": str(cfg.task.dataset_type),
        "seed": int(cfg.training.seed),
        "epoch": "" if epoch is None else int(epoch),
        "global_step": record.get("global_step", ""),
        "train_loss": record.get("train_loss", ""),
        "val_loss_logged": record.get("val_loss", ""),
        "test_mean_score": record.get("test/mean_score", ""),
        "train_mean_score": record.get("train/mean_score", ""),
    }


def robust_normalize_scores(rows: List[Dict[str, object]]) -> None:
    component_signs = {
        "support_drift": 1.0,
        "support_drift_final": 1.0,
        "support_drift_max": 1.0,
        "dynamics_uncertainty": 1.0,
        "dynamics_uncertainty_final": 1.0,
        "dynamics_uncertainty_max": 1.0,
        "dynamics_uncertainty_epi": 1.0,
        "dynamics_uncertainty_epi_final": 1.0,
        "dynamics_uncertainty_epi_max": 1.0,
        "dynamics_uncertainty_alea": 1.0,
        "dynamics_uncertainty_alea_final": 1.0,
        "dynamics_uncertainty_alea_max": 1.0,
        "dynamics_uncertainty_total": 1.0,
        "dynamics_uncertainty_total_final": 1.0,
        "dynamics_uncertainty_total_max": 1.0,
        "smoothness": 1.0,
        "progress_delta": -1.0,
        "progress_delta_final": -1.0,
        "progress_delta_min": -1.0,
    }
    mean_components = [
        "support_drift",
        "dynamics_uncertainty",
        "smoothness",
        "progress_delta",
    ]
    final_components = [
        "support_drift_final",
        "dynamics_uncertainty_final",
        "smoothness",
        "progress_delta_final",
    ]
    risk_components = [
        "support_drift_max",
        "dynamics_uncertainty_max",
        "smoothness",
        "progress_delta_final",
    ]
    risk_no_progress_components = [
        "support_drift_max",
        "dynamics_uncertainty_max",
        "smoothness",
    ]
    stability_components = [
        "dynamics_uncertainty_max",
        "smoothness",
    ]
    uncertainty_components = [
        "dynamics_uncertainty_max",
    ]
    uncertainty_epi_components = [
        "dynamics_uncertainty_epi_max",
    ]
    uncertainty_alea_components = [
        "dynamics_uncertainty_alea_max",
    ]
    uncertainty_total_components = [
        "dynamics_uncertainty_total_max",
    ]
    smoothness_components = [
        "smoothness",
    ]
    support_smooth_components = [
        "support_drift_max",
        "smoothness",
    ]
    values = {}
    for key in component_signs:
        vals = np.array([float(r[key]) for r in rows if r.get(key, "") != ""], dtype=np.float64)
        if vals.size == 0:
            continue
        med = np.median(vals)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        values[key] = (med, iqr if iqr > 1e-9 else np.std(vals) + 1e-9)

    def score_row(row, components: List[str], drop_key: Optional[str] = None):
        score = 0.0
        count = 0
        for key in components:
            if key == drop_key:
                continue
            if key in values and row.get(key, "") != "":
                med, scale = values[key]
                sign = component_signs[key]
                component = sign * (float(row[key]) - med) / scale
                score += float(np.clip(component, -5.0, 5.0))
                count += 1
        return "" if count == 0 else float(score / count)

    for row in rows:
        row["plv_lite_score_norm"] = score_row(row, mean_components)
        row["plv_no_support_norm"] = score_row(row, mean_components, drop_key="support_drift")
        row["plv_no_progress_norm"] = score_row(row, mean_components, drop_key="progress_delta")
        row["plv_no_uncertainty_norm"] = score_row(row, mean_components, drop_key="dynamics_uncertainty")
        row["plv_no_smoothness_norm"] = score_row(row, mean_components, drop_key="smoothness")
        row["plv_final_score_norm"] = score_row(row, final_components)
        row["plv_final_no_progress_norm"] = score_row(
            row,
            final_components,
            drop_key="progress_delta_final",
        )
        row["plv_risk_score_norm"] = score_row(row, risk_components)
        row["plv_risk_no_progress_norm"] = score_row(row, risk_no_progress_components)
        row["plv_stability_score_norm"] = score_row(row, stability_components)
        row["plv_uncertainty_score_norm"] = score_row(row, uncertainty_components)
        row["plv_unc_epi_score_norm"] = score_row(row, uncertainty_epi_components)
        row["plv_unc_alea_score_norm"] = score_row(row, uncertainty_alea_components)
        row["plv_unc_total_score_norm"] = score_row(row, uncertainty_total_components)
        row["plv_smoothness_score_norm"] = score_row(row, smoothness_components)
        row["plv_support_smooth_score_norm"] = score_row(row, support_smooth_components)


def robust_normalize_scores_grouped(rows: List[Dict[str, object]], group_keys: List[str]) -> None:
    if not group_keys:
        robust_normalize_scores(rows)
        return
    groups = {}
    for row in rows:
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        robust_normalize_scores(group_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["data/outputs/plv_pilot"])
    parser.add_argument("--output", default="data/robomimic/plv_assets/checkpoint_metrics.csv")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--rollout-starts", type=int, default=32)
    parser.add_argument("--plv-horizon", type=int, default=5)
    parser.add_argument("--dyn-type", choices=["linear", "mlp", "pe_mlp", "gru"], default="linear")
    parser.add_argument("--dyn-models", type=int, default=5)
    parser.add_argument("--dyn-max-transitions", type=int, default=20000)
    parser.add_argument("--dyn-episode-split", choices=["train", "all"], default="train")
    parser.add_argument("--dyn-seed", type=int, default=0)
    parser.add_argument("--dyn-ridge", type=float, default=1e-3)
    parser.add_argument("--dyn-hidden-dim", type=int, default=256)
    parser.add_argument("--dyn-epochs", type=int, default=8)
    parser.add_argument("--dyn-batch-size", type=int, default=1024)
    parser.add_argument("--dyn-lr", type=float, default=1e-3)
    parser.add_argument("--gru-sequence-len", type=int, default=10)
    parser.add_argument("--gru-scheduled-sampling-prob", type=float, default=0.5)
    parser.add_argument(
        "--progress-mode",
        choices=["semantic", "support_time"],
        default="semantic",
        help="Use task-aware object progress or nearest-support demo-time progress.",
    )
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--skip-latest", action="store_true")
    parser.add_argument(
        "--normalize-group-keys",
        nargs="*",
        default=["task", "policy", "dataset_type"],
        help="Normalize PLV component scores within these metadata groups.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpts = discover_checkpoints([Path(x) for x in args.roots], skip_latest=args.skip_latest)
    rows = []
    dynamics_cache = {}

    for ckpt_path in ckpts:
        cfg, policy, dataset, state_key = load_policy_and_dataset(ckpt_path, device=device, use_ema=args.use_ema)
        _, by_epoch = summarize_logs(ckpt_path.parent.parent / "logs.json.txt")
        row = checkpoint_metadata(ckpt_path, cfg, by_epoch)
        row["state_key"] = state_key

        dataset_key = "|".join([
            str(Path(cfg.task.dataset.dataset_path)),
            f"dyn_type={args.dyn_type}",
            f"dyn_split={args.dyn_episode_split}",
            f"dyn_seed={args.dyn_seed}",
            f"gru_context={int(cfg.n_obs_steps)}",
            f"gru_sequence={args.gru_sequence_len}",
            f"gru_sched={args.gru_scheduled_sampling_prob}",
            f"train_filter={OmegaConf.select(cfg, 'task.dataset.train_filter_key', default='')}",
            f"val_filter={OmegaConf.select(cfg, 'task.dataset.val_filter_key', default='')}",
            f"dataset_seed={OmegaConf.select(cfg, 'task.dataset.seed', default='')}",
            f"val_ratio={OmegaConf.select(cfg, 'task.dataset.val_ratio', default='')}",
            f"max_train={OmegaConf.select(cfg, 'task.dataset.max_train_episodes', default='')}",
        ])
        if dataset_key not in dynamics_cache:
            if args.dyn_type == "mlp":
                dynamics_cache[dataset_key] = build_mlp_dynamics_ensemble(
                    dataset,
                    n_models=args.dyn_models,
                    max_transitions=args.dyn_max_transitions,
                    seed=args.dyn_seed,
                    device=device,
                    hidden_dim=args.dyn_hidden_dim,
                    epochs=args.dyn_epochs,
                    batch_size=args.dyn_batch_size,
                    lr=args.dyn_lr,
                    task_name=str(cfg.task.task_name),
                    progress_mode=args.progress_mode,
                    episode_split=args.dyn_episode_split,
                )
            elif args.dyn_type == "pe_mlp":
                dynamics_cache[dataset_key] = build_pe_mlp_dynamics_ensemble(
                    dataset,
                    n_models=args.dyn_models,
                    max_transitions=args.dyn_max_transitions,
                    seed=args.dyn_seed,
                    device=device,
                    hidden_dim=args.dyn_hidden_dim,
                    epochs=args.dyn_epochs,
                    batch_size=args.dyn_batch_size,
                    lr=args.dyn_lr,
                    task_name=str(cfg.task.task_name),
                    progress_mode=args.progress_mode,
                    episode_split=args.dyn_episode_split,
                )
            elif args.dyn_type == "gru":
                dynamics_cache[dataset_key] = build_gru_dynamics_ensemble(
                    dataset,
                    n_models=args.dyn_models,
                    max_sequences=args.dyn_max_transitions,
                    seed=args.dyn_seed,
                    device=device,
                    hidden_dim=args.dyn_hidden_dim,
                    epochs=args.dyn_epochs,
                    batch_size=args.dyn_batch_size,
                    lr=args.dyn_lr,
                    task_name=str(cfg.task.task_name),
                    progress_mode=args.progress_mode,
                    episode_split=args.dyn_episode_split,
                    context_len=int(cfg.n_obs_steps),
                    sequence_len=args.gru_sequence_len,
                    scheduled_sampling_prob=args.gru_scheduled_sampling_prob,
                )
            else:
                dynamics_cache[dataset_key] = build_dynamics_ensemble(
                    dataset,
                    n_models=args.dyn_models,
                    max_transitions=args.dyn_max_transitions,
                    ridge=args.dyn_ridge,
                    seed=args.dyn_seed,
                    task_name=str(cfg.task.task_name),
                    progress_mode=args.progress_mode,
                    episode_split=args.dyn_episode_split,
                )
        dynamics = dynamics_cache[dataset_key]
        row["plv_progress_mode"] = args.progress_mode
        row["dyn_type"] = args.dyn_type
        row["dyn_seed"] = int(args.dyn_seed)
        row["dyn_episode_split"] = dynamics.episode_split
        row["dyn_support_episodes"] = dynamics.n_support_episodes
        row["dyn_support_steps"] = dynamics.n_support_steps
        row["dyn_train_transitions"] = dynamics.n_dynamics_transitions
        row.update(evaluate_action_metrics(
            policy=policy,
            cfg=cfg,
            dataset=dataset,
            device=device,
            max_samples=args.max_eval_samples,
        ))
        row.update(evaluate_teacher_forced_latent(
            policy=policy,
            cfg=cfg,
            dataset=dataset,
            dynamics=dynamics,
            device=device,
            max_samples=args.max_eval_samples,
        ))
        row.update(rollout_plv_lite(
            policy=policy,
            cfg=cfg,
            dataset=dataset,
            dynamics=dynamics,
            device=device,
            n_starts=args.rollout_starts,
            horizon=args.plv_horizon,
        ))
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    robust_normalize_scores_grouped(rows, args.normalize_group_keys)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(rows[0].keys())
        for row in rows:
            for key in row.keys():
                if key not in fields:
                    fields.append(key)
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
