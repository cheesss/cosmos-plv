"""Stage A: discover, validate and load Diffusion Policy checkpoints.

Loading reuses the original PLV helper
``diffusion_policy.scripts.plv_chunk_aware_checkpoint_metrics.load_policy_with_dataset_cache``
so that the normalizer and EMA handling are identical to the paper scripts.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .provenance import REPO_ROOT, repo_relative, sha256_file

CKPT_RE = re.compile(r"epoch=(?P<epoch>\d+)-test_mean_score=(?P<score>[0-9.]+)\.ckpt$")


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    run_name: str
    run_dir: str
    checkpoint_path: str
    task: str
    seed: int
    epoch: int
    train_rollout_success: float  # test_mean_score logged at save time (n_test rollouts)
    n_train_rollouts: int
    val_loss: Optional[float] = None
    train_loss: Optional[float] = None
    train_action_mse_error: Optional[float] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    config: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _read_logs(run_dir: Path) -> Dict[int, Dict[str, float]]:
    """Epoch -> last logged row for that epoch (from logs.json.txt)."""
    path = run_dir / "logs.json.txt"
    by_epoch: Dict[int, Dict[str, float]] = {}
    if not path.is_file():
        return by_epoch
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" not in row:
            continue
        ep = int(row["epoch"])
        cur = by_epoch.setdefault(ep, {})
        for k, v in row.items():
            if isinstance(v, (int, float)):
                cur[k] = v
    return by_epoch


def _hydra_cfg(run_dir: Path) -> Dict[str, object]:
    path = run_dir / ".hydra" / "config.yaml"
    if not path.is_file():
        return {}
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        select = lambda dotted: OmegaConf.select(cfg, dotted)  # noqa: E731
    except ImportError:  # Environment B has no omegaconf: plain YAML is enough here
        import yaml

        raw = yaml.safe_load(path.read_text())

        def select(dotted):
            cur = raw
            for part in dotted.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return None
                cur = cur[part]
            return cur
    keys = {
        "horizon": "horizon",
        "n_obs_steps": "n_obs_steps",
        "n_action_steps": "n_action_steps",
        "n_latency_steps": "n_latency_steps",
        "num_inference_steps": "policy.num_inference_steps",
        "task_name": "task.task_name",
        "dataset_type": "task.dataset_type",
        "abs_action": "task.abs_action",
        "seed": "training.seed",
        "num_epochs": "training.num_epochs",
        "n_test": "task.env_runner.n_test",
        "test_start_seed": "task.env_runner.test_start_seed",
        "policy_target": "policy._target_",
    }
    out: Dict[str, object] = {}
    for name, dotted in keys.items():
        try:
            out[name] = select(dotted)
        except Exception:
            out[name] = None
    return out


def discover_checkpoints(pool_root: Path, hash_files: bool = False) -> List[CheckpointRecord]:
    """Find every ``epoch=XXXX-test_mean_score=Y.ckpt`` under ``pool_root``."""
    pool_root = Path(pool_root)
    records: List[CheckpointRecord] = []
    for run_dir in sorted(p for p in pool_root.iterdir() if p.is_dir()):
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            continue
        logs = _read_logs(run_dir)
        cfg = _hydra_cfg(run_dir)
        task = str(cfg.get("task_name") or "unknown")
        seed = int(cfg.get("seed") or -1)
        for ckpt in sorted(ckpt_dir.glob("epoch=*.ckpt")):
            m = CKPT_RE.search(ckpt.name)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            row = logs.get(epoch, {})
            rec = CheckpointRecord(
                checkpoint_id=f"{task}_s{seed}_e{epoch:04d}",
                run_name=run_dir.name,
                run_dir=repo_relative(run_dir),
                checkpoint_path=repo_relative(ckpt),
                task=task,
                seed=seed,
                epoch=epoch,
                # The filename is rounded to 3 decimals; prefer the exact log value.
                train_rollout_success=float(row.get("test/mean_score", float(m.group("score")))),
                n_train_rollouts=int(cfg.get("n_test") or 0),
                val_loss=row.get("val_loss"),
                train_loss=row.get("train_loss"),
                train_action_mse_error=row.get("train_action_mse_error"),
                size_bytes=ckpt.stat().st_size,
                config=cfg,
            )
            if hash_files:
                rec.sha256 = sha256_file(ckpt)
            records.append(rec)
    return records


def validate_checkpoint_config(cfg) -> Dict[str, object]:
    """Read execution parameters from a loaded checkpoint config (never hardcode)."""
    horizon = int(cfg.horizon)
    n_obs = int(cfg.n_obs_steps)
    n_act = int(cfg.n_action_steps)
    n_lat = int(cfg.get("n_latency_steps", 0))
    if n_obs - 1 + n_act > horizon:
        raise ValueError(f"inconsistent chunking: n_obs={n_obs} n_act={n_act} horizon={horizon}")
    return {
        "horizon": horizon,
        "n_obs_steps": n_obs,
        "n_action_steps": n_act,
        "n_latency_steps": n_lat,
        # DP lowdim executes action_pred[:, n_obs-1 : n_obs-1+n_act]
        "action_offset": n_obs - 1 + n_lat,
        "num_inference_steps": int(cfg.policy.num_inference_steps),
        "obs_keys": list(cfg.task.obs_keys),
        "abs_action": bool(cfg.task.abs_action),
        "dataset_path": str(cfg.task.dataset_path),
        "max_steps": int(cfg.task.env_runner.max_steps),
        "action_dim": int(cfg.task.action_dim),
        "obs_dim": int(cfg.task.obs_dim),
    }


def load_policy(ckpt_path: Path, device: str = "cuda:0", dataset_cache: Optional[dict] = None):
    """Load (cfg, policy, dataset, exec_cfg). Uses EMA weights like training eval."""
    import torch

    from diffusion_policy.scripts.plv_chunk_aware_checkpoint_metrics import (
        load_policy_with_dataset_cache,
    )

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    cache = dataset_cache if dataset_cache is not None else {}
    cfg, policy, dataset, state_key = load_policy_with_dataset_cache(
        ckpt_path, torch.device(device), use_ema=True, dataset_cache=cache
    )
    exec_cfg = validate_checkpoint_config(cfg)
    exec_cfg["state_key"] = state_key
    return cfg, policy, dataset, exec_cfg
