from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


PROHIBITED_MODEL_FIELDS = frozenset(
    {
        "action_l1",
        "action_l2",
        "action_mse",
        "val_action_l1",
        "val_loss",
        "validation_loss",
        "checkpoint_epoch",
        "epoch",
        "rollout_success",
        "test_mean_score",
        "success",
    }
)


def validate_outcome_blind_fields(fields: Iterable[str]) -> tuple[str, ...]:
    names = tuple(str(value) for value in fields)
    prohibited = sorted(set(names) & PROHIBITED_MODEL_FIELDS)
    if prohibited:
        raise ValueError(f"prohibited V4 model fields: {prohibited}")
    if len(names) != len(set(names)):
        raise ValueError("V4 model fields must be unique")
    if not names:
        raise ValueError("V4 requires at least one model field")
    return names


class MultiHeadHazardMLP(nn.Module):
    """Shared trajectory encoder with binary physical-hazard heads."""

    def __init__(self, input_dim: int, heads: Sequence[str], hidden_dim: int = 64):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("model dimensions must be positive")
        self.head_names = tuple(str(value) for value in heads)
        if not self.head_names or len(set(self.head_names)) != len(self.head_names):
            raise ValueError("hazard heads must be non-empty and unique")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim, len(self.head_names))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2:
            raise ValueError("hazard input must be a two-dimensional tensor")
        return self.output(self.encoder(value))


@dataclass
class FittedHazardEnsemble:
    models: list[MultiHeadHazardMLP]
    center: np.ndarray
    scale: np.ndarray
    feature_names: tuple[str, ...]
    head_names: tuple[str, ...]

    def _normalize(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != len(self.feature_names):
            raise ValueError("hazard feature matrix has an invalid shape")
        return np.clip((value - self.center) / self.scale, -12.0, 12.0)

    def predict_members(
        self, value: np.ndarray, batch_size: int = 8192
    ) -> np.ndarray:
        normalized = self._normalize(value)
        members = []
        for model in self.models:
            model.eval()
            parts = []
            with torch.no_grad():
                for start in range(0, len(normalized), batch_size):
                    tensor = torch.from_numpy(
                        normalized[start : start + batch_size]
                    )
                    parts.append(torch.sigmoid(model(tensor)).cpu().numpy())
            members.append(np.concatenate(parts, axis=0))
        return np.stack(members, axis=0)

    def predict(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        members = self.predict_members(value)
        return np.median(members, axis=0), np.var(members, axis=0)


def _balanced_binary_weights(target: torch.Tensor) -> torch.Tensor:
    positives = target.sum(dim=0)
    negatives = target.shape[0] - positives
    weights = negatives / positives.clamp_min(1.0)
    weights[(positives == 0) | (negatives == 0)] = 1.0
    return weights


def fit_multihead_hazard_ensemble(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    feature_names: Sequence[str],
    head_names: Sequence[str],
    seeds: Sequence[int],
    hidden_dim: int = 64,
    epochs: int = 60,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> FittedHazardEnsemble:
    names = validate_outcome_blind_fields(feature_names)
    heads = tuple(str(value) for value in head_names)
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(names):
        raise ValueError("features do not match feature_names")
    if targets.ndim != 2 or targets.shape != (len(features), len(heads)):
        raise ValueError("targets do not match features and heads")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("V4 training arrays must be finite")
    if not np.isin(targets, [0.0, 1.0]).all():
        raise ValueError("V4 hazard targets must be binary")
    if not seeds:
        raise ValueError("V4 requires at least one ensemble member")

    center = np.median(features, axis=0).astype(np.float32)
    q25, q75 = np.percentile(features, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1e-4).astype(np.float32)
    normalized = np.clip((features - center) / scale, -12.0, 12.0)
    train_x = torch.from_numpy(normalized.astype(np.float32))
    train_y = torch.from_numpy(targets)
    pos_weight = _balanced_binary_weights(train_y)
    models = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = MultiHeadHazardMLP(
            input_dim=features.shape[1],
            heads=heads,
            hidden_dim=hidden_dim,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        generator = torch.Generator().manual_seed(int(seed) + 401)
        for _ in range(int(epochs)):
            order = torch.randperm(len(train_x), generator=generator)
            for start in range(0, len(order), int(batch_size)):
                indices = order[start : start + int(batch_size)]
                logits = model(train_x[indices])
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    train_y[indices],
                    pos_weight=pos_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        models.append(model)
    return FittedHazardEnsemble(
        models=models,
        center=center,
        scale=scale,
        feature_names=names,
        head_names=heads,
    )
