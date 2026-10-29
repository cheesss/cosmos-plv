from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from diffusion_policy.model.ood.outcome_blind_hazard import (
    validate_outcome_blind_fields,
)


class DirectChunkHazardNet(nn.Module):
    """Predict a compact chunk endpoint and local physical hazards."""

    def __init__(
        self,
        input_dim: int,
        endpoint_dim: int,
        hazard_heads: Sequence[str],
        hidden_dim: int = 128,
        endpoint_conditioned: bool = False,
    ):
        super().__init__()
        self.endpoint_conditioned = bool(endpoint_conditioned)
        self.hazard_heads = tuple(str(value) for value in hazard_heads)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.endpoint_head = nn.Linear(hidden_dim, endpoint_dim)
        self.recovery_head = nn.Linear(hidden_dim, 1)
        hazard_input_dim = hidden_dim + (endpoint_dim if endpoint_conditioned else 0)
        self.hazard_head = nn.Sequential(
            nn.Linear(hazard_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(self.hazard_heads)),
        )

    def forward(
        self,
        value: torch.Tensor,
        endpoint_for_hazard: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(value)
        endpoint = self.endpoint_head(encoded)
        recovery = self.recovery_head(encoded)
        if self.endpoint_conditioned:
            if endpoint_for_hazard is None:
                endpoint_for_hazard = endpoint
            hazard_input = torch.cat([encoded, endpoint_for_hazard], dim=-1)
        else:
            hazard_input = encoded
        hazard_logits = self.hazard_head(hazard_input)
        return endpoint, recovery, hazard_logits


@dataclass
class FittedDirectChunkHazardEnsemble:
    models: list[DirectChunkHazardNet]
    input_center: np.ndarray
    input_scale: np.ndarray
    endpoint_center: np.ndarray
    endpoint_scale: np.ndarray
    recovery_center: float
    recovery_scale: float
    feature_names: tuple[str, ...]
    endpoint_names: tuple[str, ...]
    hazard_heads: tuple[str, ...]
    endpoint_conditioned: bool

    def _normalize_input(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return np.clip(
            (value - self.input_center) / self.input_scale, -12.0, 12.0
        ).astype(np.float32)

    def _normalize_endpoint(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        return np.clip(
            (value - self.endpoint_center) / self.endpoint_scale, -12.0, 12.0
        ).astype(np.float32)

    def predict_members(
        self,
        value: np.ndarray,
        endpoint_for_hazard: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        normalized = torch.from_numpy(self._normalize_input(value))
        endpoint_tensor = None
        if endpoint_for_hazard is not None:
            endpoint_tensor = torch.from_numpy(
                self._normalize_endpoint(endpoint_for_hazard)
            )
        endpoints, recoveries, hazards = [], [], []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                endpoint, recovery, logits = model(
                    normalized, endpoint_for_hazard=endpoint_tensor
                )
                endpoints.append(
                    endpoint.cpu().numpy() * self.endpoint_scale
                    + self.endpoint_center
                )
                recoveries.append(
                    recovery.cpu().numpy().reshape(-1) * self.recovery_scale
                    + self.recovery_center
                )
                hazards.append(torch.sigmoid(logits).cpu().numpy())
        return (
            np.stack(endpoints, axis=0),
            np.stack(recoveries, axis=0),
            np.stack(hazards, axis=0),
        )

    def predict(
        self,
        value: np.ndarray,
        endpoint_for_hazard: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        endpoints, recoveries, hazards = self.predict_members(
            value, endpoint_for_hazard=endpoint_for_hazard
        )
        return {
            "endpoint": np.median(endpoints, axis=0),
            "endpoint_uncertainty": np.mean(np.var(endpoints, axis=0), axis=1),
            "recovery": np.median(recoveries, axis=0),
            "recovery_uncertainty": np.var(recoveries, axis=0),
            "hazard": np.median(hazards, axis=0),
            "hazard_uncertainty": np.mean(np.var(hazards, axis=0), axis=1),
        }


def _robust_location_scale(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(value, axis=0).astype(np.float32)
    q25, q75 = np.percentile(value, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1e-4).astype(np.float32)
    return center, scale


def fit_direct_chunk_hazard_ensemble(
    features: np.ndarray,
    endpoint_targets: np.ndarray,
    recovery_targets: np.ndarray,
    hazard_targets: np.ndarray,
    *,
    feature_names: Sequence[str],
    endpoint_names: Sequence[str],
    hazard_heads: Sequence[str],
    seeds: Sequence[int],
    hidden_dim: int = 128,
    epochs: int = 80,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    endpoint_conditioned: bool = False,
) -> FittedDirectChunkHazardEnsemble:
    feature_names = validate_outcome_blind_fields(feature_names)
    endpoint_names = validate_outcome_blind_fields(endpoint_names)
    hazard_heads = tuple(str(value) for value in hazard_heads)
    features = np.asarray(features, dtype=np.float32)
    endpoint_targets = np.asarray(endpoint_targets, dtype=np.float32)
    recovery_targets = np.asarray(recovery_targets, dtype=np.float32).reshape(-1)
    hazard_targets = np.asarray(hazard_targets, dtype=np.float32)
    if features.shape != (len(features), len(feature_names)):
        raise ValueError("invalid direct-chunk feature shape")
    if endpoint_targets.shape != (len(features), len(endpoint_names)):
        raise ValueError("invalid direct-chunk endpoint shape")
    if hazard_targets.shape != (len(features), len(hazard_heads)):
        raise ValueError("invalid direct-chunk hazard shape")
    arrays = (features, endpoint_targets, recovery_targets, hazard_targets)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("direct-chunk training arrays must be finite")
    if not np.isin(hazard_targets, [0.0, 1.0]).all():
        raise ValueError("direct-chunk hazards must be binary")

    input_center, input_scale = _robust_location_scale(features)
    endpoint_center, endpoint_scale = _robust_location_scale(endpoint_targets)
    recovery_center = float(np.median(recovery_targets))
    recovery_q25, recovery_q75 = np.percentile(recovery_targets, [25, 75])
    recovery_scale = float(max(recovery_q75 - recovery_q25, 1e-4))
    x = torch.from_numpy(
        np.clip((features - input_center) / input_scale, -12.0, 12.0)
    )
    endpoint_y = torch.from_numpy(
        np.clip(
            (endpoint_targets - endpoint_center) / endpoint_scale, -12.0, 12.0
        )
    )
    recovery_y = torch.from_numpy(
        ((recovery_targets - recovery_center) / recovery_scale)[:, None]
    )
    hazard_y = torch.from_numpy(hazard_targets)
    positives = hazard_y.sum(dim=0)
    negatives = len(hazard_y) - positives
    pos_weight = negatives / positives.clamp_min(1.0)
    pos_weight[(positives == 0) | (negatives == 0)] = 1.0

    models: list[DirectChunkHazardNet] = []
    for seed in seeds:
        torch.manual_seed(int(seed))
        model = DirectChunkHazardNet(
            input_dim=features.shape[1],
            endpoint_dim=endpoint_targets.shape[1],
            hazard_heads=hazard_heads,
            hidden_dim=hidden_dim,
            endpoint_conditioned=endpoint_conditioned,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        generator = torch.Generator().manual_seed(int(seed) + 211)
        for _ in range(int(epochs)):
            order = torch.randperm(len(x), generator=generator)
            for start in range(0, len(order), int(batch_size)):
                index = order[start : start + int(batch_size)]
                endpoint, recovery, hazard_logits = model(
                    x[index],
                    endpoint_for_hazard=(
                        endpoint_y[index] if endpoint_conditioned else None
                    ),
                )
                endpoint_loss = torch.nn.functional.smooth_l1_loss(
                    endpoint, endpoint_y[index]
                )
                recovery_loss = torch.nn.functional.smooth_l1_loss(
                    recovery, recovery_y[index]
                )
                hazard_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    hazard_logits, hazard_y[index], pos_weight=pos_weight
                )
                loss = endpoint_loss + recovery_loss + hazard_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        models.append(model)
    return FittedDirectChunkHazardEnsemble(
        models=models,
        input_center=input_center,
        input_scale=input_scale,
        endpoint_center=endpoint_center,
        endpoint_scale=endpoint_scale,
        recovery_center=recovery_center,
        recovery_scale=recovery_scale,
        feature_names=feature_names,
        endpoint_names=endpoint_names,
        hazard_heads=hazard_heads,
        endpoint_conditioned=bool(endpoint_conditioned),
    )
