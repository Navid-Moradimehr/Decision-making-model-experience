"""Small dynamic-candidate models for the TwoRoom Phase-3 study.

The models consume only numeric runtime observations, candidate sequences, and
validity masks. Teacher costs, ranks, IDs, and futures are kept in the training
loop and never enter these forward interfaces.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def candidate_features(candidate_sequences: np.ndarray) -> np.ndarray:
    """Encode each plan by its mean action in the first and second half.

    This is permutation-safe across candidates and invariant to repeated
    within-segment actions. It intentionally does not encode candidate row IDs.
    """
    plans = np.asarray(candidate_sequences, dtype=np.float32)
    if plans.ndim != 3 or plans.shape[0] == 0 or plans.shape[1] < 2 or plans.shape[2] != 2:
        raise ValueError("candidate sequences must have shape [N,H,2], H >= 2")
    if not np.isfinite(plans).all() or np.any(np.abs(plans) > 1.0 + 1e-6):
        raise ValueError("candidate actions must be finite and within [-1,1]")
    split = plans.shape[1] // 2
    first = plans[:, :split].mean(axis=1)
    second = plans[:, split:].mean(axis=1)
    return np.concatenate([first, second], axis=-1).astype(np.float32)


def normalize_observation(observation: np.ndarray) -> np.ndarray:
    """Normalize the ten pixel-valued deployable observation features."""
    values = np.asarray(observation, dtype=np.float32)
    if values.shape[-1] != 10 or not np.isfinite(values).all():
        raise ValueError("observation must be finite with final dimension 10")
    return values / np.float32(224.0)


try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - covered in the optional training env
    raise ImportError(
        "consequence_lab.student requires the optional training dependency: torch"
    ) from exc


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class DirectActionPolicy(nn.Module):
    """Observation-only policy over the nine legal primitive actions."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.state_encoder = _MLP(10, hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 9)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.state_encoder(normalize_observation_tensor(observation)))


class CandidateScorer(nn.Module):
    """Shared candidate MLP; optionally adds a DeepSets masked-mean context."""

    def __init__(self, hidden_dim: int = 64, set_context: bool = False):
        super().__init__()
        self.set_context = bool(set_context)
        self.state_encoder = _MLP(10, hidden_dim, hidden_dim)
        self.candidate_encoder = _MLP(4, hidden_dim, hidden_dim)
        scorer_input = hidden_dim * (3 if self.set_context else 2)
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        observation: torch.Tensor,
        candidate_features_tensor: torch.Tensor,
        valid_candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_features_tensor.ndim != 3 or candidate_features_tensor.shape[-1] != 4:
            raise ValueError("candidate features must have shape [batch,count,4]")
        batch, count, _ = candidate_features_tensor.shape
        if valid_candidate_mask.shape != (batch, count):
            raise ValueError("valid candidate mask must have shape [batch,count]")
        mask = valid_candidate_mask.to(dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every batch row must contain a valid candidate")

        state = self.state_encoder(normalize_observation_tensor(observation))
        candidate = self.candidate_encoder(candidate_features_tensor)
        expanded_state = state.unsqueeze(1).expand(-1, count, -1)
        pieces = [expanded_state, candidate]
        if self.set_context:
            weights = mask.unsqueeze(-1).to(candidate.dtype)
            context = (candidate * weights).sum(dim=1, keepdim=True) / weights.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            pieces.append(context.expand(-1, count, -1))
        logits = self.scorer(torch.cat(pieces, dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class CandidateCostRegressor(nn.Module):
    """Direct value-equivalent baseline predicting normalized candidate cost."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.state_encoder = _MLP(10, hidden_dim, hidden_dim)
        self.candidate_encoder = _MLP(4, hidden_dim, hidden_dim)
        self.cost_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        observation: torch.Tensor,
        candidate_features_tensor: torch.Tensor,
        valid_candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, _ = candidate_features_tensor.shape
        if valid_candidate_mask.shape != (batch, count):
            raise ValueError("valid candidate mask must have shape [batch,count]")
        state = self.state_encoder(normalize_observation_tensor(observation))
        candidate = self.candidate_encoder(candidate_features_tensor)
        state = state.unsqueeze(1).expand(-1, count, -1)
        costs = self.cost_head(torch.cat([state, candidate], dim=-1)).squeeze(-1)
        return costs.masked_fill(
            ~valid_candidate_mask.to(dtype=torch.bool), torch.finfo(costs.dtype).max
        )


def normalize_observation_tensor(observation: torch.Tensor) -> torch.Tensor:
    if observation.shape[-1] != 10:
        raise ValueError("observation tensor must have final dimension 10")
    return observation / 224.0


def masked_probabilities(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != mask.shape or not bool(mask.to(dtype=torch.bool).any(dim=-1).all()):
        raise ValueError("logits and mask must match and each row must have a valid action")
    return torch.softmax(logits.masked_fill(~mask.to(dtype=torch.bool), -torch.inf), dim=-1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_output_to_numpy(value: Any) -> np.ndarray:
    """Convenience helper for deterministic test and evaluation boundaries."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("expected a torch tensor")
    return value.detach().cpu().numpy()
