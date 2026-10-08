"""Small permutation-equivariant models for cached candidate-decision studies."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _validate_set(state: torch.Tensor, candidates: torch.Tensor, mask: torch.Tensor) -> None:
    if state.ndim != 2 or state.shape[-1] != 6:
        raise ValueError("state must have shape [batch,6] (qpos, qvel, goal qpos)")
    if candidates.ndim != 3 or candidates.shape[0] != state.shape[0]:
        raise ValueError("candidates must have shape [batch,count,features]")
    if candidates.shape[-1] != 54:
        raise ValueError("candidate features must contain 50 action and 4 score features")
    if mask.shape != candidates.shape[:2]:
        raise ValueError("valid-candidate mask must match [batch,count]")
    if not bool(mask.to(dtype=torch.bool).any(dim=1).all()):
        raise ValueError("each candidate set must contain at least one valid candidate")
    if not torch.isfinite(state).all() or not torch.isfinite(candidates[mask.to(torch.bool)]).all():
        raise ValueError("valid state and candidate features must be finite")


class DirectCandidateModel(nn.Module):
    """Direct state/action Q or policy head with shared candidate parameters."""

    def __init__(self, output: str = "cost", width: int = 128):
        super().__init__()
        if output not in {"cost", "logit"}:
            raise ValueError("output must be 'cost' or 'logit'")
        self.output = output
        self.net = nn.Sequential(
            nn.Linear(6 + 50, width),
            nn.GELU(),
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != 6:
            raise ValueError("state must have shape [batch,6]")
        if actions.ndim != 3 or actions.shape[0] != state.shape[0] or actions.shape[-1] != 50:
            raise ValueError("actions must have shape [batch,count,50]")
        if mask.shape != actions.shape[:2] or not bool(mask.to(torch.bool).any(dim=1).all()):
            raise ValueError("mask must match actions and each row must contain a valid candidate")
        expanded = state.unsqueeze(1).expand(-1, actions.shape[1], -1)
        output = self.net(torch.cat([expanded, actions], dim=-1)).squeeze(-1)
        fill = torch.finfo(output.dtype).max if self.output == "cost" else -torch.finfo(output.dtype).max
        return output.masked_fill(~mask.to(torch.bool), fill)


class _CandidateEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6 + input_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
        )

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        expanded = state.unsqueeze(1).expand(-1, candidates.shape[1], -1)
        return self.net(torch.cat([expanded, candidates], dim=-1))


class SharedRankCorrection(nn.Module):
    """Independent shared candidate scorer with a bounded residual rank edit."""

    def __init__(self, width: int = 64, correction_bound: float = 0.2):
        super().__init__()
        self.correction_bound = float(correction_bound)
        self.encoder = _CandidateEncoder(54, width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self, state: torch.Tensor, candidates: torch.Tensor, base_rank: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_set(state, candidates, mask)
        if base_rank.shape != mask.shape:
            raise ValueError("base ranks must match candidate mask")
        hidden = self.encoder(state, candidates)
        delta = self.correction_bound * torch.tanh(self.head(hidden).squeeze(-1))
        scores = (base_rank + delta).masked_fill(~mask.to(torch.bool), torch.finfo(base_rank.dtype).max)
        return scores, delta


class RelationalRankCorrection(nn.Module):
    """Set Transformer correction; candidate order has no positional meaning."""

    def __init__(
        self,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        correction_bound: float = 0.2,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by attention heads")
        self.correction_bound = float(correction_bound)
        self.encoder = _CandidateEncoder(54, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=2 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(width, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, state: torch.Tensor, candidates: torch.Tensor, base_rank: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_set(state, candidates, mask)
        if base_rank.shape != mask.shape:
            raise ValueError("base ranks must match candidate mask")
        valid = mask.to(torch.bool)
        hidden = self.encoder(state, candidates)
        hidden = self.relation(hidden, src_key_padding_mask=~valid)
        delta = self.correction_bound * torch.tanh(self.head(hidden).squeeze(-1))
        scores = (base_rank + delta).masked_fill(~valid, torch.finfo(base_rank.dtype).max)
        return scores, delta


def student_target_distribution(costs: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Soft preference over per-set range-normalized regret (not success probability)."""
    if costs.ndim != 2 or costs.shape[1] == 0:
        raise ValueError("costs must have shape [batch,count]")
    if not torch.isfinite(costs).all() or temperature <= 0:
        raise ValueError("costs must be finite and temperature positive")
    regret = costs - costs.min(dim=1, keepdim=True).values
    span = costs.max(dim=1, keepdim=True).values - costs.min(dim=1, keepdim=True).values
    normalized = regret / span.clamp_min(1e-12)
    return torch.softmax(-normalized / temperature, dim=1)


def ordinal_correction_loss(
    scores: torch.Tensor,
    delta: torch.Tensor,
    successes: torch.Tensor,
    base_rank: torch.Tensor,
    *,
    temperature: float = 0.05,
    margin: float = 0.02,
    pair_weight: float = 0.25,
    regularization_weight: float = 0.1,
    local_top_k: int = 16,
) -> torch.Tensor:
    """D-JEPA-inspired success-mass, local ranking and trust-region objective."""
    if scores.ndim != 2 or delta.shape != scores.shape or successes.shape != scores.shape:
        raise ValueError("scores, deltas, and success mask must share [batch,count]")
    if base_rank.shape != scores.shape:
        raise ValueError("base rank must match score shape")
    losses = []
    for row in range(scores.shape[0]):
        success = successes[row].to(torch.bool)
        if success.any():
            p = torch.softmax(-scores[row] / temperature, dim=0)
            mass = p[success].sum().clamp_min(torch.finfo(p.dtype).tiny)
            success_loss = -torch.log(mass)
        else:
            success_loss = scores[row].new_zeros(())

        order = torch.argsort(base_rank[row], stable=True)
        boundary = min(local_top_k, len(order))
        cutoff = base_rank[row, order[boundary - 1]]
        local = base_rank[row] <= cutoff
        positives = torch.nonzero(local & success, as_tuple=False).flatten()
        negatives = torch.nonzero(local & ~success, as_tuple=False).flatten()
        if len(positives) == 0 or len(negatives) == 0:
            positives = torch.nonzero(success, as_tuple=False).flatten()
            negatives = torch.nonzero(~success, as_tuple=False).flatten()
        if len(positives) and len(negatives):
            gaps = (
                scores[row, positives][:, None]
                - scores[row, negatives][None, :]
                + margin
            ) / temperature
            pair_loss = F.softplus(gaps).mean()
        else:
            pair_loss = scores[row].new_zeros(())
        regularizer = delta[row].square().mean()
        losses.append(success_loss + pair_weight * pair_loss + regularization_weight * regularizer)
    return torch.stack(losses).mean()
