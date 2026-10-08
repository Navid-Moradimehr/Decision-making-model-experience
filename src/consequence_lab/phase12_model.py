"""Small permutation-equivariant value/policy scorers for Inertial Corridor."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised in training envs
    raise ImportError("Phase 12 models require the optional torch dependency") from exc


class CorridorCandidateScorer(nn.Module):
    """A shared pairwise scorer with no candidate-row positional semantics.

    Applying the same pointwise map to each state/candidate pair makes output
    scores permutation-equivariant by construction. ``kind`` changes only the
    output interpretation; the runtime input contract remains identical.
    """

    def __init__(self, kind: str, hidden_state: int = 128, embedding: int = 64):
        super().__init__()
        if kind not in {"policy", "value"}:
            raise ValueError("kind must be 'policy' or 'value'")
        self.kind = kind
        self.state_encoder = nn.Sequential(
            nn.Linear(52, hidden_state), nn.GELU(), nn.Linear(hidden_state, embedding), nn.GELU()
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(80, embedding), nn.GELU(), nn.Linear(embedding, embedding), nn.GELU()
        )
        self.shared_scorer = nn.Sequential(
            nn.Linear(embedding * 2, embedding), nn.GELU(), nn.Linear(embedding, 1)
        )

    def forward(self, state: torch.Tensor, candidates: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != 52:
            raise ValueError("state must have shape [batch,52]")
        if candidates.ndim != 3 or candidates.shape[-1] != 80:
            raise ValueError("candidates must have shape [batch,count,80]")
        if candidates.shape[0] != state.shape[0] or mask.shape != candidates.shape[:2]:
            raise ValueError("state, candidates and mask batch dimensions must match")
        valid = mask.to(dtype=torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each candidate set must have at least one valid candidate")
        state_embedding = self.state_encoder(state).unsqueeze(1).expand(-1, candidates.shape[1], -1)
        candidate_embedding = self.candidate_encoder(candidates)
        scores = self.shared_scorer(torch.cat((state_embedding, candidate_embedding), dim=-1)).squeeze(-1)
        if self.kind == "policy":
            return scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        return scores.masked_fill(~valid, torch.finfo(scores.dtype).max)


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
