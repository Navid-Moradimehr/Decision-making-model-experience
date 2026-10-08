"""Fixed candidate-bank helpers for the Phase-2 latent diagnostic."""

from __future__ import annotations

import numpy as np


def make_candidate_bank(
    reference_actions: np.ndarray,
    *,
    uniform_seed: int,
    perturbation_seed: int,
    uniform_count: int,
    perturbation_count: int,
    perturbation_std: float,
    lower_bound: float = -1.0,
    upper_bound: float = 1.0,
) -> dict[str, np.ndarray]:
    """Generate reproducible fixed banks; the exact reference stays separate."""
    reference = np.asarray(reference_actions, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] != 2 or reference.shape[0] == 0:
        raise ValueError("reference_actions must have shape [time, 2]")
    if not np.isfinite(reference).all():
        raise ValueError("reference_actions must be finite")
    if lower_bound >= upper_bound or not np.isfinite([lower_bound, upper_bound]).all():
        raise ValueError("action bounds must be finite and ordered")
    if np.any(reference < lower_bound) or np.any(reference > upper_bound):
        raise ValueError("reference_actions exceed action bounds")
    if uniform_count <= 0 or perturbation_count <= 0:
        raise ValueError("candidate counts must be positive")
    if perturbation_std <= 0 or not np.isfinite(perturbation_std):
        raise ValueError("perturbation_std must be positive and finite")

    uniform_rng = np.random.default_rng(uniform_seed)
    perturbation_rng = np.random.default_rng(perturbation_seed)
    uniform = uniform_rng.uniform(
        lower_bound,
        upper_bound,
        size=(uniform_count, *reference.shape),
    )
    perturbed = reference + perturbation_rng.normal(
        0.0, perturbation_std, size=(perturbation_count, *reference.shape)
    )
    np.clip(perturbed, lower_bound, upper_bound, out=perturbed)
    return {
        "reference_replay": reference[None, ...].copy(),
        "uniform_random": uniform,
        "reference_perturbation": perturbed,
    }


def tie_aware_pairwise_accuracy(
    predicted: np.ndarray,
    actual: np.ndarray,
    *,
    actual_tie_tolerance: float,
    predicted_tie_tolerance: float,
) -> tuple[float | None, int]:
    """Return equal pairwise credit, excluding actual ties and half-crediting predicted ties."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(actual, dtype=np.float64)
    if pred.ndim != 1 or truth.shape != pred.shape or pred.size == 0:
        raise ValueError("predicted and actual must be same-length non-empty vectors")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("ranking costs must be finite")
    if (
        not np.isfinite(actual_tie_tolerance)
        or not np.isfinite(predicted_tie_tolerance)
        or actual_tie_tolerance < 0
        or predicted_tie_tolerance < 0
    ):
        raise ValueError("tie tolerances must be finite and non-negative")

    i, j = np.triu_indices(pred.size, k=1)
    actual_delta = truth[i] - truth[j]
    informative = np.abs(actual_delta) > actual_tie_tolerance
    count = int(informative.sum())
    if count == 0:
        return None, 0
    predicted_delta = pred[i] - pred[j]
    tied = np.abs(predicted_delta) <= predicted_tie_tolerance
    credit = np.where(
        tied,
        0.5,
        (np.sign(predicted_delta) == np.sign(actual_delta)).astype(np.float64),
    )
    return float(credit[informative].mean()), count
