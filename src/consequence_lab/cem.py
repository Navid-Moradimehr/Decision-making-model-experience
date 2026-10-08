"""Small diagonal CEM with projection for bounded continuous actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BoundedCEMResult:
    candidates: FloatArray
    costs: FloatArray
    mean: FloatArray
    std: FloatArray
    best_index: int


def run_bounded_cem(
    cost_fn: Callable[[FloatArray], NDArray[np.floating]],
    *,
    horizon_shape: tuple[int, ...],
    num_samples: int = 300,
    iterations: int = 30,
    num_elites: int = 30,
    seed: int = 0,
    lower_bound: float = -1.0,
    upper_bound: float = 1.0,
    initial_std: float = 1.0,
    minimum_std: float = 0.05,
) -> BoundedCEMResult:
    """Optimize a fixed-shape action sequence with projected diagonal CEM.

    Projection occurs before scoring and before elite statistics are updated,
    so the optimizer never scores or returns an action outside the legal set.
    This is intentionally not byte-for-byte the pinned stable-worldmodel CEM:
    that implementation samples Gaussian actions without bound projection.
    """
    if not horizon_shape or any(dimension <= 0 for dimension in horizon_shape):
        raise ValueError("horizon_shape dimensions must be positive")
    if not 1 <= num_elites <= num_samples:
        raise ValueError("num_elites must be between 1 and num_samples")
    if num_samples <= 0 or iterations <= 0:
        raise ValueError("num_samples and iterations must be positive")
    if lower_bound >= upper_bound:
        raise ValueError("lower_bound must be less than upper_bound")
    if initial_std <= 0 or minimum_std <= 0:
        raise ValueError("initial_std and minimum_std must be positive")

    rng = np.random.default_rng(seed)
    mean = np.zeros(horizon_shape, dtype=np.float64)
    std = np.full(horizon_shape, initial_std, dtype=np.float64)
    candidates: FloatArray | None = None
    costs: FloatArray | None = None

    for _ in range(iterations):
        candidates = rng.normal(
            loc=mean, scale=std, size=(num_samples, *horizon_shape)
        )
        np.clip(candidates, lower_bound, upper_bound, out=candidates)
        # Preserve the current feasible mean as the first candidate.
        candidates[0] = np.clip(mean, lower_bound, upper_bound)
        costs = np.asarray(cost_fn(candidates), dtype=np.float64)
        if costs.shape != (num_samples,):
            raise ValueError(
                f"cost_fn must return shape ({num_samples},), got {costs.shape}"
            )
        if not np.isfinite(costs).all():
            raise ValueError("cost_fn returned non-finite candidate costs")

        elite_indices = np.argsort(costs, kind="stable")[:num_elites]
        elites = candidates[elite_indices]
        mean = np.clip(elites.mean(axis=0), lower_bound, upper_bound)
        std = np.maximum(elites.std(axis=0), minimum_std)

    assert candidates is not None and costs is not None
    best_index = int(np.argmin(costs))
    return BoundedCEMResult(
        candidates=candidates,
        costs=costs,
        mean=mean,
        std=std,
        best_index=best_index,
    )
