"""Read-only D153 coverage statistics over the previously used validation block."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from consequence_lab.phase15_representation_sufficiency import mean_rank_correlation


def correlation_pair(left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    """Return Pearson and tie-aware Spearman correlations for matching vectors."""
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("D153 correlation inputs must be matching finite vectors")
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else None
    return {"pearson": pearson, "spearman": mean_rank_correlation(x, y)}


def cluster_bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float | None],
    *,
    resamples: int,
    seed: int,
) -> list[float | None]:
    """Percentile interval resampling whole layouts, never individual states."""
    x, y = np.asarray(left), np.asarray(right)
    group = np.asarray(clusters).astype(str)
    if x.ndim != 1 or y.shape != x.shape or group.shape != x.shape:
        raise ValueError("D153 cluster bootstrap arrays must align")
    names = np.asarray(sorted(set(group)))
    if len(names) < 2 or resamples < 1:
        raise ValueError("D153 cluster bootstrap needs at least two layouts and one draw")
    indices = {name: np.flatnonzero(group == name) for name in names}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(resamples):
        selected = rng.choice(names, size=len(names), replace=True)
        take = np.concatenate([indices[name] for name in selected])
        value = statistic(x[take], y[take])
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if len(values) < 0.95 * resamples:
        return [None, None]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def mean_rank_correlation_checked(left: np.ndarray, right: np.ndarray) -> float:
    """Wrapper with explicit finite-vector checks for bootstrap use."""
    value = correlation_pair(left, right)["spearman"]
    return float(value)


def nearest_distance_group_breakdown(
    train: np.ndarray, validation: np.ndarray, nearest_indices: np.ndarray,
    groups: dict[str, tuple[int, int]],
) -> dict[str, dict[str, float]]:
    """Measure each R0 block's distance to the full-R0 nearest training row."""
    tr, va = np.asarray(train, dtype=np.float64), np.asarray(validation, dtype=np.float64)
    ids = np.asarray(nearest_indices, dtype=np.int64)
    if tr.ndim != 2 or va.ndim != 2 or tr.shape[1] != va.shape[1] or ids.shape != (len(va),):
        raise ValueError("D153 nearest-distance arrays are misaligned")
    if np.any(ids < 0) or np.any(ids >= len(tr)):
        raise ValueError("D153 nearest training index is outside the train split")
    result: dict[str, dict[str, float]] = {}
    squared: dict[str, np.ndarray] = {}
    full_squared = np.zeros(len(va), dtype=np.float64)
    for name, (start, stop) in groups.items():
        if not 0 <= start < stop <= tr.shape[1]:
            raise ValueError("D153 R0 feature group is outside the representation")
        mu = tr[:, start:stop].mean(axis=0)
        sd = tr[:, start:stop].std(axis=0)
        sd[sd < 1e-8] = 1.0
        diff = (va[:, start:stop] - tr[ids, start:stop]) / sd
        contribution = np.sum(diff * diff, axis=1)
        squared[name] = contribution
        full_squared += contribution
        result[name] = {"nearest_pair_distance_median": float(np.median(np.sqrt(contribution))),
                        "mean_squared_distance_fraction": 0.0}
    denominator = max(float(np.mean(full_squared)), 1e-12)
    for name in result:
        result[name]["mean_squared_distance_fraction"] = float(np.mean(squared[name]) / denominator)
    return result
