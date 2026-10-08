"""Simulator-comparable goal costs and ranking metrics."""

from __future__ import annotations

import numpy as np


def potential_shaped_reward(
    raw_reward: float,
    current_potential: float,
    next_potential: float,
    gamma: float,
) -> float:
    """Return the potential-shaped reward ``r + gamma*Phi(s') - Phi(s)``.

    This helper does not make finite-horizon planning policy-invariant; it only
    implements the per-transition shaping term. A finite discounted sum retains
    the terminal ``gamma**H * Phi(s_H)`` boundary term.
    """
    values = np.asarray([raw_reward, current_potential, next_potential, gamma], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("reward, potentials, and gamma must be finite")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    return float(raw_reward + gamma * next_potential - current_potential)


def goal_cost(
    trajectories: np.ndarray,
    goal: np.ndarray,
    arena_diagonal: float,
    terminal_distance_weight: float = 1.0,
    mean_distance_weight: float = 0.1,
    success_distance: float = 16.0,
) -> np.ndarray:
    """Compute one bounded, scale-normalized distance cost per candidate.

    Inputs have shape ``[candidate, time, 2]`` and ``[2]``. After first
    reaching the goal tolerance, a candidate remains at the absorbing goal
    for the rest of the scored rollout.
    """
    positions = np.asarray(trajectories, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1] == 0 or positions.shape[-1] != 2:
        raise ValueError("trajectories must have shape [candidate, time, 2]")
    if target.shape != (2,):
        raise ValueError("goal must have shape [2]")
    if not np.isfinite(positions).all() or not np.isfinite(target).all():
        raise ValueError("positions and goal must be finite")
    if not np.isfinite(arena_diagonal) or arena_diagonal <= 0:
        raise ValueError("arena_diagonal must be positive and finite")
    if (
        not np.isfinite(terminal_distance_weight)
        or not np.isfinite(mean_distance_weight)
        or terminal_distance_weight < 0
        or mean_distance_weight < 0
    ):
        raise ValueError("objective weights must be non-negative")
    if not np.isfinite(success_distance) or success_distance < 0:
        raise ValueError("success_distance must be non-negative and finite")

    distances = np.linalg.norm(positions - target, axis=-1)
    reached = distances < success_distance
    any_reached = reached.any(axis=1)
    first_reached = np.argmax(reached, axis=1)
    time = np.arange(positions.shape[1])[None, :]
    distances = np.where(any_reached[:, None] & (time >= first_reached[:, None]), 0, distances)
    normalized = distances / arena_diagonal
    return (
        terminal_distance_weight * normalized[:, -1]
        + mean_distance_weight * normalized.mean(axis=1)
    )


def soft_teacher_distribution(costs: np.ndarray, temperature: float) -> np.ndarray:
    """Turn lower candidate costs into a normalized preference distribution."""
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("costs must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("costs must be finite")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    logits = -(values - values.min()) / temperature
    weights = np.exp(logits)
    return (weights / weights.sum()).astype(np.float32)


def tie_aware_pairwise_accuracy(
    predicted: np.ndarray, target: np.ndarray, tie_tolerance: float
) -> float:
    """Fraction of non-tied pairs whose ordering the prediction preserves."""
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.ndim != 1 or truth.shape != pred.shape:
        raise ValueError("predicted and target must be same-length vectors")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("rank values must be finite")
    if tie_tolerance < 0 or not np.isfinite(tie_tolerance):
        raise ValueError("tie_tolerance must be non-negative and finite")

    i, j = np.triu_indices(len(truth), k=1)
    informative = np.abs(truth[i] - truth[j]) > tie_tolerance
    if not informative.any():
        return float("nan")
    correct = (pred[i] - pred[j]) * (truth[i] - truth[j]) > 0
    return float(correct[informative].mean())
