"""Diagnostics for whether exact-teacher preferences follow runtime inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


RUNTIME_INPUT_FIELDS = frozenset(
    {"observation", "candidate_sequences", "valid_candidate_mask"}
)

TWO_ROOM_DEPENDENCIES = (
    {"dependency": "agent_position", "status": "observed", "evidence": "env.py:_get_obs lines 298-319"},
    {"dependency": "target_position", "status": "observed", "evidence": "env.py:_get_obs lines 298-319"},
    {"dependency": "door_centers", "status": "observed_and_fixed", "evidence": "env.py:_get_obs lines 298-319"},
    {"dependency": "candidate_actions", "status": "runtime_candidate_input", "evidence": "phase candidate bank"},
    {"dependency": "agent_speed", "status": "fixed", "evidence": "env.py DEFAULT_VARIATIONS and init value"},
    {"dependency": "agent_radius", "status": "fixed", "evidence": "env.py DEFAULT_VARIATIONS and init value"},
    {"dependency": "wall_axis_and_thickness", "status": "fixed", "evidence": "env.py DEFAULT_VARIATIONS and init values"},
    {"dependency": "door_count_and_half_extents", "status": "fixed", "evidence": "env.py DEFAULT_VARIATIONS and init values"},
    {"dependency": "wall_center_and_arena_bounds", "status": "fixed", "evidence": "env.py class constants and init values"},
    {"dependency": "success_distance", "status": "protocol_constant", "evidence": "Phase-3 YAML environment contract"},
    {"dependency": "arena_diagonal_and_cost_weights", "status": "protocol_constant", "evidence": "Phase-3 YAML teacher objective"},
    {"dependency": "default_variation_keys", "status": "audited", "evidence": "env.py DEFAULT_VARIATIONS and spaces.py reset_variation_space"},
)

_EXPLAINED_DEPENDENCY_STATUSES = frozenset(
    {"observed", "observed_and_fixed", "runtime_candidate_input", "fixed", "protocol_constant", "audited"}
)


def dependency_audit_report() -> dict[str, object]:
    """Return the reviewed source-dependency table and its computed gate count."""
    rows = [dict(row) for row in TWO_ROOM_DEPENDENCIES]
    unexplained = sum(
        row["status"] not in _EXPLAINED_DEPENDENCY_STATUSES for row in rows
    )
    return {"dependencies": rows, "unexplained_dependency_count": int(unexplained)}


def validate_runtime_payload(payload: Mapping[str, object]) -> None:
    """Reject undeclared/teacher-only features at the student boundary."""
    unknown = set(payload) - RUNTIME_INPUT_FIELDS
    missing = RUNTIME_INPUT_FIELDS - set(payload)
    if unknown or missing:
        raise ValueError(
            f"runtime payload fields invalid; missing={sorted(missing)}, "
            f"not_allowlisted={sorted(unknown)}"
        )

    observation = np.asarray(payload["observation"])
    candidates = np.asarray(payload["candidate_sequences"])
    mask = np.asarray(payload["valid_candidate_mask"])
    if observation.ndim != 1 or observation.shape[0] != 10:
        raise ValueError("TwoRoom runtime observation must have shape [10]")
    if candidates.ndim != 3 or candidates.shape[0] == 0 or candidates.shape[2] != 2:
        raise ValueError("candidate sequences must have shape [candidate, time, 2]")
    if mask.shape != (candidates.shape[0],) or mask.dtype != np.bool_:
        raise ValueError("valid candidate mask must be a boolean vector matching candidates")
    if not mask.any():
        raise ValueError("at least one candidate must be valid")


def near_optimal_first_action_sets(
    costs: np.ndarray,
    sequences: np.ndarray,
    regret_epsilon: float,
) -> list[frozenset[tuple[float, float]]]:
    """Map each state to the set of first actions within regret epsilon."""
    values = np.asarray(costs, dtype=np.float64)
    plans = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 2 or plans.ndim != 3 or values.shape[1] != plans.shape[0]:
        raise ValueError("costs [state,candidate] and sequences [candidate,time,2] required")
    if plans.shape[1] < 1 or plans.shape[2] != 2 or not np.isfinite(values).all():
        raise ValueError("candidate sequences/costs must be finite and non-empty")
    if not np.isfinite(regret_epsilon) or regret_epsilon < 0:
        raise ValueError("regret_epsilon must be non-negative and finite")

    result = []
    actions = np.round(plans[:, 0, :], decimals=6)
    for row in values:
        best = float(row.min())
        selected = np.flatnonzero(row - best <= regret_epsilon)
        result.append(
            frozenset((float(actions[i, 0]), float(actions[i, 1])) for i in selected)
        )
    return result


def neighborhood_consistency(
    observations: np.ndarray,
    costs: np.ndarray,
    sequences: np.ndarray,
    episode_ids: Iterable[int],
    radii_px: Iterable[float],
    regret_epsilon: float,
) -> dict[str, dict[str, float | int]]:
    """Describe cross-episode preference conflicts in nearby observations.

    This finite-sample diagnostic is descriptive. It is not an estimator of
    irreducible Bayes error or a proof of teacher realizability.
    """
    obs = np.asarray(observations, dtype=np.float64)
    values = np.asarray(costs, dtype=np.float64)
    episodes = np.asarray(list(episode_ids))
    plans = np.asarray(sequences, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != 10:
        raise ValueError("observations must have shape [state,10]")
    if values.ndim != 2 or values.shape != (len(obs), len(plans)):
        raise ValueError("cost matrix must have shape [state,candidate]")
    if episodes.shape != (len(obs),):
        raise ValueError("one episode id is required per observation")
    if not np.isfinite(obs).all() or not np.isfinite(values).all():
        raise ValueError("observations and costs must be finite")

    best_actions = near_optimal_first_action_sets(values, plans, regret_epsilon)
    pairwise_distance = np.linalg.norm(obs[:, None, :] - obs[None, :, :], axis=-1)
    cross_episode = episodes[:, None] != episodes[None, :]
    np.fill_diagonal(cross_episode, False)
    result: dict[str, dict[str, float | int]] = {}
    for radius in radii_px:
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("neighborhood radii must be non-negative and finite")
        neighbors = (pairwise_distance <= radius) & cross_episode
        pairs = np.argwhere(np.triu(neighbors, k=1))
        conflicts = [
            not bool(best_actions[i].intersection(best_actions[j]))
            for i, j in pairs
        ]
        rows_with_neighbor = neighbors.any(axis=1)
        rows_with_conflict = np.zeros(len(obs), dtype=bool)
        for (i, j), conflict in zip(pairs, conflicts, strict=True):
            if conflict:
                rows_with_conflict[i] = True
                rows_with_conflict[j] = True
        key = f"{float(radius):g}"
        result[key] = {
            "cross_episode_pair_count": int(len(pairs)),
            "conflicting_pair_count": int(sum(conflicts)),
            "conflicting_pair_fraction": (
                float(np.mean(conflicts)) if conflicts else None
            ),
            "state_coverage_with_neighbor": float(rows_with_neighbor.mean()) if len(obs) else 0.0,
            "state_coverage_with_conflict": float(rows_with_conflict.mean()) if len(obs) else 0.0,
        }
    return result
