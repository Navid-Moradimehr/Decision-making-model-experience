"""Exact Phase-12 teacher queries and leakage-safe numeric serialization.

This module deliberately duplicates the frozen Phase-11 proposal contract in
order to keep historical Phase-11 source hashes untouched.  A parity test
against ``exact_dynamics_shooting`` is a prerequisite to collection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import hypot
import random
from typing import Any

import numpy as np

from consequence_lab.corridor_baselines import _shooting_objective, exact_dynamics_shooting
from consequence_lab.corridor_tasks import CorridorScenario
from consequence_lab.inertial_corridor import ACTION_VECTORS, CorridorState, corridor_step


PAD_ACTION = 9
MAX_HORIZON = 8
NUM_CANDIDATES = 128
OUTCOME_CODES = {"success": 0, "active": 1, "timeout": 2, "collision": 3}


@dataclass(frozen=True)
class TeacherQuery:
    sequences: np.ndarray          # [128, 8], action IDs 0..8, PAD=9
    sequence_hashes: tuple[str, ...]
    rollout_states: np.ndarray    # [128, 9, 4], x/y/vx/vy; padded with zeros
    rollout_state_mask: np.ndarray  # [128, 9]
    outcome_codes: np.ndarray     # [128], OUTCOME_CODES
    objective_metric: np.ndarray  # [128], steps for success, distance otherwise
    effort: np.ndarray            # [128], executed non-coast actions
    success_available: bool
    selected_index: int
    objective_keys: tuple[tuple[int, float, float], ...]
    ranks: np.ndarray             # [128], average 1-based rank
    normalized_regret: np.ndarray  # [128]
    optimal_mask: np.ndarray      # [128]


def phase12_query_seed(scenario_seed: int, elapsed_steps: int) -> int:
    if scenario_seed < 0 or elapsed_steps < 0:
        raise ValueError("seed and elapsed_steps must be nonnegative")
    return (int(scenario_seed) * 1_000_003 + int(elapsed_steps)) % (2**32)


def propose_sequences(state: CorridorState, max_steps: int, *, seed: int) -> np.ndarray:
    """Return the exact Phase-11 constant-prefix + seeded-uniform proposal bank."""
    horizon = min(MAX_HORIZON, int(max_steps) - state.elapsed_steps)
    if horizon <= 0:
        raise ValueError("cannot propose candidates from a terminal/horizon state")
    budget = min(NUM_CANDIDATES, 9**horizon)
    proposals: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for action_id in range(len(ACTION_VECTORS)):
        candidate = (action_id,) * horizon
        if candidate not in seen:
            proposals.append(candidate)
            seen.add(candidate)
            if len(proposals) == budget:
                break
    rng = random.Random(int(seed))
    while len(proposals) < budget:
        candidate = tuple(rng.randrange(len(ACTION_VECTORS)) for _ in range(horizon))
        if candidate not in seen:
            proposals.append(candidate)
            seen.add(candidate)
    padded = np.full((budget, MAX_HORIZON), PAD_ACTION, dtype=np.uint8)
    for index, proposal in enumerate(proposals):
        padded[index, :horizon] = proposal
    return padded


def _sequence_digest(sequence: np.ndarray) -> str:
    valid = np.asarray(sequence, dtype=np.uint8)
    return hashlib.sha256(valid.tobytes()).hexdigest()


def state_features(scenario: CorridorScenario, state: CorridorState) -> np.ndarray:
    """Serialize only public numeric state and visible task geometry (52 values)."""
    spec = scenario.spec
    values = [
        state.x / spec.width,
        state.y / spec.height,
        state.vx / spec.max_speed,
        state.vy / spec.max_speed,
        spec.width,
        spec.height,
        spec.goal[0] / spec.width,
        spec.goal[1] / spec.height,
        spec.goal_radius,
        spec.safe_speed / spec.max_speed,
        spec.agent_radius,
        spec.acceleration / spec.max_speed,
        spec.damping,
        spec.max_speed,
        spec.dt,
        max(0, spec.max_steps - state.elapsed_steps) / spec.max_steps,
    ]
    if len(spec.obstacles) != 9:
        raise ValueError("Phase 12 requires the frozen nine-segment corridor geometry")
    for obstacle in spec.obstacles:
        values.extend((
            obstacle.x_min / spec.width,
            obstacle.y_min / spec.height,
            obstacle.x_max / spec.width,
            obstacle.y_max / spec.height,
        ))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (52,) or not np.isfinite(result).all():
        raise ValueError("public state feature contract violated")
    return result


def candidate_features(sequences: np.ndarray) -> np.ndarray:
    """One-hot action/PAD feature rows; sequence time is meaningful, row order is not."""
    sequence_array = np.asarray(sequences)
    if sequence_array.ndim != 2 or sequence_array.shape[1] != MAX_HORIZON:
        raise ValueError("sequences must have shape [candidate_count, 8]")
    if np.any(sequence_array > PAD_ACTION):
        raise ValueError("candidate action ID outside action/PAD vocabulary")
    encoded = np.eye(10, dtype=np.float32)[sequence_array.astype(np.int64)]
    return encoded.reshape(sequence_array.shape[0], 80)


def evaluate_candidate_set(
    state: CorridorState,
    scenario: CorridorScenario,
    sequences: np.ndarray,
) -> TeacherQuery:
    """Exact-roll out a full proposal set and retain all decision-relevant labels."""
    sequences = np.asarray(sequences, dtype=np.uint8)
    if sequences.ndim != 2 or sequences.shape[1] != MAX_HORIZON or len(sequences) == 0:
        raise ValueError("sequences must be a nonempty [N,8] array")
    horizon = min(MAX_HORIZON, scenario.spec.max_steps - state.elapsed_steps)
    trajectories = np.zeros((len(sequences), MAX_HORIZON + 1, 4), dtype=np.float32)
    masks = np.zeros((len(sequences), MAX_HORIZON + 1), dtype=np.bool_)
    outcomes = np.empty(len(sequences), dtype=np.uint8)
    metrics = np.empty(len(sequences), dtype=np.float32)
    efforts = np.empty(len(sequences), dtype=np.float32)
    keys: list[tuple[int, float, float]] = []
    any_success = False
    for index, sequence in enumerate(sequences):
        if np.any(sequence[:horizon] >= len(ACTION_VECTORS)):
            raise ValueError("valid candidate prefix contains PAD or invalid action")
        if np.any(sequence[horizon:] != PAD_ACTION):
            raise ValueError("candidate padding must start exactly after the query horizon")
        current = state
        trajectories[index, 0] = (current.x, current.y, current.vx, current.vy)
        masks[index, 0] = True
        effort = 0.0
        for step_index, action_id in enumerate(sequence[:horizon], start=1):
            action = int(action_id)
            effort += float(action != 0)
            transition = corridor_step(current, action, scenario.spec)
            current = transition.state
            trajectories[index, step_index] = (current.x, current.y, current.vx, current.vy)
            masks[index, step_index] = True
            any_success |= transition.success
            if current.status != "active":
                break
        key = _shooting_objective(state, current, effort, scenario.spec.goal)
        keys.append(key)
        outcomes[index] = OUTCOME_CODES[current.status]
        metrics[index] = key[1]
        efforts[index] = effort

    # Average 1-based ranks for exact lexicographic ties. No scalarized costs.
    ranks = np.empty(len(keys), dtype=np.float32)
    for index, key in enumerate(keys):
        strictly_better = sum(other < key for other in keys)
        tied = sum(other == key for other in keys)
        ranks[index] = strictly_better + (tied + 1) / 2.0
    best_rank = float(np.min(ranks))
    regrets = (ranks - best_rank) / max(1, len(keys) - 1)
    best_key = min(keys)
    optimal = np.asarray([key == best_key for key in keys], dtype=np.bool_)
    # Historical exact planner uses stable proposal order for exact ties.
    selected_index = next(index for index, key in enumerate(keys) if key == best_key)
    hashes = tuple(_sequence_digest(sequence) for sequence in sequences)
    if len(set(hashes)) != len(hashes):
        raise ValueError("candidate proposal bank unexpectedly contains duplicates")
    return TeacherQuery(
        sequences.copy(), hashes, trajectories, masks, outcomes, metrics, efforts,
        any_success, selected_index, tuple(keys), ranks, regrets.astype(np.float32), optimal,
    )


def make_teacher_query(scenario: CorridorScenario, state: CorridorState) -> TeacherQuery:
    sequences = propose_sequences(
        state, scenario.spec.max_steps,
        seed=phase12_query_seed(scenario.seed, state.elapsed_steps),
    )
    return evaluate_candidate_set(state, scenario, sequences)


def audit_query_parity(scenario: CorridorScenario, state: CorridorState, *, seed: int) -> dict[str, Any]:
    """Compare an independently serialized query to the frozen Phase-11 planner."""
    sequences = propose_sequences(state, scenario.spec.max_steps, seed=seed)
    query = evaluate_candidate_set(state, scenario, sequences)
    reference = exact_dynamics_shooting(
        state, scenario.spec, horizon=MAX_HORIZON, num_sequences=NUM_CANDIDATES, seed=seed,
    )
    # The reference result exposes the chosen sequence rather than its index.
    reference_sequence = np.asarray(reference.action_sequence, dtype=np.uint8)
    matching = np.flatnonzero(
        np.all(sequences[:, :len(reference_sequence)] == reference_sequence, axis=1)
        & np.all(sequences[:, len(reference_sequence):] == PAD_ACTION, axis=1)
    )
    selected = int(matching[0]) if len(matching) else -1
    return {
        "selected_sequence_match": selected >= 0 and tuple(sequences[selected, :len(reference.action_sequence)]) == reference.action_sequence,
        "selected_objective_match": selected >= 0 and query.objective_keys[selected] == reference.objective_key,
        "availability_match": query.success_available == reference.candidate_success_available,
        "candidate_count_match": len(sequences) == reference.evaluated_sequences,
        "selected_index": selected,
        "reference_first_action": reference.first_action,
        "reference_selected_sequence": reference.action_sequence,
        "query_selected_sequence": tuple(int(v) for v in sequences[query.selected_index] if v != PAD_ACTION),
        "reference_objective": reference.objective_key,
        "query_objective": query.objective_keys[selected] if selected >= 0 else None,
        "candidate_success_available": query.success_available,
        "evaluated_candidates": len(sequences),
    }
