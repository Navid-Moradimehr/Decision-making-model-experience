"""Post-hoc Phase 12 diagnostics for executable first-action equivalence.

These metrics do not alter the frozen Phase 12 sequence-regret gate. They ask
whether sequence-level regret reflects the action actually executed before the
planner replans.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_first_action_decision(
    sequences: np.ndarray,
    candidate_mask: np.ndarray,
    normalized_regret: np.ndarray,
    optimal_mask: np.ndarray,
    objective_keys: np.ndarray,
    teacher_selected_index: np.ndarray,
    student_selected_index: np.ndarray,
) -> dict[str, Any]:
    """Rank the selected first action among actions' best sampled plan keys.

    All sequences sharing their first action are execution-equivalent at the
    current decision: the environment receives that first action and the
    planner replans. The action-equivalent regret is the best available
    lexicographic planner key in the selected first-action group, ranked
    against the best key for each other represented first action.
    """
    sequences = np.asarray(sequences)
    mask = np.asarray(candidate_mask, dtype=bool)
    regret = np.asarray(normalized_regret, dtype=np.float64)
    optimal = np.asarray(optimal_mask, dtype=bool)
    keys = np.asarray(objective_keys, dtype=np.float64)
    teacher = np.asarray(teacher_selected_index, dtype=np.int64)
    student = np.asarray(student_selected_index, dtype=np.int64)
    n, count = mask.shape
    if (sequences.shape != (n, count, 8) or regret.shape != (n, count)
            or optimal.shape != (n, count) or keys.shape != (n, count, 3)):
        raise ValueError("candidate arrays have incompatible Phase 12 shapes")
    if teacher.shape != (n,) or student.shape != (n,):
        raise ValueError("selected-index arrays must have shape [query]")
    if np.any(teacher < 0) or np.any(student < 0) or np.any(teacher >= count) or np.any(student >= count):
        raise ValueError("selected candidate index out of range")
    if np.any(~mask[np.arange(n), teacher]) or np.any(~mask[np.arange(n), student]):
        raise ValueError("selected candidate must be valid")

    rows = np.arange(n)
    teacher_action = sequences[rows, teacher, 0]
    student_action = sequences[rows, student, 0]
    sequence_regret = regret[rows, student]
    action_regret = np.empty(n, dtype=np.float64)
    student_action_is_optimal = np.zeros(n, dtype=bool)
    teacher_action_is_optimal = np.zeros(n, dtype=bool)
    action_group_counts = np.empty(n, dtype=np.int16)
    distinct_optimal_actions = np.empty(n, dtype=np.int16)

    for row in range(n):
        valid = np.flatnonzero(mask[row])
        first = sequences[row, :, 0]
        action_values = sorted(set(int(x) for x in first[valid]))
        grouped_keys = {
            action: min(tuple(float(v) for v in keys[row, index])
                        for index in valid if int(first[index]) == action)
            for action in action_values
        }
        if int(student_action[row]) not in grouped_keys:
            raise ValueError("selected action has no valid candidate sequence")
        selected_key = grouped_keys[int(student_action[row])]
        strictly_better = sum(other < selected_key for other in grouped_keys.values())
        tied = sum(other == selected_key for other in grouped_keys.values())
        selected_rank = strictly_better + (tied + 1) / 2.0
        best_key = min(grouped_keys.values())
        best_rank = min(
            sum(other < value for other in grouped_keys.values())
            + (sum(other == value for other in grouped_keys.values()) + 1) / 2.0
            for value in grouped_keys.values()
        )
        action_regret[row] = (selected_rank - best_rank) / max(1, len(grouped_keys) - 1)
        best_actions = {action for action, value in grouped_keys.items() if value == best_key}
        student_action_is_optimal[row] = int(student_action[row]) in best_actions
        teacher_action_is_optimal[row] = int(teacher_action[row]) in best_actions
        action_group_counts[row] = sum(int(first[index]) == int(student_action[row]) for index in valid)
        distinct_optimal_actions[row] = len(best_actions)

    sequence_mean = float(np.mean(sequence_regret))
    action_mean = float(np.mean(action_regret))
    return {
        "queries": int(n),
        "student_first_action_matches_teacher": float(np.mean(student_action == teacher_action)),
        "student_first_action_in_optimal_action_set": float(np.mean(student_action_is_optimal)),
        "teacher_selected_action_in_optimal_action_set": float(np.mean(teacher_action_is_optimal)),
        "selected_sequence_normalized_rank_regret_sample_mean": sequence_mean,
        "teacher_action_group_normalized_regret_sample_mean": action_mean,
        "median_candidate_sequences_per_student_first_action": float(np.median(action_group_counts)),
        "mean_distinct_optimal_first_actions": float(np.mean(distinct_optimal_actions)),
        "fraction_queries_with_multiple_optimal_first_actions": float(np.mean(distinct_optimal_actions > 1)),
        "per_query_sequence_regret": sequence_regret,
        "per_query_teacher_action_group_regret": action_regret,
        "per_query_student_first_action": student_action,
        "per_query_teacher_first_action": teacher_action,
    }
