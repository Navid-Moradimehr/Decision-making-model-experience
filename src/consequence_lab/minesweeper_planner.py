"""Shallow exact belief-space planning for a finite Minesweeper candidate set."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .minesweeper import (
    Cell,
    Posterior,
    candidate_cells,
    clue_for_layout,
    posterior_layouts,
    posterior_mine_risks,
    reveal_public_cell,
)


def _one_step_value(
    public_board: np.ndarray,
    posterior: Posterior,
    *,
    mine_count: int,
    success_reward: float,
    failure_reward: float,
) -> float:
    risks = posterior_mine_risks(public_board, posterior)
    if not risks:
        return 0.0
    return max(
        (1.0 - risks[cell]) * success_reward + risks[cell] * failure_reward
        for cell in risks
    )


def two_ply_action_values(
    public_board: np.ndarray,
    posterior: Posterior,
    *,
    mine_count: int,
    success_reward: float,
    failure_reward: float,
) -> dict[Cell, float]:
    """Expected undiscounted reward for this action plus one observation-aware
    greedy action, exactly enumerating posterior layouts and clue outcomes.

    Replanning happens at each real step. If the first safe reveal makes all
    remaining covered cells mines, the environment has already terminated and
    no second action is counted.
    """
    board = np.asarray(public_board)
    rows, cols = board.shape
    candidates = candidate_cells(board)
    n_layouts = posterior.count
    if n_layouts == 0:
        raise ValueError("posterior must be non-empty")
    values: dict[Cell, float] = {}
    for first in candidates:
        layout_sets = [set(layout) for layout in posterior.layouts]
        first_mines = sum(first in layout for layout in layout_sets)
        total = (first_mines / n_layouts) * failure_reward
        safe_layouts = [
            layout for layout, layout_set in zip(posterior.layouts, layout_sets)
            if first not in layout_set
        ]
        if not safe_layouts:
            values[first] = total
            continue
        clue_groups: dict[int, list[tuple[Cell, ...]]] = defaultdict(list)
        for layout in safe_layouts:
            clue_groups[clue_for_layout(first, layout, rows, cols)].append(layout)
        for clue, group in clue_groups.items():
            group_probability = len(group) / n_layouts
            first_safe_reward = group_probability * success_reward
            child_board = reveal_public_cell(board, first, clue)
            child_candidates = candidate_cells(child_board)
            if not child_candidates or len(child_candidates) <= mine_count:
                total += first_safe_reward
                continue
            child_posterior = Posterior(tuple(group))
            continuation = _one_step_value(
                child_board,
                child_posterior,
                mine_count=mine_count,
                success_reward=success_reward,
                failure_reward=failure_reward,
            )
            total += first_safe_reward + group_probability * continuation
        values[first] = total
    return values


def choose_two_ply_action(
    public_board: np.ndarray,
    posterior: Posterior,
    *,
    mine_count: int,
    success_reward: float,
    failure_reward: float,
) -> Cell | None:
    values = two_ply_action_values(
        public_board,
        posterior,
        mine_count=mine_count,
        success_reward=success_reward,
        failure_reward=failure_reward,
    )
    if not values:
        return None
    return min(values, key=lambda cell: (-values[cell], cell))
