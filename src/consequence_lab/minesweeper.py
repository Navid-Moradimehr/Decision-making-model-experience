"""Information-limited Minesweeper baselines for a set-valued action screen.

Decision functions consume only a public board view and dynamically supplied
unrevealed cells. Hidden mine layouts are enumerated as a known prior, never
read from the simulator state.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Sequence

import numpy as np


Cell = tuple[int, int]


@dataclass(frozen=True)
class Posterior:
    layouts: tuple[tuple[Cell, ...], ...]

    @property
    def count(self) -> int:
        return len(self.layouts)


def all_mine_layouts(rows: int, cols: int, mine_count: int) -> tuple[tuple[Cell, ...], ...]:
    if rows <= 0 or cols <= 0 or not 0 <= mine_count <= rows * cols:
        raise ValueError("invalid board or mine count")
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    return tuple(tuple(layout) for layout in combinations(cells, mine_count))


def adjacent_cells(cell: Cell, rows: int, cols: int) -> tuple[Cell, ...]:
    row, col = cell
    return tuple(
        (r, c)
        for r in range(max(0, row - 1), min(rows, row + 2))
        for c in range(max(0, col - 1), min(cols, col + 2))
        if (r, c) != cell
    )


def clue_for_layout(cell: Cell, mine_layout: Sequence[Cell], rows: int, cols: int) -> int:
    mines = set(mine_layout)
    return sum(neighbor in mines for neighbor in adjacent_cells(cell, rows, cols))


def posterior_layouts(
    public_board: np.ndarray,
    *,
    mine_count: int,
    layout_universe: Sequence[Sequence[Cell]] | None = None,
) -> Posterior:
    """Filter mine layouts using revealed clues; unrevealed cells use -1."""
    board = np.asarray(public_board)
    if board.ndim != 2 or not np.issubdtype(board.dtype, np.integer):
        raise ValueError("public_board must be a 2-D integer array")
    rows, cols = board.shape
    if np.any((board < -1) | (board > 8)):
        raise ValueError("public clues must be -1 (hidden) or in [0, 8]")
    universe = layout_universe or all_mine_layouts(rows, cols, mine_count)
    revealed = [(int(r), int(c), int(board[r, c])) for r, c in np.argwhere(board >= 0)]
    possible = []
    for raw_layout in universe:
        layout = tuple((int(r), int(c)) for r, c in raw_layout)
        mine_set = set(layout)
        if any((r, c) in mine_set for r, c, _ in revealed):
            continue
        if all(clue_for_layout((r, c), layout, rows, cols) == clue for r, c, clue in revealed):
            possible.append(layout)
    if not possible:
        raise ValueError("public board is inconsistent with the supplied prior")
    return Posterior(tuple(possible))


def candidate_cells(public_board: np.ndarray) -> tuple[Cell, ...]:
    """Return the current valid action set in stable row-major order."""
    board = np.asarray(public_board)
    if board.ndim != 2:
        raise ValueError("public_board must be 2-D")
    return tuple((int(r), int(c)) for r, c in np.argwhere(board == -1))


def posterior_mine_risks(public_board: np.ndarray, posterior: Posterior) -> dict[Cell, float]:
    candidates = candidate_cells(public_board)
    if not candidates:
        return {}
    if posterior.count <= 0:
        raise ValueError("posterior must contain at least one layout")
    layout_sets = [set(layout) for layout in posterior.layouts]
    return {
        cell: sum(cell in layout for layout in layout_sets) / posterior.count
        for cell in candidates
    }


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


def expected_safe_information_gain(
    cell: Cell, posterior: Posterior, rows: int, cols: int
) -> float:
    """Expected posterior-entropy reduction, conditional on this cell being safe."""
    safe_layouts = [layout for layout in posterior.layouts if cell not in set(layout)]
    if not safe_layouts:
        return 0.0
    base_entropy = math.log2(len(safe_layouts))
    clue_counts: dict[int, int] = {}
    for layout in safe_layouts:
        clue = clue_for_layout(cell, layout, rows, cols)
        clue_counts[clue] = clue_counts.get(clue, 0) + 1
    expected_remaining_entropy = sum(
        (count / len(safe_layouts)) * math.log2(count)
        for count in clue_counts.values()
    )
    return base_entropy - expected_remaining_entropy


def choose_min_risk(public_board: np.ndarray, posterior: Posterior) -> Cell | None:
    risks = posterior_mine_risks(public_board, posterior)
    if not risks:
        return None
    return min(risks, key=lambda cell: (risks[cell], cell))


def choose_risk_then_information(public_board: np.ndarray, posterior: Posterior) -> Cell | None:
    risks = posterior_mine_risks(public_board, posterior)
    if not risks:
        return None
    rows, cols = np.asarray(public_board).shape
    gains = {
        cell: expected_safe_information_gain(cell, posterior, rows, cols)
        for cell in risks
    }
    return min(risks, key=lambda cell: (risks[cell], -gains[cell], cell))


def reveal_public_cell(public_board: np.ndarray, cell: Cell, clue: int) -> np.ndarray:
    """Pure update from the chosen public candidate and returned observation."""
    board = np.asarray(public_board).copy()
    if cell not in candidate_cells(board):
        raise ValueError("chosen cell is not a current valid candidate")
    if not 0 <= int(clue) <= 8:
        raise ValueError("safe-cell clue must be in [0, 8]")
    board[cell] = int(clue)
    return board
