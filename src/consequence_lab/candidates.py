"""Deterministic finite candidate sequences for the TwoRoom pilot."""

from __future__ import annotations

from itertools import product

import numpy as np


def primitive_actions() -> np.ndarray:
    """Return eight unit directions and a zero-velocity wait action."""
    directions = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )
    directions[:8] /= np.linalg.norm(directions[:8], axis=1, keepdims=True)
    return np.concatenate([directions, np.zeros((1, 2), np.float32)], axis=0)


def two_segment_sequences(horizon: int) -> np.ndarray:
    """Return 64 two-direction sequences and one wait sequence.

    The first and second directions each occupy half the requested horizon.
    This covers straight motions, turns, and reversals with a fixed candidate
    budget and stable row order.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 2:
        raise ValueError("horizon must be an integer of at least two steps")

    actions = primitive_actions()
    split = horizon // 2
    remainder = horizon - split
    pairs = [(direction, direction) for direction in range(8)]
    pairs.extend(
        (first, second)
        for first, second in product(range(8), repeat=2)
        if first != second
    )
    sequences = []
    for first, second in pairs:
        sequences.append(
            np.concatenate(
                [
                    np.repeat(actions[first][None, :], split, axis=0),
                    np.repeat(actions[second][None, :], remainder, axis=0),
                ],
                axis=0,
            )
        )
    sequences.append(np.zeros((horizon, 2), dtype=np.float32))
    return np.stack(sequences)


def candidate_subset(sequences: np.ndarray, count: int) -> np.ndarray:
    """Take a deterministic spread subset while preserving wait and straights."""
    if sequences.ndim != 3 or sequences.shape[1] < 1 or sequences.shape[2] != 2:
        raise ValueError("sequences must have shape [candidate, horizon, 2]")
    if count < 9 or count > len(sequences):
        raise ValueError("count must be between 9 and the available candidates")

    # The generator's first eight rows are straight motions in every direction.
    required = list(range(8)) + [len(sequences) - 1]
    remaining = [i for i in range(len(sequences)) if i not in required]
    slots = count - len(required)
    if slots:
        positions = np.linspace(0, len(remaining) - 1, slots, dtype=int)
        required.extend(remaining[i] for i in positions)
    return sequences[np.asarray(required, dtype=int)]
