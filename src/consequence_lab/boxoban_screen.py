"""Evaluation helpers for the frozen Boxoban validation qualification screen."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Callable, Sequence

from .boxoban import (
    ACTION_ORDER,
    BoxobanBoard,
    BoxobanState,
    _static_corner_deadlock,
    is_solved,
    transition,
)


ActionPolicy = Callable[[BoxobanBoard, BoxobanState], str | None]


@dataclass(frozen=True)
class EpisodeResult:
    solved: bool
    actions: int
    pushes: int
    deadlock_seen: bool
    stop_reason: str


def sample_level_ids(
    *,
    split: str,
    shard_count: int,
    levels_per_shard: int,
    count: int,
    prefix: str,
    exclude_level_ids: Sequence[str] = (),
) -> list[str]:
    """Select IDs using only the frozen complete ID universe, never level text."""
    excluded = set(exclude_level_ids)
    available_count = shard_count * levels_per_shard - len(excluded)
    if shard_count <= 0 or levels_per_shard <= 0 or not 0 < count <= available_count:
        raise ValueError("invalid level universe or selection count")
    ids = (
        f"{split}/{shard:03d}.txt#{level_index:04d}"
        for shard in range(shard_count)
        for level_index in range(levels_per_shard)
        if f"{split}/{shard:03d}.txt#{level_index:04d}" not in excluded
    )
    return sorted(
        ids,
        key=lambda level_id: (
            hashlib.sha256(f"{prefix}{level_id}".encode("ascii")).hexdigest(),
            level_id,
        ),
    )[:count]


def deterministic_level_seed(protocol: str, level_id: str) -> int:
    digest = hashlib.sha256(f"{protocol}:{level_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def random_action_policy(seed: int) -> ActionPolicy:
    rng = random.Random(seed)

    def choose(_board: BoxobanBoard, _state: BoxobanState) -> str:
        return rng.choice(ACTION_ORDER)

    return choose


def play_episode(
    board: BoxobanBoard,
    initial_state: BoxobanState,
    policy: ActionPolicy,
    *,
    action_limit: int,
) -> EpisodeResult:
    if action_limit <= 0:
        raise ValueError("action_limit must be positive")
    state = initial_state
    pushes = 0
    deadlock_seen = any(_static_corner_deadlock(board, box) for box in state.boxes)
    for action_number in range(action_limit + 1):
        if is_solved(board, state):
            return EpisodeResult(True, action_number, pushes, deadlock_seen, "solved")
        if action_number == action_limit:
            break
        action = policy(board, state)
        if action is None:
            return EpisodeResult(False, action_number, pushes, deadlock_seen, "no_action")
        if action not in ACTION_ORDER:
            raise ValueError(f"policy returned invalid action {action!r}")
        successor = transition(board, state, action)
        if successor.boxes != state.boxes:
            pushes += 1
        state = successor
        deadlock_seen = deadlock_seen or any(
            _static_corner_deadlock(board, box) for box in state.boxes
        )
    return EpisodeResult(False, action_limit, pushes, deadlock_seen, "action_limit")


def paired_bootstrap_mean_interval(
    paired_differences: Sequence[float], *, replicates: int, seed: int
) -> tuple[float, float, float]:
    """Percentile 95% bootstrap for paired level-level differences."""
    if not paired_differences or replicates <= 0:
        raise ValueError("non-empty paired data and positive replicates are required")
    rng = random.Random(seed)
    n = len(paired_differences)
    point = sum(paired_differences) / n
    means = []
    for _ in range(replicates):
        means.append(
            sum(paired_differences[rng.randrange(n)] for _ in range(n)) / n
        )
    means.sort()
    lower = means[int(0.025 * (replicates - 1))]
    upper = means[int(0.975 * (replicates - 1))]
    return point, lower, upper
