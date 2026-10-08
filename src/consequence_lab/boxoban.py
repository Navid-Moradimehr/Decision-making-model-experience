"""Small exact symbolic Boxoban mechanics and optimal primitive-action search.

This module intentionally has no Gymnasium, rendering, or Torch dependency.
The A* solver is an exact reference teacher: it uses an admissible assignment
heuristic, reopens improved states, and only prunes provable static corner
deadlocks. A resource limit is reported as ``limit``, never as unsolvable.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from heapq import heappop, heappush
from itertools import permutations
import math
import re
import time
from typing import Iterable, Sequence


ACTION_DELTAS: dict[str, tuple[int, int]] = {
    "up": (0, -1),
    "right": (1, 0),
    "down": (0, 1),
    "left": (-1, 0),
}
ACTION_ORDER: tuple[str, ...] = tuple(ACTION_DELTAS)
_LEVEL_HEADER = re.compile(r"^\s*;\s*(\d+)\s*$")
_LEVEL_SYMBOLS = frozenset("# @$.+*")


@dataclass(frozen=True)
class BoxobanState:
    """Dynamic state. Box positions are canonicalized in row-major order."""

    player: tuple[int, int]
    boxes: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BoxobanBoard:
    """Static rectangular board; coordinates are ``(x, y)``."""

    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    goals: frozenset[tuple[int, int]]

    def in_bounds(self, point: tuple[int, int]) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, point: tuple[int, int]) -> bool:
        return not self.in_bounds(point) or point in self.walls

    def validate_state(self, state: BoxobanState) -> None:
        if len(state.boxes) != len(self.goals):
            raise ValueError("the number of boxes must equal the number of goals")
        if tuple(sorted(state.boxes, key=_row_major)) != state.boxes:
            raise ValueError("box positions must be in canonical row-major order")
        occupied = (state.player, *state.boxes)
        if len(set(occupied)) != len(occupied):
            raise ValueError("player and boxes must occupy distinct cells")
        for point in occupied:
            if self.is_wall(point):
                raise ValueError(f"occupant is outside the board or inside a wall: {point}")


@dataclass(frozen=True)
class ParsedBoxobanLevel:
    level_index: int
    board: BoxobanBoard
    initial_state: BoxobanState


@dataclass(frozen=True)
class SearchResult:
    status: str
    actions: tuple[str, ...]
    expanded_states: int
    generated_states: int

    @property
    def solved(self) -> bool:
        return self.status == "solved"

    @property
    def proof_complete(self) -> bool:
        return self.status in {"solved", "proven_unsolvable"}

    @property
    def primitive_action_cost(self) -> int | None:
        return len(self.actions) if self.solved else None


def _row_major(point: tuple[int, int]) -> tuple[int, int]:
    return point[1], point[0]


def _canonical_boxes(points: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(points, key=_row_major))


def parse_boxoban_rows(rows: Sequence[str], level_index: int = 0) -> ParsedBoxobanLevel:
    """Parse one ASCII board without stripping meaningful spaces from rows."""
    if not rows or any(not isinstance(row, str) for row in rows):
        raise ValueError("rows must be a non-empty sequence of strings")
    width = len(rows[0])
    if width < 3 or any(len(row) != width for row in rows):
        raise ValueError("board rows must have one common width of at least three")
    if len(rows) < 3:
        raise ValueError("board must have at least three rows")

    walls: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    boxes: list[tuple[int, int]] = []
    players: list[tuple[int, int]] = []
    for y, row in enumerate(rows):
        for x, symbol in enumerate(row):
            if symbol not in _LEVEL_SYMBOLS:
                raise ValueError(f"unsupported level symbol {symbol!r} at {(x, y)}")
            point = (x, y)
            if symbol == "#":
                walls.add(point)
            elif symbol == ".":
                goals.add(point)
            elif symbol == "$":
                boxes.append(point)
            elif symbol == "@":
                players.append(point)
            elif symbol == "+":
                goals.add(point)
                players.append(point)
            elif symbol == "*":
                goals.add(point)
                boxes.append(point)

    if len(players) != 1:
        raise ValueError(f"expected exactly one player, found {len(players)}")
    if not goals or len(boxes) != len(goals):
        raise ValueError(
            f"box/goal counts must be equal and non-zero; found {len(boxes)}/{len(goals)}"
        )
    board = BoxobanBoard(
        width=width,
        height=len(rows),
        walls=frozenset(walls),
        goals=frozenset(goals),
    )
    state = BoxobanState(players[0], _canonical_boxes(boxes))
    board.validate_state(state)
    return ParsedBoxobanLevel(int(level_index), board, state)


def parse_boxoban_shard(text: str) -> list[ParsedBoxobanLevel]:
    """Parse the upstream ``; index`` separated, 10-row shard format."""
    lines = text.splitlines()
    result: list[ParsedBoxobanLevel] = []
    cursor = 0
    while cursor < len(lines):
        header = _LEVEL_HEADER.fullmatch(lines[cursor])
        if header is None:
            if lines[cursor] == "":
                cursor += 1
                continue
            raise ValueError(f"expected a Boxoban level header at line {cursor + 1}")
        level_index = int(header.group(1))
        cursor += 1
        while cursor < len(lines) and lines[cursor] == "":
            cursor += 1
        if cursor + 10 > len(lines):
            raise ValueError(f"level {level_index} has fewer than ten rows")
        rows = lines[cursor : cursor + 10]
        if any(_LEVEL_HEADER.fullmatch(row) for row in rows):
            raise ValueError(f"level {level_index} has fewer than ten rows")
        if any(len(row) != 10 for row in rows):
            lengths = [len(row) for row in rows]
            raise ValueError(f"level {level_index} rows must all have width 10: {lengths}")
        result.append(parse_boxoban_rows(rows, level_index))
        cursor += 10
    if not result:
        raise ValueError("shard contains no Boxoban levels")
    indices = [level.level_index for level in result]
    if len(set(indices)) != len(indices):
        raise ValueError("shard contains duplicate level indices")
    return result


def transition(
    board: BoxobanBoard, state: BoxobanState, action: str
) -> BoxobanState:
    """Apply one cardinal action; wall/blocked pushes are no-ops."""
    board.validate_state(state)
    if action not in ACTION_DELTAS:
        raise ValueError(f"unknown action {action!r}; expected one of {ACTION_ORDER}")
    dx, dy = ACTION_DELTAS[action]
    target = (state.player[0] + dx, state.player[1] + dy)
    if board.is_wall(target):
        return state

    box_set = set(state.boxes)
    if target not in box_set:
        return BoxobanState(target, state.boxes)

    beyond = (target[0] + dx, target[1] + dy)
    if board.is_wall(beyond) or beyond in box_set:
        return state
    box_set.remove(target)
    box_set.add(beyond)
    next_state = BoxobanState(target, _canonical_boxes(box_set))
    board.validate_state(next_state)
    return next_state


def is_solved(board: BoxobanBoard, state: BoxobanState) -> bool:
    board.validate_state(state)
    return set(state.boxes) == set(board.goals)


def _assignment_heuristic(board: BoxobanBoard, boxes: tuple[tuple[int, int], ...]) -> int:
    """Admissible lower bound: minimum box-goal Manhattan assignment pushes."""
    goals = tuple(sorted(board.goals, key=_row_major))
    best = math.inf
    for assignment in permutations(goals):
        total = sum(
            abs(box[0] - goal[0]) + abs(box[1] - goal[1])
            for box, goal in zip(boxes, assignment)
        )
        best = min(best, total)
    return int(best)


def _static_corner_deadlock(board: BoxobanBoard, point: tuple[int, int]) -> bool:
    if point in board.goals:
        return False
    x, y = point
    vertical = board.is_wall((x - 1, y)) or board.is_wall((x + 1, y))
    horizontal = board.is_wall((x, y - 1)) or board.is_wall((x, y + 1))
    return vertical and horizontal


def _static_box_goal_distances(
    board: BoxobanBoard,
) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    """Minimum pushes from each floor cell to each goal, ignoring other boxes.

    Reverse-push BFS respects walls on both the box destination and the player
    stance cell. It is a static heuristic only; it performs no push search.
    """
    distances: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    for goal in board.goals:
        per_goal = {goal: 0}
        queue = deque([goal])
        while queue:
            current = queue.popleft()
            for dx, dy in ACTION_DELTAS.values():
                previous_box = (current[0] - dx, current[1] - dy)
                player_stance = (previous_box[0] - dx, previous_box[1] - dy)
                if board.is_wall(previous_box) or board.is_wall(player_stance):
                    continue
                if previous_box in per_goal:
                    continue
                per_goal[previous_box] = per_goal[current] + 1
                queue.append(previous_box)
        distances[goal] = per_goal
    return distances


def _static_assignment_cost(
    boxes: tuple[tuple[int, int], ...],
    goals: tuple[tuple[int, int], ...],
    distances: dict[tuple[int, int], dict[tuple[int, int], int]],
) -> int | None:
    """Minimum wall-aware box-goal assignment cost, or None if impossible."""
    best = math.inf
    for assignment in permutations(goals):
        terms = [distances[goal].get(box) for box, goal in zip(boxes, assignment)]
        if any(value is None for value in terms):
            continue
        best = min(best, sum(terms))
    return None if math.isinf(best) else int(best)


def _reachable_player_paths(
    board: BoxobanBoard, state: BoxobanState
) -> dict[tuple[int, int], tuple[str, ...]]:
    """BFS paths through currently box-free cells in stable action order."""
    box_set = set(state.boxes)
    paths = {state.player: ()}
    queue = deque([state.player])
    while queue:
        point = queue.popleft()
        for action, (dx, dy) in ACTION_DELTAS.items():
            next_point = (point[0] + dx, point[1] + dy)
            if board.is_wall(next_point) or next_point in box_set or next_point in paths:
                continue
            paths[next_point] = (*paths[point], action)
            queue.append(next_point)
    return paths


def greedy_push_action(board: BoxobanBoard, state: BoxobanState) -> str | None:
    """Choose the next primitive action under a deterministic greedy-push rule.

    Enumerate individually reachable pushes (no multi-push lookahead), reject
    only pushes into static non-goal corners, rank resulting box layouts by
    minimum reverse-push-distance assignment cost, then by walk distance and
    stable row-major/action-order tie breaks. If the chosen stance is not yet
    reached, return the first step on its shortest player path. ``None`` means
    no admissible push remains; the episode is a baseline failure.
    """
    board.validate_state(state)
    if is_solved(board, state):
        return None
    paths = _reachable_player_paths(board, state)
    distances = _static_box_goal_distances(board)
    goals = tuple(sorted(board.goals, key=_row_major))
    choices: list[tuple[tuple[int, int, int, int, int], tuple[str, ...], str]] = []
    box_set = set(state.boxes)
    action_index = {action: index for index, action in enumerate(ACTION_ORDER)}
    for box in state.boxes:
        for action, (dx, dy) in ACTION_DELTAS.items():
            stance = (box[0] - dx, box[1] - dy)
            target = (box[0] + dx, box[1] + dy)
            if stance not in paths or board.is_wall(target) or target in box_set:
                continue
            if _static_corner_deadlock(board, target):
                continue
            next_boxes = tuple(sorted((box_set - {box}) | {target}, key=_row_major))
            assignment_cost = _static_assignment_cost(next_boxes, goals, distances)
            if assignment_cost is None:
                continue
            rank = (
                assignment_cost,
                len(paths[stance]),
                box[1] * board.width + box[0],
                action_index[action],
                target[1] * board.width + target[0],
            )
            choices.append((rank, paths[stance], action))
    if not choices:
        return None
    _, walk_path, push_action = min(choices, key=lambda item: item[0])
    return walk_path[0] if walk_path else push_action


def greedy_goal_action(board: BoxobanBoard, state: BoxobanState) -> str:
    """A deliberately simple geometry-only diagnostic action policy.

    It greedily moves the player toward the nearest goal by Manhattan distance,
    considering the player position after each cardinal transition. It ignores
    box-goal assignments and future consequences. Stable ``ACTION_ORDER`` ties
    make it deterministic. If trapped, it returns the first cardinal action,
    which is a no-op under the environment contract.
    """
    board.validate_state(state)
    if is_solved(board, state):
        return ACTION_ORDER[0]
    goals = tuple(board.goals)
    scored = []
    for index, action in enumerate(ACTION_ORDER):
        successor = transition(board, state, action)
        distance = min(
            abs(successor.player[0] - goal[0]) + abs(successor.player[1] - goal[1])
            for goal in goals
        )
        scored.append((distance, index, action))
    return min(scored)[2]


def solve_optimal(
    board: BoxobanBoard,
    initial_state: BoxobanState,
    *,
    heuristic_mode: str = "manhattan_assignment",
    max_expanded_states: int | None = None,
    timeout_seconds: float | None = None,
) -> SearchResult:
    """Find a minimum primitive-action solution with optimal A* search.

    ``proven_unsolvable`` is returned only after exhausting the finite search
    graph. A node/time limit returns ``limit`` and is not a proof.
    """
    board.validate_state(initial_state)
    if heuristic_mode not in {"manhattan_assignment", "static_push_assignment"}:
        raise ValueError("heuristic_mode must be 'manhattan_assignment' or 'static_push_assignment'")
    if max_expanded_states is not None and max_expanded_states <= 0:
        raise ValueError("max_expanded_states must be positive when supplied")
    if timeout_seconds is not None and (
        not math.isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite when supplied")
    if is_solved(board, initial_state):
        return SearchResult("solved", (), 0, 1)

    deadline = None if timeout_seconds is None else time.perf_counter() + timeout_seconds
    g_score = {initial_state: 0}
    parent: dict[BoxobanState, tuple[BoxobanState, str]] = {}
    heuristic_cache: dict[tuple[tuple[int, int], ...], int] = {}
    static_distances = (
        _static_box_goal_distances(board)
        if heuristic_mode == "static_push_assignment"
        else None
    )
    ordered_goals = tuple(sorted(board.goals, key=_row_major))

    def heuristic(boxes: tuple[tuple[int, int], ...]) -> int:
        value = heuristic_cache.get(boxes)
        if value is None:
            if static_distances is None:
                value = _assignment_heuristic(board, boxes)
            else:
                value = _static_assignment_cost(boxes, ordered_goals, static_distances)
                if value is None:
                    value = math.inf
            heuristic_cache[boxes] = value
        return value

    initial_heuristic = heuristic(initial_state.boxes)
    if math.isinf(initial_heuristic):
        return SearchResult("proven_unsolvable", (), 0, 1)

    frontier: list[tuple[int, int, int, BoxobanState]] = []
    serial = 0
    heappush(
        frontier,
        (initial_heuristic, 0, serial, initial_state),
    )
    expanded = 0
    generated = 1

    while frontier:
        if deadline is not None and time.perf_counter() >= deadline:
            return SearchResult("limit", (), expanded, generated)
        _, cost, _, state = heappop(frontier)
        if cost != g_score.get(state):
            continue
        if is_solved(board, state):
            actions: list[str] = []
            cursor = state
            while cursor != initial_state:
                previous, action = parent[cursor]
                actions.append(action)
                cursor = previous
            actions.reverse()
            return SearchResult("solved", tuple(actions), expanded, generated)
        if max_expanded_states is not None and expanded >= max_expanded_states:
            return SearchResult("limit", (), expanded, generated)
        expanded += 1

        for action in ACTION_ORDER:
            next_state = transition(board, state, action)
            if next_state == state:
                continue
            if any(
                _static_corner_deadlock(board, box)
                for box in next_state.boxes
                if box not in state.boxes
            ):
                continue
            next_heuristic = heuristic(next_state.boxes)
            if math.isinf(next_heuristic):
                continue
            tentative_cost = cost + 1
            if tentative_cost >= g_score.get(next_state, math.inf):
                continue
            g_score[next_state] = tentative_cost
            parent[next_state] = (state, action)
            serial += 1
            priority = tentative_cost + next_heuristic
            heappush(frontier, (priority, tentative_cost, serial, next_state))
            generated += 1

    return SearchResult("proven_unsolvable", (), expanded, generated)
