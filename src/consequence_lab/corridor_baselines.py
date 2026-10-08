"""Geometry-informed baselines for the inertial-corridor task.

These algorithms are references for later qualification, not tuned policies or
evidence of planner advantage. Calling them does not execute task transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import acos, hypot, isfinite, pi, sqrt
from numbers import Integral
import random

from consequence_lab.inertial_corridor import (
    ACTION_VECTORS,
    CorridorState,
    CorridorTaskSpec,
    first_swept_collision_fraction,
    corridor_step,
    validate_corridor_state,
)


@dataclass(frozen=True)
class GeometricRoute:
    waypoints: tuple[tuple[float, float], ...]
    length: float
    expanded_nodes: int


@dataclass(frozen=True)
class KinodynamicSearchResult:
    status: str
    first_action: int | None
    expanded_nodes: int
    position_resolution: float
    velocity_resolution: float


@dataclass(frozen=True)
class ShootingResult:
    """One exact-model random-shooting decision and diagnostic summary."""

    first_action: int
    action_sequence: tuple[int, ...]
    objective_key: tuple[int, float, float]
    candidate_success_available: bool
    evaluated_sequences: int


def reactive_goal_action(
    state: CorridorState,
    spec: CorridorTaskSpec,
    *,
    obstacle_influence: float = 0.10,
    clearance_weight: float = 0.01,
) -> int:
    """Choose a one-step goal/braking action with local obstacle repulsion.

    This intentionally has no route memory or multi-step search. It chooses a
    braking-limited desired velocity toward the goal, then scores each legal
    action using exact one-step dynamics, progress, velocity alignment, and
    local clearance. Gains are exposed for later design-set tuning; this is
    not a qualified controller.
    """
    validate_corridor_state(state, spec)
    if state.status != "active":
        raise ValueError("reactive controller requires an active state")
    if not isfinite(obstacle_influence) or obstacle_influence < 0:
        raise ValueError("obstacle_influence must be finite and nonnegative")
    if not isfinite(clearance_weight) or clearance_weight < 0:
        raise ValueError("clearance_weight must be finite and nonnegative")

    dx, dy = spec.goal[0] - state.x, spec.goal[1] - state.y
    goal_distance = hypot(dx, dy)
    desired_speed = min(
        spec.max_speed,
        sqrt(2.0 * spec.acceleration * max(0.0, goal_distance - spec.goal_radius)),
    )
    if goal_distance > 0:
        desired_v = (desired_speed * dx / goal_distance, desired_speed * dy / goal_distance)
    else:
        desired_v = (0.0, 0.0)

    # Repel from the nearest point on each rectangle's radius-expanded shape.
    # The small denominator floor makes the cue finite near contact.
    repulsion_x = repulsion_y = 0.0
    radius = spec.agent_radius
    for obstacle in spec.obstacles:
        nearest_x = min(max(state.x, obstacle.x_min), obstacle.x_max)
        nearest_y = min(max(state.y, obstacle.y_min), obstacle.y_max)
        rx, ry = state.x - nearest_x, state.y - nearest_y
        distance = hypot(rx, ry)
        clearance = distance - radius
        if clearance < obstacle_influence:
            if distance <= 1e-12:
                # Active states cannot overlap an obstacle. This branch only
                # handles a center exactly on a rectangle face numerically.
                rx, ry, distance = -dx, -dy, max(goal_distance, 1e-12)
            weight = max(0.0, obstacle_influence - clearance) / max(distance, 1e-6)
            repulsion_x += weight * rx / distance
            repulsion_y += weight * ry / distance

    cue = (desired_v[0] + repulsion_x, desired_v[1] + repulsion_y)
    cue_norm = hypot(*cue)
    if cue_norm > spec.max_speed:
        cue = (cue[0] * spec.max_speed / cue_norm, cue[1] * spec.max_speed / cue_norm)

    best_action = 0
    best_score = float("inf")
    for action_id in range(len(ACTION_VECTORS)):
        result = corridor_step(state, action_id, spec)
        if result.collision:
            continue
        next_state = result.state
        next_distance = hypot(spec.goal[0] - next_state.x, spec.goal[1] - next_state.y)
        velocity_error = (next_state.vx - cue[0]) ** 2 + (next_state.vy - cue[1]) ** 2
        min_clearance = spec.width + spec.height
        for obstacle in spec.obstacles:
            px = min(max(next_state.x, obstacle.x_min), obstacle.x_max)
            py = min(max(next_state.y, obstacle.y_min), obstacle.y_max)
            min_clearance = min(min_clearance, hypot(next_state.x - px, next_state.y - py) - radius)
        clearance_penalty = 1.0 / max(min_clearance, 1e-4)
        # Distance progress is the primary signal; velocity tracking and
        # clearance are local tie-breakers. No tuning/performance claim implied.
        score = next_distance + 0.25 * velocity_error + clearance_weight * clearance_penalty
        if result.success:
            score -= 10.0
        if score < best_score:
            best_action, best_score = action_id, score
    return best_action


def _shooting_objective(
    initial_state: CorridorState,
    final_state: CorridorState,
    effort: float,
    goal: tuple[float, float],
) -> tuple[int, float, float]:
    """Transparent lexicographic objective for prototype comparisons only.

    Safe success ranks first (fewer steps, then less effort). Otherwise an
    active horizon endpoint ranks ahead of timeout, which ranks ahead of
    collision; within each class prefer smaller goal distance and effort.
    This is a declared prototype convention, not a selected qualification
    objective.
    """
    if final_state.status == "success":
        return (0, float(final_state.elapsed_steps - initial_state.elapsed_steps), effort)
    category = {"active": 1, "timeout": 2, "collision": 3}[final_state.status]
    return (category, hypot(goal[0] - final_state.x, goal[1] - final_state.y), effort)


def exact_dynamics_shooting(
    initial_state: CorridorState,
    spec: CorridorTaskSpec,
    *,
    horizon: int = 12,
    num_sequences: int = 256,
    seed: int = 0,
) -> ShootingResult:
    """Select a first action by seeded random shooting in exact task dynamics.

    The proposal bank contains constant-action sequences followed by seeded
    uniform sequences (deduplicated); all are evaluated with ``corridor_step``.
    This is a reference primitive, not a tuned MPC benchmark. In particular,
    proposal distribution, horizon, sequence budget, and lexicographic scoring
    require design-stage decisions before qualification episodes.
    """
    validate_corridor_state(initial_state, spec)
    if initial_state.status != "active":
        raise ValueError("shooting requires an active initial state")
    for name, value in (("horizon", horizon), ("num_sequences", num_sequences)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rollout_horizon = min(int(horizon), spec.max_steps - initial_state.elapsed_steps)
    budget = min(int(num_sequences), 9**rollout_horizon)
    proposals: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for action_id in range(len(ACTION_VECTORS)):
        candidate = (action_id,) * rollout_horizon
        if candidate not in seen:
            proposals.append(candidate)
            seen.add(candidate)
            if len(proposals) == budget:
                break
    rng = random.Random(int(seed))
    while len(proposals) < budget:
        candidate = tuple(rng.randrange(len(ACTION_VECTORS)) for _ in range(rollout_horizon))
        if candidate not in seen:
            proposals.append(candidate)
            seen.add(candidate)

    best_sequence: tuple[int, ...] | None = None
    best_key: tuple[int, float, float] | None = None
    any_success = False
    for sequence in proposals:
        current = initial_state
        effort = 0.0
        for action_id in sequence:
            direction = ACTION_VECTORS[action_id]
            norm = hypot(*direction)
            effort += 0.0 if norm == 0.0 else 1.0
            step = corridor_step(current, action_id, spec)
            current = step.state
            if step.success:
                any_success = True
                break
            if current.status in {"collision", "timeout"}:
                break
        objective_key = _shooting_objective(initial_state, current, effort, spec.goal)
        if best_key is None or objective_key < best_key:
            best_sequence, best_key = sequence, objective_key
    assert best_sequence is not None and best_key is not None
    return ShootingResult(
        first_action=best_sequence[0],
        action_sequence=best_sequence,
        objective_key=best_key,
        candidate_success_available=any_success,
        evaluated_sequences=len(proposals),
    )


def grid_astar_route(
    start: tuple[float, float],
    goal: tuple[float, float],
    spec: CorridorTaskSpec,
    resolution: float = 0.02,
) -> GeometricRoute | None:
    """Find a shortest collision-free polyline on a disclosed geometry grid.

    The nonuniform node lattice consists of regular resolution points plus the
    exact start and goal coordinates. It plans geometry only: no velocity,
    inertia, safe-goal speed, time, or action feasibility is represented.
    """
    if not isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    if len(start) != 2 or len(goal) != 2 or not all(
        isfinite(float(v)) for v in (*start, *goal)
    ):
        raise ValueError("start and goal must be finite two-vectors")
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    margin = spec.agent_radius
    if not (margin <= start[0] <= spec.width - margin and margin <= start[1] <= spec.height - margin):
        raise ValueError("start is outside safe workspace bounds")
    if not (margin <= goal[0] <= spec.width - margin and margin <= goal[1] <= spec.height - margin):
        raise ValueError("goal is outside safe workspace bounds")

    def axis_nodes(lower: float, upper: float, exact: tuple[float, float]) -> list[float]:
        values = [lower]
        index = 1
        while lower + index * resolution < upper:
            values.append(lower + index * resolution)
            index += 1
        values.append(upper)
        values.extend(exact)
        return sorted(set(values))

    xs = axis_nodes(margin, spec.width - margin, (start[0], goal[0]))
    ys = axis_nodes(margin, spec.height - margin, (start[1], goal[1]))
    x_index = {value: index for index, value in enumerate(xs)}
    y_index = {value: index for index, value in enumerate(ys)}
    start_key = (x_index[start[0]], y_index[start[1]])
    goal_key = (x_index[goal[0]], y_index[goal[1]])

    def point(key: tuple[int, int]) -> tuple[float, float]:
        return xs[key[0]], ys[key[1]]

    def free_point(key: tuple[int, int]) -> bool:
        p = point(key)
        return first_swept_collision_fraction(p, p, spec) is None

    if not free_point(start_key) or not free_point(goal_key):
        return None

    serial = count()
    frontier = [(hypot(goal[0] - start[0], goal[1] - start[1]), next(serial), start_key)]
    cost_so_far = {start_key: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_key: None}
    expanded = 0
    while frontier:
        _, _, current = heappop(frontier)
        if current == goal_key:
            break
        expanded += 1
        current_point = point(current)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < len(xs) and 0 <= neighbor[1] < len(ys)):
                    continue
                next_point = point(neighbor)
                if first_swept_collision_fraction(current_point, next_point, spec) is not None:
                    continue
                edge_cost = hypot(next_point[0] - current_point[0], next_point[1] - current_point[1])
                new_cost = cost_so_far[current] + edge_cost
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heuristic = hypot(goal[0] - next_point[0], goal[1] - next_point[1])
                heappush(frontier, (new_cost + heuristic, next(serial), neighbor))
    else:
        return None

    keys = []
    cursor: tuple[int, int] | None = goal_key
    while cursor is not None:
        keys.append(cursor)
        cursor = came_from[cursor]
    keys.reverse()
    return GeometricRoute(
        waypoints=tuple(point(key) for key in keys),
        length=cost_so_far[goal_key],
        expanded_nodes=expanded,
    )


def waypoint_braking_action(
    state: CorridorState,
    route: GeometricRoute,
    spec: CorridorTaskSpec,
    waypoint_tolerance: float = 0.015,
) -> int:
    """Choose the discrete action whose next velocity best tracks a braking target.

    This is a geometry-route + local velocity controller baseline. It is not a
    tuned controller: gains/tolerance must be selected on a separate design set.
    """
    if not isfinite(waypoint_tolerance) or waypoint_tolerance < 0:
        raise ValueError("waypoint_tolerance must be finite and nonnegative")
    if not route.waypoints:
        raise ValueError("route must contain at least one waypoint")
    target_index = min(
        range(len(route.waypoints)),
        key=lambda index: hypot(
            state.x - route.waypoints[index][0],
            state.y - route.waypoints[index][1],
        ),
    )
    while (
        target_index < len(route.waypoints) - 1
        and hypot(
            state.x - route.waypoints[target_index][0],
            state.y - route.waypoints[target_index][1],
        ) <= waypoint_tolerance
    ):
        target_index += 1
    target = route.waypoints[target_index]
    dx, dy = target[0] - state.x, target[1] - state.y
    distance = hypot(dx, dy)
    if distance <= waypoint_tolerance:
        desired_velocity = (0.0, 0.0)
    else:
        desired_speed = min(
            spec.max_speed,
            sqrt(2.0 * spec.acceleration * distance),
            spec.safe_speed if target_index == len(route.waypoints) - 1 else spec.max_speed,
        )
        desired_velocity = (desired_speed * dx / distance, desired_speed * dy / distance)

    best_action, best_error = 0, float("inf")
    for action_id, direction in enumerate(ACTION_VECTORS):
        ax, ay = direction
        norm = hypot(ax, ay)
        if norm:
            ax, ay = ax / norm, ay / norm
        vx = spec.damping * state.vx + spec.acceleration * ax
        vy = spec.damping * state.vy + spec.acceleration * ay
        speed = hypot(vx, vy)
        if speed > spec.max_speed:
            vx, vy = vx * spec.max_speed / speed, vy * spec.max_speed / speed
        error = (vx - desired_velocity[0]) ** 2 + (vy - desired_velocity[1]) ** 2
        if error < best_error:
            best_action, best_error = action_id, error
    return best_action


def _route_projection(
    position: tuple[float, float], route: GeometricRoute
) -> tuple[int, float, tuple[float, float], float]:
    """Return closest route segment, path progress, tangent, and cross-track distance."""
    if not route.waypoints:
        raise ValueError("route must contain at least one waypoint")
    if len(route.waypoints) == 1:
        point = route.waypoints[0]
        return 0, 0.0, (0.0, 0.0), hypot(position[0] - point[0], position[1] - point[1])
    best = None
    arc = 0.0
    for index, (start, end) in enumerate(zip(route.waypoints, route.waypoints[1:])):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = hypot(dx, dy)
        if length == 0.0:
            continue
        t = max(0.0, min(1.0, ((position[0] - start[0]) * dx + (position[1] - start[1]) * dy) / (length * length)))
        px, py = start[0] + t * dx, start[1] + t * dy
        distance = hypot(position[0] - px, position[1] - py)
        candidate = (distance, index, arc + t * length, (dx / length, dy / length))
        if best is None or candidate[:3] < best[:3]:
            best = candidate
        arc += length
    if best is None:
        point = route.waypoints[-1]
        return 0, 0.0, (0.0, 0.0), hypot(position[0] - point[0], position[1] - point[1])
    distance, index, progress, tangent = best
    return index, progress, tangent, distance


def _next_route_constraint(
    route: GeometricRoute,
    segment_index: int,
    progress: float,
    corner_angle_degrees: float,
) -> tuple[float, tuple[float, float], bool]:
    """Return distance/tangent to the next material corner, or to the goal."""
    if not 0.0 < corner_angle_degrees < 180.0:
        raise ValueError("corner_angle_degrees must lie in (0, 180)")
    lengths = [
        hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(route.waypoints, route.waypoints[1:])
    ]
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    threshold = corner_angle_degrees * pi / 180.0
    for vertex_index in range(max(1, segment_index + 1), len(route.waypoints) - 1):
        incoming = route.waypoints[vertex_index][0] - route.waypoints[vertex_index - 1][0], route.waypoints[vertex_index][1] - route.waypoints[vertex_index - 1][1]
        outgoing = route.waypoints[vertex_index + 1][0] - route.waypoints[vertex_index][0], route.waypoints[vertex_index + 1][1] - route.waypoints[vertex_index][1]
        in_norm, out_norm = hypot(*incoming), hypot(*outgoing)
        if in_norm == 0.0 or out_norm == 0.0:
            continue
        cosine = max(-1.0, min(1.0, (incoming[0] * outgoing[0] + incoming[1] * outgoing[1]) / (in_norm * out_norm)))
        if acos(cosine) >= threshold:
            return max(0.0, cumulative[vertex_index] - progress), (incoming[0] / in_norm, incoming[1] / in_norm), False
    goal = route.waypoints[-1]
    if lengths:
        final = route.waypoints[-1][0] - route.waypoints[-2][0], route.waypoints[-1][1] - route.waypoints[-2][1]
        norm = hypot(*final)
        tangent = (final[0] / norm, final[1] / norm) if norm else (0.0, 0.0)
    else:
        tangent = (0.0, 0.0)
    return max(0.0, cumulative[-1] - progress), tangent, True


def _braking_distance(
    speed: float,
    direction: tuple[float, float],
    target_speed: float,
    spec: CorridorTaskSpec,
) -> float:
    """Simulate maximum discrete braking and measure along-route travel."""
    vx, vy = speed * direction[0], speed * direction[1]
    distance = 0.0
    for _ in range(256):
        current_speed = hypot(vx, vy)
        if current_speed <= target_speed + 1e-10:
            return distance
        next_options = []
        for action_id, vector in enumerate(ACTION_VECTORS):
            ax, ay = vector
            norm = hypot(ax, ay)
            if norm:
                ax, ay = ax / norm, ay / norm
            nvx, nvy = spec.damping * vx + spec.acceleration * ax, spec.damping * vy + spec.acceleration * ay
            next_speed = hypot(nvx, nvy)
            if next_speed > spec.max_speed:
                nvx, nvy = nvx * spec.max_speed / next_speed, nvy * spec.max_speed / next_speed
                next_speed = spec.max_speed
            next_options.append((next_speed, action_id, nvx, nvy))
        next_speed, _, vx, vy = min(next_options)
        if next_speed >= current_speed - 1e-12:
            return float("inf")
        distance += max(0.0, vx * direction[0] + vy * direction[1]) * spec.dt
    return float("inf")


def _braking_speed_cap(
    available_distance: float,
    direction: tuple[float, float],
    target_speed: float,
    spec: CorridorTaskSpec,
) -> float:
    """Largest route-aligned speed whose discrete braking fits the distance."""
    low = min(target_speed, spec.max_speed)
    high = spec.max_speed
    for _ in range(36):
        middle = (low + high) / 2.0
        if _braking_distance(middle, direction, target_speed, spec) <= available_distance:
            low = middle
        else:
            high = middle
    return low


def phase_space_route_action(
    state: CorridorState,
    route: GeometricRoute,
    spec: CorridorTaskSpec,
    *,
    brake_safety_margin: float = 0.04,
    corner_angle_degrees: float = 30.0,
) -> int:
    """Damping-aware route follower with one-step exact collision filtering.

    It computes an A*-route phase-space speed cap from a discrete braking
    simulation, then chooses the safe primitive action whose exact next
    velocity best tracks that reference. There is no action-sequence search.
    If every action collides, coast is the deterministic fallback.
    """
    validate_corridor_state(state, spec)
    if state.status != "active":
        raise ValueError("phase-space controller requires an active state")
    if not isfinite(brake_safety_margin) or brake_safety_margin < 0:
        raise ValueError("brake_safety_margin must be finite and nonnegative")
    segment, progress, tangent, _ = _route_projection((state.x, state.y), route)
    remaining, constraint_tangent, is_goal = _next_route_constraint(
        route, segment, progress, corner_angle_degrees
    )
    direction = tangent if hypot(*tangent) > 0 else constraint_tangent
    if hypot(*direction) == 0:
        direction = (0.0, 0.0)
    # The controller reaches a corner at no more than the task's safe-goal
    # speed; goal entry itself has the same hard simulator threshold.
    target_speed = spec.safe_speed
    available = max(0.0, remaining - brake_safety_margin)
    speed_cap = _braking_speed_cap(available, direction, target_speed, spec)
    desired_velocity = (speed_cap * direction[0], speed_cap * direction[1])

    choices = []
    for action_id in range(len(ACTION_VECTORS)):
        transition = corridor_step(state, action_id, spec)
        if transition.collision:
            continue
        next_state = transition.state
        velocity_error = (next_state.vx - desired_velocity[0]) ** 2 + (next_state.vy - desired_velocity[1]) ** 2
        # Success is always preferable to another active transition.
        choices.append((0 if transition.success else 1, velocity_error, action_id))
    return min(choices)[2] if choices else 0


def quantized_kinodynamic_first_action(
    initial_state: CorridorState,
    spec: CorridorTaskSpec,
    position_resolution: float = 0.04,
    velocity_resolution: float = 0.04,
    max_expansions: int = 20_000,
) -> KinodynamicSearchResult:
    """Approximate A* over exact one-step mechanics and quantized state keys.

    State transitions use the deterministic simulator, but states mapping to
    the same rounded (x,y,vx,vy) bin are merged. This is an approximation, not
    exact optimal control; resolution and expansion limit must be reported.
    The objective is minimum steps to safe goal entry, with collision forbidden.
    """
    validate_corridor_state(initial_state, spec)
    if initial_state.status != "active":
        raise ValueError("search requires an active initial state")
    if not isfinite(position_resolution) or position_resolution <= 0:
        raise ValueError("position_resolution must be finite and positive")
    if not isfinite(velocity_resolution) or velocity_resolution <= 0:
        raise ValueError("velocity_resolution must be finite and positive")
    if isinstance(max_expansions, bool) or not isinstance(max_expansions, Integral) or max_expansions <= 0:
        raise ValueError("max_expansions must be a positive integer")

    def key(state: CorridorState) -> tuple[int, int, int, int]:
        return (
            round(state.x / position_resolution),
            round(state.y / position_resolution),
            round(state.vx / velocity_resolution),
            round(state.vy / velocity_resolution),
        )

    def heuristic(state: CorridorState) -> int:
        distance = max(
            0.0,
            hypot(state.x - spec.goal[0], state.y - spec.goal[1]) - spec.goal_radius,
        )
        return int(distance / (spec.max_speed * spec.dt))

    start_key = key(initial_state)
    serial = count()
    frontier = [(heuristic(initial_state), next(serial), 0, start_key)]
    best_steps = {start_key: 0}
    states = {start_key: initial_state}
    first_actions: dict[tuple[int, int, int, int], int | None] = {start_key: None}
    expanded = 0

    while frontier and expanded < max_expansions:
        _, _, steps, current_key = heappop(frontier)
        if steps != best_steps.get(current_key):
            continue
        current = states[current_key]
        expanded += 1
        for action_id in range(len(ACTION_VECTORS)):
            result = corridor_step(current, action_id, spec)
            if result.collision or result.truncated:
                continue
            first_action = action_id if current_key == start_key else first_actions[current_key]
            if result.success:
                return KinodynamicSearchResult(
                    "success", first_action, expanded,
                    float(position_resolution), float(velocity_resolution),
                )
            next_steps = steps + 1
            next_key = key(result.state)
            if next_steps >= best_steps.get(next_key, 2**63 - 1):
                continue
            best_steps[next_key] = next_steps
            states[next_key] = result.state
            first_actions[next_key] = first_action
            score = next_steps + heuristic(result.state)
            heappush(frontier, (score, next(serial), next_steps, next_key))

    status = "expansion_limit" if frontier else "no_path"
    return KinodynamicSearchResult(
        status, None, expanded, float(position_resolution), float(velocity_resolution)
    )
