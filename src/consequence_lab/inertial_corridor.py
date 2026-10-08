"""Pure deterministic mechanics contract for the inertial-corridor prototype.

This module defines task dynamics only. It contains no reward, task generator,
planner, training code, or benchmark-success claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite, sqrt
from numbers import Integral


# Geometry-independent action IDs: coast, cardinal accelerations, diagonals.
ACTION_VECTORS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

TERMINAL_STATUSES = frozenset({"success", "collision", "timeout"})


@dataclass(frozen=True)
class AxisAlignedObstacle:
    """Closed axis-aligned obstacle rectangle in workspace coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("obstacle coordinates must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("obstacle must have positive width and height")


@dataclass(frozen=True)
class CorridorTaskSpec:
    """Public geometry and fixed dynamics shared by all decision methods."""

    width: float
    height: float
    obstacles: tuple[AxisAlignedObstacle, ...]
    goal: tuple[float, float]
    goal_radius: float
    safe_speed: float
    agent_radius: float = 0.02
    acceleration: float = 0.025
    damping: float = 0.98
    max_speed: float = 0.12
    dt: float = 1.0
    max_steps: int = 200

    def __post_init__(self) -> None:
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        object.__setattr__(self, "goal", tuple(self.goal))
        scalars = (
            self.width, self.height, self.goal_radius, self.safe_speed,
            self.agent_radius, self.acceleration, self.damping,
            self.max_speed, self.dt,
        )
        if not all(isfinite(float(value)) for value in scalars):
            raise ValueError("task parameters must be finite")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("workspace dimensions must be positive")
        if not 0 < self.agent_radius < min(self.width, self.height) / 2:
            raise ValueError("agent_radius must fit inside the workspace")
        if self.goal_radius <= 0 or self.safe_speed < 0:
            raise ValueError("goal_radius must be positive and safe_speed nonnegative")
        if self.acceleration <= 0 or self.max_speed <= 0 or self.dt <= 0:
            raise ValueError("acceleration, max_speed and dt must be positive")
        if not 0 <= self.damping <= 1:
            raise ValueError("damping must lie in [0, 1]")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, Integral) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if len(self.goal) != 2 or not all(isfinite(float(v)) for v in self.goal):
            raise ValueError("goal must be a finite two-vector")
        gx, gy = self.goal
        if not (self.goal_radius <= gx <= self.width - self.goal_radius):
            raise ValueError("goal circle must lie inside the workspace")
        if not (self.goal_radius <= gy <= self.height - self.goal_radius):
            raise ValueError("goal circle must lie inside the workspace")
        for obstacle in self.obstacles:
            if not isinstance(obstacle, AxisAlignedObstacle):
                raise TypeError("obstacles must be AxisAlignedObstacle values")
            if not (0 <= obstacle.x_min < obstacle.x_max <= self.width):
                raise ValueError("obstacle x bounds must lie inside the workspace")
            if not (0 <= obstacle.y_min < obstacle.y_max <= self.height):
                raise ValueError("obstacle y bounds must lie inside the workspace")
            if point_in_expanded_obstacle(
                (gx, gy), obstacle, self.agent_radius + self.goal_radius
            ):
                raise ValueError("goal region must not overlap an expanded obstacle")


@dataclass(frozen=True)
class CorridorState:
    """Complete public dynamical state; task geometry is in ``CorridorTaskSpec``."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    elapsed_steps: int = 0
    status: str = "active"

    def __post_init__(self) -> None:
        if not all(isfinite(float(v)) for v in (self.x, self.y, self.vx, self.vy)):
            raise ValueError("state position and velocity must be finite")
        if isinstance(self.elapsed_steps, bool) or not isinstance(self.elapsed_steps, Integral) or self.elapsed_steps < 0:
            raise ValueError("elapsed_steps must be a nonnegative integer")
        if self.status not in {"active", *TERMINAL_STATUSES}:
            raise ValueError("invalid terminal status")


@dataclass(frozen=True)
class CorridorStep:
    state: CorridorState
    collision: bool
    success: bool
    truncated: bool


def point_in_expanded_obstacle(
    point: tuple[float, float], obstacle: AxisAlignedObstacle, radius: float
) -> bool:
    """Whether a circular agent center overlaps a rectangle (closed contact)."""
    x, y = point
    nearest_x = min(max(x, obstacle.x_min), obstacle.x_max)
    nearest_y = min(max(y, obstacle.y_min), obstacle.y_max)
    return (x - nearest_x) ** 2 + (y - nearest_y) ** 2 <= radius**2


def _segment_aabb_entry(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> float | None:
    """First segment parameter in [0,1] intersecting a closed AABB."""
    x0, y0 = start
    x1, y1 = end
    xmin, ymin, xmax, ymax = bounds
    t_enter, t_exit = 0.0, 1.0
    for origin, destination, lower, upper in (
        (x0, x1, xmin, xmax),
        (y0, y1, ymin, ymax),
    ):
        delta = destination - origin
        if delta == 0:
            if origin < lower or origin > upper:
                return None
            continue
        t0 = (lower - origin) / delta
        t1 = (upper - origin) / delta
        enter_axis, exit_axis = min(t0, t1), max(t0, t1)
        t_enter = max(t_enter, enter_axis)
        t_exit = min(t_exit, exit_axis)
        if t_enter > t_exit:
            return None
    return t_enter if t_exit >= 0.0 and t_enter <= 1.0 else None


def _first_boundary_exit(
    start: tuple[float, float], end: tuple[float, float],
    xmin: float, ymin: float, xmax: float, ymax: float,
) -> float | None:
    """First contact with the unsafe exterior of the allowed center rectangle."""
    if xmin <= end[0] <= xmax and ymin <= end[1] <= ymax:
        return None
    candidates = []
    for origin, destination, lower, upper in (
        (start[0], end[0], xmin, xmax),
        (start[1], end[1], ymin, ymax),
    ):
        delta = destination - origin
        if delta < 0 and destination < lower:
            candidates.append((lower - origin) / delta)
        elif delta > 0 and destination > upper:
            candidates.append((upper - origin) / delta)
    return min(candidates) if candidates else None


def _segment_circle_entry(
    start: tuple[float, float], end: tuple[float, float],
    center: tuple[float, float], radius: float,
) -> float | None:
    """First segment parameter entering a closed circle, if any."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    ox, oy = start[0] - center[0], start[1] - center[1]
    c = ox * ox + oy * oy - radius * radius
    if c <= 0:
        return 0.0
    a = dx * dx + dy * dy
    if a == 0:
        return None
    b = 2.0 * (ox * dx + oy * dy)
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return None
    t = (-b - sqrt(discriminant)) / (2.0 * a)
    return t if 0.0 <= t <= 1.0 else None


def _segment_obstacle_entry(
    start: tuple[float, float], end: tuple[float, float],
    obstacle: AxisAlignedObstacle, radius: float,
) -> float | None:
    """Exact swept-circle contact with a rectangle via its rounded Minkowski sum."""
    xmin, ymin, xmax, ymax = obstacle.x_min, obstacle.y_min, obstacle.x_max, obstacle.y_max
    entries = [
        _segment_aabb_entry(start, end, (xmin - radius, ymin, xmax + radius, ymax)),
        _segment_aabb_entry(start, end, (xmin, ymin - radius, xmax, ymax + radius)),
    ]
    entries.extend(
        _segment_circle_entry(start, end, corner, radius)
        for corner in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    )
    valid = [entry for entry in entries if entry is not None]
    return min(valid) if valid else None


def first_swept_collision_fraction(
    start: tuple[float, float], end: tuple[float, float], spec: CorridorTaskSpec
) -> float | None:
    """Return the first contact fraction for a circular agent along a segment."""
    radius = spec.agent_radius
    collision_time = _first_boundary_exit(
        start, end, radius, radius, spec.width - radius, spec.height - radius
    )
    for obstacle in spec.obstacles:
        entry = _segment_obstacle_entry(start, end, obstacle, radius)
        if entry is not None and (collision_time is None or entry < collision_time):
            collision_time = entry
    return collision_time


def validate_corridor_state(state: CorridorState, spec: CorridorTaskSpec) -> None:
    """Validate the observable state contract before planning a transition."""
    if state.elapsed_steps > spec.max_steps:
        raise ValueError("elapsed_steps exceeds task horizon")
    if state.status == "active" and state.elapsed_steps >= spec.max_steps:
        raise ValueError("active state cannot remain after the task horizon")
    if hypot(state.vx, state.vy) > spec.max_speed + 1e-12:
        raise ValueError("state speed exceeds max_speed")
    radius = spec.agent_radius
    if not (radius <= state.x <= spec.width - radius):
        raise ValueError("agent center lies outside safe workspace bounds")
    if not (radius <= state.y <= spec.height - radius):
        raise ValueError("agent center lies outside safe workspace bounds")
    if state.status == "active":
        for obstacle in spec.obstacles:
            if point_in_expanded_obstacle((state.x, state.y), obstacle, radius):
                raise ValueError("active agent state overlaps an obstacle")


def corridor_step(
    state: CorridorState, action_id: int, spec: CorridorTaskSpec
) -> CorridorStep:
    """Apply one deterministic step; terminal collisions stop at contact state.

    Success is detected along the swept path at the current step's speed. A
    collision that occurs no later than goal entry takes precedence. If a
    collision occurs, the terminal state is the last collision-free state with
    zero velocity; this benchmark convention avoids inventing bounce physics.
    """
    validate_corridor_state(state, spec)
    if state.status != "active":
        raise ValueError("cannot step a terminal state")
    if isinstance(action_id, bool) or not isinstance(action_id, Integral) or not 0 <= action_id < len(ACTION_VECTORS):
        raise ValueError("action_id must be one of the nine legal integer IDs")

    ax, ay = ACTION_VECTORS[int(action_id)]
    if ax != 0 and ay != 0:
        diagonal_scale = sqrt(2.0)
        ax /= diagonal_scale
        ay /= diagonal_scale
    vx = spec.damping * state.vx + spec.acceleration * ax
    vy = spec.damping * state.vy + spec.acceleration * ay
    speed = hypot(vx, vy)
    if speed > spec.max_speed:
        scale = spec.max_speed / speed
        vx, vy = vx * scale, vy * scale
    start = (state.x, state.y)
    end = (state.x + vx * spec.dt, state.y + vy * spec.dt)
    radius = spec.agent_radius

    collision_time = first_swept_collision_fraction(start, end, spec)

    goal_time = _segment_circle_entry(start, end, spec.goal, spec.goal_radius)
    safe_speed = hypot(vx, vy) <= spec.safe_speed
    if goal_time is not None and safe_speed and (
        collision_time is None or goal_time < collision_time
    ):
        x = start[0] + goal_time * (end[0] - start[0])
        y = start[1] + goal_time * (end[1] - start[1])
        next_state = CorridorState(x, y, vx, vy, state.elapsed_steps + 1, "success")
        return CorridorStep(next_state, collision=False, success=True, truncated=False)

    if collision_time is not None:
        next_state = CorridorState(
            state.x, state.y, 0.0, 0.0, state.elapsed_steps + 1, "collision"
        )
        return CorridorStep(next_state, collision=True, success=False, truncated=False)

    elapsed = state.elapsed_steps + 1
    if elapsed >= spec.max_steps:
        next_state = CorridorState(end[0], end[1], vx, vy, elapsed, "timeout")
        return CorridorStep(next_state, collision=False, success=False, truncated=True)
    next_state = CorridorState(end[0], end[1], vx, vy, elapsed, "active")
    return CorridorStep(next_state, collision=False, success=False, truncated=False)
