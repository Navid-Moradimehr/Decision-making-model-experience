from dataclasses import FrozenInstanceError
from math import hypot, sqrt
import random

import pytest

from consequence_lab.inertial_corridor import (
    ACTION_VECTORS,
    AxisAlignedObstacle,
    CorridorState,
    CorridorTaskSpec,
    corridor_step,
)


def make_spec(**overrides):
    values = dict(
        width=1.0,
        height=1.0,
        obstacles=(),
        goal=(0.8, 0.8),
        goal_radius=0.06,
        safe_speed=0.15,
        agent_radius=0.02,
        acceleration=0.025,
        damping=0.98,
        max_speed=0.5,
        dt=1.0,
        max_steps=20,
    )
    values.update(overrides)
    return CorridorTaskSpec(**values)


def _point_rect_distance_squared(point, rect):
    dx = max(rect.x_min - point[0], 0.0, point[0] - rect.x_max)
    dy = max(rect.y_min - point[1], 0.0, point[1] - rect.y_max)
    return dx * dx + dy * dy


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a, b, c, d):
    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and ((o3 > 0 > o4) or (o3 < 0 < o4)):
        return True
    return (
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def _segment_rect_distance_squared(start, end, rect):
    corners = (
        (rect.x_min, rect.y_min), (rect.x_min, rect.y_max),
        (rect.x_max, rect.y_min), (rect.x_max, rect.y_max),
    )
    if (
        rect.x_min <= start[0] <= rect.x_max and rect.y_min <= start[1] <= rect.y_max
    ) or (
        rect.x_min <= end[0] <= rect.x_max and rect.y_min <= end[1] <= rect.y_max
    ):
        return 0.0
    edges = (
        (corners[0], corners[1]), (corners[0], corners[2]),
        (corners[1], corners[3]), (corners[2], corners[3]),
    )
    if any(_segments_intersect(start, end, a, b) for a, b in edges):
        return 0.0
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dy * dy
    distances = [_point_rect_distance_squared(start, rect), _point_rect_distance_squared(end, rect)]
    if length_squared:
        for x, y in corners:
            t = max(0.0, min(1.0, ((x - start[0]) * dx + (y - start[1]) * dy) / length_squared))
            distances.append((start[0] + t * dx - x) ** 2 + (start[1] + t * dy - y) ** 2)
    return min(distances)


def _reference_segment_obstacle_entry(start, end, obstacle, radius):
    """Slow independent closest-distance + bisection reference for tests only."""
    limit = radius * radius
    if _segment_rect_distance_squared(start, start, obstacle) <= limit:
        return 0.0
    if _segment_rect_distance_squared(start, end, obstacle) > limit:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(55):
        mid = (lo + hi) / 2.0
        point = (start[0] + mid * (end[0] - start[0]), start[1] + mid * (end[1] - start[1]))
        if _segment_rect_distance_squared(start, point, obstacle) <= limit:
            hi = mid
        else:
            lo = mid
    return hi


def _reference_goal_entry(start, end, center, radius):
    dx, dy = end[0] - start[0], end[1] - start[1]
    ox, oy = start[0] - center[0], start[1] - center[1]
    c = ox * ox + oy * oy - radius * radius
    if c <= 0:
        return 0.0
    a = dx * dx + dy * dy
    if a == 0:
        return None
    b = 2 * (ox * dx + oy * dy)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - sqrt(disc)) / (2 * a)
    return t if 0 <= t <= 1 else None


def reference_step(state, action_id, spec):
    """Slow test oracle, independently structured from production geometry code."""
    ax, ay = ACTION_VECTORS[action_id]
    direction_norm = hypot(ax, ay)
    if direction_norm:
        ax, ay = ax / direction_norm, ay / direction_norm
    vx = spec.damping * state.vx + spec.acceleration * ax
    vy = spec.damping * state.vy + spec.acceleration * ay
    speed = hypot(vx, vy)
    if speed > spec.max_speed:
        vx, vy = vx * spec.max_speed / speed, vy * spec.max_speed / speed
    start = (state.x, state.y)
    end = (state.x + vx * spec.dt, state.y + vy * spec.dt)

    collisions = []
    lo, hi = spec.agent_radius, spec.width - spec.agent_radius
    if end[0] < lo:
        collisions.append((lo - start[0]) / (end[0] - start[0]))
    elif end[0] > hi:
        collisions.append((hi - start[0]) / (end[0] - start[0]))
    lo, hi = spec.agent_radius, spec.height - spec.agent_radius
    if end[1] < lo:
        collisions.append((lo - start[1]) / (end[1] - start[1]))
    elif end[1] > hi:
        collisions.append((hi - start[1]) / (end[1] - start[1]))
    collisions.extend(
        t for obstacle in spec.obstacles
        if (t := _reference_segment_obstacle_entry(start, end, obstacle, spec.agent_radius)) is not None
    )
    collision_time = min(collisions) if collisions else None
    goal_time = _reference_goal_entry(start, end, spec.goal, spec.goal_radius)
    if goal_time is not None and speed <= spec.safe_speed and (
        collision_time is None or goal_time < collision_time
    ):
        pos = (start[0] + goal_time * (end[0] - start[0]), start[1] + goal_time * (end[1] - start[1]))
        return CorridorState(*pos, vx, vy, state.elapsed_steps + 1, "success")
    if collision_time is not None:
        return CorridorState(state.x, state.y, 0.0, 0.0, state.elapsed_steps + 1, "collision")
    elapsed = state.elapsed_steps + 1
    status = "timeout" if elapsed >= spec.max_steps else "active"
    return CorridorState(*end, vx, vy, elapsed, status)


def test_action_ids_are_fixed_geometry_independent_and_unique():
    assert len(ACTION_VECTORS) == 9
    assert len(set(ACTION_VECTORS)) == 9
    assert ACTION_VECTORS[0] == (0, 0)
    assert set(ACTION_VECTORS[1:]) == {
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    }


def test_inertia_persists_under_coast_and_step_is_deterministic():
    spec = make_spec()
    initial = CorridorState(0.3, 0.3, vx=0.2, vy=-0.1)
    first = corridor_step(initial, 0, spec)
    second = corridor_step(initial, 0, spec)
    assert first == second
    assert first.state.x == pytest.approx(0.3 + 0.98 * 0.2)
    assert first.state.y == pytest.approx(0.3 - 0.98 * 0.1)
    assert first.state.vx == pytest.approx(0.196)
    assert first.state.vy == pytest.approx(-0.098)
    assert first.state.status == "active"
    with pytest.raises(FrozenInstanceError):
        initial.x = 0.4


def test_diagonal_velocity_is_capped_by_speed_magnitude():
    spec = make_spec(acceleration=0.5, max_speed=0.2)
    result = corridor_step(CorridorState(0.4, 0.4), 8, spec)
    assert (result.state.vx**2 + result.state.vy**2) ** 0.5 == pytest.approx(0.2)


def test_cardinal_and_diagonal_actions_have_equal_acceleration_magnitude():
    spec = make_spec(acceleration=0.05)
    cardinal = corridor_step(CorridorState(0.4, 0.4), 2, spec).state
    diagonal = corridor_step(CorridorState(0.4, 0.4), 8, spec).state
    assert (cardinal.vx**2 + cardinal.vy**2) ** 0.5 == pytest.approx(0.05)
    assert (diagonal.vx**2 + diagonal.vy**2) ** 0.5 == pytest.approx(0.05)


def test_swept_collision_catches_thin_obstacle_even_when_endpoint_passes_it():
    obstacle = AxisAlignedObstacle(0.4, 0.35, 0.45, 0.65)
    spec = make_spec(obstacles=(obstacle,), max_speed=0.5)
    initial = CorridorState(0.2, 0.5, vx=0.4)
    result = corridor_step(initial, 0, spec)
    assert result.collision
    assert not result.success and not result.truncated
    assert result.state.status == "collision"
    assert (result.state.x, result.state.y, result.state.vx, result.state.vy) == (
        initial.x, initial.y, 0.0, 0.0
    )


def test_circle_agent_does_not_false_collide_with_square_inflation_corner():
    obstacle = AxisAlignedObstacle(0.5, 0.5, 0.6, 0.6)
    spec = make_spec(obstacles=(obstacle,), agent_radius=0.08)
    result = corridor_step(CorridorState(0.43, 0.43), 0, spec)
    assert not result.collision
    assert result.state.status == "active"


def test_workspace_boundary_collision_uses_agent_radius():
    spec = make_spec()
    result = corridor_step(CorridorState(0.9, 0.5, vx=0.2), 2, spec)
    assert result.collision
    assert result.state.x == 0.9


def test_goal_entry_requires_safe_speed_and_uses_swept_path():
    safe_spec = make_spec(goal=(0.5, 0.5), goal_radius=0.08, safe_speed=0.15)
    safe = corridor_step(CorridorState(0.3, 0.5, vx=0.15), 0, safe_spec)
    assert safe.success
    assert safe.state.status == "success"
    assert safe.state.x == pytest.approx(0.42)

    fast_spec = make_spec(goal=(0.5, 0.5), goal_radius=0.08, safe_speed=0.1)
    fast = corridor_step(CorridorState(0.3, 0.5, vx=0.3), 0, fast_spec)
    assert not fast.success
    assert fast.state.status == "active"


def test_collision_before_goal_takes_precedence():
    obstacle = AxisAlignedObstacle(0.42, 0.35, 0.48, 0.65)
    spec = make_spec(
        obstacles=(obstacle,), goal=(0.7, 0.5), goal_radius=0.08,
        safe_speed=0.5, max_speed=0.5,
    )
    result = corridor_step(CorridorState(0.2, 0.5, vx=0.5), 0, spec)
    assert result.collision
    assert not result.success


def test_timeout_is_truncation_and_terminal_states_cannot_step():
    spec = make_spec(max_steps=1)
    result = corridor_step(CorridorState(0.3, 0.3), 0, spec)
    assert result.truncated
    assert not result.collision and not result.success
    assert result.state.status == "timeout"
    with pytest.raises(ValueError, match="terminal"):
        corridor_step(result.state, 0, spec)
    with pytest.raises(ValueError, match="horizon"):
        corridor_step(CorridorState(0.3, 0.3, elapsed_steps=1), 0, spec)


@pytest.mark.parametrize(
    "state, action",
    [
        (CorridorState(0.3, 0.3), -1),
        (CorridorState(0.3, 0.3), 9),
        (CorridorState(0.3, 0.3), True),
        (CorridorState(0.3, 0.3, vx=0.6), 0),
        (CorridorState(0.01, 0.3), 0),
    ],
)
def test_invalid_state_or_action_is_rejected(state, action):
    with pytest.raises(ValueError):
        corridor_step(state, action, make_spec())


def test_spec_rejects_invalid_or_goal_blocking_geometry():
    with pytest.raises(ValueError):
        AxisAlignedObstacle(0.5, 0.5, 0.4, 0.7)
    with pytest.raises(ValueError, match="overlap"):
        make_spec(
            obstacles=(AxisAlignedObstacle(0.7, 0.7, 0.9, 0.9),),
            goal=(0.8, 0.8),
        )


def test_production_step_matches_independent_scalar_geometry_reference():
    rng = random.Random(110927)
    spec = make_spec(
        width=2.0,
        height=2.0,
        obstacles=(
            AxisAlignedObstacle(0.75, 0.3, 0.9, 0.8),
            AxisAlignedObstacle(1.2, 1.1, 1.35, 1.6),
        ),
        goal=(1.8, 1.8),
        goal_radius=0.08,
        agent_radius=0.04,
        acceleration=0.07,
        max_speed=0.22,
        safe_speed=0.11,
        max_steps=25,
    )
    fixed_cases = [
        (CorridorState(0.6, 0.55, vx=0.2), 0, "collision"),
        (CorridorState(1.7, 1.8, vx=0.04), 0, "success"),
        (CorridorState(0.2, 0.2, elapsed_steps=spec.max_steps - 1), 0, "timeout"),
        (CorridorState(0.2, 0.2), 0, "active"),
    ]
    checked = 0

    def assert_matches_reference(state, action_id, expected_status=None):
        actual = corridor_step(state, action_id, spec).state
        expected = reference_step(state, action_id, spec)
        assert actual.status == expected.status
        if expected_status is not None:
            assert actual.status == expected_status
        assert actual.elapsed_steps == expected.elapsed_steps
        assert (actual.x, actual.y, actual.vx, actual.vy) == pytest.approx(
            (expected.x, expected.y, expected.vx, expected.vy), abs=2e-8
        )

    for state, action_id, expected_status in fixed_cases:
        assert_matches_reference(state, action_id, expected_status)
        checked += 1

    while checked < 160:
        x, y = rng.uniform(0.05, 1.95), rng.uniform(0.05, 1.95)
        if any(_point_rect_distance_squared((x, y), obs) <= spec.agent_radius**2 for obs in spec.obstacles):
            continue
        vx, vy = rng.uniform(-0.22, 0.22), rng.uniform(-0.22, 0.22)
        norm = hypot(vx, vy)
        if norm > spec.max_speed:
            vx, vy = vx * spec.max_speed / norm, vy * spec.max_speed / norm
        state = CorridorState(x, y, vx, vy, rng.randrange(spec.max_steps))
        action_id = rng.randrange(len(ACTION_VECTORS))

        assert_matches_reference(state, action_id)
        checked += 1
