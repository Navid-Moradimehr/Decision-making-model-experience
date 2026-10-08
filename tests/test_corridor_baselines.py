import pytest

from consequence_lab.corridor_baselines import (
    _braking_distance,
    _next_route_constraint,
    _route_projection,
    exact_dynamics_shooting,
    GeometricRoute,
    grid_astar_route,
    phase_space_route_action,
    quantized_kinodynamic_first_action,
    reactive_goal_action,
    waypoint_braking_action,
)
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import (
    AxisAlignedObstacle,
    CorridorState,
    CorridorTaskSpec,
    corridor_step,
    first_swept_collision_fraction,
)


def test_geometry_astar_finds_collision_free_route_around_barrier():
    spec = CorridorTaskSpec(
        width=1.0,
        height=1.0,
        obstacles=(AxisAlignedObstacle(0.45, 0.2, 0.55, 0.8),),
        goal=(0.8, 0.5),
        goal_radius=0.06,
        safe_speed=0.1,
    )
    start, goal = (0.2, 0.5), (0.8, 0.5)
    route = grid_astar_route(start, goal, spec, resolution=0.04)
    assert route is not None
    assert route.waypoints[0] == start
    assert route.waypoints[-1] == goal
    assert route.length > goal[0] - start[0]
    assert route.expanded_nodes > 0
    for first, second in zip(route.waypoints, route.waypoints[1:]):
        assert first_swept_collision_fraction(first, second, spec) is None


def test_geometry_astar_reports_disconnected_wall():
    spec = CorridorTaskSpec(
        width=1.0,
        height=1.0,
        obstacles=(AxisAlignedObstacle(0.48, 0.0, 0.52, 1.0),),
        goal=(0.8, 0.5),
        goal_radius=0.06,
        safe_speed=0.1,
    )
    assert grid_astar_route((0.2, 0.5), (0.8, 0.5), spec, resolution=0.04) is None


def test_waypoint_braking_baseline_returns_a_legal_action_for_generated_task():
    scenario = generate_staggered_corridor(709)
    route = grid_astar_route(
        (scenario.initial_state.x, scenario.initial_state.y),
        scenario.spec.goal,
        scenario.spec,
        resolution=0.02,
    )
    assert route is not None
    action = waypoint_braking_action(scenario.initial_state, route, scenario.spec)
    assert isinstance(action, int)
    assert 0 <= action < 9
    assert waypoint_braking_action(scenario.initial_state, route, scenario.spec) == action


def test_phase_space_controller_returns_deterministic_legal_action():
    scenario = generate_staggered_corridor(1701)
    route = grid_astar_route(
        (scenario.initial_state.x, scenario.initial_state.y), scenario.spec.goal,
        scenario.spec, resolution=0.04,
    )
    assert route is not None
    action = phase_space_route_action(scenario.initial_state, route, scenario.spec)
    assert action in range(9)
    assert phase_space_route_action(scenario.initial_state, route, scenario.spec) == action


def test_damping_aware_discrete_braking_distance_is_finite_and_monotone():
    spec = CorridorTaskSpec(
        width=1.0, height=1.0, obstacles=(), goal=(0.8, 0.8),
        goal_radius=0.06, safe_speed=0.035, acceleration=0.024,
        damping=0.97, max_speed=0.12,
    )
    at_08 = _braking_distance(0.08, (1.0, 0.0), spec.safe_speed, spec)
    at_12 = _braking_distance(0.12, (1.0, 0.0), spec.safe_speed, spec)
    assert 0.0 < at_08 < at_12


def test_route_projection_finds_nearest_segment_and_upcoming_corner():
    route = GeometricRoute(
        waypoints=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        length=2.0,
        expanded_nodes=0,
    )
    segment, progress, tangent, cross_track = _route_projection((0.4, 0.2), route)
    assert segment == 0
    assert progress == pytest.approx(0.4)
    assert tangent == pytest.approx((1.0, 0.0))
    assert cross_track == pytest.approx(0.2)
    distance, constraint_tangent, is_goal = _next_route_constraint(
        route, segment, progress, corner_angle_degrees=30.0
    )
    assert distance == pytest.approx(0.6)
    assert constraint_tangent == pytest.approx((1.0, 0.0))
    assert is_goal is False


def test_phase_space_controller_uses_a_safe_action_when_one_exists():
    spec = CorridorTaskSpec(
        width=1.0, height=1.0,
        obstacles=(AxisAlignedObstacle(0.55, 0.35, 0.60, 0.65),),
        goal=(0.85, 0.5), goal_radius=0.06, safe_speed=0.08,
        agent_radius=0.02, acceleration=0.03, damping=0.98,
        max_speed=0.2, max_steps=60,
    )
    state = CorridorState(0.47, 0.5, vx=0.07)
    route = grid_astar_route((state.x, state.y), spec.goal, spec, resolution=0.03)
    assert route is not None
    safe_actions = [action for action in range(9) if not corridor_step(state, action, spec).collision]
    assert safe_actions
    assert corridor_step(state, 0, spec).collision
    chosen = phase_space_route_action(state, route, spec)
    assert not corridor_step(state, chosen, spec).collision


@pytest.mark.parametrize("margin", [-0.01, float("inf")])
def test_phase_space_controller_rejects_invalid_braking_margin(margin):
    scenario = generate_staggered_corridor(1702)
    route = grid_astar_route(
        (scenario.initial_state.x, scenario.initial_state.y), scenario.spec.goal,
        scenario.spec, resolution=0.04,
    )
    assert route is not None
    with pytest.raises(ValueError, match="brake_safety_margin"):
        phase_space_route_action(
            scenario.initial_state, route, scenario.spec,
            brake_safety_margin=margin,
        )


def test_quantized_kinodynamic_search_returns_reproducible_legal_first_action():
    spec = CorridorTaskSpec(
        width=1.0,
        height=1.0,
        obstacles=(),
        goal=(0.8, 0.5),
        goal_radius=0.08,
        safe_speed=0.4,
        agent_radius=0.02,
        acceleration=0.2,
        damping=0.9,
        max_speed=0.4,
        max_steps=30,
    )
    initial = CorridorState(0.2, 0.5)
    first = quantized_kinodynamic_first_action(
        initial, spec, position_resolution=0.05, velocity_resolution=0.1,
        max_expansions=5000,
    )
    second = quantized_kinodynamic_first_action(
        initial, spec, position_resolution=0.05, velocity_resolution=0.1,
        max_expansions=5000,
    )
    assert first == second
    assert first.status == "success"
    assert first.first_action in range(9)
    assert first.position_resolution == 0.05
    assert first.velocity_resolution == 0.1


def test_kinodynamic_search_reports_expansion_limit_without_fake_action():
    scenario = generate_staggered_corridor(711)
    result = quantized_kinodynamic_first_action(
        scenario.initial_state, scenario.spec, max_expansions=1
    )
    assert result.status == "expansion_limit"
    assert result.first_action is None
    assert result.expanded_nodes == 1


def test_reactive_controller_chooses_goalward_action_in_open_space():
    spec = CorridorTaskSpec(
        width=1.0, height=1.0, obstacles=(), goal=(0.8, 0.5),
        goal_radius=0.06, safe_speed=0.1, acceleration=0.03,
        max_speed=0.12,
    )
    state = CorridorState(0.2, 0.5)
    assert reactive_goal_action(state, spec) == 2


def test_reactive_controller_returns_a_legal_action_on_seeded_layout():
    scenario = generate_staggered_corridor(712)
    action = reactive_goal_action(scenario.initial_state, scenario.spec)
    assert action in range(9)


def test_exact_shooting_finds_successful_sequence_and_is_reproducible():
    spec = CorridorTaskSpec(
        width=1.0, height=1.0, obstacles=(), goal=(0.4, 0.5),
        goal_radius=0.1, safe_speed=0.4, agent_radius=0.02,
        acceleration=0.2, damping=0.9, max_speed=0.4, max_steps=10,
    )
    state = CorridorState(0.2, 0.5)
    first = exact_dynamics_shooting(
        state, spec, horizon=1, num_sequences=20, seed=31
    )
    second = exact_dynamics_shooting(
        state, spec, horizon=1, num_sequences=20, seed=31
    )
    assert first == second
    assert first.first_action == 2
    assert first.objective_key[0] == 0
    assert first.candidate_success_available
    assert first.evaluated_sequences == 9  # There are only 9^1 unique sequences.


def test_exact_shooting_reports_no_success_in_a_short_horizon():
    scenario = generate_staggered_corridor(713)
    result = exact_dynamics_shooting(
        scenario.initial_state, scenario.spec, horizon=1, num_sequences=9, seed=4
    )
    assert result.evaluated_sequences == 9
    assert not result.candidate_success_available
    assert result.first_action in range(9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"num_sequences": 0},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_exact_shooting_rejects_invalid_configuration(kwargs):
    scenario = generate_staggered_corridor(714)
    with pytest.raises(ValueError):
        exact_dynamics_shooting(scenario.initial_state, scenario.spec, **kwargs)


@pytest.mark.parametrize("resolution", [0.0, -0.01, float("inf")])
def test_geometry_astar_rejects_invalid_resolution(resolution):
    scenario = generate_staggered_corridor(710)
    with pytest.raises(ValueError, match="resolution"):
        grid_astar_route(
            (scenario.initial_state.x, scenario.initial_state.y),
            scenario.spec.goal,
            scenario.spec,
            resolution=resolution,
        )
