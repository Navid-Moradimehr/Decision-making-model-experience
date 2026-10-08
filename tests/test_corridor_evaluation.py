import pytest

from consequence_lab.corridor_evaluation import evaluate_corridor_episode
from consequence_lab.corridor_evaluation import (
    evaluate_corridor_block,
    paired_outcome_rate_interval,
    paired_success_gain_interval,
    summarize_corridor_results,
)
from consequence_lab.corridor_evaluation import CorridorEpisodeResult
from consequence_lab.corridor_tasks import CorridorScenario
from consequence_lab.inertial_corridor import (
    AxisAlignedObstacle,
    CorridorState,
    CorridorTaskSpec,
)


@pytest.fixture
def toy_scenario():
    spec = CorridorTaskSpec(
        width=1.0,
        height=1.0,
        obstacles=(),
        goal=(0.4, 0.5),
        goal_radius=0.1,
        safe_speed=0.1,
        agent_radius=0.02,
        acceleration=0.03,
        damping=0.9,
        max_speed=0.12,
        max_steps=60,
    )
    return CorridorScenario(
        seed=771,
        layout_family="unit_fixture",
        layout_sha256="a" * 64,
        spec=spec,
        initial_state=CorridorState(0.2, 0.5),
    )


@pytest.mark.parametrize(
    ("method", "parameters"),
    [
        ("reactive", {}),
        ("geometry_waypoint", {"resolution": 0.05}),
        ("phase_space", {"resolution": 0.04, "brake_safety_margin": 0.02}),
        (
        "kinodynamic",
            {"position_resolution": 0.05, "velocity_resolution": 0.1, "max_expansions": 100},
        ),
        ("random_shooting", {"horizon": 1, "num_sequences": 9}),
    ],
)
def test_common_episode_runner_executes_all_methods_on_unit_fixture(
    toy_scenario, method, parameters
):
    result = evaluate_corridor_episode(toy_scenario, method, parameters=parameters)
    assert result.outcome == "success"
    assert result.task_status == "success"
    assert 1 <= result.steps <= toy_scenario.spec.max_steps
    assert result.control_effort >= 1
    assert result.decision_count == result.steps
    assert result.wall_seconds >= 0.0
    assert result.wall_seconds_per_decision is not None
    assert len(result.layout_sha256) == 64


def test_episode_runner_reports_geometric_no_route_as_failure():
    spec = CorridorTaskSpec(
        width=1.0,
        height=1.0,
        obstacles=(AxisAlignedObstacle(0.48, 0.0, 0.52, 1.0),),
        goal=(0.8, 0.5),
        goal_radius=0.06,
        safe_speed=0.1,
    )
    scenario = CorridorScenario(
        seed=772,
        layout_family="unit_fixture",
        layout_sha256="b" * 64,
        spec=spec,
        initial_state=CorridorState(0.2, 0.5),
    )
    result = evaluate_corridor_episode(scenario, "geometry_waypoint")
    assert result.outcome == "planner_no_path"
    assert result.planner_failure == "no_geometric_route"
    assert result.steps == 0
    assert result.decision_count == 1


def test_episode_runner_counts_kinodynamic_expansion_limit():
    from consequence_lab.corridor_tasks import generate_staggered_corridor

    scenario = generate_staggered_corridor(773)
    result = evaluate_corridor_episode(
        scenario, "kinodynamic", parameters={"max_expansions": 1}
    )
    assert result.outcome == "planner_expansion_limit"
    assert result.kinodynamic_expansion_limit_queries == 1
    assert result.planner_failure == "expansion_limit"
    assert result.steps == 0


def test_episode_runner_rejects_unknown_method_or_parameter(toy_scenario):
    with pytest.raises(ValueError, match="unknown corridor method"):
        evaluate_corridor_episode(toy_scenario, "not_a_method")
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_corridor_episode(
            toy_scenario, "reactive", parameters={"secret_future": 1}
        )


@pytest.mark.parametrize(
    ("method", "parameters"),
    [
        ("kinodynamic", {"max_expansions": 1.5}),
        ("random_shooting", {"horizon": True}),
        ("random_shooting", {"num_sequences": 3.0}),
    ],
)
def test_episode_runner_does_not_silently_coerce_integer_parameters(
    toy_scenario, method, parameters
):
    with pytest.raises(ValueError, match="must be an integer"):
        evaluate_corridor_episode(toy_scenario, method, parameters=parameters)


def test_episode_runner_validates_phase_space_parameters(toy_scenario):
    with pytest.raises(ValueError, match="brake_safety_margin"):
        evaluate_corridor_episode(
            toy_scenario, "phase_space", parameters={"brake_safety_margin": float("nan")}
        )


def test_paired_block_runner_pairs_same_layouts_in_stable_order(toy_scenario):
    results = evaluate_corridor_block(
        [toy_scenario],
        ["reactive", "random_shooting"],
        parameters_by_method={"random_shooting": {"horizon": 1, "num_sequences": 9}},
    )
    assert [(item.seed, item.method) for item in results] == [
        (toy_scenario.seed, "reactive"),
        (toy_scenario.seed, "random_shooting"),
    ]
    assert all(item.outcome == "success" for item in results)
    assert {item.layout_sha256 for item in results} == {toy_scenario.layout_sha256}


def test_paired_block_rejects_duplicate_geometry(toy_scenario):
    duplicate = CorridorScenario(
        seed=toy_scenario.seed + 1,
        layout_family=toy_scenario.layout_family,
        layout_sha256=toy_scenario.layout_sha256,
        spec=toy_scenario.spec,
        initial_state=toy_scenario.initial_state,
    )
    with pytest.raises(ValueError, match="layout hashes must be unique"):
        evaluate_corridor_block([toy_scenario, duplicate], ["reactive"])


def test_paired_block_rotates_method_order_across_layouts(toy_scenario):
    second = CorridorScenario(
        seed=toy_scenario.seed + 1,
        layout_family=toy_scenario.layout_family,
        layout_sha256="c" * 64,
        spec=toy_scenario.spec,
        initial_state=toy_scenario.initial_state,
    )
    results = evaluate_corridor_block(
        [toy_scenario, second], ["reactive", "geometry_waypoint"]
    )
    assert [(item.seed, item.method) for item in results] == [
        (toy_scenario.seed, "reactive"),
        (toy_scenario.seed, "geometry_waypoint"),
        (second.seed, "geometry_waypoint"),
        (second.seed, "reactive"),
    ]


def _synthetic_result(method, layout, outcome):
    return CorridorEpisodeResult(
        method=method,
        seed=int(layout[-2:], 16),
        layout_sha256=layout,
        outcome=outcome,
        task_status=outcome if outcome in {"success", "collision", "timeout"} else "active",
        steps=5,
        control_effort=2,
        decision_count=5,
        wall_seconds=0.01,
        wall_seconds_per_decision=0.002,
        shooting_queries=0,
        shooting_success_available_queries=0,
        kinodynamic_expansion_limit_queries=0,
        planner_failure=(outcome.removeprefix("planner_") if outcome.startswith("planner_") else None),
    )


def test_paired_bootstrap_uses_layout_clusters_and_is_reproducible():
    results = [
        _synthetic_result("planner", "0" * 62 + "01", "success"),
        _synthetic_result("cheap", "0" * 62 + "01", "success"),
        _synthetic_result("planner", "0" * 62 + "02", "success"),
        _synthetic_result("cheap", "0" * 62 + "02", "timeout"),
        _synthetic_result("planner", "0" * 62 + "03", "collision"),
        _synthetic_result("cheap", "0" * 62 + "03", "collision"),
    ]
    first = paired_success_gain_interval(results, "planner", "cheap", resamples=200)
    second = paired_success_gain_interval(results, "planner", "cheap", resamples=200)
    assert first == second
    assert first.paired_rate_difference == pytest.approx(1 / 3)
    assert first.event == "success"
    assert first.independent_layouts == 3
    assert first.lower_95 <= first.paired_rate_difference <= first.upper_95


def test_paired_bootstrap_can_measure_collision_noninferiority():
    results = [
        _synthetic_result("planner", "0" * 62 + "01", "success"),
        _synthetic_result("cheap", "0" * 62 + "01", "collision"),
        _synthetic_result("planner", "0" * 62 + "02", "collision"),
        _synthetic_result("cheap", "0" * 62 + "02", "success"),
    ]
    interval = paired_outcome_rate_interval(
        results, "planner", "cheap", event="collision", resamples=200
    )
    assert interval.event == "collision"
    assert interval.paired_rate_difference == pytest.approx(0.0)
    assert interval.lower_95 == pytest.approx(-1.0)
    assert interval.upper_95 == pytest.approx(1.0)


def test_paired_bootstrap_rejects_unmatched_or_duplicate_layouts():
    results = [
        _synthetic_result("planner", "0" * 62 + "01", "success"),
        _synthetic_result("cheap", "0" * 62 + "02", "success"),
    ]
    with pytest.raises(ValueError, match="matching nonempty"):
        paired_success_gain_interval(results, "planner", "cheap")
    results.append(_synthetic_result("planner", "0" * 62 + "01", "timeout"))
    with pytest.raises(ValueError, match="duplicate"):
        paired_success_gain_interval(results, "planner", "cheap")


def test_summary_aggregates_rates_and_requires_unique_method_layout():
    results = [
        _synthetic_result("planner", "0" * 62 + "01", "success"),
        _synthetic_result("planner", "0" * 62 + "02", "planner_expansion_limit"),
    ]
    summary = summarize_corridor_results(results)[0]
    assert summary.method == "planner"
    assert summary.episodes == 2
    assert summary.safe_success_rate == 0.5
    assert summary.planner_failure_rate == 0.5
    with pytest.raises(ValueError, match="duplicate"):
        summarize_corridor_results(results + [results[0]])
