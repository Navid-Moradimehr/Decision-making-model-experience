"""Common episode runner for Phase 11 baseline comparisons.

This module defines reproducible evaluation mechanics only. It does not read
the phase seed-block config, launch seed sweeps, persist outcomes, or authorize
performance evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import random
from statistics import median
from time import perf_counter_ns
from typing import Literal, Mapping, Sequence

from consequence_lab.corridor_baselines import (
    exact_dynamics_shooting,
    grid_astar_route,
    phase_space_route_action,
    quantized_kinodynamic_first_action,
    reactive_goal_action,
    waypoint_braking_action,
)
from consequence_lab.corridor_tasks import CorridorScenario
from consequence_lab.inertial_corridor import corridor_step, validate_corridor_state


MethodName = Literal[
    "reactive",
    "geometry_waypoint",
    "phase_space",
    "kinodynamic",
    "random_shooting",
]


@dataclass(frozen=True)
class CorridorEpisodeResult:
    """Compact result for one method/scenario pair; no trajectory is retained."""

    method: str
    seed: int
    layout_sha256: str
    outcome: str
    task_status: str
    steps: int
    control_effort: int
    decision_count: int
    wall_seconds: float
    wall_seconds_per_decision: float | None
    shooting_queries: int
    shooting_success_available_queries: int
    kinodynamic_expansion_limit_queries: int
    planner_failure: str | None


@dataclass(frozen=True)
class PairedGainInterval:
    method_a: str
    method_b: str
    event: str
    paired_rate_difference: float
    lower_95: float
    upper_95: float
    independent_layouts: int


@dataclass(frozen=True)
class CorridorMethodSummary:
    method: str
    episodes: int
    safe_success_rate: float
    collision_rate: float
    timeout_rate: float
    planner_failure_rate: float
    median_steps_on_success: float | None
    median_wall_seconds_per_decision: float | None
    mean_wall_seconds_per_episode: float
    shooting_success_availability: float | None
    kinodynamic_expansion_limit_queries: int


def evaluate_corridor_episode(
    scenario: CorridorScenario,
    method: MethodName,
    *,
    parameters: Mapping[str, float | int] | None = None,
    proposal_seed: int | None = None,
) -> CorridorEpisodeResult:
    """Run one baseline closed-loop on one scenario and return summary metrics.

    Supported methods are ``reactive``, ``geometry_waypoint``, ``phase_space``,
    ``kinodynamic`` and ``random_shooting``. Timing uses a high-resolution monotonic wall clock
    and includes decision computation and the exact simulator step. It measures
    observed latency on the current host, not hardware-independent compute. The
    function has no file or global-state side effects.
    """
    if method not in {"reactive", "geometry_waypoint", "phase_space", "kinodynamic", "random_shooting"}:
        raise ValueError(f"unknown corridor method: {method}")
    params = dict(parameters or {})
    allowed = {
        "reactive": {"obstacle_influence", "clearance_weight"},
        "geometry_waypoint": {"resolution", "waypoint_tolerance"},
        "phase_space": {"resolution", "brake_safety_margin", "corner_angle_degrees"},
        "kinodynamic": {
            "position_resolution", "velocity_resolution", "max_expansions"
        },
        "random_shooting": {"horizon", "num_sequences"},
    }[method]
    unexpected = set(params) - allowed
    if unexpected:
        raise ValueError(f"unsupported {method} parameters: {sorted(unexpected)}")
    integer_parameters = {
        "kinodynamic": {"max_expansions"},
        "random_shooting": {"horizon", "num_sequences"},
    }.get(method, set())
    for name in integer_parameters & set(params):
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if proposal_seed is None:
        proposal_seed = scenario.seed
    if isinstance(proposal_seed, bool) or not isinstance(proposal_seed, Integral) or proposal_seed < 0:
        raise ValueError("proposal_seed must be a nonnegative integer")

    state = scenario.initial_state
    spec = scenario.spec
    validate_corridor_state(state, spec)
    route = None
    started_wall_ns = perf_counter_ns()
    if method in {"geometry_waypoint", "phase_space"}:
        route = grid_astar_route(
            (state.x, state.y),
            spec.goal,
            spec,
            resolution=float(params.get("resolution", 0.02)),
        )
        if route is None:
            elapsed_wall = (perf_counter_ns() - started_wall_ns) / 1_000_000_000
            return CorridorEpisodeResult(
                method, scenario.seed, scenario.layout_sha256,
                "planner_no_path", state.status, 0, 0, 1, elapsed_wall, elapsed_wall,
                0, 0, 0, "no_geometric_route",
            )

    effort = 0
    decision_count = 0
    shooting_queries = 0
    shooting_success_available = 0
    expansion_limit_queries = 0
    planner_failure: str | None = None

    while state.status == "active":
        if method == "reactive":
            action = reactive_goal_action(
                state,
                spec,
                obstacle_influence=float(params.get("obstacle_influence", 0.10)),
                clearance_weight=float(params.get("clearance_weight", 0.01)),
            )
        elif method == "geometry_waypoint":
            assert route is not None
            action = waypoint_braking_action(
                state,
                route,
                spec,
                waypoint_tolerance=float(params.get("waypoint_tolerance", 0.015)),
            )
        elif method == "phase_space":
            assert route is not None
            action = phase_space_route_action(
                state,
                route,
                spec,
                brake_safety_margin=float(params.get("brake_safety_margin", 0.04)),
                corner_angle_degrees=float(params.get("corner_angle_degrees", 30.0)),
            )
        elif method == "kinodynamic":
            search = quantized_kinodynamic_first_action(
                state,
                spec,
                position_resolution=float(params.get("position_resolution", 0.04)),
                velocity_resolution=float(params.get("velocity_resolution", 0.04)),
                max_expansions=int(params.get("max_expansions", 20_000)),
            )
            decision_count += 1
            if search.status != "success" or search.first_action is None:
                if search.status == "expansion_limit":
                    expansion_limit_queries += 1
                planner_failure = search.status
                break
            action = search.first_action
        else:
            query_seed = (int(proposal_seed) * 1_000_003 + state.elapsed_steps) % (2**32)
            shooting = exact_dynamics_shooting(
                state,
                spec,
                horizon=int(params.get("horizon", 12)),
                num_sequences=int(params.get("num_sequences", 256)),
                seed=query_seed,
            )
            decision_count += 1
            shooting_queries += 1
            shooting_success_available += int(shooting.candidate_success_available)
            action = shooting.first_action

        decision_count += int(method in {"reactive", "geometry_waypoint", "phase_space"})
        effort += int(action != 0)
        state = corridor_step(state, action, spec).state

    wall_seconds = (perf_counter_ns() - started_wall_ns) / 1_000_000_000
    if planner_failure is not None:
        outcome = "planner_expansion_limit" if planner_failure == "expansion_limit" else "planner_no_path"
    else:
        outcome = state.status
    return CorridorEpisodeResult(
        method=method,
        seed=scenario.seed,
        layout_sha256=scenario.layout_sha256,
        outcome=outcome,
        task_status=state.status,
        steps=state.elapsed_steps - scenario.initial_state.elapsed_steps,
        control_effort=effort,
        decision_count=decision_count,
        wall_seconds=wall_seconds,
        wall_seconds_per_decision=(
            wall_seconds / decision_count if decision_count else None
        ),
        shooting_queries=shooting_queries,
        shooting_success_available_queries=shooting_success_available,
        kinodynamic_expansion_limit_queries=expansion_limit_queries,
        planner_failure=planner_failure,
    )


def evaluate_corridor_block(
    scenarios: Sequence[CorridorScenario],
    methods: Sequence[MethodName],
    *,
    parameters_by_method: Mapping[str, Mapping[str, float | int]] | None = None,
) -> tuple[CorridorEpisodeResult, ...]:
    """Evaluate each method on every scenario in deterministic paired order.

    This is an execution primitive only. It neither reads nor enforces the
    draft seed protocol, writes artifacts, nor provides authorization to run a
    seed block. Callers must perform protocol/hash checks before invoking it.
    Duplicate seeds/layouts or method names are rejected so one outcome per
    independent layout and method is explicit.
    """
    scenarios = tuple(scenarios)
    methods = tuple(methods)
    if not scenarios:
        raise ValueError("at least one scenario is required")
    if not methods:
        raise ValueError("at least one method is required")
    if len({scenario.seed for scenario in scenarios}) != len(scenarios):
        raise ValueError("scenario seeds must be unique within a block")
    if len({scenario.layout_sha256 for scenario in scenarios}) != len(scenarios):
        raise ValueError("layout hashes must be unique within a block")
    if len(set(methods)) != len(methods):
        raise ValueError("method names must be unique")
    parameter_map = parameters_by_method or {}
    if set(parameter_map) - set(methods):
        raise ValueError("parameters supplied for a method outside the evaluation block")
    results = []
    for scenario_index, scenario in enumerate(scenarios):
        rotation = scenario_index % len(methods)
        ordered_methods = methods[rotation:] + methods[:rotation]
        for method in ordered_methods:
            results.append(
                evaluate_corridor_episode(
                    scenario,
                    method,
                    parameters=parameter_map.get(method),
                )
            )
    return tuple(results)


def summarize_corridor_results(
    results: Sequence[CorridorEpisodeResult],
) -> tuple[CorridorMethodSummary, ...]:
    """Summarize one method per layout; reject duplicate/missing labels."""
    results = tuple(results)
    if not results:
        raise ValueError("at least one result is required")
    by_method: dict[str, list[CorridorEpisodeResult]] = {}
    seen: set[tuple[str, str]] = set()
    for result in results:
        key = (result.method, result.layout_sha256)
        if key in seen:
            raise ValueError("duplicate method/layout result")
        seen.add(key)
        by_method.setdefault(result.method, []).append(result)
    summaries = []
    for method, group in sorted(by_method.items()):
        outcomes = [item.outcome for item in group]
        successful_steps = [item.steps for item in group if item.outcome == "success"]
        wall_decision_times = [
            item.wall_seconds_per_decision
            for item in group
            if item.wall_seconds_per_decision is not None
        ]
        shooting_total = sum(item.shooting_queries for item in group)
        shooting_available = sum(item.shooting_success_available_queries for item in group)
        summaries.append(
            CorridorMethodSummary(
                method=method,
                episodes=len(group),
                safe_success_rate=outcomes.count("success") / len(group),
                collision_rate=outcomes.count("collision") / len(group),
                timeout_rate=outcomes.count("timeout") / len(group),
                planner_failure_rate=sum(item.planner_failure is not None for item in group) / len(group),
                median_steps_on_success=(median(successful_steps) if successful_steps else None),
                median_wall_seconds_per_decision=(
                    median(wall_decision_times) if wall_decision_times else None
                ),
                mean_wall_seconds_per_episode=sum(item.wall_seconds for item in group) / len(group),
                shooting_success_availability=(
                    shooting_available / shooting_total if shooting_total else None
                ),
                kinodynamic_expansion_limit_queries=sum(
                    item.kinodynamic_expansion_limit_queries for item in group
                ),
            )
        )
    return tuple(summaries)


def paired_outcome_rate_interval(
    results: Sequence[CorridorEpisodeResult],
    method_a: str,
    method_b: str,
    *,
    event: Literal["success", "collision", "timeout", "planner_failure"] = "success",
    bootstrap_seed: int = 110011,
    resamples: int = 10_000,
) -> PairedGainInterval:
    """Paired percentile bootstrap of a binary episode rate difference.

    There must be exactly one outcome per method/layout. The bootstrap samples
    layout hashes with replacement, keeping method outcomes paired. This
    supports the proposed one-episode-per-geometry protocol; repeated episodes
    per geometry need a hierarchical bootstrap and are intentionally rejected.
    ``event`` supports safe success, collision, timeout, or planner failure.
    """
    if method_a == method_b:
        raise ValueError("paired methods must be distinct")
    if event not in {"success", "collision", "timeout", "planner_failure"}:
        raise ValueError("unsupported binary outcome event")
    if isinstance(resamples, bool) or not isinstance(resamples, Integral) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, Integral) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    values: dict[str, dict[str, int]] = {method_a: {}, method_b: {}}
    for result in results:
        if result.method not in values:
            continue
        if result.layout_sha256 in values[result.method]:
            raise ValueError("duplicate method/layout result")
        if event == "planner_failure":
            value = int(result.planner_failure is not None)
        else:
            value = int(result.outcome == event)
        values[result.method][result.layout_sha256] = value
    hashes_a, hashes_b = set(values[method_a]), set(values[method_b])
    if not hashes_a or hashes_a != hashes_b:
        raise ValueError("methods must have matching nonempty layout sets")
    layout_hashes = sorted(hashes_a)
    paired = [
        values[method_a][layout_hash] - values[method_b][layout_hash]
        for layout_hash in layout_hashes
    ]
    observed = sum(paired) / len(paired)
    rng = random.Random(int(bootstrap_seed))
    bootstrap_values = []
    for _ in range(int(resamples)):
        bootstrap_values.append(
            sum(paired[rng.randrange(len(paired))] for _ in paired) / len(paired)
        )
    bootstrap_values.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(bootstrap_values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(bootstrap_values) - 1)
        fraction = position - lower
        return bootstrap_values[lower] * (1.0 - fraction) + bootstrap_values[upper] * fraction

    return PairedGainInterval(
        method_a=method_a,
        method_b=method_b,
        event=event,
        paired_rate_difference=observed,
        lower_95=percentile(0.025),
        upper_95=percentile(0.975),
        independent_layouts=len(layout_hashes),
    )


def paired_success_gain_interval(
    results: Sequence[CorridorEpisodeResult],
    method_a: str,
    method_b: str,
    *,
    bootstrap_seed: int = 110011,
    resamples: int = 10_000,
) -> PairedGainInterval:
    """Convenience wrapper for the paired safe-success-rate difference."""
    return paired_outcome_rate_interval(
        results,
        method_a,
        method_b,
        event="success",
        bootstrap_seed=bootstrap_seed,
        resamples=resamples,
    )
