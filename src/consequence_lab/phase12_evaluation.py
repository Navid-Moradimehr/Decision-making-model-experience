"""Closed-loop evaluation for frozen Phase-12 state/candidate students."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from typing import Any

import numpy as np
import torch

from consequence_lab.corridor_baselines import grid_astar_route
from consequence_lab.corridor_evaluation import (
    CorridorEpisodeResult, evaluate_corridor_block, paired_success_gain_interval,
    summarize_corridor_results,
)
from consequence_lab.corridor_tasks import CorridorScenario
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase12_teacher import (
    candidate_features, make_teacher_query, phase12_query_seed, propose_sequences, state_features,
)
from consequence_lab.phase12_training import evaluate_offline


def load_student(checkpoint_path: Path, *, device: str = "cpu"):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = __import__("consequence_lab.phase12_model", fromlist=["CorridorCandidateScorer"]).CorridorCandidateScorer(payload["kind"])
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device)).eval()
    return model, payload


def evaluate_student_episode(
    scenario: CorridorScenario, model: torch.nn.Module, *, kind: str,
) -> tuple[CorridorEpisodeResult, dict[str, Any]]:
    state = scenario.initial_state
    effort = decisions = 0
    latencies_ms: list[float] = []
    teacher_disagreements = action_disagreements = 0
    regret_on_disagreement: list[float] = []
    all_regrets: list[float] = []
    entropies: list[float] = []
    successes_available = 0
    counterfactual_wall = 0.0
    while state.status == "active":
        started = perf_counter_ns()
        query_seed = phase12_query_seed(scenario.seed, state.elapsed_steps)
        sequences = propose_sequences(state, scenario.spec.max_steps, seed=query_seed)
        state_array = state_features(scenario, state)
        candidate_array = candidate_features(sequences)
        state_tensor = torch.from_numpy(state_array).unsqueeze(0)
        candidate_tensor = torch.from_numpy(candidate_array).unsqueeze(0)
        candidate_mask = torch.ones((1, len(sequences)), dtype=torch.bool)
        with torch.no_grad():
            scores = model(state_tensor, candidate_tensor, candidate_mask)[0].cpu().numpy()
            if kind == "value":
                preference_logits = -scores
                best_score = scores.min()
            else:
                preference_logits = scores
                best_score = scores.max()
        tied = np.flatnonzero(scores == best_score)
        selected = min(tied.tolist(), key=lambda index: tuple(int(x) for x in sequences[index]))
        selected_action = int(sequences[selected, 0])
        next_state = corridor_step(state, selected_action, scenario.spec).state
        latency_ms = (perf_counter_ns() - started) / 1_000_000.0
        latencies_ms.append(latency_ms)
        effort += int(selected_action != 0)
        decisions += 1

        # Counterfactual exact teacher query is audit-only and excluded from
        # student decision latency. It sees the current state, not the outcome.
        audit_started = perf_counter_ns()
        teacher = make_teacher_query(scenario, state)
        counterfactual_wall += (perf_counter_ns() - audit_started) / 1_000_000_000.0
        sequence_disagreement = selected != teacher.selected_index
        action_disagreement = selected_action != int(teacher.sequences[teacher.selected_index, 0])
        regret = float(teacher.normalized_regret[selected])
        teacher_disagreements += int(sequence_disagreement)
        action_disagreements += int(action_disagreement)
        all_regrets.append(regret)
        successes_available += int(teacher.success_available)
        logits = preference_logits - np.max(preference_logits)
        probs = np.exp(logits) / np.exp(logits).sum()
        entropies.append(float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum()))
        if sequence_disagreement:
            regret_on_disagreement.append(regret)
        state = next_state

    wall_seconds = sum(latencies_ms) / 1000.0
    result = CorridorEpisodeResult(
        method="student", seed=scenario.seed, layout_sha256=scenario.layout_sha256,
        outcome=state.status, task_status=state.status,
        steps=state.elapsed_steps - scenario.initial_state.elapsed_steps,
        control_effort=effort, decision_count=decisions,
        wall_seconds=wall_seconds,
        wall_seconds_per_decision=(wall_seconds / decisions if decisions else None),
        shooting_queries=0, shooting_success_available_queries=0,
        kinodynamic_expansion_limit_queries=0, planner_failure=None,
    )
    diagnostics = {
        "latency_ms_per_decision": latencies_ms,
        "median_student_latency_ms": median(latencies_ms) if latencies_ms else None,
        "teacher_sequence_disagreement_count": teacher_disagreements,
        "teacher_sequence_disagreement_rate": teacher_disagreements / decisions if decisions else 0.0,
        "teacher_first_action_disagreement_count": action_disagreements,
        "teacher_first_action_disagreement_rate": action_disagreements / decisions if decisions else 0.0,
        "mean_selected_normalized_teacher_regret": float(np.mean(all_regrets)) if all_regrets else None,
        "mean_regret_at_sequence_disagreement": float(np.mean(regret_on_disagreement)) if regret_on_disagreement else 0.0,
        "mean_policy_entropy": float(np.mean(entropies)) if entropies else None,
        "teacher_success_candidate_availability": successes_available / decisions if decisions else 0.0,
        "counterfactual_teacher_audit_wall_seconds_excluded_from_latency": counterfactual_wall,
    }
    return result, diagnostics


def run_phase12_development(
    scenarios: list[CorridorScenario], model: torch.nn.Module, *, kind: str,
    parameters: dict[str, dict[str, float | int]] | None = None,
) -> dict[str, Any]:
    if len(scenarios) != 40 or len({item.layout_sha256 for item in scenarios}) != 40:
        raise ValueError("Phase 12 development requires exactly 40 unique frozen layouts")
    base_results = list(evaluate_corridor_block(
        scenarios, ("geometry_waypoint", "random_shooting"),
        parameters_by_method=parameters or {
            "geometry_waypoint": {"resolution": 0.04, "waypoint_tolerance": 0.03},
            "random_shooting": {"horizon": 8, "num_sequences": 128},
        },
    ))
    student_rows = []
    student_diagnostics = []
    for scenario in scenarios:
        outcome, diagnostic = evaluate_student_episode(scenario, model, kind=kind)
        student_rows.append(outcome)
        student_diagnostics.append({
            "seed": scenario.seed, "layout_sha256": scenario.layout_sha256, **diagnostic,
        })
    all_results = base_results + student_rows
    summaries = [asdict(item) for item in summarize_corridor_results(all_results)]
    student_geometry = asdict(paired_success_gain_interval(all_results, "student", "geometry_waypoint", bootstrap_seed=143011, resamples=10000))
    student_planner = asdict(paired_success_gain_interval(all_results, "student", "random_shooting", bootstrap_seed=143011, resamples=10000))
    collision_gain = asdict(__import__("consequence_lab.corridor_evaluation", fromlist=["paired_outcome_rate_interval"]).paired_outcome_rate_interval(
        all_results, "student", "geometry_waypoint", event="collision", bootstrap_seed=143012, resamples=10000,
    ))
    summary_by_method = {item["method"]: item for item in summaries}
    student = summary_by_method["student"]
    exact = summary_by_method["random_shooting"]
    gain = student_geometry["paired_rate_difference"]
    offline_trigger = (
        gain >= 0.10 and student_geometry["lower_95"] > 0.0
        and collision_gain["upper_95"] <= 0.05
        and student["median_wall_seconds_per_decision"] is not None
        and exact["median_wall_seconds_per_decision"] is not None
        and student["median_wall_seconds_per_decision"] <= exact["median_wall_seconds_per_decision"] / 3.0
    )
    return {
        "schema": "phase12_closed_loop_development_v1",
        "episodes": len(scenarios),
        "methods": summaries,
        "student_vs_geometry_paired_success": student_geometry,
        "student_vs_exact_shooting_paired_success": student_planner,
        "student_vs_geometry_paired_collision": collision_gain,
        "student_diagnostics": student_diagnostics,
        "episode_results": [asdict(item) for item in all_results],
        "dagger_trigger_passed": offline_trigger,
        "reserve_accessed": False,
        "training_performed": False,
    }
