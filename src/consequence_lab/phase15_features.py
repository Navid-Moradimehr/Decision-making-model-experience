"""Leakage-explicit feature construction for the locked Phase15 offline study.

R1/R2/R3 are privileged simulator diagnostics, not deployable model inputs.
R4 additionally includes exact planner-continuation outcomes and is an oracle
diagnostic. This module never generates episodes or invokes a planner.
"""

from __future__ import annotations

from math import hypot
from typing import Any

import numpy as np

from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import CorridorState, corridor_step
from consequence_lab.phase12_teacher import state_features
from consequence_lab.phase14c import build_feature_matrix
from consequence_lab.phase15 import Phase15AuditError, validate_trace_branch_contract


REPRESENTATIONS = ("R0", "R1", "R2", "R3", "R4")
R2_HORIZON = 4
_OUTCOME_IDS = {"success": 0, "active": 1, "timeout": 2, "collision": 3}


def _clearance(x: float, y: float, scenario: Any) -> float:
    """Signed center clearance after accounting for the circular agent radius."""
    spec = scenario.spec
    radius = spec.agent_radius
    result = min(x - radius, spec.width - radius - x,
                 y - radius, spec.height - radius - y)
    for obstacle in spec.obstacles:
        dx = max(obstacle.x_min - x, 0.0, x - obstacle.x_max)
        dy = max(obstacle.y_min - y, 0.0, y - obstacle.y_max)
        if dx == 0.0 and dy == 0.0:
            # Interior signed distance: negative distance to the nearest face.
            center_clearance = -min(x - obstacle.x_min, obstacle.x_max - x,
                                    y - obstacle.y_min, obstacle.y_max - y)
        else:
            center_clearance = hypot(dx, dy)
        result = min(result, center_clearance - radius)
    return float(result)


def _goal_distance(x: float, y: float, scenario: Any) -> float:
    return hypot(scenario.spec.goal[0] - x, scenario.spec.goal[1] - y)


def _event_bits(status: str) -> tuple[float, float, float]:
    return (float(status == "collision"), float(status == "success"), float(status == "timeout"))


def _one_step_block(branch: dict[str, Any], source_xy: tuple[float, float], scenario: Any) -> np.ndarray:
    spec = scenario.spec
    x, y, vx, vy = map(float, branch["forced_successor"])
    # Inputs and physical summaries are normalized by public task constants.
    diagonal = hypot(spec.width, spec.height)
    goal_progress = (_goal_distance(source_xy[0], source_xy[1], scenario) - _goal_distance(x, y, scenario)) / diagonal
    return np.asarray((
        x / spec.width, y / spec.height, vx / spec.max_speed, vy / spec.max_speed,
        _goal_distance(x, y, scenario) / diagonal, goal_progress,
        hypot(vx, vy) / spec.max_speed, _clearance(x, y, scenario) / diagonal,
        *_event_bits(str(branch["terminal_status"]) if branch["forced_collision"] or branch["forced_success"] or branch["forced_timeout"] else "active"),
    ), dtype=np.float64)


def _pulse_features(trace: dict[str, Any], scenario: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """Return all sampled R2 step features and the compact R3 action blocks."""
    spec = scenario.spec
    source = trace["state_before"]
    initial = CorridorState(float(source[0]), float(source[1]), float(source[2]), float(source[3]),
                            int(source[4]), str(source[5]))
    diagonal = hypot(spec.width, spec.height)
    pulse_by_action: list[np.ndarray] = []
    compact: list[list[float]] = []
    additional_transitions = 0
    for branch in trace["teacher_branches"]:
        action = int(branch["forced_action_id"])
        # Reuse the already-saved exact forced first transition, then advance
        # at most three more times. This both avoids recomputing the saved step
        # and makes R2's transition budget explicit.
        sx, sy, svx, svy = map(float, branch["forced_successor"])
        if branch["forced_collision"]:
            first_status = "collision"
        elif branch["forced_success"]:
            first_status = "success"
        elif branch["forced_timeout"]:
            first_status = "timeout"
        else:
            first_status = "active"
        current = CorridorState(sx, sy, svx, svy, initial.elapsed_steps + 1, first_status)
        step_rows: list[list[float]] = []
        clearances: list[float] = []
        distances: list[float] = []
        for step_index in range(R2_HORIZON):
            if step_index > 0 and current.status == "active":
                transition = corridor_step(current, action, spec)
                current = transition.state
                additional_transitions += 1
            clearance = _clearance(current.x, current.y, scenario)
            distance = _goal_distance(current.x, current.y, scenario)
            clearances.append(clearance)
            distances.append(distance)
            # State, goal distance, clearance, speed, and active/terminal mask.
            step_rows.append([
                current.x / spec.width, current.y / spec.height,
                current.vx / spec.max_speed, current.vy / spec.max_speed,
                distance / diagonal, clearance / diagonal,
                hypot(current.vx, current.vy) / spec.max_speed,
                float(current.status == "active"), float(current.status == "success"),
                float(current.status == "collision" or current.status == "timeout"),
            ])
        block = np.asarray(step_rows, dtype=np.float64)
        pulse_by_action.append(block.reshape(-1))
        velocity = np.asarray((current.vx, current.vy), dtype=np.float64)
        goal_vec = np.asarray((spec.goal[0] - current.x, spec.goal[1] - current.y), dtype=np.float64)
        denom = hypot(*velocity) * hypot(*goal_vec)
        alignment = float(np.dot(velocity, goal_vec) / denom) if denom > 1e-12 else 0.0
        stopping_distance = hypot(*velocity) ** 2 / (2.0 * spec.acceleration)
        compact.append([
            min(clearances) / diagonal,
            (distances[0] - distances[-1]) / diagonal,
            hypot(current.vx, current.vy) / spec.max_speed,
            alignment,
            float(current.status == "collision"), float(current.status == "success"),
            float(current.status == "timeout"),
            (min(clearances) - stopping_distance) / diagonal,
        ])
    return np.asarray(pulse_by_action), np.asarray(compact), additional_transitions


def build_phase15_representations(traces: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Build R0-R4 matrices for exactly the supplied already-saved traces.

    R2 incurs up to three additional deterministic simulator transitions per
    saved action branch after the already-saved forced first step. It does not
    rerun Phase14G episodes, exact planners, or touch other layouts.
    """
    if not traces:
        raise ValueError("Phase15 representation construction needs saved traces")
    representations: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATIONS}
    targets: list[np.ndarray] = []
    additional_transitions = 0
    for trace in traces:
        validate_trace_branch_contract(trace)
        scenario = generate_staggered_corridor(int(trace["episode_seed"]))
        if scenario.layout_sha256 != trace["layout_sha256"]:
            raise Phase15AuditError("trace seed does not reconstruct its locked public layout")
        raw = np.asarray(trace["student_runtime_input_52"], dtype=np.float64)
        if raw.shape != (52,) or not np.isfinite(raw).all():
            raise Phase15AuditError("saved runtime state must be finite 52D input")
        source = trace["state_before"]
        rebuilt = state_features(scenario, CorridorState(
            float(source[0]), float(source[1]), float(source[2]), float(source[3]),
            int(source[4]), str(source[5]),
        )).astype(np.float64)
        if not np.allclose(rebuilt, raw, rtol=0.0, atol=2e-7):
            raise Phase15AuditError("saved runtime state disagrees with reconstructed public state")
        base = build_feature_matrix(raw[None, :], "raw52_plus_physics9")[0]
        r1_blocks = []
        r4_blocks = []
        for branch in trace["teacher_branches"]:
            if (int(branch["source_episode_seed"]) != int(trace["episode_seed"])
                    or branch["source_layout_sha256"] != trace["layout_sha256"]
                    or int(branch["source_decision_index"]) != int(trace["decision_index"])):
                raise Phase15AuditError("branch provenance does not match its saved source trace")
            successor = _one_step_block(branch, (float(source[0]), float(source[1])), scenario)
            if branch["forced_collision"]:
                status = "collision"
            elif branch["forced_success"]:
                status = "success"
            elif branch["forced_timeout"]:
                status = "timeout"
            else:
                status = "active"
            terminal_status = str(branch["terminal_status"])
            terminal_code = np.eye(4, dtype=np.float64)[_OUTCOME_IDS[terminal_status]]
            terminal_steps = float(branch["terminal_steps_from_source"]) / scenario.spec.max_steps
            terminal_distance = float(branch["terminal_goal_distance"]) / hypot(scenario.spec.width, scenario.spec.height)
            effort = float(branch["total_noncoast_effort"]) / scenario.spec.max_steps
            r1_blocks.append(successor)
            r4_blocks.append(np.concatenate((terminal_code, (terminal_steps, terminal_distance, effort))))
            # Saved branch terminal labels are outputs for R4 only. Check local
            # one-step event indicators against saved forced-event flags.
            assert tuple(successor[-3:]) == _event_bits(status)
        pulse, compact, trace_transitions = _pulse_features(trace, scenario)
        additional_transitions += trace_transitions
        representations["R0"].append(base)
        representations["R1"].append(np.concatenate((base, np.asarray(r1_blocks).reshape(-1))))
        representations["R2"].append(np.concatenate((base, pulse.reshape(-1))))
        representations["R3"].append(np.concatenate((base, compact.reshape(-1))))
        representations["R4"].append(np.concatenate((base, np.asarray(r4_blocks).reshape(-1))))
        targets.append(np.asarray([float(branch["normalized_regret"]) for branch in trace["teacher_branches"]]))
    matrices = {name: np.stack(rows) for name, rows in representations.items()}
    y = np.stack(targets)
    if any(not np.isfinite(value).all() for value in (*matrices.values(), y)):
        raise Phase15AuditError("non-finite Phase15 feature/target values")
    return matrices, y, additional_transitions
