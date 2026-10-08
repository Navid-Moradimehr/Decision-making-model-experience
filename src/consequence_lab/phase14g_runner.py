"""Locked Phase14G trace-attribution run; one authorized block, no training."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from time import perf_counter_ns
from typing import Any

import numpy as np
import yaml

from consequence_lab.corridor_baselines import exact_dynamics_shooting
from consequence_lab.corridor_evaluation import (
    CorridorEpisodeResult,
    evaluate_corridor_episode,
    paired_outcome_rate_interval,
    summarize_corridor_results,
)
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step, validate_corridor_state
from consequence_lab.phase12_teacher import phase12_query_seed, state_features
from consequence_lab.phase14 import add_state_ranks, sha256_file
from consequence_lab.phase14f import FrozenCartPolicy
from consequence_lab.phase14g import (
    Phase14GIntegrityError,
    build_support_counts,
    forced_first_action_values,
    trace_frozen_cart,
)
from consequence_lab.phase14c import build_feature_matrix


CONFIG_PATH = "configs/phase14g_trace.yaml"
SOURCE_FILES = (
    CONFIG_PATH,
    "reports/phase14g_trace_attribution_protocol_v2.md",
    "src/consequence_lab/phase14g_runner.py",
    "src/consequence_lab/phase14g.py",
    "src/consequence_lab/phase14f.py",
    "src/consequence_lab/phase14.py",
    "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/corridor_evaluation.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "scripts/run_phase14g_trace.py",
    "scripts/freeze_phase14g_trace_lock.py",
    "tests/test_phase14g_runner.py",
)


class Phase14GRunError(RuntimeError):
    """Phase14G trace-run config, lock, parent, or output integrity failure."""


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14GRunError(f"git {' '.join(args)} failed") from exc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_hashes(root: Path) -> dict[str, str]:
    missing = [relative for relative in SOURCE_FILES if not (root / relative).is_file()]
    if missing:
        raise Phase14GRunError(f"Phase14G run source files missing: {missing}")
    return {relative: sha256_file(root / relative) for relative in SOURCE_FILES}


def _validate_run_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "phase14g_trace_run_config_v1":
        raise Phase14GRunError("Phase14G trace run config schema mismatch")
    if config.get("status") != "authorized_trace_development_pending_lock":
        raise Phase14GRunError("Phase14G trace run is not in the authorized pre-lock state")
    expected_scope = {
        "trace_development": True,
        "task_transitions": True,
        "teacher_counterfactual_branches": True,
        "student_fit_or_tuning": False,
        "dagger": False,
        "learned_world_model": False,
        "reserve_access": False,
        "final_reserve_access": False,
        "further_runs_authorized": False,
    }
    if config.get("scope") != expected_scope:
        raise Phase14GRunError("Phase14G trace scope differs from its one-run authorization")
    block = config["block"]
    if (block.get("seed_start") != 193000 or block.get("layout_count") != 100
            or block.get("methods") != ["student", "geometry_waypoint", "random_shooting"]):
        raise Phase14GRunError("Phase14G run block/method contract changed")
    if config["resources"].get("wall_time_seconds") is not None or config["resources"].get("storage_bytes") is not None:
        raise Phase14GRunError("project-wide time/storage caps were removed; do not introduce run caps")
    return config


def _load_config(root: Path) -> dict[str, Any]:
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    return _validate_run_config(config)


def _verified_audit_and_parents(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    audit_path = root / config["parent"]["layout_audit"]
    audit = _read_json(audit_path)
    block = config["block"]
    if (audit.get("schema") != block["audit_schema"]
            or audit.get("status") != "hash_only_passed_no_outcomes_run_not_authorized"
            or audit.get("task_transitions_evaluated") is not False
            or audit.get("episode_outcomes_read") is not False
            or audit.get("teacher_counterfactual_branches_evaluated") is not False
            or audit.get("model_fit_or_tuning") is not False
            or audit.get("reserve_access") is not False
            or audit.get("run_authorized") is not False):
        raise Phase14GRunError("Phase14G parent layout audit is invalid or has wider scope")
    layouts = audit.get("candidate_block", {}).get("layouts", [])
    hashes = [row.get("layout_sha256") for row in layouts]
    seeds = [row.get("seed") for row in layouts]
    if (seeds != list(range(int(block["seed_start"]), int(block["seed_start"]) + int(block["layout_count"])))
            or len(set(hashes)) != int(block["layout_count"])
            or audit.get("unique_candidate_layout_hashes") != int(block["layout_count"])
            or not audit.get("candidate_block_disjoint_from_all_historical_layouts")
            or not audit.get("candidate_block_disjoint_from_prior_outcome_layouts")
            or not audit.get("candidate_block_outside_sealed_reserve_seed_range")):
        raise Phase14GRunError("audited layout identities do not exactly match the run block")
    audit_sha = sha256_file(audit_path)
    for relative, expected in audit.get("source_sha256", {}).items():
        if not (root / relative).is_file() or sha256_file(root / relative) != expected:
            raise Phase14GRunError(f"hash-audited source changed: {relative}")
    for relative, expected in audit.get("historical_hash_source_sha256", {}).items():
        if not (root / relative).is_file() or sha256_file(root / relative) != expected:
            raise Phase14GRunError(f"historical layout-provenance source changed: {relative}")
    phase14f = {
        key: sha256_file(root / config["parent"][key])
        for key in (
            "phase14f_lock", "phase14f_manifest", "phase14f_metrics", "phase14f_episodes",
            "phase14f_student", "phase14f_student_manifest", "training_records", "training_manifest",
        )
    }
    model_sha = phase14f["phase14f_student"]
    if model_sha != audit.get("phase14f_parent", {}).get("student_sha256"):
        raise Phase14GRunError("frozen Phase14F student differs from hash-audited parent")
    if audit.get("source_sha256", {}).get(config["parent"]["training_records"]) != phase14f["training_records"]:
        raise Phase14GRunError("train-support records differ from the hash-audited source")
    if audit.get("phase14f_parent", {}).get("failed_gate_confirmed") is not True:
        raise Phase14GRunError("Phase14F failed parent result was not verified")
    forbidden_ranges = (
        (int(config["integrity"]["reserve_seed_start"]), int(config["integrity"]["reserve_layout_count"])),
        (int(config["integrity"]["consumed_seed_block_forbidden"]["seed_start"]),
         int(config["integrity"]["consumed_seed_block_forbidden"]["count"])),
    )
    run_start, run_end = int(block["seed_start"]), int(block["seed_start"]) + int(block["layout_count"])
    for forbidden_start, forbidden_count in forbidden_ranges:
        if run_start < forbidden_start + forbidden_count and forbidden_start < run_end:
            raise Phase14GRunError("Phase14G run block overlaps reserve or consumed Phase14F block")
    return {
        "audit_sha256": audit_sha,
        "audit_source_commit": audit.get("generation_git_commit"),
        "ordered_layout_hashes": hashes,
        "parent_artifact_sha256": phase14f,
        "layout_audit_report_sha256": sha256_file(root / config["parent"]["layout_audit_report"]),
    }


def build_run_lock(root: Path) -> dict[str, Any]:
    """Build the one-run lock; does not generate a transition or outcome."""
    root = Path(root).resolve()
    if _git(root, "status", "--porcelain"):
        raise Phase14GRunError("commit Phase14G protocol/source/tests before freezing the run lock")
    config = _load_config(root)
    parents = _verified_audit_and_parents(root, config)
    lock_path = root / config["outputs"]["lock"]
    output_dir = root / config["outputs"]["run_dir"]
    report_path = root / config["outputs"]["report"]
    if lock_path.exists() or output_dir.exists() or report_path.exists():
        raise Phase14GRunError("refusing to overwrite Phase14G lock or run outputs")
    return {
        "schema": "phase14g_trace_run_lock_v1",
        "status": "authorized_one_shot_trace_development",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_reference": config["authorization"]["reference"],
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "source_sha256": _source_hashes(root),
        "config_sha256": sha256_file(root / CONFIG_PATH),
        "parent": parents,
        "block": {
            "seed_start": int(config["block"]["seed_start"]),
            "layout_count": int(config["block"]["layout_count"]),
            "ordered_layout_hashes": parents["ordered_layout_hashes"],
        },
        "methods": config["block"]["methods"],
        "scope": config["scope"],
        "integrity": config["integrity"],
        "resources": config["resources"],
        "outputs": config["outputs"],
        "overwrite_forbidden": True,
    }


def validate_run_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any], FrozenCartPolicy]:
    root = Path(root).resolve()
    config = _load_config(root)
    lock_path = root / config["outputs"]["lock"]
    if not lock_path.is_file():
        raise Phase14GRunError("Phase14G trace run lock is missing")
    if _git(root, "status", "--porcelain"):
        raise Phase14GRunError("working tree must be clean before the one-shot run")
    lock = _read_json(lock_path)
    if lock.get("schema") != "phase14g_trace_run_lock_v1" or lock.get("status") != "authorized_one_shot_trace_development":
        raise Phase14GRunError("Phase14G run lock schema/status mismatch")
    if lock.get("source_commit") != _git(root, "rev-parse", "HEAD"):
        try:
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", str(lock.get("source_commit")), "HEAD"],
                cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Phase14GRunError("run-lock source commit is not an ancestor of HEAD") from exc
    parents = _verified_audit_and_parents(root, config)
    expected = {
        "authorization_reference": config["authorization"]["reference"],
        "source_sha256": _source_hashes(root),
        "config_sha256": sha256_file(root / CONFIG_PATH),
        "parent": parents,
        "block": {
            "seed_start": int(config["block"]["seed_start"]),
            "layout_count": int(config["block"]["layout_count"]),
            "ordered_layout_hashes": parents["ordered_layout_hashes"],
        },
        "methods": config["block"]["methods"],
        "scope": config["scope"],
        "integrity": config["integrity"],
        "resources": config["resources"],
        "outputs": config["outputs"],
        "overwrite_forbidden": True,
    }
    mismatches = [key for key, value in expected.items() if lock.get(key) != value]
    if mismatches:
        raise Phase14GRunError(f"Phase14G run lock binding mismatch: {', '.join(mismatches)}")
    output_dir = root / config["outputs"]["run_dir"]
    if output_dir.exists() or (root / config["outputs"]["report"]).exists():
        raise Phase14GRunError("Phase14G outputs already exist; the one-shot run cannot be repeated")
    model = _read_json(root / config["parent"]["phase14f_student"])
    if sha256_file(root / config["parent"]["phase14f_student"]) != parents["parent_artifact_sha256"]["phase14f_student"]:
        raise Phase14GRunError("frozen CART changed since lock freeze")
    return config, lock, FrozenCartPolicy(model)


def _load_training_inputs(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_index, line in enumerate(stream):
            if not line.strip():
                continue
            record = json.loads(line)
            values = record.get("student_input_52")
            if not isinstance(values, list) or len(values) != 52:
                raise Phase14GRunError(f"train-only state record {line_index} violates 52-feature contract")
            vector = np.asarray(values, dtype=np.float64)
            if not np.isfinite(vector).all():
                raise Phase14GRunError(f"train-only state record {line_index} has non-finite features")
            rows.append(vector)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (2095, 52):
        raise Phase14GRunError(f"expected 2,095 train-only support states, got {matrix.shape}")
    return matrix


def _student_episode_with_traces(scenario: Any, policy: FrozenCartPolicy,
                                 training_normalized: np.ndarray, support: dict[int, int],
                                 trace_stream: Any, branch_counter: list[int]) -> CorridorEpisodeResult:
    state = scenario.initial_state
    validate_corridor_state(state, scenario.spec)
    decision_count = 0
    effort = 0
    total_policy_seconds = 0.0
    pending_trace_rows: list[dict[str, Any]] = []
    while state.status == "active":
        state_before = state
        started = perf_counter_ns()
        raw = state_features(scenario, state_before)
        action = policy.action_for_features(raw)
        if not 0 <= action < 9:
            raise Phase14GRunError("frozen CART selected an illegal action")
        live_step = corridor_step(state_before, action, scenario.spec)
        decision_seconds = (perf_counter_ns() - started) / 1_000_000_000
        total_policy_seconds += decision_seconds
        state = live_step.state
        decision_count += 1
        effort += int(action != 0)

        traced = trace_frozen_cart(policy.payload, raw)
        if traced.action_id != action or len(traced.normalized_features) != 61:
            raise Phase14GRunError("trace evaluator disagrees with frozen deployable CART path")
        query_vector = np.asarray(traced.normalized_features, dtype=np.float64)
        nearest = float(np.min(np.linalg.norm(training_normalized - query_vector[None, :], axis=1)))

        # Shadow planner and counterfactual branches occur after the real step and
        # operate only on immutable copies/values, outside deployable latency.
        shadow_started = perf_counter_ns()
        shooting = exact_dynamics_shooting(
            state_before, scenario.spec, horizon=8, num_sequences=128,
            seed=phase12_query_seed(scenario.seed, state_before.elapsed_steps),
        )
        shooting_shadow_seconds = (perf_counter_ns() - shadow_started) / 1_000_000_000
        state_tuple = (state_before.x, state_before.y, state_before.vx, state_before.vy,
                       state_before.elapsed_steps, state_before.status)
        branch_started = perf_counter_ns()
        branch_rows = forced_first_action_values(
            scenario, state_before, decision_index=decision_count - 1,
            expected_student_features=raw,
        )
        branch_shadow_seconds = (perf_counter_ns() - branch_started) / 1_000_000_000
        ranked = add_state_ranks(branch_rows)
        branch_counter[0] += len(ranked)
        selected_branch = ranked[action]
        best_tuple = min(tuple(row["objective_tuple"]) for row in ranked)
        best_actions = [int(row["forced_action_id"]) for row in ranked
                        if tuple(row["objective_tuple"]) == best_tuple]
        distinct_tuples = sorted({tuple(row["objective_tuple"]) for row in ranked})
        second_tuple = distinct_tuples[1] if len(distinct_tuples) > 1 else None
        trace_row = {
            "schema": "phase14g_student_decision_trace_v1",
            "episode_seed": scenario.seed,
            "layout_sha256": scenario.layout_sha256,
            "decision_index": decision_count - 1,
            "student_runtime_input_52": [float(value) for value in raw],
            "student_selected_action_id": int(action),
            "student_action_applied_matches_selected": True,
            "state_before": list(state_tuple),
            "state_after": [state.x, state.y, state.vx, state.vy, state.elapsed_steps, state.status],
            "live_transition": {"collision": live_step.collision, "success": live_step.success,
                                "timeout": live_step.truncated},
            "student_decision_latency_seconds": decision_seconds,
            "shadow_exact_shooting_latency_seconds": shooting_shadow_seconds,
            "counterfactual_branch_latency_seconds": branch_shadow_seconds,
            "cart": {
                "predicted_normalized_regret9": list(traced.predicted_regret),
                "normalized_features_61": list(traced.normalized_features),
                "node_path": list(traced.node_path),
                "leaf_id": traced.leaf_id,
                "training_leaf_support": int(support.get(traced.leaf_id, 0)),
                "nearest_training_distance_normalized_61d_l2": nearest,
            },
            "shadow_exact_shooting_action_id": int(shooting.first_action),
            "shadow_exact_shooting_queries": int(shooting.evaluated_sequences),
            "teacher_optimal_action_ids": best_actions,
            "teacher_best_objective_tuple": list(best_tuple),
            "teacher_second_objective_tuple": None if second_tuple is None else list(second_tuple),
            "teacher_branches": [
                {key: value for key, value in row.items() if key not in {"student_input", "continuation_action_trace"}}
                for row in ranked
            ],
            "selected_action_teacher_normalized_regret": float(selected_branch["normalized_regret"]),
            "source_state_unchanged_by_shadow_evaluations": state_tuple == (
                state_before.x, state_before.y, state_before.vx, state_before.vy,
                state_before.elapsed_steps, state_before.status,
            ),
        }
        if trace_row["student_runtime_input_52"] != [float(value) for value in state_features(scenario, state_before)]:
            raise Phase14GRunError("serialized runtime input differs from public state feature function")
        if not trace_row["source_state_unchanged_by_shadow_evaluations"]:
            raise Phase14GRunError("counterfactual diagnostics mutated live source state")
        pending_trace_rows.append(trace_row)

    outcome = state.status
    for i, row in enumerate(pending_trace_rows):
        row["episode_outcome"] = outcome
        row["episode_steps"] = state.elapsed_steps - scenario.initial_state.elapsed_steps
        row["collision_within_next_three_student_decisions"] = any(
            pending_trace_rows[j]["live_transition"]["collision"]
            for j in range(i, min(len(pending_trace_rows), i + 3))
        )
        trace_stream.write(json.dumps(row, sort_keys=True) + "\n")
    return CorridorEpisodeResult(
        method="student", seed=scenario.seed, layout_sha256=scenario.layout_sha256,
        outcome=outcome, task_status=outcome,
        steps=state.elapsed_steps - scenario.initial_state.elapsed_steps,
        control_effort=effort, decision_count=decision_count,
        wall_seconds=total_policy_seconds,
        wall_seconds_per_decision=total_policy_seconds / decision_count if decision_count else None,
        shooting_queries=0, shooting_success_available_queries=0,
        kinodynamic_expansion_limit_queries=0, planner_failure=None,
    )


def run_trace_study(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config, lock, policy = validate_run_lock(root)
    parents = lock["parent"]
    output = root / config["outputs"]["run_dir"]
    report_path = root / config["outputs"]["report"]
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / config["outputs"]["manifest"]
    episodes_path = output / config["outputs"]["episodes"]
    traces_path = output / config["outputs"]["student_traces"]
    metrics_path = output / config["outputs"]["metrics"]
    audit = _read_json(root / config["parent"]["layout_audit"])
    training_raw = _load_training_inputs(root / config["parent"]["training_records"])
    support = dict(build_support_counts(policy.payload, training_raw))
    variant = policy.payload["selected_model"]["feature_variant"]
    normalized_train = build_feature_matrix(training_raw, variant)
    normalized_train = (normalized_train - np.asarray(policy.payload["x_mean"])) / np.asarray(policy.payload["x_std"])
    manifest: dict[str, Any] = {
        "schema": "phase14g_trace_result_manifest_v1",
        "status": "collecting_locked_trace_development",
        "source_commit": lock["source_commit"],
        "run_lock_sha256": sha256_file(root / config["outputs"]["lock"]),
        "layout_audit_sha256": parents["audit_sha256"],
        "ordered_layout_hashes": lock["block"]["ordered_layout_hashes"],
        "student_sha256": parents["parent_artifact_sha256"]["phase14f_student"],
        "train_support_records_sha256": parents["parent_artifact_sha256"]["training_records"],
        "scope": config["scope"],
        "reserve_access": False,
        "model_fit_or_tuning": False,
        "runtime_environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    start_time = time.perf_counter()
    result_rows: list[CorridorEpisodeResult] = []
    branch_counter = [0]
    try:
        with episodes_path.open("x", encoding="utf-8", newline="\n") as episode_stream, traces_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as trace_stream:
            methods = list(config["block"]["methods"])
            for offset, (seed, expected_hash) in enumerate(zip(
                range(int(config["block"]["seed_start"]), int(config["block"]["seed_start"]) + int(config["block"]["layout_count"])),
                lock["block"]["ordered_layout_hashes"], strict=True,
            )):
                scenario = generate_staggered_corridor(seed)
                if scenario.layout_sha256 != expected_hash:
                    raise Phase14GRunError(f"audited layout identity mismatch at seed {seed}")
                rotation = offset % len(methods)
                method_order = methods[rotation:] + methods[:rotation]
                for method in method_order:
                    if method == "student":
                        result = _student_episode_with_traces(
                            scenario, policy, normalized_train, support,
                            trace_stream, branch_counter,
                        )
                    elif method == "geometry_waypoint":
                        result = evaluate_corridor_episode(
                            scenario, method,
                            parameters={"resolution": 0.04, "waypoint_tolerance": 0.03},
                        )
                    else:
                        result = evaluate_corridor_episode(
                            scenario, method,
                            parameters={"horizon": 8, "num_sequences": 128},
                            proposal_seed=scenario.seed,
                        )
                    result_rows.append(result)
                    episode_stream.write(json.dumps(asdict(result), sort_keys=True) + "\n")

        if len(result_rows) != int(config["block"]["layout_count"]) * 3:
            raise Phase14GRunError("episode record count does not match the frozen paired design")
        if branch_counter[0] != sum(
            len(row["teacher_branches"]) for row in _read_jsonl(traces_path)
        ) or branch_counter[0] % 9:
            raise Phase14GRunError("forced-action branch count does not match persisted nine-action traces")
        summaries = {row.method: asdict(row) for row in summarize_corridor_results(result_rows)}
        metrics = {
            "schema": "phase14g_trace_metrics_v1",
            "status": "completed_descriptive_trace_attribution",
            "layout_count": int(config["block"]["layout_count"]),
            "episode_record_count": len(result_rows),
            "student_decision_trace_count": sum(row.decision_count for row in result_rows if row.method == "student"),
            "forced_action_branch_count": branch_counter[0],
            "student": summaries["student"],
            "geometry_waypoint": summaries["geometry_waypoint"],
            "exact_random_shooting": summaries["random_shooting"],
            "student_minus_geometry_safe_success_paired_95": asdict(paired_outcome_rate_interval(
                result_rows, "student", "geometry_waypoint", event="success",
                bootstrap_seed=193011, resamples=10000,
            )),
            "student_minus_exact_shooting_safe_success_paired_95": asdict(paired_outcome_rate_interval(
                result_rows, "student", "random_shooting", event="success",
                bootstrap_seed=193011, resamples=10000,
            )),
            "student_minus_geometry_collision_paired_95": asdict(paired_outcome_rate_interval(
                result_rows, "student", "geometry_waypoint", event="collision",
                bootstrap_seed=193011, resamples=10000,
            )),
            "student_minus_exact_shooting_collision_paired_95": asdict(paired_outcome_rate_interval(
                result_rows, "student", "random_shooting", event="collision",
                bootstrap_seed=193011, resamples=10000,
            )),
            "student_action_frequencies": [0] * 9,
            "exact_shooting_action_frequencies_at_student_states": [0] * 9,
            "mean_selected_teacher_regret": 0.0,
            "teacher_regret_by_student_action": [[] for _ in range(9)],
            "candidate_layout_hashes": lock["block"]["ordered_layout_hashes"],
            "student_model_sha256": lock["parent"]["parent_artifact_sha256"]["phase14f_student"],
            "no_gate_or_model_selection": True,
        }
        # Derive decision diagnostics from the persisted trace, not live state.
        trace_rows = _read_jsonl(traces_path)
        action_frequency = [0] * 9
        shooting_frequency = [0] * 9
        teacher_optimal_frequency = [0] * 9
        regrets: list[float] = []
        regret_by_action: list[list[float]] = [[] for _ in range(9)]
        agreement = 0
        immediate_collisions = 0
        decision_latencies: list[float] = []
        shooting_shadow_latencies: list[float] = []
        branch_shadow_latencies: list[float] = []
        optimal_set_sizes: list[int] = []
        by_episode_outcome: dict[str, list[dict[str, Any]]] = {
            "success": [], "collision": [], "timeout": []
        }
        for row in trace_rows:
            action = int(row["student_selected_action_id"])
            teacher_action = int(row["shadow_exact_shooting_action_id"])
            action_frequency[action] += 1
            shooting_frequency[teacher_action] += 1
            for best_action in row["teacher_optimal_action_ids"]:
                teacher_optimal_frequency[int(best_action)] += 1
            regret = float(row["selected_action_teacher_normalized_regret"])
            regrets.append(regret)
            regret_by_action[action].append(regret)
            agreement += int(action == teacher_action)
            immediate_collisions += int(row["live_transition"]["collision"])
            decision_latencies.append(float(row["student_decision_latency_seconds"]))
            shooting_shadow_latencies.append(float(row["shadow_exact_shooting_latency_seconds"]))
            branch_shadow_latencies.append(float(row["counterfactual_branch_latency_seconds"]))
            optimal_set_sizes.append(len(row["teacher_optimal_action_ids"]))
            by_episode_outcome[row["episode_outcome"]].append(row)
        metrics.update({
            "student_action_frequencies": action_frequency,
            "exact_shooting_action_frequencies_at_student_states": shooting_frequency,
            "teacher_optimal_action_incidence_at_student_states": teacher_optimal_frequency,
            "exact_shooting_action_agreement_fraction_at_student_states": agreement / len(trace_rows) if trace_rows else None,
            "mean_selected_teacher_regret": float(np.mean(regrets)) if regrets else None,
            "median_selected_teacher_regret": float(np.median(regrets)) if regrets else None,
            "selected_teacher_regret_by_student_action": [
                float(np.mean(values)) if values else None for values in regret_by_action
            ],
            "immediate_student_transition_collisions": immediate_collisions,
            "collision_within_three_student_decisions": sum(
                bool(row["collision_within_next_three_student_decisions"]) for row in trace_rows
            ),
            "teacher_multi_optimum_state_fraction": (
                float(np.mean(np.asarray(optimal_set_sizes) > 1)) if optimal_set_sizes else None
            ),
            "student_decision_latency_median_seconds": float(np.median(decision_latencies)) if decision_latencies else None,
            "shadow_exact_shooting_latency_median_seconds": float(np.median(shooting_shadow_latencies)) if shooting_shadow_latencies else None,
            "counterfactual_branch_latency_median_seconds": float(np.median(branch_shadow_latencies)) if branch_shadow_latencies else None,
            "decision_diagnostics_by_episode_outcome": {
                name: {
                    "decision_count": len(values),
                    "mean_selected_teacher_regret": (
                        float(np.mean([value["selected_action_teacher_normalized_regret"] for value in values]))
                        if values else None
                    ),
                    "median_nearest_training_state_distance": (
                        float(np.median([value["cart"]["nearest_training_distance_normalized_61d_l2"] for value in values]))
                        if values else None
                    ),
                    "mean_routed_leaf_training_support": (
                        float(np.mean([value["cart"]["training_leaf_support"] for value in values]))
                        if values else None
                    ),
                }
                for name, values in by_episode_outcome.items()
            },
            "student_state_trace_count": len(trace_rows),
        })
        _write_json(metrics_path, metrics)
        manifest.update({
            "status": "completed_trace_attribution_development",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - start_time,
            "episode_record_count": len(result_rows),
            "student_decision_trace_count": len(trace_rows),
            "forced_action_branch_count": branch_counter[0],
            "episodes_sha256": sha256_file(episodes_path),
            "student_traces_sha256": sha256_file(traces_path),
            "metrics_sha256": sha256_file(metrics_path),
            "reserve_access": False,
        })
        _write_json(manifest_path, manifest)
        _write_report(report_path, metrics, manifest)
        return metrics
    except BaseException as exc:
        manifest.update({
            "status": "aborted_or_incomplete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - start_time,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "episodes_sha256": sha256_file(episodes_path) if episodes_path.is_file() else None,
            "student_traces_sha256": sha256_file(traces_path) if traces_path.is_file() else None,
            "reserve_access": False,
        })
        _write_json(manifest_path, manifest)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_report(path: Path, metrics: dict[str, Any], manifest: dict[str, Any]) -> None:
    rows = []
    for key, label in (("student", "Frozen CART student"), ("geometry_waypoint", "Geometry waypoint"),
                       ("exact_random_shooting", "Exact shooting")):
        item = metrics[key]
        rows.append(
            f"| {label} | {item['safe_success_rate']:.3f} | {item['collision_rate']:.3f} | "
            f"{item['timeout_rate']:.3f} | {item['median_wall_seconds_per_decision'] * 1000:.4f} ms |"
        )
    text = "\n".join([
        "# Phase 14G trace-attribution result", "",
        "**Interpretation:** descriptive diagnosis only; Phase14F remains failed and closed. This run does not authorize training, DAgger, learned dynamics, reserve evaluation, or another run.", "",
        f"Layouts: {metrics['layout_count']}; paired episodes: {metrics['episode_record_count']}; student states traced: {metrics['student_state_trace_count']}; forced-action branches: {metrics['forced_action_branch_count']}.", "",
        "| Method | Safe success | Collision | Timeout | Median decision latency |", "|---|---:|---:|---:|---:|", *rows, "",
        f"Student vs geometry paired success interval: `{metrics['student_minus_geometry_safe_success_paired_95']}`.",
        f"Student vs exact shooting paired success interval: `{metrics['student_minus_exact_shooting_safe_success_paired_95']}`.",
        f"Student/exact-shooting action agreement at student states: {metrics['exact_shooting_action_agreement_fraction_at_student_states']:.3f}.",
        f"Mean selected-action teacher normalized regret: {metrics['mean_selected_teacher_regret']:.4f}; immediate student-action collisions: {metrics['immediate_student_transition_collisions']}.",
        "Support/regret associations are descriptive, not causal. Student latency excludes shadow planning and counterfactual branches.", "",
        f"Run lock SHA-256: `{manifest['run_lock_sha256']}`.",
        f"Episodes SHA-256: `{manifest['episodes_sha256']}`; traces SHA-256: `{manifest['student_traces_sha256']}`; metrics SHA-256: `{manifest['metrics_sha256']}`.",
        "",
    ])
    path.write_text(text, encoding="utf-8")
