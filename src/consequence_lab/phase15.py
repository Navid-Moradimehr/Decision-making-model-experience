"""Read-only audits for Phase15 use of the locked Phase14G diagnostic set."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from consequence_lab.phase14c import build_feature_matrix


class Phase15AuditError(RuntimeError):
    """Phase14G parent artifacts do not satisfy the Phase15A input contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=np.float64), [0, .25, .5, .75, .9, .95, 1])
    return {key: float(value) for key, value in zip(
        ("min", "q25", "median", "q75", "q90", "q95", "max"), q, strict=True
    )}


def validate_trace_branch_contract(trace: dict[str, Any]) -> None:
    """Check branch identity/visibility without requiring any run artifacts."""
    branches = trace.get("teacher_branches", [])
    if [int(b.get("forced_action_id", -1)) for b in branches] != list(range(9)):
        raise Phase15AuditError("trace must contain all nine legal primitive action branches in ID order")
    if not trace.get("student_action_applied_matches_selected") or not trace.get("source_state_unchanged_by_shadow_evaluations"):
        raise Phase15AuditError("trace failed student-action or source-immutability assertion")
    for branch in branches:
        if "student_input" in branch or "continuation_action_trace" in branch:
            raise Phase15AuditError("teacher branch contains a forbidden serialized field")
        if not all(key in branch for key in ("forced_successor", "objective_tuple", "normalized_regret")):
            raise Phase15AuditError("teacher branch is missing a saved consequence or target field")


def audit_phase14g_inputs(root: Path) -> dict[str, Any]:
    """Validate saved Phase14G data and characterize its relation to Phase14B train states.

    This reads saved records only. It does not instantiate the simulator, generate
    layouts, run teacher branches, fit models, or access any reserve split.
    """
    root = Path(root).resolve()
    lock_path = root / "experiments/phase14g_trace_v1.lock.json"
    run_dir = root / "artifacts/runs/phase14g_trace_v1"
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    episodes_path = run_dir / "episodes.jsonl"
    traces_path = run_dir / "student_decisions.jsonl"
    student_path = root / "experiments/phase14f_student_cart_v1.json"
    train_path = root / "artifacts/runs/phase14b_training_labels_resume_v1/first_action_values_complete.jsonl"
    for path in (lock_path, manifest_path, metrics_path, episodes_path, traces_path, student_path, train_path):
        if not path.is_file():
            raise Phase15AuditError(f"required locked Phase14G/Phase14B artifact missing: {path.relative_to(root)}")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    student = json.loads(student_path.read_text(encoding="utf-8"))
    episodes = _jsonl(episodes_path)
    traces = _jsonl(traces_path)
    train_rows = _jsonl(train_path)

    expected_lock_sha = _sha256(lock_path)
    expected_layouts = lock.get("block", {}).get("ordered_layout_hashes", [])
    expected_scopes = {
        "dagger": False, "final_reserve_access": False, "learned_world_model": False,
        "reserve_access": False, "student_fit_or_tuning": False,
    }
    if lock.get("schema") != "phase14g_trace_run_lock_v1" or lock.get("status") != "authorized_one_shot_trace_development":
        raise Phase15AuditError("unexpected Phase14G lock identity/status")
    if manifest.get("status") != "completed_trace_attribution_development":
        raise Phase15AuditError("Phase14G manifest does not record a completed one-shot run")
    if manifest.get("run_lock_sha256", "").upper() != expected_lock_sha:
        raise Phase15AuditError("Phase14G manifest lock hash mismatch")
    if manifest.get("scope", {}) != lock.get("scope", {}) or any(lock.get("scope", {}).get(k) is not v for k, v in expected_scopes.items()):
        raise Phase15AuditError("Phase14G scope does not preserve closed downstream/reserve gates")
    hash_checks = {
        "episodes": (episodes_path, manifest.get("episodes_sha256")),
        "traces": (traces_path, manifest.get("student_traces_sha256")),
        "metrics": (metrics_path, manifest.get("metrics_sha256")),
        "student": (student_path, lock.get("parent", {}).get("parent_artifact_sha256", {}).get("phase14f_student")),
        "training_records": (train_path, lock.get("parent", {}).get("parent_artifact_sha256", {}).get("training_records")),
    }
    actual_hashes = {}
    for label, (path, expected) in hash_checks.items():
        actual = _sha256(path)
        actual_hashes[label] = actual
        if not isinstance(expected, str) or actual != expected.upper():
            raise Phase15AuditError(f"locked {label} artifact hash mismatch")

    layout_set = set(expected_layouts)
    if len(layout_set) != 100 or len(expected_layouts) != 100:
        raise Phase15AuditError("Phase14G lock must identify exactly 100 unique layouts")
    if len(episodes) != 300 or len({(r.get("method"), r.get("layout_sha256")) for r in episodes}) != 300:
        raise Phase15AuditError("Phase14G episode records are incomplete or duplicated")
    methods = Counter(r.get("method") for r in episodes)
    if methods != Counter({"student": 100, "geometry_waypoint": 100, "random_shooting": 100}):
        raise Phase15AuditError("Phase14G method/layout balance differs from locked design")
    if {r.get("layout_sha256") for r in episodes} != layout_set:
        raise Phase15AuditError("episode layouts differ from the consumed locked block")
    if len(traces) != 922 or len({(r.get("episode_seed"), r.get("decision_index")) for r in traces}) != len(traces):
        raise Phase15AuditError("Phase14G trace records are incomplete or duplicated")
    if any(t.get("layout_sha256") not in layout_set for t in traces):
        raise Phase15AuditError("trace references a layout outside the consumed Phase14G block")

    student_episode = {(r["seed"], r["layout_sha256"]): r for r in episodes if r["method"] == "student"}
    by_episode: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        key = (int(trace["episode_seed"]), trace["layout_sha256"])
        if key not in student_episode:
            raise Phase15AuditError("trace has no paired student episode")
        by_episode[key].append(trace)
        validate_trace_branch_contract(trace)
        if trace.get("cart", {}).get("training_leaf_support") < 1:
            raise Phase15AuditError("student trace references an unoccupied CART leaf")
    for key, rows in by_episode.items():
        rows.sort(key=lambda r: int(r["decision_index"]))
        if [int(r["decision_index"]) for r in rows] != list(range(len(rows))):
            raise Phase15AuditError("student decision trace indices are not a contiguous episode prefix")
        episode = student_episode[key]
        if len(rows) != int(episode["decision_count"]):
            raise Phase15AuditError("trace count differs from paired student episode decision count")
        if any(r["episode_outcome"] != episode["outcome"] for r in rows):
            raise Phase15AuditError("trace outcome differs from paired episode outcome")
    if len(train_rows) != 2095:
        raise Phase15AuditError("expected exactly 2,095 Phase14B train-only support states")

    regrets = np.asarray([float(t["selected_action_teacher_normalized_regret"]) for t in traces])
    student_actions = np.asarray([int(t["student_selected_action_id"]) for t in traces])
    action_counts = np.bincount(student_actions, minlength=9)
    immediate_collisions = np.asarray([bool(t["live_transition"]["collision"]) for t in traces])
    tie_counts = Counter(len(t["teacher_optimal_action_ids"]) for t in traces)
    outcome_diagnostics: dict[str, Any] = {}
    for outcome in ("collision", "success", "timeout"):
        rows = [t for t in traces if t["episode_outcome"] == outcome]
        outcome_diagnostics[outcome] = {
            "trace_count": len(rows),
            "episode_count": sum(r["outcome"] == outcome for r in episodes if r["method"] == "student"),
            "mean_selected_regret": float(np.mean([t["selected_action_teacher_normalized_regret"] for t in rows])) if rows else None,
            "catastrophic_regret_ge_0_5_fraction": float(np.mean([t["selected_action_teacher_normalized_regret"] >= .5 for t in rows])) if rows else None,
            "median_nearest_training_distance": float(np.median([t["cart"]["nearest_training_distance_normalized_61d_l2"] for t in rows])) if rows else None,
            "mean_leaf_support": float(np.mean([t["cart"]["training_leaf_support"] for t in rows])) if rows else None,
            "immediate_collision_fraction": float(np.mean([t["live_transition"]["collision"] for t in rows])) if rows else None,
        }
    action_diagnostics = []
    for action_id in range(9):
        selected = [i for i, value in enumerate(student_actions) if value == action_id]
        action_diagnostics.append({
            "action_id": action_id,
            "selected_count": len(selected),
            "immediate_collision_count": int(immediate_collisions[selected].sum()) if selected else 0,
            "mean_selected_regret": float(regrets[selected].mean()) if selected else None,
        })

    raw_train = np.asarray([r["student_input_52"] for r in train_rows], dtype=np.float64)
    raw_student = np.asarray([t["student_runtime_input_52"] for t in traces], dtype=np.float64)
    selected_model = student["selected_model"]
    variant = selected_model["feature_variant"]
    x_mean = np.asarray(student["x_mean"], dtype=np.float64)
    x_std = np.asarray(student["x_std"], dtype=np.float64)
    z_train = (build_feature_matrix(raw_train, variant) - x_mean) / x_std
    z_student = (build_feature_matrix(raw_student, variant) - x_mean) / x_std
    if z_student.shape[1] != 61 or z_train.shape[1] != 61:
        raise Phase15AuditError("Phase14F frozen student feature contract is not the expected 61D representation")
    # Chunked nearest-neighbor distances keep this read-only audit bounded in memory.
    loo_distance = np.empty(len(z_train), dtype=np.float64)
    for start in range(0, len(z_train), 64):
        stop = min(start + 64, len(z_train))
        delta = z_train[start:stop, None, :] - z_train[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        loo_distance[start:stop] = distances.min(axis=1)
    student_distance = np.empty(len(z_student), dtype=np.float64)
    for start in range(0, len(z_student), 64):
        stop = min(start + 64, len(z_student))
        distances = np.sqrt(np.sum((z_student[start:stop, None, :] - z_train[None, :, :]) ** 2, axis=2))
        student_distance[start:stop] = distances.min(axis=1)
    # Compare on-policy feature centroid shift in the training-normalized coordinates.
    centroid_delta = np.abs(z_student.mean(axis=0) - z_train.mean(axis=0))
    top_shifted = sorted(
        [{"feature_index": i, "absolute_standardized_centroid_shift": float(value)}
         for i, value in enumerate(centroid_delta)],
        key=lambda row: row["absolute_standardized_centroid_shift"], reverse=True,
    )[:10]

    outcome_counts = {name: dict(Counter(r["outcome"] for r in episodes if r["method"] == name))
                       for name in ("student", "geometry_waypoint", "random_shooting")}
    computed = {
        "schema": "phase15a_phase14g_locked_dataset_audit_v1",
        "status": "passed_read_only_phase14g_audit",
        "phase14g_layout_block_consumed": True,
        "reserve_accessed": False,
        "simulator_transitions_generated": False,
        "teacher_branches_generated": False,
        "models_fit": False,
        "phase14g_lock_sha256": expected_lock_sha,
        "artifact_sha256": actual_hashes,
        "layout_count": len(layout_set),
        "episode_count": len(episodes),
        "student_trace_count": len(traces),
        "forced_action_branch_count": sum(len(t["teacher_branches"]) for t in traces),
        "outcomes_by_method": outcome_counts,
        "student_regret_quantiles": _quantiles(regrets),
        "student_catastrophic_regret_ge_0_5_fraction": float(np.mean(regrets >= .5)),
        "student_teacher_action_agreement": float(np.mean(
            student_actions == np.asarray([int(t["shadow_exact_shooting_action_id"]) for t in traces])
        )),
        "teacher_optimal_action_tie_state_counts": dict(sorted(tie_counts.items())),
        "student_action_diagnostics": action_diagnostics,
        "outcome_diagnostics": outcome_diagnostics,
        "original_train_state_count": len(train_rows),
        "train_61d_leave_one_out_nearest_distance": _quantiles(loo_distance),
        "student_61d_nearest_train_distance_recomputed": _quantiles(student_distance),
        "student_61d_nearest_train_distance_logged": _quantiles([
            float(t["cart"]["nearest_training_distance_normalized_61d_l2"]) for t in traces
        ]),
        "student_vs_train_61d_centroid_shift_top10": top_shifted,
        "phase14g_metrics_sha256": actual_hashes["metrics"],
    }
    if computed["forced_action_branch_count"] != 8298:
        raise Phase15AuditError("Phase14G forced-action branch count differs from expected 9 x trace count")
    if not np.allclose(student_distance, [t["cart"]["nearest_training_distance_normalized_61d_l2"] for t in traces], rtol=1e-10, atol=1e-10):
        raise Phase15AuditError("recomputed student-to-train nearest distances differ from logged values")
    return computed
