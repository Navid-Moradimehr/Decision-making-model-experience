"""Phase14D independent offline gate lock, evaluation and paired statistics."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14c import extract_arrays
from consequence_lab.phase14d import (
    _fit_candidate,
    collect_model_selection_rows,
    _source_hashes as phase14d_source_hashes,
    _write_json,
)


CONFIG_PATH = "configs/phase14d_offline_gate.yaml"
SOURCE_FILES = (
    CONFIG_PATH,
    "reports/phase14d_decision_function_generalization_protocol.md",
    "reports/phase14d_offline_gate_protocol.md",
    "reports/phase14d_offline_gate_v1_pre_data_failure.md",
    "configs/phase14d.yaml",
    "reports/phase14d_model_selection_v2_result.md",
    "src/consequence_lab/phase14e.py",
    "src/consequence_lab/phase14d.py",
    "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase14b.py",
    "src/consequence_lab/phase14.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "scripts/freeze_phase14d_offline_gate.py",
    "scripts/run_phase14d_offline_gate.py",
    "tests/test_phase14e.py",
)


class Phase14EIntegrityError(RuntimeError):
    """Offline-gate source lock, selected model, or outcome integrity failure."""


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14EIntegrityError(f"git {' '.join(args)} failed") from exc


def _hash_sources(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def _parent_artifacts(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent"]
    source_lock_path = root / parent["phase14d_selection_lock"]
    selection_manifest_path = root / parent["phase14d_selection_manifest"]
    metrics_path = root / parent["phase14d_selection_metrics"]
    labels_path = root / parent["phase14d_selection_labels"]
    report_path = root / parent["phase14d_selection_report"]
    audit_path = root / parent["phase14d_layout_audit"]
    training_path = root / parent["training_records"]
    training_manifest_path = root / parent["training_manifest"]
    paths = (source_lock_path, selection_manifest_path, metrics_path, labels_path,
             report_path, audit_path, training_path, training_manifest_path)
    if any(not path.is_file() for path in paths):
        raise Phase14EIntegrityError("one or more Phase14D parent artifacts are missing")
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    selection_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    phase14d_config = yaml.safe_load((root / parent["phase14d_config"]).read_text(encoding="utf-8"))
    phase14b_lock_path = root / phase14d_config["source"]["phase14b_lock"]
    previous_lock_path = root / parent["previous_v1_lock"]
    previous_manifest_path = root / parent["previous_v1_manifest"]
    previous_report_path = root / parent["previous_v1_failure_report"]
    if any(not path.is_file() for path in (previous_lock_path, previous_manifest_path, previous_report_path)):
        raise Phase14EIntegrityError("preserved v1 pre-data failure artifacts are missing")
    previous_lock = json.loads(previous_lock_path.read_text(encoding="utf-8"))
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    previous_files = sorted(path.name for path in (root / parent["previous_v1_run_dir"]).iterdir())
    offline = config["offline_gate"]
    expected_exact_teacher = {
        "horizon": int(phase14d_config["teacher"]["planner"]["horizon"]),
        "sequences": int(phase14d_config["teacher"]["planner"]["sequences"]),
        "seed_schedule": phase14d_config["teacher"]["seed_schedule"],
    }
    expected_geometry = {
        "route": phase14d_config["geometry_comparator"]["route"],
        "resolution": float(phase14d_config["geometry_comparator"]["resolution"]),
        "waypoint_tolerance": float(phase14d_config["geometry_comparator"]["waypoint_tolerance"]),
    }
    if (source_lock.get("schema") != "phase14d_model_selection_lock_v2"
            or source_lock.get("status") != "authorized_model_selection_only"
            or selection_manifest.get("status") != "completed_model_selection"
            or selection_manifest.get("offline_gate_labels_collected") is not False
            or selection_manifest.get("reserve_access") is not False
            or selection_manifest.get("scope", {}).get("model_selection_only") is not True
            or audit.get("status") != "hash_only_passed_no_outcomes"
            or audit.get("task_transitions_evaluated") is not False
            or audit.get("episode_outcomes_read") is not False
            or audit.get("reserve_performance_outcomes_read") is not False
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("all_phase14d_blocks_pairwise_disjoint") is not True
            or audit.get("all_blocks_disjoint_from_prior_outcomes") is not True
            or training_manifest.get("status") != "completed_teacher_values_gate_passed"
            or training_manifest.get("reserve_accessed") is not False
            or training_manifest.get("heldout_access") is not False
            or training_manifest.get("student_training") is not False
            or training_manifest.get("world_model") is not False
            or not phase14b_lock_path.is_file()
            or offline["exact_teacher"] != expected_exact_teacher
            or offline["geometry"] != expected_geometry
            or offline["trajectory_fractions"] != phase14d_config["splits"]["trajectory_fractions"]
            or previous_lock.get("schema") != "phase14d_offline_gate_lock_v1"
            or previous_lock.get("status") != "authorized_independent_offline_gate_only"
            or previous_lock.get("reserve_access_authorized") is not False
            or previous_lock.get("authorized_block", {}).get("seed_start") != int(offline["seed_start"])
            or previous_lock.get("authorized_block", {}).get("count") != int(offline["layout_count"])
            or previous_lock.get("authorized_block", {}).get("layout_sha256")
                != [item.get("layout_sha256") for item in audit.get("blocks", {}).get("offline_gate", [])]
            or previous_lock.get("parent_artifacts", {}).get("layout_audit_sha256") != sha256_file(audit_path)
            or previous_lock.get("gate") != offline
            or previous_lock.get("selected_model", {}).get("learner") != config["selected_model"]["learner"]
            or previous_lock.get("selected_model", {}).get("feature_variant") != config["selected_model"]["feature_variant"]
            or previous_lock.get("selected_model", {}).get("target_variant") != config["selected_model"]["target_variant"]
            or previous_manifest.get("status") != "aborted_or_incomplete"
            or previous_manifest.get("error_type") != "KeyError"
            or previous_manifest.get("error") != "'count'"
            or previous_manifest.get("offline_gate_only") is not True
            or previous_manifest.get("reserve_access") is not None
            or previous_manifest.get("source_commit") != previous_lock.get("source_commit")
            or previous_manifest.get("scope") != {
                "closed_loop": False, "dagger": False,
                "learned_world_model": False, "reserve_access": False,
            }
            or previous_files != ["manifest.json"]
            or previous_manifest.get("lock_sha256") != sha256_file(previous_lock_path)
            or previous_lock.get("outputs", {}).get("run_dir") != parent["previous_v1_run_dir"]):
        raise Phase14EIntegrityError("Phase14D selection/train parent gate or scope did not pass")
    prior_config = phase14d_config["splits"]["model_selection"]
    selection_audit_rows = audit.get("blocks", {}).get("model_selection", [])
    if (selection_manifest.get("lock_sha256") != sha256_file(source_lock_path)
            or selection_manifest.get("source_commit") != source_lock.get("source_commit")
            or selection_manifest.get("hash_audit_sha256") != source_lock.get("phase14d_hash_audit_sha256")
            or source_lock.get("phase14d_hash_audit_sha256") != sha256_file(audit_path)
            or source_lock.get("phase14d_source_sha256") != phase14d_source_hashes(root)
            or audit.get("generation_source_sha256") != source_lock.get("phase14d_source_sha256")
            or audit.get("generation_git_commit") != source_lock.get("source_commit")
            or source_lock.get("authorized_block") != prior_config
            or selection_manifest.get("authorized_seed_start") != int(prior_config["seed_start"])
            or selection_manifest.get("authorized_seed_count") != int(prior_config["count"])
            or [item.get("seed") for item in selection_audit_rows]
                != list(range(int(prior_config["seed_start"]), int(prior_config["seed_start"]) + int(prior_config["count"])))
            or source_lock.get("authorized_layout_sha256")
                != [item.get("layout_sha256") for item in selection_audit_rows]
            or selection_manifest.get("layout_count") != int(prior_config["count"])
            or selection_manifest.get("state_count") != 2 * int(prior_config["count"])
            or selection_manifest.get("teacher_action_value_count") != 18 * int(prior_config["count"])
            or selection_manifest.get("metrics_sha256") != sha256_file(metrics_path)
            or selection_manifest.get("teacher_labels_sha256") != sha256_file(labels_path)
            or selection_metrics.get("selection_labels_sha256") != sha256_file(labels_path)
            or selection_metrics.get("selection_layout_count") != int(prior_config["count"])
            or selection_metrics.get("selection_state_count") != selection_manifest.get("state_count")
            or selection_metrics.get("training_record_sha256") != sha256_file(training_path)
            or source_lock.get("phase14b_training_records_sha256") != sha256_file(training_path)
            or source_lock.get("phase14b_training_manifest_sha256") != sha256_file(training_manifest_path)
            or training_manifest.get("records_sha256") != sha256_file(training_path)
            or training_manifest.get("continuation_lock_sha256") != sha256_file(phase14b_lock_path)):
        raise Phase14EIntegrityError("selection or training artifacts fail parent hash verification")
    selected = selection_metrics.get("selected_candidate")
    expected_selected = config["selected_model"]
    if selected is None or any(selected.get(key) != expected_selected[key] for key in (
        "learner", "feature_variant", "target_variant",
    )):
        raise Phase14EIntegrityError("selected model config does not match the selection result")
    offline_rows = audit.get("blocks", {}).get("offline_gate", [])
    offline = config["offline_gate"]
    expected_offline_seeds = list(range(int(offline["seed_start"]), int(offline["seed_start"]) + int(offline["layout_count"])))
    if ([item.get("seed") for item in offline_rows] != expected_offline_seeds
            or len({item.get("layout_sha256") for item in offline_rows}) != len(offline_rows)):
        raise Phase14EIntegrityError("offline gate hash block count mismatch")
    return {
        "source_lock": source_lock,
        "selected_candidate": selected,
        "selection_manifest_sha256": sha256_file(selection_manifest_path),
        "selection_metrics_sha256": sha256_file(metrics_path),
        "selection_labels_sha256": sha256_file(labels_path),
        "selection_report_sha256": sha256_file(report_path),
        "layout_audit_sha256": sha256_file(audit_path),
        "offline_layout_hashes": [item["layout_sha256"] for item in offline_rows],
        "training_records_sha256": sha256_file(training_path),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "preserved_v1_lock_sha256": sha256_file(previous_lock_path),
        "preserved_v1_manifest_sha256": sha256_file(previous_manifest_path),
        "preserved_v1_failure_report_sha256": sha256_file(previous_report_path),
    }


def create_offline_gate_lock(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if _git(root, "status", "--porcelain"):
        raise Phase14EIntegrityError("commit Phase14E source before freezing offline gate lock")
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema") != "phase14d_offline_gate_config_v2":
        raise Phase14EIntegrityError("offline-gate v2 config schema is required")
    parent = _parent_artifacts(root, config)
    return {
        "schema": "phase14d_offline_gate_lock_v2",
        "status": "authorized_independent_offline_gate_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "authorization_reference": "User authorized progression through evidence-gated Phase14 student training; model selection completed under D124, independent gate remains conditional on its own lock.",
        "config_sha256": sha256_file(root / CONFIG_PATH),
        "source_sha256": _hash_sources(root),
        "phase14d_selection_lock_sha256": sha256_file(root / config["parent"]["phase14d_selection_lock"]),
        "parent_artifacts": {key: value for key, value in parent.items() if key != "source_lock"},
        "selected_model": {
            key: parent["selected_candidate"][key]
            for key in ("learner", "feature_variant", "target_variant")
        } | {
            "fit_data_sha256": parent["training_records_sha256"],
            "fit_only_on_training_split": True,
            "action_selection": config["selected_model"]["action_selection"],
            "hyperparameter_changes": False,
        },
        "authorized_block": {
            "seed_start": int(config["offline_gate"]["seed_start"]),
            "count": int(config["offline_gate"]["layout_count"]),
            "layout_sha256": parent["offline_layout_hashes"],
        },
        "gate": config["offline_gate"],
        "closed_loop_authorized": False,
        "dagger_authorized": False,
        "learned_world_model_authorized": False,
        "reserve_access_authorized": False,
        "resource_caps": {"wall_time_seconds": None, "storage_bytes": None},
        "outputs": config["outputs"],
        "overwrite_forbidden": True,
    }


def _expected_lock_bindings(
    config: dict[str, Any],
    parent: dict[str, Any],
    *,
    source_commit: str,
    source_sha256: dict[str, str],
    selection_lock_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Return every lock field that must remain identical before outcomes."""
    return {
        "source_commit": source_commit,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "phase14d_selection_lock_sha256": selection_lock_sha256,
        "parent_artifacts": {key: value for key, value in parent.items() if key != "source_lock"},
        "selected_model": {
            key: parent["selected_candidate"][key]
            for key in ("learner", "feature_variant", "target_variant")
        } | {
            "fit_data_sha256": parent["training_records_sha256"],
            "fit_only_on_training_split": True,
            "action_selection": config["selected_model"]["action_selection"],
            "hyperparameter_changes": False,
        },
        "authorized_block": {
            "seed_start": int(config["offline_gate"]["seed_start"]),
            "count": int(config["offline_gate"]["layout_count"]),
            "layout_sha256": parent["offline_layout_hashes"],
        },
        "gate": config["offline_gate"],
        "closed_loop_authorized": False,
        "dagger_authorized": False,
        "learned_world_model_authorized": False,
        "reserve_access_authorized": False,
        "resource_caps": {"wall_time_seconds": None, "storage_bytes": None},
        "outputs": config["outputs"],
        "overwrite_forbidden": True,
    }


def _validate_lock_bindings(lock: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = [key for key, value in expected.items() if lock.get(key) != value]
    if mismatches:
        raise Phase14EIntegrityError(f"offline lock binding mismatch: {', '.join(mismatches)}")


def validate_offline_gate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(root).resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema") != "phase14d_offline_gate_config_v2":
        raise Phase14EIntegrityError("offline-gate v2 config schema is required")
    lock_path = root / config["outputs"]["lock"]
    if not lock_path.is_file():
        raise Phase14EIntegrityError("offline-gate lock is missing")
    if _git(root, "status", "--porcelain"):
        raise Phase14EIntegrityError("working tree must be clean before offline-gate collection")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("schema") != "phase14d_offline_gate_lock_v2"
            or lock.get("status") != "authorized_independent_offline_gate_only"):
        raise Phase14EIntegrityError("offline-gate lock has wrong schema/status")
    parent = _parent_artifacts(root, config)
    if lock.get("source_commit") is None:
        raise Phase14EIntegrityError("offline lock has no source commit")
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", str(lock["source_commit"]), "HEAD"],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14EIntegrityError("offline lock source commit is not an ancestor of HEAD") from exc
    expected = _expected_lock_bindings(
        config, parent, source_commit=lock["source_commit"], source_sha256=_hash_sources(root),
        selection_lock_sha256=sha256_file(root / config["parent"]["phase14d_selection_lock"]),
        config_sha256=sha256_file(root / CONFIG_PATH),
    )
    _validate_lock_bindings(lock, expected)
    output_dir = root / config["outputs"]["run_dir"]
    report_path = root / config["outputs"]["report"]
    if output_dir.exists() or report_path.exists():
        raise Phase14EIntegrityError("refusing to overwrite existing offline-gate outputs")
    return config, lock, parent


def paired_layout_bootstrap(
    layout_improvements: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(layout_improvements, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires a finite vector of at least two layout differences")
    if resamples < 1000:
        raise ValueError("bootstrap resamples must be at least 1000")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[samples].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "layout_count": int(len(values)),
        "resamples": int(resamples),
        "seed": int(seed),
        "mean_improvement": float(values.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "layout_improvements": values.tolist(),
    }


def _adapt_phase14d_config_for_offline_gate(
    phase14d_config: dict[str, Any], offline_gate: dict[str, Any],
) -> dict[str, Any]:
    phase14d_config = deepcopy(phase14d_config)
    phase14d_config["splits"]["model_selection"] = {
        "seed_start": int(offline_gate["seed_start"]),
        "count": int(offline_gate["layout_count"]),
        "purpose": "independent_offline_gate_only",
    }
    phase14d_config["splits"]["trajectory_fractions"] = offline_gate["trajectory_fractions"]
    return phase14d_config


def _offline_gate_rows(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    parent_config = yaml.safe_load((root / config["parent"]["phase14d_config"]).read_text(encoding="utf-8"))
    phase14d_config = _adapt_phase14d_config_for_offline_gate(parent_config, config["offline_gate"])
    rows = collect_model_selection_rows(phase14d_config)
    for row in rows:
        row["split"] = "offline_gate"
    return rows


def run_offline_gate(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config, lock, parent = validate_offline_gate_lock(root)
    output_dir = root / config["outputs"]["run_dir"]
    report_path = root / config["outputs"]["report"]
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.json"
    labels_path = output_dir / "teacher_labels.jsonl"
    metrics_path = output_dir / "metrics.json"
    clock_start = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema": "phase14d_offline_gate_result_v2",
        "status": "collecting_locked_offline_gate_labels",
        "source_commit": lock["source_commit"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": sha256_file(root / config["outputs"]["lock"]),
        "offline_gate_only": True,
        "scope": {"closed_loop": False, "dagger": False, "learned_world_model": False, "reserve_access": False},
        "runtime_environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    _write_json(manifest_path, manifest)
    try:
        rows = _offline_gate_rows(config, root)
        actual_hashes = sorted({row["source_layout_sha256"] for row in rows})
        expected_hashes = sorted(lock["authorized_block"]["layout_sha256"])
        if actual_hashes != expected_hashes or len(actual_hashes) != 40:
            raise Phase14EIntegrityError("generated offline layouts differ from frozen lock")
        labels_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        labels_sha = sha256_file(labels_path)
        training_rows = [
            json.loads(line) for line in
            (root / config["parent"]["training_records"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_raw, train_regret, train_objective = extract_arrays(training_rows)
        selected = lock["selected_model"]
        student = _fit_candidate(
            selected["learner"], selected["feature_variant"], selected["target_variant"],
            train_raw, train_regret, train_objective, rows,
            int(config["selected_model"]["fit_seed"]),
        )
        choices = np.asarray(student["selected_action_ids"], dtype=np.int64)
        _, eval_regret, _ = extract_arrays(rows)
        geometry_actions = np.asarray([row["geometry_waypoint_action_id"] for row in rows], dtype=np.int64)
        exact_actions = np.asarray([row["exact_shooting_action_id"] for row in rows], dtype=np.int64)
        student_regret = eval_regret[np.arange(len(rows)), choices]
        geometry_regret = eval_regret[np.arange(len(rows)), geometry_actions]
        exact_regret = eval_regret[np.arange(len(rows)), exact_actions]
        per_layout: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            per_layout.setdefault(row["source_layout_sha256"], []).append(index)
        student_layout = np.asarray([student_regret[indices].mean() for indices in per_layout.values()])
        geometry_layout = np.asarray([geometry_regret[indices].mean() for indices in per_layout.values()])
        exact_layout = np.asarray([exact_regret[indices].mean() for indices in per_layout.values()])
        bootstrap = paired_layout_bootstrap(
            geometry_layout - student_layout,
            resamples=int(config["offline_gate"]["bootstrap"]["resamples"]),
            seed=int(config["offline_gate"]["bootstrap"]["seed"]),
        )
        point_improvement = float(geometry_layout.mean() - student_layout.mean())
        passed = (
            point_improvement >= float(config["offline_gate"]["minimum_point_improvement"])
            and bootstrap["lower_95"] > 0.0
        )
        chosen_set = [
            {int(action["forced_action_id"]) for action in row["actions"] if float(action["normalized_regret"]) == 0.0}
            for row in rows
        ]
        best_agreement = float(np.mean([int(choice in best) for choice, best in zip(choices, chosen_set)]))
        metrics = {
            "schema": "phase14d_offline_gate_metrics_v2",
            "status": "passed" if passed else "failed",
            "gate_passed": bool(passed),
            "selected_model": selected,
            "layout_count": len(per_layout),
            "state_count": len(rows),
            "forced_action_value_count": 9 * len(rows),
            "student_layout_macro_mean_regret": float(student_layout.mean()),
            "geometry_layout_macro_mean_regret": float(geometry_layout.mean()),
            "exact_shooting_action_layout_macro_mean_regret": float(exact_layout.mean()),
            "point_improvement_over_geometry": point_improvement,
            "paired_layout_bootstrap": bootstrap,
            "student_median_regret": float(np.median(student_regret)),
            "student_best_action_set_agreement": best_agreement,
            "student_clear_pair_order_accuracy": student["clear_pair_order_accuracy"],
            "student_catastrophic_regret_fraction_ge_0_5": float(np.mean(student_regret >= 0.5)),
            "student_action_frequency": np.bincount(choices, minlength=9).tolist(),
            "geometry_action_frequency": np.bincount(geometry_actions, minlength=9).tolist(),
            "exact_shooting_action_frequency": np.bincount(exact_actions, minlength=9).tolist(),
            "student_median_inference_seconds": student["median_inference_seconds_per_state"],
            "student_p95_inference_seconds": student["p95_inference_seconds_per_state"],
            "student_serialized_model_bytes": student["serialized_model_bytes"],
            "student_node_count": student["parameter_count_or_node_count"],
            "training_record_sha256": parent["training_records_sha256"],
            "offline_labels_sha256": labels_sha,
        }
        _write_json(metrics_path, metrics)
        manifest.update({
            "status": "completed_offline_gate",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - clock_start,
            "layout_count": len(per_layout),
            "state_count": len(rows),
            "teacher_labels_sha256": labels_sha,
            "metrics_sha256": sha256_file(metrics_path),
            "gate_passed": bool(passed),
            "reserve_access": False,
        })
        _write_json(manifest_path, manifest)
        _write_offline_report(report_path, metrics, manifest)
        return metrics
    except BaseException as exc:
        manifest.update({
            "status": "aborted_or_incomplete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - clock_start,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        _write_json(manifest_path, manifest)
        raise


def _write_offline_report(path: Path, metrics: dict[str, Any], manifest: dict[str, Any]) -> None:
    bootstrap = metrics["paired_layout_bootstrap"]
    lines = [
        "# Phase 14D independent offline-gate result", "",
        f"**Gate:** {metrics['status']}.", "",
        f"Layouts/states/action values: {metrics['layout_count']} / {metrics['state_count']} / {metrics['forced_action_value_count']}.",
        f"Selected model: `{metrics['selected_model']['learner']} / {metrics['selected_model']['feature_variant']} / {metrics['selected_model']['target_variant']}`.", "",
        "| Controller | Layout-macro mean normalized regret |", "|---|---:|",
        f"| Student | {metrics['student_layout_macro_mean_regret']:.6f} |",
        f"| Geometry waypoint | {metrics['geometry_layout_macro_mean_regret']:.6f} |",
        f"| Exact-shooting action | {metrics['exact_shooting_action_layout_macro_mean_regret']:.6f} |", "",
        f"Student improvement over geometry: {metrics['point_improvement_over_geometry']:.6f}; paired layout-bootstrap 95% CI [{bootstrap['lower_95']:.6f}, {bootstrap['upper_95']:.6f}].",
        f"Gate requires improvement >= 0.05 and lower 95% bound > 0; result: **{'PASS' if metrics['gate_passed'] else 'FAIL'}**.", "",
        f"Student best-action-set agreement: {metrics['student_best_action_set_agreement']:.4f}; clear-pair order accuracy: {metrics['student_clear_pair_order_accuracy']:.4f}; catastrophic regret fraction: {metrics['student_catastrophic_regret_fraction_ge_0_5']:.4f}.",
        f"Median/p95 inference latency: {1000*metrics['student_median_inference_seconds']:.4f}/{1000*metrics['student_p95_inference_seconds']:.4f} ms; serialized payload {metrics['student_serialized_model_bytes']} bytes; tree nodes {metrics['student_node_count']}.", "",
        f"Offline label SHA-256: `{manifest.get('teacher_labels_sha256')}`; metrics SHA-256: `{manifest.get('metrics_sha256')}`; lock SHA-256: `{manifest.get('lock_sha256')}`.", "",
        "No closed-loop, DAgger, learned-world-model, or reserve outcome was accessed. If this gate passes, closed-loop development still requires a separate lock; if it fails, stop this branch.", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
