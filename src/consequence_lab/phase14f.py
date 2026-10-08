"""Locked closed-loop development evaluation for the Phase14D selected CART."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
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

from consequence_lab.corridor_baselines import grid_astar_route, waypoint_braking_action
from consequence_lab.corridor_evaluation import (
    CorridorEpisodeResult,
    evaluate_corridor_episode,
    paired_outcome_rate_interval,
    summarize_corridor_results,
)
from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step, validate_corridor_state
from consequence_lab.phase12_teacher import state_features
from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14c import _SquaredErrorTree, build_feature_matrix, extract_arrays
from consequence_lab.phase14d import _source_hashes as phase14d_source_hashes
from consequence_lab.phase14e import _hash_sources as phase14e_source_hashes, _parent_artifacts


CONFIG_PATH = "configs/phase14f_closed_loop.yaml"
SOURCE_FILES = (
    CONFIG_PATH,
    "reports/phase14f_closed_loop_protocol.md",
    "reports/phase14d_decision_function_generalization_protocol.md",
    "reports/phase14d_offline_gate_protocol.md",
    "reports/phase14d_offline_gate_v1_pre_data_failure.md",
    "reports/phase14d_offline_gate_v2_result.md",
    "reports/phase13a_baseline_headroom_result.md",
    "configs/phase14d.yaml",
    "configs/phase14d_offline_gate.yaml",
    "configs/phase13.yaml",
    "src/consequence_lab/phase14f.py",
    "src/consequence_lab/phase14e.py",
    "src/consequence_lab/phase14d.py",
    "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase14b.py",
    "src/consequence_lab/phase14.py",
    "src/consequence_lab/phase13.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/corridor_evaluation.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "scripts/freeze_phase14f_student.py",
    "scripts/freeze_phase14f_lock.py",
    "scripts/run_phase14f_closed_loop.py",
    "tests/test_phase14f.py",
    "second-brain/00_Project_Home.md",
    "second-brain/14_Decisions.md",
    "second-brain/16_Experiment_Log.md",
    "second-brain/phases/Phase_14_First_Action_Value_Learnability.md",
)


class Phase14FIntegrityError(RuntimeError):
    """Closed-loop lock, parent evidence, model artifact, or result integrity error."""


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14FIntegrityError(f"git {' '.join(args)} failed") from exc


def _hash_sources(root: Path) -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise Phase14FIntegrityError(f"locked source files missing: {missing}")
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _serialize_tree(estimator: _SquaredErrorTree) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    indices = {id(estimator.root): 0}
    pending = deque([estimator.root])
    while pending:
        node = pending.popleft()
        for child in (node.left, node.right):
            if child is not None and id(child) not in indices:
                indices[id(child)] = len(indices)
                pending.append(child)
        nodes.append({
            "value": [float(value) for value in node.value],
            "feature": int(node.feature),
            "threshold": float(node.threshold),
            "left": None if node.left is None else indices[id(node.left)],
            "right": None if node.right is None else indices[id(node.right)],
        })
    return nodes


def _make_student_payload(root: Path, config: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    training_path = root / config["parent"]["training_records"]
    if sha256_file(training_path) != parent["training_records_sha256"]:
        raise Phase14FIntegrityError("training records differ from the verified Phase14D parent")
    rows = _read_jsonl(training_path)
    if len(rows) != 2095:
        raise Phase14FIntegrityError("selected model fit source is not the complete Phase14B train split")
    raw, regrets, _ = extract_arrays(rows)
    if not np.isfinite(raw).all() or not np.isfinite(regrets).all():
        raise Phase14FIntegrityError("training arrays contain non-finite values")
    variant = config["model"]["feature_variant"]
    x = build_feature_matrix(raw, variant)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    estimator = _SquaredErrorTree().fit((x - x_mean) / x_std, regrets)
    selected = parent["selected_candidate"]
    if (estimator.node_count != int(config["model"]["expected_nodes"])
            or any(selected.get(key) != config["model"][key] for key in (
                "learner", "feature_variant", "target_variant",
            ))):
        raise Phase14FIntegrityError("training-only refit differs from the locked selected CART")
    return {
        "schema": "phase14f_frozen_student_cart_v1",
        "selected_model": {
            "learner": config["model"]["learner"],
            "feature_variant": config["model"]["feature_variant"],
            "target_variant": config["model"]["target_variant"],
            "action_selection": config["model"]["action_selection"],
        },
        "training_records_sha256": parent["training_records_sha256"],
        "training_record_count": len(rows),
        "offline_gate_lock_sha256": parent["offline_lock_sha256"],
        "offline_metrics_sha256": parent["offline_metrics_sha256"],
        "offline_labels_sha256": parent["offline_labels_sha256"],
        "state_feature_count": 52,
        "action_count": 9,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "nodes": _serialize_tree(estimator),
        "node_count": estimator.node_count,
    }


def verify_payload_against_offline_gate(root: Path, config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Confirm deterministic training-only refit exactly reproduces locked offline predictions."""
    labels_path = root / config["parent"]["phase14d_offline_run_dir"] / "teacher_labels.jsonl"
    metrics_path = root / config["parent"]["phase14d_offline_run_dir"] / "metrics.json"
    rows = _read_jsonl(labels_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    policy = FrozenCartPolicy(payload)
    choices = np.asarray([policy.action_for_features(row["student_input_52"]) for row in rows], dtype=np.int64)
    _, regrets, _ = extract_arrays(rows)
    actual_frequency = np.bincount(choices, minlength=9).tolist()
    if actual_frequency != metrics["student_action_frequency"]:
        raise Phase14FIntegrityError("frozen CART does not reproduce offline-gate action frequencies")
    if len(rows) != int(metrics["state_count"]):
        raise Phase14FIntegrityError("offline consistency rows differ from the frozen offline state count")
    chosen_regret = regrets[np.arange(len(rows)), choices]
    per_layout: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        per_layout.setdefault(row["source_layout_sha256"], []).append(index)
    layout_macro = float(np.mean([chosen_regret[ids].mean() for ids in per_layout.values()]))
    if not np.isclose(layout_macro, metrics["student_layout_macro_mean_regret"], atol=1e-12, rtol=0.0):
        raise Phase14FIntegrityError("frozen CART does not reproduce offline-gate regret")
    return {
        "checked_states": len(rows),
        "action_frequency": actual_frequency,
        "layout_macro_mean_regret": layout_macro,
        "matches_offline_gate_model": True,
    }


class FrozenCartPolicy:
    """Tiny deployable evaluator for the canonical serialized multi-output CART."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if (payload.get("schema") != "phase14f_frozen_student_cart_v1"
                or payload.get("state_feature_count") != 52
                or payload.get("action_count") != 9):
            raise Phase14FIntegrityError("frozen student model schema or dimensions are invalid")
        self.payload = payload
        self.mean = np.asarray(payload["x_mean"], dtype=np.float64)
        self.std = np.asarray(payload["x_std"], dtype=np.float64)
        self.nodes = payload["nodes"]
        if (self.mean.shape != (61,) or self.std.shape != (61,)
                or len(self.nodes) != int(payload["node_count"])
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise Phase14FIntegrityError("frozen CART transform/tree payload is malformed")

    def action_for_features(self, raw_features: Any) -> int:
        raw = np.asarray(raw_features, dtype=np.float64)
        if raw.shape != (52,) or not np.isfinite(raw).all():
            raise Phase14FIntegrityError("student runtime features must be 52 finite public values")
        x = build_feature_matrix(raw[None, :], self.payload["selected_model"]["feature_variant"])
        row = ((x - self.mean) / self.std)[0]
        node_index = 0
        while True:
            node = self.nodes[node_index]
            feature = int(node["feature"])
            if feature < 0:
                values = np.asarray(node["value"], dtype=np.float64)
                if values.shape != (9,) or not np.isfinite(values).all():
                    raise Phase14FIntegrityError("CART leaf has invalid nine-action values")
                return int(np.argmin(values))
            if feature >= len(row):
                raise Phase14FIntegrityError("CART split feature is outside the declared feature vector")
            node_index = int(node["left"] if row[feature] <= float(node["threshold"]) else node["right"])
            if node_index < 0 or node_index >= len(self.nodes):
                raise Phase14FIntegrityError("CART child index is invalid")


def fit_and_freeze_student(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema") != "phase14f_closed_loop_config_v1":
        raise Phase14FIntegrityError("Phase14F v1 config required")
    output = root / config["model"]["artifact"]
    manifest_path = root / config["outputs"]["model_freeze_manifest"]
    if output.exists() or manifest_path.exists():
        raise Phase14FIntegrityError("refusing to overwrite frozen Phase14F student artifact")
    if _git(root, "status", "--porcelain"):
        raise Phase14FIntegrityError("commit Phase14F model-export source before freezing the model")
    parent = _verified_parent(root, config)
    payload = _make_student_payload(root, config, parent)
    consistency = verify_payload_against_offline_gate(root, config, payload)
    artifact_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(artifact_bytes)
    manifest = {
        "schema": "phase14f_student_freeze_manifest_v1",
        "status": "frozen_train_only_student",
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "config_sha256": sha256_file(root / CONFIG_PATH),
        "source_sha256": _hash_sources(root),
        "model_sha256": hashlib.sha256(artifact_bytes).hexdigest().upper(),
        "model_bytes": len(artifact_bytes),
        "model_nodes": payload["node_count"],
        "fit_records_sha256": parent["training_records_sha256"],
        "offline_consistency_check": consistency,
        "fit_only_on_training_split": True,
        "offline_gate_read_for_consistency_only": True,
        "reserve_access": False,
        "closed_loop_evaluation": False,
        "dagger": False,
        "learned_world_model": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _verified_parent(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    offline_config_path = root / config["parent"]["phase14d_offline_config"]
    offline_config = yaml.safe_load(offline_config_path.read_text(encoding="utf-8"))
    offline_lock_path = root / config["parent"]["phase14d_offline_lock"]
    offline_lock = json.loads(offline_lock_path.read_text(encoding="utf-8"))
    parent = _parent_artifacts(root, offline_config)
    if (offline_config.get("schema") != "phase14d_offline_gate_config_v2"
            or offline_lock.get("schema") != "phase14d_offline_gate_lock_v2"
            or offline_lock.get("status") != "authorized_independent_offline_gate_only"
            or offline_lock.get("source_sha256") != phase14e_source_hashes(root)
            or offline_lock.get("config_sha256") != sha256_file(offline_config_path)
            or offline_lock.get("parent_artifacts") != {key: value for key, value in parent.items() if key != "source_lock"}
            or any(offline_lock.get(key) is not False for key in (
                "closed_loop_authorized", "dagger_authorized", "learned_world_model_authorized", "reserve_access_authorized",
            ))):
        raise Phase14FIntegrityError("Phase14D offline source lock is invalid or scope-expanded")
    run_dir = root / config["parent"]["phase14d_offline_run_dir"]
    labels_path = run_dir / "teacher_labels.jsonl"
    metrics_path = run_dir / "metrics.json"
    manifest_path = run_dir / "manifest.json"
    report_path = root / config["parent"]["phase14d_offline_report"]
    for path in (labels_path, metrics_path, manifest_path, report_path):
        if not path.is_file():
            raise Phase14FIntegrityError(f"offline gate parent artifact is missing: {path.name}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (metrics.get("schema") != "phase14d_offline_gate_metrics_v2"
            or metrics.get("gate_passed") is not True
            or metrics.get("status") != "passed"
            or manifest.get("status") != "completed_offline_gate"
            or manifest.get("gate_passed") is not True
            or manifest.get("reserve_access") is not False
            or manifest.get("lock_sha256") != sha256_file(offline_lock_path)
            or manifest.get("metrics_sha256") != sha256_file(metrics_path)
            or manifest.get("teacher_labels_sha256") != sha256_file(labels_path)
            or manifest.get("scope") != {
                "closed_loop": False, "dagger": False,
                "learned_world_model": False, "reserve_access": False,
            }):
        raise Phase14FIntegrityError("Phase14D offline gate did not pass with closed downstream scope")
    selected = config["model"]
    for key in ("learner", "feature_variant", "target_variant"):
        if metrics.get("selected_model", {}).get(key) != selected[key]:
            raise Phase14FIntegrityError("Phase14F selected model differs from the offline-gate model")

    d_config = yaml.safe_load((root / config["parent"]["phase14d_config"]).read_text(encoding="utf-8"))
    d_audit_path = root / config["parent"]["phase14d_layout_audit"]
    d_selection_lock_path = root / config["parent"]["phase14d_selection_lock"]
    d_audit = json.loads(d_audit_path.read_text(encoding="utf-8"))
    d_lock = json.loads(d_selection_lock_path.read_text(encoding="utf-8"))
    closed_spec = config["closed_loop"]
    audited = d_audit.get("blocks", {}).get("closed_loop_development", [])
    expected_seeds = list(range(int(closed_spec["seed_start"]), int(closed_spec["seed_start"]) + int(closed_spec["layout_count"])))
    if (d_config.get("splits", {}).get("closed_loop_development") != {
            "seed_start": int(closed_spec["seed_start"]),
            "count": int(closed_spec["layout_count"]),
            "purpose": "conditional_closed_loop_development",
        }
            or [item.get("seed") for item in audited] != expected_seeds
            or len({item.get("layout_sha256") for item in audited}) != len(audited)
            or d_audit.get("status") != "hash_only_passed_no_outcomes"
            or d_audit.get("task_transitions_evaluated") is not False
            or d_audit.get("episode_outcomes_read") is not False
            or d_audit.get("reserve_performance_outcomes_read") is not False
            or d_audit.get("reserve_generated_or_accessed") is not False
            or d_audit.get("all_phase14d_blocks_pairwise_disjoint") is not True
            or d_audit.get("all_blocks_disjoint_from_prior_outcomes") is not True
            or d_lock.get("reserve_access_authorized") is not False):
        raise Phase14FIntegrityError("conditional closed-loop layout block fails the prior hash audit")
    if (d_lock.get("phase14d_source_sha256") != phase14d_source_hashes(root)
            or d_lock.get("phase14d_hash_audit_sha256") != sha256_file(d_audit_path)):
        raise Phase14FIntegrityError("Phase14D layout hash audit/source do not match its committed lock")

    p13_config = yaml.safe_load((root / config["parent"]["phase13a_config"]).read_text(encoding="utf-8"))
    p13_rules = p13_config.get("phase13A", {}).get("independent_comparison", {})
    p14_rules = d_config.get("closed_loop_gate_if_offline_passes", {})
    ref = config["reference_phase13a"]
    p13_report_path = root / config["parent"]["phase13a_report"]
    p13_report = p13_report_path.read_text(encoding="utf-8")
    if (int(ref["independent_layout_count"]) != 100
            or float(ref["exact_shooting_safe_success"]) != 1.0
            or float(ref["geometry_safe_success"]) != 0.63
            or float(ref["safe_success_gap"]) != 0.37
            or int(p13_config["phase13A"]["independent_comparison"]["seed_start"]) != 190000
            or int(p13_config["phase13A"]["independent_comparison"]["count"]) != 100
            or p14_rules.get("layouts") != {"seed_start": 192000, "count": 100}
            or float(p14_rules.get("retain_fraction_of_phase13a_shooting_minus_geometry_safe_success_gap", -1))
                != float(config["gate"]["retain_fraction_of_phase13a_planner_minus_geometry_success_gap"])
            or float(p14_rules.get("paired_success_lower_95_above_geometry", -1)) != 0.0
            or float(p14_rules.get("collision_difference_upper_95_maximum", -1))
                != float(config["gate"]["paired_student_minus_geometry_collision_upper_95_maximum"])
            or float(p14_rules.get("median_latency_maximum_fraction_of_shooting", -1))
                != float(config["gate"]["student_median_decision_latency_maximum_fraction_of_exact_shooting"])
            or "Exact shooting | 100/100 (100%)" not in p13_report
            or "Geometry waypoint | 63/100 (63%)" not in p13_report):
        raise Phase14FIntegrityError("Phase13A historical reference/Phase14D closed-loop criteria mismatch")

    training_path = root / config["parent"]["training_records"]
    training_manifest_path = root / config["parent"]["training_manifest"]
    training_lock_path = root / config["parent"]["training_lock"]
    if (sha256_file(training_path) != d_lock.get("phase14b_training_records_sha256")
            or sha256_file(training_manifest_path) != d_lock.get("phase14b_training_manifest_sha256")
            or sha256_file(training_lock_path) != d_lock.get("phase14b_lock_sha256")):
        raise Phase14FIntegrityError("Phase14B train-only source differs from Phase14D selection lock")
    return {
        "selected_candidate": {
            key: metrics["selected_model"][key]
            for key in ("learner", "feature_variant", "target_variant")
        },
        "offline_lock_sha256": sha256_file(offline_lock_path),
        "offline_manifest_sha256": sha256_file(manifest_path),
        "offline_metrics_sha256": sha256_file(metrics_path),
        "offline_labels_sha256": sha256_file(labels_path),
        "offline_report_sha256": sha256_file(report_path),
        "phase14d_audit_sha256": sha256_file(d_audit_path),
        "phase14d_selection_lock_sha256": sha256_file(d_selection_lock_path),
        "phase14d_source_sha256": d_lock["phase14d_source_sha256"],
        "offline_layout_hashes": parent["offline_layout_hashes"],
        "closed_loop_layout_hashes": [item["layout_sha256"] for item in audited],
        "training_records_sha256": sha256_file(training_path),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "training_lock_sha256": sha256_file(training_lock_path),
        "phase13a_report_sha256": sha256_file(p13_report_path),
        "phase13a_config_sha256": sha256_file(root / config["parent"]["phase13a_config"]),
        "offline_metrics": metrics,
    }


def _model_freeze_manifest(root: Path, config: dict[str, Any], parent: dict[str, Any]) -> tuple[dict[str, Any], str]:
    model_path = root / config["model"]["artifact"]
    manifest_path = root / config["outputs"]["model_freeze_manifest"]
    if not model_path.is_file() or not manifest_path.is_file():
        raise Phase14FIntegrityError("frozen student CART or freeze manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_sha = sha256_file(model_path)
    if (manifest.get("schema") != "phase14f_student_freeze_manifest_v1"
            or manifest.get("status") != "frozen_train_only_student"
            or manifest.get("source_commit") is None
            or manifest.get("source_sha256") != _hash_sources(root)
            or manifest.get("config_sha256") != sha256_file(root / CONFIG_PATH)
            or manifest.get("model_sha256") != model_sha
            or manifest.get("model_bytes") != model_path.stat().st_size
            or manifest.get("model_nodes") != json.loads(model_path.read_text(encoding="utf-8")).get("node_count")
            or manifest.get("fit_records_sha256") != parent["training_records_sha256"]
            or manifest.get("fit_only_on_training_split") is not True
            or manifest.get("offline_gate_read_for_consistency_only") is not True
            or manifest.get("reserve_access") is not False
            or manifest.get("closed_loop_evaluation") is not False
            or manifest.get("dagger") is not False
            or manifest.get("learned_world_model") is not False):
        raise Phase14FIntegrityError("frozen model manifest fails source/data/scope verification")
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", str(manifest["source_commit"]), "HEAD"],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14FIntegrityError("frozen model source commit is not an ancestor of HEAD") from exc
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    expected_payload = _make_student_payload(root, config, parent)
    if payload != expected_payload:
        raise Phase14FIntegrityError("frozen model does not exactly reproduce deterministic train-only CART")
    consistency = verify_payload_against_offline_gate(root, config, payload)
    if manifest.get("offline_consistency_check") != consistency:
        raise Phase14FIntegrityError("frozen model consistency report differs from rechecked offline predictions")
    return manifest, model_sha


def _expected_lock_bindings(
    config: dict[str, Any],
    parent: dict[str, Any],
    model_manifest: dict[str, Any],
    model_sha: str,
    *,
    source_commit: str,
    source_hashes: dict[str, str],
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "source_commit": source_commit,
        "source_sha256": source_hashes,
        "config_sha256": config_sha256,
        "parent_artifacts": {key: value for key, value in parent.items() if key != "offline_metrics"},
        "student_model": {
            "artifact_sha256": model_sha,
            "manifest_sha256": sha256_file(Path(config["_root"]) / config["outputs"]["model_freeze_manifest"]),
            "model": config["model"],
            "offline_consistency_check": model_manifest["offline_consistency_check"],
        },
        "authorized_block": {
            "seed_start": int(config["closed_loop"]["seed_start"]),
            "count": int(config["closed_loop"]["layout_count"]),
            "layout_sha256": parent["closed_loop_layout_hashes"],
        },
        "gate": config["gate"],
        "reference_phase13a": config["reference_phase13a"],
        "closed_loop_authorized": True,
        "dagger_authorized": False,
        "learned_world_model_authorized": False,
        "reserve_access_authorized": False,
        "resources": config["resources"],
        "outputs": config["outputs"],
        "overwrite_forbidden": True,
    }


def _validate_lock_bindings(lock: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = [key for key, value in expected.items() if lock.get(key) != value]
    if mismatches:
        raise Phase14FIntegrityError(f"Phase14F lock binding mismatch: {', '.join(mismatches)}")


def create_closed_loop_lock(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if _git(root, "status", "--porcelain"):
        raise Phase14FIntegrityError("commit Phase14F protocol, frozen model, and notes before lock freeze")
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema") != "phase14f_closed_loop_config_v1":
        raise Phase14FIntegrityError("Phase14F closed-loop config v1 required")
    if config["closed_loop"].get("methods") != ["student", "geometry_waypoint", "random_shooting"]:
        raise Phase14FIntegrityError("Phase14F method list must match the execution API exactly")
    config["_root"] = str(root)
    parent = _verified_parent(root, config)
    model_manifest, model_sha = _model_freeze_manifest(root, config, parent)
    lock = {
        "schema": "phase14f_closed_loop_lock_v1",
        "status": "authorized_conditional_closed_loop_development_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_reference": "Phase14D independent offline gate passed; only the separately locked conditional closed-loop development criterion is opened.",
        **_expected_lock_bindings(
            config, parent, model_manifest, model_sha,
            source_commit=_git(root, "rev-parse", "HEAD"),
            source_hashes=_hash_sources(root),
            config_sha256=sha256_file(root / CONFIG_PATH),
        ),
    }
    return lock


def validate_closed_loop_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], FrozenCartPolicy]:
    root = Path(root).resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema") != "phase14f_closed_loop_config_v1":
        raise Phase14FIntegrityError("Phase14F closed-loop config v1 required")
    if config["closed_loop"].get("methods") != ["student", "geometry_waypoint", "random_shooting"]:
        raise Phase14FIntegrityError("Phase14F method list must match the execution API exactly")
    config["_root"] = str(root)
    lock_path = root / config["outputs"]["lock"]
    if not lock_path.is_file():
        raise Phase14FIntegrityError("Phase14F closed-loop lock is missing")
    if _git(root, "status", "--porcelain"):
        raise Phase14FIntegrityError("working tree must be clean before Phase14F evaluation")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("schema") != "phase14f_closed_loop_lock_v1"
            or lock.get("status") != "authorized_conditional_closed_loop_development_only"):
        raise Phase14FIntegrityError("Phase14F lock has the wrong schema or status")
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", str(lock.get("source_commit")), "HEAD"],
            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14FIntegrityError("Phase14F source commit is not an ancestor of HEAD") from exc
    parent = _verified_parent(root, config)
    model_manifest, model_sha = _model_freeze_manifest(root, config, parent)
    _validate_lock_bindings(lock, _expected_lock_bindings(
        config, parent, model_manifest, model_sha,
        source_commit=lock["source_commit"], source_hashes=_hash_sources(root),
        config_sha256=sha256_file(root / CONFIG_PATH),
    ))
    output = root / config["outputs"]["run_dir"]
    report = root / config["outputs"]["report"]
    if output.exists() or report.exists():
        raise Phase14FIntegrityError("refusing to overwrite Phase14F closed-loop outputs")
    model_payload = json.loads((root / config["model"]["artifact"]).read_text(encoding="utf-8"))
    return config, lock, parent, FrozenCartPolicy(model_payload)


def evaluate_student_episode(scenario: CorridorScenario, policy: FrozenCartPolicy) -> CorridorEpisodeResult:
    """Run the frozen deployable state-only policy, timing features + tree + step."""
    state = scenario.initial_state
    validate_corridor_state(state, scenario.spec)
    decision_count = 0
    effort = 0
    started = perf_counter_ns()
    while state.status == "active":
        raw = state_features(scenario, state)
        action = policy.action_for_features(raw)
        if action < 0 or action >= 9:
            raise Phase14FIntegrityError("student produced an illegal primitive action")
        effort += int(action != 0)
        state = corridor_step(state, action, scenario.spec).state
        decision_count += 1
    wall_seconds = (perf_counter_ns() - started) / 1_000_000_000
    return CorridorEpisodeResult(
        method="student",
        seed=scenario.seed,
        layout_sha256=scenario.layout_sha256,
        outcome=state.status,
        task_status=state.status,
        steps=state.elapsed_steps - scenario.initial_state.elapsed_steps,
        control_effort=effort,
        decision_count=decision_count,
        wall_seconds=wall_seconds,
        wall_seconds_per_decision=wall_seconds / decision_count if decision_count else None,
        shooting_queries=0,
        shooting_success_available_queries=0,
        kinodynamic_expansion_limit_queries=0,
        planner_failure=None,
    )


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_closed_loop(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config, lock, parent, policy = validate_closed_loop_lock(root)
    output = root / config["outputs"]["run_dir"]
    report = root / config["outputs"]["report"]
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    episodes_path = output / "episodes.jsonl"
    metrics_path = output / "metrics.json"
    manifest: dict[str, Any] = {
        "schema": "phase14f_closed_loop_result_v1",
        "status": "collecting_locked_closed_loop_development",
        "source_commit": lock["source_commit"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_sha256": sha256_file(root / config["outputs"]["lock"]),
        "student_model_sha256": lock["student_model"]["artifact_sha256"],
        "authorized_seed_start": lock["authorized_block"]["seed_start"],
        "authorized_layout_count": lock["authorized_block"]["count"],
        "scope": {
            "closed_loop_development": True,
            "dagger": False,
            "learned_world_model": False,
            "reserve_access": False,
        },
        "reserve_access": False,
        "runtime_environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    _write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        expected_by_seed = dict(zip(
            range(lock["authorized_block"]["seed_start"], lock["authorized_block"]["seed_start"] + lock["authorized_block"]["count"]),
            lock["authorized_block"]["layout_sha256"], strict=True,
        ))
        results: list[CorridorEpisodeResult] = []
        params = {
            "geometry_waypoint": {
                "resolution": float(config["closed_loop"]["geometry"]["resolution"]),
                "waypoint_tolerance": float(config["closed_loop"]["geometry"]["waypoint_tolerance"]),
            },
            "random_shooting": {
                "horizon": int(config["closed_loop"]["exact_shooting"]["horizon"]),
                "num_sequences": int(config["closed_loop"]["exact_shooting"]["sequences"]),
            },
        }
        for offset, (seed, expected_layout) in enumerate(expected_by_seed.items()):
            scenario = generate_staggered_corridor(seed)
            if scenario.layout_sha256 != expected_layout:
                raise Phase14FIntegrityError(f"layout hash mismatch for locked seed {seed}")
            method_order = ["student", "geometry_waypoint", "random_shooting"]
            rotation = offset % len(method_order)
            method_order = method_order[rotation:] + method_order[:rotation]
            for method in method_order:
                if method == "student":
                    result = evaluate_student_episode(scenario, policy)
                else:
                    result = evaluate_corridor_episode(scenario, method, parameters=params[method])
                results.append(result)
                with episodes_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(asdict(result), sort_keys=True) + "\n")

        if len(results) != 3 * int(config["closed_loop"]["layout_count"]):
            raise Phase14FIntegrityError("closed-loop result cardinality differs from the locked plan")
        summaries = {item.method: asdict(item) for item in summarize_corridor_results(results)}
        success_gain = paired_outcome_rate_interval(
            results, "student", "geometry_waypoint", event="success",
            bootstrap_seed=int(config["gate"]["bootstrap"]["seed"]),
            resamples=int(config["gate"]["bootstrap"]["resamples"]),
        )
        collision_diff = paired_outcome_rate_interval(
            results, "student", "geometry_waypoint", event="collision",
            bootstrap_seed=int(config["gate"]["bootstrap"]["seed"]),
            resamples=int(config["gate"]["bootstrap"]["resamples"]),
        )
        shooting_success_diff = paired_outcome_rate_interval(
            results, "student", "random_shooting", event="success",
            bootstrap_seed=int(config["gate"]["bootstrap"]["seed"]),
            resamples=int(config["gate"]["bootstrap"]["resamples"]),
        )
        shooting_collision_diff = paired_outcome_rate_interval(
            results, "student", "random_shooting", event="collision",
            bootstrap_seed=int(config["gate"]["bootstrap"]["seed"]),
            resamples=int(config["gate"]["bootstrap"]["resamples"]),
        )
        student_summary = summaries["student"]
        geometry_summary = summaries["geometry_waypoint"]
        shooting_summary = summaries["random_shooting"]
        phase13_gap = float(config["reference_phase13a"]["safe_success_gap"])
        required_gain = phase13_gap * float(config["gate"]["retain_fraction_of_phase13a_planner_minus_geometry_success_gap"])
        latency_ratio = (
            float(student_summary["median_wall_seconds_per_decision"])
            / float(shooting_summary["median_wall_seconds_per_decision"])
        )
        point_gain = float(student_summary["safe_success_rate"] - geometry_summary["safe_success_rate"])
        retained_fraction = point_gain / phase13_gap
        passed = (
            point_gain >= required_gain
            and success_gain.lower_95 > 0.0
            and collision_diff.upper_95 <= float(config["gate"]["paired_student_minus_geometry_collision_upper_95_maximum"])
            and latency_ratio <= float(config["gate"]["student_median_decision_latency_maximum_fraction_of_exact_shooting"])
        )
        metrics = {
            "schema": "phase14f_closed_loop_metrics_v1",
            "status": "passed" if passed else "failed",
            "gate_passed": bool(passed),
            "layout_count": len(expected_by_seed),
            "episodes_per_method": len(expected_by_seed),
            "student": student_summary,
            "geometry_waypoint": geometry_summary,
            "exact_random_shooting": shooting_summary,
            "student_minus_geometry_safe_success_point_gain": point_gain,
            "student_minus_geometry_safe_success_paired_95": asdict(success_gain),
            "student_minus_geometry_collision_paired_95": asdict(collision_diff),
            "student_minus_exact_shooting_safe_success_paired_95": asdict(shooting_success_diff),
            "student_minus_exact_shooting_collision_paired_95": asdict(shooting_collision_diff),
            "phase13a_reference_success_gap": phase13_gap,
            "required_student_success_gain": required_gain,
            "fraction_of_phase13a_success_gap_retained": retained_fraction,
            "student_to_shooting_median_decision_latency_ratio": latency_ratio,
            "thresholds": config["gate"],
            "model_sha256": lock["student_model"]["artifact_sha256"],
            "offline_gate_mean_regret_gain": parent["offline_metrics"]["point_improvement_over_geometry"],
            "offline_student_best_action_set_agreement": parent["offline_metrics"]["student_best_action_set_agreement"],
            "episode_record_count": len(results),
            "layout_sha256": lock["authorized_block"]["layout_sha256"],
        }
        _write_json(metrics_path, metrics)
        manifest.update({
            "status": "completed_closed_loop_development",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "episode_record_count": len(results),
            "metrics_sha256": sha256_file(metrics_path),
            "episodes_sha256": sha256_file(episodes_path),
            "gate_passed": bool(passed),
            "reserve_access": False,
        })
        _write_json(manifest_path, manifest)
        _write_closed_loop_report(report, metrics, manifest)
        return metrics
    except BaseException as exc:
        manifest.update({
            "status": "aborted_or_incomplete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        _write_json(manifest_path, manifest)
        raise


def _write_closed_loop_report(path: Path, metrics: dict[str, Any], manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase 14F conditional closed-loop development result", "",
        f"**Gate:** {metrics['status']}.", "",
        f"Layouts: {metrics['layout_count']}; episodes per method: {metrics['episodes_per_method']}; paired records: {metrics['episode_record_count']}.",
        f"Student model SHA-256: `{metrics['model_sha256']}`.", "",
        "| Controller | Safe success | Collision | Timeout | Median successful steps | Median decision latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("student", "Frozen CART student"), ("geometry_waypoint", "Geometry waypoint"), ("exact_random_shooting", "Exact shooting")):
        summary = metrics[key]
        latency_ms = summary["median_wall_seconds_per_decision"]
        latency = "n/a" if latency_ms is None else f"{1000*latency_ms:.4f} ms"
        steps = "n/a" if summary["median_steps_on_success"] is None else f"{summary['median_steps_on_success']:.1f}"
        lines.append(
            f"| {label} | {summary['safe_success_rate']:.3f} ({round(summary['safe_success_rate']*summary['episodes'])}/{summary['episodes']}) | "
            f"{summary['collision_rate']:.3f} | {summary['timeout_rate']:.3f} | {steps} | {latency} |"
        )
    success = metrics["student_minus_geometry_safe_success_paired_95"]
    collision = metrics["student_minus_geometry_collision_paired_95"]
    shooting_success = metrics["student_minus_exact_shooting_safe_success_paired_95"]
    shooting_collision = metrics["student_minus_exact_shooting_collision_paired_95"]
    lines.extend([
        "",
        f"Student minus geometry success: {metrics['student_minus_geometry_safe_success_point_gain']:+.3f}, paired 95% interval [{success['lower_95']:+.3f}, {success['upper_95']:+.3f}].",
        f"Student minus geometry collision: {collision['paired_rate_difference']:+.3f}, paired 95% interval [{collision['lower_95']:+.3f}, {collision['upper_95']:+.3f}].",
        f"Student minus exact-shooting success: {shooting_success['paired_rate_difference']:+.3f}, paired 95% interval [{shooting_success['lower_95']:+.3f}, {shooting_success['upper_95']:+.3f}].",
        f"Student minus exact-shooting collision: {shooting_collision['paired_rate_difference']:+.3f}, paired 95% interval [{shooting_collision['lower_95']:+.3f}, {shooting_collision['upper_95']:+.3f}].",
        f"Historical Phase13A success gap retained: {metrics['fraction_of_phase13a_success_gap_retained']:.3f}; required gain {metrics['required_student_success_gain']:.3f}.",
        f"Student/exact-shooting median decision latency ratio: {metrics['student_to_shooting_median_decision_latency_ratio']:.4f}; limit {metrics['thresholds']['student_median_decision_latency_maximum_fraction_of_exact_shooting']:.2f}.",
        f"All four gates: **{'PASS' if metrics['gate_passed'] else 'FAIL'}**.", "",
        f"Lock SHA-256: `{manifest.get('lock_sha256')}`; episode SHA-256: `{manifest.get('episodes_sha256')}`; metrics SHA-256: `{manifest.get('metrics_sha256')}`.",
        "No DAgger, learned-world-model, or reserve outcome was accessed. A passing development gate is not final reserve confirmation.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
