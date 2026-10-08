"""Phase14G trace-attribution preparation and hash-only layout audit.

This module deliberately separates pure CART diagnostics and hash-only layout
qualification from episode execution. The v1 config authorizes only the
layout-identity audit; no function here runs a closed-loop episode.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import yaml

from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.inertial_corridor import CorridorState
from consequence_lab.phase13 import _is_outcome_artifact, _read_layout_hashes
from consequence_lab.phase14 import SourceDecision, evaluate_forced_action, sha256_file
from consequence_lab.phase14c import build_feature_matrix
from consequence_lab.phase12_teacher import state_features


CONFIG_PATH = "configs/phase14g.yaml"
SOURCE_FILES = (
    CONFIG_PATH,
    "reports/phase14g_trace_attribution_protocol_draft.md",
    "reports/phase14f_tree_support_postmortem.md",
    "reports/phase14f_closed_loop_v1_result.md",
    "configs/phase14f_closed_loop.yaml",
    "experiments/phase14f_closed_loop_v1.lock.json",
    "experiments/phase14f_student_cart_v1.json",
    "experiments/phase14f_student_cart_v1.manifest.json",
    "experiments/phase14d_offline_gate_v2.lock.json",
    "artifacts/runs/phase14f_closed_loop_v1/manifest.json",
    "artifacts/runs/phase14f_closed_loop_v1/metrics.json",
    "artifacts/runs/phase14f_closed_loop_v1/episodes.jsonl",
    "artifacts/runs/phase14b_training_labels_resume_v1/first_action_values_complete.jsonl",
    "src/consequence_lab/phase14g.py",
    "src/consequence_lab/phase14f.py",
    "src/consequence_lab/phase14.py",
    "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase13.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "scripts/audit_phase14g_layouts.py",
    "tests/test_phase14g.py",
)


class Phase14GIntegrityError(RuntimeError):
    """Phase14G config, parent artifacts, or hash-audit integrity failure."""


@dataclass(frozen=True)
class CartDecisionTrace:
    action_id: int
    predicted_regret: tuple[float, ...]
    normalized_features: tuple[float, ...]
    node_path: tuple[int, ...]
    leaf_id: int


def _normalized_features(policy: dict[str, Any], raw_features: Any) -> np.ndarray:
    raw = np.asarray(raw_features, dtype=np.float64)
    if raw.shape != (52,) or not np.isfinite(raw).all():
        raise ValueError("student state must be 52 finite deployable features")
    selected = policy.get("selected_model", {})
    variant = selected.get("feature_variant")
    x = build_feature_matrix(raw[None, :], variant)[0]
    mean = np.asarray(policy.get("x_mean"), dtype=np.float64)
    std = np.asarray(policy.get("x_std"), dtype=np.float64)
    if (mean.shape != x.shape or std.shape != x.shape or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise ValueError("frozen CART normalization does not match the feature vector")
    return (x - mean) / std


def trace_frozen_cart(policy: dict[str, Any], raw_features: Any) -> CartDecisionTrace:
    """Return the exact frozen-CART action, values, path, and leaf for one state."""
    if (policy.get("schema") != "phase14f_frozen_student_cart_v1"
            or policy.get("state_feature_count") != 52 or policy.get("action_count") != 9):
        raise ValueError("unsupported frozen CART schema")
    x = _normalized_features(policy, raw_features)
    nodes = policy.get("nodes", [])
    if not nodes or len(nodes) != policy.get("node_count"):
        raise ValueError("CART node array is empty or has inconsistent count")
    index = 0
    path: list[int] = []
    while True:
        if index < 0 or index >= len(nodes) or index in path:
            raise ValueError("CART path contains a cycle or invalid node reference")
        path.append(index)
        node = nodes[index]
        feature = int(node["feature"])
        if feature < 0:
            values = np.asarray(node["value"], dtype=np.float64)
            if values.shape != (9,) or not np.isfinite(values).all():
                raise ValueError("CART leaf must contain nine finite action values")
            return CartDecisionTrace(
                action_id=int(np.argmin(values)),
                predicted_regret=tuple(float(value) for value in values),
                normalized_features=tuple(float(value) for value in x),
                node_path=tuple(path),
                leaf_id=index,
            )
        if feature >= len(x):
            raise ValueError("CART split feature is outside normalized input")
        child = node["left"] if x[feature] <= float(node["threshold"]) else node["right"]
        if child is None:
            raise ValueError("CART internal node is missing a child")
        index = int(child)


def build_leaf_support_index(policy: dict[str, Any], training_raw: np.ndarray) -> dict[int, tuple[int, ...]]:
    """Map each frozen CART leaf to the indices of fitting examples reaching it."""
    rows = np.asarray(training_raw, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 52 or not np.isfinite(rows).all():
        raise ValueError("training states must be a finite [N,52] array")
    supports: dict[int, list[int]] = {}
    for row_index, row in enumerate(rows):
        leaf = trace_frozen_cart(policy, row).leaf_id
        supports.setdefault(leaf, []).append(row_index)
    return {leaf: tuple(indices) for leaf, indices in supports.items()}


def nearest_training_state_distance(policy: dict[str, Any], raw_features: Any, training_raw: np.ndarray) -> float:
    """Euclidean distance in the model's normalized 61-D input space."""
    rows = np.asarray(training_raw, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 52 or len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError("training states must be a nonempty finite [N,52] array")
    query = _normalized_features(policy, raw_features)
    matrix = build_feature_matrix(rows, policy["selected_model"]["feature_variant"])
    mean = np.asarray(policy["x_mean"], dtype=np.float64)
    std = np.asarray(policy["x_std"], dtype=np.float64)
    normalized = (matrix - mean) / std
    distances = np.linalg.norm(normalized - query[None, :], axis=1)
    return float(np.min(distances))


def build_support_counts(policy: dict[str, Any], training_raw: np.ndarray) -> Counter[int]:
    """Compact occupancy counter for a previously declared train-only set."""
    rows = np.asarray(training_raw, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 52 or len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError("training states must be a nonempty finite [N,52] array")
    return Counter(trace_frozen_cart(policy, row).leaf_id for row in rows)


def forced_first_action_values(
    scenario: CorridorScenario,
    state: CorridorState,
    *,
    decision_index: int,
    expected_student_features: Any | None = None,
) -> list[dict[str, Any]]:
    """Produce exact Q_T(s,a) diagnostic labels for all nine primitive actions.

    Calling this function performs simulator branching and is intentionally not
    reachable through the Phase14G hash-audit CLI. A future separately
    authorized trace runner may call it only for student-visited states.
    """
    if state.status != "active":
        raise ValueError("counterfactual action values require an active state")
    if isinstance(decision_index, bool) or not isinstance(decision_index, Integral) or decision_index < 0:
        raise ValueError("decision_index must be a nonnegative integer")
    features = tuple(float(value) for value in state_features(scenario, state))
    if expected_student_features is not None:
        expected = np.asarray(expected_student_features, dtype=np.float64)
        if expected.shape != (52,) or not np.array_equal(expected, np.asarray(features)):
            raise ValueError("runtime student features disagree with exact observable state features")
    source = SourceDecision(
        episode_seed=int(scenario.seed),
        layout_sha256=scenario.layout_sha256,
        decision_index=int(decision_index),
        global_query_index=int(decision_index),
        state=state,
        scenario=scenario,
        features=features,
    )
    state_before = (state.x, state.y, state.vx, state.vy, state.elapsed_steps, state.status)
    rows = [evaluate_forced_action(source, action_id) for action_id in range(9)]
    state_after = (state.x, state.y, state.vx, state.vy, state.elapsed_steps, state.status)
    if state_before != state_after:
        raise Phase14GIntegrityError("counterfactual evaluation mutated its source state")
    return rows


def _collect_prior_layout_hashes(root: Path, exclude_path: Path) -> tuple[set[str], set[str], list[str], dict[str, str]]:
    run_root = root / "artifacts" / "runs"
    if not run_root.is_dir():
        raise Phase14GIntegrityError("artifacts/runs is unavailable for layout provenance audit")
    all_hashes: set[str] = set()
    spent_hashes: set[str] = set()
    sources: list[str] = []
    source_hashes: dict[str, str] = {}
    exclude_resolved = exclude_path.resolve()
    for path in sorted(run_root.rglob("*")):
        if (not path.is_file() or path.resolve() == exclude_resolved
                or path.name.startswith("phase13_layout_hash_audit_v")
                or path.suffix.lower() not in {".json", ".jsonl"}):
            continue
        hashes = _read_layout_hashes(path)
        if not hashes:
            continue
        relative = path.relative_to(root).as_posix()
        all_hashes.update(hashes)
        sources.append(relative)
        source_hashes[relative] = sha256_file(path)
        if path.name.lower() == "manifest.json":
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if _is_outcome_artifact(path, metadata):
                spent_hashes.update(hashes)
        elif _is_outcome_artifact(path, {}):
            spent_hashes.update(hashes)
    return all_hashes, spent_hashes, sources, source_hashes


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14GIntegrityError(f"git {' '.join(args)} failed") from exc


def build_layout_audit(root: Path) -> dict[str, Any]:
    """Hash-only audit: generate the configured layout identities and no outcomes."""
    root = Path(root).resolve()
    config_path = root / CONFIG_PATH
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (config.get("schema") != "phase14g_trace_attribution_config_v1"
            or config.get("status") != "hash_audit_only_run_not_authorized"
            or config.get("scope", {}).get("run_authorized") is not False
            or config.get("scope", {}).get("simulator_transitions") is not False
            or config.get("scope", {}).get("reserve_access") is not False):
        raise Phase14GIntegrityError("Phase14G config is not hash-audit-only")
    if _git(root, "status", "--porcelain"):
        raise Phase14GIntegrityError("commit Phase14G protocol/source/tests before hash audit")
    audit_path = root / config["outputs"]["layout_audit"]
    if audit_path.exists():
        raise Phase14GIntegrityError("refusing to overwrite Phase14G layout audit")
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise Phase14GIntegrityError(f"Phase14G source files missing: {missing}")
    phase14f_manifest_path = root / config["parent"]["phase14f_manifest"]
    phase14f_metrics_path = root / config["parent"]["phase14f_metrics"]
    phase14f_lock_path = root / config["parent"]["phase14f_lock"]
    phase14f_episodes_path = root / config["parent"]["phase14f_episodes"]
    phase14f_student_path = root / config["parent"]["phase14f_student"]
    phase14f_student_manifest_path = root / config["parent"]["phase14f_student_manifest"]
    phase14f_manifest = json.loads(phase14f_manifest_path.read_text(encoding="utf-8"))
    phase14f_metrics = json.loads(phase14f_metrics_path.read_text(encoding="utf-8"))
    phase14f_student_manifest = json.loads(phase14f_student_manifest_path.read_text(encoding="utf-8"))
    phase14f_lock = json.loads(phase14f_lock_path.read_text(encoding="utf-8"))
    phase14f_model_sha256 = sha256_file(phase14f_student_path)
    if (phase14f_manifest.get("status") != "completed_closed_loop_development"
            or phase14f_manifest.get("gate_passed") is not False
            or phase14f_manifest.get("metrics_sha256") != sha256_file(phase14f_metrics_path)
            or phase14f_manifest.get("episodes_sha256") != sha256_file(phase14f_episodes_path)
            or phase14f_manifest.get("lock_sha256") != sha256_file(phase14f_lock_path)
            or phase14f_manifest.get("episode_record_count") != 300
            or phase14f_manifest.get("reserve_access") is not False
            or phase14f_manifest.get("scope") != {
                "closed_loop_development": True, "dagger": False,
                "learned_world_model": False, "reserve_access": False,
            }
            or phase14f_lock.get("reserve_access_authorized") is not False
            or phase14f_lock.get("dagger_authorized") is not False
            or phase14f_lock.get("learned_world_model_authorized") is not False
            or phase14f_student_manifest.get("model_sha256") != phase14f_model_sha256
            or phase14f_model_sha256 != config["student"]["model_artifact_sha256"]
            or phase14f_metrics.get("gate_passed") is not False
            or phase14f_metrics.get("student", {}).get("safe_success_rate") != 0.01):
        raise Phase14GIntegrityError("Phase14F parent result is incomplete or inconsistent")

    block = config["block"]
    start, count = int(block["seed_start"]), int(block["layout_count"])
    reserve = config["block"]["seed_range_must_be_outside_reserve"]
    reserve_start, reserve_stop = int(reserve["seed_start"]), int(reserve["seed_start"]) + int(reserve["count"])
    if start < reserve_stop and reserve_start < start + count:
        raise Phase14GIntegrityError("candidate layout block overlaps sealed reserve seed range")
    records = [
        {"seed": seed, "layout_sha256": generate_staggered_corridor(seed).layout_sha256}
        for seed in range(start, start + count)
    ]
    block_hashes = {row["layout_sha256"] for row in records}
    if len(records) != count or len(block_hashes) != count:
        raise Phase14GIntegrityError("candidate layout block is incomplete or contains duplicate layouts")
    prior_all, prior_spent, historical_sources, historical_hashes = _collect_prior_layout_hashes(root, audit_path)
    if block_hashes & prior_all:
        raise Phase14GIntegrityError("candidate layouts overlap an already audited historical layout")
    if block_hashes & prior_spent:
        raise Phase14GIntegrityError("candidate layouts overlap a prior outcome-bearing layout")
    source_hashes = {name: sha256_file(root / name) for name in SOURCE_FILES}
    phase14d_audit_path = root / "artifacts/runs/phase14d_layout_hash_audit_v2.json"
    phase14d_audit = json.loads(phase14d_audit_path.read_text(encoding="utf-8"))
    if (phase14d_audit.get("status") != "hash_only_passed_no_outcomes"
            or phase14d_audit.get("task_transitions_evaluated") is not False
            or phase14d_audit.get("reserve_performance_outcomes_read") is not False
            or phase14d_audit.get("reserve_generated_or_accessed") is not False):
        raise Phase14GIntegrityError("Phase14D historical hash audit scope is invalid")
    return {
        "schema": "phase14g_layout_hash_audit_v1",
        "status": "hash_only_passed_no_outcomes_run_not_authorized",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_git_commit": _git(root, "rev-parse", "HEAD"),
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_hashes,
        "phase14f_parent": {
            "lock_sha256": sha256_file(phase14f_lock_path),
            "manifest_sha256": sha256_file(phase14f_manifest_path),
            "metrics_sha256": sha256_file(phase14f_metrics_path),
            "episodes_sha256": sha256_file(phase14f_episodes_path),
            "student_sha256": phase14f_model_sha256,
            "student_manifest_sha256": sha256_file(phase14f_student_manifest_path),
            "report_sha256": sha256_file(root / config["parent"]["phase14f_report"]),
            "failed_gate_confirmed": True,
        },
        "historical_hash_source_files": historical_sources,
        "historical_hash_source_sha256": historical_hashes,
        "prior_layout_hash_count": len(prior_all),
        "prior_outcome_layout_hash_count": len(prior_spent),
        "candidate_block": {"seed_start": start, "count": count, "layouts": records},
        "unique_candidate_layout_hashes": len(block_hashes),
        "candidate_block_disjoint_from_all_historical_layouts": True,
        "candidate_block_disjoint_from_prior_outcome_layouts": True,
        "candidate_block_outside_sealed_reserve_seed_range": True,
        "task_transitions_evaluated": False,
        "episode_outcomes_read": False,
        "student_traces_collected": False,
        "teacher_counterfactual_branches_evaluated": False,
        "model_fit_or_tuning": False,
        "reserve_access": False,
        "run_authorized": False,
    }
