"""Phase 14D split integrity, exact teacher labels, and train-only model selection."""

from __future__ import annotations

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

from consequence_lab.corridor_baselines import grid_astar_route, waypoint_braking_action
from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.inertial_corridor import CorridorState, corridor_step
from consequence_lab.phase13 import _collect_historical_layouts
from consequence_lab.phase12_teacher import make_teacher_query, state_features
from consequence_lab.phase14 import (
    OUTCOME_ORDER,
    SourceDecision,
    sha256_file,
)
from consequence_lab.phase14b import make_compact_state_record
from consequence_lab.phase14c import (
    _OneNearestNeighbor,
    _SquaredErrorTree,
    _TinyMLP,
    _q_target_scale,
    build_feature_matrix,
    evaluate_predictions,
    extract_arrays,
    model_parameter_count,
)


CONFIG_PATH = "configs/phase14d.yaml"
SOURCE_FILES = (
    CONFIG_PATH,
    "reports/phase14d_decision_function_generalization_protocol.md",
    "src/consequence_lab/phase14d.py",
    "src/consequence_lab/phase13.py",
    "scripts/audit_phase14d_layouts.py",
    "scripts/freeze_phase14d_lock.py",
    "scripts/run_phase14d_model_selection.py",
    "tests/test_phase14d.py",
    "src/consequence_lab/phase14.py",
    "src/consequence_lab/phase14b.py",
    "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
)


class Phase14DIntegrityError(RuntimeError):
    """Hash split, protocol, or source-lock integrity failure."""


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14DIntegrityError(f"git {' '.join(args)} failed") from exc


def _source_hashes(root: Path) -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise Phase14DIntegrityError(f"locked source files missing: {missing}")
    return {name: sha256_file(root / name) for name in SOURCE_FILES}


def build_layout_audit(root: Path, *, require_clean_git: bool = True) -> dict[str, Any]:
    """Audit only layout identities and prior outcome hashes; do not step tasks."""
    root = Path(root).resolve()
    if require_clean_git and _git(root, "status", "--porcelain"):
        raise Phase14DIntegrityError("commit Phase14D protocol and audit source before hash audit")
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    prior_path = root / config["source"]["phase13_layout_audit"]
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (prior.get("schema") != "phase13_layout_hash_audit_v1"
            or prior.get("status") != "hash_only_passed_no_outcomes"
            or prior.get("task_transitions_evaluated") is not False):
        raise Phase14DIntegrityError("Phase13 layout audit is not a valid no-outcome hash audit")

    prior_all, prior_spent, history_sources = _collect_historical_layouts(root)
    split_specs = {
        "model_selection": config["splits"]["model_selection"],
        "offline_gate": config["splits"]["offline_gate"],
        "closed_loop_development": config["splits"]["closed_loop_development"],
    }
    prior_names = {
        "model_selection": "phase13E_model_selection",
        "offline_gate": "phase13E_offline_gate",
        "closed_loop_development": "phase13F_closed_loop_development",
    }
    blocks: dict[str, list[dict[str, Any]]] = {}
    block_sets: dict[str, set[str]] = {}
    for name, spec in split_specs.items():
        start, count = int(spec["seed_start"]), int(spec["count"])
        records = []
        hashes: set[str] = set()
        for seed in range(start, start + count):
            scenario = generate_staggered_corridor(seed)
            record = {"seed": seed, "layout_sha256": scenario.layout_sha256}
            records.append(record)
            hashes.add(scenario.layout_sha256)
        if len(records) != count or len(hashes) != count:
            raise Phase14DIntegrityError(f"{name} has missing seeds or duplicate layouts")
        old_records = prior.get("blocks", {}).get(prior_names[name], [])
        if records != old_records:
            raise Phase14DIntegrityError(f"{name} differs from its Phase13 hash-only audit")
        if hashes & prior_spent:
            raise Phase14DIntegrityError(f"{name} overlaps a previously outcome-bearing layout")
        blocks[name] = records
        block_sets[name] = hashes

    names = list(block_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if block_sets[left] & block_sets[right]:
                raise Phase14DIntegrityError(f"Phase14D split overlap: {left} / {right}")

    return {
        "schema": "phase14d_layout_hash_audit_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_git_commit": _git(root, "rev-parse", "HEAD"),
        "generation_source_sha256": _source_hashes(root),
        "phase13_hash_audit_sha256": sha256_file(prior_path),
        "status": "hash_only_passed_no_outcomes",
        "task_transitions_evaluated": False,
        "episode_outcomes_read": False,
        "reserve_performance_outcomes_read": False,
        "reserve_generated_or_accessed": False,
        "previous_outcome_layout_hash_count": len(prior_spent),
        "historical_hash_source_files": history_sources,
        "blocks": blocks,
        "all_phase14d_blocks_pairwise_disjoint": True,
        "all_blocks_disjoint_from_prior_outcomes": True,
        "model_selection_and_offline_gate_are_previously_hash_audited_only": True,
    }


def create_selection_lock(root: Path) -> dict[str, Any]:
    """Freeze only model-selection label collection after source/hash audit."""
    root = Path(root).resolve()
    if _git(root, "status", "--porcelain"):
        raise Phase14DIntegrityError("commit hash audit before freezing Phase14D lock")
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    audit_path = root / config["outputs"]["hash_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "hash_only_passed_no_outcomes" or audit.get("task_transitions_evaluated") is not False:
        raise Phase14DIntegrityError("Phase14D hash audit did not pass")
    if audit.get("generation_source_sha256") != _source_hashes(root):
        raise Phase14DIntegrityError("Phase14D source changed since hash audit")
    if audit.get("generation_git_commit") != _git(root, "rev-parse", "HEAD"):
        raise Phase14DIntegrityError("Phase14D source commit changed since hash audit")
    c_lock_path = root / config["source"]["phase14c_lock"]
    c_lock = json.loads(c_lock_path.read_text(encoding="utf-8"))
    c_manifest_path = root / config["source"]["phase14c_manifest"]
    c_metrics_path = root / config["source"]["phase14c_metrics"]
    c_manifest = json.loads(c_manifest_path.read_text(encoding="utf-8"))
    c_metrics = json.loads(c_metrics_path.read_text(encoding="utf-8"))
    if (c_lock.get("schema") != "phase14c_memorization_lock_v2"
            or c_lock.get("status") != "authorized_phase14c_train_split_memorization_only"
            or c_lock.get("reserve_access") is not False
            or c_lock.get("heldout_access") is not False
            or c_manifest.get("status") != "passed"):
        raise Phase14DIntegrityError("Phase14C memorization gate is not verified as passed")
    if c_metrics.get("memorization_ladder", {}).get("status") != "passed":
        raise Phase14DIntegrityError("Phase14C metrics do not show the full memorization ladder pass")
    c_scope = c_manifest.get("scope", {})
    if (c_scope.get("heldout_access") is not False
            or c_scope.get("reserve_access") is not False
            or c_scope.get("training_split_only") is not True):
        raise Phase14DIntegrityError("Phase14C manifest has invalid held-out/reserve access flags")

    training_path = root / config["source"]["training_records"]
    training_manifest_path = root / config["source"]["training_manifest"]
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    phase14b_lock_path = root / config["source"]["phase14b_lock"]
    training_hash = sha256_file(training_path)
    if (training_hash != c_lock.get("phase14b_records_sha256")
            or training_manifest.get("records_sha256") != training_hash
            or training_manifest.get("status") != "completed_teacher_values_gate_passed"
            or training_manifest.get("heldout_access") is not False
            or training_manifest.get("reserve_accessed") is not False
            or training_manifest.get("student_training") is not False):
        raise Phase14DIntegrityError("Phase14B training-only teacher dataset provenance mismatch")
    outputs = {
        "manifest": config["outputs"]["model_selection_dir"] + "/manifest.json",
        "labels": config["outputs"]["model_selection_dir"] + "/teacher_labels.jsonl",
        "metrics": config["outputs"]["model_selection_dir"] + "/metrics.json",
        "report": config["outputs"]["model_selection_report"],
    }
    return {
        "schema": "phase14d_model_selection_lock_v2",
        "status": "authorized_model_selection_only",
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "authorization_reference": "User authorized continued gated execution after positive Phase14C memorization; scope is model-selection only.",
        "phase14d_config_sha256": sha256_file(root / CONFIG_PATH),
        "phase14d_protocol_sha256": sha256_file(root / config["source"]["protocol"]),
        "phase14d_source_sha256": _source_hashes(root),
        "phase14d_hash_audit_sha256": sha256_file(audit_path),
        "phase14c_lock_sha256": sha256_file(c_lock_path),
        "phase14c_manifest_sha256": sha256_file(c_manifest_path),
        "phase14c_metrics_sha256": sha256_file(c_metrics_path),
        "phase14c_training_records_sha256": c_lock["phase14b_records_sha256"],
        "phase14b_training_records_sha256": training_hash,
        "phase14b_training_manifest_sha256": sha256_file(training_manifest_path),
        "phase14b_lock_sha256": sha256_file(phase14b_lock_path),
        "authorized_block": config["splits"]["model_selection"],
        "authorized_layout_sha256": [
            row["layout_sha256"] for row in audit["blocks"]["model_selection"]
        ],
        "offline_gate_labels_authorized": False,
        "closed_loop_authorized": False,
        "dagger_authorized": False,
        "learned_world_model_authorized": False,
        "reserve_access_authorized": False,
        "resource_caps": {"wall_time_seconds": None, "storage_bytes": None},
        "outputs": outputs,
        "overwrite_forbidden": True,
    }


def validate_selection_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify committed lock/source/splits before any selection transition."""
    root = Path(root).resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    lock_path = root / config["outputs"]["selection_lock"]
    audit_path = root / config["outputs"]["hash_audit"]
    if not lock_path.is_file() or not audit_path.is_file():
        raise Phase14DIntegrityError("Phase14D selection lock or hash audit is missing")
    if _git(root, "status", "--porcelain"):
        raise Phase14DIntegrityError("working tree must be clean before Phase14D data collection")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (lock.get("schema") != "phase14d_model_selection_lock_v2"
            or lock.get("status") != "authorized_model_selection_only"):
        raise Phase14DIntegrityError("Phase14D lock is not the corrected v2 model-selection lock")
    if lock.get("source_commit") != audit.get("generation_git_commit"):
        try:
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", str(lock.get("source_commit")), "HEAD"],
                cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Phase14DIntegrityError("locked source commit is not an ancestor of current HEAD") from exc
    if lock.get("phase14d_source_sha256") != _source_hashes(root):
        raise Phase14DIntegrityError("Phase14D sources changed after selection lock")
    if (lock.get("phase14d_config_sha256") != sha256_file(root / CONFIG_PATH)
            or lock.get("phase14d_protocol_sha256") != sha256_file(root / config["source"]["protocol"])
            or lock.get("phase14d_hash_audit_sha256") != sha256_file(audit_path)):
        raise Phase14DIntegrityError("Phase14D config/protocol/hash audit differs from lock")
    if (audit.get("status") != "hash_only_passed_no_outcomes"
            or audit.get("task_transitions_evaluated") is not False
            or audit.get("reserve_generated_or_accessed") is not False
            or lock.get("offline_gate_labels_authorized") is not False
            or lock.get("closed_loop_authorized") is not False
            or lock.get("dagger_authorized") is not False
            or lock.get("learned_world_model_authorized") is not False
            or lock.get("reserve_access_authorized") is not False):
        raise Phase14DIntegrityError("Phase14D lock/audit has an invalid scope or reserve flag")
    if lock.get("authorized_block") != config["splits"]["model_selection"]:
        raise Phase14DIntegrityError("authorized seed block differs from Phase14D config")
    if lock.get("resource_caps") != {"wall_time_seconds": None, "storage_bytes": None}:
        raise Phase14DIntegrityError("selection lock has unexpected time/storage caps")
    if lock.get("authorized_layout_sha256") != [
        row["layout_sha256"] for row in audit["blocks"]["model_selection"]
    ]:
        raise Phase14DIntegrityError("authorized selection layout list differs from hash audit")
    if lock.get("phase14b_training_records_sha256") != sha256_file(root / config["source"]["training_records"]):
        raise Phase14DIntegrityError("Phase14B training records changed after Phase14D lock")
    for source_name, lock_key in (
        ("phase14c_lock", "phase14c_lock_sha256"),
        ("phase14c_manifest", "phase14c_manifest_sha256"),
        ("phase14c_metrics", "phase14c_metrics_sha256"),
        ("phase14b_lock", "phase14b_lock_sha256"),
        ("training_manifest", "phase14b_training_manifest_sha256"),
    ):
        if sha256_file(root / config["source"][source_name]) != lock.get(lock_key):
            raise Phase14DIntegrityError(f"locked source artifact changed: {source_name}")
    output_dir = root / config["outputs"]["model_selection_dir"]
    report_path = root / config["outputs"]["model_selection_report"]
    if output_dir.exists() or report_path.exists():
        raise Phase14DIntegrityError("refusing to overwrite existing Phase14D selection outputs")
    return lock, config


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_model_selection(root: Path) -> dict[str, Any]:
    """Generate only model-selection labels, fit frozen candidates, and report."""
    root = Path(root).resolve()
    lock, config = validate_selection_lock(root)
    output_dir = root / config["outputs"]["model_selection_dir"]
    report_path = root / config["outputs"]["model_selection_report"]
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "manifest.json"
    labels_path = output_dir / "teacher_labels.jsonl"
    metrics_path = output_dir / "metrics.json"
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    manifest = {
        "schema": "phase14d_model_selection_result_v2",
        "status": "collecting_exact_counterfactual_labels",
        "source_commit": lock["source_commit"],
        "started_at_utc": started_at.isoformat(),
        "lock_sha256": sha256_file(root / config["outputs"]["selection_lock"]),
        "hash_audit_sha256": lock["phase14d_hash_audit_sha256"],
        "authorized_seed_start": int(config["splits"]["model_selection"]["seed_start"]),
        "authorized_seed_count": int(config["splits"]["model_selection"]["count"]),
        "scope": {
            "model_selection_only": True,
            "offline_gate_labels": False,
            "closed_loop": False,
            "dagger": False,
            "learned_world_model": False,
            "reserve_access": False,
        },
        "runtime_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    _write_json(manifest_path, manifest)
    try:
        rows = collect_model_selection_rows(config)
        expected_hashes = set(lock["authorized_layout_sha256"])
        actual_hashes = {row["source_layout_sha256"] for row in rows}
        if actual_hashes != expected_hashes:
            raise Phase14DIntegrityError("generated selection layouts differ from frozen lock")
        if not rows or len(rows) > 2 * int(config["splits"]["model_selection"]["count"]):
            raise Phase14DIntegrityError("selection state count violates the frozen block contract")
        labels_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        labels_path.write_text(labels_text, encoding="utf-8")
        labels_sha256 = sha256_file(labels_path)

        training_rows = [
            json.loads(line)
            for line in (root / config["source"]["training_records"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_raw, train_regret, train_objective = extract_arrays(training_rows)
        if len(training_rows) != 2095 or labels_sha256 != sha256_file(labels_path):
            raise Phase14DIntegrityError("training source or selection labels failed structural checks")

        results: list[dict[str, Any]] = []
        for family in config["student"]["learners"]:
            for feature_variant in config["student"]["feature_variants"]:
                for target_variant in config["student"]["target_variants"]:
                    print(
                        f"Phase14D fitting family={family} features={feature_variant} "
                        f"target={target_variant}", flush=True,
                    )
                    results.append(_fit_candidate(
                        family, feature_variant, target_variant,
                        train_raw, train_regret, train_objective, rows,
                        int(config["student"]["tiny_mlp"]["seed"]),
                    ))
        selected = select_model_candidate(results)
        metrics = {
            "schema": "phase14d_model_selection_metrics_v2",
            "candidate_results": results,
            "selected_candidate": selected,
            "selection_rule": config["selection"],
            "selection_state_count": len(rows),
            "selection_layout_count": len(actual_hashes),
            "mean_geometry_regret": selected["geometry_layout_macro_mean_normalized_regret"],
            "selected_student_mean_regret": selected["layout_macro_mean_normalized_regret"],
            "selected_improvement_over_geometry": selected["improvement_over_geometry"],
            "training_record_count": len(training_rows),
            "training_record_sha256": lock["phase14b_training_records_sha256"],
            "selection_labels_sha256": labels_sha256,
        }
        _write_json(metrics_path, metrics)
        manifest.update({
            "status": "completed_model_selection",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started_clock,
            "state_count": len(rows),
            "layout_count": len(actual_hashes),
            "teacher_action_value_count": 9 * len(rows),
            "teacher_labels_sha256": labels_sha256,
            "metrics_sha256": sha256_file(metrics_path),
            "offline_gate_labels_collected": False,
            "reserve_access": False,
        })
        _write_json(manifest_path, manifest)
        _write_selection_report(report_path, lock, metrics, manifest)
        return metrics
    except BaseException as exc:
        manifest.update({
            "status": "aborted_or_incomplete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started_clock,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        _write_json(manifest_path, manifest)
        raise


def _write_selection_report(path: Path, lock: dict[str, Any], metrics: dict[str, Any], manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase 14D model-selection result", "",
        f"**Status:** {manifest['status']} (development/model-selection only).", "",
        f"Source commit: `{lock['source_commit']}`  ",
        f"Lock SHA-256: `{manifest['lock_sha256']}`  ",
        f"Hash-audit SHA-256: `{manifest['hash_audit_sha256']}`  ",
        f"Label SHA-256: `{manifest.get('teacher_labels_sha256', 'not-completed')}`", "",
        f"Layouts/states/action values: {metrics['selection_layout_count']} / {metrics['selection_state_count']} / {9 * metrics['selection_state_count']}", "",
        "| Learner | Features | Target | Layout-macro regret | Geometry regret | Improvement | Median latency (ms) | Bytes |", "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["candidate_results"]:
        lines.append(
            f"| {item['learner']} | {item['feature_variant']} | {item['target_variant']} | "
            f"{item['layout_macro_mean_normalized_regret']:.6f} | "
            f"{item['geometry_layout_macro_mean_normalized_regret']:.6f} | "
            f"{item['improvement_over_geometry']:.6f} | "
            f"{1000 * item['median_inference_seconds_per_state']:.4f} | {item['serialized_model_bytes']} |"
        )
    selected = metrics["selected_candidate"]
    lines.extend([
        "", f"Selected by the frozen empirical-regret rule: `{selected['learner']} / {selected['feature_variant']} / {selected['target_variant']}`.",
        f"Selection improvement over geometry: {selected['improvement_over_geometry']:.6f} normalized regret.", "",
        "Selection data select the candidate only. They are not the offline gate; all model fitting used the Phase14B training split only. No offline-gate, closed-loop, DAgger, learned-world-model, or reserve outcomes were accessed.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _active_teacher_trajectory(scenario: CorridorScenario) -> tuple[list[CorridorState], str]:
    """Replay exact Phase12 shooting and return one public state per decision."""
    current = scenario.initial_state
    states: list[CorridorState] = []
    while current.status == "active":
        states.append(current)
        query = make_teacher_query(scenario, current)
        action = int(query.sequences[query.selected_index, 0])
        current = corridor_step(current, action, scenario.spec).state
        if len(states) > scenario.spec.max_steps:
            raise Phase14DIntegrityError("teacher trajectory exceeded the task horizon")
    if current.status not in OUTCOME_ORDER:
        raise Phase14DIntegrityError(f"teacher trajectory ended in unexpected state {current.status}")
    return states, current.status


def _selected_trajectory_indices(count: int, fractions: list[float]) -> list[tuple[float, int]]:
    if count <= 0:
        raise Phase14DIntegrityError("teacher trajectory contains no active decisions")
    values = []
    for fraction in fractions:
        index = int(np.floor(float(fraction) * (count - 1) + 0.5))
        values.append((float(fraction), min(max(index, 0), count - 1)))
    return values


def collect_model_selection_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect two teacher-trajectory states/layout and nine forced-action labels."""
    block = config["splits"]["model_selection"]
    start, count = int(block["seed_start"]), int(block["count"])
    fractions = [float(value) for value in config["splits"]["trajectory_fractions"]]
    rows: list[dict[str, Any]] = []
    seen_layouts: set[str] = set()
    for layout_offset, seed in enumerate(range(start, start + count)):
        scenario = generate_staggered_corridor(seed)
        if scenario.layout_sha256 in seen_layouts:
            raise Phase14DIntegrityError("model-selection block contains a repeated layout")
        seen_layouts.add(scenario.layout_sha256)
        trajectory, trajectory_terminal_status = _active_teacher_trajectory(scenario)
        route = grid_astar_route(
            (scenario.initial_state.x, scenario.initial_state.y),
            scenario.spec.goal,
            scenario.spec,
            resolution=float(config["geometry_comparator"]["resolution"]),
        )
        if route is None:
            raise Phase14DIntegrityError(f"geometry waypoint has no route for layout {seed}")
        selected = _selected_trajectory_indices(len(trajectory), fractions)
        selected_by_index: dict[int, list[float]] = {}
        for fraction, decision_index in selected:
            selected_by_index.setdefault(decision_index, []).append(fraction)
        for decision_index, requested_fractions in selected_by_index.items():
            state = trajectory[decision_index]
            source = SourceDecision(
                episode_seed=seed,
                layout_sha256=scenario.layout_sha256,
                decision_index=decision_index,
                global_query_index=len(rows),
                state=state,
                scenario=scenario,
                features=tuple(float(v) for v in state_features(scenario, state)),
            )
            record, _ = make_compact_state_record(source)
            if ([int(action["forced_action_id"]) for action in record["actions"]] != list(range(9))
                    or len(record["actions"]) != 9):
                raise Phase14DIntegrityError("forced-action target order differs from Phase14B contract")
            exact_query = make_teacher_query(scenario, state)
            geometry_action = int(waypoint_braking_action(
                state,
                route,
                scenario.spec,
                waypoint_tolerance=float(config["geometry_comparator"]["waypoint_tolerance"]),
            ))
            rows.append({
                "schema": "phase14d_first_action_value_state_v1",
                "split": "model_selection",
                "layout_offset": layout_offset,
                "trajectory_fractions_requested": requested_fractions,
                "trajectory_decision_index": decision_index,
                "trajectory_decision_count": len(trajectory),
                "source_episode_seed": seed,
                "source_layout_sha256": scenario.layout_sha256,
                "student_input_52": list(source.features),
                "actions": record["actions"],
                "geometry_waypoint_action_id": geometry_action,
                "exact_shooting_action_id": int(exact_query.sequences[exact_query.selected_index, 0]),
                "teacher_trajectory_terminal_status": trajectory_terminal_status,
                "teacher_trajectory_total_decisions": len(trajectory),
            })
    return rows


def _fit_candidate(
    family: str,
    feature_variant: str,
    target_variant: str,
    train_raw: np.ndarray,
    train_regret: np.ndarray,
    train_objective: np.ndarray,
    eval_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    eval_raw, eval_regret, eval_objective = extract_arrays(eval_rows)
    train_x = build_feature_matrix(train_raw, feature_variant)
    eval_x = build_feature_matrix(eval_raw, feature_variant)
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std[x_std < 1e-12] = 1.0
    train_x = (train_x - x_mean) / x_std
    eval_x = (eval_x - x_mean) / x_std
    if target_variant == "within_state_normalized_regret9":
        train_y = train_regret
        q_mean = q_std = None
    elif target_variant == "standardized_structured_objective27":
        train_y, q_mean, q_std = _q_target_scale(train_objective)
    else:
        raise Phase14DIntegrityError(f"unknown target variant: {target_variant}")
    if family == "nearest_neighbor":
        estimator = _OneNearestNeighbor()
    elif family == "decision_tree":
        estimator = _SquaredErrorTree()
    elif family == "tiny_mlp":
        estimator = _TinyMLP(seed=seed)
    else:
        raise Phase14DIntegrityError(f"unknown learner: {family}")
    import time
    started = time.perf_counter()
    estimator.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predicted = np.asarray(estimator.predict(eval_x), dtype=np.float64)
    predict_seconds = time.perf_counter() - started
    per_state_latencies = []
    for raw_query in eval_raw:
        query_started = time.perf_counter()
        deploy_x = build_feature_matrix(raw_query[None, :], feature_variant)
        deploy_x = (deploy_x - x_mean) / x_std
        estimator.predict(deploy_x)
        per_state_latencies.append(time.perf_counter() - query_started)
    metrics = evaluate_predictions(
        family, predicted, eval_regret, eval_objective, q_mean, q_std
    )
    choices = np.asarray(metrics["selected_action_ids"], dtype=np.int64)
    geometry_actions = np.asarray(
        [int(row["geometry_waypoint_action_id"]) for row in eval_rows], dtype=np.int64
    )
    geometry_regret = eval_regret[np.arange(len(eval_rows)), geometry_actions]
    exact_actions = np.asarray(
        [int(row["exact_shooting_action_id"]) for row in eval_rows], dtype=np.int64
    )
    exact_regret = eval_regret[np.arange(len(eval_rows)), exact_actions]
    # Every layout has two requested fractions except a recorded short-episode
    # duplicate; aggregate state regret by layout first to keep the layout unit.
    per_layout: dict[str, list[int]] = {}
    for index, row in enumerate(eval_rows):
        per_layout.setdefault(row["source_layout_sha256"], []).append(index)
    layout_student = [float(np.mean(metrics["selected_action_regret"][ids]))
                      if "selected_action_regret" in metrics else float(np.mean(eval_regret[ids, choices[ids]]))
                      for ids in per_layout.values()]
    layout_geometry = [float(np.mean(geometry_regret[ids])) for ids in per_layout.values()]
    metrics.update({
        "learner": family,
        "feature_variant": feature_variant,
        "target_variant": target_variant,
        "selection_layout_count": len(per_layout),
        "selection_state_count": len(eval_rows),
        "layout_macro_mean_normalized_regret": float(np.mean(layout_student)),
        "geometry_layout_macro_mean_normalized_regret": float(np.mean(layout_geometry)),
        "improvement_over_geometry": float(np.mean(layout_geometry) - np.mean(layout_student)),
        "mean_normalized_regret_exact_shooting_action": float(exact_regret.mean()),
        "geometry_selected_action_frequency": np.bincount(geometry_actions, minlength=9).tolist(),
        "exact_shooting_action_frequency": np.bincount(exact_actions, minlength=9).tolist(),
        "fit_seconds": fit_seconds,
        "prediction_batch_seconds": predict_seconds,
        "median_inference_seconds_per_state": float(np.median(per_state_latencies)),
        "p95_inference_seconds_per_state": float(np.quantile(per_state_latencies, 0.95)),
        "parameter_count_or_node_count": model_parameter_count(
            family, estimator, train_x.shape[1], train_y.shape[1]
        ),
        "serialized_model_bytes": _serialized_estimator_bytes(
            family, estimator, x_mean, x_std, q_mean, q_std
        ),
    })
    return metrics


def _serialized_estimator_bytes(
    family: str,
    estimator: Any,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    q_mean: np.ndarray | None,
    q_std: np.ndarray | None,
) -> int:
    """Size of a canonical JSON deployment payload, including transforms."""
    payload: dict[str, Any] = {
        "family": family,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "q_mean": None if q_mean is None else q_mean.tolist(),
        "q_std": None if q_std is None else q_std.tolist(),
    }
    if family == "nearest_neighbor":
        payload["x"] = estimator.x.tolist()
        payload["y"] = estimator.y.tolist()
    elif family == "decision_tree":
        nodes: list[dict[str, Any]] = []
        indices = {id(estimator.root): 0}
        pending = [estimator.root]
        while pending:
            node = pending.pop(0)
            for child in (node.left, node.right):
                if child is not None and id(child) not in indices:
                    indices[id(child)] = len(indices)
                    pending.append(child)
            nodes.append({
                "value": node.value.tolist(),
                "feature": int(node.feature),
                "threshold": float(node.threshold),
                "left": None if node.left is None else indices[id(node.left)],
                "right": None if node.right is None else indices[id(node.right)],
            })
        payload["nodes"] = nodes
    elif family == "tiny_mlp":
        payload["weights"] = [weight.tolist() for weight in estimator.weights]
        payload["biases"] = [bias.tolist() for bias in estimator.biases]
    else:
        raise Phase14DIntegrityError(f"cannot serialize learner {family}")
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def select_model_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered empirical-regret / latency / size tie-break."""
    if not results:
        raise Phase14DIntegrityError("no candidate results to select")
    return min(results, key=lambda item: (
        float(item["layout_macro_mean_normalized_regret"]),
        float(item["median_inference_seconds_per_state"]),
        int(item["serialized_model_bytes"]),
        str(item["learner"]), str(item["feature_variant"]), str(item["target_variant"]),
    ))
