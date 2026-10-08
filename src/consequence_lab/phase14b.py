"""Fail-closed Phase14B full-train-label source lock and target generation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import yaml

from consequence_lab.phase14 import (
    CONFIG_PATH as PHASE14_CONFIG,
    PROTOCOL_PATH,
    SOURCE_CODE as PHASE14_SOURCE_CODE,
    SOURCE_DIR,
    SOURCE_FILES,
    Phase14IntegrityError,
    add_state_ranks,
    evaluate_forced_action,
    phase14_python_environment,
    replay_training_states,
    sha256_file,
    summarize_audit,
)


CONFIG_PATH = "configs/phase14b.yaml"
LOCK_PATH = "experiments/phase14b_training_labels_v1.lock.json"
PHASE14B_SOURCE_CODE = (
    PHASE14_CONFIG,
    CONFIG_PATH,
    PROTOCOL_PATH,
    "reports/phase14a_counterfactual_teacher_audit_result.md",
    "reports/phase14b_full_training_labels_protocol.md",
    "src/consequence_lab/phase14.py",
    "src/consequence_lab/phase14b.py",
    "scripts/freeze_phase14b_lock.py",
    "scripts/run_phase14b_labels.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "configs/phase12_teacher_student.yaml",
    "tests/test_phase14b.py",
)


class Phase14BLockError(RuntimeError):
    """Phase14B data-stage authorization/provenance failure."""


def _git_state(root: Path) -> tuple[str, str]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14BLockError("cannot verify Phase14B Git state") from exc
    return head, dirty


def _read_phase14a(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    a_dir = root / "artifacts/runs/phase14_teacher_audit_v2"
    manifest_path = a_dir / "manifest.json"
    records_path = a_dir / "forced_action_values.jsonl"
    lock_path = root / "experiments/phase14_source_lock_v2.json"
    audit_path = root / "artifacts/runs/phase14_training_source_audit_v2.json"
    if not all(path.is_file() for path in (manifest_path, records_path, lock_path, audit_path)):
        raise Phase14BLockError("Phase14A passed audit artifacts/lock are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "phase14a_teacher_audit_v2"
            or manifest.get("status") != "completed_gate_passed"
            or manifest.get("phase14a_gate_passed") is not True
            or manifest.get("reserve_accessed") is not False
            or manifest.get("phase12_validation_opened") is not False
            or manifest.get("model_training") is not False):
        raise Phase14BLockError("Phase14A did not pass the required teacher-audit gate")
    if (sha256_file(records_path) != manifest.get("output_jsonl_sha256")
            or records_path.stat().st_size != manifest.get("output_jsonl_bytes")):
        raise Phase14BLockError("Phase14A diagnostic output does not match its manifest")
    if lock.get("schema") != "phase14_source_lock_v2" or lock.get("status") != "authorized_phase14a_audit_only":
        raise Phase14BLockError("Phase14A source lock is missing or not the audit-only lock")
    if (lock.get("authorized_stage") != "phase14A_32_state_teacher_audit_only"
            or lock.get("reserve_access") is not False
            or lock.get("model_training") is not False):
        raise Phase14BLockError("Phase14A lock must remain audit-only and deny downstream work")
    if (manifest.get("source_lock_sha256") != sha256_file(lock_path)
            or manifest.get("source_audit_sha256") != sha256_file(audit_path)):
        raise Phase14BLockError("Phase14A result is not bound to its audit-only source lock")
    if lock.get("source_code_sha256") != {
        name: sha256_file(root / name) for name in PHASE14_SOURCE_CODE
    }:
        raise Phase14BLockError("Phase14A source files differ from its audit-only lock")
    if (source_audit.get("generation_git_commit") != lock.get("source_commit")
            or source_audit.get("generation_source_sha256") != lock.get("source_code_sha256")):
        raise Phase14BLockError("Phase14 source audit/lock provenance mismatch")
    if source_audit.get("status") != "exact_training_archive_replay_passed_no_phase14_labels":
        raise Phase14BLockError("Phase14 source replay audit did not pass")
    if (source_audit.get("counterfactual_transitions_evaluated") is not False
            or source_audit.get("reserve_accessed") is not False
            or source_audit.get("phase12_validation_opened") is not False):
        raise Phase14BLockError("Phase14 source audit has invalid split/outcome flags")
    archive_hashes = {name: sha256_file(root / SOURCE_DIR / name) for name in SOURCE_FILES}
    if (manifest.get("phase12_training_archive_sha256") != archive_hashes
            or source_audit.get("training_archive_sha256") != archive_hashes):
        raise Phase14BLockError("Phase12 training archive differs from Phase14A provenance")
    if config["source"]["expected_states"] != source_audit.get("training_decision_state_count"):
        raise Phase14BLockError("Phase14B training state count differs from source audit")
    return manifest, lock, archive_hashes


def projected_full_label_seconds(phase14a_manifest: dict[str, Any], full_state_count: int) -> float:
    # Phase14A evaluated 32 unique states plus 8 deliberate replay states.
    measured_equivalents = 32 + int(phase14a_manifest["metric_summary"]["repeatability_states"])
    return float(phase14a_manifest["wall_seconds"]) * full_state_count / measured_equivalents


def create_training_lock(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    lock_path = root / config["outputs"]["lock"]
    if lock_path.exists():
        raise Phase14BLockError("refusing to overwrite Phase14B training-label lock")
    head, dirty = _git_state(root)
    if dirty:
        raise Phase14BLockError("commit Phase14B source/config/tests before freezing label lock")
    phase14a_manifest, phase14a_lock, archive_hashes = _read_phase14a(root, config)
    estimate = projected_full_label_seconds(phase14a_manifest, int(config["source"]["expected_states"]))
    if estimate >= float(config["resources"]["wall_seconds"]):
        raise Phase14BLockError("measured projection does not fit the frozen full-label wall cap")
    source_audit_path = root / config["source"]["replay_audit"]
    phase14a_manifest_path = root / "artifacts/runs/phase14_teacher_audit_v2/manifest.json"
    phase14a_records_path = root / "artifacts/runs/phase14_teacher_audit_v2/forced_action_values.jsonl"
    lock = {
        "schema": "phase14b_training_label_lock_v1",
        "status": "authorized_phase14b_full_training_labels_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_reference": "user instruction opening Phase14 and authorizing gated progression on 2026-09-28",
        "source_commit": head,
        "config_sha256": sha256_file(root / CONFIG_PATH),
        "phase14_config_sha256": sha256_file(root / PHASE14_CONFIG),
        "protocol_sha256": sha256_file(root / PROTOCOL_PATH),
        "source_code_sha256": {path: sha256_file(root / path) for path in PHASE14B_SOURCE_CODE},
        "phase14a_manifest_sha256": sha256_file(phase14a_manifest_path),
        "phase14a_records_sha256": sha256_file(phase14a_records_path),
        "phase14a_lock_sha256": sha256_file(root / "experiments/phase14_source_lock_v2.json"),
        "phase14_source_audit_sha256": sha256_file(source_audit_path),
        "phase12_training_archive_sha256": archive_hashes,
        "phase14a_observed_wall_seconds": phase14a_manifest["wall_seconds"],
        "full_label_wall_projection_seconds": estimate,
        "training_states": int(config["source"]["expected_states"]),
        "actions_per_state": 9,
        "authorized_stage": "full_training_labels_only",
        "student_training": False,
        "heldout_access": False,
        "dagger": False,
        "world_model": False,
        "reserve_access": False,
        "resources": config["resources"],
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return lock


def validate_training_lock(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(root).resolve()
    lock_path = root / config["outputs"]["lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("schema") != "phase14b_training_label_lock_v1"
            or lock.get("status") != "authorized_phase14b_full_training_labels_only"
            or lock.get("authorized_stage") != "full_training_labels_only"):
        raise Phase14BLockError("Phase14B lock does not authorize the label-generation stage")
    if any(lock.get(name) is not False for name in (
        "student_training", "heldout_access", "dagger", "world_model", "reserve_access"
    )):
        raise Phase14BLockError("Phase14B lock improperly authorizes downstream work")
    if lock.get("config_sha256") != sha256_file(root / CONFIG_PATH):
        raise Phase14BLockError("Phase14B config hash mismatch")
    if lock.get("phase14_config_sha256") != sha256_file(root / PHASE14_CONFIG):
        raise Phase14BLockError("Phase14 config hash mismatch")
    if lock.get("protocol_sha256") != sha256_file(root / PROTOCOL_PATH):
        raise Phase14BLockError("Phase14 protocol hash mismatch")
    if lock.get("source_code_sha256") != {
        path: sha256_file(root / path) for path in PHASE14B_SOURCE_CODE
    }:
        raise Phase14BLockError("Phase14B source code hash mismatch")
    phase14a_manifest, _, archive_hashes = _read_phase14a(root, config)
    if lock.get("phase14a_manifest_sha256") != sha256_file(root / "artifacts/runs/phase14_teacher_audit_v2/manifest.json"):
        raise Phase14BLockError("Phase14A manifest changed after Phase14B lock")
    if lock.get("phase14a_records_sha256") != phase14a_manifest.get("output_jsonl_sha256"):
        raise Phase14BLockError("Phase14A records hash mismatch")
    if lock.get("phase14a_lock_sha256") != sha256_file(root / "experiments/phase14_source_lock_v2.json"):
        raise Phase14BLockError("Phase14A source lock hash mismatch")
    if lock.get("phase14_source_audit_sha256") != sha256_file(root / config["source"]["replay_audit"]):
        raise Phase14BLockError("Phase14 train-only source audit hash mismatch")
    if lock.get("phase12_training_archive_sha256") != archive_hashes:
        raise Phase14BLockError("Phase12 training archive changed after Phase14B lock")
    if (lock.get("training_states") != int(config["source"]["expected_states"])
            or lock.get("actions_per_state") != 9):
        raise Phase14BLockError("Phase14B dataset cardinality differs from the frozen contract")
    if lock.get("resources") != config.get("resources"):
        raise Phase14BLockError("Phase14B resource limits differ from config")
    head, dirty = _git_state(root)
    if head != lock.get("source_commit"):
        try:
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"],
                cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Phase14BLockError("Phase14B locked source commit is not an ancestor") from exc
    if dirty:
        raise Phase14BLockError("working tree must be clean after committing Phase14B lock")
    return lock


def make_compact_state_record(source: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [evaluate_forced_action(source, action_id) for action_id in range(9)]
    ranked = add_state_ranks(rows)
    compact_actions = []
    for row in ranked:
        compact_actions.append({
            key: row[key]
            for key in (
                "forced_action_id", "forced_collision", "forced_success", "forced_timeout",
                "forced_successor", "terminal_status", "terminal_state",
                "terminal_steps_from_source", "terminal_goal_distance",
                "total_noncoast_effort", "objective_tuple", "continuation_decision_count",
                "average_rank", "normalized_regret", "optimal",
            )
        })
    record = {
        "source_episode_seed": source.episode_seed,
        "source_layout_sha256": source.layout_sha256,
        "source_decision_index": source.decision_index,
        "global_query_index": source.global_query_index,
        "student_input_52": list(source.features),
        "actions": compact_actions,
    }
    return record, compact_actions
