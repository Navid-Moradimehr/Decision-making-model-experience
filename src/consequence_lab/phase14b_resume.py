"""Source-bound continuation for the Phase14B training-only label dataset."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import yaml

from consequence_lab.phase14 import replay_training_states, sha256_file
from consequence_lab.phase14b import (
    CONFIG_PATH as BASE_CONFIG_PATH,
    LOCK_PATH as BASE_LOCK_PATH,
    PHASE14B_SOURCE_CODE,
    make_compact_state_record,
    validate_training_lock,
)


CONFIG_PATH = "configs/phase14b_resume.yaml"
LOCK_PATH = "experiments/phase14b_training_labels_resume_v1.lock.json"
SOURCE_FILES = tuple(dict.fromkeys((
    BASE_CONFIG_PATH,
    BASE_LOCK_PATH,
    "configs/phase14.yaml",
    "configs/phase14b_resume.yaml",
    "configs/phase12_teacher_student.yaml",
    "reports/phase14_first_action_value_learnability_protocol.md",
    "reports/phase14a_counterfactual_teacher_audit_result.md",
    "reports/phase14b_full_training_labels_protocol.md",
    "reports/phase14b_full_training_labels_result.md",
    "reports/phase14b_resume_protocol.md",
    *PHASE14B_SOURCE_CODE,
    "src/consequence_lab/phase14b_resume.py",
    "scripts/freeze_phase14b_resume_lock.py",
    "scripts/run_phase14b_resume.py",
    "tests/test_phase14b_resume.py",
)))


class Phase14BResumeError(RuntimeError):
    """Continuation provenance, source-alignment, or data-integrity failure."""


def read_config(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))


def _git_state(root: Path) -> tuple[str, str]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14BResumeError("cannot verify continuation Git state") from exc
    return head, dirty


def _source_fields_match(row: dict[str, Any], source: Any, query_index: int) -> bool:
    if row.get("global_query_index") != query_index:
        return False
    if row.get("source_episode_seed") != source.episode_seed:
        return False
    if row.get("source_layout_sha256") != source.layout_sha256:
        return False
    if row.get("source_decision_index") != source.decision_index:
        return False
    try:
        features = np.asarray(row["student_input_52"], dtype=np.float64)
        expected = np.asarray(source.features, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    return features.shape == (52,) and np.array_equal(features, expected)


def validate_record_range(
    rows: list[dict[str, Any]], sources: list[Any], start: int, stop: int,
) -> list[list[dict[str, Any]]]:
    """Validate ordered state identities/features and each fixed nine-action target row."""
    if start < 0 or stop > len(sources) or stop < start or len(rows) != stop - start:
        raise Phase14BResumeError("record range has invalid bounds or cardinality")
    action_targets: list[list[dict[str, Any]]] = []
    for query_index, row in zip(range(start, stop), rows, strict=True):
        source = sources[query_index]
        if not _source_fields_match(row, source, query_index):
            raise Phase14BResumeError(f"query {query_index} does not match its train-source state")
        actions = row.get("actions")
        if not isinstance(actions, list) or len(actions) != 9:
            raise Phase14BResumeError(f"query {query_index} must have exactly nine action records")
        try:
            action_ids = [int(action["forced_action_id"]) for action in actions]
            regrets = [float(action["normalized_regret"]) for action in actions]
            ranks = [float(action["average_rank"]) for action in actions]
            tuples = [action["objective_tuple"] for action in actions]
            statuses = [action["terminal_status"] for action in actions]
        except (KeyError, TypeError, ValueError) as exc:
            raise Phase14BResumeError(f"query {query_index} has malformed action targets") from exc
        if action_ids != list(range(9)):
            raise Phase14BResumeError(f"query {query_index} action IDs are missing, duplicated, or reordered")
        if (not np.isfinite(regrets).all() or not np.isfinite(ranks).all()
                or any(len(value) != 3 for value in tuples)
                or any(status not in {"success", "timeout", "collision"} for status in statuses)):
            raise Phase14BResumeError(f"query {query_index} contains invalid targets")
        action_targets.append(actions)
    return action_targets


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise Phase14BResumeError(f"JSONL line {line_number} is not an object")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase14BResumeError(f"cannot read valid JSONL: {path}") from exc
    return rows


def validate_prior_run(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]]]:
    src = config["source"]
    base_config = yaml.safe_load((root / src["base_config"]).read_text(encoding="utf-8"))
    base_lock_path = root / src["base_lock"]
    if sha256_file(base_lock_path) != src["base_lock_sha256"]:
        raise Phase14BResumeError("frozen Phase14B v1 lock hash differs")
    base_lock = validate_training_lock(root, base_config)
    if base_lock.get("student_training") is not False:
        raise Phase14BResumeError("base Phase14B lock must remain labels-only")

    partial_path = root / src["partial_records"]
    manifest_path = root / src["partial_manifest"]
    if (sha256_file(partial_path) != src["partial_records_sha256"]
            or sha256_file(manifest_path) != src["partial_manifest_sha256"]):
        raise Phase14BResumeError("preserved Phase14B v1 output hash differs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_flags = {
        "status": "aborted_or_incomplete",
        "completed_state_rows": int(src["expected_partial_rows"]),
        "phase14b_lock_sha256": src["base_lock_sha256"],
        "student_training": False,
        "heldout_access": False,
        "dagger": False,
        "world_model": False,
        "reserve_accessed": False,
    }
    if any(manifest.get(key) != value for key, value in expected_flags.items()):
        raise Phase14BResumeError("preserved Phase14B manifest does not match the failed run")

    sources = replay_training_states(root)
    if len(sources) != int(src["total_training_rows"]):
        raise Phase14BResumeError("training archive replay count differs from frozen continuation config")
    prefix = read_jsonl(partial_path)
    validate_record_range(prefix, sources, 0, int(src["expected_partial_rows"]))
    return base_lock, sources, prefix


def _validate_config(root: Path, config: dict[str, Any]) -> None:
    if config.get("schema") != "phase14b_resume_config_v1":
        raise Phase14BResumeError("unsupported continuation config schema")
    if config.get("authorization", {}).get("student_training") is not False:
        raise Phase14BResumeError("continuation cannot authorize student training")
    if any(config.get("authorization", {}).get(flag) is not False for flag in (
        "heldout_access", "dagger", "world_model", "reserve_access"
    )):
        raise Phase14BResumeError("continuation improperly opens a forbidden research scope")
    if config.get("resources", {}).get("wall_time_cap") is not None:
        raise Phase14BResumeError("user-authorized continuation must not restore the old time cap")
    if config.get("resources", {}).get("generated_storage_cap") is not None:
        raise Phase14BResumeError("user-authorized continuation must not restore the old storage cap")


def create_continuation_lock(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config = read_config(root)
    _validate_config(root, config)
    lock_path = root / config["outputs"]["lock"]
    if lock_path.exists():
        raise Phase14BResumeError("refusing to overwrite continuation lock")
    head, dirty = _git_state(root)
    if dirty:
        raise Phase14BResumeError("commit continuation code/config/tests before freezing the lock")
    base_lock, _, _ = validate_prior_run(root, config)
    src = config["source"]
    files = {path: sha256_file(root / path) for path in SOURCE_FILES}
    lock = {
        "schema": "phase14b_training_labels_resume_lock_v1",
        "status": "authorized_complete_training_labels_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_reference": config["authorization"]["reference"],
        "source_commit": head,
        "source_code_sha256": files,
        "base_phase14b_lock_sha256": sha256_file(root / src["base_lock"]),
        "base_phase14b_source_commit": base_lock["source_commit"],
        "failed_partial_records_sha256": sha256_file(root / src["partial_records"]),
        "failed_partial_manifest_sha256": sha256_file(root / src["partial_manifest"]),
        "config_sha256": sha256_file(root / CONFIG_PATH),
        "protocol_sha256": sha256_file(root / "reports/phase14b_resume_protocol.md"),
        "training_archive": src["train_archive"],
        "training_seed_range": src["train_seed_range"],
        "total_training_rows": int(src["total_training_rows"]),
        "prefix_rows": int(src["expected_partial_rows"]),
        "continuation_start_query": int(src["expected_partial_rows"]),
        "continuation_rows": int(src["total_training_rows"]) - int(src["expected_partial_rows"]),
        "actions_per_state": int(src["actions_per_state"]),
        "authorized_stage": "complete_phase14b_training_labels_only",
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


def validate_continuation_lock(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]]]:
    root = Path(root).resolve()
    _validate_config(root, config)
    lock_path = root / config["outputs"]["lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_status = {
        "schema": "phase14b_training_labels_resume_lock_v1",
        "status": "authorized_complete_training_labels_only",
        "authorized_stage": "complete_phase14b_training_labels_only",
        "student_training": False,
        "heldout_access": False,
        "dagger": False,
        "world_model": False,
        "reserve_access": False,
    }
    if any(lock.get(key) != value for key, value in expected_status.items()):
        raise Phase14BResumeError("continuation lock does not enforce labels-only scope")
    if lock.get("config_sha256") != sha256_file(root / CONFIG_PATH):
        raise Phase14BResumeError("continuation config hash mismatch")
    if lock.get("protocol_sha256") != sha256_file(root / "reports/phase14b_resume_protocol.md"):
        raise Phase14BResumeError("continuation protocol hash mismatch")
    if lock.get("source_code_sha256") != {path: sha256_file(root / path) for path in SOURCE_FILES}:
        raise Phase14BResumeError("continuation source-code hash mismatch")
    if lock.get("resources") != config["resources"]:
        raise Phase14BResumeError("continuation resource declaration differs from config")
    head, dirty = _git_state(root)
    if head != lock.get("source_commit"):
        try:
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"],
                cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Phase14BResumeError("continuation lock commit is not an ancestor") from exc
    if dirty:
        raise Phase14BResumeError("working tree must be clean after committing continuation lock")
    base_lock, sources, prefix = validate_prior_run(root, config)
    if lock.get("base_phase14b_lock_sha256") != sha256_file(root / config["source"]["base_lock"]):
        raise Phase14BResumeError("base Phase14B lock changed after continuation lock")
    if lock.get("base_phase14b_source_commit") != base_lock["source_commit"]:
        raise Phase14BResumeError("base Phase14B source commit changed")
    return lock, sources, prefix


def iter_remaining_rows(sources: list[Any], start: int):
    if start < 0 or start > len(sources):
        raise Phase14BResumeError("continuation start index outside source state list")
    for expected_index, source in enumerate(sources[start:], start=start):
        if source.global_query_index != expected_index:
            raise Phase14BResumeError("training-source query index changed during replay")
        record, _ = make_compact_state_record(source)
        validate_record_range([record], sources, expected_index, expected_index + 1)
        yield record


def generate_remaining_rows(sources: list[Any], start: int) -> list[dict[str, Any]]:
    """Small test/helper API; production runner streams with iter_remaining_rows."""
    return list(iter_remaining_rows(sources, start))
