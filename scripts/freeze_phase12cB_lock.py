"""Freeze Phase 12c-B config, state sample and source/data hashes after approval."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    config_path = ROOT / "configs/phase12cB_locked.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lock_path = ROOT / config["runner_contract"]["authorization_lock"]
    if lock_path.exists():
        raise SystemExit("authorization lock already exists; refusing overwrite")
    if config.get("phase") != "12c-B" or config.get("authorization_status") != "authorized":
        raise SystemExit("config is not the authorized Phase 12c-B protocol")
    output_path = ROOT / config["runner_contract"]["output_path"]
    if output_path.exists():
        raise SystemExit("run output path already exists")

    audit_path = ROOT / config["source_data"]["replay_audit"]
    sample_path = ROOT / config["state_sampling"]["sample_manifest"]
    source_lock_path = ROOT / config["source_data"]["execution_lock"]
    teacher_manifest_path = ROOT / config["source_data"]["teacher_qualification_manifest"]
    teacher_analysis_path = ROOT / config["source_data"]["teacher_qualification_analysis"]
    for path in (audit_path, sample_path, source_lock_path, teacher_manifest_path, teacher_analysis_path):
        if not path.is_file():
            raise SystemExit(f"required source artifact missing: {path.relative_to(ROOT)}")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sample_manifest = json.loads(sample_path.read_text(encoding="utf-8"))
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    teacher_analysis = json.loads(teacher_analysis_path.read_text(encoding="utf-8"))
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if audit.get("status") != "exact_replay_passed" or audit.get("reserve_accessed") is not False:
        raise SystemExit("fitting-data replay audit did not pass its information boundary")
    if audit.get("total_exactly_replayed_decision_states") != 2828:
        raise SystemExit("unexpected exact replay state count")
    if sample_manifest.get("sample_count") != 160 or sample_manifest.get("layout_count") != 80:
        raise SystemExit("sample manifest count mismatch")
    if sample_manifest.get("source_split_layout_counts") != {
        "phase12_train_teacher_v1": 60,
        "phase12_validation_teacher_v1": 20,
    }:
        raise SystemExit("sample manifest split allocation mismatch")
    if sample_manifest.get("progress_speed_strata_counts") != {
        "early_low": 40, "early_high": 40, "late_low": 40, "late_high": 40,
    }:
        raise SystemExit("sample manifest is not balanced across frozen strata")
    if sample_manifest.get("source_data_replay_audit_sha256") != sha256(audit_path):
        raise SystemExit("sample manifest does not match the preflight replay audit")
    if sha256(source_lock_path) != config["source_data"]["execution_lock_sha256"]:
        raise SystemExit("fitting-data source lock hash mismatch")
    if sha256(audit_path) != config["source_data"]["replay_audit_sha256"]:
        raise SystemExit("replay audit hash mismatch")
    if sha256(teacher_manifest_path) != config["source_data"]["teacher_qualification_manifest_sha256"]:
        raise SystemExit("Phase 12c-A manifest hash mismatch")
    if sha256(teacher_analysis_path) != config["source_data"]["teacher_qualification_analysis_sha256"]:
        raise SystemExit("Phase 12c-A analysis hash mismatch")
    if teacher_manifest.get("status") != "completed_phase12c_a_only" or not teacher_analysis.get("all_gates_pass"):
        raise SystemExit("Phase 12c-A qualification gate is not verified as passed")
    if source_lock.get("reserve_accessed") is not False:
        raise SystemExit("source data lock does not preserve the reserve boundary")

    input_hashes = dict(audit["splits"]["phase12_train_teacher_v1"]["files_sha256"])
    input_hashes.update(audit["splits"]["phase12_validation_teacher_v1"]["files_sha256"])
    input_hashes[source_lock_path.relative_to(ROOT).as_posix()] = sha256(source_lock_path)
    input_hashes[audit_path.relative_to(ROOT).as_posix()] = sha256(audit_path)
    input_hashes[teacher_manifest_path.relative_to(ROOT).as_posix()] = sha256(teacher_manifest_path)
    input_hashes[teacher_analysis_path.relative_to(ROOT).as_posix()] = sha256(teacher_analysis_path)

    source_hashes: dict[str, str] = {}
    for relative in config["source_files"]:
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise SystemExit(f"unsafe or missing source file: {relative}")
        source_hashes[relative] = sha256(path)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True

    payload = {
        "format": "phase12c_b_execution_lock_v1",
        "phase": "12c-B",
        "experiment": config["experiment"],
        "authorization_status": "authorized",
        "approval_reference": (
            "User explicitly authorized Phase 12c-B on 2026-09-28; scope is the frozen "
            "160-state/80-fitting-layout repeated-bank audit only."
        ),
        "approved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_git_commit": commit,
        "working_tree_was_dirty": dirty,
        "execution_authorized": True,
        "training_authorized": False,
        "dagger_authorized": False,
        "development_authorized": False,
        "reserve_accessed": False,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "sample_manifest_path": sample_path.relative_to(ROOT).as_posix(),
        "sample_manifest_sha256": sha256(sample_path),
        "replay_audit_path": audit_path.relative_to(ROOT).as_posix(),
        "replay_audit_sha256": sha256(audit_path),
        "training_source_lock_path": source_lock_path.relative_to(ROOT).as_posix(),
        "training_source_lock_sha256": sha256(source_lock_path),
        "input_data_sha256": dict(sorted(input_hashes.items())),
        "source_sha256": dict(sorted(source_hashes.items())),
        "caps": {
            "wall_seconds": int(config["compute_caps"]["wall_seconds"]),
            "artifact_bytes": int(config["compute_caps"]["generated_artifact_bytes"]),
        },
        "scope": {
            "states": 160,
            "layouts": 80,
            "banks_per_state": 12,
            "expected_bank_records": 1920,
            "allowed_operation": "balanced exact proposal banks at archived fitting-only states",
            "forbidden": ["student_training", "dagger", "development_test_access", "reserve_access"],
        },
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
