"""Freeze the approved Phase 12c-E source/data lock before any new outcome access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase12cE.yaml"
LOCK = ROOT / "experiments/phase12cE_execution_v2.lock.json"

SOURCE_FILES = (
    "configs/phase12cE.yaml",
    "reports/phase12cE_geometry_relative_protocol_draft.md",
    "src/consequence_lab/phase12cE.py",
    "src/consequence_lab/phase12c_c.py",
    "src/consequence_lab/phase12c_c_training.py",
    "src/consequence_lab/phase12_model.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/phase12_data.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "scripts/freeze_phase12cE_lock.py",
    "scripts/audit_phase12cE_layouts.py",
    "scripts/collect_phase12cE_dataset.py",
    "scripts/verify_phase12cE_replay.py",
    "scripts/train_phase12cE_students.py",
    "scripts/run_phase12cE_offline_gate.py",
    "tests/test_phase12cE.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    if args.confirmation != "FREEZE_PHASE12CE_EXECUTION_LOCK":
        raise SystemExit("lock creation requires --confirmation FREEZE_PHASE12CE_EXECUTION_LOCK")
    if LOCK.exists():
        raise SystemExit(f"refusing to overwrite {LOCK}")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise SystemExit("freeze the lock only from a clean committed worktree")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config.get("phase") != "12c-E" or config.get("status") != "user-approved-matched-representation-comparison":
        raise SystemExit("Phase 12c-E config is not in its approved status")
    source_hashes = {}
    for relative in SOURCE_FILES:
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise SystemExit(f"locked source missing/outside repository: {relative}")
        source_hashes[relative] = digest(path)
    data_dir = ROOT / config["training_source"]["archived_dataset"]
    data_hashes = {}
    for relative in ("manifest.json", "student_inputs.npz", "teacher_targets.npz", "sample_metadata.json"):
        path = data_dir / relative
        if not path.is_file():
            raise SystemExit(f"fitting source missing: {path}")
        data_hashes[relative] = digest(path)
    replay_data_hashes = {}
    for split in ("phase12_train_teacher_v1", "phase12_validation_teacher_v1"):
        directory = ROOT / "artifacts/runs" / split
        for relative in ("manifest.json", "student_inputs.npz", "sample_metadata.npz"):
            path = directory / relative
            if not path.is_file():
                raise SystemExit(f"Phase 12 replay source missing: {path}")
            replay_data_hashes[f"{split}/{relative}"] = digest(path)
    if data_hashes["manifest.json"] != config["training_source"]["expected_manifest_sha256"]:
        raise SystemExit("archived fitting source manifest differs from approved protocol")
    phase12_lock_path = ROOT / config["trajectory_collection"]["source_lock"]
    phase12_lock = json.loads(phase12_lock_path.read_text(encoding="utf-8"))
    phase12_config_path = ROOT / config["trajectory_collection"]["source_config"]
    if digest(phase12_config_path) != phase12_lock["source_sha256"].get(
        config["trajectory_collection"]["source_config"]
    ):
        raise SystemExit("Phase 12 trajectory source config no longer matches its execution lock")
    for relative, expected in phase12_lock["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise SystemExit(f"Phase 12 trajectory source changed: {relative}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    lock = {
        "schema": "phase12cE_execution_lock_v2",
        "supersedes_lock": "experiments/phase12cE_execution.lock.json",
        "amendment": "Version 1 selection runner stopped before output creation due to a config-key lookup error; no new outcomes were accessed. Version 2 corrects only path wiring and uses distinct hash/replay audit artifact paths; protocol, seeds, thresholds, model and data scope are unchanged.",
        "authorization": "User approved all Phase 12c-E study steps on 2026-09-28; CPU has no fixed runtime cap. Approval is limited to approved selection/offline-gate protocol; no closed-loop development, DAgger, reserve, or world-model work.",
        "base_git_commit": commit,
        "config_sha256": digest(CONFIG),
        "source_sha256": source_hashes,
        "training_data_sha256": data_hashes,
        "training_manifest_sha256": data_hashes["manifest.json"],
        "replay_source_data_sha256": replay_data_hashes,
        "phase12_trajectory_lock_sha256": digest(phase12_lock_path),
        "phase12_trajectory_source_config_sha256": digest(phase12_config_path),
        "layout_audit_created": False,
        "selection_outcomes_accessed": False,
        "offline_gate_outcomes_accessed": False,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
        "cpu_runtime_cap_seconds": None,
        "artifact_cap_bytes": int(config["resources"]["generated_artifact_cap_bytes"]),
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "phase12cE_execution_lock_frozen", "lock": str(LOCK.relative_to(ROOT)),
                      "base_git_commit": commit, "source_count": len(source_hashes),
                      "training_manifest_sha256": lock["training_manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
