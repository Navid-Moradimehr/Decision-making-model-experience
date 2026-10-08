"""Freeze Phase 12c-C fitting/model-selection authorization after integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="experiments/phase12cC_training.lock.json")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    if args.confirmation != "FREEZE_PHASE12C_C_TRAINING_LOCK":
        raise SystemExit("lock creation requires --confirmation FREEZE_PHASE12C_C_TRAINING_LOCK")
    config_path = ROOT / "configs/phase12cC_locked.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_path = ROOT / config["seed_splits"]["hash_audit"]["output"]
    result_path = ROOT / config["source_data"]["repeated_bank_analysis"]
    cA_lock_path = ROOT / config["seed_splits"]["hash_audit"]["prior_phase12cA_authorization_lock"]
    cA_audit_path = ROOT / config["seed_splits"]["hash_audit"]["prior_phase12cA_layout_audit"]
    required = (audit_path, result_path, cA_lock_path, cA_audit_path)
    if not all(path.is_file() for path in required):
        raise SystemExit("Phase 12c-C hash audit or prerequisite result is missing")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cA_lock = json.loads(cA_lock_path.read_text(encoding="utf-8"))
    if result.get("primary_gate_pass") is not True:
        raise SystemExit("conditional student-fitting authorization not satisfied: Phase 12c-B gate failed")
    if result.get("reserve_accessed") is not False or result.get("dagger_performed") is not False:
        raise SystemExit("Phase 12c-B prerequisite has invalid scope")
    if not audit.get("new_hashes_disjoint_from_all_phase11_phase12_phase12b_phase12cA_blocks"):
        raise SystemExit("Phase 12c-C layout audit did not pass")
    if audit.get("reserve_accessed") is not False or audit.get("development_outcomes_accessed") is not False:
        raise SystemExit("Phase 12c-C layout audit touched protected outcomes")
    if cA_lock.get("layout_audit_sha256") != digest(cA_audit_path):
        raise SystemExit("Phase 12c-A layout audit no longer matches its lock")
    if audit.get("config_sha256") != digest(config_path):
        raise SystemExit("Phase 12c-C layout audit was made against a different config")
    if digest(result_path) != config["source_data"]["repeated_bank_analysis_sha256"]:
        raise SystemExit("Phase 12c-B prerequisite result hash mismatch")
    replay_path = ROOT / config["source_data"]["exact_replay_audit"]
    data_audit_path = ROOT / config["source_data"]["source_data_audit"]
    teacher_manifest_path = ROOT / config["source_data"]["balanced_teacher_manifest"]
    old_lock_path = ROOT / config["source_data"]["training_source_lock"]
    if digest(replay_path) != config["source_data"]["exact_replay_audit_sha256"]:
        raise SystemExit("exact-replay audit hash mismatch")
    if digest(data_audit_path) != config["source_data"]["source_data_audit_sha256"]:
        raise SystemExit("archived Phase 12 data audit hash mismatch")
    if digest(teacher_manifest_path) != config["source_data"]["balanced_teacher_manifest_sha256"]:
        raise SystemExit("Phase 12c-A balanced-teacher manifest hash mismatch")
    if digest(old_lock_path) != config["source_data"]["training_source_lock_sha256"]:
        raise SystemExit("archived Phase 12 training lock hash mismatch")
    data_audit = json.loads(data_audit_path.read_text(encoding="utf-8"))
    if data_audit.get("status") != "passed" or data_audit.get("reserve_accessed") is not False:
        raise SystemExit("archived Phase 12 data audit does not pass the frozen source contract")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if (replay.get("status") != "exact_replay_passed"
            or replay.get("reserve_accessed") is not False
            or replay.get("student_training_performed") is not False
            or replay.get("total_exactly_replayed_decision_states") != config["source_data"]["expected_replayed_states"]):
        raise SystemExit("exact replay audit does not match the frozen source replay contract")

    source_hashes = {}
    for relative in config["source_files"]:
        path = (ROOT / relative).resolve()
        if not inside(path, ROOT) or not path.is_file():
            raise SystemExit(f"locked source missing/outside repository: {relative}")
        source_hashes[relative] = digest(path)

    training_data_hashes = {}
    old_lock = json.loads(old_lock_path.read_text(encoding="utf-8"))
    for split in ("phase12_train_teacher_v1", "phase12_validation_teacher_v1"):
        directory = ROOT / "artifacts/runs" / split
        for filename in ("student_inputs.npz", "sample_metadata.npz", "teacher_targets.npz", "episodes.json", "manifest.json"):
            relative = (Path("artifacts/runs") / split / filename).as_posix()
            path = ROOT / relative
            if not path.is_file():
                raise SystemExit(f"archived fitting source missing: {relative}")
            expected = old_lock.get("training_data_sha256", {}).get(relative)
            if expected is None or digest(path) != expected:
                raise SystemExit(f"archived fitting source differs from Phase 12b training lock: {relative}")
            training_data_hashes[relative] = digest(path)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except Exception:
        commit, dirty = None, True
    lock = {
        "schema": "phase12c_c_training_authorization_v1",
        "phase": "12c-C",
        "authorization_reference": "User authorized the Phase 12c-C plan and conditional student fitting; condition satisfied by Phase 12c-B primary_gate_pass=true.",
        "training_authorized": True,
        "offline_gate_labels_collected": False,
        "development_authorized": False,
        "dagger_authorized": False,
        "reserve_accessed": False,
        "base_git_commit": commit,
        "working_tree_was_dirty": dirty,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_policy": "cpu",
        },
        "config": "configs/phase12cC_locked.yaml",
        "config_sha256": digest(config_path),
        "phase12cB_analysis": config["source_data"]["repeated_bank_analysis"],
        "phase12cB_analysis_sha256": digest(result_path),
        "phase12cC_layout_audit": config["seed_splits"]["hash_audit"]["output"],
        "phase12cC_layout_audit_sha256": digest(audit_path),
        "phase12cA_layout_audit_sha256": digest(cA_audit_path),
        "phase12cA_lock_sha256": digest(cA_lock_path),
        "training_source_lock_sha256": digest(old_lock_path),
        "training_data_sha256": training_data_hashes,
        "source_sha256": source_hashes,
        "split_hashes": audit["new_blocks"],
        "caps": {
            "wall_seconds": config["resource_caps"]["max_total_wall_seconds"],
            "fit_wall_seconds": config["resource_caps"]["max_fit_wall_seconds"],
            "artifact_bytes": config["resource_caps"]["max_generated_artifact_bytes"],
            "model_parameters": config["resource_caps"]["max_parameters"],
        },
        "scope": {
            "training_states": config["source_data"]["expected_replayed_states"],
            "model_selection_layouts": config["seed_splits"]["model_selection"]["count"],
            "offline_gate_authorized_after_checkpoint_lock": True,
            "closed_loop_development": False,
            "dagger": False,
            "reserve": False,
        },
    }
    output_path = (ROOT / args.output).resolve()
    if not inside(output_path, ROOT / "experiments") or output_path.exists():
        raise SystemExit("lock output must be a new file under experiments/")
    output_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "phase12cC_training_lock_frozen",
        "output": str(output_path),
        "lock_sha256": digest(output_path),
        "training_authorized": True,
        "offline_gate_labels_collected": False,
        "reserve_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
