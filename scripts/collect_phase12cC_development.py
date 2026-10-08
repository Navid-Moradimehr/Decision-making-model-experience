"""Collect fitting targets and model-selection labels only; never opens gate/dev/reserve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12c_c import (
    build_archived_training_dataset,
    collect_balanced_teacher_dataset,
    persist_teacher_dataset,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--config", default="configs/phase12cC_locked.yaml")
    parser.add_argument("--lock", default="experiments/phase12cC_training.lock.json")
    parser.add_argument("--training-output")
    parser.add_argument("--selection-output")
    args = parser.parse_args()
    if args.confirmation != "COLLECT_PHASE12C_C_TRAIN_AND_SELECTION_ONLY":
        raise SystemExit("collection requires --confirmation COLLECT_PHASE12C_C_TRAIN_AND_SELECTION_ONLY")
    config_path, lock_path = ROOT / args.config, ROOT / args.lock
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase12c_c_training_authorization_v1" or lock.get("training_authorized") is not True:
        raise SystemExit("training/model-selection lock does not authorize this stage")
    if lock.get("offline_gate_labels_collected") is not False or lock.get("reserve_accessed") is not False:
        raise SystemExit("lock has already opened a forbidden data split")
    if digest(config_path) != lock.get("config_sha256"):
        raise SystemExit("Phase 12c-C config/lock integrity failure")
    for relative, expected in lock["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"locked source changed: {relative}")
    for relative, expected in lock["training_data_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"archived training source changed: {relative}")
    audit_path = ROOT / config["seed_splits"]["hash_audit"]["output"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if digest(audit_path) != lock.get("phase12cC_layout_audit_sha256"):
        raise SystemExit("layout audit hash differs from frozen fitting lock")
    if audit.get("reserve_accessed") is not False or not audit.get("new_blocks_pairwise_disjoint"):
        raise SystemExit("layout audit violates split protection")

    train_output = ROOT / (args.training_output or config["data_generation"]["outputs"]["archived_training"])
    selection_output = ROOT / (args.selection_output or config["data_generation"]["outputs"]["model_selection"])
    for path in (train_output, selection_output):
        if path.exists() or not path.resolve().is_relative_to((ROOT / "artifacts/runs").resolve()):
            raise SystemExit("outputs must be new paths under artifacts/runs")
    if lock["scope"].get("offline_gate_authorized_after_checkpoint_lock") is not True:
        raise SystemExit("lock does not preserve the checkpoint-before-gate boundary")

    started = time.monotonic()
    master = int(config["data_generation"]["master_seed"])
    wall_cap = int(config["resource_caps"]["max_total_wall_seconds"])
    train = build_archived_training_dataset(
        ROOT, master_seed=master, banks=12,
        suffixes_per_action=int(config["teacher"]["suffixes_per_first_action"]),
        time_cap_seconds=wall_cap,
    )
    train_manifest = persist_teacher_dataset(
        train_output, train, split="archived_training", lock_sha256=digest(lock_path),
        config_sha256=digest(config_path), max_bytes=int(config["resource_caps"]["max_generated_artifact_bytes"]),
    )

    remaining_time = wall_cap - (time.monotonic() - started)
    if remaining_time <= 0:
        raise TimeoutError("Phase 12c-C total collection time cap reached before selection stage")
    selection_block = audit["new_blocks"]["model_selection"]
    selection = collect_balanced_teacher_dataset(
        selection_block["seeds"], selection_block["layout_sha256"], split="model_selection",
        master_seed=master, banks=12,
        suffixes_per_action=int(config["teacher"]["suffixes_per_first_action"]),
        time_cap_seconds=int(remaining_time),
        geometry_resolution=float(config["evaluation"]["geometry_parameters"]["resolution"]),
        waypoint_tolerance=float(config["evaluation"]["geometry_parameters"]["waypoint_tolerance"]),
    )
    selection_manifest = persist_teacher_dataset(
        selection_output, selection, split="model_selection", lock_sha256=digest(lock_path),
        config_sha256=digest(config_path), max_bytes=int(config["resource_caps"]["max_generated_artifact_bytes"]),
    )
    total_artifacts = train_manifest["artifact_bytes"] + selection_manifest["artifact_bytes"]
    if total_artifacts > int(config["resource_caps"]["max_generated_artifact_bytes"]):
        raise OSError("combined Phase 12c-C train/selection artifact cap exceeded")
    print(json.dumps({
        "status": "phase12cC_train_and_selection_data_collected",
        "training_queries": train_manifest["query_count"],
        "training_layouts": train_manifest["layout_count"],
        "selection_queries": selection_manifest["query_count"],
        "selection_layouts": selection_manifest["layout_count"],
        "training_artifact_bytes": train_manifest["artifact_bytes"],
        "selection_artifact_bytes": selection_manifest["artifact_bytes"],
        "total_artifact_bytes": total_artifacts,
        "wall_seconds": time.monotonic() - started,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "offline_gate_labels_collected": False,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
