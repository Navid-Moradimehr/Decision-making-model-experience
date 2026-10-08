"""Fit the two frozen Phase 12c-C model families using train/selection only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12c_c_training import load_dataset, sha256, train_phase12c_c


def check_dataset(path: Path, split: str, cap: int) -> dict:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing data manifest for {split}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != split or manifest.get("reserve_accessed") is not False:
        raise SystemExit(f"unexpected data scope in {split}")
    if manifest.get("offline_gate_labels_collected") is not False:
        raise SystemExit(f"offline gate labels were exposed before model lock ({split})")
    for name, entry in manifest["files"].items():
        file_path = path / name
        if not file_path.is_file() or sha256(file_path) != entry["sha256"]:
            raise SystemExit(f"data file hash mismatch: {split}/{name}")
    if manifest.get("artifact_bytes", cap + 1) > cap:
        raise SystemExit(f"{split} data artifact cap exceeded")
    actual_bytes = sum((path / name).stat().st_size for name in manifest["files"]) + manifest_path.stat().st_size
    if int(manifest.get("artifact_bytes", -1)) != actual_bytes:
        raise SystemExit(f"{split} manifest artifact-byte total mismatch")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--config", default="configs/phase12cC_locked.yaml")
    parser.add_argument("--lock", default="experiments/phase12cC_training.lock.json")
    parser.add_argument("--training-data")
    parser.add_argument("--selection-data")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.confirmation != "TRAIN_PHASE12C_C_STUDENTS_ONLY":
        raise SystemExit("fitting requires --confirmation TRAIN_PHASE12C_C_STUDENTS_ONLY")
    config_path, lock_path = ROOT / args.config, ROOT / args.lock
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase12c_c_training_authorization_v1" or lock.get("training_authorized") is not True:
        raise SystemExit("frozen Phase 12c-C lock does not authorize student fitting")
    if lock.get("offline_gate_labels_collected") is not False or lock.get("reserve_accessed") is not False:
        raise SystemExit("student-fitting lock violates split boundary")
    runtime = lock.get("runtime", {})
    observed_runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_policy": "cpu",
    }
    for key, value in observed_runtime.items():
        if runtime.get(key) != value:
            raise SystemExit(f"training runtime differs from frozen lock for {key}")
    if sha256(config_path) != lock.get("config_sha256"):
        raise SystemExit("Phase 12c-C config hash mismatch")
    for relative, expected in lock["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"locked source file changed: {relative}")
    for relative, expected in lock["training_data_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"archived fitting source changed: {relative}")

    train_path = ROOT / (args.training_data or config["data_generation"]["outputs"]["archived_training"])
    selection_path = ROOT / (args.selection_data or config["data_generation"]["outputs"]["model_selection"])
    cap = int(config["resource_caps"]["max_generated_artifact_bytes"])
    train_manifest = check_dataset(train_path, "archived_training", cap)
    selection_manifest = check_dataset(selection_path, "model_selection", cap)
    if train_manifest["query_count"] != config["source_data"]["expected_replayed_states"]:
        raise SystemExit("training query count differs from frozen exact-replay scope")
    if train_manifest["layout_count"] != config["source_data"]["expected_layouts"]:
        raise SystemExit("training layout count differs from frozen exact-replay scope")
    if selection_manifest["layout_count"] != config["seed_splits"]["model_selection"]["count"]:
        raise SystemExit("selection layout count differs from audited block")
    started = time.monotonic()
    train, selection = load_dataset(train_path), load_dataset(selection_path)
    output_dir = ROOT / (args.output or config["data_generation"]["outputs"]["selected_models"])
    result = train_phase12c_c(
        train, selection, config=config, output_dir=output_dir,
        time_cap_seconds=int(config["resource_caps"]["max_fit_wall_seconds"]),
    )
    if time.monotonic() - started > int(config["resource_caps"]["max_fit_wall_seconds"]):
        raise SystemExit("Phase 12c-C fitting exceeded its frozen time cap")
    manifest = {
        "schema": "phase12cC_fit_manifest_v1",
        "training_lock_sha256": sha256(lock_path),
        "config_sha256": sha256(config_path),
        "training_manifest_sha256": sha256(train_path / "manifest.json"),
        "selection_manifest_sha256": sha256(selection_path / "manifest.json"),
        "selection_results_sha256": sha256(output_dir / "selection_results.json"),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "fit_wall_seconds": time.monotonic() - started,
        "offline_gate_read": False,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
        "artifact_bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
    }
    manifest_path = output_dir / "fit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    generated_roots = (train_path, selection_path, output_dir)
    total_generated = sum(
        file_path.stat().st_size
        for base in generated_roots for file_path in base.rglob("*") if file_path.is_file()
    )
    if total_generated > int(config["resource_caps"]["max_generated_artifact_bytes"]):
        raise OSError("combined Phase 12c-C train/selection/model artifact cap exceeded")
    print(json.dumps({
        "status": "phase12cC_student_fitting_complete",
        "fit_wall_seconds": manifest["fit_wall_seconds"],
        "fit_artifact_bytes": manifest["artifact_bytes"] + manifest_path.stat().st_size,
        "total_generated_artifact_bytes": total_generated,
        "training_queries": result["training_queries"],
        "selection_queries": result["selection_queries"],
        "selected_models": result["selected_models"],
        "offline_gate_read": False,
        "development_accessed": False,
        "reserve_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
