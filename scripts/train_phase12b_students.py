"""Train the frozen Phase-12b comparison using only old train + selection data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_training import load_phase12_split
from consequence_lab.phase12b import merge_training_sources, train_phase12b

CONFIG = ROOT / "configs" / "phase12b_action_aligned.yaml"
LOCK = ROOT / "experiments" / "phase12b_training.lock.json"
COLLECTION = ROOT / "artifacts" / "runs" / "phase12b_model_selection_v1"
OUTPUT = ROOT / "artifacts" / "runs" / "phase12b_training_v1"
AUDIT = ROOT / "artifacts" / "runs" / "phase12b_layout_audit_v2.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.plan_only:
        print(json.dumps({
            "status": "plan_only_no_training",
            "variants": [item["id"] for item in config["models"]["variants"]],
            "fit_seeds": config["models"]["fit"]["seeds"],
            "fit_source_layouts": 160,
            "selection_layouts": 40,
            "offline_gate_read": False,
            "reserve_accessed": False,
        }, indent=2))
        return 0
    if args.confirmation != "TRAIN_PHASE12B_FROZEN_COMPARISON":
        raise SystemExit("training requires --confirmation TRAIN_PHASE12B_FROZEN_COMPARISON")
    if not LOCK.is_file() or not COLLECTION.is_dir():
        raise SystemExit("frozen Phase-12b training lock and selection data are required")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("training_authorized") is not True or lock.get("offline_gate_read") is not False:
        raise SystemExit("training lock must authorize fitting but prohibit offline-gate access")
    if lock.get("reserve_accessed") is not False:
        raise SystemExit("invalid lock: reserve is not sealed")
    for relpath, expected in lock["source_sha256"].items():
        path = (ROOT / relpath).resolve()
        if ROOT not in path.parents or not path.is_file() or digest(path) != expected:
            raise SystemExit(f"frozen Phase-12b training source mismatch: {relpath}")
    for relpath, expected in lock["training_data_sha256"].items():
        path = (ROOT / relpath).resolve()
        if ROOT not in path.parents or not path.is_file() or digest(path) != expected:
            raise SystemExit(f"frozen Phase-12b training data mismatch: {relpath}")
    if digest(AUDIT) != lock["layout_audit_sha256"]:
        raise SystemExit("Phase-12b layout audit changed after lock")
    manifest_path = COLLECTION / "manifest.json"
    if digest(manifest_path) != lock["selection_manifest_sha256"]:
        raise SystemExit("model-selection collection manifest changed after lock")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("authorized_split") != "model_selection" or manifest.get("offline_gate_labels_collected") is not False:
        raise SystemExit("unexpected data split or offline-gate access in selection manifest")
    if manifest.get("reserve_accessed") is not False or manifest.get("closed_loop_development_accessed") is not False:
        raise SystemExit("selection manifest indicates forbidden split access")
    if manifest.get("config_sha256") != lock.get("collection_config_sha256"):
        raise SystemExit("selection data were not collected under the pinned collection config")
    layout_audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if manifest.get("layout_sha256") != layout_audit["new_blocks"]["model_selection"]["layout_sha256"]:
        raise SystemExit("selection collection layout hashes differ from the pinned split audit")
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")

    old_train = merge_training_sources(
        ROOT / "artifacts" / "runs" / "phase12_train_teacher_v1",
        ROOT / "artifacts" / "runs" / "phase12_validation_teacher_v1",
    )
    selection = load_phase12_split(COLLECTION)
    if len(set(manifest["layout_sha256"])) != 40:
        raise SystemExit("model-selection collection has an incomplete/non-unique layout set")
    result = train_phase12b(
        old_train, selection, config=config, output_dir=OUTPUT, device_name=args.device,
    )
    result.update({
        "training_lock_sha256": digest(LOCK),
        "selection_manifest_sha256": digest(manifest_path),
        "reserve_accessed": False,
        "offline_gate_read": False,
    })
    results_path = OUTPUT / "selection_results.json"
    results_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "phase12b_model_selection_complete",
        "device": result["device"],
        "cuda_available": result["cuda_available"],
        "training_layouts": result["training_layouts"],
        "training_queries": result["training_queries"],
        "selection_layouts": result["selection_layouts"],
        "selection_queries": result["selection_queries"],
        "runs": len(result["runs"]),
        "selected": result["selected"],
        "offline_gate_read": False,
        "reserve_accessed": False,
        "results_path": str(results_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
