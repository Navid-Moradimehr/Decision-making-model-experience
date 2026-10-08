"""Run the frozen Phase-12 student comparison after the data audit gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_training import train_phase12

LOCK = ROOT / "experiments" / "phase12_training.lock.json"
CONFIG = ROOT / "configs" / "phase12_teacher_student.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output", default="artifacts/runs/phase12_training_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps({
            "status": "plan_only_no_training",
            "variants": ["value", "hard", "soft_0.05", "soft_0.15", "soft_0.30", "expected_regret"],
            "fit_seeds": [1201, 1202, 1203],
            "reserve_accessed": False,
        }, indent=2))
        return 0
    if args.confirmation != "TRAIN_PHASE12_FROZEN_STUDENTS":
        raise SystemExit("training requires --confirmation TRAIN_PHASE12_FROZEN_STUDENTS")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("reserve_accessed") is not False or lock.get("training_authorized") is not True:
        raise SystemExit("invalid training execution lock")
    for relpath, expected in lock["source_sha256"].items():
        path = (ROOT / relpath).resolve()
        if ROOT not in path.parents or sha256(path) != expected:
            raise SystemExit(f"frozen training source hash mismatch: {relpath}")
    data_audit = ROOT / "reports" / "phase12_teacher_data_audit.json"
    if sha256(data_audit) != lock["teacher_data_audit_sha256"]:
        raise SystemExit("teacher-data audit hash differs from frozen training lock")
    audit = json.loads(data_audit.read_text(encoding="utf-8"))
    if audit["status"] != "passed" or audit["reserve_accessed"]:
        raise SystemExit("teacher-data audit failed or reserve was accessed")
    for split, block in (("train", audit["train"]), ("validation", audit["validation"])):
        for filename, data in block["files"].items():
            path = ROOT / "artifacts" / "runs" / f"phase12_{'train_teacher' if split == 'train' else 'validation_teacher'}_v1" / filename
            if sha256(path) != data["sha256"]:
                raise SystemExit(f"teacher data hash mismatch: {path.name}")
    for relpath, expected in lock["dataset_sha256"].items():
        path = (ROOT / relpath).resolve()
        if ROOT not in path.parents or sha256(path) != expected:
            raise SystemExit(f"frozen dataset artifact hash mismatch: {relpath}")
    output = (ROOT / args.output).resolve()
    artifacts_root = (ROOT / "artifacts" / "runs").resolve()
    if artifacts_root not in output.parents:
        raise SystemExit("output must remain under artifacts/runs")
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    result = train_phase12(
        ROOT / "artifacts" / "runs" / "phase12_train_teacher_v1",
        ROOT / "artifacts" / "runs" / "phase12_validation_teacher_v1",
        output,
        config=config,
        device_name=args.device,
    )
    result["training_lock_sha256"] = sha256(LOCK)
    result["data_audit_sha256"] = sha256(data_audit)
    result["config_sha256"] = sha256(CONFIG)
    result["reserve_accessed"] = False
    result_path = output / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed_training_comparison",
        "device": result["device"],
        "cuda_available": result["cuda_available"],
        "train_samples": result["train_samples"],
        "validation_samples": result["validation_samples"],
        "runs": len(result["all_runs"]),
        "selected": result["selected"],
        "offline_competence_gate": result["offline_competence_gate"],
        "training_wall_seconds": result["training_wall_seconds"],
        "results_path": str(result_path),
        "reserve_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
