"""Freeze the Phase 12c-D1 fitting-only layout fold without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase12cD.yaml"
PROTOCOL = "reports/phase12cD_protocol.md"
TRAINER = "scripts/run_phase12cD1.py"
PINNED_SOURCES = [
    CONFIG,
    PROTOCOL,
    "scripts/prepare_phase12cD1.py",
    TRAINER,
    "src/consequence_lab/phase12c_c_training.py",
    "src/consequence_lab/phase12_model.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "tests/test_phase12c_c.py",
    "tests/test_phase12cD.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output", default="experiments/phase12cD1_fold_v3.lock.json")
    args = parser.parse_args()
    if args.confirmation != "FREEZE_PHASE12CD1_FITTING_FOLD":
        raise SystemExit("fold lock requires --confirmation FREEZE_PHASE12CD1_FITTING_FOLD")

    config_path = ROOT / CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage = config["stage_D1"]
    dataset = ROOT / stage["input_dataset"]
    manifest_path = dataset / "manifest.json"
    if digest(manifest_path) != stage["input_manifest_sha256"]:
        raise SystemExit("fitting dataset manifest hash differs from the frozen protocol")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != "archived_training" or manifest.get("layout_count") != 160:
        raise SystemExit("D1 input must be exactly the archived fitting-only dataset")
    if manifest.get("reserve_accessed") is not False or manifest.get("offline_gate_labels_collected") is not False:
        raise SystemExit("fitting dataset manifest violates the authorized scope")
    for relative, entry in manifest["files"].items():
        if digest(dataset / relative) != entry["sha256"]:
            raise SystemExit(f"fitting dataset file digest mismatch: {relative}")

    metadata = json.loads((dataset / "sample_metadata.json").read_text(encoding="utf-8"))
    layouts = sorted({str(row["layout_sha256"]) for row in metadata})
    if len(layouts) != 160:
        raise SystemExit(f"expected 160 fitting layouts, observed {len(layouts)}")
    namespace = stage["fold"]["namespace"]
    ranked = sorted(layouts, key=lambda item: (hashlib.sha256(f"{namespace}:{item}".encode("ascii")).hexdigest(), item))
    n_val = int(stage["fold"]["internal_validation_layouts"])
    validation = ranked[:n_val]
    training = ranked[n_val:]
    if len(training) != 128 or len(validation) != 32 or set(training) & set(validation):
        raise SystemExit("invalid deterministic D1 layout fold")

    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / "experiments").resolve()) or output.exists():
        raise SystemExit("fold lock must be a new file under experiments/")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"could not record git commit: {exc}") from exc
    lock = {
        "schema": "phase12cD1_layout_fold_v3",
        "authorization": "User authorized the staged Phase 12c-D program on 2026-09-28; D1 is first and later stages remain gate-conditional.",
        "git_commit": commit,
        "config_sha256": digest(config_path),
        "training_manifest_sha256": digest(manifest_path),
        "training_data_file_sha256": {name: digest(dataset / name) for name in manifest["files"]},
        "source_sha256": {relative: digest(ROOT / relative) for relative in PINNED_SOURCES},
        "fold_namespace": namespace,
        "training_layout_sha256": training,
        "internal_validation_layout_sha256": validation,
        "training_layout_count": len(training),
        "internal_validation_layout_count": len(validation),
        "source_query_count": len(metadata),
        "old_phase12cC_selection_gate_read": False,
        "new_rollouts_or_labels": False,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "phase12cD1_fold_frozen",
        "output": str(output.relative_to(ROOT)),
        "training_layouts": lock["training_layout_count"],
        "validation_layouts": lock["internal_validation_layout_count"],
        "source_queries": len(metadata),
        "outcomes_read": False,
        "old_phase12cC_selection_gate_read": False,
        "reserve_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
