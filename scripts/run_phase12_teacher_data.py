"""Fail-closed Phase-12 layout audit / exact teacher-data runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_data import audit_phase12_layout_splits, collect_teacher_split

CONFIG = ROOT / "configs" / "phase12_teacher_student.yaml"
LOCK = ROOT / "experiments" / "phase12_teacher_data.lock.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_frozen():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for relpath, expected in lock["source_sha256"].items():
        path = (ROOT / relpath).resolve()
        if ROOT not in path.parents or digest(path) != expected:
            raise SystemExit(f"frozen source/config hash mismatch: {relpath}")
    if lock.get("reserve_accessed") is not False or lock.get("training_authorized") is not True:
        raise SystemExit("invalid Phase 12 execution lock")
    return config, lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "audit-layouts", "collect"), default="plan")
    parser.add_argument("--split", choices=("train_teacher", "validation_teacher"))
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.mode == "plan":
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        print(json.dumps({
            "status": "plan_only_no_layout_generation_or_outcomes",
            "phase": 12,
            "splits": config["seed_splits"],
            "reserve_accessed": False,
            "training_performed": False,
        }, indent=2))
        return 0
    config, lock = load_frozen()
    if args.mode == "audit-layouts":
        if args.confirmation != "AUDIT_PHASE12_LAYOUT_HASHES":
            raise SystemExit("layout audit requires --confirmation AUDIT_PHASE12_LAYOUT_HASHES")
        result = audit_phase12_layout_splits(config)
        out = ROOT / "artifacts" / "runs" / "phase12_layout_audit.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "layout_audit_passed", "output": str(out), **result}, indent=2))
        return 0
    if args.split not in {"train_teacher", "validation_teacher"}:
        raise SystemExit("--split is required for collection")
    expected_token = f"COLLECT_PHASE12_{args.split.upper()}"
    if args.confirmation != expected_token:
        raise SystemExit(f"collection requires --confirmation {expected_token}")
    out = Path(args.output).resolve() if args.output else (
        ROOT / "artifacts" / "runs" / f"phase12_{args.split}_v1"
    ).resolve()
    artifacts_root = (ROOT / "artifacts" / "runs").resolve()
    if artifacts_root not in out.parents:
        raise SystemExit("output path must remain under artifacts/runs")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    # The prior hash-only audit is a required provenance input to collection.
    layout_manifest_path = ROOT / "artifacts" / "runs" / "phase12_layout_audit.json"
    if not layout_manifest_path.exists():
        raise SystemExit("run and pass --mode audit-layouts before teacher collection")
    layout_manifest = json.loads(layout_manifest_path.read_text(encoding="utf-8"))
    split_manifest = layout_manifest["blocks"].get(args.split)
    if not split_manifest:
        raise SystemExit("requested split missing from frozen layout manifest")
    block = config["seed_splits"][args.split]
    seeds = list(range(block["start"], block["start"] + block["count"]))
    if seeds != split_manifest["seeds"]:
        raise SystemExit("layout manifest seed scope mismatch")
    result = collect_teacher_split(
        seeds, out,
        artifact_cap_bytes=config["compute_caps"]["teacher_dataset_artifact_bytes"],
        wall_cap_seconds=config["compute_caps"]["teacher_dataset_wall_seconds"],
    )
    manifest = {
        "experiment": f"phase12_{args.split}_v1",
        "config_sha256": digest(CONFIG),
        "execution_lock_sha256": digest(LOCK),
        "source_sha256": lock["source_sha256"],
        "layout_manifest_sha256": digest(layout_manifest_path),
        "reserve_accessed": False,
        "training_performed": False,
        **result,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
