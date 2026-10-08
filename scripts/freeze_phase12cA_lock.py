"""Create the source-pinned authorization lock for approved Phase 12c-A only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase12cA_locked.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/phase12cA_authorization_v4.lock.json")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config_path, output = config_path.resolve(), output.resolve()
    if output.exists():
        parser.error("authorization lock already exists; refusing overwrite")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_path = (ROOT / config["layout_audit"]["path"]).resolve()
    if not audit_path.is_file():
        parser.error("run the hash-only audit first")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("pairwise_disjoint") or audit.get("reserve_generated_or_accessed") is not False:
        parser.error("layout audit failed integrity checks")
    if digest(ROOT / "src/consequence_lab/corridor_tasks.py") != config["task"]["generator_sha256"]:
        parser.error("task generator hash differs from locked protocol")
    if digest(audit_path) != config["layout_audit"]["sha256"]:
        parser.error("layout audit digest differs from locked protocol")
    source_sha256 = {
        relative: digest(ROOT / relative)
        for relative in config["source_files"]
        if relative != "configs/phase12cA_locked.yaml"
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    lock = {
        "format": "phase12c_a_execution_lock_v1",
        "authorization_status": "authorized",
        "stage": "phase12c_a_teacher_qualification_only",
        "execution_authorized": True,
        "performance_data_authorized": True,
        "training_authorized": False,
        "approval_reference": config["authorization"]["approval_reference"],
        "approved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_git_commit": git,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": digest(config_path),
        "seed_block": config["seed_block"],
        "layout_audit_path": config["layout_audit"]["path"],
        "layout_audit_sha256": digest(audit_path),
        "caps": {
            "wall_seconds": config["compute_caps"]["total_wall_seconds"],
            "artifact_bytes": config["compute_caps"]["generated_artifact_bytes"],
        },
        "source_sha256": source_sha256,
        "reserve_accessed": False,
        "training_performed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
