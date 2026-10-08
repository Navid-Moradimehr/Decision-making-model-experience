"""Freeze source/config/layout-audit hashes for the approved Phase 13A stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

from consequence_lab.phase13 import (
    ROOT_RELATIVE,
    file_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--approval-reference",
        required=True,
        help="brief reference to the user's Phase 13 implementation/execution authorization",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / "configs/phase13.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = root / config["phase13A"]["provenance"]["authorization_lock"]
    audit_path = root / config["phase13A"]["provenance"]["layout_audit"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing lock: {output}")
    if not audit_path.is_file():
        raise SystemExit("run the hash-only Phase 13 layout audit first")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if status:
        raise SystemExit("commit all Phase 13 source/config/tests before freezing authorization")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "hash_only_passed_no_outcomes":
        raise SystemExit("layout audit did not pass without outcomes")
    if audit.get("reserve_performance_outcomes_read") is not False:
        raise SystemExit("audit did not attest reserve outcomes remained unopened")
    if not audit.get("all_new_blocks_pairwise_disjoint"):
        raise SystemExit("new layout blocks are not pairwise disjoint")
    if audit.get("generation_git_commit") != head:
        raise SystemExit("layout audit was not generated from the current clean source commit")
    expected_audit_sources = {
        relative: file_sha256(root / relative)
        for relative in (
            "configs/phase13.yaml",
            "scripts/audit_phase13_layouts.py",
            "src/consequence_lab/phase13.py",
            "src/consequence_lab/corridor_tasks.py",
            "src/consequence_lab/inertial_corridor.py",
        )
    }
    if audit.get("generation_source_sha256") != expected_audit_sources:
        raise SystemExit("layout audit source/config hashes do not match current source")
    source_hashes = {}
    for relative in ROOT_RELATIVE:
        source = root / relative
        if not source.is_file():
            raise SystemExit(f"locked source is missing: {relative}")
        source_hashes[relative] = file_sha256(source)
    lock = {
        "schema": "phase13a_authorization_lock_v2",
        "authorization_status": "authorized_with_preregistered_gates",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_reference": args.approval_reference,
        "base_git_commit": head,
        "config_sha256": file_sha256(config_path),
        "layout_audit_path": config["phase13A"]["provenance"]["layout_audit"],
        "layout_audit_sha256": file_sha256(audit_path),
        "source_sha256": source_hashes,
        "authorized_stages": {
            "phase13A_development": {"start": 143000, "count": 40},
            "phase13A_independent_conditional": {"start": 190000, "count": 100},
        },
        "phase13a_execution_authorized": True,
        "independent_stage_conditional_on_development_completion": True,
        "reserve_access_authorized": False,
        "student_training_authorized": False,
        "dagger_authorized": False,
        "world_model_or_jepa_authorized": False,
        "caps_per_stage": {
            "wall_seconds": config["integrity"]["cpu_wall_cap_seconds_per_stage"],
            "generated_bytes": config["integrity"]["generated_artifact_cap_bytes_per_stage"],
            "device": "cpu",
        },
    }
    output.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"lock_created={output.relative_to(root)} source_commit={head} "
        f"pinned_sources={len(source_hashes)} reserve_authorized=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
