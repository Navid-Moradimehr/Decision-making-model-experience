"""Freeze a clean source/data lock for the Phase14A diagnostic only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

from consequence_lab.phase14 import (
    CONFIG_PATH, PROTOCOL_PATH, SOURCE_CODE, SOURCE_DIR, SOURCE_FILES,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    lock_path = root / config["provenance"]["source_lock"]
    audit_path = root / config["provenance"]["source_audit"]
    if lock_path.exists():
        raise SystemExit(f"refusing to overwrite source lock: {lock_path}")
    if not audit_path.is_file():
        raise SystemExit("Phase14 training-source replay audit must pass before locking")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != "phase14_training_source_audit_v2" or audit.get("status") != "exact_training_archive_replay_passed_no_phase14_labels":
        raise SystemExit("training-source replay audit has not passed")
    if audit.get("reserve_accessed") is not False or audit.get("phase12_validation_opened") is not False:
        raise SystemExit("forbidden split was accessed")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if dirty:
        raise SystemExit("commit Phase14 source/config/tests before freezing the lock")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if audit.get("generation_git_commit") != commit:
        raise SystemExit("source audit must be generated from the exact clean source commit")
    source_dir = root / SOURCE_DIR
    archive_hashes = {name: sha256_file(source_dir / name) for name in SOURCE_FILES}
    if archive_hashes != audit.get("training_archive_sha256"):
        raise SystemExit("Phase12 training source changed after Phase14 source audit")
    lock = {
        "schema": "phase14_source_lock_v2",
        "status": "authorized_phase14a_audit_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_reference": "user instruction opening separate Phase 14 on 2026-09-28",
        "source_commit": commit,
        "config_sha256": sha256_file(root / CONFIG_PATH),
        "protocol_sha256": sha256_file(root / PROTOCOL_PATH),
        "source_code_sha256": {name: sha256_file(root / name) for name in SOURCE_CODE},
        "source_audit_path": config["provenance"]["source_audit"],
        "source_audit_sha256": sha256_file(audit_path),
        "phase12_training_archive_sha256": archive_hashes,
        "selected_state_keys": audit["diagnostic_state_keys"],
        "training_layout_count": audit["training_layout_count"],
        "training_state_count": audit["training_decision_state_count"],
        "authorized_stage": "phase14A_32_state_teacher_audit_only",
        "reserve_access": False,
        "model_training": False,
        "resources": {"device": "cpu", "wall_seconds": 10800, "generated_bytes": 20971520},
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"lock": str(lock_path), "source_commit": commit,
                      "source_audit_sha256": lock["source_audit_sha256"],
                      "authorized_stage": lock["authorized_stage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
