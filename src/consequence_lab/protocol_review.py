"""Read-only integrity checks for non-executable Phase 11 review snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def verify_phase11_review_snapshot(root: Path) -> dict[str, Any]:
    """Verify pinned files and draft/no-authorization invariants.

    This function never evaluates a scenario or changes protocol state. A
    successful result establishes snapshot consistency only, not permission to
    collect performance data or train models.
    """
    root = Path(root).resolve()
    lock_path = root / "experiments" / "phase11_corridor_protocol_review.lock.json"
    config_path = root / "configs" / "phase11_corridor_development_draft.yaml"
    errors: list[str] = []
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return {
            "valid": False,
            "files_verified": 0,
            "execution_authorized": False,
            "performance_data_authorized": False,
            "training_authorized": False,
            "errors": [f"could not read snapshot/config: {exc}"],
        }
    if not isinstance(lock, dict) or not isinstance(config, dict):
        return {
            "valid": False,
            "files_verified": 0,
            "execution_authorized": False,
            "performance_data_authorized": False,
            "training_authorized": False,
            "errors": ["snapshot and config roots must be mappings"],
        }

    if lock.get("experiment") != "phase11_corridor_baseline_headroom_screen":
        errors.append("unexpected experiment identifier")
    if lock.get("status") != "protocol_review_snapshot_no_execution":
        errors.append("snapshot status is not review-only")
    if lock.get("protocol_status") != "draft_for_review":
        errors.append("snapshot protocol is not a draft")
    if config.get("protocol_status") != "draft_for_review":
        errors.append("config protocol is not a draft")

    auth_fields = (
        "execution_authorized",
        "performance_data_authorized",
        "training_authorized",
    )
    if any(
        lock.get(field) is True or config.get(field) is True
        for field in auth_fields
    ):
        errors.append("review snapshot/config must not authorize execution or training")

    pins = lock.get("sha256")
    if not isinstance(pins, dict) or not pins:
        errors.append("snapshot has no scoped SHA-256 pins")
        pins = {}
    files_verified = 0
    for relative_name, expected_hash in pins.items():
        candidate = (root / relative_name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"pinned path escapes project root: {relative_name}")
            continue
        if not candidate.is_file():
            errors.append(f"pinned file is missing: {relative_name}")
            continue
        actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest().upper()
        if actual_hash != str(expected_hash).upper():
            errors.append(f"pinned hash mismatch: {relative_name}")
            continue
        files_verified += 1

    return {
        "valid": not errors,
        "files_verified": files_verified,
        "execution_authorized": False,
        "performance_data_authorized": False,
        "training_authorized": False,
        "errors": errors,
    }
