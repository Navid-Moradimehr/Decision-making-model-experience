"""Create the frozen 160-state fitting-only sample manifest for Phase 12c-B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12c_b import sample_manifest_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase12cB_locked.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    audit_path = (ROOT / config["source_data"]["replay_audit"]).resolve()
    if not audit_path.is_file():
        raise SystemExit("fitting-data exact-replay audit is missing")
    audit_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest().upper()
    if audit_digest != config["source_data"]["replay_audit_sha256"]:
        raise SystemExit("fitting-data replay-audit hash mismatch")
    payload = sample_manifest_payload(ROOT, config)
    payload["source_data_replay_audit_sha256"] = audit_digest
    payload["source_training_lock_sha256"] = config["source_data"]["execution_lock_sha256"]
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = args.output or Path(config["state_sampling"]["sample_manifest"])
    output = output if output.is_absolute() else ROOT / output
    output = output.resolve()
    if output.exists() or ROOT not in output.parents:
        raise SystemExit("refusing overwrite or output outside repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest().upper()
    print(json.dumps({
        "status": "sample_manifest_frozen_no_new_teacher_labels",
        "sample_manifest": output.relative_to(ROOT).as_posix(),
        "sample_manifest_sha256": digest,
        "sample_count": payload["sample_count"],
        "layout_count": payload["layout_count"],
        "source_split_layout_counts": payload["source_split_layout_counts"],
        "progress_speed_strata_counts": payload["progress_speed_strata_counts"],
        "source_data_replay_audit_sha256": audit_digest,
        "reserve_accessed": False,
        "repeated_banks_generated": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
