"""Run exactly one gated Phase 13A block after source/hash lock verification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

from consequence_lab.phase13 import run_phase13a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", choices=("development", "independent"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("pass --execute to collect the explicitly authorized Phase 13A outcomes")
    root = args.root.resolve()
    config = yaml.safe_load((root / "configs/phase13.yaml").read_text(encoding="utf-8"))
    output = root / config["phase13A"]["outputs"][args.stage]
    lock = root / config["phase13A"]["provenance"]["authorization_lock"]
    audit = root / config["phase13A"]["provenance"]["layout_audit"]
    try:
        result = run_phase13a(
            root,
            stage=args.stage,
            output_path=output,
            lock_path=lock,
            audit_path=audit,
        )
    except Exception as exc:
        partial_episodes = output / "episodes.jsonl"
        partial_manifest = output / "manifest.json"
        if output.is_dir() and partial_episodes.is_file() and not partial_manifest.exists():
            lines = partial_episodes.read_text(encoding="utf-8").splitlines()
            lock_digest = hashlib.sha256(lock.read_bytes()).hexdigest().upper() if lock.is_file() else None
            partial_manifest.write_text(json.dumps({
                "experiment": f"phase13A_{args.stage}",
                "status": "aborted_or_preflight_failed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": str(exc),
                "episode_artifact": "episodes.jsonl",
                "episode_records": len(lines),
                "source_lock_sha256": lock_digest,
                "reserve_accessed": False,
                "student_training_performed": False,
                "dagger_performed": False,
                "world_model_used": False,
            }, indent=2, sort_keys=True), encoding="utf-8")
        raise SystemExit(f"Phase 13A stopped: {exc}") from exc
    print(json.dumps({
        "status": result["status"],
        "stage": result["stage"],
        "episodes": result["episode_records"],
        "selected_brake_safety_margin": result["selected_brake_safety_margin"],
        "headroom_gate": result["headroom_gate"],
        "reserve_accessed": result["reserve_accessed"],
        "output": output.relative_to(root).as_posix(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
