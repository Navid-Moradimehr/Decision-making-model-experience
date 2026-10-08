"""Execute only the explicitly authorized Phase 12c-B repeated-bank audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12c_b import run_phase12c_b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase12cB_locked.yaml")
    parser.add_argument("--authorization-lock", type=Path, default=ROOT / "experiments/phase12cB_authorization.lock.json")
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sample_path = args.sample_manifest or Path(config["state_sampling"]["sample_manifest"])
    sample_path = sample_path if sample_path.is_absolute() else ROOT / sample_path
    output_path = args.output_dir or Path(config["runner_contract"]["output_path"])
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    lock_path = args.authorization_lock if args.authorization_lock.is_absolute() else ROOT / args.authorization_lock
    lock_path = lock_path.resolve()
    if not args.execute:
        print(json.dumps({
            "mode": "plan_only_no_repeated_banks_generated",
            "phase": config["phase"],
            "sample_manifest": str(sample_path),
            "states": config["state_sampling"]["total_states"],
            "layouts": config["state_sampling"]["total_layouts"],
            "banks_per_state": config["bank_sampling"]["banks_per_state"],
            "expected_records": config["runner_contract"]["expected_bank_records"],
            "caps": config["compute_caps"],
            "training_authorized": False,
            "reserve_accessed": False,
        }, indent=2))
        return 0
    if args.confirmation != config["runner_contract"]["required_cli_confirmation"]:
        print("Refusing execution: exact Phase 12c-B confirmation required.", file=sys.stderr)
        return 2
    try:
        manifest = run_phase12c_b(
            ROOT, config_path, lock_path, sample_path.resolve(), output_path.resolve(),
            confirmed=True,
        )
    except (OSError, ValueError, KeyError, TimeoutError) as exc:
        print(f"Phase 12c-B stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    print(json.dumps({
        key: manifest[key] for key in (
            "status", "completed_bank_records", "completed_state_count",
            "unique_proposal_bank_seeds", "elapsed_wall_seconds", "artifact_bytes",
            "reserve_accessed", "student_training_performed", "dagger_performed",
        )
    }, indent=2))
    return 0 if manifest["status"] == "completed_phase12c_b_only" else 3


if __name__ == "__main__":
    raise SystemExit(main())
