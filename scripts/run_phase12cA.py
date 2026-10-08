"""Plan or execute only the authorized Phase 12c-A comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12c import Phase12cAuthorizationError, run_phase12c_a


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase12cA_locked.yaml")
    parser.add_argument("--authorization-lock", type=Path, default=ROOT / "experiments/phase12cA_authorization_v4.lock.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps({
            "mode": "plan_only_no_outcomes_generated",
            "phase": config["phase"], "seed_block": config["seed_block"],
            "methods": config["methods"],
            "expected_episode_records": config["seed_block"]["count"] * len(config["methods"]),
            "caps": config["compute_caps"], "training_authorized": False,
            "reserve_accessed": False,
        }, indent=2))
        return 0
    token = config["runner_contract"]["required_cli_confirmation"]
    if args.confirmation != token:
        print("Refusing execution: exact confirmation token required.", file=sys.stderr)
        return 2
    output_path = args.output_dir or ROOT / config["runner_contract"]["output_path"]
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    lock_path = args.authorization_lock if args.authorization_lock.is_absolute() else ROOT / args.authorization_lock
    try:
        manifest = run_phase12c_a(
            ROOT, config_path, lock_path, output_path, confirmed=True
        )
    except Phase12cAuthorizationError as exc:
        print(f"Refusing execution: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "completed_phase12c_a_only" else 3


if __name__ == "__main__":
    raise SystemExit(main())
