"""Plan or execute Phase 11's authorized 100-layout development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase11_development import run_development_screen
from consequence_lab.phase11_runner import Phase11AuthorizationError


DEFAULT_CONFIG = ROOT / "configs" / "phase11_development_draft.yaml"


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--authorization-lock", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()

    config_path = _rooted(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    block = config["seed_block"]
    frozen = config["frozen_comparison"]
    if not args.execute:
        print(json.dumps({
            "mode": "plan_only_no_scenarios_generated",
            "stage": config["stage"],
            "protocol_status": config["protocol_status"],
            "seed_start": block["start"],
            "seed_count": block["count"],
            "frozen_methods": [
                {"method": frozen["comparator_method"], "role": "cheap_comparator",
                 "parameters": frozen["comparator_parameters"]},
                {"method": frozen["primary_method"], "role": "planner",
                 "parameters": frozen["primary_parameters"]},
            ],
            "expected_episode_records": block["count"] * 2,
            "wall_cap_seconds": config["compute_caps"]["total_wall_seconds"],
            "artifact_cap_bytes": config["compute_caps"]["generated_artifact_bytes"],
            "execution_authorized": False,
            "performance_data_authorized": False,
            "training_authorized": False,
            "reserve_accessed": False,
        }, indent=2))
        return 0

    if args.authorization_lock is None or args.output_dir is None:
        parser.error("--execute requires --authorization-lock and --output-dir")
    token = config["runner_contract"]["required_cli_confirmation"]
    if args.confirmation != token:
        print("Refusing execution: exact --confirmation token is required.", file=sys.stderr)
        return 2
    try:
        manifest = run_development_screen(
            ROOT,
            config_path,
            _rooted(args.authorization_lock),
            _rooted(args.output_dir),
            confirmed=True,
        )
    except Phase11AuthorizationError as exc:
        print(f"Refusing execution: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "completed_development_screen_only" else 3


if __name__ == "__main__":
    raise SystemExit(main())
