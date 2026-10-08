"""Run the immutable Phase17C fresh-layout exact-planner evaluation."""
from __future__ import annotations

import argparse
import json

from consequence_lab.phase17_proposal_guidance import freeze_fresh_execution_lock, run_fresh_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-lock", "run"))
    args = parser.parse_args()
    result = freeze_fresh_execution_lock() if args.command == "freeze-lock" else run_fresh_evaluation()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
