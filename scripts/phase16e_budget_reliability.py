from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase16e_budget_reliability import (  # noqa: E402
    build_layout_audit,
    run_locked,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "run"))
    args = parser.parse_args()
    result = build_layout_audit(ROOT) if args.stage == "audit" else run_locked(ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
