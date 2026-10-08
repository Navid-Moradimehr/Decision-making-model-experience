"""Run only the locked Phase14D independent offline regret gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase14e import run_offline_gate


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    metrics = run_offline_gate(args.root.resolve())
    print(json.dumps({
        "status": metrics["status"],
        "gate_passed": metrics["gate_passed"],
        "point_improvement": metrics["point_improvement_over_geometry"],
        "bootstrap_lower_95": metrics["paired_layout_bootstrap"]["lower_95"],
        "bootstrap_upper_95": metrics["paired_layout_bootstrap"]["upper_95"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
