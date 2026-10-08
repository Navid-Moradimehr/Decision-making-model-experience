"""Run only the committed, locked Phase14F conditional development comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase14f import run_closed_loop


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    metrics = run_closed_loop(args.root.resolve())
    print(json.dumps({
        "status": metrics["status"],
        "gate_passed": metrics["gate_passed"],
        "student_success": metrics["student"]["safe_success_rate"],
        "geometry_success": metrics["geometry_waypoint"]["safe_success_rate"],
        "exact_shooting_success": metrics["exact_random_shooting"]["safe_success_rate"],
        "student_minus_geometry_gain": metrics["student_minus_geometry_safe_success_point_gain"],
        "student_to_shooting_latency_ratio": metrics["student_to_shooting_median_decision_latency_ratio"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
