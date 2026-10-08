"""Run the separately locked, one-shot Phase14G trace-attribution diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase14g_runner import run_trace_study


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    metrics = run_trace_study(args.root.resolve())
    print(json.dumps({
        "status": metrics["status"],
        "student_success": metrics["student"]["safe_success_rate"],
        "geometry_success": metrics["geometry_waypoint"]["safe_success_rate"],
        "exact_shooting_success": metrics["exact_random_shooting"]["safe_success_rate"],
        "student_state_trace_count": metrics["student_state_trace_count"],
        "mean_selected_teacher_regret": metrics["mean_selected_teacher_regret"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
