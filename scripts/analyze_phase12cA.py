"""Validate and summarize only the completed Phase 12c-A development block."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.corridor_evaluation import (
    paired_outcome_rate_interval,
    paired_success_gain_interval,
    summarize_corridor_results,
)
from consequence_lab.phase12c import Phase12cEpisodeResult


def analyze(run_dir: Path, bootstrap_seed: int = 160011) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    expected_seeds = list(range(160000, 160100))
    expected_methods = {"geometry_waypoint", "random_shooting", "balanced_shooting"}
    if manifest.get("status") != "completed_phase12c_a_only":
        raise ValueError("run did not complete the authorized Phase 12c-A stage")
    if manifest.get("training_performed") is not False or manifest.get("reserve_accessed") is not False:
        raise ValueError("run manifest violates Phase 12c-A scope")
    if manifest.get("seeds") != expected_seeds or len(rows) != 300:
        raise ValueError("run is incomplete or contains an unexpected seed block")
    results = [Phase12cEpisodeResult(**row["result"]) for row in rows]
    if {row["method"] for row in rows} != expected_methods:
        raise ValueError("unexpected method set")
    for seed in expected_seeds:
        matching = [row for row in rows if row["seed"] == seed]
        if len(matching) != 3 or {row["method"] for row in matching} != expected_methods:
            raise ValueError(f"incomplete/duplicate paired layout {seed}")
    summaries = {s.method: asdict(s) for s in summarize_corridor_results(results)}
    bg = paired_success_gain_interval(results, "balanced_shooting", "geometry_waypoint", bootstrap_seed=bootstrap_seed)
    bo = paired_success_gain_interval(results, "balanced_shooting", "random_shooting", bootstrap_seed=bootstrap_seed)
    collision = paired_outcome_rate_interval(
        results, "balanced_shooting", "random_shooting", event="collision", bootstrap_seed=bootstrap_seed
    )
    balanced_rows = [row for row in rows if row["method"] == "balanced_shooting"]
    successes = sum(row["result"]["outcome"] == "success" for row in balanced_rows)
    collisions = sum(row["result"]["outcome"] == "collision" for row in balanced_rows)
    balanced_queries = sum(row["result"]["shooting_queries"] for row in balanced_rows)
    total_candidates = sum(row["result"]["shooting_candidate_evaluations"] for row in balanced_rows)
    total_ties = sum(row["result"]["shooting_objective_tie_candidates"] for row in balanced_rows)
    gates = {
        "balanced_successes_at_least_90": successes >= 90,
        "balanced_gain_point_at_least_20pp": bg.paired_rate_difference >= 0.20,
        "balanced_gain_lower_ci_above_10pp": bg.lower_95 > 0.10,
        "balanced_noninferiority_lower_ci_above_minus_10pp": bo.lower_95 > -0.10,
        "balanced_collision_count_at_most_10": collisions <= 10,
        "balanced_collision_upper_ci_at_most_plus_10pp": collision.upper_95 <= 0.10,
        "complete_300_records": len(rows) == 300,
        "training_not_performed": manifest["training_performed"] is False,
        "reserve_not_accessed": manifest["reserve_accessed"] is False,
    }
    return {
        "status": "analyzed_phase12c_a_only",
        "summaries": summaries,
        "paired_success_gain_balanced_minus_geometry": asdict(bg),
        "paired_success_gain_balanced_minus_original": asdict(bo),
        "paired_collision_difference_balanced_minus_original": asdict(collision),
        "balanced_teacher_diagnostics": {
            "successes": successes, "collisions": collisions,
            "shooting_queries": balanced_queries,
            "mean_candidates_per_query": total_candidates / balanced_queries if balanced_queries else None,
            "mean_objective_tied_candidates_per_query": total_ties / balanced_queries if balanced_queries else None,
            "mean_success_candidate_availability": (
                sum(row["result"]["shooting_success_available_queries"] for row in balanced_rows) / balanced_queries
                if balanced_queries else None
            ),
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "interpretation": "teacher_qualification_only_no_student_or_world_model_evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    report = analyze(run_dir)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
