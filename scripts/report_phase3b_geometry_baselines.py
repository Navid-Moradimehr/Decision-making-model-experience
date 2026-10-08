"""Create a concise, paired validation report for the Phase-3B baseline screen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts/runs/phase3b_tworoom_geometry_baselines_v1/summary.json"
REPORT = ROOT / "reports/phase3b_tworoom_geometry_baselines.md"
BOOTSTRAP_SEED = 501091
BOOTSTRAP_REPLICATES = 10000


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054):
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)) / denominator
    return [float(center - half), float(center + half)]


def paired_difference(a: np.ndarray, b: np.ndarray):
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(difference), size=(BOOTSTRAP_REPLICATES, len(difference)))
    means = difference[draws].mean(axis=1)
    return {
        "mean": float(difference.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "paired_success_cells": {
            "both_success": int(np.sum((a == 1) & (b == 1))),
            "a_only": int(np.sum((a == 1) & (b == 0))),
            "b_only": int(np.sum((a == 0) & (b == 1))),
            "both_fail": int(np.sum((a == 0) & (b == 0))),
        },
    }


def create_report(summary_path: Path, report_path: Path):
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("locked_test_opened") is not False:
        raise ValueError("baseline summary is incomplete or indicates locked-test access")
    methods = summary["baselines"]
    required = {"greedy", "doorway_waypoint", "random", "exact_planner"}
    if set(methods) != required:
        raise ValueError("required validation baseline results are missing")
    expected_seeds = list(range(*summary["validation_seed_range"]))
    outcomes = {}
    for method, result in methods.items():
        rows = result["episode_results"]
        seeds = [row["seed"] for row in rows]
        if seeds != expected_seeds:
            raise ValueError(f"{method} does not use the complete frozen validation seed range")
        outcomes[method] = np.asarray([row["success"] for row in rows], dtype=np.int8)

    rows = []
    for method in ("greedy", "doorway_waypoint", "exact_planner", "random"):
        result = methods[method]
        n = result["episode_count"]
        latency_text = (
            "not timed"
            if result["decision_latency_median_ms"] is None
            else f"{result['decision_latency_median_ms']:.1f}"
        )
        rows.append(
            f"| {method.replace('_', ' ')} | {result['success_count']}/{n} "
            f"({100.0 * result['success_rate']:.1f}%) | "
            f"[{100.0 * wilson_interval(result['success_count'], n)[0]:.1f}, "
            f"{100.0 * wilson_interval(result['success_count'], n)[1]:.1f}] | "
            f"{result['mean_steps']:.1f} | {latency_text} |"
        )
    comparisons = {
        "waypoint minus greedy": paired_difference(outcomes["doorway_waypoint"], outcomes["greedy"]),
        "exact planner minus greedy": paired_difference(outcomes["exact_planner"], outcomes["greedy"]),
        "waypoint minus exact planner": paired_difference(outcomes["doorway_waypoint"], outcomes["exact_planner"]),
    }
    contrast_lines = []
    for label, result in comparisons.items():
        low, high = result["ci95"]
        cells = result["paired_success_cells"]
        contrast_lines.append(
            f"- `{label}`: {result['mean']:+.3f} (paired episode-bootstrap 95% interval "
            f"[{low:+.3f}, {high:+.3f}]); cells both-success/a-only/b-only/both-fail = "
            f"{cells['both_success']}/{cells['a_only']}/{cells['b_only']}/{cells['both_fail']}."
        )
    exact = methods["exact_planner"]
    waypoint = methods["doorway_waypoint"]
    report = f"""# Phase 3B validation baseline screen

**Result:** stop before student fitting. On the 60 frozen validation episodes, the fixed-doorway waypoint controller succeeded on every seed (60/60), while exact simulator-truth shooting succeeded on 58/60. The waypoint is both at least as successful on this block and shorter in mean episode length (23.7 vs. 29.6 actions). A cheap, deployable geometric policy therefore removes the decision-performance headroom needed to justify distilling this planner in this task. This is a task/baseline stop, not evidence that a student failed.

## Protocol and results

- Paired validation seeds `{summary['validation_seed_range'][0]}–{summary['validation_seed_range'][1] - 1}`; the independent final-test range remains unopened.
- All methods use the pinned TwoRoom simulator, fixed visible doorway, 200-step episode cap, and the same success-distance rule. The waypoint targets the observed door center when current position and goal lie on opposite sides; otherwise it moves directly to the goal.
- Paired uncertainty resamples complete episode seeds ({BOOTSTRAP_REPLICATES:,} replicates, seed `{BOOTSTRAP_SEED}`). Wilson intervals below are marginal binomial intervals.

| Method | Success | Wilson 95% interval (%) | Mean steps | Median decision latency (ms) |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

Paired success differences (first method minus second):

{chr(10).join(contrast_lines)}

The exact planner made {exact['teacher_cost_evaluations']:,} 65-candidate cost evaluations; median/p95 planning-call latency was {exact['decision_latency_median_ms']:.1f}/{exact['decision_latency_p95_ms']:.1f} ms on CPU. Waypoint latency was not instrumented, so no numerical speedup is claimed.

## Decision and limits

- Do not train the direct policy, candidate scorer, candidate-cost model, oracle-future model, or DAgger variant for this fixed-geometry task. They cannot be justified over a 100%-successful geometric baseline using these validation outcomes.
- Do not calibrate or open `[244000,244200)`. No final-test outcome is needed to make this development-stage stop.
- This does not show that planner distillation fails generally, that the exact planner is useless in other environments, or that world models/JEPA are ineffective. It shows this known-goal TwoRoom task’s apparent headroom over greedy disappears against an audited structure-aware controller.
- The collected train/validation label data are retained as ignored artifacts, with integrity audit and hashes. They must not be reused to make a confirmatory claim about a redesigned task.

Artifacts: baseline summary SHA-256 `{__import__('hashlib').sha256(summary_path.read_bytes()).hexdigest()}`, validation data SHA-256 `{summary['validation_data_sha256']}`, config SHA-256 `{summary['config_sha256']}`. Full per-episode outcomes and provenance are in `artifacts/runs/phase3b_tworoom_geometry_baselines_v1/summary.json`.
"""
    report_path.write_text(report, encoding="utf-8")
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    comparisons = create_report(args.summary, args.report)
    print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
