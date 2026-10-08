"""Check whether a hand-designed doorway policy removes the student headroom."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase3a_tworoom_realizability import _git, _load_upstream_environment, _sha256
from train_phase3b_tworoom_students import _load_npz
from evaluate_phase3b_tworoom_validation import _paired_bootstrap_difference, _rollout_method
from consequence_lab.candidates import two_segment_sequences


CONFIG_PATH = ROOT / "configs/phase3_tworoom_amortization.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = args.data_dir or (ROOT / config["artifacts"]["dataset_directory"])
    output_dir = args.output_dir or (ROOT / "artifacts/runs/phase3b_tworoom_geometry_baselines_v1")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    data = _load_npz(data_dir / "validation.npz", "validation")
    low, high = config["data"]["seed_ranges"]["validation"]
    seeds = np.arange(low, high, dtype=np.int64)
    if not np.array_equal(np.unique(data["episode_seeds"]), seeds):
        raise ValueError("validation file does not contain exactly the frozen validation seeds")
    locked_low, locked_high = config["data"]["seed_ranges"]["locked_test"]
    if max(low, locked_low) < min(high, locked_high):
        raise ValueError("validation interval overlaps locked test")
    output_dir.mkdir(parents=True)
    env_type = _load_upstream_environment(args.upstream_root.resolve())
    sequences = two_segment_sequences(config["environment"]["candidate_horizon_steps"])
    begin = time.perf_counter()
    results = {}
    for method in ("greedy", "doorway_waypoint", "random", "exact_planner"):
        results[method] = _rollout_method(env_type, method, seeds, config, sequences)
        print(
            f"{method}: {results[method]['success_count']}/{len(seeds)} "
            f"({results[method]['success_rate']:.3f}), mean steps "
            f"{results[method]['mean_steps']:.1f}",
            flush=True,
        )
    exact_success = np.asarray([row["success"] for row in results["exact_planner"]["episode_results"]])
    paired = {}
    for method in ("greedy", "doorway_waypoint", "random"):
        method_success = np.asarray([row["success"] for row in results[method]["episode_results"]])
        paired[method] = _paired_bootstrap_difference(method_success, exact_success)
    waypoint = results["doorway_waypoint"]
    exact = results["exact_planner"]
    summary = {
        "experiment": "Phase 3B validation-only geometry baseline screen",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_seed_range": [low, high],
        "locked_test_opened": False,
        "validation_data_sha256": _sha256(data_dir / "validation.npz"),
        "config_sha256": _sha256(CONFIG_PATH),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "supporting_evaluator_sha256": _sha256(ROOT / "scripts/evaluate_phase3b_tworoom_validation.py"),
        "baselines": results,
        "paired_success_difference_vs_exact_planner": paired,
        "paired_bootstrap": {"unit": "episode_seed", "replicates": 10000, "seed": 501091},
        "waypoint_planner_point_gap": float(exact["success_rate"] - waypoint["success_rate"]),
        "interpretation": (
            "Validation-only screening diagnostic. The point comparison and paired interval are not a final gate; "
            "if the waypoint is near the planner, reassess whether there is enough task headroom "
            "before spending compute on student training."
        ),
        "wall_seconds": round(time.perf_counter() - begin, 3),
        "device": "cpu",
        "gpu_hours": 0,
        "git": {"commit": _git("rev-parse", "HEAD"), "status": _git("status", "--porcelain")},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "phase": "phase_3b_geometry_baseline_screen",
        "status": "completed",
        "data_split": "validation",
        "locked_test_opened": False,
        "validation_data_sha256": summary["validation_data_sha256"],
        "config_sha256": summary["config_sha256"],
        "script_sha256": summary["script_sha256"],
        "supporting_evaluator_sha256": summary["supporting_evaluator_sha256"],
        "summary_path": str(summary_path.relative_to(ROOT)),
        "git": summary["git"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "waypoint_planner_point_gap": summary["waypoint_planner_point_gap"],
        "paired_waypoint_vs_exact": paired["doorway_waypoint"],
    }, indent=2))


if __name__ == "__main__":
    main()
