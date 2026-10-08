"""Independent contract/provenance check for completed Phase-3B datasets."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import gymnasium
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from consequence_lab.candidates import two_segment_sequences
from consequence_lab.environment import TwoRoomAdapter
from run_phase3a_tworoom_realizability import (
    _cost,
    _git,
    _load_upstream_environment,
    _reset_observable_state,
    _sha256,
)
from collect_phase3b_tworoom_data import _assign_behaviors


CONFIG_PATH = ROOT / "configs/phase3_tworoom_amortization.yaml"
EXPECTED_UPSTREAM_HASHES = {
    "stable_worldmodel/spaces.py": "69e583dd06fa754e226f2430d4ff459e2dc8d7534fdb27cf7f9f5921bdec8018",
    "stable_worldmodel/envs/two_room/env.py": "5e1d392de5b02472062dbe872aded67fd465fcc8f7eaa1c02a753b2fc31c61f0",
}


def audit(config: dict, env_type, run_dir: Path, upstream_root: Path):
    output_path = run_dir / "saved_data_audit.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    sequences = two_segment_sequences(config["environment"]["candidate_horizon_steps"])
    expected_files = {}
    errors = []
    split_results = {}
    checks_sampled = []
    for split in ("train", "validation"):
        data_path = run_dir / f"{split}.npz"
        progress_path = run_dir / f"{split}_progress.jsonl"
        if not data_path.is_file() or not progress_path.is_file():
            raise FileNotFoundError(f"completed {split} data/progress is required")
        with np.load(data_path, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        required = {
            "observations", "candidate_costs", "candidate_reachability", "episode_seeds",
            "sampled_step_indices", "behavior", "candidate_sequences",
        }
        if set(data) != required:
            errors.append(f"{split}: unexpected/missing data fields")
        state_count = len(data["observations"])
        if data["observations"].shape != (state_count, 10):
            errors.append(f"{split}: observation shape mismatch")
        if data["candidate_costs"].shape != (state_count, 65):
            errors.append(f"{split}: candidate-cost shape mismatch")
        if data["candidate_reachability"].shape != (state_count, 65):
            errors.append(f"{split}: reachability shape mismatch")
        if not np.array_equal(data["candidate_sequences"], sequences):
            errors.append(f"{split}: candidate bank differs from canonical generator")
        if not np.isfinite(data["observations"]).all() or not np.isfinite(data["candidate_costs"]).all():
            errors.append(f"{split}: non-finite observations/costs")
        if data["candidate_reachability"].dtype != np.bool_:
            errors.append(f"{split}: reachability field is not boolean")
        if data["candidate_costs"].min() < -1e-6 or data["candidate_costs"].max() > 1.1 + 1e-6:
            errors.append(f"{split}: cost outside protocol objective bounds")

        low, high = config["data"]["seed_ranges"][split]
        expected_seeds = np.arange(low, high, dtype=np.int64)
        observed_seeds = np.unique(data["episode_seeds"])
        if not np.array_equal(observed_seeds, expected_seeds):
            errors.append(f"{split}: seed coverage differs from its complete-episode range")
        assigned = _assign_behaviors(low, high, config["data"]["behavior_assignment"][split])
        expected_behavior = dict(assigned)
        if any(
            str(actual) != expected_behavior[int(seed)]
            for seed, actual in zip(data["episode_seeds"], data["behavior"], strict=True)
        ):
            errors.append(f"{split}: behavior assignment does not match frozen schedule")
        progress_rows = [
            json.loads(line)
            for line in progress_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(progress_rows) != high - low:
            errors.append(f"{split}: progress episode count mismatch")
        for row, (seed, behavior) in zip(progress_rows, assigned, strict=True):
            if row.get("seed") != seed or row.get("behavior") != behavior:
                errors.append(f"{split}: progress provenance differs at seed {seed}")
            recorded_steps = sorted(
                map(int, data["sampled_step_indices"][data["episode_seeds"] == seed])
            )
            if recorded_steps != sorted(map(int, row.get("captured_steps", []))):
                errors.append(f"{split}: captured state rows do not match progress at seed {seed}")

        unique_state_seeds = set(map(int, observed_seeds))
        locked_low, locked_high = config["data"]["seed_ranges"]["locked_test"]
        if any(locked_low <= seed < locked_high for seed in unique_state_seeds):
            errors.append(f"{split}: locked-test seed leakage")
        expected_files[split] = {
            "data_sha256": _sha256(data_path),
            "data_bytes": data_path.stat().st_size,
            "progress_sha256": _sha256(progress_path),
            "episode_count": int(high - low),
            "state_count": int(state_count),
            "behavior_episodes": config["data"]["behavior_assignment"][split],
            "states_by_behavior": {
                name: int(np.sum(data["behavior"] == name))
                for name in config["data"]["behavior_assignment"][split]
            },
            "cost_min_max": [float(data["candidate_costs"].min()), float(data["candidate_costs"].max())],
            "locked_test_seed_leaks": 0,
        }
        split_results[split] = data

    # Recompute exact teacher costs at a deterministic, spread sample of states
    # in each split. This independently checks the saved label matrix without
    # spending a full second rollout over thousands of states.
    objective_config = {
        "environment": config["environment"],
        "teacher": {"objective": config["environment"]["objective"]},
    }
    for split, data in split_results.items():
        chosen = np.linspace(0, len(data["observations"]) - 1, min(12, len(data["observations"])), dtype=int)
        for index in chosen:
            obs = data["observations"][index]
            env, reconstructed = _reset_observable_state(env_type, obs)
            adapter = TwoRoomAdapter(env, success_distance=config["environment"]["success_distance_px"])
            costs, reachability = _cost(adapter, adapter.snapshot(), sequences, obs[2:4], objective_config)
            obs_error = float(np.max(np.abs(reconstructed - obs)))
            cost_error = float(np.max(np.abs(costs - data["candidate_costs"][index])))
            reach_equal = bool(np.array_equal(reachability, data["candidate_reachability"][index]))
            checks_sampled.append({
                "split": split,
                "seed": int(data["episode_seeds"][index]),
                "step": int(data["sampled_step_indices"][index]),
                "observation_max_abs_error": obs_error,
                "candidate_cost_max_abs_error": cost_error,
                "candidate_reachability_equal": reach_equal,
            })
            if obs_error > 1e-6 or cost_error > 1e-6 or not reach_equal:
                errors.append(f"{split}: exact simulator reconstruction mismatch at row {index}")
            env.close()

    source_paths = {
        "config": CONFIG_PATH,
        "collector": ROOT / "scripts/collect_phase3b_tworoom_data.py",
        "collector_helper": ROOT / "scripts/run_phase3a_tworoom_realizability.py",
        "auditor": Path(__file__).resolve(),
        "candidate_generator": ROOT / "src/consequence_lab/candidates.py",
        "objective": ROOT / "src/consequence_lab/objectives.py",
        "environment_adapter": ROOT / "src/consequence_lab/environment.py",
        "source_pins": ROOT / "experiments/source_pins.json",
    }
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    upstream_hashes = getattr(env_type, "_phase3_source_hashes", {})
    if upstream_hashes != EXPECTED_UPSTREAM_HASHES:
        errors.append("pinned upstream source hashes do not match reviewed values")

    result = {
        "experiment": "Phase 3B saved-data integrity and source audit",
        "status": "passed" if not errors else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "errors": errors,
        "split_results": expected_files,
        "exact_cost_reconstruction_sample": {
            "sample_count": len(checks_sampled),
            "max_observation_error": max((row["observation_max_abs_error"] for row in checks_sampled), default=0.0),
            "max_candidate_cost_error": max((row["candidate_cost_max_abs_error"] for row in checks_sampled), default=0.0),
            "reachability_mismatches": sum(not row["candidate_reachability_equal"] for row in checks_sampled),
            "rows": checks_sampled,
        },
        "config_sha256": source_hashes["config"],
        "source_sha256": source_hashes,
        "upstream_source_sha256": upstream_hashes,
        "git": {"commit": _git("rev-parse", "HEAD"), "status": _git("status", "--porcelain")},
        "locked_test_opened": False,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    run_dir = args.run_dir or (ROOT / config["artifacts"]["dataset_directory"])
    env_type = _load_upstream_environment(args.upstream_root.resolve())
    result = audit(config, env_type, run_dir, args.upstream_root.resolve())
    print(json.dumps({
        "status": result["status"],
        "errors": result["errors"],
        "reconstruction": result["exact_cost_reconstruction_sample"],
    }, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
