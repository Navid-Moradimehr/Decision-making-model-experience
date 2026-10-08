"""Collect teacher cost matrices for Phase 3B without opening locked test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from consequence_lab.candidates import primitive_actions, two_segment_sequences
from consequence_lab.environment import TwoRoomAdapter
from run_phase3a_tworoom_realizability import _cost, _git, _load_upstream_environment, _sha256


CONFIG_PATH = ROOT / "configs" / "phase3_tworoom_amortization.yaml"
BEHAVIORS = ("simulator_truth_shooting", "greedy", "random")
ALLOWED_SPLITS = {"train", "validation", "calibration", "matched_offline_control"}


def _assign_behaviors(seed_start: int, seed_end: int, counts: dict[str, int]) -> list[tuple[int, str]]:
    labels = []
    for name in BEHAVIORS:
        labels.extend([name] * int(counts[name]))
    if len(labels) != seed_end - seed_start:
        raise ValueError("behavior allocation does not match the selected seed range")
    rng = np.random.default_rng(seed_start + 7331)
    permutation = rng.permutation(len(labels))
    shuffled = [labels[int(index)] for index in permutation]
    return [(seed, shuffled[seed - seed_start]) for seed in range(seed_start, seed_end)]


def collect_split(config: dict, env_type, split: str, output_dir: Path, max_minutes: float) -> dict:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}; locked_test is never collectable here")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / f"{split}.npz"
    summary_path = output_dir / f"{split}_summary.json"
    manifest_path = output_dir / f"{split}_manifest.json"
    progress_path = output_dir / f"{split}_progress.jsonl"
    if any(path.exists() for path in (data_path, summary_path, manifest_path, progress_path)):
        raise FileExistsError(f"refusing to overwrite existing {split} artifacts in {output_dir}")

    low, high = config["data"]["seed_ranges"][split]
    counts = config["data"]["behavior_assignment"][split]
    assignments = _assign_behaviors(low, high, counts)
    env_cfg = config["environment"]
    sequences = two_segment_sequences(env_cfg["candidate_horizon_steps"])
    if len(sequences) != env_cfg["candidate_count"]:
        raise AssertionError("candidate bank size differs from frozen protocol")
    capture_steps = set(config["data"]["sampled_step_indices"])
    observations: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    reachability: list[np.ndarray] = []
    episode_seeds: list[int] = []
    step_indices: list[int] = []
    behavior_rows: list[str] = []
    progress_records = []
    started = time.perf_counter()
    deadline = started + max_minutes * 60.0

    with progress_path.open("x", encoding="utf-8") as progress:
        for episode_index, (seed, behavior) in enumerate(assignments, start=1):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Phase-3B data collection exceeded its explicit CPU-time budget")
            env = env_type(render_mode="rgb_array")
            adapter = TwoRoomAdapter(env, success_distance=env_cfg["success_distance_px"])
            observation, _ = adapter.reset(seed=seed)
            rng = np.random.default_rng(seed + 170000)
            captured = []
            try:
                for step in range(env_cfg["max_episode_steps"]):
                    is_capture = step in capture_steps
                    costs_for_step = None
                    if behavior == "simulator_truth_shooting" or is_capture:
                        costs_for_step, reached = _cost(
                            adapter,
                            adapter.snapshot(),
                            sequences,
                            observation[2:4],
                            {
                                "environment": env_cfg,
                                "teacher": {"objective": env_cfg["objective"]},
                            },
                        )
                    if is_capture:
                        if costs_for_step is None:
                            raise AssertionError("capture state did not receive teacher labels")
                        observations.append(observation.copy())
                        costs.append(costs_for_step.copy())
                        reachability.append(reached.copy())
                        episode_seeds.append(seed)
                        step_indices.append(step)
                        behavior_rows.append(behavior)
                        captured.append(step)

                    if behavior == "simulator_truth_shooting":
                        assert costs_for_step is not None
                        action = sequences[int(np.argmin(costs_for_step)), 0]
                    elif behavior == "greedy":
                        delta = observation[2:4] - observation[:2]
                        norm = float(np.linalg.norm(delta))
                        action = np.zeros(2, dtype=np.float32) if norm == 0 else delta / norm
                    else:
                        action = primitive_actions()[int(rng.integers(0, 9))]

                    observation, _, terminated, truncated, _ = adapter.step(action)
                    if terminated or truncated:
                        break
            finally:
                env.close()
            row = {
                "episode_index": episode_index,
                "seed": seed,
                "behavior": behavior,
                "captured_steps": captured,
            }
            progress.write(json.dumps(row, sort_keys=True) + "\n")
            progress.flush()
            progress_records.append(row)
            if episode_index % 10 == 0 or episode_index == len(assignments):
                elapsed = time.perf_counter() - started
                print(
                    f"{split}: {episode_index}/{len(assignments)} episodes; "
                    f"{len(observations)} states; {elapsed:.1f}s",
                    flush=True,
                )

    obs_array = np.asarray(observations, dtype=np.float32)
    costs_array = np.asarray(costs, dtype=np.float32)
    reach_array = np.asarray(reachability, dtype=bool)
    seed_array = np.asarray(episode_seeds, dtype=np.int64)
    step_array = np.asarray(step_indices, dtype=np.int16)
    behavior_array = np.asarray(behavior_rows, dtype="U32")
    if not len(obs_array):
        raise RuntimeError("no nonterminal states were captured")
    np.savez_compressed(
        data_path,
        observations=obs_array,
        candidate_costs=costs_array,
        candidate_reachability=reach_array,
        episode_seeds=seed_array,
        sampled_step_indices=step_array,
        behavior=behavior_array,
        candidate_sequences=sequences,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "experiment": "Phase 3B exact-simulator teacher data collection",
        "split": split,
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment_revision": env_cfg["revision"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device": "cpu",
        "seed_range": [low, high],
        "episode_count": len(assignments),
        "state_count": int(len(obs_array)),
        "states_by_behavior": {name: int(np.sum(behavior_array == name)) for name in BEHAVIORS},
        "episodes_by_behavior": counts,
        "step_histogram": {
            str(step): int(np.sum(step_array == step))
            for step in config["data"]["sampled_step_indices"]
        },
        "cost_range": [float(costs_array.min()), float(costs_array.max())],
        "data_sha256": _sha256(data_path),
        "data_bytes": data_path.stat().st_size,
        "wall_seconds": round(elapsed, 3),
        "max_minutes": max_minutes,
        "source_sha256": {
            "config": _sha256(CONFIG_PATH),
            "collector": _sha256(Path(__file__).resolve()),
            "student_module": _sha256(ROOT / "src/consequence_lab/student.py"),
            "environment_adapter": _sha256(ROOT / "src/consequence_lab/environment.py"),
        },
        "episode_records_sha256": hashlib.sha256(
            json.dumps(progress_records, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "git": {"commit": _git("rev-parse", "HEAD"), "status": _git("status", "--porcelain")},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "run_id": config["artifacts"]["phase3b_run_id"],
        "phase": "phase_3b_data",
        "split": split,
        "status": "completed",
        "seed_range": [low, high],
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": summary["source_sha256"]["config"],
        "data_path": str(data_path.relative_to(ROOT)),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "data_sha256": summary["data_sha256"],
        "sources_sha256": summary["source_sha256"],
        "git": summary["git"],
        "resources": {"device": "cpu", "wall_seconds": summary["wall_seconds"], "gpu_hours": 0},
        "notes": [
            "The exact teacher supplies costs only; these are not world-model predictions.",
            "Student training code receives only observation, candidate sequence features, and validity mask.",
            "This collector refuses the locked final-test split by construction.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=sorted(ALLOWED_SPLITS))
    parser.add_argument("--max-minutes", type=float, default=360.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    env_type = _load_upstream_environment(args.upstream_root.resolve())
    output_dir = args.output_dir or (ROOT / config["artifacts"]["dataset_directory"])
    result = collect_split(config, env_type, args.split, output_dir, args.max_minutes)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
