"""Compare greedy navigation with simulator-truth finite candidate shooting."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers the swm/ Gymnasium environments
import torch

from consequence_lab.candidates import two_segment_sequences
from consequence_lab.environment import TwoRoomAdapter
from consequence_lab.objectives import goal_cost


def evaluate_episode(adapter, method: str, seed: int, sequences: np.ndarray) -> dict:
    observation, _ = adapter.reset(seed=seed)
    candidate_transition_evaluations = 0
    for step in range(1, 201):
        if method == "greedy":
            direction = observation[2:4] - observation[:2]
            norm = float(np.linalg.norm(direction))
            action = np.zeros(2, dtype=np.float32) if norm == 0 else direction / norm
        else:
            futures = adapter.rollout_candidates(adapter.snapshot(), sequences)
            distances = np.linalg.norm(futures - observation[2:4], axis=-1)
            reached = distances < 16.0
            first_reached = np.argmax(reached, axis=1)
            candidate_transition_evaluations += int(
                np.where(reached.any(axis=1), first_reached + 1, sequences.shape[1]).sum()
            )
            costs = goal_cost(futures, observation[2:4], arena_diagonal=float(np.sqrt(2) * 224))
            action = sequences[int(np.argmin(costs)), 0]

        observation, _, terminated, truncated, _ = adapter.step(action)
        if terminated or truncated:
            return {
                "seed": seed,
                "method": method,
                "success": bool(terminated),
                "steps": step,
                "candidate_transition_evaluations": candidate_transition_evaluations,
            }
    return {
        "seed": seed,
        "method": method,
        "success": False,
        "steps": 200,
        "candidate_transition_evaluations": candidate_transition_evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[601, 602, 603])
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=0, help="Report each N completed seed pairs to stderr; 0 disables progress.")
    parser.add_argument("--output", type=Path, help="Optional path for the complete JSON result.")
    parser.add_argument("--source-loaded-temporarily", action="store_true", help="Record that the pinned simulator source was loaded from a disposable checkout.")
    args = parser.parse_args()
    if args.horizon < 2:
        parser.error("--horizon must be at least 2")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("episode seeds must be unique")

    env = gym.make("swm/TwoRoom-v1")
    adapter = TwoRoomAdapter(env)
    sequences = two_segment_sequences(args.horizon)
    started = time.perf_counter()
    episodes = []
    for seed_index, seed in enumerate(args.seeds, start=1):
        for method in ("greedy", "simulator_shooting"):
            episodes.append(evaluate_episode(adapter, method, seed, sequences))
        if args.progress_every and seed_index % args.progress_every == 0:
            print(f"completed {seed_index}/{len(args.seeds)} paired seeds", file=sys.stderr, flush=True)
    env.close()
    planner_calls = sum(
        row["steps"] for row in episodes if row["method"] == "simulator_shooting"
    )

    summary = {
        "environment": "swm/TwoRoom-v1",
        "stable_worldmodel_revision": "63988116d34cde56aea1240d5e58eb158ac67dc0",
        "stable_worldmodel_version": _package_version("stable-worldmodel"),
        "horizon": args.horizon,
        "candidate_count": len(sequences),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gym.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "source_mode": "temporary_pinned_checkout" if args.source_loaded_temporarily else "installed_package",
        "seeds": args.seeds,
        "episodes_per_method": len(args.seeds),
        "success_by_method": {
            method: sum(row["success"] for row in episodes if row["method"] == method)
            for method in ("greedy", "simulator_shooting")
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "planner_calls": planner_calls,
        "candidate_transition_evaluations": sum(
            row["candidate_transition_evaluations"] for row in episodes
        ),
        "episodes": episodes,
        "interpretation": "Candidate-ceiling feasibility only; not a learned-model or policy result.",
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        config_path = Path("configs/phase0_tworoom.yaml")
        pins_path = Path("experiments/source_pins.json")
        manifest = {
            "run_id": args.output.parent.name,
            "phase": "phase_0",
            "status": "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git": {"commit": None, "dirty": True},
            "config_path": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_pins_path": str(pins_path),
            "source_pins_sha256": hashlib.sha256(pins_path.read_bytes()).hexdigest(),
            "environment": {
                "id": summary["environment"],
                "revision": summary["stable_worldmodel_revision"],
                "episode_seed_start": args.seeds[0],
                "episode_count": len(args.seeds),
            },
            "method": "paired_greedy_vs_simulator_truth_shooting",
            "seed": args.seeds[0],
            "model_sha256": None,
            "metrics_path": None,
            "summary_path": str(args.output),
            "resources": {
                "device": "cpu",
                "wall_seconds": summary["elapsed_seconds"],
                "peak_vram_bytes": 0,
                "peak_ram_bytes": None,
                "artifact_bytes": args.output.stat().st_size,
                "planner_calls": summary["planner_calls"],
                "candidate_transition_evaluations": summary["candidate_transition_evaluations"],
            },
            "notes": [
                "Paired fixed-seed candidate-coverage evaluation; exploratory 3-seed pilot is excluded from this 200-seed set.",
                "Source mode: " + summary["source_mode"],
                "Oracle only: simulator truth, not a learned model or policy.",
            ],
        }
        (args.output.parent / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(serialized)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
