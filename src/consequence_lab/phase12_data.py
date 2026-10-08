"""Phase-12 layout audit and exact-shooting teacher dataset collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase12_teacher import (
    MAX_HORIZON, NUM_CANDIDATES, PAD_ACTION, make_teacher_query, state_features,
)


def _blocks(config: dict[str, Any], *, include_dagger: bool = True) -> dict[str, list[int]]:
    splits = config["seed_splits"]
    names = ["train_teacher", "validation_teacher"]
    if include_dagger:
        names.extend(("dagger_collection_round_1", "dagger_collection_round_2", "dagger_collection_round_3"))
    names.append("closed_loop_development")
    return {
        name: list(range(splits[name]["start"], splits[name]["start"] + splits[name]["count"]))
        for name in names
    }


def audit_phase12_layout_splits(config: dict[str, Any]) -> dict[str, Any]:
    """Hash-only check for all Phase-12 opened blocks and previously used layouts.

    The sealed reserve is deliberately not generated or read here.
    """
    blocks = _blocks(config)
    all_new_seeds = [seed for values in blocks.values() for seed in values]
    if len(all_new_seeds) != len(set(all_new_seeds)):
        raise ValueError("Phase 12 seed blocks overlap")
    hashes: dict[str, list[str]] = {}
    for name, seeds in blocks.items():
        scenarios = [generate_staggered_corridor(seed) for seed in seeds]
        values = [scenario.layout_sha256 for scenario in scenarios]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate layout within {name}")
        hashes[name] = values
    # These are already-open design/development layouts. Re-generation only
    # audits deterministic layout identity; it does not inspect reserve data.
    previous_seeds = list(range(110000, 110040)) + list(range(120000, 120100))
    previous_hashes = [generate_staggered_corridor(seed).layout_sha256 for seed in previous_seeds]
    if len(previous_hashes) != len(set(previous_hashes)):
        raise ValueError("historical Phase 11 layout collision")
    occupied = set(previous_hashes)
    for name, values in hashes.items():
        overlap = occupied.intersection(values)
        if overlap:
            raise ValueError(f"{name} overlaps a spent Phase 11 layout")
        occupied.update(values)
    return {
        "schema": "phase12_layout_split_manifest_v1",
        "reserve_generated_or_accessed": False,
        "historical_phase11_seeds_audited": [110000, 110039, 120000, 120099],
        "blocks": {
            name: {
                "seeds": values,
                "layout_sha256": hashes[name],
                "unique_layouts": len(set(hashes[name])),
            }
            for name, values in blocks.items()
        },
        "pairwise_disjoint": True,
        "sha256": {
            name: hashlib.sha256("\n".join(values).encode("ascii")).hexdigest().upper()
            for name, values in hashes.items()
        },
    }


def _write_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest().upper()}


def collect_teacher_split(
    seeds: list[int],
    output_dir: Path,
    *,
    artifact_cap_bytes: int,
    wall_cap_seconds: int,
) -> dict[str, Any]:
    """Collect exact planner queries on complete, receding-horizon teacher episodes."""
    started = time.monotonic()
    states: list[np.ndarray] = []
    sequences: list[np.ndarray] = []
    candidate_masks: list[np.ndarray] = []
    layout_indices: list[int] = []
    decision_indices: list[int] = []
    episode_seeds: list[int] = []
    optima: list[np.ndarray] = []
    regrets: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    objective_keys: list[np.ndarray] = []
    selected_indices: list[int] = []
    success_available: list[bool] = []
    trajectories: list[np.ndarray] = []
    trajectory_masks: list[np.ndarray] = []
    episode_rows: list[dict[str, Any]] = []
    layout_hashes: list[str] = []

    for layout_index, seed in enumerate(seeds):
        if time.monotonic() - started > wall_cap_seconds:
            raise TimeoutError("Phase 12 teacher data wall-time cap reached")
        scenario = generate_staggered_corridor(seed)
        layout_hashes.append(scenario.layout_sha256)
        state = scenario.initial_state
        decision_index = 0
        success_queries = 0
        while state.status == "active":
            if time.monotonic() - started > wall_cap_seconds:
                raise TimeoutError("Phase 12 teacher data wall-time cap reached")
            query = make_teacher_query(scenario, state)
            count = len(query.sequences)
            # Bound live in-memory arrays as well as final compressed files.
            # Raw exact rollouts dominate: 128 candidates x 9 states x 4 float32.
            estimated_per_query = (
                52 * 4 + NUM_CANDIDATES * MAX_HORIZON + NUM_CANDIDATES
                + NUM_CANDIDATES * (1 + 4 + 4 + 3 * 8)
                + NUM_CANDIDATES * (MAX_HORIZON + 1) * (4 * 4 + 1)
            )
            if (len(states) + 1) * estimated_per_query > artifact_cap_bytes:
                raise OSError("Phase 12 teacher data estimated in-memory artifact cap reached")
            state_array = state_features(scenario, state)
            seq_array = np.full((NUM_CANDIDATES, MAX_HORIZON), PAD_ACTION, dtype=np.uint8)
            seq_array[:count] = query.sequences
            cand_mask = np.zeros(NUM_CANDIDATES, dtype=np.bool_)
            cand_mask[:count] = True
            opt = np.zeros(NUM_CANDIDATES, dtype=np.bool_)
            opt[:count] = query.optimal_mask
            regret = np.zeros(NUM_CANDIDATES, dtype=np.float32)
            regret[:count] = query.normalized_regret
            rank = np.zeros(NUM_CANDIDATES, dtype=np.float32)
            rank[:count] = query.ranks
            outcome = np.full(NUM_CANDIDATES, 255, dtype=np.uint8)
            outcome[:count] = query.outcome_codes
            keys = np.zeros((NUM_CANDIDATES, 3), dtype=np.float64)
            keys[:count] = np.asarray(query.objective_keys, dtype=np.float64)
            rollout = np.zeros((NUM_CANDIDATES, MAX_HORIZON + 1, 4), dtype=np.float32)
            rollout[:count] = query.rollout_states
            rollout_mask = np.zeros((NUM_CANDIDATES, MAX_HORIZON + 1), dtype=np.bool_)
            rollout_mask[:count] = query.rollout_state_mask

            states.append(state_array)
            sequences.append(seq_array)
            candidate_masks.append(cand_mask)
            layout_indices.append(layout_index)
            decision_indices.append(decision_index)
            episode_seeds.append(seed)
            optima.append(opt)
            regrets.append(regret)
            ranks.append(rank)
            outcomes.append(outcome)
            objective_keys.append(keys)
            selected_indices.append(query.selected_index)
            success_available.append(query.success_available)
            trajectories.append(rollout)
            trajectory_masks.append(rollout_mask)
            success_queries += int(query.success_available)

            action = int(query.sequences[query.selected_index, 0])
            state = corridor_step(state, action, scenario.spec).state
            decision_index += 1
        episode_rows.append({
            "seed": seed,
            "layout_sha256": scenario.layout_sha256,
            "outcome": state.status,
            "steps": state.elapsed_steps - scenario.initial_state.elapsed_steps,
            "decision_queries": decision_index,
            "queries_with_success_sequence": success_queries,
        })

        # Enforce the configured artifact ceiling during collection. Existing
        # files in the dedicated output directory count toward the same cap.
        if output_dir.exists():
            current_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
            if current_bytes > artifact_cap_bytes:
                raise OSError("Phase 12 teacher data artifact cap reached")

    input_arrays = {
        "states": np.stack(states),
        "sequences": np.stack(sequences),
        "candidate_mask": np.stack(candidate_masks),
    }
    metadata_arrays = {
        "layout_index": np.asarray(layout_indices, dtype=np.int32),
        "decision_index": np.asarray(decision_indices, dtype=np.int16),
        "episode_seed": np.asarray(episode_seeds, dtype=np.int64),
    }
    target_arrays = {
        "optimal_mask": np.stack(optima),
        "normalized_regret": np.stack(regrets),
        "average_rank": np.stack(ranks),
        "outcome_code": np.stack(outcomes),
        "objective_key": np.stack(objective_keys),
        "teacher_selected_index": np.asarray(selected_indices, dtype=np.int16),
        "success_candidate_available": np.asarray(success_available, dtype=np.bool_),
    }
    rollout_arrays = {
        "states": np.stack(trajectories),
        "state_mask": np.stack(trajectory_masks),
    }
    files = {
        "student_inputs": _write_npz(output_dir / "student_inputs.npz", **input_arrays),
        "sample_metadata": _write_npz(output_dir / "sample_metadata.npz", **metadata_arrays),
        "teacher_targets": _write_npz(output_dir / "teacher_targets.npz", **target_arrays),
        "teacher_rollouts": _write_npz(output_dir / "teacher_rollouts.npz", **rollout_arrays),
    }
    import json as _json
    episode_path = output_dir / "episodes.json"
    episode_path.write_text(_json.dumps(episode_rows, indent=2) + "\n", encoding="utf-8")
    files["episodes"] = {
        "path": str(episode_path), "bytes": episode_path.stat().st_size,
        "sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest().upper(),
    }
    total_bytes = sum(item["bytes"] for item in files.values())
    if total_bytes > artifact_cap_bytes:
        raise OSError("Phase 12 teacher data artifact cap exceeded")
    return {
        "status": "completed_exact_teacher_collection",
        "episode_count": len(episode_rows),
        "decision_query_count": len(states),
        "layout_sha256": layout_hashes,
        "episodes": episode_rows,
        "candidate_availability_fraction": float(np.mean(success_available)),
        "success_rate": sum(row["outcome"] == "success" for row in episode_rows) / len(episode_rows),
        "files": files,
        "artifact_bytes": total_bytes,
        "wall_seconds": time.monotonic() - started,
        "teacher_selected_action_replans_each_step": True,
    }
