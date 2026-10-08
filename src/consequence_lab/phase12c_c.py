"""Shared, auditable data and target utilities for Phase 12c-C."""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any
from pathlib import Path

import numpy as np

from consequence_lab.corridor_baselines import grid_astar_route, waypoint_braking_action
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import ACTION_VECTORS, corridor_step
from consequence_lab.phase12_teacher import PAD_ACTION, MAX_HORIZON, candidate_features, evaluate_candidate_set, state_features
from consequence_lab.phase12c import _phase12c_shooting_decision, balanced_continuation_suffixes
from consequence_lab.phase12c_b import replay_fitting_queries


SEALED_RESERVE = (130000, 130200)
PHASE11_12CA_AUDIT = "artifacts/runs/phase12cA_layout_audit_v2_160000_160099.json"


def audit_phase12c_c_layouts(config: dict[str, Any], prior_audit: dict[str, Any]) -> dict[str, Any]:
    """Hash-only audit of fresh selection/gate layouts against all audited blocks."""
    if prior_audit.get("schema") != "phase12c_a_layout_audit_v1":
        raise ValueError("prior Phase 12c-A layout audit schema mismatch")
    if prior_audit.get("reserve_generated_or_accessed") is not False:
        raise ValueError("prior audit does not attest sealed reserve")
    occupied: set[str] = set()
    prior_blocks: dict[str, int] = {}
    for name, block in prior_audit.get("historical_blocks", {}).items():
        hashes = block.get("layout_sha256", [])
        if not hashes or len(hashes) != len(set(hashes)):
            raise ValueError(f"invalid or duplicate hashes in prior block {name}")
        overlap = occupied.intersection(hashes)
        if overlap:
            raise ValueError(f"prior blocks overlap at {name}")
        occupied.update(hashes)
        prior_blocks[name] = len(hashes)

    selected_hashes = prior_audit.get("layout_sha256", [])
    seed_block = prior_audit.get("seed_block", {})
    if seed_block != {"start": 160000, "count": 100} or len(selected_hashes) != 100:
        raise ValueError("Phase 12c-A selection block is incomplete or unexpected")
    if len(set(selected_hashes)) != 100 or occupied.intersection(selected_hashes):
        raise ValueError("Phase 12c-A layouts overlap another audited block")
    occupied.update(selected_hashes)

    blocks: dict[str, dict[str, Any]] = {}
    new_union: set[str] = set()
    for name in ("model_selection", "offline_gate"):
        spec = config["seed_splits"][name]
        start, count = int(spec["start"]), int(spec["count"])
        seeds = list(range(start, start + count))
        if start < 0 or count <= 0:
            raise ValueError(f"invalid seed block for {name}")
        if set(seeds).intersection(range(*SEALED_RESERVE)):
            raise ValueError(f"{name} overlaps sealed reserve seeds")
        hashes = [generate_staggered_corridor(seed).layout_sha256 for seed in seeds]
        if len(set(hashes)) != count:
            raise ValueError(f"duplicate layouts within {name}")
        if occupied.intersection(hashes) or new_union.intersection(hashes):
            raise ValueError(f"layout hash overlap in {name}")
        new_union.update(hashes)
        blocks[name] = {
            "seeds": seeds,
            "layout_sha256": hashes,
            "unique_layouts": count,
            "layout_list_sha256": hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest().upper(),
        }

    return {
        "schema": "phase12c_c_layout_audit_v1",
        "prior_audit_schema": prior_audit["schema"],
        "prior_historical_blocks_checked": prior_blocks,
        "phase12cA_layouts_checked": len(selected_hashes),
        "new_blocks": blocks,
        "new_hashes_disjoint_from_all_phase11_phase12_phase12b_phase12cA_blocks": True,
        "new_blocks_pairwise_disjoint": True,
        "layout_generation_only_no_outcomes": True,
        "development_outcomes_accessed": False,
        "dagger_outcomes_accessed": False,
        "reserve_accessed": False,
    }


def first_action_candidate_features(n: int) -> np.ndarray:
    """Return the same nine legal action candidates in canonical row order."""
    if isinstance(n, bool) or int(n) != n or n <= 0:
        raise ValueError("n must be a positive integer")
    action = np.full((9, 8), 9, dtype=np.uint8)
    action[:, 0] = np.arange(9, dtype=np.uint8)
    return np.broadcast_to(action, (int(n), 9, 8)).copy()


def bank_regret_and_preference(group_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact tie-aware rank-regret and tie-split optimum mass per bank."""
    keys = np.asarray(group_keys, dtype=np.float64)
    if keys.shape != (9, 3) or not np.isfinite(keys).all():
        raise ValueError("group_keys must be a finite [9,3] exact-objective array")
    values = [tuple(float(value) for value in row) for row in keys]
    ranks = np.empty(9, dtype=np.float32)
    for index, key in enumerate(values):
        better = sum(other < key for other in values)
        tied = sum(other == key for other in values)
        ranks[index] = better + (tied + 1) / 2.0
    regret = (ranks - float(ranks.min())) / 8.0
    best = min(values)
    optimal = np.asarray([key == best for key in values], dtype=np.float32)
    preference = optimal / optimal.sum()
    return regret.astype(np.float32), preference.astype(np.float32)


def aggregate_bank_targets(
    bank_group_keys: np.ndarray, *, bank_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
) -> dict[str, np.ndarray]:
    """Aggregate full exact bank landscapes into value and policy targets."""
    keys = np.asarray(bank_group_keys, dtype=np.float64)
    if keys.ndim != 4 or keys.shape[2:] != (9, 3):
        raise ValueError("bank_group_keys must have shape [queries,banks,9,3]")
    chosen = np.asarray(bank_indices, dtype=np.int64)
    if len(chosen) == 0 or np.any(chosen < 0) or np.any(chosen >= keys.shape[1]):
        raise ValueError("invalid target-estimation bank indices")
    regrets = np.empty(keys.shape[:3], dtype=np.float32)
    preferences = np.empty_like(regrets)
    for query in range(keys.shape[0]):
        for bank in range(keys.shape[1]):
            regrets[query, bank], preferences[query, bank] = bank_regret_and_preference(keys[query, bank])
    return {
        "per_bank_regret": regrets,
        "per_bank_optimal_mass": preferences,
        "mean_regret_target": regrets[:, chosen].mean(axis=1, dtype=np.float64).astype(np.float32),
        "preference_target": preferences[:, chosen].mean(axis=1, dtype=np.float64).astype(np.float32),
    }


def balanced_candidate_bank(
    scenario: Any, state: Any, *, seed: int, suffixes_per_action: int = 14
) -> dict[str, Any]:
    """Evaluate and preserve every exact candidate objective in one balanced bank."""
    horizon = min(MAX_HORIZON, scenario.spec.max_steps - state.elapsed_steps)
    suffixes = balanced_continuation_suffixes(
        horizon, suffixes_per_action=suffixes_per_action, seed=seed
    )
    sequences = np.full((9 * len(suffixes), MAX_HORIZON), PAD_ACTION, dtype=np.uint8)
    for action in range(9):
        for suffix_index, suffix in enumerate(suffixes):
            sequences[action * len(suffixes) + suffix_index, :horizon] = (action, *suffix)
    query = evaluate_candidate_set(state, scenario, sequences)
    keys = np.asarray(query.objective_keys, dtype=np.float64)
    if keys.shape != (len(sequences), 3) or not np.isfinite(keys).all():
        raise ValueError("exact candidate objectives are malformed")
    # Match the balanced teacher's order-independent SHA-256 content tie-break.
    best_indices = np.flatnonzero(query.optimal_mask)
    selected_index = min(
        best_indices.tolist(),
        key=lambda index: hashlib.sha256(bytes(sequences[index, :horizon])).digest(),
    )
    padded_suffixes = np.full((14, MAX_HORIZON - 1), PAD_ACTION, dtype=np.uint8)
    for index, suffix in enumerate(suffixes):
        padded_suffixes[index, : len(suffix)] = suffix
    candidate_keys = np.full((126, 3), np.nan, dtype=np.float64)
    candidate_keys[: len(keys)] = keys
    candidate_mask = np.zeros(126, dtype=np.bool_)
    candidate_mask[: len(keys)] = True
    return {
        "proposal_seed": np.uint64(seed),
        "proposal_bank_sha256": hashlib.sha256(sequences.tobytes()).hexdigest().encode("ascii"),
        "horizon": np.uint8(horizon),
        "suffix_count": np.uint8(len(suffixes)),
        "suffixes": padded_suffixes,
        "candidate_keys": candidate_keys,
        "candidate_mask": candidate_mask,
        "group_objective_keys": reduce_candidate_keys(keys, len(suffixes)),
        "candidate_success_available": bool(query.success_available),
        "teacher_selected_action": int(sequences[selected_index, 0]),
    }


def reduce_candidate_keys(candidate_keys: np.ndarray, suffix_count: int) -> np.ndarray:
    """Reduce action-major sequence keys to each first-action's lexicographic minimum."""
    keys = np.asarray(candidate_keys, dtype=np.float64)
    if isinstance(suffix_count, bool) or int(suffix_count) != suffix_count or not 1 <= suffix_count <= 14:
        raise ValueError("suffix_count must be an integer in [1,14]")
    if keys.shape != (9 * int(suffix_count), 3) or not np.isfinite(keys).all():
        raise ValueError("candidate_keys must match nine action-major suffix groups")
    groups = []
    for action in range(9):
        begin = action * int(suffix_count)
        end = begin + int(suffix_count)
        best = min(tuple(float(value) for value in key) for key in keys[begin:end])
        groups.append(best)
    return np.asarray(groups, dtype=np.float64)


def derive_bank_seed(master_seed: int, split: str, layout_hash: str, decision_index: int, bank_index: int) -> int:
    material = f"{master_seed}|{split}|{layout_hash}|{decision_index}|{bank_index}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:8], "big")


def build_archived_training_dataset(root: Path, *, master_seed: int, banks: int = 12,
                                   suffixes_per_action: int = 14,
                                   time_cap_seconds: int = 10800) -> dict[str, Any]:
    """Replay archived fitting trajectories and collect independent balanced banks."""
    started = time.monotonic()
    replayed = replay_fitting_queries(root)
    query_rows: list[tuple[str, Any]] = []
    for split in ("phase12_train_teacher_v1", "phase12_validation_teacher_v1"):
        for seed in sorted(replayed[split]):
            for row in replayed[split][seed]:
                query_rows.append((split, row))
    query_rows.sort(key=lambda item: (item[0], item[1].episode_seed, item[1].decision_index))
    if len(query_rows) != 2828 or len({row.layout_sha256 for _, row in query_rows}) != 160:
        raise ValueError("archived fitting source does not match frozen 2,828-query/160-layout contract")
    arrays = allocate_dataset_arrays(len(query_rows), banks)
    metadata: list[dict[str, Any]] = []
    for query_index, (split, row) in enumerate(query_rows):
        if time.monotonic() - started > time_cap_seconds:
            raise TimeoutError("Phase 12c-C archived target collection time cap reached")
        metadata.append({
            "query_index": query_index,
            "source_split": split,
            "episode_seed": row.episode_seed,
            "layout_sha256": row.layout_sha256,
            "decision_index": row.decision_index,
            "progress": row.progress,
        })
        arrays["states"][query_index] = state_features(row.scenario, row.state)
        for bank_index in range(banks):
            seed = derive_bank_seed(master_seed, "archived_train", row.layout_sha256,
                                    row.decision_index, bank_index)
            bank = balanced_candidate_bank(
                row.scenario, row.state, seed=seed, suffixes_per_action=suffixes_per_action
            )
            put_bank_in_arrays(arrays, query_index, bank_index, bank)
    finalize_dataset_arrays(arrays)
    return {**arrays, "metadata": metadata, "elapsed_wall_seconds": time.monotonic() - started}


def allocate_dataset_arrays(count: int, banks: int) -> dict[str, np.ndarray]:
    if count <= 0 or banks != 12:
        raise ValueError("Phase 12c-C requires positive query count and exactly 12 banks/query")
    return {
        "states": np.zeros((count, 52), dtype=np.float32),
        "candidate_action_ids": np.broadcast_to(np.arange(9, dtype=np.uint8), (count, 9)).copy(),
        "candidate_mask": np.ones((count, 9), dtype=np.bool_),
        "candidate_objective_keys": np.full((count, banks, 126, 3), np.nan, dtype=np.float64),
        "candidate_objective_mask": np.zeros((count, banks, 126), dtype=np.bool_),
        "suffixes": np.full((count, banks, 14, 7), PAD_ACTION, dtype=np.uint8),
        "suffix_count": np.zeros((count, banks), dtype=np.uint8),
        "horizon": np.zeros((count, banks), dtype=np.uint8),
        "proposal_seed": np.zeros((count, banks), dtype=np.uint64),
        "proposal_bank_sha256": np.zeros((count, banks), dtype="S64"),
        "candidate_success_available": np.zeros((count, banks), dtype=np.bool_),
        "teacher_selected_action": np.zeros((count, banks), dtype=np.uint8),
    }


def put_bank_in_arrays(arrays: dict[str, np.ndarray], query: int, bank_index: int,
                       bank: dict[str, Any]) -> None:
    arrays["candidate_objective_keys"][query, bank_index] = bank["candidate_keys"]
    arrays["candidate_objective_mask"][query, bank_index] = bank["candidate_mask"]
    arrays["suffixes"][query, bank_index] = bank["suffixes"]
    arrays["suffix_count"][query, bank_index] = bank["suffix_count"]
    arrays["horizon"][query, bank_index] = bank["horizon"]
    arrays["proposal_seed"][query, bank_index] = bank["proposal_seed"]
    arrays["proposal_bank_sha256"][query, bank_index] = bank["proposal_bank_sha256"]
    arrays["candidate_success_available"][query, bank_index] = bank["candidate_success_available"]
    arrays["teacher_selected_action"][query, bank_index] = bank["teacher_selected_action"]


def finalize_dataset_arrays(arrays: dict[str, np.ndarray]) -> None:
    keys, mask = arrays["candidate_objective_keys"], arrays["candidate_objective_mask"]
    if not np.any(mask) or np.any(~np.isfinite(keys[mask])) or np.any(np.isfinite(keys[~mask])):
        raise ValueError("candidate objective arrays contain malformed valid/padded entries")
    if not np.all(arrays["candidate_mask"]) or not np.array_equal(
        arrays["candidate_action_ids"], np.broadcast_to(np.arange(9), arrays["candidate_action_ids"].shape)
    ):
        raise ValueError("student runtime candidate set is not the complete nine-action set")


def collect_balanced_teacher_dataset(
    seeds: list[int], layout_hashes: list[str], *, split: str, master_seed: int,
    banks: int = 12, suffixes_per_action: int = 14, time_cap_seconds: int = 10800,
    geometry_resolution: float = 0.04, waypoint_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Collect balanced-teacher visited queries and independently resampled target banks."""
    if split not in {"model_selection", "offline_gate"} or len(seeds) != len(layout_hashes):
        raise ValueError("invalid split or seed/hash list")
    started = time.monotonic()
    states: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    bank_rows: list[list[dict[str, Any]]] = []
    for seed, expected_hash in zip(seeds, layout_hashes):
        scenario = generate_staggered_corridor(int(seed))
        if scenario.layout_sha256 != expected_hash:
            raise ValueError("generated layout differs from frozen hash-only audit")
        state = scenario.initial_state
        decision_index = 0
        while state.status == "active":
            if time.monotonic() - started > time_cap_seconds:
                raise TimeoutError(f"Phase 12c-C {split} collection time cap reached")
            query_index = len(states)
            states.append(state_features(scenario, state))
            route = grid_astar_route((state.x, state.y), scenario.spec.goal, scenario.spec,
                                     resolution=geometry_resolution)
            if route is None:
                raise RuntimeError("geometry baseline could not find a route on a valid corridor")
            geometry_action = waypoint_braking_action(
                state, route, scenario.spec, waypoint_tolerance=waypoint_tolerance
            )
            rollout_seed = derive_bank_seed(master_seed, f"{split}_rollout", expected_hash,
                                             decision_index, 0)
            teacher_decision = _phase12c_shooting_decision(
                state, scenario.spec, horizon=8, seed=rollout_seed, balanced=True,
                suffixes_per_action=suffixes_per_action,
            )
            metadata.append({
                "query_index": query_index,
                "episode_seed": int(seed),
                "layout_sha256": expected_hash,
                "decision_index": decision_index,
                "source": "balanced_teacher_closed_loop",
                "teacher_action": int(teacher_decision.first_action),
                "geometry_action": int(geometry_action),
                "rollout_proposal_seed": rollout_seed,
                "outcome_at_collection_state": state.status,
            })
            banks_for_query: list[dict[str, Any]] = []
            for bank_index in range(banks):
                bank_seed = derive_bank_seed(master_seed, split, expected_hash, decision_index, bank_index)
                banks_for_query.append(balanced_candidate_bank(
                    scenario, state, seed=bank_seed, suffixes_per_action=suffixes_per_action
                ))
            bank_rows.append(banks_for_query)
            state = corridor_step(state, teacher_decision.first_action, scenario.spec).state
            decision_index += 1
        episodes.append({
            "seed": int(seed),
            "layout_sha256": expected_hash,
            "outcome": state.status,
            "steps": state.elapsed_steps - scenario.initial_state.elapsed_steps,
            "decision_queries": decision_index,
        })
    arrays = allocate_dataset_arrays(len(states), banks)
    arrays["states"][:] = np.asarray(states, dtype=np.float32)
    for query_index, row in enumerate(bank_rows):
        for bank_index, bank in enumerate(row):
            put_bank_in_arrays(arrays, query_index, bank_index, bank)
    finalize_dataset_arrays(arrays)
    return {
        **arrays,
        "metadata": metadata,
        "episodes": episodes,
        "elapsed_wall_seconds": time.monotonic() - started,
    }


def target_arrays_from_dataset(arrays: dict[str, np.ndarray], *, target_banks: tuple[int, ...] = (0, 1, 2, 3, 4, 5)) -> dict[str, np.ndarray]:
    keys, mask = arrays["candidate_objective_keys"], arrays["candidate_objective_mask"]
    count, banks = keys.shape[:2]
    group_keys = np.full((count, banks, 9, 3), np.nan, dtype=np.float64)
    bank_regret = np.full((count, banks, 9), np.nan, dtype=np.float32)
    bank_preference = np.full_like(bank_regret, np.nan)
    for query in range(count):
        for bank_index in range(banks):
            suffix_count = int(arrays["suffix_count"][query, bank_index])
            candidate_count = 9 * suffix_count
            if int(mask[query, bank_index].sum()) != candidate_count:
                raise ValueError("candidate mask and suffix count disagree")
            group_keys[query, bank_index] = reduce_candidate_keys(
                keys[query, bank_index, :candidate_count], suffix_count
            )
            bank_regret[query, bank_index], bank_preference[query, bank_index] = bank_regret_and_preference(
                group_keys[query, bank_index]
            )
    chosen = np.asarray(target_banks, dtype=np.int64)
    if len(chosen) == 0 or np.any(chosen < 0) or np.any(chosen >= banks):
        raise ValueError("target bank selection outside available range")
    return {
        "group_objective_keys": group_keys,
        "per_bank_normalized_regret": bank_regret,
        "per_bank_tie_split_optimal_mass": bank_preference,
        "mean_regret_target": np.nanmean(bank_regret[:, chosen], axis=1).astype(np.float32),
        "preference_target": np.nanmean(bank_preference[:, chosen], axis=1).astype(np.float32),
        "train_bank_indices": chosen,
        "heldout_bank_indices": np.asarray([i for i in range(banks) if i not in set(chosen)], dtype=np.int64),
    }


def persist_teacher_dataset(
    output: Path, dataset: dict[str, Any], *, split: str, lock_sha256: str,
    config_sha256: str, max_bytes: int,
) -> dict[str, Any]:
    """Write deployable inputs separately from complete teacher objective labels."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    input_path = output / "student_inputs.npz"
    target_path = output / "teacher_targets.npz"
    metadata_path = output / "sample_metadata.json"
    episodes_path = output / "episodes.json"
    np.savez_compressed(
        input_path, states=dataset["states"], candidate_action_ids=dataset["candidate_action_ids"],
        candidate_mask=dataset["candidate_mask"],
    )
    targets = target_arrays_from_dataset(dataset)
    excluded = {"states", "candidate_action_ids", "candidate_mask", "metadata", "episodes", "elapsed_wall_seconds"}
    teacher_fields = {key: value for key, value in dataset.items() if key not in excluded}
    np.savez_compressed(target_path, **teacher_fields, **targets)
    metadata_path.write_text(
        json.dumps(dataset["metadata"], separators=(",", ":")) + "\n", encoding="utf-8"
    )
    episode_rows = dataset.get("episodes", [])
    episodes_path.write_text(json.dumps(episode_rows, separators=(",", ":")) + "\n", encoding="utf-8")
    files: dict[str, dict[str, Any]] = {}
    for path in (input_path, target_path, metadata_path, episodes_path):
        files[path.name] = {"bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper()}
    bank_seeds = dataset["proposal_seed"]
    manifest: dict[str, Any] = {
        "schema": "phase12cC_teacher_dataset_v1",
        "split": split,
        "query_count": int(len(dataset["states"])),
        "layout_count": len({row["layout_sha256"] for row in dataset["metadata"]}),
        "bank_count_per_query": int(bank_seeds.shape[1]),
        "unique_bank_seed_count": int(np.unique(bank_seeds).size),
        "expected_bank_seed_count": int(bank_seeds.size),
        "unique_bank_seeds": bool(np.unique(bank_seeds).size == bank_seeds.size),
        "unique_query_layout_decisions": len({
            (row["layout_sha256"], row["decision_index"]) for row in dataset["metadata"]
        }) == len(dataset["metadata"]),
        "student_input_allowlist": ["states", "candidate_action_ids", "candidate_mask"],
        "teacher_targets_separate": True,
        "source_lock_sha256": lock_sha256,
        "config_sha256": config_sha256,
        "reserve_accessed": False,
        "offline_gate_labels_collected": split == "offline_gate",
        "closed_loop_development_accessed": False,
        "dagger_accessed": False,
        "elapsed_wall_seconds": float(dataset["elapsed_wall_seconds"]),
        "files": files,
        "artifact_bytes": 0,
    }
    manifest_path = output / "manifest.json"
    for _ in range(4):
        manifest["artifact_bytes"] = sum(item["bytes"] for item in files.values()) + (
            manifest_path.stat().st_size if manifest_path.exists() else 0
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["artifact_bytes"] = sum(item["bytes"] for item in files.values()) + manifest_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if manifest["artifact_bytes"] > max_bytes:
        raise OSError(f"Phase 12c-C {split} generated-artifact cap exceeded")
    return manifest
