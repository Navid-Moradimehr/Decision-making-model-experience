"""Repeated-bank teacher realizability audit for Phase 12c-B.

This module replays only previously collected fitting episodes, selects a
frozen state sample, and evaluates balanced proposal banks at those fixed
states. It never reads Phase 12c-A outcomes as labels or touches reserve data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
import yaml

from consequence_lab.inertial_corridor import CorridorState, corridor_step
from consequence_lab.phase12_teacher import PAD_ACTION, MAX_HORIZON, evaluate_candidate_set, state_features
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.phase12c import balanced_continuation_suffixes


SPLITS = {
    "phase12_train_teacher_v1": {"start": 140000, "count": 120, "selected": 60},
    "phase12_validation_teacher_v1": {"start": 141000, "count": 40, "selected": 20},
}


@dataclass(frozen=True)
class ReplayedQuery:
    split: str
    episode_seed: int
    layout_sha256: str
    decision_index: int
    global_query_index: int
    progress: float
    normalized_speed: float
    scenario: Any
    state: CorridorState


def _stable_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def replay_fitting_queries(root: Path) -> dict[str, dict[int, list[ReplayedQuery]]]:
    """Reconstruct all train/former-validation states, failing on any mismatch."""
    root = root.resolve()
    result: dict[str, dict[int, list[ReplayedQuery]]] = {}
    for split, spec in SPLITS.items():
        directory = root / "artifacts/runs" / split
        inputs = np.load(directory / "student_inputs.npz", allow_pickle=False)
        metadata = np.load(directory / "sample_metadata.npz", allow_pickle=False)
        targets = np.load(directory / "teacher_targets.npz", allow_pickle=False)
        episodes = json.loads((directory / "episodes.json").read_text(encoding="utf-8"))
        expected_seeds = list(range(spec["start"], spec["start"] + spec["count"]))
        if [int(item["seed"]) for item in episodes] != expected_seeds:
            raise ValueError(f"unexpected seed block in {split}")
        by_seed: dict[int, list[ReplayedQuery]] = {}
        query_offset = 0
        for layout_index, episode in enumerate(episodes):
            scenario = generate_staggered_corridor(int(episode["seed"]))
            if scenario.layout_sha256 != episode["layout_sha256"]:
                raise ValueError("layout hash changed while replaying fitting data")
            state = scenario.initial_state
            count = int(episode["decision_queries"])
            rows: list[ReplayedQuery] = []
            for local_index in range(count):
                query_index = query_offset + local_index
                if (int(metadata["episode_seed"][query_index]) != int(episode["seed"])
                        or int(metadata["decision_index"][query_index]) != local_index
                        or int(metadata["layout_index"][query_index]) != layout_index):
                    raise ValueError("query metadata does not align with its source episode")
                if not np.array_equal(state_features(scenario, state), inputs["states"][query_index]):
                    raise ValueError("state does not exactly replay from archived teacher actions")
                if not state.status == "active":
                    raise ValueError("archived decision state is not active")
                rows.append(ReplayedQuery(
                    split=split,
                    episode_seed=int(episode["seed"]),
                    layout_sha256=episode["layout_sha256"],
                    decision_index=local_index,
                    global_query_index=query_index,
                    progress=local_index / max(1, count - 1),
                    normalized_speed=math.hypot(state.vx, state.vy) / scenario.spec.max_speed,
                    scenario=scenario,
                    state=state,
                ))
                chosen = int(targets["teacher_selected_index"][query_index])
                if not 0 <= chosen < int(np.sum(inputs["candidate_mask"][query_index])):
                    raise ValueError("archived teacher choice points outside the candidate mask")
                action = int(inputs["sequences"][query_index, chosen, 0])
                state = corridor_step(state, action, scenario.spec).state
            if (state.status != episode["outcome"]
                    or state.elapsed_steps - scenario.initial_state.elapsed_steps != int(episode["steps"])):
                raise ValueError("episode status/length does not replay exactly")
            by_seed[int(episode["seed"])] = rows
            query_offset += count
        if query_offset != int(inputs["states"].shape[0]):
            raise ValueError(f"unmatched query data in {split}")
        result[split] = by_seed
    return result


def _speed_halves(rows: list[ReplayedQuery]) -> dict[str, list[ReplayedQuery]]:
    ordered = sorted(rows, key=lambda row: (row.normalized_speed, row.decision_index))
    cut = (len(ordered) + 1) // 2
    if cut <= 0 or cut >= len(ordered):
        raise ValueError("progress half must contain at least two states for speed stratification")
    return {"low": ordered[:cut], "high": ordered[cut:]}


def select_state_samples(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Select exactly two states/layout with counterbalanced progress-speed strata."""
    queries = replay_fitting_queries(root)
    selection = config["state_sampling"]
    selection_seed = int(selection["selection_seed"])
    chosen_layouts: list[tuple[str, int, str]] = []
    for split, split_spec in SPLITS.items():
        eligible = [
            (seed, rows[0].layout_sha256)
            for seed, rows in queries[split].items()
        ]
        take = int(split_spec["selected"])
        ordered = sorted(
            eligible,
            key=lambda item: (_stable_digest(f"layout|{selection_seed}|{split}|{item[0]}|{item[1]}"), item[0]),
        )
        chosen_layouts.extend((split, seed, layout_hash) for seed, layout_hash in ordered[:take])
    chosen_layouts.sort(
        key=lambda item: (_stable_digest(f"strata|{selection_seed}|{item[0]}|{item[1]}|{item[2]}"), item[1])
    )
    if len(chosen_layouts) != 80:
        raise ValueError("Phase 12c-B protocol requires exactly 80 source layouts")
    samples: list[dict[str, Any]] = []
    patterns = (
        ("low", "high"),
        ("high", "low"),
        ("low", "low"),
        ("high", "high"),
    )
    for ordinal, (split, seed, layout_hash) in enumerate(chosen_layouts):
        rows = queries[split][seed]
        midpoint = len(rows) // 2
        progress_groups = {
            "early": rows[:midpoint],
            "late": rows[midpoint:],
        }
        speed_pattern = patterns[ordinal % len(patterns)]
        for progress_bin, target_progress, speed_bin in (
            ("early", 0.25, speed_pattern[0]),
            ("late", 0.75, speed_pattern[1]),
        ):
            pool = _speed_halves(progress_groups[progress_bin])[speed_bin]
            target = min(
                pool,
                key=lambda row: (
                    abs(row.progress - target_progress),
                    abs(row.normalized_speed - float(np.median([item.normalized_speed for item in pool]))),
                    row.decision_index,
                ),
            )
            samples.append({
                "sample_id": f"{ordinal:03d}_{progress_bin}",
                "layout_ordinal": ordinal,
                "source_split": split,
                "episode_seed": seed,
                "layout_sha256": layout_hash,
                "decision_index": target.decision_index,
                "global_query_index": target.global_query_index,
                "episode_decision_count": len(rows),
                "progress": target.progress,
                "progress_bin": progress_bin,
                "speed_stratum": speed_bin,
                "normalized_speed": target.normalized_speed,
                "state_features_sha256": hashlib.sha256(
                    state_features(target.scenario, target.state).tobytes()
                ).hexdigest().upper(),
            })
    if len(samples) != 160 or len({(r["episode_seed"], r["decision_index"]) for r in samples}) != 160:
        raise ValueError("state sample count or uniqueness invariant failed")
    strata_counts = {
        f"{progress}_{speed}": sum(
            row["progress_bin"] == progress and row["speed_stratum"] == speed
            for row in samples
        )
        for progress in ("early", "late") for speed in ("low", "high")
    }
    if any(count != 40 for count in strata_counts.values()):
        raise ValueError(f"progress/speed strata are not exactly balanced: {strata_counts}")
    return samples


def _bank_seed(master_seed: int, sample: dict[str, Any], bank_index: int) -> int:
    material = (
        f"{master_seed}|{sample['sample_id']}|{sample['layout_sha256']}|"
        f"{sample['episode_seed']}|{sample['decision_index']}|{bank_index}"
    )
    return int.from_bytes(_stable_digest(material)[:8], "big", signed=False)


def _average_ranks(keys: list[tuple[int, float, float]]) -> tuple[list[float], list[float], list[int]]:
    ranks: list[float] = []
    for key in keys:
        strictly_better = sum(other < key for other in keys)
        tied = sum(other == key for other in keys)
        ranks.append(strictly_better + (tied + 1) / 2.0)
    best = min(ranks)
    regrets = [(rank - best) / 8.0 for rank in ranks]
    best_key = min(keys)
    optimal = [index for index, key in enumerate(keys) if key == best_key]
    return ranks, regrets, optimal


def evaluate_balanced_bank(
    scenario: Any,
    state: CorridorState,
    *,
    seed: int,
    suffixes_per_action: int = 14,
) -> dict[str, Any]:
    horizon = min(MAX_HORIZON, scenario.spec.max_steps - state.elapsed_steps)
    suffixes = balanced_continuation_suffixes(
        horizon, suffixes_per_action=suffixes_per_action, seed=seed
    )
    sequences = np.full((9 * len(suffixes), MAX_HORIZON), PAD_ACTION, dtype=np.uint8)
    for action in range(9):
        for suffix_index, suffix in enumerate(suffixes):
            sequence = (action,) + suffix
            row_index = action * len(suffixes) + suffix_index
            sequences[row_index, :horizon] = sequence
    query = evaluate_candidate_set(state, scenario, sequences)
    group_keys: list[tuple[int, float, float]] = []
    group_ties: list[int] = []
    for action in range(9):
        begin, end = action * len(suffixes), (action + 1) * len(suffixes)
        values = list(query.objective_keys[begin:end])
        key = min(values)
        group_keys.append(key)
        group_ties.append(sum(value == key for value in values))
    ranks, regrets, optimal = _average_ranks(group_keys)
    bank_hash = hashlib.sha256(sequences.tobytes()).hexdigest().upper()
    return {
        "proposal_seed": int(seed),
        "proposal_bank_sha256": bank_hash,
        "horizon": horizon,
        "suffixes_per_action": len(suffixes),
        "candidate_count": int(len(sequences)),
        "candidate_success_available": bool(query.success_available),
        "group_objective_keys": [list(key) for key in group_keys],
        "group_best_candidate_ties": group_ties,
        "group_average_ranks": ranks,
        "group_normalized_regrets": regrets,
        "optimal_action_ids": optimal,
    }


def evaluate_repeated_banks(
    root: Path,
    samples: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    record_callback,
    time_cap_seconds: int,
) -> dict[str, Any]:
    started = time.monotonic()
    replayed = replay_fitting_queries(root)
    master_seed = int(config["bank_sampling"]["master_seed"])
    bank_count = int(config["bank_sampling"]["banks_per_state"])
    records_written = 0
    bank_seed_values: set[int] = set()
    for sample in samples:
        source = replayed[sample["source_split"]][int(sample["episode_seed"])][int(sample["decision_index"])]
        if source.layout_sha256 != sample["layout_sha256"]:
            raise ValueError("selected state provenance changed after sample manifest creation")
        feature_hash = hashlib.sha256(state_features(source.scenario, source.state).tobytes()).hexdigest().upper()
        if feature_hash != sample["state_features_sha256"]:
            raise ValueError("selected state feature hash mismatch")
        for bank_index in range(bank_count):
            if time.monotonic() - started > time_cap_seconds:
                raise TimeoutError("Phase 12c-B wall-time cap reached")
            seed = _bank_seed(master_seed, sample, bank_index)
            if seed in bank_seed_values:
                raise ValueError("proposal-bank seed collision")
            bank_seed_values.add(seed)
            record = dict(sample)
            record["bank_index"] = bank_index
            record.update(evaluate_balanced_bank(
                source.scenario, source.state, seed=seed,
                suffixes_per_action=int(config["bank_sampling"]["suffixes_per_action"]),
            ))
            record_callback(record)
            records_written += 1
    return {
        "completed_state_count": len(samples),
        "completed_bank_records": records_written,
        "unique_proposal_bank_seeds": len(bank_seed_values),
        "elapsed_wall_seconds": time.monotonic() - started,
    }


def sample_manifest_payload(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    samples = select_state_samples(root, config)
    return {
        "schema": "phase12c_b_state_sample_manifest_v1",
        "state_sampling": config["state_sampling"],
        "sample_count": len(samples),
        "layout_count": len({row["episode_seed"] for row in samples}),
        "source_split_layout_counts": {
            split: len({row["episode_seed"] for row in samples if row["source_split"] == split})
            for split in SPLITS
        },
        "progress_speed_strata_counts": {
            f"{progress}_{speed}": sum(
                row["progress_bin"] == progress and row["speed_stratum"] == speed
                for row in samples
            )
            for progress in ("early", "late") for speed in ("low", "high")
        },
        "samples": samples,
    }


def authorize_phase12c_b(
    root: Path,
    config_path: Path,
    lock_path: Path,
    sample_manifest_path: Path,
    output_path: Path,
    *,
    confirmed: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    config_path, lock_path = config_path.resolve(), lock_path.resolve()
    sample_manifest_path, output_path = sample_manifest_path.resolve(), output_path.resolve()
    if not confirmed:
        raise ValueError("explicit Phase 12c-B execution confirmation required")
    if output_path.exists() or not output_path.is_relative_to(root / "artifacts/runs"):
        raise ValueError("output must be a new path under artifacts/runs")
    if any(root not in path.parents for path in (config_path, lock_path, sample_manifest_path)):
        raise ValueError("config, lock, and sample manifest must remain inside the project")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if config.get("phase") != "12c-B" or lock.get("phase") != "12c-B":
        raise ValueError("wrong phase authorization")
    if lock.get("authorization_status") != "authorized" or lock.get("training_authorized") is not False:
        raise ValueError("lock does not authorize only the repeated-bank audit")
    if lock.get("reserve_accessed") is not False or lock.get("dagger_authorized") is not False:
        raise ValueError("lock exceeds Phase 12c-B scope")
    if (config.get("scope", {}).get("reserve_accessed") is not False
            or config.get("scope", {}).get("student_training_authorized") is not False
            or config.get("scope", {}).get("dagger_authorized") is not False
            or config.get("scope", {}).get("development_or_test_layouts_read") is not False):
        raise ValueError("config does not maintain the Phase 12c-B information/scope boundary")
    caps = lock.get("caps", {})
    expected_caps = {
        "wall_seconds": int(config["compute_caps"]["wall_seconds"]),
        "artifact_bytes": int(config["compute_caps"]["generated_artifact_bytes"]),
    }
    if caps != expected_caps:
        raise ValueError("execution lock caps differ from frozen config")
    if lock.get("scope", {}).get("expected_bank_records") != int(config["runner_contract"]["expected_bank_records"]):
        raise ValueError("lock record count differs from frozen protocol")
    if _sha256(config_path) != lock.get("config_sha256"):
        raise ValueError("locked config hash mismatch")
    replay_audit_path = (root / config["source_data"]["replay_audit"]).resolve()
    training_lock_path = (root / config["source_data"]["execution_lock"]).resolve()
    if (root not in replay_audit_path.parents or not replay_audit_path.is_file()
            or _sha256(replay_audit_path) != lock.get("replay_audit_sha256")
            or _sha256(replay_audit_path) != config["source_data"]["replay_audit_sha256"]):
        raise ValueError("exact-replay audit hash mismatch")
    if (root not in training_lock_path.parents or not training_lock_path.is_file()
            or _sha256(training_lock_path) != lock.get("training_source_lock_sha256")
            or _sha256(training_lock_path) != config["source_data"]["execution_lock_sha256"]):
        raise ValueError("fitting-data source lock hash mismatch")
    if sample_manifest_path != (root / config["state_sampling"]["sample_manifest"]).resolve():
        raise ValueError("sample manifest path differs from locked config")
    if _sha256(sample_manifest_path) != lock.get("sample_manifest_sha256"):
        raise ValueError("sample manifest hash mismatch")
    for relative, expected in lock.get("source_sha256", {}).items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source hash mismatch: {relative}")
    for relative, expected in lock.get("input_data_sha256", {}).items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"input-data hash mismatch: {relative}")
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    if sample_manifest.get("sample_count") != 160 or sample_manifest.get("layout_count") != 80:
        raise ValueError("sample manifest does not match frozen 160-state/80-layout scope")
    if sample_manifest.get("source_data_replay_audit_sha256") != lock.get("replay_audit_sha256"):
        raise ValueError("sample manifest is not linked to the exact-replay audit")
    return config, lock, sample_manifest


def run_phase12c_b(
    root: Path,
    config_path: Path,
    lock_path: Path,
    sample_manifest_path: Path,
    output_path: Path,
    *,
    confirmed: bool,
) -> dict[str, Any]:
    config, lock, sample_manifest = authorize_phase12c_b(
        root, config_path, lock_path, sample_manifest_path, output_path,
        confirmed=confirmed,
    )
    root, output_path = root.resolve(), output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    records_path = output_path / "bank_records.jsonl"
    manifest_path = output_path / "manifest.json"
    started = time.monotonic()
    cap_seconds = int(lock["caps"]["wall_seconds"])
    cap_bytes = int(lock["caps"]["artifact_bytes"])
    bank_parameters = config["bank_sampling"]
    manifest: dict[str, Any] = {
        "schema": "phase12c_b_run_manifest_v1",
        "experiment": config["experiment"],
        "phase": "12c-B",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        },
        "authorization_reference": lock["approval_reference"],
        "authorization_lock_sha256": _sha256(lock_path),
        "config_sha256": _sha256(config_path),
        "sample_manifest_sha256": _sha256(sample_manifest_path),
        "replay_audit_sha256": lock["replay_audit_sha256"],
        "training_source_lock_sha256": lock["training_source_lock_sha256"],
        "input_data_sha256": lock["input_data_sha256"],
        "source_sha256": lock["source_sha256"],
        "state_count": len(sample_manifest["samples"]),
        "layout_count": sample_manifest["layout_count"],
        "banks_per_state": int(bank_parameters["banks_per_state"]),
        "expected_bank_records": int(config["runner_contract"]["expected_bank_records"]),
        "reserve_accessed": False,
        "student_training_performed": False,
        "dagger_performed": False,
        "errors": [],
        "completed_bank_records": 0,
        "artifact_bytes": 0,
    }

    def write_manifest() -> None:
        manifest["elapsed_wall_seconds"] = time.monotonic() - started
        current = records_path.stat().st_size if records_path.exists() else 0
        for _ in range(3):
            manifest["artifact_bytes"] = current + (manifest_path.stat().st_size if manifest_path.exists() else 0)
            rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            if len(rendered.encode("utf-8")) > int(config["runner_contract"]["maximum_manifest_bytes"]):
                raise OSError("manifest size cap exceeded")
            manifest_path.write_text(rendered, encoding="utf-8")
            exact_bytes = current + manifest_path.stat().st_size
            if exact_bytes == manifest["artifact_bytes"]:
                break
            manifest["artifact_bytes"] = exact_bytes
        manifest["artifact_bytes"] = current + manifest_path.stat().st_size
        if manifest["artifact_bytes"] > cap_bytes:
            raise OSError("generated artifact byte cap reached")

    try:
        write_manifest()
        with records_path.open("w", encoding="utf-8", newline="\n") as stream:
            def callback(record: dict[str, Any]) -> None:
                line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                if len(line.encode("utf-8")) > int(config["runner_contract"]["maximum_record_bytes"]):
                    raise OSError("bank record size cap exceeded")
                stream.write(line)
                stream.flush()
                manifest["completed_bank_records"] += 1
                if time.monotonic() - started > cap_seconds:
                    raise TimeoutError("Phase 12c-B wall-time cap reached")
                write_manifest()

            result = evaluate_repeated_banks(
                root, sample_manifest["samples"], config,
                record_callback=callback, time_cap_seconds=cap_seconds,
            )
        manifest.update(result)
        manifest["elapsed_wall_seconds"] = time.monotonic() - started
        manifest["status"] = "completed_phase12c_b_only"
        write_manifest()
        if manifest["completed_bank_records"] != manifest["expected_bank_records"]:
            raise ValueError("completed record count differs from frozen expectation")
        return manifest
    except Exception as exc:
        manifest["status"] = "failed_phase12c_b"
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        try:
            write_manifest()
        except Exception:
            pass
        raise
