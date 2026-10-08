"""Verify fitting-only Phase 12 trajectories can be replayed exactly for 12c-B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase12_teacher import state_features


INPUT_FILES = (
    "student_inputs.npz",
    "sample_metadata.npz",
    "teacher_targets.npz",
    "episodes.json",
    "manifest.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    lock_path = root / "experiments/phase12b_training.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("reserve_accessed") is not False or lock.get("offline_gate_read") is not False:
        raise ValueError("Phase 12b source lock does not attest fitting-only provenance")
    expected_hashes = lock.get("training_data_sha256")
    if not isinstance(expected_hashes, dict):
        raise ValueError("Phase 12b training lock lacks input hashes")

    split_specs = (
        ("phase12_train_teacher_v1", 140000, 120),
        ("phase12_validation_teacher_v1", 141000, 40),
    )
    split_results: dict[str, Any] = {}
    for directory, seed_start, expected_episodes in split_specs:
        split_path = root / "artifacts/runs" / directory
        paths = {name: split_path / name for name in INPUT_FILES}
        if any(not path.is_file() for path in paths.values()):
            raise ValueError(f"missing required fitting-only artifact in {directory}")
        relative_hashes = {
            path.relative_to(root).as_posix(): _sha256(path)
            for path in paths.values()
        }
        for relative, digest in relative_hashes.items():
            if expected_hashes.get(relative) != digest:
                raise ValueError(f"training-lock hash mismatch: {relative}")

        manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
        if (manifest.get("status") != "completed_exact_teacher_collection"
                or manifest.get("reserve_accessed") is not False
                or manifest.get("training_performed") is not False):
            raise ValueError(f"source collection is not a completed fitting-only dataset: {directory}")
        episodes = json.loads(paths["episodes.json"].read_text(encoding="utf-8"))
        if len(episodes) != expected_episodes:
            raise ValueError(f"unexpected episode count in {directory}")
        wanted_seeds = list(range(seed_start, seed_start + expected_episodes))
        if [int(row["seed"]) for row in episodes] != wanted_seeds:
            raise ValueError(f"unexpected seed order in {directory}")

        inputs = np.load(paths["student_inputs.npz"], allow_pickle=False)
        metadata = np.load(paths["sample_metadata.npz"], allow_pickle=False)
        targets = np.load(paths["teacher_targets.npz"], allow_pickle=False)
        query_count = int(inputs["states"].shape[0])
        if any(int(metadata[key].shape[0]) != query_count for key in ("episode_seed", "decision_index", "layout_index")):
            raise ValueError(f"metadata row count mismatch in {directory}")
        if int(targets["teacher_selected_index"].shape[0]) != query_count:
            raise ValueError(f"teacher target row count mismatch in {directory}")

        query_offset = 0
        for layout_index, episode in enumerate(episodes):
            scenario = generate_staggered_corridor(int(episode["seed"]))
            if scenario.layout_sha256 != episode["layout_sha256"]:
                raise ValueError("scenario/layout provenance mismatch")
            state = scenario.initial_state
            count = int(episode["decision_queries"])
            for local_index in range(count):
                query_index = query_offset + local_index
                if (int(metadata["episode_seed"][query_index]) != int(episode["seed"])
                        or int(metadata["decision_index"][query_index]) != local_index
                        or int(metadata["layout_index"][query_index]) != layout_index):
                    raise ValueError("query metadata cannot be aligned to episode replay")
                if not np.array_equal(state_features(scenario, state), inputs["states"][query_index]):
                    raise ValueError("saved deployable state does not exactly replay")
                chosen_index = int(targets["teacher_selected_index"][query_index])
                if not 0 <= chosen_index < int(np.sum(inputs["candidate_mask"][query_index])):
                    raise ValueError("teacher-selected candidate is outside the valid mask")
                action = int(inputs["sequences"][query_index, chosen_index, 0])
                state = corridor_step(state, action, scenario.spec).state
            if (state.status != episode["outcome"]
                    or state.elapsed_steps - scenario.initial_state.elapsed_steps != int(episode["steps"])):
                raise ValueError("saved episode outcome/length does not exactly replay")
            query_offset += count
        if query_offset != query_count:
            raise ValueError(f"unmatched query rows in {directory}")
        split_results[directory] = {
            "seed_start": seed_start,
            "episodes": expected_episodes,
            "decision_states_exactly_replayed": query_count,
            "reserve_accessed": False,
            "files_sha256": relative_hashes,
        }

    return {
        "schema": "phase12c_b_fitting_data_replay_audit_v1",
        "status": "exact_replay_passed",
        "source_lock": "experiments/phase12b_training.lock.json",
        "source_lock_sha256": _sha256(lock_path),
        "splits": split_results,
        "total_episodes": sum(row["episodes"] for row in split_results.values()),
        "total_exactly_replayed_decision_states": sum(
            row["decision_states_exactly_replayed"] for row in split_results.values()
        ),
        "generated_repeated_banks": False,
        "reserve_accessed": False,
        "student_training_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output = output.resolve()
        if output.exists() or ROOT not in output.parents:
            raise SystemExit("refusing overwrite or output outside repository")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
