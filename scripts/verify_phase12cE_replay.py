"""Replay the locked exact planner on archived Phase 12 episodes and verify state parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase12_teacher import make_teacher_query, state_features
from audit_phase12cE_layouts import verify_lock
from freeze_phase12cE_lock import CONFIG, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    if args.confirmation != "VERIFY_PHASE12CE_REPLAY":
        raise SystemExit("replay parity requires --confirmation VERIFY_PHASE12CE_REPLAY")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lock, lock_path = verify_lock(config)
    source_root = ROOT / "artifacts/runs"
    source_paths = [source_root / "phase12_train_teacher_v1",
                    source_root / "phase12_validation_teacher_v1"]
    references: dict[int, dict[int, np.ndarray]] = {}
    layout_hashes: dict[int, str] = {}
    for directory in source_paths:
        with np.load(directory / "student_inputs.npz", allow_pickle=False) as archive:
            rows = archive["states"]
        with np.load(directory / "sample_metadata.npz", allow_pickle=False) as archive:
            seeds, decisions = archive["episode_seed"], archive["decision_index"]
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("teacher_selected_action_replans_each_step") is not True:
            raise SystemExit(f"archived Phase 12 source is not a receding-horizon dataset: {directory}")
        for state_row, seed_raw, decision_raw in zip(rows, seeds, decisions):
            seed, decision = int(seed_raw), int(decision_raw)
            if decision in references.setdefault(seed, {}):
                raise SystemExit("duplicate episode decision in archived replay source")
            references[seed][decision] = state_row
        for episode in manifest["episodes"]:
            seed = int(episode["seed"])
            layout_hashes[seed] = str(episode["layout_sha256"])

    state_count = 0
    action_count = 0
    episode_count = 0
    for seed in sorted(references):
        scenario = generate_staggered_corridor(seed)
        if scenario.layout_sha256 != layout_hashes[seed]:
            raise SystemExit(f"archived source layout hash mismatch for seed {seed}")
        state = scenario.initial_state
        decision = 0
        while state.status == "active":
            expected = references[seed].get(decision)
            if expected is None:
                raise SystemExit(f"missing archived state for seed {seed}, decision {decision}")
            actual = state_features(scenario, state)
            if not np.array_equal(actual, expected):
                difference = float(np.max(np.abs(actual - expected)))
                raise SystemExit(f"state replay mismatch seed={seed} decision={decision} max_abs={difference}")
            query = make_teacher_query(scenario, state)
            action = int(query.sequences[query.selected_index, 0])
            state = corridor_step(state, action, scenario.spec).state
            decision += 1
            state_count += 1
            action_count += 1
        if decision != len(references[seed]):
            raise SystemExit(f"replay query count differs for seed {seed}")
        episode_count += 1

    output = ROOT / config["seed_splits"]["replay_output"]
    if output.exists():
        raise SystemExit("refusing to overwrite replay parity artifact")
    result = {
        "schema": "phase12cE_replay_parity_v2",
        "status": "passed_exact_state_feature_replay",
        "execution_lock_sha256": digest(lock_path),
        "archived_sources": [str(path.relative_to(ROOT)) for path in source_paths],
        "archived_source_data_sha256": lock["replay_source_data_sha256"],
        "episodes_replayed": episode_count,
        "states_exactly_matched": state_count,
        "exact_planner_actions_replayed": action_count,
        "maximum_absolute_state_feature_difference": 0.0,
        "new_phase12cE_selection_or_gate_outcomes_accessed": False,
        "reserve_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "episodes": episode_count,
                      "states": state_count, "artifact": str(output.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
