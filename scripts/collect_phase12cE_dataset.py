"""Collect one hash-locked Phase 12c-E exact-teacher state/label split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12cE import (
    collect_phase12_visited_balanced_dataset,
    phase12cE_artifact_bytes,
    save_phase12cE_dataset,
)
from audit_phase12cE_layouts import verify_lock
from freeze_phase12cE_lock import CONFIG, LOCK, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("model_selection", "offline_gate"), required=True)
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    expected = ("RUN_PHASE12CE_SELECTION_ONLY" if args.split == "model_selection"
                else "RUN_PHASE12CE_OFFLINE_GATE_ONLY")
    if args.confirmation != expected:
        raise SystemExit(f"collection requires --confirmation {expected}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lock, lock_path = verify_lock(config)
    audit_path = ROOT / config["seed_splits"]["audit_output"]
    if not audit_path.is_file():
        raise SystemExit("run the approved hash-only layout audit before collection")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "phase12cE_layout_audit_v1"
            or audit.get("execution_lock_sha256") != digest(lock_path)
            or audit.get("layout_generation_only_no_rollouts_or_labels") is not True
            or audit.get("reserve_layouts_generated_or_accessed") is not False):
        raise SystemExit("Phase 12c-E layout audit is invalid or not source-matched")
    replay_path = ROOT / config["seed_splits"]["replay_output"]
    if not replay_path.is_file():
        raise SystemExit("run exact archived-trajectory replay parity before fresh collection")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if (replay.get("schema") != "phase12cE_replay_parity_v2"
            or replay.get("status") != "passed_exact_state_feature_replay"
            or replay.get("execution_lock_sha256") != digest(lock_path)
            or replay.get("maximum_absolute_state_feature_difference") != 0.0):
        raise SystemExit("Phase 12c-E archived trajectory replay parity did not pass")
    block = config["seed_splits"][args.split]
    seeds = list(range(int(block["start"]), int(block["start"]) + int(block["count"])))
    if audit["blocks"][args.split]["seeds"] != seeds:
        raise SystemExit("layout audit block does not match approved seed scope")
    if args.split == "offline_gate":
        selected_lock = ROOT / config["seed_splits"]["selected_model_lock"]
        if not selected_lock.is_file():
            raise SystemExit("offline gate collection is forbidden before model lock")
        selected = json.loads(selected_lock.read_text(encoding="utf-8"))
        if selected.get("schema") != "phase12cE_selected_models_lock_v1" or selected.get("status") != "frozen_before_gate":
            raise SystemExit("selected model lock is missing or not frozen")
        if (selected.get("gate_outcomes_accessed") is not False
                or selected.get("reserve_accessed") is not False
                or selected.get("config_sha256") != digest(CONFIG)
                or selected.get("execution_lock_sha256") != digest(lock_path)
                or selected.get("selection_manifest_sha256") != digest(
                    ROOT / config["seed_splits"]["selection_output"] / "manifest.json"
                )
                or selected.get("training_manifest_sha256") != lock["training_manifest_sha256"]
                or set(selected.get("models", {})) != {"absolute", "agent_relative"}):
            raise SystemExit("selected-model lock already records gate access")
        for representation, model in selected["models"].items():
            checkpoint = ROOT / model["checkpoint_path"]
            normalizer = ROOT / model["normalizer_path"]
            if (not checkpoint.is_file() or not normalizer.is_file()
                    or digest(checkpoint) != model["checkpoint_sha256"]
                    or digest(normalizer) != model["normalizer_sha256"]
                    or int(model["parameter_count"]) != int(config["student"]["parameter_count"])):
                raise SystemExit(f"frozen {representation} model artifacts failed pre-gate verification")
    output_key = "selection_output" if args.split == "model_selection" else "gate_output"
    output = ROOT / config["seed_splits"][output_key]
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    hashes = audit["blocks"][args.split]["layout_sha256"]
    dataset = collect_phase12_visited_balanced_dataset(
        seeds, hashes, split=args.split,
        master_seed=int(config["trajectory_collection"]["bank_master_seed"]),
        suffixes_per_action=int(config["trajectory_collection"]["suffixes_per_action"]),
        wall_cap_seconds=None,
    )
    manifest = save_phase12cE_dataset(
        output, dataset, split=args.split, config_sha256=digest(CONFIG),
        execution_lock_sha256=digest(lock_path), layout_audit_sha256=digest(audit_path),
        replay_parity_sha256=digest(replay_path),
        artifact_cap_bytes=int(config["resources"]["generated_artifact_cap_bytes"]),
    )
    # The offline gate split is collected only after model freeze, but its outcomes
    # remain unread by training or selection code until the separate gate runner.
    if args.split == "offline_gate":
        manifest["selected_model_lock_sha256"] = digest(ROOT / config["seed_splits"]["selected_model_lock"])
        path = output / "manifest.json"
        for _ in range(10):
            path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            actual = sum(item.stat().st_size for item in output.iterdir() if item.is_file())
            if actual == manifest["artifact_bytes"]:
                break
            manifest["artifact_bytes"] = actual
        else:
            raise SystemExit("offline gate artifact accounting did not converge")
        total_bytes = sum(item.stat().st_size for item in output.iterdir() if item.is_file())
        if total_bytes > int(config["resources"]["generated_artifact_cap_bytes"]):
            raise SystemExit("Phase 12c-E aggregate gate artifact cap exceeded")
    aggregate_bytes = phase12cE_artifact_bytes(ROOT)
    if aggregate_bytes > int(config["resources"]["generated_artifact_cap_bytes"]):
        raise SystemExit("Phase 12c-E total generated-artifact cap exceeded")
    print(json.dumps({
        "status": f"phase12cE_{args.split}_collection_complete",
        "layouts": len(dataset["episodes"]),
        "queries": len(dataset["metadata"]),
        "wall_seconds": dataset["elapsed_wall_seconds"],
        "artifact_bytes": manifest["artifact_bytes"],
        "reserve_accessed": False,
        "gate_evaluated": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
