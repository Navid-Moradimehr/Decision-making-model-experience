"""Locked Phase15 feature dataset construction and layout-group splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from consequence_lab.phase15 import audit_phase14g_inputs
from consequence_lab.phase15_features import build_phase15_representations


def layout_split(layout_hashes: list[str]) -> dict[str, list[str]]:
    unique = sorted(set(layout_hashes))
    if len(unique) != 100:
        raise ValueError("Phase15 split requires the 100 consumed Phase14G layouts")
    ordered = sorted(unique, key=lambda value: hashlib.sha256(
        ("phase15-layout-split-v1:" + value).encode("ascii")
    ).hexdigest())
    return {"test": ordered[:20], "validation": ordered[20:40], "train": ordered[40:]}


def split_manifest(layouts: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "scheme": "sha256_layout_group_phase15_v1",
        "layout_counts": {name: len(values) for name, values in layouts.items()},
        "layout_hashes": layouts,
        "split_sha256": {
            name: hashlib.sha256("\n".join(values).encode("ascii")).hexdigest().upper()
            for name, values in layouts.items()
        },
    }


def load_locked_inputs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(root).resolve()
    audit = audit_phase14g_inputs(root)
    run_dir = root / "artifacts/runs/phase14g_trace_v1"
    traces = [json.loads(line) for line in (run_dir / "student_decisions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    source = [json.loads(line) for line in (root / "artifacts/runs/phase14b_training_labels_resume_v1/first_action_values_complete.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return traces, source, audit


def build_phase15_dataset(root: Path, output_dir: Path) -> dict[str, Any]:
    """Build R0-R4 on locked saved Phase14G traces; does no episode/planner run."""
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if root not in output_dir.parents:
        raise ValueError("Phase15 output must be inside repository")
    traces, source, audit = load_locked_inputs(root)
    layouts = split_manifest(layout_split([str(row["layout_sha256"]) for row in traces]))
    x, y, r2_transitions = build_phase15_representations(traces)
    trace_layouts = np.asarray([str(row["layout_sha256"]) for row in traces])
    trace_seeds = np.asarray([int(row["episode_seed"]) for row in traces], dtype=np.int64)
    decision_index = np.asarray([int(row["decision_index"]) for row in traces], dtype=np.int64)
    # Preserve source training rows and their same-format action regrets for the
    # equal-size H1 distribution comparison. No Phase14G test rows are read as
    # fitting targets by this function; split membership is stored separately.
    source_x = np.asarray([row["student_input_52"] for row in source], dtype=np.float64)
    source_y = np.asarray([[action["normalized_regret"] for action in row["actions"]] for row in source], dtype=np.float64)
    if source_x.shape != (2095, 52) or source_y.shape != (2095, 9):
        raise ValueError("Phase14B source teacher rows do not match the locked input contract")
    output_dir.mkdir(parents=True, exist_ok=False)
    arrays_path = output_dir / "phase15_features.npz"
    np.savez_compressed(
        arrays_path,
        **{f"x_{name}": values.astype(np.float32) for name, values in x.items()},
        y=y.astype(np.float32), layouts=trace_layouts, seeds=trace_seeds,
        decision_index=decision_index, source_x=source_x.astype(np.float32),
        source_y=source_y.astype(np.float32),
    )
    data_sha = hashlib.sha256(arrays_path.read_bytes()).hexdigest().upper()
    manifest = {
        "schema": "phase15_offline_feature_dataset_v1",
        "status": "built_locked_phase14g_offline_features",
        "phase14g_run_lock_sha256": audit["phase14g_lock_sha256"],
        "phase14g_artifact_sha256": audit["artifact_sha256"],
        "phase14b_training_records_sha256": audit["artifact_sha256"]["training_records"],
        "trace_count": len(traces), "action_branches": len(traces) * 9,
        "r2_horizon": 4, "r2_additional_transitions_max": len(traces) * 9 * 3,
        "r2_additional_transitions_actual": int(r2_transitions),
        "representations": {name: {"dimension": int(values.shape[1]), "runtime_status": (
            "deployable_current_state" if name == "R0" else
            "privileged_exact_transition_diagnostic" if name in {"R1", "R2", "R3"} else
            "non_deployable_exact_continuation_oracle"
        )} for name, values in x.items()},
        "split": layouts,
        "feature_dataset_sha256": data_sha,
        "phase14g_rerun": False, "exact_planner_calls": 0,
        "new_episodes": 0, "reserve_access": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
