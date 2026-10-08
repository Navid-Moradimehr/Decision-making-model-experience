"""Read-only support audit for the frozen Phase14F CART.

This diagnostic reads only the train-only fit records, already-open Phase14D
offline-gate labels, and completed Phase14F summaries. It never generates a
layout, advances the simulator, fits a model, or reads reserve data.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase14c import build_feature_matrix, extract_arrays


CONFIG = "configs/phase14f_closed_loop.yaml"
DEFAULT_OUTPUT = "artifacts/runs/phase14f_postmortem_v1/tree_support.json"


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _leaf(policy: dict[str, Any], raw: list[float] | np.ndarray) -> tuple[int, int]:
    selected = policy["selected_model"]
    x = build_feature_matrix(np.asarray(raw, dtype=np.float64)[None, :], selected["feature_variant"])[0]
    x = (x - np.asarray(policy["x_mean"], dtype=np.float64)) / np.asarray(policy["x_std"], dtype=np.float64)
    nodes = policy["nodes"]
    index = 0
    visited: set[int] = set()
    while True:
        if index in visited or index < 0 or index >= len(nodes):
            raise ValueError("frozen CART contains a cycle or invalid node reference")
        visited.add(index)
        node = nodes[index]
        feature = int(node["feature"])
        if feature < 0:
            value = np.asarray(node["value"], dtype=np.float64)
            if value.shape != (9,) or not np.isfinite(value).all():
                raise ValueError("invalid multi-action CART leaf")
            return index, int(np.argmin(value))
        if feature >= len(x):
            raise ValueError("CART split feature is outside its input dimensions")
        child = node["left"] if x[feature] <= float(node["threshold"]) else node["right"]
        if child is None:
            raise ValueError("internal CART node has a missing child")
        index = int(child)


def _distribution(counts: Counter[int], total: int) -> dict[str, Any]:
    values = np.asarray(list(counts.values()), dtype=np.int64)
    return {
        "occupied_leaf_count": len(counts),
        "sample_count": int(total),
        "min": int(values.min()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p90": float(np.quantile(values, 0.90)) if len(values) else None,
        "max": int(values.max()) if len(values) else None,
        "singleton_leaf_count": int(np.sum(values == 1)),
        "singleton_leaf_fraction": float(np.mean(values == 1)) if len(values) else None,
    }


def analyze(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    artifact_path = root / config["model"]["artifact"]
    model_manifest_path = root / config["outputs"]["model_freeze_manifest"]
    result_dir = root / config["outputs"]["run_dir"]
    model = json.loads(artifact_path.read_text(encoding="utf-8"))
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    result_manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    result_metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    episodes_path = result_dir / "episodes.jsonl"

    offline_labels_path = root / config["parent"]["phase14d_offline_run_dir"] / "teacher_labels.jsonl"
    offline_result_dir = offline_labels_path.parent
    offline_manifest = json.loads((offline_result_dir / "manifest.json").read_text(encoding="utf-8"))
    offline_metrics_path = offline_result_dir / "metrics.json"
    offline_metrics = json.loads(offline_metrics_path.read_text(encoding="utf-8"))
    train_path = root / config["parent"]["training_records"]
    if (hashlib.sha256(artifact_path.read_bytes()).hexdigest().upper() != model_manifest["model_sha256"]
            or hashlib.sha256(train_path.read_bytes()).hexdigest().upper() != model["training_records_sha256"]
            or hashlib.sha256(offline_labels_path.read_bytes()).hexdigest().upper() != model["offline_labels_sha256"]
            or hashlib.sha256(offline_labels_path.read_bytes()).hexdigest().upper() != offline_manifest["teacher_labels_sha256"]
            or hashlib.sha256(offline_metrics_path.read_bytes()).hexdigest().upper() != offline_manifest["metrics_sha256"]
            or hashlib.sha256(episodes_path.read_bytes()).hexdigest().upper() != result_manifest["episodes_sha256"]
            or hashlib.sha256((result_dir / "metrics.json").read_bytes()).hexdigest().upper() != result_manifest["metrics_sha256"]
            or result_manifest["status"] != "completed_closed_loop_development"
            or result_metrics["model_sha256"] != model_manifest["model_sha256"]):
        raise ValueError("Phase14F parent/result artifact hash or status mismatch")

    train_rows = _rows(train_path)
    offline_rows = _rows(offline_labels_path)
    phase14f_episodes = _rows(episodes_path)
    student_decision_counts = [
        int(row["decision_count"]) for row in phase14f_episodes if row["method"] == "student"
    ]
    if len(student_decision_counts) != result_metrics["student"]["episodes"]:
        raise ValueError("Phase14F student episode count does not match metrics")
    if len(phase14f_episodes) != result_manifest["episode_record_count"]:
        raise ValueError("Phase14F episode record count does not match result manifest")
    train_raw, _, _ = extract_arrays(train_rows)
    train_leaf_counts: Counter[int] = Counter(_leaf(model, row)[0] for row in train_raw)
    offline_leaf_rows = [(_leaf(model, row["student_input_52"]), row) for row in offline_rows]
    offline_reached_support = [train_leaf_counts[leaf] for (leaf, _), _row in offline_leaf_rows]
    offline_regrets = np.asarray([
        [float(action["normalized_regret"]) for action in row["actions"]]
        for row in offline_rows
    ], dtype=np.float64)
    offline_actions = np.asarray([action for (_, action), _row in offline_leaf_rows], dtype=np.int64)
    selected_regret = offline_regrets[np.arange(len(offline_rows)), offline_actions]
    offline_by_layout: dict[str, list[int]] = {}
    for index, row in enumerate(offline_rows):
        offline_by_layout.setdefault(row["source_layout_sha256"], []).append(index)
    layout_macro_regret = float(np.mean([
        selected_regret[indices].mean() for indices in offline_by_layout.values()
    ]))
    node_count = int(model["node_count"])
    leaf_count = sum(int(int(node["feature"]) < 0) for node in model["nodes"])
    if len(model["nodes"]) != node_count or node_count != 2 * leaf_count - 1:
        raise ValueError("frozen CART node/leaf structure is inconsistent with a full binary tree")
    reached_support = np.asarray(offline_reached_support, dtype=np.int64)
    return {
        "schema": "phase14f_tree_support_diagnostic_v1",
        "scope": {
            "training_records_read": True,
            "already_open_offline_labels_read": True,
            "completed_phase14f_metrics_read": True,
            "new_layouts_generated": False,
            "simulator_transitions": False,
            "model_fit_or_tuning": False,
            "reserve_access": False,
        },
        "artifacts": {
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper(),
            "closed_loop_config_sha256": hashlib.sha256((root / CONFIG).read_bytes()).hexdigest().upper(),
            "student_model_sha256": model_manifest["model_sha256"],
            "training_records_sha256": model["training_records_sha256"],
            "offline_labels_sha256": model["offline_labels_sha256"],
            "offline_metrics_sha256": offline_manifest["metrics_sha256"],
            "phase14f_lock_sha256": result_manifest["lock_sha256"],
            "phase14f_metrics_sha256": result_manifest["metrics_sha256"],
        },
        "model": {
            "node_count": node_count,
            "leaf_count": leaf_count,
            "training_state_count": len(train_rows),
            "training_state_to_leaf_ratio": len(train_rows) / leaf_count,
            "training_occupancy_by_leaf": _distribution(train_leaf_counts, len(train_rows)),
        },
        "offline_gate": {
            "state_count": len(offline_rows),
            "distinct_leaves_reached": len({leaf for (leaf, _), _row in offline_leaf_rows}),
            "per_query_training_leaf_support": {
                "min": int(reached_support.min()),
                "median": float(np.median(reached_support)),
                "p90": float(np.quantile(reached_support, 0.90)),
                "max": int(reached_support.max()),
                "queries_in_singleton_training_leaves": int(np.sum(reached_support == 1)),
                "fraction_in_singleton_training_leaves": float(np.mean(reached_support == 1)),
            },
            "reproduced_student_action_frequency": np.bincount(offline_actions, minlength=9).tolist(),
            "reproduced_layout_macro_mean_regret": layout_macro_regret,
            "matches_phase14d_offline_gate": bool(np.isclose(
                layout_macro_regret, offline_metrics["student_layout_macro_mean_regret"], atol=1e-12, rtol=0.0
            )),
        },
        "phase14f_closed_loop": {
            "student_safe_success_rate": result_metrics["student"]["safe_success_rate"],
            "student_collision_rate": result_metrics["student"]["collision_rate"],
            "student_median_decisions": float(np.median(student_decision_counts)),
            "performance_gate_passed": result_metrics["gate_passed"],
        },
        "interpretation": (
            "Tree support is a model-capacity diagnostic, not a causal explanation of the closed-loop failure."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = analyze(args.root)
    output = args.root.resolve() / args.output
    if output.exists():
        raise SystemExit(f"refusing to overwrite diagnostic artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
