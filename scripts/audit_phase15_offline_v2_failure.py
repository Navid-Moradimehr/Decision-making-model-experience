"""Read-only provenance and fitting-input audit for the stopped Phase15 v2 run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14c import build_feature_matrix
from consequence_lab.phase15_data import layout_split


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/runs/phase15_offline_v2"
V1 = ROOT / "artifacts/runs/phase15_offline_v1"
PHASE14G_TRACES = ROOT / "artifacts/runs/phase14g_trace_v1/student_decisions.jsonl"
LOCK = ROOT / "experiments/phase15_offline_v2.lock.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit() -> dict:
    result_path = RUN / "result.json"
    result_manifest_path = RUN / "manifest.json"
    feature_manifest_path = RUN / "features/manifest.json"
    feature_path = RUN / "features/phase15_features.npz"
    result = _json(result_path)
    result_manifest = _json(result_manifest_path)
    feature_manifest = _json(feature_manifest_path)
    lock = _json(LOCK)
    v1_failure = _json(V1 / "failure.json")
    traces = _jsonl(PHASE14G_TRACES)

    assert result["status"] == "stopped_memorization_gate_failed"
    assert result["heldout_evaluated"] is False and result["reserve_access"] is False
    assert result_manifest["result_sha256"] == sha256_file(result_path)
    assert result_manifest["run_lock_sha256"] == sha256_file(LOCK)
    assert feature_manifest["copied_npz_sha256"] == sha256_file(feature_path)
    assert feature_manifest["feature_dataset_sha256"] == sha256_file(feature_path)
    assert feature_manifest["copied_npz_sha256"] == sha256_file(V1 / "features/phase15_features.npz")
    assert feature_manifest["new_r2_transitions"] == 0
    assert lock["phase15_v1_failure_record_sha256"] == sha256_file(V1 / "failure.json")
    assert v1_failure["heldout_metrics_computed"] is False

    with np.load(feature_path, allow_pickle=False) as dataset:
        x_r0 = np.asarray(dataset["x_R0"], dtype=np.float64)
        y = np.asarray(dataset["y"], dtype=np.float64)
        layouts = np.asarray(dataset["layouts"]).astype(str)
    raw = np.asarray([trace["student_runtime_input_52"] for trace in traces], dtype=np.float64)
    trace_targets = np.asarray([
        [float(branch["normalized_regret"]) for branch in trace["teacher_branches"]]
        for trace in traces
    ], dtype=np.float64)
    rebuilt_r0 = build_feature_matrix(raw, "raw52_plus_physics9")
    max_r0_error = float(np.max(np.abs(x_r0 - rebuilt_r0)))
    target_mismatch_count = int(np.sum(np.max(np.abs(y - trace_targets), axis=1) > 0.0))
    assert x_r0.shape == (len(traces), 61)
    assert y.shape == (len(traces), 9)
    assert max_r0_error <= 2e-7
    assert target_mismatch_count == 0

    split = layout_split(layouts.tolist())
    train_ids = np.asarray([i for i, layout in enumerate(layouts) if layout in set(split["train"])])
    rung_ids = train_ids[:32]
    rung_target = y[rung_ids]
    sorted_values = np.sort(rung_target, axis=1)
    best_margin = sorted_values[:, 1] - sorted_values[:, 0]
    best_sets = [np.flatnonzero(np.isclose(row, row.min(), rtol=0.0, atol=1e-12)).tolist()
                 for row in rung_target]
    ladder = result["memorization"]["results"]
    assert len(ladder) == 31 and sum(bool(cell["passed"]) for cell in ladder) == 30
    failed_cell = ladder[-1]
    assert (failed_cell["rung"], failed_cell["representation"], failed_cell["objective"]) == (
        "32", "R0", "regression"
    )

    pair_feature_hashes = hashlib.sha256(x_r0[rung_ids].tobytes()).hexdigest().upper()
    return {
        "schema": "phase15_offline_v2_stopped_run_audit_v1",
        "status": "verified_stopped_at_memorization_gate",
        "run_lock_sha256": sha256_file(LOCK),
        "result_sha256": sha256_file(result_path),
        "result_manifest_sha256": sha256_file(result_manifest_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "feature_dataset_sha256": sha256_file(feature_path),
        "v1_abort_record_sha256": sha256_file(V1 / "failure.json"),
        "provenance": {
            "phase14g_trace_count": len(traces),
            "r0_reconstruction_max_abs_error": max_r0_error,
            "r0_reconstruction_tolerance": 2e-7,
            "exact_teacher_target_row_mismatches": target_mismatch_count,
            "feature_dataset_reused_byte_identically": True,
            "new_r2_transitions_in_v2": 0,
            "exact_planner_calls": 0,
            "new_episodes": 0,
            "reserve_access": False,
        },
        "memorization": {
            "cells_completed": len(ladder),
            "cells_passed_before_stop": sum(bool(cell["passed"]) for cell in ladder),
            "failed_cell": failed_cell,
            "rung32_layout_count": len(set(layouts[rung_ids])),
            "rung32_unique_r0_rows": len(np.unique(x_r0[rung_ids], axis=0)),
            "rung32_r0_rows_sha256": pair_feature_hashes,
            "rung32_teacher_best_action_set_counts": {
                str(action): sum(action in best_set for best_set in best_sets) for action in range(9)
            },
            "rung32_teacher_best_action_margin": {
                "min": float(best_margin.min()),
                "median": float(np.median(best_margin)),
                "max": float(best_margin.max()),
                "tied_state_count": int(np.sum(best_margin == 0.0)),
            },
            "heldout_metrics_computed": False,
        },
        "reporting_note": (
            "The v2 result manifest has a null feature_dataset_sha256 because the early-stop result omitted "
            "the feature_dataset object. The companion feature manifest contains and verifies the exact copied "
            "dataset hash; this audit binds both artifacts without modifying the run outputs."
        ),
    }


def main() -> None:
    output = RUN / "postmortem_audit.json"
    if output.exists():
        raise SystemExit("refusing to overwrite Phase15 v2 postmortem audit")
    report = audit()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "audit": str(output),
                      "audit_sha256": sha256_file(output)}, indent=2))


if __name__ == "__main__":
    main()
