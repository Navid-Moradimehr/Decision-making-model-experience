"""Read-only row-level audit of the reproduced Phase15 32-row R0 fit."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_learning import layout_weights

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "artifacts/runs/phase15_offline_v2/features/phase15_features.npz"
FIT = ROOT / "artifacts/runs/phase15_offline_v2_fit_diagnostic_v2/fit_arrays.npz"
DIAGNOSTIC = ROOT / "artifacts/runs/phase15_offline_v2_fit_diagnostic_v2/diagnostic.json"
MANIFEST = ROOT / "artifacts/runs/phase15_offline_v2_fit_diagnostic_v2/manifest.json"
EXPECTED_FEATURE_SHA = "26D532A918E5AE913EE7F5A5E58BC60E68ED71D6CE38889CCDB7684363191386"
OUTPUT_DIR = ROOT / "artifacts/runs/phase15_v2_fit_error_audit_v1"


def audit() -> dict:
    if sha256_file(FEATURES) != EXPECTED_FEATURE_SHA:
        raise RuntimeError("source feature dataset changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for key, path in (("diagnostic_sha256", DIAGNOSTIC), ("fit_arrays_sha256", FIT)):
        if sha256_file(path) != manifest[key]:
            raise RuntimeError(f"saved replay artifact hash mismatch: {key}")
    with np.load(FIT, allow_pickle=False) as data:
        ids = data["row_indices"].astype(int)
        y = np.asarray(data["targets"], dtype=np.float64)
        pred = np.asarray(data["predictions"], dtype=np.float64)
        layouts = data["layouts"].astype(str)
    with np.load(FEATURES, allow_pickle=False) as data:
        x = np.asarray(data["x_R0"][ids], dtype=np.float64)
        source_y = np.asarray(data["y"][ids], dtype=np.float64)
    if not np.array_equal(y, source_y) or len(ids) != 32 or x.shape != (32, 61):
        raise RuntimeError("fit artifact does not match the fixed 32 training rows")
    error = pred - y
    per_layout_weight = layout_weights(layouts)
    row_mse_contribution = per_layout_weight * np.mean(error * error, axis=1)
    chosen = np.argmin(pred, axis=1)
    best = np.argmin(y, axis=1)
    margins = np.sort(y, axis=1)[:, 1] - np.sort(y, axis=1)[:, 0]
    row_by_id = {int(row): i for i, row in enumerate(ids)}
    mismatch_id = 39
    mismatch = row_by_id[mismatch_id]

    # Distances use the same training-only standardization convention as the MLP.
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    z = (x - x.mean(axis=0)) / x_scale
    distances = np.sqrt(np.sum((z - z[mismatch]) ** 2, axis=1))
    distances[mismatch] = np.inf
    nearest = np.argsort(distances)[:5]
    inversions = []
    for a in range(9):
        for b in range(a + 1, 9):
            target_order = np.sign(y[mismatch, b] - y[mismatch, a])
            predicted_order = np.sign(pred[mismatch, b] - pred[mismatch, a])
            if target_order != 0 and target_order != predicted_order:
                inversions.append({"action_a": a, "action_b": b,
                                   "teacher_regret_a": float(y[mismatch, a]),
                                   "teacher_regret_b": float(y[mismatch, b]),
                                   "predicted_a": float(pred[mismatch, a]),
                                   "predicted_b": float(pred[mismatch, b]),
                                   "teacher_gap": float(abs(y[mismatch, a] - y[mismatch, b]))})
    layout_mse: dict[str, float] = {}
    for layout in sorted(set(layouts)):
        rows = np.flatnonzero(layouts == layout)
        layout_mse[layout] = float(np.mean(error[rows] ** 2))
    best_sets = [np.flatnonzero(np.isclose(row, row.min(), rtol=0.0, atol=1e-12)) for row in y]
    return {
        "status": "descriptive_training_subset_audit",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "feature_sha256": EXPECTED_FEATURE_SHA,
        "fit_arrays_sha256": sha256_file(FIT),
        "row_count": len(ids), "layout_count": len(set(layouts)),
        "mismatch": {
            "feature_row_index": mismatch_id, "layout_hash": str(layouts[mismatch]),
            "teacher_best_action": int(best[mismatch]), "predicted_action": int(chosen[mismatch]),
            "teacher_best_margin": float(margins[mismatch]),
            "selected_regret": float(y[mismatch, chosen[mismatch]]),
            "squared_error_mean": float(np.mean(error[mismatch] ** 2)),
            "layout_weighted_loss_contribution": float(row_mse_contribution[mismatch]),
            "pairwise_inversions": inversions,
            "teacher_regrets": y[mismatch].tolist(), "predicted_values": pred[mismatch].tolist(),
        },
        "fit_subset_summary": {
            "best_action_set_agreement_count": int(sum(chosen[i] in best_sets[i] for i in range(len(ids)))),
            "teacher_margin_min_median_max": [float(margins.min()), float(np.median(margins)), float(margins.max())],
            "mismatch_margin_rank_among_rows": int(np.sum(margins <= margins[mismatch])),
            "mismatch_mse_rank_descending": int(np.sum(np.mean(error * error, axis=1) >= np.mean(error[mismatch] ** 2))),
            "mismatch_layout_row_count": int(np.sum(layouts == layouts[mismatch])),
            "r0_unique_rows": int(len(np.unique(x, axis=0))),
            "nearest_training_rows": [
                {"feature_row_index": int(ids[i]), "distance_standardized_r0": float(distances[i]),
                 "teacher_best_set": np.flatnonzero(np.isclose(y[i], y[i].min(), rtol=0.0, atol=1e-12)).tolist(),
                 "predicted_action": int(chosen[i]), "selected_regret": float(y[i, chosen[i]])}
                for i in nearest
            ],
            "layout_mean_squared_error": layout_mse,
            "objective_loss_mean": float(np.mean(row_mse_contribution) * len(row_mse_contribution)),
        },
        "scope": {"training_rows_only": True, "heldout_metrics": False, "reserve_access": False,
                  "new_transitions": 0, "new_episodes": 0, "planner_calls": 0},
    }


if __name__ == "__main__":
    result = audit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    output_path = OUTPUT_DIR / "audit.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit_path": str(output_path), "audit": result}, indent=2, sort_keys=True))
