"""Run the one locked Phase15 failed-cell fit after dtype-explicit hash validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_data import layout_split
from consequence_lab.phase15_learning import TinyRegretMLP, decision_metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_offline_v2_fit_diagnostic_v2.yaml"
LOCK = "experiments/phase15_offline_v2_fit_diagnostic_v2.lock.json"


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest().upper()


def _validate() -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    lock = json.loads((ROOT / LOCK).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase15_v2_failed_cell_replay_lock_v2" or lock.get("status") != "authorized_single_diagnostic_replay_after_hash_preflight_correction":
        raise RuntimeError("corrected replay lock status mismatch")
    if lock.get("config_path") != CONFIG or lock.get("config_sha256") != sha256_file(ROOT / CONFIG):
        raise RuntimeError("corrected replay config hash mismatch")
    if lock.get("source_sha256") != {p: sha256_file(ROOT / p) for p in lock["source_sha256"]}:
        raise RuntimeError("corrected replay source hash mismatch")
    if lock.get("parent_sha256") != {p: sha256_file(ROOT / config["parent"][p]) for p in lock["parent_sha256"]}:
        raise RuntimeError("parent artifact hash mismatch")
    if lock.get("failed_preflight_lock_sha256") != sha256_file(ROOT / "experiments/phase15_offline_v2_fit_diagnostic.lock.json"):
        raise RuntimeError("first preflight abort lock provenance mismatch")
    if lock.get("scope") != config["scope"] or lock.get("resource_caps") != config["resource_caps"]:
        raise RuntimeError("scope/resource cap mismatch")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != lock["source_commit"]:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"], cwd=ROOT)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("corrected replay requires a clean tree")
    for name, expected in config["scope"].items():
        if name not in {"exact_parent_cell_only", "reuse_saved_phase14g_features_only"} and expected is not False:
            raise RuntimeError(f"forbidden scope enabled: {name}")
    return config, lock


def main() -> None:
    config, lock = _validate()
    started = time.perf_counter()
    source_path = ROOT / config["parent"]["feature_dataset"]
    ids = config["cell"]["row_indices"]
    with np.load(source_path, allow_pickle=False) as data:
        all_layouts = data["layouts"].astype(str)
        split = layout_split(all_layouts.tolist())
        reproduced = [i for i, name in enumerate(all_layouts) if name in set(split["train"])][:32]
        if reproduced != ids:
            raise RuntimeError("frozen 32 indices differ from original memorization ladder")
        x_source = np.ascontiguousarray(data["x_R0"][ids])
        y_source = np.ascontiguousarray(data["y"][ids])
        layouts = all_layouts[ids]
    x = np.asarray(x_source, dtype=np.float64)
    y = np.asarray(y_source, dtype=np.float64)
    cell = config["cell"]
    observed = {
        "source_input_dtype": str(x_source.dtype), "source_input_sha256": _sha(x_source),
        "fit_input_dtype": str(x.dtype), "fit_input_sha256": _sha(x),
        "source_target_dtype": str(y_source.dtype), "source_target_sha256": _sha(y_source),
        "fit_target_dtype": str(y.dtype), "fit_target_sha256": _sha(y),
    }
    for field, value in observed.items():
        if cell[field] != value:
            raise RuntimeError(f"dtype-aware fitting data mismatch for {field}")
    # Create outputs only after every source/target/split preflight has passed.
    run_dir = ROOT / config["outputs"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=False)
    train = config["training"]
    model = TinyRegretMLP(
        train["seed"], cell["objective"], max_epochs=train["max_epochs"], patience=train["patience"],
        learning_rate=train["learning_rate"], l2=train["l2"], record_history=True,
    ).fit(x, y, layouts, x, y, layouts)
    prediction = model.predict(x)
    metrics = decision_metrics(y, prediction, layouts)
    sorted_target = np.sort(y, axis=1)
    rows = []
    for row_index, layout, target, pred, margin in zip(ids, layouts, y, prediction,
                                                       sorted_target[:, 1] - sorted_target[:, 0], strict=True):
        action = int(np.argmin(pred))
        best_set = np.flatnonzero(np.isclose(target, target.min(), rtol=0.0, atol=1e-12)).tolist()
        rows.append({"feature_row_index": int(row_index), "layout_hash": str(layout),
                     "target_normalized_regret": target.tolist(),
                     "predicted_normalized_regret": pred.tolist(),
                     "teacher_best_action_set": [int(a) for a in best_set],
                     "predicted_action": action, "selected_teacher_regret": float(target[action]),
                     "best_vs_second_best_margin": float(margin),
                     "best_action_set_match": bool(action in best_set)})
    expected = config["expected_original"]
    reproduced = (model.best_epoch == expected["best_epoch"]
                  and metrics["best_action_set_agreement"] == expected["best_action_set_agreement"]
                  and metrics["mean_normalized_regret_state_micro"] == expected["mean_normalized_regret_state_micro"])
    np.savez_compressed(run_dir / "fit_arrays.npz", row_indices=np.asarray(ids), layouts=layouts,
                        targets=y, predictions=prediction,
                        selected_actions=np.asarray(metrics["selected_actions"]),
                        selected_regret=np.asarray(metrics["selected_regret"]))
    (run_dir / "training_history.json").write_text(json.dumps(model.history, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema": "phase15_v2_failed_cell_fit_diagnostic_v2", "status": "reproduced" if reproduced else "non_reproducing_stop",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
        "feature_dataset_sha256": sha256_file(source_path), "cell": cell, "observed_hashes": observed,
        "training": train, "row_details": rows, "metrics": metrics,
        "optimization": {"best_epoch": model.best_epoch, "epochs_run": model.epochs_run,
                          "best_validation_loss": model.validation_loss,
                          "best_validation_decision_regret": model.validation_decision_regret,
                          "parameter_count": model.parameter_count, "history_rows": len(model.history),
                          "last_history_row": model.history[-1]},
        "reproduction_comparison": {"expected": expected, "matches_all_frozen_summaries": reproduced},
        "scope": {"heldout_metrics_computed": False, "new_transitions": 0, "new_episodes": 0,
                  "exact_planner_calls": 0, "closed_loop": False, "dagger": False,
                  "world_model": False, "reserve_access": False},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "wall_seconds": time.perf_counter() - started,
    }
    (run_dir / "diagnostic.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / config["outputs"]["report"]
    report_path.write_text(_render(result), encoding="utf-8")
    manifest = {
        "schema": "phase15_v2_failed_cell_fit_diagnostic_manifest_v1",
        "status": result["status"], "run_lock_sha256": result["lock_sha256"],
        "source_commit": result["source_commit"],
        "feature_dataset_sha256": result["feature_dataset_sha256"],
        "diagnostic_sha256": sha256_file(run_dir / "diagnostic.json"),
        "fit_arrays_sha256": sha256_file(run_dir / "fit_arrays.npz"),
        "training_history_sha256": sha256_file(run_dir / "training_history.json"),
        "report_sha256": sha256_file(report_path),
        "heldout_metrics_computed": False, "reserve_access": False,
        "new_transitions": 0, "new_episodes": 0, "exact_planner_calls": 0,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_bytes = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file()) + report_path.stat().st_size
    if result["wall_seconds"] > config["resource_caps"]["wall_seconds"]:
        raise RuntimeError("authorized 3-hour wall-time cap exceeded")
    if output_bytes > config["resource_caps"]["output_bytes"]:
        raise RuntimeError("authorized 20-MiB output cap exceeded")
    print(json.dumps({"status": result["status"], "metrics": metrics,
                      "output_bytes": output_bytes, "run_dir": str(run_dir),
                      "report": config["outputs"]["report"]}, indent=2))
    if not reproduced:
        raise SystemExit("corrected replay did not reproduce the frozen result; stop")


def _render(result: dict) -> str:
    metrics = result["metrics"]
    bad = [row for row in result["row_details"] if not row["best_action_set_match"]]
    lines = ["# Phase15 v2 failed-cell diagnostic replay result (corrected hash preflight)", "",
             f"**Status: {result['status']}**. One exact replay of the v2 32-state R0/regression fit cell.", "",
             "## Outcome", "",
             f"- Best epoch: {result['optimization']['best_epoch']} (expected {result['reproduction_comparison']['expected']['best_epoch']})",
             f"- Best-action-set agreement: {metrics['best_action_set_agreement']:.8f} (expected {result['reproduction_comparison']['expected']['best_action_set_agreement']:.8f})",
             f"- Mean fitting regret: {metrics['mean_normalized_regret_state_micro']:.8f} (expected {result['reproduction_comparison']['expected']['mean_normalized_regret_state_micro']:.8f})",
             f"- Pairwise ranking accuracy: {metrics['pairwise_ranking_accuracy_macro']:.6f}",
             f"- Recorded epochs / parameters: {result['optimization']['history_rows']} / {result['optimization']['parameter_count']}",
             f"- Runtime / saved bytes: {result['wall_seconds']:.3f}s / {result['output_bytes'] if 'output_bytes' in result else 'recorded in runner stdout'}", "",
             "## Disagreements", ""]
    if not bad:
        lines.append("No predicted action fell outside the teacher best-action sets.")
    for row in bad:
        lines.extend([f"- Row `{row['feature_row_index']}` (layout `{row['layout_hash']}`): predicted action `{row['predicted_action']}`, teacher best set `{row['teacher_best_action_set']}`, regret `{row['selected_teacher_regret']:.8f}`, best/runner-up margin `{row['best_vs_second_best_margin']:.8f}`.",
                      f"  - Targets: `{row['target_normalized_regret']}`",
                      f"  - Prediction: `{row['predicted_normalized_regret']}`"])
    lines.extend(["", "## Scope and interpretation", "",
                  "Only the 32 frozen training rows were fitted/scored. No held-out metric, new transition/episode/planner call, closed-loop test, DAgger, world model, or reserve access occurred. The prior v1 diagnostic attempt stopped before `.fit()` on a dtype-specific hash mismatch; that separate abort is preserved in [[phase15_offline_v2_fit_diagnostic_preflight_abort]]. This replay does not open subsequent analysis.", "",
                  f"Feature dataset SHA-256: `{result['feature_dataset_sha256']}`. Lock SHA-256: `{result['lock_sha256']}`.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
