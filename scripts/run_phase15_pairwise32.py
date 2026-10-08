"""Fit the one frozen Phase15 pairwise objective diagnostic on 32 training rows."""

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
CONFIG = "configs/phase15_v2_pairwise32.yaml"
LOCK = "experiments/phase15_v2_pairwise32.lock.json"


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest().upper()


def _validate() -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    lock = json.loads((ROOT / LOCK).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase15_v2_objective_pairwise32_lock_v1" or lock.get("status") != "authorized_single_pairwise_memorization_diagnostic":
        raise RuntimeError("pairwise lock is not authorized")
    if lock.get("config_path") != CONFIG or lock.get("config_sha256") != sha256_file(ROOT / CONFIG):
        raise RuntimeError("pairwise config hash mismatch")
    if lock.get("source_sha256") != {p: sha256_file(ROOT / p) for p in lock["source_sha256"]}:
        raise RuntimeError("pairwise source hash mismatch")
    if lock.get("parent_sha256") != {key: sha256_file(ROOT / path) for key, path in config["parents"].items()}:
        raise RuntimeError("pairwise parent hash mismatch")
    if lock.get("scope") != config["scope"] or lock.get("resource_caps") != config["resource_caps"]:
        raise RuntimeError("pairwise scope/cap mismatch")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != lock["source_commit"]:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"], cwd=ROOT)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("pairwise run requires clean tree")
    return config, lock


def main() -> None:
    config, lock = _validate()
    started = time.perf_counter()
    ids = config["cell"]["row_indices"]
    with np.load(ROOT / config["parents"]["phase15_feature_dataset"], allow_pickle=False) as data:
        layouts_all = data["layouts"].astype(str)
        split = layout_split(layouts_all.tolist())
        actual_ids = [i for i, name in enumerate(layouts_all) if name in set(split["train"])][:32]
        if actual_ids != ids:
            raise RuntimeError("frozen row IDs do not match original Phase15 rung")
        raw_x = np.ascontiguousarray(data["x_R0"][ids])
        raw_y = np.ascontiguousarray(data["y"][ids])
        layouts = layouts_all[ids]
    x, y = raw_x.astype(np.float64), raw_y.astype(np.float64)
    cell = config["cell"]
    observed = {"source_x_dtype": str(raw_x.dtype), "source_x_sha256": _sha(raw_x),
                "fit_x_dtype": str(x.dtype), "fit_x_sha256": _sha(x),
                "source_y_dtype": str(raw_y.dtype), "source_y_sha256": _sha(raw_y),
                "fit_y_dtype": str(y.dtype), "fit_y_sha256": _sha(y)}
    expected = {"source_x_dtype": cell["source_dtype"], "source_x_sha256": cell["source_x_sha256"],
                "fit_x_dtype": cell["fit_dtype"], "fit_x_sha256": cell["fit_x_sha256"],
                "source_y_dtype": cell["source_dtype"], "source_y_sha256": cell["source_y_sha256"],
                "fit_y_dtype": cell["fit_dtype"], "fit_y_sha256": cell["fit_y_sha256"]}
    if observed != expected:
        raise RuntimeError("pairwise fit data dtype/hash mismatch")
    if sha256_file(ROOT / config["parents"]["phase15_feature_dataset"]) != lock["parent_sha256"]["phase15_feature_dataset"]:
        raise RuntimeError("feature data changed after lock")
    baseline = json.loads((ROOT / config["parents"]["regression_diagnostic"]).read_text(encoding="utf-8"))
    baseline_arrays_path = ROOT / config["parents"]["regression_arrays"]
    with np.load(baseline_arrays_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["row_indices"], np.asarray(ids)) or not np.array_equal(saved["targets"], y):
            raise RuntimeError("regression baseline rows/targets do not match this comparison")
    # No output directory is created until all lock, split, source and baseline checks pass.
    run_dir = ROOT / config["outputs"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=False)
    train = config["training"]
    model = TinyRegretMLP(train["seed"], train["objective"], max_epochs=train["max_epochs"],
                          patience=train["patience"], learning_rate=train["learning_rate"],
                          l2=train["l2"], record_history=True)
    model.fit(x, y, layouts, x, y, layouts)
    prediction = model.predict(x)
    metrics = decision_metrics(y, prediction, layouts)
    baseline_metrics = baseline["metrics"]
    rows = []
    margins = np.sort(y, axis=1)[:, 1] - np.sort(y, axis=1)[:, 0]
    for i, row_id in enumerate(ids):
        best_set = np.flatnonzero(np.isclose(y[i], y[i].min(), rtol=0.0, atol=1e-12)).tolist()
        action = int(np.argmin(prediction[i]))
        rows.append({"feature_row_index": int(row_id), "teacher_best_action_set": [int(a) for a in best_set],
                     "predicted_action": action, "selected_regret": float(y[i, action]),
                     "best_vs_second_margin": float(margins[i]),
                     "best_action_set_match": bool(action in best_set),
                     "teacher_regrets": y[i].tolist(), "predicted_regrets": prediction[i].tolist()})
    gate = config["gate"]
    passed = (metrics["best_action_set_agreement"] >= gate["best_action_set_agreement_minimum"]
              and metrics["mean_normalized_regret_state_micro"] <= gate["mean_normalized_regret_state_micro_maximum"])
    np.savez_compressed(run_dir / "fit_arrays.npz", row_indices=np.asarray(ids), layouts=layouts,
                        targets=y, predictions=prediction,
                        selected_actions=np.asarray(metrics["selected_actions"]),
                        selected_regret=np.asarray(metrics["selected_regret"]))
    (run_dir / "training_history.json").write_text(json.dumps(model.history, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema": "phase15_v2_pairwise32_result_v1", "status": "memorization_gate_pass" if passed else "memorization_gate_fail",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
        "feature_dataset_sha256": lock["parent_sha256"]["phase15_feature_dataset"],
        "parent_audit_sha256": lock["parent_sha256"]["fit_error_audit"],
        "row_indices": ids, "observed_data_hashes": observed, "training": train,
        "metrics": metrics, "rows": rows,
        "baseline_regression": {"status": baseline["status"], "metrics": baseline_metrics,
                                "best_epoch": baseline["optimization"]["best_epoch"]},
        "comparison": {
            "best_action_agreement_delta": metrics["best_action_set_agreement"] - baseline_metrics["best_action_set_agreement"],
            "mean_regret_delta": metrics["mean_normalized_regret_state_micro"] - baseline_metrics["mean_normalized_regret_state_micro"],
            "pairwise_accuracy_delta": metrics["pairwise_ranking_accuracy_macro"] - baseline_metrics["pairwise_ranking_accuracy_macro"],
        },
        "gate": {"thresholds": gate, "passed": passed,
                 "note": "In-sample memorization only; a pass does not establish generalization or closed-loop control."},
        "optimization": {"best_epoch": model.best_epoch, "epochs_run": model.epochs_run,
                         "best_validation_loss": model.validation_loss,
                         "best_validation_decision_regret": model.validation_decision_regret,
                         "history_rows": len(model.history), "parameter_count": model.parameter_count},
        "scope": {"rows_fit": 32, "heldout_metrics": False, "reserve_access": False,
                  "new_transitions": 0, "new_episodes": 0, "planner_calls": 0,
                  "closed_loop": False, "dagger": False, "world_model": False},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "wall_seconds": time.perf_counter() - started,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / config["outputs"]["report"]
    report_path.write_text(_render(result), encoding="utf-8")
    manifest = {"schema": "phase15_v2_pairwise32_manifest_v1", "status": result["status"],
                "run_lock_sha256": result["lock_sha256"], "source_commit": lock["source_commit"],
                "feature_dataset_sha256": result["feature_dataset_sha256"],
                "result_sha256": sha256_file(run_dir / "result.json"),
                "fit_arrays_sha256": sha256_file(run_dir / "fit_arrays.npz"),
                "training_history_sha256": sha256_file(run_dir / "training_history.json"),
                "report_sha256": sha256_file(report_path), "heldout_metrics": False,
                "reserve_access": False, "planner_calls": 0}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_bytes = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file()) + report_path.stat().st_size
    if result["wall_seconds"] > config["resource_caps"]["wall_seconds"] or output_bytes > config["resource_caps"]["output_bytes"]:
        raise RuntimeError("authorized Phase15 pairwise diagnostic resource cap exceeded")
    print(json.dumps({"status": result["status"], "metrics": metrics, "comparison": result["comparison"],
                      "best_epoch": model.best_epoch, "wall_seconds": result["wall_seconds"],
                      "output_bytes": output_bytes, "report": config["outputs"]["report"]}, indent=2))


def _render(result: dict) -> str:
    m, base = result["metrics"], result["baseline_regression"]["metrics"]
    lines = ["# Phase15 pairwise objective 32-row result", "",
             f"**Status: {result['status']}**. Training-set memorization only.", "",
             "| Metric | Saved regression | Pairwise objective | Delta |",
             "|---|---:|---:|---:|",
             f"| Best-action-set agreement | {base['best_action_set_agreement']:.6f} | {m['best_action_set_agreement']:.6f} | {result['comparison']['best_action_agreement_delta']:+.6f} |",
             f"| Mean normalized regret | {base['mean_normalized_regret_state_micro']:.6f} | {m['mean_normalized_regret_state_micro']:.6f} | {result['comparison']['mean_regret_delta']:+.6f} |",
             f"| Pairwise ranking accuracy | {base['pairwise_ranking_accuracy_macro']:.6f} | {m['pairwise_ranking_accuracy_macro']:.6f} | {result['comparison']['pairwise_accuracy_delta']:+.6f} |",
             f"| Catastrophic fraction | {base['catastrophic_fraction_state_micro']:.6f} | {m['catastrophic_fraction_state_micro']:.6f} | — |", "",
             f"Pairwise fit best epoch: {result['optimization']['best_epoch']}; epochs run: {result['optimization']['epochs_run']}; parameters: {result['optimization']['parameter_count']}.",
             f"Frozen gate (`agreement >= {result['gate']['thresholds']['best_action_set_agreement_minimum']}` and `mean regret <= {result['gate']['thresholds']['mean_normalized_regret_state_micro_maximum']}`): **{'PASS' if result['gate']['passed'] else 'FAIL'}**.", "",
             "Only the frozen 32 training rows were fit/scored. No held-out metric, reserve access, transition, episode, planner call, closed-loop test, DAgger, or world model was used. No generalization or deployment conclusion follows.", "",
             "## Rows outside the teacher best-action set", ""]
    bad = [row for row in result["rows"] if not row["best_action_set_match"]]
    lines.extend([f"- Row {row['feature_row_index']}: teacher set {row['teacher_best_action_set']}; selected {row['predicted_action']}; regret {row['selected_regret']:.6f}; margin {row['best_vs_second_margin']:.6f}." for row in bad] or ["None."])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
