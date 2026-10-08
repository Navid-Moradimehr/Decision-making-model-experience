"""Run the single locked full-split regression-versus-pairwise comparison."""

from __future__ import annotations

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
from consequence_lab.phase15_generalization import layout_list_sha256, paired_layout_bootstrap
from consequence_lab.phase15_learning import TinyRegretMLP, decision_metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_fullsplit_objective_generalization.yaml"
LOCK = "experiments/phase15_fullsplit_objective_generalization.lock.json"


def _validate() -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    lock = json.loads((ROOT / LOCK).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase15_fullsplit_objective_generalization_lock_v1" or lock.get("status") != "authorized_single_offline_objective_comparison":
        raise RuntimeError("full-split lock status mismatch")
    if lock.get("config_path") != CONFIG or lock.get("config_sha256") != sha256_file(ROOT / CONFIG):
        raise RuntimeError("full-split config hash mismatch")
    if lock.get("source_sha256") != {p: sha256_file(ROOT / p) for p in lock["source_sha256"]}:
        raise RuntimeError("full-split source hash mismatch")
    if lock.get("parent_sha256") != {key: sha256_file(ROOT / path) for key, path in config["parents"].items()}:
        raise RuntimeError("full-split parent artifact hash mismatch")
    if lock.get("scope") != config["scope"] or lock.get("resource_caps") != config["resource_caps"]:
        raise RuntimeError("full-split scope/caps mismatch")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != lock["source_commit"]:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"], cwd=ROOT)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("full-split run requires clean tree")
    for key, value in config["scope"].items():
        if key not in {"new_training_only_two_fixed_objectives", "heldout_test_metrics_once_after_checkpoint_freeze"} and value is not False:
            raise RuntimeError(f"forbidden full-split scope enabled: {key}")
    return config, lock


def _save_checkpoint(path: Path, model: TinyRegretMLP) -> None:
    np.savez_compressed(path, x_mean=model.x_mean, x_std=model.x_std,
                        w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2, w3=model.w3, b3=model.b3)


def _predict_with_latency(model: TinyRegretMLP, x: np.ndarray) -> tuple[np.ndarray, float]:
    for row in x[:min(10, len(x))]:
        model.predict(row[None, :])
    elapsed = []
    predictions = []
    for row in x:
        start = time.perf_counter_ns()
        value = model.predict(row[None, :])
        elapsed.append((time.perf_counter_ns() - start) / 1e9)
        predictions.append(value[0])
    return np.asarray(predictions), float(np.median(elapsed)) if elapsed else 0.0


def _fit(objective: str, train: dict, x_train: np.ndarray, y_train: np.ndarray, train_layouts: np.ndarray,
         x_val: np.ndarray, y_val: np.ndarray, val_layouts: np.ndarray) -> tuple[TinyRegretMLP, dict]:
    start = time.perf_counter()
    model = TinyRegretMLP(train["seed"], objective, max_epochs=train["max_epochs"],
                          patience=train["patience"], learning_rate=train["learning_rate"], l2=train["l2"])
    model.fit(x_train, y_train, train_layouts, x_val, y_val, val_layouts)
    validation_prediction = model.predict(x_val)
    metrics = decision_metrics(y_val, validation_prediction, val_layouts)
    return model, {"best_epoch": model.best_epoch, "epochs_run": model.epochs_run,
                   "parameters": model.parameter_count, "validation_metrics": metrics,
                   "validation_objective_loss": model.validation_loss,
                   "validation_layout_macro_regret_checkpoint": model.validation_decision_regret,
                   "fit_wall_seconds": time.perf_counter() - start}


def main() -> None:
    config, lock = _validate()
    started = time.perf_counter()
    with np.load(ROOT / config["parents"]["feature_dataset"], allow_pickle=False) as data:
        layouts_all = data["layouts"].astype(str)
        split = layout_split(layouts_all.tolist())
        for name in ("train", "validation", "test"):
            if sorted(split[name]) != lock["split"][name]["layouts"] or layout_list_sha256(split[name]) != lock["split"][name]["layout_list_sha256"]:
                raise RuntimeError(f"locked {name} layout identities changed")
        indices = {name: np.asarray([i for i, layout in enumerate(layouts_all) if layout in set(split[name])], dtype=np.int64)
                   for name in ("train", "validation", "test")}
        if {key: len(value) for key, value in indices.items()} != {key: lock["split"][key]["row_count"] for key in indices}:
            raise RuntimeError("locked row split cardinality changed")
        # Train/validation arrays are the only model-fitting inputs. Test arrays are not indexed here;
        # they are extracted below only after both selected checkpoints are written.
        x_train = np.asarray(data["x_R0"][indices["train"]], dtype=np.float64)
        y_train = np.asarray(data["y"][indices["train"]], dtype=np.float64)
        x_val = np.asarray(data["x_R0"][indices["validation"]], dtype=np.float64)
        y_val = np.asarray(data["y"][indices["validation"]], dtype=np.float64)
        train_layouts, val_layouts = layouts_all[indices["train"]], layouts_all[indices["validation"]]
    run_dir = ROOT / config["outputs"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=False)
    models: dict[str, TinyRegretMLP] = {}
    fit_records: dict[str, dict] = {}
    for model_config in config["models"]:
        objective = model_config["objective"]
        model, record = _fit(objective, model_config, x_train, y_train, train_layouts,
                             x_val, y_val, val_layouts)
        models[objective] = model
        fit_records[objective] = record
        _save_checkpoint(run_dir / f"{objective}_checkpoint.npz", model)
    # Test rows are first indexed after both checkpoints are frozen to disk.
    with np.load(ROOT / config["parents"]["feature_dataset"], allow_pickle=False) as data:
        test_ids = indices["test"]
        x_test = np.asarray(data["x_R0"][test_ids], dtype=np.float64)
        y_test = np.asarray(data["y"][test_ids], dtype=np.float64)
        test_layouts = data["layouts"][test_ids].astype(str)
    test_predictions: dict[str, np.ndarray] = {}
    inference_latency: dict[str, float] = {}
    test_metrics: dict[str, dict] = {}
    for objective, model in models.items():
        prediction = model.predict(x_test)
        test_predictions[objective] = prediction
        _, latency = _predict_with_latency(model, x_val[:min(50, len(x_val))])
        inference_latency[objective] = latency
        test_metrics[objective] = decision_metrics(y_test, prediction, test_layouts)
    bootstrap_cfg = config["metrics"]
    paired = paired_layout_bootstrap(
        test_metrics["pairwise"]["per_layout_mean_regret"],
        test_metrics["regression"]["per_layout_mean_regret"],
        bootstrap_cfg["bootstrap_seed"], bootstrap_cfg["bootstrap_resamples"],
    )
    decision = (paired["pairwise_minus_regression_mean"] < 0.0 and paired["ci_95_upper"] < 0.0)
    rows = []
    for i, row_id in enumerate(test_ids):
        rows.append({"feature_row_index": int(row_id), "layout_hash": str(test_layouts[i]),
                     "teacher_regrets": y_test[i].tolist(),
                     "regression_scores": test_predictions["regression"][i].tolist(),
                     "pairwise_scores": test_predictions["pairwise"][i].tolist(),
                     "regression_action": int(np.argmin(test_predictions["regression"][i])),
                     "pairwise_action": int(np.argmin(test_predictions["pairwise"][i])),
                     "regression_selected_regret": float(test_metrics["regression"]["selected_regret"][i]),
                     "pairwise_selected_regret": float(test_metrics["pairwise"]["selected_regret"][i])})
    np.savez_compressed(run_dir / "heldout_predictions.npz", row_indices=test_ids, layouts=test_layouts,
                        targets=y_test, regression_scores=test_predictions["regression"],
                        pairwise_scores=test_predictions["pairwise"])
    result = {
        "schema": "phase15_fullsplit_objective_generalization_result_v1",
        "status": "completed_single_offline_test",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
        "feature_dataset_sha256": lock["feature_dataset_sha256"],
        "split": {key: {"layout_count": value["layout_count"], "row_count": value["row_count"],
                        "layout_list_sha256": value["layout_list_sha256"]}
                  for key, value in lock["split"].items()},
        "fit": fit_records, "test_metrics": test_metrics, "validation_inference_latency_seconds": inference_latency,
        "primary_paired_layout_bootstrap": paired,
        "decision_rule": {"pairwise_improvement_demonstrated": decision,
                          "criterion": "negative mean pairwise-minus-regression layout regret and 95% CI upper below zero"},
        "test_rows": rows,
        "test_exposure_disclosure": lock["test_exposure_disclosure"],
        "scope": {"test_metrics_computed_once": True, "test_model_selection": False,
                  "reserve_access": False, "new_transitions": 0, "new_episodes": 0,
                  "planner_calls": 0, "closed_loop": False, "dagger": False, "world_model": False},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "wall_seconds": time.perf_counter() - started,
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / config["outputs"]["report"]
    report_path.write_text(_render(result), encoding="utf-8")
    manifest = {"schema": "phase15_fullsplit_objective_generalization_manifest_v1",
                "status": result["status"], "run_lock_sha256": result["lock_sha256"],
                "source_commit": lock["source_commit"], "feature_dataset_sha256": result["feature_dataset_sha256"],
                "result_sha256": sha256_file(result_path),
                "regression_checkpoint_sha256": sha256_file(run_dir / "regression_checkpoint.npz"),
                "pairwise_checkpoint_sha256": sha256_file(run_dir / "pairwise_checkpoint.npz"),
                "heldout_predictions_sha256": sha256_file(run_dir / "heldout_predictions.npz"),
                "report_sha256": sha256_file(report_path), "reserve_access": False,
                "closed_loop": False, "test_metrics_computed_once": True}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_bytes = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file()) + report_path.stat().st_size
    if result["wall_seconds"] > config["resource_caps"]["wall_seconds"] or output_bytes > config["resource_caps"]["output_bytes"]:
        raise RuntimeError("authorized full-split resource cap exceeded")
    print(json.dumps({"status": result["status"], "primary": {k: paired[k] for k in
                      ("layout_count", "pairwise_minus_regression_mean", "ci_95_lower", "ci_95_upper")},
                      "pairwise_improvement_demonstrated": decision,
                      "test_regret": {k: v["mean_normalized_regret_layout_macro"] for k, v in test_metrics.items()},
                      "wall_seconds": result["wall_seconds"], "output_bytes": output_bytes,
                      "report": config["outputs"]["report"]}, indent=2))


def _render(result: dict) -> str:
    metrics, ci = result["test_metrics"], result["primary_paired_layout_bootstrap"]
    lines = ["# Phase15 full-split objective generalization result", "",
             "**Status: completed once.** This is a layout-held-out offline decision-quality comparison, not closed-loop control.", "",
             "## Frozen grouped split", "",
             "| Split | Layouts | Decision rows |",
             "|---|---:|---:|"]
    for name, split in result["split"].items():
        lines.append(f"| {name} | {split['layout_count']} | {split['row_count']} |")
    lines.extend(["", "## Diagnostic test metrics", "",
                  "| Metric | Regression | Pairwise |", "|---|---:|---:|"])
    for label, key in (("Layout-macro mean regret", "mean_normalized_regret_layout_macro"),
                       ("State-micro mean regret", "mean_normalized_regret_state_micro"),
                       ("Catastrophic fraction (layout macro)", "catastrophic_fraction_layout_macro"),
                       ("Best-action-set agreement", "best_action_set_agreement"),
                       ("Pairwise ranking accuracy", "pairwise_ranking_accuracy_macro")):
        lines.append(f"| {label} | {metrics['regression'][key]:.6f} | {metrics['pairwise'][key]:.6f} |")
    lines.extend(["", f"Paired layout-bootstrap pairwise-minus-regression regret: `{ci['pairwise_minus_regression_mean']:.6f}`, 95% interval `[{ci['ci_95_lower']:.6f}, {ci['ci_95_upper']:.6f}]` over {ci['layout_count']} layouts ({ci['resamples']} resamples, seed {ci['seed']}).",
                  f"Predeclared rule: **{'pairwise improvement demonstrated' if result['decision_rule']['pairwise_improvement_demonstrated'] else 'no pairwise improvement demonstrated'}**.", "",
                  "## Validation-selected checkpoints and compute", "",
                  "| Objective | Best epoch | Parameters | Fit seconds | Validation-sample median inference ms |",
                  "|---|---:|---:|---:|---:|"])
    for objective in ("regression", "pairwise"):
        fit = result["fit"][objective]
        lines.append(f"| {objective} | {fit['best_epoch']} | {fit['parameters']} | {fit['fit_wall_seconds']:.3f} | {1000*result['validation_inference_latency_seconds'][objective]:.6f} |")
    lines.extend(["", "## Scope and interpretation", "",
                  "Checkpoints were selected using validation only and saved before test-row extraction/scoring. The feature archive already contained the diagnostic test features/targets, and the earlier v2 loader read the full NPZ before stopping; no test metric or test-driven model choice was made before this run. This is therefore a metric-unconsumed diagnostic holdout, not an entirely unexposed external test.", "",
                  "No reserve access, new transition/episode/planner call, closed-loop test, DAgger, calibration, or world-model use occurred. Pairwise outputs are ranking scores, not calibrated regret values. No follow-on phase is authorized by this result.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
