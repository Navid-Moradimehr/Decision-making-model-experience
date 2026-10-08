"""One-shot, train/validation-only D151 R4-summary prediction feasibility screen."""

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
from consequence_lab.phase15_learning import decision_metrics
from consequence_lab.phase15_outcome_summary import (
    OutcomeSummaryMLP, constant_summary, restore_decision_head, screen_prediction,
    summary_metrics, validate_summary_targets,
)
from consequence_lab.phase15_representation_screen import train_validation_indices

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_r4_summary_feasibility.yaml"
LOCK = "experiments/phase15_r4_summary_feasibility.lock.json"


def validate_lock() -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    lock = json.loads((ROOT / LOCK).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase15_r4_summary_feasibility_lock_v1" or lock.get("status") != "authorized_offline_summary_prediction_screen":
        raise RuntimeError("D151 lock status mismatch")
    if lock.get("config_sha256") != sha256_file(ROOT / CONFIG):
        raise RuntimeError("D151 config hash mismatch")
    if lock.get("source_sha256") != {path: sha256_file(ROOT / path) for path in lock["source_sha256"]}:
        raise RuntimeError("D151 source hash mismatch")
    if lock.get("parent_sha256") != {key: sha256_file(ROOT / path) for key, path in config["parents"].items()}:
        raise RuntimeError("D151 parent hash mismatch")
    if lock.get("scope") != config["scope"] or lock.get("screen_gate") != config["screen_gate"]:
        raise RuntimeError("D151 scope/screen mismatch")
    if lock.get("frozen_head_checkpoint_sha256") != {
            name: sha256_file(ROOT / config["representation_checkpoint_dir"] / f"{name}_checkpoint.npz")
            for name in ("R0", "R4")}:
        raise RuntimeError("frozen D150 head checkpoint hash mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("D151 must run from a clean committed source tree")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not (source_commit == lock["source_commit"] or subprocess.run(
            ["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"], cwd=ROOT).returncode == 0):
        raise RuntimeError("D151 source commit is not an ancestor of current HEAD")
    return config, lock


def _verify_reproduction(name: str, metrics: dict, config: dict, *, epoch: int | None = None) -> None:
    gate = config["head_reproduction_gate"][name]
    tolerance = config["head_reproduction_gate"]["tolerance"]
    if epoch is not None and epoch != gate["best_epoch"]:
        raise RuntimeError(f"D150 {name} best epoch did not reproduce")
    keys = {"validation_layout_macro_regret": "mean_normalized_regret_layout_macro",
            "validation_catastrophic_fraction_state_micro": "catastrophic_fraction_state_micro",
            "validation_best_action_set_agreement": "best_action_set_agreement"}
    for expected_key, metric_key in keys.items():
        if abs(float(metrics[metric_key]) - float(gate[expected_key])) > tolerance:
            raise RuntimeError(f"D150 {name} frozen-head reproduction failed at {metric_key}")


def _predict_metrics(head, x: np.ndarray, y: np.ndarray, layouts: np.ndarray) -> tuple[dict, np.ndarray]:
    predicted = head.predict(x)
    return decision_metrics(y, predicted, layouts), predicted


def _render(result: dict) -> str:
    lines = ["# Phase 15 D151 R4-summary prediction feasibility result", "",
             "**Status: exploratory offline feasibility screen; validation reused.** No reserve or consumed D148 test rows were scored.", "",
             "## Frozen decision-head reproduction", "",
             "| Head | Macro regret | Catastrophic fraction | Best-action agreement |", "|---|---:|---:|---:|"]
    for name in ("R0", "R4"):
        m = result["head_reproduction"][name]
        lines.append(f"| {name} | {m['mean_normalized_regret_layout_macro']:.6f} | {m['catastrophic_fraction_state_micro']:.6f} | {m['best_action_set_agreement']:.6f} |")
    lines.extend(["", "## R4 outcome-summary prediction", "",
                  "| Predictor | RMSE | MAE | Outcome argmax accuracy | Steps MAE | Goal-distance MAE | Effort MAE |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name in ("constant", "learned"):
        m = result["summary_prediction"][name]["validation"]
        lines.append(f"| {name} | {m['rmse']:.6f} | {m['mae']:.6f} | {m['outcome_argmax_accuracy']:.6f} | {m['terminal_steps_mae']:.6f} | {m['terminal_goal_distance_mae']:.6f} | {m['noncoast_effort_mae']:.6f} |")
    lines.extend(["", "## Frozen-head downstream decisions", "",
                  "| Input to frozen head | Macro regret | State-micro regret | Catastrophic fraction | Best-action agreement | Weighted pairwise accuracy |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, record in result["downstream_decisions"].items():
        m = record["validation"]
        lines.append(f"| {name} | {m['mean_normalized_regret_layout_macro']:.6f} | {m['mean_normalized_regret_state_micro']:.6f} | {m['catastrophic_fraction_state_micro']:.6f} | {m['best_action_set_agreement']:.6f} | {m['regret_gap_weighted_pairwise_accuracy_macro']:.6f} |")
    screen = result["screen"]
    gate_status = "promising for fresh independent feasibility confirmation only" if screen["promising_for_fresh_confirmation_only"] else "not met"
    lines.extend(["", f"Screen: **{gate_status}**. {screen['summary_rmse_relative_improvement_vs_constant']:.1%} RMSE improvement versus constant; gates were frozen before the run.", "",
                  "## Scope and interpretation", "",
                  "This predicts saved exact teacher continuation summaries from current R0 state, then passes them through a frozen D150 R4 decision head. It is not recursive dynamics, JEPA, planner integration, online world-model use, or closed-loop control. Decision heads were not updated. Results are exploratory because the validation group was previously used for D148/D150 selection. A pass only nominates this representation for a separately locked fresh-layout confirmation.", "",
                  "The compressed NPZ may materialize complete feature members internally. Only train/validation row indices entered fitting, frozen-head inference, and metrics; D148 test rows were not indexed or scored. No simulator/planner calls, new episodes/transitions, DAgger, reserve access, or recursive rollout occurred.", ""])
    return "\n".join(lines)


def main() -> None:
    config, lock = validate_lock()
    out_dir = ROOT / config["outputs"]["run_dir"]
    dataset_path = ROOT / config["parents"]["feature_dataset"]
    with np.load(dataset_path, allow_pickle=False) as archive:
        layouts_all = archive["layouts"].astype(str).tolist()
        train_idx, val_idx, train_names, val_names = train_validation_indices(layouts_all, config["split"])
        if lock["split"]["train"]["layouts"] != train_names or lock["split"]["validation"]["layouts"] != val_names:
            raise RuntimeError("D151 locked layout membership mismatch")
        layout_array = np.asarray(layouts_all, dtype=str)
        train_layouts, val_layouts = layout_array[train_idx], layout_array[val_idx]
        # The source NPZ is compressed, so NumPy can materialize a complete member internally.
        # Only train/validation indices are explicitly passed downstream.
        y_train = np.asarray(archive["y"][train_idx], dtype=np.float64)
        y_val = np.asarray(archive["y"][val_idx], dtype=np.float64)
        r0_all, r4_all = archive["x_R0"], archive["x_R4"]
        if r0_all.shape != (len(layouts_all), 61) or r4_all.shape != (len(layouts_all), 124):
            raise RuntimeError("D151 feature dimensions differ from frozen contract")
        x0_train, x0_val = np.asarray(r0_all[train_idx], dtype=np.float64), np.asarray(r0_all[val_idx], dtype=np.float64)
        x4_train, x4_val = np.asarray(r4_all[train_idx], dtype=np.float64), np.asarray(r4_all[val_idx], dtype=np.float64)
    if not np.array_equal(x4_train[:, :61], x0_train) or not np.array_equal(x4_val[:, :61], x0_val):
        raise RuntimeError("R4 state prefix differs from R0 input")
    if y_train.shape != (len(train_idx), 9) or y_val.shape != (len(val_idx), 9):
        raise RuntimeError("teacher regret target dimensions differ from contract")
    summary_train, summary_val = x4_train[:, 61:124], x4_val[:, 61:124]
    validate_summary_targets(summary_train)
    validate_summary_targets(summary_val)

    ckpt_dir = ROOT / config["representation_checkpoint_dir"]
    r0_head = restore_decision_head(ckpt_dir / "R0_checkpoint.npz")
    r4_head = restore_decision_head(ckpt_dir / "R4_checkpoint.npz")
    if r0_head.x_mean.shape != (61,) or r4_head.x_mean.shape != (124,):
        raise RuntimeError("frozen R0/R4 input dimensions do not match their representation")
    # Reproduce frozen D150 validation baselines before fitting the summary predictor.
    r0_metrics, r0_val_scores = _predict_metrics(r0_head, x0_val, y_val, val_layouts)
    r4_metrics, r4_val_scores = _predict_metrics(r4_head, x4_val, y_val, val_layouts)
    _verify_reproduction("R0", r0_metrics, config, epoch=113)
    _verify_reproduction("R4", r4_metrics, config)

    out_dir.mkdir(parents=True, exist_ok=False)
    constant_val = constant_summary(summary_train, train_layouts, len(val_idx))
    constant_train = constant_summary(summary_train, train_layouts, len(train_idx))
    model_cfg = config["model"]
    predictor = OutcomeSummaryMLP(seed=model_cfg["seed"], hidden=tuple(model_cfg["architecture"][1:3]),
                                  max_epochs=model_cfg["max_epochs"], patience=model_cfg["patience"],
                                  learning_rate=model_cfg["learning_rate"], l2=model_cfg["l2"])
    start = time.perf_counter()
    predictor.fit(x0_train, summary_train, train_layouts, x0_val, summary_val, val_layouts)
    fit_seconds = time.perf_counter() - start
    predicted_train, predicted_val = predictor.predict(x0_train), predictor.predict(x0_val)
    constant_train_metrics = summary_metrics(summary_train, constant_train)
    constant_val_metrics = summary_metrics(summary_val, constant_val)
    learned_train_metrics = summary_metrics(summary_train, predicted_train)
    learned_val_metrics = summary_metrics(summary_val, predicted_val)

    constant_scores = r4_head.predict(np.concatenate((x0_val, constant_val), axis=1))
    predicted_scores = r4_head.predict(np.concatenate((x0_val, predicted_val), axis=1))
    downstream = {
        "constant_summary_plus_frozen_R4_head": {"validation": decision_metrics(y_val, constant_scores, val_layouts)},
        "predicted_summary_plus_frozen_R4_head": {"validation": decision_metrics(y_val, predicted_scores, val_layouts)},
        "exact_R4_summary_oracle_plus_frozen_R4_head": {"validation": r4_metrics},
        "direct_R0_frozen_head": {"validation": r0_metrics},
    }
    screen = screen_prediction(learned_val_metrics["rmse"], constant_val_metrics["rmse"],
                               downstream["predicted_summary_plus_frozen_R4_head"]["validation"],
                               config["screen_gate"])
    result = {
        "schema": "phase15_r4_summary_feasibility_result_v1", "status": "completed_exploratory_screen",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
        "config_sha256": lock["config_sha256"], "feature_dataset_sha256": lock["parent_sha256"]["feature_dataset"],
        "split": {"train_layouts": len(train_names), "train_rows": len(train_idx),
                  "validation_layouts": len(val_names), "validation_rows": len(val_idx), "test_rows_scored": 0},
        "head_reproduction": {"R0": r0_metrics, "R4": r4_metrics},
        "summary_prediction": {"constant": {"train": constant_train_metrics, "validation": constant_val_metrics},
                               "learned": {"train": learned_train_metrics, "validation": learned_val_metrics}},
        "downstream_decisions": downstream, "screen": screen,
        "predictor": {"architecture": model_cfg["architecture"], "parameters": predictor.parameter_count,
                      "best_epoch": predictor.best_epoch, "epochs_run": predictor.epochs_run,
                      "validation_feature_mse": predictor.validation_mse, "fit_wall_seconds": fit_seconds,
                      "frozen_R0_head_updated": False, "frozen_R4_head_updated": False},
        "scope": {"train_rows_fit": len(train_idx), "validation_rows_checkpoint_selected_and_scored": len(val_idx),
                  "test_rows_indexed_or_scored": 0, "new_transitions": 0, "new_episodes": 0,
                  "simulator_or_planner_calls": 0, "recursive_rollout": False, "closed_loop": False,
                  "dagger": False, "jepa": False, "reserve_access": False},
        "validation_reuse_disclosure": lock["validation_reuse_disclosure"],
        "test_exposure_disclosure": lock["test_disclosure"],
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    checkpoint = out_dir / "summary_predictor.npz"
    np.savez_compressed(checkpoint, x_mean=predictor.x_mean, x_std=predictor.x_std,
                        w1=predictor.w1, b1=predictor.b1, w2=predictor.w2, b2=predictor.b2,
                        w3=predictor.w3, b3=predictor.b3)
    predictions_path = out_dir / "train_validation_predictions.npz"
    np.savez_compressed(predictions_path, train_indices=train_idx, validation_indices=val_idx,
                        train_summary_prediction=predicted_train, validation_summary_prediction=predicted_val,
                        validation_constant_summary=constant_val, validation_r0_scores=r0_val_scores,
                        validation_r4_oracle_scores=r4_val_scores, validation_predicted_summary_scores=predicted_scores,
                        validation_teacher_regret=y_val)
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / config["outputs"]["report"]
    report_path.write_text(_render(result), encoding="utf-8")
    manifest = {"schema": "phase15_r4_summary_feasibility_manifest_v1", "status": result["status"],
                "source_commit": result["source_commit"], "lock_sha256": result["lock_sha256"],
                "config_sha256": result["config_sha256"], "feature_dataset_sha256": result["feature_dataset_sha256"],
                "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
                "predictor_sha256": sha256_file(checkpoint), "predictions_sha256": sha256_file(predictions_path),
                "train_rows_fit": len(train_idx), "validation_rows_scored": len(val_idx),
                "test_rows_indexed_or_scored": 0, "new_transitions": 0, "new_episodes": 0,
                "simulator_or_planner_calls": 0, "closed_loop": False, "dagger": False, "reserve_access": False}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "screen": screen,
                      "predicted_summary_validation": learned_val_metrics,
                      "predicted_summary_downstream": downstream["predicted_summary_plus_frozen_R4_head"]["validation"],
                      "report": config["outputs"]["report"]}, indent=2))


if __name__ == "__main__":
    main()
