"""Exploratory Phase15 R0-R4 pairwise screen on train/validation rows only."""

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
from consequence_lab.phase15_learning import TinyRegretMLP, decision_metrics
from consequence_lab.phase15_representation_screen import (
    representation_screen, train_validation_indices, validate_r0_reproduction,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_representation_screen.yaml"
LOCK = "experiments/phase15_representation_screen.lock.json"


def _validate() -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    lock = json.loads((ROOT / LOCK).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase15_representation_screen_lock_v1" or lock.get("status") != "authorized_exploratory_train_validation_screen":
        raise RuntimeError("representation-screen lock status mismatch")
    if lock.get("config_sha256") != sha256_file(ROOT / CONFIG):
        raise RuntimeError("representation-screen config hash mismatch")
    if lock.get("source_sha256") != {p: sha256_file(ROOT / p) for p in lock["source_sha256"]}:
        raise RuntimeError("representation-screen source hash mismatch")
    if lock.get("parent_sha256") != {key: sha256_file(ROOT / path) for key, path in config["parents"].items()}:
        raise RuntimeError("representation-screen parent hash mismatch")
    if lock.get("checkpoint_sha256") != {
            name: sha256_file(ROOT / config["checkpoint_dir"] / f"{name}_checkpoint.npz")
            for name in ("regression", "pairwise")}:
        raise RuntimeError("D148 checkpoint hash mismatch")
    if config["scope"] != lock["scope"]:
        raise RuntimeError("representation-screen scope mismatch")
    if config["screen_gate"] != lock["screen_gate"]:
        raise RuntimeError("representation-screen development gate mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("representation screen requires clean tree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not (head == lock["source_commit"] or subprocess.run(
            ["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"], cwd=ROOT).returncode == 0):
        raise RuntimeError("screen source commit is not an ancestor of current HEAD")
    return config, lock


def _save_checkpoint(path: Path, model: TinyRegretMLP) -> None:
    np.savez_compressed(path, x_mean=model.x_mean, x_std=model.x_std,
                        w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2, w3=model.w3, b3=model.b3)


def _render(result: dict) -> str:
    lines = ["# Phase 15 R0-R4 representation screen result", "",
             "**Status: exploratory development screen.** Validation was reused from D148; these results are not independent confirmation.", "",
             "## Train metrics", "",
             "| Representation | Dim | Parameters | Best epoch | Train layout-macro regret | Train best-action agreement | Train catastrophic fraction |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, record in result["models"].items():
        metric = record["train_metrics"]
        lines.append(f"| {name} | {record['input_dimension']} | {record['parameters']} | {record['best_epoch']} | "
                     f"{metric['mean_normalized_regret_layout_macro']:.6f} | {metric['best_action_set_agreement']:.6f} | "
                     f"{metric['catastrophic_fraction_state_micro']:.6f} |")
    lines.extend(["", "## Validation development screen", "",
                  "| Representation | Layout-macro regret | State-micro regret | Best-action agreement | Catastrophic fraction | Pairwise rank accuracy | Improvement vs R0 gate |",
                  "|---|---:|---:|---:|---:|---:|---|"])
    for name, record in result["models"].items():
        metric = record["validation_metrics"]
        screen = "baseline" if name == "R0" else (
            "promising for fresh confirmation only" if result["screen"][name]["promising_for_fresh_confirmation_only"] else "screen not met")
        lines.append(f"| {name} | {metric['mean_normalized_regret_layout_macro']:.6f} | "
                     f"{metric['mean_normalized_regret_state_micro']:.6f} | {metric['best_action_set_agreement']:.6f} | "
                     f"{metric['catastrophic_fraction_state_micro']:.6f} | {metric['pairwise_ranking_accuracy_macro']:.6f} | {screen} |")
    lines.extend(["", "## Relative screen values", "",
                  "| Representation | Macro regret gain vs R0 | Catastrophic fraction gain vs R0 |", "|---|---:|---:|"])
    for name, metrics in result["screen"].items():
        lines.append(f"| {name} | {metrics['validation_macro_regret_gain_vs_R0']:.6f} | "
                     f"{metrics['validation_catastrophic_fraction_gain_vs_R0']:.6f} |")
    lines.extend(["", "## R0 integrity check and fitting cost", "",
                  f"R0 reproduction gate: **{'passed' if result['r0_reproduction_gate_passed'] else 'failed'}**. "
                  f"Expected epoch 113 / validation macro regret 0.3510361523 / agreement 0.2673267327; "
                  f"observed epoch {result['models']['R0']['best_epoch']} / "
                  f"{result['models']['R0']['validation_metrics']['mean_normalized_regret_layout_macro']:.10f} / "
                  f"{result['models']['R0']['validation_metrics']['best_action_set_agreement']:.10f}.", "",
                  "| Representation | Fit seconds |", "|---|---:|"])
    for name, record in result["models"].items():
        lines.append(f"| {name} | {record['fit_wall_seconds']:.3f} |")
    lines.extend(["", "## Information boundary and interpretation", "",
                  "R0 is the only deployable input. R1-R3 use privileged exact transition/pulse consequences; R4 is an exact teacher-continuation oracle and is non-deployable. Improvements in these branches diagnose information value, not a functioning world model or policy.", "",
                  "Only train and validation indices were passed to learners and metrics. The compressed NPZ may materialize full feature/target members internally; no test rows were indexed, scored, or used for selection. The D148 diagnostic test block remains consumed and was not revisited. No new transition, episode, planner call, closed-loop action, DAgger, learned world model, or reserve data was used.", "",
                  "Validation was already used for R0 checkpoint selection in D148, so this comparison is a development screen only. A positive result nominates a representation for a separately locked fresh-layout confirmation; it does not authorize that run.", ""])
    return "\n".join(lines)


def main() -> None:
    config, lock = _validate()
    output_dir = ROOT / config["outputs"]["run_dir"]
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = ROOT / config["parents"]["feature_dataset"]
    with np.load(dataset, allow_pickle=False) as archive:
        layouts = archive["layouts"].astype(str).tolist()
        train_idx, val_idx, _, _ = train_validation_indices(layouts, config["split"])
        # The compressed NPZ materializes each requested member; only train/validation
        # indices are passed downstream. The consumed test split is never indexed.
        y_train = np.asarray(archive["y"][train_idx], dtype=np.float64)
        y_val = np.asarray(archive["y"][val_idx], dtype=np.float64)
        train_layouts = np.asarray(layouts, dtype=str)[train_idx]
        val_layouts = np.asarray(layouts, dtype=str)[val_idx]
        x_splits = {}
        for name, dim in config["feature_dimensions"].items():
            full = archive[f"x_{name}"]
            if full.shape != (len(layouts), dim):
                raise RuntimeError(f"{name} feature shape mismatch")
            x_splits[name] = (np.asarray(full[train_idx], dtype=np.float64),
                              np.asarray(full[val_idx], dtype=np.float64))
    model_cfg = config["model"]
    records = {}
    for name in config["feature_dimensions"]:
        x_train, x_val = x_splits[name]
        start = time.perf_counter()
        model = TinyRegretMLP(model_cfg["seed"], model_cfg["objective"],
                              max_epochs=model_cfg["max_epochs"], patience=model_cfg["patience"],
                              learning_rate=model_cfg["learning_rate"], l2=model_cfg["l2"])
        model.fit(x_train, y_train, train_layouts, x_val, y_val, val_layouts)
        elapsed = time.perf_counter() - start
        train_metrics = decision_metrics(y_train, model.predict(x_train), train_layouts)
        validation_metrics = decision_metrics(y_val, model.predict(x_val), val_layouts)
        record = {"input_dimension": int(x_train.shape[1]), "parameters": model.parameter_count,
                  "best_epoch": model.best_epoch, "epochs_run": model.epochs_run,
                  "fit_wall_seconds": elapsed, "validation_objective_loss": model.validation_loss,
                  "train_metrics": train_metrics, "validation_metrics": validation_metrics}
        if name == "R0":
            try:
                validate_r0_reproduction(record, config["r0_reproduction_gate"])
            except ValueError as exc:
                abort = {"schema": "phase15_representation_screen_abort_v1", "status": "stopped_r0_reproduction_gate",
                         "reason": str(exc), "observed_R0": record,
                         "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
                         "test_rows_scored": 0, "reserve_access": False}
                (output_dir / "abort.json").write_text(json.dumps(abort, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(json.dumps({"status": abort["status"], "reason": str(exc)}, indent=2))
                raise SystemExit(2)
        records[name] = record
        _save_checkpoint(output_dir / f"{name}_checkpoint.npz", model)
    screen = representation_screen(
        {name: records[name]["validation_metrics"] for name in records},
        config["screen_gate"]["relative_to_R0_validation"]["minimum_absolute_macro_regret_improvement"],
        config["screen_gate"]["relative_to_R0_validation"]["minimum_absolute_catastrophic_fraction_improvement"])
    result = {"schema": "phase15_representation_screen_result_v1", "status": "completed_exploratory_screen",
              "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
              "feature_dataset_sha256": lock["parent_sha256"]["feature_dataset"],
              "split": {name: {key: value for key, value in lock["split"][name].items() if key != "layouts"}
                        for name in ("train", "validation")},
              "models": records, "screen": screen,
              "r0_reproduction_gate_passed": True,
              "scope": {"train_rows_scored": len(train_idx), "validation_rows_scored": len(val_idx),
                        "test_rows_scored": 0, "weights_updated": True, "new_transitions": 0,
                        "new_episodes": 0, "planner_calls": 0, "closed_loop": False,
                        "dagger": False, "world_model": False, "reserve_access": False},
              "test_exposure_disclosure": lock["prior_test_disclosure"],
              "validation_reuse_disclosure": lock["validation_reuse_disclosure"],
              "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}}
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / config["outputs"]["report"]
    report_path.write_text(_render(result), encoding="utf-8")
    checkpoint_hashes = {name: sha256_file(output_dir / f"{name}_checkpoint.npz") for name in records}
    manifest = {"schema": "phase15_representation_screen_manifest_v1", "status": result["status"],
                "source_commit": lock["source_commit"], "lock_sha256": result["lock_sha256"],
                "feature_dataset_sha256": result["feature_dataset_sha256"],
                "checkpoint_sha256": checkpoint_hashes, "result_sha256": sha256_file(result_path),
                "report_sha256": sha256_file(report_path), "train_rows_scored": len(train_idx),
                "validation_rows_scored": len(val_idx), "test_rows_scored": 0,
                "new_transitions": 0, "new_episodes": 0, "planner_calls": 0,
                "reserve_access": False, "closed_loop": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "screen": screen,
                      "r0_reproduction_gate_passed": True,
                      "report": config["outputs"]["report"]}, indent=2))


if __name__ == "__main__":
    main()
