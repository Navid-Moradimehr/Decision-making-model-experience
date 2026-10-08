"""Read-only train-split audit of the two frozen Phase15 checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import platform

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_data import layout_split
from consequence_lab.phase15_generalization import layout_list_sha256
from consequence_lab.phase15_fit_audit import interpret_train_fit, restore_frozen_mlp
from consequence_lab.phase15_learning import decision_metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_training_fit_audit.yaml"
LOCK = "experiments/phase15_training_fit_audit.lock.json"


def _validate() -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    lock = json.loads((ROOT / LOCK).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase15_training_fit_audit_lock_v1" or lock.get("status") != "authorized_read_only_training_fit_audit":
        raise RuntimeError("training-fit audit lock status mismatch")
    if lock.get("config_sha256") != sha256_file(ROOT / CONFIG):
        raise RuntimeError("training-fit config hash mismatch")
    if lock.get("source_sha256") != {p: sha256_file(ROOT / p) for p in lock["source_sha256"]}:
        raise RuntimeError("training-fit source hash mismatch")
    if lock.get("parent_sha256") != {key: sha256_file(ROOT / path) for key, path in config["parents"].items()}:
        raise RuntimeError("training-fit parent hash mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("training-fit audit requires clean tree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not (head == lock["source_commit"] or subprocess.run(
            ["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"], cwd=ROOT).returncode == 0):
        raise RuntimeError("audit source commit is not an ancestor of current HEAD")
    if (config["scope"] != lock["scope"]
            or config["scope"].get("read_validation_rows") is not False
            or config["scope"].get("read_test_rows") is not False
            or config["diagnostic_thresholds"] != lock["interpretation_bands"]):
        raise RuntimeError("audit scope differs from lock or permits held-out reads")
    return config, lock


def _render(result: dict) -> str:
    rows = ["# Phase 15 frozen-checkpoint training-fit audit", "",
            "**Status: completed.** This is an in-sample, read-only checkpoint audit; no training or held-out scoring occurred.", "",
            "| Metric | Regression | Pairwise |", "|---|---:|---:|"]
    metrics = result["train_metrics"]
    for label, key in (("Layout-macro mean regret", "mean_normalized_regret_layout_macro"),
                       ("State-micro mean regret", "mean_normalized_regret_state_micro"),
                       ("Median selected regret", "median_normalized_regret"),
                       ("Catastrophic fraction (state micro)", "catastrophic_fraction_state_micro"),
                       ("Best-action-set agreement", "best_action_set_agreement"),
                       ("Pairwise ranking accuracy", "pairwise_ranking_accuracy_macro"),
                       ("Regret-gap-weighted pairwise accuracy", "regret_gap_weighted_pairwise_accuracy_macro")):
        rows.append(f"| {label} | {metrics['regression'][key]:.6f} | {metrics['pairwise'][key]:.6f} |")
    rows.extend(["", "## Training-to-validation regret gap", "",
                 "Validation figures below are copied from the prior locked result; the audit did not read validation rows or recompute those metrics.", "",
                 "| Objective | Train macro regret | Recorded validation macro regret | Gap (validation minus train) |", "|---|---:|---:|---:|"])
    for objective in ("regression", "pairwise"):
        train = metrics[objective]["mean_normalized_regret_layout_macro"]
        validation = result["prior_validation_macro_regret"][objective]
        rows.append(f"| {objective} | {train:.6f} | {validation:.6f} | {validation - train:.6f} |")
    rows.extend(["", f"Train action counts, primitive IDs 0-8: regression `{metrics['regression']['predicted_action_frequency']}`; pairwise `{metrics['pairwise']['predicted_action_frequency']}`.", "",
                 "## Interpretation", "", result["interpretation"], "",
                 "Only the 571 train indices from the 60 frozen training layouts were passed to inference and metrics. Because the source is compressed NPZ, NumPy may materialize full x_R0/y members internally; no held-out row was scored or used. No weights changed, and no simulation, planner, closed-loop, DAgger, or world-model work occurred; reserve remains sealed.", ""])
    return "\n".join(rows)


def main() -> None:
    config, lock = _validate()
    data_path = ROOT / config["parents"]["feature_dataset"]
    train_ids: np.ndarray
    train_layouts_ids: list[str]
    with np.load(data_path, allow_pickle=False) as data:
        layouts = data["layouts"].astype(str)
        split = layout_split(layouts.tolist())
        train_layouts_ids = sorted(split["train"])
        if (sorted(split["train"]) != lock["train_layouts"]
                or layout_list_sha256(split["train"]) != config["train_split"]["layouts_sha256"]):
            raise RuntimeError("training layout split mismatch")
        train_ids = np.asarray([i for i, layout in enumerate(layouts) if layout in set(split["train"])], dtype=np.int64)
        if len(train_ids) != config["train_split"]["row_count"]:
            raise RuntimeError("training row count mismatch")
        # Deliberately index only the train split; never request val/test feature/target rows.
        x_train = np.asarray(data["x_R0"][train_ids], dtype=np.float64)
        y_train = np.asarray(data["y"][train_ids], dtype=np.float64)
        train_layouts = layouts[train_ids]
    train_metrics = {}
    for objective in config["models"]:
        model_path = ROOT / config["checkpoint_dir"] / f"{objective}_checkpoint.npz"
        model = restore_frozen_mlp(model_path, objective)
        train_metrics[objective] = decision_metrics(y_train, model.predict(x_train), train_layouts)
    run_dir = ROOT / config["outputs"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "phase15_frozen_checkpoint_training_fit_audit_result_v1",
        "status": "completed_read_only_train_audit",
        "source_commit": lock["source_commit"],
        "audit_lock_sha256": sha256_file(ROOT / LOCK),
        "dataset_sha256": lock["parent_sha256"]["feature_dataset"],
        "checkpoint_sha256": lock["checkpoint_sha256"],
        "train_split": {"layout_count": len(train_layouts_ids), "row_count": len(train_ids),
                        "layouts_sha256": config["train_split"]["layouts_sha256"]},
        "train_metrics": train_metrics,
        "prior_validation_macro_regret": config["diagnostic_thresholds"]["recorded_validation_macro_regret"],
        "prior_validation_reference_only": True,
        "heldout_rows_passed_to_inference_or_metrics": False,
        "compressed_npz_member_materialization_disclosed": True,
        "weights_updated": False,
        "reserve_access": False,
        "interpretation": interpret_train_fit(
            train_metrics["pairwise"],
            config["diagnostic_thresholds"]["recorded_validation_macro_regret"]["pairwise"],
            config["diagnostic_thresholds"]),
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / config["outputs"]["report"]
    report_path.write_text(_render(result), encoding="utf-8")
    manifest = {"schema": "phase15_training_fit_audit_manifest_v1",
                "status": result["status"], "audit_lock_sha256": result["audit_lock_sha256"],
                "source_commit": lock["source_commit"], "dataset_sha256": result["dataset_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
                "train_rows_scored": len(train_ids), "heldout_rows_scored": 0,
                "weights_updated": False, "reserve_access": False}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "train_macro_regret": {
        key: value["mean_normalized_regret_layout_macro"] for key, value in train_metrics.items()},
        "train_best_action_agreement": {key: value["best_action_set_agreement"] for key, value in train_metrics.items()},
        "interpretation": result["interpretation"]}, indent=2))




if __name__ == "__main__":
    main()
