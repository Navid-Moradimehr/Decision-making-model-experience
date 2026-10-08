"""Freeze the scope and exact artifacts for the Phase15 train-fit audit."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_data import layout_split
from consequence_lab.phase15_generalization import layout_list_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_training_fit_audit.yaml"
LOCK = "experiments/phase15_training_fit_audit.lock.json"
SOURCES = (CONFIG, "reports/phase15_training_fit_audit_protocol.md",
           "scripts/freeze_phase15_training_fit_audit.py", "scripts/run_phase15_training_fit_audit.py",
           "src/consequence_lab/phase15_fit_audit.py", "src/consequence_lab/phase15_learning.py",
           "src/consequence_lab/phase15_data.py", "src/consequence_lab/phase15_generalization.py",
           "tests/test_phase15_fit_audit.py",
           "second-brain/phases/Phase_15_Closed_Loop_Failure_Diagnosis.md",
           "second-brain/16_Experiment_Log.md")


def main() -> None:
    lock_path = ROOT / LOCK
    if lock_path.exists():
        raise SystemExit("refusing to overwrite training-fit audit lock")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit protocol, runner, helper, and tests before locking")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    expected_scope = {"load_frozen_checkpoints": True, "score_training_rows_only": True,
                      "fit_or_update_weights": False, "read_validation_rows": False,
                      "read_test_rows": False, "recompute_any_heldout_metric": False,
                      "simulator_or_planner_calls": False, "reserve_access": False,
                      "closed_loop": False, "dagger": False, "world_model": False}
    if config.get("schema") != "phase15_frozen_checkpoint_training_fit_audit_v1" or config["scope"] != expected_scope:
        raise SystemExit("training-fit audit schema/scope mismatch")
    expected_bands = {"strong_train_fit_mean_layout_macro_regret_max": 0.1,
                      "strong_train_fit_best_action_agreement_min": 0.9,
                      "large_validation_train_regret_gap_min": 0.15,
                      "recorded_validation_macro_regret": {
                          "regression": 0.41029432236604313,
                          "pairwise": 0.3510361523030522}}
    if config["diagnostic_thresholds"] != expected_bands or config["models"] != ["regression", "pairwise"]:
        raise SystemExit("training-fit interpretation bands/model list changed")
    parents = config["parents"]
    parent_hashes = {key: sha256_file(ROOT / path) for key, path in parents.items()}
    prior_manifest = json.loads((ROOT / parents["prior_manifest"]).read_text(encoding="utf-8"))
    if (prior_manifest.get("status") != "completed_single_offline_test"
            or prior_manifest.get("result_sha256") != parent_hashes["prior_result"]
            or prior_manifest.get("run_lock_sha256") != parent_hashes["prior_lock"]
            or prior_manifest.get("regression_checkpoint_sha256") != parent_hashes["regression_checkpoint"]
            or prior_manifest.get("pairwise_checkpoint_sha256") != parent_hashes["pairwise_checkpoint"]
            or prior_manifest.get("feature_dataset_sha256") != parent_hashes["feature_dataset"]
            or prior_manifest.get("test_metrics_computed_once") is not True):
        raise SystemExit("prior frozen checkpoint/dataset hashes do not match manifest")
    with np.load(ROOT / parents["feature_dataset"], allow_pickle=False) as data:
        layouts = data["layouts"].astype(str).tolist()
    split = layout_split(layouts)
    train_layouts = sorted(split["train"])
    train_rows = sum(layout in set(train_layouts) for layout in layouts)
    train_cfg = config["train_split"]
    if (len(train_layouts) != train_cfg["layout_count"] or train_rows != train_cfg["row_count"]
            or layout_list_sha256(train_layouts) != train_cfg["layouts_sha256"]):
        raise SystemExit("frozen training layout split mismatch")
    if config["models"] != ["regression", "pairwise"]:
        raise SystemExit("model list changed")
    lock = {"schema": "phase15_training_fit_audit_lock_v1",
            "status": "authorized_read_only_training_fit_audit",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": sha256_file(ROOT / CONFIG),
            "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
            "parent_sha256": parent_hashes,
            "checkpoint_sha256": {name: parent_hashes[f"{name}_checkpoint"] for name in config["models"]},
            "train_layouts": train_layouts, "train_row_count": train_rows,
            "scope": expected_scope,
            "interpretation_bands": config["diagnostic_thresholds"],
            "test_exposure": "The prior diagnostic holdout was consumed once under the D148 protocol; this audit indexes no validation/test feature or target rows and performs no held-out scoring.",
            "authorization_reference": "User authorized the read-only training-fit audit and conditional continuation if results are promising."}
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": LOCK, "source_commit": lock["source_commit"],
                      "train_layouts": len(train_layouts), "train_rows": train_rows,
                      "scope": expected_scope}, indent=2))


if __name__ == "__main__":
    main()
