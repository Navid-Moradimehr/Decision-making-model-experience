"""Freeze the exploratory R0-R4 Phase15 representation screen."""

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
CONFIG = "configs/phase15_representation_screen.yaml"
LOCK = "experiments/phase15_representation_screen.lock.json"
SOURCES = (CONFIG, "reports/phase15_representation_screen_protocol.md",
           "scripts/freeze_phase15_representation_screen.py", "scripts/run_phase15_representation_screen.py",
           "src/consequence_lab/phase15_representation_screen.py", "src/consequence_lab/phase15_learning.py",
           "src/consequence_lab/phase15_data.py", "src/consequence_lab/phase15_generalization.py",
           "tests/test_phase15_representation_screen.py",
           "second-brain/phases/Phase_15_Closed_Loop_Failure_Diagnosis.md",
           "second-brain/16_Experiment_Log.md")


def main() -> None:
    lock_path = ROOT / LOCK
    if lock_path.exists():
        raise SystemExit("refusing to overwrite representation-screen lock")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit representation protocol, runner, tests, and vault before locking")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    expected_scope = {"train_rows_used_for_fit": True,
                      "validation_rows_used_for_checkpoint_selection_and_development_screen": True,
                      "test_rows_used_for_fit_selection_or_metrics": False,
                      "rerun_or_recompute_test_metrics": False,
                      "refit_R0_reproduction": True, "fit_R1_to_R4": True,
                      "simulator_or_planner_calls": False, "new_transitions_or_episodes": False,
                      "closed_loop": False, "dagger": False, "world_model": False,
                      "reserve_access": False}
    if config.get("schema") != "phase15_representation_screen_v1" or config["scope"] != expected_scope:
        raise SystemExit("representation-screen schema/scope mismatch")
    expected_dims = {"R0": 61, "R1": 160, "R2": 421, "R3": 133, "R4": 124}
    if config["feature_dimensions"] != expected_dims or list(config["feature_dimensions"]) != ["R0", "R1", "R2", "R3", "R4"]:
        raise SystemExit("feature dimensions/order changed")
    expected_status = {"R0": "deployable_current_state",
                       "R1": "privileged_exact_transition_diagnostic",
                       "R2": "privileged_exact_transition_diagnostic",
                       "R3": "privileged_exact_transition_diagnostic",
                       "R4": "non_deployable_exact_continuation_oracle"}
    if config["feature_status"] != expected_status:
        raise SystemExit("representation deployability labels changed")
    if config["model"] != {"objective": "pairwise", "seed": 15015, "hidden": [32, 32],
                           "max_epochs": 1200, "patience": 100, "learning_rate": 0.002,
                           "l2": 0.00001, "weighting": "equal_total_weight_per_train_layout",
                           "checkpoint": "minimum_validation_layout_macro_regret_then_pairwise_loss",
                           "optimizer": "existing_numpy_adam",
                           "target": "nine_saved_normalized_teacher_forced_action_regrets",
                           "checkpoint_selection_split": "validation"}:
        raise SystemExit("representation-screen learner settings changed")
    if config["screen_gate"] != {"relative_to_R0_validation": {
            "minimum_absolute_macro_regret_improvement": 0.1,
            "minimum_absolute_catastrophic_fraction_improvement": 0.1},
            "purpose": "nominate_only_for_fresh_independent_confirmation",
            "validation_comparison_is_confirmatory": False}:
        raise SystemExit("representation-screen interpretation gate changed")
    if config["r0_reproduction_gate"] != {
            "best_epoch": 113, "validation_layout_macro_regret": 0.3510361523030522,
            "validation_best_action_set_agreement": 0.26732673267326734, "tolerance": 1e-10}:
        raise SystemExit("R0 reproduction gate changed")
    if config["split"]["scheme"] != "sha256_layout_group_phase15_v1" or config["split"]["salt"] != "phase15-layout-split-v1:":
        raise SystemExit("layout split algorithm changed")
    parents = config["parents"]
    parent_hashes = {key: sha256_file(ROOT / path) for key, path in parents.items()}
    obj_manifest = json.loads((ROOT / parents["prior_objective_manifest"]).read_text(encoding="utf-8"))
    fit_manifest = json.loads((ROOT / parents["prior_fit_audit_manifest"]).read_text(encoding="utf-8"))
    feature_manifest = json.loads((ROOT / parents["feature_manifest"]).read_text(encoding="utf-8"))
    if (obj_manifest.get("status") != "completed_single_offline_test"
            or obj_manifest.get("result_sha256") != parent_hashes["prior_objective_result"]
            or obj_manifest.get("run_lock_sha256") != parent_hashes["prior_objective_lock"]
            or obj_manifest.get("feature_dataset_sha256") != parent_hashes["feature_dataset"]):
        raise SystemExit("prior objective manifest/result/lock/dataset mismatch")
    checkpoint_hashes = {name: sha256_file(ROOT / config["checkpoint_dir"] / f"{name}_checkpoint.npz")
                         for name in ("regression", "pairwise")}
    if (obj_manifest.get("pairwise_checkpoint_sha256") != checkpoint_hashes["pairwise"]
            or obj_manifest.get("regression_checkpoint_sha256") != checkpoint_hashes["regression"]):
        raise SystemExit("frozen D148 checkpoint mismatch")
    if (fit_manifest.get("status") != "completed_read_only_train_audit"
            or fit_manifest.get("result_sha256") != parent_hashes["prior_fit_audit_result"]
            or fit_manifest.get("audit_lock_sha256") != parent_hashes["prior_fit_audit_lock"]
            or fit_manifest.get("dataset_sha256") != parent_hashes["feature_dataset"]):
        raise SystemExit("prior D149 audit artifact mismatch")
    if feature_manifest.get("feature_dataset_sha256") != parent_hashes["feature_dataset"]:
        raise SystemExit("feature manifest does not bind dataset")
    for name, dim in expected_dims.items():
        if feature_manifest["representations"][name]["dimension"] != dim:
            raise SystemExit(f"feature manifest dimension mismatch for {name}")
        if feature_manifest["representations"][name]["runtime_status"] != expected_status[name]:
            raise SystemExit(f"feature manifest availability mismatch for {name}")
    with np.load(ROOT / parents["feature_dataset"], allow_pickle=False) as data:
        layouts = data["layouts"].astype(str).tolist()
    split = layout_split(layouts)
    split_cfg = config["split"]
    split_lock = {}
    for name in ("train", "validation", "test"):
        values = sorted(split[name])
        digest = layout_list_sha256(values)
        row_count = sum(layout in set(values) for layout in layouts)
        expected = split_cfg[f"{name}_layout_count"], split_cfg[f"{name}_row_count"], split_cfg[f"{name}_layouts_sha256"]
        actual = len(values), row_count, digest
        if actual != expected:
            raise SystemExit(f"Phase15 {name} split mismatch")
        split_lock[name] = {"layouts": values, "row_count": row_count, "layout_list_sha256": digest}
    sources = {path: sha256_file(ROOT / path) for path in SOURCES}
    lock = {"schema": "phase15_representation_screen_lock_v1",
            "status": "authorized_exploratory_train_validation_screen",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": sha256_file(ROOT / CONFIG), "source_sha256": sources,
            "parent_sha256": parent_hashes, "checkpoint_sha256": checkpoint_hashes,
            "split": split_lock, "scope": expected_scope,
            "screen_gate": config["screen_gate"],
            "prior_test_disclosure": "The D148 diagnostic test was consumed once. This runner must not index its rows or compute any test metric. NPZ compression may materialize full feature/target members in memory; only train/validation indices are passed to models and metrics.",
            "validation_reuse_disclosure": "Validation was used for D148 checkpoint selection and is reused for this exploratory screen. No independent inference follows from the comparison.",
            "authorization_reference": "User said keep going after D149 recommended an offline saved R1-R4 representation diagnosis; no world model, DAgger, closed-loop, or new environment."}
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": LOCK, "source_commit": lock["source_commit"],
                      "train_rows": split_lock["train"]["row_count"],
                      "validation_rows": split_lock["validation"]["row_count"],
                      "test_rows_indexed": 0, "scope": expected_scope}, indent=2))


if __name__ == "__main__":
    main()
