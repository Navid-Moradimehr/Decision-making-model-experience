"""Freeze inputs and scope for the Phase15 R4-summary predictor screen."""

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
CONFIG = "configs/phase15_r4_summary_feasibility.yaml"
LOCK = "experiments/phase15_r4_summary_feasibility.lock.json"
SOURCES = (CONFIG, "reports/phase15_r4_summary_feasibility_protocol.md",
           "scripts/freeze_phase15_r4_summary_feasibility.py", "scripts/run_phase15_r4_summary_feasibility.py",
           "src/consequence_lab/phase15_outcome_summary.py",
           "src/consequence_lab/phase15_representation_screen.py",
           "src/consequence_lab/phase15_learning.py", "src/consequence_lab/phase15_data.py",
           "src/consequence_lab/phase15_generalization.py", "src/consequence_lab/phase15_fit_audit.py",
           "src/consequence_lab/phase14.py",
           "tests/test_phase15_outcome_summary.py", "tests/test_phase15_representation_screen.py",
           "second-brain/phases/Phase_15_Closed_Loop_Failure_Diagnosis.md",
           "second-brain/16_Experiment_Log.md")


def main() -> None:
    lock_path = ROOT / LOCK
    if lock_path.exists():
        raise SystemExit("refusing to overwrite R4-summary feasibility lock")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit D151 protocol, helper, runner, tests, and vault before locking")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    expected_scope = {"fit_summary_predictor": True, "fit_or_update_decision_heads": False,
                      "train_rows_used_for_fit": True,
                      "validation_used_for_checkpoint_selection_and_development_screen": True,
                      "test_rows_used_for_fit_selection_or_metrics": False,
                      "rerun_or_recompute_test_metrics": False,
                      "simulator_or_planner_calls": False, "new_transitions_or_episodes": False,
                      "recursive_rollout": False, "closed_loop": False, "dagger": False,
                      "jepa": False, "reserve_access": False}
    if config.get("schema") != "phase15_r4_summary_feasibility_v1" or config["scope"] != expected_scope:
        raise SystemExit("D151 schema/scope mismatch")
    if config["feature_contract"] != {"student_input": "R0_state_only", "input_dimension": 61,
            "target": "R4[61:124]_exact_per_action_continuation_summary", "output_dimension": 63,
            "action_count": 9, "per_action_fields": ["terminal_outcome_onehot_4", "terminal_steps",
            "terminal_goal_distance", "noncoast_effort"]}:
        raise SystemExit("D151 feature contract changed")
    if config["model"] != {"architecture": [61, 32, 32, 63], "output_activation": "sigmoid",
            "loss": "layout_weighted_mean_squared_error", "seed": 15151,
            "max_epochs": 1200, "patience": 100, "learning_rate": 0.002, "l2": 0.00001,
            "input_standardization": "train_only", "target_standardization": "none_all_targets_in_0_1",
            "checkpoint": "minimum_validation_feature_mse"}:
        raise SystemExit("D151 model settings changed")
    if config["screen_gate"] != {"summary_rmse_relative_improvement_vs_constant_minimum": 0.25,
            "downstream_macro_regret_maximum": 0.18,
            "downstream_catastrophic_fraction_maximum": 0.15,
            "outcome": "nominate_only_for_fresh_independent_feasibility_confirmation"}:
        raise SystemExit("D151 screen thresholds changed")
    if config["head_reproduction_gate"] != {
            "tolerance": 1e-10,
            "R0": {"best_epoch": 113,
                   "validation_layout_macro_regret": 0.3510361523030522,
                   "validation_catastrophic_fraction_state_micro": 0.32673267326732675,
                   "validation_best_action_set_agreement": 0.26732673267326734},
            "R4": {"validation_layout_macro_regret": 0.07915851363150343,
                   "validation_catastrophic_fraction_state_micro": 0.04950495049504951,
                   "validation_best_action_set_agreement": 0.6485148514851485}}:
        raise SystemExit("D151 frozen-head reproduction criteria changed")
    parents = config["parents"]
    parent_hashes = {key: sha256_file(ROOT / path) for key, path in parents.items()}
    rep_manifest = json.loads((ROOT / parents["representation_manifest"]).read_text(encoding="utf-8"))
    rep_result = json.loads((ROOT / parents["representation_result"]).read_text(encoding="utf-8"))
    feature_manifest = json.loads((ROOT / parents["feature_manifest"]).read_text(encoding="utf-8"))
    if (rep_manifest.get("status") != "completed_exploratory_screen"
            or rep_manifest.get("result_sha256") != parent_hashes["representation_result"]
            or rep_manifest.get("lock_sha256") != parent_hashes["representation_lock"]
            or rep_manifest.get("feature_dataset_sha256") != parent_hashes["feature_dataset"]
            or rep_manifest.get("report_sha256") != parent_hashes["representation_report"]
            or rep_result.get("r0_reproduction_gate_passed") is not True
            or rep_result.get("screen", {}).get("R4", {}).get("promising_for_fresh_confirmation_only") is not True):
        raise SystemExit("D150 parent artifacts do not show the locked positive R4 screen")
    for name in ("R0", "R4"):
        checkpoint_path = ROOT / config["representation_checkpoint_dir"] / f"{name}_checkpoint.npz"
        if sha256_file(checkpoint_path) != rep_manifest["checkpoint_sha256"][name]:
            raise SystemExit(f"D150 {name} checkpoint hash mismatch")
    gate = config["head_reproduction_gate"]
    r0, r4 = rep_result["models"]["R0"], rep_result["models"]["R4"]
    metric_keys = {"validation_layout_macro_regret": "mean_normalized_regret_layout_macro",
                   "validation_catastrophic_fraction_state_micro": "catastrophic_fraction_state_micro",
                   "validation_best_action_set_agreement": "best_action_set_agreement"}
    if (r0["best_epoch"] != gate["R0"]["best_epoch"]
            or any(abs(r0["validation_metrics"][metric_keys[key]] - value) > gate["tolerance"]
                   for key, value in gate["R0"].items() if key != "best_epoch")
            or any(abs(r4["validation_metrics"][metric_keys[key]] - value) > gate["tolerance"]
                   for key, value in gate["R4"].items())):
        raise SystemExit("D150 baseline metrics differ from the locked record")
    if feature_manifest.get("feature_dataset_sha256") != parent_hashes["feature_dataset"]:
        raise SystemExit("feature manifest dataset hash mismatch")
    if feature_manifest["representations"]["R0"]["dimension"] != 61 or feature_manifest["representations"]["R4"]["dimension"] != 124:
        raise SystemExit("R0/R4 feature dimensions do not match protocol")
    with np.load(ROOT / parents["feature_dataset"], allow_pickle=False) as data:
        layouts = data["layouts"].astype(str).tolist()
    actual_split = layout_split(layouts)
    split_cfg = config["split"]
    split_lock = {}
    for name in ("train", "validation", "test"):
        names = sorted(actual_split[name])
        rows = sum(layout in set(names) for layout in layouts)
        digest = layout_list_sha256(names)
        expected = (split_cfg[f"{name}_layout_count"], split_cfg[f"{name}_row_count"], split_cfg[f"{name}_layouts_sha256"])
        if (len(names), rows, digest) != expected:
            raise SystemExit(f"D151 {name} split mismatch")
        split_lock[name] = {"layouts": names, "row_count": rows, "layout_list_sha256": digest}
    lock = {"schema": "phase15_r4_summary_feasibility_lock_v1",
            "status": "authorized_offline_summary_prediction_screen",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": sha256_file(ROOT / CONFIG),
            "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
            "parent_sha256": parent_hashes, "split": split_lock,
            "frozen_head_checkpoint_sha256": {
                name: sha256_file(ROOT / config["representation_checkpoint_dir"] / f"{name}_checkpoint.npz")
                for name in ("R0", "R4")},
            "scope": expected_scope, "screen_gate": config["screen_gate"],
            "test_disclosure": "D148 test metrics were consumed once and are not accessed. Compressed NPZ may materialize arrays internally; only train/validation indices enter fit, head inference, or metrics.",
            "validation_reuse_disclosure": "The 20-layout validation group was already used in D148/D150. This is exploratory and not independent confirmation.",
            "authorization_reference": "D150 oracle R4 passed the user-authorized screen; user requested continuing to next steps. D151 is limited to saved outcome-summary feasibility, not a full world model."}
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": LOCK, "source_commit": lock["source_commit"],
                      "train_rows": split_lock["train"]["row_count"],
                      "validation_rows": split_lock["validation"]["row_count"],
                      "test_rows_indexed": 0, "scope": expected_scope}, indent=2))


if __name__ == "__main__":
    main()
