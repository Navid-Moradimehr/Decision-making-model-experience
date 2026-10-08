"""Freeze the one full-split regression-vs-pairwise offline comparison."""

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
CONFIG = "configs/phase15_fullsplit_objective_generalization.yaml"
LOCK = "experiments/phase15_fullsplit_objective_generalization.lock.json"
SOURCES = (
    CONFIG,
    "reports/phase15_fullsplit_objective_generalization_protocol.md",
    "scripts/freeze_phase15_fullsplit_lock.py",
    "scripts/run_phase15_fullsplit_objective_generalization.py",
    "src/consequence_lab/phase15_learning.py",
    "src/consequence_lab/phase15_data.py",
    "src/consequence_lab/phase15_generalization.py",
    "tests/test_phase15_generalization.py",
    "second-brain/phases/Phase_15_Closed_Loop_Failure_Diagnosis.md",
    "second-brain/16_Experiment_Log.md",
)


def main() -> None:
    lock_path = ROOT / LOCK
    if lock_path.exists():
        raise SystemExit("refusing to overwrite full-split lock")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit full-split protocol, runner, tests, and vault before locking")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    scope = {"new_training_only_two_fixed_objectives": True,
             "heldout_test_metrics_once_after_checkpoint_freeze": True,
             "heldout_model_selection": False, "test_reruns_or_tuning": False,
             "new_transitions": False, "new_episodes": False, "planner_calls": False,
             "closed_loop": False, "dagger": False, "calibration": False,
             "world_model": False, "reserve_access": False}
    if config.get("schema") != "phase15_fullsplit_objective_generalization_v1" or config["scope"] != scope:
        raise SystemExit("full-split protocol/scope mismatch")
    if config["resource_caps"] != {"wall_seconds": 10800, "output_bytes": 20971520}:
        raise SystemExit("full-split resource caps mismatch")
    expected_model = {"seed": 15015, "hidden": [32, 32], "max_epochs": 1200, "patience": 100,
                      "learning_rate": 0.002, "l2": 0.00001,
                      "weighting": "equal_total_weight_per_train_layout",
                      "checkpoint": "minimum_validation_layout_macro_regret_then_objective_loss"}
    if config["models"] != [{"objective": objective, **expected_model} for objective in ("regression", "pairwise")]:
        raise SystemExit("model settings differ from frozen two-objective plan")
    if config["decision_rule"] != {"pairwise_improvement_requires_mean_difference_below_zero": True,
                                   "pairwise_improvement_requires_ci_upper_below_zero": True}:
        raise SystemExit("held-out decision rule was modified")
    if config["features"] != {"representation": "R0", "dimension": 61,
                              "target": "nine saved normalized teacher forced-action regrets",
                              "student_input_allowlist": ["R0_state_only"]}:
        raise SystemExit("feature/information contract changed")
    if config["metrics"] != {"primary": "paired_test_layout_macro_regret_difference_pairwise_minus_regression",
                             "bootstrap_resamples": 10000, "bootstrap_seed": 150151,
                             "confidence": 0.95, "catastrophic_regret_threshold": 0.5,
                             "inference_latency_probe_split": "validation"}:
        raise SystemExit("test metrics or bootstrap protocol changed")
    if config["split"]["scheme"] != "sha256_layout_group_phase15_v1" or config["split"]["salt"] != "phase15-layout-split-v1:":
        raise SystemExit("layout-group split algorithm changed")
    parents = config["parents"]
    parent_hashes = {key: sha256_file(ROOT / path) for key, path in parents.items()}
    regression32 = json.loads((ROOT / parents["regression32_result"]).read_text(encoding="utf-8"))
    pairwise32 = json.loads((ROOT / parents["pairwise32_result"]).read_text(encoding="utf-8"))
    regression_manifest = json.loads((ROOT / parents["regression32_manifest"]).read_text(encoding="utf-8"))
    pairwise_manifest = json.loads((ROOT / parents["pairwise32_manifest"]).read_text(encoding="utf-8"))
    pairwise_lock = json.loads((ROOT / parents["pairwise32_lock"]).read_text(encoding="utf-8"))
    if (regression32.get("status") != "reproduced"
            or pairwise32.get("status") != "memorization_gate_pass"
            or regression_manifest.get("diagnostic_sha256") != parent_hashes["regression32_result"]
            or pairwise_manifest.get("result_sha256") != parent_hashes["pairwise32_result"]
            or pairwise_manifest.get("run_lock_sha256") != parent_hashes["pairwise32_lock"]
            or pairwise_lock.get("status") != "authorized_single_pairwise_memorization_diagnostic"):
        raise SystemExit("parent 32-row diagnostic artifacts are not in the expected terminal state")
    with np.load(ROOT / parents["feature_dataset"], allow_pickle=False) as data:
        layouts = data["layouts"].astype(str).tolist()
    split = layout_split(layouts)
    split_cfg = config["split"]
    for name in ("train", "validation", "test"):
        keys = split[name]
        count_rows = sum(value in set(keys) for value in layouts)
        if (len(keys) != split_cfg[f"{name}_layout_count"]
                or count_rows != split_cfg[f"{name}_row_count"]
                or layout_list_sha256(keys) != split_cfg[f"{name}_layouts_sha256"]):
            raise SystemExit(f"frozen {name} split changed")
    if set(split["train"]) & set(split["validation"]) or set(split["train"]) & set(split["test"]) or set(split["validation"]) & set(split["test"]):
        raise SystemExit("layout split is not disjoint")
    missing = [path for path in SOURCES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing lock sources: {missing}")
    lock = {
        "schema": "phase15_fullsplit_objective_generalization_lock_v1",
        "status": "authorized_single_offline_objective_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_path": CONFIG, "config_sha256": sha256_file(ROOT / CONFIG),
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
        "parent_sha256": parent_hashes,
        "feature_dataset_sha256": parent_hashes["feature_dataset"],
        "split": {name: {"layouts": sorted(split[name]), "layout_count": len(split[name]),
                         "row_count": sum(value in set(split[name]) for value in layouts),
                         "layout_list_sha256": layout_list_sha256(split[name])}
                  for name in ("train", "validation", "test")},
        "scope": scope, "resource_caps": config["resource_caps"],
        "reserve_seed_block": {"start": 130000, "count": 200, "access": False},
        "authorization_reference": "User said continue after the Phase15 pairwise 32-row gate passed; prior plan excludes tuning, reserve, and closed-loop work.",
        "test_exposure_disclosure": "Test-layout R0/target arrays were materialized in the parent NPZ and loaded by the earlier v2 runner before its training gate stopped. No held-out metrics or test-driven selection were performed before this locked one-time diagnostic.",
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": LOCK, "source_commit": lock["source_commit"],
                      "split_rows": {k: v["row_count"] for k, v in lock["split"].items()},
                      "scope": scope}, indent=2))


if __name__ == "__main__":
    main()
