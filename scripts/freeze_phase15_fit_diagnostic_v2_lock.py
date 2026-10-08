"""Freeze the dtype-explicit correction for the same single Phase15 fit replay."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

from consequence_lab.phase14 import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_offline_v2_fit_diagnostic_v2.yaml"
LOCK = "experiments/phase15_offline_v2_fit_diagnostic_v2.lock.json"
SOURCES = (
    CONFIG,
    "reports/phase15_offline_v2_fit_diagnostic_protocol.md",
    "reports/phase15_offline_v2_fit_diagnostic_preflight_abort.md",
    "scripts/freeze_phase15_fit_diagnostic_v2_lock.py",
    "scripts/run_phase15_fit_diagnostic_v2.py",
    "src/consequence_lab/phase15_learning.py",
    "tests/test_phase15.py",
    "second-brain/phases/Phase_15_Closed_Loop_Failure_Diagnosis.md",
    "second-brain/16_Experiment_Log.md",
)


def main() -> None:
    lock_path = ROOT / LOCK
    if lock_path.exists():
        raise SystemExit("refusing to overwrite corrected diagnostic lock")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit the corrected protocol, runner, tests, and vault before locking")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    if config["schema"] != "phase15_v2_failed_cell_replay_v2":
        raise SystemExit("wrong corrected diagnostic schema")
    scope = {"exact_parent_cell_only": True, "reuse_saved_phase14g_features_only": True,
             "compute_heldout_metrics": False, "heldout_rows_for_model_selection": False,
             "new_transitions": False, "new_episodes": False, "exact_planner_calls": False,
             "closed_loop": False, "dagger": False, "world_model": False, "reserve_access": False}
    if config["scope"] != scope or config["resource_caps"] != {"wall_seconds": 10800, "output_bytes": 20971520}:
        raise SystemExit("scope or user resource caps differ")
    cell = config["cell"]
    if cell["row_indices"] != [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 28, 29, 30, 31, 32,
                               33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 58]:
        raise SystemExit("corrected cell row IDs changed")
    training = {"seed": 15015, "hidden": [32, 32], "max_epochs": 1200, "patience": 100,
                "learning_rate": 0.002, "l2": 0.00001,
                "layout_weighting": "equal_total_weight_per_layout",
                "model_selection": "minimum_regret_then_validation_loss"}
    if config["training"] != training:
        raise SystemExit("training settings changed")
    parent = config["parent"]
    parent_hashes = {key: sha256_file(ROOT / parent[key]) for key in
                     ("config", "lock", "result", "result_manifest", "feature_manifest", "feature_dataset")}
    if (parent_hashes["lock"] != "5CFDCDD2024DFC4C7A6105F47B2BFE8C96EE3845EE911F14FF13A939902D7729"
            or parent_hashes["result"] != "3AC34BAF90B5060CA4741C504F5824107B53E1814267A5D10078966D21728015"
            or parent_hashes["feature_dataset"] != "26D532A918E5AE913EE7F5A5E58BC60E68ED71D6CE38889CCDB7684363191386"):
        raise SystemExit("Phase15 v2 parent artifact changed")
    sources = {path: sha256_file(ROOT / path) for path in SOURCES}
    lock = {
        "schema": "phase15_v2_failed_cell_replay_lock_v2",
        "status": "authorized_single_diagnostic_replay_after_hash_preflight_correction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_path": CONFIG, "config_sha256": sha256_file(ROOT / CONFIG),
        "source_sha256": sources, "parent_sha256": parent_hashes,
        "failed_preflight_lock_sha256": sha256_file(ROOT / "experiments/phase15_offline_v2_fit_diagnostic.lock.json"),
        "failed_preflight_report_sha256": sha256_file(ROOT / "reports/phase15_offline_v2_fit_diagnostic_preflight_abort.md"),
        "feature_dataset_sha256": parent_hashes["feature_dataset"],
        "scope": scope, "resource_caps": config["resource_caps"],
        "reserve_seed_block": {"start": 130000, "count": 200, "access": False},
        "authorization_reference": "Same single-cell replay authorization; first attempt aborted before fit on dtype-specific hash serialization. This lock allows only that fit once, with no changes to rows/model/training settings.",
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": LOCK, "source_commit": lock["source_commit"], "scope": scope,
                      "resource_caps": config["resource_caps"]}, indent=2))


if __name__ == "__main__":
    main()
