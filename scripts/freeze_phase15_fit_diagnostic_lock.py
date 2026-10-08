"""Freeze provenance for the single authorized Phase15 failed-cell replay."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

from consequence_lab.phase14 import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_offline_v2_fit_diagnostic.yaml"
LOCK = "experiments/phase15_offline_v2_fit_diagnostic.lock.json"
SOURCES = (
    CONFIG,
    "reports/phase15_offline_v2_fit_diagnostic_protocol.md",
    "scripts/freeze_phase15_fit_diagnostic_lock.py",
    "scripts/run_phase15_fit_diagnostic.py",
    "src/consequence_lab/phase15_learning.py",
    "tests/test_phase15.py",
    "second-brain/phases/Phase_15_Closed_Loop_Failure_Diagnosis.md",
    "second-brain/16_Experiment_Log.md",
)


def main() -> None:
    lock_path = ROOT / LOCK
    if lock_path.exists():
        raise SystemExit("refusing to overwrite diagnostic lock")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise SystemExit("commit protocol, runner, tests, and vault before freezing lock")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    if config["schema"] != "phase15_v2_failed_cell_replay_v1":
        raise SystemExit("wrong protocol schema")
    allowed_scope = {"exact_parent_cell_only": True, "reuse_saved_phase14g_features_only": True,
                     "compute_heldout_metrics": False, "heldout_rows_for_model_selection": False,
                     "new_transitions": False, "new_episodes": False, "exact_planner_calls": False,
                     "closed_loop": False, "dagger": False, "world_model": False, "reserve_access": False}
    if config["scope"] != allowed_scope:
        raise SystemExit("scope differs from this single-cell authorization")
    if config["resource_caps"] != {"wall_seconds": 10800, "output_bytes": 20971520}:
        raise SystemExit("resource caps differ from explicit authorization")
    if config["cell"] != {
        "rung": 32, "representation": "R0", "objective": "regression",
        "row_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 28, 29, 30, 31, 32,
                        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 58],
        "input_sha256": "B38F4909252F6F2036C4B15A040974F8EE40E42ABE53094270BDB1560CCB6028",
        "target_sha256": "9628CC2356C52FB4EE3554716C99DB4085B8200093383ADACCBDB56E37294045",
    }:
        raise SystemExit("cell differs from the only authorized Phase15 v2 failure")
    expected_training = {"seed": 15015, "hidden": [32, 32], "max_epochs": 1200, "patience": 100,
                         "learning_rate": 0.002, "l2": 0.00001,
                         "layout_weighting": "equal_total_weight_per_layout",
                         "model_selection": "minimum_regret_then_validation_loss"}
    if config["training"] != expected_training:
        raise SystemExit("training settings differ from the failed v2 cell")
    v2_lock = ROOT / config["parent"]["lock"]
    parent_result = ROOT / config["parent"]["result"]
    feature = ROOT / config["parent"]["feature_dataset"]
    if (sha256_file(v2_lock) != "5CFDCDD2024DFC4C7A6105F47B2BFE8C96EE3845EE911F14FF13A939902D7729"
            or sha256_file(parent_result) != "3AC34BAF90B5060CA4741C504F5824107B53E1814267A5D10078966D21728015"
            or sha256_file(feature) != "26D532A918E5AE913EE7F5A5E58BC60E68ED71D6CE38889CCDB7684363191386"):
        raise SystemExit("Phase15 v2 parent artifact changed")
    missing = [path for path in SOURCES if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"missing locked sources: {missing}")
    lock = {
        "schema": "phase15_v2_failed_cell_replay_lock_v1",
        "status": "authorized_single_diagnostic_replay",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_path": CONFIG,
        "config_sha256": sha256_file(ROOT / CONFIG),
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
        "parent_sha256": {
            key: sha256_file(ROOT / config["parent"][key])
            for key in ("config", "lock", "result", "result_manifest", "feature_manifest", "feature_dataset")
        },
        "feature_dataset_sha256": sha256_file(feature),
        "scope": config["scope"],
        "reserve_seed_block": {"start": 130000, "count": 200, "access": False},
        "authorization_reference": "User approved only deterministic replay of the failed 32-state R0/regression cell, with 3-hour and 20-MiB user caps previously cited; no training changes or other data split.",
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": LOCK, "source_commit": lock["source_commit"], "scope": lock["scope"]}, indent=2))


if __name__ == "__main__":
    main()
