"""Freeze one loss-only Phase15 pairwise memorization comparison."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

from consequence_lab.phase14 import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_v2_pairwise32.yaml"
LOCK = "experiments/phase15_v2_pairwise32.lock.json"
SOURCES = (
    CONFIG,
    "reports/phase15_v2_pairwise32_protocol.md",
    "reports/phase15_v2_fit_error_audit.md",
    "scripts/freeze_phase15_pairwise32_lock.py",
    "scripts/run_phase15_pairwise32.py",
    "scripts/audit_phase15_v2_fit_error.py",
    "src/consequence_lab/phase15_learning.py",
    "tests/test_phase15.py",
    "second-brain/phases/Phase_15_Closed_Loop_Failure_Diagnosis.md",
    "second-brain/16_Experiment_Log.md",
)


def main() -> None:
    path = ROOT / LOCK
    if path.exists():
        raise SystemExit("refusing to overwrite Phase15 pairwise lock")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit protocol, runner, and second-brain updates before locking")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    expected_scope = {"fit_only_frozen_32_training_rows": True, "heldout_metrics": False,
                      "heldout_model_selection": False, "new_transitions": False,
                      "new_episodes": False, "planner_calls": False, "closed_loop": False,
                      "dagger": False, "world_model": False, "reserve_access": False}
    if config.get("schema") != "phase15_v2_objective_pairwise32_v1" or config["scope"] != expected_scope:
        raise SystemExit("Phase15 pairwise protocol/scope mismatch")
    if config["resource_caps"] != {"wall_seconds": 10800, "output_bytes": 20971520}:
        raise SystemExit("resource caps mismatch")
    if config["training"] != {"objective": "pairwise", "seed": 15015, "hidden": [32, 32],
                               "max_epochs": 1200, "patience": 100, "learning_rate": 0.002,
                               "l2": 0.00001, "layout_weighting": "equal_total_weight_per_layout",
                               "model_selection": "minimum_layout_macro_regret_then_objective_validation_loss",
                               "exact_tied_pairs": "excluded"}:
        raise SystemExit("pairwise training configuration differs from protocol")
    expected_rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 28, 29, 30, 31, 32,
                     33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 58]
    if config["cell"] != {
        "row_indices": expected_rows, "representation": "R0", "feature_count": 61, "action_count": 9,
        "source_dtype": "float32", "fit_dtype": "float64",
        "source_x_sha256": "AECE48C1A055078D51A4252FCB187BD019C8FDEC980C1425CD712BEA039853A4",
        "fit_x_sha256": "B38F4909252F6F2036C4B15A040974F8EE40E42ABE53094270BDB1560CCB6028",
        "source_y_sha256": "9628CC2356C52FB4EE3554716C99DB4085B8200093383ADACCBDB56E37294045",
        "fit_y_sha256": "6A90D24FF22AD408181F957AD2E1587537A33E67D01E3EB419D200F7E49502BF",
    }:
        raise SystemExit("pairwise data cell differs from the audited 32-row set")
    if config["gate"] != {"best_action_set_agreement_minimum": 0.99,
                          "mean_normalized_regret_state_micro_maximum": 0.01}:
        raise SystemExit("original memorization gate was changed")
    parents = config["parents"]
    parent_hashes = {key: sha256_file(ROOT / relative) for key, relative in parents.items()}
    baseline = json.loads((ROOT / parents["regression_diagnostic"]).read_text(encoding="utf-8"))
    audit = json.loads((ROOT / parents["fit_error_audit"]).read_text(encoding="utf-8"))
    if (baseline.get("status") != "reproduced"
            or baseline["metrics"]["best_action_set_agreement"] != 0.96875
            or baseline["metrics"]["mean_normalized_regret_state_micro"] != 0.00390625
            or audit.get("status") != "descriptive_training_subset_audit"
            or audit["mismatch"]["feature_row_index"] != 39):
        raise SystemExit("parent regression/audit evidence does not match this conditional protocol")
    sources = {relative: sha256_file(ROOT / relative) for relative in SOURCES}
    lock = {
        "schema": "phase15_v2_objective_pairwise32_lock_v1",
        "status": "authorized_single_pairwise_memorization_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_path": CONFIG, "config_sha256": sha256_file(ROOT / CONFIG),
        "source_sha256": sources, "parent_sha256": parent_hashes,
        "scope": config["scope"], "resource_caps": config["resource_caps"],
        "frozen_row_indices": config["cell"]["row_indices"],
        "reserve_seed_block": {"start": 130000, "count": 200, "access": False},
        "authorization_reference": "User authorized the conditional objective-alignment protocol after the read-only Phase15 row-39 audit supported testing objective mismatch.",
    }
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": LOCK, "source_commit": lock["source_commit"], "scope": lock["scope"]}, indent=2))


if __name__ == "__main__":
    main()
