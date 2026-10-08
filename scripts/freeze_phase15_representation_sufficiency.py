"""Freeze D152's train/validation-only consequence-predictability audit."""

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
CONFIG = "configs/phase15_representation_sufficiency.yaml"
LOCK = "experiments/phase15_representation_sufficiency.lock.json"
SOURCES = (
    CONFIG, "reports/phase15_representation_sufficiency_protocol.md",
    "scripts/freeze_phase15_representation_sufficiency.py",
    "scripts/run_phase15_representation_sufficiency.py",
    "src/consequence_lab/phase15_representation_sufficiency.py",
    "src/consequence_lab/phase15_outcome_summary.py",
    "src/consequence_lab/phase15_learning.py",
    "src/consequence_lab/phase15_representation_screen.py",
    "src/consequence_lab/phase15_data.py",
    "src/consequence_lab/phase15_generalization.py",
    "tests/test_phase15_representation_sufficiency.py",
    "second-brain/phases/Phase_15_Closed_Loop_Failure_Diagnosis.md",
    "second-brain/16_Experiment_Log.md",
)


def main() -> None:
    if (ROOT / LOCK).exists():
        raise SystemExit("refusing to overwrite D152 lock")
    status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).splitlines()
    if status != ['?? "new Prompt.md"']:
        raise SystemExit(f"D152 lock requires committed source; unexpected Git status: {status}")
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    if config.get("schema") != "phase15_representation_sufficiency_v1":
        raise SystemExit("D152 schema mismatch")
    expected_scope = {"train_rows_used_for_predictor_fit": True,
                      "validation_rows_used_for_exploratory_metrics": True,
                      "validation_used_for_checkpoint_selection": False,
                      "test_rows_used_or_indexed": False,
                      "consumed_test_metrics_recomputed": False, "reserve_access": False,
                      "simulator_or_planner_calls": False, "new_transitions_or_episodes": False,
                      "closed_loop": False, "dagger": False, "learned_world_model": False,
                      "jepa": False, "d151_result_modified": False}
    if config.get("scope") != expected_scope:
        raise SystemExit("D152 scope mismatch")
    parents = config["parents"]
    parent_hashes = {key: sha256_file(ROOT / path) for key, path in parents.items()}
    prompt_hash = sha256_file(ROOT / config["prompt_source"])
    d151 = json.loads((ROOT / parents["d151_result"]).read_text(encoding="utf-8"))
    d151_manifest = json.loads((ROOT / parents["d151_manifest"]).read_text(encoding="utf-8"))
    d150 = json.loads((ROOT / parents["d150_result"]).read_text(encoding="utf-8"))
    d150_manifest = json.loads((ROOT / parents["d150_manifest"]).read_text(encoding="utf-8"))
    feature_manifest = json.loads((ROOT / parents["feature_manifest"]).read_text(encoding="utf-8"))
    trace_manifest = json.loads((ROOT / parents["phase14g_manifest"]).read_text(encoding="utf-8"))
    if (d151.get("status") != "completed_exploratory_screen"
            or d151_manifest.get("result_sha256") != parent_hashes["d151_result"]
            or d151_manifest.get("report_sha256") != parent_hashes["d151_report"]
            or d151.get("screen", {}).get("promising_for_fresh_confirmation_only") is not False
            or d151.get("scope", {}).get("test_rows_indexed_or_scored") != 0
            or d151.get("scope", {}).get("reserve_access") is not False):
        raise SystemExit("D151 parent is inconsistent with the preserved closed result")
    if (d150.get("status") != "completed_exploratory_screen"
            or d150_manifest.get("result_sha256") != parent_hashes["d150_result"]
            or d150_manifest.get("report_sha256") != parent_hashes["d150_report"]
            or d150.get("models", {}).get("R4", {}).get("input_dimension") != 124
            or d150_manifest.get("checkpoint_sha256", {}).get("R4") is None):
        raise SystemExit("D150 R4 oracle parent is inconsistent")
    if (feature_manifest.get("feature_dataset_sha256") != parent_hashes["feature_dataset"]
            or trace_manifest.get("student_traces_sha256") is None
            or trace_manifest.get("student_decision_trace_count") != 922
            or trace_manifest.get("reserve_access") is not False):
        raise SystemExit("Phase15/Phase14G parent provenance mismatch")
    for model_name in ("R0", "R4"):
        checkpoint = ROOT / parents[f"d150_{model_name.lower()}_checkpoint"]
        expected_hash = d150_manifest["checkpoint_sha256"][model_name]
        if sha256_file(checkpoint) != expected_hash:
            raise SystemExit(f"D150 {model_name} checkpoint hash mismatch")
    with np.load(ROOT / parents["feature_dataset"], allow_pickle=False) as data:
        layouts = data["layouts"].astype(str).tolist()
    split = layout_split(layouts)
    split_lock = {}
    for name in ("train", "validation", "test"):
        names = sorted(split[name])
        count = sum(value in set(names) for value in layouts)
        digest = layout_list_sha256(names)
        expected = (config["split"][f"{name}_layout_count"], config["split"][f"{name}_row_count"],
                    config["split"][f"{name}_layouts_sha256"])
        if (len(names), count, digest) != expected:
            raise SystemExit(f"D152 {name} layout split mismatch")
        split_lock[name] = {"layouts": names, "row_count": count, "layout_list_sha256": digest}
    lock = {
        "schema": "phase15_representation_sufficiency_lock_v1",
        "status": "authorized_exploratory_train_validation_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCES},
        "config_sha256": sha256_file(ROOT / CONFIG),
        "prompt_sha256": prompt_hash,
        "parent_sha256": parent_hashes,
        "frozen_d150_head_sha256": {name: sha256_file(ROOT / parents[f"d150_{name.lower()}_checkpoint"])
                                     for name in ("R0", "R4")},
        "split": split_lock,
        "scope": expected_scope,
        "aliasing": config["aliasing"],
        "predictors": config["predictors"],
        "target_contract": config["target_contract"],
        "decision_head": config["decision_head"],
        "validation_reuse_disclosure": "D148/D150/D151 validation was reused; all D152 validation analyses are exploratory and not independent confirmation.",
        "test_boundary_disclosure": "The D148 test metrics are consumed. The D152 loader returns only train and validation row indices. The compressed NPZ may materialize full members internally; test values are never indexed, scored, fitted, or selected.",
        "prompt_status_disclosure": "new Prompt.md is retained as a user-provided untracked input and pinned by SHA-256; it is not modified by D152.",
    }
    (ROOT / LOCK).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": LOCK, "source_commit": lock["source_commit"],
                      "train_rows": split_lock["train"]["row_count"],
                      "validation_rows": split_lock["validation"]["row_count"],
                      "test_rows_indexed": 0, "prompt_sha256": prompt_hash,
                      "scope": expected_scope}, indent=2))


if __name__ == "__main__":
    main()
