"""Freeze the Phase15 offline-only source/data/config lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from consequence_lab.phase15 import audit_phase14g_inputs


DEFAULT_CONFIG = "configs/phase15_offline_v1.yaml"
DEFAULT_LOCK = "experiments/phase15_offline_v1.lock.json"
SOURCES = (
    "reports/phase15_failure_diagnosis_protocol_v1.md",
    "reports/phase15a_locked_dataset_audit.md",
    "reports/phase15_offline_v1_predata_failure.md",
    "src/consequence_lab/phase15.py",
    "src/consequence_lab/phase15_data.py",
    "src/consequence_lab/phase15_features.py",
    "src/consequence_lab/phase15_learning.py",
    "src/consequence_lab/phase14g.py",
    "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase13.py",
    "src/consequence_lab/phase14.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "scripts/audit_phase15_dataset.py",
    "scripts/freeze_phase15_lock.py",
    "scripts/run_phase15_offline.py",
    "tests/test_phase15.py",
    "second-brain/phases/Phase_15_Closed_Loop_Failure_Diagnosis.md",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, lock_relative = args.config, args.lock
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    lock_path = root / lock_relative
    if lock_path.exists():
        raise SystemExit("refusing to overwrite Phase15 lock")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if dirty:
        raise SystemExit("commit Phase15 protocol, implementation, and tests before freezing its lock")
    audit = audit_phase14g_inputs(root)
    if audit["reserve_accessed"] or audit["models_fit"] or audit["simulator_transitions_generated"]:
        raise SystemExit("Phase15A audit is not read-only/within scope")
    scope = config.get("scope", {})
    if (config.get("schema") not in {"phase15_offline_run_config_v1", "phase15_offline_run_config_v2"}
            or scope.get("saved_phase14g_data_only") is not True
            or scope.get("fit_offline_models") is not True
            or scope.get("heldout_diagnostic_layouts") is not True
            or any(scope.get(name) is not False for name in (
                "new_episodes", "rerun_phase14g", "exact_planner_calls", "closed_loop_evaluation",
                "dagger", "learned_world_model", "representation_distillation", "reserve_access",
            ))):
        raise SystemExit("Phase15 config identity or scope is invalid")
    if config.get("schema") == "phase15_offline_run_config_v1" and scope.get("build_features_from_saved_states") is not True:
        raise SystemExit("Phase15 v1 must build the locked saved-state feature data")
    if config.get("schema") == "phase15_offline_run_config_v2" and (
            scope.get("reuse_phase15_v1_features_only_if_hashes_match") is not True
            or scope.get("build_features_from_saved_states") is not False
            or scope.get("regenerate_exact_continuation_labels") is not False):
        raise SystemExit("Phase15 v2 must reuse the locked v1 feature artifact without rerunning transitions")
    if config.get("schema") == "phase15_offline_run_config_v2":
        failure_record = json.loads((root / config["parents"]["phase15_v1_failure_record"]).read_text(encoding="utf-8"))
        if (failure_record.get("status") != "aborted_before_completed_result"
                or failure_record.get("heldout_metrics_computed") is not False
                or failure_record.get("reserve_access") is not False
                or failure_record.get("lock_sha256") != sha(root / config["parents"]["phase15_v1_lock"])):
            raise SystemExit("Phase15 v2 must bind the preserved v1 abort record")
    amendment_sources = ("reports/phase15_offline_v2_amendment.md",) if config.get("schema") == "phase15_offline_run_config_v2" else ()
    source_files = (config_path, *SOURCES, *amendment_sources)
    missing = [relative for relative in source_files if not (root / relative).is_file()]
    if missing:
        raise SystemExit(f"missing Phase15 lock source files: {missing}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    lock = {
        "schema": "phase15_offline_run_lock_v1",
        "status": "authorized_phase15_offline_diagnosis_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": head,
        "config_path": config_path,
        "lock_path": lock_relative,
        "config_sha256": sha(root / config_path),
        "source_sha256": {relative: sha(root / relative) for relative in source_files},
        "phase14g_lock_sha256": sha(root / config["parents"]["phase14g_lock"]),
        "phase14g_manifest_sha256": sha(root / config["parents"]["phase14g_manifest"]),
        "phase14g_traces_sha256": sha(root / config["parents"]["phase14g_traces"]),
        "phase14g_episodes_sha256": sha(root / config["parents"]["phase14g_episodes"]),
        "phase14g_metrics_sha256": sha(root / config["parents"]["phase14g_metrics"]),
        "phase14f_student_sha256": sha(root / config["parents"]["phase14f_student"]),
        "phase14b_records_sha256": sha(root / config["parents"]["phase14b_records"]),
        "phase14g_layout_count_consumed": audit["layout_count"],
        "phase14g_trace_count": audit["student_trace_count"],
        "phase14g_branch_count": audit["forced_action_branch_count"],
        "phase15_v1_feature_dataset_sha256": sha(root / config["parents"]["phase15_v1_feature_dataset"]) if config.get("schema") == "phase15_offline_run_config_v2" else None,
        "phase15_v1_feature_manifest_sha256": sha(root / config["parents"]["phase15_v1_feature_manifest"]) if config.get("schema") == "phase15_offline_run_config_v2" else None,
        "phase15_v1_lock_sha256": sha(root / config["parents"]["phase15_v1_lock"]) if config.get("schema") == "phase15_offline_run_config_v2" else None,
        "phase15_v1_failure_record_sha256": sha(root / config["parents"]["phase15_v1_failure_record"]) if config.get("schema") == "phase15_offline_run_config_v2" else None,
        "reserve_seed_block": {"start": 130000, "count": 200, "access": False},
        "phase14g_consumed_seed_block": {"start": 193000, "count": 100, "reused_read_only": True},
        "scope": scope,
        "authorization_reference": "User authorized continuing the Phase15 offline diagnosis; live evaluation remains closed.",
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": lock_relative, "source_commit": head, "scope": scope}, indent=2))


if __name__ == "__main__":
    main()
