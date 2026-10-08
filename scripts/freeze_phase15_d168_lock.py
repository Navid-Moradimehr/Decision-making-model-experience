from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

from consequence_lab.phase14 import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_d168_latent_intervention.yaml"
PROTOCOL = "reports/phase15_d168_protocol.md"
OUTPUT = "experiments/phase15_d168_latent_intervention_v3.lock.json"
SOURCES = (
    CONFIG, PROTOCOL, "reports/phase15_d168_prelock_abort_v1.md",
    "reports/phase15_d168_g1_runtime_abort_v1.md", "reports/phase15_d168_runtime_amendment_v1.md",
    "src/consequence_lab/phase15_d168.py",
    "src/consequence_lab/phase15_d153_scaling.py", "src/consequence_lab/phase15_d164.py",
    "src/consequence_lab/phase15_d157.py", "src/consequence_lab/phase15_decision_aligned.py",
    "src/consequence_lab/phase14.py", "src/consequence_lab/phase14g.py",
    "src/consequence_lab/phase12_teacher.py", "src/consequence_lab/inertial_corridor.py",
    "src/consequence_lab/corridor_tasks.py", "src/consequence_lab/phase14c.py",
    "tests/test_phase15_d168.py", "tests/test_config.py",
    "scripts/freeze_phase15_d168_lock.py", "scripts/run_phase15_d168.py", "uv.lock",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build_lock() -> dict:
    if _git("status", "--porcelain"):
        raise RuntimeError("commit D168 source/config/tests before lock freeze")
    cfg = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    if cfg.get("schema") != "phase15_d168_latent_intervention_v1":
        raise RuntimeError("D168 config schema mismatch")
    if cfg.get("status") != "locked_g1_runtime_amendment_v1":
        raise RuntimeError("D168 design status changed after lock")
    d153_lock_path = ROOT / cfg["parents"]["d153_lock"]
    d153_manifest_path = ROOT / cfg["parents"]["d153_manifest"]
    d153_lock = json.loads(d153_lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(d153_manifest_path.read_text(encoding="utf-8"))
    rows_path = ROOT / cfg["parents"]["d153_rows"]
    rows_sha = sha256_file(rows_path)
    if (d153_lock.get("schema") != "phase15_d153_coverage_scaling_lock_v3"
            or d153_lock.get("reused_corpus", {}).get("sha256") != rows_sha
            or d153_lock.get("reused_corpus", {}).get("expected_state_rows") != 1365
            or manifest.get("status") != "completed_exploratory_scaling"
            or manifest.get("fresh_dataset_sha256") != rows_sha
            or manifest.get("reserve_access") is not False
            or manifest.get("test_rows_indexed_or_scored") != 0):
        raise RuntimeError("D153 training-only parent integrity/scope audit failed")
    missing = [name for name in SOURCES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"D168 lock source files missing: {missing}")
    g0_dir = ROOT / cfg["outputs"]["run_dir"]
    g0_result_path, g0_data_path = g0_dir / "g0_result.json", g0_dir / "training_trajectories.npz"
    if not g0_result_path.is_file() or not g0_data_path.is_file():
        raise RuntimeError("amended G1 lock requires a completed G0 artifact")
    g0_result = json.loads(g0_result_path.read_text(encoding="utf-8"))
    if (g0_result.get("status") != "G0_passed_training_replay_parity"
            or g0_result.get("training_data_sha256") != sha256_file(g0_data_path)
            or g0_result.get("integrity", {}).get("source_sha256") != rows_sha
            or g0_result.get("integrity", {}).get("max_state_feature_abs_error", float("inf"))
            > float(cfg["data"]["replay_state_feature_abs_tolerance"])
            or g0_result.get("integrity", {}).get("max_summary_abs_error", float("inf"))
            > float(cfg["data"]["branch_summary_max_abs_tolerance"])):
        raise RuntimeError("D168 G0 result/data do not satisfy amended lock prerequisites")
    parent_hashes = {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCES}
    return {
        "schema": "phase15_d168_latent_intervention_lock_v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "locked_training_feasibility_and_conditional_fresh_development",
        "source_commit": _git("rev-parse", "HEAD"),
        "config_sha256": sha256_file(ROOT / CONFIG),
        "protocol_sha256": sha256_file(ROOT / PROTOCOL),
        "source_sha256": source_hashes,
        "parent_sha256": parent_hashes,
        "d153_training": {"path": cfg["parents"]["d153_rows"], "sha256": rows_sha,
                          "rows": 1365, "layouts": 140, "reserve_access": False},
        "g0": {"result_sha256": sha256_file(g0_result_path),
               "dataset_sha256": sha256_file(g0_data_path),
               "max_state_feature_abs_error": g0_result["integrity"]["max_state_feature_abs_error"],
               "max_summary_abs_error": g0_result["integrity"]["max_summary_abs_error"],
               "planner_queries": g0_result["integrity"]["planner_queries"]},
        "split": cfg["data"]["grouped_split"],
        "scope": {"training_layout_replay_only": True, "fresh_layout_outcomes_before_G1": False,
                  "reserve_access": False, "test_access": False,
                  "closed_loop_student_control": False, "dagger": False},
        "fresh_development_block": cfg["data"]["fresh_development_block"],
        "overwrite_forbidden": True,
        "runtime": cfg["runtime"],
    }


def main() -> None:
    output = ROOT / OUTPUT
    if output.exists():
        raise RuntimeError("refusing to overwrite D168 lock")
    lock = build_lock()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": OUTPUT, "source_commit": lock["source_commit"],
                      "d153_training_sha256": lock["d153_training"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
