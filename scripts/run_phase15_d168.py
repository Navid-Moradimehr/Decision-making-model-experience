from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_d168 import (
    D168Error,
    fit_training_feasibility,
    replay_training_rows,
    save_reconstructed_training,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK = "experiments/phase15_d168_latent_intervention_v3.lock.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _validate_lock() -> tuple[dict, dict]:
    lock_path = ROOT / LOCK
    if not lock_path.is_file():
        raise D168Error("D168 immutable lock is missing")
    lock = _json(lock_path)
    cfg_path = ROOT / "configs/phase15_d168_latent_intervention.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    try:
        lock_parent = _git("rev-parse", "HEAD^")
    except subprocess.CalledProcessError as exc:
        raise D168Error("D168 source commit has no committed lock parent") from exc
    if (lock.get("schema") != "phase15_d168_latent_intervention_lock_v3"
            or lock.get("status") != "locked_training_feasibility_and_conditional_fresh_development"
            or lock.get("source_commit") != lock_parent
            or lock.get("config_sha256") != sha256_file(cfg_path)
            or lock.get("protocol_sha256") != sha256_file(ROOT / "reports/phase15_d168_protocol.md")):
        raise D168Error("D168 source/config/protocol do not match the committed lock")
    for relative, digest in lock.get("source_sha256", {}).items():
        if sha256_file(ROOT / relative) != digest:
            raise D168Error(f"locked source changed: {relative}")
    for relative, digest in lock.get("parent_sha256", {}).items():
        if sha256_file(ROOT / cfg["parents"][relative]) != digest:
            raise D168Error(f"locked parent changed: {relative}")
    if _git("status", "--porcelain"):
        raise D168Error("D168 runner requires a clean worktree matching the lock")
    return lock, cfg


def run_g0() -> None:
    lock, cfg = _validate_lock()
    out = ROOT / cfg["outputs"]["run_dir"]
    if out.exists():
        raise D168Error("refusing to overwrite D168 run directory")
    out.mkdir(parents=True)
    start = time.perf_counter()
    rows = ROOT / cfg["parents"]["d153_rows"]
    replay = replay_training_rows(rows, lock["d153_training"]["sha256"],
                                  float(cfg["data"]["replay_state_feature_abs_tolerance"]),
                                  float(cfg["data"]["branch_summary_max_abs_tolerance"]))
    data_path = out / "training_trajectories.npz"
    dataset_sha = save_reconstructed_training(replay, data_path)
    integrity = replay["integrity"]
    integrity["dataset_sha256"] = dataset_sha
    integrity["serialized_trajectory_state_count"] = int(len(replay["branch_states"]))
    integrity["planner_queries"] = int(replay["branch_continuation_query_counts"].sum())
    integrity["branch_environment_transitions"] = int(len(replay["branch_states"]))
    integrity["collector_replay_transitions"] = int(len(replay["x_R0"]))
    integrity["elapsed_seconds"] = time.perf_counter() - start
    result = {"schema": "phase15_d168_g0_result_v1", "status": "G0_passed_training_replay_parity",
              "lock_sha256": sha256_file(ROOT / LOCK), "integrity": integrity,
              "training_data_path": str(data_path.relative_to(ROOT)),
              "training_data_sha256": dataset_sha,
              "training_started": False, "fresh_layout_outcomes_read": False,
              "reserve_access": False, "test_access": False}
    path = out / "g0_result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def run_g1() -> None:
    lock, cfg = _validate_lock()
    g0_dir = ROOT / cfg["outputs"]["run_dir"]
    out = ROOT / cfg["outputs"]["g1_run_dir"]
    result_path, manifest_path = out / "g1_result.json", out / "g1_manifest.json"
    if result_path.exists() or manifest_path.exists():
        raise D168Error("refusing to overwrite D168 G1 result/manifest")
    g0_path = g0_dir / "g0_result.json"
    if not g0_path.is_file():
        raise D168Error("D168 G0 replay/parity result is required before fitting")
    g0 = _json(g0_path)
    data_path = g0_dir / "training_trajectories.npz"
    if (g0.get("status") != "G0_passed_training_replay_parity"
            or g0.get("training_data_sha256") != sha256_file(data_path)
            or sha256_file(g0_path) != lock.get("g0", {}).get("result_sha256")
            or sha256_file(data_path) != lock.get("g0", {}).get("dataset_sha256")
            or g0.get("integrity", {}).get("source_sha256") != lock.get("d153_training", {}).get("sha256")
            or g0.get("integrity", {}).get("max_state_feature_abs_error", float("inf"))
            > float(cfg["data"]["replay_state_feature_abs_tolerance"])
            or g0.get("integrity", {}).get("max_summary_abs_error", float("inf"))
            > float(cfg["data"]["branch_summary_max_abs_tolerance"])):
        raise D168Error("D168 G0 artifact does not match amended lock or data")
    if out.exists():
        raise D168Error("refusing to reuse G1 output directory from another attempt")
    out.mkdir(parents=True)
    start = time.perf_counter()
    result = fit_training_feasibility(data_path, cfg, out)
    result.update({"schema": "phase15_d168_g1_result_v1", "lock_sha256": sha256_file(ROOT / LOCK),
                   "g0_result_sha256": sha256_file(g0_path), "g0_dataset_sha256": sha256_file(data_path),
                   "elapsed_seconds": time.perf_counter() - start,
                   "fresh_layout_outcomes_read": False, "reserve_access": False, "test_access": False})
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoints = {p.name: sha256_file(p) for p in out.glob("*.pt")}
    manifest = {
        "schema": "phase15_d168_g1_manifest_v1", "status": result["status"],
        "source_commit": _git("rev-parse", "HEAD"), "lock_sha256": sha256_file(ROOT / LOCK),
        "config_sha256": sha256_file(ROOT / "configs/phase15_d168_latent_intervention.yaml"),
        "g0_dataset_sha256": sha256_file(data_path), "g1_result_sha256": sha256_file(result_path),
        "checkpoints": checkpoints, "training_rows": 1365, "training_layouts": 140,
        "internal_validation_fold": 0, "reserve_access": False, "test_rows_accessed": 0,
        "fresh_layout_outcomes_read": False, "simulator_or_planner_calls_during_g1": 0,
        "closed_loop": False, "dagger": False,
        "runtime_environment": {"python": sys.version, "platform": platform.platform(),
                                "torch": __import__("torch").__version__,
                                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                                "cuda_available": bool(__import__("torch").cuda.is_available()),
                                "cuda_device": (__import__("torch").cuda.get_device_name(0)
                                                if __import__("torch").cuda.is_available() else None)},
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(result_path.relative_to(ROOT)),
                      "elapsed_seconds": result["elapsed_seconds"], "checkpoints": checkpoints}, indent=2))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("g0", "g1"))
    args = parser.parse_args()
    if args.stage == "g0":
        run_g0()
    else:
        run_g1()


if __name__ == "__main__":
    main()
