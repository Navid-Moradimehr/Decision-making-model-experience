"""Execute the frozen Phase-12 student/geometry/exact-planner development block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.phase12_evaluation import load_student, run_phase12_development

LOCK = ROOT / "experiments" / "phase12_development.lock.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output", default="artifacts/runs/phase12_development_v1")
    args = parser.parse_args()
    if args.confirmation != "EVAL_PHASE12_DEVELOPMENT":
        raise SystemExit("development evaluation requires --confirmation EVAL_PHASE12_DEVELOPMENT")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["reserve_accessed"] or lock["training_performed"]:
        raise SystemExit("invalid development lock flags")
    checks = {
        "configs/phase12_teacher_student.yaml": lock["config_sha256"],
        "experiments/phase12_training.lock.json": lock["training_lock_sha256"],
        "artifacts/runs/phase12_training_v1/results.json": lock["training_results_sha256"],
        "reports/phase12_teacher_data_audit.json": lock["teacher_data_audit_sha256"],
        "artifacts/runs/phase12_layout_audit.json": lock["layout_manifest_sha256"],
    }
    for relpath, expected in checks.items():
        if sha(ROOT / relpath) != expected:
            raise SystemExit(f"frozen development input hash mismatch: {relpath}")
    for relpath, expected in lock["evaluation_source_sha256"].items():
        if sha(ROOT / relpath) != expected:
            raise SystemExit(f"frozen development code hash mismatch: {relpath}")
    checkpoint = Path(lock["selected_checkpoint"])
    if sha(checkpoint) != lock["selected_checkpoint_sha256"]:
        raise SystemExit("selected checkpoint hash mismatch")
    out = (ROOT / args.output).resolve()
    artifacts_root = (ROOT / "artifacts" / "runs").resolve()
    if artifacts_root not in out.parents or out.exists():
        raise SystemExit("refusing unsafe or existing development output path")
    config = yaml.safe_load((ROOT / "configs" / "phase12_teacher_student.yaml").read_text(encoding="utf-8"))
    seed_block = config["seed_splits"]["closed_loop_development"]
    if seed_block != lock["development_seeds"]:
        raise SystemExit("development seed block changed after lock")
    layout = json.loads((ROOT / "artifacts" / "runs" / "phase12_layout_audit.json").read_text(encoding="utf-8"))
    layouts = layout["blocks"]["closed_loop_development"]
    if layouts["layout_sha256"] != lock["development_layout_sha256"]:
        raise SystemExit("development layout hashes changed after lock")
    scenarios = [generate_staggered_corridor(seed) for seed in layouts["seeds"]]
    if [scenario.layout_sha256 for scenario in scenarios] != layouts["layout_sha256"]:
        raise SystemExit("development scenario regeneration mismatch")
    model, payload = load_student(checkpoint, device="cpu")
    started = time.monotonic()
    results = run_phase12_development(scenarios, model, kind=payload["kind"])
    wall = time.monotonic() - started
    methods = {row["method"]: row for row in results["methods"]}
    gate_cfg = config["gates"]["final_phase12_amortization_pass"]
    student = methods["student"]
    exact = methods["random_shooting"]
    success_gain = results["student_vs_geometry_paired_success"]
    collision_gain = results["student_vs_geometry_paired_collision"]
    final_gate = (
        success_gain["paired_rate_difference"] >= gate_cfg["student_success_point_gain_vs_geometry_waypoint_at_least"]
        and success_gain["lower_95"] > gate_cfg["paired_success_gain_95_percent_lower_bound_above"]
        and collision_gain["upper_95"] <= gate_cfg["paired_collision_difference_95_percent_upper_bound_at_most"]
        and student["safe_success_rate"] >= exact["safe_success_rate"] - gate_cfg["student_success_rate_within_0.15_of_exact_shooting_on_same_development_layouts"]
        and student["median_wall_seconds_per_decision"] <= exact["median_wall_seconds_per_decision"] * gate_cfg["median_student_decision_latency_at_most_0.20_of_exact_shooting"]
    )
    results["final_phase12_amortization_gate_passed"] = bool(final_gate)
    results["student_vs_geometry_dagger_trigger_passed"] = results["dagger_trigger_passed"]
    results["phase12_stopping_recommendation"] = (
        "advance_to_dagger" if results["dagger_trigger_passed"] and not final_gate
        else "phase12_amortization_passed" if final_gate
        else "stop_before_dagger"
    )
    results["selected_model"] = {
        "variant": lock["selected_variant"], "fit_seed": lock["selected_fit_seed"],
        "checkpoint_sha256": lock["selected_checkpoint_sha256"], "kind": payload["kind"],
    }
    results["development_layout_hashes"] = layouts["layout_sha256"]
    results["development_wall_seconds"] = wall
    results["training_performed"] = False
    results["reserve_accessed"] = False
    out.mkdir(parents=True)
    result_path = out / "results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "phase12_development_manifest_v1",
        "development_lock_sha256": sha(LOCK),
        "selected_checkpoint_sha256": lock["selected_checkpoint_sha256"],
        "result_sha256": sha(result_path),
        "torch_version": torch.__version__,
        "device": "CPU",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "reserve_accessed": False,
        "training_performed": False,
        "wall_seconds": wall,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "completed_phase12_development",
        "student_method": student,
        "exact_shooting": exact,
        "geometry_waypoint": methods["geometry_waypoint"],
        "student_vs_geometry_paired_success": success_gain,
        "student_vs_geometry_paired_collision": collision_gain,
        "dagger_trigger_passed": results["dagger_trigger_passed"],
        "final_phase12_amortization_gate_passed": final_gate,
        "next": results["phase12_stopping_recommendation"],
        "result_path": str(result_path),
        "reserve_accessed": False,
        "training_performed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
