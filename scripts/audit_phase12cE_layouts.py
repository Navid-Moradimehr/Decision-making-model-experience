"""Hash-only layout audit for the approved Phase 12c-E selection/gate blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.phase12c_c import audit_phase12c_c_layouts
from freeze_phase12cE_lock import CONFIG, LOCK, digest


def collect_layout_hashes(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "layout_sha256":
                if isinstance(item, str) and re.fullmatch(r"[0-9A-Fa-f]{64}", item):
                    found.add(item)
                elif isinstance(item, list):
                    found.update(x for x in item if isinstance(x, str) and re.fullmatch(r"[0-9A-Fa-f]{64}", x))
            else:
                found.update(collect_layout_hashes(item))
    elif isinstance(value, list):
        for item in value:
            found.update(collect_layout_hashes(item))
    return found


def verify_lock(config: dict) -> tuple[dict, Path]:
    lock_path = ROOT / LOCK.relative_to(ROOT)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase12cE_execution_lock_v2" or lock.get("config_sha256") != digest(CONFIG):
        raise SystemExit("Phase 12c-E lock/config mismatch")
    for relative, expected in lock["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise SystemExit(f"Phase 12c-E source changed since lock: {relative}")
    training_dir = ROOT / config["training_source"]["archived_dataset"]
    for relative, expected in lock["training_data_sha256"].items():
        if digest(training_dir / relative) != expected:
            raise SystemExit(f"Phase 12c-E archived fitting file changed: {relative}")
    for relative, expected in lock["replay_source_data_sha256"].items():
        if digest(ROOT / "artifacts/runs" / relative) != expected:
            raise SystemExit(f"Phase 12 replay source changed since Phase 12c-E lock: {relative}")
    if lock.get("cpu_runtime_cap_seconds") is not None or lock.get("reserve_accessed") is not False:
        raise SystemExit("Phase 12c-E resource/split lock is invalid")
    if lock.get("selection_outcomes_accessed") is not False or lock.get("offline_gate_outcomes_accessed") is not False:
        raise SystemExit("Phase 12c-E execution lock has prior outcomes")
    if config["resources"].get("fixed_runtime_cap_seconds") is not None:
        raise SystemExit("approved CPU runtime contract changed")
    return lock, lock_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not args.audit:
        print(json.dumps({"status": "plan_only_no_layout_generation", "blocks": {
            key: config["seed_splits"][key] for key in ("model_selection", "offline_gate")},
            "reserve_accessed": False}, indent=2))
        return 0
    if args.confirmation != "AUDIT_PHASE12CE_LAYOUT_HASHES":
        raise SystemExit("hash audit requires --confirmation AUDIT_PHASE12CE_LAYOUT_HASHES")
    lock, lock_path = verify_lock(config)
    prior_audit_path = ROOT / "artifacts/runs/phase12cC_layout_audit_v4.json"
    prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
    if prior_audit.get("new_hashes_disjoint_from_all_phase11_phase12_phase12b_phase12cA_blocks") is not True:
        raise SystemExit("the historical Phase 12c-C layout audit did not attest disjointness")
    broad = audit_phase12c_c_layouts(config, json.loads(
        (ROOT / "artifacts/runs/phase12cA_layout_audit_v2_160000_160099.json").read_text(encoding="utf-8")
    ))
    already_used = collect_layout_hashes(prior_audit)
    if set(broad["new_blocks"]["model_selection"]["layout_sha256"]) & already_used:
        raise SystemExit("Phase 12c-E layouts collide with previously audited Phase 12c-C layouts")
    if set(broad["new_blocks"]["offline_gate"]["layout_sha256"]) & already_used:
        raise SystemExit("Phase 12c-E gate layouts collide with previously audited Phase 12c-C layouts")

    d2_spec = config["seed_splits"]["conditional_D2_not_repurposed"]
    d2_seeds = list(range(int(d2_spec["start"]), int(d2_spec["start"]) + int(d2_spec["count"])))
    d2_hashes = [generate_staggered_corridor(seed).layout_sha256 for seed in d2_seeds]
    if len(set(d2_hashes)) != len(d2_hashes):
        raise SystemExit("duplicate layouts inside unopened conditional D2 seed block")
    e_hashes = set(broad["new_blocks"]["model_selection"]["layout_sha256"] +
                   broad["new_blocks"]["offline_gate"]["layout_sha256"])
    if e_hashes.intersection(d2_hashes):
        raise SystemExit("Phase 12c-E layout hash overlaps unopened conditional D2 block")
    reserve = config["seed_splits"]["final_reserve_sealed"]
    reserve_range = range(int(reserve["start"]), int(reserve["start"]) + int(reserve["count"]))
    e_seeds = (broad["new_blocks"]["model_selection"]["seeds"] +
               broad["new_blocks"]["offline_gate"]["seeds"])
    if set(e_seeds).intersection(reserve_range):
        raise SystemExit("Phase 12c-E seed numbers overlap sealed reserve; reserve layouts are not generated")

    output = (ROOT / (args.output or config["seed_splits"]["audit_output"])).resolve()
    if not output.is_relative_to((ROOT / "artifacts/runs").resolve()) or output.exists():
        raise SystemExit("layout audit output must be new and remain under artifacts/runs")
    result = {
        "schema": "phase12cE_layout_audit_v1",
        "config_sha256": digest(CONFIG),
        "execution_lock_sha256": digest(lock_path),
        "historical_phase12cC_audit_sha256": digest(prior_audit_path),
        "phase12cA_audit_sha256": digest(ROOT / "artifacts/runs/phase12cA_layout_audit_v2_160000_160099.json"),
        "blocks": broad["new_blocks"],
        "unopened_conditional_D2_hashes_checked": len(d2_hashes),
        "unopened_conditional_D2_overlap": False,
        "reserve_layouts_generated_or_accessed": False,
        "development_or_dagger_outcomes_accessed": False,
        "layout_generation_only_no_rollouts_or_labels": True,
        "pairwise_disjoint": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "phase12cE_hash_audit_passed", "output": str(output),
                      "selection_layouts": len(result["blocks"]["model_selection"]["seeds"]),
                      "gate_layouts": len(result["blocks"]["offline_gate"]["seeds"]),
                      "D2_layout_hashes_checked": len(d2_hashes),
                      "reserve_generated_or_accessed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
