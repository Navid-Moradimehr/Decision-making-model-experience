"""Collect only the fresh Phase-12b model-selection teacher split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_data import collect_teacher_split

CONFIG = ROOT / "configs" / "phase12b_action_aligned.yaml"
LOCK = ROOT / "experiments" / "phase12b_model_selection_data.lock.json"
LAYOUT_AUDIT = ROOT / "artifacts" / "runs" / "phase12b_layout_audit.json"
OUTPUT = ROOT / "artifacts" / "runs" / "phase12b_model_selection_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def within(root: Path, path: Path) -> bool:
    return root == path or root in path.parents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.plan_only:
        print(json.dumps({
            "status": "plan_only_no_collection",
            "split": "model_selection",
            "seeds": config["seed_splits"]["model_selection"],
            "output": str(OUTPUT),
            "reserve_accessed": False,
        }, indent=2))
        return 0
    if args.confirmation != "COLLECT_PHASE12B_MODEL_SELECTION":
        raise SystemExit("collection requires --confirmation COLLECT_PHASE12B_MODEL_SELECTION")
    if not LOCK.is_file() or not LAYOUT_AUDIT.is_file():
        raise SystemExit("frozen Phase-12b source lock and layout audit are required")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("authorized_split") != "model_selection" or lock.get("offline_gate_authorized") is not False:
        raise SystemExit("execution lock must authorize model_selection only")
    if lock.get("reserve_accessed") is not False:
        raise SystemExit("invalid lock: reserve is not sealed")
    for relpath, expected in lock["source_sha256"].items():
        path = (ROOT / relpath).resolve()
        if not within(ROOT, path) or not path.is_file() or digest(path) != expected:
            raise SystemExit(f"frozen Phase-12b source mismatch: {relpath}")
    if digest(LAYOUT_AUDIT) != lock["layout_audit_sha256"]:
        raise SystemExit("Phase-12b layout audit changed after lock")
    audit = json.loads(LAYOUT_AUDIT.read_text(encoding="utf-8"))
    if audit.get("new_hashes_disjoint_from_all_historical_blocks") is not True:
        raise SystemExit("Phase-12b layout audit did not pass")
    block = config["seed_splits"]["model_selection"]
    seeds = list(range(int(block["start"]), int(block["start"]) + int(block["count"])))
    audited = audit["new_blocks"]["model_selection"]
    if seeds != audited["seeds"] or len(audited["layout_sha256"]) != len(seeds):
        raise SystemExit("model-selection seed/layout scope mismatch")
    if OUTPUT.exists():
        raise SystemExit(f"refusing to overwrite {OUTPUT}")
    start = time.monotonic()
    result = collect_teacher_split(
        seeds,
        OUTPUT,
        artifact_cap_bytes=int(config["resource_caps"]["teacher_collection_artifact_bytes"]),
        wall_cap_seconds=int(config["resource_caps"]["teacher_collection_wall_seconds"]),
    )
    actual_hashes = result["layout_sha256"]
    if actual_hashes != audited["layout_sha256"]:
        raise RuntimeError("collected layouts differ from the pre-data layout audit")
    manifest = {
        "schema": "phase12b_model_selection_collection_v1",
        "experiment": "phase12b_model_selection_v1",
        "config_sha256": digest(CONFIG),
        "execution_lock_sha256": digest(LOCK),
        "layout_audit_sha256": digest(LAYOUT_AUDIT),
        "authorized_split": "model_selection",
        "offline_gate_labels_collected": False,
        "closed_loop_development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
        **result,
        "runner_wall_seconds": time.monotonic() - start,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Report only provenance/size, not teacher performance outcomes.
    print(json.dumps({
        "status": "model_selection_teacher_data_collected",
        "layouts": result["episode_count"],
        "queries": result["decision_query_count"],
        "artifact_bytes": result["artifact_bytes"],
        "manifest_sha256": digest(OUTPUT / "manifest.json"),
        "output": str(OUTPUT),
        "offline_gate_labels_collected": False,
        "reserve_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
