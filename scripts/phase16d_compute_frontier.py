from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase16d_compute_frontier import (  # noqa: E402
    CONFIG, execution_order, run_locked,
)
from consequence_lab.phase16c_prior_guided import layout_hash_audit  # noqa: E402


def audit() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    start = int(cfg["environment"]["proposed_seed_start"])
    count = int(cfg["environment"]["proposed_layout_count"])
    report = layout_hash_audit(start, count, ROOT)
    if (report["unique_hashes"] != count or report["overlap_historical"] != 0
            or report["overlap_outcome_bearing"] != 0
            or report["reserve_generated_or_accessed"] is not False
            or report["d168_seed_block_accessed"] is not False
            or report["outcomes_or_transitions_computed"] is not False):
        raise RuntimeError("Phase16D hash-only freshness audit failed")
    run_dir = ROOT / cfg["outputs"]["proposed_run_dir"]
    if run_dir.exists():
        raise RuntimeError("refusing to overwrite a Phase16D audit/output directory")
    run_dir.mkdir(parents=True)
    audit_path = ROOT / cfg["outputs"]["layout_audit"]
    audit_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    schedule = execution_order(report["layouts"], int(cfg["execution"]["counterbalance_seed"]))
    schedule_path = ROOT / cfg["outputs"]["execution_order"]
    schedule_path.write_text(json.dumps(schedule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "seed_start": start, "layout_count": count, "unique_hashes": report["unique_hashes"],
        "historical_source_count": report["historical_source_count"],
        "overlap_historical": report["overlap_historical"],
        "overlap_outcome_bearing": report["overlap_outcome_bearing"],
        "outcomes_or_transitions_computed": report["outcomes_or_transitions_computed"],
        "execution_arms": len(schedule),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "run"))
    args = parser.parse_args()
    result = audit() if args.stage == "audit" else run_locked(ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
