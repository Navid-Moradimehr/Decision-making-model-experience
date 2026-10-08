"""Run Phase-12 pre-data parity and layout-split gates; never reads reserve."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase12_data import audit_phase12_layout_splits
from consequence_lab.phase12_teacher import audit_query_parity


def run() -> dict:
    config = yaml.safe_load((ROOT / "configs" / "phase12_teacher_student.yaml").read_text(encoding="utf-8"))
    layout = audit_phase12_layout_splits(config)
    parity_rows = []
    for offset in range(200):
        scenario = generate_staggered_corridor(200000 + offset)
        rng = random.Random(912_000 + offset)
        state = scenario.initial_state
        for _ in range(rng.randrange(0, 33)):
            if state.status != "active":
                break
            state = corridor_step(state, rng.randrange(9), scenario.spec).state
        if state.status != "active":
            state = scenario.initial_state
        query_seed = rng.randrange(2**32)
        parity_rows.append(audit_query_parity(scenario, state, seed=query_seed))
    checks = {
        "exact_proposal_selected_sequence": all(row["selected_sequence_match"] for row in parity_rows),
        "exact_selected_objective": all(row["selected_objective_match"] for row in parity_rows),
        "candidate_success_availability": all(row["availability_match"] for row in parity_rows),
        "candidate_count": all(row["candidate_count_match"] for row in parity_rows),
        "200_queries": len(parity_rows) == 200,
        "layout_split_disjointness": layout["pairwise_disjoint"],
        "reserve_unaccessed": not layout["reserve_generated_or_accessed"],
    }
    return {
        "schema": "phase12_predata_gate_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "parity_query_count": len(parity_rows),
        "parity_mismatches": {
            key: [index for index, row in enumerate(parity_rows) if not row[key]]
            for key in ("selected_sequence_match", "selected_objective_match", "availability_match", "candidate_count_match")
        },
        "layout_split_hashes": layout["sha256"],
        "opened_layout_counts": {name: block["unique_layouts"] for name, block in layout["blocks"].items()},
        "reserve_accessed": False,
        "training_performed": False,
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)
