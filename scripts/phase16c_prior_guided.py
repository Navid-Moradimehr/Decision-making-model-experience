from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase16c_prior_guided import layout_hash_audit, run_locked  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "run"))
    args = parser.parse_args()
    if args.stage == "audit":
        config = yaml.safe_load((ROOT / "configs/phase16c_prior_guided_planner_v1.yaml").read_text())
        audit = layout_hash_audit(int(config["environment"]["seed_start"]),
                                  int(config["environment"]["layout_count"]), ROOT)
        path = ROOT / config["outputs"]["layout_audit"]
        if path.exists():
            raise SystemExit("refusing to overwrite existing Phase 16C layout audit")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({k: audit[k] for k in (
            "seed_start", "layout_count", "unique_hashes", "historical_source_count",
            "overlap_historical", "overlap_outcome_bearing", "outcomes_or_transitions_computed")},
            indent=2))
    else:
        result = run_locked(ROOT)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
