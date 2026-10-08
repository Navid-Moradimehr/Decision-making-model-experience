"""Hash-only Phase14G layout audit; it evaluates no simulator transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase14g import CONFIG_PATH, build_layout_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = build_layout_audit(root)
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    output = root / config["outputs"]["layout_audit"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite Phase14G audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "seed_start": audit["candidate_block"]["seed_start"],
        "count": audit["candidate_block"]["count"],
        "prior_outcome_layout_hash_count": audit["prior_outcome_layout_hash_count"],
        "transitions": audit["task_transitions_evaluated"],
        "run_authorized": audit["run_authorized"],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
