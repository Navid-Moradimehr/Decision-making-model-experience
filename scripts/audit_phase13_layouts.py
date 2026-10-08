"""Hash-only split audit for Phase 13; never evaluates simulator transitions."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import json
from pathlib import Path

from consequence_lab.phase13 import PHASE13_AUDIT_NAME, build_layout_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--confirm-hash-only", action="store_true")
    args = parser.parse_args()
    if not args.confirm_hash_only:
        parser.error("pass --confirm-hash-only; this command generates hashes only and runs no task transitions")
    root = args.root.resolve()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if dirty:
        raise SystemExit("commit all audited source/config files before generating layout hashes")
    output = root / "artifacts" / "runs" / PHASE13_AUDIT_NAME
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    result = build_layout_audit(root)
    result["generation_git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    result["generation_source_sha256"] = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest().upper()
        for relative in (
            "configs/phase13.yaml",
            "scripts/audit_phase13_layouts.py",
            "src/consequence_lab/phase13.py",
            "src/consequence_lab/corridor_tasks.py",
            "src/consequence_lab/inertial_corridor.py",
        )
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"status={result['status']} prior_hashes={result['previous_layout_hash_count']} "
        f"spent_hashes={result['previous_outcome_layout_hash_count']} "
        f"new_blocks={len(result['blocks'])} output={output.relative_to(root)}"
    )
    print("task_transitions_evaluated=false episode_outcomes_read=false reserve_performance_outcomes_read=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
