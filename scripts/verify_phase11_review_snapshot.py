"""Verify the Phase 11 review snapshot without running any task episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from consequence_lab.protocol_review import verify_phase11_review_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project root (defaults to this checkout)",
    )
    args = parser.parse_args()
    result = verify_phase11_review_snapshot(args.root)
    print(
        f"snapshot_valid={result['valid']} files_verified={result['files_verified']} "
        "execution_authorized=false performance_data_authorized=false training_authorized=false"
    )
    for error in result["errors"]:
        print(f"ERROR: {error}")
    if result["valid"]:
        print("Integrity only; this review snapshot grants no execution permission.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
