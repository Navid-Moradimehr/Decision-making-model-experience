"""Read-only Phase15A audit of the consumed Phase14G trace dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase15 import audit_phase14g_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="artifacts/runs/phase15a_audit_v1",
        help="new output directory; existing directories are never overwritten",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = audit_phase14g_inputs(root)
    output_dir = (root / args.output_dir).resolve()
    if root not in output_dir.parents:
        raise ValueError("audit output must be inside the repository")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "audit.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
