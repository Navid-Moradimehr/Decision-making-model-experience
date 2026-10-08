"""Exact replay audit of Phase14's permitted Phase12 training-only source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase14 import build_source_audit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/runs/phase14_training_source_audit_v2.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing audit: {output}")
    result = build_source_audit(args.root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "layouts": result["training_layout_count"],
        "states": result["training_decision_state_count"],
        "diagnostic_states": len(result["diagnostic_state_keys"]),
        "one_layout_seed": result["one_layout_seed"],
        "audit": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
