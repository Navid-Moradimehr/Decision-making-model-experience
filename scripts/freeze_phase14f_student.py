"""Export the fixed train-only Phase14D CART for deployment evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase14f import fit_and_freeze_student


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = fit_and_freeze_student(args.root.resolve())
    print(json.dumps({
        "status": manifest["status"],
        "model_sha256": manifest["model_sha256"],
        "model_nodes": manifest["model_nodes"],
        "model_bytes": manifest["model_bytes"],
        "offline_consistency_check": manifest["offline_consistency_check"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
