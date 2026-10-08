"""Freeze Phase14D model-selection-only source/split authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from consequence_lab.phase14d import CONFIG_PATH, create_selection_lock


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    output = root / config["outputs"]["selection_lock"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    lock = create_selection_lock(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": lock["status"], "output": str(output), "authorized_block": lock["authorized_block"]}, sort_keys=True))


if __name__ == "__main__":
    main()
