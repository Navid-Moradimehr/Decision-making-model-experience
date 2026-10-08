"""Freeze the selected Phase14D model and authorize only offline-gate outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from consequence_lab.phase14e import CONFIG_PATH, create_offline_gate_lock


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    output = root / config["outputs"]["lock"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    lock = create_offline_gate_lock(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": lock["status"],
        "output": str(output),
        "selected_model": lock["selected_model"],
        "authorized_seed_start": lock["authorized_block"]["seed_start"],
        "authorized_layout_count": lock["authorized_block"]["count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
