"""Create the Phase14B full-training-label-only source lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from consequence_lab.phase14b import CONFIG_PATH, create_training_lock


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    lock = create_training_lock(root)
    lock_path = root / config["outputs"]["lock"]
    print(json.dumps({
        "lock": str(lock_path),
        "source_commit": lock["source_commit"],
        "authorized_stage": lock["authorized_stage"],
        "states": lock["training_states"],
        "actions_per_state": lock["actions_per_state"],
        "projected_wall_seconds": lock["full_label_wall_projection_seconds"],
        "training_authorized": lock["student_training"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
