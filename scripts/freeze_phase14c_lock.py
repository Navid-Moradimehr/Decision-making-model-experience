from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase14c import create_training_lock


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Phase14C train-only memorization lock")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    lock = create_training_lock(args.root)
    print(json.dumps(lock, sort_keys=True))


if __name__ == "__main__":
    main()
