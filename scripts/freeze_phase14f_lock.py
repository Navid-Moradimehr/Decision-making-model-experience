"""Freeze the separate source-bound Phase14F closed-loop development lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase14f import create_closed_loop_lock


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = create_closed_loop_lock(root)
    path = root / "experiments/phase14f_closed_loop_v1.lock.json"
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": lock["status"],
        "source_commit": lock["source_commit"],
        "authorized_block": lock["authorized_block"],
        "lock_path": str(path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
