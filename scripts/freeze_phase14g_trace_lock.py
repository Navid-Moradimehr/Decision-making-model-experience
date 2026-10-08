"""Freeze the source/data-bound one-shot Phase14G trace lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14g_runner import build_run_lock


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = build_run_lock(root)
    path = root / lock["outputs"]["lock"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing lock: {path}")
    path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"lock": str(path), "sha256": sha256_file(path), "status": lock["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
