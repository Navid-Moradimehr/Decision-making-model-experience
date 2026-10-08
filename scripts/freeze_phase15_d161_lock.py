"""Freeze the source and artifact lineage for D161."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d161 import D161Error, build_lock  # noqa: E402


def main() -> None:
    try:
        lock = build_lock()
    except (D161Error, ValueError) as exc:
        raise SystemExit(f"D161 lock preflight stopped: {exc}") from exc
    path = ROOT / lock["outputs"]["lock"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"D161 lock already exists: {path}")
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"D161 lock created: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
