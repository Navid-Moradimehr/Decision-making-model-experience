"""Freeze the training-only D159 fit-audit protocol."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d159 import D159Error, build_lock  # noqa: E402


def main() -> None:
    try:
        lock = build_lock()
    except (D159Error, ValueError) as exc:
        raise SystemExit(f"D159 lock preflight stopped: {exc}") from exc
    path = ROOT / lock["outputs"]["lock"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"D159 lock already exists: {path}")
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"D159 lock created: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
