"""Freeze the D154 offline study after committed protocol/source/tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_decision_aligned import D154Error, LOCK, ROOT as PROJECT_ROOT, build_lock  # noqa: E402


def main() -> None:
    try:
        lock = build_lock()
    except (D154Error, ValueError) as exc:
        raise SystemExit(f"D154 lock preflight stopped: {exc}") from exc
    path = PROJECT_ROOT / LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"D154 lock created: {LOCK}")


if __name__ == "__main__":
    main()
