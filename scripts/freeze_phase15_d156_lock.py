"""Freeze the no-training D155 predicted/oracle component-swap audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d156 import D156Error, build_lock  # noqa: E402


def main() -> None:
    try:
        lock = build_lock()
    except (D156Error, ValueError) as exc:
        raise SystemExit(f"D156 lock preflight stopped: {exc}") from exc
    path = ROOT / lock["outputs"]["lock"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"D156 lock created: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
