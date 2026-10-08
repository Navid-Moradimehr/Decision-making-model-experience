"""Freeze the score-only D157 correction; no simulator/planner calls are made."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d157_score_finalizer import D157FinalizerError, build_lock  # noqa: E402


def main() -> None:
    try:
        lock = build_lock()
    except (D157FinalizerError, ValueError) as exc:
        raise SystemExit(f"D157 score-only lock preflight stopped: {exc}") from exc
    path = ROOT / lock["outputs"]["lock"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"D157 score-only lock created: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
