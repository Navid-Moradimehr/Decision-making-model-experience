"""Run the single locked D154 offline decision-target comparison."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_decision_aligned import D154Error, run_d154  # noqa: E402


def main() -> None:
    try:
        result = run_d154()
    except (D154Error, ValueError) as exc:
        raise SystemExit(f"D154 stopped: {exc}") from exc
    print(f"D154 completed: {result['decision_summary']}")


if __name__ == "__main__":
    main()
