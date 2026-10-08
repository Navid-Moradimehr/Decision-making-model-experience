"""Run the single locked D157 fresh-layout confirmation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d157 import D157Error, run_d157  # noqa: E402


def main() -> None:
    try:
        result = run_d157()
    except (D157Error, ValueError) as exc:
        raise SystemExit(f"D157 stopped: {exc}") from exc
    print(f"D157 complete; states={result['collection']['state_rows']}; gates={result['gates']}")


if __name__ == "__main__":
    main()
