"""Run the single locked D158 offline relational candidate study."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d158 import D158Error, run_d158  # noqa: E402


def main() -> None:
    try:
        result = run_d158()
    except (D158Error, ValueError) as exc:
        raise SystemExit(f"D158 stopped: {exc}") from exc
    print(f"D158 complete; training layouts={result['split']['train_layout_count']}; "
          f"held-out layouts={result['split']['d157_heldout_layout_count']}; gates={result['advancement_gates']}")


if __name__ == "__main__":
    main()
