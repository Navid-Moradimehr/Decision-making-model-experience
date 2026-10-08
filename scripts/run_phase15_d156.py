"""Run the single locked D156 component-swap audit."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d156 import D156Error, run_d156  # noqa: E402


def main() -> None:
    try:
        result = run_d156()
    except (D156Error, ValueError) as exc:
        raise SystemExit(f"D156 stopped: {exc}") from exc
    print(f"D156 complete; component swap rows={sum(len(v) for v in result['predictors'].values())}")


if __name__ == "__main__":
    main()
