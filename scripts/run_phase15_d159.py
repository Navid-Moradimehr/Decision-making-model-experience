"""Execute the one-shot training-only D159 fit audit."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d159 import D159Error, run_d159  # noqa: E402


def main() -> None:
    try:
        result = run_d159()
    except (D159Error, ValueError) as exc:
        raise SystemExit(f"D159 stopped: {exc}") from exc
    print(f"D159 v2 status={result['status']}; oracle-fit={result['oracle_capacity_control_passed']}; "
          f"models={len(result['models'])}; validation/test/reserve read=0")


if __name__ == "__main__":
    main()
