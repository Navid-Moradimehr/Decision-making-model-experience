"""Execute or resume the locked D161 training-only capacity study."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d161 import D161Error, run_d161  # noqa: E402


def main() -> None:
    try:
        result = run_d161()
    except (D161Error, ValueError) as exc:
        raise SystemExit(f"D161 stopped: {exc}") from exc
    passed = [key for key, record in result["models"].items() if record["fit_gate_passed"]]
    print(f"D161 status={result['status']}; fits={len(result['models'])}; gate_passed={passed}; "
          "validation/test/reserve read=0")


if __name__ == "__main__":
    main()
