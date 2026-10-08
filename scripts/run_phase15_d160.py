"""Execute the one-shot D160 training-only convergence continuation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d160 import D160Error, run_d160  # noqa: E402


def main() -> None:
    try:
        result = run_d160()
    except (D160Error, ValueError) as exc:
        raise SystemExit(f"D160 stopped: {exc}") from exc
    states = ", ".join(f"{name}={record['status']}@{record['epochs_run']}"
                        for name, record in result["models"].items())
    print(f"D160 status={result['status']}; {states}; validation/test/reserve read=0")


if __name__ == "__main__":
    main()
