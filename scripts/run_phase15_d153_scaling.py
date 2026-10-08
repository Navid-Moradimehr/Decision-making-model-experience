"""Execute the corrected, locked D153 coverage-scaling study from preserved data."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d153_scaling import D153ScalingError, run_scaling  # noqa: E402


def main() -> None:
    try:
        result = run_scaling()
    except D153ScalingError as exc:
        raise SystemExit(f"D153 scaling stopped: {exc}") from exc
    print(f"D153 completed: {result['interpretation']}")


if __name__ == "__main__":
    main()
