"""Run the single locked Stage-C compact-consequence prediction screen."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d155_stage_c import D155StageCError, run_stage_c  # noqa: E402


def main() -> None:
    try:
        result = run_stage_c()
    except (D155StageCError, ValueError) as exc:
        raise SystemExit(f"D155 Stage C stopped: {exc}") from exc
    print(f"D155 Stage C complete; offline advancement gate={result['offline_gate_passed']}")


if __name__ == "__main__":
    main()
