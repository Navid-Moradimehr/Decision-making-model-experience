"""Run the single locked D155 Stage-A oracle consequence ablation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d155 import D155Error, run_d155  # noqa: E402


def main() -> None:
    try:
        result = run_d155()
    except (D155Error, ValueError) as exc:
        raise SystemExit(f"D155 Stage A stopped: {exc}") from exc
    gate = result["stage_c_gate"]
    print(f"D155 Stage A complete; compact subset={gate['selected_compact_subset']}; "
          f"Stage C executed={gate['stage_c_executed']}")


if __name__ == "__main__":
    main()
