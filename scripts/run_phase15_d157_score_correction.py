"""Score the preserved D157 dataset once without new labels or simulation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d157_score_finalizer import D157FinalizerError, run_score_correction  # noqa: E402


def main() -> None:
    try:
        result = run_score_correction()
    except (D157FinalizerError, ValueError) as exc:
        raise SystemExit(f"D157 score-only correction stopped: {exc}") from exc
    print(f"D157 score-only correction complete; states={result['collection']['state_rows']}; gates={result['gates']}")


if __name__ == "__main__":
    main()
