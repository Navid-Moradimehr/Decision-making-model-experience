"""Run only the corrected D60/D152 prediction and metric reproduction gate."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d153_scaling import D153ScalingError, run_d60_reproduction_gate  # noqa: E402


def main() -> None:
    try:
        result = run_d60_reproduction_gate()
    except D153ScalingError as exc:
        raise SystemExit(f"D153 corrected D60 reproduction stopped: {exc}") from exc
    print("D153 corrected D60 reproduction passed")
    print(f"maximum absolute prediction difference: {result['maximum_absolute_prediction_difference']:.12g}")
    print(f"weighted validation RMSE: {result['weighted_validation_rmse']:.12g}")
    print(f"decision metric max difference: {result['saved_d152_decision_metric_maximum_absolute_difference']:.12g}")


if __name__ == "__main__":
    main()
