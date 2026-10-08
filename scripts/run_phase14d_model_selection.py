"""Generate/train only the locked Phase14D model-selection block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from consequence_lab.phase14d import run_model_selection


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run_model_selection(args.root.resolve())
    selected = result["selected_candidate"]
    print(json.dumps({
        "status": "completed_model_selection",
        "selection_layout_count": result["selection_layout_count"],
        "selection_state_count": result["selection_state_count"],
        "selected_candidate": {
            key: selected[key]
            for key in (
                "learner", "feature_variant", "target_variant",
                "layout_macro_mean_normalized_regret", "improvement_over_geometry",
            )
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
