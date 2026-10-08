"""Pure split and screening helpers for exploratory Phase15 representation study."""

from __future__ import annotations

import numpy as np

from consequence_lab.phase15_data import layout_split
from consequence_lab.phase15_generalization import layout_list_sha256


def train_validation_indices(layouts: list[str], expected: dict) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return only train/validation row indices; never expose the test indices."""
    split = layout_split([str(value) for value in layouts])
    train_layouts, validation_layouts = sorted(split["train"]), sorted(split["validation"])
    if (len(train_layouts) != expected["train_layout_count"]
            or len(validation_layouts) != expected["validation_layout_count"]
            or layout_list_sha256(train_layouts) != expected["train_layouts_sha256"]
            or layout_list_sha256(validation_layouts) != expected["validation_layouts_sha256"]):
        raise ValueError("Phase15 train/validation layout split mismatch")
    train_set, validation_set = set(train_layouts), set(validation_layouts)
    train_idx = np.asarray([i for i, value in enumerate(layouts) if str(value) in train_set], dtype=np.int64)
    validation_idx = np.asarray([i for i, value in enumerate(layouts) if str(value) in validation_set], dtype=np.int64)
    if len(train_idx) != expected["train_row_count"] or len(validation_idx) != expected["validation_row_count"]:
        raise ValueError("Phase15 train/validation row count mismatch")
    if set(train_idx.tolist()) & set(validation_idx.tolist()):
        raise ValueError("train/validation row overlap")
    return train_idx, validation_idx, train_layouts, validation_layouts


def validate_r0_reproduction(record: dict, expected: dict) -> None:
    """Raise unless the R0/pairwise refit reproduces its locked D148 checkpoint screen."""
    tol = float(expected["tolerance"])
    if record["best_epoch"] != expected["best_epoch"]:
        raise ValueError("R0 reproduction best epoch mismatch")
    checks = (
        (record["validation_metrics"]["mean_normalized_regret_layout_macro"], expected["validation_layout_macro_regret"]),
        (record["validation_metrics"]["best_action_set_agreement"], expected["validation_best_action_set_agreement"]),
    )
    if any(abs(float(observed) - float(reference)) > tol for observed, reference in checks):
        raise ValueError("R0 reproduction validation metric mismatch")


def representation_screen(metrics: dict, minimum_regret_improvement: float,
                           minimum_catastrophic_improvement: float) -> dict[str, dict]:
    """Development-only nominations relative to R0 validation metrics."""
    baseline = metrics["R0"]
    baseline_regret = float(baseline["mean_normalized_regret_layout_macro"])
    baseline_cat = float(baseline["catastrophic_fraction_state_micro"])
    result = {}
    for name in ("R1", "R2", "R3", "R4"):
        current = metrics[name]
        regret_gain = baseline_regret - float(current["mean_normalized_regret_layout_macro"])
        cat_gain = baseline_cat - float(current["catastrophic_fraction_state_micro"])
        result[name] = {"validation_macro_regret_gain_vs_R0": regret_gain,
                        "validation_catastrophic_fraction_gain_vs_R0": cat_gain,
                        "promising_for_fresh_confirmation_only": (
                            regret_gain >= minimum_regret_improvement
                            and cat_gain >= minimum_catastrophic_improvement)}
    return result
