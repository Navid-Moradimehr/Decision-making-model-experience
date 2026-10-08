"""Helpers for post-hoc, read-only audits of frozen Phase15 checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from consequence_lab.phase15_learning import TinyRegretMLP


def restore_frozen_mlp(path: Path, objective: str) -> TinyRegretMLP:
    """Restore exact saved arrays without constructing an optimizer or training."""
    model = TinyRegretMLP(seed=15015, objective=objective)
    with np.load(path, allow_pickle=False) as saved:
        for key in ("x_mean", "x_std", "w1", "b1", "w2", "b2", "w3", "b3"):
            setattr(model, key, np.asarray(saved[key], dtype=np.float64))
    expected = {"x_mean": (61,), "x_std": (61,), "w1": (61, 32), "b1": (32,),
                "w2": (32, 32), "b2": (32,), "w3": (32, 9), "b3": (9,)}
    for name, shape in expected.items():
        value = getattr(model, name)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid frozen checkpoint array {name}: expected {shape}, got {value.shape}")
    model.parameter_count = int(sum(getattr(model, name).size for name in ("w1", "b1", "w2", "b2", "w3", "b3")))
    return model


def interpret_train_fit(pairwise_metrics: dict, validation_macro_regret: float, bands: dict) -> str:
    """Apply frozen descriptive bands; never authorizes a downstream run."""
    train = float(pairwise_metrics["mean_normalized_regret_layout_macro"])
    strong = (train <= float(bands["strong_train_fit_mean_layout_macro_regret_max"])
              and float(pairwise_metrics["best_action_set_agreement"])
              >= float(bands["strong_train_fit_best_action_agreement_min"]))
    large_gap = float(validation_macro_regret) - train >= float(bands["large_validation_train_regret_gap_min"])
    if strong and large_gap:
        return ("Strong pairwise in-sample fit with a large recorded validation gap: layout/generalization limitation is plausible. "
                "This is not causal evidence and does not authorize DAgger; next, design a fresh-layout representation/coverage diagnostic.")
    if strong:
        return ("Strong pairwise in-sample fit without the preregistered large validation gap. The recorded validation regret remains poor, "
                "so no closed-loop or DAgger step is justified; prioritize a consequence-feature diagnostic.")
    return ("Pairwise did not meet the preregistered strong in-sample-fit band. The current R0 student does not fit adequately; "
            "distribution shift is not established as the main remedy. Next, diagnose richer saved consequence representations before DAgger.")
