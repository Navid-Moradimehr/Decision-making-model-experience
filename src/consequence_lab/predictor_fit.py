"""NumPy data transforms and ridge baseline for latent-dynamics fitting."""

from __future__ import annotations

import numpy as np


def make_predictor_features(start_latents: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Concatenate one start latent with each flattened action sequence."""
    start = np.asarray(start_latents, dtype=np.float64)
    sequence = np.asarray(actions, dtype=np.float64)
    if start.ndim != 2 or sequence.ndim < 2 or len(start) != len(sequence):
        raise ValueError("start_latents and actions must share a candidate dimension")
    if not np.isfinite(start).all() or not np.isfinite(sequence).all():
        raise ValueError("predictor inputs must be finite")
    return np.concatenate([start, sequence.reshape(len(sequence), -1)], axis=1)


def fit_standardized_ridge(
    features: np.ndarray, targets: np.ndarray, *, l2: float
) -> dict[str, np.ndarray | float]:
    """Fit standardized linear regression with an unpenalized intercept."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("features and targets must be aligned 2D arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge data must be finite")
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and non-negative")
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale = np.where(x_scale > 1e-8, x_scale, 1.0)
    y_mean = y.mean(axis=0)
    y_scale = y.std(axis=0)
    y_scale = np.where(y_scale > 1e-8, y_scale, 1.0)
    x_norm = (x - x_mean) / x_scale
    y_norm = (y - y_mean) / y_scale
    design = np.concatenate([x_norm, np.ones((len(x_norm), 1))], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y_norm)
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "weights": weights,
        "l2": float(l2),
    }


def predict_standardized_ridge(
    model: dict[str, np.ndarray | float], features: np.ndarray
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    x_norm = (x - model["x_mean"]) / model["x_scale"]
    design = np.concatenate([x_norm, np.ones((len(x_norm), 1))], axis=1)
    return (design @ model["weights"]) * model["y_scale"] + model["y_mean"]
