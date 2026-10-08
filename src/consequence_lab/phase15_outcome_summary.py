"""Small offline predictor for saved Phase15 R4 continuation summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from pathlib import Path

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _layout_weights(layouts: np.ndarray) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for layout in layouts:
        counts[str(layout)] += 1
    weights = np.asarray([1.0 / counts[str(layout)] for layout in layouts], dtype=np.float64)
    return weights / weights.sum()


class OutcomeSummaryMLP:
    """Train-only-standardized R0-to-[9,7] sigmoid summary predictor."""

    def __init__(self, seed: int = 15151, hidden: tuple[int, int] = (32, 32),
                 max_epochs: int = 1200, patience: int = 100,
                 learning_rate: float = 0.002, l2: float = 1e-5) -> None:
        self.seed, self.hidden = int(seed), tuple(map(int, hidden))
        self.max_epochs, self.patience = int(max_epochs), int(patience)
        self.learning_rate, self.l2 = float(learning_rate), float(l2)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0.0)
        out = _sigmoid(h2 @ self.w3 + self.b3)
        return out, (x, h1, h2), (z1, z2)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, train_layouts: np.ndarray,
            x_val: np.ndarray, y_val: np.ndarray, val_layouts: np.ndarray) -> "OutcomeSummaryMLP":
        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        x_val = np.asarray(x_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)
        if (x_train.ndim != 2 or x_val.ndim != 2 or x_train.shape[1] != x_val.shape[1]
                or y_train.shape != (len(x_train), 63) or y_val.shape != (len(x_val), 63)
                or len(train_layouts) != len(x_train) or len(val_layouts) != len(x_val)):
            raise ValueError("summary predictor requires X=[N,D], Y=[N,63], and matching layout rows")
        if any(not np.isfinite(value).all() for value in (x_train, y_train, x_val, y_val)):
            raise ValueError("summary predictor inputs must be finite")
        if np.any(y_train < 0.0) or np.any(y_train > 1.0) or np.any(y_val < 0.0) or np.any(y_val > 1.0):
            raise ValueError("R4 summary targets must lie in [0,1]")
        self.x_mean = x_train.mean(axis=0)
        self.x_std = x_train.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        x = (x_train - self.x_mean) / self.x_std
        xv = (x_val - self.x_mean) / self.x_std
        wt = _layout_weights(np.asarray(train_layouts))
        wv = _layout_weights(np.asarray(val_layouts))
        dimensions = (x.shape[1], *self.hidden, 63)
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / dimensions[0]), (dimensions[0], dimensions[1]))
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / dimensions[1]), (dimensions[1], dimensions[2]))
        self.w3 = rng.normal(0.0, np.sqrt(1.0 / dimensions[2]), (dimensions[2], 63))
        self.b1, self.b2, self.b3 = np.zeros(dimensions[1]), np.zeros(dimensions[2]), np.zeros(63)
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        best_params = None
        best_val = float("inf")
        stale = 0
        for epoch in range(self.max_epochs):
            pred, acts, zs = self._forward(x)
            delta = (2.0 / 63.0) * wt[:, None] * (pred - y_train) * pred * (1.0 - pred)
            x0, h1, h2 = acts
            z1, z2 = zs
            grads = [None] * 6
            grads[4] = h2.T @ delta + self.l2 * self.w3
            grads[5] = delta.sum(axis=0)
            dz2 = (delta @ self.w3.T) * (z2 > 0.0)
            grads[2] = h1.T @ dz2 + self.l2 * self.w2
            grads[3] = dz2.sum(axis=0)
            dz1 = (dz2 @ self.w2.T) * (z1 > 0.0)
            grads[0] = x0.T @ dz1 + self.l2 * self.w1
            grads[1] = dz1.sum(axis=0)
            t = epoch + 1
            for i, (parameter, grad) in enumerate(zip(params, grads, strict=True)):
                first[i] = 0.9 * first[i] + 0.1 * grad
                second[i] = 0.999 * second[i] + 0.001 * grad * grad
                parameter -= self.learning_rate * (first[i] / (1.0 - 0.9**t)) / (
                    np.sqrt(second[i] / (1.0 - 0.999**t)) + 1e-8)
            validation_prediction = self._forward(xv)[0]
            val_loss = float(np.sum(wv[:, None] * (validation_prediction - y_val) ** 2) / 63.0)
            if val_loss < best_val - 1e-12:
                best_val = val_loss
                best_params = [p.copy() for p in params]
                self.best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
            if stale >= self.patience:
                break
        if best_params is None:
            raise RuntimeError("summary predictor failed to select a validation checkpoint")
        for name, value in zip(("w1", "b1", "w2", "b2", "w3", "b3"), best_params, strict=True):
            setattr(self, name, value)
        self.epochs_run = epoch + 1
        self.validation_mse = best_val
        self.parameter_count = int(sum(p.size for p in params))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        rows = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        return self._forward(rows)[0]


def summary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 63:
        raise ValueError("summary metrics require matching [N,63] arrays")
    t = target.reshape(-1, 9, 7)
    p = prediction.reshape(-1, 9, 7)
    return {"rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "mae": float(np.mean(np.abs(prediction - target))),
            "outcome_argmax_accuracy": float(np.mean(np.argmax(t[:, :, :4], axis=2) == np.argmax(p[:, :, :4], axis=2))),
            "terminal_steps_mae": float(np.mean(np.abs(t[:, :, 4] - p[:, :, 4]))),
            "terminal_goal_distance_mae": float(np.mean(np.abs(t[:, :, 5] - p[:, :, 5]))),
            "noncoast_effort_mae": float(np.mean(np.abs(t[:, :, 6] - p[:, :, 6])))}


def constant_summary(train_target: np.ndarray, train_layouts: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(train_target, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 63 or len(train_layouts) != len(values):
        raise ValueError("constant summary baseline requires [N,63] targets and matching layouts")
    counts: dict[str, int] = defaultdict(int)
    for layout in train_layouts:
        counts[str(layout)] += 1
    weights = np.asarray([1.0 / counts[str(layout)] for layout in train_layouts], dtype=np.float64)
    weights /= weights.sum()
    return np.repeat(np.sum(values * weights[:, None], axis=0, keepdims=True), count, axis=0)


def screen_prediction(rmse: float, constant_rmse: float, decision_metrics: dict,
                      thresholds: dict) -> dict[str, Any]:
    relative_gain = 1.0 - float(rmse) / max(float(constant_rmse), 1e-12)
    pass_screen = (relative_gain >= thresholds["summary_rmse_relative_improvement_vs_constant_minimum"]
                   and decision_metrics["mean_normalized_regret_layout_macro"] <= thresholds["downstream_macro_regret_maximum"]
                   and decision_metrics["catastrophic_fraction_state_micro"] <= thresholds["downstream_catastrophic_fraction_maximum"])
    return {"summary_rmse_relative_improvement_vs_constant": relative_gain,
            "promising_for_fresh_confirmation_only": bool(pass_screen)}


def restore_decision_head(path: Path) -> Any:
    """Restore a frozen Phase15 MLP checkpoint, validating dimensions from its arrays."""
    # Import locally to keep the summary predictor usable without decision-head tooling.
    from consequence_lab.phase15_learning import TinyRegretMLP

    model = TinyRegretMLP(seed=0, objective="pairwise")
    with np.load(path, allow_pickle=False) as saved:
        required = ("x_mean", "x_std", "w1", "b1", "w2", "b2", "w3", "b3")
        if set(required) - set(saved.files):
            raise ValueError("frozen decision checkpoint is missing arrays")
        for key in required:
            setattr(model, key, np.asarray(saved[key], dtype=np.float64))
    input_dim = model.x_mean.shape[0] if model.x_mean.ndim == 1 else -1
    hidden1 = model.b1.shape[0] if model.b1.ndim == 1 else -1
    hidden2 = model.b2.shape[0] if model.b2.ndim == 1 else -1
    expected = {"x_std": (input_dim,), "w1": (input_dim, hidden1), "b1": (hidden1,),
                "w2": (hidden1, hidden2), "b2": (hidden2,), "w3": (hidden2, 9), "b3": (9,)}
    for name, shape in expected.items():
        value = getattr(model, name)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid frozen decision checkpoint array {name}: expected {shape}, got {value.shape}")
    if input_dim not in (61, 124):
        raise ValueError(f"unexpected D150 decision-head input dimension {input_dim}")
    model.parameter_count = int(sum(getattr(model, key).size for key in
                                    ("w1", "b1", "w2", "b2", "w3", "b3")))
    return model


def validate_summary_targets(target: np.ndarray) -> None:
    """Check the declared 9x7 outcome-summary contract on rows actually used."""
    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 63 or not np.isfinite(values).all():
        raise ValueError("R4 summary targets must be finite [N,63]")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("R4 summary targets must lie in [0,1]")
    outcome = values.reshape(-1, 9, 7)[:, :, :4]
    if not np.all((outcome == 0.0) | (outcome == 1.0)) or not np.all(outcome.sum(axis=2) == 1.0):
        raise ValueError("each action's terminal outcome must be a four-way one-hot vector")
