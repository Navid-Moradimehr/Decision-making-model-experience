"""Small NumPy learners and layout-macro metrics for Phase15 offline diagnosis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from consequence_lab.phase14c import _SquaredErrorTree


def standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std[std < 1e-8] = 1.0
    return tuple((np.asarray(item, dtype=np.float64) - mean) / std for item in (train, *others))


def layout_weights(layouts: np.ndarray) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for value in layouts:
        counts[str(value)] += 1
    weights = np.asarray([1.0 / counts[str(value)] for value in layouts], dtype=np.float64)
    return weights / weights.sum()


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


class TinyRegretMLP:
    """Two-layer 32/32 ReLU network with regression or listwise-pair loss."""

    def __init__(self, seed: int, objective: str, max_epochs: int = 1200, patience: int = 100,
                 learning_rate: float = 0.002, l2: float = 1e-5,
                 record_history: bool = False) -> None:
        if objective not in {"regression", "pairwise", "weighted_pairwise"}:
            raise ValueError("unknown Phase15 objective")
        self.seed, self.objective = int(seed), objective
        self.max_epochs, self.patience = int(max_epochs), int(patience)
        self.learning_rate, self.l2 = float(learning_rate), float(l2)
        self.record_history = bool(record_history)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0.0)
        out = h2 @ self.w3 + self.b3
        return out, (x, h1, h2), (z1, z2)

    def _loss_gradient(self, pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        if self.objective == "regression":
            error = pred - target
            loss = float(np.sum(weights[:, None] * error * error) / 9.0)
            return loss, (2.0 / 9.0) * weights[:, None] * error
        left, right = np.triu_indices(9, 1)
        yi, yj = target[:, left], target[:, right]
        keep = yi != yj
        if not np.any(keep):
            # A fully tied batch contains no preference information. Returning
            # a zero objective/gradient preserves the intended tie semantics.
            return 0.0, np.zeros_like(pred)
        sign = np.sign(yj - yi)
        gaps = np.abs(yj - yi)
        pair_weight = keep.astype(np.float64)
        if self.objective == "weighted_pairwise":
            pair_weight *= gaps
        row_sums = pair_weight.sum(axis=1, keepdims=True)
        active_rows = row_sums[:, 0] > 0.0
        effective_weights = weights * active_rows
        effective_weights /= effective_weights.sum()
        pair_coefficients = effective_weights[:, None] * pair_weight / np.maximum(row_sums, 1e-12)
        margin = sign * (pred[:, right] - pred[:, left])
        # stable softplus(-margin)
        loss_rows = np.logaddexp(0.0, -margin)
        loss = float(np.sum(pair_coefficients * loss_rows))
        derivative = -_sigmoid_stable(-margin) * sign * pair_coefficients
        grad = np.zeros_like(pred)
        pair_shape = margin.shape
        row_ids = np.broadcast_to(np.arange(len(target))[:, None], pair_shape)
        left_ids = np.broadcast_to(left[None, :], row_ids.shape)
        right_ids = np.broadcast_to(right[None, :], row_ids.shape)
        np.add.at(grad, (row_ids, left_ids), -derivative)
        np.add.at(grad, (row_ids, right_ids), derivative)
        return loss, grad

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, train_layouts: np.ndarray,
            x_val: np.ndarray, y_val: np.ndarray, val_layouts: np.ndarray) -> "TinyRegretMLP":
        self.x_mean = np.mean(x_train, axis=0)
        self.x_std = np.std(x_train, axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        x = (np.asarray(x_train, dtype=np.float64) - self.x_mean) / self.x_std
        xv = (np.asarray(x_val, dtype=np.float64) - self.x_mean) / self.x_std
        y = np.asarray(y_train, dtype=np.float64)
        yv = np.asarray(y_val, dtype=np.float64)
        wt = layout_weights(np.asarray(train_layouts))
        wv = layout_weights(np.asarray(val_layouts))
        rng = np.random.default_rng(self.seed)
        dims = (x.shape[1], 32, 32, 9)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / dims[0]), (dims[0], 32))
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / 32), (32, 32))
        self.w3 = rng.normal(0.0, np.sqrt(2.0 / 32), (32, 9))
        self.b1, self.b2, self.b3 = np.zeros(32), np.zeros(32), np.zeros(9)
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        best = None
        best_loss = (float("inf"), float("inf"))
        stale = 0
        self.history: list[dict[str, float | int | bool]] = []
        for epoch in range(self.max_epochs):
            pred, acts, zs = self._forward(x)
            train_loss, delta = self._loss_gradient(pred, y, wt)
            x0, h1, h2 = acts
            z1, z2 = zs
            gw3 = h2.T @ delta + self.l2 * self.w3
            gb3 = delta.sum(axis=0)
            dh2 = delta @ self.w3.T
            dz2 = dh2 * (z2 > 0.0)
            gw2 = h1.T @ dz2 + self.l2 * self.w2
            gb2 = dz2.sum(axis=0)
            dh1 = dz2 @ self.w2.T
            dz1 = dh1 * (z1 > 0.0)
            gw1 = x0.T @ dz1 + self.l2 * self.w1
            gb1 = dz1.sum(axis=0)
            grads = [gw1, gb1, gw2, gb2, gw3, gb3]
            t = epoch + 1
            for i, (parameter, grad) in enumerate(zip(params, grads, strict=True)):
                first[i] = 0.9 * first[i] + 0.1 * grad
                second[i] = 0.999 * second[i] + 0.001 * grad * grad
                m = first[i] / (1.0 - 0.9**t)
                v = second[i] / (1.0 - 0.999**t)
                parameter -= self.learning_rate * m / (np.sqrt(v) + 1e-8)
            val_pred, _, _ = self._forward(xv)
            val_loss, _ = self._loss_gradient(val_pred, yv, wv)
            val_decision_regret = decision_metrics(yv, val_pred, np.asarray(val_layouts))[
                "mean_normalized_regret_layout_macro"
            ]
            criterion = (float(val_decision_regret), float(val_loss))
            improved = (criterion[0] < best_loss[0] - 1e-9
                        or (abs(criterion[0] - best_loss[0]) <= 1e-9
                            and criterion[1] < best_loss[1] - 1e-7))
            if improved:
                best_loss, stale = criterion, 0
                best = [p.copy() for p in params]
                self.best_epoch = epoch + 1
                self.train_loss_at_best = train_loss
            else:
                stale += 1
            if self.record_history:
                self.history.append({
                    "epoch": epoch + 1,
                    "train_loss_pre_update": float(train_loss),
                    "gradient_l2_norm": float(np.sqrt(sum(float(np.sum(g * g)) for g in grads))),
                    "validation_loss": float(val_loss),
                    "validation_decision_regret": float(val_decision_regret),
                    "best_checkpoint_updated": bool(improved),
                    "stale_epochs": int(stale),
                })
            if not improved and stale >= self.patience:
                break
        if best is None:
            raise RuntimeError("MLP failed to produce a validation checkpoint")
        for name, value in zip(("w1", "b1", "w2", "b2", "w3", "b3"), best, strict=True):
            setattr(self, name, value)
        self.validation_loss = best_loss[1]
        self.validation_decision_regret = best_loss[0]
        self.epochs_run = epoch + 1
        self.parameter_count = int(sum(p.size for p in params))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        rows = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        return self._forward(rows)[0]


def decision_metrics(y: np.ndarray, predicted: np.ndarray, layouts: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    if target.shape != pred.shape or target.ndim != 2 or target.shape[1] != 9:
        raise ValueError("Phase15 metrics require matching [N,9] target and score arrays")
    chosen = np.argmin(pred, axis=1)
    regret = target[np.arange(len(target)), chosen]
    best = target.min(axis=1)
    agreement = np.isclose(target[np.arange(len(target)), chosen], best, rtol=0.0, atol=1e-12)
    pair_ok: list[float] = []
    weighted_ok: list[float] = []
    for yi, pi in zip(target, pred, strict=True):
        i, j = np.triu_indices(9, 1)
        keep = yi[i] != yi[j]
        correct = (np.sign(yi[j[keep]] - yi[i[keep]]) == np.sign(pi[j[keep]] - pi[i[keep]])).astype(float)
        gaps = np.abs(yi[j[keep]] - yi[i[keep]])
        pair_ok.append(float(correct.mean()) if len(correct) else 1.0)
        weighted_ok.append(float(np.average(correct, weights=gaps)) if len(correct) and gaps.sum() > 0 else 1.0)
    per_layout: dict[str, list[int]] = defaultdict(list)
    for i, layout in enumerate(layouts):
        per_layout[str(layout)].append(i)
    layout_means = {key: float(np.mean(regret[idx])) for key, idx in per_layout.items()}
    layout_catastrophic = {key: float(np.mean(regret[idx] >= 0.5)) for key, idx in per_layout.items()}
    macro_regret = float(np.mean(list(layout_means.values())))
    macro_catastrophic = float(np.mean(list(layout_catastrophic.values())))
    return {
        "state_count": len(target), "layout_count": len(per_layout),
        "mean_normalized_regret_state_micro": float(regret.mean()),
        "mean_normalized_regret_layout_macro": macro_regret,
        "median_normalized_regret": float(np.median(regret)),
        "catastrophic_fraction_state_micro": float(np.mean(regret >= 0.5)),
        "catastrophic_fraction_layout_macro": macro_catastrophic,
        "best_action_set_agreement": float(np.mean(agreement)),
        "pairwise_ranking_accuracy_macro": float(np.mean(pair_ok)),
        "regret_gap_weighted_pairwise_accuracy_macro": float(np.mean(weighted_ok)),
        "predicted_action_frequency": np.bincount(chosen, minlength=9).tolist(),
        "per_layout_mean_regret": layout_means,
        "selected_actions": chosen.tolist(),
        "selected_regret": regret.tolist(),
    }


def fit_cart(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, int]:
    x_mean = np.mean(train_x, axis=0)
    x_std = np.std(train_x, axis=0)
    x_std[x_std < 1e-8] = 1.0
    xt = (train_x - x_mean) / x_std
    xv = (test_x - x_mean) / x_std
    tree = _SquaredErrorTree().fit(xt, train_y)
    return tree.predict(xv), int(tree.node_count)
