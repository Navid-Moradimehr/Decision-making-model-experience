"""D152 offline audits for R0 consequence sufficiency and action-conditioned prediction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from consequence_lab.phase15_learning import decision_metrics, layout_weights


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def expand_action_examples(states: np.ndarray, summaries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand state x 9 summaries into shared (state, action-onehot) examples."""
    state = np.asarray(states, dtype=np.float64)
    target = np.asarray(summaries, dtype=np.float64)
    if state.ndim != 2 or target.shape != (len(state), 9, 7):
        raise ValueError("D152 expansion expects states [N,D] and summaries [N,9,7]")
    action = np.eye(9, dtype=np.float64)
    repeated_state = np.repeat(state[:, None, :], 9, axis=1)
    repeated_action = np.repeat(action[None, :, :], len(state), axis=0)
    features = np.concatenate((repeated_state, repeated_action), axis=2).reshape(len(state) * 9, -1)
    expanded_target = target.reshape(len(state) * 9, 7)
    action_ids = np.tile(np.arange(9, dtype=np.int64), len(state))
    return features, expanded_target, action_ids


def expand_layout_weights(layouts: np.ndarray) -> np.ndarray:
    """Give each layout equal total weight and divide it evenly over nine actions."""
    layout = np.asarray(layouts).astype(str)
    return np.repeat(layout_weights(layout) / 9.0, 9)


class ActionSummaryMLP:
    """Small train-only-standardized shared state/action MLP, seven sigmoid outputs."""

    def __init__(self, seed: int, hidden: tuple[int, int] = (32, 32), epochs: int = 800,
                 learning_rate: float = 0.002, l2: float = 1e-5) -> None:
        self.seed, self.hidden, self.epochs = int(seed), tuple(hidden), int(epochs)
        self.learning_rate, self.l2 = float(learning_rate), float(l2)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0.0)
        output = _sigmoid(h2 @ self.w3 + self.b3)
        return output, (x, h1, h2), (z1, z2)

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> "ActionSummaryMLP":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weights = np.asarray(sample_weights, dtype=np.float64)
        if (x.ndim != 2 or y.shape != (len(x), 7) or weights.shape != (len(x),)
                or not all(np.isfinite(item).all() for item in (x, y, weights))
                or np.any(y < 0.0) or np.any(y > 1.0) or np.any(weights < 0.0)
                or not np.isclose(weights.sum(), 1.0, atol=1e-10)):
            raise ValueError("invalid D152 MLP data/weight contract")
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        xn = (x - self.x_mean) / self.x_std
        dimensions = (x.shape[1], *self.hidden, 7)
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / dimensions[0]), (dimensions[0], dimensions[1]))
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / dimensions[1]), (dimensions[1], dimensions[2]))
        self.w3 = rng.normal(0.0, np.sqrt(1.0 / dimensions[2]), (dimensions[2], 7))
        self.b1, self.b2, self.b3 = np.zeros(dimensions[1]), np.zeros(dimensions[2]), np.zeros(7)
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        first = [np.zeros_like(parameter) for parameter in params]
        second = [np.zeros_like(parameter) for parameter in params]
        best, best_loss = None, float("inf")
        self.training_loss_history: list[float] = []
        for epoch in range(self.epochs):
            predicted, activations, preactivations = self._forward(xn)
            loss = float(np.sum(weights[:, None] * (predicted - y) ** 2) / 7.0)
            delta = (2.0 / 7.0) * weights[:, None] * (predicted - y) * predicted * (1.0 - predicted)
            x0, h1, h2 = activations
            z1, z2 = preactivations
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
            for index, (parameter, gradient) in enumerate(zip(params, grads, strict=True)):
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
                parameter -= self.learning_rate * (first[index] / (1.0 - 0.9**t)) / (
                    np.sqrt(second[index] / (1.0 - 0.999**t)) + 1e-8)
            # The fixed-fit checkpoint is selected only on training objective.
            post_update = self._forward(xn)[0]
            post_loss = float(np.sum(weights[:, None] * (post_update - y) ** 2) / 7.0)
            self.training_loss_history.append(post_loss)
            if post_loss < best_loss:
                best_loss, best = post_loss, [parameter.copy() for parameter in params]
                self.best_epoch = epoch + 1
        if best is None:
            raise RuntimeError("D152 MLP did not produce a finite training checkpoint")
        for name, value in zip(("w1", "b1", "w2", "b2", "w3", "b3"), best, strict=True):
            setattr(self, name, value)
        self.training_loss = best_loss
        self.parameter_count = int(sum(parameter.size for parameter in params))
        return self

    def predict_examples(self, x: np.ndarray) -> np.ndarray:
        return self._forward((np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std)[0]

    def predict(self, states: np.ndarray) -> np.ndarray:
        features, _, _ = expand_action_examples(states, np.zeros((len(states), 9, 7)))
        return self.predict_examples(features).reshape(len(states), 9, 7)


@dataclass
class _CartNode:
    value: np.ndarray
    feature: int = -1
    threshold: float = 0.0
    left: "_CartNode | None" = None
    right: "_CartNode | None" = None


class WeightedMultiOutputCART:
    """Deterministic, bounded-depth multi-output CART with layout sample weights."""

    def __init__(self, max_depth: int = 6, min_leaf: int = 32) -> None:
        self.max_depth, self.min_leaf = int(max_depth), int(min_leaf)

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> "WeightedMultiOutputCART":
        self.x, self.y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.x.ndim != 2 or self.y.shape != (len(self.x), 7) or self.weights.shape != (len(self.x),):
            raise ValueError("invalid D152 CART shapes")
        indices = np.arange(len(self.x), dtype=np.int64)
        self.root = self._grow(indices, 0)
        self.node_count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            self.node_count += 1
            if node.left is not None and node.right is not None:
                stack.extend((node.left, node.right))
        return self

    def _weighted_stats(self, indices: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        w = self.weights[indices]
        total = float(w.sum())
        means = np.sum(self.y[indices] * w[:, None], axis=0) / max(total, 1e-15)
        squares = np.sum(self.y[indices] * self.y[indices] * w[:, None], axis=0)
        sums = np.sum(self.y[indices] * w[:, None], axis=0)
        sse = float(np.sum(squares - sums * sums / max(total, 1e-15)))
        return sse, means, w

    def _grow(self, indices: np.ndarray, depth: int) -> _CartNode:
        _, mean, _ = self._weighted_stats(indices)
        node = _CartNode(mean)
        if depth >= self.max_depth or len(indices) < 2 * self.min_leaf:
            return node
        total_sse, _, _ = self._weighted_stats(indices)
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        for feature in range(self.x.shape[1]):
            order = np.argsort(self.x[indices, feature], kind="mergesort")
            ids = indices[order]
            xs = self.x[ids, feature]
            boundaries = np.flatnonzero((xs[:-1] < xs[1:]) &
                                        (np.arange(1, len(xs)) >= self.min_leaf) &
                                        (np.arange(1, len(xs)) <= len(xs) - self.min_leaf)) + 1
            if not len(boundaries):
                continue
            row_w = self.weights[ids]
            weighted_y = self.y[ids] * row_w[:, None]
            prefix_w = np.cumsum(row_w)
            prefix_y = np.cumsum(weighted_y, axis=0)
            prefix_sq = np.cumsum(self.y[ids] * self.y[ids] * row_w[:, None], axis=0)
            total_w = prefix_w[-1]
            total_y = prefix_y[-1]
            total_sq = prefix_sq[-1]
            left_w = prefix_w[boundaries - 1]
            right_w = total_w - left_w
            left_y = prefix_y[boundaries - 1]
            right_y = total_y - left_y
            left_sq = prefix_sq[boundaries - 1]
            right_sq = total_sq - left_sq
            losses = np.sum(left_sq - left_y * left_y / np.maximum(left_w[:, None], 1e-15), axis=1)
            losses += np.sum(right_sq - right_y * right_y / np.maximum(right_w[:, None], 1e-15), axis=1)
            offset = int(np.argmin(losses))
            boundary = int(boundaries[offset])
            candidate = (float(losses[offset]), feature,
                         float(xs[boundary - 1] + (xs[boundary] - xs[boundary - 1]) / 2.0),
                         ids[:boundary], ids[boundary:])
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None or best[0] >= total_sse - 1e-12:
            return node
        _, node.feature, node.threshold, left, right = best
        node.left, node.right = self._grow(left, depth + 1), self._grow(right, depth + 1)
        return node

    def predict(self, x: np.ndarray) -> np.ndarray:
        rows = np.asarray(x, dtype=np.float64)
        result = np.empty((len(rows), 7), dtype=np.float64)
        for index, row in enumerate(rows):
            node = self.root
            while node.feature >= 0:
                assert node.left is not None and node.right is not None
                node = node.left if row[node.feature] <= node.threshold else node.right
            result[index] = node.value
        return result


def nearest_action_summary(train_states: np.ndarray, train_summary: np.ndarray,
                           query_states: np.ndarray, *, k: int = 1,
                           exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same-action nearest-neighbor summaries, using train-only standardized state distance."""
    train = np.asarray(train_states, dtype=np.float64)
    query = np.asarray(query_states, dtype=np.float64)
    target = np.asarray(train_summary, dtype=np.float64)
    if target.shape != (len(train), 9, 7) or train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ValueError("invalid D152 nearest-neighbor arrays")
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std < 1e-8] = 1.0
    tr, qu = (train - mean) / std, (query - mean) / std
    distances = np.sqrt(np.sum((qu[:, None, :] - tr[None, :, :]) ** 2, axis=2))
    if exclude_self:
        if len(query) != len(train):
            raise ValueError("self-exclusion requires matching query/train rows")
        distances[np.arange(len(query)), np.arange(len(train))] = np.inf
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    predictions = target[order[:, :, None], np.arange(9)[None, None, :], :].mean(axis=1)
    return predictions, distances[np.arange(len(query))[:, None], order], order


def expand_examples_prediction(prediction: np.ndarray, states_count: int) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != (states_count * 9, 7):
        raise ValueError("action-conditioned prediction shape mismatch")
    rows = values.reshape(states_count, 9, 7).copy()
    # A sigmoid head predicts each outcome bit independently; project to a valid
    # categorical distribution before scoring or downstream substitution.
    outcomes = np.maximum(rows[:, :, :4], 1e-12)
    rows[:, :, :4] = outcomes / outcomes.sum(axis=2, keepdims=True)
    return rows


def summary_component_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(target, dtype=np.float64).reshape(-1, 9, 7)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1, 9, 7)
    if truth.shape != pred.shape:
        raise ValueError("D152 component metrics require matching [N,9,7] arrays")
    true_class = np.argmax(truth[:, :, :4], axis=2).ravel()
    pred_class = np.argmax(pred[:, :, :4], axis=2).ravel()
    classes = {}
    for name, class_id in (("success", 0), ("active", 1), ("timeout", 2), ("collision", 3)):
        tp = int(np.sum((true_class == class_id) & (pred_class == class_id)))
        fp = int(np.sum((true_class != class_id) & (pred_class == class_id)))
        fn = int(np.sum((true_class == class_id) & (pred_class != class_id)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes[name] = {"precision": precision, "recall": recall, "f1": f1,
                         "support": int(np.sum(true_class == class_id))}
    metrics: dict[str, Any] = {"outcome_accuracy": float(np.mean(true_class == pred_class)),
                               "outcome_classification": classes}
    for field, index in (("terminal_steps", 4), ("terminal_goal_distance", 5), ("noncoast_effort", 6)):
        error = pred[:, :, index] - truth[:, :, index]
        metrics[f"{field}_mae"] = float(np.mean(np.abs(error)))
        metrics[f"{field}_rmse"] = float(np.sqrt(np.mean(error * error)))
    return metrics


def _best_set(values: np.ndarray, tolerance: float = 1e-12) -> set[int]:
    row = np.asarray(values, dtype=np.float64)
    return set(np.flatnonzero(row <= row.min() + tolerance).tolist())


def _pair_rank_disagreement(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left), np.asarray(right)
    i, j = np.triu_indices(9, 1)
    da, db = a[i] - a[j], b[i] - b[j]
    keep = (da != 0.0) & (db != 0.0)
    return float(np.mean(np.sign(da[keep]) != np.sign(db[keep]))) if np.any(keep) else 0.0


def _catastrophic_jaccard(left: np.ndarray, right: np.ndarray, threshold: float = 0.5) -> float:
    a, b = np.asarray(left) >= threshold, np.asarray(right) >= threshold
    union = np.logical_or(a, b).sum()
    return float(1.0 - np.logical_and(a, b).sum() / union) if union else 0.0


def neighbor_consequence_metrics(r4: np.ndarray, regrets: np.ndarray,
                                 query_indices: np.ndarray, neighbor_indices: np.ndarray) -> dict[str, float]:
    summary = np.asarray(r4, dtype=np.float64)[:, 61:124].reshape(-1, 9, 7)
    values = np.asarray(regrets, dtype=np.float64)
    records = []
    for q, n in zip(query_indices, neighbor_indices, strict=True):
        left, right = values[int(q)], values[int(n)]
        left_set, right_set = _best_set(left), _best_set(right)
        a, b = left >= 0.5, right >= 0.5
        union = np.logical_or(a, b).sum()
        records.append({
            "r4_summary_rmse": float(np.sqrt(np.mean((summary[int(q)] - summary[int(n)]) ** 2))),
            "teacher_regret_mean_absolute_difference": float(np.mean(np.abs(left - right))),
            "teacher_pairwise_ranking_disagreement": _pair_rank_disagreement(left, right),
            "best_action_set_disagreement": float(not bool(left_set & right_set)),
            "catastrophic_action_mask_jaccard_distance": float(1.0 - np.logical_and(a, b).sum() / union) if union else 0.0,
        })
    return {key: float(np.mean([row[key] for row in records])) if records else float("nan")
            for key in records[0]} if records else {}


def mean_rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("rank correlation requires matching vectors")
    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        result = np.empty(len(v), dtype=np.float64)
        start = 0
        while start < len(v):
            end = start + 1
            while end < len(v) and v[order[end]] == v[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return result
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def build_history_features(layouts: list[str], seeds: np.ndarray, decision_indices: np.ndarray,
                           states: np.ndarray, trace_rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Add only the prior observable kinematics/action for contiguous same-episode traces."""
    layout = np.asarray(layouts).astype(str)
    seed_array, decision = np.asarray(seeds), np.asarray(decision_indices)
    if not (len(layout) == len(seed_array) == len(decision) == len(states)):
        raise ValueError("history metadata does not align with state rows")
    traces: dict[tuple[str, int, int], dict[str, Any]] = {}
    for trace in trace_rows:
        key = (str(trace["layout_sha256"]), int(trace["episode_seed"]), int(trace["decision_index"]))
        if key in traces:
            raise ValueError("duplicate Phase14G trace key in D152 allowed layouts")
        traces[key] = trace
    index_by_key = {(layout[i], int(seed_array[i]), int(decision[i])): i for i in range(len(layout))}
    if len(index_by_key) != len(layout) or set(index_by_key) != set(traces):
        raise ValueError("train/validation Phase15 rows do not exactly align with filtered Phase14G traces")
    previous_kinematics = np.zeros((len(layout), 4), dtype=np.float64)
    previous_action = np.zeros((len(layout), 9), dtype=np.float64)
    mask = np.zeros((len(layout), 1), dtype=np.float64)
    for key, current_index in index_by_key.items():
        prior_key = (key[0], key[1], key[2] - 1)
        prior_index = index_by_key.get(prior_key)
        if prior_index is None:
            continue
        prior_trace = traces[prior_key]
        action = int(prior_trace["student_selected_action_id"])
        if not 0 <= action < 9 or not prior_trace["student_action_applied_matches_selected"]:
            raise ValueError("prior applied action violates the observable history contract")
        previous_kinematics[current_index] = states[prior_index, :4]
        previous_action[current_index, action] = 1.0
        mask[current_index, 0] = 1.0
    return {"R0_plus_previous_kinematics": np.concatenate((states, previous_kinematics, mask), axis=1),
            "R0_plus_previous_action": np.concatenate((states, previous_action, mask), axis=1),
            "R0_plus_history": np.concatenate((states, previous_kinematics, previous_action, mask), axis=1),
            "history_available_fraction": np.asarray([float(mask.mean())])}


def component_ablation(exact_r4: np.ndarray, predicted_summary: np.ndarray,
                       r0: np.ndarray, y: np.ndarray, layouts: np.ndarray, head: Any) -> dict[str, Any]:
    exact = np.asarray(exact_r4, dtype=np.float64).copy()
    predicted = np.asarray(predicted_summary, dtype=np.float64).reshape(-1, 9, 7)
    target = np.asarray(y, dtype=np.float64)
    groups = {"terminal_outcome": slice(0, 4), "terminal_steps": slice(4, 5),
              "terminal_goal_distance": slice(5, 6), "noncoast_effort": slice(6, 7)}
    result: dict[str, Any] = {}
    def score(matrix: np.ndarray) -> dict[str, Any]:
        return decision_metrics(target, head.predict(matrix), layouts)
    exact_metrics = score(exact)
    predicted_all = exact.copy()
    predicted_all[:, 61:124] = predicted.reshape(len(predicted), 63)
    result["exact_all_components"] = exact_metrics
    result["predicted_all_components"] = score(predicted_all)
    for name, field_slice in groups.items():
        # The D150 scorer input is [R0(61), flattened summaries(63)], not interleaved.
        forward_matrix = exact.copy()
        view = forward_matrix[:, 61:124].reshape(-1, 9, 7)
        view[:, :, field_slice] = predicted[:, :, field_slice]
        result[f"predict_{name}_keep_others_exact"] = score(forward_matrix)
        reverse = predicted_all.copy()
        reverse_view = reverse[:, 61:124].reshape(-1, 9, 7)
        reverse_view[:, :, field_slice] = exact[:, 61:124].reshape(-1, 9, 7)[:, :, field_slice]
        result[f"restore_exact_{name}_among_predicted"] = score(reverse)
    return result
