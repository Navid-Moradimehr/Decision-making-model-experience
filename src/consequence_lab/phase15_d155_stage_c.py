"""D155 Stage-C prediction screen for the selected outcome+steps subset."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_d155 import (
    D155Error, D155ListwiseMLP, _cfg as _stage_a_cfg, _load_data,
    _read_json, _softmax,
)
from consequence_lab.phase15_decision_aligned import decision_detail_metrics
from consequence_lab.phase15_learning import layout_weights

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d155_stage_c.yaml"
PROTOCOL = "reports/phase15_d155_stage_c_protocol.md"
LOCK = "experiments/phase15_d155_stage_c.lock.json"
SOURCE_FILES = (
    CONFIG, PROTOCOL, "reports/phase15_d155_stage_c_amendment_v1.md",
    "src/consequence_lab/phase15_d155_stage_c.py",
    "src/consequence_lab/phase15_d155.py",
    "src/consequence_lab/phase15_decision_aligned.py",
    "src/consequence_lab/phase15_learning.py",
    "scripts/freeze_phase15_d155_stage_c_lock.py",
    "scripts/run_phase15_d155_stage_c.py",
    "tests/test_phase15_d155_stage_c.py",
)
CLASS_COLUMNS = (0, 2, 3)  # success, timeout, collision; active is absent.


class D155StageCError(RuntimeError):
    """D155 Stage-C integrity or execution failure."""


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D155StageCError(f"git {' '.join(args)} failed") from exc


def _config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _parent_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}


def _audit_stage_a(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _read_json(ROOT / cfg["parents"]["stage_a_lock"])
    result = _read_json(ROOT / cfg["parents"]["stage_a_result"])
    manifest = _read_json(ROOT / cfg["parents"]["stage_a_manifest"])
    if lock.get("schema") != "phase15_d155_lock_v2":
        raise D155StageCError("Stage-C parent is not the corrected D155 Stage-A v2 lock")
    if (result.get("status") != "completed_exploratory_oracle_consequence_ablation"
            or result.get("stage_c_gate", {}).get("selected_compact_subset") != "F_outcome_steps"
            or result.get("stage_c_gate", {}).get("stage_c_condition_satisfied") is not True
            or result.get("stage_c_gate", {}).get("stage_c_executed") is not False):
        raise D155StageCError("Stage-A result does not authorize the frozen selected subset")
    if (manifest.get("result_sha256", "").upper() != sha256_file(ROOT / cfg["parents"]["stage_a_result"])
            or manifest.get("report_sha256", "").upper() != sha256_file(ROOT / "reports/phase15_d155_stage_a_v2_result.md")
            or manifest.get("lock_sha256", "").upper() != sha256_file(ROOT / cfg["parents"]["stage_a_lock"])
            or manifest.get("test_rows_dereferenced") != 0
            or manifest.get("reserve_access") is not False):
        raise D155StageCError("Stage-A manifest/hash/scope check failed")
    checkpoint_name = "F_outcome_steps.npz"
    checkpoint_hash = manifest.get("checkpoint_sha256", {}).get(checkpoint_name)
    if (not checkpoint_hash
            or checkpoint_hash.upper() != sha256_file(ROOT / cfg["parents"]["selected_decision_head"])):
        raise D155StageCError("selected Stage-A decision head hash differs from manifest")
    d154 = _read_json(ROOT / cfg["parents"]["d154_result"])
    refs = result["reference_metrics"]
    for reference_name in ("direct_D150_R0", "exact_R4_then_D150_head_oracle"):
        keys = ("layout_macro_normalized_selected_regret", "catastrophic_fraction_state_micro",
                "best_action_set_agreement")
        if any(abs(float(refs[reference_name]["metrics"][key]) -
                   float(d154["references"][reference_name]["metrics"][key])) > 1e-10 for key in keys):
            raise D155StageCError(f"Stage-A frozen reference mismatch: {reference_name}")
    return lock, result


def _d153_historical_component_metrics(path: Path) -> dict[str, Any]:
    """Read the saved D200 full-summary component metrics without refitting D153."""
    result = _read_json(path)
    if (result.get("status") != "completed_exploratory_scaling"
            or result.get("test_rows_indexed_or_scored") != 0
            or result.get("reserve_access") is not False
            or result.get("scope", {}).get("test_rows_indexed_or_scored") is not False
            or result.get("scope", {}).get("reserve_access") is not False):
        raise D155StageCError("D153 historical component reference scope/status is invalid")
    rows = [row for row in result.get("scale_results", []) if row.get("layout_count") == 200]
    if len(rows) != 1:
        raise D155StageCError("D153 historical D200 component row is missing or ambiguous")
    row = rows[0]
    components, summary = row["validation_summary_components"], row["validation_summary_metrics"]
    return {
        "source": "D153 D200 saved result; descriptive, not controlled against Stage-C models",
        "outcome_argmax_accuracy": float(components["outcome_accuracy"]),
        "normalized_terminal_steps_rmse": float(components["terminal_steps_rmse"]),
        "full_R4_summary_rmse": float(summary["rmse"]),
        "predictor_checkpoint_available": False,
        "prediction_tensor_available": False,
        "latency_available": False,
    }


def build_lock() -> dict[str, Any]:
    cfg = _config()
    if _git("status", "--porcelain"):
        raise D155StageCError("commit Stage-C protocol/source/tests before locking")
    if cfg.get("schema") != "phase15_d155_compact_consequence_prediction":
        raise D155StageCError("Stage-C config schema mismatch")
    scope = cfg["scope"]
    if scope.get("train_and_reused_validation_only") is not True or scope.get("selected_subset_only") is not True:
        raise D155StageCError("Stage-C required scope flags disabled")
    if any(value is not False for value in scope.values()
           if value not in (True,)):
        raise D155StageCError("Stage-C scope enables a forbidden operation")
    parent_lock, parent_result = _audit_stage_a(cfg)
    historical_component_reference = _d153_historical_component_metrics(ROOT / cfg["parents"]["d153_result"])
    expected_split = parent_lock["split_audit"]
    if (expected_split["d152_validation_rows"] != 202
            or expected_split["d152_validation_layouts"] != 20
            or expected_split["validation_layouts_sha256"] != cfg["split"]["validation_layouts_sha256"]):
        raise D155StageCError("Stage-C validation identity differs from D155 Stage A")
    for path in (cfg["outputs"]["lock"], cfg["outputs"]["run_dir"], cfg["outputs"]["report"]):
        if (ROOT / path).exists():
            raise D155StageCError(f"refusing to overwrite Stage-C output: {path}")
    return {
        "schema": "phase15_d155_stage_c_lock_v1",
        "status": "locked_train_reused_validation_prediction_screen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "config_sha256": sha256_file(ROOT / CONFIG),
        "parent_sha256": _parent_hashes(cfg),
        "selected_subset": parent_result["stage_c_gate"]["selected_compact_subset"],
        "feature_sets": ["F_outcome_steps_only"],
        "split": cfg["split"], "target": cfg["target"], "predictors": cfg["predictors"],
        "historical_full_r4_prediction_components": historical_component_reference,
        "decision_gate": cfg["decision_gate"], "metrics": cfg["metrics"],
        "scope": cfg["scope"], "outputs": cfg["outputs"],
        "validation_reuse": "D152 validation reused in D152-D155; every Stage-C validation estimate exploratory",
        "test_or_reserve_used": False,
    }


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _config()
    lock = _read_json(ROOT / cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d155_stage_c_lock_v1":
        raise D155StageCError("Stage-C lock schema mismatch")
    if (lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("source_sha256") != {path: sha256_file(ROOT / path) for path in SOURCE_FILES}
            or lock.get("parent_sha256") != _parent_hashes(cfg)
            or lock.get("scope") != cfg["scope"]
            or lock.get("test_or_reserve_used") is not False):
        raise D155StageCError("Stage-C source/config/parent/scope lock mismatch")
    if lock.get("historical_full_r4_prediction_components") != _d153_historical_component_metrics(
            ROOT / cfg["parents"]["d153_result"]):
        raise D155StageCError("Stage-C historical D153 component reference mismatch")
    if _git("status", "--porcelain"):
        raise D155StageCError("Stage-C run requires a clean committed tree")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    _audit_stage_a(cfg)
    for path in (cfg["outputs"]["run_dir"], cfg["outputs"]["report"]):
        if (ROOT / path).exists():
            raise D155StageCError(f"refusing second Stage-C execution: {path}")
    return cfg, lock


def action_conditioned_inputs(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 61 or not np.isfinite(values).all():
        raise ValueError("Stage-C state batch must be finite [N,61]")
    action = np.eye(9, dtype=np.float64)
    tiled_state = np.repeat(values[:, None, :], 9, axis=1)
    tiled_action = np.repeat(action[None, :, :], len(values), axis=0)
    return np.concatenate((tiled_state, tiled_action), axis=2).reshape(-1, 70)


def encode_targets(summaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(summaries, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (9, 7) or not np.isfinite(values).all():
        raise ValueError("Stage-C summaries must be finite [N,9,7]")
    outcome = values[:, :, :4]
    if (np.any(np.abs(outcome[:, :, 1]) > 1e-12)
            or not np.allclose(outcome.sum(axis=2), 1.0, atol=1e-12, rtol=0.0)
            or np.any((outcome < 0.0) | (outcome > 1.0))):
        raise ValueError("Stage-C outcomes must be terminal one-hot with active=0")
    classes = np.argmax(outcome[:, :, CLASS_COLUMNS], axis=2).reshape(-1)
    steps = values[:, :, 4].reshape(-1)
    if np.any((steps < 0.0) | (steps > 1.0)):
        raise ValueError("Stage-C normalized steps must lie in [0,1]")
    return classes.astype(np.int64), steps


class PerActionConstant:
    def fit(self, classes: np.ndarray, steps: np.ndarray, layouts: np.ndarray) -> "PerActionConstant":
        labels = np.asarray(layouts).astype(str)
        n = len(labels)
        if len(classes) != 9 * n or len(steps) != 9 * n:
            raise ValueError("constant predictor targets must be flattened state-action arrays")
        base_weight = layout_weights(labels)
        weights = np.repeat(base_weight, 9)
        self.probabilities = np.zeros((9, 3), dtype=np.float64)
        self.mean_steps = np.zeros(9, dtype=np.float64)
        for action_id in range(9):
            ids = np.arange(n) * 9 + action_id
            for klass in range(3):
                self.probabilities[action_id, klass] = weights[ids][np.asarray(classes)[ids] == klass].sum()
            self.probabilities[action_id] /= self.probabilities[action_id].sum()
            self.mean_steps[action_id] = np.average(np.asarray(steps)[ids], weights=weights[ids])
        return self

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(features, dtype=np.float64)
        action_ids = np.argmax(rows[:, 61:], axis=1)
        return self.probabilities[action_ids], self.mean_steps[action_ids]

    @property
    def parameter_count(self) -> int:
        return int(self.probabilities.size + self.mean_steps.size)


class _TreeNode:
    def __init__(self, value: np.ndarray, feature: int = -1, threshold: float = 0.0,
                 left: "_TreeNode | None" = None, right: "_TreeNode | None" = None) -> None:
        self.value, self.feature, self.threshold = value, feature, threshold
        self.left, self.right = left, right


class BoundedCART:
    """Small weighted CART; one-hot variance equals Gini for outcome classes."""

    def __init__(self, task: str, classes: int = 3, max_depth: int = 4,
                 min_samples_leaf: int = 64, quantiles: tuple[float, ...] =
                 (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("unknown bounded CART task")
        self.task, self.classes = task, classes
        self.max_depth, self.min_samples_leaf, self.quantiles = max_depth, min_samples_leaf, quantiles

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> "BoundedCART":
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int64 if self.task == "classification" else np.float64)
        self.weights = np.asarray(sample_weights, dtype=np.float64)
        if (self.x.ndim != 2 or len(self.y) != len(self.x) or self.weights.shape != (len(self.x),)
                or not np.isfinite(self.x).all() or not np.isfinite(self.y).all()
                or not np.isfinite(self.weights).all() or np.any(self.weights < 0.0)):
            raise ValueError("invalid bounded CART training arrays")
        self.node_count = 0
        self.root = self._grow(np.arange(len(self.x), dtype=np.int64), 0)
        return self

    def _value_impurity(self, ids: np.ndarray) -> tuple[np.ndarray, float]:
        weights = self.weights[ids]
        total = weights.sum()
        if total <= 0.0:
            raise ValueError("CART node has no positive sample weight")
        if self.task == "classification":
            counts = np.bincount(self.y[ids], weights=weights, minlength=self.classes).astype(float)
            value = counts / total
            impurity = total - float(np.dot(counts, counts) / total)
        else:
            value = np.asarray([np.average(self.y[ids], weights=weights)])
            impurity = float(np.dot(weights, (self.y[ids] - value[0]) ** 2))
        return value, impurity

    def _grow(self, ids: np.ndarray, depth: int) -> _TreeNode:
        value, parent_impurity = self._value_impurity(ids)
        self.node_count += 1
        node = _TreeNode(value)
        if depth >= self.max_depth or len(ids) < 2 * self.min_samples_leaf or parent_impurity <= 1e-12:
            return node
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        for feature in range(self.x.shape[1]):
            column = self.x[ids, feature]
            thresholds = np.unique(np.quantile(column, self.quantiles))
            for threshold in thresholds:
                left_mask = column <= threshold
                left, right = ids[left_mask], ids[~left_mask]
                if min(len(left), len(right)) < self.min_samples_leaf:
                    continue
                _, left_impurity = self._value_impurity(left)
                _, right_impurity = self._value_impurity(right)
                loss = left_impurity + right_impurity
                candidate = (loss, feature, float(threshold), left, right)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None or best[0] >= parent_impurity - 1e-12:
            return node
        _, node.feature, node.threshold, left_ids, right_ids = best
        node.left = self._grow(left_ids, depth + 1)
        node.right = self._grow(right_ids, depth + 1)
        return node

    def _predict_one(self, row: np.ndarray) -> np.ndarray:
        node = self.root
        while node.feature >= 0:
            assert node.left is not None and node.right is not None
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, x: np.ndarray) -> np.ndarray:
        rows = np.asarray(x, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.x.shape[1] or not np.isfinite(rows).all():
            raise ValueError("invalid bounded CART prediction matrix")
        return np.stack([self._predict_one(row) for row in rows])


class CARTOutcomeSteps:
    """Paired bounded trees for outcome probabilities and step regression."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings

    def fit(self, x: np.ndarray, classes: np.ndarray, steps: np.ndarray,
            sample_weights: np.ndarray) -> "CARTOutcomeSteps":
        values = np.asarray(x, dtype=np.float64)
        self.x_mean, self.x_std = values.mean(axis=0), values.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        train_x = (values - self.x_mean) / self.x_std
        args = {"max_depth": int(self.settings["max_depth"]),
                "min_samples_leaf": int(self.settings["min_samples_leaf"]),
                "quantiles": tuple(self.settings["candidate_quantiles"])}
        self.class_tree = BoundedCART("classification", **args).fit(train_x, classes, sample_weights)
        self.steps_tree = BoundedCART("regression", **args).fit(train_x, steps, sample_weights)
        self.node_count = self.class_tree.node_count + self.steps_tree.node_count
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        probabilities = self.class_tree.predict(rows)
        predicted_steps = np.clip(self.steps_tree.predict(rows)[:, 0], 0.0, 1.0)
        return probabilities, predicted_steps


class OutcomeStepsMLP:
    def __init__(self, config: dict[str, Any]) -> None:
        self.settings = config["predictors"]["mlp"]

    def _forward(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0.0)
        return h2 @ self.w3 + self.b3, (x, h1, h2), (z1, z2)

    def fit(self, x: np.ndarray, classes: np.ndarray, steps: np.ndarray, weights: np.ndarray,
            validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> "OutcomeStepsMLP":
        x = np.asarray(x, dtype=np.float64)
        classes = np.asarray(classes, dtype=np.int64)
        steps = np.asarray(steps, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        xv, yv_class, yv_steps, wv = validation
        xv, yv_steps, wv = np.asarray(xv, float), np.asarray(yv_steps, float), np.asarray(wv, float)
        yv_class = np.asarray(yv_class, dtype=np.int64)
        if (x.shape != (len(classes), 70) or steps.shape != classes.shape or weights.shape != classes.shape
                or xv.shape != (len(yv_class), 70) or yv_steps.shape != yv_class.shape
                or wv.shape != yv_class.shape or not all(np.isfinite(a).all() for a in (x, steps, weights, xv, yv_steps, wv))):
            raise ValueError("invalid Stage-C MLP fit arrays")
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        xt, xval = (x - self.x_mean) / self.x_std, (xv - self.x_mean) / self.x_std
        self.step_mean = float(np.average(steps, weights=weights))
        self.step_std = float(np.sqrt(np.average((steps - self.step_mean) ** 2, weights=weights)))
        if self.step_std < 1e-8:
            self.step_std = 1.0
        ystep, ystep_val = (steps - self.step_mean) / self.step_std, (yv_steps - self.step_mean) / self.step_std
        onehot = np.eye(3)[classes]
        rng = np.random.default_rng(int(self.settings["seed"]))
        d1, d2 = map(int, self.settings["hidden"])
        self.w1 = rng.normal(0.0, np.sqrt(2 / 70), (70, d1))
        self.w2 = rng.normal(0.0, np.sqrt(2 / d1), (d1, d2))
        self.w3 = rng.normal(0.0, np.sqrt(1 / d2), (d2, 4))
        self.b1, self.b2, self.b3 = np.zeros(d1), np.zeros(d2), np.zeros(4)
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        best_params = None
        best_loss = float("inf")
        stale = 0
        for epoch in range(int(self.settings["max_epochs"])):
            output, activations, pre = self._forward(xt)
            probs = _softmax(output[:, :3])
            delta = np.empty_like(output)
            delta[:, :3] = float(self.settings["outcome_loss_weight"]) * weights[:, None] * (probs - onehot)
            delta[:, 3] = (float(self.settings["steps_loss_weight"]) * 2.0 * weights * (output[:, 3] - ystep))
            x0, h1, h2 = activations
            z1, z2 = pre
            grads = [None] * 6
            grads[4] = h2.T @ delta + float(self.settings["l2"]) * self.w3
            grads[5] = delta.sum(axis=0)
            dz2 = (delta @ self.w3.T) * (z2 > 0.0)
            grads[2] = h1.T @ dz2 + float(self.settings["l2"]) * self.w2
            grads[3] = dz2.sum(axis=0)
            dz1 = (dz2 @ self.w2.T) * (z1 > 0.0)
            grads[0] = x0.T @ dz1 + float(self.settings["l2"]) * self.w1
            grads[1] = dz1.sum(axis=0)
            step = epoch + 1
            for index, (parameter, gradient) in enumerate(zip(params, grads, strict=True)):
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
                parameter -= float(self.settings["learning_rate"]) * (first[index] / (1.0 - 0.9**step)) / (
                    np.sqrt(second[index] / (1.0 - 0.999**step)) + 1e-8)
            val = self._forward(xval)[0]
            val_prob = _softmax(val[:, :3])
            val_ce = -float(np.sum(wv * np.log(np.maximum(val_prob[np.arange(len(wv)), yv_class], 1e-12))))
            val_mse = float(np.sum(wv * (val[:, 3] - ystep_val) ** 2))
            criterion = (float(self.settings["outcome_loss_weight"]) * val_ce
                         + float(self.settings["steps_loss_weight"]) * val_mse)
            if criterion < best_loss - 1e-9:
                best_loss, stale = criterion, 0
                best_params = [p.copy() for p in params]
                self.best_epoch = epoch + 1
            else:
                stale += 1
            if stale >= int(self.settings["patience"]):
                break
        if best_params is None:
            raise RuntimeError("Stage-C MLP failed to produce a checkpoint")
        for name, value in zip(("w1", "b1", "w2", "b2", "w3", "b3"), best_params, strict=True):
            setattr(self, name, value)
        self.epochs_run = epoch + 1
        self.validation_loss = best_loss
        self.parameter_count = int(sum(p.size for p in params))
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        output = self._forward((np.asarray(x, dtype=float) - self.x_mean) / self.x_std)[0]
        probs = _softmax(output[:, :3])
        steps = np.clip(output[:, 3] * self.step_std + self.step_mean, 0.0, 1.0)
        return probs, steps


def _weights_per_action(layouts: np.ndarray) -> np.ndarray:
    return np.repeat(layout_weights(np.asarray(layouts)), 9) / 9.0


def _decision_features(states: np.ndarray, prob: np.ndarray, steps: np.ndarray) -> np.ndarray:
    n = len(states)
    outcome = np.zeros((n, 9, 4), dtype=np.float64)
    outcome[:, :, 0] = prob[:, :, 0]
    outcome[:, :, 2] = prob[:, :, 1]
    outcome[:, :, 3] = prob[:, :, 2]
    result = np.concatenate((states, outcome.reshape(n, 36), np.asarray(steps).reshape(n, 9)), axis=1)
    if result.shape != (n, 106) or not np.isfinite(result).all():
        raise ValueError("Stage-C predicted consequence-to-head feature contract failed")
    return result


def _load_stage_a_head(path: Path) -> D155ListwiseMLP:
    model = D155ListwiseMLP(106, _stage_a_cfg())
    with np.load(path, allow_pickle=False) as arrays:
        required = ("x_mean", "x_std", "w1", "b1", "w2", "b2", "w3", "b3")
        if set(required) - set(arrays.files):
            raise D155StageCError("selected Stage-A head checkpoint arrays missing")
        for name in required:
            setattr(model, name, np.asarray(arrays[name], dtype=np.float64))
    if (model.x_mean.shape != (106,) or model.x_std.shape != (106,)
            or model.w1.shape != (106, 32) or model.w2.shape != (32, 32)
            or model.w3.shape != (32, 9) or not all(np.isfinite(getattr(model, n)).all() for n in required)):
        raise D155StageCError("selected Stage-A head dimensions/data invalid")
    model.parameter_count = int(sum(getattr(model, name).size for name in ("w1", "b1", "w2", "b2", "w3", "b3")))
    return model


def _outcome_metrics(target_class: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target_class, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    onehot = np.eye(3)[target]
    output: dict[str, Any] = {
        "accuracy": float(np.sum(weights * (np.argmax(p, axis=1) == target))),
        "cross_entropy": float(-np.sum(weights * np.log(np.maximum(p[np.arange(len(p)), target], 1e-12)))),
        "brier_score": float(np.sum(weights * np.sum((p - onehot) ** 2, axis=1))),
        "per_class": {},
    }
    prediction = np.argmax(p, axis=1)
    for klass, name in enumerate(("success", "timeout", "collision")):
        class_weight = weights * (target == klass)
        mass = float(class_weight.sum())
        if mass <= 0.0:
            output["per_class"][name] = {"support_weight": 0.0, "recall": None,
                                         "conditional_cross_entropy": None,
                                         "one_vs_rest_brier": None}
            continue
        output["per_class"][name] = {
            "support_weight": mass,
            "recall": float(np.sum(class_weight * (prediction == klass)) / mass),
            "conditional_cross_entropy": float(-np.sum(
                class_weight * np.log(np.maximum(p[:, klass], 1e-12))) / mass),
            "one_vs_rest_brier": float(np.sum(weights * (p[:, klass] - onehot[:, klass]) ** 2)),
        }
    return output


def _eval_model(name: str, predictor: Any, data: dict[str, Any], cfg: dict[str, Any],
                stage_a_head: D155ListwiseMLP) -> dict[str, Any]:
    states = data["x_val"]
    exact = data["r4_val"][:, 61:].reshape(-1, 9, 7)
    target_class, target_steps = encode_targets(exact)
    flat_x = action_conditioned_inputs(states)
    prob_flat, steps_flat = predictor.predict(flat_x)
    prob = np.asarray(prob_flat).reshape(-1, 9, 3)
    prob = np.maximum(prob, 1e-12)
    prob /= prob.sum(axis=2, keepdims=True)
    pred_steps = np.clip(np.asarray(steps_flat).reshape(-1, 9), 0.0, 1.0)
    sample_weights = _weights_per_action(data["val_layouts"])
    out_metric = _outcome_metrics(target_class, prob.reshape(-1, 3), sample_weights)
    steps_flat_true = target_steps
    steps_flat_pred = pred_steps.reshape(-1)
    steps_metric = {
        "rmse": float(np.sqrt(np.sum(sample_weights * (steps_flat_pred - steps_flat_true) ** 2))),
        "mae": float(np.sum(sample_weights * np.abs(steps_flat_pred - steps_flat_true))),
    }
    decision_x = _decision_features(states, prob, pred_steps)
    logits = stage_a_head.predict_logits(decision_x)
    decision = decision_detail_metrics(data["y_val"], -logits, data["val_layouts"], cfg["metrics"])
    latency = []
    for row in states:
        rows = action_conditioned_inputs(row[None, :])
        begin = time.perf_counter_ns()
        row_prob, row_steps = predictor.predict(rows)
        row_prob = np.asarray(row_prob).reshape(1, 9, 3)
        row_steps = np.asarray(row_steps).reshape(1, 9)
        head_input = _decision_features(row[None, :], row_prob, row_steps)
        stage_a_head.predict_logits(head_input)
        latency.append((time.perf_counter_ns() - begin) / 1e9)
    return {"outcome": out_metric, "steps": steps_metric, "downstream": {
        key: value for key, value in decision.items()
        if key not in {"selected_regret", "per_layout_mean_regret"}},
        "median_end_to_end_latency_seconds_per_state": float(np.median(latency))}


def run_stage_c() -> dict[str, Any]:
    cfg, lock = validate_lock()
    started = time.perf_counter()
    data = _load_data(_stage_a_cfg())
    train_summary = data["r4_train"][:, 61:].reshape(-1, 9, 7)
    val_summary = data["r4_val"][:, 61:].reshape(-1, 9, 7)
    train_class, train_steps = encode_targets(train_summary)
    val_class, val_steps = encode_targets(val_summary)
    train_x, val_x = action_conditioned_inputs(data["x_train"]), action_conditioned_inputs(data["x_val"])
    stage_a_head = _load_stage_a_head(ROOT / cfg["parents"]["selected_decision_head"])
    train_layouts_flat = np.repeat(data["train_layouts"], 9)
    val_layouts_flat = np.repeat(data["val_layouts"], 9)
    train_weights, val_weights = _weights_per_action(data["train_layouts"]), _weights_per_action(data["val_layouts"])
    output = ROOT / cfg["outputs"]["run_dir"]
    output.mkdir(parents=True, exist_ok=False)
    # Constant outcome frequencies and step means are calculated independently per action.
    fit_start = time.perf_counter()
    constant = PerActionConstant().fit(train_class, train_steps, data["train_layouts"])
    constant_seconds = time.perf_counter() - fit_start
    # Bounded CART predicts each target with an independently impurity-matched tree.
    fit_start = time.perf_counter()
    tree = CARTOutcomeSteps(cfg["predictors"]["bounded_cart"]).fit(
        train_x, train_class, train_steps, train_weights)
    tree_seconds = time.perf_counter() - fit_start
    # The shared MLP has a categorical outcome head and a separately standardized scalar head.
    fit_start = time.perf_counter()
    mlp = OutcomeStepsMLP(cfg).fit(
        train_x, train_class, train_steps, train_weights,
        (val_x, val_class, val_steps, val_weights))
    mlp_seconds = time.perf_counter() - fit_start
    predictor_records = {
        "per_action_constant": {"predictor": constant, "parameter_count": constant.parameter_count,
                                "fit_seconds": constant_seconds},
        "bounded_cart": {"predictor": tree, "parameter_count": tree.node_count,
                         "node_count": tree.node_count, "fit_seconds": tree_seconds},
        "action_conditioned_mlp": {"predictor": mlp, "parameter_count": mlp.parameter_count,
                                   "best_epoch": mlp.best_epoch, "epochs_run": mlp.epochs_run,
                                   "validation_joint_loss": mlp.validation_loss,
                                   "fit_seconds": mlp_seconds},
    }
    predictor_checkpoint_hashes = _save_predictor_checkpoints(output, constant, tree, mlp)
    results = {}
    for name, record in predictor_records.items():
        evaluated = _eval_model(name, record["predictor"], data, cfg, stage_a_head)
        if name == "bounded_cart":
            evaluated["node_count"] = record["node_count"]
        evaluated["parameter_count"] = record["parameter_count"]
        evaluated["fit_seconds"] = record["fit_seconds"]
        if name == "action_conditioned_mlp":
            evaluated["best_epoch"] = record["best_epoch"]
            evaluated["epochs_run"] = record["epochs_run"]
            evaluated["validation_joint_loss"] = record["validation_joint_loss"]
        checkpoint_names = {"per_action_constant": "per_action_constant.npz",
                            "bounded_cart": "bounded_cart.json",
                            "action_conditioned_mlp": "action_conditioned_mlp.npz"}
        evaluated["checkpoint_sha256"] = predictor_checkpoint_hashes[checkpoint_names[name]]
        results[name] = evaluated
    stage_a = _read_json(ROOT / cfg["parents"]["stage_a_result"])
    d153_component_reference = _d153_historical_component_metrics(ROOT / cfg["parents"]["d153_result"])
    reference = stage_a["reference_metrics"]
    r0 = reference["direct_D150_R0"]["metrics"]
    exact_r4 = reference["exact_R4_then_D150_head_oracle"]["metrics"]
    gap = (float(r0["layout_macro_normalized_selected_regret"])
           - float(exact_r4["layout_macro_normalized_selected_regret"]))
    cat_limit = float(r0["catastrophic_fraction_state_micro"]) - float(
        cfg["decision_gate"]["minimum_catastrophic_fraction_absolute_improvement_vs_frozen_r0"])
    for result in results.values():
        regret = result["downstream"]["layout_macro_normalized_selected_regret"]
        cat = result["downstream"]["catastrophic_fraction_state_micro"]
        result["recovered_fraction_of_frozen_oracle_gap"] = float(
            (float(r0["layout_macro_normalized_selected_regret"]) - regret) / gap)
        result["catastrophic_absolute_improvement_vs_frozen_r0"] = float(
            float(r0["catastrophic_fraction_state_micro"]) - cat)
        result["passes_preregistered_offline_gate"] = bool(
            result["recovered_fraction_of_frozen_oracle_gap"] >= cfg["decision_gate"]["minimum_oracle_regret_gap_recovered"]
            and result["catastrophic_absolute_improvement_vs_frozen_r0"] >=
            cfg["decision_gate"]["minimum_catastrophic_fraction_absolute_improvement_vs_frozen_r0"])
    result = {
        "schema": "phase15_d155_stage_c_result_v1",
        "status": "completed_exploratory_compact_consequence_prediction_screen",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
        "parent_sha256": lock["parent_sha256"], "selected_subset": "F_outcome_steps",
        "target_contract": {"train_state_action_examples": len(train_x),
                            "validation_state_action_examples": len(val_x),
                            "outcome_class_order": ["success", "timeout", "collision"],
                            "active_outcomes": 0,
                            "training_layouts": int(len(set(train_layouts_flat.tolist()))),
                            "validation_layouts": int(len(set(val_layouts_flat.tolist()))),
                            "train_target_sha256": _hash_arrays(train_class, train_steps),
                            "validation_target_sha256": _hash_arrays(val_class, val_steps),
                            "test_rows_dereferenced": 0},
        "references": {name: row["metrics"] for name, row in reference.items()},
        "historical_full_r4_prediction_components": d153_component_reference,
        "reference_gap": {"macro_regret_r0_to_exact_r4": gap,
                          "catastrophic_fraction_r0": float(r0["catastrophic_fraction_state_micro"]),
                          "catastrophic_gate_maximum": cat_limit,
                          "macro_regret_gate_maximum": float(r0["layout_macro_normalized_selected_regret"])
                          - float(cfg["decision_gate"]["minimum_oracle_regret_gap_recovered"]) * gap},
        "predictors": results,
        "offline_gate_passed": any(row["passes_preregistered_offline_gate"] for row in results.values()),
        "scope": {"simulator_calls": 0, "planner_calls": 0, "new_transitions": 0, "new_episodes": 0,
                  "test_rows_dereferenced": 0, "reserve_access": False, "closed_loop": False,
                  "student_controlled_simulation": False, "dagger": False, "world_model": False, "jepa": False},
        "wall_seconds": time.perf_counter() - started,
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(result)
    report_path = ROOT / cfg["outputs"]["report"]
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "schema": "phase15_d155_stage_c_manifest_v1", "status": result["status"],
        "lock_sha256": result["lock_sha256"], "source_commit": result["source_commit"],
        "parent_sha256": lock["parent_sha256"], "result_sha256": sha256_file(result_path),
        "report_sha256": sha256_file(report_path),
        "predictor_checkpoint_sha256": predictor_checkpoint_hashes,
        "test_rows_dereferenced": 0, "reserve_access": False, "simulator_calls": 0,
        "planner_calls": 0, "new_transitions": 0, "new_episodes": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _hash_arrays(*arrays: np.ndarray) -> str:
    import hashlib
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest().upper()


def _serialize_tree(node: _TreeNode) -> dict[str, Any]:
    record: dict[str, Any] = {"value": np.asarray(node.value, dtype=float).tolist(),
                              "feature": int(node.feature), "threshold": float(node.threshold)}
    if node.feature >= 0:
        assert node.left is not None and node.right is not None
        record["left"], record["right"] = _serialize_tree(node.left), _serialize_tree(node.right)
    return record


def _save_predictor_checkpoints(output: Path, constant: PerActionConstant,
                                tree: CARTOutcomeSteps, mlp: OutcomeStepsMLP) -> dict[str, str]:
    constant_path = output / "per_action_constant.npz"
    np.savez_compressed(constant_path, probabilities=constant.probabilities, mean_steps=constant.mean_steps)
    tree_path = output / "bounded_cart.json"
    tree_record = {"schema": "phase15_d155_cart_pair_v1",
                   "class_tree": _serialize_tree(tree.class_tree.root),
                   "steps_tree": _serialize_tree(tree.steps_tree.root),
                   "x_mean": tree.x_mean.tolist(), "x_std": tree.x_std.tolist(),
                   "node_count": tree.node_count}
    tree_path.write_text(json.dumps(tree_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mlp_path = output / "action_conditioned_mlp.npz"
    np.savez_compressed(mlp_path, **{name: getattr(mlp, name) for name in
                                    ("x_mean", "x_std", "step_mean", "step_std", "w1", "b1", "w2", "b2", "w3", "b3")})
    return {path.name: sha256_file(path) for path in (constant_path, tree_path, mlp_path)}


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D155 Stage C — Compact outcome-and-steps prediction", "",
             "**Exploratory offline screen only.** Fixed validation has been reused; reserve/test remain sealed. No closed-loop, DAgger, simulator, planner, world model, or JEPA work occurred.", "",
             f"Selected subset: `{result['selected_subset']}`. State-action examples: {result['target_contract']['train_state_action_examples']} train / {result['target_contract']['validation_state_action_examples']} reused-validation; active outcomes observed: {result['target_contract']['active_outcomes']}.", "",
             f"Frozen macro-regret gate maximum: `{result['reference_gap']['macro_regret_gate_maximum']:.5f}` (>=25% R0-to-R4 gap recovered); catastrophic maximum: `{result['reference_gap']['catastrophic_gate_maximum']:.5f}` (>=.10 absolute gain vs D150 R0).", "",
             "| Predictor | Outcome accuracy | Outcome CE | Brier | Steps RMSE | Steps MAE | Downstream macro regret | Catastrophic | Recovered oracle gap | Cat reduction vs R0 | Pass | Params/nodes | Fit seconds | ms/state end-to-end |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|"]
    for name, row in result["predictors"].items():
        d = row["downstream"]
        lines.append(f"| {name} | {row['outcome']['accuracy']:.4f} | {row['outcome']['cross_entropy']:.4f} | {row['outcome']['brier_score']:.4f} | {row['steps']['rmse']:.4f} | {row['steps']['mae']:.4f} | {d['layout_macro_normalized_selected_regret']:.5f} | {d['catastrophic_fraction_state_micro']:.4f} | {row['recovered_fraction_of_frozen_oracle_gap']:.3f} | {row['catastrophic_absolute_improvement_vs_frozen_r0']:.3f} | {row['passes_preregistered_offline_gate']} | {row.get('parameter_count', row.get('node_count'))} | {row['fit_seconds']:.3f} | {1000*row['median_end_to_end_latency_seconds_per_state']:.4f} |")
    lines.extend(["", "## Per-class outcome diagnostics", "",
                  "| Predictor | Class | Recall | Conditional CE | One-vs-rest Brier | Weighted support |",
                  "|---|---|---:|---:|---:|---:|"])
    for name, row in result["predictors"].items():
        for klass, metric in row["outcome"]["per_class"].items():
            display = lambda value: "n/a" if value is None else f"{value:.4f}"
            lines.append(f"| {name} | {klass} | {display(metric['recall'])} | {display(metric['conditional_cross_entropy'])} | {display(metric['one_vs_rest_brier'])} | {metric['support_weight']:.4f} |")
    historical = result["historical_full_r4_prediction_components"]
    lines.extend(["", "## Historical full-R4 prediction comparison", "",
                  f"The D153 D200 joint full-R4 predictor recorded outcome argmax accuracy `{historical['outcome_argmax_accuracy']:.4f}`, normalized-steps RMSE `{historical['normalized_terminal_steps_rmse']:.4f}`, and aggregate full-summary RMSE `{historical['full_R4_summary_rmse']:.4f}`. D153 used a different joint 63-output predictor/training protocol; the prediction tensor/checkpoint is absent, so this is descriptive, not a controlled comparison.", "",
                  f"Overall preregistered offline gate: **{result['offline_gate_passed']}**. This does not authorize a closed-loop or DAgger phase.", "",
                  "Metrics are descriptive on the reused validation block. The carried D153 predicted-R4 decision anchor also lacks a saved prediction tensor/checkpoint, so its original decision metrics are cited from D154 and end-to-end latency is unavailable.", ""])
    return "\n".join(lines)
