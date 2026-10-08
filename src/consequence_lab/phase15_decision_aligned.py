"""D154 offline decision-target comparison on frozen Phase15 artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
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
from consequence_lab.phase15_data import layout_split
from consequence_lab.phase15_generalization import layout_list_sha256
from consequence_lab.phase15_learning import TinyRegretMLP, decision_metrics, layout_weights
from consequence_lab.phase15_outcome_summary import restore_decision_head
from consequence_lab.phase15_representation_screen import train_validation_indices
from consequence_lab.phase15_representation_sufficiency import component_ablation


ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d154_decision_aligned_v2.yaml"
PROTOCOL = "reports/phase15_d154_decision_aligned_target_protocol.md"
AMENDMENT = "reports/phase15_d154_runner_amendment_v2.md"
LOCK = "experiments/phase15_d154_decision_aligned_v2.lock.json"
SOURCE_FILES = (
    CONFIG, PROTOCOL, AMENDMENT, "src/consequence_lab/phase15_decision_aligned.py",
    "scripts/freeze_phase15_d154_lock.py", "scripts/run_phase15_d154.py",
    "tests/test_phase15_decision_aligned.py",
)
STATUS_TO_TUPLE = {0: 0, 2: 1, 3: 2}  # success, timeout, collision; active is invalid terminal data


class D154Error(RuntimeError):
    """D154 integrity or execution contract failed."""


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D154Error(f"git {' '.join(args)} failed") from exc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_d153_parent_manifest(manifest: dict[str, Any]) -> None:
    """Validate the completed D153 manifest's actual test/reserve field contract."""
    if (manifest.get("status") != "completed_exploratory_scaling"
            or manifest.get("test_rows_indexed_or_scored") != 0
            or manifest.get("reserve_access") is not False
            or manifest.get("scope", {}).get("test_rows_indexed_or_scored") is not False
            or manifest.get("scope", {}).get("reserve_access") is not False):
        raise D154Error("D153 parent manifest status or scope is invalid")


def sha256_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    return hashlib.sha256(values.dtype.str.encode("ascii") + str(values.shape).encode("ascii")
                          + values.tobytes()).hexdigest().upper()


def regrets_from_exact_summaries(summary: np.ndarray) -> np.ndarray:
    """Recreate Phase14 average-rank regret from exact [state, action, 7] R4 fields."""
    values = np.asarray(summary, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (9, 7) or not np.isfinite(values).all():
        raise ValueError("D154 exact summary targets must have shape [N,9,7] and be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("D154 exact summary values must be normalized to [0,1]")
    outcomes = values[:, :, :4]
    if (not np.all((outcomes == 0.0) | (outcomes == 1.0))
            or not np.all(outcomes.sum(axis=2) == 1.0)):
        raise ValueError("D154 requires exact one-hot terminal outcome fields")
    status = np.argmax(outcomes, axis=2)
    if np.any(status == 1):
        raise ValueError("active outcome cannot be a terminal forced-action continuation")
    target = np.empty((len(values), 9), dtype=np.float64)
    for row in range(len(values)):
        objectives = []
        for action in range(9):
            outcome = int(status[row, action])
            # Within outcome classes, multiplying steps/distance by the same
            # positive task constant preserves the exact Phase14 ordering.
            primary = float(values[row, action, 4] if outcome == 0 else values[row, action, 5])
            effort = float(values[row, action, 6])
            objectives.append((STATUS_TO_TUPLE[outcome], primary, effort))
        ranks = np.asarray([
            sum(other < value for other in objectives)
            + (sum(other == value for other in objectives) + 1.0) / 2.0
            for value in objectives
        ], dtype=np.float64)
        target[row] = (ranks - ranks.min()) / 8.0
    return target


def _target_parity_audit(data_path: Path, corpus_path: Path,
                         split_config: dict[str, Any]) -> dict[str, Any]:
    with np.load(data_path, allow_pickle=False) as data:
        all_layouts = data["layouts"].astype(str).tolist()
        train_ids, val_ids, train_layouts, val_layouts = train_validation_indices(all_layouts, split_config)
        # Never create test indices or read test feature/target rows.
        train_summary = data["x_R4"][train_ids, 61:124].astype(np.float64).reshape(-1, 9, 7)
        val_summary = data["x_R4"][val_ids, 61:124].astype(np.float64).reshape(-1, 9, 7)
        train_y = data["y"][train_ids].astype(np.float64)
        val_y = data["y"][val_ids].astype(np.float64)
    train_reconstructed = regrets_from_exact_summaries(train_summary)
    val_reconstructed = regrets_from_exact_summaries(val_summary)
    train_diff = float(np.max(np.abs(train_reconstructed - train_y)))
    val_diff = float(np.max(np.abs(val_reconstructed - val_y)))
    if train_diff != 0.0 or val_diff != 0.0:
        raise D154Error("exact R4-to-regret target reconstruction was not bit-exact on D152 train/validation")
    with np.load(corpus_path, allow_pickle=False) as corpus:
        fresh_x = corpus["x_R0"].astype(np.float64)
        fresh_summary = corpus["summary_R4"].astype(np.float64)
        fresh_layouts = corpus["layouts"].astype(str)
    fresh_y = regrets_from_exact_summaries(fresh_summary)
    if (fresh_x.shape != (1365, 61) or fresh_summary.shape != (1365, 9, 7)
            or len(fresh_layouts) != 1365 or len(set(fresh_layouts.tolist())) != 140):
        raise D154Error("D153 fresh corpus shape/layout contract mismatch")
    if set(fresh_layouts.tolist()).intersection(train_layouts + val_layouts):
        raise D154Error("D153 fresh training layouts overlap D152 train/validation")
    return {
        "d152_train_rows": len(train_ids), "d152_validation_rows": len(val_ids),
        "d152_train_layouts": len(train_layouts), "d152_validation_layouts": len(val_layouts),
        "d152_train_target_max_abs_difference": train_diff,
        "d152_validation_target_max_abs_difference": val_diff,
        "d153_fresh_rows": len(fresh_x), "d153_fresh_layouts": len(set(fresh_layouts.tolist())),
        "combined_train_rows": len(train_ids) + len(fresh_x),
        "combined_train_layouts": len(train_layouts) + len(set(fresh_layouts.tolist())),
        "validation_layouts_sha256": layout_list_sha256(val_layouts),
        "train_regret_sha256": sha256_array(np.concatenate((train_y, fresh_y))),
        "validation_regret_sha256": sha256_array(val_y),
        "fresh_y_range": [float(fresh_y.min()), float(fresh_y.max())],
    }


def load_config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def build_lock() -> dict[str, Any]:
    cfg = load_config()
    if _git("status", "--porcelain"):
        raise D154Error("commit D154 protocol, config, source and tests before locking")
    if cfg.get("schema") != "phase15_d154_decision_aligned_v2":
        raise D154Error("D154 config schema mismatch")
    if cfg.get("status") != "exploratory_offline_runner_amendment_pending_lock":
        raise D154Error("D154 protocol status mismatch")
    for key, value in cfg["scope"].items():
        if key in {"train_validation_only", "fixed_reused_validation"}:
            if value is not True:
                raise D154Error(f"D154 required scope flag disabled: {key}")
        elif value is not False:
            raise D154Error(f"D154 forbidden scope flag enabled: {key}")
    parent_hashes = {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}
    d153_manifest = _read_json(ROOT / cfg["parents"]["d153_manifest"])
    validate_d153_parent_manifest(d153_manifest)
    d153_result = _read_json(ROOT / cfg["parents"]["d153_result"])
    if d153_result.get("interpretation") != "mixed_or_inconclusive":
        raise D154Error("D153 result parent does not match closed interpretation")
    data_path = ROOT / cfg["parents"]["d152_features"]
    with np.load(data_path, allow_pickle=False) as data:
        layouts = data["layouts"].astype(str).tolist()
        split = layout_split(layouts)
        if (len(split["train"]) != 60 or len(split["validation"]) != 20
                or len(split["test"]) != 20
                or layout_list_sha256(split["validation"]) != cfg["split"]["validation_layouts_sha256"]):
            raise D154Error("D154 D152 split does not match frozen reused validation identity")
    parity = _target_parity_audit(data_path, ROOT / cfg["parents"]["d153_corpus"], {
        "train_layout_count": 60, "train_row_count": 571,
        "train_layouts_sha256": "1FF811CF3169CA339C23FC73A22617C10249297565702A592D2BD3C9C683769B",
        "validation_layout_count": 20, "validation_row_count": 202,
        "validation_layouts_sha256": cfg["split"]["validation_layouts_sha256"],
    })
    for rel in (cfg["outputs"]["lock"], cfg["outputs"]["report"], cfg["outputs"]["run_dir"]):
        if (ROOT / rel).exists():
            raise D154Error(f"refusing to overwrite pre-existing D154 output: {rel}")
    return {
        "schema": "phase15_d154_decision_aligned_lock_v2",
        "status": "locked_exploratory_train_validation_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "config_sha256": sha256_file(ROOT / CONFIG),
        "parent_sha256": parent_hashes,
        "target_reconstruction_audit": parity,
        "split": {"train_layouts": 200, "train_rows": 1936,
                  "validation_layouts": 20, "validation_rows": 202,
                  "validation_layouts_sha256": cfg["split"]["validation_layouts_sha256"],
                  "test_or_reserve_used": False},
        "student": cfg["student"], "references": cfg["references"],
        "metrics": cfg["metrics"], "component_diagnostic": cfg["component_diagnostic"],
        "scope": cfg["scope"], "outputs": cfg["outputs"],
        "validation_reuse": "D152 validation block reused through D153; all D154 results are exploratory",
    }


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_config()
    lock = _read_json(ROOT / cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d154_decision_aligned_lock_v2":
        raise D154Error("D154 lock schema mismatch")
    if lock.get("config_sha256") != sha256_file(ROOT / CONFIG):
        raise D154Error("D154 config differs from frozen lock")
    if lock.get("source_sha256") != {path: sha256_file(ROOT / path) for path in lock["source_sha256"]}:
        raise D154Error("D154 source differs from frozen lock")
    if lock.get("parent_sha256") != {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}:
        raise D154Error("D154 parent artifact hash mismatch")
    if lock.get("scope") != cfg["scope"] or lock.get("test_or_reserve_used") is True:
        raise D154Error("D154 scope/lock mismatch")
    if _git("status", "--porcelain"):
        raise D154Error("D154 execution requires a clean committed tree")
    # A non-ancestor makes git return nonzero and _git raises D154Error.
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    if (ROOT / cfg["outputs"]["run_dir"]).exists() or (ROOT / cfg["outputs"]["report"]).exists():
        raise D154Error("D154 output exists; refusing a second run")
    return cfg, lock


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def build_r4_reference_inputs(r0: np.ndarray, exact_r4: np.ndarray,
                              predicted_summary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build frozen R4-head inputs without duplicating exact R0 fields."""
    state = np.asarray(r0, dtype=np.float64)
    exact = np.asarray(exact_r4, dtype=np.float64)
    predicted = np.asarray(predicted_summary, dtype=np.float64)
    if (state.ndim != 2 or state.shape[1] != 61 or exact.shape != (len(state), 124)
            or predicted.shape != (len(state), 9, 7)
            or not all(np.isfinite(value).all() for value in (state, exact, predicted))):
        raise ValueError("D154 R4 references require R0 [N,61], exact R4 [N,124], predicted [N,9,7]")
    predicted_input = np.concatenate((state, predicted.reshape(len(state), 63)), axis=1)
    if predicted_input.shape != exact.shape:
        raise ValueError("D154 predicted and exact R4 reference inputs do not align")
    return predicted_input, exact


def _layout_macro(values: np.ndarray, layouts: np.ndarray) -> float:
    groups = np.asarray(layouts).astype(str)
    return float(np.mean([np.mean(values[groups == layout]) for layout in sorted(set(groups.tolist()))]))


def decision_detail_metrics(target: np.ndarray, lower_is_better_scores: np.ndarray,
                            layouts: np.ndarray, thresholds: dict[str, Any]) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64)
    score = np.asarray(lower_is_better_scores, dtype=np.float64)
    base = decision_metrics(y, score, layouts)
    chosen = np.argmin(score, axis=1)
    chosen_regret = y[np.arange(len(y)), chosen]
    high_margin_accuracy: list[float] = []
    high_margin_layouts: list[str] = []
    safe_cat_accuracy: list[float] = []
    safe_cat_layouts: list[str] = []
    for row_y, row_score, layout in zip(y, score, np.asarray(layouts).astype(str), strict=True):
        left, right = np.triu_indices(9, 1)
        gap = np.abs(row_y[left] - row_y[right])
        keep = gap >= float(thresholds["high_margin_pair_gap_threshold"])
        if np.any(keep):
            true_sign = np.sign(row_y[right[keep]] - row_y[left[keep]])
            predicted_sign = np.sign(row_score[right[keep]] - row_score[left[keep]])
            high_margin_accuracy.append(float(np.mean(true_sign == predicted_sign)))
            high_margin_layouts.append(str(layout))
        safe = (row_y[left] <= float(thresholds["safe_action_regret_at_most"]))
        catastrophic = (row_y[right] >= float(thresholds["catastrophic_action_regret_at_least"]))
        safe_left = safe & catastrophic
        cat_left_safe_right = ((row_y[left] >= float(thresholds["catastrophic_action_regret_at_least"]))
                               & (row_y[right] <= float(thresholds["safe_action_regret_at_most"])))
        # In each pair, lower score must be assigned to the safe action.
        pair_i = np.concatenate((left[safe_left], right[cat_left_safe_right]))
        pair_j = np.concatenate((right[safe_left], left[cat_left_safe_right]))
        if len(pair_i):
            safe_cat_accuracy.append(float(np.mean(row_score[pair_i] < row_score[pair_j])))
            safe_cat_layouts.append(str(layout))
    return {
        "layout_macro_normalized_selected_regret": base["mean_normalized_regret_layout_macro"],
        "state_micro_normalized_selected_regret": base["mean_normalized_regret_state_micro"],
        "catastrophic_fraction_state_micro": base["catastrophic_fraction_state_micro"],
        "catastrophic_fraction_layout_macro": base["catastrophic_fraction_layout_macro"],
        "best_action_set_agreement": base["best_action_set_agreement"],
        "epsilon_optimal_rate": float(np.mean(chosen_regret <= float(thresholds["epsilon_optimal_regret_threshold"]) + 1e-12)),
        "high_margin_pairwise_accuracy_macro": _layout_macro(np.asarray(high_margin_accuracy), np.asarray(high_margin_layouts))
            if high_margin_accuracy else None,
        "safe_vs_catastrophic_order_accuracy_macro": _layout_macro(np.asarray(safe_cat_accuracy), np.asarray(safe_cat_layouts))
            if safe_cat_accuracy else None,
        "all_pairwise_ranking_accuracy_macro": base["pairwise_ranking_accuracy_macro"],
        "regret_gap_weighted_pairwise_accuracy_macro": base["regret_gap_weighted_pairwise_accuracy_macro"],
        "selected_action_frequency": base["predicted_action_frequency"],
        "selected_regret": chosen_regret.tolist(),
        "per_layout_mean_regret": base["per_layout_mean_regret"],
    }


class ListwiseRegretMLP:
    """Same 61-32-32-9 MLP trained against a soft teacher distribution."""

    def __init__(self, seed: int, temperature: float, max_epochs: int = 1200,
                 patience: int = 100, learning_rate: float = 0.002,
                 l2: float = 1e-5) -> None:
        self.seed, self.temperature = int(seed), float(temperature)
        self.max_epochs, self.patience = int(max_epochs), int(patience)
        self.learning_rate, self.l2 = float(learning_rate), float(l2)

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0.0)
        logits = h2 @ self.w3 + self.b3
        return logits, (x, h1, h2), (z1, z2)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, train_layouts: np.ndarray,
            x_val: np.ndarray, y_val: np.ndarray, val_layouts: np.ndarray) -> "ListwiseRegretMLP":
        x_train, y_train = np.asarray(x_train, dtype=np.float64), np.asarray(y_train, dtype=np.float64)
        x_val, y_val = np.asarray(x_val, dtype=np.float64), np.asarray(y_val, dtype=np.float64)
        if (x_train.ndim != 2 or x_train.shape[1] != 61 or y_train.shape != (len(x_train), 9)
                or x_val.ndim != 2 or x_val.shape[1] != 61 or y_val.shape != (len(x_val), 9)
                or len(train_layouts) != len(x_train) or len(val_layouts) != len(x_val)
                or not all(np.isfinite(v).all() for v in (x_train, y_train, x_val, y_val))):
            raise ValueError("invalid D154 listwise fit arrays")
        self.x_mean = x_train.mean(axis=0)
        self.x_std = x_train.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        x, xv = (x_train - self.x_mean) / self.x_std, (x_val - self.x_mean) / self.x_std
        teacher_train = _softmax(-y_train / self.temperature)
        teacher_val = _softmax(-y_val / self.temperature)
        wt, wv = layout_weights(np.asarray(train_layouts)), layout_weights(np.asarray(val_layouts))
        rng = np.random.default_rng(self.seed)
        dims = (61, 32, 32, 9)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / dims[0]), (dims[0], 32))
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / 32), (32, 32))
        self.w3 = rng.normal(0.0, np.sqrt(2.0 / 32), (32, 9))
        self.b1, self.b2, self.b3 = np.zeros(32), np.zeros(32), np.zeros(9)
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        def cross_entropy(logits: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
            logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
            return float(-np.sum(weights[:, None] * targets * logp))
        best = None
        best_criterion = (float("inf"), float("inf"))
        stale = 0
        for epoch in range(self.max_epochs):
            logits, acts, zs = self._forward(x)
            probs = _softmax(logits)
            delta = wt[:, None] * (probs - teacher_train)
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
            for index, (parameter, gradient) in enumerate(zip(params, grads, strict=True)):
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
                parameter -= self.learning_rate * (first[index] / (1.0 - 0.9**t)) / (
                    np.sqrt(second[index] / (1.0 - 0.999**t)) + 1e-8)
            val_logits = self._forward(xv)[0]
            val_loss = cross_entropy(val_logits, teacher_val, wv)
            val_regret = decision_metrics(y_val, -val_logits, np.asarray(val_layouts))[
                "mean_normalized_regret_layout_macro"]
            criterion = (float(val_regret), val_loss)
            improved = (criterion[0] < best_criterion[0] - 1e-9
                        or (abs(criterion[0] - best_criterion[0]) <= 1e-9
                            and criterion[1] < best_criterion[1] - 1e-7))
            if improved:
                best_criterion, stale = criterion, 0
                best = [p.copy() for p in params]
                self.best_epoch = epoch + 1
            else:
                stale += 1
            if stale >= self.patience:
                break
        if best is None:
            raise RuntimeError("D154 listwise model did not produce a checkpoint")
        for name, value in zip(("w1", "b1", "w2", "b2", "w3", "b3"), best, strict=True):
            setattr(self, name, value)
        self.epochs_run = epoch + 1
        self.validation_loss = best_criterion[1]
        self.validation_decision_regret = best_criterion[0]
        self.parameter_count = int(sum(p.size for p in params))
        return self

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        return self._forward((np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std)[0]


def _predict_latency(model: Any, x: np.ndarray, listwise: bool = False) -> tuple[np.ndarray, float]:
    predict = model.predict_logits if listwise else model.predict
    warm_count = min(10, len(x))
    for row in x[:warm_count]:
        predict(row[None, :])
    times, values = [], []
    for row in x:
        start = time.perf_counter_ns()
        value = predict(row[None, :])[0]
        times.append((time.perf_counter_ns() - start) / 1e9)
        values.append(value)
    return np.asarray(values), float(np.median(times))


def _model_metrics(model: Any, x: np.ndarray, y: np.ndarray, layouts: np.ndarray,
                   cfg: dict[str, Any], listwise: bool = False) -> dict[str, Any]:
    raw, latency = _predict_latency(model, x, listwise)
    scores = -raw if listwise else raw
    detail = decision_detail_metrics(y, scores, layouts, cfg["metrics"])
    if not listwise:
        detail["regret_score_rmse"] = float(np.sqrt(np.mean((scores - y) ** 2)))
        detail["regret_score_mae"] = float(np.mean(np.abs(scores - y)))
    detail["inference_median_seconds_per_state"] = latency
    detail["latency_scope"] = "end_to_end_state_to_action"
    return detail


def _reference_metrics(cfg: dict[str, Any], x_val: np.ndarray, y_val: np.ndarray,
                       val_layouts: np.ndarray, exact_r4: np.ndarray,
                       predicted_summary: np.ndarray) -> dict[str, Any]:
    r0 = restore_decision_head(ROOT / cfg["parents"]["d150_r0_checkpoint"])
    r4 = restore_decision_head(ROOT / cfg["parents"]["d150_r4_checkpoint"])
    predicted_input, exact_input = build_r4_reference_inputs(x_val, exact_r4, predicted_summary)
    predictions = {}
    for name, head, features in (
        ("direct_D150_R0", r0, x_val),
        ("D153_predicted_R4_then_D150_head", r4, predicted_input),
        ("exact_R4_then_D150_head_oracle", r4, exact_input),
    ):
        scores, latency = _predict_latency(head, features)
        detail = decision_detail_metrics(y_val, scores, val_layouts, cfg["metrics"])
        if name == "direct_D150_R0":
            detail["inference_median_seconds_per_state"] = latency
            detail["latency_scope"] = "end_to_end_state_to_action"
            detail["parameter_count"] = head.parameter_count
        elif name == "D153_predicted_R4_then_D150_head":
            # D153 persisted validation predictions, not the predictor checkpoint.
            # Do not misreport head-only time as total latency.
            detail["inference_median_seconds_per_state"] = None
            detail["r4_head_only_median_seconds_per_state"] = latency
            detail["latency_scope"] = "end_to_end_unavailable_predictor_checkpoint_not_saved"
            detail["parameter_count"] = 3559 + head.parameter_count
        else:
            detail["inference_median_seconds_per_state"] = None
            detail["r4_head_only_median_seconds_per_state"] = latency
            detail["latency_scope"] = "oracle_R4_features_supplied_scorer_only"
            detail["parameter_count"] = head.parameter_count
        predictions[name] = {"scores": scores, "metrics": detail}
    return predictions


def _load_data(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_path = ROOT / cfg["parents"]["d152_features"]
    with np.load(dataset_path, allow_pickle=False) as data:
        all_layouts = data["layouts"].astype(str).tolist()
        split_cfg = {
            "train_layout_count": 60, "train_row_count": 571,
            "train_layouts_sha256": "1FF811CF3169CA339C23FC73A22617C10249297565702A592D2BD3C9C683769B",
            "validation_layout_count": 20, "validation_row_count": 202,
            "validation_layouts_sha256": cfg["split"]["validation_layouts_sha256"],
        }
        train_ids, val_ids, old_train_layouts, val_layouts_list = train_validation_indices(all_layouts, split_cfg)
        # Explicit allowlist: only old-train and validation rows are dereferenced.
        old_x = data["x_R0"][train_ids].astype(np.float64)
        old_y = data["y"][train_ids].astype(np.float64)
        old_r4 = data["x_R4"][train_ids].astype(np.float64)
        val_x = data["x_R0"][val_ids].astype(np.float64)
        val_y = data["y"][val_ids].astype(np.float64)
        exact_r4_full = data["x_R4"][val_ids].astype(np.float64)
        val_layouts = np.asarray(all_layouts, dtype=str)[val_ids]
        train_layouts = np.asarray(all_layouts, dtype=str)[train_ids]
        old_reconstructed = regrets_from_exact_summaries(old_r4[:, 61:124].reshape(-1, 9, 7))
        val_reconstructed = regrets_from_exact_summaries(exact_r4_full[:, 61:124].reshape(-1, 9, 7))
        if not np.array_equal(old_reconstructed, old_y) or not np.array_equal(val_reconstructed, val_y):
            raise D154Error("D152 exact R4 summary does not reproduce saved normalized regret targets")
    corpus_path = ROOT / cfg["parents"]["d153_corpus"]
    with np.load(corpus_path, allow_pickle=False) as corpus:
        fresh_x = corpus["x_R0"].astype(np.float64)
        fresh_summary = corpus["summary_R4"].astype(np.float64)
        fresh_layouts = corpus["layouts"].astype(str)
    fresh_y = regrets_from_exact_summaries(fresh_summary)
    if set(fresh_layouts.tolist()).intersection(set(train_layouts) | set(val_layouts)):
        raise D154Error("D153 added training layouts overlap D152 layouts")
    x_train = np.concatenate((old_x, fresh_x), axis=0)
    y_train = np.concatenate((old_y, fresh_y), axis=0)
    layout_train = np.concatenate((train_layouts, fresh_layouts))
    if len(x_train) != 1936 or len(set(layout_train.tolist())) != 200 or len(val_x) != 202:
        raise D154Error("D154 D200 training / reused validation cardinality mismatch")
    val_summary_prediction_path = ROOT / cfg["parents"]["d153_validation_predictions"]
    with np.load(val_summary_prediction_path, allow_pickle=False) as saved:
        predicted_summary = saved["200"].astype(np.float64).reshape(-1, 9, 7)
    if predicted_summary.shape != (202, 9, 7):
        raise D154Error("D153 D200 validation prediction shape mismatch")
    return {"x_train": x_train, "y_train": y_train, "train_layouts": layout_train,
            "x_val": val_x, "y_val": val_y, "val_layouts": val_layouts,
            "exact_r4": exact_r4_full, "predicted_summary": predicted_summary}


def _train_memorization(cfg: dict[str, Any], x: np.ndarray, y: np.ndarray,
                        layouts: np.ndarray) -> dict[str, Any]:
    # Deterministic 8-state sample: first state from each of the first 8 layouts.
    first_by_layout: dict[str, int] = {}
    for index, layout in enumerate(np.asarray(layouts).astype(str)):
        first_by_layout.setdefault(str(layout), index)
    order = [first_by_layout[layout] for layout in sorted(first_by_layout)[:8]]
    mx, my, ml = x[order], y[order], layouts[order]
    results = {}
    settings = cfg["student"]
    for name, objective in (("T1_regret_regression", "regression"),
                            ("T2_margin_weighted_pairwise", "weighted_pairwise")):
        model = TinyRegretMLP(settings["seed"], objective, max_epochs=settings["max_epochs"],
                              patience=settings["patience"], learning_rate=settings["learning_rate"], l2=settings["l2"])
        model.fit(mx, my, ml, mx, my, ml)
        detail = decision_detail_metrics(my, model.predict(mx), ml, cfg["metrics"])
        results[name] = {"best_epoch": model.best_epoch, "mean_selected_regret": detail["state_micro_normalized_selected_regret"],
                         "best_action_set_agreement": detail["best_action_set_agreement"],
                         "passed": detail["state_micro_normalized_selected_regret"] <= settings["memorization"]["maximum_mean_selected_regret"]
                         and detail["best_action_set_agreement"] >= settings["memorization"]["minimum_best_action_agreement"]}
    for tau in settings["formulations"][2]["temperatures"]:
        name = f"T3_listwise_tau_{tau:g}"
        model = ListwiseRegretMLP(settings["seed"], tau, settings["max_epochs"], settings["patience"],
                                  settings["learning_rate"], settings["l2"]).fit(mx, my, ml, mx, my, ml)
        detail = decision_detail_metrics(my, -model.predict_logits(mx), ml, cfg["metrics"])
        results[name] = {"best_epoch": model.best_epoch, "mean_selected_regret": detail["state_micro_normalized_selected_regret"],
                         "best_action_set_agreement": detail["best_action_set_agreement"],
                         "passed": detail["state_micro_normalized_selected_regret"] <= settings["memorization"]["maximum_mean_selected_regret"]
                         and detail["best_action_set_agreement"] >= settings["memorization"]["minimum_best_action_agreement"]}
    return {"states": len(order), "results": results,
            "all_passed": all(value["passed"] for value in results.values())}


def _save_model(path: Path, model: Any) -> None:
    fields = {name: getattr(model, name) for name in ("x_mean", "x_std", "w1", "b1", "w2", "b2", "w3", "b3")}
    np.savez_compressed(path, **fields)


def _strip_selected(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"selected_regret", "per_layout_mean_regret"}}


def run_d154() -> dict[str, Any]:
    cfg, lock = validate_lock()
    data = _load_data(cfg)
    started = time.perf_counter()
    output = ROOT / cfg["outputs"]["run_dir"]
    output.mkdir(parents=True, exist_ok=False)
    mem = _train_memorization(cfg, data["x_train"], data["y_train"], data["train_layouts"])
    (output / "memorization.json").write_text(json.dumps(mem, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not mem["all_passed"]:
        raise D154Error("D154 8-state memorization gate failed; stopped before full model fitting")

    refs = _reference_metrics(cfg, data["x_val"], data["y_val"], data["val_layouts"],
                              data["exact_r4"], data["predicted_summary"])
    frozen_d153 = _read_json(ROOT / cfg["parents"]["d153_result"])
    expected_refs = {
        "direct_D150_R0": frozen_d153["fixed_references"]["direct_r0_decisions"],
        "exact_R4_then_D150_head_oracle": frozen_d153["fixed_references"]["exact_r4_decisions"],
        "D153_predicted_R4_then_D150_head": next(
            record["validation_downstream_decisions"] for record in frozen_d153["scale_results"]
            if int(record["layout_count"]) == 200),
    }
    reference_parity = {}
    for name, expected in expected_refs.items():
        actual = refs[name]["metrics"]
        pairs = (("layout_macro_normalized_selected_regret", "layout_macro_normalized_regret"),
                 ("catastrophic_fraction_state_micro", "state_micro_catastrophic_fraction_regret_ge_0_5"),
                 ("best_action_set_agreement", "best_action_set_agreement"))
        reference_parity[name] = {
            metric: abs(float(actual[current]) - float(expected[saved]))
            for current, saved in pairs for metric in [current]
        }
    if max(value for record in reference_parity.values() for value in record.values()) > 1e-10:
        raise D154Error(f"frozen D150/D153 reference reproduction failed: {reference_parity}")
    settings = cfg["student"]
    models: dict[str, Any] = {}
    fit_results: dict[str, Any] = {}
    start_all = time.perf_counter()
    for name, objective in (("T1_regret_regression", "regression"),
                            ("T2_margin_weighted_pairwise", "weighted_pairwise")):
        fit_start = time.perf_counter()
        model = TinyRegretMLP(settings["seed"], objective,
                              max_epochs=settings["max_epochs"], patience=settings["patience"],
                              learning_rate=settings["learning_rate"], l2=settings["l2"])
        model.fit(data["x_train"], data["y_train"], data["train_layouts"],
                  data["x_val"], data["y_val"], data["val_layouts"])
        models[name] = model
        _save_model(output / f"{name}.npz", model)
        fit_results[name] = {
            "objective": objective, "best_epoch": model.best_epoch, "epochs_run": model.epochs_run,
            "parameter_count": model.parameter_count, "fit_seconds": time.perf_counter() - fit_start,
            "train": _model_metrics(model, data["x_train"], data["y_train"], data["train_layouts"], cfg),
            "validation": _model_metrics(model, data["x_val"], data["y_val"], data["val_layouts"], cfg),
        }
    for tau in settings["formulations"][2]["temperatures"]:
        name = f"T3_listwise_tau_{tau:g}"
        fit_start = time.perf_counter()
        model = ListwiseRegretMLP(settings["seed"], tau, settings["max_epochs"], settings["patience"],
                                  settings["learning_rate"], settings["l2"])
        model.fit(data["x_train"], data["y_train"], data["train_layouts"],
                  data["x_val"], data["y_val"], data["val_layouts"])
        models[name] = model
        _save_model(output / f"{name}.npz", model)
        fit_results[name] = {
            "objective": "listwise_cross_entropy", "temperature": tau,
            "best_epoch": model.best_epoch, "epochs_run": model.epochs_run,
            "parameter_count": model.parameter_count, "fit_seconds": time.perf_counter() - fit_start,
            "train": _model_metrics(model, data["x_train"], data["y_train"], data["train_layouts"], cfg, True),
            "validation": _model_metrics(model, data["x_val"], data["y_val"], data["val_layouts"], cfg, True),
            "teacher_distribution_cross_entropy_validation": model.validation_loss,
        }
    component = component_ablation(data["exact_r4"], data["predicted_summary"].reshape(-1, 63),
                                  data["x_val"], data["y_val"], data["val_layouts"],
                                  restore_decision_head(ROOT / cfg["parents"]["d150_r4_checkpoint"]))
    component_metrics = {name: _strip_selected(value) for name, value in component.items()}
    baseline_regret = baseline_metrics["D153_predicted_R4_then_D150_head"]["layout_macro_normalized_selected_regret"]
    best_direct = min((record["validation"]["layout_macro_normalized_selected_regret"], name)
                      for name, record in fit_results.items())
    result = {
        "schema": "phase15_d154_decision_aligned_result_v1", "status": "completed_exploratory_offline_screen",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
        "parent_sha256": lock["parent_sha256"], "target_reconstruction_audit": lock["target_reconstruction_audit"],
        "split": lock["split"], "memorization": mem,
        "frozen_reference_reproduction_max_abs_differences": reference_parity,
        "references": {name: {"metrics": value["metrics"], "parameter_count": value["parameter_count"]}
                       for name, value in refs.items()},
        "models": fit_results, "component_ablation": component_metrics,
        "decision_summary": {
            "best_direct_formulation": best_direct[1],
            "best_direct_validation_macro_regret": best_direct[0],
            "d153_predicted_r4_validation_macro_regret": baseline_regret,
            "best_direct_minus_predicted_r4_regret": best_direct[0] - baseline_regret,
            "direct_improves_regret_vs_predicted_r4": best_direct[0] < baseline_regret,
            "validation_reuse_precludes_independent_confirmation": True,
        },
        "protocol_deviations": [],
        "scope": {"simulator_calls": 0, "planner_calls": 0, "new_transitions": 0,
                  "new_episodes": 0, "closed_loop": False, "dagger": False,
                  "world_model": False, "jepa": False, "test_rows_scored": 0,
                  "reserve_access": False},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "wall_seconds": time.perf_counter() - started,
        "fit_total_seconds": time.perf_counter() - start_all,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = render_report(result)
    report_path = ROOT / cfg["outputs"]["report"]
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "schema": "phase15_d154_decision_aligned_manifest_v1", "status": result["status"],
        "lock_sha256": result["lock_sha256"], "source_commit": result["source_commit"],
        "parent_sha256": result["parent_sha256"], "result_sha256": sha256_file(result_path),
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": {path.name: sha256_file(path) for path in sorted(output.glob("*.npz"))},
        "memorization_sha256": sha256_file(output / "memorization.json"),
        "test_rows_scored": 0, "reserve_access": False,
        "simulator_calls": 0, "planner_calls": 0, "closed_loop": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D154 — Decision-Aligned Target Study", "",
             "**Status: exploratory offline mechanism study on the reused D152 validation block.** No independent confirmation is claimed; test/reserve remain sealed.", "",
             "## Target integrity and split", "",
             f"D152 target reconstruction max absolute difference: train `{result['target_reconstruction_audit']['d152_train_target_max_abs_difference']:.1f}`, validation `{result['target_reconstruction_audit']['d152_validation_target_max_abs_difference']:.1f}`.",
             f"Training: {result['split']['train_rows']} states / {result['split']['train_layouts']} layouts (D152 train plus D153 fresh corpus); validation: {result['split']['validation_rows']} states / {result['split']['validation_layouts']} reused layouts.", "",
             "## Eight-state memorization gate", "", "| Formulation | Best epoch | Train selected regret | Best-action agreement | Passed |", "|---|---:|---:|---:|---:|"]
    for name, value in result["memorization"]["results"].items():
        lines.append(f"| {name} | {value['best_epoch']} | {value['mean_selected_regret']:.6f} | {value['best_action_set_agreement']:.4f} | {value['passed']} |")
    lines.extend(["", "## Decision metrics", "", "| Method | Macro regret | Catastrophic | Best action | Epsilon-optimal | High-margin ranking | Safe-vs-catastrophic order | Latency (ms/state) | Latency scope | Params |", "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"])
    methods = [(name, record["metrics"], record["parameter_count"])
               for name, record in result["references"].items()]
    methods += [(name, record["validation"], record["parameter_count"])
                for name, record in result["models"].items()]
    for name, metrics, params in methods:
        latency = metrics.get("inference_median_seconds_per_state")
        latency_text = "n/a" if latency is None else f"{1000*latency:.5f}"
        high_margin = metrics["high_margin_pairwise_accuracy_macro"]
        safe_cat = metrics["safe_vs_catastrophic_order_accuracy_macro"]
        high_margin_text = "n/a" if high_margin is None else f"{high_margin:.4f}"
        safe_cat_text = "n/a" if safe_cat is None else f"{safe_cat:.4f}"
        lines.append(f"| {name} | {metrics['layout_macro_normalized_selected_regret']:.6f} | {metrics['catastrophic_fraction_state_micro']:.4f} | {metrics['best_action_set_agreement']:.4f} | {metrics['epsilon_optimal_rate']:.4f} | {high_margin_text} | {safe_cat_text} | {latency_text} | {metrics.get('latency_scope', 'unknown')} | {params} |")
    lines.extend(["", "## Fit and training diagnostics", "", "| Formulation | Best epoch | Train macro regret | Train catastrophic | Validation macro regret | Fit seconds |", "|---|---:|---:|---:|---:|---:|"])
    for name, record in result["models"].items():
        lines.append(f"| {name} | {record['best_epoch']} | {record['train']['layout_macro_normalized_selected_regret']:.6f} | {record['train']['catastrophic_fraction_state_micro']:.4f} | {record['validation']['layout_macro_normalized_selected_regret']:.6f} | {record['fit_seconds']:.3f} |")
    lines.extend(["", "## Frozen R4 component diagnostic", "", "This is a scorer-specific feature substitution test, not causal component importance.", "", "| Feature substitution | Macro regret | Catastrophic | Best action |", "|---|---:|---:|---:|"])
    for name, metrics in result["component_ablation"].items():
        lines.append(f"| {name} | {metrics['mean_normalized_regret_layout_macro']:.6f} | {metrics['catastrophic_fraction_state_micro']:.4f} | {metrics['best_action_set_agreement']:.4f} |")
    summary = result["decision_summary"]
    lines.extend(["", "## Interpretation and boundary", "",
                  f"Best direct formulation by validation macro regret: **{summary['best_direct_formulation']}** (`{summary['best_direct_validation_macro_regret']:.6f}`); D153 predicted-R4 pipeline: `{summary['d153_predicted_r4_validation_macro_regret']:.6f}`. The difference is exploratory because validation has been reused.",
                  "",
                  "Teacher softmax outputs are preference distributions, not calibrated success probabilities. No causal claim follows from summary RMSE or feature substitution. No closed-loop test, DAgger, model/world-model training, JEPA, reserve/test scoring, or new simulator/planner call occurred. D154 stops here pending a separately selected next protocol.", ""])
    return "\n".join(lines)
