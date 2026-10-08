"""D166 asymmetric collision-risk learning and risk-veto evaluation."""

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
from consequence_lab.phase15_d155_stage_c import encode_targets
from consequence_lab.phase15_d158 import _action_inputs, _candidate_metrics, _feature_set, _probabilities_from_exact
from consequence_lab.phase15_d162 import _load_student
from consequence_lab.phase15_d165 import ActionEffectMLP, _load_data, paired_layout_split
from consequence_lab.phase15_decision_aligned import sha256_array
from consequence_lab.phase15_d157 import ACTION_COUNT
from consequence_lab.phase15_learning import layout_weights

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d166_failure_aware_risk.yaml"
SOURCE_FILES = (
    CONFIG, "reports/phase15_d166_protocol.md", "src/consequence_lab/phase15_d166.py",
    "scripts/freeze_phase15_d166_lock.py", "scripts/run_phase15_d166.py",
    "tests/test_phase15_d166.py", "src/consequence_lab/phase15_d165.py",
    "src/consequence_lab/phase15_d162.py", "src/consequence_lab/phase15_d158.py",
    "src/consequence_lab/phase15_d155_stage_c.py", "src/consequence_lab/phase15_d157.py",
    "src/consequence_lab/phase15_decision_aligned.py", "src/consequence_lab/phase15_learning.py",
    "src/consequence_lab/phase14.py", "configs/phase15_d155_stage_c.yaml",
    "experiments/phase15_d165_action_effect_alignment_v2.lock.json",
    "reports/phase15_d165_interpretation.md", "reports/phase15_d165_v2_result.md",
    "second-brain/00_Project_Home.md", "second-brain/14_Decisions.md",
    "second-brain/16_Experiment_Log.md",
    "second-brain/phases/Phase_15_Decision_Aligned_Target_Study.md",
)
PARAMS = ("w1", "b1", "w2", "b2", "w3", "b3")


class D166Error(RuntimeError):
    """D166 provenance or risk-model contract error."""


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def select_threshold(probability: np.ndarray, target: np.ndarray, weights: np.ndarray,
                     grid: list[float], minimum_recall: float) -> dict[str, Any]:
    p, y, w = np.asarray(probability, float).reshape(-1), np.asarray(target, bool).reshape(-1), np.asarray(weights, float).reshape(-1)
    if p.shape != y.shape or w.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(w).all():
        raise ValueError("D166 threshold arrays must be aligned and finite")
    positive_weight = float(np.sum(w * y))
    if positive_weight <= 0:
        return {"threshold": 0.0, "feasible": False, "validation_recall": 0.0}
    feasible = []
    for threshold in grid:
        predicted = p >= float(threshold)
        recall = float(np.sum(w * (y & predicted)) / positive_weight)
        if recall + 1e-12 >= minimum_recall:
            feasible.append((float(threshold), recall))
    if not feasible:
        threshold = 0.0
        recall = float(np.sum(w * y * (p >= threshold)) / positive_weight)
        return {"threshold": threshold, "feasible": False, "validation_recall": recall}
    threshold, recall = feasible[-1]
    return {"threshold": threshold, "feasible": True, "validation_recall": recall}


class CollisionRiskMLP:
    def __init__(self, hidden: tuple[int, int], seed: int, learning_rate: float,
                 l2: float, beta1: float, beta2: float, epsilon: float) -> None:
        self.hidden, self.seed, self.lr, self.l2 = hidden, seed, learning_rate, l2
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon

    def _forward(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0.0)
        return (h2 @ self.w3 + self.b3)[:, 0], (x, h1, h2), (z1, z2)

    def _backward(self, grad: np.ndarray, cache: tuple[np.ndarray, ...], pre: tuple[np.ndarray, ...]):
        x, h1, h2 = cache
        z1, z2 = pre
        g = grad[:, None]
        gw3 = h2.T @ g + self.l2 * self.w3
        gb3 = g.sum(axis=0)
        dz2 = (g @ self.w3.T) * (z2 > 0)
        gw2 = h1.T @ dz2 + self.l2 * self.w2
        gb2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.w2.T) * (z1 > 0)
        gw1 = x.T @ dz1 + self.l2 * self.w1
        gb1 = dz1.sum(axis=0)
        return [gw1, gb1, gw2, gb2, gw3, gb3]

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
            validation: tuple[np.ndarray, np.ndarray, np.ndarray],
            settings: dict[str, Any]) -> "CollisionRiskMLP":
        values, labels, sample_weights = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
        vx, vy, vw = (np.asarray(a, float) for a in validation)
        if values.ndim != 2 or values.shape[1] != 65 or labels.shape != (len(values),) or sample_weights.shape != labels.shape:
            raise ValueError("invalid D166 risk training arrays")
        if vx.shape != (len(vy), 65) or vw.shape != vy.shape or not np.isfinite(values).all():
            raise ValueError("invalid D166 risk validation arrays")
        self.x_mean, self.x_std = values.mean(axis=0), values.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        xt, xv = (values - self.x_mean) / self.x_std, (vx - self.x_mean) / self.x_std
        positive = np.flatnonzero(labels > .5)
        negative = np.flatnonzero(labels <= .5)
        if not len(positive) or not len(negative):
            raise ValueError("D166 training requires both collision and noncollision branches")
        multiplier = int(settings["positive_sampling_multiplier"])
        target_positive = min(len(negative), len(positive) * multiplier)
        extra_count = max(0, target_positive - len(positive))
        h1, h2 = self.hidden
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, np.sqrt(2 / 65), (65, h1))
        self.w2 = rng.normal(0.0, np.sqrt(2 / h1), (h1, h2))
        self.w3 = rng.normal(0.0, np.sqrt(1 / h2), (h2, 1))
        self.b1, self.b2, self.b3 = np.zeros(h1), np.zeros(h2), np.zeros(1)
        params = [getattr(self, name) for name in PARAMS]
        first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        best_params = None
        best_loss, best_epoch, stale = float("inf"), 0, 0
        history = []
        for epoch in range(int(settings["max_epochs"])):
            epoch_rng = np.random.default_rng(self.seed + epoch + 1)
            sampled_pos = np.concatenate((positive, epoch_rng.choice(positive, extra_count, replace=True))) if extra_count else positive.copy()
            indices = np.concatenate((negative, sampled_pos))
            epoch_rng.shuffle(indices)
            epoch_weights = sample_weights[indices].copy()
            epoch_weights /= max(float(epoch_weights.sum()), 1e-12)
            logits, cache, pre = self._forward(xt[indices])
            probability = _sigmoid(logits)
            positive_weight = float(settings["positive_collision_loss_weight"])
            loss_terms = (positive_weight * labels[indices] * np.logaddexp(0.0, -logits)
                          + (1.0 - labels[indices]) * np.logaddexp(0.0, logits))
            grad = epoch_weights * (positive_weight * labels[indices] * (probability - 1.0)
                                    + (1.0 - labels[indices]) * probability)
            grads = self._backward(grad, cache, pre)
            for idx, (param, derivative) in enumerate(zip(params, grads, strict=True)):
                first[idx] = self.beta1 * first[idx] + (1 - self.beta1) * derivative
                second[idx] = self.beta2 * second[idx] + (1 - self.beta2) * derivative**2
                t = epoch + 1
                param -= self.lr * (first[idx] / (1 - self.beta1**t)) / (
                    np.sqrt(second[idx] / (1 - self.beta2**t)) + self.epsilon)
            val_logits = self._forward(xv)[0]
            val_probability = _sigmoid(val_logits)
            val_loss = (float(settings["positive_collision_loss_weight"]) * vy * np.logaddexp(0.0, -val_logits)
                        + (1.0 - vy) * np.logaddexp(0.0, val_logits))
            objective = float(np.sum(vw * val_loss))
            if epoch == 0 or (epoch + 1) % 25 == 0:
                history.append({"epoch": epoch + 1, "sampled_train_asymmetric_bce": float(np.sum(epoch_weights * loss_terms)),
                                "validation_asymmetric_bce": objective,
                                "validation_collision_recall_at_0_5": risk_metrics(val_probability, vy, vw, .5)["recall"]})
            if objective < best_loss - 1e-9:
                best_loss, best_epoch, stale = objective, epoch + 1, 0
                best_params = [p.copy() for p in params]
            else:
                stale += 1
            if stale >= int(settings["patience"]):
                break
        if best_params is None:
            raise RuntimeError("D166 risk model did not checkpoint")
        for name, value in zip(PARAMS, best_params, strict=True):
            setattr(self, name, value)
        self.best_epoch, self.epochs_run, self.validation_loss = best_epoch, epoch + 1, best_loss
        self.history = history
        self.parameter_count = int(sum(p.size for p in params))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, float)
        if values.ndim != 2 or values.shape[1] != 65 or not np.isfinite(values).all():
            raise ValueError("D166 risk inputs must be finite [N,65]")
        return _sigmoid(self._forward((values - self.x_mean) / self.x_std)[0])


def risk_metrics(probability: np.ndarray, target: np.ndarray, weights: np.ndarray,
                 threshold: float) -> dict[str, Any]:
    p, y, w = np.asarray(probability, float).reshape(-1), np.asarray(target, bool).reshape(-1), np.asarray(weights, float).reshape(-1)
    if p.shape != y.shape or w.shape != p.shape or np.any(w < 0):
        raise ValueError("D166 risk metrics arrays must be aligned")
    predicted = p >= threshold
    tp = float(np.sum(w * (y & predicted)))
    fp = float(np.sum(w * (~y & predicted)))
    fn = float(np.sum(w * (y & ~predicted)))
    tn = float(np.sum(w * (~y & ~predicted)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    order = np.argsort(-p, kind="stable")
    tp_curve = np.cumsum(w[order] * y[order])
    fp_curve = np.cumsum(w[order] * ~y[order])
    positive = max(float(np.sum(w * y)), 1e-12)
    recall_curve = tp_curve / positive
    precision_curve = tp_curve / np.maximum(tp_curve + fp_curve, 1e-12)
    pr_auc = float(np.sum(np.diff(np.concatenate(([0.0], recall_curve))) * precision_curve))
    return {"threshold": float(threshold), "recall": recall, "false_negative_rate": 1 - recall,
            "precision": precision, "pr_auc": pr_auc,
            "false_negative_weight": fn, "false_positive_weight": fp,
            "true_positive_weight": tp, "true_negative_weight": tn,
            "predicted_positive_rate": float(np.sum(w * predicted))}


def apply_risk_veto(preference_score: np.ndarray, risk: np.ndarray,
                    threshold: float) -> np.ndarray:
    preference = np.asarray(preference_score, dtype=np.float64)
    probability = np.asarray(risk, dtype=np.float64)
    if preference.shape != probability.shape or preference.ndim != 2 or preference.shape[1] != ACTION_COUNT:
        raise ValueError("D166 policy inputs must be aligned [N,9]")
    output = np.empty_like(preference)
    for row in range(len(preference)):
        allowed = probability[row] < threshold
        if np.any(allowed):
            output[row] = preference[row]
            output[row, ~allowed] = np.max(preference[row]) + 1.0
        else:
            tie_break = (preference[row] - np.min(preference[row])) / max(
                float(np.ptp(preference[row])), 1.0)
            output[row] = probability[row] + 1e-6 * tie_break
    return output


def _effect_predictor(path: Path, settings: dict[str, Any]) -> ActionEffectMLP:
    model = ActionEffectMLP(tuple(settings["hidden"]), int(settings["seed"]),
                            float(settings["learning_rate"]), float(settings["l2"]))
    with np.load(path, allow_pickle=False) as saved:
        for key in ("x_mean", "x_std", "step_mean", "step_std", *PARAMS):
            if key not in saved.files:
                raise D166Error(f"D165 effect checkpoint missing {key}")
        model.x_mean = np.asarray(saved["x_mean"], float)
        model.x_std = np.asarray(saved["x_std"], float)
        model.step_mean, model.step_std = float(saved["step_mean"]), float(saved["step_std"])
        for name in PARAMS:
            setattr(model, name, np.asarray(saved[name], float))
    return model


def _parents_ok(cfg: dict[str, Any]) -> dict[str, str]:
    p = cfg["parents"]
    d153_lock, d153_manifest = _json(p["d153_lock"]), _json(p["d153_manifest"])
    d162_manifest = _json(p["d162_manifest"])
    d165_lock, d165_result, d165_manifest = (_json(p[k]) for k in ("d165_lock", "d165_result", "d165_manifest"))
    if (d153_lock.get("reused_corpus", {}).get("sha256") != sha256_file(ROOT / p["d153_training_rows"])
            or d153_manifest.get("reserve_access") is not False or d153_manifest.get("test_rows_indexed_or_scored") != 0
            or d162_manifest.get("dataset_sha256") != sha256_file(ROOT / p["d162_query_rows"])
            or d162_manifest.get("reserve_access") is not False or d162_manifest.get("test_access") is not False
            or d165_lock.get("schema") != "phase15_d165_action_effect_alignment_lock_v2"
            or d165_result.get("decision_gate", {}).get("passed") is not False
            or d165_manifest.get("result_sha256") != sha256_file(ROOT / p["d165_result"])
            or d165_manifest.get("reserve_access") is not False or d165_manifest.get("test_access") is not False):
        raise D166Error("D153/D162/D165 parent integrity or scope validation failed")
    return {key: sha256_file(ROOT / value) for key, value in p.items()}


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D166Error("commit D166 protocol/source/tests/vault before locking")
    data = _load_data(cfg)
    tr, va, folds = paired_layout_split(data["train_layouts"], cfg["split"]["fold_count"], cfg["split"]["validation_fold"])
    parents = _parents_ok(cfg)
    return {"schema": "phase15_d166_failure_aware_risk_lock_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": _git("rev-parse", "HEAD"),
            "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
            "config_sha256": sha256_file(ROOT / CONFIG), "protocol_sha256": sha256_file(ROOT / cfg["protocol"]),
            "parent_sha256": parents, "d153_state_sha256": sha256_array(data["train_r0"]),
            "d153_summary_sha256": sha256_array(data["train_summary"]),
            "d162_state_sha256": sha256_array(data["query_r0"]),
            "d162_summary_sha256": sha256_array(data["query_summary"]),
            "split": {"fold_sha256": sha256_array(folds), "train_state_count": len(tr),
                      "validation_state_count": len(va),
                      "train_layout_count": len(set(data["train_layouts"][tr].tolist())),
                      "validation_layout_count": len(set(data["train_layouts"][va].tolist()))},
            "scope": cfg["scope"], "outputs": cfg["outputs"]}


def freeze_lock() -> dict[str, Any]:
    cfg = _cfg()
    path = ROOT / cfg["outputs"]["lock"]
    if path.exists():
        raise D166Error("refusing to overwrite D166 lock")
    value = build_lock()
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def run() -> dict[str, Any]:
    cfg = _cfg()
    lock = _json(cfg["outputs"]["lock"])
    if _git("status", "--porcelain"):
        raise D166Error("D166 execution requires a clean committed tree")
    if (lock.get("schema") != "phase15_d166_failure_aware_risk_lock_v1"
            or lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("protocol_sha256") != sha256_file(ROOT / cfg["protocol"])
            or lock.get("source_sha256") != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
            or lock.get("parent_sha256") != _parents_ok(cfg)):
        raise D166Error("D166 protocol/source/parent hash mismatch")
    output = ROOT / cfg["outputs"]["run_dir"]
    if output.exists():
        raise D166Error("refusing to overwrite D166 output")
    data = _load_data(cfg)
    train_idx, val_idx, folds = paired_layout_split(data["train_layouts"], cfg["split"]["fold_count"], cfg["split"]["validation_fold"])
    if lock["split"]["fold_sha256"] != sha256_array(folds):
        raise D166Error("D166 fixed layout split changed")
    output.mkdir(parents=True)
    lock_hash = sha256_file(ROOT / cfg["outputs"]["lock"])
    started = time.perf_counter()
    try:
        classes, _ = encode_targets(data["train_summary"])
        classes = classes.reshape(-1, ACTION_COUNT)
        target = (classes == 2).astype(np.float64)
        train_x = data["train_x"][train_idx].reshape(-1, 65)
        train_y = target[train_idx].reshape(-1)
        train_layouts_by_state = data["train_layouts"][train_idx]
        state_weight = layout_weights(train_layouts_by_state)
        base_weight = np.broadcast_to(state_weight[:, None] / 9.0, (len(train_idx), 9)).reshape(-1)
        vx = data["train_x"][val_idx].reshape(-1, 65)
        vy = target[val_idx].reshape(-1)
        val_weight = np.broadcast_to(layout_weights(data["train_layouts"][val_idx])[:, None] / 9.0,
                                     (len(val_idx), 9)).reshape(-1)
        settings = cfg["model"]
        model = CollisionRiskMLP(tuple(settings["hidden"]), int(settings["seed"]),
                                 float(settings["learning_rate"]), float(settings["l2"]),
                                 float(settings["beta1"]), float(settings["beta2"]), float(settings["epsilon"]))
        model.fit(train_x, train_y, base_weight, (vx, vy, val_weight), settings)
        validation_probability = model.predict(vx)
        threshold_cfg = cfg["threshold"]
        threshold = select_threshold(validation_probability, vy, val_weight, threshold_cfg["grid"],
                                     float(threshold_cfg["minimum_validation_collision_recall"]))
        threshold_metric = risk_metrics(validation_probability, vy, val_weight, threshold["threshold"])
        np.savez_compressed(output / "risk_head.npz", x_mean=model.x_mean, x_std=model.x_std,
                            best_epoch=np.asarray(model.best_epoch), **{name: getattr(model, name) for name in PARAMS})

        query_target = (data["query_classes"] == 2)
        query_x = data["query_x"].reshape(-1, 65)
        risk_probability = model.predict(query_x).reshape(len(data["query_x"]), ACTION_COUNT)
        query_weights = np.broadcast_to(layout_weights(data["query_layouts"])[:, None] / 9.0,
                                        (len(data["query_layouts"]), 9))
        query_risk_metrics = risk_metrics(risk_probability, query_target, query_weights,
                                          threshold["threshold"])
        latency_batch = data["query_x"][:min(32, len(data["query_x"]))].reshape(-1, 65)
        for _ in range(3):
            model.predict(latency_batch)
        latency_samples = []
        for _ in range(20):
            tick = time.perf_counter_ns()
            model.predict(latency_batch)
            latency_samples.append((time.perf_counter_ns() - tick) / (len(data["query_x"][:32]) * 1e6))

        settings165 = yaml.safe_load((ROOT / cfg["parents"]["d155_stage_c_config"]).read_text(encoding="utf-8"))["predictors"]["mlp"]
        effect_predictor = _effect_predictor(ROOT / cfg["parents"]["d165_effect_checkpoint"], settings165)
        p_effect, s_effect = effect_predictor.predict(data["query_x"])
        stage_cfg = yaml.safe_load((ROOT / cfg["parents"]["d155_stage_c_config"]).read_text(encoding="utf-8"))
        metric_cfg = stage_cfg["metrics"]
        predicted_head = _load_student("r0_action_predicted_outcome_steps",
                                       ROOT / cfg["parents"]["d161_predicted_checkpoint"], data["train_r0"])
        actions, _ = _action_inputs(data["query_r0"])
        consequence, mask = _feature_set(p_effect, s_effect, "r0_action_predicted_outcome_steps")
        utility, _ = predicted_head.predict(data["query_r0"], actions, consequence, mask)
        preference_score = -utility
        policy_score = apply_risk_veto(preference_score, risk_probability, threshold["threshold"])
        risk_policy_score = _candidate_metrics(data["query_regret"], policy_score, data["query_summary"],
                                               data["query_layouts"], metric_cfg)
        scorer_only = _candidate_metrics(data["query_regret"], preference_score, data["query_summary"],
                                         data["query_layouts"], metric_cfg)
        # Frozen R0 and exact-consequence references are fixed, not refit.
        zero_p, zero_s = np.zeros_like(data["query_steps"][:, :, None]).repeat(3, axis=2), np.zeros_like(data["query_steps"])
        exact_p, exact_s = _probabilities_from_exact(data["query_summary"])
        from consequence_lab.phase15_d158 import _feature_set as make_features
        r0_head = _load_student("r0_action_only", ROOT / cfg["parents"]["d161_r0_checkpoint"], data["train_r0"])
        oracle_head = _load_student("r0_action_exact_outcome_steps_oracle",
                                    ROOT / cfg["parents"]["d161_oracle_checkpoint"], data["train_r0"])
        r0_consequence, r0_mask = make_features(zero_p, zero_s, "r0_action_only")
        exact_consequence, exact_mask = make_features(exact_p, exact_s, "r0_action_exact_outcome_steps_oracle")
        r0_utility, _ = r0_head.predict(data["query_r0"], actions, r0_consequence, r0_mask)
        oracle_utility, _ = oracle_head.predict(data["query_r0"], actions, exact_consequence, exact_mask)
        r0_metrics = _candidate_metrics(data["query_regret"], -r0_utility, data["query_summary"],
                                        data["query_layouts"], metric_cfg)
        oracle_metrics = _candidate_metrics(data["query_regret"], -oracle_utility, data["query_summary"],
                                            data["query_layouts"], metric_cfg)
        gate_cfg = cfg["evaluation"]
        collision_selection_gain = scorer_only["collision_action_selection_rate"] - risk_policy_score["collision_action_selection_rate"]
        cat_gain = scorer_only["catastrophic_fraction_layout_macro"] - risk_policy_score["catastrophic_fraction_layout_macro"]
        regret_vs_r0 = risk_policy_score["layout_macro_normalized_selected_regret"] - r0_metrics["layout_macro_normalized_selected_regret"]
        gate = (threshold["feasible"]
                and collision_selection_gain >= gate_cfg["minimum_collision_selection_reduction_vs_policy_a"]
                and cat_gain >= gate_cfg["minimum_catastrophic_fraction_reduction_vs_policy_a"]
                and regret_vs_r0 <= gate_cfg["maximum_macro_regret_vs_R0_plus"])
        result = {"schema": "phase15_d166_failure_aware_risk_result_v1",
                  "status": "completed_exploratory_failure_aware_risk_screen",
                  "source_commit": lock["source_commit"], "lock_sha256": lock_hash,
                  "split": lock["split"], "dataset": {"d153_rows": len(data["train_r0"]),
                      "d153_layouts": len(set(data["train_layouts"])), "d162_rows": len(data["query_r0"]),
                      "d162_layouts": len(set(data["query_layouts"])), "layout_overlap": 0,
                      "d153_sha256": lock["parent_sha256"]["d153_training_rows"],
                      "d162_sha256": lock["parent_sha256"]["d162_query_rows"]},
                  "training": {"best_epoch": model.best_epoch, "epochs_run": model.epochs_run,
                      "validation_asymmetric_bce": model.validation_loss, "parameter_count": model.parameter_count,
                      "training_collision_rate": float(np.mean(train_y)),
                      "positive_sampling_multiplier": settings["positive_sampling_multiplier"],
                      "positive_loss_weight": settings["positive_collision_loss_weight"],
                      "history": model.history, "checkpoint_sha256": sha256_file(output / "risk_head.npz")},
                  "inference_latency_ms_per_decision": {"median": float(np.median(latency_samples)),
                      "p95": float(np.quantile(latency_samples, .95)),
                      "measurement": "CPU NumPy risk head only; batch of up to 32 states, nine actions each"},
                  "threshold_selection": {**threshold, "validation_metrics": threshold_metric,
                      "rule": threshold_cfg["selection"]},
                  "risk_metrics": {"internal_validation": risk_metrics(validation_probability, vy, val_weight,
                                                                            threshold["threshold"]),
                                   "d162_exploratory": query_risk_metrics},
                  "decision_metrics": {"policy_a_scorer_only": scorer_only,
                                       "policy_b_risk_veto": risk_policy_score,
                                       "frozen_R0": r0_metrics, "exact_consequence_oracle": oracle_metrics},
                  "advancement_gate": {"collision_selection_reduction": collision_selection_gain,
                      "catastrophic_fraction_reduction": cat_gain, "risk_macro_regret_minus_R0": regret_vs_r0,
                      "passed": bool(gate), "thresholds": gate_cfg},
                  "interpretation": ("risk veto passes the frozen exploratory gate; a fresh-layout D167 protocol may be prepared"
                                     if gate else "risk veto fails the frozen exploratory gate; do not open D167"),
                  "scope": {"new_labels": False, "simulator_calls": 0, "planner_calls": 0,
                            "new_layouts": 0, "closed_loop": False, "dagger": False,
                            "world_model_training": False, "jepa": False,
                            "reserve_access": False, "test_access": False},
                  "wall_time_seconds": time.perf_counter() - started,
                  "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}}
        path = output / "result.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        report_path = ROOT / cfg["outputs"]["report"]
        report_path.write_text(render_report(result), encoding="utf-8")
        manifest = {"schema": "phase15_d166_failure_aware_risk_manifest_v1", "status": result["status"],
                    "source_commit": lock["source_commit"], "lock_sha256": lock_hash,
                    "result_sha256": sha256_file(path), "report_sha256": sha256_file(report_path),
                    "risk_head_sha256": result["training"]["checkpoint_sha256"],
                    "parent_sha256": lock["parent_sha256"], "simulator_calls": 0, "planner_calls": 0,
                    "reserve_access": False, "test_access": False, "closed_loop": False, "dagger": False}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        (output / "abort.json").write_text(json.dumps({"status": "aborted", "error_type": type(exc).__name__,
                                                        "error": str(exc), "lock_sha256": lock_hash,
                                                        "reserve_access": False, "test_access": False}, indent=2) + "\n",
                                       encoding="utf-8")
        raise


def render_report(result: dict[str, Any]) -> str:
    risk, decision, gate = result["risk_metrics"], result["decision_metrics"], result["advancement_gate"]
    lines = ["# D166 — Failure-aware collision prediction", "", "**Status:** exploratory offline screen; D162 is reused.", "",
             f"Threshold selected on D153 fold-0 only: `{result['threshold_selection']['threshold']:.2f}`; feasible recall constraint: `{result['threshold_selection']['feasible']}`.", "",
             "## Collision metrics", "", "| Split | Recall | FNR | Precision | PR-AUC |", "|---|---:|---:|---:|---:|"]
    for split, values in risk.items():
        lines.append(f"| {split} | {values['recall']:.4f} | {values['false_negative_rate']:.4f} | {values['precision']:.4f} | {values['pr_auc']:.4f} |")
    lines += ["", "## Decision comparison", "", "| Policy | Macro regret | Catastrophic | Collision selected | Safe-vs-collision accuracy |", "|---|---:|---:|---:|---:|"]
    for name, values in decision.items():
        lines.append(f"| {name} | {values['layout_macro_normalized_selected_regret']:.5f} | {values['catastrophic_fraction_layout_macro']:.5f} | {values['collision_action_selection_rate']:.5f} | {values['safe_vs_collision_pair_accuracy']} |")
    lines += ["", "## Advancement", "", f"Collision-selection reduction: {gate['collision_selection_reduction']:.5f}; catastrophic-fraction reduction: {gate['catastrophic_fraction_reduction']:.5f}; risk regret minus R0: {gate['risk_macro_regret_minus_R0']:.5f}; gate: **{'PASS' if gate['passed'] else 'FAIL'}**.", "", result["interpretation"], "", "No new simulator/planner labels, layouts, closed-loop, DAgger, JEPA, reserve or test data.", "", f"Lock SHA-256: {result['lock_sha256']}.", ""]
    return "\n".join(lines)
