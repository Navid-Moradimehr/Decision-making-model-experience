"""D165 paired absolute-vs-intervention-effect compact consequence learning."""

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
from consequence_lab.phase15_d155_stage_c import encode_targets
from consequence_lab.phase15_d158 import (
    _action_inputs, _candidate_metrics, _feature_set, _probabilities_from_exact,
)
from consequence_lab.phase15_d162 import _load_student
from consequence_lab.phase15_d164 import canonical_agent_relative, normalized_action_vectors
from consequence_lab.phase15_decision_aligned import regrets_from_exact_summaries, sha256_array
from consequence_lab.phase15_learning import layout_weights

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d165_action_effect_alignment.yaml"
SOURCE_FILES = (
    CONFIG, "reports/phase15_d165_protocol.md", "reports/phase15_d165_protocol_amendment_v1.md",
    "reports/phase15_d165_prelock_and_metric_abort.md", "src/consequence_lab/phase15_d165.py",
    "scripts/freeze_phase15_d165_lock.py", "scripts/run_phase15_d165.py",
    "tests/test_phase15_d165.py", "src/consequence_lab/phase15_d164.py",
    "src/consequence_lab/phase15_d162.py", "src/consequence_lab/phase15_d158.py",
    "src/consequence_lab/phase15_d155_stage_c.py", "src/consequence_lab/phase15_d157.py",
    "src/consequence_lab/phase15_decision_aligned.py", "src/consequence_lab/phase15_learning.py",
    "src/consequence_lab/phase14.py", "configs/phase15_d155_stage_c.yaml",
    "second-brain/00_Project_Home.md", "second-brain/14_Decisions.md",
    "second-brain/16_Experiment_Log.md",
    "second-brain/phases/Phase_15_Decision_Aligned_Target_Study.md",
)
PARAMS = ("w1", "b1", "w2", "b2", "w3", "b3")


class D165Error(RuntimeError):
    """D165 provenance or model-contract error."""


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D165Error(f"git {' '.join(args)} failed") from exc


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    return exp / exp.sum(axis=-1, keepdims=True)


def paired_layout_split(layouts: np.ndarray, fold_count: int = 5,
                        validation_fold: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(layouts).astype(str)
    folds = np.asarray([int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) % fold_count
                        for x in labels], dtype=np.int64)
    train, validation = np.flatnonzero(folds != validation_fold), np.flatnonzero(folds == validation_fold)
    if not len(train) or not len(validation):
        raise ValueError("D165 layout split must have nonempty train and validation folds")
    if set(labels[train].tolist()) & set(labels[validation].tolist()):
        raise ValueError("D165 layout split leakage")
    return train, validation, folds


def pair_importance(regret: np.ndarray, classes: np.ndarray, safe_max: float = .125,
                    catastrophic_min: float = .5, gap_scale: float = .5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(regret, dtype=np.float64)
    outcome = np.asarray(classes, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 9 or outcome.shape != values.shape:
        raise ValueError("D165 pair targets must be aligned [N,9]")
    i, j = np.triu_indices(values.shape[1], 1)
    gap = np.minimum(np.abs(values[:, i] - values[:, j]) / gap_scale, 1.0)
    safe_cat = (((values[:, i] <= safe_max) & (values[:, j] >= catastrophic_min))
                | ((values[:, j] <= safe_max) & (values[:, i] >= catastrophic_min)))
    collision_pair = (outcome[:, i] == 2) != (outcome[:, j] == 2)
    weight = gap * np.where(safe_cat, 2.0, 1.0) * np.where(collision_pair, 2.0, 1.0)
    # Keep explicitly safety-relevant pairs represented even when regret margins are modest.
    weight = np.where(collision_pair | safe_cat, np.maximum(weight, .25), weight)
    return i, j, weight


class ActionEffectMLP:
    """Shared state/action predictor; effect loss couples actions within each state."""

    def __init__(self, hidden: tuple[int, int] = (32, 32), seed: int = 16551,
                 learning_rate: float = .002, l2: float = 1e-5,
                 beta1: float = .9, beta2: float = .999, epsilon: float = 1e-8) -> None:
        self.hidden, self.seed, self.lr, self.l2 = hidden, seed, learning_rate, l2
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0.0)
        return h2 @ self.w3 + self.b3, (x, h1, h2), (z1, z2)

    def _backward(self, grad: np.ndarray, cache: tuple[np.ndarray, ...],
                  pre: tuple[np.ndarray, ...]) -> list[np.ndarray]:
        x, h1, h2 = cache
        z1, z2 = pre
        gw3 = h2.T @ grad + self.l2 * self.w3
        gb3 = grad.sum(axis=0)
        dz2 = (grad @ self.w3.T) * (z2 > 0.0)
        gw2 = h1.T @ dz2 + self.l2 * self.w2
        gb2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.w2.T) * (z1 > 0.0)
        gw1 = x.T @ dz1 + self.l2 * self.w1
        gb1 = dz1.sum(axis=0)
        return [gw1, gb1, gw2, gb2, gw3, gb3]

    def fit(self, x: np.ndarray, classes: np.ndarray, steps: np.ndarray, regret: np.ndarray,
            state_weights: np.ndarray, *, effect: bool, config: dict[str, Any],
            validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            ) -> "ActionEffectMLP":
        features = np.asarray(x, dtype=np.float64)
        klass = np.asarray(classes, dtype=np.int64)
        step_target = np.asarray(steps, dtype=np.float64)
        regrets = np.asarray(regret, dtype=np.float64)
        state_w = np.asarray(state_weights, dtype=np.float64)
        n = len(features)
        if (features.shape != (n, 9, 65) or klass.shape != (n, 9) or step_target.shape != (n, 9)
                or regrets.shape != (n, 9) or state_w.shape != (n,) or np.any(state_w < 0)
                or not all(np.isfinite(a).all() for a in (features, step_target, regrets, state_w))):
            raise ValueError("invalid D165 fit arrays")
        vx, vc, vs, vr, vw = validation
        vx, vc, vs, vr, vw = (np.asarray(a) for a in (vx, vc, vs, vr, vw))
        if vx.shape[1:] != (9, 65) or vc.shape != vx.shape[:2] or vs.shape != vc.shape or vr.shape != vc.shape or vw.shape != (len(vx),):
            raise ValueError("invalid D165 validation arrays")

        flat = features.reshape(-1, 65)
        self.x_mean, self.x_std = flat.mean(axis=0), flat.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        step_scale = np.broadcast_to(state_w[:, None] / 9.0, (n, 9))
        self.step_mean = float(np.sum(step_scale * step_target))
        self.step_std = float(np.sqrt(np.sum(step_scale * (step_target - self.step_mean) ** 2)))
        if self.step_std < 1e-8:
            self.step_std = 1.0
        train_x = (features - self.x_mean) / self.x_std
        val_x = (vx - self.x_mean) / self.x_std
        target_step_z = (step_target - self.step_mean) / self.step_std
        val_step_z = (vs - self.step_mean) / self.step_std
        output_weight = state_w[:, None] / 9.0
        val_weight = vw[:, None] / 9.0
        onehot = np.eye(3)[klass]

        h1, h2 = self.hidden
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / 65), (65, h1))
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / h1), (h1, h2))
        self.w3 = rng.normal(0.0, np.sqrt(1.0 / h2), (h2, 4))
        self.b1, self.b2, self.b3 = np.zeros(h1), np.zeros(h2), np.zeros(4)
        params = [getattr(self, name) for name in PARAMS]
        first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        pi, pj, pweight = pair_importance(regrets, klass, config["safe_regret_max"],
                                          config["catastrophic_regret_min"], config["pair_regret_gap_scale"])
        pweight = pweight / np.maximum(pweight.sum(axis=1, keepdims=True), 1e-12)
        trace: list[dict[str, float | int]] = []
        best_params = None
        best_criterion, stale, best_epoch = float("inf"), 0, 0
        max_epochs, patience = int(config["max_epochs"]), int(config["patience"])
        for epoch in range(max_epochs):
            raw, cache, pre = self._forward(train_x.reshape(-1, 65))
            raw = raw.reshape(n, 9, 4)
            prob = _softmax(raw[:, :, :3])
            grad = np.zeros_like(raw)
            abs_ce = -np.sum(output_weight * np.log(np.maximum(prob[np.arange(n)[:, None],
                                                                         np.arange(9)[None, :], klass], 1e-12)))
            grad[:, :, :3] = output_weight[:, :, None] * (prob - onehot)
            abs_steps = float(np.sum(output_weight * (raw[:, :, 3] - target_step_z) ** 2))
            grad[:, :, 3] = 2.0 * output_weight * (raw[:, :, 3] - target_step_z)
            effect_loss = 0.0
            if effect:
                delta_prob = prob[:, pi] - prob[:, pj]
                delta_onehot = onehot[:, pi] - onehot[:, pj]
                residual_p = delta_prob - delta_onehot
                delta_step = raw[:, pi, 3] - raw[:, pj, 3]
                residual_s = delta_step - (target_step_z[:, pi] - target_step_z[:, pj])
                pair_w = state_w[:, None] * pweight
                outcome_loss = .5 * float(np.sum(pair_w * np.mean(residual_p**2, axis=2)))
                step_loss = .5 * float(np.sum(pair_w * residual_s**2))
                effect_loss = outcome_loss + step_loss
                grad_prob = np.zeros_like(prob)
                grad_step = np.zeros((n, 9), dtype=np.float64)
                gp = pair_w[:, :, None] * residual_p / 3.0
                gs = pair_w * residual_s
                for k in range(len(pi)):
                    grad_prob[:, pi[k]] += gp[:, k]
                    grad_prob[:, pj[k]] -= gp[:, k]
                    grad_step[:, pi[k]] += gs[:, k]
                    grad_step[:, pj[k]] -= gs[:, k]
                grad_logits = prob * (grad_prob - np.sum(grad_prob * prob, axis=2, keepdims=True))
                grad[:, :, :3] += grad_logits
                grad[:, :, 3] += grad_step
            gradients = self._backward(grad.reshape(-1, 4), cache, pre)
            for idx, (param, g) in enumerate(zip(params, gradients, strict=True)):
                first[idx] = self.beta1 * first[idx] + (1.0 - self.beta1) * g
                second[idx] = self.beta2 * second[idx] + (1.0 - self.beta2) * g * g
                t = epoch + 1
                param -= self.lr * (first[idx] / (1.0 - self.beta1**t)) / (
                    np.sqrt(second[idx] / (1.0 - self.beta2**t)) + self.epsilon)

            val_raw = self._forward(val_x.reshape(-1, 65))[0].reshape(len(vx), 9, 4)
            val_prob = _softmax(val_raw[:, :, :3])
            selected_val_prob = np.take_along_axis(val_prob, vc.astype(int)[..., None], axis=2)[..., 0]
            val_ce = -float(np.sum(val_weight * np.log(np.maximum(selected_val_prob, 1e-12))))
            val_mse = float(np.sum(val_weight * (val_raw[:, :, 3] - val_step_z) ** 2))
            criterion = val_ce + val_mse
            if epoch == 0 or (epoch + 1) % 25 == 0:
                trace.append({"epoch": epoch + 1, "train_outcome_ce": float(abs_ce),
                              "train_steps_mse": abs_steps, "train_effect_loss": effect_loss,
                              "validation_absolute_loss": criterion})
            if criterion < best_criterion - 1e-9:
                best_criterion, best_epoch, stale = criterion, epoch + 1, 0
                best_params = [p.copy() for p in params]
            else:
                stale += 1
            if stale >= patience:
                break
        if best_params is None:
            raise RuntimeError("D165 failed to checkpoint")
        for name, value in zip(PARAMS, best_params, strict=True):
            setattr(self, name, value)
        self.best_epoch, self.epochs_run = best_epoch, epoch + 1
        self.validation_loss, self.training_trace = best_criterion, trace
        self.parameter_count = int(sum(p.size for p in params))
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (9, 65) or not np.isfinite(values).all():
            raise ValueError("D165 prediction inputs must be finite [N,9,65]")
        raw = self._forward(((values - self.x_mean) / self.x_std).reshape(-1, 65))[0]
        raw = raw.reshape(len(values), 9, 4)
        probabilities = _softmax(raw[:, :, :3])
        steps = np.clip(raw[:, :, 3] * self.step_std + self.step_mean, 0.0, 1.0)
        return probabilities, steps


def _load_data(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["parents"]
    with np.load(ROOT / p["d153_training_rows"], allow_pickle=False) as arr:
        train_r0 = np.asarray(arr["x_R0"], dtype=np.float64)
        train_summary = np.asarray(arr["summary_R4"], dtype=np.float64)
        train_layouts = np.asarray(arr["layouts"]).astype(str)
    with np.load(ROOT / p["d162_query_rows"], allow_pickle=False) as arr:
        query_r0 = np.asarray(arr["x_R0"], dtype=np.float64)
        query_summary = np.asarray(arr["summary_R4"], dtype=np.float64)
        query_layouts = np.asarray(arr["layouts"]).astype(str)
    if (train_r0.shape != (1365, 61) or train_summary.shape != (1365, 9, 7)
            or len(set(train_layouts.tolist())) != 140 or query_r0.shape != (1046, 61)
            or query_summary.shape != (1046, 9, 7) or len(set(query_layouts.tolist())) != 100
            or set(train_layouts.tolist()) & set(query_layouts.tolist())):
        raise D165Error("D153/D162 saved-data shape or layout-boundary mismatch")
    from consequence_lab.phase14c import build_feature_matrix
    for r0, name in ((train_r0, "D153"), (query_r0, "D162")):
        if not np.allclose(build_feature_matrix(r0[:, :52], "raw52_plus_physics9"), r0,
                           atol=1e-6, rtol=0.0):
            raise D165Error(f"{name} deployable R0 reconstruction failed")
        if not np.isfinite(r0).all():
            raise D165Error(f"{name} has non-finite student state")
    train_canon = canonical_agent_relative(train_r0[:, :52], train_r0[:, 52:])
    query_canon = canonical_agent_relative(query_r0[:, :52], query_r0[:, 52:])
    actions = normalized_action_vectors()
    train_x = np.concatenate((np.repeat(train_canon[:, None, :], 9, axis=1),
                              np.broadcast_to(actions, (len(train_r0), 9, 2))), axis=2)
    query_x = np.concatenate((np.repeat(query_canon[:, None, :], 9, axis=1),
                              np.broadcast_to(actions, (len(query_r0), 9, 2))), axis=2)
    train_classes, train_steps = encode_targets(train_summary)
    query_classes, query_steps = encode_targets(query_summary)
    train_classes, train_steps = train_classes.reshape(-1, 9), train_steps.reshape(-1, 9)
    query_classes, query_steps = query_classes.reshape(-1, 9), query_steps.reshape(-1, 9)
    train_regret = regrets_from_exact_summaries(train_summary)
    query_regret = regrets_from_exact_summaries(query_summary)
    if not np.isfinite(train_x).all() or not np.isfinite(query_x).all():
        raise D165Error("non-finite D165 action-conditioned input")
    return {"train_r0": train_r0, "train_x": train_x, "train_summary": train_summary,
            "train_classes": train_classes, "train_steps": train_steps, "train_regret": train_regret,
            "train_layouts": train_layouts, "query_r0": query_r0, "query_x": query_x,
            "query_summary": query_summary, "query_classes": query_classes,
            "query_steps": query_steps, "query_regret": query_regret, "query_layouts": query_layouts}


def _config_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D165Error("commit D165 protocol/source/tests/vault before locking")
    if (cfg.get("schema") != "phase15_d165_action_effect_alignment_v2"
            or cfg.get("status") != "locked_v2_metric_broadcast_correction"):
        raise D165Error("D165 v2 metric-correction config/status mismatch")
    d164_lock = _json(cfg["parents"]["d164_lock"])
    d164_result = _json(cfg["parents"]["d164_result"])
    d164_manifest = _json(cfg["parents"]["d164_manifest"])
    d162_result = _json(cfg["parents"]["d162_result"])
    d162_manifest = _json(cfg["parents"]["d162_manifest"])
    d153_lock = _json(cfg["parents"]["d153_lock"])
    d153_manifest = _json(cfg["parents"]["d153_manifest"])
    if (d164_lock.get("schema") != "phase15_d164_representation_symmetry_lock_v1"
            or d164_result.get("canonical_vs_r0", {}).get("gate_passed") is not True
            or d164_manifest.get("lock_sha256") != sha256_file(ROOT / cfg["parents"]["d164_lock"])
            or d164_manifest.get("result_sha256") != sha256_file(ROOT / cfg["parents"]["d164_result"])
            or d162_manifest.get("lock_sha256") != sha256_file(ROOT / cfg["parents"]["d162_lock"])
            or d162_manifest.get("dataset_sha256") != sha256_file(ROOT / cfg["parents"]["d162_query_rows"])
            or d162_manifest.get("reserve_access") is not False
            or d162_manifest.get("test_access") is not False
            or d162_result.get("scope", {}).get("reserve_access") is not False
            or d153_lock.get("reused_corpus", {}).get("sha256") != sha256_file(ROOT / cfg["parents"]["d153_training_rows"])
            or d153_manifest.get("lock_sha256") != sha256_file(ROOT / cfg["parents"]["d153_lock"])
            or d153_manifest.get("reserve_access") is not False
            or d153_manifest.get("test_rows_indexed_or_scored") != 0):
        raise D165Error("D164 advancement gate or D162 scope verification failed")
    data = _load_data(cfg)
    tr, va, folds = paired_layout_split(data["train_layouts"], cfg["data"]["fold_count"],
                                        cfg["data"]["internal_validation_fold"])
    return {"schema": "phase15_d165_action_effect_alignment_lock_v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": _git("rev-parse", "HEAD"),
            "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
            "config_sha256": sha256_file(ROOT / CONFIG),
            "protocol_sha256": sha256_file(ROOT / cfg["protocol"]),
            "parent_sha256": _config_hashes(cfg),
            "train_r0_sha256": sha256_array(data["train_r0"]),
            "train_consequence_sha256": sha256_array(data["train_summary"]),
            "train_layout_sha256": sha256_array(data["train_layouts"]),
            "query_r0_sha256": sha256_array(data["query_r0"]),
            "query_consequence_sha256": sha256_array(data["query_summary"]),
            "query_layout_sha256": sha256_array(data["query_layouts"]),
            "split": {"folds_sha256": sha256_array(folds), "train_state_count": len(tr),
                      "validation_state_count": len(va), "train_layout_count": len(set(data["train_layouts"][tr])),
                      "validation_layout_count": len(set(data["train_layouts"][va])),
                      "train_layouts": sorted(set(data["train_layouts"][tr].tolist())),
                      "validation_layouts": sorted(set(data["train_layouts"][va].tolist()))},
            "scope": cfg["scope"], "outputs": cfg["outputs"]}


def freeze_lock() -> dict[str, Any]:
    cfg = _cfg()
    path = ROOT / cfg["outputs"]["lock"]
    if path.exists():
        raise D165Error("refusing to overwrite D165 lock")
    lock = build_lock()
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def _validate_lock() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    lock = _json(cfg["outputs"]["lock"])
    if _git("status", "--porcelain"):
        raise D165Error("D165 run requires clean committed sources and a frozen lock")
    if (lock.get("schema") != "phase15_d165_action_effect_alignment_lock_v2"
            or lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("protocol_sha256") != sha256_file(ROOT / cfg["protocol"])
            or lock.get("source_sha256") != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
            or lock.get("parent_sha256") != _config_hashes(cfg)):
        raise D165Error("D165 source/protocol/parent hash mismatch")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    output = ROOT / cfg["outputs"]["run_dir"]
    if output.exists():
        raise D165Error("refusing to rerun/overwrite D165 output")
    data = _load_data(cfg)
    tr, va, folds = paired_layout_split(data["train_layouts"], cfg["data"]["fold_count"],
                                        cfg["data"]["internal_validation_fold"])
    if (lock.get("train_r0_sha256") != sha256_array(data["train_r0"])
            or lock.get("train_consequence_sha256") != sha256_array(data["train_summary"])
            or lock.get("query_r0_sha256") != sha256_array(data["query_r0"])
            or lock.get("query_consequence_sha256") != sha256_array(data["query_summary"])
            or lock.get("split", {}).get("folds_sha256") != sha256_array(folds)
            or lock.get("split", {}).get("train_state_count") != len(tr)
            or lock.get("split", {}).get("validation_state_count") != len(va)):
        raise D165Error("D165 data or split fingerprint mismatch")
    return cfg, lock, data


def _outcome_metrics(prob: np.ndarray, classes: np.ndarray, steps: np.ndarray,
                     step_target: np.ndarray, layouts: np.ndarray) -> dict[str, Any]:
    p = np.asarray(prob, dtype=float)
    y = np.asarray(classes, dtype=int)
    pred = np.argmax(p, axis=2)
    weights = np.broadcast_to(layout_weights(layouts)[:, None] / 9.0, (len(y), 9))
    collision = y == 2
    selected_collision = pred == 2
    tp = float(np.sum(weights * (collision & selected_collision)))
    fp = float(np.sum(weights * (~collision & selected_collision)))
    fn = float(np.sum(weights * (collision & ~selected_collision)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    order = np.argsort(-p[:, :, 2].reshape(-1), kind="stable")
    flat_y, flat_w = collision.reshape(-1), weights.reshape(-1)
    tpw = np.cumsum(flat_w[order] * flat_y[order])
    fpw = np.cumsum(flat_w[order] * ~flat_y[order])
    total_positive = max(float(np.sum(flat_w * flat_y)), 1e-12)
    recall_at_rank = tpw / total_positive
    precision_at_rank = tpw / np.maximum(tpw + fpw, 1e-12)
    pr_auc = float(np.sum(np.diff(np.concatenate(([0.0], recall_at_rank))) * precision_at_rank))
    selected_probability = np.take_along_axis(p, y[..., None], axis=2)[..., 0]
    ce = -float(np.sum(weights * np.log(np.maximum(selected_probability, 1e-12))))
    return {"outcome_cross_entropy": ce, "outcome_accuracy": float(np.sum(weights * (pred == y))),
            "collision_recall": recall, "collision_false_negative_rate": 1.0 - recall,
            "collision_precision": precision, "collision_pr_auc": pr_auc,
            "step_rmse": float(np.sqrt(np.sum(weights * (steps - step_target) ** 2))),
            "step_mae": float(np.sum(weights * np.abs(steps - step_target))),
            "collision_count_weighted": float(np.sum(weights * collision)),
            "predicted_collision_rate": float(np.sum(weights * selected_collision))}


def _effect_error(prob: np.ndarray, steps: np.ndarray, target_prob: np.ndarray,
                  target_steps: np.ndarray) -> dict[str, float]:
    i, j = np.triu_indices(9, 1)
    dp = (prob[:, i] - prob[:, j]) - (target_prob[:, i] - target_prob[:, j])
    ds = (steps[:, i] - steps[:, j]) - (target_steps[:, i] - target_steps[:, j])
    return {"outcome_difference_rmse": float(np.sqrt(np.mean(dp**2))),
            "steps_difference_rmse": float(np.sqrt(np.mean(ds**2)))}


def _score_with_head(name: str, model: Any, states: np.ndarray, consequence_prob: np.ndarray,
                     consequence_steps: np.ndarray, summaries: np.ndarray, regret: np.ndarray,
                     layouts: np.ndarray, metric_cfg: dict[str, Any]) -> dict[str, Any]:
    actions, _ = _action_inputs(states)
    consequence, mask = _feature_set(consequence_prob, consequence_steps, name)
    utility, _ = model.predict(states, actions, consequence, mask)
    return _candidate_metrics(regret, -utility, summaries, layouts, metric_cfg)


def _evaluate_predictions(data: dict[str, Any], model24: ActionEffectMLP,
                          model_effect: ActionEffectMLP, cfg: dict[str, Any]) -> dict[str, Any]:
    stage_cfg = yaml.safe_load((ROOT / cfg["parents"]["d155_stage_c_config"]).read_text(encoding="utf-8"))
    metric_cfg = stage_cfg["metrics"]
    val_rows = data["validation_indices"]
    models = {"absolute": model24, "absolute_plus_effect": model_effect}
    results: dict[str, Any] = {}
    for split_name, state_key, summary_key, regret_key, layouts_key in (
        ("internal_validation", "train_x", "train_summary", "train_regret", "train_layouts"),
        ("d162_exploratory", "query_x", "query_summary", "query_regret", "query_layouts"),
    ):
        row_ids = val_rows if split_name == "internal_validation" else np.arange(len(data["query_x"]))
        x = data[state_key][row_ids]
        summary = data[summary_key][row_ids]
        regret = data[regret_key][row_ids]
        layouts = data[layouts_key][row_ids]
        exact_prob, exact_steps = _probabilities_from_exact(summary)
        local: dict[str, Any] = {}
        for model_name, predictor in models.items():
            pred_prob, pred_steps = predictor.predict(x)
            local[model_name] = {
                "prediction": _outcome_metrics(pred_prob, np.argmax(summary[:, :, [0, 2, 3]], axis=2),
                                               pred_steps, summary[:, :, 4], layouts),
                "effect_error": _effect_error(pred_prob, pred_steps, exact_prob, exact_steps),
                "training": {"best_epoch": predictor.best_epoch, "epochs_run": predictor.epochs_run,
                             "validation_absolute_loss": predictor.validation_loss,
                             "parameter_count": predictor.parameter_count},
            }
            for head_name in ("r0_action_predicted_outcome_steps", "r0_action_exact_outcome_steps_oracle"):
                head = _load_student(head_name, ROOT / cfg["parents"]["d161_predicted_checkpoint" if "predicted" in head_name else "d161_oracle_checkpoint"],
                                     data["train_r0"])
                local[model_name].setdefault("decision", {})[head_name] = _score_with_head(
                    head_name, head, data["query_r0"] if split_name == "d162_exploratory" else data["train_r0"][row_ids],
                    pred_prob, pred_steps, summary, regret, layouts, metric_cfg)
        # Exact oracle and fixed R0 anchors use the same states as this split.
        r0_head = _load_student("r0_action_only", ROOT / cfg["parents"]["d161_r0_checkpoint"], data["train_r0"])
        exact_head = _load_student("r0_action_exact_outcome_steps_oracle",
                                   ROOT / cfg["parents"]["d161_oracle_checkpoint"], data["train_r0"])
        r0_states = data["query_r0"] if split_name == "d162_exploratory" else data["train_r0"][row_ids]
        zeros_prob, zeros_steps = np.zeros_like(exact_prob), np.zeros_like(exact_steps)
        local["frozen_references"] = {
            "R0_direct": _score_with_head("r0_action_only", r0_head, r0_states, zeros_prob,
                                          zeros_steps, summary, regret, layouts, metric_cfg),
            "exact_consequences_oracle": _score_with_head("r0_action_exact_outcome_steps_oracle", exact_head,
                                                          r0_states, exact_prob, exact_steps,
                                                          summary, regret, layouts, metric_cfg),
        }
        results[split_name] = local
    return results


def run() -> dict[str, Any]:
    cfg, lock, data = _validate_lock()
    out = ROOT / cfg["outputs"]["run_dir"]
    out.mkdir(parents=True, exist_ok=False)
    lock_sha = sha256_file(ROOT / cfg["outputs"]["lock"])
    tr, va, folds = paired_layout_split(data["train_layouts"], cfg["data"]["fold_count"],
                                        cfg["data"]["internal_validation_fold"])
    data["validation_indices"] = va
    settings = cfg["model"]
    train_weights = layout_weights(data["train_layouts"][tr])
    val_weights = layout_weights(data["train_layouts"][va])
    validation = (data["train_x"][va], data["train_classes"][va], data["train_steps"][va],
                  data["train_regret"][va], val_weights)
    started = time.perf_counter()
    results: dict[str, Any] = {}
    models = {}
    try:
        for name, use_effect in (("absolute", False), ("absolute_plus_effect", True)):
            model = ActionEffectMLP(tuple(settings["hidden"]), int(settings["seed"]),
                                    float(settings["learning_rate"]), float(settings["l2"]),
                                    float(settings["beta1"]), float(settings["beta2"]),
                                    float(settings["epsilon"]))
            model.fit(data["train_x"][tr], data["train_classes"][tr], data["train_steps"][tr],
                      data["train_regret"][tr], train_weights, effect=use_effect,
                      config={"max_epochs": settings["max_epochs"], "patience": settings["patience"],
                              "safe_regret_max": settings["safe_regret_max"],
                              "catastrophic_regret_min": settings["catastrophic_regret_min"],
                              "pair_regret_gap_scale": settings["pair_regret_gap_scale"]},
                      validation=validation)
            models[name] = model
            probs, pred_steps = model.predict(data["train_x"][va])
            results[name] = {"best_epoch": model.best_epoch, "epochs_run": model.epochs_run,
                             "validation_absolute_loss": model.validation_loss,
                             "training_trace": model.training_trace,
                             "parameter_count": model.parameter_count,
                             "internal_validation_prediction": _outcome_metrics(
                                 probs, data["train_classes"][va], pred_steps, data["train_steps"][va],
                                 data["train_layouts"][va])}
            latency_batch = data["query_x"][:min(32, len(data["query_x"]))]
            for _ in range(3):
                model.predict(latency_batch)
            latency = []
            for _ in range(20):
                tic = time.perf_counter_ns()
                model.predict(latency_batch)
                latency.append((time.perf_counter_ns() - tic) / (len(latency_batch) * 1e6))
            results[name]["inference_latency_ms_per_decision"] = {
                "median": float(np.median(latency)), "p95": float(np.quantile(latency, .95)),
                "measurement": "batch of up to 32 states, CPU NumPy predictor only"}
            ckpt = out / f"{name}.npz"
            np.savez_compressed(ckpt, x_mean=model.x_mean, x_std=model.x_std,
                                step_mean=np.asarray(model.step_mean), step_std=np.asarray(model.step_std),
                                best_epoch=np.asarray(model.best_epoch), **{n: getattr(model, n) for n in PARAMS})
            results[name]["checkpoint_sha256"] = sha256_file(ckpt)
        evaluated = _evaluate_predictions(data, models["absolute"], models["absolute_plus_effect"], cfg)
        # The training set fit gate is a diagnostic, not a reason to tune this locked run.
        for name, model in models.items():
            p, s = model.predict(data["train_x"][tr])
            results[name]["training_fit"] = _outcome_metrics(
                p, data["train_classes"][tr], s, data["train_steps"][tr], data["train_layouts"][tr])
        d162 = _json(cfg["parents"]["d162_result"])
        effect_metrics = evaluated["d162_exploratory"]["absolute_plus_effect"]["decision"][
            "r0_action_predicted_outcome_steps"]
        absolute_metrics = evaluated["d162_exploratory"]["absolute"]["decision"][
            "r0_action_predicted_outcome_steps"]
        gate = cfg["evaluation"]
        regret_gain = (absolute_metrics["layout_macro_normalized_selected_regret"]
                       - effect_metrics["layout_macro_normalized_selected_regret"])
        catastrophic_gain = (absolute_metrics["catastrophic_fraction_layout_macro"]
                             - effect_metrics["catastrophic_fraction_layout_macro"])
        safe_delta = (effect_metrics["safe_vs_collision_pair_accuracy"]
                      - absolute_metrics["safe_vs_collision_pair_accuracy"])
        advance = (regret_gain >= gate["minimum_macro_regret_improvement"]
                   and catastrophic_gain >= gate["minimum_catastrophic_fraction_improvement"]
                   and safe_delta >= -gate["maximum_safe_collision_accuracy_drop"])
        result = {"schema": "phase15_d165_action_effect_alignment_result_v1",
                  "status": "completed_exploratory_paired_offline_action_effect_screen",
                  "source_commit": lock["source_commit"], "lock_sha256": lock_sha,
                  "dataset": {"d153_rows": 1365, "d153_layouts": 140, "d162_rows": 1046,
                              "d162_layouts": 100, "layout_overlap": 0,
                              "d153_sha256": lock["parent_sha256"]["d153_training_rows"],
                              "d162_sha256": lock["parent_sha256"]["d162_query_rows"]},
                  "split": lock["split"], "pairwise_training": {"candidate_pairs_per_state": 36,
                      "same_state_only": True, "importance_formula": "regret-gap clipped at 0.5; x2 safe/catastrophic; x2 collision/noncollision; safety pairs minimum weight .25",
                      "effect_loss_coefficient": settings["effect_coefficient"]},
                  "models": results, "evaluation": evaluated,
                  "d162_historical_reference": d162.get("evaluation", {}).get("model_metrics", {}),
                  "decision_gate": {"macro_regret_gain_absolute_minus_effect": regret_gain,
                                    "catastrophic_gain_absolute_minus_effect": catastrophic_gain,
                                    "safe_collision_accuracy_change": safe_delta,
                                    "passed": bool(advance), "thresholds": gate},
                  "interpretation": ("effect-aligned compact prediction improved the preregistered exploratory D162 decision screen"
                                     if advance else "intervention-effect alignment did not pass the preregistered D162 decision screen"),
                  "scope": {"new_labels": False, "simulator_calls": 0, "planner_calls": 0,
                            "new_layouts": 0, "closed_loop": False, "dagger": False,
                            "world_model_training": False, "jepa": False,
                            "reserve_access": False, "test_access": False},
                  "wall_time_seconds": time.perf_counter() - started,
                  "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}}
        rp = out / "result.json"
        rp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        report = render_report(result)
        report_path = ROOT / cfg["outputs"]["report"]
        report_path.write_text(report, encoding="utf-8")
        manifest = {"schema": "phase15_d165_action_effect_alignment_manifest_v1",
                    "status": result["status"], "source_commit": lock["source_commit"],
                    "lock_sha256": lock_sha, "result_sha256": sha256_file(rp),
                    "report_sha256": sha256_file(report_path), "parent_sha256": lock["parent_sha256"],
                    "planner_calls": 0, "simulator_calls": 0, "reserve_access": False,
                    "test_access": False, "closed_loop": False, "dagger": False}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        (out / "abort.json").write_text(json.dumps({"status": "aborted", "error_type": type(exc).__name__,
                                                       "error": str(exc), "lock_sha256": lock_sha,
                                                       "reserve_access": False, "test_access": False},
                                                      indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D165 — Action-effect alignment", "", "**Status:** completed exploratory offline screen. D162 is reused and not independent confirmation.", "",
             "## Paired models", "", "| Model | Validation outcome CE | Collision recall | Steps RMSE | Best epoch | Parameters |", "|---|---:|---:|---:|---:|---:|"]
    for name, model in result["models"].items():
        m = model["internal_validation_prediction"]
        lines.append(f"| {name} | {m['outcome_cross_entropy']:.5f} | {m['collision_recall']:.4f} | {m['step_rmse']:.5f} | {model['best_epoch']} | {model['parameter_count']} |")
    lines += ["", "## D162 exploratory decision screen", "", "| Model | Frozen scorer | Macro regret | Catastrophic | Epsilon optimal | Safe/collision pairs | Selected collision rate |", "|---|---|---:|---:|---:|---:|---:|"]
    for model_name in ("absolute", "absolute_plus_effect"):
        for head, metrics in result["evaluation"]["d162_exploratory"][model_name]["decision"].items():
            lines.append(f"| {model_name} | {head} | {metrics['layout_macro_normalized_selected_regret']:.5f} | {metrics['catastrophic_fraction_layout_macro']:.5f} | {metrics['epsilon_optimal_rate']:.5f} | {metrics['safe_vs_collision_pair_accuracy']} | {metrics['collision_action_selection_rate']:.5f} |")
    gate = result["decision_gate"]
    lines += ["", "## Advancement gate", "", f"Macro-regret gain: {gate['macro_regret_gain_absolute_minus_effect']:.5f}; catastrophic-fraction gain: {gate['catastrophic_gain_absolute_minus_effect']:.5f}; safe-vs-collision accuracy change: {gate['safe_collision_accuracy_change']}; gate: **{'PASS' if gate['passed'] else 'FAIL'}**.", "", "## Interpretation", "", result["interpretation"], "", "No reserve/test, simulator/planner, closed-loop, or DAgger work was performed. This is an exploratory reused-layout screen.", "", f"Lock SHA-256: {result['lock_sha256']}.", ""]
    return "\n".join(lines)
