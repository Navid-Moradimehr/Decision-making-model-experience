"""D155 exact consequence-subset study using only locked Phase15 artifacts."""

from __future__ import annotations

from collections import defaultdict
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
from consequence_lab.phase15_decision_aligned import (
    _read_json, decision_detail_metrics, regrets_from_exact_summaries,
)
from consequence_lab.phase15_generalization import layout_list_sha256
from consequence_lab.phase15_learning import layout_weights
from consequence_lab.phase15_outcome_summary import restore_decision_head
from consequence_lab.phase15_representation_screen import train_validation_indices

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d155_v2.yaml"
PROTOCOL = "reports/phase15_d155_protocol_amendment_v2.md"
LOCK = "experiments/phase15_d155_v2.lock.json"
SOURCE_FILES = (
    CONFIG, PROTOCOL, "src/consequence_lab/phase15_d155.py",
    "src/consequence_lab/phase15_decision_aligned.py",
    "src/consequence_lab/phase15_representation_screen.py",
    "src/consequence_lab/phase15_outcome_summary.py",
    "src/consequence_lab/phase15_learning.py",
    "src/consequence_lab/inertial_corridor.py",
    "scripts/freeze_phase15_d155_lock.py", "scripts/run_phase15_d155.py",
    "tests/test_phase15_d155.py",
)
GROUP_INDEX = {"outcome": tuple(range(4)), "steps": (4,), "goal_distance": (5,), "effort": (6,)}
FEATURES = (
    ("F0_R0", ()),
    ("F_outcome", ("outcome",)),
    ("F_steps", ("steps",)),
    ("F_goal_distance", ("goal_distance",)),
    ("F_effort", ("effort",)),
    ("F_outcome_steps", ("outcome", "steps")),
    ("F_outcome_steps_distance", ("outcome", "steps", "goal_distance")),
    ("F_full", ("outcome", "steps", "goal_distance", "effort")),
)


class D155Error(RuntimeError):
    """D155 integrity or execution failure."""


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D155Error(f"git {' '.join(args)} failed") from exc


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _hash_parents(cfg: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}


def _validate_parent_chain(cfg: dict[str, Any]) -> dict[str, Any]:
    parent_hash = _hash_parents(cfg)
    d154_lock = _read_json(ROOT / cfg["parents"]["d154_training_lock"])
    d154_result_lock = _read_json(ROOT / cfg["parents"]["d154_result_lock"])
    d154_result = _read_json(ROOT / cfg["parents"]["d154_result"])
    manifest = _read_json(ROOT / cfg["parents"]["d154_manifest"])
    if d154_lock.get("schema") != "phase15_d154_decision_aligned_lock_v2":
        raise D155Error("D154 training lock schema mismatch")
    if d154_result_lock.get("schema") != "phase15_d154_finalize_lock_v2":
        raise D155Error("D154 result lock schema mismatch")
    if (d154_result.get("status") != "exploratory_recomputed_from_preserved_fits"
            or d154_result.get("training_repeated") is not False
            or d154_result.get("scope", {}).get("reserve_access") is not False
            or d154_result.get("scope", {}).get("test_rows_scored") != 0):
        raise D155Error("D154 result is not the closed, train/reused-validation result")
    if (manifest.get("result_sha256") != sha256_file(ROOT / cfg["parents"]["d154_result"])
            or manifest.get("reserve_access") is not False
            or manifest.get("test_rows_scored") != 0):
        raise D155Error("D154 result manifest/hash/scope mismatch")
    for lock_path, lock_obj in ((cfg["parents"]["d154_training_lock"], d154_lock),
                                (cfg["parents"]["d154_result_lock"], d154_result_lock)):
        if not lock_obj.get("source_commit"):
            raise D155Error(f"parent lock has no source commit: {lock_path}")
    d153_manifest = _read_json(ROOT / cfg["parents"]["d153_manifest"])
    if (d153_manifest.get("status") != "completed_exploratory_scaling"
            or d153_manifest.get("test_rows_indexed_or_scored") != 0
            or d153_manifest.get("reserve_access") is not False):
        raise D155Error("D153 corpus manifest is not closed train-only evidence")
    d153_result = _read_json(ROOT / cfg["parents"]["d153_result"])
    if d153_result.get("interpretation") != "mixed_or_inconclusive":
        raise D155Error("D153 interpretation changed from the closed result")
    if set(d154_result["references"]) != {
            "direct_D150_R0", "D153_predicted_R4_then_D150_head", "exact_R4_then_D150_head_oracle"}:
        raise D155Error("D154 reference set is incomplete")
    if max((abs(float(difference))
            for row in d154_result["frozen_reference_reproduction_max_abs_differences"].values()
            for difference in row.values()), default=0.0) > 1e-10:
        raise D155Error("D154 frozen reference parity is not exact")
    prior_lock = _read_json(ROOT / cfg["parents"]["d155_v1_lock"])
    prior_abort = (ROOT / cfg["parents"]["d155_v1_abort_report"]).read_text(encoding="utf-8")
    if prior_lock.get("schema") != "phase15_d155_lock_v1" or not prior_lock.get("source_commit"):
        raise D155Error("D155 v1 abort lock is not the expected preserved parent")
    if ("failed before fitting" not in prior_abort.lower()
            or "no valid decision metric" not in prior_abort.lower()):
        raise D155Error("D155 v1 abort report does not preserve the pre-fit stop")
    return {"parent_sha256": parent_hash, "d154_result": d154_result,
            "d154_training_lock": d154_lock, "d154_result_lock": d154_result_lock}


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D155Error("commit D155 protocol/source/tests before freezing lock")
    if cfg.get("schema") != "phase15_d155_minimal_consequence_v2":
        raise D155Error("D155 config schema mismatch")
    scope = cfg["scope"]
    if scope.get("train_and_reused_validation_only") is not True or scope.get("exact_r4_is_privileged_diagnostic") is not True:
        raise D155Error("D155 required scope flags disabled")
    if any(value is not False for key, value in scope.items()
           if key not in {"train_and_reused_validation_only", "exact_r4_is_privileged_diagnostic"}):
        raise D155Error("D155 scope permits a forbidden downstream operation")
    parents = _validate_parent_chain(cfg)
    split_meta = audit_split(cfg)
    for path in (cfg["outputs"]["lock"], cfg["outputs"]["run_dir"], cfg["outputs"]["report"]):
        if (ROOT / path).exists():
            raise D155Error(f"refusing to overwrite D155 output {path}")
    return {
        "schema": "phase15_d155_lock_v2",
        "status": "locked_stage_a_conditional_stage_c",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "config_sha256": sha256_file(ROOT / CONFIG),
        "parent_sha256": parents["parent_sha256"],
        "split_audit": split_meta,
        "feature_sets": cfg["feature_sets"],
        "stage_a": cfg["stage_a"],
        "stage_c_gate": cfg["stage_c_gate"],
        "scope": cfg["scope"],
        "outputs": cfg["outputs"],
        "validation_reuse": "D152 validation reused through D153/D154; all D155 metrics exploratory",
        "test_or_reserve_used": False,
    }


def audit_split(cfg: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / cfg["parents"]["d152_features"]
    with np.load(path, allow_pickle=False) as data:
        layouts = data["layouts"].astype(str).tolist()
        split = train_validation_indices(layouts, {
            "train_layout_count": 60, "train_row_count": 571,
            "train_layouts_sha256": "1FF811CF3169CA339C23FC73A22617C10249297565702A592D2BD3C9C683769B",
            "validation_layout_count": 20, "validation_row_count": 202,
            "validation_layouts_sha256": cfg["split"]["validation_layouts_sha256"],
        })
    train_ids, val_ids, train_layouts, val_layouts = split
    if len(train_ids) != 571 or len(val_ids) != 202:
        raise D155Error("D155 D152 train/validation counts differ")
    return {
        "d152_train_rows": len(train_ids), "d152_train_layouts": len(train_layouts),
        "d152_validation_rows": len(val_ids), "d152_validation_layouts": len(val_layouts),
        "validation_layouts_sha256": layout_list_sha256(val_layouts),
        "test_indices_created": False,
        "test_rows_dereferenced": 0,
    }


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    lock = _read_json(ROOT / cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d155_lock_v2":
        raise D155Error("D155 lock schema mismatch")
    if lock.get("config_sha256") != sha256_file(ROOT / CONFIG):
        raise D155Error("D155 config changed after lock")
    if lock.get("source_sha256") != {path: sha256_file(ROOT / path) for path in SOURCE_FILES}:
        raise D155Error("D155 source changed after lock")
    if lock.get("parent_sha256") != _hash_parents(cfg):
        raise D155Error("D155 parent data/checkpoint hash changed")
    if lock.get("scope") != cfg["scope"] or lock.get("test_or_reserve_used") is not False:
        raise D155Error("D155 scope/lock mismatch")
    if _git("status", "--porcelain"):
        raise D155Error("D155 run requires clean committed source")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    for path in (cfg["outputs"]["run_dir"], cfg["outputs"]["report"]):
        if (ROOT / path).exists():
            raise D155Error(f"refusing second D155 run: {path}")
    return cfg, lock


def extract_feature_matrix(exact_r4: np.ndarray, groups: tuple[str, ...]) -> np.ndarray:
    values = np.asarray(exact_r4, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 124 or not np.isfinite(values).all():
        raise ValueError("D155 exact R4 must be finite [N,124]")
    state = values[:, :61]
    summary = values[:, 61:].reshape(-1, 9, 7)
    columns = [state]
    for group in groups:
        if group == "outcome":
            columns.append(summary[:, :, :4].reshape(len(values), 36))
        elif group == "steps":
            columns.append(summary[:, :, 4])
        elif group == "goal_distance":
            columns.append(summary[:, :, 5])
        elif group == "effort":
            columns.append(summary[:, :, 6])
        else:
            raise ValueError(f"unknown D155 R4 group {group}")
    result = np.concatenate(columns, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("D155 extracted features are not finite")
    return result


def row_layout_labels(all_layouts: list[str], row_indices: np.ndarray) -> np.ndarray:
    """Return one permitted layout label per selected state row, in row order."""
    labels = np.asarray(all_layouts, dtype=str)
    indices = np.asarray(row_indices, dtype=np.int64)
    if (labels.ndim != 1 or indices.ndim != 1 or len(indices) == 0
            or np.any(indices < 0) or np.any(indices >= len(labels))
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("invalid D155 row-layout selection")
    return labels[indices]


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


class D155ListwiseMLP:
    """Fixed-width state-to-nine-action scorer; only input dimension varies."""

    def __init__(self, input_dim: int, cfg: dict[str, Any]) -> None:
        self.input_dim = int(input_dim)
        self.settings = cfg["stage_a"]
        self.metrics_cfg = cfg["metrics"]

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.w2 + self.b2
        h2 = np.maximum(z2, 0.0)
        return h2 @ self.w3 + self.b3, (x, h1, h2), (z1, z2)

    @staticmethod
    def _cross_entropy(logits: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> float:
        logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        return float(-np.sum(weights[:, None] * targets * logp))

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, train_layouts: np.ndarray,
            x_val: np.ndarray, y_val: np.ndarray, val_layouts: np.ndarray) -> "D155ListwiseMLP":
        x_train, y_train = np.asarray(x_train, float), np.asarray(y_train, float)
        x_val, y_val = np.asarray(x_val, float), np.asarray(y_val, float)
        if (x_train.ndim != 2 or x_train.shape[1] != self.input_dim or y_train.shape != (len(x_train), 9)
                or x_val.ndim != 2 or x_val.shape[1] != self.input_dim or y_val.shape != (len(x_val), 9)
                or len(train_layouts) != len(x_train) or len(val_layouts) != len(x_val)
                or not all(np.isfinite(v).all() for v in (x_train, y_train, x_val, y_val))):
            raise ValueError("invalid D155 listwise fit arrays")
        self.x_mean, self.x_std = x_train.mean(axis=0), x_train.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        xt, xv = (x_train - self.x_mean) / self.x_std, (x_val - self.x_mean) / self.x_std
        wt, wv = layout_weights(np.asarray(train_layouts)), layout_weights(np.asarray(val_layouts))
        teacher_train = _softmax(-y_train / float(self.settings["listwise_temperature"]))
        teacher_val = _softmax(-y_val / float(self.settings["listwise_temperature"]))
        rng = np.random.default_rng(int(self.settings["seed"]))
        hidden = tuple(map(int, self.settings["hidden"]))
        dims = (self.input_dim, *hidden, 9)
        self.w1 = rng.normal(0, np.sqrt(2 / dims[0]), (dims[0], dims[1]))
        self.w2 = rng.normal(0, np.sqrt(2 / dims[1]), (dims[1], dims[2]))
        self.w3 = rng.normal(0, np.sqrt(2 / dims[2]), (dims[2], 9))
        self.b1, self.b2, self.b3 = np.zeros(dims[1]), np.zeros(dims[2]), np.zeros(9)
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        m = [np.zeros_like(v) for v in params]
        v = [np.zeros_like(p) for p in params]
        best_params: list[np.ndarray] | None = None
        best_criterion = (float("inf"), float("inf"))
        stale = 0
        for epoch in range(int(self.settings["max_epochs"])):
            logits, activations, preactivations = self._forward(xt)
            delta = wt[:, None] * (_softmax(logits) - teacher_train)
            x0, h1, h2 = activations
            z1, z2 = preactivations
            grad = [None] * 6
            grad[4] = h2.T @ delta + float(self.settings["l2"]) * self.w3
            grad[5] = delta.sum(axis=0)
            dz2 = (delta @ self.w3.T) * (z2 > 0)
            grad[2] = h1.T @ dz2 + float(self.settings["l2"]) * self.w2
            grad[3] = dz2.sum(axis=0)
            dz1 = (dz2 @ self.w2.T) * (z1 > 0)
            grad[0] = x0.T @ dz1 + float(self.settings["l2"]) * self.w1
            grad[1] = dz1.sum(axis=0)
            step = epoch + 1
            for j, (parameter, g) in enumerate(zip(params, grad, strict=True)):
                m[j] = .9 * m[j] + .1 * g
                v[j] = .999 * v[j] + .001 * g * g
                parameter -= float(self.settings["learning_rate"]) * (m[j] / (1 - .9**step)) / (
                    np.sqrt(v[j] / (1 - .999**step)) + 1e-8)
            val_logits = self._forward(xv)[0]
            val_ce = self._cross_entropy(val_logits, teacher_val, wv)
            val_regret = decision_detail_metrics(y_val, -val_logits, np.asarray(val_layouts), self.metrics_cfg)[
                "layout_macro_normalized_selected_regret"]
            criterion = (float(val_regret), float(val_ce))
            improved = (criterion[0] < best_criterion[0] - 1e-9 or
                        (abs(criterion[0] - best_criterion[0]) <= 1e-9 and criterion[1] < best_criterion[1] - 1e-7))
            if improved:
                best_criterion, stale = criterion, 0
                best_params = [parameter.copy() for parameter in params]
                self.best_epoch = epoch + 1
            else:
                stale += 1
            if stale >= int(self.settings["patience"]):
                break
        if best_params is None:
            raise RuntimeError("D155 learner produced no checkpoint")
        for name, parameter in zip(("w1", "b1", "w2", "b2", "w3", "b3"), best_params, strict=True):
            setattr(self, name, parameter)
        self.epochs_run = epoch + 1
        self.validation_loss = best_criterion[1]
        self.parameter_count = int(sum(p.size for p in params))
        return self

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        rows = (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_std
        return self._forward(rows)[0]


def _load_data(cfg: dict[str, Any]) -> dict[str, Any]:
    # Only the explicit D152 train and validation indices are ever dereferenced.
    with np.load(ROOT / cfg["parents"]["d152_features"], allow_pickle=False) as saved:
        layouts = saved["layouts"].astype(str).tolist()
        train_ids, val_ids, old_train_layouts, val_layouts = train_validation_indices(layouts, {
            "train_layout_count": 60, "train_row_count": 571,
            "train_layouts_sha256": "1FF811CF3169CA339C23FC73A22617C10249297565702A592D2BD3C9C683769B",
            "validation_layout_count": 20, "validation_row_count": 202,
            "validation_layouts_sha256": cfg["split"]["validation_layouts_sha256"],
        })
        old_x = saved["x_R0"][train_ids].astype(np.float64)
        old_r4 = saved["x_R4"][train_ids].astype(np.float64)
        old_y = saved["y"][train_ids].astype(np.float64)
        val_x = saved["x_R0"][val_ids].astype(np.float64)
        val_r4 = saved["x_R4"][val_ids].astype(np.float64)
        val_y = saved["y"][val_ids].astype(np.float64)
        old_train_row_layouts = row_layout_labels(layouts, train_ids)
        validation_row_layouts = row_layout_labels(layouts, val_ids)
    with np.load(ROOT / cfg["parents"]["d153_corpus"], allow_pickle=False) as fresh:
        fresh_x = fresh["x_R0"].astype(np.float64)
        fresh_summary = fresh["summary_R4"].astype(np.float64)
        fresh_layouts = fresh["layouts"].astype(str)
    if (fresh_x.shape != (1365, 61) or fresh_summary.shape != (1365, 9, 7)
            or len(set(fresh_layouts.tolist())) != 140
            or set(fresh_layouts.tolist()).intersection(set(layouts))):
        raise D155Error("D153 fresh training corpus violates D155 split/schema")
    fresh_r4 = np.concatenate((fresh_x, fresh_summary.reshape(-1, 63)), axis=1)
    fresh_y = regrets_from_exact_summaries(fresh_summary)
    x_train = np.concatenate((old_x, fresh_x), axis=0)
    r4_train = np.concatenate((old_r4, fresh_r4), axis=0)
    y_train = np.concatenate((old_y, fresh_y), axis=0)
    train_layouts = np.concatenate((old_train_row_layouts, fresh_layouts))
    if (x_train.shape != (1936, 61) or r4_train.shape != (1936, 124) or y_train.shape != (1936, 9)
            or train_layouts.shape != (1936,) or len(set(train_layouts.tolist())) != 200
            or val_x.shape != (202, 61) or validation_row_layouts.shape != (202,)
            or val_r4.shape != (202, 124) or val_y.shape != (202, 9)):
        raise D155Error("D155 assembled D200 train/reused validation dimensions differ")
    if not np.array_equal(r4_train[:, :61], x_train) or not np.array_equal(val_r4[:, :61], val_x):
        raise D155Error("D155 R4 R0 prefix differs from the corresponding deployable state")
    if not np.array_equal(regrets_from_exact_summaries(r4_train[:, 61:].reshape(-1, 9, 7)), y_train):
        raise D155Error("D155 D200 exact target reconstruction mismatch")
    if not np.array_equal(regrets_from_exact_summaries(val_r4[:, 61:].reshape(-1, 9, 7)), val_y):
        raise D155Error("D155 validation exact target reconstruction mismatch")
    summary = r4_train[:, 61:].reshape(-1, 9, 7)
    if np.any(np.argmax(summary[:, :, :4], axis=2) == 1):
        raise D155Error("active continuation outcome found in D155 train labels")
    summary_val = val_r4[:, 61:].reshape(-1, 9, 7)
    if np.any(np.argmax(summary_val[:, :, :4], axis=2) == 1):
        raise D155Error("active continuation outcome found in D155 validation labels")
    return {"x_train": x_train, "r4_train": r4_train, "y_train": y_train,
            "train_layouts": train_layouts, "x_val": val_x, "r4_val": val_r4,
            "y_val": val_y, "val_layouts": validation_row_layouts,
            "target_audit": {"d152_train_rows": 571, "d152_validation_rows": 202,
                             "d153_fresh_rows": 1365, "d200_train_rows": 1936,
                             "d200_train_layouts": 200, "d152_validation_layouts": 20,
                             "train_target_max_abs_diff": 0.0, "validation_target_max_abs_diff": 0.0,
                             "active_outcomes_observed": 0,
                             "test_indices_created": False, "test_rows_dereferenced": 0}}


def _metrics(model: D155ListwiseMLP, x: np.ndarray, y: np.ndarray,
             layouts: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    raw = model.predict_logits(x)
    detail = decision_detail_metrics(y, -raw, layouts, cfg["metrics"])
    teacher = _softmax(-np.asarray(y, dtype=np.float64) /
                       float(cfg["stage_a"]["listwise_temperature"]))
    layout_weight = layout_weights(np.asarray(layouts))
    detail["listwise_cross_entropy"] = D155ListwiseMLP._cross_entropy(raw, teacher, layout_weight)
    times = []
    for row in x:
        start = time.perf_counter_ns()
        model.predict_logits(row[None, :])
        times.append((time.perf_counter_ns() - start) / 1e9)
    detail["median_latency_seconds_per_state"] = float(np.median(times))
    detail["parameter_count"] = model.parameter_count
    return detail


def _strip(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items()
            if key not in {"selected_regret", "per_layout_mean_regret"}}


def _write_model(path: Path, model: D155ListwiseMLP) -> str:
    np.savez_compressed(path, **{name: getattr(model, name) for name in
                                ("x_mean", "x_std", "w1", "b1", "w2", "b2", "w3", "b3")})
    return sha256_file(path)


def _reference_metrics(data: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    r0 = restore_decision_head(ROOT / cfg["parents"]["d150_r0_checkpoint"])
    r4 = restore_decision_head(ROOT / cfg["parents"]["d150_r4_checkpoint"])
    d154 = _read_json(ROOT / cfg["parents"]["d154_result"])
    rows = {}
    for name, head, x in (("direct_D150_R0", r0, data["x_val"]),
                          ("exact_R4_then_D150_head_oracle", r4, data["r4_val"])):
        metrics = decision_detail_metrics(data["y_val"], head.predict(x), data["val_layouts"], cfg["metrics"])
        rows[name] = {"metrics": _strip(metrics), "parameter_count": head.parameter_count,
                      "source": "recomputed_from_hash_pinned_D150_checkpoint_and_D155_validation_arrays",
                      "scores": head.predict(x)}
    # D153 did not preserve its predicted-summary tensor/checkpoint. Carry its
    # already frozen D154 metric record as a cited anchor; do not imply that it
    # was recomputed by D155.
    predicted_name = "D153_predicted_R4_then_D150_head"
    predicted_record = d154["references"][predicted_name]
    rows[predicted_name] = {
        "metrics": _strip(predicted_record["metrics"]),
        "parameter_count": predicted_record["parameter_count"],
        "source": "carried_forward_from_hash_pinned_D154_result; D153_predictions_not_saved",
    }
    checks = ("layout_macro_normalized_selected_regret", "catastrophic_fraction_state_micro",
              "best_action_set_agreement")
    diffs = {name: {key: abs(float(rows[name]["metrics"][key]) -
                              float(d154["references"][name]["metrics"][key])) for key in checks}
             for name in ("direct_D150_R0", "exact_R4_then_D150_head_oracle")}
    if max(value for record in diffs.values() for value in record.values()) > float(cfg["reference_parity_tolerance"]):
        raise D155Error(f"D155 frozen reference parity failed: {diffs}")
    for row in rows.values():
        row.pop("scores", None)
    return {"metrics": rows, "parity_max_abs_differences": diffs}


def _memorization(data: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    first_by_layout: dict[str, int] = {}
    for idx, layout in enumerate(data["train_layouts"].astype(str)):
        first_by_layout.setdefault(str(layout), idx)
    indices = [first_by_layout[name] for name in sorted(first_by_layout)[:8]]
    x, y, layouts = data["x_train"], data["y_train"], data["train_layouts"]
    results = {}
    setting = cfg["stage_a"]
    for name, groups in FEATURES:
        full_x = extract_feature_matrix(data["r4_train"], groups)
        mx, my, ml = full_x[indices], y[indices], layouts[indices]
        model = D155ListwiseMLP(full_x.shape[1], cfg).fit(mx, my, ml, mx, my, ml)
        metrics = decision_detail_metrics(my, -model.predict_logits(mx), ml, cfg["metrics"])
        results[name] = {
            "input_dimension": full_x.shape[1], "best_epoch": model.best_epoch,
            "mean_selected_regret": metrics["state_micro_normalized_selected_regret"],
            "best_action_set_agreement": metrics["best_action_set_agreement"],
            "passed": metrics["state_micro_normalized_selected_regret"] <= setting["memorization"]["maximum_selected_regret"]
            and metrics["best_action_set_agreement"] >= setting["memorization"]["minimum_best_action_agreement"],
        }
    return {"states": len(indices), "results": results,
            "all_passed": all(record["passed"] for record in results.values())}


def run_d155() -> dict[str, Any]:
    cfg, lock = validate_lock()
    data = _load_data(cfg)
    output = ROOT / cfg["outputs"]["run_dir"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    refs = _reference_metrics(data, cfg)
    mem = _memorization(data, cfg)
    (output / "memorization.json").write_text(json.dumps(mem, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not mem["all_passed"]:
        raise D155Error("D155 8-state feature-set memorization gate failed; stopped before full fits")
    results = {}
    fits_started = time.perf_counter()
    for name, groups in FEATURES:
        x_train = extract_feature_matrix(data["r4_train"], groups)
        x_val = extract_feature_matrix(data["r4_val"], groups)
        fit_start = time.perf_counter()
        model = D155ListwiseMLP(x_train.shape[1], cfg).fit(
            x_train, data["y_train"], data["train_layouts"], x_val, data["y_val"], data["val_layouts"])
        train_metrics = _metrics(model, x_train, data["y_train"], data["train_layouts"], cfg)
        val_metrics = _metrics(model, x_val, data["y_val"], data["val_layouts"], cfg)
        fname = name + ".npz"
        results[name] = {"groups": list(groups), "input_dimension": int(x_train.shape[1]),
                         "added_consequence_dimensions": int(x_train.shape[1] - 61),
                         "best_epoch": int(model.best_epoch), "epochs_run": int(model.epochs_run),
                         "fit_seconds": time.perf_counter() - fit_start,
                         "parameter_count": int(model.parameter_count),
                         "checkpoint_sha256": _write_model(output / fname, model),
                         "train": _strip(train_metrics), "validation": _strip(val_metrics),
                         "validation_selected_regret": val_metrics["selected_regret"],
                         "validation_per_layout_regret": val_metrics["per_layout_mean_regret"]}
    frozen_r0_regret = refs["metrics"]["direct_D150_R0"]["metrics"][
        "layout_macro_normalized_selected_regret"]
    frozen_full_regret = refs["metrics"]["exact_R4_then_D150_head_oracle"]["metrics"][
        "layout_macro_normalized_selected_regret"]
    gap = frozen_r0_regret - frozen_full_regret
    for row in results.values():
        regret = row["validation"]["layout_macro_normalized_selected_regret"]
        row["recovered_fraction_of_frozen_oracle_gap"] = float((frozen_r0_regret - regret) / gap)
    gate = cfg["stage_c_gate"]
    full = results["F_full"]
    full_pass = (full["recovered_fraction_of_frozen_oracle_gap"] >= gate["exact_full_controlled_model_min_recovered_fraction"]
                 and full["validation"]["catastrophic_fraction_state_micro"] <= gate["maximum_catastrophic_fraction"])
    candidates = [name for name, row in results.items()
                  if name != "F_full" and row["input_dimension"] < gate["compact_means_feature_dimension_lt"]
                  and row["recovered_fraction_of_frozen_oracle_gap"] >= gate["compact_subset_min_recovered_fraction"]
                  and row["validation"]["catastrophic_fraction_state_micro"] <= gate["maximum_catastrophic_fraction"]]
    candidates.sort(key=lambda name: (results[name]["added_consequence_dimensions"],
                                      results[name]["validation"]["layout_macro_normalized_selected_regret"],
                                      results[name]["validation"]["catastrophic_fraction_state_micro"]))
    selected = candidates[0] if full_pass and candidates else None
    result = {
        "schema": "phase15_d155_stage_a_result_v1",
        "status": "completed_exploratory_oracle_consequence_ablation",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
        "parent_sha256": lock["parent_sha256"], "split_audit": lock["split_audit"],
        "target_audit": data["target_audit"], "reference_metrics": refs["metrics"],
        "reference_parity_max_abs_differences": refs["parity_max_abs_differences"],
        "memorization": mem, "feature_results": results,
        "frozen_anchor_regret_gap": gap,
        "stage_c_gate": {"full_feature_model_passed": bool(full_pass),
                         "compact_candidates": candidates, "selected_compact_subset": selected,
                         "stage_c_condition_satisfied": selected is not None,
                         "stage_c_executed": False,
                         "reason": "requires separate implementation/tests/lock" if selected else
                         "no compact subset met the frozen thresholds"},
        "training_repeated": False,
        "scope": {"simulator_calls": 0, "planner_calls": 0, "new_transitions": 0,
                  "new_episodes": 0, "test_rows_dereferenced": 0, "reserve_access": False,
                  "student_controlled_simulation": False, "closed_loop": False, "dagger": False,
                  "world_model": False, "jepa": False},
        "fit_total_seconds": time.perf_counter() - fits_started,
        "wall_seconds": time.perf_counter() - started,
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / cfg["outputs"]["report"]
    report_path.write_text(render_report(result), encoding="utf-8")
    manifest = {
        "schema": "phase15_d155_stage_a_manifest_v1", "status": result["status"],
        "lock_sha256": result["lock_sha256"], "source_commit": result["source_commit"],
        "parent_sha256": lock["parent_sha256"], "result_sha256": sha256_file(result_path),
        "report_sha256": sha256_file(report_path),
        "checkpoint_sha256": {name + ".npz": row["checkpoint_sha256"] for name, row in results.items()},
        "test_rows_dereferenced": 0, "reserve_access": False,
        "simulator_calls": 0, "planner_calls": 0, "training_repeated": False,
        "stage_c_executed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D155 — Minimal Decision-Critical Consequence Study", "",
             "**Status: exploratory Stage-A oracle ablation.** Fixed validation is heavily reused; test/reserve remain sealed. Exact future features are privileged diagnostics, not deployable inputs.", "",
             "## Integrity and references", "",
             f"D200 train: {result['target_audit']['d200_train_rows']} rows / {result['target_audit']['d200_train_layouts']} layouts; fixed validation: {result['split_audit']['d152_validation_rows']} rows / {result['split_audit']['d152_validation_layouts']} reused layouts.",
             f"Target reconstruction max difference: train `{result['target_audit']['train_target_max_abs_diff']}`, validation `{result['target_audit']['validation_target_max_abs_diff']}`. D150 frozen reference parity max difference: `{max(x for r in result['reference_parity_max_abs_differences'].values() for x in r.values()):.2e}`.", "",
             "## Oracle consequence ablation", "",
             "Each subset uses the same listwise MLP trained with exact continuation features. Frozen D150 direct-R0 and exact-R4 heads are recomputed anchors; the D153 predicted-R4 metric is carried forward from the hash-pinned D154 result because D153 saved neither predictions nor a predictor checkpoint. Its end-to-end latency is unavailable. Fraction recovered is `(R0 regret - subset regret)/(R0 regret - full-R4 regret)`.", "",
             "| Feature set | Consequence fields | D | Added D | Params | Train regret | Val macro regret | Recovered fraction | Catastrophic | Epsilon-optimal | Best-action set | High-margin rank | Safe/cat order |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["feature_results"].items():
        v = row["validation"]
        lines.append(f"| {name} | {', '.join(row['groups']) or 'none'} | {row['input_dimension']} | {row['added_consequence_dimensions']} | {row['parameter_count']} | {row['train']['layout_macro_normalized_selected_regret']:.4f} | {v['layout_macro_normalized_selected_regret']:.4f} | {row['recovered_fraction_of_frozen_oracle_gap']:.3f} | {v['catastrophic_fraction_state_micro']:.3f} | {v['epsilon_optimal_rate']:.3f} | {v['best_action_set_agreement']:.3f} | {v['high_margin_pairwise_accuracy_macro']:.3f} | {v['safe_vs_catastrophic_order_accuracy_macro']:.3f} |")
    for name, row in result["reference_metrics"].items():
        v = row["metrics"]
        lines.append(f"| {name} (frozen) | anchor | {61 if name.startswith('direct') else 124} | — | {row['parameter_count']} | — | {v['layout_macro_normalized_selected_regret']:.4f} | {((result['frozen_anchor_regret_gap'] if name.startswith('exact') else 0.0)):.3f} | {v['catastrophic_fraction_state_micro']:.3f} | {v['epsilon_optimal_rate']:.3f} | {v['best_action_set_agreement']:.3f} | {v['high_margin_pairwise_accuracy_macro']:.3f} | {v['safe_vs_catastrophic_order_accuracy_macro']:.3f} |")
    gate = result["stage_c_gate"]
    lines.extend(["", "## Conditional prediction screen", "",
                  f"F-full adequacy gate: `{gate['full_feature_model_passed']}`. Compact eligible subsets: `{gate['compact_candidates']}`. Selected: `{gate['selected_compact_subset']}`. Stage C executed: `{gate['stage_c_executed']}` ({gate['reason']}).", "",
                  "No Stage-C consequence predictor, simulation, new transition, closed-loop evaluation, DAgger, test/reserve access, or world-model/JEPA training occurred in this Stage-A run.", ""])
    return "\n".join(lines)
