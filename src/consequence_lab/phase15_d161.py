"""Bounded D161 relational student fit-boundary study."""

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
from consequence_lab.phase15_d158 import (
    RelationalCandidateModel, _action_inputs, _candidate_metrics,
    _crossfit_train_consequences, _feature_set, _preference_loss_grad,
    _probabilities_from_exact, _risk_metrics, _sigmoid, _softmax,
)
from consequence_lab.phase15_d159 import _training_payload, _json_safe
from consequence_lab.phase15_decision_aligned import sha256_array
from consequence_lab.phase15_learning import layout_weights

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d161_fit_boundary.yaml"
SOURCE_FILES = (
    CONFIG, "reports/phase15_d161_protocol.md", "src/consequence_lab/phase15_d161.py",
    "scripts/freeze_phase15_d161_lock.py", "scripts/run_phase15_d161.py",
    "tests/test_phase15_d161.py", "src/consequence_lab/phase15_d158.py",
    "src/consequence_lab/phase15_d159.py", "src/consequence_lab/phase15_d160.py",
    "src/consequence_lab/phase15_decision_aligned.py", "src/consequence_lab/phase15_learning.py",
    "src/consequence_lab/phase15_d155_stage_c.py", "src/consequence_lab/phase14.py",
    "configs/phase15_d155_stage_c.yaml", "configs/phase15_d159_training_fit_v2.yaml",
    "second-brain/00_Project_Home.md", "second-brain/14_Decisions.md",
    "second-brain/16_Experiment_Log.md",
    "second-brain/phases/Phase_15_Decision_Aligned_Target_Study.md",
)
PARAMETER_NAMES = ("ws", "bs", "wc", "bc", "wctx", "bctx", "wscore", "bscore", "wrisk", "brisk")


class D161Error(RuntimeError):
    """D161 provenance, resume, or execution error."""


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D161Error(f"git {' '.join(args)} failed") from exc


def _parent_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}


def _d160_lineage(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["parents"]
    d153_lock, d153_manifest = _read_json(ROOT / p["d153_lock"]), _read_json(ROOT / p["d153_manifest"])
    d159_lock, d159_manifest = _read_json(ROOT / p["d159_lock"]), _read_json(ROOT / p["d159_manifest"])
    d159_result = _read_json(ROOT / p["d159_result"])
    d160_lock = _read_json(ROOT / p["d160_lock"])
    d160_manifest, d160_result = _read_json(ROOT / p["d160_manifest"]), _read_json(ROOT / p["d160_result"])
    dataset_sha = sha256_file(ROOT / p["d153_training_rows"])
    if (d153_lock.get("schema") != "phase15_d153_coverage_scaling_lock_v3"
            or d153_lock.get("reused_corpus", {}).get("path") != p["d153_training_rows"]
            or d153_lock.get("reused_corpus", {}).get("sha256") != dataset_sha
            or d153_manifest.get("fresh_dataset_sha256") != dataset_sha
            or d153_manifest.get("reserve_access") is not False
            or d153_manifest.get("test_rows_indexed_or_scored") != 0):
        raise D161Error("D153 training-only lineage mismatch")
    if (d159_lock.get("schema") != "phase15_d159_training_fit_lock_v2"
            or d159_lock.get("dataset_sha256") != dataset_sha
            or d159_manifest.get("lock_sha256") != sha256_file(ROOT / p["d159_lock"])
            or d159_manifest.get("result_sha256") != sha256_file(ROOT / p["d159_result"])
            or d159_result.get("status") != "completed_training_only_fit_audit_oracle_control_passed"
            or d159_result.get("scope", {}).get("reserve_access") is not False
            or d159_result.get("scope", {}).get("test_rows_dereferenced") != 0):
        raise D161Error("D159 source audit failed")
    if (d160_lock.get("schema") != "phase15_d160_convergence_lock_v1"
            or d160_lock.get("dataset_sha256") != dataset_sha
            or d160_manifest.get("lock_sha256") != sha256_file(ROOT / p["d160_lock"])
            or d160_manifest.get("result_sha256") != sha256_file(ROOT / p["d160_result"])
            or d160_result.get("status") != "completed_training_only_convergence_audit"
            or d160_result.get("scope", {}).get("reserve_access") is not False
            or d160_result.get("scope", {}).get("validation_rows_read") != 0
            or d160_result.get("scope", {}).get("test_rows_dereferenced") != 0):
        raise D161Error("D160 source audit failed")
    checkpoint_names = {
        "r0_action_only": "d160_r0_checkpoint",
        "r0_action_predicted_outcome_steps": "d160_predicted_checkpoint",
        "r0_action_exact_outcome_steps_oracle": "d160_oracle_checkpoint",
    }
    for model_name, parent_name in checkpoint_names.items():
        digest = sha256_file(ROOT / p[parent_name])
        if d160_manifest.get("model_checkpoint_sha256", {}).get(model_name) != digest:
            raise D161Error(f"D160 checkpoint hash mismatch: {model_name}")
        if model_name not in d160_result.get("models", {}):
            raise D161Error(f"D160 selected result missing: {model_name}")
    return {"d153_training_sha256": dataset_sha, "d159_status": d159_result["status"],
            "d160_status": d160_result["status"], "reserve_access": False,
            "validation_rows_read": 0, "test_rows_dereferenced": 0}


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D161Error("commit D161 protocol/config/source/tests/vault before locking")
    if cfg.get("schema") != "phase15_d161_fit_boundary_v1" or cfg.get("status") != "training_only_capacity_study_frozen":
        raise D161Error("D161 schema/status mismatch")
    scope = cfg["scope"]
    if scope.get("training_only") is not True or any(v is not False for k, v in scope.items() if k != "training_only"):
        raise D161Error("D161 must remain training-only")
    lineage = _d160_lineage(cfg)
    data = _training_payload()
    if (data["states"].shape != (1365, 61) or data["summary"].shape != (1365, 9, 7)
            or len(set(data["layouts"].tolist())) != 140):
        raise D161Error("D161 D153 training corpus shape changed")
    return {
        "schema": "phase15_d161_fit_boundary_lock_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "config_sha256": sha256_file(ROOT / CONFIG), "protocol_sha256": sha256_file(ROOT / cfg["protocol"]),
        "parent_sha256": _parent_hashes(cfg), "lineage": lineage,
        "dataset_sha256": sha256_file(ROOT / cfg["parents"]["d153_training_rows"]),
        "state_sha256": sha256_array(data["states"]), "summary_sha256": sha256_array(data["summary"]),
        "regret_sha256": sha256_array(data["regret"]),
        "layout_set_sha256": hashlib.sha256("\n".join(sorted(set(data["layouts"].tolist()))).encode()).hexdigest(),
        "row_count": int(len(data["states"])), "layout_count": len(set(data["layouts"].tolist())),
        "scope": {**scope, "validation_rows_read": 0, "test_rows_dereferenced": 0, "reserve_access": False},
        "outputs": cfg["outputs"],
    }


def _validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D161Error("D161 execution requires a clean committed tree")
    lock_path = ROOT / cfg["outputs"]["lock"]
    lock = _read_json(lock_path)
    if lock.get("schema") != "phase15_d161_fit_boundary_lock_v1":
        raise D161Error("D161 lock schema mismatch")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    if (lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("protocol_sha256") != sha256_file(ROOT / cfg["protocol"])
            or lock.get("source_sha256") != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
            or lock.get("parent_sha256") != _parent_hashes(cfg)
            or lock.get("dataset_sha256") != sha256_file(ROOT / cfg["parents"]["d153_training_rows"])):
        raise D161Error("D161 source/config/parent/dataset hash mismatch")
    _d160_lineage(cfg)
    data = _training_payload()
    if (lock.get("state_sha256") != sha256_array(data["states"])
            or lock.get("summary_sha256") != sha256_array(data["summary"])
            or lock.get("regret_sha256") != sha256_array(data["regret"])):
        raise D161Error("D161 state/target fingerprint mismatch")
    out = ROOT / cfg["outputs"]["run_dir"]
    if out.exists():
        marker = out / "run_state.json"
        if not marker.exists() or _read_json(marker).get("lock_sha256") != sha256_file(lock_path):
            raise D161Error("existing output is not a matching resumable D161 run")
    return cfg, lock


def _atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def _named_state(model: RelationalCandidateModel, first: list[np.ndarray], second: list[np.ndarray],
                 best_params: list[np.ndarray], best_first: list[np.ndarray], best_second: list[np.ndarray],
                 epoch: int, best_loss: float, best_epoch: int, metadata: dict[str, Any],
                 history: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"x_mean": model.x_mean, "x_std": model.x_std,
                              "epoch": np.asarray(epoch, dtype=np.int64),
                              "best_loss": np.asarray(best_loss), "best_epoch": np.asarray(best_epoch, dtype=np.int64),
                              "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
                              "history_json": np.asarray(json.dumps(_json_safe(history), sort_keys=True)),
                              "scheduler_state_json": np.asarray("null")}
    for idx, name in enumerate(PARAMETER_NAMES):
        output[f"param_{name}"] = getattr(model, name)
        output[f"adam_first_{idx}"] = first[idx]
        output[f"adam_second_{idx}"] = second[idx]
        output[f"best_param_{name}"] = best_params[idx]
        output[f"best_adam_first_{idx}"] = best_first[idx]
        output[f"best_adam_second_{idx}"] = best_second[idx]
    return output


def _restore_state(path: Path, model: RelationalCandidateModel,
                   expected_metadata: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray],
                                                                list[np.ndarray], list[np.ndarray],
                                                                list[np.ndarray], int, float, int,
                                                                list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        history = json.loads(str(data["history_json"].item()))
        if metadata != expected_metadata or str(data["scheduler_state_json"].item()) != "null":
            raise D161Error("resume checkpoint metadata/scheduler mismatch")
        for key in ("x_mean", "x_std"):
            value = np.asarray(data[key])
            if value.shape != getattr(model, key).shape or not np.array_equal(value, getattr(model, key)):
                raise D161Error(f"resume checkpoint {key} mismatch")
        first, second, best_params, best_first, best_second = [], [], [], [], []
        for idx, name in enumerate(PARAMETER_NAMES):
            param = np.asarray(data[f"param_{name}"])
            if param.shape != getattr(model, name).shape or not np.isfinite(param).all():
                raise D161Error(f"resume parameter contract failed: {name}")
            getattr(model, name)[...] = param
            for target, key in ((first, f"adam_first_{idx}"), (second, f"adam_second_{idx}"),
                                (best_params, f"best_param_{name}"), (best_first, f"best_adam_first_{idx}"),
                                (best_second, f"best_adam_second_{idx}")):
                value = np.asarray(data[key], dtype=np.float64)
                if value.shape != param.shape or not np.isfinite(value).all():
                    raise D161Error(f"resume optimizer/best-state contract failed: {key}")
                target.append(value.copy())
        return (first, second, best_params, best_first, best_second, int(data["epoch"]),
                float(data["best_loss"]), int(data["best_epoch"]), history)


def _save_best(path: Path, x_mean: np.ndarray, x_std: np.ndarray, best_params: list[np.ndarray],
               first: list[np.ndarray], second: list[np.ndarray], epoch: int,
               objective: float, metadata: dict[str, Any]) -> None:
    arrays = {"x_mean": x_mean, "x_std": x_std,
              "epoch": np.asarray(epoch), "objective": np.asarray(objective),
              "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))}
    for idx, name in enumerate(PARAMETER_NAMES):
        arrays[f"param_{name}"] = best_params[idx]
        arrays[f"adam_first_{idx}"] = first[idx]
        arrays[f"adam_second_{idx}"] = second[idx]
    _atomic_npz(path, **arrays)


def _convergence_check(history: list[dict[str, Any]], epoch: int, rule: dict[str, Any]) -> dict[str, Any]:
    spacing = int(rule["check_spacing_epochs"])
    checks = [item for item in history if item["epoch"] % spacing == 0]
    if epoch < int(rule["min_epochs_before_check"]):
        return {"converged": False, "reason": "minimum_epoch_requirement_not_met"}
    latest = checks[-5:]
    expected = [epoch - 4 * spacing, epoch - 3 * spacing, epoch - 2 * spacing, epoch - spacing, epoch]
    if len(latest) != 5 or [r["epoch"] for r in latest] != expected:
        return {"converged": False, "reason": "noncontiguous_stability_checks"}
    window = int(rule["convergence_window_epochs"])
    recent_start = epoch - window
    previous_start = recent_start - window
    recent = [r["objective"] for r in history if recent_start < r["epoch"] <= epoch]
    previous = [r["objective"] for r in history if previous_start < r["epoch"] <= recent_start]
    if not recent or not previous:
        return {"converged": False, "reason": "objective_window_missing"}
    prev_best, recent_best = min(previous), min(recent)
    relative_improvement = (prev_best - recent_best) / max(abs(prev_best), 1e-12)
    regret_range = max(r["macro_regret"] for r in latest) - min(r["macro_regret"] for r in latest)
    catastrophic_range = max(r["catastrophic_fraction"] for r in latest) - min(r["catastrophic_fraction"] for r in latest)
    return {"converged": bool(relative_improvement <= float(rule["objective_relative_improvement_tolerance"])
                               and regret_range <= float(rule["decision_metric_range_tolerance"])
                               and catastrophic_range <= float(rule["decision_metric_range_tolerance"])),
            "relative_objective_improvement": float(relative_improvement),
            "previous_window_best_objective": float(prev_best), "recent_window_best_objective": float(recent_best),
            "macro_regret_range_last_five_checks": float(regret_range),
            "catastrophic_range_last_five_checks": float(catastrophic_range),
            "check_epochs": [int(r["epoch"]) for r in latest]}


def _train_one(name: str, width: int, states: np.ndarray, actions: np.ndarray,
               consequences: np.ndarray, mask: tuple[float, float], regret: np.ndarray,
               collision: np.ndarray, summary: np.ndarray, layouts: np.ndarray,
               stage_metrics: dict[str, Any], cfg: dict[str, Any], lock: dict[str, Any],
               out: Path, warm_checkpoint: Path | None) -> dict[str, Any]:
    model_cfg = cfg["models"]
    seed = int(model_cfg["width_64"]["seed"] if width == 64 else 15900)
    model = RelationalCandidateModel(width, seed, model_cfg["learning_rate"], model_cfg["l2"],
                                     model_cfg["beta1"], model_cfg["beta2"], model_cfg["epsilon"])
    model.initialize(states)
    warm_sha = None
    if warm_checkpoint is not None:
        expected_mean, expected_std = model.x_mean.copy(), model.x_std.copy()
        with np.load(warm_checkpoint, allow_pickle=False) as saved:
            warm_sha = sha256_file(warm_checkpoint)
            if not (np.array_equal(saved["x_mean"], expected_mean) and np.array_equal(saved["x_std"], expected_std)):
                raise D161Error("D160 warm-start normalization differs from D153 training statistics")
            for name_param in ("ws", "bs", "wc", "bc", "wctx", "bctx", "wscore", "bscore", "wrisk", "brisk"):
                source = np.asarray(saved[name_param], dtype=np.float64)
                if source.shape != getattr(model, name_param).shape:
                    raise D161Error(f"D160 width-24 checkpoint shape mismatch: {name_param}")
                getattr(model, name_param)[...] = source
    params = model.parameters()
    state_weight = layout_weights(layouts)
    candidate_weight = state_weight[:, None] / 9.0
    teacher = _softmax(-regret / float(model_cfg["listwise_temperature"]))
    pair_i, pair_j = np.triu_indices(9, 1)
    zero_pairs = np.zeros((len(states), len(pair_i)), dtype=np.float64)
    collision_target = collision
    maximum = int(model_cfg["width_24"]["maximum_additional_epochs"] if width == 24
                  else model_cfg["width_64"]["maximum_epochs"])
    representation_dir = out / f"width_{width}"
    representation_dir.mkdir(parents=True, exist_ok=True)
    stem = name
    latest_path = representation_dir / f"{stem}_latest_resume.npz"
    best_path = representation_dir / f"{stem}_best_training_objective.npz"
    history_path = representation_dir / f"{stem}_history.json"
    completed_path = representation_dir / f"{stem}_completed.json"
    metadata = {"name": name, "width": width, "seed": seed, "warm_start_sha256": warm_sha,
                "dataset_sha256": lock["dataset_sha256"], "regret_sha256": lock["regret_sha256"],
                "config_sha256": lock["config_sha256"], "source_commit": lock["source_commit"],
                "listwise_temperature": model_cfg["listwise_temperature"],
                "collision_loss_weight": model_cfg["collision_loss_weight"],
                "learning_rate": model_cfg["learning_rate"], "l2": model_cfg["l2"],
                "beta1": model_cfg["beta1"], "beta2": model_cfg["beta2"], "epsilon": model_cfg["epsilon"],
                "scheduler": None, "optimizer": "full_batch_adam", "mask": list(mask)}
    if completed_path.exists():
        completed = _read_json(completed_path)
        if completed.get("metadata") != metadata or completed.get("checkpoint_sha256") != sha256_file(best_path):
            raise D161Error(f"completed D161 fit provenance mismatch: {name}/width{width}")
        return completed["result"]
    if latest_path.exists():
        first, second, best_params, best_first, best_second, start_epoch, best_loss, best_epoch, history = _restore_state(
            latest_path, model, metadata)
        if not history or history[-1]["epoch"] != start_epoch:
            raise D161Error("resume state has invalid embedded training history")
        _atomic_json(history_path, history)
        _save_best(best_path, model.x_mean, model.x_std, best_params, best_first, best_second,
                   best_epoch, best_loss, metadata)
        run_start = time.perf_counter()
        prior_elapsed = float(history[-1].get("elapsed_seconds", 0.0))
    else:
        first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        best_params = [p.copy() for p in params]
        best_first, best_second = [p.copy() for p in first], [p.copy() for p in second]
        start_epoch, best_loss, best_epoch = 0, float("inf"), -1
        history = []
        run_start, prior_elapsed = time.perf_counter(), 0.0

    risk_weight = float(model_cfg["collision_loss_weight"])

    def evaluate(epoch: int, elapsed: float) -> dict[str, Any]:
        utility, risk_logits = model._forward(states, actions, consequences, mask)
        pref, _ = _preference_loss_grad(utility, regret, teacher, state_weight, zero_pairs,
                                        pair_i, pair_j, "listwise_tau_025", 0.0)
        risk_prob = _sigmoid(risk_logits)
        bce = -float(np.sum(candidate_weight * (collision_target * np.log(np.maximum(risk_prob, 1e-12))
                                                + (1.0 - collision_target) * np.log(np.maximum(1.0 - risk_prob, 1e-12)))))
        objective = float(pref + risk_weight * bce)
        metrics = _candidate_metrics(regret, -utility, summary, layouts, stage_metrics)
        risk_metrics = _risk_metrics(risk_prob, collision_target > 0.5, layouts, 0.5)
        return {"epoch": epoch, "objective": objective, "preference_loss": float(pref), "collision_bce": bce,
                "macro_regret": float(metrics["layout_macro_normalized_selected_regret"]),
                "catastrophic_fraction": float(metrics["catastrophic_fraction_layout_macro"]),
                "epsilon_optimal_rate": float(metrics["epsilon_optimal_rate"]),
                "best_action_set_agreement": float(metrics["best_action_set_agreement"]),
                "high_margin_pairwise_accuracy": metrics.get("high_margin_pairwise_accuracy_macro"),
                "safe_vs_collision_pair_accuracy": metrics.get("safe_vs_collision_pair_accuracy"),
                "collision_recall": risk_metrics["collision_recall"], "collision_precision": risk_metrics["precision"],
                "elapsed_seconds": elapsed}

    if not history:
        record = evaluate(0, prior_elapsed)
        history.append(record)
        best_loss, best_epoch = record["objective"], 0
        best_params = [p.copy() for p in params]
        best_first, best_second = [p.copy() for p in first], [p.copy() for p in second]
        _atomic_json(history_path, history)
        _atomic_npz(latest_path, **_named_state(model, first, second, best_params, best_first, best_second,
                                               0, best_loss, best_epoch, metadata, history))
        _save_best(best_path, model.x_mean, model.x_std, best_params, best_first, best_second,
                   best_epoch, best_loss, metadata)

    convergence = cfg["convergence"]
    checks: list[dict[str, Any]] = []
    true_check_epochs = [item["epoch"] for item in history
                         if item["epoch"] >= convergence["min_epochs_before_check"]
                         and item["epoch"] % convergence["check_spacing_epochs"] == 0
                         and item.get("converged_check") is True]
    streak = 0
    for idx, check_epoch in enumerate(true_check_epochs):
        if idx and check_epoch == true_check_epochs[idx - 1] + convergence["check_spacing_epochs"]:
            streak += 1
        else:
            streak = 1
    for epoch in range(start_epoch + 1, maximum + 1):
        utility, risk_logits, cache = model._forward(states, actions, consequences, mask, cache=True)
        _, grad_utility = _preference_loss_grad(utility, regret, teacher, state_weight, zero_pairs,
                                                 pair_i, pair_j, "listwise_tau_025", 0.0)
        risk_prob = _sigmoid(risk_logits)
        grad_risk = risk_weight * candidate_weight * (risk_prob - collision_target)
        gradients = model._backward(grad_utility, grad_risk, cache)
        step = epoch
        for idx, (param, grad) in enumerate(zip(params, gradients, strict=True)):
            if param.ndim > 0 and id(param) not in {id(model.bs), id(model.bc), id(model.bctx)}:
                grad = grad + float(model_cfg["l2"]) * param
            first[idx] = model_cfg["beta1"] * first[idx] + (1.0 - model_cfg["beta1"]) * grad
            second[idx] = model_cfg["beta2"] * second[idx] + (1.0 - model_cfg["beta2"]) * grad * grad
            param -= model_cfg["learning_rate"] * (first[idx] / (1.0 - model_cfg["beta1"] ** step)) / (
                np.sqrt(second[idx] / (1.0 - model_cfg["beta2"] ** step)) + model_cfg["epsilon"])
        if epoch % convergence["evaluate_every_epochs"]:
            continue
        elapsed = prior_elapsed + time.perf_counter() - run_start
        record = evaluate(epoch, elapsed)
        best_updated = False
        if record["objective"] < best_loss - 1e-12:
            best_loss, best_epoch = record["objective"], epoch
            best_params = [p.copy() for p in params]
            best_first, best_second = [p.copy() for p in first], [p.copy() for p in second]
            best_updated = True
        check = {"converged": False, "reason": "not_check_epoch"}
        spacing = int(convergence["check_spacing_epochs"])
        if epoch >= convergence["min_epochs_before_check"] and epoch % spacing == 0:
            check = _convergence_check(history + [record], epoch, convergence)
            streak = streak + 1 if check["converged"] else 0
            checks.append({"epoch": epoch, **check, "consecutive_converged_checks": streak})
            record["converged_check"] = bool(check["converged"])
        history.append(record)
        _atomic_json(history_path, history)
        _atomic_npz(latest_path, **_named_state(model, first, second, best_params, best_first, best_second,
                                               epoch, best_loss, best_epoch, metadata, history))
        if best_updated:
            _save_best(best_path, model.x_mean, model.x_std, best_params, best_first, best_second,
                       best_epoch, best_loss, metadata)
        if streak >= convergence["required_consecutive_converged_checks"]:
            break

    # Restore the minimum-objective checkpoint, not the last optimization iterate.
    for param, value in zip(params, best_params, strict=True):
        param[...] = value
    utility, risk_logits = model._forward(states, actions, consequences, mask)
    decision_metrics = _candidate_metrics(regret, -utility, summary, layouts, stage_metrics)
    decision_metrics.update(_risk_metrics(_sigmoid(risk_logits), collision_target > 0.5, layouts, 0.5))
    gate_passed = (decision_metrics["layout_macro_normalized_selected_regret"] <= cfg["fit_gate"]["max_train_macro_regret"]
                   and decision_metrics["catastrophic_fraction_layout_macro"] <= cfg["fit_gate"]["max_train_catastrophic_fraction"])
    epochs_run = max((item["epoch"] for item in history), default=0)
    result = {"status": "converged" if streak >= convergence["required_consecutive_converged_checks"]
              else "not_converged_within_budget", "width": width, "parameter_count": int(sum(p.size for p in params)),
              "epochs_run": epochs_run, "best_epoch": best_epoch, "best_training_objective": best_loss,
              "metrics_training_only": decision_metrics, "fit_gate_passed": bool(gate_passed),
              "convergence_checks": checks, "consecutive_checks_at_stop": int(streak),
              "wall_time_seconds": float(history[-1].get("elapsed_seconds", 0.0)),
              "checkpoint_path": best_path.relative_to(ROOT).as_posix(),
              "checkpoint_sha256": sha256_file(best_path), "latest_resume_path": latest_path.relative_to(ROOT).as_posix(),
              "latest_resume_sha256": sha256_file(latest_path), "history_path": history_path.relative_to(ROOT).as_posix(),
              "history_sha256": sha256_file(history_path), "metadata": metadata,
              "optimizer_state_saved": True, "scheduler": None}
    _atomic_json(completed_path, {"metadata": metadata, "checkpoint_sha256": result["checkpoint_sha256"], "result": result})
    return result


def run_d161() -> dict[str, Any]:
    cfg, lock = _validate_lock()
    started = time.perf_counter()
    out = ROOT / cfg["outputs"]["run_dir"]
    out.mkdir(parents=True, exist_ok=True)
    lock_sha = sha256_file(ROOT / cfg["outputs"]["lock"])
    marker_path = out / "run_state.json"
    if not marker_path.exists():
        _atomic_json(marker_path, {"schema": "phase15_d161_run_state_v1", "lock_sha256": lock_sha,
                                   "dataset_sha256": lock["dataset_sha256"], "status": "running"})
    payload = _training_payload()
    actions, _ = _action_inputs(payload["states"])
    cross_cfg = {"parents": {"d155_stage_c_config": cfg["parents"]["d155_stage_c_config"]}}
    train_data = {"train_r0": payload["states"], "train_summary": payload["summary"],
                  "train_layouts": payload["layouts"]}
    predicted_outcome, predicted_steps, crossfit_audit = _crossfit_train_consequences(train_data, cross_cfg)
    exact_outcome, exact_steps = _probabilities_from_exact(payload["summary"])
    representations = {
        "r0_action_only": _feature_set(predicted_outcome, predicted_steps, "r0_action_only"),
        "r0_action_predicted_outcome_steps": _feature_set(predicted_outcome, predicted_steps, "r0_action_predicted_outcome_steps"),
        "r0_action_exact_outcome_steps_oracle": _feature_set(exact_outcome, exact_steps, "r0_action_exact_outcome_steps"),
    }
    stage_metrics = yaml.safe_load((ROOT / cfg["parents"]["d155_stage_c_config"]).read_text(encoding="utf-8"))["metrics"]
    if out.joinpath("result.json").exists():
        raise D161Error("D161 final result already exists; do not rerun a completed experiment")
    try:
        fit_results: dict[str, dict[str, Any]] = {}
        for width in cfg["models"]["widths"]:
            for name, (consequence, mask) in representations.items():
                warm = None
                if width == 24:
                    parent_key = {"r0_action_only": "d160_r0_checkpoint",
                                  "r0_action_predicted_outcome_steps": "d160_predicted_checkpoint",
                                  "r0_action_exact_outcome_steps_oracle": "d160_oracle_checkpoint"}[name]
                    warm = ROOT / cfg["parents"][parent_key]
                fit_results[f"width_{width}/{name}"] = _train_one(
                    name, int(width), payload["states"], actions, consequence, mask,
                    payload["regret"], payload["collision"], payload["summary"], payload["layouts"],
                    stage_metrics, cfg, lock, out, warm)
        d159 = _read_json(ROOT / cfg["parents"]["d159_result"])
        d160 = _read_json(ROOT / cfg["parents"]["d160_result"])
        prior = {"D159": {name: d159["models"][name]["metrics_training_only"] for name in representations},
                 "D160": {name: d160["models"][name]["metrics_training_only"] for name in representations}}
        result = {"schema": "phase15_d161_fit_boundary_result_v1", "status": "completed_training_only_fit_boundary",
                  "source_commit": lock["source_commit"], "lock_sha256": lock_sha,
                  "dataset_sha256": lock["dataset_sha256"], "row_count": lock["row_count"],
                  "layout_count": lock["layout_count"], "state_sha256": lock["state_sha256"],
                  "regret_sha256": lock["regret_sha256"], "crossfit_predictor_audit": crossfit_audit,
                  "models": fit_results, "prior_training_references": prior,
                  "fit_gate": cfg["fit_gate"], "scope": {**cfg["scope"], "validation_rows_read": 0,
                      "test_rows_dereferenced": 0, "reserve_access": False, "simulator_calls": 0,
                      "planner_calls": 0, "closed_loop": False},
                  "wall_time_seconds": time.perf_counter() - started,
                  "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}}
        result_path = out / "result.json"
        report_path = ROOT / cfg["outputs"]["report"]
        _atomic_json(result_path, result)
        report_path.write_text(render_report(result), encoding="utf-8")
        manifest = {"schema": "phase15_d161_fit_boundary_manifest_v1", "status": result["status"],
                    "source_commit": lock["source_commit"], "lock_sha256": lock_sha,
                    "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
                    "parent_sha256": lock["parent_sha256"], "dataset_sha256": lock["dataset_sha256"],
                    "checkpoints": {key: {"best_sha256": val["checkpoint_sha256"],
                        "latest_sha256": val["latest_resume_sha256"], "history_sha256": val["history_sha256"]}
                        for key, val in fit_results.items()},
                    "reserve_access": False, "validation_rows_read": 0, "test_rows_dereferenced": 0,
                    "simulator_calls": 0, "planner_calls": 0, "closed_loop": False,
                    "wall_time_seconds": result["wall_time_seconds"]}
        _atomic_json(out / "manifest.json", manifest)
        _atomic_json(marker_path, {"schema": "phase15_d161_run_state_v1", "lock_sha256": lock_sha,
                                   "dataset_sha256": lock["dataset_sha256"], "status": "completed"})
        return result
    except Exception as exc:
        abort_path = out / "abort_attempts.json"
        attempts = _read_json(abort_path) if abort_path.exists() else {"schema": "phase15_d161_abort_attempts_v1", "attempts": []}
        attempts["attempts"].append({"at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__, "error": str(exc), "reserve_access": False,
            "validation_rows_read": 0, "test_rows_dereferenced": 0})
        _atomic_json(abort_path, attempts)
        raise


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D161 — Relational Fit Boundary", "",
             f"**Status:** `{result['status']}`. Training-only; no generalization claim.", "",
             "Only the D153 1,365-row / 140-layout training corpus was used. Validation, D157, test/reserve, simulator/planner, and closed-loop access were excluded.", "",
             "| Width | Representation | Status | Gate | Parameters | Epochs | Wall sec | Train macro regret | Catastrophic | Epsilon-optimal | High-margin pair | Best-set agreement | Collision recall |",
             "|---:|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, record in result["models"].items():
        metric = record["metrics_training_only"]
        width, name = key.split("/", 1)
        lines.append(f"| {record['width']} | `{name}` | `{record['status']}` | {record['fit_gate_passed']} | {record['parameter_count']} | {record['epochs_run']} | {record['wall_time_seconds']:.1f} | {metric['layout_macro_normalized_selected_regret']:.5f} | {metric['catastrophic_fraction_layout_macro']:.5f} | {metric['epsilon_optimal_rate']:.5f} | {metric.get('high_margin_pairwise_accuracy_macro')} | {metric['best_action_set_agreement']:.5f} | {metric['collision_recall']} |")
    lines.extend(["", "## Interpretation", "",
                  "The frozen joint fit gate is layout-macro regret <= .05 and catastrophic fraction <= .05. `not_converged_within_budget` means the fixed cap was reached without three consecutive plateau checks; it is not evidence of impossibility. No held-out or closed-loop claim is made.", "",
                  "The six fit records and their optimizer-state checkpoints are in the ignored run directory; the tracked lock and manifest pin their hashes.", ""])
    return "\n".join(lines)
