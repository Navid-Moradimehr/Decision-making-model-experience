"""Training-only audit of D158 relational decision-model fit."""

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
    RelationalCandidateModel, _action_inputs, _candidate_metrics, _crossfit_train_consequences,
    _feature_set, _fit_settings, _preference_loss_grad, _probabilities_from_exact, _sigmoid,
    _softmax,
)
from consequence_lab.phase15_decision_aligned import regrets_from_exact_summaries, sha256_array
from consequence_lab.phase15_learning import layout_weights

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d159_training_fit_v2.yaml"
LOCK = "experiments/phase15_d159_training_fit_v2.lock.json"
SOURCE_FILES = (
    "configs/phase15_d159_training_fit.yaml", CONFIG, "reports/phase15_d159_protocol.md",
    "reports/phase15_d159_prelock_amendment_v1.md", "src/consequence_lab/phase15_d159.py",
    "scripts/freeze_phase15_d159_lock.py", "scripts/run_phase15_d159.py",
    "tests/test_phase15_d159.py", "src/consequence_lab/phase15_d158.py",
    "src/consequence_lab/phase15_d155_stage_c.py", "src/consequence_lab/phase15_decision_aligned.py",
    "src/consequence_lab/phase15_learning.py", "src/consequence_lab/phase15_d155.py",
    "src/consequence_lab/phase14.py", "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase15_d157.py", "src/consequence_lab/phase15_d156.py",
    "src/consequence_lab/phase15_outcome_summary.py", "src/consequence_lab/phase15_representation_screen.py",
    "configs/phase15_d155_stage_c.yaml",
    "second-brain/00_Project_Home.md", "second-brain/14_Decisions.md",
    "second-brain/16_Experiment_Log.md", "second-brain/phases/Phase_15_Decision_Aligned_Target_Study.md",
)


class D159Error(RuntimeError):
    """D159 protocol, integrity, or execution error."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D159Error(f"git {' '.join(args)} failed") from exc


def _config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _file_hashes(paths: dict[str, str]) -> dict[str, str]:
    return {key: sha256_file(ROOT / value) for key, value in paths.items()}


def _training_payload() -> dict[str, Any]:
    cfg = _config()
    path = ROOT / cfg["parents"]["d153_training_rows"]
    with np.load(path, allow_pickle=False) as data:
        # This is the only state/label archive opened by D159.
        states = np.asarray(data["x_R0"], dtype=np.float64)
        summary = np.asarray(data["summary_R4"], dtype=np.float64)
        layouts = np.asarray(data["layouts"]).astype(str)
    if (states.shape != (1365, 61) or summary.shape != (1365, 9, 7)
            or layouts.shape != (1365,) or len(set(layouts.tolist())) != 140
            or not np.isfinite(states).all() or not np.isfinite(summary).all()):
        raise D159Error("D153 training-only corpus shape/layout/finite check failed")
    if not np.allclose(summary[:, :, :4].sum(axis=2), 1.0, atol=0.0, rtol=0.0):
        raise D159Error("training corpus outcome categories are not normalized")
    if np.any(summary[:, :, 1] != 0.0):
        raise D159Error("training corpus contains active-zero outcome branches")
    regrets = regrets_from_exact_summaries(summary)
    if regrets.shape != (1365, 9) or not np.isfinite(regrets).all():
        raise D159Error("teacher regret target contract failed")
    return {"states": states, "summary": summary, "layouts": layouts, "regret": regrets,
            "collision": summary[:, :, 3].copy()}


def _train_only_fit(model: RelationalCandidateModel, states: np.ndarray, actions: np.ndarray,
                    consequences: np.ndarray, mask: tuple[float, float], regret: np.ndarray,
                    collision: np.ndarray, layouts: np.ndarray, settings: dict[str, Any]
                    ) -> dict[str, Any]:
    """Fit/checkpoint from training loss only; this function accepts no eval arrays."""
    model.initialize(states)
    state_weight = layout_weights(layouts)
    candidate_weight = state_weight[:, None] / 9.0
    teacher = _softmax(-np.asarray(regret, dtype=np.float64) / float(settings["listwise_temperature"]))
    params = model.parameters()
    first = [np.zeros_like(param) for param in params]
    second = [np.zeros_like(param) for param in params]
    best_params: list[np.ndarray] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    trace: list[dict[str, float | int]] = []
    pair_i, pair_j = np.triu_indices(9, 1)
    zero_pairs = np.zeros((len(states), len(pair_i)), dtype=np.float64)
    for epoch in range(int(settings["max_epochs"])):
        utility, risk_logit, cache = model._forward(states, actions, consequences, mask, cache=True)
        preference_loss, grad_utility = _preference_loss_grad(
            utility, regret, teacher, state_weight, zero_pairs, pair_i, pair_j,
            "listwise_tau_025", 0.0,
        )
        risk = _sigmoid(risk_logit)
        risk_target = np.asarray(collision, dtype=np.float64)
        grad_risk = float(settings["collision_loss_weight"]) * candidate_weight * (risk - risk_target)
        gradients = model._backward(grad_utility, grad_risk, cache)
        step = epoch + 1
        for idx, (param, grad) in enumerate(zip(params, gradients, strict=True)):
            if param.ndim > 0 and id(param) not in {id(model.bs), id(model.bc), id(model.bctx)}:
                grad = grad + model.l2 * param
            first[idx] = model.beta1 * first[idx] + (1.0 - model.beta1) * grad
            second[idx] = model.beta2 * second[idx] + (1.0 - model.beta2) * grad * grad
            param -= model.lr * (first[idx] / (1.0 - model.beta1**step)) / (
                np.sqrt(second[idx] / (1.0 - model.beta2**step)) + model.epsilon)

        # Use only the frozen training objective to select checkpoints.
        updated_utility, updated_risk_logits = model._forward(states, actions, consequences, mask)
        updated_preference, _ = _preference_loss_grad(
            updated_utility, regret, teacher, state_weight, zero_pairs, pair_i, pair_j,
            "listwise_tau_025", 0.0,
        )
        updated_risk = _sigmoid(updated_risk_logits)
        updated_collision = -float(np.sum(candidate_weight * (
            risk_target * np.log(np.maximum(updated_risk, 1e-12))
            + (1.0 - risk_target) * np.log(np.maximum(1.0 - updated_risk, 1e-12)))))
        objective = float(updated_preference + settings["collision_loss_weight"] * updated_collision)
        if epoch == 0 or (epoch + 1) % 25 == 0:
            trace.append({"epoch": epoch + 1, "preference_loss": float(updated_preference),
                          "collision_bce": float(updated_collision), "total_loss": objective})
        if objective < best_loss - 1e-9:
            best_loss = objective
            best_epoch = epoch + 1
            best_params = [param.copy() for param in params]
            stale = 0
        else:
            stale += 1
        if stale >= int(settings["patience"]):
            break
    if best_params is None:
        raise D159Error("train-only optimizer did not produce a checkpoint")
    for param, value in zip(params, best_params, strict=True):
        param[...] = value
    model.epochs_run = epoch + 1
    model.best_epoch = best_epoch
    model.training_objective_loss = best_loss
    model.parameter_count = int(sum(param.size for param in model.parameters()))
    return {"epochs_run": model.epochs_run, "best_epoch": best_epoch,
            "best_training_objective_loss": best_loss, "training_loss_trace": trace,
            "parameter_count": model.parameter_count}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_lock() -> dict[str, Any]:
    cfg = _config()
    if cfg.get("schema") != "phase15_d159_training_fit_v2" or cfg.get("status") != "training_only_fit_diagnostic_path_amendment_frozen":
        raise D159Error("D159 config schema/status mismatch")
    if cfg.get("scope", {}).get("training_only") is not True:
        raise D159Error("D159 must be training-only")
    for flag in ("d152_validation_read", "d157_read", "test_rows", "reserve_access", "simulator_calls",
                 "planner_calls", "new_transitions_or_episodes", "closed_loop", "dagger", "world_model", "jepa"):
        if cfg["scope"].get(flag) is not False:
            raise D159Error(f"D159 forbidden scope flag enabled: {flag}")
    if _git("status", "--porcelain"):
        raise D159Error("commit D159 protocol, config, implementation, and tests before locking")
    parent_hashes = _file_hashes(cfg["parents"])
    dataset = ROOT / cfg["parents"]["d153_training_rows"]
    d153_manifest = _read_json(ROOT / cfg["parents"]["d153_manifest"])
    d153_lock = _read_json(ROOT / cfg["parents"]["d153_lock"])
    reused_corpus = d153_lock.get("reused_corpus", {})
    if (reused_corpus.get("path") != cfg["parents"]["d153_training_rows"]
            or reused_corpus.get("sha256") != sha256_file(dataset)
            or d153_manifest.get("fresh_dataset_sha256") != sha256_file(dataset)
            or d153_lock.get("schema") != "phase15_d153_coverage_scaling_lock_v3"
            or d153_manifest.get("reserve_access") is not False
            or d153_manifest.get("test_rows_indexed_or_scored") != 0):
        raise D159Error("D153 training corpus lineage/scope check failed")
    payload = _training_payload()
    _, actions = _action_inputs(payload["states"])
    actions_hash = sha256_array(actions)
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    return {
        "schema": "phase15_d159_training_fit_lock_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": source_hashes,
        "config_sha256": sha256_file(ROOT / CONFIG),
        "protocol_sha256": sha256_file(ROOT / cfg["protocol"]),
        "parent_sha256": parent_hashes,
        "dataset_sha256": sha256_file(dataset),
        "state_sha256": sha256_array(payload["states"]),
        "summary_sha256": sha256_array(payload["summary"]),
        "regret_sha256": sha256_array(payload["regret"]),
        "action_identity_sha256": actions_hash,
        "layout_count": len(set(payload["layouts"].tolist())),
        "row_count": int(len(payload["states"])),
        "layout_set_sha256": hashlib.sha256("\n".join(sorted(set(payload["layouts"].tolist()))).encode()).hexdigest(),
        "scope": {**cfg["scope"], "test_rows_dereferenced": 0, "reserve_access": False},
        "outputs": cfg["outputs"],
    }


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _config()
    if _git("status", "--porcelain"):
        raise D159Error("D159 execution requires a clean committed tree")
    lock = _read_json(ROOT / cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d159_training_fit_lock_v2" or lock.get("scope") != {
            **cfg["scope"], "test_rows_dereferenced": 0, "reserve_access": False}:
        raise D159Error("D159 lock schema mismatch")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    if (lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("protocol_sha256") != sha256_file(ROOT / cfg["protocol"])
            or lock.get("dataset_sha256") != sha256_file(ROOT / cfg["parents"]["d153_training_rows"])
            or lock.get("source_sha256") != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
            or lock.get("parent_sha256") != _file_hashes(cfg["parents"])):
        raise D159Error("D159 lock/source/parent hash mismatch")
    payload = _training_payload()
    if (len(payload["states"]) != lock["row_count"]
            or len(set(payload["layouts"].tolist())) != lock["layout_count"]
            or sha256_array(payload["states"]) != lock["state_sha256"]
            or sha256_array(payload["summary"]) != lock["summary_sha256"]
            or sha256_array(payload["regret"]) != lock["regret_sha256"]):
        raise D159Error("D159 training-only data fingerprint mismatch")
    out = ROOT / cfg["outputs"]["run_dir"]
    if out.exists():
        raise D159Error("D159 is one-shot; output path already exists")
    return cfg, lock


def run_d159() -> dict[str, Any]:
    cfg, lock = validate_lock()
    started = time.perf_counter()
    out = ROOT / cfg["outputs"]["run_dir"]
    out.mkdir(parents=True, exist_ok=False)
    payload = _training_payload()
    actions, _ = _action_inputs(payload["states"])
    cross_cfg = {"parents": {"d155_stage_c_config": cfg["parents"]["d155_stage_c_config"]}}
    crossfit_data = {"train_r0": payload["states"], "train_summary": payload["summary"],
                     "train_layouts": payload["layouts"]}
    predicted_outcome, predicted_steps, crossfit_audit = _crossfit_train_consequences(crossfit_data, cross_cfg)
    exact_outcome, exact_steps = _probabilities_from_exact(payload["summary"])
    representations = {
        "r0_action_only": _feature_set(predicted_outcome, predicted_steps, "r0_action_only"),
        "r0_action_predicted_outcome_steps": _feature_set(predicted_outcome, predicted_steps, "r0_action_predicted_outcome_steps"),
        "r0_action_exact_outcome_steps_oracle": _feature_set(exact_outcome, exact_steps,
                                                               "r0_action_exact_outcome_steps"),
    }
    with (ROOT / cfg["parents"]["d155_stage_c_config"]).open(encoding="utf-8") as file:
        metrics_cfg = yaml.safe_load(file)["metrics"]
    model_cfg = cfg["model"]
    objective_cfg = {"pairwise_weight": 0.0}
    settings = _fit_settings(model_cfg, {
        "listwise_temperature": model_cfg["listwise_temperature"],
        "pairwise_minimum_regret_gap": 0.125, "safe_regret_at_most": 0.125,
        "catastrophic_regret_threshold": model_cfg["catastrophic_regret_threshold"],
        "pairwise_safe_collision_weight_multiplier": 3.0,
        "pairwise_safe_catastrophic_weight_multiplier": 3.0,
    }, objective_cfg)
    settings["max_epochs"] = int(model_cfg["max_epochs"])
    settings["patience"] = int(model_cfg["patience"])
    models: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    trace_by_model: dict[str, Any] = {}
    for idx, (name, (consequence, mask)) in enumerate(representations.items()):
        model = RelationalCandidateModel(
            model_cfg["width"], model_cfg["seed"], model_cfg["learning_rate"], model_cfg["l2"],
            model_cfg["beta1"], model_cfg["beta2"], model_cfg["epsilon"],
        )
        fit = _train_only_fit(model, payload["states"], actions, consequence, mask,
                              payload["regret"], payload["collision"], payload["layouts"], settings)
        utility, risk = model.predict(payload["states"], actions, consequence, mask)
        metrics = _candidate_metrics(payload["regret"], -utility, payload["summary"],
                                     payload["layouts"], metrics_cfg)
        selected = np.argmin(-utility, axis=1)
        selected_risk = risk[np.arange(len(risk)), selected]
        metrics["selected_collision_risk_mean"] = float(np.mean(selected_risk))
        metrics["collision_recall_at_0_5"] = float(np.sum((risk >= 0.5) & (payload["collision"] > 0.5))
                                                      / max(1, np.sum(payload["collision"] > 0.5)))
        metrics["collision_precision_at_0_5"] = float(np.sum((risk >= 0.5) & (payload["collision"] > 0.5))
                                                        / max(1, np.sum(risk >= 0.5)))
        gate = (metrics["layout_macro_normalized_selected_regret"] <= cfg["fit_gate"]["max_train_macro_regret"]
                and metrics["catastrophic_fraction_layout_macro"] <= cfg["fit_gate"]["max_train_catastrophic_fraction"])
        checkpoint = out / f"{name}.npz"
        np.savez_compressed(checkpoint, **{key: getattr(model, key) for key in
                                           ("x_mean", "x_std", "ws", "bs", "wc", "bc", "wctx", "bctx",
                                            "wscore", "bscore", "wrisk", "brisk")})
        checkpoint_hashes[name] = sha256_file(checkpoint)
        trace_by_model[name] = fit.pop("training_loss_trace")
        models[name] = {"fit": fit, "metrics_training_only": metrics,
                        "fit_gate_passed": bool(gate), "parameter_count": model.parameter_count,
                        "checkpoint_sha256": checkpoint_hashes[name]}
    oracle_name = "r0_action_exact_outcome_steps_oracle"
    oracle_passed = models[oracle_name]["fit_gate_passed"]
    overall_status = ("completed_training_only_fit_audit_oracle_control_passed"
                      if oracle_passed else "aborted_training_only_oracle_fit_gate_failed")
    result = {
        "schema": "phase15_d159_training_fit_result_v2", "status": overall_status,
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
        "dataset_sha256": lock["dataset_sha256"], "row_count": lock["row_count"],
        "layout_count": lock["layout_count"], "state_sha256": lock["state_sha256"],
        "regret_sha256": lock["regret_sha256"], "crossfit_predictor_audit": crossfit_audit,
        "models": models, "fit_gate": cfg["fit_gate"], "oracle_capacity_control_passed": oracle_passed,
        "scope": {**cfg["scope"], "test_rows_dereferenced": 0, "reserve_access": False,
                  "validation_rows_read": 0, "d157_rows_read": 0},
        "time_seconds": time.perf_counter() - started,
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    result_path = out / "result.json"
    trace_path = out / "training_loss_traces.json"
    report_path = ROOT / cfg["outputs"]["report"]
    result_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trace_path.write_text(json.dumps(_json_safe(trace_by_model), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_hash = sha256_file(result_path)
    report_path.write_text(render_report(result), encoding="utf-8")
    manifest = {
        "schema": "phase15_d159_training_fit_manifest_v1", "status": overall_status,
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
        "result_sha256": result_hash, "report_sha256": sha256_file(report_path),
        "loss_trace_sha256": sha256_file(trace_path), "checkpoint_sha256": checkpoint_hashes,
        "dataset_sha256": lock["dataset_sha256"], "parent_sha256": lock["parent_sha256"],
        "scope": result["scope"], "elapsed_seconds": result["time_seconds"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D159 — Training-only relational fit audit", "",
             f"**Status:** `{result['status']}`. Training-only exploratory diagnostic; no generalization claim.", "",
             "Only the 1,365-row / 140-layout D153 training corpus was opened. D152 validation, D157, test, and reserve data were not read. No simulator or planner was run.", "",
             "| Representation | Train macro regret | Catastrophic | Epsilon-optimal | Best-set agreement | Safe/collision pair | Fit gate | Epoch |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, record in result["models"].items():
        m, f = record["metrics_training_only"], record["fit"]
        lines.append(f"| `{name}` | {m['layout_macro_normalized_selected_regret']:.4f} | "
                     f"{m['catastrophic_fraction_layout_macro']:.4f} | {m['epsilon_optimal_rate']:.4f} | "
                     f"{m['best_action_set_agreement']:.4f} | {m['safe_vs_collision_pair_accuracy']} | "
                     f"{record['fit_gate_passed']} | {f['best_epoch']} / {f['epochs_run']} |")
    lines.extend(["", "## Interpretation", "",
                  "This audit can distinguish a train-only checkpoint-selection effect from the fixed model's ability to fit using these inputs. It cannot establish held-out transfer, causal feature necessity, or fundamental information insufficiency.", "",
                  f"Exact-consequence capacity control passed: **{result['oracle_capacity_control_passed']}**.", "",
                  "Reserve/test remains sealed. D159 closes after this training-only result; no D157 scoring, validation tuning, DAgger, closed-loop control, or world-model branch is authorized by this report.", "",
             "See `artifacts/runs/phase15_d159_training_fit_v2/result.json` and `manifest.json` for hashes, detailed metrics, and training traces.", ""])
    return "\n".join(lines)
