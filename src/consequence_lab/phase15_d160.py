"""Training-only convergence continuation for the D159 fits."""

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
    _crossfit_train_consequences, _feature_set,
    _preference_loss_grad, _probabilities_from_exact, _risk_metrics, _sigmoid, _softmax,
)
from consequence_lab.phase15_d159 import _training_payload, _json_safe
from consequence_lab.phase15_decision_aligned import sha256_array
from consequence_lab.phase15_learning import layout_weights

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d160_convergence.yaml"
LOCK = "experiments/phase15_d160_convergence.lock.json"
SOURCE_FILES = (
    CONFIG, "reports/phase15_d160_protocol.md", "src/consequence_lab/phase15_d160.py",
    "scripts/freeze_phase15_d160_lock.py", "scripts/run_phase15_d160.py",
    "tests/test_phase15_d160.py", "src/consequence_lab/phase15_d158.py",
    "src/consequence_lab/phase15_d159.py", "src/consequence_lab/phase15_decision_aligned.py",
    "src/consequence_lab/phase15_learning.py", "src/consequence_lab/phase15_d155_stage_c.py",
    "src/consequence_lab/phase14.py", "configs/phase15_d155_stage_c.yaml",
    "second-brain/00_Project_Home.md", "second-brain/14_Decisions.md",
    "second-brain/16_Experiment_Log.md",
    "second-brain/phases/Phase_15_Decision_Aligned_Target_Study.md",
)


class D160Error(RuntimeError):
    """D160 integrity or execution failure."""


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D160Error(f"git {' '.join(args)} failed") from exc


def _parents(cfg: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}


def _lineage(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["parents"]
    d153_lock, d153_manifest = _read_json(ROOT / p["d153_lock"]), _read_json(ROOT / p["d153_manifest"])
    d159_lock = _read_json(ROOT / p["d159_lock"])
    d159_result, d159_manifest = _read_json(ROOT / p["d159_result"]), _read_json(ROOT / p["d159_manifest"])
    if (d153_lock.get("schema") != "phase15_d153_coverage_scaling_lock_v3"
            or d153_lock.get("reused_corpus", {}).get("path") != p["d153_training_rows"]
            or d153_lock.get("reused_corpus", {}).get("sha256") != sha256_file(ROOT / p["d153_training_rows"])
            or d153_manifest.get("fresh_dataset_sha256") != sha256_file(ROOT / p["d153_training_rows"])
            or d153_manifest.get("reserve_access") is not False
            or d153_manifest.get("test_rows_indexed_or_scored") != 0):
        raise D160Error("D153 training-only lineage check failed")
    if (d159_lock.get("schema") != "phase15_d159_training_fit_lock_v2"
            or d159_lock.get("dataset_sha256") != sha256_file(ROOT / p["d153_training_rows"])
            or d159_manifest.get("schema") != "phase15_d159_training_fit_manifest_v1"
            or d159_manifest.get("status") != d159_result.get("status")
            or d159_manifest.get("lock_sha256") != sha256_file(ROOT / p["d159_lock"])
            or d159_manifest.get("result_sha256") != sha256_file(ROOT / p["d159_result"])
            or d159_result.get("scope", {}).get("reserve_access") is not False
            or d159_result.get("scope", {}).get("test_rows_dereferenced") != 0
            or d159_result.get("row_count") != 1365 or d159_result.get("layout_count") != 140):
        raise D160Error("D159 parent result/manifest/lock/scope check failed")
    expected_ckpts = {
        "r0_action_only": p["d159_r0_checkpoint"],
        "r0_action_predicted_outcome_steps": p["d159_predicted_checkpoint"],
        "r0_action_exact_outcome_steps_oracle": p["d159_oracle_checkpoint"],
    }
    for name, path in expected_ckpts.items():
        digest = sha256_file(ROOT / path)
        if (d159_manifest.get("checkpoint_sha256", {}).get(name) != digest
                or d159_result.get("models", {}).get(name, {}).get("checkpoint_sha256") != digest):
            raise D160Error(f"D159 checkpoint hash mismatch: {name}")
    # Check the result is the known completed training-only fit, not a partial abort.
    if d159_result.get("status") != "completed_training_only_fit_audit_oracle_control_passed":
        raise D160Error("D159 did not complete with its oracle capacity control")
    return {"d153_status": d153_manifest.get("status"), "d159_status": d159_result["status"],
            "reserve_access": False, "test_rows_dereferenced": 0}


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D160Error("commit D160 protocol/config/source/tests/vault before freezing")
    if cfg.get("schema") != "phase15_d160_convergence_v1" or cfg.get("status") != "training_only_convergence_audit_frozen":
        raise D160Error("D160 config schema/status mismatch")
    scope = cfg.get("scope", {})
    if scope.get("training_only") is not True or any(value is not False for key, value in scope.items() if key != "training_only"):
        raise D160Error("D160 scope must be training-only with all forbidden flags false")
    lineage = _lineage(cfg)
    payload = _training_payload()
    if len(payload["states"]) != cfg["continuation"]["rows"] or len(set(payload["layouts"].tolist())) != cfg["continuation"]["layouts"]:
        raise D160Error("D160 source training corpus dimensions changed")
    return {
        "schema": "phase15_d160_convergence_lock_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "config_sha256": sha256_file(ROOT / CONFIG), "protocol_sha256": sha256_file(ROOT / cfg["protocol"]),
        "parent_sha256": _parents(cfg), "lineage_audit": lineage,
        "dataset_sha256": sha256_file(ROOT / cfg["parents"]["d153_training_rows"]),
        "state_sha256": sha256_array(payload["states"]), "summary_sha256": sha256_array(payload["summary"]),
        "regret_sha256": sha256_array(payload["regret"]),
        "layout_set_sha256": hashlib.sha256("\n".join(sorted(set(payload["layouts"].tolist()))).encode()).hexdigest(),
        "row_count": int(len(payload["states"])), "layout_count": len(set(payload["layouts"].tolist())),
        "scope": {**scope, "validation_rows_read": 0, "test_rows_dereferenced": 0, "reserve_access": False},
        "outputs": cfg["outputs"], "continuation": cfg["continuation"],
    }


def _validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D160Error("D160 execution requires a clean committed tree")
    lock_path = ROOT / cfg["outputs"]["lock"]
    if not lock_path.exists():
        raise D160Error("D160 lock is missing")
    lock = _read_json(lock_path)
    if lock.get("schema") != "phase15_d160_convergence_lock_v1":
        raise D160Error("D160 lock schema mismatch")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    if (lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("protocol_sha256") != sha256_file(ROOT / cfg["protocol"])
            or lock.get("source_sha256") != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
            or lock.get("parent_sha256") != _parents(cfg)
            or lock.get("dataset_sha256") != sha256_file(ROOT / cfg["parents"]["d153_training_rows"])):
        raise D160Error("D160 source/config/parent hashes changed after lock")
    _lineage(cfg)
    payload = _training_payload()
    if (lock.get("state_sha256") != sha256_array(payload["states"])
            or lock.get("summary_sha256") != sha256_array(payload["summary"])
            or lock.get("regret_sha256") != sha256_array(payload["regret"])):
        raise D160Error("D160 training corpus fingerprint mismatch")
    out = ROOT / cfg["outputs"]["run_dir"]
    if out.exists():
        raise D160Error("D160 is one-shot; output directory already exists")
    return cfg, lock


def _convergence_status(history: list[dict[str, Any]], epoch: int, cfg: dict[str, Any]) -> dict[str, Any]:
    rule = cfg["continuation"]
    if epoch < int(rule["min_additional_epochs_before_check"]):
        return {"converged": False, "reason": "minimum_epoch_requirement_not_met"}
    spacing = int(rule["convergence_check_spacing_epochs"])
    checks = [r for r in history if r["epoch"] <= epoch and r["epoch"] % spacing == 0]
    if len(checks) < 5:
        return {"converged": False, "reason": "insufficient_250_epoch_metric_history"}
    latest = checks[-5:]
    if [r["epoch"] for r in latest] != [epoch - 4 * spacing, epoch - 3 * spacing, epoch - 2 * spacing, epoch - spacing, epoch]:
        return {"converged": False, "reason": "noncontiguous_stability_checks"}
    recent_start = epoch - int(rule["convergence_window_epochs"])
    previous_start = recent_start - int(rule["convergence_window_epochs"])
    recent = [r["objective"] for r in history if recent_start < r["epoch"] <= epoch]
    previous = [r["objective"] for r in history if previous_start < r["epoch"] <= recent_start]
    if not recent or not previous:
        return {"converged": False, "reason": "objective_window_missing"}
    previous_best, recent_best = min(previous), min(recent)
    relative_improvement = (previous_best - recent_best) / max(abs(previous_best), 1e-12)
    regret_range = max(r["macro_regret"] for r in latest) - min(r["macro_regret"] for r in latest)
    catastrophic_range = max(r["catastrophic_fraction"] for r in latest) - min(r["catastrophic_fraction"] for r in latest)
    return {"converged": bool(relative_improvement <= float(rule["objective_relative_improvement_tolerance"])
                               and regret_range <= float(rule["decision_metric_range_tolerance"])
                               and catastrophic_range <= float(rule["decision_metric_range_tolerance"])),
            "relative_objective_improvement": float(relative_improvement),
            "previous_window_best_objective": float(previous_best),
            "recent_window_best_objective": float(recent_best),
            "macro_regret_range_last_five_checks": float(regret_range),
            "catastrophic_range_last_five_checks": float(catastrophic_range),
            "check_epochs": [int(r["epoch"]) for r in latest]}


def _load_checkpoint(model: RelationalCandidateModel, path: Path) -> None:
    with np.load(path, allow_pickle=False) as checkpoint:
        expected = ("x_mean", "x_std", "ws", "bs", "wc", "bc", "wctx", "bctx", "wscore", "bscore", "wrisk", "brisk")
        if set(checkpoint.files) != set(expected):
            raise D160Error("D159 checkpoint field set mismatch")
        for name in expected:
            value = np.asarray(checkpoint[name], dtype=np.float64)
            if value.shape != np.asarray(getattr(model, name)).shape or not np.isfinite(value).all():
                raise D160Error(f"D159 checkpoint shape/finite mismatch for {name}")
            getattr(model, name)[...] = value


def _continue_one(name: str, checkpoint_path: Path, states: np.ndarray, actions: np.ndarray,
                  consequences: np.ndarray, mask: tuple[float, float], regret: np.ndarray,
                  collision: np.ndarray, summary: np.ndarray, layouts: np.ndarray, model_cfg: dict[str, Any],
                  cfg: dict[str, Any], metrics_cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    model = RelationalCandidateModel(model_cfg["width"], model_cfg["seed"],
                                     model_cfg["learning_rate"], model_cfg["l2"],
                                     model_cfg["beta1"], model_cfg["beta2"], model_cfg["epsilon"])
    model.initialize(states)
    expected_mean, expected_std = model.x_mean.copy(), model.x_std.copy()
    _load_checkpoint(model, checkpoint_path)
    if not (np.array_equal(model.x_mean, expected_mean) and np.array_equal(model.x_std, expected_std)):
        raise D160Error("pinned D159 checkpoint normalization differs from D153 training-only statistics")
    # Optimizer moments deliberately reset because D159 did not serialize them.
    params = model.parameters()
    first, second = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
    state_weight = layout_weights(layouts)
    candidate_weight = state_weight[:, None] / 9.0
    teacher = _softmax(-regret / float(cfg["continuation"]["listwise_temperature"]))
    pair_i, pair_j = np.triu_indices(9, 1)
    zero_pairs = np.zeros((len(states), len(pair_i)), dtype=np.float64)
    risk_target = collision
    rate = float(cfg["continuation"]["learning_rate"])
    beta1, beta2, epsilon = (float(cfg["continuation"][k]) for k in ("beta1", "beta2", "epsilon"))
    l2, risk_weight = float(cfg["continuation"]["l2"]), float(cfg["continuation"]["collision_loss_weight"])
    def evaluate(epoch: int) -> dict[str, Any]:
        utility, risk_logits = model._forward(states, actions, consequences, mask)
        pref, _ = _preference_loss_grad(utility, regret, teacher, state_weight,
                                        zero_pairs, pair_i, pair_j, "listwise_tau_025", 0.0)
        risk = _sigmoid(risk_logits)
        bce = -float(np.sum(candidate_weight * (risk_target * np.log(np.maximum(risk, 1e-12))
                                               + (1.0 - risk_target) * np.log(np.maximum(1.0 - risk, 1e-12)))))
        objective = float(pref + risk_weight * bce)
        metrics = _candidate_metrics(regret, -utility, summary, layouts, metrics_cfg)
        risk_metrics = _risk_metrics(risk, collision > 0.5, layouts, 0.5)
        return {"epoch": epoch, "objective": objective, "preference_loss": float(pref),
                "collision_bce": bce,
                "macro_regret": float(metrics["layout_macro_normalized_selected_regret"]),
                "catastrophic_fraction": float(metrics["catastrophic_fraction_layout_macro"]),
                "epsilon_optimal_rate": float(metrics["epsilon_optimal_rate"]),
                "best_action_set_agreement": float(metrics["best_action_set_agreement"]),
                "high_margin_pairwise_accuracy": metrics.get("high_margin_pairwise_accuracy_macro"),
                "safe_vs_collision_pair_accuracy": metrics.get("safe_vs_collision_pair_accuracy"),
                "collision_action_selection_rate": metrics["collision_action_selection_rate"],
                "collision_risk_recall_at_0_5": risk_metrics["collision_recall"],
                "collision_risk_precision_at_0_5": risk_metrics["precision"]}

    history = [evaluate(0)]
    best_record = dict(history[0])
    best_params = [p.copy() for p in params]
    consecutive = 0
    convergence_checks: list[dict[str, Any]] = []
    total_epochs = int(cfg["continuation"]["max_additional_epochs"])
    eval_every = int(cfg["continuation"]["evaluate_every_epochs"])
    for epoch in range(1, total_epochs + 1):
        utility, risk_logits, cache = model._forward(states, actions, consequences, mask, cache=True)
        pref, grad_utility = _preference_loss_grad(utility, regret, teacher, state_weight,
                                                  zero_pairs, pair_i, pair_j, "listwise_tau_025", 0.0)
        risk = _sigmoid(risk_logits)
        grad_risk = risk_weight * candidate_weight * (risk - risk_target)
        grads = model._backward(grad_utility, grad_risk, cache)
        for idx, (param, grad) in enumerate(zip(params, grads, strict=True)):
            if param.ndim > 0 and id(param) not in {id(model.bs), id(model.bc), id(model.bctx)}:
                grad = grad + l2 * param
            first[idx] = beta1 * first[idx] + (1.0 - beta1) * grad
            second[idx] = beta2 * second[idx] + (1.0 - beta2) * grad * grad
            param -= rate * (first[idx] / (1.0 - beta1**epoch)) / (
                np.sqrt(second[idx] / (1.0 - beta2**epoch)) + epsilon)
        if epoch % eval_every == 0:
            record = evaluate(epoch)
            history.append(record)
            if record["objective"] < best_record["objective"] - 1e-12:
                best_record, best_params = dict(record), [p.copy() for p in params]
            spacing = int(cfg["continuation"]["convergence_check_spacing_epochs"])
            if epoch >= int(cfg["continuation"]["min_additional_epochs_before_check"]) and epoch % spacing == 0:
                check = _convergence_status(history, epoch, cfg)
                convergence_checks.append({"epoch": epoch, **check})
                consecutive = consecutive + 1 if check["converged"] else 0
                if consecutive >= int(cfg["continuation"]["required_consecutive_converged_checks"]):
                    break
    for param, value in zip(params, best_params, strict=True):
        param[...] = value
    final_metrics = evaluate(int(best_record["epoch"]))
    utility, risk_logits = model._forward(states, actions, consequences, mask)
    risk_prob = _sigmoid(risk_logits)
    decision_metrics = _candidate_metrics(regret, -utility, summary, layouts, metrics_cfg)
    decision_metrics.update(_risk_metrics(risk_prob, collision > 0.5, layouts, 0.5))
    decision_metrics["selected_risk_mean"] = float(np.mean(risk_prob[np.arange(len(states)), np.argmax(utility, axis=1)]))
    ckpt = out / f"{name}_converged_or_budget_checkpoint.npz"
    np.savez_compressed(ckpt, **{key: getattr(model, key) for key in
        ("x_mean", "x_std", "ws", "bs", "wc", "bc", "wctx", "bctx", "wscore", "bscore", "wrisk", "brisk")})
    trace_path = out / f"{name}_history.json"
    trace_path.write_text(json.dumps(_json_safe(history), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "converged" if consecutive >= int(cfg["continuation"]["required_consecutive_converged_checks"])
            else "not_converged_within_budget", "epochs_run": int(history[-1]["epoch"]),
            "best_added_epoch": int(best_record["epoch"]), "best_training_objective": float(best_record["objective"]),
            "metrics_training_only": {**final_metrics, **decision_metrics}, "convergence_checks": convergence_checks,
            "checkpoint_path": ckpt.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256_file(ckpt),
            "history_path": trace_path.relative_to(ROOT).as_posix(), "history_sha256": sha256_file(trace_path),
            "parameter_count": int(sum(p.size for p in params)),
            "optimizer_moments_reset": True, "adam_state_resumed": False}


def run_d160() -> dict[str, Any]:
    cfg, lock = _validate_lock()
    started = time.perf_counter()
    out = ROOT / cfg["outputs"]["run_dir"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        payload = _training_payload()
        actions, _ = _action_inputs(payload["states"])
        cross_cfg = {"parents": {"d155_stage_c_config": cfg["parents"]["d155_stage_c_config"]}}
        crossfit_data = {"train_r0": payload["states"], "train_summary": payload["summary"],
                         "train_layouts": payload["layouts"]}
        predicted_outcome, predicted_steps, crossfit_audit = _crossfit_train_consequences(crossfit_data, cross_cfg)
        exact_outcome, exact_steps = _probabilities_from_exact(payload["summary"])
        reps = {
            "r0_action_only": _feature_set(predicted_outcome, predicted_steps, "r0_action_only"),
            "r0_action_predicted_outcome_steps": _feature_set(predicted_outcome, predicted_steps, "r0_action_predicted_outcome_steps"),
            "r0_action_exact_outcome_steps_oracle": _feature_set(exact_outcome, exact_steps, "r0_action_exact_outcome_steps"),
        }
        d159_cfg = yaml.safe_load((ROOT / "configs/phase15_d159_training_fit_v2.yaml").read_text(encoding="utf-8"))
        metrics_cfg = yaml.safe_load((ROOT / cfg["parents"]["d155_stage_c_config"]).read_text(encoding="utf-8"))["metrics"]
        results = {}
        for name, (consequences, mask) in reps.items():
            source_key = {"r0_action_only": "d159_r0_checkpoint",
                          "r0_action_predicted_outcome_steps": "d159_predicted_checkpoint",
                          "r0_action_exact_outcome_steps_oracle": "d159_oracle_checkpoint"}[name]
            results[name] = _continue_one(name, ROOT / cfg["parents"][source_key], payload["states"], actions,
                consequences, mask, payload["regret"], payload["collision"], payload["summary"], payload["layouts"],
                d159_cfg["model"], cfg, metrics_cfg, out)
        result = {"schema": "phase15_d160_convergence_result_v1", "status": "completed_training_only_convergence_audit",
                  "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
                  "dataset_sha256": lock["dataset_sha256"], "row_count": lock["row_count"],
                  "layout_count": lock["layout_count"], "state_sha256": lock["state_sha256"],
                  "regret_sha256": lock["regret_sha256"], "crossfit_predictor_audit": crossfit_audit,
                  "models": results, "scope": {**cfg["scope"], "validation_rows_read": 0,
                    "test_rows_dereferenced": 0, "reserve_access": False, "simulator_calls": 0,
                    "planner_calls": 0, "closed_loop": False},
                  "time_seconds": time.perf_counter() - started,
                  "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}}
        result_path = out / "result.json"
        report_path = ROOT / cfg["outputs"]["report"]
        result_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_path.write_text(render_report(result), encoding="utf-8")
        manifest = {"schema": "phase15_d160_convergence_manifest_v1", "status": result["status"],
                    "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
                    "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
                    "parent_sha256": lock["parent_sha256"], "dataset_sha256": lock["dataset_sha256"],
                    "model_checkpoint_sha256": {k: v["checkpoint_sha256"] for k, v in results.items()},
                    "history_sha256": {k: v["history_sha256"] for k, v in results.items()},
                    "reserve_access": False, "test_rows_dereferenced": 0, "validation_rows_read": 0,
                    "simulator_calls": 0, "planner_calls": 0, "closed_loop": False,
                    "elapsed_seconds": result["time_seconds"]}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        (out / "abort.json").write_text(json.dumps({"schema": "phase15_d160_abort_v1",
            "status": "aborted_or_incomplete", "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
            "error_type": type(exc).__name__, "error": str(exc), "reserve_access": False,
            "test_rows_dereferenced": 0}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D160 — Training-only convergence continuation", "",
             f"**Status:** `{result['status']}`. This is a training-fit diagnostic, not a generalization result.", "",
             "Only the 1,365-row / 140-layout D153 training corpus was opened. D152 validation, D157, test, and reserve rows were not read. No simulator, planner, or closed-loop interaction occurred.", "",
             "| Representation | Status | Added epochs | Best epoch | Best objective | Macro regret | Catastrophic | Epsilon-optimal | Best-set agreement | Collision recall |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, rec in result["models"].items():
        m = rec["metrics_training_only"]
        lines.append(f"| `{name}` | `{rec['status']}` | {rec['epochs_run']} | {rec['best_added_epoch']} | {rec['best_training_objective']:.6f} | {m['layout_macro_normalized_selected_regret']:.5f} | {m['catastrophic_fraction_layout_macro']:.5f} | {m['epsilon_optimal_rate']:.5f} | {m['best_action_set_agreement']:.5f} | {m['collision_recall']:.5f} |")
    lines.extend(["", "## Interpretation", "",
        "The D159 checkpoint was warm-started and Adam moments were reset because optimizer state was not saved. A `not_converged_within_budget` status means only that this frozen continuation rule did not trigger within 10,000 additional epochs; it is not an impossibility or representation-insufficiency result.", "",
        "This same-corpus continuation is descriptive only. It does not authorize held-out scoring, DAgger, closed-loop control, validation/D157 access, world-model/JEPA work, or reserve/test evaluation.", ""])
    return "\n".join(lines)
