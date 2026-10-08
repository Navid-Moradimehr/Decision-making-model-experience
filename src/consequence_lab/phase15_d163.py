"""D163: offline consequence-prediction and frozen-scorer sensitivity audit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_d155_stage_c import _TreeNode
from consequence_lab.phase15_d158 import (
    _action_inputs, _candidate_metrics, _feature_set,
)
from consequence_lab.phase15_d157 import ACTION_COUNT
from consequence_lab.phase15_decision_aligned import regrets_from_exact_summaries
from consequence_lab.phase15_d162 import CHECKPOINTS, _load_student

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d163_error_decision_audit.yaml"
SOURCES = (
    CONFIG, "reports/phase15_d163_protocol.md", "src/consequence_lab/phase15_d163.py",
    "reports/phase15_d163_prelock_abort.md",
    "scripts/freeze_phase15_d163_lock.py", "scripts/run_phase15_d163.py", "tests/test_phase15_d163.py",
    "src/consequence_lab/phase15_d162.py", "src/consequence_lab/phase15_d158.py",
    "src/consequence_lab/phase15_d155_stage_c.py", "src/consequence_lab/phase15_decision_aligned.py",
    "src/consequence_lab/phase15_d157.py", "src/consequence_lab/phase14.py",
    "configs/phase15_d162_fresh_layout_relational.yaml",
    "configs/phase15_d155_stage_c.yaml", "reports/phase15_d162_result.md",
    "second-brain/00_Project_Home.md", "second-brain/14_Decisions.md",
    "second-brain/16_Experiment_Log.md",
    "second-brain/phases/Phase_15_Decision_Aligned_Target_Study.md",
)
PREDICTED_SCORER = "r0_action_predicted_outcome_steps"
EXACT_SCORER = "r0_action_exact_outcome_steps_oracle"


class D163Error(RuntimeError):
    """D163 provenance, parity, or analysis failure."""


def _read(path: str | Path) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _parent_paths(cfg: dict[str, Any]) -> dict[str, str]:
    p = cfg["parents"]
    paths = {k: v for k, v in p.items() if k != "d162_fold_predictors"}
    paths.update({f"d162_fold_{i}": path for i, path in enumerate(p["d162_fold_predictors"])})
    return paths


def _verify_saved_fold_digest(result: dict[str, Any], fold: int, path: str) -> None:
    """D162 stores serialized predictor-file hashes in result.json, not manifest.json."""
    expected = result.get("predictor_artifact_sha256", {}).get(f"fold_{fold}")
    if expected != sha256_file(ROOT / path):
        raise D163Error(f"serialized D162 fold predictor {fold} file digest mismatch")


def _verify_checkpoint_digest(d161_manifest: dict[str, Any], d161_result: dict[str, Any],
                              name: str, path: str) -> None:
    digest = sha256_file(ROOT / path)
    if (d161_manifest.get("checkpoints", {}).get(f"width_64/{name}", {}).get("best_sha256") != digest
            or d161_result.get("models", {}).get(f"width_64/{name}", {}).get("checkpoint_sha256") != digest):
        raise D163Error(f"frozen D161 checkpoint hash mismatch: {name}")


def _audit_inputs(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    p = cfg["parents"]
    lock, result, manifest = (_read(p[k]) for k in ("d162_lock", "d162_result", "d162_manifest"))
    if (lock.get("schema") != "phase15_d162_fresh_layout_relational_lock_v1"
            or result.get("status") != "completed_offline_fresh_layout_relational_evaluation"
            or manifest.get("status") != result.get("status")
            or manifest.get("lock_sha256") != sha256_file(ROOT / p["d162_lock"])
            or manifest.get("result_sha256") != sha256_file(ROOT / p["d162_result"])
            or manifest.get("dataset_sha256") != sha256_file(ROOT / p["d162_rows"])
            or manifest.get("accounting_sha256") != sha256_file(ROOT / p["d162_accounting"])
            or manifest.get("reserve_access") is not False or manifest.get("test_access") is not False
            or manifest.get("student_closed_loop") is not False or manifest.get("student_updates") is not False):
        raise D163Error("D162 artifact hash/status/scope audit failed")
    d161 = _read(p["d161_manifest"])
    d161_result = _read(p["d161_result"])
    if (d161.get("status") != d161_result.get("status")
            or d161.get("reserve_access") is not False
            or d161.get("test_rows_dereferenced") != 0):
        raise D163Error("D161 frozen scorer provenance/scope audit failed")
    with np.load(ROOT / p["d162_rows"], allow_pickle=False) as rows:
        states, summaries = rows["x_R0"].astype(np.float64), rows["summary_R4"].astype(np.float64)
        layouts = rows["layouts"].astype(str)
    if (states.shape != (1046, 61) or summaries.shape != (1046, 9, 7)
            or len(set(layouts.tolist())) != 100 or not np.isfinite(states).all()
            or not np.isfinite(summaries).all()):
        raise D163Error("D162 saved dataset shape/layout/value contract failed")
    if result.get("dataset_sha256") != sha256_file(ROOT / p["d162_rows"]):
        raise D163Error("D162 result dataset digest mismatch")
    for i, path in enumerate(p["d162_fold_predictors"]):
        payload = _read(path)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        expected = result["predictor_model_sha256"][i]
        if hashlib.sha256(canonical).hexdigest().upper() != expected:
            raise D163Error(f"serialized D162 fold predictor {i} digest mismatch")
        _verify_saved_fold_digest(result, i, path)
    for name, path_key in (("r0_action_only", "d161_r0_checkpoint"),
                           (PREDICTED_SCORER, "d161_predicted_checkpoint"),
                           (EXACT_SCORER, "d161_exact_checkpoint")):
        _verify_checkpoint_digest(d161, d161_result, name, p[path_key])
    return {"states": states, "summary": summaries, "layouts": layouts}, {
        "d162_lock_sha256": sha256_file(ROOT / p["d162_lock"]),
        "d162_result_sha256": sha256_file(ROOT / p["d162_result"]),
        "d162_manifest_sha256": sha256_file(ROOT / p["d162_manifest"]),
        "d162_rows_sha256": sha256_file(ROOT / p["d162_rows"]),
        "d162_accounting_sha256": sha256_file(ROOT / p["d162_accounting"]),
        "d161_result_sha256": sha256_file(ROOT / p["d161_result"]),
        "d161_manifest_sha256": sha256_file(ROOT / p["d161_manifest"]),
        "reserve_access": False, "test_access": False,
    }


def _load_tree(payload: dict[str, Any]) -> _TreeNode:
    value = np.asarray(payload["value"], dtype=np.float64)
    feature = int(payload["feature"])
    if feature < 0:
        return _TreeNode(value)
    return _TreeNode(value, feature, float(payload["threshold"]),
                     _load_tree(payload["left"]), _load_tree(payload["right"]))


def _predict_tree(root: _TreeNode, rows: np.ndarray, output_size: int) -> np.ndarray:
    x = np.asarray(rows, dtype=np.float64)
    out = np.empty((len(x), output_size), dtype=np.float64)
    for i, row in enumerate(x):
        node = root
        while node.feature >= 0:
            node = node.left if row[node.feature] <= node.threshold else node.right
            if node is None:
                raise D163Error("serialized CART tree has missing child")
        out[i] = node.value
    return out


def load_saved_predictions(cfg: dict[str, Any], states: np.ndarray, layouts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer with saved fold trees; this function never fits a predictor."""
    from consequence_lab.phase15_d155_stage_c import action_conditioned_inputs

    paths = cfg["parents"]["d162_fold_predictors"]
    fold_ids = np.asarray([int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) % len(paths)
                           for x in layouts], dtype=np.int64)
    inputs = action_conditioned_inputs(states)
    out_prob = np.empty((len(states), ACTION_COUNT, 3), dtype=np.float64)
    out_steps = np.empty((len(states), ACTION_COUNT), dtype=np.float64)
    for fold, path in enumerate(paths):
        rows = np.flatnonzero(fold_ids == fold)
        if not len(rows):
            continue
        payload = _read(path)
        mean, std = np.asarray(payload["x_mean"], float), np.asarray(payload["x_std"], float)
        if mean.shape != (70,) or std.shape != (70,) or np.any(std <= 0):
            raise D163Error("serialized predictor normalization is invalid")
        flat_ids = (rows[:, None] * ACTION_COUNT + np.arange(ACTION_COUNT)[None, :]).reshape(-1)
        normalized = (inputs.reshape(-1, 70)[flat_ids] - mean) / std
        class_values = _predict_tree(_load_tree(payload["outcome_tree"]), normalized, 3)
        step_values = _predict_tree(_load_tree(payload["steps_tree"]), normalized, 1)[:, 0]
        class_values = np.maximum(class_values, 1e-12)
        class_values /= class_values.sum(axis=1, keepdims=True)
        out_prob[rows] = class_values.reshape(len(rows), ACTION_COUNT, 3)
        out_steps[rows] = np.clip(step_values.reshape(len(rows), ACTION_COUNT), 0.0, 1.0)
    return out_prob, out_steps


def _component_metrics(exact: np.ndarray, predicted_prob: np.ndarray,
                       predicted_steps: np.ndarray, regrets: np.ndarray) -> dict[str, Any]:
    true_class = np.argmax(exact[:, :, [0, 2, 3]], axis=2)
    pred_class = np.argmax(predicted_prob, axis=2)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (true_class.ravel(), pred_class.ravel()), 1)
    class_scores = []
    for klass in range(3):
        tp = int(confusion[klass, klass])
        precision = tp / int(confusion[:, klass].sum()) if confusion[:, klass].sum() else 0.0
        recall = tp / int(confusion[klass, :].sum()) if confusion[klass, :].sum() else None
        class_scores.append({"class": ("success", "timeout", "collision")[klass],
                             "support": int(confusion[klass, :].sum()),
                             "precision": precision, "recall": recall})
    collision = true_class == 2
    p_collision = predicted_prob[:, :, 2]
    brier = float(np.mean((p_collision - collision.astype(float)) ** 2))
    p_true = np.take_along_axis(predicted_prob, true_class[:, :, None], axis=2)[:, :, 0]
    log_loss = float(-np.mean(np.log(np.maximum(p_true, 1e-12))))
    error = predicted_steps - exact[:, :, 4]
    by_outcome = {}
    for klass, name in enumerate(("success", "timeout", "collision")):
        mask = true_class == klass
        by_outcome[name] = {"candidate_count": int(mask.sum()),
                            "steps_mae": float(np.mean(np.abs(error[mask]))) if mask.any() else None,
                            "steps_rmse": float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None}
    collision_missed = collision & (pred_class != 2)
    bands = {"regret_lt_0.125": regrets < .125,
             "regret_0.125_to_0.5": (regrets >= .125) & (regrets < .5),
             "regret_ge_0.5": regrets >= .5}
    false_negative_by_regret = {}
    for name, band in bands.items():
        selected = collision & band
        false_negative_by_regret[name] = {"collision_count": int(selected.sum()),
                                          "missed_count": int((collision_missed & band).sum()),
                                          "recall": (float((selected & (pred_class == 2)).sum() / selected.sum())
                                                     if selected.any() else None)}
    return {"outcome_candidate_count": int(true_class.size), "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
            "per_class": class_scores, "collision_recall": class_scores[2]["recall"],
            "collision_false_negative_rate": (1.0 - class_scores[2]["recall"]
                                               if class_scores[2]["recall"] is not None else None),
            "collision_brier": brier, "multiclass_log_loss": log_loss,
            "steps_mae": float(np.mean(np.abs(error))),
            "steps_rmse": float(np.sqrt(np.mean(error ** 2))),
            "steps_mae_by_exact_terminal_outcome": by_outcome,
            "collision_misses_by_teacher_regret": false_negative_by_regret,
            "collision_probability_mean_on_true_collision": float(p_collision[collision].mean()) if collision.any() else None,
            "collision_probability_mean_on_safe_branch": float(p_collision[~collision].mean()) if (~collision).any() else None}


def _score_swaps(data: dict[str, np.ndarray], predicted_prob: np.ndarray,
                 predicted_steps: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    states, exact, layouts = data["states"], data["summary"], data["layouts"]
    regret = regrets_from_exact_summaries(exact)
    exact_prob, exact_steps = exact[:, :, [0, 2, 3]], exact[:, :, 4]
    actions, _ = _action_inputs(states)
    train_path = ROOT / cfg["parents"]["d153_training_rows"]
    with np.load(train_path, allow_pickle=False) as train:
        train_states = train["x_R0"].astype(np.float64)
    metrics_cfg = yaml.safe_load((ROOT / cfg["parents"]["d155_config"]).read_text(encoding="utf-8"))["metrics"]
    models = {}
    for name in (PREDICTED_SCORER, EXACT_SCORER):
        models[name] = _load_student(name, ROOT / CHECKPOINTS[name], train_states)
    components = _component_metrics(exact, predicted_prob, predicted_steps, regret)
    outcomes = {"predicted": predicted_prob, "exact": exact_prob}
    steps = {"predicted": predicted_steps, "exact": exact_steps}
    swap_metrics: dict[str, Any] = {}
    for outcome_source in ("predicted", "exact"):
        for steps_source in ("predicted", "exact"):
            case = f"outcome_{outcome_source}__steps_{steps_source}"
            consequence = np.concatenate((outcomes[outcome_source], steps[steps_source][:, :, None]), axis=2)
            swap_metrics[case] = {}
            for scorer_name, model in models.items():
                utility, _ = model.predict(states, actions, consequence, (1.0, 1.0))
                scores = -utility
                swap_metrics[case][scorer_name] = _candidate_metrics(
                    regret, scores, exact, layouts, metrics_cfg)
    # Recompute the unchanged D162 R0 arm as a software/data parity anchor.
    r0_name = "r0_action_only"
    r0 = _load_student(r0_name, ROOT / CHECKPOINTS[r0_name], train_states)
    blank = np.zeros((len(states), ACTION_COUNT, 4), dtype=np.float64)
    utility, _ = r0.predict(states, actions, blank, (0.0, 0.0))
    r0_metrics = _candidate_metrics(regret, -utility, exact, layouts, metrics_cfg)
    return {"component_prediction": components, "frozen_scorer_input_swaps": swap_metrics,
            "r0_parity_metrics": r0_metrics}


def analyze(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or _cfg()
    data, parent_hashes = _audit_inputs(cfg)
    pred_prob, pred_steps = load_saved_predictions(cfg, data["states"], data["layouts"])
    result = _score_swaps(data, pred_prob, pred_steps, cfg)
    regret = regrets_from_exact_summaries(data["summary"])
    train_path = ROOT / cfg["parents"]["d153_training_rows"]
    with np.load(train_path, allow_pickle=False) as train:
        train_states = train["x_R0"].astype(np.float64)
    actions, _ = _action_inputs(data["states"])
    metrics_cfg = yaml.safe_load((ROOT / cfg["parents"]["d155_config"]).read_text(encoding="utf-8"))["metrics"]
    d162_result = _read(cfg["parents"]["d162_result"])
    parity = {}
    for name, probs, steps, mask_key in (
        (PREDICTED_SCORER, pred_prob, pred_steps, "predicted"),
        (EXACT_SCORER, data["summary"][:, :, [0, 2, 3]], data["summary"][:, :, 4], "exact"),
    ):
        model = _load_student(name, ROOT / CHECKPOINTS[name], train_states)
        feature_name = "r0_action_predicted_outcome_steps" if mask_key == "predicted" else "r0_action_exact_outcome_steps_oracle"
        consequence, mask = _feature_set(probs, steps, feature_name)
        utility, _ = model.predict(data["states"], actions, consequence, mask)
        measured = _candidate_metrics(regret, -utility, data["summary"], data["layouts"], metrics_cfg)
        expected = d162_result["evaluation"]["model_metrics"][name]
        keys = ("layout_macro_normalized_selected_regret", "catastrophic_fraction_layout_macro",
                "epsilon_optimal_rate", "best_action_set_agreement", "collision_action_selection_rate")
        diffs = {key: abs(float(measured[key]) - float(expected[key])) for key in keys}
        parity[name] = {"max_absolute_metric_difference": max(diffs.values()), "differences": diffs,
                        "passed": max(diffs.values()) <= float(cfg["analysis"]["d162_parity_tolerance"])}
    r0_expected = d162_result["evaluation"]["model_metrics"]["r0_action_only"]
    keys = ("layout_macro_normalized_selected_regret", "catastrophic_fraction_layout_macro",
            "epsilon_optimal_rate", "best_action_set_agreement", "collision_action_selection_rate")
    r0_diff = max(abs(float(result["r0_parity_metrics"][k]) - float(r0_expected[k])) for k in keys)
    parity["r0_action_only"] = {"max_absolute_metric_difference": r0_diff,
                                "passed": r0_diff <= float(cfg["analysis"]["d162_parity_tolerance"])}
    if not all(item["passed"] for item in parity.values()):
        raise D163Error(f"D162 prediction/scorer parity failed: {parity}")
    return {"schema": "phase15_d163_error_decision_audit_result_v1",
            "status": "completed_exploratory_offline_component_sensitivity_audit",
            "parent_hashes": parent_hashes, "state_count": len(data["states"]),
            "layout_count": len(set(data["layouts"].tolist())),
            "branch_count": int(len(data["states"]) * ACTION_COUNT),
            "component_prediction": result["component_prediction"],
            "parity": parity, "r0_metrics": result["r0_parity_metrics"],
            "frozen_scorer_input_swaps": result["frozen_scorer_input_swaps"],
            "scope": {"new_simulator_or_planner_calls": False, "model_training_or_updates": False,
                      "student_closed_loop": False, "dagger": False, "world_model_or_jepa": False,
                      "reserve_or_test_access": False, "causal_claims": False}}


def _source_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in SOURCES}


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D163Error("commit D163 protocol/source/tests/vault before locking")
    if cfg.get("schema") != "phase15_d163_error_decision_audit_v1":
        raise D163Error("D163 config schema mismatch")
    for name in ("lock", "run_dir", "report"):
        if (ROOT / cfg["outputs"][name]).exists():
            raise D163Error(f"D163 output already exists: {name}")
    data, parents = _audit_inputs(cfg)
    return {"schema": "phase15_d163_error_decision_audit_lock_v1",
            "status": "locked_one_shot_offline_diagnostic",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": _git("rev-parse", "HEAD"), "source_sha256": _source_hashes(),
            "config_sha256": sha256_file(ROOT / CONFIG),
            "parent_artifact_sha256": parents,
            "parent_file_sha256": {k: sha256_file(ROOT / v) for k, v in _parent_paths(cfg).items()},
            "data_shape": {"states": list(data["states"].shape), "summary": list(data["summary"].shape),
                           "layouts": len(set(data["layouts"].tolist()))},
            "scope": cfg["scope"], "reserve_access": False, "test_access": False}


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    lock = _read(cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d163_error_decision_audit_lock_v1":
        raise D163Error("D163 lock schema mismatch")
    if _git("status", "--porcelain"):
        raise D163Error("D163 run requires a clean committed tree")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    if (lock["source_sha256"] != _source_hashes()
            or lock["config_sha256"] != sha256_file(ROOT / CONFIG)
            or lock["scope"] != cfg["scope"] or lock.get("reserve_access") is not False
            or lock.get("test_access") is not False):
        raise D163Error("D163 lock/source/config/scope mismatch")
    _, parents = _audit_inputs(cfg)
    if parents != lock["parent_artifact_sha256"]:
        raise D163Error("D163 parent artifact changed after lock")
    for key in ("run_dir", "report"):
        if (ROOT / cfg["outputs"][key]).exists():
            raise D163Error(f"D163 one-shot output already exists: {key}")
    return cfg, lock


def render_report(result: dict[str, Any], lock_sha: str) -> str:
    lines = ["# D163 — Consequence-error-to-decision audit", "",
             "**Status:** completed exploratory, offline-only, fixed D162 artifacts; not causal or closed-loop evidence.", "",
             f"Data: {result['layout_count']} layouts, {result['state_count']} states, {result['branch_count']} action branches. New simulator/planner calls: 0. Reserve/test access: no.", "",
             "## D162 parity gate", "", "| Arm | Max absolute metric difference | Passed |", "|---|---:|:---:|"]
    for name, row in result["parity"].items():
        lines.append(f"| {name} | {row['max_absolute_metric_difference']:.3g} | {row['passed']} |")
    c = result["component_prediction"]
    lines += ["", "## Component prediction", "",
              f"Outcome confusion (true rows, predicted columns; success/timeout/collision): `{c['confusion_matrix_true_rows_predicted_columns']}`.",
              f"Collision recall/FNR: {c['collision_recall']} / {c['collision_false_negative_rate']}; Brier: {c['collision_brier']:.5f}; multiclass log loss: {c['multiclass_log_loss']:.5f}.",
              f"Steps MAE/RMSE: {c['steps_mae']:.5f} / {c['steps_rmse']:.5f}.", "",
              "Collision false negatives by exact teacher regret:", "",
              "| Regret band | Collisions | Missed | Recall |", "|---|---:|---:|---:|"]
    for band, row in c["collision_misses_by_teacher_regret"].items():
        lines.append(f"| {band} | {row['collision_count']} | {row['missed_count']} | {row['recall']} |")
    lines += ["", "## Frozen scorer × input component swaps", "",
              "Lower regret/catastrophic fraction is better. Crossed cells are frozen-input sensitivity diagnostics and may be shifted from the scorer's training distribution.", "",
              "| Outcome input | Steps input | Scorer | Macro regret | Catastrophic | Safe/collision pair | Low-regret/catastrophic pair | Collision selected |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for case, scorers in result["frozen_scorer_input_swaps"].items():
        parts = case.split("__")
        outcome = parts[0].replace("outcome_", "")
        steps = parts[1].replace("steps_", "")
        for scorer, m in scorers.items():
            lines.append(f"| {outcome} | {steps} | {scorer} | {m['layout_macro_normalized_selected_regret']:.5f} | {m['catastrophic_fraction_layout_macro']:.5f} | {m['safe_vs_collision_pair_accuracy']} | {m['low_regret_vs_catastrophic_pair_accuracy']} | {m['collision_action_selection_rate']:.5f} |")
    lines += ["", "## Interpretation limits", "",
              "The swaps do not establish causal component importance: they replace inputs to already-trained scorers on the same D162 rows. Exact/predicted crossed cells can be distribution-shifted. Outcomes on this block are not independent confirmation, and the fixed CART collector ended in collision on 93/100 layouts. No model was fit or updated.", "",
              f"Lock SHA-256: {lock_sha}.", ""]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    cfg, lock = validate_lock()
    run_dir = ROOT / cfg["outputs"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=False)
    result = analyze(cfg)
    result["lock_sha256"] = sha256_file(ROOT / cfg["outputs"]["lock"])
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report_path = ROOT / cfg["outputs"]["report"]
    report_path.write_text(render_report(result, result["lock_sha256"]), encoding="utf-8")
    manifest = {"schema": "phase15_d163_error_decision_audit_manifest_v1",
                "status": result["status"], "source_commit": lock["source_commit"],
                "lock_sha256": result["lock_sha256"], "result_sha256": sha256_file(result_path),
                "report_sha256": sha256_file(report_path), "parent_artifact_sha256": result["parent_hashes"],
                "new_simulator_calls": 0, "new_planner_calls": 0, "new_training": False,
                "reserve_access": False, "test_access": False, "student_closed_loop": False}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
