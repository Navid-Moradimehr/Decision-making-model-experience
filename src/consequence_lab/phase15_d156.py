"""D156 no-training swap audit for predicted versus oracle consequence inputs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_d155 import _cfg as _stage_a_cfg, _read_json, row_layout_labels
from consequence_lab.phase15_decision_aligned import decision_detail_metrics, regrets_from_exact_summaries
from consequence_lab.phase15_representation_screen import train_validation_indices
from consequence_lab.phase15_d155_stage_c import (
    BoundedCART, CARTOutcomeSteps, OutcomeStepsMLP, PerActionConstant,
    _config as _stage_c_cfg, _decision_features, _load_stage_a_head,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d156_component_swap.yaml"
PROTOCOL = "reports/phase15_d156_protocol.md"
LOCK = "experiments/phase15_d156.lock.json"
SOURCE_FILES = (
    CONFIG, PROTOCOL, "src/consequence_lab/phase15_d156.py",
    "src/consequence_lab/phase15_d155.py", "src/consequence_lab/phase15_d155_stage_c.py",
    "src/consequence_lab/phase15_decision_aligned.py",
    "src/consequence_lab/phase15_learning.py",
    "src/consequence_lab/phase15_representation_screen.py",
    "scripts/freeze_phase15_d156_lock.py", "scripts/run_phase15_d156.py",
    "tests/test_phase15_d156.py",
)
SWAP_NAMES = ("exact_exact", "predicted_exact", "exact_predicted", "predicted_predicted")
DECISION_KEYS = (
    "layout_macro_normalized_selected_regret", "state_micro_normalized_selected_regret",
    "catastrophic_fraction_state_micro", "catastrophic_fraction_layout_macro",
    "epsilon_optimal_rate", "best_action_set_agreement", "high_margin_pairwise_accuracy_macro",
    "safe_vs_catastrophic_order_accuracy_macro", "all_pairwise_ranking_accuracy_macro",
    "regret_gap_weighted_pairwise_accuracy_macro",
)


class D156Error(RuntimeError):
    """D156 integrity or execution failure."""


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D156Error(f"git {' '.join(args)} failed") from exc


def _config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _parent_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}


def _audit_stage_c(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock_path = ROOT / cfg["parents"]["stage_c_lock"]
    result_path = ROOT / cfg["parents"]["stage_c_result"]
    manifest_path = ROOT / cfg["parents"]["stage_c_manifest"]
    lock, result, manifest = _read_json(lock_path), _read_json(result_path), _read_json(manifest_path)
    if (lock.get("schema") != "phase15_d155_stage_c_lock_v1"
            or result.get("schema") != "phase15_d155_stage_c_result_v1"
            or result.get("status") != "completed_exploratory_compact_consequence_prediction_screen"
            or result.get("offline_gate_passed") is not False):
        raise D156Error("D155 Stage-C parent does not match its completed failed-gate result")
    if (result.get("lock_sha256", "").upper() != sha256_file(lock_path)
            or manifest.get("lock_sha256", "").upper() != sha256_file(lock_path)
            or manifest.get("result_sha256", "").upper() != sha256_file(result_path)
            or manifest.get("report_sha256", "").upper() != sha256_file(ROOT / "reports/phase15_d155_stage_c_result.md")
            or manifest.get("test_rows_dereferenced") != 0
            or manifest.get("reserve_access") is not False
            or manifest.get("simulator_calls") != 0 or manifest.get("planner_calls") != 0):
        raise D156Error("D155 Stage-C manifest/hash/scope check failed")
    checkpoint_map = {"stage_c_constant": "per_action_constant.npz",
                      "stage_c_cart": "bounded_cart.json",
                      "stage_c_mlp": "action_conditioned_mlp.npz"}
    for parent_name, filename in checkpoint_map.items():
        expected = manifest.get("predictor_checkpoint_sha256", {}).get(filename, "")
        if not expected or expected.upper() != sha256_file(ROOT / cfg["parents"][parent_name]):
            raise D156Error(f"D155 Stage-C checkpoint hash mismatch: {filename}")
    if set(result.get("predictors", {})) != {"per_action_constant", "bounded_cart", "action_conditioned_mlp"}:
        raise D156Error("D155 Stage-C predictor set changed")
    if any(row.get("passes_preregistered_offline_gate") is not False
           for row in result["predictors"].values()):
        raise D156Error("D156 selected parent contains a Stage-C predictor that passed the gate")
    return lock, result, manifest


def build_lock() -> dict[str, Any]:
    cfg = _config()
    if _git("status", "--porcelain"):
        raise D156Error("commit D156 protocol/source/tests before freezing its lock")
    if cfg.get("schema") != "phase15_d156_predicted_oracle_component_swap":
        raise D156Error("D156 config schema mismatch")
    if (cfg["scope"].get("validation_only") is not True
            or any(value is not False for key, value in cfg["scope"].items()
                   if key != "validation_only")):
        raise D156Error("D156 enables a forbidden scope operation")
    if (cfg["scope"].get("model_fitting") is not False
            or cfg["split"].get("test_reserve_rows_dereferenced") != 0):
        raise D156Error("D156 scope/split contract invalid")
    stage_c_lock, stage_c_result, _ = _audit_stage_c(cfg)
    split = stage_c_lock.get("split", {})
    if (split.get("validation_rows") != cfg["split"]["validation_rows"]
            or split.get("validation_layouts") != cfg["split"]["validation_layouts"]
            or split.get("validation_layouts_sha256") != cfg["split"]["validation_layouts_sha256"]
            or cfg["split"]["reused"] is not True):
        raise D156Error("D156 validation block does not match D155 Stage-C lock")
    for output in (cfg["outputs"]["lock"], cfg["outputs"]["run_dir"], cfg["outputs"]["report"]):
        if (ROOT / output).exists():
            raise D156Error(f"refusing to overwrite D156 output: {output}")
    return {
        "schema": "phase15_d156_lock_v1", "status": "locked_validation_only_feature_swap_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "config_sha256": sha256_file(ROOT / CONFIG), "parent_sha256": _parent_hashes(cfg),
        "split": cfg["split"], "feature_swaps": cfg["feature_swaps"],
        "parity": cfg["parity"], "metrics": cfg["metrics"], "scope": cfg["scope"],
        "outputs": cfg["outputs"], "stage_c_offline_gate_passed": stage_c_result["offline_gate_passed"],
        "test_or_reserve_used": False,
        "reuse_disclosure": "The same D152 validation block was reused through D152-D156; descriptive only.",
    }


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _config()
    lock = _read_json(ROOT / cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d156_lock_v1":
        raise D156Error("D156 lock schema mismatch")
    if (lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("source_sha256") != {path: sha256_file(ROOT / path) for path in SOURCE_FILES}
            or lock.get("parent_sha256") != _parent_hashes(cfg)
            or lock.get("scope") != cfg["scope"]
            or lock.get("test_or_reserve_used") is not False):
        raise D156Error("D156 source/config/parent/scope lock mismatch")
    if _git("status", "--porcelain"):
        raise D156Error("D156 run requires a clean committed tree")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    _audit_stage_c(cfg)
    for output in (cfg["outputs"]["run_dir"], cfg["outputs"]["report"]):
        if (ROOT / output).exists():
            raise D156Error(f"refusing second D156 execution: {output}")
    return cfg, lock


def load_validation_only(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Dereference only the explicitly split validation feature rows."""
    split_expected = {
        "train_layout_count": 60, "train_row_count": 571,
        "train_layouts_sha256": "1FF811CF3169CA339C23FC73A22617C10249297565702A592D2BD3C9C683769B",
        "validation_layout_count": int(expected["validation_layouts"]),
        "validation_row_count": int(expected["validation_rows"]),
        "validation_layouts_sha256": expected["validation_layouts_sha256"],
    }
    with np.load(path, allow_pickle=False) as saved:
        layouts = saved["layouts"].astype(str).tolist()
        _, validation_ids, _, validation_layout_ids = train_validation_indices(layouts, split_expected)
        states = saved["x_R0"][validation_ids].astype(np.float64)
        exact_r4 = saved["x_R4"][validation_ids].astype(np.float64)
        target = saved["y"][validation_ids].astype(np.float64)
    row_layouts = row_layout_labels(layouts, validation_ids)
    if (states.shape != (expected["validation_rows"], 61)
            or exact_r4.shape != (expected["validation_rows"], 124)
            or target.shape != (expected["validation_rows"], 9)
            or len(validation_layout_ids) != expected["validation_layouts"]
            or len(row_layouts) != expected["validation_rows"]
            or not np.array_equal(exact_r4[:, :61], states)
            or not np.array_equal(regrets_from_exact_summaries(exact_r4[:, 61:].reshape(-1, 9, 7)), target)):
        raise D156Error("D156 validation feature/target/split contract mismatch")
    outcome = exact_r4[:, 61:].reshape(-1, 9, 7)[:, :, :4]
    if (np.any(np.argmax(outcome, axis=2) == 1)
            or not np.all((outcome == 0.0) | (outcome == 1.0))
            or not np.all(outcome.sum(axis=2) == 1.0)):
        raise D156Error("D156 exact validation outcomes violate the one-hot terminal contract")
    return {"states": states, "exact_r4": exact_r4, "target_regret": target,
            "layouts": row_layouts, "validation_layout_ids": validation_layout_ids,
            "test_rows_dereferenced": 0}


def _tree_from_record(record: dict[str, Any]) -> Any:
    from consequence_lab.phase15_d155_stage_c import _TreeNode
    feature = int(record["feature"])
    left = _tree_from_record(record["left"]) if feature >= 0 else None
    right = _tree_from_record(record["right"]) if feature >= 0 else None
    return _TreeNode(np.asarray(record["value"], dtype=np.float64), feature,
                     float(record["threshold"]), left, right)


def load_predictors(cfg: dict[str, Any]) -> dict[str, Any]:
    constant = PerActionConstant()
    with np.load(ROOT / cfg["parents"]["stage_c_constant"], allow_pickle=False) as saved:
        constant.probabilities = np.asarray(saved["probabilities"], dtype=np.float64)
        constant.mean_steps = np.asarray(saved["mean_steps"], dtype=np.float64)
    if constant.probabilities.shape != (9, 3) or constant.mean_steps.shape != (9,):
        raise D156Error("Stage-C constant checkpoint shape mismatch")

    tree_record = _read_json(ROOT / cfg["parents"]["stage_c_cart"])
    if tree_record.get("schema") != "phase15_d155_cart_pair_v1":
        raise D156Error("Stage-C CART checkpoint schema mismatch")
    stage_cfg = _stage_c_cfg()
    tree = CARTOutcomeSteps(stage_cfg["predictors"]["bounded_cart"])
    tree.x_mean = np.asarray(tree_record["x_mean"], dtype=np.float64)
    tree.x_std = np.asarray(tree_record["x_std"], dtype=np.float64)
    tree.node_count = int(tree_record["node_count"])
    tree.class_tree = BoundedCART("classification", classes=3)
    tree.class_tree.x = np.empty((0, 70), dtype=np.float64)
    tree.class_tree.root = _tree_from_record(tree_record["class_tree"])
    tree.steps_tree = BoundedCART("regression")
    tree.steps_tree.x = np.empty((0, 70), dtype=np.float64)
    tree.steps_tree.root = _tree_from_record(tree_record["steps_tree"])
    if tree.x_mean.shape != (70,) or tree.x_std.shape != (70,) or np.any(tree.x_std <= 0):
        raise D156Error("Stage-C CART normalization checkpoint shape/value mismatch")

    mlp = OutcomeStepsMLP(stage_cfg)
    with np.load(ROOT / cfg["parents"]["stage_c_mlp"], allow_pickle=False) as saved:
        required = ("x_mean", "x_std", "step_mean", "step_std", "w1", "b1", "w2", "b2", "w3", "b3")
        if set(required) - set(saved.files):
            raise D156Error("Stage-C MLP checkpoint is incomplete")
        for key in required:
            value = np.asarray(saved[key], dtype=np.float64)
            setattr(mlp, key, float(value) if value.ndim == 0 else value)
    expected_shapes = {"x_mean": (70,), "x_std": (70,), "w1": (70, 32), "b1": (32,),
                       "w2": (32, 32), "b2": (32,), "w3": (32, 4), "b3": (4,)}
    if any(np.shape(getattr(mlp, key)) != shape for key, shape in expected_shapes.items()):
        raise D156Error("Stage-C MLP checkpoint parameter dimensions mismatch")
    mlp.parameter_count = int(sum(np.size(getattr(mlp, key)) for key in expected_shapes))
    return {"per_action_constant": constant, "bounded_cart": tree, "action_conditioned_mlp": mlp}


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest().upper()


def _score(features: np.ndarray, head: Any, target: np.ndarray, layouts: np.ndarray,
           metrics_cfg: dict[str, Any]) -> dict[str, Any]:
    logits = head.predict_logits(features)
    measured = decision_detail_metrics(target, -logits, layouts, metrics_cfg)
    return {**{key: measured[key] for key in DECISION_KEYS},
            "selected_action_frequency": measured["selected_action_frequency"]}


def component_swaps(states: np.ndarray, exact_r4: np.ndarray, target: np.ndarray,
                    layouts: np.ndarray, predictor: Any, head: Any,
                    metrics_cfg: dict[str, Any]) -> dict[str, Any]:
    summary = np.asarray(exact_r4[:, 61:], dtype=np.float64).reshape(-1, 9, 7)
    exact_outcomes = summary[:, :, :4]
    if (not np.all((exact_outcomes == 0.0) | (exact_outcomes == 1.0))
            or not np.all(exact_outcomes.sum(axis=2) == 1.0)
            or np.any(exact_outcomes[:, :, 1] != 0.0)):
        raise D156Error("exact swap outcome contract violated")
    exact_probs = exact_outcomes[:, :, [0, 2, 3]]
    exact_steps = summary[:, :, 4]
    action_rows = np.zeros((len(states) * 9, 70), dtype=np.float64)
    action_rows[:, :61] = np.repeat(states, 9, axis=0)
    action_rows[:, 61:] = np.tile(np.eye(9), (len(states), 1))
    pred_flat_probs, pred_flat_steps = predictor.predict(action_rows)
    pred_probs = np.asarray(pred_flat_probs, dtype=np.float64).reshape(-1, 9, 3)
    pred_probs = np.maximum(pred_probs, 1e-12)
    pred_probs /= pred_probs.sum(axis=2, keepdims=True)
    pred_steps = np.clip(np.asarray(pred_flat_steps, dtype=np.float64).reshape(-1, 9), 0.0, 1.0)
    combinations = {
        "exact_exact": _decision_features(states, exact_probs, exact_steps),
        "predicted_exact": _decision_features(states, pred_probs, exact_steps),
        "exact_predicted": _decision_features(states, exact_probs, pred_steps),
        "predicted_predicted": _decision_features(states, pred_probs, pred_steps),
    }
    scores = {name: _score(features, head, target, layouts, metrics_cfg)
              for name, features in combinations.items()}
    oracle, out_swap, steps_swap, both_swap = (scores[key] for key in SWAP_NAMES)
    for metric in ("layout_macro_normalized_selected_regret", "catastrophic_fraction_state_micro"):
        out_delta = float(out_swap[metric] - oracle[metric])
        step_delta = float(steps_swap[metric] - oracle[metric])
        combined_delta = float(both_swap[metric] - oracle[metric])
        both_swap.setdefault("component_deltas", {})[metric] = {
            "outcome_only": out_delta, "steps_only": step_delta,
            "combined": combined_delta,
            "interaction": float(combined_delta - out_delta - step_delta),
        }
    return {
        "metrics": scores,
        "prediction_hash": _hash_arrays(pred_probs, pred_steps),
        "component_prediction": {
            "outcome_accuracy": float(np.mean(np.argmax(pred_probs, axis=2).reshape(-1)
                                               == np.argmax(exact_probs, axis=2).reshape(-1))),
            "steps_rmse": float(np.sqrt(np.mean((pred_steps - exact_steps) ** 2))),
        },
    }


def _assert_metric_parity(observed: dict[str, Any], expected: dict[str, Any], tol: float,
                          label: str) -> None:
    for key in DECISION_KEYS:
        if key not in expected or abs(float(observed[key]) - float(expected[key])) > tol:
            raise D156Error(f"{label} parity failed at metric {key}")
    if ("selected_action_frequency" in expected
            and list(observed["selected_action_frequency"]) != list(expected["selected_action_frequency"])):
        raise D156Error(f"{label} parity failed at selected_action_frequency")


def run_d156() -> dict[str, Any]:
    cfg, lock = validate_lock()
    started = time.perf_counter()
    d155cfg = _stage_a_cfg()
    data = load_validation_only(ROOT / d155cfg["parents"]["d152_features"], cfg["split"])
    stage_a_result = _read_json(ROOT / cfg["parents"]["stage_a_result"])
    stage_c_result = _read_json(ROOT / cfg["parents"]["stage_c_result"])
    head = _load_stage_a_head(ROOT / cfg["parents"]["stage_a_head"])
    predictors = load_predictors(cfg)
    exact_summary = data["exact_r4"][:, 61:].reshape(-1, 9, 7)
    exact_prob = exact_summary[:, :, [0, 2, 3]]
    exact_steps = exact_summary[:, :, 4]
    exact_features = _decision_features(data["states"], exact_prob, exact_steps)
    exact_metrics = _score(exact_features, head, data["target_regret"], data["layouts"], cfg["metrics"])
    tolerance = float(cfg["parity"]["absolute_tolerance"])
    _assert_metric_parity(exact_metrics,
                          stage_a_result["feature_results"]["F_outcome_steps"]["validation"],
                          tolerance, "Stage-A exact/exact")
    results: dict[str, Any] = {}
    for name, predictor in predictors.items():
        record = component_swaps(data["states"], data["exact_r4"], data["target_regret"],
                                 data["layouts"], predictor, head, cfg["metrics"])
        _assert_metric_parity(record["metrics"]["predicted_predicted"],
                              stage_c_result["predictors"][name]["downstream"], tolerance,
                              f"Stage-C {name} predicted/predicted")
        results[name] = record
    output = ROOT / cfg["outputs"]["run_dir"]
    output.mkdir(parents=True, exist_ok=False)
    r0 = stage_c_result["references"]["direct_D150_R0"]["layout_macro_normalized_selected_regret"]
    exact_r4 = stage_c_result["references"]["exact_R4_then_D150_head_oracle"]["layout_macro_normalized_selected_regret"]
    result = {
        "schema": "phase15_d156_result_v1", "status": "completed_exploratory_component_swap_audit",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
        "parent_sha256": lock["parent_sha256"],
        "scope": {"validation_rows_scored": len(data["states"]), "validation_layouts": len(data["validation_layout_ids"]),
                  "test_rows_dereferenced": 0, "reserve_access": False, "model_fitting": False,
                  "hyperparameter_tuning": False, "new_data": False, "simulator_calls": 0,
                  "planner_calls": 0, "new_transitions": 0, "new_episodes": 0,
                  "closed_loop": False, "dagger": False, "world_model": False, "jepa": False},
        "validation_layout_ids_sha256": cfg["split"]["validation_layouts_sha256"],
        "validation_target_sha256": _hash_arrays(data["states"], data["exact_r4"], data["target_regret"]),
        "parity": {"tolerance": tolerance, "stage_a_exact_subset_parity": True,
                   "stage_c_predictions_reproduced": list(results)},
        "references": {"d155_exact_F_outcome_steps_macro_regret": exact_r4,
                       "d150_direct_R0_macro_regret": r0,
                       "exact_swap_metrics": exact_metrics},
        "predictors": results,
        "timing_seconds": time.perf_counter() - started,
        "reuse_disclosure": "D152 validation was reused through D152-D156; all D156 results are descriptive.",
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / cfg["outputs"]["report"]
    report_path.write_text(render_report(result), encoding="utf-8")
    manifest = {
        "schema": "phase15_d156_manifest_v1", "status": result["status"],
        "lock_sha256": result["lock_sha256"], "source_commit": result["source_commit"],
        "parent_sha256": lock["parent_sha256"], "result_sha256": sha256_file(result_path),
        "report_sha256": sha256_file(report_path), "validation_target_sha256": result["validation_target_sha256"],
        "test_rows_dereferenced": 0, "reserve_access": False, "model_fitting": False,
        "simulator_calls": 0, "planner_calls": 0, "new_transitions": 0, "new_episodes": 0,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D156 — Predicted-versus-oracle consequence component swaps", "",
             "**Exploratory offline diagnostic, no training.** The same validation block was reused through D152-D156; reserve/test remain sealed.", "",
             f"Parity passed for Stage-A exact/exact and all {len(result['parity']['stage_c_predictions_reproduced'])} Stage-C predicted/predicted checkpoints (absolute tolerance {result['parity']['tolerance']:.0e}). Validation: {result['scope']['validation_rows_scored']} states / {result['scope']['validation_layouts']} layouts.", "",
             f"References: exact outcome+steps regret `{result['references']['d155_exact_F_outcome_steps_macro_regret']:.5f}`; direct-R0 regret `{result['references']['d150_direct_R0_macro_regret']:.5f}`.", ""]
    for metric in ("layout_macro_normalized_selected_regret", "catastrophic_fraction_state_micro"):
        lines.extend([f"## {metric}", "",
                      "| Predictor | Exact/exact | Predicted outcome, exact steps | Exact outcome, predicted steps | Predicted/predicted | Outcome swap Δ | Steps swap Δ | Interaction Δ |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for name, record in result["predictors"].items():
            scores = record["metrics"]
            deltas = scores["predicted_predicted"]["component_deltas"][metric]
            vals = [scores[key][metric] for key in SWAP_NAMES]
            lines.append(f"| {name} | {vals[0]:.5f} | {vals[1]:.5f} | {vals[2]:.5f} | {vals[3]:.5f} | {deltas['outcome_only']:+.5f} | {deltas['steps_only']:+.5f} | {deltas['interaction']:+.5f} |")
        lines.append("")
    lines.extend(["## Component prediction quality", "",
                  "| Predictor | Outcome argmax accuracy | Steps RMSE | Prediction hash |",
                  "|---|---:|---:|---|"])
    for name, record in result["predictors"].items():
        quality = record["component_prediction"]
        lines.append(f"| {name} | {quality['outcome_accuracy']:.4f} | {quality['steps_rmse']:.5f} | `{record['prediction_hash']}` |")
    lines.extend(["", "## Selected action frequencies", "",
                  "Each frequency vector follows the fixed primitive action ID order; ordering is identical across swap cases.", "",
                  "| Predictor | Exact/exact | Predicted outcome, exact steps | Exact outcome, predicted steps | Predicted/predicted |",
                  "|---|---|---|---|---|"])
    for name, record in result["predictors"].items():
        freq = [record["metrics"][case]["selected_action_frequency"] for case in SWAP_NAMES]
        lines.append(f"| {name} | {freq[0]} | {freq[1]} | {freq[2]} | {freq[3]} |")
    lines.extend(["", "Component deltas are single-feature-group swaps through the same frozen scorer, relative to exact/exact. Interaction is the combined-prediction regret/catastrophe change minus the two isolated changes. This local feature intervention does not establish out-of-sample causal importance.", "",
                  "No model was fit, no new data/transition/episode or simulator/planner call occurred, and no test/reserve row was dereferenced. D156 closes here; further work requires a separate protocol.", ""])
    return "\n".join(lines)
