"""Run the locked D152 train/validation-only representation-sufficiency audit."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_learning import decision_metrics, layout_weights
from consequence_lab.phase15_outcome_summary import restore_decision_head
from consequence_lab.phase15_representation_screen import train_validation_indices
from consequence_lab.phase15_representation_sufficiency import (
    ActionSummaryMLP, WeightedMultiOutputCART, build_history_features, component_ablation,
    expand_action_examples, expand_examples_prediction, expand_layout_weights,
    mean_rank_correlation, nearest_action_summary, neighbor_consequence_metrics,
    summary_component_metrics,
)
from consequence_lab.phase15_data import layout_split

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_representation_sufficiency.yaml"
LOCK = "experiments/phase15_representation_sufficiency.lock.json"
_LAYOUT_RE = re.compile(r'"layout_sha256"\s*:\s*"([0-9a-f]{64})"')


def _validate_lock() -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    lock = json.loads((ROOT / LOCK).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase15_representation_sufficiency_lock_v1" or lock.get("status") != "authorized_exploratory_train_validation_audit":
        raise RuntimeError("D152 lock status mismatch")
    if lock.get("config_sha256") != sha256_file(ROOT / CONFIG):
        raise RuntimeError("D152 config hash mismatch")
    if lock.get("source_sha256") != {path: sha256_file(ROOT / path) for path in lock["source_sha256"]}:
        raise RuntimeError("D152 source hash mismatch")
    if lock.get("parent_sha256") != {key: sha256_file(ROOT / path) for key, path in config["parents"].items()}:
        raise RuntimeError("D152 parent hash mismatch")
    if lock.get("prompt_sha256") != sha256_file(ROOT / config["prompt_source"]):
        raise RuntimeError("user-provided D152 prompt hash changed")
    if lock.get("scope") != config["scope"]:
        raise RuntimeError("D152 scope mismatch")
    if lock.get("frozen_d150_head_sha256") != {
            name: sha256_file(ROOT / config["parents"][f"d150_{name.lower()}_checkpoint"])
            for name in ("R0", "R4")}:
        raise RuntimeError("D150 frozen checkpoint hash mismatch")
    git_status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).splitlines()
    if git_status != ['?? "new Prompt.md"']:
        raise RuntimeError(f"D152 run requires clean committed source except pinned user prompt: {git_status}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not (commit == lock["source_commit"] or subprocess.run(
            ["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"], cwd=ROOT).returncode == 0):
        raise RuntimeError("D152 lock source commit is not an ancestor of HEAD")
    return config, lock


def _weighted_summary_rmse(target: np.ndarray, prediction: np.ndarray, layouts: np.ndarray) -> float:
    y, p = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 3 or y.shape[1:] != (9, 7):
        raise ValueError("D152 weighted summary RMSE requires matching [N,9,7]")
    weights = layout_weights(np.asarray(layouts).astype(str))
    return float(np.sqrt(np.sum(weights[:, None, None] * (y - p) ** 2) / 63.0))


def _constant_prediction(target: np.ndarray, layouts: np.ndarray, count: int) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    weights = layout_weights(np.asarray(layouts).astype(str))
    mean = np.sum(y * weights[:, None, None], axis=0, keepdims=True)
    return np.repeat(mean, count, axis=0)


def _normalize_outcomes(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64).reshape(-1, 9, 7).copy()
    outcomes = np.maximum(rows[:, :, :4], 1e-12)
    rows[:, :, :4] = outcomes / outcomes.sum(axis=2, keepdims=True)
    return rows


def _tree_json(node) -> dict:
    result = {"value": np.asarray(node.value).tolist(), "feature": int(node.feature),
              "threshold": float(node.threshold)}
    if node.feature >= 0:
        result["left"] = _tree_json(node.left)
        result["right"] = _tree_json(node.right)
    return result


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return mean_rank_correlation(np.asarray(x), np.asarray(y))


def _load_allowed_traces(path: Path, allowed_layouts: set[str]) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            match = _LAYOUT_RE.search(line)
            if match is None or match.group(1) not in allowed_layouts:
                continue
            record = json.loads(line)
            if record["layout_sha256"] not in allowed_layouts:
                raise RuntimeError("D152 trace layout filter/parser disagreement")
            rows.append(record)
    return rows


def _render(result: dict) -> str:
    lines = ["# D152 Representation Sufficiency and Consequence Predictability Audit", "",
             "**Status: exploratory train/reused-validation audit.** Test and reserve remain sealed; no simulation or planning was run.", "",
             "## Frozen-head integrity", "",
             "| Head | Validation macro regret | Catastrophic fraction | Best-action agreement |", "|---|---:|---:|---:|"]
    for name, metrics in result["frozen_head_reproduction"].items():
        lines.append(f"| {name} | {metrics['mean_normalized_regret_layout_macro']:.6f} | {metrics['catastrophic_fraction_state_micro']:.6f} | {metrics['best_action_set_agreement']:.6f} |")
    alias = result["aliasing_audit"]
    lines.extend(["", "## R0 neighborhood aliasing diagnostic", "",
                  f"Training leave-one-out R0 distance p90: **{alias['train_loo_distance_p90']:.6f}**. "
                  f"Validation nearest-neighbor distance median: **{alias['validation_nn_distance_median']:.6f}**. "
                  f"Close-neighbor rows: **{alias['close_neighbor_count']} / {alias['validation_rows']}**. "
                  f"Predeclared history trigger: **{'passed' if alias['conditional_history_trigger_passed'] else 'not passed'}**.", "",
                  "| Pair set | R0 distance | R4 summary RMSE | Regret-vector MAE | Ranking disagreement | Best-set disagreement | Catastrophic-mask Jaccard distance |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name in ("nearest_1_all", "nearest_1_close", "nearest_5_all", "random_cross_split"):
        metrics = alias["pair_metrics"].get(name, {})
        distance = alias["distance_summary"].get(name, {}).get("r0_distance_median", float("nan"))
        if metrics:
            lines.append(f"| {name} | {distance:.6f} | {metrics['r4_summary_rmse']:.6f} | {metrics['teacher_regret_mean_absolute_difference']:.6f} | {metrics['teacher_pairwise_ranking_disagreement']:.6f} | {metrics['best_action_set_disagreement']:.6f} | {metrics['catastrophic_action_mask_jaccard_distance']:.6f} |")
    lines.extend(["", "## Action-conditioned continuation-summary predictors", "",
                  "| Input/model | Train weighted RMSE | Validation weighted RMSE | Outcome accuracy | Success F1 | Collision F1 | Steps MAE | Goal-distance MAE | Effort MAE | Frozen-head val macro regret | Val catastrophic fraction |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, record in result["predictors"].items():
        train, val = record["train"], record["validation"]
        val_metrics = val["component_metrics"]
        downstream = val["downstream_decision"]
        lines.append(f"| {name} | {train['weighted_rmse']:.6f} | {val['weighted_rmse']:.6f} | {val_metrics['outcome_accuracy']:.4f} | {val_metrics['outcome_classification']['success']['f1']:.4f} | {val_metrics['outcome_classification']['collision']['f1']:.4f} | {val_metrics['terminal_steps_mae']:.5f} | {val_metrics['terminal_goal_distance_mae']:.5f} | {val_metrics['noncoast_effort_mae']:.5f} | {downstream['mean_normalized_regret_layout_macro']:.6f} | {downstream['catastrophic_fraction_state_micro']:.6f} |")
    lines.extend(["", "## Conditional history representation", "",
                  f"Executed: **{result['history_representations']['executed']}**. "
                  f"History available on {result['history_representations'].get('available_fraction', 0.0):.1%} of allowed rows.", "",
                  "| Input | Train weighted RMSE | Validation weighted RMSE | Validation outcome accuracy | Validation macro regret | Catastrophic fraction |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, record in result["history_representations"].get("models", {}).items():
        val = record["validation"]
        lines.append(f"| {name} | {record['train']['weighted_rmse']:.6f} | {val['weighted_rmse']:.6f} | {val['component_metrics']['outcome_accuracy']:.4f} | {val['downstream_decision']['mean_normalized_regret_layout_macro']:.6f} | {val['downstream_decision']['catastrophic_fraction_state_micro']:.6f} |")
    lines.extend(["", "## Frozen R4-head component substitutions", "",
                  "The full per-predictor forward/reverse ablation values are in `result.json`. These feature substitutions diagnose the frozen scorer and are not deployed policies.", "",
                  "## Interpretation and boundary", "",
                  result["interpretation"], "",
                  "The validation group was reused in D148-D151 and is not an independent test. The consumed D148 test rows were not indexed or scored; the compressed archive may materialize full members internally. Only train/validation indices entered predictor fitting and evaluation. The conditional history reader parses only rows whose layout hash is in the locked train/validation groups. No new transitions/episodes, simulator/planner calls, DAgger, closed-loop work, learned world model, JEPA, or reserve access occurred.", ""])
    return "\n".join(lines)


def main() -> None:
    config, lock = _validate_lock()
    out_dir = ROOT / config["outputs"]["run_dir"]
    dataset = ROOT / config["parents"]["feature_dataset"]
    with np.load(dataset, allow_pickle=False) as data:
        layouts_all = data["layouts"].astype(str).tolist()
        train_idx, val_idx, train_names, val_names = train_validation_indices(layouts_all, {
            "train_layout_count": config["split"]["train_layout_count"],
            "validation_layout_count": config["split"]["validation_layout_count"],
            "train_layouts_sha256": config["split"]["train_layouts_sha256"],
            "validation_layouts_sha256": config["split"]["validation_layouts_sha256"],
            "train_row_count": config["split"]["train_row_count"],
            "validation_row_count": config["split"]["validation_row_count"],
        })
        if lock["split"]["train"]["layouts"] != train_names or lock["split"]["validation"]["layouts"] != val_names:
            raise RuntimeError("D152 train/validation layout membership mismatch")
        layouts = np.asarray(layouts_all, dtype=str)
        all_allowed_idx = np.concatenate((train_idx, val_idx))
        allowed_layouts = set(train_names + val_names)
        x0_train, x0_val = np.asarray(data["x_R0"][train_idx], dtype=np.float64), np.asarray(data["x_R0"][val_idx], dtype=np.float64)
        x4_train, x4_val = np.asarray(data["x_R4"][train_idx], dtype=np.float64), np.asarray(data["x_R4"][val_idx], dtype=np.float64)
        y_train, y_val = np.asarray(data["y"][train_idx], dtype=np.float64), np.asarray(data["y"][val_idx], dtype=np.float64)
        train_layouts, val_layouts = layouts[train_idx], layouts[val_idx]
        allowed_layout_vector = layouts[all_allowed_idx]
        allowed_x0 = np.concatenate((x0_train, x0_val), axis=0)
        allowed_x4 = np.concatenate((x4_train, x4_val), axis=0)
        allowed_y = np.concatenate((y_train, y_val), axis=0)
        allowed_seeds = np.asarray(data["seeds"][all_allowed_idx])
        allowed_decisions = np.asarray(data["decision_index"][all_allowed_idx])
    if (x0_train.shape != (len(train_idx), 61) or x0_val.shape != (len(val_idx), 61)
            or x4_train.shape != (len(train_idx), 124) or x4_val.shape != (len(val_idx), 124)
            or not np.array_equal(x4_train[:, :61], x0_train)
            or not np.array_equal(x4_val[:, :61], x0_val)
            or y_train.shape != (len(train_idx), 9) or y_val.shape != (len(val_idx), 9)):
        raise RuntimeError("D152 feature/target contract mismatch")
    summary_train = x4_train[:, 61:124].reshape(-1, 9, 7)
    summary_val = x4_val[:, 61:124].reshape(-1, 9, 7)
    for values in (summary_train, summary_val):
        if (not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0)
                or not np.all((values[:, :, :4] == 0.0) | (values[:, :, :4] == 1.0))
                or not np.all(values[:, :, :4].sum(axis=2) == 1.0)):
            raise RuntimeError("D152 exact target violates seven-field one-hot contract")

    head_dir = ROOT / "artifacts/runs/phase15_representation_screen_v1"
    r0_head = restore_decision_head(head_dir / "R0_checkpoint.npz")
    r4_head = restore_decision_head(head_dir / "R4_checkpoint.npz")
    r0_val_scores, r4_val_scores = r0_head.predict(x0_val), r4_head.predict(x4_val)
    r0_val_metrics = decision_metrics(y_val, r0_val_scores, val_layouts)
    r4_val_metrics = decision_metrics(y_val, r4_val_scores, val_layouts)
    gate = config["parents"]
    d150_result = json.loads((ROOT / gate["d150_result"]).read_text(encoding="utf-8"))
    expected = {"R0": d150_result["models"]["R0"]["validation_metrics"],
                "R4": d150_result["models"]["R4"]["validation_metrics"]}
    for name, actual in (("R0", r0_val_metrics), ("R4", r4_val_metrics)):
        source = expected[name]
        for key in ("mean_normalized_regret_layout_macro", "catastrophic_fraction_state_micro", "best_action_set_agreement"):
            if abs(actual[key] - source[key]) > 1e-10:
                raise RuntimeError(f"D150 frozen {name} head reproduction failed at {key}")

    # R0 aliasing: nearest training neighbors and a fixed random nonlocal reference.
    loo_predictions, loo_distances, loo_ids = nearest_action_summary(
        x0_train, summary_train, x0_train, k=1, exclude_self=True)
    del loo_predictions
    val_nn1, val_dist1, val_ids1 = nearest_action_summary(x0_train, summary_train, x0_val, k=1)
    _, val_dist5, val_ids5 = nearest_action_summary(x0_train, summary_train, x0_val, k=5)
    train_p90 = float(np.quantile(loo_distances[:, 0], 0.90))
    joint_r4, joint_y = allowed_x4, allowed_y
    query_global = np.arange(len(train_idx), len(train_idx) + len(val_idx), dtype=np.int64)
    train_global = np.arange(len(train_idx), dtype=np.int64)
    nearest_metrics = neighbor_consequence_metrics(joint_r4, joint_y, query_global, val_ids1[:, 0])
    close_mask = val_dist1[:, 0] <= train_p90
    close_rows = np.flatnonzero(close_mask)
    close_metrics = neighbor_consequence_metrics(joint_r4, joint_y, query_global[close_mask], val_ids1[close_mask, 0])
    q_rep = np.repeat(query_global, val_ids5.shape[1])
    k5_metrics = neighbor_consequence_metrics(joint_r4, joint_y, q_rep, val_ids5.reshape(-1))
    rng = np.random.default_rng(config["aliasing"]["random_pair_seed"])
    random_q = rng.integers(0, len(val_idx), size=config["aliasing"]["random_cross_split_pairs"])
    random_t = rng.integers(0, len(train_idx), size=len(random_q))
    random_metrics = neighbor_consequence_metrics(joint_r4, joint_y, len(train_idx) + random_q, random_t)
    random_dist = np.sqrt(np.sum(((x0_val[random_q] - x0_train[random_t]) / np.where(x0_train.std(axis=0) < 1e-8, 1.0, x0_train.std(axis=0))) ** 2, axis=1))
    pair_metrics = {"nearest_1_all": nearest_metrics, "nearest_1_close": close_metrics,
                    "nearest_5_all": k5_metrics, "random_cross_split": random_metrics}
    distance_summary = {
        "train_leave_one_out": {"median": float(np.median(loo_distances[:, 0])), "p90": train_p90},
        "nearest_1_all": {"r0_distance_median": float(np.median(val_dist1[:, 0])),
                           "r0_distance_p90": float(np.quantile(val_dist1[:, 0], 0.90))},
        "nearest_1_close": {"r0_distance_median": float(np.median(val_dist1[close_mask, 0])) if len(close_rows) else float("nan")},
        "nearest_5_all": {"r0_distance_median": float(np.median(val_dist5))},
        "random_cross_split": {"r0_distance_median": float(np.median(random_dist))},
    }
    nn_summary_rmse = np.asarray([
        np.sqrt(np.mean((allowed_x4[len(train_idx) + i, 61:124] -
                         allowed_x4[val_ids1[i, 0], 61:124]) ** 2))
        for i in range(len(val_idx))])
    nn_regret_mae = np.asarray([
        np.mean(np.abs(y_val[i] - y_train[val_ids1[i, 0]])) for i in range(len(val_idx))])
    nn_best_disagree = np.asarray([
        float(not bool(set(np.flatnonzero(y_val[i] <= y_val[i].min() + 1e-12)) &
                       set(np.flatnonzero(y_train[val_ids1[i, 0]] <= y_train[val_ids1[i, 0]].min() + 1e-12))))
        for i in range(len(val_idx))])
    aliasing = {"train_loo_distance_median": float(np.median(loo_distances[:, 0])),
                "train_loo_distance_p90": train_p90,
                "validation_nn_distance_median": float(np.median(val_dist1[:, 0])),
                "validation_rows": len(val_idx), "close_neighbor_count": int(len(close_rows)),
                "close_neighbor_fraction": float(np.mean(close_mask)),
                "distance_summary": distance_summary, "pair_metrics": pair_metrics,
                "spearman_r0_vs_r4_summary_distance": _spearman(val_dist1[:, 0], nn_summary_rmse),
                "spearman_r0_vs_regret_disagreement": _spearman(val_dist1[:, 0], nn_regret_mae),
                "spearman_r0_vs_best_action_disagreement": _spearman(val_dist1[:, 0], nn_best_disagree),
                "conditional_history_trigger_passed": bool(
                    len(close_rows) >= config["aliasing"]["conditional_history_gate"]["minimum_close_validation_rows"]
                    and close_metrics.get("r4_summary_rmse", 0.0) >= config["aliasing"]["conditional_history_gate"]["close_neighbor_r4_summary_rmse_minimum"]
                    and close_metrics.get("best_action_set_disagreement", 0.0) >= config["aliasing"]["conditional_history_gate"]["close_neighbor_best_action_set_disagreement_minimum"])}
    top_pairs = []
    for i in np.argsort(nn_summary_rmse)[::-1][:20]:
        top_pairs.append({"validation_row": int(i), "training_row": int(val_ids1[i, 0]),
                          "r0_distance": float(val_dist1[i, 0]),
                          "r4_summary_rmse": float(nn_summary_rmse[i]),
                          "teacher_regret_mae": float(nn_regret_mae[i])})
    aliasing["nearest_r0_high_r4_disagreement_examples"] = top_pairs

    out_dir = ROOT / config["outputs"]["run_dir"]
    out_dir.mkdir(parents=True, exist_ok=False)
    result_predictors: dict[str, Any] = {}
    predictions_out: dict[str, np.ndarray] = {"validation_r0_scores": r0_val_scores,
                                               "validation_r4_oracle_scores": r4_val_scores}
    checkpoints: dict[str, np.ndarray] = {}
    tree_artifacts: dict[str, Any] = {}

    def record_predictor(name: str, train_prediction: np.ndarray, val_prediction: np.ndarray,
                         train_model_name: str, model_meta: dict[str, Any]) -> None:
        train_prediction = _normalize_outcomes(train_prediction)
        val_prediction = _normalize_outcomes(val_prediction)
        train_weighted_rmse = _weighted_summary_rmse(summary_train, train_prediction, train_layouts)
        val_weighted_rmse = _weighted_summary_rmse(summary_val, val_prediction, val_layouts)
        train_input = np.concatenate((x0_train, train_prediction.reshape(len(train_idx), 63)), axis=1)
        val_input = np.concatenate((x0_val, val_prediction.reshape(len(val_idx), 63)), axis=1)
        train_downstream = decision_metrics(y_train, r4_head.predict(train_input), train_layouts)
        val_downstream = decision_metrics(y_val, r4_head.predict(val_input), val_layouts)
        result_predictors[name] = {
            "model": model_meta,
            "train": {"weighted_rmse": train_weighted_rmse,
                      "component_metrics": summary_component_metrics(summary_train, train_prediction),
                      "downstream_decision": train_downstream},
            "validation": {"weighted_rmse": val_weighted_rmse,
                           "component_metrics": summary_component_metrics(summary_val, val_prediction),
                           "downstream_decision": val_downstream,
                           "component_ablation": component_ablation(x4_val, val_prediction, x0_val, y_val, val_layouts, r4_head)},
        }
        predictions_out[f"train_{train_model_name}"] = train_prediction
        predictions_out[f"validation_{train_model_name}"] = val_prediction

    const_train = _constant_prediction(summary_train, train_layouts, len(train_idx))
    const_val = _constant_prediction(summary_train, train_layouts, len(val_idx))
    record_predictor("train_layout_constant", const_train, const_val, "constant", {"kind": "equal_layout_weighted_constant"})

    for k in config["predictors"]["nearest_neighbor"]["k_values"]:
        train_prediction, _, _ = nearest_action_summary(x0_train, summary_train, x0_train, k=k, exclude_self=True)
        val_prediction, _, _ = nearest_action_summary(x0_train, summary_train, x0_val, k=k)
        record_predictor(f"same_action_knn_k{k}", train_prediction, val_prediction, f"knn_k{k}",
                         {"kind": "same_action_state_knn", "k": int(k), "train_metric": "leave_one_state_out"})

    action_x_train, action_y_train, _ = expand_action_examples(x0_train, summary_train)
    action_x_val, _, _ = expand_action_examples(x0_val, summary_val)
    expanded_weights = expand_layout_weights(train_layouts)
    cart_cfg = config["predictors"]["cart"]
    cart = WeightedMultiOutputCART(cart_cfg["maximum_depth"], cart_cfg["minimum_leaf_examples"])
    start = time.perf_counter()
    cart.fit(action_x_train, action_y_train, expanded_weights)
    cart_seconds = time.perf_counter() - start
    cart_train = expand_examples_prediction(cart.predict(action_x_train), len(train_idx))
    cart_val = expand_examples_prediction(cart.predict(action_x_val), len(val_idx))
    tree_artifacts["R0_CART"] = _tree_json(cart.root)
    record_predictor("shared_weighted_cart", cart_train, cart_val, "cart",
                     {"kind": "shared_multioutput_cart", "max_depth": cart_cfg["maximum_depth"],
                      "minimum_leaf_examples": cart_cfg["minimum_leaf_examples"], "node_count": cart.node_count,
                      "fit_seconds": cart_seconds})

    mlp_cfg = config["predictors"]["small_mlp"]
    mlp = ActionSummaryMLP(mlp_cfg["seed"], tuple(mlp_cfg["architecture"][1:3]),
                           mlp_cfg["epochs"], mlp_cfg["learning_rate"], mlp_cfg["l2"])
    start = time.perf_counter()
    mlp.fit(action_x_train, action_y_train, expanded_weights)
    mlp_seconds = time.perf_counter() - start
    mlp_train = expand_examples_prediction(mlp.predict_examples(action_x_train), len(train_idx))
    mlp_val = expand_examples_prediction(mlp.predict_examples(action_x_val), len(val_idx))
    record_predictor("shared_action_mlp_32", mlp_train, mlp_val, "mlp32",
                     {"kind": "shared_action_conditioned_mlp", "architecture": mlp_cfg["architecture"],
                      "parameters": mlp.parameter_count, "best_train_epoch": mlp.best_epoch,
                      "epochs": mlp.epochs, "train_objective": mlp.training_loss, "fit_seconds": mlp_seconds})
    for field in ("x_mean", "x_std", "w1", "b1", "w2", "b2", "w3", "b3"):
        checkpoints[f"R0_mlp32_{field}"] = getattr(mlp, field)
    small_train_rmse = result_predictors["shared_action_mlp_32"]["train"]["weighted_rmse"]
    constant_train_rmse = result_predictors["train_layout_constant"]["train"]["weighted_rmse"]
    larger_trigger = small_train_rmse >= 1.05 * constant_train_rmse
    if larger_trigger:
        large_cfg = config["predictors"]["larger_mlp"]
        large = ActionSummaryMLP(large_cfg["seed"], tuple(large_cfg["architecture"][1:3]),
                                 large_cfg["epochs"], large_cfg["learning_rate"], large_cfg["l2"])
        start = time.perf_counter()
        large.fit(action_x_train, action_y_train, expanded_weights)
        large_seconds = time.perf_counter() - start
        large_train = expand_examples_prediction(large.predict_examples(action_x_train), len(train_idx))
        large_val = expand_examples_prediction(large.predict_examples(action_x_val), len(val_idx))
        record_predictor("shared_action_mlp_64_conditional", large_train, large_val, "mlp64",
                         {"kind": "conditional_larger_mlp", "architecture": large_cfg["architecture"],
                          "parameters": large.parameter_count, "best_train_epoch": large.best_epoch,
                          "epochs": large_cfg["epochs"], "train_objective": large.training_loss,
                          "fit_seconds": large_seconds, "trigger_passed": True})
        for field in ("x_mean", "x_std", "w1", "b1", "w2", "b2", "w3", "b3"):
            checkpoints[f"R0_mlp64_{field}"] = getattr(large, field)

    history_result: dict[str, Any] = {"executed": False, "available_fraction": 0.0, "models": {}}
    if aliasing["conditional_history_trigger_passed"]:
        trace_path = ROOT / config["parents"]["phase14g_traces"]
        allowed_traces = _load_allowed_traces(trace_path, allowed_layouts)
        history = build_history_features(allowed_layout_vector.tolist(), allowed_seeds, allowed_decisions,
                                         allowed_x0, allowed_traces)
        history_result["executed"] = True
        history_result["available_fraction"] = float(history["history_available_fraction"][0])
        history_rows = {name: value for name, value in history.items() if name.startswith("R0_plus_")}
        for name, rich in history_rows.items():
            rich_train, rich_val = rich[:len(train_idx)], rich[len(train_idx):]
            rich_x_train, rich_y_train, _ = expand_action_examples(rich_train, summary_train)
            rich_x_val, _, _ = expand_action_examples(rich_val, summary_val)
            model = ActionSummaryMLP(mlp_cfg["seed"], tuple(mlp_cfg["architecture"][1:3]),
                                     mlp_cfg["epochs"], mlp_cfg["learning_rate"], mlp_cfg["l2"])
            start = time.perf_counter()
            model.fit(rich_x_train, rich_y_train, expanded_weights)
            elapsed = time.perf_counter() - start
            rich_train_prediction = expand_examples_prediction(model.predict_examples(rich_x_train), len(train_idx))
            rich_val_prediction = expand_examples_prediction(model.predict_examples(rich_x_val), len(val_idx))
            record_predictor(name, rich_train_prediction, rich_val_prediction, name,
                             {"kind": "shared_action_mlp_with_deployable_history", "input_dimension": rich.shape[1] + 9,
                              "parameters": model.parameter_count, "best_train_epoch": model.best_epoch,
                              "epochs": mlp_cfg["epochs"], "fit_seconds": elapsed})
            history_result["models"][name] = result_predictors[name]
            for field in ("x_mean", "x_std", "w1", "b1", "w2", "b2", "w3", "b3"):
                checkpoints[f"{name}_{field}"] = getattr(model, field)
    else:
        history_result["reason"] = "predeclared close-neighbor aliasing gate not met; no richer representation was fitted"

    result = {
        "schema": "phase15_representation_sufficiency_result_v1", "status": "completed_exploratory_audit",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / LOCK),
        "config_sha256": lock["config_sha256"], "prompt_sha256": lock["prompt_sha256"],
        "parent_sha256": lock["parent_sha256"], "train_rows": len(train_idx), "validation_rows": len(val_idx),
        "test_rows_indexed_or_scored": 0, "reserve_access": False,
        "frozen_head_reproduction": {"R0": r0_val_metrics, "R4": r4_val_metrics},
        "aliasing_audit": aliasing,
        "predictors": result_predictors,
        "small_mlp_underfit_trigger": {"train_rmse": small_train_rmse,
                                       "train_constant_rmse": constant_train_rmse,
                                       "triggered": bool(larger_trigger)},
        "history_representations": history_result,
        "scope": config["scope"],
        "validation_reuse_disclosure": lock["validation_reuse_disclosure"],
        "test_boundary_disclosure": lock["test_boundary_disclosure"],
        "training_layout_action_examples": {"train": len(action_x_train), "validation": len(action_x_val)},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "interpretation": "D152 is descriptive and exploratory. Close R0 neighbors with divergent exact summaries/rankings make local aliasing plausible, not fundamentally non-identifiable. Action-conditioned in-sample versus validation fit separates formulation/optimization signals from layout coverage only suggestively. Component substitutions diagnose the frozen D150 head and do not establish causal component importance outside it.",
    }
    predictions_path = out_dir / "train_validation_predictions.npz"
    predictions_out["validation_exact_summary"] = summary_val
    predictions_out["validation_teacher_regrets"] = y_val
    np.savez_compressed(predictions_path, **predictions_out)
    checkpoint_path = out_dir / "action_mlp_checkpoints.npz"
    if checkpoints:
        np.savez_compressed(checkpoint_path, **checkpoints)
    else:
        checkpoint_path.write_bytes(b"")
    tree_path = out_dir / "cart_tree.json"
    tree_path.write_text(json.dumps(tree_artifacts, sort_keys=True) + "\n", encoding="utf-8")
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / config["outputs"]["report"]
    report_path.write_text(_render(result), encoding="utf-8")
    manifest = {"schema": "phase15_representation_sufficiency_manifest_v1", "status": result["status"],
                "source_commit": result["source_commit"], "lock_sha256": result["lock_sha256"],
                "config_sha256": result["config_sha256"], "prompt_sha256": result["prompt_sha256"],
                "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
                "predictions_sha256": sha256_file(predictions_path),
                "checkpoints_sha256": sha256_file(checkpoint_path), "cart_sha256": sha256_file(tree_path),
                "feature_dataset_sha256": lock["parent_sha256"]["feature_dataset"],
                "train_rows": len(train_idx), "validation_rows": len(val_idx), "test_rows_indexed_or_scored": 0,
                "new_transitions": 0, "new_episodes": 0, "simulator_or_planner_calls": 0,
                "closed_loop": False, "dagger": False, "world_model": False, "reserve_access": False}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "aliasing": {
        "close_neighbor_count": aliasing["close_neighbor_count"],
        "conditional_history_trigger_passed": aliasing["conditional_history_trigger_passed"]},
        "predictors": {name: {"train_rmse": value["train"]["weighted_rmse"],
                              "validation_rmse": value["validation"]["weighted_rmse"],
                              "validation_macro_regret": value["validation"]["downstream_decision"]["mean_normalized_regret_layout_macro"]}
                       for name, value in result_predictors.items()},
        "test_rows_indexed_or_scored": 0, "report": config["outputs"]["report"]}, indent=2))


if __name__ == "__main__":
    main()
