"""Locked D153 nested-layout coverage scaling and fixed-predictor evaluation."""

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

from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step, validate_corridor_state
from consequence_lab.phase12_teacher import state_features
from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14f import FrozenCartPolicy
from consequence_lab.phase14g import forced_first_action_values
from consequence_lab.phase14c import build_feature_matrix
from consequence_lab.phase15_d153 import correlation_pair
from consequence_lab.phase15_outcome_summary import restore_decision_head, summary_metrics
from consequence_lab.phase15_representation_sufficiency import (
    ActionSummaryMLP, expand_action_examples, expand_layout_weights,
)
from consequence_lab.phase15_representation_screen import train_validation_indices


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = "configs/phase15_d153_coverage_scaling_v3.yaml"
PROTOCOL_PATH = "reports/phase15_d153_coverage_scaling_amendment_v3.md"
LOCK_PATH = "experiments/phase15_d153_coverage_scaling_v3.lock.json"
SOURCE_FILES = (
    CONFIG_PATH, PROTOCOL_PATH, "src/consequence_lab/phase15_d153_scaling.py",
    "src/consequence_lab/phase15_d153.py", "src/consequence_lab/phase15_outcome_summary.py",
    "src/consequence_lab/phase15_representation_screen.py", "src/consequence_lab/phase15_data.py",
    "src/consequence_lab/phase15_features.py", "src/consequence_lab/phase14g.py",
    "src/consequence_lab/phase14g_runner.py", "src/consequence_lab/phase14f.py",
    "src/consequence_lab/phase14.py", "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase12_teacher.py", "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py", "scripts/freeze_phase15_d153_scaling.py",
    "scripts/run_phase15_d153_scaling.py", "scripts/check_phase15_d153_d60_reproduction.py",
    "tests/test_phase15_d153_scaling.py",
)
OUTCOME_INDEX = {"success": 0, "active": 1, "timeout": 2, "collision": 3}


class D153ScalingError(RuntimeError):
    """D153 scaling lock, data, or run contract error."""


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D153ScalingError(f"git {' '.join(args)} failed") from exc


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config() -> dict[str, Any]:
    raw = yaml.safe_load((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    base_path = raw.get("base_config")
    if not base_path:
        return raw
    config = yaml.safe_load((ROOT / base_path).read_text(encoding="utf-8"))
    config["schema"] = raw["schema"]
    config["status"] = raw["status"]
    config["parents"] = {**config["parents"], **raw.get("parents", {})}
    config["outputs"] = raw["outputs"]
    return config


def _parent_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in config["parents"].items()}


def _normalize_outcomes(values: np.ndarray) -> np.ndarray:
    """Match D152's post-prediction normalization exactly (not a training transform)."""
    rows = np.asarray(values, dtype=np.float64).reshape(-1, 9, 7).copy()
    outcomes = np.maximum(rows[:, :, :4], 1e-12)
    rows[:, :, :4] = outcomes / outcomes.sum(axis=2, keepdims=True)
    return rows


def _load_d152_split(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    path = ROOT / config["parents"]["d152_features"]
    with np.load(path, allow_pickle=False) as data:
        layouts = data["layouts"].astype(str).tolist()
        train_ids, val_ids, train_layouts, val_layouts = train_validation_indices(layouts, config["fixed_split"])
    return train_ids, val_ids, train_layouts, val_layouts


def _ranked_layouts(layouts: list[str], salt: str) -> list[str]:
    return sorted(layouts, key=lambda value: (hashlib.sha256((salt + value).encode("ascii")).hexdigest(), value))


def candidate_layout_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Hash-only fresh-layout check. It never steps a simulator or reads outcome data."""
    train_ids, val_ids, train_layouts, val_layouts = _load_d152_split(config)
    del train_ids, val_ids
    block = config["layout_design"]
    start, count = int(block["fresh_seed_start"]), int(block["fresh_layout_count"])
    reserve_start, reserve_count = int(block["reserve_seed_start"]), int(block["reserve_layout_count"])
    if start < reserve_start + reserve_count and reserve_start < start + count:
        raise D153ScalingError("fresh D153 seed block overlaps sealed reserve range")
    for forbidden_start, forbidden_count in block["historical_forbidden_ranges"]:
        if start < int(forbidden_start) + int(forbidden_count) and int(forbidden_start) < start + count:
            raise D153ScalingError("fresh D153 seed block overlaps a consumed historical range")
    fresh = []
    for seed in range(start, start + count):
        scenario = generate_staggered_corridor(seed)
        fresh.append({"seed": seed, "layout_sha256": scenario.layout_sha256})
    hashes = [item["layout_sha256"] for item in fresh]
    prior = set(train_layouts) | set(val_layouts)
    if len(set(hashes)) != count or prior.intersection(hashes):
        raise D153ScalingError("fresh layout identities are duplicated or overlap D152 train/validation")
    old_order = _ranked_layouts(train_layouts, block["old_layout_ranking_salt"])
    fresh_order = _ranked_layouts(hashes, block["fresh_layout_ranking_salt"])
    nested = {}
    for size in (10, 20, 40, 60, 100, 150, 200):
        if size <= 60:
            selected = old_order[:size]
        else:
            selected = old_order + fresh_order[:size - 60]
        nested[str(size)] = selected
    if any(not set(nested[str(left)]).issubset(nested[str(right)])
           for left, right in zip((10, 20, 40, 60, 100, 150), (20, 40, 60, 100, 150, 200), strict=True)):
        raise D153ScalingError("D153 layout subsets are not nested")
    return {
        "status": "hash_only_layout_audit_passed",
        "fresh_layouts": fresh,
        "fresh_layout_count": len(fresh),
        "unique_fresh_layout_hashes": len(set(hashes)),
        "d152_train_layout_count": len(train_layouts),
        "d152_validation_layout_count": len(val_layouts),
        "fresh_overlap_with_d152_train_validation": 0,
        "nested_layout_hashes": nested,
    }


def build_run_lock() -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise D153ScalingError("commit D153 sources, protocol, config, and tests before freezing lock")
    cfg = _load_config()
    if cfg.get("schema") != "phase15_d153_coverage_scaling_v3" or cfg.get("status") != "protocol_amended_prelock":
        raise D153ScalingError("D153 scaling config schema/status mismatch")
    parents = _parent_hashes(cfg)
    audit = _json(ROOT / cfg["parents"]["d153_audit_result"])
    if (audit.get("status") != "completed_read_only_exploratory_analysis"
            or audit.get("test_rows_indexed_or_scored") != 0 or audit.get("reserve_access") is not False):
        raise D153ScalingError("D153 preliminary validation audit scope is invalid")
    layout_audit = candidate_layout_audit(cfg)
    reused = cfg["reused_corpus"]
    corpus_path = ROOT / reused["dataset"]
    original_manifest = _json(ROOT / reused["original_manifest"])
    original_lock = _json(ROOT / cfg["parents"]["d153_aborted_lock"])
    if (sha256_file(corpus_path) != reused["dataset_sha256"]
            or original_manifest.get("status") != "aborted_or_incomplete"
            or original_manifest.get("lock_sha256") != sha256_file(ROOT / cfg["parents"]["d153_aborted_lock"])
            or original_manifest.get("source_commit") != original_lock.get("source_commit")
            or original_manifest.get("test_rows_indexed_or_scored") != 0
            or original_manifest.get("reserve_access") is not False):
        raise D153ScalingError("preserved D153 corpus does not match the aborted run manifest")
    with np.load(corpus_path, allow_pickle=False) as corpus:
        corpus_layouts = set(corpus["layouts"].astype(str).tolist())
        if (len(corpus["x_R0"]) != reused["expected_state_rows"]
                or corpus["summary_R4"].shape != (reused["expected_state_rows"], 9, 7)
                or corpus_layouts != {item["layout_sha256"] for item in layout_audit["fresh_layouts"]}):
            raise D153ScalingError("reused D153 corpus rows/layout identities differ from locked audit")
    accounting_path = ROOT / reused["original_accounting"]
    if sha256_file(accounting_path) != reused["collector_accounting_sha256"]:
        raise D153ScalingError("preserved D153 collector accounting hash mismatch")
    output = ROOT / cfg["outputs"]["run_dir"]
    lock_path = ROOT / cfg["outputs"]["lock"]
    report_path = ROOT / cfg["outputs"]["report"]
    gate_path = ROOT / cfg["outputs"]["d60_gate"]
    if output.exists() or lock_path.exists() or report_path.exists() or gate_path.exists():
        raise D153ScalingError("refusing to overwrite D153 scaling outputs or lock")
    return {
        "schema": "phase15_d153_coverage_scaling_lock_v3",
        "status": "locked_corrected_reproduction_and_exploratory_coverage_scaling",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "config_sha256": sha256_file(ROOT / CONFIG_PATH),
        "parent_sha256": parents,
        "reused_corpus": {"path": reused["dataset"], "sha256": reused["dataset_sha256"],
                          "expected_state_rows": reused["expected_state_rows"],
                          "expected_branch_count": reused["expected_branch_count"]},
        "layout_audit": layout_audit,
        "fixed_d152_split": {"train_layouts": _load_d152_split(cfg)[2], "validation_layouts": _load_d152_split(cfg)[3]},
        "predictor": cfg["predictor"], "coverage": cfg["coverage"], "metrics": cfg["metrics"],
        "gates": cfg["gates"], "resources": cfg["resources"], "scope": cfg["scope"],
        "outputs": cfg["outputs"], "overwrite_forbidden": True,
    }


def validate_run_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _load_config()
    lock = _json(ROOT / cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d153_coverage_scaling_lock_v3":
        raise D153ScalingError("D153 scaling lock schema mismatch")
    if _git("status", "--porcelain"):
        raise D153ScalingError("D153 scaling run requires clean committed source and lock")
    if lock.get("config_sha256") != sha256_file(ROOT / CONFIG_PATH):
        raise D153ScalingError("D153 config changed after lock")
    if lock.get("source_sha256") != {name: sha256_file(ROOT / name) for name in lock["source_sha256"]}:
        raise D153ScalingError("D153 source hash differs from lock")
    if lock.get("parent_sha256") != _parent_hashes(cfg):
        raise D153ScalingError("D153 parent artifact hash differs from lock")
    corpus = cfg["reused_corpus"]
    if (lock.get("reused_corpus", {}).get("sha256") != sha256_file(ROOT / corpus["dataset"])
            or lock["reused_corpus"].get("sha256") != corpus["dataset_sha256"]):
        raise D153ScalingError("D153 preserved training corpus hash differs from amendment lock")
    if sha256_file(ROOT / corpus["original_accounting"]) != corpus["collector_accounting_sha256"]:
        raise D153ScalingError("D153 preserved collector accounting differs from amendment lock")
    current_layout_audit = candidate_layout_audit(cfg)
    if current_layout_audit != lock.get("layout_audit"):
        raise D153ScalingError("D153 hash-only candidate layout audit differs from lock")
    if (lock.get("scope") != cfg["scope"] or lock.get("reserve_access") is True
            or cfg["scope"].get("test_rows_indexed_or_scored") is not False
            or cfg["scope"].get("reserve_access") is not False):
        raise D153ScalingError("D153 run scope mismatch or forbidden access enabled")
    for rel in (cfg["outputs"]["run_dir"], cfg["outputs"]["report"]):
        if (ROOT / rel).exists():
            raise D153ScalingError(f"D153 output already exists; one-shot run will not overwrite: {rel}")
    return cfg, lock


def _load_reused_corpus(cfg: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    spec = lock["reused_corpus"]
    path = ROOT / spec["path"]
    if sha256_file(path) != spec["sha256"]:
        raise D153ScalingError("preserved D153 training corpus hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    expected_layouts = {item["layout_sha256"] for item in lock["layout_audit"]["fresh_layouts"]}
    layouts = arrays["layouts"].astype(str)
    if (len(arrays["x_R0"]) != spec["expected_state_rows"]
            or len(arrays["summary_R4"]) * 9 != spec["expected_branch_count"]
            or set(layouts.tolist()) != expected_layouts):
        raise D153ScalingError("preserved D153 corpus fails locked shape/layout audit")
    accounting = _json(ROOT / cfg["reused_corpus"]["original_accounting"])
    return {"path": path, "array_rows": len(layouts), "sha256": spec["sha256"],
            "bytes": path.stat().st_size, "continuation_branch_count": spec["expected_branch_count"],
            "arrays": arrays, "episodes": accounting}


def _layout_weighted_summary_rmse(target: np.ndarray, prediction: np.ndarray,
                                  layouts: np.ndarray) -> float:
    truth, estimate = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    groups = np.asarray(layouts).astype(str)
    if truth.shape != estimate.shape or truth.ndim != 3 or truth.shape[1:] != (9, 7) or groups.shape != (len(truth),):
        raise ValueError("D153 layout-weighted summary RMSE requires matching [N,9,7] arrays and layout IDs")
    weights = _layout_balanced_weights(groups)
    squared = (truth - estimate) ** 2
    return float(np.sqrt(np.sum(weights[:, None, None] * squared) / 63.0))


def run_d60_reproduction_gate() -> dict[str, Any]:
    """Fit only the original D60 cell and verify D152 prediction/metric parity."""
    cfg, lock = validate_run_lock()
    gate_path = ROOT / cfg["outputs"]["d60_gate"]
    if gate_path.exists():
        raise D153ScalingError("D60 reproduction gate artifact already exists; refusing to overwrite")
    original_path = ROOT / cfg["parents"]["d152_features"]
    predictions_path = ROOT / cfg["parents"]["d152_predictions"]
    result_path = ROOT / cfg["parents"]["d152_result"]
    with np.load(original_path, allow_pickle=False) as source:
        source_layouts = source["layouts"].astype(str).tolist()
        train_ids, val_ids, _, _ = train_validation_indices(source_layouts, cfg["fixed_split"])
        train_x = source["x_R0"][train_ids].astype(np.float64)
        train_summary = source["x_R4"][train_ids, 61:124].astype(np.float64).reshape(-1, 9, 7)
        train_layouts = np.asarray(source_layouts, dtype=str)[train_ids]
        val_x = source["x_R0"][val_ids].astype(np.float64)
        val_y = source["y"][val_ids].astype(np.float64)
        val_layouts = np.asarray(source_layouts, dtype=str)[val_ids]
    old_order = lock["layout_audit"]["nested_layout_hashes"]["60"]
    if set(train_layouts.tolist()) != set(old_order):
        raise D153ScalingError("D60 reproduction train layouts do not match frozen D152 split")
    examples, targets, _ = expand_action_examples(train_x, train_summary)
    weights = expand_layout_weights(train_layouts)
    model = ActionSummaryMLP(
        seed=int(cfg["predictor"]["seed"]), hidden=tuple(cfg["predictor"]["hidden"]),
        epochs=int(cfg["predictor"]["epochs"]), learning_rate=float(cfg["predictor"]["learning_rate"]),
        l2=float(cfg["predictor"]["l2"]),
    ).fit(examples, targets, weights)
    prediction = _normalize_outcomes(model.predict(val_x)).reshape(-1, 63)
    with np.load(predictions_path, allow_pickle=False) as saved:
        reference = saved["validation_mlp32"].astype(np.float64).reshape(-1, 63)
        exact_summary = saved["validation_exact_summary"].astype(np.float64).reshape(-1, 63)
        reference_y = saved["validation_teacher_regrets"].astype(np.float64)
    if not np.array_equal(reference_y, val_y):
        raise D153ScalingError("D60 gate validation labels differ from D152 frozen prediction artifact")
    max_abs = float(np.max(np.abs(prediction - reference)))
    tolerance = float(cfg["predictor"]["d60_saved_d152_prediction_reproduction_absolute_tolerance"])
    if max_abs > tolerance:
        failed_gate = {"schema": "phase15_d153_d60_reproduction_gate_v3", "status": "failed_prediction_reproduction",
                       "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
                       "reused_dataset_sha256": lock["reused_corpus"]["sha256"],
                       "maximum_absolute_prediction_difference": max_abs, "prediction_tolerance": tolerance,
                       "test_rows_indexed_or_scored": 0, "reserve_access": False}
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(failed_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise D153ScalingError(f"corrected D60 prediction reproduction failed: max_abs={max_abs}, tolerance={tolerance}")
    head_r4 = restore_decision_head(ROOT / cfg["parents"]["d150_r4_checkpoint"])
    decisions = _macro_metrics(val_y, head_r4.predict(np.concatenate((val_x, prediction), axis=1)), val_layouts)
    reference_decisions = _macro_metrics(val_y, head_r4.predict(np.concatenate((val_x, reference), axis=1)), val_layouts)
    decision_parity = {name: abs(decisions[key] - float(value)) for name, key, value in (
        ("macro_regret", "layout_macro_normalized_regret", reference_decisions["layout_macro_normalized_regret"]),
        ("catastrophic_fraction", "state_micro_catastrophic_fraction_regret_ge_0_5", reference_decisions["state_micro_catastrophic_fraction_regret_ge_0_5"]),
        ("best_action_agreement", "best_action_set_agreement", reference_decisions["best_action_set_agreement"]),
    )}
    d152_result = _json(result_path)["predictors"]["shared_action_mlp_32"]["validation"]
    weighted_rmse = _layout_weighted_summary_rmse(
        exact_summary.reshape(-1, 9, 7), prediction.reshape(-1, 9, 7), val_layouts)
    metric_diff = abs(weighted_rmse - float(d152_result["weighted_rmse"]))
    gate = {
        "schema": "phase15_d153_d60_reproduction_gate_v3", "status": "passed",
        "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
        "d152_prediction_parent_sha256": sha256_file(predictions_path),
        "reused_dataset_sha256": lock["reused_corpus"]["sha256"],
        "maximum_absolute_prediction_difference": max_abs, "prediction_tolerance": tolerance,
        "weighted_validation_rmse": weighted_rmse, "d152_weighted_validation_rmse": float(d152_result["weighted_rmse"]),
        "weighted_validation_rmse_absolute_difference": metric_diff,
        "frozen_r4_decision_metrics": decisions,
        "reference_r4_decision_metrics": reference_decisions,
        "decision_metric_maximum_absolute_difference": max(decision_parity.values()),
        "test_rows_indexed_or_scored": 0, "reserve_access": False,
        "validation_reuse_disclosure": "Fixed D152 validation was previously reused through D153 partial D10-D60 diagnostics; this is reproduction only, not fresh evidence.",
    }
    metric_tol = float(cfg["predictor"]["d60_saved_d152_metric_reproduction_absolute_tolerance"])
    saved_macro = d152_result["downstream_decision"]
    saved_decision_diffs = {
        "macro_regret": abs(reference_decisions["layout_macro_normalized_regret"] - float(saved_macro["mean_normalized_regret_layout_macro"])),
        "catastrophic_fraction": abs(reference_decisions["state_micro_catastrophic_fraction_regret_ge_0_5"] - float(saved_macro["catastrophic_fraction_state_micro"])),
        "best_action_agreement": abs(reference_decisions["best_action_set_agreement"] - float(saved_macro["best_action_set_agreement"])),
    }
    gate["saved_d152_decision_metric_maximum_absolute_difference"] = max(saved_decision_diffs.values())
    gate["saved_d152_decision_metric_differences"] = saved_decision_diffs
    if (metric_diff > metric_tol or max(decision_parity.values()) > metric_tol
            or max(saved_decision_diffs.values()) > metric_tol):
        gate["schema"] = "phase15_d153_d60_reproduction_gate_v3"
        gate["status"] = "failed_metric_parity"
        gate["metric_tolerance"] = metric_tol
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise D153ScalingError("D60 predictions reproduced but D152 summary/decision metric parity failed")
    gate["metric_tolerance"] = metric_tol
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def _summary_from_branches(branches: list[dict[str, Any]], max_steps: int, diagonal: float) -> np.ndarray:
    rows = []
    for branch in branches:
        status = str(branch["terminal_status"])
        if status not in OUTCOME_INDEX:
            raise D153ScalingError(f"unknown exact continuation terminal status: {status}")
        onehot = np.eye(4, dtype=np.float64)[OUTCOME_INDEX[status]]
        rows.append(np.concatenate((onehot, (
            float(branch["terminal_steps_from_source"]) / max_steps,
            float(branch["terminal_goal_distance"]) / diagonal,
            float(branch["total_noncoast_effort"]) / max_steps,
        ))))
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (9, 7) or not np.isfinite(result).all():
        raise D153ScalingError("exact continuation summary does not match the 9x7 contract")
    return result


def collect_fresh_training_layouts(cfg: dict[str, Any], lock: dict[str, Any], output: Path,
                                   started: float) -> dict[str, Any]:
    """Collect training targets on audited layouts with the frozen historical CART only."""
    parent = cfg["parents"]
    policy_payload = _json(ROOT / parent["phase14f_student"])
    policy = FrozenCartPolicy(policy_payload)
    records: list[tuple[np.ndarray, np.ndarray, str, int, int, int, str]] = []
    episode_records = []
    block = lock["layout_audit"]["fresh_layouts"]
    max_bytes = int(cfg["resources"]["generated_bytes_max"])
    max_seconds = int(cfg["resources"]["collection_wall_seconds_max"])
    for layout_index, item in enumerate(block):
        if time.monotonic() - started > max_seconds:
            raise D153ScalingError("D153 collection exceeded protocol wall-time cap")
        scenario = generate_staggered_corridor(int(item["seed"]))
        if scenario.layout_sha256 != item["layout_sha256"]:
            raise D153ScalingError("fresh layout identity changed since lock")
        state = scenario.initial_state
        decision = 0
        actions: list[int] = []
        while state.status == "active":
            raw = state_features(scenario, state).astype(np.float64)
            if raw.shape != (52,) or not np.isfinite(raw).all():
                raise D153ScalingError("frozen collector state violates deployable 52-feature contract")
            action = int(policy.action_for_features(raw))
            if not 0 <= action < 9:
                raise D153ScalingError("frozen collector selected illegal primitive action")
            branches = forced_first_action_values(scenario, state, decision_index=decision,
                                                   expected_student_features=raw)
            if [int(branch["forced_action_id"]) for branch in branches] != list(range(9)):
                raise D153ScalingError("forced-action teacher action IDs/order changed")
            summaries = _summary_from_branches(branches, scenario.spec.max_steps,
                                                float(np.hypot(scenario.spec.width, scenario.spec.height)))
            x0 = build_feature_matrix(raw[None, :], "raw52_plus_physics9")[0]
            records.append((x0, summaries, scenario.layout_sha256, int(item["seed"]), decision, action, "student_runtime_input_52"))
            state = corridor_step(state, action, scenario.spec).state
            decision += 1
            if len(records) * 9 * 7 * 8 > max_bytes:
                raise D153ScalingError("D153 generated target payload exceeded protocol storage cap")
        episode_records.append({"seed": int(item["seed"]), "layout_sha256": scenario.layout_sha256,
                                "decision_state_count": decision, "collector_terminal_status": state.status,
                                "collector_terminal_steps": int(state.elapsed_steps)})
    if not records:
        raise D153ScalingError("D153 fresh collector produced no active states")
    x, y, layouts, seeds, decisions, collector_actions, _ = zip(*records, strict=True)
    arrays = {
        "x_R0": np.stack(x).astype(np.float32), "summary_R4": np.stack(y).astype(np.float32),
        "layouts": np.asarray(layouts, dtype="U64"), "seeds": np.asarray(seeds, dtype=np.int64),
        "decision_index": np.asarray(decisions, dtype=np.int64),
        "collector_action_ids_audit_only": np.asarray(collector_actions, dtype=np.int8),
    }
    dataset_path = output / "fresh_training_rows.npz"
    np.savez_compressed(dataset_path, **arrays)
    if dataset_path.stat().st_size > max_bytes:
        raise D153ScalingError("D153 compressed dataset exceeded protocol storage cap")
    episode_path = output / "collector_accounting.json"
    episode_path.write_text(json.dumps(episode_records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": dataset_path, "array_rows": len(records), "episodes": episode_records,
            "sha256": sha256_file(dataset_path), "bytes": dataset_path.stat().st_size,
            "continuation_branch_count": len(records) * 9}


def _layout_balanced_weights(layouts: np.ndarray) -> np.ndarray:
    group = np.asarray(layouts).astype(str)
    unique, counts = np.unique(group, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    weights = np.asarray([1.0 / count_map[name] for name in group], dtype=np.float64)
    weights /= weights.sum()
    return weights


def _distances(train: np.ndarray, val: np.ndarray, fixed_mean: np.ndarray, fixed_std: np.ndarray) -> np.ndarray:
    tr = (np.asarray(train, dtype=np.float64) - fixed_mean) / fixed_std
    va = (np.asarray(val, dtype=np.float64) - fixed_mean) / fixed_std
    result = np.empty(len(va), dtype=np.float64)
    for start in range(0, len(va), 32):
        delta = va[start:start + 32, None, :] - tr[None, :, :]
        result[start:start + 32] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    return result


def _per_component_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    t, p = np.asarray(target).reshape(-1, 9, 7), np.asarray(predicted).reshape(-1, 9, 7)
    outcome_true = np.argmax(t[:, :, :4], axis=2)
    outcome_pred = np.argmax(p[:, :, :4], axis=2)
    result = {"outcome_accuracy": float(np.mean(outcome_true == outcome_pred))}
    for status, index in (("success", 0), ("collision", 3)):
        truth = outcome_true == index
        pred = outcome_pred == index
        tp = int(np.sum(truth & pred)); fp = int(np.sum(~truth & pred)); fn = int(np.sum(truth & ~pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[f"{status}_precision"] = precision
        result[f"{status}_recall"] = recall
        result[f"{status}_f1"] = f1
    for index, name in ((4, "terminal_steps"), (5, "terminal_goal_distance"), (6, "noncoast_effort")):
        err = p[:, :, index] - t[:, :, index]
        result[f"{name}_mae"] = float(np.mean(np.abs(err)))
        result[f"{name}_rmse"] = float(np.sqrt(np.mean(err * err)))
    return result


def _macro_metrics(y: np.ndarray, prediction: np.ndarray, layouts: np.ndarray) -> dict[str, float]:
    target, scores = np.asarray(y, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    selected = np.argmin(scores, axis=1)
    regret = target[np.arange(len(target)), selected]
    unique = np.unique(np.asarray(layouts).astype(str))
    layout_regrets = [float(np.mean(regret[np.asarray(layouts).astype(str) == name])) for name in unique]
    return {
        "layout_macro_normalized_regret": float(np.mean(layout_regrets)),
        "state_micro_catastrophic_fraction_regret_ge_0_5": float(np.mean(regret >= 0.5)),
        "best_action_set_agreement": float(np.mean(regret <= 1e-12)),
        "state_micro_normalized_regret": float(np.mean(regret)),
        "selected_action_frequency": np.bincount(selected, minlength=9).tolist(),
    }


def run_scaling() -> dict[str, Any]:
    cfg, lock = validate_run_lock()
    gate_path = ROOT / cfg["outputs"]["d60_gate"]
    if not gate_path.exists():
        raise D153ScalingError("corrected scaling cannot continue before D60-only reproduction gate passes")
    gate = _json(gate_path)
    if (gate.get("status") != "passed"
            or gate.get("lock_sha256") != sha256_file(ROOT / cfg["outputs"]["lock"])
            or gate.get("reused_dataset_sha256") != lock["reused_corpus"]["sha256"]):
        raise D153ScalingError("D60 reproduction gate is not a passing artifact for this lock/corpus")
    output = ROOT / cfg["outputs"]["run_dir"]
    output.mkdir(parents=True, exist_ok=False)
    started_wall = time.monotonic()
    manifest_path = output / "manifest.json"
    manifest = {
        "schema": "phase15_d153_coverage_scaling_manifest_v3", "status": "loading_preserved_training_rows",
        "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]), "source_commit": lock["source_commit"],
        "parent_sha256": lock["parent_sha256"], "layout_audit_sha256": hashlib.sha256(
            json.dumps(lock["layout_audit"], sort_keys=True).encode("utf-8")).hexdigest().upper(),
        "scope": cfg["scope"], "reserve_access": False, "test_rows_indexed_or_scored": 0,
        "d60_reproduction_gate_sha256": sha256_file(gate_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        fresh = _load_reused_corpus(cfg, lock)
        original_path = ROOT / cfg["parents"]["d152_features"]
        predictions_path = ROOT / cfg["parents"]["d152_predictions"]
        with np.load(original_path, allow_pickle=False) as source:
            source_layouts = source["layouts"].astype(str).tolist()
            split_cfg = cfg["fixed_split"]
            train_ids, val_ids, train_names, val_names = train_validation_indices(source_layouts, split_cfg)
            old_train_x = source["x_R0"][train_ids].astype(np.float64)
            old_train_summary = source["x_R4"][train_ids, 61:124].astype(np.float64).reshape(-1, 9, 7)
            old_train_layouts = np.asarray(source_layouts, dtype=str)[train_ids]
            val_x = source["x_R0"][val_ids].astype(np.float64)
            val_y = source["y"][val_ids].astype(np.float64)
            val_layouts = np.asarray(source_layouts, dtype=str)[val_ids]
        with np.load(predictions_path, allow_pickle=False) as saved:
            val_summary = saved["validation_exact_summary"].astype(np.float64).reshape(-1, 9, 7)
            if not np.array_equal(saved["validation_teacher_regrets"], val_y):
                raise D153ScalingError("fixed D152 validation teacher labels do not match feature parent")
        new = fresh["arrays"]
        fresh_x = new["x_R0"].astype(np.float64)
        fresh_summary = new["summary_R4"].astype(np.float64)
        fresh_layouts = new["layouts"].astype(str)
        old_order = lock["layout_audit"]["nested_layout_hashes"]["60"]
        fresh_order = [value for value in lock["layout_audit"]["nested_layout_hashes"]["200"] if value not in set(old_order)]
        fixed_mean, fixed_std = old_train_x.mean(axis=0), old_train_x.std(axis=0)
        fixed_std[fixed_std < 1e-8] = 1.0
        base_loo = np.empty(len(old_train_x), dtype=np.float64)
        standardized_old = (old_train_x - fixed_mean) / fixed_std
        for i in range(len(old_train_x)):
            ds = np.linalg.norm(standardized_old - standardized_old[i], axis=1)
            ds[i] = np.inf
            base_loo[i] = float(np.min(ds))
        fixed_cutoff = float(np.quantile(base_loo, 0.90))
        if not np.isclose(fixed_cutoff, cfg["coverage"]["train_loo_p90_cutoff"], atol=1e-9, rtol=0.0):
            raise D153ScalingError("fixed D60 train leave-one-out support cutoff mismatch")
        head_r0 = restore_decision_head(ROOT / cfg["parents"]["d150_r0_checkpoint"])
        head_r4 = restore_decision_head(ROOT / cfg["parents"]["d150_r4_checkpoint"])
        exact_r4_decisions = _macro_metrics(val_y, head_r4.predict(np.concatenate((val_x, val_summary.reshape(-1, 63)), axis=1)), val_layouts)
        direct_r0_decisions = _macro_metrics(val_y, head_r0.predict(val_x), val_layouts)
        result_rows = []
        predictions_by_size = {}
        d60_reproduction = None
        for size in cfg["layout_design"]["nested_sizes"]:
            selected_hashes = lock["layout_audit"]["nested_layout_hashes"][str(size)]
            chosen = set(selected_hashes)
            old_mask = np.isin(old_train_layouts, list(chosen))
            new_mask = np.isin(fresh_layouts, list(chosen))
            train_x = np.concatenate((old_train_x[old_mask], fresh_x[new_mask]), axis=0)
            train_summary = np.concatenate((old_train_summary[old_mask], fresh_summary[new_mask]), axis=0)
            train_layout = np.concatenate((old_train_layouts[old_mask], fresh_layouts[new_mask]), axis=0)
            if len(train_x) == 0 or set(train_layout.tolist()) != chosen:
                raise D153ScalingError(f"D{size} does not contain exactly its locked layout set")
            examples, targets, _ = expand_action_examples(train_x, train_summary)
            weights = expand_layout_weights(train_layout)
            model = ActionSummaryMLP(
                seed=int(cfg["predictor"]["seed"]), hidden=tuple(cfg["predictor"]["hidden"]),
                epochs=int(cfg["predictor"]["epochs"]), learning_rate=float(cfg["predictor"]["learning_rate"]),
                l2=float(cfg["predictor"]["l2"]),
            ).fit(examples, targets, weights)
            # D152 applied this exact post-processing to all recorded predictions.
            # Keep training unchanged; normalize only for parity, inference, and metrics.
            pred_train = _normalize_outcomes(model.predict(train_x)).reshape(-1, 63)
            pred_val = _normalize_outcomes(model.predict(val_x)).reshape(-1, 63)
            if int(size) == 60:
                with np.load(predictions_path, allow_pickle=False) as saved:
                    reference = saved["validation_mlp32"].astype(np.float64).reshape(-1, 63)
                max_abs = float(np.max(np.abs(pred_val - reference)))
                tolerance = float(cfg["predictor"]["d60_saved_d152_prediction_reproduction_absolute_tolerance"])
                if max_abs > tolerance:
                    raise D153ScalingError(f"D60 predictor does not reproduce D152 saved outputs: max_abs={max_abs}")
                d60_reproduction = {"maximum_absolute_prediction_difference": max_abs,
                                    "tolerance": tolerance, "passed": True}
            train_m = summary_metrics(train_summary.reshape(-1, 63), pred_train)
            val_m = summary_metrics(val_summary.reshape(-1, 63), pred_val)
            val_components = _per_component_metrics(val_summary, pred_val.reshape(-1, 9, 7))
            decisions = _macro_metrics(val_y, head_r4.predict(np.concatenate((val_x, pred_val), axis=1)), val_layouts)
            distances = _distances(train_x, val_x, fixed_mean, fixed_std)
            result_rows.append({
                "layout_count": int(size), "training_state_count": len(train_x), "training_layout_count": len(chosen),
                "validation_state_count": len(val_x), "train_summary_metrics": train_m,
                "validation_summary_metrics": val_m, "validation_summary_components": val_components,
                "validation_downstream_decisions": decisions,
                "coverage": {"nearest_distance_median": float(np.median(distances)),
                             "nearest_distance_p90": float(np.quantile(distances, .9)),
                             "fraction_inside_fixed_train_loo_p90": float(np.mean(distances <= fixed_cutoff)),
                             "fraction_inside_twice_fixed_train_loo_p90": float(np.mean(distances <= 2 * fixed_cutoff))},
                "training": {"best_epoch": model.best_epoch, "training_objective": model.training_loss,
                             "parameter_count": model.parameter_count},
            })
            predictions_by_size[str(size)] = pred_val.astype(np.float32)
            print(f"D{size}: train_rows={len(train_x)} val_RMSE={val_m['rmse']:.5f} "
                  f"macro_regret={decisions['layout_macro_normalized_regret']:.5f}", flush=True)
        all_layouts = np.concatenate((old_train_layouts, fresh_layouts))
        all_x = np.concatenate((old_train_x, fresh_x))
        all_summary = np.concatenate((old_train_summary, fresh_summary))
        per_layout_counts = {str(name): int(np.sum(all_layouts == name)) for name in sorted(set(all_layouts))}
        physical_correlations = {}
        largest_train = all_x[np.isin(all_layouts, lock["layout_audit"]["nested_layout_hashes"]["200"])]
        final_dist = _distances(largest_train, val_x, fixed_mean, fixed_std)
        for index, name in ((0, "x_norm"), (1, "y_norm"), (2, "vx_norm"), (3, "vy_norm"),
                            (15, "remaining_time_fraction"), (52, "goal_dx_norm"), (53, "goal_dy_norm"),
                            (54, "goal_distance_norm"), (55, "speed_norm"), (56, "forward_speed_norm"),
                            (57, "lateral_speed_norm"), (58, "velocity_goal_alignment"),
                            (59, "obstacle_clearance_norm"), (60, "stopping_distance_norm")):
            physical_correlations[name] = correlation_pair(final_dist, val_x[:, index])["spearman"]
        decisions60 = result_rows[3]["validation_downstream_decisions"]
        metrics60 = result_rows[3]["validation_summary_metrics"]
        coverage60 = result_rows[3]["coverage"]
        decisions200 = result_rows[-1]["validation_downstream_decisions"]
        metrics200 = result_rows[-1]["validation_summary_metrics"]
        coverage200 = result_rows[-1]["coverage"]
        coverage_gate = ((coverage60["nearest_distance_median"] - coverage200["nearest_distance_median"])
                         / coverage60["nearest_distance_median"] >= cfg["gates"]["coverage_material"]["median_distance_relative_drop_min"]
                         and (coverage60["nearest_distance_p90"] - coverage200["nearest_distance_p90"])
                         / coverage60["nearest_distance_p90"] >= cfg["gates"]["coverage_material"]["p90_distance_relative_drop_min"])
        rmse_gain = 1.0 - metrics200["rmse"] / max(metrics60["rmse"], 1e-12)
        regret_gain = decisions60["layout_macro_normalized_regret"] - decisions200["layout_macro_normalized_regret"]
        cat_gain = decisions60["state_micro_catastrophic_fraction_regret_ge_0_5"] - decisions200["state_micro_catastrophic_fraction_regret_ge_0_5"]
        rules = cfg["gates"]
        prediction_gate = rmse_gain >= rules["prediction_material"]["aggregate_validation_rmse_relative_drop_min"]
        decision_gate = (regret_gain >= rules["decision_material"]["macro_regret_absolute_drop_min"]
                         and cat_gain >= rules["decision_material"]["catastrophic_fraction_absolute_drop_min"])
        plateau = result_rows[-2]
        d150_d200 = {
            "distance_median_relative_gain": (plateau["coverage"]["nearest_distance_median"] - coverage200["nearest_distance_median"])
                / plateau["coverage"]["nearest_distance_median"],
            "summary_rmse_relative_gain": 1.0 - metrics200["rmse"] / max(plateau["validation_summary_metrics"]["rmse"], 1e-12),
            "regret_absolute_change": plateau["validation_downstream_decisions"]["layout_macro_normalized_regret"]
                - decisions200["layout_macro_normalized_regret"],
            "catastrophic_absolute_change": plateau["validation_downstream_decisions"]["state_micro_catastrophic_fraction_regret_ge_0_5"]
                - decisions200["state_micro_catastrophic_fraction_regret_ge_0_5"],
        }
        saturation = (d150_d200["distance_median_relative_gain"] < rules["saturation"]["d150_to_d200_distance_relative_gain_below"]
                      and d150_d200["summary_rmse_relative_gain"] < rules["saturation"]["rmse_relative_gain_below"]
                      and abs(d150_d200["regret_absolute_change"]) < rules["saturation"]["regret_absolute_change_below"]
                      and abs(d150_d200["catastrophic_absolute_change"]) < rules["saturation"]["catastrophic_absolute_change_below"])
        if coverage_gate and prediction_gate and decision_gate:
            interpretation = "consistent_with_coverage_driven_downstream_recovery"
        elif coverage_gate and prediction_gate and not decision_gate:
            interpretation = "coverage_and_prediction_improved_without_material_decision_recovery"
        elif saturation:
            interpretation = "uniform_layout_scaling_saturated_by_d150"
        elif coverage_gate and rmse_gain < rules["summary_not_material_below"]:
            interpretation = "coverage_improved_but_summary_prediction_remains_limited"
        else:
            interpretation = "mixed_or_inconclusive"
        result = {
            "schema": "phase15_d153_coverage_scaling_result_v3", "status": "completed_exploratory_scaling",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": lock["source_commit"],
            "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]), "config_sha256": sha256_file(ROOT / CONFIG_PATH),
            "parent_sha256": lock["parent_sha256"], "fresh_dataset": {
                "array_rows": fresh["array_rows"], "sha256": fresh["sha256"],
                "bytes": fresh["bytes"], "continuation_branch_count": fresh["continuation_branch_count"]},
            "fresh_layouts": len(lock["layout_audit"]["fresh_layouts"]),
            "training_layout_state_counts": per_layout_counts,
            "validation_layout_count": len(val_names), "validation_state_count": len(val_x),
            "scale_results": result_rows,
            "fixed_references": {"direct_r0_decisions": direct_r0_decisions, "exact_r4_decisions": exact_r4_decisions},
            "d60_saved_d152_prediction_reproduction": d60_reproduction,
            "fixed_support_cutoff": fixed_cutoff,
            "physical_variable_distance_correlations_at_d200": physical_correlations,
            "d60_to_d200_changes": {"coverage_material_gate_passed": bool(coverage_gate),
                                    "validation_rmse_relative_gain": float(rmse_gain), "prediction_material_gate_passed": bool(prediction_gate),
                                    "macro_regret_absolute_gain": float(regret_gain), "catastrophic_fraction_absolute_gain": float(cat_gain),
                                    "decision_material_gate_passed": bool(decision_gate), "d150_to_d200": d150_d200,
                                    "saturation_gate_passed": bool(saturation)},
            "interpretation": interpretation,
            "validation_reuse_disclosure": "The fixed 20-layout validation block was reused in D148-D152; all D153 validation results are exploratory and descriptive.",
            "scope": cfg["scope"], "test_rows_indexed_or_scored": 0, "reserve_access": False,
            "simulator_data_collection": "preserved frozen Phase14F state-corpus collector plus exact counterfactual continuation targets on locked training layouts only; reused from the preserved aborted D153 run",
            "elapsed_seconds": time.monotonic() - started_wall,
        }
        np.savez_compressed(output / "validation_predictions.npz", **predictions_by_size)
        result_path = output / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = render_report(result)
        report_path = ROOT / cfg["outputs"]["report"]
        report_path.write_text(report, encoding="utf-8")
        manifest.update({"status": "completed_exploratory_scaling", "fresh_dataset_sha256": fresh["sha256"],
                         "fresh_dataset_bytes": fresh["bytes"], "fresh_state_rows": fresh["array_rows"],
                         "episodes_collected": len(fresh["episodes"]), "result_sha256": sha256_file(result_path),
                         "report_sha256": sha256_file(report_path), "validation_predictions_sha256": sha256_file(output / "validation_predictions.npz"),
                         "finished_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": result["elapsed_seconds"]})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        manifest.update({"status": "aborted_or_incomplete", "failure": f"{type(exc).__name__}: {exc}",
                         "finished_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started_wall})
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D153 - Coverage Scaling Results", "", "**Status: exploratory, fixed reused validation.** No D152 test outcomes or reserve artifacts were accessed. This is not an independent confirmation.", "",
             f"Interpretation under frozen descriptive rules: **{result['interpretation']}**.", "",
             "## Scaling curve", "",
             "| Layouts | Train states | NN distance median / p90 | Inside p90 / 2x | Train RMSE | Validation RMSE | Macro regret | Catastrophic fraction | Best-set agreement |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["scale_results"]:
        c, tr, va, d = row["coverage"], row["train_summary_metrics"], row["validation_summary_metrics"], row["validation_downstream_decisions"]
        lines.append(f"| {row['layout_count']} | {row['training_state_count']} | {c['nearest_distance_median']:.3f} / {c['nearest_distance_p90']:.3f} | {c['fraction_inside_fixed_train_loo_p90']:.1%} / {c['fraction_inside_twice_fixed_train_loo_p90']:.1%} | {tr['rmse']:.4f} | {va['rmse']:.4f} | {d['layout_macro_normalized_regret']:.4f} | {d['state_micro_catastrophic_fraction_regret_ge_0_5']:.3f} | {d['best_action_set_agreement']:.3f} |")
    changes = result["d60_to_d200_changes"]
    lines += ["", "## Frozen gate readout", "",
              f"Coverage gate: `{changes['coverage_material_gate_passed']}`; D200 vs D60 validation RMSE relative gain: `{changes['validation_rmse_relative_gain']:.3%}` (gate `{changes['prediction_material_gate_passed']}`); macro-regret gain: `{changes['macro_regret_absolute_gain']:.4f}`; catastrophic-fraction gain: `{changes['catastrophic_fraction_absolute_gain']:.4f}` (decision gate `{changes['decision_material_gate_passed']}`).",
              "", "## Fixed references", "",
              f"Direct R0: macro regret `{result['fixed_references']['direct_r0_decisions']['layout_macro_normalized_regret']:.4f}`, catastrophic fraction `{result['fixed_references']['direct_r0_decisions']['state_micro_catastrophic_fraction_regret_ge_0_5']:.3f}`. Exact R4: macro regret `{result['fixed_references']['exact_r4_decisions']['layout_macro_normalized_regret']:.4f}`, catastrophic fraction `{result['fixed_references']['exact_r4_decisions']['state_micro_catastrophic_fraction_regret_ge_0_5']:.3f}`.",
              "", "## Physical regime diagnostics", "", "Spearman correlations between fixed-D200 nearest-train distance and predeclared deployable physical fields:", "", "| Field | Spearman |", "|---|---:|"]
    for name, value in result["physical_variable_distance_correlations_at_d200"].items():
        lines.append(f"| {name} | {value if value is not None else 'NA'} |")
    lines += ["", "## Scope and interpretation", "", result["validation_reuse_disclosure"],
              f"Fresh layouts: {result['fresh_layouts']}; fresh state rows: {result['fresh_dataset']['array_rows']}; forced continuation branches: {result['fresh_dataset']['continuation_branch_count']}. Test rows scored: 0; reserve accessed: false.",
              "", "The gates are predeclared descriptive thresholds, not significance tests. Coverage, summary prediction, and decision quality are reported separately; no causal claim follows from this fixed validation comparison.", ""]
    return "\n".join(lines)
