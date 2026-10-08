"""D157 one-shot fresh-layout evaluation of frozen D155 consequence models."""

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
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase12_teacher import state_features
from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14c import build_feature_matrix
from consequence_lab.phase14f import FrozenCartPolicy
from consequence_lab.phase14g import _collect_prior_layout_hashes, forced_first_action_values
from consequence_lab.phase15_d153_scaling import _summary_from_branches
from consequence_lab.phase15_d155_stage_c import _outcome_metrics, _weights_per_action
from consequence_lab.phase15_decision_aligned import decision_detail_metrics, regrets_from_exact_summaries
from consequence_lab.phase15_d156 import _decision_features, _read_json, load_predictors
from consequence_lab.phase15_outcome_summary import restore_decision_head
from consequence_lab.phase15_representation_screen import train_validation_indices

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d157_fresh_confirmation.yaml"
PROTOCOL = "reports/phase15_d157_protocol.md"
LOCK = "experiments/phase15_d157_fresh_confirmation.lock.json"
SOURCE_FILES = (
    CONFIG, PROTOCOL, "src/consequence_lab/phase15_d157.py",
    "src/consequence_lab/phase15_d153_scaling.py", "src/consequence_lab/phase15_d156.py",
    "src/consequence_lab/phase15_d155.py", "src/consequence_lab/phase15_d155_stage_c.py",
    "src/consequence_lab/phase15_decision_aligned.py", "src/consequence_lab/phase15_outcome_summary.py",
    "src/consequence_lab/phase15_representation_screen.py", "src/consequence_lab/phase15_generalization.py",
    "src/consequence_lab/phase14.py", "src/consequence_lab/phase14f.py",
    "src/consequence_lab/phase14g.py", "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase12_teacher.py", "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py", "src/consequence_lab/phase15_data.py",
    "scripts/freeze_phase15_d157_lock.py", "scripts/run_phase15_d157.py",
    "tests/test_phase15_d157.py",
)
TRAIN_SPLIT = {
    "train_layout_count": 60, "train_row_count": 571,
    "train_layouts_sha256": "1FF811CF3169CA339C23FC73A22617C10249297565702A592D2BD3C9C683769B",
    "validation_layout_count": 20, "validation_row_count": 202,
    "validation_layouts_sha256": "8B14D46BE6EF7C62150D0D584F1D171A33AC7BAF3E8A58991F97BA98980863EB",
}
ACTION_COUNT = 9


class D157Error(RuntimeError):
    """D157 provenance, protocol, or execution failure."""


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D157Error(f"git {' '.join(args)} failed") from exc


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _hash_parents(cfg: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}


def _audit_parents(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["parents"]
    d156_lock_path, d156_result_path, d156_manifest_path = (
        ROOT / p["d156_lock"], ROOT / p["d156_result"], ROOT / p["d156_manifest"])
    d156_lock, d156_result, d156_manifest = (
        _read_json(d156_lock_path), _read_json(d156_result_path), _read_json(d156_manifest_path))
    if (d156_lock.get("schema") != "phase15_d156_lock_v1"
            or d156_result.get("status") != "completed_exploratory_component_swap_audit"
            or d156_manifest.get("status") != d156_result.get("status")
            or d156_result.get("lock_sha256") != sha256_file(d156_lock_path)
            or d156_manifest.get("lock_sha256") != sha256_file(d156_lock_path)
            or d156_manifest.get("result_sha256") != sha256_file(d156_result_path)
            or d156_manifest.get("report_sha256") != sha256_file(ROOT / "reports/phase15_d156_result.md")
            or d156_manifest.get("test_rows_dereferenced") != 0
            or d156_manifest.get("reserve_access") is not False):
        raise D157Error("D156 parent chain/hash/scope audit failed")

    stage_a = _read_json(ROOT / p["d155_stage_a_result"])
    stage_a_manifest = _read_json(ROOT / p["d155_stage_a_manifest"])
    stage_a_lock = _read_json(ROOT / p["d155_stage_a_lock"])
    if (stage_a.get("status") != "completed_exploratory_oracle_consequence_ablation"
            or stage_a_lock.get("schema") != "phase15_d155_lock_v2"
            or stage_a.get("lock_sha256") != sha256_file(ROOT / p["d155_stage_a_lock"])
            or stage_a_manifest.get("lock_sha256") != sha256_file(ROOT / p["d155_stage_a_lock"])
            or stage_a_manifest.get("result_sha256") != sha256_file(ROOT / p["d155_stage_a_result"])
            or stage_a_manifest.get("checkpoint_sha256", {}).get("F_outcome_steps.npz")
                != sha256_file(ROOT / p["stage_a_head"])
            or stage_a_manifest.get("reserve_access") is not False
            or stage_a_manifest.get("test_rows_dereferenced") != 0):
        raise D157Error("D155 Stage-A head/result/manifest audit failed")

    stage_c = _read_json(ROOT / p["d155_stage_c_result"])
    stage_c_manifest = _read_json(ROOT / p["d155_stage_c_manifest"])
    stage_c_lock = _read_json(ROOT / p["d155_stage_c_lock"])
    checkpoint_files = {"per_action_constant.npz": p["stage_c_constant"],
                        "bounded_cart.json": p["stage_c_cart"],
                        "action_conditioned_mlp.npz": p["stage_c_mlp"]}
    for filename, relative in checkpoint_files.items():
        expected = stage_c_manifest.get("predictor_checkpoint_sha256", {}).get(filename)
        if not expected or expected != sha256_file(ROOT / relative):
            raise D157Error(f"D155 Stage-C predictor hash mismatch: {filename}")
    if (stage_c.get("status") != "completed_exploratory_compact_consequence_prediction_screen"
            or stage_c.get("offline_gate_passed") is not False
            or stage_c_lock.get("schema") != "phase15_d155_stage_c_lock_v1"
            or stage_c.get("lock_sha256") != sha256_file(ROOT / p["d155_stage_c_lock"])
            or stage_c_manifest.get("lock_sha256") != sha256_file(ROOT / p["d155_stage_c_lock"])
            or stage_c_manifest.get("result_sha256") != sha256_file(ROOT / p["d155_stage_c_result"])
            or stage_c_manifest.get("reserve_access") is not False
            or stage_c_manifest.get("test_rows_dereferenced") != 0):
        raise D157Error("D155 Stage-C result/manifest/scope audit failed")

    d153 = _read_json(ROOT / p["d153_result"])
    d153_manifest = _read_json(ROOT / p["d153_manifest"])
    if (d153.get("status") != "completed_exploratory_scaling"
            or d153_manifest.get("status") != "completed_exploratory_scaling"
            or d153.get("lock_sha256") != sha256_file(ROOT / p["d153_lock"])
            or d153_manifest.get("lock_sha256") != sha256_file(ROOT / p["d153_lock"])
            or d153_manifest.get("result_sha256") != sha256_file(ROOT / p["d153_result"])
            or d153_manifest.get("reserve_access") is not False
            or d153_manifest.get("test_rows_indexed_or_scored") != 0):
        raise D157Error("D153 training-corpus provenance audit failed")
    d153_lock = _read_json(ROOT / p["d153_lock"])
    if (d153_lock.get("schema") != "phase15_d153_coverage_scaling_lock_v3"
            or d153_lock.get("reused_corpus", {}).get("sha256") != sha256_file(ROOT / p["d153_train_corpus"])):
        raise D157Error("D153 frozen training-corpus lock mismatch")

    phase14f_manifest = _read_json(ROOT / p["phase14f_manifest"])
    phase14f_lock = _read_json(ROOT / p["phase14f_lock"])
    phase14f_cart_manifest = _read_json(ROOT / p["phase14f_cart_manifest"])
    if (phase14f_manifest.get("status") != "completed_closed_loop_development"
            or phase14f_manifest.get("reserve_access") is not False
            or phase14f_manifest.get("lock_sha256") != sha256_file(ROOT / p["phase14f_lock"])
            or phase14f_lock.get("reserve_access_authorized") is not False
            or phase14f_cart_manifest.get("model_sha256") != sha256_file(ROOT / p["phase14f_cart"])):
        raise D157Error("frozen Phase14F collector provenance audit failed")
    return {"d156_status": d156_result["status"],
            "d155_stage_c_gate_passed": stage_c["offline_gate_passed"],
            "d153_status": d153["status"],
            "phase14f_collector_gate_passed": phase14f_manifest.get("gate_passed")}


def candidate_layout_audit(cfg: dict[str, Any]) -> dict[str, Any]:
    """Hash-only uniqueness/reserve audit; no transition or outcome is computed."""
    block = cfg["layout_block"]
    start, stop = validate_seed_range(block)
    count = stop - start
    records = [{"seed": seed, "layout_sha256": generate_staggered_corridor(seed).layout_sha256}
               for seed in range(start, stop)]
    hashes = [row["layout_sha256"] for row in records]
    if len(set(hashes)) != count:
        raise D157Error("D157 candidate layout identities are not unique")
    prior_all, prior_spent, source_files, source_hashes = _collect_prior_layout_hashes(
        ROOT, ROOT / block_run_path(cfg))
    overlap_all, overlap_spent = set(hashes) & prior_all, set(hashes) & prior_spent
    if overlap_all or overlap_spent:
        raise D157Error("D157 candidate layouts overlap an existing historical layout artifact")
    d152_layouts = _d152_development_layouts(ROOT / cfg["parents"]["d152_features"])
    d153_lock = _read_json(ROOT / cfg["parents"]["d153_lock"])
    d153_hashes = {item["layout_sha256"] for item in d153_lock["layout_audit"]["fresh_layouts"]}
    if d153_hashes & set(hashes) or d152_layouts & set(hashes):
        raise D157Error("D157 candidate layout overlaps D152 train/validation or D153 training identities")
    return {"seed_start": start, "layout_count": count, "layouts": records,
            "unique_layout_hashes": len(set(hashes)), "historical_layout_hashes": len(prior_all),
            "historical_outcome_layout_hashes": len(prior_spent), "historical_source_count": len(source_files),
            "historical_source_sha256": source_hashes, "overlap_with_historical": 0,
            "overlap_with_outcome_bearing": 0, "overlap_with_d152_development_or_d153_train": 0,
            "reserve_range_checked_not_generated": block["reserved_range"],
            "outcomes_or_transitions_computed": False}


def validate_seed_range(block: dict[str, Any]) -> tuple[int, int]:
    """Validate positive-sized candidate interval against all sealed/used ranges."""
    start, count = int(block["seed_start"]), int(block["layout_count"])
    if start < 0 or count <= 0:
        raise D157Error("D157 candidate seed range must have nonnegative start and positive count")
    stop = start + count
    ranges = [block["reserved_range"], *block["forbidden_seed_ranges"]]
    for item in ranges:
        other_start, other_count = int(item["start"]), int(item["count"])
        if other_start < 0 or other_count <= 0:
            raise D157Error("D157 forbidden seed range is malformed")
        other_stop = other_start + other_count
        if start < other_stop and other_start < stop:
            raise D157Error(f"D157 candidate seeds overlap forbidden range [{other_start},{other_stop})")
    return start, stop


def block_run_path(cfg: dict[str, Any]) -> str:
    return cfg["outputs"]["run_dir"]


def _d152_development_layouts(path: Path) -> set[str]:
    with np.load(path, allow_pickle=False) as data:
        layouts = data["layouts"].astype(str).tolist()
        _, _, train_layouts, validation_layouts = train_validation_indices(layouts, TRAIN_SPLIT)
    return set(train_layouts) | set(validation_layouts)


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D157Error("commit D157 protocol/source/tests and vault updates before freezing lock")
    if cfg.get("schema") != "phase15_d157_fresh_layout_confirmation_v1":
        raise D157Error("D157 config schema mismatch")
    scope = cfg["scope"]
    if any(value is not True for key, value in scope.items()
           if key in {"fresh_layout_development_confirmation", "simulator_calls_for_locked_teacher_targets",
                      "exact_planner_calls_for_locked_teacher_continuations", "new_teacher_labeled_states",
                      "new_episodes"}):
        raise D157Error("D157 required scope authorization is disabled")
    if any(value is not False for key, value in scope.items()
           if key not in {"fresh_layout_development_confirmation", "simulator_calls_for_locked_teacher_targets",
                          "exact_planner_calls_for_locked_teacher_continuations", "new_teacher_labeled_states",
                          "new_episodes"}):
        raise D157Error("D157 enables a forbidden operation")
    parent_status = _audit_parents(cfg)
    for relative in (cfg["outputs"]["lock"], cfg["outputs"]["run_dir"], cfg["outputs"]["report"]):
        if (ROOT / relative).exists():
            raise D157Error(f"refusing to overwrite D157 output: {relative}")
    audit = candidate_layout_audit(cfg)
    if audit["layout_count"] != 100 or audit["unique_layout_hashes"] != 100:
        raise D157Error("D157 fresh block is incomplete")
    return {"schema": "phase15_d157_fresh_confirmation_lock_v1",
            "status": "locked_one_shot_frozen_predictor_confirmation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": _git("rev-parse", "HEAD"),
            "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
            "config_sha256": sha256_file(ROOT / CONFIG), "parent_sha256": _hash_parents(cfg),
            "parent_status": parent_status, "layout_audit": audit,
            "collector": cfg["state_collection"], "predictors": cfg["predictors"],
            "scoring": cfg["scoring"], "statistics": cfg["statistics"],
            "scope": cfg["scope"], "outputs": cfg["outputs"],
            "reserve_generated_or_accessed": False,
            "training_or_validation_labels_used_for_selection": False}


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    lock = _read_json(ROOT / cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d157_fresh_confirmation_lock_v1":
        raise D157Error("D157 lock schema mismatch")
    if (lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("source_sha256") != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
            or lock.get("parent_sha256") != _hash_parents(cfg)
            or lock.get("scope") != cfg["scope"]
            or lock.get("reserve_generated_or_accessed") is not False):
        raise D157Error("D157 source/config/parent/scope hash mismatch")
    if _git("status", "--porcelain"):
        raise D157Error("D157 run requires a clean committed tree")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    _audit_parents(cfg)
    current = candidate_layout_audit(cfg)
    if current != lock["layout_audit"]:
        raise D157Error("D157 candidate layout or historical-hash audit changed after lock")
    for relative in (cfg["outputs"]["run_dir"], cfg["outputs"]["report"]):
        if (ROOT / relative).exists():
            raise D157Error(f"refusing a second D157 execution: {relative}")
    return cfg, lock


def _collect_dataset(cfg: dict[str, Any], lock: dict[str, Any], output: Path) -> dict[str, Any]:
    p = cfg["parents"]
    policy_payload = _read_json(ROOT / p["phase14f_cart"])
    policy = FrozenCartPolicy(policy_payload)
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    layouts: list[str] = []
    seeds: list[int] = []
    decision_ids: list[int] = []
    collector_actions: list[int] = []
    episodes: list[dict[str, Any]] = []
    teacher_planner_queries = 0
    simulator_transitions = 0
    layout_block = lock["layout_audit"]["layouts"]
    for item in layout_block:
        seed, expected_hash = int(item["seed"]), str(item["layout_sha256"])
        scenario = generate_staggered_corridor(seed)
        if scenario.layout_sha256 != expected_hash:
            raise D157Error("D157 generated layout hash differs from locked hash-only audit")
        state, decision = scenario.initial_state, 0
        while state.status == "active":
            raw = state_features(scenario, state).astype(np.float64)
            if raw.shape != (52,) or not np.isfinite(raw).all():
                raise D157Error("D157 collector state violates raw deployable 52-D contract")
            action = int(policy.action_for_features(raw))
            if not 0 <= action < ACTION_COUNT:
                raise D157Error("D157 frozen collector returned an invalid primitive action")
            branches = forced_first_action_values(scenario, state, decision_index=decision,
                                                 expected_student_features=raw)
            if [int(branch["forced_action_id"]) for branch in branches] != list(range(ACTION_COUNT)):
                raise D157Error("D157 forced-action teacher order changed")
            summary = _summary_from_branches(branches, scenario.spec.max_steps,
                                              float(np.hypot(scenario.spec.width, scenario.spec.height)))
            teacher_planner_queries += sum(int(branch["continuation_decision_count"]) for branch in branches)
            simulator_transitions += sum(1 + int(branch["continuation_decision_count"]) for branch in branches)
            r0 = build_feature_matrix(raw[None, :], "raw52_plus_physics9")[0].astype(np.float64)
            rows_x.append(r0)
            rows_y.append(summary)
            layouts.append(expected_hash)
            seeds.append(seed)
            decision_ids.append(decision)
            collector_actions.append(action)
            state = corridor_step(state, action, scenario.spec).state
            simulator_transitions += 1
            decision += 1
            if decision > scenario.spec.max_steps:
                raise D157Error("D157 collector exceeded the task's frozen episode horizon")
        if state.status not in {"success", "collision", "timeout"}:
            raise D157Error("D157 collector episode has an invalid terminal state")
        episodes.append({"seed": seed, "layout_sha256": expected_hash,
                         "decision_state_count": decision, "collector_terminal_status": state.status,
                         "collector_terminal_steps": int(state.elapsed_steps)})
    if len(episodes) != int(cfg["layout_block"]["layout_count"]):
        raise D157Error("D157 collector did not complete the locked layout block")
    x = np.stack(rows_x).astype(np.float32)
    y = np.stack(rows_y).astype(np.float32)
    outcomes = y[:, :, :4]
    if (x.ndim != 2 or x.shape[1] != 61 or y.shape != (len(x), 9, 7)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.all((outcomes == 0.0) | (outcomes == 1.0))
            or not np.all(outcomes.sum(axis=2) == 1.0)
            or np.any(outcomes[:, :, 1] != 0.0)):
        raise D157Error("D157 collected arrays violate the finite R0/R4 terminal-outcome contract")
    array_path = output / "fresh_confirmation_rows.npz"
    np.savez_compressed(array_path, x_R0=x, summary_R4=y,
                        layouts=np.asarray(layouts, dtype="U64"),
                        seeds=np.asarray(seeds, dtype=np.int64),
                        decision_index=np.asarray(decision_ids, dtype=np.int64),
                        collector_action_ids_audit_only=np.asarray(collector_actions, dtype=np.int8))
    accounting_path = output / "episode_accounting.json"
    accounting_path.write_text(json.dumps(episodes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"dataset_path": array_path, "dataset_sha256": sha256_file(array_path),
            "dataset_bytes": array_path.stat().st_size, "accounting_path": accounting_path,
            "accounting_sha256": sha256_file(accounting_path), "state_rows": len(x),
            "teacher_action_branches": len(x) * ACTION_COUNT, "episodes": episodes,
            "teacher_planner_queries": teacher_planner_queries,
            "simulator_transitions": simulator_transitions,
            "x_R0": x.astype(np.float64), "summary_R4": y.astype(np.float64),
            "layouts": np.asarray(layouts, dtype=str), "seeds": np.asarray(seeds, dtype=np.int64)}


def _stage_c_config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs/phase15_d155_stage_c.yaml").read_text(encoding="utf-8"))


def _decision_record(scores: np.ndarray, target: np.ndarray, layouts: np.ndarray,
                     metrics_cfg: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    details = decision_detail_metrics(target, scores, layouts, metrics_cfg)
    selected_regret = np.asarray(details.pop("selected_regret"), dtype=np.float64)
    details.pop("per_layout_mean_regret", None)
    catastrophic = (selected_regret >= float(metrics_cfg["catastrophic_regret_threshold"])).astype(np.float64)
    return details, selected_regret, catastrophic


def _paired_layout_bootstrap(left: np.ndarray, right: np.ndarray, layouts: np.ndarray,
                             *, statistic: str, resamples: int, seed: int) -> dict[str, float]:
    """CI for left-minus-right improvement, resampling whole layouts in pairs."""
    x, y, group = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64), np.asarray(layouts).astype(str)
    if x.shape != y.shape or x.shape != group.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("D157 paired bootstrap arrays must be aligned finite vectors")
    names = np.asarray(sorted(set(group.tolist())))
    if len(names) < 2 or resamples < 1:
        raise ValueError("D157 bootstrap needs multiple layouts and positive resamples")
    row_idx = {name: np.flatnonzero(group == name) for name in names}
    if statistic == "layout_macro":
        per_layout = np.asarray([float(np.mean(x[row_idx[name]] - y[row_idx[name]])) for name in names])
        point = float(np.mean(per_layout))
        rng = np.random.default_rng(seed)
        draws = np.mean(per_layout[rng.integers(0, len(names), size=(resamples, len(names)))], axis=1)
    elif statistic == "state_micro":
        x_pos = np.asarray([float(np.sum(x[row_idx[name]])) for name in names])
        y_pos = np.asarray([float(np.sum(y[row_idx[name]])) for name in names])
        counts = np.asarray([len(row_idx[name]) for name in names], dtype=np.float64)
        point = float((x_pos.sum() - y_pos.sum()) / counts.sum())
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(names), size=(resamples, len(names)))
        draws = ((x_pos[picks] - y_pos[picks]).sum(axis=1)
                 / counts[picks].sum(axis=1))
    else:
        raise ValueError("D157 bootstrap statistic must be layout_macro or state_micro")
    return {"point_improvement_left_minus_right": point,
            "ci95_lower": float(np.quantile(draws, 0.025)),
            "ci95_upper": float(np.quantile(draws, 0.975)),
            "layout_count": int(len(names)), "resamples": int(resamples), "seed": int(seed)}


def _predictor_output(predictor: Any, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    action_x = np.zeros((len(states) * ACTION_COUNT, 70), dtype=np.float64)
    action_x[:, :61] = np.repeat(states, ACTION_COUNT, axis=0)
    action_x[:, 61:] = np.tile(np.eye(ACTION_COUNT), (len(states), 1))
    probability, steps = predictor.predict(action_x)
    probabilities = np.asarray(probability, dtype=np.float64).reshape(-1, ACTION_COUNT, 3)
    probabilities = np.maximum(probabilities, 1e-12)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    steps = np.clip(np.asarray(steps, dtype=np.float64).reshape(-1, ACTION_COUNT), 0.0, 1.0)
    if not np.isfinite(probabilities).all() or not np.isfinite(steps).all():
        raise D157Error("D157 frozen predictor returned non-finite outputs")
    return probabilities, steps


def _training_support(cfg: dict[str, Any], r0_head: Any) -> tuple[np.ndarray, str]:
    with np.load(ROOT / cfg["parents"]["d152_features"], allow_pickle=False) as saved:
        layouts = saved["layouts"].astype(str).tolist()
        train_idx, _, train_layouts, _ = train_validation_indices(layouts, TRAIN_SPLIT)
        original = saved["x_R0"][train_idx].astype(np.float64)
    if len(original) != 571 or len(train_layouts) != 60:
        raise D157Error("D157 D152 training support does not match frozen 571/60 contract")
    with np.load(ROOT / cfg["parents"]["d153_train_corpus"], allow_pickle=False) as saved:
        fresh = saved["x_R0"].astype(np.float64)
        fresh_layouts = saved["layouts"].astype(str)
    if fresh.shape != (1365, 61) or len(set(fresh_layouts.tolist())) != 140:
        raise D157Error("D157 D153 training support does not match frozen 1365/140 contract")
    support = np.concatenate((original, fresh), axis=0)
    if support.shape != (1936, 61):
        raise D157Error("D157 frozen training support cardinality changed")
    digest = hashlib.sha256(np.ascontiguousarray(support).tobytes()).hexdigest().upper()
    return support, digest


def _nearest_training_distance(query: np.ndarray, train: np.ndarray,
                               mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    q, t = (np.asarray(query, dtype=np.float64) - mean) / std, (np.asarray(train, dtype=np.float64) - mean) / std
    result = np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), 32):
        delta = q[start:start + 32, None, :] - t[None, :, :]
        result[start:start + 32] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    return result


def _decision_features_by_case(states: np.ndarray, exact_summary: np.ndarray,
                               predictions: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    exact_outcomes = exact_summary[:, :, :4]
    exact_prob = exact_outcomes[:, :, [0, 2, 3]]
    exact_steps = exact_summary[:, :, 4]
    cases: dict[str, np.ndarray] = {
        "exact_exact": _decision_features(states, exact_prob, exact_steps),
    }
    for name, (pred_prob, pred_steps) in predictions.items():
        cases[f"{name}_predicted_exact"] = _decision_features(states, pred_prob, exact_steps)
        cases[f"{name}_exact_predicted"] = _decision_features(states, exact_prob, pred_steps)
        cases[f"{name}_predicted_predicted"] = _decision_features(states, pred_prob, pred_steps)
    return cases


def _component_prediction_metrics(summary: np.ndarray, layouts: np.ndarray,
                                  probabilities: np.ndarray, steps: np.ndarray) -> dict[str, Any]:
    target_summary = np.asarray(summary, dtype=np.float64).reshape(-1, ACTION_COUNT, 7)
    true_class = np.argmax(target_summary[:, :, [0, 2, 3]], axis=2).reshape(-1)
    flat_probability = probabilities.reshape(-1, 3)
    weights = _weights_per_action(layouts)
    outcome = _outcome_metrics(true_class, flat_probability, weights)
    true_steps = target_summary[:, :, 4].reshape(-1)
    pred_steps = steps.reshape(-1)
    steps_metric = {"rmse": float(np.sqrt(np.sum(weights * (pred_steps - true_steps) ** 2))),
                    "mae": float(np.sum(weights * np.abs(pred_steps - true_steps)))}
    return {"outcome": outcome, "steps": steps_metric}


def _strip_metrics(details: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in details.items() if key != "selected_regret"}


def run_d157() -> dict[str, Any]:
    cfg, lock = validate_lock()
    start_time = time.perf_counter()
    out = ROOT / cfg["outputs"]["run_dir"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        collected = _collect_dataset(cfg, lock, out)
        states = collected["x_R0"]
        summary = collected["summary_R4"]
        layouts = collected["layouts"]
        target = regrets_from_exact_summaries(summary)
        if (target.shape != (len(states), ACTION_COUNT) or len(set(layouts.tolist())) != 100
                or not np.isfinite(target).all()):
            raise D157Error("D157 regret target/layout contract failed")
        d156_cfg = yaml.safe_load((ROOT / cfg["parents"]["d156_config"]).read_text(encoding="utf-8"))
        predictors = load_predictors(d156_cfg)
        d155_metrics_cfg = _stage_c_config()["metrics"]
        stage_a_head = restore_decision_head(ROOT / cfg["parents"]["stage_a_head"])
        r0_head = restore_decision_head(ROOT / cfg["parents"]["d150_r0_head"])
        if (stage_a_head.x_mean.shape != (106,) or r0_head.x_mean.shape != (61,)
                or set(predictors) != {"per_action_constant", "bounded_cart", "action_conditioned_mlp"}):
            raise D157Error("D157 frozen model input/checkpoint contract failed")

        predictions = {name: _predictor_output(model, states)
                       for name, model in predictors.items()}
        feature_cases = _decision_features_by_case(states, summary, predictions)
        decision_vectors: dict[str, dict[str, Any]] = {}
        for case, feature in feature_cases.items():
            logits = stage_a_head.predict_logits(feature)
            details, regret, catastrophe = _decision_record(-logits, target, layouts, d155_metrics_cfg)
            decision_vectors[case] = {"metrics": _strip_metrics(details),
                                      "_regret": regret, "_catastrophic": catastrophe}
        direct_scores = r0_head.predict(states)
        direct_details, direct_regret, direct_cat = _decision_record(
            direct_scores, target, layouts, d155_metrics_cfg)
        decision_vectors["direct_D150_R0"] = {"metrics": _strip_metrics(direct_details),
                                               "_regret": direct_regret,
                                               "_catastrophic": direct_cat}

        stats_cfg = cfg["statistics"]
        n_boot = int(stats_cfg["bootstrap_resamples"])
        seed = int(stats_cfg["bootstrap_seed"])
        comparisons: dict[str, Any] = {}
        oracle = decision_vectors["exact_exact"]
        direct = decision_vectors["direct_D150_R0"]
        comparisons["exact_exact_advantage_over_direct_R0"] = {
            "regret": _paired_layout_bootstrap(direct["_regret"], oracle["_regret"], layouts,
                                                statistic="layout_macro", resamples=n_boot, seed=seed),
            "catastrophic_fraction": _paired_layout_bootstrap(direct["_catastrophic"], oracle["_catastrophic"], layouts,
                                                               statistic="state_micro", resamples=n_boot, seed=seed + 1),
        }
        cart_both = decision_vectors["bounded_cart_predicted_predicted"]
        comparisons["bounded_cart_predicted_both_advantage_over_direct_R0"] = {
            "regret": _paired_layout_bootstrap(direct["_regret"], cart_both["_regret"], layouts,
                                                statistic="layout_macro", resamples=n_boot, seed=seed + 2),
            "catastrophic_fraction": _paired_layout_bootstrap(direct["_catastrophic"], cart_both["_catastrophic"], layouts,
                                                               statistic="state_micro", resamples=n_boot, seed=seed + 3),
        }
        cart_outcome = decision_vectors["bounded_cart_predicted_exact"]
        cart_steps = decision_vectors["bounded_cart_exact_predicted"]
        comparisons["bounded_cart_component_swap_penalties_vs_exact"] = {
            "outcome_swap_regret": _paired_layout_bootstrap(cart_outcome["_regret"], oracle["_regret"], layouts,
                                                             statistic="layout_macro", resamples=n_boot, seed=seed + 4),
            "steps_swap_regret": _paired_layout_bootstrap(cart_steps["_regret"], oracle["_regret"], layouts,
                                                           statistic="layout_macro", resamples=n_boot, seed=seed + 5),
            "outcome_minus_steps_regret_penalty": _paired_layout_bootstrap(cart_outcome["_regret"], cart_steps["_regret"], layouts,
                                                                            statistic="layout_macro", resamples=n_boot, seed=seed + 6),
            "outcome_swap_catastrophe": _paired_layout_bootstrap(cart_outcome["_catastrophic"], oracle["_catastrophic"], layouts,
                                                                  statistic="state_micro", resamples=n_boot, seed=seed + 7),
            "steps_swap_catastrophe": _paired_layout_bootstrap(cart_steps["_catastrophic"], oracle["_catastrophic"], layouts,
                                                                statistic="state_micro", resamples=n_boot, seed=seed + 8),
            "outcome_minus_steps_catastrophe_penalty": _paired_layout_bootstrap(cart_outcome["_catastrophic"], cart_steps["_catastrophic"], layouts,
                                                                                statistic="state_micro", resamples=n_boot, seed=seed + 9),
        }

        support, support_hash = _training_support(cfg, r0_head)
        distances = _nearest_training_distance(states, support, r0_head.x_mean, r0_head.x_std)
        component_metrics = {name: _component_prediction_metrics(summary, layouts, *predictions[name])
                             for name in predictions}
        public_decisions = {name: value["metrics"] for name, value in decision_vectors.items()}
        oracle_gate = comparisons["exact_exact_advantage_over_direct_R0"]
        predictor_gate = comparisons["bounded_cart_predicted_both_advantage_over_direct_R0"]
        oracle_pass = (oracle_gate["regret"]["ci95_lower"] > 0.0
                       and oracle_gate["catastrophic_fraction"]["ci95_lower"] > 0.0)
        cart_pass = (predictor_gate["regret"]["point_improvement_left_minus_right"] >= 0.05
                     and predictor_gate["catastrophic_fraction"]["point_improvement_left_minus_right"] >= 0.05
                     and predictor_gate["regret"]["ci95_lower"] > 0.0
                     and predictor_gate["catastrophic_fraction"]["ci95_lower"] > 0.0)
        dominance = comparisons["bounded_cart_component_swap_penalties_vs_exact"]
        outcome_dominant = (dominance["outcome_minus_steps_regret_penalty"]["ci95_lower"] > 0.0
                            and dominance["outcome_minus_steps_catastrophe_penalty"]["ci95_lower"] > 0.0)
        steps_dominant = (dominance["outcome_minus_steps_regret_penalty"]["ci95_upper"] < 0.0
                          and dominance["outcome_minus_steps_catastrophe_penalty"]["ci95_upper"] < 0.0)
        result = {
            "schema": "phase15_d157_fresh_confirmation_result_v1",
            "status": "completed_fresh_layout_frozen_predictor_confirmation",
            "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
            "parent_sha256": lock["parent_sha256"], "layout_audit": lock["layout_audit"],
            "collection": {"layout_count": len(collected["episodes"]), "state_rows": len(states),
                           "forced_action_teacher_branches": len(states) * ACTION_COUNT,
                           "teacher_planner_queries": collected["teacher_planner_queries"],
                           "simulator_transitions": collected["simulator_transitions"],
                           "collector_terminal_outcomes": _count_map([row["collector_terminal_status"] for row in collected["episodes"]]),
            "branch_terminal_outcomes": _terminal_outcome_counts(summary),
                           "dataset_sha256": collected["dataset_sha256"], "dataset_bytes": collected["dataset_bytes"],
                           "accounting_sha256": collected["accounting_sha256"]},
            "scope": {"fresh_development_layouts": 100, "test_rows_dereferenced": 0,
                      "reserve_generated_or_accessed": False, "model_fitting": False,
                      "hyperparameter_tuning": False, "predictor_or_head_updates": False,
                      "student_closed_loop": False, "dagger": False, "world_model": False,
                      "jepa": False, "other_environment": False},
            "support": {"train_state_count": len(support), "train_state_sha256": support_hash,
                        "distance_normalization": "frozen_D150_R0_head_x_mean_x_std",
                        "nearest_train_distance_median": float(np.median(distances)),
                        "nearest_train_distance_p90": float(np.quantile(distances, 0.90)),
                        "nearest_train_distance_p99": float(np.quantile(distances, 0.99))},
            "decision_metrics": public_decisions, "component_prediction_metrics": component_metrics,
            "paired_layout_bootstrap": comparisons,
            "gates": {"oracle_advantage_replicates": bool(oracle_pass),
                      "bounded_cart_predicted_both_beats_R0_materially": bool(cart_pass),
                      "dominant_component": "outcome" if outcome_dominant else "steps" if steps_dominant else "mixed_or_uncertain"},
            "interpretation": _interpret(oracle_pass, cart_pass),
            "timing_seconds": time.perf_counter() - start_time,
            "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
            "reserve_access": False,
        }
        result_path = out / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_path = ROOT / cfg["outputs"]["report"]
        report_path.write_text(render_report(result), encoding="utf-8")
        manifest = {
            "schema": "phase15_d157_fresh_confirmation_manifest_v1", "status": result["status"],
            "source_commit": result["source_commit"], "lock_sha256": result["lock_sha256"],
            "parent_sha256": result["parent_sha256"], "layout_hashes_sha256": _hash_json(lock["layout_audit"]["layouts"]),
            "dataset_sha256": collected["dataset_sha256"], "accounting_sha256": collected["accounting_sha256"],
            "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
            "test_rows_dereferenced": 0, "reserve_access": False, "model_fitting": False,
            "simulator_transitions": collected["simulator_transitions"],
            "planner_queries": collected["teacher_planner_queries"],
            "new_transitions": collected["simulator_transitions"],
            "new_episodes": len(collected["episodes"]), "student_closed_loop": False,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        abort = {"schema": "phase15_d157_abort_v1", "status": "aborted_or_incomplete",
                 "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
                 "error_type": type(exc).__name__, "error": str(exc),
                 "reserve_access": False, "test_rows_dereferenced": 0,
                 "note": "Preserve partial output; this one-shot lock cannot be resumed."}
        (out / "abort.json").write_text(json.dumps(abort, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def _count_map(values: list[Any]) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    return {str(name): int(count) for name, count in zip(unique.tolist(), counts.tolist(), strict=True)}


def _terminal_outcome_counts(summary: np.ndarray) -> dict[str, int]:
    class_names = ("success", "active_zero", "timeout", "collision")
    indices = np.argmax(np.asarray(summary)[:, :, :4], axis=2).reshape(-1)
    return {name: int(np.sum(indices == index)) for index, name in enumerate(class_names)}


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _interpret(oracle_pass: bool, cart_pass: bool) -> str:
    if not oracle_pass:
        return "exact oracle advantage did not meet the fresh-layout confirmation gate; stop the current oracle-headroom interpretation"
    if not cart_pass:
        return "oracle advantage replicated but the frozen primary CART consequence predictor did not meet the material transfer gate"
    return "oracle advantage and frozen primary CART consequence transfer both passed; stop and require a separate closed-loop protocol"


def render_report(result: dict[str, Any]) -> str:
    lines = ["# D157 — Fresh-layout confirmation of frozen consequence predictors", "",
             "**One-shot development confirmation.** Predictors and heads were frozen; reserve/test remain sealed. Results apply only to the locked corridor layout family and the frozen-CART state-collection distribution.", "",
             f"Block: `{result['layout_audit']['seed_start']}`–`{result['layout_audit']['seed_start'] + result['layout_audit']['layout_count'] - 1}` ({result['collection']['layout_count']} layouts), {result['collection']['state_rows']} states and {result['collection']['forced_action_teacher_branches']} exact forced-action continuation branches.", "",
             "## Decision results", "",
             "| Method | Macro regret | Catastrophic | Epsilon-optimal | Best-action set | High-margin pairs |", "|---|---:|---:|---:|---:|---:|"]
    order = ["direct_D150_R0", "exact_exact"]
    for predictor in ("per_action_constant", "bounded_cart", "action_conditioned_mlp"):
        order.extend((f"{predictor}_predicted_exact", f"{predictor}_exact_predicted", f"{predictor}_predicted_predicted"))
    for name in order:
        metric = result["decision_metrics"][name]
        lines.append(f"| {name} | {metric['layout_macro_normalized_selected_regret']:.5f} | {metric['catastrophic_fraction_state_micro']:.4f} | {metric['epsilon_optimal_rate']:.4f} | {metric['best_action_set_agreement']:.4f} | {metric['high_margin_pairwise_accuracy_macro']:.4f} |")
    lines.extend(["", "## Paired layout-bootstrap comparisons", "",
                  "Positive values mean the left/reference method has lower regret or fewer catastrophic decisions than the right/comparator.", "",
                  "| Comparison | Metric | Improvement | Paired 95% CI |", "|---|---|---:|---:|"])
    for comparison, metrics in result["paired_layout_bootstrap"].items():
        for metric, row in metrics.items():
            lines.append(f"| {comparison} | {metric} | {row['point_improvement_left_minus_right']:.5f} | [{row['ci95_lower']:.5f}, {row['ci95_upper']:.5f}] |")
    lines.extend(["", "## Component prediction", "",
                  "| Predictor | Outcome accuracy | Outcome CE | Brier | Collision recall | Timeout recall | Steps RMSE | Steps MAE |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, record in result["component_prediction_metrics"].items():
        per_class = record["outcome"]["per_class"]
        def recall(key: str) -> str:
            value = per_class[key]["recall"]
            return "n/a" if value is None else f"{value:.4f}"

        lines.append(f"| {name} | {record['outcome']['accuracy']:.4f} | {record['outcome']['cross_entropy']:.4f} | {record['outcome']['brier_score']:.4f} | {recall('collision')} | {recall('timeout')} | {record['steps']['rmse']:.5f} | {record['steps']['mae']:.5f} |")
    lines.extend(["", "## Gates and interpretation", "",
                  f"Oracle advantage replicates: **{result['gates']['oracle_advantage_replicates']}**. Frozen primary CART materially beats direct R0: **{result['gates']['bounded_cart_predicted_both_beats_R0_materially']}**. Component assessment: **{result['gates']['dominant_component']}**.", "",
                  result["interpretation"], "",
                  f"Training-support nearest R0 distance (D150 frozen normalization): median `{result['support']['nearest_train_distance_median']:.4f}`, p90 `{result['support']['nearest_train_distance_p90']:.4f}`, p99 `{result['support']['nearest_train_distance_p99']:.4f}`. This is descriptive only.", "",
                  "No predictor/head was fit or changed. The 100 fresh layouts were used once; no test/reserve data, student control, DAgger, world model, JEPA, or other environment was involved. D157 closes here; subsequent work needs a separate protocol and decision.", ""])
    return "\n".join(lines)
