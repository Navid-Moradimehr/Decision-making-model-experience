"""D162 frozen relational-model evaluation on fresh collector states."""
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
from consequence_lab.phase15_d155_stage_c import CARTOutcomeSteps, _weights_per_action, encode_targets, action_conditioned_inputs
from consequence_lab.phase15_d157 import ACTION_COUNT, _paired_layout_bootstrap
from consequence_lab.phase15_d158 import (
    RelationalCandidateModel, _action_inputs, _candidate_metrics, _feature_set, _probabilities_from_exact,
)
from consequence_lab.phase15_decision_aligned import regrets_from_exact_summaries

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d162_fresh_layout_relational.yaml"
AUDIT = "artifacts/runs/phase15_d162_layout_audit.json"
PARAMS = ("ws", "bs", "wc", "bc", "wctx", "bctx", "wscore", "bscore", "wrisk", "brisk")
CHECKPOINTS = {
    "r0_action_only": "artifacts/runs/phase15_d161_fit_boundary_v1/width_64/r0_action_only_best_training_objective.npz",
    "r0_action_predicted_outcome_steps": "artifacts/runs/phase15_d161_fit_boundary_v1/width_64/r0_action_predicted_outcome_steps_best_training_objective.npz",
    "r0_action_exact_outcome_steps_oracle": "artifacts/runs/phase15_d161_fit_boundary_v1/width_64/r0_action_exact_outcome_steps_oracle_best_training_objective.npz",
}
SOURCE_FILES = (
    CONFIG, "reports/phase15_d162_fresh_layout_relational_protocol.md",
    "reports/phase15_d162_prelock_scope_abort.md",
    "src/consequence_lab/phase15_d162.py", "scripts/audit_phase15_d162_layouts.py",
    "scripts/freeze_phase15_d162_lock.py", "scripts/run_phase15_d162.py", "tests/test_phase15_d162.py", "uv.lock",
    "src/consequence_lab/corridor_tasks.py", "src/consequence_lab/inertial_corridor.py",
    "src/consequence_lab/phase12_teacher.py", "src/consequence_lab/phase14.py",
    "src/consequence_lab/phase14c.py", "src/consequence_lab/phase14f.py", "src/consequence_lab/phase14g.py",
    "src/consequence_lab/phase15_d153_scaling.py", "src/consequence_lab/phase15_d155_stage_c.py",
    "src/consequence_lab/phase15_d157.py", "src/consequence_lab/phase15_d158.py",
    "src/consequence_lab/phase15_decision_aligned.py", "src/consequence_lab/phase15_learning.py",
    "experiments/phase15_d161_fit_boundary.lock.json",
    "artifacts/runs/phase15_d161_fit_boundary_v1/manifest.json",
    "artifacts/runs/phase15_d161_fit_boundary_v1/result.json",
    "reports/phase15_d161_result.md", "configs/phase15_d155_stage_c.yaml",
    "configs/phase15_d153_coverage_scaling_v3.yaml",
    "experiments/phase15_d153_coverage_scaling_v3.lock.json",
    "artifacts/runs/phase15_d153_coverage_scaling_v3/manifest.json",
    "artifacts/runs/phase15_d153_coverage_scaling_v1/fresh_training_rows.npz",
    "experiments/phase14f_student_cart_v1.json", "experiments/phase14f_student_cart_v1.manifest.json",
    "second-brain/00_Project_Home.md", "second-brain/14_Decisions.md",
    "second-brain/16_Experiment_Log.md",
    "second-brain/phases/Phase_15_Decision_Aligned_Target_Study.md",
)


class D162Error(RuntimeError):
    """D162 provenance or execution failure."""


def validate_scope(scope: dict[str, Any]) -> bool:
    """Allow only D162's declared predictor-training and fresh-scoring operations."""
    permitted_true = {"training_rows_only_for_predictor_reconstruction",
                      "fresh_layout_development_scoring"}
    return {key for key, value in scope.items() if value is True} == permitted_true


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D162Error(f"git {' '.join(args)} failed") from exc


def _parents(cfg: dict[str, Any]) -> dict[str, str]:
    return {k: sha256_file(ROOT / v) for k, v in cfg["parents"].items()}


def candidate_layout_audit(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hash-only identity audit; does not evaluate transitions or outcomes."""
    cfg = cfg or _cfg()
    block = cfg["candidate_layout_block"]
    start, count = int(block["seed_start"]), int(block["layout_count"])
    stop = start + count
    if start < 0 or count <= 0:
        raise D162Error("invalid candidate seed interval")
    for row in block["previously_consumed_or_excluded_ranges"]:
        lo, hi = int(row["start"]), int(row["start"]) + int(row["count"])
        if start < hi and lo < stop:
            raise D162Error(f"candidate interval overlaps {row['label']}")
    layouts = [{"seed": seed, "layout_sha256": generate_staggered_corridor(seed).layout_sha256}
               for seed in range(start, stop)]
    hashes = [r["layout_sha256"] for r in layouts]
    if len(set(hashes)) != count:
        raise D162Error("candidate layout hashes are not unique")
    prior, spent, sources, source_hashes = _collect_prior_layout_hashes(ROOT, ROOT / AUDIT)
    overlap = set(hashes) & prior
    if overlap or set(hashes) & spent:
        raise D162Error(f"candidate hashes overlap historical layout artifacts: {sorted(overlap)}")
    reserve = next((r for r in block["previously_consumed_or_excluded_ranges"]
                    if r["label"] == "sealed reserve"), None)
    if reserve is None:
        raise D162Error("sealed reserve range is missing")
    return {"schema": "phase15_d162_hash_only_layout_audit_v1",
            "seed_start": start, "layout_count": count, "layouts": layouts,
            "unique_layout_hashes": len(set(hashes)), "historical_layout_hashes": len(prior),
            "historical_outcome_layout_hashes": len(spent), "historical_source_count": len(sources),
            "historical_source_sha256": source_hashes, "overlap_with_historical": 0,
            "overlap_with_outcome_bearing": 0, "reserve_range_checked_not_generated": reserve,
            "outcomes_or_transitions_computed": False, "reserve_outcomes_read": False}


def write_layout_audit() -> dict[str, Any]:
    path = ROOT / AUDIT
    if path.exists():
        raise D162Error("refusing to overwrite D162 layout audit")
    audit = candidate_layout_audit()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _validate_parents(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["parents"]
    d161_lock, d161_result, d161_manifest = (_json(ROOT / p[k]) for k in
                                               ("d161_lock", "d161_result", "d161_manifest"))
    if (d161_lock.get("schema") != "phase15_d161_fit_boundary_lock_v1"
            or d161_result.get("status") != "completed_training_only_fit_boundary"
            or d161_manifest.get("status") != d161_result.get("status")
            or d161_manifest.get("lock_sha256") != sha256_file(ROOT / p["d161_lock"])
            or d161_manifest.get("result_sha256") != sha256_file(ROOT / p["d161_result"])
            or d161_manifest.get("reserve_access") is not False
            or d161_manifest.get("validation_rows_read") != 0
            or d161_manifest.get("test_rows_dereferenced") != 0
            or d161_manifest.get("simulator_calls") != 0
            or d161_manifest.get("planner_calls") != 0
            or d161_manifest.get("closed_loop") is not False):
        raise D162Error("D161 status/hash/scope audit failed")
    if d161_lock.get("row_count") != 1365 or d161_lock.get("layout_count") != 140:
        raise D162Error("D161 training lineage mismatch")
    checkpoint_hashes = {}
    for name, relative in CHECKPOINTS.items():
        digest = sha256_file(ROOT / relative)
        record = d161_result["models"].get(f"width_64/{name}", {})
        manifest_record = d161_manifest.get("checkpoints", {}).get(f"width_64/{name}", {})
        if record.get("checkpoint_sha256") != digest or manifest_record.get("best_sha256") != digest:
            raise D162Error(f"D161 checkpoint hash mismatch: {name}")
        checkpoint_hashes[name] = digest
    d153_lock = _json(ROOT / p["d153_lock"])
    d153_manifest = _json(ROOT / "artifacts/runs/phase15_d153_coverage_scaling_v3/manifest.json")
    rows_path = ROOT / p["d153_training_rows"]
    if (d153_lock.get("schema") != "phase15_d153_coverage_scaling_lock_v3"
            or d153_lock.get("reused_corpus", {}).get("sha256") != sha256_file(rows_path)
            or d153_manifest.get("fresh_dataset_sha256") != sha256_file(rows_path)
            or d153_manifest.get("reserve_access") is not False
            or d153_manifest.get("test_rows_indexed_or_scored") != 0):
        raise D162Error("D153 training corpus lineage mismatch")
    with np.load(rows_path, allow_pickle=False) as data:
        train = {"states": data["x_R0"].astype(np.float64),
                 "summary": data["summary_R4"].astype(np.float64),
                 "layouts": data["layouts"].astype(str)}
    if (train["states"].shape != (1365, 61) or train["summary"].shape != (1365, 9, 7)
            or len(set(train["layouts"].tolist())) != 140):
        raise D162Error("D153 training corpus shape mismatch")
    lineage = {"d161_lock_sha256": sha256_file(ROOT / p["d161_lock"]),
               "d161_result_sha256": sha256_file(ROOT / p["d161_result"]),
               "d161_manifest_sha256": sha256_file(ROOT / p["d161_manifest"]),
               "checkpoint_sha256": checkpoint_hashes,
               "d153_training_sha256": sha256_file(rows_path),
               "d153_training_rows": 1365, "d153_training_layouts": 140,
               "reserve_access": False, "test_rows_accessed": 0}
    return train | {"lineage": lineage}


def _fold_predictors(train: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Rebuild D158's same five-fold D153-only CART construction."""
    nfold = int(cfg["consequence_predictor"]["fold_count"])
    layout = train["layouts"]
    fold_id = np.array([int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) % nfold for x in layout])
    classes, steps = encode_targets(train["summary"])
    x = action_conditioned_inputs(train["states"])
    settings = yaml.safe_load((ROOT / cfg["parents"]["d155_stage_c_config"]).read_text(encoding="utf-8"))[
        "predictors"]["bounded_cart"]
    if settings != cfg["consequence_predictor"]["model_settings"]:
        raise D162Error("D162 CART settings differ from D155 frozen settings")
    models, records = [], []
    for fold in range(nfold):
        held = np.flatnonzero(fold_id == fold)
        fit = np.flatnonzero(fold_id != fold)
        mask = np.repeat(np.isin(np.arange(len(layout)), fit), ACTION_COUNT)
        model = CARTOutcomeSteps(settings).fit(x[mask], classes.reshape(-1)[mask],
                                                steps.reshape(-1)[mask], _weights_per_action(layout[fit]))
        models.append(model)
        records.append({"fold": fold, "fit_rows": int(len(fit)), "fit_layouts": int(len(set(layout[fit]))),
                        "heldout_rows": int(len(held)), "heldout_layouts": int(len(set(layout[held]))),
                        "node_count": int(model.node_count)})
    return models, records


def _tree_payload(node: Any) -> dict[str, Any]:
    value = np.asarray(node.value, dtype=np.float64).tolist()
    if node.feature < 0:
        return {"feature": -1, "value": value}
    return {"feature": int(node.feature), "threshold": float(node.threshold), "value": value,
            "left": _tree_payload(node.left), "right": _tree_payload(node.right)}


def _predictor_payload(model: Any, fold: int) -> dict[str, Any]:
    return {"schema": "phase15_d162_crossfit_cart_fold_v1", "fold": fold,
            "x_mean": model.x_mean.tolist(), "x_std": model.x_std.tolist(),
            "node_count": int(model.node_count),
            "outcome_tree": _tree_payload(model.class_tree.root),
            "steps_tree": _tree_payload(model.steps_tree.root)}


def _predictor_hashes(models: list[Any]) -> list[str]:
    result = []
    for fold, model in enumerate(models):
        payload = json.dumps(_predictor_payload(model, fold), sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        result.append(hashlib.sha256(payload).hexdigest().upper())
    return result


def _predict(model: Any, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs, steps = model.predict(action_conditioned_inputs(states))
    probs = np.asarray(probs, dtype=np.float64).reshape(len(states), 9, 3)
    probs = np.maximum(probs, 1e-12)
    probs /= probs.sum(axis=2, keepdims=True)
    steps = np.clip(np.asarray(steps, dtype=np.float64).reshape(len(states), 9), 0.0, 1.0)
    if not np.isfinite(probs).all() or not np.isfinite(steps).all():
        raise D162Error("CART predictor output is non-finite")
    return probs, steps


def _collect(cfg: dict[str, Any], lock: dict[str, Any], out: Path) -> dict[str, Any]:
    policy = FrozenCartPolicy(_json(ROOT / cfg["parents"]["collector_policy"]))
    xx, yy, layout_ids, seed_ids, decisions, actions_audit, episodes = [], [], [], [], [], [], []
    planner_queries = transitions = 0
    for item in lock["layout_audit"]["layouts"]:
        seed, layout_hash = int(item["seed"]), str(item["layout_sha256"])
        scenario = generate_staggered_corridor(seed)
        if scenario.layout_sha256 != layout_hash:
            raise D162Error("layout hash changed after hash-only audit")
        state, decision = scenario.initial_state, 0
        while state.status == "active":
            raw = state_features(scenario, state).astype(np.float64)
            if raw.shape != (52,) or not np.isfinite(raw).all():
                raise D162Error("invalid collector-observable state")
            action = int(policy.action_for_features(raw))
            if action not in range(9):
                raise D162Error("collector action ID outside legal range")
            branches = forced_first_action_values(scenario, state, decision_index=decision,
                                                   expected_student_features=raw)
            if [int(b["forced_action_id"]) for b in branches] != list(range(9)):
                raise D162Error("teacher branch ordering changed")
            summary = _summary_from_branches(branches, scenario.spec.max_steps,
                                               float(np.hypot(scenario.spec.width, scenario.spec.height)))
            planner_queries += sum(int(b["continuation_decision_count"]) for b in branches)
            transitions += sum(1 + int(b["continuation_decision_count"]) for b in branches)
            r0 = build_feature_matrix(raw[None], "raw52_plus_physics9")[0].astype(np.float64)
            xx.append(r0)
            yy.append(summary)
            layout_ids.append(layout_hash)
            seed_ids.append(seed)
            decisions.append(decision)
            actions_audit.append(action)
            state = corridor_step(state, action, scenario.spec).state
            transitions += 1
            decision += 1
            if decision > scenario.spec.max_steps:
                raise D162Error("collector exceeded task horizon")
        if state.status not in {"success", "collision", "timeout"}:
            raise D162Error("unsupported collector terminal state")
        episodes.append({"seed": seed, "layout_sha256": layout_hash, "decision_state_count": decision,
                         "collector_terminal_status": state.status, "collector_terminal_steps": state.elapsed_steps})
    x, y = np.stack(xx).astype(np.float32), np.stack(yy).astype(np.float32)
    outc = y[:, :, :4]
    if (x.shape[1:] != (61,) or y.shape != (len(x), 9, 7) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.all((outc == 0) | (outc == 1))
            or not np.all(outc.sum(axis=2) == 1) or np.any(outc[:, :, 1] != 0)):
        raise D162Error("D162 collected array contract failed")
    ds = out / "fresh_layout_rows.npz"
    np.savez_compressed(ds, x_R0=x, summary_R4=y, layouts=np.asarray(layout_ids, dtype="U64"),
                        seeds=np.asarray(seed_ids, dtype=np.int64), decision_index=np.asarray(decisions, dtype=np.int64),
                        collector_action_ids_audit_only=np.asarray(actions_audit, dtype=np.int8))
    ep = out / "episode_accounting.json"
    ep.write_text(json.dumps(episodes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"states": x.astype(np.float64), "summary": y.astype(np.float64),
            "layouts": np.asarray(layout_ids, dtype=str), "episodes": episodes,
            "dataset_path": ds, "dataset_sha256": sha256_file(ds),
            "accounting_path": ep, "accounting_sha256": sha256_file(ep),
            "planner_queries": planner_queries, "transitions": transitions}


def _load_student(name: str, path: Path, training_states: np.ndarray) -> RelationalCandidateModel:
    result = _json(ROOT / "artifacts/runs/phase15_d161_fit_boundary_v1/result.json")
    meta = result["models"][f"width_64/{name}"]["metadata"]
    model = RelationalCandidateModel(64, int(meta["seed"]), meta["learning_rate"], meta["l2"],
                                     meta["beta1"], meta["beta2"], meta["epsilon"])
    model.initialize(training_states)
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["x_mean"], model.x_mean) or not np.array_equal(saved["x_std"], model.x_std):
            raise D162Error("D161 state normalization differs from training corpus")
        for param in PARAMS:
            value = np.asarray(saved[f"param_{param}"], dtype=np.float64)
            if value.shape != getattr(model, param).shape or not np.isfinite(value).all():
                raise D162Error(f"invalid frozen model parameter {param}")
            getattr(model, param)[...] = value
    return model


def _score(cfg: dict[str, Any], data: dict[str, Any], train: dict[str, Any],
           predictors: list[Any]) -> dict[str, Any]:
    states, summary, layouts = data["states"], data["summary"], data["layouts"]
    regret = regrets_from_exact_summaries(summary)
    exact_prob, exact_steps = _probabilities_from_exact(summary)
    folds = np.array([int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) % len(predictors) for x in layouts])
    predicted_prob, predicted_steps = np.empty((len(states), 9, 3)), np.empty((len(states), 9))
    for fold, model in enumerate(predictors):
        ids = np.flatnonzero(folds == fold)
        if len(ids):
            predicted_prob[ids], predicted_steps[ids] = _predict(model, states[ids])
    reps = {
        "r0_action_only": _feature_set(predicted_prob, predicted_steps, "r0_action_only"),
        "r0_action_predicted_outcome_steps": _feature_set(predicted_prob, predicted_steps, "r0_action_predicted_outcome_steps"),
        "r0_action_exact_outcome_steps_oracle": _feature_set(exact_prob, exact_steps, "r0_action_exact_outcome_steps_oracle"),
    }
    metric_cfg = yaml.safe_load((ROOT / cfg["parents"]["d155_stage_c_config"]).read_text(encoding="utf-8"))["metrics"]
    scores, metrics, runtime = {}, {}, {}
    for name, ckpt in CHECKPOINTS.items():
        model = _load_student(name, ROOT / ckpt, train["states"])
        consequence, mask = reps[name]
        actions, _ = _action_inputs(states)
        utility, _ = model.predict(states, actions, consequence, mask)
        scores[name] = -utility
        metrics[name] = _candidate_metrics(regret, scores[name], summary, layouts, metric_cfg)
        millis = []
        for i in range(len(states)):
            tic = time.perf_counter_ns()
            if name == "r0_action_predicted_outcome_steps":
                p, s = _predict(predictors[int(folds[i])], states[i:i+1])
                c, m = _feature_set(p, s, name)
            elif name == "r0_action_only":
                c, m = _feature_set(predicted_prob[i:i+1], predicted_steps[i:i+1], name)
            else:
                c, m = _feature_set(exact_prob[i:i+1], exact_steps[i:i+1], name)
            model.predict(states[i:i+1], actions[i:i+1], c, m)
            millis.append((time.perf_counter_ns() - tic) / 1e6)
        runtime[name] = {"median_decision_latency_ms": float(np.median(millis)),
                         "p95_decision_latency_ms": float(np.quantile(millis, .95)),
                         "parameter_count": int(model.parameter_count)}
    selected_regret, catastrophic = {}, {}
    for name, cost in scores.items():
        chosen = np.argmin(cost, axis=1)
        selected_regret[name] = regret[np.arange(len(regret)), chosen]
        catastrophic[name] = (selected_regret[name] >= cfg["evaluation"]["thresholds"]["catastrophic_regret"]).astype(float)
    boot = {}
    base = "r0_action_only"
    for name in list(CHECKPOINTS)[1:]:
        boot[name] = {
            "regret_improvement_R0_minus_model": _paired_layout_bootstrap(
                selected_regret[base], selected_regret[name], layouts, statistic="layout_macro",
                resamples=cfg["evaluation"]["statistics"]["resamples"], seed=cfg["evaluation"]["statistics"]["seed"]),
            "catastrophic_improvement_R0_minus_model": _paired_layout_bootstrap(
                catastrophic[base], catastrophic[name], layouts, statistic="layout_macro",
                resamples=cfg["evaluation"]["statistics"]["resamples"], seed=cfg["evaluation"]["statistics"]["seed"]),
        }
    return {"model_metrics": metrics, "runtime": runtime, "paired_layout_bootstrap": boot}


def _checkpoint_equivariance_audit(train: dict[str, Any], predictors: list[Any]) -> dict[str, float]:
    """Exercise the actual frozen checkpoints before any fresh-layout transition."""
    count = min(16, len(train["states"]))
    states = train["states"][:count]
    summary = train["summary"][:count]
    layouts = train["layouts"][:count]
    fold_ids = np.array([int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) % len(predictors)
                         for x in layouts])
    predicted_p = np.empty((count, 9, 3), dtype=np.float64)
    predicted_s = np.empty((count, 9), dtype=np.float64)
    for fold, predictor in enumerate(predictors):
        ids = np.flatnonzero(fold_ids == fold)
        if len(ids):
            predicted_p[ids], predicted_s[ids] = _predict(predictor, states[ids])
    exact_p, exact_s = _probabilities_from_exact(summary)
    reps = {
        "r0_action_only": _feature_set(predicted_p, predicted_s, "r0_action_only"),
        "r0_action_predicted_outcome_steps": _feature_set(predicted_p, predicted_s, "r0_action_predicted_outcome_steps"),
        "r0_action_exact_outcome_steps_oracle": _feature_set(exact_p, exact_s, "r0_action_exact_outcome_steps_oracle"),
    }
    actions, _ = _action_inputs(states)
    permutation = np.array([5, 1, 8, 3, 0, 7, 2, 6, 4])
    deviations = {}
    for name, path in CHECKPOINTS.items():
        model = _load_student(name, ROOT / path, train["states"])
        consequence, mask = reps[name]
        utility, risk = model.predict(states, actions, consequence, mask)
        pu, pr = model.predict(states, actions[:, permutation], consequence[:, permutation], mask)
        deviation = max(float(np.max(np.abs(pu - utility[:, permutation]))),
                        float(np.max(np.abs(pr - risk[:, permutation]))))
        if deviation > 1e-10:
            raise D162Error(f"frozen checkpoint is not permutation-equivariant: {name} ({deviation})")
        deviations[name] = deviation
    return deviations


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D162Error("commit protocol/source/tests/vault before locking")
    if cfg.get("schema") != "phase15_d162_fresh_layout_relational_v1":
        raise D162Error("D162 config schema mismatch")
    if cfg["frozen_models"]["checkpoint_paths"] != CHECKPOINTS:
        raise D162Error("D162 checkpoint config and runner paths differ")
    scope = cfg["scope"]
    if not validate_scope(scope):
        raise D162Error("D162 scope opens an unapproved operation")
    audit_path = ROOT / AUDIT
    if not audit_path.is_file():
        raise D162Error("hash-only layout audit must be completed before lock")
    audit = _json(audit_path)
    if audit != candidate_layout_audit(cfg) or audit["layout_count"] != 100:
        raise D162Error("D162 candidate layout audit failed or changed")
    train = _validate_parents(cfg)
    predictors, folds = _fold_predictors(train, cfg)
    if len(predictors) != 5:
        raise D162Error("five-fold predictor reconstruction failed")
    equivariance = _checkpoint_equivariance_audit(train, predictors)
    for key in ("lock", "run_dir", "report"):
        if (ROOT / cfg["outputs"][key]).exists():
            raise D162Error(f"refusing to overwrite D162 {key}")
    return {"schema": "phase15_d162_fresh_layout_relational_lock_v1",
            "status": "locked_one_shot_fresh_layout_development_evaluation",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": _git("rev-parse", "HEAD"),
            "source_sha256": {p: sha256_file(ROOT / p) for p in SOURCE_FILES},
            "config_sha256": sha256_file(ROOT / CONFIG),
            "protocol_sha256": sha256_file(ROOT / cfg["protocol"]),
            "audit_sha256": sha256_file(audit_path), "layout_audit": audit,
            "parent_sha256": _parents(cfg), "lineage": train["lineage"],
            "crossfit_predictor": {"folds": folds, "model_sha256": _predictor_hashes(predictors),
                                   "training_rows": 1365, "training_layouts": 140,
                                   "fits_on_fresh_data": False},
            "frozen_checkpoint_permutation_max_deviation": equivariance,
            "scope": cfg["scope"], "outputs": cfg["outputs"], "reserve_access": False, "test_access": False}


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    if cfg["frozen_models"]["checkpoint_paths"] != CHECKPOINTS:
        raise D162Error("D162 checkpoint config and runner paths differ")
    lock = _json(ROOT / cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d162_fresh_layout_relational_lock_v1":
        raise D162Error("D162 lock schema mismatch")
    if _git("status", "--porcelain"):
        raise D162Error("D162 execution requires a clean committed tree")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    if (lock["source_sha256"] != {p: sha256_file(ROOT / p) for p in SOURCE_FILES}
            or lock["config_sha256"] != sha256_file(ROOT / CONFIG)
            or lock["protocol_sha256"] != sha256_file(ROOT / cfg["protocol"])
            or lock["audit_sha256"] != sha256_file(ROOT / AUDIT)
            or lock["parent_sha256"] != _parents(cfg) or lock["scope"] != cfg["scope"]
            or lock.get("reserve_access") is not False or lock.get("test_access") is not False):
        raise D162Error("D162 lock hash/scope mismatch")
    if lock["layout_audit"] != candidate_layout_audit(cfg):
        raise D162Error("D162 fresh-layout hash audit changed")
    train = _validate_parents(cfg)
    if lock["lineage"] != train["lineage"]:
        raise D162Error("D162 parent checkpoint lineage changed")
    predictors, folds = _fold_predictors(train, cfg)
    if (lock["crossfit_predictor"]["folds"] != folds
            or lock["crossfit_predictor"]["model_sha256"] != _predictor_hashes(predictors)
            or lock["frozen_checkpoint_permutation_max_deviation"]
            != _checkpoint_equivariance_audit(train, predictors)):
        raise D162Error("D162 reconstructed predictor/checkpoint diagnostic differs from lock")
    for key in ("run_dir", "report"):
        if (ROOT / cfg["outputs"][key]).exists():
            raise D162Error("D162 is one-shot; output already exists")
    return cfg, lock


def _report(result: dict[str, Any]) -> str:
    out = ["# D162 — Fresh-layout relational generalization", "",
           "**Status:** completed frozen offline development evaluation. No student closed loop or reserve/test access.", "",
           f"Layouts: {result['layout_count']}; decision states: {result['state_count']}; exact branches: {result['teacher_action_branches']}; planner queries: {result['planner_queries']}; simulator transitions: {result['simulator_transitions']}.", "",
           "| Model | Macro regret | Catastrophic | Epsilon optimal | Best-set agreement | High-margin pair | Safe/collision pair | Collision action | p50 ms | p95 ms | Params |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in result["evaluation"]["model_metrics"].items():
        r = result["evaluation"]["runtime"][name]
        out.append(f"| {name} | {m['layout_macro_normalized_selected_regret']:.5f} | {m['catastrophic_fraction_layout_macro']:.5f} | {m['epsilon_optimal_rate']:.5f} | {m['best_action_set_agreement']:.5f} | {m.get('high_margin_pairwise_accuracy_macro')} | {m.get('safe_vs_collision_pair_accuracy')} | {m['collision_action_selection_rate']:.5f} | {r['median_decision_latency_ms']:.4f} | {r['p95_decision_latency_ms']:.4f} | {r['parameter_count']} |")
    out += ["", "## Paired layout bootstrap", "", "Improvement is R0 minus comparator; positive favors the comparator."]
    for name, pair in result["evaluation"]["paired_layout_bootstrap"].items():
        out.append(f"- {name}: regret {pair['regret_improvement_R0_minus_model']}; catastrophic {pair['catastrophic_improvement_R0_minus_model']}")
    out += ["", "## Interpretation", "", result["interpretation"], "",
           "This is a first-action preference evaluation on fixed Phase14F collector trajectories, not student-induced occupancy or closed-loop performance. The exact-consequence oracle latency is scorer-only and is not a deployable runtime claim.", "",
            f"Lock SHA-256: {result['lock_sha256']}; dataset SHA-256: {result['dataset_sha256']}; accounting SHA-256: {result['accounting_sha256']}.", ""]
    return "\n".join(out)


def run_d162() -> dict[str, Any]:
    cfg, lock = validate_lock()
    out = ROOT / cfg["outputs"]["run_dir"]
    out.mkdir(parents=True, exist_ok=False)
    lock_sha = sha256_file(ROOT / cfg["outputs"]["lock"])
    state_path = out / "run_state.json"
    state_path.write_text(json.dumps({"lock_sha256": lock_sha, "status": "running"}, indent=2) + "\n")
    started = time.perf_counter()
    planner_calls = transitions = 0
    try:
        train = _validate_parents(cfg)
        predictors, folds = _fold_predictors(train, cfg)
        predictor_hashes = _predictor_hashes(predictors)
        if predictor_hashes != lock["crossfit_predictor"]["model_sha256"]:
            raise D162Error("reconstructed CART model hashes differ from frozen lock")
        for fold, predictor in enumerate(predictors):
            payload = _predictor_payload(predictor, fold)
            path = out / f"crossfit_cart_fold_{fold}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                            encoding="utf-8")
            canonical_hash = hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")).hexdigest().upper()
            if canonical_hash != predictor_hashes[fold]:
                raise D162Error("saved cross-fit predictor file hash mismatch")
        reconstructed = out / "crossfit_predictor_reconstruction.json"
        reconstructed.write_text(json.dumps({"fold_records": folds, "model_sha256": predictor_hashes},
                                             indent=2, sort_keys=True) + "\n", encoding="utf-8")
        data = _collect(cfg, lock, out)
        planner_calls, transitions = data["planner_queries"], data["transitions"]
        evaluation = _score(cfg, data, train, predictors)
        gate = cfg["evaluation"]["material_predicted_consequence_gate"]
        pred = evaluation["paired_layout_bootstrap"]["r0_action_predicted_outcome_steps"]
        oracle = evaluation["paired_layout_bootstrap"]["r0_action_exact_outcome_steps_oracle"]
        preg, pcat = pred["regret_improvement_R0_minus_model"], pred["catastrophic_improvement_R0_minus_model"]
        material = (preg["point_improvement_left_minus_right"] >= gate["minimum_absolute_improvement_on_each_primary_metric"]
                    and pcat["point_improvement_left_minus_right"] >= gate["minimum_absolute_improvement_on_each_primary_metric"]
                    and preg["ci95_lower"] > 0 and pcat["ci95_lower"] > 0)
        oracle_pass = (oracle["regret_improvement_R0_minus_model"]["ci95_lower"] > 0
                       and oracle["catastrophic_improvement_R0_minus_model"]["ci95_lower"] > 0)
        if not oracle_pass:
            case, interpretation = "oracle_does_not_beat_R0", "Oracle headroom did not replicate on this fixed collector/layout block; stop without attributing the gap to consequence quality."
        elif material:
            case, interpretation = "predicted_consequences_help", "Predicted consequences pass the frozen material-gain gate over R0 on both primary metrics. This supports a separately locked confirmation, not closed-loop or final-test performance."
        elif (preg["point_improvement_left_minus_right"] <= -gate["minimum_absolute_improvement_on_each_primary_metric"]
              or pcat["point_improvement_left_minus_right"] <= -gate["minimum_absolute_improvement_on_each_primary_metric"]
              or preg["ci95_upper"] < 0 or pcat["ci95_upper"] < 0):
            case, interpretation = "predicted_arm_worse_or_not_material", "Oracle headroom replicates, but predicted consequences fail the material-gain gate and show harm on at least one primary point estimate."
        else:
            case, interpretation = "only_oracle_or_no_material_predicted_gain", "Oracle headroom replicates, but predicted consequences do not pass the frozen material-gain gate; no practical benefit is demonstrated."
        result = {"schema": "phase15_d162_fresh_layout_relational_result_v1",
                  "status": "completed_offline_fresh_layout_relational_evaluation",
                  "source_commit": lock["source_commit"], "lock_sha256": lock_sha,
                  "layout_count": len(data["episodes"]), "state_count": len(data["states"]),
                  "teacher_action_branches": len(data["states"]) * 9,
                  "planner_queries": planner_calls, "simulator_transitions": transitions,
                  "collector_terminal_outcomes": {s: sum(x["collector_terminal_status"] == s for x in data["episodes"])
                                                   for s in ("success", "collision", "timeout")},
                  "per_layout_state_counts": [
                      {"seed": episode["seed"], "layout_sha256": episode["layout_sha256"],
                       "decision_state_count": episode["decision_state_count"],
                       "collector_terminal_status": episode["collector_terminal_status"]}
                      for episode in data["episodes"]],
                  "branch_terminal_outcomes": {s: int(np.sum(np.argmax(data["summary"][:, :, :4], axis=2) == i))
                                               for i, s in ((0, "success"), (2, "timeout"), (3, "collision"))},
                  "dataset_sha256": data["dataset_sha256"], "accounting_sha256": data["accounting_sha256"],
                    "predictor_reconstruction_sha256": sha256_file(reconstructed),
                    "predictor_model_sha256": predictor_hashes,
                    "predictor_artifact_sha256": {
                        f"fold_{i}": sha256_file(out / f"crossfit_cart_fold_{i}.json")
                        for i in range(len(predictors))},
                  "evaluation": evaluation, "decision_rule": {"case": case, "predicted_material_gain": material,
                                                                 "oracle_headroom_replicated": oracle_pass},
                  "interpretation": interpretation,
                  "scope": {"student_closed_loop": False, "student_updates": False, "dagger": False,
                            "world_model_training": False, "jepa": False, "test_access": False,
                            "reserve_access": False},
                  "wall_time_seconds": time.perf_counter() - started,
                  "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}}
        rp = out / "result.json"
        rp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = _report(result)
        report_path = ROOT / cfg["outputs"]["report"]
        report_path.write_text(report, encoding="utf-8")
        manifest = {"schema": "phase15_d162_fresh_layout_relational_manifest_v1",
                    "status": result["status"], "source_commit": lock["source_commit"],
                    "lock_sha256": lock_sha, "result_sha256": sha256_file(rp),
                    "report_sha256": sha256_file(report_path), "dataset_sha256": data["dataset_sha256"],
                    "accounting_sha256": data["accounting_sha256"], "parent_sha256": lock["parent_sha256"],
                    "checkpoint_sha256": lock["lineage"]["checkpoint_sha256"],
                    "planner_calls": planner_calls, "simulator_transitions": transitions,
                    "student_closed_loop": False, "student_updates": False, "dagger": False,
                    "test_access": False, "reserve_access": False}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        state_path.write_text(json.dumps({"lock_sha256": lock_sha, "status": "completed",
                                          "result_sha256": sha256_file(rp)}, indent=2) + "\n")
        return result
    except Exception as exc:
        abort = {"schema": "phase15_d162_abort_v1", "status": "aborted_or_incomplete",
                 "lock_sha256": lock_sha, "error_type": type(exc).__name__, "error": str(exc),
                 "planner_queries_completed": planner_calls, "simulator_transitions_completed": transitions,
                 "test_access": False, "reserve_access": False}
        ap = out / "abort.json"
        ap.write_text(json.dumps(abort, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        state_path.write_text(json.dumps({"lock_sha256": lock_sha, "status": "aborted",
                                          "abort_sha256": sha256_file(ap)}, indent=2) + "\n")
        raise
