"""Score-only finalization of the fully preserved but post-collection-aborted D157 run."""

from __future__ import annotations

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
from consequence_lab.phase15_d155_stage_c import _load_stage_a_head
from consequence_lab.phase15_d156 import _read_json, load_predictors
from consequence_lab.phase15_d157 import (
    ACTION_COUNT, SOURCE_FILES as D157_SOURCE_FILES, _component_prediction_metrics,
    _decision_features_by_case, _decision_record, _interpret,
    _paired_layout_bootstrap, _predictor_output, _strip_metrics, _training_support,
)
from consequence_lab.phase15_decision_aligned import regrets_from_exact_summaries
from consequence_lab.phase15_outcome_summary import restore_decision_head

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d157_score_correction.yaml"
PROTOCOL = "reports/phase15_d157_score_correction_amendment.md"
LOCK = "experiments/phase15_d157_score_correction.lock.json"
SCORE_SOURCE_FILES = tuple(dict.fromkeys((
    CONFIG, PROTOCOL, "src/consequence_lab/phase15_d157_score_finalizer.py",
    "scripts/freeze_phase15_d157_score_correction.py",
    "scripts/run_phase15_d157_score_correction.py", "tests/test_phase15_d157_score_finalizer.py",
    *D157_SOURCE_FILES,
    "src/consequence_lab/phase15_d156.py",
    "src/consequence_lab/phase15_d155_stage_c.py", "src/consequence_lab/phase15_d155.py",
    "src/consequence_lab/phase15_decision_aligned.py", "src/consequence_lab/phase15_outcome_summary.py",
    "src/consequence_lab/phase15_representation_screen.py", "src/consequence_lab/phase14.py",
)))
EXPECTED_DATASET_SHA256 = "C67840935F44F58EC90D5B636155BBFB4022ABCA6CD0E6E1344ECE7B57667ADD"
EXPECTED_ACCOUNTING_SHA256 = "90832EC147A9B8E33DE675F825EC2105D81861E27BF68626E3FBA66201164248"


class D157FinalizerError(RuntimeError):
    """Fail-closed provenance or scoring error in the D157 score-only amendment."""


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D157FinalizerError(f"git {' '.join(args)} failed") from exc


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _parent_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}


def _validate_preserved(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    p = cfg["parents"]
    original_lock_path = ROOT / p["d157_lock"]
    original_lock = _read_json(original_lock_path)
    original_cfg = yaml.safe_load((ROOT / p["d157_config"]).read_text(encoding="utf-8"))
    abort_path = ROOT / p["d157_abort"]
    abort = _read_json(abort_path)
    if (original_lock.get("schema") != "phase15_d157_fresh_confirmation_lock_v1"
            or original_lock.get("status") != "locked_one_shot_frozen_predictor_confirmation"
            or abort.get("schema") != "phase15_d157_abort_v1"
            or abort.get("status") != "aborted_or_incomplete"
            or abort.get("source_commit") != original_lock.get("source_commit")
            or abort.get("lock_sha256") != sha256_file(original_lock_path)
            or abort.get("error") != "unexpected D150 decision-head input dimension 106"
            or abort.get("reserve_access") is not False
            or abort.get("test_rows_dereferenced") != 0):
        raise D157FinalizerError("original D157 lock/abort lineage does not match the score-only amendment")
    if (original_lock.get("reserve_generated_or_accessed") is not False
            or original_lock.get("scope", {}).get("test_data") is not False
            or original_lock.get("scope", {}).get("reserve_access") is not False):
        raise D157FinalizerError("original D157 lock has unexpected test/reserve scope")
    for name in D157_SOURCE_FILES:
        if original_lock.get("source_sha256", {}).get(name) != sha256_file(ROOT / name):
            raise D157FinalizerError(f"original D157 source drift: {name}")
    for name, expected in original_lock.get("parent_sha256", {}).items():
        parent_path = original_cfg.get("parents", {}).get(name)
        if parent_path is None or sha256_file(ROOT / parent_path) != expected:
            raise D157FinalizerError(f"original D157 frozen parent drift: {name}")

    dataset_path, accounting_path = ROOT / p["d157_dataset"], ROOT / p["d157_accounting"]
    if (sha256_file(dataset_path) != EXPECTED_DATASET_SHA256
            or sha256_file(accounting_path) != EXPECTED_ACCOUNTING_SHA256):
        raise D157FinalizerError("preserved D157 collection artifacts do not match audited hashes")
    layout_records = original_lock["layout_audit"]["layouts"]
    expected_layouts = {int(row["seed"]): str(row["layout_sha256"]) for row in layout_records}
    episode_records = json.loads(accounting_path.read_text(encoding="utf-8"))
    episode_layouts = {int(row["seed"]): str(row["layout_sha256"]) for row in episode_records}
    if (len(expected_layouts) != 100 or len(episode_records) != 100
            or episode_layouts != expected_layouts):
        raise D157FinalizerError("preserved episode accounting does not match the locked 100 layouts")
    with np.load(dataset_path, allow_pickle=False) as saved:
        states = saved["x_R0"].astype(np.float64)
        summary = saved["summary_R4"].astype(np.float64)
        layouts = saved["layouts"].astype(str)
        seeds = saved["seeds"].astype(np.int64)
        decisions = saved["decision_index"].astype(np.int64)
    row_pairs = {(int(seed), str(layout)) for seed, layout in zip(seeds, layouts, strict=True)}
    if (states.shape != (794, 61) or summary.shape != (794, ACTION_COUNT, 7)
            or len(set(layouts.tolist())) != 100 or len(decisions) != 794
            or any(expected_layouts.get(seed) != layout for seed, layout in row_pairs)
            or not np.isfinite(states).all() or not np.isfinite(summary).all()):
        raise D157FinalizerError("preserved D157 feature/summary/layout data contract failed")
    outcome = summary[:, :, :4]
    if (not np.all((outcome == 0.0) | (outcome == 1.0))
            or not np.all(outcome.sum(axis=2) == 1.0)
            or np.any(outcome[:, :, 1] != 0.0)):
        raise D157FinalizerError("preserved D157 outcomes violate terminal one-hot contract")
    observed_seeds = sorted(set(seeds.tolist()))
    if observed_seeds != sorted(expected_layouts):
        raise D157FinalizerError("preserved D157 rows do not cover every locked seed")
    _validate_row_accounting(seeds, layouts, decisions, episode_records, expected_layouts)
    collection = {
        "layout_count": 100, "state_rows": len(states),
        "forced_action_teacher_branches": len(states) * ACTION_COUNT,
        "teacher_planner_queries": None, "simulator_transitions": None,
        "unavailable_count_reason": "original runner aborted before counters were serialized; not reconstructed",
        "collector_terminal_outcomes": _count_map([row["collector_terminal_status"] for row in episode_records]),
        "branch_terminal_outcomes": _terminal_outcome_counts(summary),
        "dataset_sha256": EXPECTED_DATASET_SHA256, "dataset_bytes": dataset_path.stat().st_size,
        "accounting_sha256": EXPECTED_ACCOUNTING_SHA256,
        "episode_accounting": episode_records,
        "states": states, "summary": summary, "layouts": layouts,
    }
    return original_lock, abort, collection


def _validate_row_accounting(seeds: np.ndarray, layouts: np.ndarray, decisions: np.ndarray,
                             episode_records: list[dict[str, Any]],
                             expected_layouts: dict[int, str]) -> None:
    """Ensure each stored decision row is represented exactly once in episode accounting."""
    records = {int(row["seed"]): row for row in episode_records}
    for seed, expected_layout in expected_layouts.items():
        idx = np.flatnonzero(np.asarray(seeds) == seed)
        row_layouts = set(np.asarray(layouts).astype(str)[idx].tolist())
        row_decisions = np.sort(np.asarray(decisions)[idx])
        record = records[seed]
        count = int(record["decision_state_count"])
        if (row_layouts != {expected_layout} or len(idx) != count
                or not np.array_equal(row_decisions, np.arange(count, dtype=row_decisions.dtype))):
            raise D157FinalizerError(f"preserved D157 rows do not reconcile with seed {seed} accounting")


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D157FinalizerError("commit score-only amendment/provenance/tests/vault before freezing")
    if cfg.get("schema") != "phase15_d157_score_correction_v1":
        raise D157FinalizerError("D157 score-only amendment config schema mismatch")
    expected_scope = {
        "use_preserved_D157_dataset_only": True, "frozen_predictor_inference": True,
        "new_simulator_calls": False, "new_exact_planner_calls": False,
        "new_transitions_or_episodes": False, "model_fitting": False,
        "hyperparameter_tuning": False, "predictor_or_head_updates": False,
        "student_closed_loop": False, "dagger": False, "world_model": False,
        "jepa": False, "test_data": False, "reserve_access": False,
    }
    if cfg.get("scope") != expected_scope:
        raise D157FinalizerError("score-only amendment scope changed")
    original_lock, abort, collection = _validate_preserved(cfg)
    del abort
    for output in cfg["outputs"].values():
        if (ROOT / output).exists():
            raise D157FinalizerError(f"refusing to overwrite score-only output: {output}")
    return {
        "schema": "phase15_d157_score_correction_lock_v1",
        "status": "locked_score_only_finalization_of_preserved_dataset",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SCORE_SOURCE_FILES},
        "config_sha256": sha256_file(ROOT / CONFIG),
        "parent_sha256": _parent_hashes(cfg),
        "original_d157_lock_sha256": sha256_file(ROOT / cfg["parents"]["d157_lock"]),
        "original_d157_source_commit": original_lock["source_commit"],
        "dataset_sha256": collection["dataset_sha256"],
        "accounting_sha256": collection["accounting_sha256"],
        "metrics": _d157_config()["scoring"], "statistics": cfg["statistics"],
        "scope": cfg["scope"], "outputs": cfg["outputs"],
        "original_d157_abort_preserved": True,
        "new_teacher_or_simulator_data": False,
    }


def _d157_config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs/phase15_d157_fresh_confirmation.yaml").read_text(encoding="utf-8"))


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    lock = _read_json(ROOT / cfg["outputs"]["lock"])
    if (lock.get("schema") != "phase15_d157_score_correction_lock_v1"
            or lock.get("status") != "locked_score_only_finalization_of_preserved_dataset"):
        raise D157FinalizerError("D157 score-only correction lock schema mismatch")
    if (lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("source_sha256") != {name: sha256_file(ROOT / name) for name in SCORE_SOURCE_FILES}
            or lock.get("parent_sha256") != _parent_hashes(cfg)
            or lock.get("scope") != cfg["scope"]
            or lock.get("metrics") != _d157_config()["scoring"]
            or lock.get("statistics") != cfg["statistics"]
            or lock.get("outputs") != cfg["outputs"]
            or lock.get("new_teacher_or_simulator_data") is not False
            or lock.get("original_d157_abort_preserved") is not True):
        raise D157FinalizerError("D157 score-only correction source/parent/scope hash mismatch")
    if _git("status", "--porcelain"):
        raise D157FinalizerError("score-only finalization requires a clean committed tree")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    original_lock, _, _ = _validate_preserved(cfg)
    if (sha256_file(ROOT / cfg["parents"]["d157_lock"]) != lock["original_d157_lock_sha256"]
            or original_lock["source_commit"] != lock["original_d157_source_commit"]
            or sha256_file(ROOT / cfg["parents"]["d157_dataset"]) != lock["dataset_sha256"]
            or sha256_file(ROOT / cfg["parents"]["d157_accounting"]) != lock["accounting_sha256"]):
        raise D157FinalizerError("preserved D157 run artifacts changed after score-only lock")
    for output_name, output in cfg["outputs"].items():
        if output_name == "lock":
            continue
        if (ROOT / output).exists():
            raise D157FinalizerError(f"refusing second score-only finalization: {output}")
    return cfg, lock


def run_score_correction() -> dict[str, Any]:
    cfg, lock = validate_lock()
    started = time.perf_counter()
    out = ROOT / cfg["outputs"]["run_dir"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        original_lock, abort, collected = _validate_preserved(cfg)
        states, summary, layouts = collected["states"], collected["summary"], collected["layouts"]
        target = regrets_from_exact_summaries(summary)
        d157_cfg = _d157_config()
        d156_cfg = yaml.safe_load((ROOT / cfg["parents"]["d156_config"]).read_text(encoding="utf-8"))
        predictors = load_predictors(d156_cfg)
        stage_cfg = yaml.safe_load((ROOT / cfg["parents"]["stage_c_config"]).read_text(encoding="utf-8"))
        stage_a_head = _load_stage_a_head(ROOT / cfg["parents"]["stage_a_head"])
        r0_head = restore_decision_head(ROOT / cfg["parents"]["d150_r0_head"])
        if (stage_a_head.x_mean.shape != (106,) or r0_head.x_mean.shape != (61,)
                or set(predictors) != {"per_action_constant", "bounded_cart", "action_conditioned_mlp"}):
            raise D157FinalizerError("dedicated frozen model loader produced unexpected head/predictor contract")

        predictions = {name: _predictor_output(model, states) for name, model in predictors.items()}
        feature_cases = _decision_features_by_case(states, summary, predictions)
        vectors: dict[str, dict[str, Any]] = {}
        for case, feature in feature_cases.items():
            logits = stage_a_head.predict_logits(feature)
            details, regret, catastrophic = _decision_record(-logits, target, layouts, stage_cfg["metrics"])
            vectors[case] = {"metrics": _strip_metrics(details), "_regret": regret,
                             "_catastrophic": catastrophic}
        direct_detail, direct_regret, direct_cat = _decision_record(
            r0_head.predict(states), target, layouts, stage_cfg["metrics"])
        vectors["direct_D150_R0"] = {"metrics": _strip_metrics(direct_detail),
                                      "_regret": direct_regret, "_catastrophic": direct_cat}

        n_boot, seed = int(cfg["statistics"]["bootstrap_resamples"]), int(cfg["statistics"]["bootstrap_seed"])
        direct, exact = vectors["direct_D150_R0"], vectors["exact_exact"]
        comparisons: dict[str, Any] = {
            "exact_exact_advantage_over_direct_R0": {
                "regret": _paired_layout_bootstrap(direct["_regret"], exact["_regret"], layouts,
                                                     statistic="layout_macro", resamples=n_boot, seed=seed),
                "catastrophic_fraction": _paired_layout_bootstrap(direct["_catastrophic"], exact["_catastrophic"], layouts,
                                                                    statistic="state_micro", resamples=n_boot, seed=seed + 1),
            },
        }
        cart, out_only, steps_only = (vectors["bounded_cart_predicted_predicted"],
                                      vectors["bounded_cart_predicted_exact"],
                                      vectors["bounded_cart_exact_predicted"])
        comparisons["bounded_cart_predicted_both_advantage_over_direct_R0"] = {
            "regret": _paired_layout_bootstrap(direct["_regret"], cart["_regret"], layouts,
                                                 statistic="layout_macro", resamples=n_boot, seed=seed + 2),
            "catastrophic_fraction": _paired_layout_bootstrap(direct["_catastrophic"], cart["_catastrophic"], layouts,
                                                                statistic="state_micro", resamples=n_boot, seed=seed + 3),
        }
        comparisons["bounded_cart_component_swap_penalties_vs_exact"] = {
            "outcome_swap_regret": _paired_layout_bootstrap(out_only["_regret"], exact["_regret"], layouts,
                                                             statistic="layout_macro", resamples=n_boot, seed=seed + 4),
            "steps_swap_regret": _paired_layout_bootstrap(steps_only["_regret"], exact["_regret"], layouts,
                                                           statistic="layout_macro", resamples=n_boot, seed=seed + 5),
            "outcome_minus_steps_regret_penalty": _paired_layout_bootstrap(out_only["_regret"], steps_only["_regret"], layouts,
                                                                            statistic="layout_macro", resamples=n_boot, seed=seed + 6),
            "outcome_swap_catastrophe": _paired_layout_bootstrap(out_only["_catastrophic"], exact["_catastrophic"], layouts,
                                                                  statistic="state_micro", resamples=n_boot, seed=seed + 7),
            "steps_swap_catastrophe": _paired_layout_bootstrap(steps_only["_catastrophic"], exact["_catastrophic"], layouts,
                                                                statistic="state_micro", resamples=n_boot, seed=seed + 8),
            "outcome_minus_steps_catastrophe_penalty": _paired_layout_bootstrap(out_only["_catastrophic"], steps_only["_catastrophic"], layouts,
                                                                                statistic="state_micro", resamples=n_boot, seed=seed + 9),
        }
        support, support_hash = _training_support(d157_cfg, r0_head)
        distances = _nearest_distance(states, support, r0_head.x_mean, r0_head.x_std)
        component_metrics = {name: _component_prediction_metrics(summary, layouts, *predictions[name])
                             for name in predictions}
        oracle_cmp = comparisons["exact_exact_advantage_over_direct_R0"]
        cart_cmp = comparisons["bounded_cart_predicted_both_advantage_over_direct_R0"]
        oracle_pass = oracle_cmp["regret"]["ci95_lower"] > 0 and oracle_cmp["catastrophic_fraction"]["ci95_lower"] > 0
        cart_pass = (cart_cmp["regret"]["point_improvement_left_minus_right"] >= .05
                     and cart_cmp["catastrophic_fraction"]["point_improvement_left_minus_right"] >= .05
                     and cart_cmp["regret"]["ci95_lower"] > 0
                     and cart_cmp["catastrophic_fraction"]["ci95_lower"] > 0)
        dominance = comparisons["bounded_cart_component_swap_penalties_vs_exact"]
        outcome_dominant = (dominance["outcome_minus_steps_regret_penalty"]["ci95_lower"] > 0
                            and dominance["outcome_minus_steps_catastrophe_penalty"]["ci95_lower"] > 0)
        steps_dominant = (dominance["outcome_minus_steps_regret_penalty"]["ci95_upper"] < 0
                          and dominance["outcome_minus_steps_catastrophe_penalty"]["ci95_upper"] < 0)
        result = {
            "schema": "phase15_d157_score_correction_result_v1",
            "status": "completed_score_only_finalization_from_preserved_d157_dataset",
            "original_d157_status": "aborted_after_complete_collection_before_scoring",
            "original_abort_sha256": sha256_file(ROOT / cfg["parents"]["d157_abort"]),
            "source_commit": lock["source_commit"], "original_d157_source_commit": original_lock["source_commit"],
            "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
            "original_d157_lock_sha256": sha256_file(ROOT / cfg["parents"]["d157_lock"]),
            "parent_sha256": lock["parent_sha256"],
            "collection": {key: value for key, value in collected.items()
                            if key not in {"states", "summary", "layouts"}},
            "scope": {"new_simulator_calls": 0, "new_planner_calls": 0, "new_transitions": 0,
                      "new_episodes": 0, "model_fitting": False, "hyperparameter_tuning": False,
                      "student_closed_loop": False, "dagger": False, "world_model": False,
                      "jepa": False, "test_rows_dereferenced": 0, "reserve_access": False},
            "support": {"train_state_count": len(support), "train_state_sha256": support_hash,
                        "distance_normalization": "frozen_D150_R0_head_x_mean_x_std",
                        "nearest_train_distance_median": float(np.median(distances)),
                        "nearest_train_distance_p90": float(np.quantile(distances, .90)),
                        "nearest_train_distance_p99": float(np.quantile(distances, .99))},
            "decision_metrics": {name: value["metrics"] for name, value in vectors.items()},
            "component_prediction_metrics": component_metrics,
            "paired_layout_bootstrap": comparisons,
            "gates": {"oracle_advantage_replicates": bool(oracle_pass),
                      "bounded_cart_predicted_both_beats_R0_materially": bool(cart_pass),
                      "dominant_component": "outcome" if outcome_dominant else "steps" if steps_dominant else "mixed_or_uncertain"},
            "interpretation": _interpret(oracle_pass, cart_pass),
            "timing_seconds_score_only": time.perf_counter() - started,
            "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
            "reserve_access": False,
        }
        result_path = out / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_path = ROOT / cfg["outputs"]["report"]
        report_path.write_text(render_report(result), encoding="utf-8")
        manifest = {
            "schema": "phase15_d157_score_correction_manifest_v1", "status": result["status"],
            "source_commit": lock["source_commit"], "lock_sha256": result["lock_sha256"],
            "original_d157_lock_sha256": result["original_d157_lock_sha256"],
            "original_d157_abort_sha256": result["original_abort_sha256"],
            "parent_sha256": lock["parent_sha256"],
            "dataset_sha256": collected["dataset_sha256"], "accounting_sha256": collected["accounting_sha256"],
            "original_d157_teacher_planner_queries": None,
            "original_d157_simulator_transitions": None,
            "original_d157_count_unavailable_reason": collected["unavailable_count_reason"],
            "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
            "new_simulator_calls": 0, "new_planner_calls": 0, "new_transitions": 0,
            "new_episodes": 0, "model_fitting": False, "test_rows_dereferenced": 0,
            "reserve_access": False, "original_d157_aborted": True,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        abort = {"schema": "phase15_d157_score_correction_abort_v1", "status": "aborted_or_incomplete",
                 "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
                 "error_type": type(exc).__name__, "error": str(exc),
                 "new_simulator_calls": 0, "new_planner_calls": 0,
                 "reserve_access": False, "test_rows_dereferenced": 0,
                 "note": "Preserve this score-only output; do not re-run without a separate correction protocol."}
        (out / "abort.json").write_text(json.dumps(abort, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def _nearest_distance(query: np.ndarray, train: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    q, t = (np.asarray(query, dtype=np.float64) - mean) / std, (np.asarray(train, dtype=np.float64) - mean) / std
    result = np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), 32):
        delta = q[start:start + 32, None, :] - t[None, :, :]
        result[start:start + 32] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
    return result


def _count_map(values: list[Any]) -> dict[str, int]:
    names, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    return {str(name): int(count) for name, count in zip(names.tolist(), counts.tolist(), strict=True)}


def _terminal_outcome_counts(summary: np.ndarray) -> dict[str, int]:
    class_names = ("success", "active_zero", "timeout", "collision")
    indices = np.argmax(np.asarray(summary)[:, :, :4], axis=2).reshape(-1)
    return {name: int(np.sum(indices == index)) for index, name in enumerate(class_names)}


def render_report(result: dict[str, Any]) -> str:
    rows = [
        "# D157 score-only correction - fresh-layout results", "",
        "The original D157 one-shot run is preserved as aborted after data collection and before scoring. This report comes from a separate, source-locked score-only finalization of the unchanged dataset. No new teacher labels, simulator/planner calls, fitting, or tuning occurred.", "",
        f"Original abort SHA: `{result['original_abort_sha256']}`; preserved dataset SHA: `{result['collection']['dataset_sha256']}`; {result['collection']['layout_count']} layouts, {result['collection']['state_rows']} states, {result['collection']['forced_action_teacher_branches']} exact action summaries.", "",
        "## Decision metrics", "",
        "| Method | Macro regret | Catastrophic | Epsilon-optimal | Best-action set | High-margin pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = ["direct_D150_R0", "exact_exact"]
    for predictor in ("per_action_constant", "bounded_cart", "action_conditioned_mlp"):
        order.extend((f"{predictor}_predicted_exact", f"{predictor}_exact_predicted", f"{predictor}_predicted_predicted"))
    for name in order:
        metric = result["decision_metrics"][name]
        rows.append(f"| {name} | {metric['layout_macro_normalized_selected_regret']:.5f} | {metric['catastrophic_fraction_state_micro']:.4f} | {metric['epsilon_optimal_rate']:.4f} | {metric['best_action_set_agreement']:.4f} | {metric['high_margin_pairwise_accuracy_macro']:.4f} |")
    rows.extend(["", "## Paired layout bootstrap", "",
                 "Positive values mean the left/reference method has lower regret or fewer catastrophic decisions.", "",
                 "| Comparison | Metric | Improvement | 95% CI |", "|---|---|---:|---:|"])
    for comparison, metrics in result["paired_layout_bootstrap"].items():
        for metric, value in metrics.items():
            rows.append(f"| {comparison} | {metric} | {value['point_improvement_left_minus_right']:.5f} | [{value['ci95_lower']:.5f}, {value['ci95_upper']:.5f}] |")
    rows.extend(["", "## Gates and interpretation", "",
                 f"Oracle advantage replicated: **{result['gates']['oracle_advantage_replicates']}**. Frozen CART transfer gate passed: **{result['gates']['bounded_cart_predicted_both_beats_R0_materially']}**. Component assessment: **{result['gates']['dominant_component']}**.", "",
                 result["interpretation"], "",
                 "## Component prediction diagnostics", "",
                 "These weighted component metrics are descriptive and use the unchanged D157 summary targets.", "",
                 "| Predictor | Outcome accuracy | Outcome CE | Outcome Brier | Success recall | Timeout recall | Collision recall | Steps RMSE | Steps MAE |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, metrics in result["component_prediction_metrics"].items():
        outcome = metrics["outcome"]
        recall = outcome["per_class"]
        rows.append(
            f"| {name} | {outcome['accuracy']:.4f} | {outcome['cross_entropy']:.5f} | "
            f"{outcome['brier_score']:.5f} | {recall['success']['recall']} | "
            f"{recall['timeout']['recall']} | {recall['collision']['recall']} | "
            f"{metrics['steps']['rmse']:.5f} | {metrics['steps']['mae']:.5f} |"
        )
    rows.extend(["", "## Collection and provenance", "",
                 f"Collector terminal outcomes: `{result['collection']['collector_terminal_outcomes']}`.", "",
                 f"Forced-action branch terminal outcomes: `{result['collection']['branch_terminal_outcomes']}`.", "",
                 "The original D157 teacher-planner query and simulator-transition counters were not serialized before the original runner aborted; they remain unavailable and are not reconstructed. The score-only amendment itself made zero simulator calls, planner calls, transitions, or episodes.", "",
                 "No next phase is opened. DAgger, live student control, new labels, world-model/JEPA work, and reserve/test evaluation remain unauthorized by this score-only correction.", ""])
    return "\n".join(rows)
