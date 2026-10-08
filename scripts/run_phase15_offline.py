"""Run the frozen Phase15 offline representation/objective diagnosis only."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14g import trace_frozen_cart
from consequence_lab.phase15_data import build_phase15_dataset, layout_split
from consequence_lab.phase15_learning import TinyRegretMLP, decision_metrics, fit_cart


DEFAULT_CONFIG = "configs/phase15_offline_v1.yaml"
DEFAULT_LOCK = "experiments/phase15_offline_v1.lock.json"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_lock(root: Path, config: dict, config_path: str, lock_path: str) -> dict:
    lock = json.loads((root / lock_path).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase15_offline_run_lock_v1" or lock.get("status") != "authorized_phase15_offline_diagnosis_only":
        raise RuntimeError("Phase15 lock does not authorize the offline diagnostic")
    if lock.get("config_path") != config_path or lock.get("lock_path") != lock_path:
        raise RuntimeError("Phase15 config/lock path differs from the frozen run")
    if lock.get("config_sha256") != sha256_file(root / config_path):
        raise RuntimeError("Phase15 config hash differs from the lock")
    if lock.get("source_sha256") != {path: sha256_file(root / path) for path in lock["source_sha256"]}:
        raise RuntimeError("Phase15 source code hash differs from the lock")
    for key, path in (("phase14g_lock_sha256", config["parents"]["phase14g_lock"]),
                      ("phase14g_manifest_sha256", config["parents"]["phase14g_manifest"]),
                      ("phase14g_traces_sha256", config["parents"]["phase14g_traces"]),
                      ("phase14g_episodes_sha256", config["parents"]["phase14g_episodes"]),
                      ("phase14g_metrics_sha256", config["parents"]["phase14g_metrics"]),
                      ("phase14f_student_sha256", config["parents"]["phase14f_student"]),
                      ("phase14b_records_sha256", config["parents"]["phase14b_records"])):
        if lock[key] != sha256_file(root / path):
            raise RuntimeError(f"Phase15 parent artifact hash changed: {key}")
    if config.get("schema") == "phase15_offline_run_config_v2":
        for key, parent_key in (("phase15_v1_feature_dataset_sha256", "phase15_v1_feature_dataset"),
                                ("phase15_v1_feature_manifest_sha256", "phase15_v1_feature_manifest"),
                                ("phase15_v1_lock_sha256", "phase15_v1_lock"),
                                ("phase15_v1_failure_record_sha256", "phase15_v1_failure_record")):
            if lock.get(key) != sha256_file(root / config["parents"][parent_key]):
                raise RuntimeError(f"Phase15 v1 provenance artifact changed: {key}")
    if lock.get("scope") != config.get("scope") or lock["scope"].get("reserve_access") is not False:
        raise RuntimeError("Phase15 scope mismatch or reserve permission detected")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != lock["source_commit"]:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"], cwd=root)
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if dirty:
        raise RuntimeError("Phase15 run requires a clean tree after lock commit")
    return lock


def _indices(layouts: np.ndarray, selected: list[str]) -> np.ndarray:
    wanted = set(selected)
    return np.asarray([i for i, value in enumerate(layouts) if str(value) in wanted], dtype=np.int64)


def _balanced_source_indices(rows: list[dict], count: int) -> np.ndarray:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[str(row["source_layout_sha256"])].append(i)
    keys = sorted(groups)
    for key in keys:
        groups[key].sort(key=lambda i: (int(rows[i]["source_episode_seed"]), int(rows[i]["source_decision_index"])))
    selected: list[int] = []
    depth = 0
    while len(selected) < count:
        added = False
        for key in keys:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError("not enough Phase14B source rows for size-matched diagnostic")
        depth += 1
    return np.asarray(selected, dtype=np.int64)


def _bootstrap_difference(left: dict[str, float], right: dict[str, float], seed: int, count: int) -> dict:
    common = sorted(set(left) & set(right))
    if not common:
        raise ValueError("paired bootstrap requires shared test layouts")
    delta = np.asarray([right[key] - left[key] for key in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.mean(delta[rng.integers(0, len(delta), size=(count, len(delta)))], axis=1)
    return {
        "layout_count": len(common), "mean_difference_right_minus_left": float(delta.mean()),
        "lower_95": float(np.quantile(samples, 0.025)), "upper_95": float(np.quantile(samples, 0.975)),
        "resamples": int(count), "seed": int(seed),
    }


def _train_cell(x: np.ndarray, y: np.ndarray, layouts: np.ndarray, train_ids: np.ndarray,
                val_ids: np.ndarray, seed: int, objective: str, config: dict) -> tuple[TinyRegretMLP, dict]:
    model = TinyRegretMLP(
        seed, objective, max_epochs=int(config["training"]["max_epochs"]),
        patience=int(config["training"]["patience"]),
        learning_rate=float(config["training"]["learning_rate"]),
        l2=float(config["training"]["l2"]),
    ).fit(x[train_ids], y[train_ids], layouts[train_ids], x[val_ids], y[val_ids], layouts[val_ids])
    validation = decision_metrics(y[val_ids], model.predict(x[val_ids]), layouts[val_ids])
    latency_seconds = _inference_latency(model.predict, x[val_ids[:min(50, len(val_ids))]])
    metadata = {
        "objective": objective, "seed": seed, "best_epoch": model.best_epoch,
        "epochs_run": model.epochs_run, "parameter_count": model.parameter_count,
        "validation_loss": model.validation_loss,
        "validation_layout_macro_mean_regret": validation["mean_normalized_regret_layout_macro"],
        "validation_layout_macro_catastrophic_fraction": validation["catastrophic_fraction_layout_macro"],
        "validation_best_action_set_agreement": validation["best_action_set_agreement"],
        "median_single_state_inference_seconds": latency_seconds,
    }
    return model, metadata


def _inference_latency(predict, rows: np.ndarray) -> float:
    for row in rows[:5]:
        predict(row[None, :])
    samples = []
    for row in rows:
        started = time.perf_counter_ns()
        predict(row[None, :])
        samples.append((time.perf_counter_ns() - started) / 1e9)
    return float(np.median(samples)) if samples else 0.0


def _save_model(path: Path, model: TinyRegretMLP) -> None:
    np.savez_compressed(path, x_mean=model.x_mean, x_std=model.x_std,
                        w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2, w3=model.w3, b3=model.b3)


def _fit_memorization_ladder(x_by_rep: dict[str, np.ndarray], y: np.ndarray,
                             layouts: np.ndarray, train_ids: np.ndarray,
                             config: dict) -> dict:
    train_layouts = sorted(set(str(layouts[i]) for i in train_ids))
    per_layout = {layout: [int(i) for i in train_ids if str(layouts[i]) == layout] for layout in train_layouts}
    fullest_layout = max(train_layouts, key=lambda layout: (len(per_layout[layout]), layout))
    rungs = {
        "1": train_ids[:1], "8": train_ids[:min(8, len(train_ids))],
        "32": train_ids[:min(32, len(train_ids))],
        "one_layout": np.asarray(per_layout[fullest_layout], dtype=np.int64),
        "full_train": train_ids,
    }
    records = []
    gate = True
    for rung, ids in rungs.items():
        for rep, matrix in x_by_rep.items():
            for objective in config["training"]["objectives"]:
                model = TinyRegretMLP(
                    int(config["training"]["seed"]), objective,
                    max_epochs=int(config["training"]["max_epochs"]),
                    patience=int(config["training"]["patience"]),
                    learning_rate=float(config["training"]["learning_rate"]),
                    l2=float(config["training"]["l2"]),
                ).fit(matrix[ids], y[ids], layouts[ids], matrix[ids], y[ids], layouts[ids])
                metric = decision_metrics(y[ids], model.predict(matrix[ids]), layouts[ids])
                passed = (
                    metric["mean_normalized_regret_state_micro"] <= float(config["training"]["memorization_maximum_mean_decision_regret"])
                    and metric["best_action_set_agreement"] >= float(config["training"]["memorization_minimum_best_action_agreement"])
                )
                records.append({"rung": rung, "representation": rep, "objective": objective,
                                "state_count": len(ids), "mean_regret": metric["mean_normalized_regret_state_micro"],
                                "best_action_set_agreement": metric["best_action_set_agreement"],
                                "passed": bool(passed), "best_epoch": model.best_epoch})
                print(f"memorization rung={rung} rep={rep} objective={objective} pass={passed}", flush=True)
                gate &= passed
                if not passed:
                    return {"status": "failed", "one_layout_seed_count": len(per_layout[fullest_layout]),
                            "one_layout_hash": fullest_layout, "results": records}
    return {"status": "passed" if gate else "failed", "one_layout_seed_count": len(per_layout[fullest_layout]),
            "one_layout_hash": fullest_layout, "results": records}


def _main_impl() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, lock_path = args.config, args.lock
    config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
    lock = _validate_lock(root, config, config_path, lock_path)
    run_dir = root / config["outputs"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    traces = _jsonl(root / config["parents"]["phase14g_traces"])
    source_rows = _jsonl(root / config["parents"]["phase14b_records"])
    if config.get("schema") == "phase15_offline_run_config_v1":
        feature_manifest = build_phase15_dataset(root, run_dir / config["outputs"]["feature_dir"])
    else:
        parent_npz = root / config["parents"]["phase15_v1_feature_dataset"]
        parent_manifest_path = root / config["parents"]["phase15_v1_feature_manifest"]
        parent_lock_path = root / config["parents"]["phase15_v1_lock"]
        parent_failure_path = root / config["parents"]["phase15_v1_failure_record"]
        parent_failure = json.loads(parent_failure_path.read_text(encoding="utf-8"))
        if (parent_failure.get("status") != "aborted_before_completed_result"
                or parent_failure.get("heldout_metrics_computed") is not False
                or parent_failure.get("reserve_access") is not False
                or parent_failure.get("lock_sha256") != sha256_file(parent_lock_path)):
            raise RuntimeError("Phase15 v1 failure provenance is missing or inconsistent")
        parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
        if sha256_file(parent_npz) != parent_manifest.get("feature_dataset_sha256"):
            raise RuntimeError("Phase15 v1 feature NPZ does not match its parent manifest")
        if (parent_manifest.get("phase14g_run_lock_sha256") != lock["phase14g_lock_sha256"]
                or parent_manifest.get("phase14g_artifact_sha256", {}).get("traces") != lock["phase14g_traces_sha256"]
                or parent_manifest.get("phase14b_training_records_sha256") != lock["phase14b_records_sha256"]):
            raise RuntimeError("Phase15 v1 feature dataset parents differ from v2 lock")
        feature_dir = run_dir / config["outputs"]["feature_dir"]
        feature_dir.mkdir(parents=True, exist_ok=False)
        copied_npz = feature_dir / "phase15_features.npz"
        shutil.copyfile(parent_npz, copied_npz)
        feature_manifest = dict(parent_manifest)
        feature_manifest.update({
            "status": "reused_verified_phase15_v1_feature_dataset",
            "reused_from": config["parents"]["phase15_v1_feature_dataset"],
            "reused_parent_manifest_sha256": sha256_file(parent_manifest_path),
            "copied_npz_sha256": sha256_file(copied_npz),
            "new_r2_transitions": 0,
        })
        (feature_dir / "manifest.json").write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with np.load(run_dir / config["outputs"]["feature_dir"] / "phase15_features.npz", allow_pickle=False) as feature_file:
        x_by_rep = {rep: np.asarray(feature_file[f"x_{rep}"], dtype=np.float64) for rep in ("R0", "R1", "R2", "R3", "R4")}
        y = np.asarray(feature_file["y"], dtype=np.float64)
        layouts = np.asarray(feature_file["layouts"]).astype(str)
    split = layout_split(layouts.tolist())
    split_ids = {name: _indices(layouts, values) for name, values in split.items()}
    train_ids, val_ids, test_ids = split_ids["train"], split_ids["validation"], split_ids["test"]
    y = np.asarray(y, dtype=np.float64)

    ladder = _fit_memorization_ladder(x_by_rep, y, layouts, train_ids, config)
    if ladder["status"] != "passed":
        result = {"status": "stopped_memorization_gate_failed", "memorization": ladder,
                  "heldout_evaluated": False, "reserve_access": False}
        result.update({"run_config_path": config_path, "run_lock_path": lock_path})
        _write_result(root, run_dir, result, lock, lock_path, config["outputs"]["report"])
        return

    model_results = []
    selected_validation = None
    for rep, matrix in x_by_rep.items():
        for objective in config["training"]["objectives"]:
            model, meta = _train_cell(matrix, y, layouts, train_ids, val_ids,
                                      int(config["training"]["seed"]), objective, config)
            val_prediction = model.predict(matrix[val_ids])
            test_prediction = model.predict(matrix[test_ids])
            val_metric = decision_metrics(y[val_ids], val_prediction, layouts[val_ids])
            test_metric = decision_metrics(y[test_ids], test_prediction, layouts[test_ids])
            cell = {"representation": rep, **meta, "validation_metrics": val_metric,
                    "test_metrics": test_metric, "runtime_status": (
                        "deployable" if rep == "R0" else "privileged_diagnostic_only" if rep != "R4" else "oracle_non_deployable"
                    )}
            model_results.append(cell)
            safe_config = rep == "R0"
            if safe_config and (selected_validation is None or (
                val_metric["mean_normalized_regret_layout_macro"], val_metric["catastrophic_fraction_layout_macro"]
            ) < (selected_validation["validation_metrics"]["mean_normalized_regret_layout_macro"],
                 selected_validation["validation_metrics"]["catastrophic_fraction_layout_macro"])):
                selected_validation = cell
                _save_model(run_dir / "selected_r0_student.npz", model)
            print(f"fit rep={rep} objective={objective} val_regret={val_metric['mean_normalized_regret_layout_macro']:.4f} test_regret={test_metric['mean_normalized_regret_layout_macro']:.4f}", flush=True)

    # CART remains a non-neural regression diagnostic on R0 and compact R3.
    cart_results = []
    for rep in ("R0", "R3"):
        pred, nodes = fit_cart(x_by_rep[rep][train_ids], y[train_ids], x_by_rep[rep][test_ids])
        cart_results.append({"representation": rep, "tree_nodes": nodes,
                             "test_metrics": decision_metrics(y[test_ids], pred, layouts[test_ids])})

    # H1 source-vs-on-policy comparison uses Phase14B's matching exact
    # forced-first-action regret labels and the same learner/state representation.
    source_ids = _balanced_source_indices(source_rows, len(train_ids))
    sx_raw = np.asarray([source_rows[i]["student_input_52"] for i in source_ids], dtype=np.float64)
    sy = np.asarray([[a["normalized_regret"] for a in source_rows[i]["actions"]] for i in source_ids], dtype=np.float64)
    from consequence_lab.phase14c import build_feature_matrix
    sx = build_feature_matrix(sx_raw, "raw52_plus_physics9")
    slayouts = np.asarray([source_rows[i]["source_layout_sha256"] for i in source_ids])
    source_model = TinyRegretMLP(
        int(config["training"]["seed"]), "regression",
        max_epochs=int(config["training"]["max_epochs"]), patience=int(config["training"]["patience"]),
        learning_rate=float(config["training"]["learning_rate"]), l2=float(config["training"]["l2"]),
    ).fit(sx, sy, slayouts, x_by_rep["R0"][val_ids], y[val_ids], layouts[val_ids])
    source_meta = {
        "best_epoch": source_model.best_epoch, "epochs_run": source_model.epochs_run,
        "parameter_count": source_model.parameter_count, "validation_loss": source_model.validation_loss,
        "median_single_state_inference_seconds": _inference_latency(
            source_model.predict, x_by_rep["R0"][val_ids[:min(50, len(val_ids))]]),
    }
    source_val = decision_metrics(y[val_ids], source_model.predict(x_by_rep["R0"][val_ids]), layouts[val_ids])
    source_test = decision_metrics(y[test_ids], source_model.predict(x_by_rep["R0"][test_ids]), layouts[test_ids])
    onpolicy_key = "R0:regression"
    onpolicy_test = next(row["test_metrics"] for row in model_results if f"{row['representation']}:{row['objective']}" == onpolicy_key)
    # Frozen Phase14F CART baseline on the same student-visited test states.
    frozen = json.loads((root / config["parents"]["phase14f_student"]).read_text(encoding="utf-8"))
    frozen_predictions = []
    for i in test_ids:
        frozen_predictions.append(trace_frozen_cart(frozen, traces[int(i)]["student_runtime_input_52"]).predicted_regret)
    frozen_metric = decision_metrics(y[test_ids], np.asarray(frozen_predictions), layouts[test_ids])
    source_h1 = {
        "source_fit_count": len(source_ids), "source_unique_layouts": len(set(slayouts)),
        "source_record_indices": source_ids.tolist(),
        "source_layout_row_counts": {key: int(np.sum(slayouts == key)) for key in sorted(set(slayouts))},
        "source_train_metadata": source_meta, "source_validation_metrics": source_val,
        "source_test_metrics": source_test, "onpolicy_r0_test_metrics": onpolicy_test,
        "frozen_phase14f_cart_test_metrics": frozen_metric,
        "source_minus_onpolicy_regret_bootstrap": _bootstrap_difference(
            onpolicy_test["per_layout_mean_regret"], source_test["per_layout_mean_regret"],
            int(config["training"]["bootstrap"]["seed"]), int(config["training"]["bootstrap"]["resamples"])),
    }

    selected_test = selected_validation["test_metrics"]
    frozen_layout_regret = frozen_metric["per_layout_mean_regret"]
    selected_layout_regret = selected_test["per_layout_mean_regret"]
    improvement_ci = _bootstrap_difference(
        selected_layout_regret, frozen_layout_regret,
        int(config["training"]["bootstrap"]["seed"]), int(config["training"]["bootstrap"]["resamples"]),
    )
    catastrophic_gain = (frozen_metric["catastrophic_fraction_layout_macro"]
                        - selected_test["catastrophic_fraction_layout_macro"])
    gate = config["gate"]
    advancement_gate = bool(
        improvement_ci["mean_difference_right_minus_left"] >= float(gate["mean_regret_absolute_improvement_over_frozen_cart"])
        and improvement_ci["lower_95"] > float(gate["paired_bootstrap_lower_95_improvement"])
        and selected_test["catastrophic_fraction_layout_macro"] <= float(gate["catastrophic_fraction_maximum"])
        and catastrophic_gain >= float(gate["catastrophic_fraction_absolute_improvement_over_frozen_cart"])
        and selected_test["best_action_set_agreement"] >= float(gate["best_action_set_agreement_minimum"])
    )
    result = {
        "status": "completed_offline_diagnosis",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(root / lock_path),
        "run_config_path": config_path, "run_lock_path": lock_path,
        "phase14g_reexecuted": False, "new_episodes": 0, "exact_planner_calls": 0,
        "reserve_access": False, "closed_loop_evaluation": False,
        "split": {name: {"layout_count": len(values), "layout_hashes": values} for name, values in split.items()},
        "memorization": ladder,
        "feature_dataset": feature_manifest,
        "representation_objective_results": model_results,
        "cart_diagnostics": cart_results,
        "distribution_shift_h1": source_h1,
        "selected_by_validation": {"representation": selected_validation["representation"],
                                    "objective": selected_validation["objective"]},
        "frozen_cart_gate_comparison": {
            "selected_student_test_metrics": selected_test,
            "frozen_cart_test_metrics": frozen_metric,
            "mean_regret_improvement_paired_bootstrap": improvement_ci,
            "catastrophic_fraction_absolute_improvement": catastrophic_gain,
            "proposed_for_separate_live_protocol": advancement_gate,
            "note": "Even if true, this does not authorize or initiate closed-loop evaluation.",
        },
        "selected_r0_checkpoint": {
            "path": "selected_r0_student.npz",
            "sha256": sha256_file(run_dir / "selected_r0_student.npz"),
        } if (run_dir / "selected_r0_student.npz").is_file() else None,
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "wall_seconds": time.perf_counter() - started,
    }
    _write_result(root, run_dir, result, lock, lock_path, config["outputs"]["report"])


def main() -> None:
    try:
        _main_impl()
    except Exception as exc:
        root = Path(__file__).resolve().parents[1]
        config_path, lock_path = DEFAULT_CONFIG, DEFAULT_LOCK
        args = sys.argv[1:]
        for index, value in enumerate(args[:-1]):
            if value == "--config":
                config_path = args[index + 1]
            elif value == "--lock":
                lock_path = args[index + 1]
        try:
            config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
            run_dir = root / config["outputs"]["run_dir"]
            feature_manifest = run_dir / config["outputs"]["feature_dir"] / "manifest.json"
            if run_dir.is_dir() and not (run_dir / "failure.json").exists():
                failure = {
                    "schema": "phase15_offline_abort_record_v1", "status": "aborted_before_completed_result",
                    "config_path": config_path, "lock_path": lock_path,
                    "lock_sha256": sha256_file(root / lock_path) if (root / lock_path).is_file() else None,
                    "exception_type": type(exc).__name__, "exception": str(exc),
                    "feature_manifest_sha256": sha256_file(feature_manifest) if feature_manifest.is_file() else None,
                    "reserve_access": False, "closed_loop_evaluation": False,
                }
                (run_dir / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
        raise


def _write_result(root: Path, run_dir: Path, result: dict, lock: dict,
                  lock_relative: str, report_relative: str) -> None:
    path = run_dir / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_sha = sha256_file(path)
    manifest = {
        "schema": "phase15_offline_result_manifest_v1", "status": result["status"],
        "run_lock_sha256": sha256_file(root / lock_relative), "source_commit": lock["source_commit"],
        "result_sha256": result_sha,
        "feature_dataset_sha256": result.get("feature_dataset", {}).get("feature_dataset_sha256"),
        "selected_r0_checkpoint_sha256": (result.get("selected_r0_checkpoint") or {}).get("sha256"),
        "phase14g_reexecuted": False, "new_episodes": 0, "exact_planner_calls": 0,
        "closed_loop_evaluation": False, "dagger": False, "learned_world_model": False,
        "reserve_access": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase 15B-D offline diagnosis result", "",
        f"**Status:** {result['status']}. No new episodes, planner calls, closed-loop evaluation, DAgger, learned world model, or reserve access.", "",
    ]
    if result.get("memorization", {}).get("status") != "passed":
        lines.extend(["## Memorization gate", "", "The fitting gate failed. Held-out results were not evaluated; stop and debug the reported rung.", ""])
    else:
        lines.extend(["## Representation/objective comparison", "",
                      "All metrics below are on the predeclared held-out Phase14G test layouts; hyperparameters/checkpoints were selected using training/validation only.", "",
                      "| Representation | Objective | Runtime status | Test regret (layout macro) | Catastrophic fraction | Best-set agreement | Pairwise ranking | Median inference ms | Parameters |",
                      "|---|---|---|---:|---:|---:|---:|---:|---:|"])
        for row in result["representation_objective_results"]:
            m = row["test_metrics"]
            lines.append(f"| {row['representation']} | {row['objective']} | {row['runtime_status']} | {m['mean_normalized_regret_layout_macro']:.4f} | {m['catastrophic_fraction_layout_macro']:.4f} | {m['best_action_set_agreement']:.4f} | {m['pairwise_ranking_accuracy_macro']:.4f} | {1000*row['median_single_state_inference_seconds']:.4f} | {row['parameter_count']} |")
        lines.extend(["", "## Distribution-shift diagnostic", "",
                      "| Training distribution / learner | Test regret | Catastrophic fraction | Best-set agreement |",
                      "|---|---:|---:|---:|"])
        h1 = result["distribution_shift_h1"]
        for name, metric in (("Frozen Phase14F CART", h1["frozen_phase14f_cart_test_metrics"]),
                             ("Phase14B source-trained MLP", h1["source_test_metrics"]),
                             ("Phase14G student-visited R0 MLP", h1["onpolicy_r0_test_metrics"])):
            lines.append(f"| {name} | {metric['mean_normalized_regret_layout_macro']:.4f} | {metric['catastrophic_fraction_layout_macro']:.4f} | {metric['best_action_set_agreement']:.4f} |")
        h1_ci = h1["source_minus_onpolicy_regret_bootstrap"]
        lines.append("")
        lines.append(f"Paired layout-bootstrap source-minus-on-policy regret difference: {h1_ci['mean_difference_right_minus_left']:.4f} (95% CI [{h1_ci['lower_95']:.4f}, {h1_ci['upper_95']:.4f}]); positive values mean the source-trained model has higher test regret.")
        lines.extend(["", "## CART diagnostics", "", "| Representation | Nodes | Test regret | Catastrophic fraction | Best-set agreement |", "|---|---:|---:|---:|---:|"])
        for row in result["cart_diagnostics"]:
            metric = row["test_metrics"]
            lines.append(f"| {row['representation']} | {row['tree_nodes']} | {metric['mean_normalized_regret_layout_macro']:.4f} | {metric['catastrophic_fraction_layout_macro']:.4f} | {metric['best_action_set_agreement']:.4f} |")
        gate_data = result["frozen_cart_gate_comparison"]
        ci = gate_data["mean_regret_improvement_paired_bootstrap"]
        lines.extend(["", "## Gate and scope", "",
                      f"Validation selected `{result['selected_by_validation']['representation']}` / `{result['selected_by_validation']['objective']}`. Frozen-CART-relative test regret improvement: {ci['mean_difference_right_minus_left']:.4f}, paired layout-bootstrap 95% CI [{ci['lower_95']:.4f}, {ci['upper_95']:.4f}].",
                      f"Catastrophic fraction absolute improvement: {gate_data['catastrophic_fraction_absolute_improvement']:.4f}. Live-protocol proposal gate: **{'PASS' if gate_data['proposed_for_separate_live_protocol'] else 'FAIL'}**. A pass permits only a separate protocol proposal, not a live run.", "",
                      "R1-R3 use exact simulator-derived features and are privileged; R4 contains exact-teacher continuation summaries and is explicitly an oracle. Results are diagnostic, not deployable-policy claims. Descriptive associations do not establish causal explanations.",
                      f"Run lock SHA-256: `{manifest['run_lock_sha256']}`; result SHA-256: `{result_sha}`.", ""])
    (root / report_relative).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "result": str(path), "report": report_relative}, indent=2))


if __name__ == "__main__":
    main()
