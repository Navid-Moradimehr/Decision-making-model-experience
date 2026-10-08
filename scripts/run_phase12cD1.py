"""Run the authorized Phase 12c-D1 fitting-data-only learnability check."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_model import CorridorCandidateScorer, model_parameter_count
from consequence_lab.corridor_baselines import grid_astar_route, waypoint_braking_action
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import CorridorState
from consequence_lab.phase12c_c_training import (
    _fit_one,
    _layout_arrays,
    _layout_macro,
    _one_sided_cluster_bounds,
    evaluate_model,
    load_dataset,
    permutation_consistency,
    score_model,
    sha256,
)


ROOT_REL = Path("artifacts/runs")
FOLD_LOCK = "experiments/phase12cD1_fold_v3.lock.json"
CONFIG = "configs/phase12cD.yaml"


def subset_dataset(data: dict, rows: np.ndarray, *, split: str) -> dict:
    selected = np.asarray(rows, dtype=np.int64)
    query_count = len(data["metadata"])
    targets = {}
    for key, value in data["targets"].items():
        if value.ndim > 0 and value.shape[0] == query_count:
            targets[key] = value[selected].copy()
        else:
            # Bank-index/config arrays describe the entire archive, not rows.
            targets[key] = value.copy()
    result = {
        "inputs": {key: value[selected].copy() for key, value in data["inputs"].items()},
        "targets": targets,
        "metadata": [data["metadata"][int(index)] for index in selected],
        "manifest": copy.deepcopy(data["manifest"]),
    }
    result["manifest"]["split"] = split
    result["manifest"]["query_count"] = len(selected)
    result["manifest"]["layout_count"] = len({row["layout_sha256"] for row in result["metadata"]})
    result["manifest"]["offline_gate_labels_collected"] = False
    result["manifest"]["reserve_accessed"] = False
    return result


def normalize_pair(train: dict, validation: dict, floor: float) -> tuple[np.ndarray, np.ndarray]:
    train_states = train["inputs"]["states"].astype(np.float64)
    mean = train_states.mean(axis=0)
    scale = train_states.std(axis=0)
    scale = np.where(scale < floor, 1.0, scale)
    for split in (train, validation):
        split["inputs"]["states"] = ((split["inputs"]["states"] - mean) / scale).astype(np.float32)
    return mean.astype(np.float64), scale.astype(np.float64)


def add_geometry_actions(data: dict, *, resolution: float, waypoint_tolerance: float) -> None:
    """Reconstruct each public dynamic state and evaluate the frozen geometry baseline."""
    scenario_cache = {}
    actions = []
    for features, metadata in zip(data["inputs"]["states"], data["metadata"]):
        seed = int(metadata["episode_seed"])
        scenario = scenario_cache.get(seed)
        if scenario is None:
            scenario = generate_staggered_corridor(seed)
            if scenario.layout_sha256 != str(metadata["layout_sha256"]):
                raise ValueError("fitting metadata layout hash does not match its seed")
            scenario_cache[seed] = scenario
        spec = scenario.spec
        elapsed = int(round((1.0 - float(features[15])) * spec.max_steps))
        state = CorridorState(
            x=float(features[0]) * spec.width,
            y=float(features[1]) * spec.height,
            vx=float(features[2]) * spec.max_speed,
            vy=float(features[3]) * spec.max_speed,
            elapsed_steps=elapsed,
            status="active",
        )
        route = grid_astar_route(
            (state.x, state.y), spec.goal, spec, resolution=resolution
        )
        if route is None:
            raise ValueError("geometry baseline could not route a valid fitting state")
        actions.append(waypoint_braking_action(
            state, route, spec, waypoint_tolerance=waypoint_tolerance
        ))
    for row, action in zip(data["metadata"], actions):
        row["geometry_action"] = int(action)


def baseline_summary(validation: dict) -> dict:
    targets = validation["targets"]
    regrets = targets["per_bank_normalized_regret"][:, 6:12, :].mean(axis=1)
    layout_ids, _ = _layout_arrays(validation["metadata"])
    geometry_action = np.asarray([row["geometry_action"] for row in validation["metadata"]], dtype=np.int64)
    choices = {
        "constant_coast": np.zeros(len(regrets), dtype=np.int64),
        "geometry_waypoint": geometry_action,
    }
    summaries = {}
    layout_regrets = {}
    for name, action in choices.items():
        selected = regrets[np.arange(len(regrets)), action]
        mean, per_layout = _layout_macro(selected, layout_ids)
        lower, upper = _one_sided_cluster_bounds(per_layout, seed=133701)
        summaries[name] = {
            "layout_macro_heldout_regret": mean,
            "one_sided_98_75_lower": lower,
            "one_sided_98_75_upper": upper,
            "per_layout_regret": per_layout.tolist(),
        }
        layout_regrets[name] = per_layout
    return {"summary": summaries, "per_layout": layout_regrets}


def load_frozen_model(checkpoint_path: Path, variant: str, config: dict) -> tuple[torch.nn.Module, dict]:
    kind = "policy" if variant == "bounded_preference_policy" else "value"
    model = CorridorCandidateScorer(
        kind,
        hidden_state=int(config["model"]["hidden_state"]),
        embedding=int(config["model"]["embedding"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output", default="artifacts/runs/phase12cD1_learnability_v3")
    args = parser.parse_args()
    if args.confirmation != "RUN_PHASE12CD1_FIT_ONLY":
        raise SystemExit("D1 fitting requires --confirmation RUN_PHASE12CD1_FIT_ONLY")
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        # CPU is part of this phase's preregistered compute contract.
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(1)

    config_path = ROOT / CONFIG
    fold_lock_path = ROOT / FOLD_LOCK
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    lock = json.loads(fold_lock_path.read_text(encoding="utf-8"))
    stage = config["stage_D1"]
    if lock.get("schema") != "phase12cD1_layout_fold_v3":
        raise SystemExit("missing/invalid D1 layout fold lock")
    if lock.get("config_sha256") != sha256(config_path):
        raise SystemExit("D1 config hash differs from frozen fold lock")
    if lock.get("source_sha256", {}).get("scripts/run_phase12cD1.py") != sha256(Path(__file__).resolve()):
        raise SystemExit("D1 runner changed after layout-fold lock")
    for relative, expected in lock["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"D1 locked source changed: {relative}")
    if lock.get("old_phase12cC_selection_gate_read") is not False:
        raise SystemExit("D1 lock violates old selection/gate boundary")
    if lock.get("reserve_accessed") is not False or lock.get("development_accessed") is not False:
        raise SystemExit("D1 lock violates sealed-split boundary")

    input_dir = ROOT / stage["input_dataset"]
    manifest_path = input_dir / "manifest.json"
    if sha256(manifest_path) != lock.get("training_manifest_sha256"):
        raise SystemExit("D1 fitting data manifest changed")
    for name, expected in lock["training_data_file_sha256"].items():
        if sha256(input_dir / name) != expected:
            raise SystemExit(f"D1 fitting file changed: {name}")
    data = load_dataset(input_dir)
    hashes = np.asarray([str(row["layout_sha256"]) for row in data["metadata"]])
    training_hashes = set(lock["training_layout_sha256"])
    validation_hashes = set(lock["internal_validation_layout_sha256"])
    if training_hashes & validation_hashes or len(training_hashes) != 128 or len(validation_hashes) != 32:
        raise SystemExit("D1 fold lock has invalid layout assignment")
    if set(np.unique(hashes)) != training_hashes | validation_hashes:
        raise SystemExit("D1 layout fold does not match fitting dataset")
    train_rows = np.flatnonzero(np.isin(hashes, list(training_hashes)))
    validation_rows = np.flatnonzero(np.isin(hashes, list(validation_hashes)))
    train = subset_dataset(data, train_rows, split="archived_training")
    validation = subset_dataset(data, validation_rows, split="model_selection")
    geometry = stage["geometry_parameters"]
    add_geometry_actions(
        train, resolution=float(geometry["resolution"]),
        waypoint_tolerance=float(geometry["waypoint_tolerance"]),
    )
    add_geometry_actions(
        validation, resolution=float(geometry["resolution"]),
        waypoint_tolerance=float(geometry["waypoint_tolerance"]),
    )
    mean, scale = normalize_pair(train, validation, float(stage["state_preprocessing"]["std_floor"]))

    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / ROOT_REL).resolve()) or output.exists():
        raise SystemExit("D1 output must be a new path under artifacts/runs")
    started = time.monotonic()
    fit_config = {
        "models": {
            "common": {
                "hidden_state": int(stage["model"]["hidden_state"]),
                "embedding": int(stage["model"]["embedding"]),
                "maximum_parameters": int(stage["model"]["maximum_parameters"]),
            },
            "fitting": {
                "learning_rate": float(stage["model"]["learning_rate"]),
                "weight_decay": float(stage["model"]["weight_decay"]),
                "batch_decision_sets": int(stage["model"]["batch_decision_sets"]),
                "max_epochs": int(stage["model"]["max_epochs"]),
                "early_stopping_patience": int(stage["model"]["early_stopping_patience"]),
                "gradient_norm_clip": float(stage["model"]["gradient_norm_clip"]),
                "seeds": [int(seed) for seed in stage["model"]["seeds"]],
            },
        }
    }
    cap = int(stage["resources"]["wall_seconds"])
    fit_runs: dict[str, list[dict]] = {}
    selected: dict[str, tuple[torch.nn.Module, dict]] = {}
    for variant in ("direct_candidate_value", "bounded_preference_policy"):
        variant_runs = []
        best = None
        for seed in fit_config["models"]["fitting"]["seeds"]:
            model, summary = _fit_one(
                train, validation, variant=variant, seed=seed, config=fit_config,
                device=device, started=started, time_cap=cap,
            )
            variant_runs.append(summary)
            if best is None or summary["best_selection_heldout_regret"] < best[1]["best_selection_heldout_regret"]:
                best = (model, summary)
        assert best is not None
        selected[variant] = best
        fit_runs[variant] = variant_runs

    if time.monotonic() - started > cap:
        raise SystemExit("D1 exceeded frozen wall-time cap")
    baselines = baseline_summary(validation)
    promotion = stage["advancement_gate"]
    variant_results = {}
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_hashes = {}
    for variant, (model, summary) in selected.items():
        kind = "policy" if variant == "bounded_preference_policy" else "value"
        checkpoint_path = checkpoint_dir / f"{variant}_seed{summary['fit_seed']}.pt"
        torch.save({
            "variant": variant,
            "fit_seed": int(summary["fit_seed"]),
            "state_dict": model.state_dict(),
            "input_mean": torch.as_tensor(mean, dtype=torch.float64),
            "input_scale": torch.as_tensor(scale, dtype=torch.float64),
            "config_sha256": sha256(config_path),
            "fold_lock_sha256": sha256(fold_lock_path),
        }, checkpoint_path)
        checkpoint_hashes[str(checkpoint_path.relative_to(ROOT))] = sha256(checkpoint_path)
        validation_metrics = evaluate_model(model, validation, variant=variant, device=device)
        fitting_metrics = evaluate_model(model, train, variant=variant, device=device)
        scores, _ = score_model(model, validation, variant=variant, device=device)
        chosen = scores.argmax(axis=1) if kind == "policy" else scores.argmin(axis=1)
        permutation = permutation_consistency(
            model, variant=variant, state_rows=validation["inputs"]["states"], device=device,
        )
        layout_ids, _ = _layout_arrays(validation["metadata"])
        model_regret = np.asarray(validation_metrics["per_layout_regret"], dtype=np.float64)
        geometry_regret = baselines["per_layout"]["geometry_waypoint"]
        coast_regret = baselines["per_layout"]["constant_coast"]
        improvement_geometry = geometry_regret - model_regret
        improvement_coast = coast_regret - model_regret
        geo_lower, geo_upper = _one_sided_cluster_bounds(improvement_geometry, seed=133702)
        coast_lower, coast_upper = _one_sided_cluster_bounds(improvement_coast, seed=133703)
        passed = bool(
            validation_metrics["layout_macro_argmax_heldout_regret"] <= promotion["mean_regret_maximum"]
            and validation_metrics["regret_one_sided_98_75_upper"] <= promotion["one_sided_98_75_regret_upper_maximum"]
            and validation_metrics["mean_improvement_over_geometry"] >= promotion["improvement_over_geometry_minimum"]
            and validation_metrics["improvement_one_sided_98_75_lower"] > promotion["one_sided_98_75_improvement_lower_minimum"]
            and permutation["max_probability_deviation"] <= promotion["permutation_max_probability_deviation_maximum"]
            and permutation["semantic_choice_flips"] <= promotion["semantic_choice_flips_maximum"]
        )
        variant_results[variant] = {
            "selected_fit_seed": int(summary["fit_seed"]),
            "parameter_count": model_parameter_count(model),
            "validation_metrics": validation_metrics,
            "fitting_metrics": fitting_metrics,
            "baseline_metrics": baselines["summary"],
            "improvement_over_geometry_one_sided_98_75": [geo_lower, geo_upper],
            "improvement_over_coast_one_sided_98_75": [coast_lower, coast_upper],
            "validation_action_choice_counts": np.bincount(chosen, minlength=9).tolist(),
            "permutation": permutation,
            "promotion_gate_pass": passed,
            "checkpoint_sha256": sha256(checkpoint_path),
        }

    np.savez_compressed(output / "state_normalizer.npz", mean=mean, scale=scale)
    elapsed = time.monotonic() - started
    result = {
        "schema": "phase12cD1_learnability_result_v1",
        "config_sha256": sha256(config_path),
        "fold_lock_sha256": sha256(fold_lock_path),
        "training_manifest_sha256": sha256(manifest_path),
        "training_layouts": len(training_hashes),
        "internal_validation_layouts": len(validation_hashes),
        "training_queries": len(train_rows),
        "internal_validation_queries": len(validation_rows),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_used": False,
        "elapsed_wall_seconds": elapsed,
        "variants": variant_results,
        "all_fit_runs": fit_runs,
        "d1_advance": any(item["promotion_gate_pass"] for item in variant_results.values()),
        "new_rollouts_or_labels": False,
        "old_phase12cC_selection_gate_read": False,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema": "phase12cD1_run_manifest_v1",
        "result_sha256": sha256(result_path),
        "checkpoint_sha256": checkpoint_hashes,
        "normalizer_sha256": sha256(output / "state_normalizer.npz"),
        "artifact_bytes": sum(item.stat().st_size for item in output.rglob("*") if item.is_file()),
        "new_rollouts_or_labels": False,
        "old_phase12cC_selection_gate_read": False,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
    }
    manifest_path_out = output / "manifest.json"
    for _ in range(3):
        manifest["artifact_bytes"] = sum(
            item.stat().st_size for item in output.rglob("*") if item.is_file()
        ) + (manifest_path_out.stat().st_size if manifest_path_out.exists() else 0)
        manifest_path_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total_bytes = sum(item.stat().st_size for item in output.rglob("*") if item.is_file())
    if total_bytes > int(stage["resources"]["generated_bytes"]):
        raise OSError("D1 artifact cap exceeded")
    print(json.dumps({
        "status": "phase12cD1_complete",
        "d1_advance": result["d1_advance"],
        "variants": {key: {
            "pass": value["promotion_gate_pass"],
            "regret": value["validation_metrics"]["layout_macro_argmax_heldout_regret"],
            "geometry": value["baseline_metrics"]["geometry_waypoint"]["layout_macro_heldout_regret"],
            "actions": value["validation_action_choice_counts"],
        } for key, value in variant_results.items()},
        "wall_seconds": elapsed,
        "artifact_bytes": total_bytes,
        "old_phase12cC_selection_gate_read": False,
        "reserve_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
