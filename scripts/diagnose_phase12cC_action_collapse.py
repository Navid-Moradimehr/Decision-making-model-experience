"""Read-only audit of the frozen Phase 12c-C constant-action collapse.

This diagnostic may read only the already-spent Phase 12c-C fitting,
model-selection, and offline-gate datasets and the two frozen checkpoints.
It performs no fitting, rollout, relabeling, or access to development/DAgger/
reserve data. It writes a descriptive JSON report under reports/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_model import CorridorCandidateScorer, model_parameter_count
from consequence_lab.phase12c_c_training import load_dataset, score_model, sha256


SPLITS = {
    "fit": "artifacts/runs/phase12cC_archived_training_v1",
    "selection": "artifacts/runs/phase12cC_model_selection_v1",
    "offline_gate": "artifacts/runs/phase12cC_offline_gate_v1",
}
CHECKPOINTS = {
    "direct_candidate_value": (
        "artifacts/runs/phase12cC_selected_models_v3/checkpoints/direct_candidate_value_seed1321.pt",
        "052F020154B4FA569AA7D33366C24290A2754E2E2BEC41A5CE28E177C67B52B4",
        "value",
    ),
    "bounded_preference_policy": (
        "artifacts/runs/phase12cC_selected_models_v3/checkpoints/bounded_preference_policy_seed1322.pt",
        "46DBE9124EFB4568C3B3261F035010163C41BED724A8DB916972FF74CDBA0CBF",
        "policy",
    ),
}
FEATURE_NAMES = [
    "agent_x_norm", "agent_y_norm", "velocity_x_norm", "velocity_y_norm",
    "world_width_raw", "world_height_raw", "goal_x_norm", "goal_y_norm",
    "goal_radius_raw", "safe_speed_norm", "agent_radius_raw",
    "acceleration_norm", "damping_raw", "max_speed_raw", "dt_raw",
    "remaining_horizon_fraction",
]
for _obstacle in range(9):
    FEATURE_NAMES.extend([
        f"obstacle{_obstacle}_xmin_norm", f"obstacle{_obstacle}_ymin_norm",
        f"obstacle{_obstacle}_xmax_norm", f"obstacle{_obstacle}_ymax_norm",
    ])


def verify_dataset(directory: Path, expected_split: str) -> dict:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != expected_split:
        raise SystemExit(f"unexpected split label in {manifest_path}")
    if manifest.get("reserve_accessed") is not False:
        raise SystemExit(f"dataset manifest does not attest sealed reserve: {manifest_path}")
    for name, entry in manifest["files"].items():
        path = directory / name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise SystemExit(f"dataset digest mismatch: {path}")
    return manifest


def layout_means(data: dict) -> np.ndarray:
    x = data["inputs"]["states"].astype(np.float64)
    names = np.asarray([str(row["layout_sha256"]) for row in data["metadata"]])
    unique = np.unique(names)
    return np.stack([x[names == name].mean(axis=0) for name in unique])


def feature_shift(reference: np.ndarray, other: np.ndarray) -> dict:
    # Equal-weight each layout to match the layout-level experimental unit.
    train_layout = layout_means(reference)
    other_layout = layout_means(other)
    mean = train_layout.mean(axis=0)
    std = train_layout.std(axis=0, ddof=1)
    safe_std = np.where(std > 1e-10, std, 1.0)
    standardized_delta = (other_layout.mean(axis=0) - mean) / safe_std
    # Range violations are query-level descriptive coverage diagnostics.
    train_x = reference["inputs"]["states"].astype(np.float64)
    other_x = other["inputs"]["states"].astype(np.float64)
    low, high = train_x.min(axis=0), train_x.max(axis=0)
    outside = (other_x < low) | (other_x > high)
    max_z = np.max(np.abs((other_x - mean) / safe_std), axis=0)
    top = np.argsort(np.abs(standardized_delta))[::-1][:12]
    return {
        "query_count": int(len(other_x)),
        "layout_count": int(len(other_layout)),
        "per_feature_layout_macro_standardized_mean_delta": {
            FEATURE_NAMES[i]: float(standardized_delta[i]) for i in range(len(FEATURE_NAMES))
        },
        "largest_absolute_layout_macro_deltas": [
            {"feature": FEATURE_NAMES[i], "standardized_delta": float(standardized_delta[i])}
            for i in top
        ],
        "query_fraction_outside_fit_feature_range": {
            FEATURE_NAMES[i]: float(outside[:, i].mean()) for i in range(len(FEATURE_NAMES))
        },
        "features_with_any_range_violation": int(np.count_nonzero(outside.any(axis=0))),
        "query_fraction_with_any_feature_outside_fit_range": float(outside.any(axis=1).mean()),
        "max_absolute_fit_standardized_value_by_feature": {
            FEATURE_NAMES[i]: float(max_z[i]) for i in range(len(FEATURE_NAMES))
        },
        "selected_feature_query_quantiles": {
            FEATURE_NAMES[i]: {
                "fit": np.quantile(train_x[:, i], [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
                "heldout": np.quantile(other_x[:, i], [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
            }
            for i in (0, 1, 2, 3, 15)
        },
    }


def action_target_summary(data: dict, split: str) -> dict:
    targets = data["targets"]
    if split == "fit":
        regrets = targets["mean_regret_target"].astype(np.float64)
        preferences = targets["preference_target"].astype(np.float64)
    else:
        # First six proposal banks are used to estimate a modal action;
        # the second six are used only to measure its held-out regret.
        regrets = targets["per_bank_normalized_regret"][:, 6:12, :].mean(axis=1)
        preferences = targets["per_bank_tie_split_optimal_mass"][:, 0:6, :].mean(axis=1)
    modal = preferences.argmax(axis=1)
    per_action_regret = regrets.mean(axis=0)
    modal_regret = regrets[np.arange(len(modal)), modal]
    layout_ids = np.asarray([str(row["layout_sha256"]) for row in data["metadata"]])
    unique_layouts = np.unique(layout_ids)
    per_layout_action_regret = np.stack([
        regrets[layout_ids == layout].mean(axis=0) for layout in unique_layouts
    ])
    per_layout_modal_regret = np.asarray([
        modal_regret[layout_ids == layout].mean() for layout in unique_layouts
    ])
    return {
        "modal_action_counts_from_preference_targets": np.bincount(modal, minlength=9).tolist(),
        "modal_action_fraction": (np.bincount(modal, minlength=9) / len(modal)).tolist(),
        "mean_regret_if_fixed_action": per_action_regret.tolist(),
        "lowest_mean_regret_fixed_action": int(per_action_regret.argmin()),
        "lowest_fixed_action_mean_regret": float(per_action_regret.min()),
        "layout_macro_regret_if_fixed_action": per_layout_action_regret.mean(axis=0).tolist(),
        "layout_macro_regret_if_action0": float(per_layout_action_regret.mean(axis=0)[0]),
        "per_query_modal_action_mean_regret": float(modal_regret.mean()),
        "layout_macro_per_query_modal_action_regret": float(per_layout_modal_regret.mean()),
        "action0_mean_regret": float(per_action_regret[0]),
        "mean_regret_advantage_modal_over_action0": float(per_action_regret[0] - modal_regret.mean()),
    }


def checkpoint_action_summary(data: dict, scores: np.ndarray, variant: str) -> dict:
    choice = scores.argmax(axis=1) if variant == "policy" else scores.argmin(axis=1)
    chosen_score = scores[np.arange(len(scores)), choice]
    order = np.sort(scores, axis=1)
    if variant == "policy":
        margin = order[:, -1] - order[:, -2]
    else:
        margin = order[:, 1] - order[:, 0]
    return {
        "choice_counts": np.bincount(choice, minlength=9).tolist(),
        "unique_actions_selected": int(np.unique(choice).size),
        "per_candidate_mean_raw_score": scores.mean(axis=0).tolist(),
        "per_candidate_std_raw_score": scores.std(axis=0).tolist(),
        "within_query_score_range_mean": float((scores.max(axis=1) - scores.min(axis=1)).mean()),
        "within_query_score_range_median": float(np.median(scores.max(axis=1) - scores.min(axis=1))),
        "selected_vs_runnerup_raw_score_margin_mean": float(margin.mean()),
        "selected_vs_runnerup_raw_score_margin_median": float(np.median(margin)),
        "selected_score_std_across_queries": float(chosen_score.std()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/phase12cC_constant_action_diagnostic.json")
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise SystemExit("diagnostic must run CPU-only")
    config_path = ROOT / "configs/phase12cC_locked.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    datasets = {}
    manifest_hashes = {}
    for split, rel in SPLITS.items():
        path = ROOT / rel
        manifest_name = {"fit": "archived_training", "selection": "model_selection",
                         "offline_gate": "offline_gate"}[split]
        manifest = verify_dataset(path, manifest_name)
        datasets[split] = load_dataset(path)
        manifest_hashes[split] = sha256(path / "manifest.json")
        if manifest.get("closed_loop_development_accessed") is not False:
            raise SystemExit(f"unexpected development access flag in {split}")

    checkpoint_results = {}
    for label, (rel, expected_hash, kind) in CHECKPOINTS.items():
        checkpoint_path = ROOT / rel
        if sha256(checkpoint_path) != expected_hash:
            raise SystemExit(f"frozen checkpoint hash mismatch: {rel}")
        model = CorridorCandidateScorer(
            kind,
            hidden_state=int(config["models"]["common"]["hidden_state"]),
            embedding=int(config["models"]["common"]["embedding"]),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        if model_parameter_count(model) != 32705:
            raise SystemExit("unexpected frozen model parameter count")
        per_split = {}
        for split, data in datasets.items():
            scores, _ = score_model(model, data, variant=label, device=torch.device("cpu"))
            per_split[split] = checkpoint_action_summary(data, scores, kind)
        checkpoint_results[label] = {
            "checkpoint_sha256": expected_hash,
            "parameter_count": model_parameter_count(model),
            "per_split": per_split,
        }

    shift_results = {
        split: feature_shift(datasets["fit"], datasets[split])
        for split in ("selection", "offline_gate")
    }
    result = {
        "schema": "phase12cC_read_only_constant_action_diagnostic_v1",
        "config_sha256": sha256(config_path),
        "diagnostic_script_sha256": sha256(Path(__file__).resolve()),
        "scope": {
            "purpose": "diagnose state-distribution shift and action-score collapse",
            "read_splits": list(SPLITS),
            "read_checkpoints": list(CHECKPOINTS),
            "training_performed": False,
            "rollouts_performed": False,
            "new_labels_generated": False,
            "development_accessed": False,
            "dagger_accessed": False,
            "reserve_accessed": False,
            "cuda_used": False,
        },
        "dataset_manifest_sha256": manifest_hashes,
        "fit_to_heldout_state_feature_shift": shift_results,
        "teacher_action_landscape": {
            split: action_target_summary(data, split) for split, data in datasets.items()
        },
        "frozen_model_action_scores": checkpoint_results,
        "interpretive_limits": [
            "Feature-range and standardized-mean diagnostics do not establish causal distribution shift.",
            "State-only offline scoring does not establish closed-loop behavior.",
            "The teacher modal action is an empirical proposal-bank reference, not an oracle deployable policy.",
        ],
    }
    output = ROOT / args.output
    if not output.resolve().is_relative_to((ROOT / "reports").resolve()):
        raise SystemExit("output must remain under reports/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "read_only_diagnostic_complete",
        "output": str(output.relative_to(ROOT)),
        "files_read": [*SPLITS.values(), *[item[0] for item in CHECKPOINTS.values()]],
        "training_performed": False,
        "rollouts_performed": False,
        "reserve_accessed": False,
        "offline_gate_constant_action_count": {
            name: values["per_split"]["offline_gate"]["choice_counts"]
            for name, values in checkpoint_results.items()
        },
        "fit_to_gate_any_feature_range_violation_fraction":
            shift_results["offline_gate"]["query_fraction_with_any_feature_outside_fit_range"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
