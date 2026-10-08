"""Recompute explicitly post-hoc paired and selected-action diagnostics."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np


def mean_ci(values: np.ndarray, *, seed: int, replicates: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def analyze(rows: list[dict]) -> dict:
    groups = ("uniform_random", "reference_perturbation")
    differences = {}
    selected = {}
    availability = {}
    for group in groups:
        differences[group] = {}
        for metric in (
            "endpoint_latent_coordinate_rmse",
            "predicted_latent_goal_vs_qpos_pairwise_accuracy",
        ):
            delta = np.asarray(
                [
                    row["groups"][group]["methods"]["mlp"]["metrics"][metric]
                    - row["groups"][group]["methods"]["native_lewm"]["metrics"][metric]
                    for row in rows
                ],
                dtype=np.float64,
            )
            differences[group][metric] = {
                "mlp_minus_native_mean": float(delta.mean()),
                "paired_seed_bootstrap_95_ci": mean_ci(
                    delta, seed=404030 + len(metric)
                ),
            }

        group_availability = 0
        selected[group] = {}
        for method in ("mlp", "native_lewm", "ridge", "persistence"):
            success_values, regret_values = [], []
            for row in rows:
                data = row["groups"][group]
                costs = np.asarray(data["qpos_cost"], dtype=np.float64)
                success = np.asarray(data["success_bits"], dtype=bool)
                predicted = np.asarray(
                    data["methods"][method]["predicted_goal_cost"], dtype=np.float64
                )
                selected_index = int(np.argmin(predicted))
                success_values.append(float(success[selected_index]))
                regret_values.append(float(max(0.0, costs[selected_index] - costs.min())))
            success_array = np.asarray(success_values)
            regret_array = np.asarray(regret_values)
            selected[group][method] = {
                "selected_success_fraction": float(success_array.mean()),
                "selected_success_seed_bootstrap_95_ci": mean_ci(
                    success_array, seed=404040 + len("selected_success")
                ),
                "mean_selected_normalized_regret": float(regret_array.mean()),
                "selected_regret_seed_bootstrap_95_ci": mean_ci(
                    regret_array, seed=404040 + len("selected_regret")
                ),
            }
        episode_success = [
            any(row["groups"][group]["success_bits"]) for row in rows
        ]
        group_availability = sum(episode_success)
        availability[group] = {
            "episodes_with_any_success": int(group_availability),
            "episode_count": len(rows),
            "candidate_success_count": int(
                sum(sum(row["groups"][group]["success_bits"]) for row in rows)
            ),
            "candidate_count": int(
                sum(len(row["groups"][group]["success_bits"]) for row in rows)
            ),
        }
    return {
        "status": "explicitly_posthoc_descriptive_analysis",
        "episodes": len(rows),
        "paired_model_differences": differences,
        "selected_candidate_diagnostics": selected,
        "candidate_availability": availability,
    }


def unpaired_run_difference(baseline_rows: list[dict], scaled_rows: list[dict]) -> dict:
    """Descriptive difference between disjoint test splits; not a paired test."""
    result = {}
    for group_index, group in enumerate(("uniform_random", "reference_perturbation")):
        result[group] = {}
        for metric_index, metric in enumerate(
            (
                "endpoint_latent_coordinate_rmse",
                "predicted_latent_goal_vs_qpos_pairwise_accuracy",
            )
        ):
            base = np.asarray(
                [row["groups"][group]["methods"]["mlp"]["metrics"][metric] for row in baseline_rows],
                dtype=np.float64,
            )
            scaled = np.asarray(
                [row["groups"][group]["methods"]["mlp"]["metrics"][metric] for row in scaled_rows],
                dtype=np.float64,
            )
            rng = np.random.default_rng(404200 + group_index * 20 + metric_index)
            base_indices = rng.integers(0, len(base), size=(5000, len(base)))
            scaled_indices = rng.integers(0, len(scaled), size=(5000, len(scaled)))
            draws = scaled[scaled_indices].mean(axis=1) - base[base_indices].mean(axis=1)
            result[group][metric] = {
                "scaled_minus_original_mlp_mean": float(scaled.mean() - base.mean()),
                "independent_seed_bootstrap_95_ci": [
                    float(np.quantile(draws, 0.025)),
                    float(np.quantile(draws, 0.975)),
                ],
                "baseline_test_episodes": len(base),
                "scaled_test_episodes": len(scaled),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "runs"
        / "phase2_reacher_predictor_fit",
    )
    parser.add_argument("--baseline-run-dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    source = root / "test_progress.jsonl"
    destination = root / (
        "posthoc_scaling_comparison.json"
        if args.baseline_run_dir
        else "posthoc_analysis.json"
    )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    result = analyze(rows)
    if args.baseline_run_dir:
        baseline_source = args.baseline_run_dir.resolve() / "test_progress.jsonl"
        baseline_rows = [
            json.loads(line) for line in baseline_source.read_text(encoding="utf-8").splitlines()
        ]
        result["cross_run_comparison"] = {
            "status": "posthoc_unpaired_disjoint_test_seed_comparison",
            "baseline_test_split": str(args.baseline_run_dir.resolve()),
            "scaled_test_split": str(root),
            "metrics": unpaired_run_difference(baseline_rows, rows),
        }
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Post-hoc analysis: {destination}")


if __name__ == "__main__":
    main()
