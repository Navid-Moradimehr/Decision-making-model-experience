"""Fresh-seed, development-only horizon diagnostic for a frozen Phase-1 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from consequence_lab.candidates import two_segment_sequences  # noqa: E402
from consequence_lab.objectives import goal_cost  # noqa: E402
from phase1_tworoom_dynamics import (  # noqa: E402
    EPISODE_CAP,
    OBS_SCALE,
    SNAPSHOT_STEPS,
    NumpyPredictor,
    collect_episode,
    load_environment,
    rollout_predictor,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_bootstrap(values: np.ndarray, reps: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return [float("nan"), float("nan")]
    indices = rng.integers(0, len(array), size=(reps, len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def score_group(true_costs, predicted_costs, reachable, tolerance, regret_divisor):
    per_state = []
    for truth, prediction, can_reach in zip(true_costs, predicted_costs, reachable):
        i, j = np.triu_indices(len(truth), 1)
        informative = np.abs(truth[i] - truth[j]) > tolerance / math.sqrt(2 * 224.0**2)
        correct = (prediction[i] - prediction[j]) * (truth[i] - truth[j]) > 0
        selected = int(np.argmin(prediction))
        per_state.append({
            "pairwise_accuracy": float(correct[informative].mean()) if informative.any() else float("nan"),
            "informative_pairs": int(informative.sum()),
            "top1_agreement": float(selected == int(np.argmin(truth))),
            "normalized_regret": float(max(0.0, truth[selected] - np.min(truth)) / regret_divisor),
            "oracle_candidate_available": bool(can_reach.any()),
        })
    return per_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "phase1_horizon_diagnostic.yaml")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "runs" / "phase1_horizon_diagnostic_v1")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = config["project"]["device"]
    if device != "cpu" or config["frozen_model"]["train_or_finetune"]:
        raise ValueError("This diagnostic is frozen-model CPU-only; training is forbidden")
    if config["environment"]["observation_scale_px"] != OBS_SCALE:
        raise ValueError("observation scale differs from the frozen predictor implementation")
    if config["environment"]["max_episode_steps"] != EPISODE_CAP:
        raise ValueError("episode cap differs from the pinned collector implementation")
    if config["environment"]["success_distance_px"] != 16.0:
        raise ValueError("success tolerance differs from the frozen rollout implementation")
    if config["development"]["snapshot_steps"] != list(SNAPSHOT_STEPS):
        raise ValueError("snapshot schedule differs from the preregistered collector")
    if config["frozen_model"]["architecture"] != [12, 128, 128, 10]:
        raise ValueError("checkpoint architecture does not match the pinned predictor")
    if config["planning"]["candidate_count"] != 65 or config["planning"]["execution_steps_before_replanning"] != 1:
        raise ValueError("candidate bank or receding-horizon contract changed")
    if config["planning"]["action_sequence"] != "two_equal_segments_plus_wait":
        raise ValueError("candidate generator contract changed")
    try:
        actual_revision = subprocess.run(
            ["git", "-C", str(args.source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot verify the pinned simulator source revision") from error
    if actual_revision != config["environment"]["revision"]:
        raise ValueError(f"simulator revision mismatch: {actual_revision}")

    checkpoint = ROOT / config["frozen_model"]["checkpoint"]
    if sha256(checkpoint) != config["frozen_model"]["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint hash does not match preregistration")
    start_seed, end_seed = config["development"]["episode_seeds"]
    old_ranges = [(202700, 202900), (202900, 202940), (203000, 203200), (204000, 204110)]
    if any(start_seed < end and start < end_seed for start, end in old_ranges):
        raise ValueError("development seeds overlap a previous protocol")

    torch.set_num_threads(2)
    model = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    predictor = NumpyPredictor(network=model)
    env, adapter = load_environment(args.source_root)
    sequences_by_horizon = {
        horizon: two_segment_sequences(horizon)
        for horizon in config["planning"]["horizons_steps"]
    }
    if any(len(sequences) != config["planning"]["candidate_count"] for sequences in sequences_by_horizon.values()):
        raise ValueError("generated candidate count differs from preregistration")

    episodes = []
    snapshots = []
    if config["development"]["policy_mix_by_episode"] != "alternating_noisy_geometric_and_persistent_random":
        raise ValueError("unsupported development policy mix")
    if config["development"]["persistent_random_resample_steps"] != [1, 8]:
        raise ValueError("persistent-random resampling differs from the pinned collector")
    for offset, episode_seed in enumerate(range(start_seed, end_seed)):
        mode = "noisy_geometric" if offset % 2 == 0 else "persistent_random"
        episode = collect_episode(adapter, episode_seed, mode, keep_snapshots=True)
        episodes.append({"seed": episode_seed, "steps": len(episode["actions"]), "mode": mode})
        snapshots.extend((episode_seed, step, obs, snapshot) for step, obs, snapshot in episode["snapshots"])

    per_horizon = {}
    detailed_true_costs = []
    detailed_predicted_costs = []
    detailed_reachable = []
    detailed_seed = []
    detailed_step = []
    started = time.perf_counter()
    for horizon, sequences in sequences_by_horizon.items():
        true_rows, predicted_rows, reachable_rows, seeds, steps = [], [], [], [], []
        for episode_seed, step, observation, snapshot in snapshots:
            true_positions = adapter.rollout_candidates(snapshot, sequences)
            predicted_positions = rollout_predictor(predictor, observation, sequences)
            true_cost = goal_cost(
                true_positions, observation[2:4], math.sqrt(2.0) * config["environment"]["observation_scale_px"],
                terminal_distance_weight=config["planning"]["objective"]["terminal_distance_weight"],
                mean_distance_weight=config["planning"]["objective"]["mean_distance_weight"],
                success_distance=config["environment"]["success_distance_px"],
            )
            predicted_cost = goal_cost(
                predicted_positions, observation[2:4], math.sqrt(2.0) * config["environment"]["observation_scale_px"],
                terminal_distance_weight=config["planning"]["objective"]["terminal_distance_weight"],
                mean_distance_weight=config["planning"]["objective"]["mean_distance_weight"],
                success_distance=config["environment"]["success_distance_px"],
            )
            reachable = (np.linalg.norm(true_positions - observation[2:4], axis=-1) < config["environment"]["success_distance_px"]).any(axis=1)
            true_rows.append(true_cost.astype(np.float32))
            predicted_rows.append(predicted_cost.astype(np.float32))
            reachable_rows.append(reachable)
            seeds.append(episode_seed)
            steps.append(step)

        rows = score_group(
            true_rows,
            predicted_rows,
            reachable_rows,
            config["planning"]["ranking_tie_tolerance_px"],
            config["planning"]["normalized_regret_divisor"],
        )
        by_seed = {}
        for seed, row in zip(seeds, rows):
            by_seed.setdefault(seed, []).append(row)
        seed_metrics = []
        for seed, values in by_seed.items():
            pair = [v["pairwise_accuracy"] for v in values if np.isfinite(v["pairwise_accuracy"])]
            seed_metrics.append({
                "seed": seed,
                "pairwise_accuracy": float(np.mean(pair)) if pair else float("nan"),
                "normalized_regret": float(np.mean([v["normalized_regret"] for v in values])),
                "oracle_candidate_availability": float(np.mean([v["oracle_candidate_available"] for v in values])),
                "top1_agreement": float(np.mean([v["top1_agreement"] for v in values])),
            })
        per_horizon[str(horizon)] = {
            "states": len(rows),
            "pairwise_accuracy_macro": float(np.nanmean([r["pairwise_accuracy"] for r in rows])),
            "pairwise_accuracy_seed_cluster_ci95": seed_bootstrap(np.asarray([r["pairwise_accuracy"] for r in seed_metrics]), config["statistics"]["bootstrap_replicates"], config["statistics"]["bootstrap_seed"] + horizon),
            "normalized_regret_mean": float(np.mean([r["normalized_regret"] for r in rows])),
            "normalized_regret_seed_cluster_ci95": seed_bootstrap(np.asarray([r["normalized_regret"] for r in seed_metrics]), config["statistics"]["bootstrap_replicates"], config["statistics"]["bootstrap_seed"] + 100 + horizon),
            "oracle_candidate_availability": float(np.mean([r["oracle_candidate_available"] for r in rows])),
            "top1_agreement": float(np.mean([r["top1_agreement"] for r in rows])),
            "per_seed": seed_metrics,
        }
        detailed_true_costs.append(np.asarray(true_rows, dtype=np.float32))
        detailed_predicted_costs.append(np.asarray(predicted_rows, dtype=np.float32))
        detailed_reachable.append(np.asarray(reachable_rows, dtype=np.bool_))
        detailed_seed = seeds
        detailed_step = steps
    env.close()

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "candidate_costs.npz",
        horizons=np.asarray(list(sequences_by_horizon), dtype=np.int32),
        candidate_id=np.arange(config["planning"]["candidate_count"], dtype=np.int32),
        true_costs=np.stack(detailed_true_costs, axis=0),
        predicted_costs=np.stack(detailed_predicted_costs, axis=0),
        oracle_reachable=np.stack(detailed_reachable, axis=0),
        episode_seed=np.asarray(detailed_seed, dtype=np.int64),
        step_index=np.asarray(detailed_step, dtype=np.int32),
    )
    summary = {
        "protocol": config["project"]["protocol"],
        "config_sha256": sha256(args.config),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_revision": config["environment"]["revision"],
        "checkpoint_sha256": sha256(checkpoint),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": "cpu"},
        "episodes": episodes,
        "ranking_states": len(snapshots),
        "horizons": per_horizon,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "interpretation": "Development diagnostic only. No training, teacher labels, student, or test-set claim.",
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "run_id": args.output.name,
        "phase": "phase_1_diagnostic",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "config_sha256": summary["config_sha256"],
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": summary["runner_sha256"],
        "source_revision": summary["source_revision"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "summary_sha256": sha256(summary_path),
        "costs_sha256": sha256(args.output / "candidate_costs.npz"),
        "resources": {"device": "cpu", "wall_seconds": summary["elapsed_seconds"], "peak_vram_bytes": 0},
        "notes": ["Seed-cluster bootstrap intervals are descriptive; horizon choice requires independent confirmation."],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"horizons": per_horizon, "ranking_states": len(snapshots), "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
