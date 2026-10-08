"""Decompose LeWM endpoint prediction and latent-goal objective alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import types

import gymnasium as gym
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.latent_diagnostic import (  # noqa: E402
    make_candidate_bank,
    tie_aware_pairwise_accuracy,
)
from scripts.run_phase2_reacher_ranking_dev import (  # noqa: E402
    load_frozen_model,
    namespace,
    preprocess_frame,
    replay_goal,
    sha256,
)


def encode_frames(model, frames: list[np.ndarray], batch_size: int) -> np.ndarray:
    vectors = []
    for start in range(0, len(frames), batch_size):
        images = torch.stack(
            [preprocess_frame(frame) for frame in frames[start : start + batch_size]]
        ).unsqueeze(1)
        with torch.inference_mode():
            encoded = model.encode({"pixels": images})["emb"][:, -1]
        vectors.append(encoded.cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)


def predict_latents(
    model,
    current_frame: np.ndarray,
    actions: np.ndarray,
    *,
    rolling_window: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_image = preprocess_frame(current_frame).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        current_embedding = model.encode({"pixels": current_image})["emb"]
    predictions = []
    for start in range(0, len(actions), batch_size):
        action_batch = torch.as_tensor(
            actions[start : start + batch_size], dtype=torch.float32
        ).reshape(-1, 5, 10)
        count = action_batch.shape[0]
        info = {
            "pixels": current_image.unsqueeze(1).expand(
                1, count, *current_image.shape[1:]
            ),
            "emb": current_embedding.unsqueeze(1).expand(
                1, count, *current_embedding.shape[1:]
            ),
        }
        with torch.inference_mode():
            prediction = model.rollout(
                info,
                action_batch.unsqueeze(0),
                history_size=rolling_window,
            )["predicted_emb"][:, :, -1]
        predictions.append(prediction.squeeze(0).cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32), current_embedding.cpu().numpy()


def simulate_endpoints(
    env,
    *,
    seed: int,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    sentinel_offset: list[float],
    actions: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    frames: list[np.ndarray] = []
    endpoint_qpos = np.empty((len(actions), len(start_qpos)), dtype=np.float64)
    sentinel_target = start_qpos + np.asarray(sentinel_offset, dtype=np.float64)
    for index, sequence in enumerate(actions):
        env.reset(seed=seed)
        env.unwrapped.set_state(start_qpos, start_qvel)
        env.unwrapped.set_target_qpos(sentinel_target)
        info = env.unwrapped.info
        for action in sequence:
            _, _, terminated, truncated, info = env.step(
                action.astype(env.action_space.dtype)
            )
            if terminated or truncated:
                raise RuntimeError(
                    "sentinel-target diagnostic branch terminated before the fixed horizon"
                )
        endpoint_qpos[index] = np.asarray(info["qpos"], dtype=np.float64)
        frames.append(env.unwrapped.render(width=224, height=224))
    return frames, endpoint_qpos


def bootstrap_ci(values: list[float], *, replicates: int, seed: int):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_group_rows(rows: list[dict], config: dict) -> dict:
    names = ("uniform_random", "reference_perturbation")
    result = {}
    reps = config["measurements"]["bootstrap_replicates"]
    base_seed = config["measurements"]["bootstrap_seed"]
    metric_names = (
        "candidate_success_fraction",
        "prediction_coordinate_rmse",
        "prediction_cosine_similarity",
        "predicted_vs_realized_latent_cost_pairwise",
        "predicted_latent_cost_vs_physical_pairwise",
        "realized_latent_cost_vs_physical_pairwise",
    )
    for group_index, name in enumerate(names):
        values_by_metric = {metric: [] for metric in metric_names}
        defined_states = {metric: 0 for metric in metric_names}
        for row in rows:
            group = row["groups"][name]
            values_by_metric["candidate_success_fraction"].append(
                float(np.mean(group["success_bits"]))
            )
            for metric in metric_names[1:]:
                value = group["episode_metrics"][metric]
                if value is not None and np.isfinite(value):
                    values_by_metric[metric].append(float(value))
                    defined_states[metric] += 1
        group_summary = {}
        for metric_index, metric in enumerate(metric_names):
            values = values_by_metric[metric]
            group_summary[metric] = {
                "mean_equal_episode_weight": float(np.mean(values)) if values else None,
                "cluster_bootstrap_95_ci": bootstrap_ci(
                    values,
                    replicates=reps,
                    seed=base_seed + group_index * 20 + metric_index,
                ),
                "episodes_with_defined_value": len(values),
            }
        group_summary["candidate_availability"] = {
            "episodes_with_at_least_one_success": sum(
                bool(row["groups"][name]["success_bits"].count(True))
                for row in rows
            ),
            "episodes": len(rows),
        }
        result[name] = group_summary
    return result


def run_one_episode(env, model, config: dict, seed: int) -> dict:
    environment = config["environment"]
    bank_config = config["candidate_bank"]
    start_qpos, start_qvel, start_frame, goal_qpos, goal_frame = replay_goal(
        env,
        seed,
        seed + environment["goal_action_seed_offset"],
        environment["sentinel_qpos_offset"],
    )
    reference_actions = np.random.default_rng(
        seed + environment["goal_action_seed_offset"]
    ).uniform(-1.0, 1.0, size=(25, 2))
    banks = make_candidate_bank(
        reference_actions,
        uniform_seed=seed + bank_config["uniform_random"]["seed_offset"],
        perturbation_seed=seed + bank_config["reference_perturbation"]["seed_offset"],
        uniform_count=bank_config["uniform_random"]["count"],
        perturbation_count=bank_config["reference_perturbation"]["count"],
        perturbation_std=bank_config["reference_perturbation"]["standard_deviation"],
        lower_bound=bank_config["uniform_random"]["bounds"][0],
        upper_bound=bank_config["uniform_random"]["bounds"][1],
    )

    current_image = preprocess_frame(start_frame).unsqueeze(0).unsqueeze(0)
    goal_image = preprocess_frame(goal_frame).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        goal_embedding = model.encode({"pixels": goal_image})["emb"][:, -1]
    goal_vector = goal_embedding.squeeze(0).cpu().numpy().astype(np.float32)

    group_results = {}
    batch_size = config["runtime"]["candidate_batch_size"]
    for group_name in config["candidate_bank"]["stable_order"]:
        candidates = banks[group_name]
        predicted, _ = predict_latents(
            model,
            start_frame,
            candidates,
            rolling_window=config["planning"]["predictor_rolling_window"],
            batch_size=batch_size,
        )
        end_frames, end_qpos = simulate_endpoints(
            env,
            seed=seed,
            start_qpos=start_qpos,
            start_qvel=start_qvel,
            sentinel_offset=environment["sentinel_qpos_offset"],
            actions=candidates,
        )
        realized = encode_frames(model, end_frames, batch_size)
        if predicted.shape != realized.shape:
            raise RuntimeError(
                f"predicted/realized latent shape mismatch: {predicted.shape} vs {realized.shape}"
            )
        if goal_vector.shape != predicted.shape[1:]:
            raise RuntimeError(
                f"goal latent shape {goal_vector.shape} does not match candidate latent {predicted.shape[1:]}"
            )

        forecast_rmse = np.sqrt(np.mean((predicted - realized) ** 2, axis=1))
        norms = np.linalg.norm(predicted, axis=1) * np.linalg.norm(realized, axis=1)
        cosine = np.divide(
            (predicted * realized).sum(axis=1),
            norms,
            out=np.zeros(len(norms), dtype=np.float64),
            where=norms > 0,
        )
        predicted_goal_cost = np.square(predicted - goal_vector).sum(axis=1)
        realized_goal_cost = np.square(realized - goal_vector).sum(axis=1)
        qpos_error = np.abs(end_qpos - goal_qpos)
        physical_cost = qpos_error.mean(axis=1) / (2.0 * np.pi)
        success = np.all(qpos_error < 0.05, axis=1)
        physical_cost[success] = 0.0

        pairwise_config = config["measurements"]
        pred_to_realized_rank, pred_to_realized_pairs = tie_aware_pairwise_accuracy(
            predicted_goal_cost,
            realized_goal_cost,
            actual_tie_tolerance=pairwise_config["latent_cost_tie_tolerance"],
            predicted_tie_tolerance=pairwise_config["predicted_cost_tie_tolerance"],
        )
        pred_to_physical_rank, pred_physical_pairs = tie_aware_pairwise_accuracy(
            predicted_goal_cost,
            physical_cost,
            actual_tie_tolerance=pairwise_config["physical_cost_tie_tolerance"],
            predicted_tie_tolerance=pairwise_config["predicted_cost_tie_tolerance"],
        )
        realized_to_physical_rank, realized_physical_pairs = tie_aware_pairwise_accuracy(
            realized_goal_cost,
            physical_cost,
            actual_tie_tolerance=pairwise_config["physical_cost_tie_tolerance"],
            predicted_tie_tolerance=pairwise_config["latent_cost_tie_tolerance"],
        )
        group_results[group_name] = {
            "candidate_sequences": candidates.tolist(),
            "predicted_endpoint_latents": predicted.tolist(),
            "realized_endpoint_latents": realized.tolist(),
            "endpoint_qpos": end_qpos.tolist(),
            "prediction_coordinate_rmse": forecast_rmse.tolist(),
            "prediction_cosine_similarity": cosine.tolist(),
            "predicted_latent_goal_cost": predicted_goal_cost.tolist(),
            "realized_latent_goal_cost": realized_goal_cost.tolist(),
            "physical_cost": physical_cost.tolist(),
            "success_bits": success.tolist(),
            "episode_metrics": {
                "prediction_coordinate_rmse": float(forecast_rmse.mean()),
                "prediction_cosine_similarity": float(cosine.mean()),
                "predicted_vs_realized_latent_cost_pairwise": pred_to_realized_rank,
                "predicted_latent_cost_vs_physical_pairwise": pred_to_physical_rank,
                "realized_latent_cost_vs_physical_pairwise": realized_to_physical_rank,
                "pair_counts": {
                    "predicted_vs_realized_latent_cost": pred_to_realized_pairs,
                    "predicted_latent_vs_physical": pred_physical_pairs,
                    "realized_latent_vs_physical": realized_physical_pairs,
                },
            },
        }
    return {
        "seed": seed,
        "start_qpos": start_qpos.tolist(),
        "start_qvel": start_qvel.tolist(),
        "goal_qpos": goal_qpos.tolist(),
        "groups": group_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one out-of-protocol seed only")
    args = parser.parse_args()
    config_path = ROOT / "configs" / "phase2_reacher_latent_diagnostic.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = Path(os.environ["STABLE_WORLD_MODEL_SOURCE"]).resolve()
    checkpoint_dir = Path(os.environ["LEWM_REACHER_CHECKPOINT"]).resolve()
    output_root = Path(
        os.environ.get(
            "PHASE2_DIAGNOSTIC_OUTPUT_DIR",
            ROOT / "artifacts" / "runs" / "phase2_reacher_latent_diagnostic",
        )
    ).resolve()
    if args.smoke:
        config["environment"]["episode_seed_range"] = [220902, 220903]
        for key in ("uniform_random", "reference_perturbation"):
            config["candidate_bank"][key]["count"] = 3
        config["measurements"]["bootstrap_replicates"] = 20
        output_root = output_root / "smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.jsonl"
    summary_path = output_root / "summary.json"
    if progress_path.exists() or summary_path.exists():
        raise FileExistsError(f"Refusing to reuse diagnostic output at {output_root}")

    weights_path = checkpoint_dir / "weights.pt"
    if sha256(weights_path) != config["source"]["checkpoint_sha256"]:
        raise ValueError("checkpoint weight hash differs from frozen config")
    if sha256(checkpoint_dir / "config.json") != config["source"]["checkpoint_config_sha256"]:
        raise ValueError("checkpoint config hash differs from frozen config")

    torch.set_num_threads(2)
    torch.manual_seed(0)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU-only diagnostic")
    model = load_frozen_model(source_root, checkpoint_dir)
    sys.path.insert(0, str(source_root))
    dmcontrol_path = source_root / "stable_worldmodel" / "envs" / "dmcontrol"
    namespace("stable_worldmodel.envs.dmcontrol", dmcontrol_path)
    import stable_worldmodel.envs  # noqa: F401 - register exact pinned env ID

    env = gym.make(config["environment"]["id"], task=config["environment"]["task"])
    seed_range = config["environment"]["episode_seed_range"]
    rows = []
    started = time.perf_counter()
    try:
        with progress_path.open("x", encoding="utf-8") as stream:
            for seed in range(*seed_range):
                row = run_one_episode(env, model, config, seed)
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if progress_path.stat().st_size > config["artifacts"]["stop_bytes"]:
                    raise RuntimeError("diagnostic artifact stop limit exceeded")
                random_success = sum(row["groups"]["uniform_random"]["success_bits"])
                perturbed_success = sum(
                    row["groups"]["reference_perturbation"]["success_bits"]
                )
                print(
                    f"seed={seed} random_success={random_success} "
                    f"reference_perturbation_success={perturbed_success} "
                    f"random_latent_rmse="
                    f"{row['groups']['uniform_random']['episode_metrics']['prediction_coordinate_rmse']:.4f}",
                    flush=True,
                )

        summary = {
            "status": "out_of_protocol_smoke_only" if args.smoke else "completed_diagnostic",
            "interpretation": "no_performance_or_distillation_claim",
            "seed_range_half_open": seed_range,
            "episodes": len(rows),
            "group_metrics": summarize_group_rows(rows, config),
            "positive_control_reference_success": [
                bool(row["groups"]["reference_replay"]["success_bits"][0])
                for row in rows
            ],
            "compute": {
                "device": "cpu",
                "cuda_initialized": bool(torch.cuda.is_initialized()),
                "wall_seconds": time.perf_counter() - started,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_hours": 0,
            },
            "rows": rows,
        }
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        manifest = {
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__)),
            "diagnostic_module_sha256": sha256(ROOT / "src" / "consequence_lab" / "latent_diagnostic.py"),
            "checkpoint_sha256": sha256(weights_path),
            "checkpoint_config_sha256": sha256(checkpoint_dir / "config.json"),
            "lewm_source_revision": config["source"]["lewm_source_revision"],
            "stable_worldmodel_revision": config["source"]["stable_worldmodel_revision"],
            "progress_sha256": sha256(progress_path),
            "summary_sha256": sha256(summary_path),
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
        print(f"Artifacts: {output_root}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
