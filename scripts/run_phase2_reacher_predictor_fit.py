"""Fit small frozen-encoder Reacher latent predictors and evaluate held-out ranks."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.latent_diagnostic import (  # noqa: E402
    make_candidate_bank,
    tie_aware_pairwise_accuracy,
)
from consequence_lab.predictor_fit import (  # noqa: E402
    fit_standardized_ridge,
    make_predictor_features,
    predict_standardized_ridge,
)
from scripts.run_phase2_reacher_latent_diagnostic import (  # noqa: E402
    encode_frames,
    predict_latents,
    simulate_endpoints,
)
from scripts.run_phase2_reacher_ranking_dev import (  # noqa: E402
    load_frozen_model,
    namespace,
    preprocess_frame,
    replay_goal,
    sha256,
)


def generate_bank_for_seed(config: dict, seed: int, *, training: bool):
    env_cfg = config["environment"]
    bank_cfg = config["data"]
    reference = np.random.default_rng(seed + env_cfg["goal_action_seed_offset"]).uniform(
        -1.0, 1.0, size=(25, 2)
    )
    if training:
        random_count = env_cfg["train_validation_uniform_count_per_seed"]
        perturbation_count = env_cfg["train_validation_perturbation_count_per_seed"]
    else:
        random_count = env_cfg["test_uniform_count_per_seed"]
        perturbation_count = env_cfg["test_perturbation_count_per_seed"]
    return reference, make_candidate_bank(
        reference,
        uniform_seed=seed + bank_cfg["candidate_generation_seed_offsets"]["uniform_random"],
        perturbation_seed=seed
        + bank_cfg["candidate_generation_seed_offsets"]["reference_perturbation"],
        uniform_count=random_count,
        perturbation_count=perturbation_count,
        perturbation_std=bank_cfg["reference_perturbation_std"],
        lower_bound=env_cfg["input_action_bounds"][0],
        upper_bound=env_cfg["input_action_bounds"][1],
    )


def collect_seed(env, encoder, config: dict, seed: int, *, training: bool):
    env_cfg = config["environment"]
    start_qpos, start_qvel, start_frame, goal_qpos, _ = replay_goal(
        env,
        seed,
        seed + env_cfg["goal_action_seed_offset"],
        env_cfg["sentinel_qpos_offset"],
    )
    reference, bank = generate_bank_for_seed(config, seed, training=training)
    start_latent = encode_frames(encoder, [start_frame], batch_size=1)[0]
    group_rows = {}
    for name in ("uniform_random", "reference_perturbation"):
        actions = bank[name]
        endpoint_frames, endpoint_qpos = simulate_endpoints(
            env,
            seed=seed,
            start_qpos=start_qpos,
            start_qvel=start_qvel,
            sentinel_offset=env_cfg["sentinel_qpos_offset"],
            actions=actions,
        )
        endpoint_latents = encode_frames(
            encoder,
            endpoint_frames,
            batch_size=config["runtime"]["candidate_batch_size"],
        )
        qpos_error = np.abs(endpoint_qpos - goal_qpos)
        group_rows[name] = {
            "actions": actions.astype(np.float32).tolist(),
            "endpoint_latents": endpoint_latents.tolist(),
            "endpoint_qpos": endpoint_qpos.tolist(),
            "qpos_cost": (qpos_error.mean(axis=1) / (2 * np.pi)).tolist(),
            "success_bits": np.all(qpos_error < 0.05, axis=1).tolist(),
        }
    reference_frame, reference_qpos = simulate_endpoints(
        env,
        seed=seed,
        start_qpos=start_qpos,
        start_qvel=start_qvel,
        sentinel_offset=env_cfg["sentinel_qpos_offset"],
        actions=bank["reference_replay"],
    )
    reference_error = np.abs(reference_qpos[0] - goal_qpos)
    return {
        "seed": int(seed),
        "split": "train_or_validation" if training else "test",
        "start_qpos": start_qpos.tolist(),
        "start_qvel": start_qvel.tolist(),
        "start_latent": start_latent.tolist(),
        "goal_qpos": goal_qpos.tolist(),
        "reference_actions": reference.tolist(),
        "reference_replay_success": bool(np.all(reference_error < 0.05)),
        "groups": group_rows,
    }


def stack_supervision(rows: list[dict]):
    starts, actions, endpoints = [], [], []
    for row in rows:
        start = np.asarray(row["start_latent"], dtype=np.float32)
        for group in ("uniform_random", "reference_perturbation"):
            action = np.asarray(row["groups"][group]["actions"], dtype=np.float32)
            endpoint = np.asarray(row["groups"][group]["endpoint_latents"], dtype=np.float32)
            starts.append(np.repeat(start[None, :], len(action), axis=0))
            actions.append(action)
            endpoints.append(endpoint)
    start_array = np.concatenate(starts)
    action_array = np.concatenate(actions)
    endpoint_array = np.concatenate(endpoints)
    x = make_predictor_features(start_array, action_array).astype(np.float32)
    delta = endpoint_array - start_array
    return x, delta.astype(np.float32), start_array, action_array


def train_mlp(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, cfg: dict):
    torch.manual_seed(cfg["seed"])
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0)
    y_scale = y_train.std(axis=0)
    y_scale[y_scale < 1e-6] = 1.0
    train_x = torch.as_tensor((x_train - x_mean) / x_scale, dtype=torch.float32)
    train_y = torch.as_tensor((y_train - y_mean) / y_scale, dtype=torch.float32)
    val_x = torch.as_tensor((x_val - x_mean) / x_scale, dtype=torch.float32)
    val_y = torch.as_tensor((y_val - y_mean) / y_scale, dtype=torch.float32)

    hidden1, hidden2 = cfg["hidden_dimensions"]
    network = nn.Sequential(
        nn.Linear(train_x.shape[1], hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, train_y.shape[1]),
    )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    rng = torch.Generator().manual_seed(cfg["seed"] + 1)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    minimum_delta = cfg["minimum_delta"]
    for epoch in range(cfg["max_epochs"]):
        network.train()
        permutation = torch.randperm(len(train_x), generator=rng)
        for start in range(0, len(train_x), cfg["batch_size"]):
            indices = permutation[start : start + cfg["batch_size"]]
            prediction = network(train_x[indices])
            loss = ((prediction - train_y[indices]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.inference_mode():
            val_loss = float(((network(val_x) - val_y) ** 2).mean().item())
        history.append({"epoch": epoch + 1, "validation_latent_mse_standardized": val_loss})
        if val_loss < best_loss - minimum_delta:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(network.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= cfg["patience_epochs"]:
            break
    if best_state is None:
        raise RuntimeError("MLP validation did not produce a finite selected checkpoint")
    network.load_state_dict(best_state)
    return {
        "network": network.eval(),
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
    }


def mlp_predict(model: dict, features: np.ndarray) -> np.ndarray:
    normalized = torch.as_tensor(
        (features - model["x_mean"]) / model["x_scale"], dtype=torch.float32
    )
    with torch.inference_mode():
        delta = model["network"](normalized).cpu().numpy()
    return delta * model["y_scale"] + model["y_mean"]


def cosine_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny out-of-protocol end-to-end run")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "phase2_reacher_predictor_fit.yaml",
        help="versioned predictor experiment config",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = Path(os.environ["STABLE_WORLD_MODEL_SOURCE"]).resolve()
    checkpoint_dir = Path(os.environ["LEWM_REACHER_CHECKPOINT"]).resolve()
    output_root = Path(
        os.environ.get(
            "PHASE2_PREDICTOR_OUTPUT_DIR",
            ROOT / "artifacts" / "runs" / config["project"]["phase"],
        )
    ).resolve()
    if args.smoke:
        config["environment"]["train_episode_seeds"] = [220930, 220932]
        config["environment"]["validation_episode_seeds"] = [220932, 220933]
        config["environment"]["test_episode_seeds"] = [220933, 220934]
        config["environment"]["train_validation_uniform_count_per_seed"] = 2
        config["environment"]["train_validation_perturbation_count_per_seed"] = 2
        config["environment"]["test_uniform_count_per_seed"] = 3
        config["environment"]["test_perturbation_count_per_seed"] = 3
        config["predictors"]["mlp"].update(max_epochs=3, patience_epochs=2, batch_size=8)
        output_root = output_root / "smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    for filename in ("dataset_progress.jsonl", "test_progress.jsonl", "summary.json"):
        if (output_root / filename).exists():
            raise FileExistsError(f"Refusing to reuse predictor artifact: {output_root / filename}")

    weight_path = checkpoint_dir / "weights.pt"
    if sha256(weight_path) != config["source"]["checkpoint_sha256"]:
        raise ValueError("checkpoint weight hash differs from frozen config")
    if sha256(checkpoint_dir / "config.json") != config["source"]["checkpoint_config_sha256"]:
        raise ValueError("checkpoint config hash differs from frozen config")
    torch.set_num_threads(2)
    torch.manual_seed(config["predictors"]["mlp"]["seed"])
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU-only predictor study")
    encoder = load_frozen_model(source_root, checkpoint_dir)
    sys.path.insert(0, str(source_root))
    dmcontrol_path = source_root / "stable_worldmodel" / "envs" / "dmcontrol"
    namespace("stable_worldmodel.envs.dmcontrol", dmcontrol_path)
    import stable_worldmodel.envs  # noqa: F401 - register exact pinned env ID
    env = gym.make(config["environment"]["id"], task=config["environment"]["task"])
    started = time.perf_counter()
    train_rows, val_rows = [], []
    dataset_progress = output_root / "dataset_progress.jsonl"
    try:
        with dataset_progress.open("x", encoding="utf-8") as stream:
            for split_name in ("train", "validation"):
                bounds = config["environment"][f"{split_name}_episode_seeds"]
                target_rows = train_rows if split_name == "train" else val_rows
                for seed in range(*bounds):
                    row = collect_seed(env, encoder, config, seed, training=True)
                    row["split"] = split_name
                    target_rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    if dataset_progress.stat().st_size > config["artifacts"]["stop_bytes"]:
                        raise RuntimeError("dataset artifact stop limit exceeded")
                print(f"collected {split_name} episode seeds {bounds}", flush=True)

        x_train, y_train, _, _ = stack_supervision(train_rows)
        x_val, y_val, _, _ = stack_supervision(val_rows)
        ridge_cfg = config["predictors"]["ridge"]
        ridge = fit_standardized_ridge(x_train, y_train, l2=ridge_cfg["l2_regularization"])
        mlp = train_mlp(x_train, y_train, x_val, y_val, config["predictors"]["mlp"])
        ridge_val_pred = predict_standardized_ridge(ridge, x_val)
        mlp_val_pred = mlp_predict(mlp, x_val)
        validation_metrics = {
            "ridge_latent_delta_mse": float(np.mean((ridge_val_pred - y_val) ** 2)),
            "mlp_latent_delta_mse": float(np.mean((mlp_val_pred - y_val) ** 2)),
            "persistence_latent_delta_mse": float(
                np.mean((np.zeros_like(y_val) - y_val) ** 2)
            ),
        }

        checkpoint_payload = {
            "network": mlp["network"].state_dict(),
            "x_mean": mlp["x_mean"],
            "x_scale": mlp["x_scale"],
            "y_mean": mlp["y_mean"],
            "y_scale": mlp["y_scale"],
            "ridge": ridge,
        }
        trained_predictor_path = output_root / "predictors.pt"
        torch.save(checkpoint_payload, trained_predictor_path)

        test_rows = []
        test_progress = output_root / "test_progress.jsonl"
        with test_progress.open("x", encoding="utf-8") as stream:
            bounds = config["environment"]["test_episode_seeds"]
            for seed in range(*bounds):
                # Collect once, including the exact start frame, then evaluate
                # every predictor on that same visual/action input.
                row = collect_seed(env, encoder, config, seed, training=False)
                row = evaluate_collected_seed(env, encoder, mlp, ridge, config, row, seed)
                test_rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if test_progress.stat().st_size + dataset_progress.stat().st_size > config["artifacts"]["stop_bytes"]:
                    raise RuntimeError("combined predictor data artifact stop limit exceeded")
                print(
                    f"test_seed={seed} reference_success={row['reference_replay_success']} "
                    f"mlp_local_rank={row['groups']['reference_perturbation']['methods']['mlp']['metrics']['predicted_latent_goal_vs_qpos_pairwise_accuracy']}",
                    flush=True,
                )

        summary = make_summary(
            train_rows,
            val_rows,
            test_rows,
            mlp,
            validation_metrics,
            config,
            smoke=args.smoke,
            elapsed=time.perf_counter() - started,
        )
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        manifest = {
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__)),
            "predictor_module_sha256": sha256(ROOT / "src" / "consequence_lab" / "predictor_fit.py"),
            "latent_diagnostic_module_sha256": sha256(ROOT / "src" / "consequence_lab" / "latent_diagnostic.py"),
            "checkpoint_sha256": sha256(weight_path),
            "checkpoint_config_sha256": sha256(checkpoint_dir / "config.json"),
            "lewm_source_revision": config["source"]["lewm_source_revision"],
            "stable_worldmodel_revision": config["source"]["stable_worldmodel_revision"],
            "dataset_progress_sha256": sha256(dataset_progress),
            "test_progress_sha256": sha256(test_progress),
            "summary_sha256": sha256(summary_path),
            "predictors_sha256": sha256(trained_predictor_path),
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in summary.items() if k != "test_group_metrics"}, indent=2))
        print(f"Artifacts: {output_root}")
    finally:
        env.close()


def evaluate_collected_seed(env, encoder, mlp, ridge, config, row, seed):
    """Evaluate native/model baselines using the exact stored start observation."""
    env_cfg = config["environment"]
    _, _, start_frame, _, goal_frame = replay_goal(
        env,
        seed,
        seed + env_cfg["goal_action_seed_offset"],
        env_cfg["sentinel_qpos_offset"],
    )
    start_latent = np.asarray(row["start_latent"], dtype=np.float32)
    goal_latent = encode_frames(encoder, [goal_frame], batch_size=1)[0]
    all_groups = {}
    model_names = ("persistence", "ridge", "mlp", "native_lewm")
    for group_name in ("uniform_random", "reference_perturbation"):
        group = row["groups"][group_name]
        actions = np.asarray(group["actions"], dtype=np.float32)
        realized = np.asarray(group["endpoint_latents"], dtype=np.float32)
        qpos_cost = np.asarray(group["qpos_cost"], dtype=np.float64)
        success = np.asarray(group["success_bits"], dtype=bool)
        starts = np.repeat(start_latent[None, :], len(actions), axis=0)
        features = make_predictor_features(starts, actions).astype(np.float32)
        native, _ = predict_latents(
            encoder,
            start_frame,
            actions,
            rolling_window=config["planning"]["predictor_rolling_window"],
            batch_size=config["runtime"]["candidate_batch_size"],
        )
        predictions = {
            "persistence": starts,
            "ridge": starts + predict_standardized_ridge(ridge, features).astype(np.float32),
            "mlp": starts + mlp_predict(mlp, features).astype(np.float32),
            "native_lewm": native,
        }
        realized_goal_cost = np.square(realized - goal_latent).sum(axis=1)
        methods = {}
        for method_name in model_names:
            predicted = predictions[method_name]
            predicted_goal_cost = np.square(predicted - goal_latent).sum(axis=1)
            forecast_rmse = np.sqrt(np.mean((predicted - realized) ** 2, axis=1))
            forecast_cosine = cosine_rows(predicted, realized)
            predicted_realized_rank, predicted_realized_pairs = tie_aware_pairwise_accuracy(
                predicted_goal_cost,
                realized_goal_cost,
                actual_tie_tolerance=config["evaluation"]["latent_cost_tie_tolerance"],
                predicted_tie_tolerance=config["evaluation"]["latent_cost_tie_tolerance"],
            )
            predicted_physical_rank, predicted_physical_pairs = tie_aware_pairwise_accuracy(
                predicted_goal_cost,
                qpos_cost,
                actual_tie_tolerance=config["evaluation"]["physical_cost_tie_tolerance"],
                predicted_tie_tolerance=config["evaluation"]["latent_cost_tie_tolerance"],
            )
            realized_physical_rank, realized_physical_pairs = tie_aware_pairwise_accuracy(
                realized_goal_cost,
                qpos_cost,
                actual_tie_tolerance=config["evaluation"]["physical_cost_tie_tolerance"],
                predicted_tie_tolerance=config["evaluation"]["latent_cost_tie_tolerance"],
            )
            methods[method_name] = {
                "predicted_latents": predicted.tolist(),
                "predicted_goal_cost": predicted_goal_cost.tolist(),
                "endpoint_latent_coordinate_rmse": forecast_rmse.tolist(),
                "endpoint_latent_cosine_similarity": forecast_cosine.tolist(),
                "metrics": {
                    "endpoint_latent_coordinate_rmse": float(forecast_rmse.mean()),
                    "endpoint_latent_cosine_similarity": float(forecast_cosine.mean()),
                    "predicted_vs_realized_latent_cost_pairwise_accuracy": predicted_realized_rank,
                    "predicted_latent_goal_vs_qpos_pairwise_accuracy": predicted_physical_rank,
                    "realized_latent_goal_vs_qpos_pairwise_accuracy": realized_physical_rank,
                    "pair_counts": {
                        "predicted_vs_realized_latent_cost": predicted_realized_pairs,
                        "predicted_latent_vs_qpos": predicted_physical_pairs,
                        "realized_latent_vs_qpos": realized_physical_pairs,
                    },
                },
            }
        all_groups[group_name] = {
            "candidate_actions": group["actions"],
            "endpoint_latents": group["endpoint_latents"],
            "endpoint_qpos": group["endpoint_qpos"],
            "qpos_cost": group["qpos_cost"],
            "success_bits": success.tolist(),
            "methods": methods,
        }
    row["groups"] = all_groups
    return row


def make_summary(train_rows, val_rows, test_rows, mlp, validation_metrics, config, *, smoke, elapsed):
    bootstrap_cfg = config["evaluation"]
    groups = ("uniform_random", "reference_perturbation")
    method_names = ("persistence", "ridge", "mlp", "native_lewm")
    metric_names = (
        "endpoint_latent_coordinate_rmse",
        "endpoint_latent_cosine_similarity",
        "predicted_vs_realized_latent_cost_pairwise_accuracy",
        "predicted_latent_goal_vs_qpos_pairwise_accuracy",
    )
    summary = {}
    for group_index, group_name in enumerate(groups):
        summary[group_name] = {}
        for method_index, method in enumerate(method_names):
            summary[group_name][method] = {}
            for metric_index, metric in enumerate(metric_names):
                values = []
                for row in test_rows:
                    value = row["groups"][group_name]["methods"][method]["metrics"][metric]
                    if value is not None and np.isfinite(value):
                        values.append(float(value))
                summary[group_name][method][metric] = {
                    "mean_equal_episode_weight": float(np.mean(values)) if values else None,
                    "cluster_bootstrap_95_ci": bootstrap_ci(
                        values,
                        replicates=bootstrap_cfg["bootstrap_replicates"],
                        seed=bootstrap_cfg["bootstrap_seed"]
                        + group_index * 100
                        + method_index * 10
                        + metric_index,
                    ),
                    "episodes_with_defined_value": len(values),
                }
            success_fraction = [
                float(np.mean(row["groups"][group_name]["success_bits"]))
                for row in test_rows
            ]
            summary[group_name][method]["candidate_success_fraction"] = {
                "mean_equal_episode_weight": float(np.mean(success_fraction)),
                "cluster_bootstrap_95_ci": bootstrap_ci(
                    success_fraction,
                    replicates=bootstrap_cfg["bootstrap_replicates"],
                    seed=bootstrap_cfg["bootstrap_seed"] + group_index * 100 + method_index * 10 + 8,
                ),
                "episodes_with_defined_value": len(success_fraction),
            }
            summary[group_name][method]["candidate_availability_episodes"] = sum(
                any(row["groups"][group_name]["success_bits"]) for row in test_rows
            )
    return {
        "status": "out_of_protocol_smoke_only" if smoke else "completed_predictor_development",
        "interpretation": "predictor_diagnostic_only_no_student_or_teacher_claim",
        "seed_splits": {
            name: config["environment"][f"{name}_episode_seeds"]
            for name in ("train", "validation", "test")
        },
        "episodes": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "samples": {
            "train": len(stack_supervision(train_rows)[0]),
            "validation": len(stack_supervision(val_rows)[0]),
        },
        "validation_latent_delta_mse": validation_metrics,
        "mlp_early_stopping": {
            "selected_epoch": mlp["best_epoch"],
            "best_validation_loss_standardized": mlp["best_validation_loss"],
            "curve": mlp["history"],
        },
        "test_group_metrics": summary,
        "positive_control_replay_success_count": sum(
            row["reference_replay_success"] for row in test_rows
        ),
        "compute": {
            "device": "cpu",
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "wall_seconds": elapsed,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "gpu_hours": 0,
        },
    }


def bootstrap_ci(values, *, replicates, seed):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_ids = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[sample_ids].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


if __name__ == "__main__":
    main()
