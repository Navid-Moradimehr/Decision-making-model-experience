"""Cross-scored frozen-model CEM/random planner development screen."""

from __future__ import annotations

import argparse
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

from consequence_lab.cem import run_bounded_cem  # noqa: E402
from consequence_lab.latent_diagnostic import tie_aware_pairwise_accuracy  # noqa: E402
from consequence_lab.predictor_fit import (  # noqa: E402
    make_predictor_features,
)
from scripts.run_phase2_reacher_latent_diagnostic import (  # noqa: E402
    encode_frames,
    predict_latents,
)
from scripts.run_phase2_reacher_ranking_dev import (  # noqa: E402
    load_frozen_model,
    namespace,
    preprocess_frame,
    replay_goal,
    sha256,
)


def load_mlp_predictor(path: Path, input_dim: int, output_dim: int, hidden: list[int]):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden[0]),
        nn.ReLU(),
        nn.Linear(hidden[0], hidden[1]),
        nn.ReLU(),
        nn.Linear(hidden[1], output_dim),
    )
    model.load_state_dict(payload["network"], strict=True)
    model.eval().requires_grad_(False)
    return {
        "network": model,
        "x_mean": np.asarray(payload["x_mean"], dtype=np.float32),
        "x_scale": np.asarray(payload["x_scale"], dtype=np.float32),
        "y_mean": np.asarray(payload["y_mean"], dtype=np.float32),
        "y_scale": np.asarray(payload["y_scale"], dtype=np.float32),
    }


def mlp_endpoints(predictor: dict, start_latent: np.ndarray, actions: np.ndarray):
    start = np.asarray(start_latent, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32).reshape(len(actions), 25, 2)
    starts = np.repeat(start[None, :], len(action_array), axis=0)
    features = make_predictor_features(starts, action_array).astype(np.float32)
    x = torch.as_tensor((features - predictor["x_mean"]) / predictor["x_scale"])
    with torch.inference_mode():
        normalized_delta = predictor["network"](x).cpu().numpy()
    delta = normalized_delta * predictor["y_scale"] + predictor["y_mean"]
    return starts + delta


def score_population(
    model_name: str,
    candidates: np.ndarray,
    *,
    mlp_predictor: dict,
    native_model,
    start_frame: np.ndarray,
    start_latent: np.ndarray,
    goal_latent: np.ndarray,
    rolling_window: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    actions = np.asarray(candidates, dtype=np.float32).reshape(len(candidates), 25, 2)
    if model_name == "supervised_mlp":
        endpoint_latent = mlp_endpoints(mlp_predictor, start_latent, actions)
    elif model_name == "native_lewm":
        endpoint_latent, _ = predict_latents(
            native_model,
            start_frame,
            actions,
            rolling_window=rolling_window,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"unknown scorer: {model_name}")
    costs = np.square(endpoint_latent - goal_latent[None, :]).sum(axis=1)
    if costs.shape != (len(candidates),) or not np.isfinite(costs).all():
        raise RuntimeError(f"{model_name} returned invalid model costs")
    return costs.astype(np.float64), time.perf_counter() - started


def get_candidate_populations(
    config: dict,
    *,
    start_latent: np.ndarray,
    goal_latent: np.ndarray,
    start_frame: np.ndarray,
    mlp_predictor: dict,
    native_model,
    episode_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    optimizer = config["candidates"]["optimizer"]
    lower, upper = config["candidates"]["bounds"]
    populations = {}
    optimizer_latency = {}
    scorer_args = {
        "mlp_predictor": mlp_predictor,
        "native_model": native_model,
        "start_frame": start_frame,
        "start_latent": start_latent,
        "goal_latent": goal_latent,
        "rolling_window": config["models"]["native_lewm"]["predictor_rolling_window"],
        "batch_size": config["runtime"]["candidate_batch_size"],
    }
    for generator, scorer, offset in (
        ("mlp_cem", "supervised_mlp", optimizer["mlp_seed_offset"]),
        ("native_lewm_cem", "native_lewm", optimizer["native_lewm_seed_offset"]),
    ):
        planner_started = time.perf_counter()
        result = run_bounded_cem(
            lambda batch: score_population(scorer, batch.reshape(len(batch), 25, 2), **scorer_args)[0],
            horizon_shape=tuple(config["candidates"]["flattened_optimizer_shape"]),
            num_samples=optimizer["samples_per_iteration"],
            iterations=optimizer["iterations"],
            num_elites=optimizer["elites"],
            seed=optimizer["seed_base"] + offset + episode_index,
            lower_bound=lower,
            upper_bound=upper,
            initial_std=optimizer["initial_std"],
            minimum_std=optimizer["minimum_std"],
        )
        populations[generator] = result.candidates.reshape(-1, 25, 2)
        optimizer_latency[generator] = time.perf_counter() - planner_started

    random_seed = config["candidates"]["uniform_random_seed_base"] + episode_index
    rng = np.random.default_rng(random_seed)
    populations["uniform_random"] = rng.uniform(
        lower,
        upper,
        size=(config["candidates"]["uniform_random_count"], 25, 2),
    )
    return populations, optimizer_latency


def simulate_terminal_qpos(
    env,
    *,
    seed: int,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    sentinel_offset: list[float],
    actions: np.ndarray,
) -> np.ndarray:
    endpoints = np.empty((len(actions), len(start_qpos)), dtype=np.float64)
    sentinel = start_qpos + np.asarray(sentinel_offset, dtype=np.float64)
    for index, sequence in enumerate(actions):
        env.reset(seed=seed)
        env.unwrapped.set_state(start_qpos, start_qvel)
        env.unwrapped.set_target_qpos(sentinel)
        info = env.unwrapped.info
        for action in sequence:
            _, _, terminated, truncated, info = env.step(
                action.astype(env.action_space.dtype)
            )
            if terminated or truncated:
                raise RuntimeError("sentinel target failed to disable early termination")
        endpoints[index] = np.asarray(info["qpos"], dtype=np.float64)
    return endpoints


def evaluate_group(
    candidates: np.ndarray,
    costs_by_scorer: dict[str, np.ndarray],
    true_cost: np.ndarray,
    successes: np.ndarray,
    config: dict,
) -> dict:
    result = {}
    for scorer, predicted_cost in costs_by_scorer.items():
        best = int(np.argmin(predicted_cost))
        oracle = int(np.argmin(true_cost))
        accuracy, pair_count = tie_aware_pairwise_accuracy(
            predicted_cost,
            true_cost,
            actual_tie_tolerance=config["ground_truth"]["physical_cost_tie_tolerance"],
            predicted_tie_tolerance=config["ground_truth"]["model_cost_tie_tolerance"],
        )
        result[scorer] = {
            "model_costs": predicted_cost.tolist(),
            "model_best_index": best,
            "oracle_best_index": oracle,
            "candidate_availability": bool(successes.any()),
            "model_best_success": bool(successes[best]),
            "selected_true_cost": float(true_cost[best]),
            "oracle_true_cost": float(true_cost[oracle]),
            "normalized_regret": float(max(0.0, true_cost[best] - true_cost[oracle])),
            "top1_agreement_within_tolerance": bool(
                true_cost[best] - true_cost[oracle]
                <= config["ground_truth"]["physical_cost_tie_tolerance"]
            ),
            "pairwise_accuracy": accuracy,
            "ranked_pair_count": pair_count,
        }
    return result


def bootstrap_ci(values: list[float], *, replicates: int, seed: int):
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[index].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(rows: list[dict], config: dict):
    names = ("candidate_availability", "model_best_success", "normalized_regret", "pairwise_accuracy", "top1_agreement_within_tolerance")
    aggregate_result = {}
    seed_base = config["statistics"]["bootstrap_seed"]
    replicates = config["statistics"]["bootstrap_replicates"]
    for group_index, group_name in enumerate(config["candidates"]["generators"]):
        aggregate_result[group_name] = {}
        for scorer_index, scorer in enumerate(config["candidates"]["cross_score_with"]):
            group_rows = [row["groups"][group_name][scorer] for row in rows]
            cell = {}
            for metric_index, metric in enumerate(names):
                values = [
                    float(row[metric])
                    for row in group_rows
                    if row[metric] is not None and np.isfinite(row[metric])
                ]
                cell[metric] = {
                    "mean_equal_episode_weight": float(np.mean(values)) if values else None,
                    "cluster_bootstrap_95_ci": bootstrap_ci(
                        values,
                        replicates=replicates,
                        seed=seed_base + group_index * 100 + scorer_index * 10 + metric_index,
                    ),
                    "episodes_with_defined_value": len(values),
                }
            cell["selected_success_episode_count"] = int(
                sum(row["model_best_success"] for row in group_rows)
            )
            cell["episodes_with_any_successful_candidate"] = int(
                sum(row["candidate_availability"] for row in group_rows)
            )
            cell["total_ranked_pairs"] = int(sum(row["ranked_pair_count"] for row in group_rows))
            aggregate_result[group_name][scorer] = cell
    return aggregate_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny out-of-protocol check")
    args = parser.parse_args()
    config_path = ROOT / "configs" / "phase2_reacher_planner_dev.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = Path(os.environ["STABLE_WORLD_MODEL_SOURCE"]).resolve()
    checkpoint_dir = Path(os.environ["LEWM_REACHER_CHECKPOINT"]).resolve()
    predictor_run = ROOT / config["source"]["supervised_predictor_run"]
    predictor_path = predictor_run / "predictors.pt"
    predictor_manifest = json.loads((predictor_run / "manifest.json").read_text(encoding="utf-8"))
    if sha256(predictor_path) != predictor_manifest["predictors_sha256"]:
        raise ValueError("trained predictor artifact hash does not match Phase-2E manifest")
    output_root = Path(
        os.environ.get(
            "PHASE2_PLANNER_OUTPUT_DIR",
            ROOT / "artifacts" / "runs" / "phase2_reacher_planner_dev",
        )
    ).resolve()
    if args.smoke:
        config["environment"]["episode_seed_range"] = [220950, 220951]
        config["candidates"]["uniform_random_count"] = 8
        config["candidates"]["optimizer"].update(
            samples_per_iteration=8, iterations=2, elites=2
        )
        config["statistics"]["bootstrap_replicates"] = 20
        output_root = output_root / "smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("progress.jsonl", "summary.json"):
        if (output_root / name).exists():
            raise FileExistsError(f"Refusing to overwrite planner artifact: {output_root / name}")

    if sha256(checkpoint_dir / "weights.pt") != config["source"]["checkpoint_sha256"]:
        raise ValueError("LeWM checkpoint hash mismatch")
    if sha256(checkpoint_dir / "config.json") != config["source"]["checkpoint_config_sha256"]:
        raise ValueError("LeWM config hash mismatch")
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized before CPU-only planner study")
    native_model = load_frozen_model(source_root, checkpoint_dir)
    dimension = native_model.encode(
        {"pixels": preprocess_frame(np.zeros((224, 224, 3), dtype=np.uint8)).unsqueeze(0).unsqueeze(0)}
    )["emb"].shape[-1]
    mlp_predictor = load_mlp_predictor(
        predictor_path,
        input_dim=dimension
        + config["models"]["supervised_mlp"]["action_sequence_flattened_dimension"],
        output_dim=dimension,
        hidden=config["models"]["supervised_mlp"]["hidden_dimensions"],
    )
    sys.path.insert(0, str(source_root))
    dmcontrol_path = source_root / "stable_worldmodel" / "envs" / "dmcontrol"
    namespace("stable_worldmodel.envs.dmcontrol", dmcontrol_path)
    import stable_worldmodel.envs  # noqa: F401 - register pinned environment
    env = gym.make(config["environment"]["id"], task=config["environment"]["task"])
    rows = []
    started = time.perf_counter()
    progress_path = output_root / "progress.jsonl"
    try:
        with progress_path.open("x", encoding="utf-8") as stream:
            for episode_index, seed in enumerate(range(*config["environment"]["episode_seed_range"])):
                start_qpos, start_qvel, start_frame, goal_qpos, goal_frame = replay_goal(
                    env,
                    seed,
                    seed + config["environment"]["goal_action_seed_offset"],
                    config["environment"]["sentinel_qpos_offset"],
                )
                with torch.inference_mode():
                    start_embedding = native_model.encode(
                        {"pixels": preprocess_frame(start_frame).unsqueeze(0).unsqueeze(0)}
                    )["emb"][:, -1]
                    goal_embedding = native_model.encode(
                        {"pixels": preprocess_frame(goal_frame).unsqueeze(0).unsqueeze(0)}
                    )["emb"][:, -1]
                start_latent = start_embedding.squeeze(0).cpu().numpy().astype(np.float32)
                goal_latent = goal_embedding.squeeze(0).cpu().numpy().astype(np.float32)
                populations, planner_times = get_candidate_populations(
                    config,
                    start_latent=start_latent,
                    goal_latent=goal_latent,
                    start_frame=start_frame,
                    mlp_predictor=mlp_predictor,
                    native_model=native_model,
                    episode_index=episode_index,
                )

                group_results = {}
                for generator in config["candidates"]["generators"]:
                    candidate_actions = populations[generator]
                    endpoint_qpos = simulate_terminal_qpos(
                        env,
                        seed=seed,
                        start_qpos=start_qpos,
                        start_qvel=start_qvel,
                        sentinel_offset=config["environment"]["sentinel_qpos_offset"],
                        actions=candidate_actions,
                    )
                    qpos_error = np.abs(endpoint_qpos - goal_qpos)
                    true_cost = qpos_error.mean(axis=1) / (2.0 * np.pi)
                    success = np.all(qpos_error < 0.05, axis=1)
                    true_cost[success] = 0.0
                    costs_by_scorer = {}
                    scorer_times = {}
                    for scorer in config["candidates"]["cross_score_with"]:
                        model_cost, elapsed = score_population(
                            scorer,
                            candidate_actions,
                            mlp_predictor=mlp_predictor,
                            native_model=native_model,
                            start_frame=start_frame,
                            start_latent=start_latent,
                            goal_latent=goal_latent,
                            rolling_window=config["models"]["native_lewm"]["predictor_rolling_window"],
                            batch_size=config["runtime"]["candidate_batch_size"],
                        )
                        costs_by_scorer[scorer] = model_cost
                        scorer_times[scorer] = elapsed
                    evaluated = evaluate_group(
                        candidate_actions,
                        costs_by_scorer,
                        true_cost,
                        success,
                        config,
                    )
                    group_results[generator] = {
                        "candidate_sequences": candidate_actions.tolist(),
                        "simulator_costs": true_cost.tolist(),
                        "success_bits": success.tolist(),
                        "endpoint_qpos": endpoint_qpos.tolist(),
                        "scorer_seconds": scorer_times,
                        "planner_seconds": planner_times.get(generator),
                        **evaluated,
                    }

                reference_actions = np.random.default_rng(
                    seed + config["environment"]["goal_action_seed_offset"]
                ).uniform(-1.0, 1.0, size=(1, 25, 2))
                reference_qpos = simulate_terminal_qpos(
                    env,
                    seed=seed,
                    start_qpos=start_qpos,
                    start_qvel=start_qvel,
                    sentinel_offset=config["environment"]["sentinel_qpos_offset"],
                    actions=reference_actions,
                )[0]
                reference_success = bool(np.all(np.abs(reference_qpos - goal_qpos) < 0.05))
                row = {
                    "seed": int(seed),
                    "start_qpos": start_qpos.tolist(),
                    "start_qvel": start_qvel.tolist(),
                    "goal_qpos": goal_qpos.tolist(),
                    "reference_replay_success": reference_success,
                    "groups": group_results,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if progress_path.stat().st_size > config["artifacts"]["stop_bytes"]:
                    raise RuntimeError("planner artifact stop limit exceeded")
                print(
                    f"seed={seed} mlp_cem_available={group_results['mlp_cem']['supervised_mlp']['candidate_availability']} "
                    f"mlp_selected_success={group_results['mlp_cem']['supervised_mlp']['model_best_success']} "
                    f"mlp_rank={group_results['mlp_cem']['supervised_mlp']['pairwise_accuracy']}",
                    flush=True,
                )

        metrics = aggregate(rows, config)
        primary = metrics[config["primary_screen"]["generator"]][config["primary_screen"]["scorer"]]
        availability = primary["candidate_availability"]["mean_equal_episode_weight"]
        ranking = primary["pairwise_accuracy"]["mean_equal_episode_weight"]
        regret = primary["normalized_regret"]["mean_equal_episode_weight"]
        summary_path = output_root / "summary.json"
        summary = {
            "status": "out_of_protocol_smoke_only" if args.smoke else "completed_planner_development_screen",
            "interpretation": "no_teacher_labels_or_student_training",
            "seed_range_half_open": config["environment"]["episode_seed_range"],
            "episodes": len(rows),
            "candidate_generator_scorer_metrics": metrics,
            "primary_screen": {
                "generator": config["primary_screen"]["generator"],
                "scorer": config["primary_screen"]["scorer"],
                "metrics": {
                    "candidate_availability": primary["candidate_availability"],
                    "pairwise_accuracy": primary["pairwise_accuracy"],
                    "normalized_regret": primary["normalized_regret"],
                },
                "gates_passed": {
                    "candidate_availability": availability is not None
                    and availability >= config["primary_screen"]["candidate_availability_at_least"],
                    "pairwise_accuracy": ranking is not None
                    and ranking >= config["primary_screen"]["pairwise_accuracy_at_least"],
                    "normalized_regret": regret is not None
                    and regret <= config["primary_screen"]["normalized_regret_at_most"],
                },
            },
            "positive_control": "tested separately via exact action replay in goal construction; excluded from candidate populations",
            "positive_control_success_count": int(
                sum(row["reference_replay_success"] for row in rows)
            ),
            "compute": {
                "device": "cpu",
                "cuda_initialized": bool(torch.cuda.is_initialized()),
                "wall_seconds": time.perf_counter() - started,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_hours": 0,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        manifest = {
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__)),
            "cem_sha256": sha256(ROOT / "src" / "consequence_lab" / "cem.py"),
            "latent_diagnostic_sha256": sha256(ROOT / "src" / "consequence_lab" / "latent_diagnostic.py"),
            "predictor_run_manifest_sha256": sha256(predictor_run / "manifest.json"),
            "predictor_artifact_sha256": sha256(predictor_path),
            "checkpoint_sha256": sha256(checkpoint_dir / "weights.pt"),
            "checkpoint_config_sha256": sha256(checkpoint_dir / "config.json"),
            "lewm_source_revision": config["source"]["lewm_source_revision"],
            "stable_worldmodel_revision": config["source"]["stable_worldmodel_revision"],
            "progress_sha256": sha256(progress_path),
            "summary_sha256": sha256(summary_path),
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Artifacts: {output_root}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
