"""Run preregistered frozen-LeWM Reacher candidate ranking (CPU only)."""

from __future__ import annotations

import hashlib
import argparse
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
from torchvision.transforms import v2 as transforms
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consequence_lab.cem import run_bounded_cem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def load_frozen_model(source_root: Path, checkpoint_dir: Path):
    import stable_pretraining

    sp_root = Path(stable_pretraining.__file__).resolve().parent
    namespace("stable_pretraining.backbone", sp_root / "backbone")
    from stable_pretraining.backbone.utils import vit_hf

    namespace("stable_worldmodel", source_root / "stable_worldmodel")
    namespace("stable_worldmodel.wm", source_root / "stable_worldmodel" / "wm")
    namespace(
        "stable_worldmodel.wm.lewm",
        source_root / "stable_worldmodel" / "wm" / "lewm",
    )
    from stable_worldmodel.wm.lewm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor

    cfg = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    enc = cfg["encoder"]
    encoder = vit_hf(
        size=enc["size"],
        patch_size=enc["patch_size"],
        image_size=enc["image_size"],
        pretrained=enc["pretrained"],
        use_mask_token=enc["use_mask_token"],
    )
    predictor = Predictor(**{k: v for k, v in cfg["predictor"].items() if k != "_target_"})
    action_encoder = Embedder(
        **{k: v for k, v in cfg["action_encoder"].items() if k != "_target_"}
    )

    def projection(name: str) -> MLP:
        values = cfg[name]
        return MLP(
            input_dim=values["input_dim"],
            hidden_dim=values["hidden_dim"],
            output_dim=values["output_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = LeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projection("projector"),
        pred_proj=projection("pred_proj"),
    )
    state = torch.load(
        checkpoint_dir / "weights.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    pipeline = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            transforms.Resize(size=224),
        ]
    )
    return pipeline(np.ascontiguousarray(frame))


def replay_goal(
    env, seed: int, action_seed: int, sentinel_qpos_offset: list[float]
):
    _, start_info = env.reset(seed=seed)
    start_frame = env.unwrapped.render(width=224, height=224)
    start_qpos = np.asarray(start_info["qpos"]).copy()
    start_qvel = np.asarray(start_info["qvel"]).copy()
    # qpos_match target_qpos survives an environment reset in this pinned
    # wrapper. Keep goal-generation rollouts from inheriting a prior target.
    env.unwrapped.set_target_qpos(
        start_qpos + np.asarray(sentinel_qpos_offset, dtype=np.float64)
    )
    rng = np.random.default_rng(action_seed)
    last_info = start_info
    for action in rng.uniform(-1.0, 1.0, size=(25, 2)):
        _, _, terminated, truncated, last_info = env.step(
            action.astype(env.action_space.dtype)
        )
        if terminated or truncated:
            raise RuntimeError("goal-generation trajectory ended before 25 actions")
    goal_qpos = np.asarray(last_info["qpos"]).copy()
    goal_frame = env.unwrapped.render(width=224, height=224)
    return start_qpos, start_qvel, start_frame, goal_qpos, goal_frame


def evaluate_candidates(
    env,
    seed: int,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    goal_qpos: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    costs = np.empty(len(candidates), dtype=np.float64)
    successes = np.zeros(len(candidates), dtype=bool)
    for index, sequence in enumerate(candidates):
        env.reset(seed=seed)
        env.unwrapped.set_state(start_qpos, start_qvel)
        env.unwrapped.set_target_qpos(goal_qpos)
        terminated = False
        truncated = False
        info = env.unwrapped.info
        actions = sequence.reshape(25, 2)
        for action in actions:
            _, _, terminated, truncated, info = env.step(
                action.astype(env.action_space.dtype)
            )
            if terminated or truncated:
                break
        successes[index] = bool(terminated)
        if terminated:
            costs[index] = 0.0
        else:
            qpos_error = np.abs(np.asarray(info["qpos"]) - goal_qpos).mean()
            costs[index] = float(qpos_error / (2.0 * np.pi))
    return costs, successes


def tie_aware_pairwise_accuracy(
    model_cost: np.ndarray,
    true_cost: np.ndarray,
    true_tolerance: float,
    model_tolerance: float,
) -> tuple[float | None, int]:
    upper = np.triu(np.ones((len(true_cost), len(true_cost)), dtype=bool), k=1)
    delta_true = true_cost[:, None] - true_cost[None, :]
    eligible = upper & (np.abs(delta_true) > true_tolerance)
    if not eligible.any():
        return None, 0
    delta_model = model_cost[:, None] - model_cost[None, :]
    credit = np.where(
        np.abs(delta_model) <= model_tolerance,
        0.5,
        (np.sign(delta_model) == np.sign(delta_true)).astype(np.float64),
    )
    return float(credit[eligible].mean()), int(eligible.sum())


def bootstrap_mean_ci(
    values: list[float], *, replicates: int, seed: int
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one out-of-protocol seed with a tiny CEM only to verify integration",
    )
    args = parser.parse_args()
    config_path = ROOT / "configs" / "phase2_reacher_ranking_dev.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_root = Path(os.environ["STABLE_WORLD_MODEL_SOURCE"]).resolve()
    checkpoint_dir = Path(os.environ["LEWM_REACHER_CHECKPOINT"]).resolve()
    output_root = Path(
        os.environ.get(
            "PHASE2_OUTPUT_DIR",
            ROOT / "artifacts" / "runs" / "phase2_reacher_ranking_dev",
        )
    ).resolve()
    if args.smoke:
        output_root = output_root / "smoke"
        config["environment"]["episode_seed_range"] = [220900, 220902]
        config["planning"]["cem"].update(
            samples_per_iteration=8, iterations=2, elites=2, seed=229000
        )
        config["statistics"]["bootstrap_replicates"] = 20
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.jsonl"
    if progress_path.exists():
        raise FileExistsError(
            f"Refusing to reuse existing run progress: {progress_path}"
        )
    if not args.smoke and (output_root / "summary.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite a completed run: {output_root / 'summary.json'}"
        )
    weights_path = checkpoint_dir / "weights.pt"
    assert sha256(weights_path) == config["source"]["checkpoint_sha256"]
    assert sha256(checkpoint_dir / "config.json") == config["source"]["checkpoint_config_sha256"]

    torch.set_num_threads(2)
    torch.manual_seed(0)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before the CPU-only experiment")
    model = load_frozen_model(source_root, checkpoint_dir)

    # Reuse the exact pinned environment/task source; skip only its unrelated
    # eager DMC expert-policy import, as in the compatibility smoke.
    sys.path.insert(0, str(source_root))
    dmcontrol_path = source_root / "stable_worldmodel" / "envs" / "dmcontrol"
    namespace("stable_worldmodel.envs.dmcontrol", dmcontrol_path)
    import stable_worldmodel.envs  # noqa: F401 - register the pinned env ID

    from consequence_lab.cem import run_bounded_cem

    env = gym.make(config["environment"]["id"], task=config["environment"]["task"])
    seeds = range(*config["environment"]["episode_seed_range"])
    rows: list[dict] = []
    progress_stream = progress_path.open("x", encoding="utf-8")
    wall_start = time.perf_counter()
    try:
        for state_index, seed in enumerate(seeds):
            start_qpos, start_qvel, start_frame, goal_qpos, goal_frame = replay_goal(
                env,
                seed,
                seed + config["environment"]["goal_action_seed_offset"],
                config["environment"]["goal_generation_sentinel_qpos_offset"],
            )
            current_image = preprocess_frame(start_frame).unsqueeze(0).unsqueeze(0)
            goal_image = preprocess_frame(goal_frame).unsqueeze(0).unsqueeze(0)
            with torch.inference_mode():
                current_emb = model.encode({"pixels": current_image})["emb"]
                goal_emb = model.encode({"pixels": goal_image})["emb"]

            cem = config["planning"]["cem"]
            samples = cem["samples_per_iteration"]

            def model_cost(candidate_array: np.ndarray) -> np.ndarray:
                actions = torch.as_tensor(
                    candidate_array, dtype=torch.float32
                ).unsqueeze(0)
                sample_count = actions.shape[1]
                info = {
                    "pixels": current_image.unsqueeze(1).expand(
                        1, sample_count, *current_image.shape[1:]
                    ),
                    "emb": current_emb.unsqueeze(1).expand(
                        1, sample_count, *current_emb.shape[1:]
                    ),
                }
                with torch.inference_mode():
                    prediction = model.rollout(
                    info,
                    actions,
                    history_size=config["planning"]["predictor_rolling_window"],
                    )["predicted_emb"]
                    final = prediction[:, :, -1]
                    goal = goal_emb[:, -1].unsqueeze(1)
                    costs = (final - goal).square().sum(dim=-1).squeeze(0)
                return costs.cpu().numpy().astype(np.float64)

            started = time.perf_counter()
            result = run_bounded_cem(
                model_cost,
                horizon_shape=(
                    config["planning"]["horizon_blocks"],
                    config["planning"]["flattened_action_dimensions"],
                ),
                num_samples=samples,
                iterations=cem["iterations"],
                num_elites=cem["elites"],
                seed=cem["seed"] + state_index,
                lower_bound=config["planning"]["action_bounds"][0],
                upper_bound=config["planning"]["action_bounds"][1],
                initial_std=cem["initial_std"],
                minimum_std=cem["minimum_std"],
            )
            model_wall = time.perf_counter() - started
            if not np.isfinite(result.costs).all():
                raise RuntimeError(f"non-finite model cost at seed {seed}")

            true_costs, successes = evaluate_candidates(
                env,
                seed,
                start_qpos,
                start_qvel,
                goal_qpos,
                result.candidates,
            )
            selected = result.best_index
            oracle = int(np.argmin(true_costs))
            pairwise, pair_count = tie_aware_pairwise_accuracy(
                result.costs,
                true_costs,
                config["ground_truth"]["true_cost_tie_tolerance"],
                config["ground_truth"]["predicted_cost_tie_tolerance"],
            )
            mean_cost = model_cost(result.mean[None, ...])[0]
            mean_true_cost, mean_success = evaluate_candidates(
                env,
                seed,
                start_qpos,
                start_qvel,
                goal_qpos,
                result.mean[None, ...],
            )

            row = {
                    "seed": int(seed),
                    "start_qpos": start_qpos.tolist(),
                    "start_qvel": start_qvel.tolist(),
                    "goal_qpos": goal_qpos.tolist(),
                    "candidate_sequences": result.candidates.tolist(),
                    "model_costs": result.costs.tolist(),
                    "simulator_costs": true_costs.tolist(),
                    "success_bits": successes.tolist(),
                    "model_best_index": int(selected),
                    "oracle_best_index": oracle,
                    "model_best_success": bool(successes[selected]),
                    "candidate_availability": bool(successes.any()),
                    "normalized_regret": float(
                        max(0.0, true_costs[selected] - true_costs[oracle])
                    ),
                    "top1_agreement_within_tolerance": bool(
                        true_costs[selected] - true_costs[oracle]
                        <= config["ground_truth"]["true_cost_tie_tolerance"]
                    ),
                    "pairwise_accuracy": pairwise,
                    "ranked_pair_count": pair_count,
                    "cem_mean_model_cost": float(mean_cost),
                    "cem_mean_simulator_cost": float(mean_true_cost[0]),
                    "cem_mean_success": bool(mean_success[0]),
                    "model_cem_wall_seconds": float(model_wall),
                }
            rows.append(row)
            progress_stream.write(json.dumps(row, allow_nan=False) + "\n")
            progress_stream.flush()
            print(
                f"seed={seed} available={bool(successes.any())} "
                f"rank={pairwise} regret={rows[-1]['normalized_regret']:.4f} "
                f"model-best-success={bool(successes[selected])} "
                f"cem-cpu={model_wall:.2f}s",
                flush=True,
            )

        metric_values = {
            "candidate_availability": [float(row["candidate_availability"]) for row in rows],
            "pairwise_accuracy": [
                float(row["pairwise_accuracy"])
                for row in rows
                if row["pairwise_accuracy"] is not None
            ],
            "normalized_regret": [float(row["normalized_regret"]) for row in rows],
            "model_best_success": [float(row["model_best_success"]) for row in rows],
            "cem_mean_success": [float(row["cem_mean_success"]) for row in rows],
            "cem_mean_simulator_cost": [float(row["cem_mean_simulator_cost"]) for row in rows],
        }
        bootstrap = config["statistics"]["bootstrap_replicates"]
        base_seed = config["statistics"]["bootstrap_seed"]
        metrics = {}
        for offset, (name, values) in enumerate(metric_values.items()):
            metrics[name] = {
                "mean": float(np.mean(values)) if values else None,
                "cluster_bootstrap_95_ci": bootstrap_mean_ci(
                    values, replicates=bootstrap, seed=base_seed + offset
                ),
            }

        summary = {
            "status": "integration_smoke_only" if args.smoke else "completed_development_screen",
            "interpretation": "out_of_protocol_no_ranking_claim"
            if args.smoke
            else "not_confirmation_not_distillation_evidence",
            "seed_range_half_open": list(config["environment"]["episode_seed_range"]),
            "episodes": len(rows),
            "metrics": metrics,
            "screening_thresholds": config["screening_criteria"],
            "phase_screens_passed": {
                "candidate_availability": metrics["candidate_availability"]["mean"]
                is not None
                and metrics["candidate_availability"]["mean"]
                >= config["screening_criteria"]["candidate_availability_at_least"],
                "pairwise_accuracy": metrics["pairwise_accuracy"]["mean"]
                is not None
                and metrics["pairwise_accuracy"]["mean"]
                >= config["screening_criteria"]["pairwise_accuracy_at_least"],
                "normalized_regret": metrics["normalized_regret"]["mean"]
                is not None
                and metrics["normalized_regret"]["mean"]
                <= config["screening_criteria"]["normalized_regret_at_most"],
            },
            "compute": {
                "device": "cpu",
                "cuda_initialized": bool(torch.cuda.is_initialized()),
                "wall_seconds": time.perf_counter() - wall_start,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "gpu_hours": 0,
            },
            "rows": rows,
        }
        summary_path = output_root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        manifest = {
            "config_sha256": sha256(config_path),
            "runner_sha256": sha256(Path(__file__)),
            "local_cem_sha256": sha256(ROOT / "src" / "consequence_lab" / "cem.py"),
            "checkpoint_sha256": sha256(weights_path),
            "checkpoint_config_sha256": sha256(checkpoint_dir / "config.json"),
            "lewm_source_revision": config["source"]["lewm_source_revision"],
            "stable_worldmodel_revision": config["source"]["stable_worldmodel_revision"],
            "progress_sha256": sha256(progress_path),
            "summary_sha256": sha256(summary_path),
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
        print(f"Artifacts: {output_root}")
    finally:
        progress_stream.close()
        env.close()


if __name__ == "__main__":
    main()
