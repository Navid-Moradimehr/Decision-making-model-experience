"""Bounded CPU dynamics/ranking experiment for the preregistered Phase 1.

The pinned upstream source is loaded from --source-root; no package install,
checkpoint, rendering, or GPU use is required. See configs/phase1_tworoom_dynamics.yaml.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.candidates import two_segment_sequences  # noqa: E402
from consequence_lab.environment import TwoRoomAdapter  # noqa: E402
from consequence_lab.objectives import goal_cost  # noqa: E402

OBS_SCALE = 224.0
ARENA_DIAGONAL = math.sqrt(2.0) * 224.0
HORIZON = 64
EPISODE_CAP = 200
TEST_SEEDS = range(204090, 204110)
TRAIN_SEEDS = range(204000, 204070)
VALIDATION_SEEDS = range(204070, 204080)
CALIBRATION_SEEDS = range(204080, 204090)
SNAPSHOT_STEPS = (0, 40, 80, 120, 160)


def load_environment(source_root: Path):
    """Register the pinned env while bypassing its unrelated expert import."""
    source_root = source_root.resolve()
    if not (source_root / "stable_worldmodel" / "envs" / "two_room" / "env.py").is_file():
        raise FileNotFoundError(f"not a stable-worldmodel source root: {source_root}")
    sys.path.insert(0, str(source_root))
    package = types.ModuleType("stable_worldmodel.envs.two_room")
    package.__path__ = [str(source_root / "stable_worldmodel" / "envs" / "two_room")]
    package.__package__ = "stable_worldmodel.envs.two_room"
    sys.modules[package.__name__] = package

    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    env = gym.make("swm/TwoRoom-v1")
    return env, TwoRoomAdapter(env)


def behavior_action(observation: np.ndarray, rng: np.random.Generator, mode: str, held: dict):
    if mode == "noisy_geometric":
        direction = observation[2:4] - observation[:2]
        direction = direction / max(float(np.linalg.norm(direction)), 1e-6)
        noisy = direction + rng.normal(0.0, 0.8, size=2)
        norm = float(np.linalg.norm(noisy))
        return (noisy / norm if norm > 1e-8 else np.zeros(2)).astype(np.float32)
    if held["remaining"] <= 0:
        angle = rng.uniform(0.0, 2.0 * np.pi)
        held["action"] = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        held["remaining"] = int(rng.integers(1, 9))
    held["remaining"] -= 1
    return held["action"].copy()


def collect_episode(adapter, seed: int, mode: str, keep_snapshots: bool = False) -> dict:
    rng = np.random.default_rng(seed ^ 0x5EED)
    observation, _ = adapter.reset(seed=seed)
    observations = [observation]
    actions = []
    snapshots = []
    held = {"remaining": 0, "action": np.zeros(2, np.float32)}
    for step in range(EPISODE_CAP):
        if keep_snapshots and step in SNAPSHOT_STEPS:
            snapshots.append((step, observation.copy(), adapter.snapshot()))
        action = behavior_action(observation, rng, mode, held)
        next_observation, _, terminated, truncated, _ = adapter.step(action)
        actions.append(action)
        observations.append(next_observation)
        observation = next_observation
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "mode": mode,
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "snapshots": snapshots,
    }


def transitions(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states, actions, deltas, seeds = [], [], [], []
    for episode in episodes:
        obs = episode["observations"]
        act = episode["actions"]
        states.append(obs[:-1])
        actions.append(act)
        deltas.append((obs[1:] - obs[:-1]) / OBS_SCALE)
        seeds.append(np.full(len(act), episode["seed"], dtype=np.int64))
    return tuple(np.concatenate(items, axis=0) for items in (states, actions, deltas, seeds))


def fit_ridge(states, actions, targets, alpha: float = 1e-4):
    x = np.concatenate([states / OBS_SCALE, actions, np.ones((len(states), 1))], axis=1)
    regularizer = np.eye(x.shape[1]) * alpha
    regularizer[-1, -1] = 0.0
    weights = np.linalg.solve(x.T @ x + regularizer, x.T @ targets)
    return weights


class NumpyPredictor:
    def __init__(self, weights: np.ndarray | None = None, network: nn.Module | None = None):
        self.weights = weights
        self.layers = None
        if network is not None:
            self.layers = [
                (layer.weight.detach().cpu().numpy().T.copy(), layer.bias.detach().cpu().numpy().copy())
                for layer in network.modules()
                if isinstance(layer, nn.Linear)
            ]

    def delta(self, states_px: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = np.concatenate([states_px / OBS_SCALE, actions], axis=1)
        if self.weights is not None:
            x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
            return (x @ self.weights) * OBS_SCALE
        for index, (weight, bias) in enumerate(self.layers):
            x = x @ weight + bias
            if index + 1 < len(self.layers):
                x = np.maximum(x, 0.0)
        return x * OBS_SCALE


def train_mlp(train_states, train_actions, train_targets, val_states, val_actions, val_targets):
    torch.manual_seed(204001)
    torch.set_num_threads(2)
    model = nn.Sequential(nn.Linear(12, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_train = torch.from_numpy(np.concatenate([train_states / OBS_SCALE, train_actions], axis=1)).float()
    y_train = torch.from_numpy(train_targets).float()
    x_val = torch.from_numpy(np.concatenate([val_states / OBS_SCALE, val_actions], axis=1)).float()
    y_val = torch.from_numpy(val_targets).float()
    generator = torch.Generator().manual_seed(204001)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, 101):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 256):
            indices = order[start : start + 256]
            prediction = model(x_train[indices])
            loss = torch.mean((prediction - y_train[indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(x_val) - y_val) ** 2))
        history.append({"epoch": epoch, "validation_delta_mse": val_loss})
        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_validation_delta_mse": best_val, "history": history}


def rmse(prediction, target) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def free_rollout_metrics(predictor, episodes, horizons=(16, 32, 64)):
    output = {}
    for horizon in horizons:
        errors = []
        for episode in episodes:
            observations, actions = episode["observations"], episode["actions"]
            for start in (0, 40, 80, 120):
                if start + horizon >= len(observations):
                    continue
                state = observations[start : start + 1].astype(np.float64)
                for action in actions[start : start + horizon]:
                    state += predictor.delta(state, action[None, :])
                errors.append((state[0] - observations[start + horizon]) / OBS_SCALE)
        error_array = np.asarray(errors)
        output[str(horizon)] = {
            "rollouts": len(errors),
            "endpoint_rmse_all_features_normalized": rmse(error_array, np.zeros_like(error_array)) if errors else None,
            "endpoint_rmse_agent_xy_pixels": rmse(error_array[:, :2] * OBS_SCALE, np.zeros_like(error_array[:, :2])) if errors else None,
        }
    return output


def rollout_predictor(predictor: NumpyPredictor, observation: np.ndarray, sequences: np.ndarray) -> np.ndarray:
    states = np.repeat(observation[None, :], len(sequences), axis=0).astype(np.float64)
    positions = np.empty((len(sequences), sequences.shape[1], 2), dtype=np.float32)
    absorbing = np.zeros(len(sequences), dtype=bool)
    for t in range(sequences.shape[1]):
        before = states.copy()
        predicted = states + predictor.delta(states, sequences[:, t]).astype(np.float64)
        states = np.where(absorbing[:, None], before, predicted)
        reached = np.linalg.norm(states[:, :2] - states[:, 2:4], axis=1) < 16.0
        absorbing |= reached
        positions[:, t] = states[:, :2]
    return positions


def ranking_metrics(predictor, adapter, samples, sequences):
    per_state = []
    for observation, snapshot in samples:
        predicted_positions = rollout_predictor(predictor, observation, sequences)
        actual_positions = adapter.rollout_candidates(snapshot, sequences)
        true_cost = goal_cost(actual_positions, observation[2:4], ARENA_DIAGONAL)
        predicted_cost = goal_cost(predicted_positions, observation[2:4], ARENA_DIAGONAL)
        i, j = np.triu_indices(len(true_cost), 1)
        informative = np.abs(true_cost[i] - true_cost[j]) > (1.0 / ARENA_DIAGONAL)
        correct = (predicted_cost[i] - predicted_cost[j]) * (true_cost[i] - true_cost[j]) > 0
        pair_accuracy = float(correct[informative].mean()) if informative.any() else float("nan")
        chosen = int(np.argmin(predicted_cost))
        reaches_goal = np.linalg.norm(actual_positions - observation[2:4], axis=-1) < 16.0
        per_state.append({
            "pairwise_accuracy": pair_accuracy,
            "top1_agreement": float(chosen == int(np.argmin(true_cost))),
            "normalized_regret": float((true_cost[chosen] - np.min(true_cost)) / 1.1),
            "oracle_candidate_success": bool(reaches_goal.any()),
            "informative_pairs": int(informative.sum()),
        })
    finite = [row for row in per_state if np.isfinite(row["pairwise_accuracy"])]
    return {
        "sampled_states": len(per_state),
        "pairwise_accuracy_macro": float(np.mean([row["pairwise_accuracy"] for row in finite])) if finite else None,
        "top1_agreement": float(np.mean([row["top1_agreement"] for row in per_state])) if per_state else None,
        "normalized_regret_mean": float(np.mean([row["normalized_regret"] for row in per_state])) if per_state else None,
        "oracle_candidate_success_rate": float(np.mean([row["oracle_candidate_success"] for row in per_state])) if per_state else None,
        "per_state": per_state,
    }


def planner_episode(adapter, seed, sequences, predictor=None):
    observation, _ = adapter.reset(seed=seed)
    for step in range(1, EPISODE_CAP + 1):
        if predictor is None:
            futures = adapter.rollout_candidates(adapter.snapshot(), sequences)
        else:
            futures = rollout_predictor(predictor, observation, sequences)
        costs = goal_cost(futures, observation[2:4], ARENA_DIAGONAL)
        observation, _, terminated, truncated, _ = adapter.step(sequences[int(np.argmin(costs)), 0])
        if terminated or truncated:
            return {"seed": seed, "success": bool(terminated), "steps": step}
    return {"seed": seed, "success": False, "steps": EPISODE_CAP}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "runs" / "phase1_tworoom_dynamics_v1")
    args = parser.parse_args()

    env, adapter = load_environment(args.source_root)
    started = time.perf_counter()
    all_episodes = []
    test_episodes = []
    for index, seed in enumerate(TRAIN_SEEDS):
        mode = "noisy_geometric" if index % 2 == 0 else "persistent_random"
        all_episodes.append(collect_episode(adapter, seed, mode))
    for index, seed in enumerate(VALIDATION_SEEDS):
        mode = "noisy_geometric" if index % 2 == 0 else "persistent_random"
        all_episodes.append(collect_episode(adapter, seed, mode))
    for index, seed in enumerate(CALIBRATION_SEEDS):
        mode = "noisy_geometric" if index % 2 == 0 else "persistent_random"
        all_episodes.append(collect_episode(adapter, seed, mode))
    for index, seed in enumerate(TEST_SEEDS):
        mode = "noisy_geometric" if index % 2 == 0 else "persistent_random"
        episode = collect_episode(adapter, seed, mode, keep_snapshots=True)
        all_episodes.append(episode)
        test_episodes.append(episode)

    train_episodes = [e for e in all_episodes if e["seed"] in TRAIN_SEEDS]
    val_episodes = [e for e in all_episodes if e["seed"] in VALIDATION_SEEDS]
    train_s, train_a, train_y, train_seed_ids = transitions(train_episodes)
    val_s, val_a, val_y, _ = transitions(val_episodes)
    if len(train_s) > 22000:
        raise RuntimeError("transition cap exceeded")

    network, train_report = train_mlp(train_s, train_a, train_y, val_s, val_a, val_y)
    mlp_predictor = NumpyPredictor(network=network)
    ridge = fit_ridge(train_s, train_a, train_y)
    ridge_predictor = NumpyPredictor(weights=ridge)
    persistence_predictor = NumpyPredictor(weights=np.zeros((13, 10), dtype=np.float64))

    test_s, test_a, test_y, _ = transitions(test_episodes)
    validation_metrics = {}
    for name, predictor in (("persistence", persistence_predictor), ("ridge", ridge_predictor), ("residual_mlp", mlp_predictor)):
        prediction = predictor.delta(val_s, val_a) / OBS_SCALE
        validation_metrics[name] = {
            "one_step_delta_rmse_all_features_normalized": rmse(prediction, val_y),
            "one_step_delta_rmse_agent_xy_pixels": rmse(prediction[:, :2] * OBS_SCALE, val_y[:, :2] * OBS_SCALE),
        }
    prediction_metrics = {}
    for name, predictor in (("persistence", persistence_predictor), ("ridge", ridge_predictor), ("residual_mlp", mlp_predictor)):
        prediction = predictor.delta(test_s, test_a) / OBS_SCALE
        prediction_metrics[name] = {
            "one_step_delta_rmse_all_features_normalized": rmse(prediction, test_y),
            "one_step_delta_rmse_agent_xy_pixels": rmse(prediction[:, :2] * OBS_SCALE, test_y[:, :2] * OBS_SCALE),
        }

    sequences = two_segment_sequences(HORIZON)
    ranking_samples = [(obs, snapshot) for episode in test_episodes for _, obs, snapshot in episode["snapshots"]]
    rank_results = {
        name: ranking_metrics(predictor, adapter, ranking_samples, sequences)
        for name, predictor in (("persistence", persistence_predictor), ("ridge", ridge_predictor), ("residual_mlp", mlp_predictor))
    }
    rollout_errors = {
        name: free_rollout_metrics(predictor, test_episodes)
        for name, predictor in (("persistence", persistence_predictor), ("ridge", ridge_predictor), ("residual_mlp", mlp_predictor))
    }

    closed_loop = {
        "greedy": [greedy_episode(adapter, seed) for seed in TEST_SEEDS],
        "simulator_shooting": [planner_episode(adapter, seed, sequences) for seed in TEST_SEEDS],
        "ridge_mpc": [planner_episode(adapter, seed, sequences, ridge_predictor) for seed in TEST_SEEDS],
        "residual_mlp_mpc": [planner_episode(adapter, seed, sequences, mlp_predictor) for seed in TEST_SEEDS],
    }
    env.close()

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "transitions.npz",
        observations=np.concatenate([e["observations"][:-1] for e in all_episodes]),
        actions=np.concatenate([e["actions"] for e in all_episodes]),
        next_observations=np.concatenate([e["observations"][1:] for e in all_episodes]),
        episode_seeds=np.concatenate([np.full(len(e["actions"]), e["seed"], dtype=np.int64) for e in all_episodes]),
    )
    torch.save(network.state_dict(), args.output / "residual_mlp.pt")
    summary = {
        "protocol": "phase1_tworoom_dynamics_v1",
        "environment_revision": "63988116d34cde56aea1240d5e58eb158ac67dc0",
        "device": "cpu",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "episode_seed_ranges": {"train": [204000, 204070], "validation": [204070, 204080], "calibration_reserved": [204080, 204090], "test": [204090, 204110]},
        "episodes": len(all_episodes),
        "transitions": {"train": len(train_s), "validation": len(val_s), "test": len(test_s), "total": sum(len(e["actions"]) for e in all_episodes)},
        "model_selection": train_report,
        "held_out_prediction_metrics": prediction_metrics,
        "validation_prediction_metrics": validation_metrics,
        "held_out_free_rollout_error": rollout_errors,
        "held_out_ranking": rank_results,
        "closed_loop_episodes": closed_loop,
        "closed_loop_success": {name: sum(row["success"] for row in rows) for name, rows in closed_loop.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "interpretation": "Numeric predictive-model pilot only; no teacher distribution or decision student trained.",
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "run_id": args.output.name,
        "phase": "phase_1",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": None, "dirty": True},
        "config_path": "configs/phase1_tworoom_dynamics.yaml",
        "config_sha256": hashlib.sha256((ROOT / "configs" / "phase1_tworoom_dynamics.yaml").read_bytes()).hexdigest(),
        "source_revision": "63988116d34cde56aea1240d5e58eb158ac67dc0",
        "summary_path": str(summary_path),
        "resources": {"device": "cpu", "wall_seconds": summary["elapsed_seconds"], "peak_vram_bytes": 0, "transitions_file_bytes": (args.output / "transitions.npz").stat().st_size},
        "notes": ["Calibration seed range is reserved and unused.", "No planner teacher labels or student trained."],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def greedy_episode(adapter, seed):
    observation, _ = adapter.reset(seed=seed)
    for step in range(1, EPISODE_CAP + 1):
        direction = observation[2:4] - observation[:2]
        norm = float(np.linalg.norm(direction))
        action = np.zeros(2, np.float32) if norm < 1e-8 else direction / norm
        observation, _, terminated, truncated, _ = adapter.step(action)
        if terminated or truncated:
            return {"seed": seed, "success": bool(terminated), "steps": step}
    return {"seed": seed, "success": False, "steps": EPISODE_CAP}


if __name__ == "__main__":
    main()
