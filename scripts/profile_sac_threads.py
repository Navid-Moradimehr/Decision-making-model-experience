"""Profile SB3 SAC optimizer threading on a reward-free synthetic Gymnasium env."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from gymnasium import spaces
from stable_baselines3 import SAC


class SyntheticEnv(gym.Env[np.ndarray, np.ndarray]):
    """Cheap deterministic shape-matched task; deliberately has no task reward."""

    metadata = {"render_modes": []}

    def __init__(self, episode_length: int) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.episode_length = episode_length
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        super().reset(seed=seed)
        self._step = 0
        obs = self.np_random.uniform(-1.0, 1.0, size=(12,)).astype(np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite 2D action")
        self._step += 1
        obs = self.np_random.uniform(-1.0, 1.0, size=(12,)).astype(np.float32)
        truncated = self._step >= self.episode_length
        return obs, 0.0, False, truncated, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert cfg["scope"] == "synthetic_optimizer_throughput_only_no_task_performance"
    assert cfg["environment"]["task_environment_imported"] is False
    assert cfg["environment"]["reward"] == 0.0
    assert platform.python_version() == "3.12.3"
    assert torch.__version__ == "2.7.1+cpu"
    assert not torch.cuda.is_available()

    order = list(cfg["thread_counts"])
    random.Random(int(cfg["seed"])).shuffle(order)
    results = []
    total_cpu = 0.0
    total_wall = 0.0
    for threads in order:
        torch.set_num_threads(int(threads))
        env = SyntheticEnv(int(cfg["environment"]["episode_length"]))
        env.action_space.seed(int(cfg["seed"]))
        sac = cfg["sac"]
        model = SAC(
            sac["policy"], env,
            learning_rate=float(sac["learning_rate"]),
            buffer_size=int(sac["buffer_size"]),
            learning_starts=int(sac["learning_starts"]),
            batch_size=int(sac["batch_size"]),
            train_freq=int(sac["train_frequency"]),
            gradient_steps=int(sac["gradient_steps"]),
            gamma=float(sac["gamma"]),
            tau=float(sac["tau"]),
            policy_kwargs={"net_arch": list(sac["network"])},
            seed=int(cfg["seed"]),
            device="cpu",
            verbose=0,
        )
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        model.learn(
            total_timesteps=int(cfg["training_steps_per_thread_count"]),
            log_interval=None,
            progress_bar=False,
        )
        elapsed_wall = time.perf_counter() - wall_start
        elapsed_cpu = time.process_time() - cpu_start
        results.append({
            "torch_threads": int(threads),
            "steps": int(cfg["training_steps_per_thread_count"]),
            "wall_seconds": round(elapsed_wall, 4),
            "process_cpu_seconds": round(elapsed_cpu, 4),
            "process_cpu_seconds_per_step": round(elapsed_cpu / int(cfg["training_steps_per_thread_count"]), 6),
        })
        total_cpu += elapsed_cpu
        total_wall += elapsed_wall
        env.close()
        del model
        assert elapsed_cpu < float(cfg["limits"]["max_process_cpu_seconds_per_run"])
        assert total_cpu < float(cfg["limits"]["max_process_cpu_seconds_total"])
        assert total_wall < float(cfg["limits"]["max_wall_seconds_total"])

    print(json.dumps({
        "status": "PASS",
        "scope": cfg["scope"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "synthetic_reward_only": 0.0,
        "task_resets_or_returns": False,
        "run_order": order,
        "results": results,
        "total_wall_seconds": round(total_wall, 4),
        "total_process_cpu_seconds": round(total_cpu, 4),
        "saved_model_or_logs": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
