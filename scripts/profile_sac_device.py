"""Bounded CPU-vs-CUDA SAC throughput profile on a reward-free synthetic env."""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
import yaml
from gymnasium import spaces
from stable_baselines3 import SAC


class SyntheticEnv(gym.Env[np.ndarray, np.ndarray]):
    """Deterministic API/shape fixture with no task semantics or reward."""

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
        observation = self.np_random.uniform(-1.0, 1.0, size=(12,)).astype(np.float32)
        return observation, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite two-dimensional action")
        self._step += 1
        observation = self.np_random.uniform(-1.0, 1.0, size=(12,)).astype(np.float32)
        truncated = self._step >= self.episode_length
        return observation, 0.0, False, truncated, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert config["scope"] == "synthetic_optimizer_throughput_only_no_task_performance"
    assert config["environment"]["task_environment_imported"] is False
    assert config["environment"]["reward"] == 0.0
    assert platform.python_version() == config["runtime"]["python"]
    assert torch.__version__ == config["runtime"]["torch"]
    assert torch.version.cuda == config["runtime"]["torch_cuda_runtime"]
    assert stable_baselines3.__version__ == config["runtime"]["stable_baselines3"]
    assert gym.__version__ == config["runtime"]["gymnasium"]
    assert np.__version__ == config["runtime"]["numpy"]
    assert yaml.__version__ == config["runtime"]["pyyaml"]
    assert torch.cuda.is_available()

    free_cuda_bytes, total_cuda_bytes = torch.cuda.mem_get_info(0)
    free_cuda_mib = free_cuda_bytes / (1024**2)
    total_cuda_mib = total_cuda_bytes / (1024**2)
    assert free_cuda_mib >= float(config["minimum_free_cuda_memory_mib"]), (
        f"Insufficient free CUDA memory: {free_cuda_mib:.1f} MiB"
    )

    run_order = [
        {"device": device, "replicate": replicate}
        for replicate in range(int(config["repeats_per_device"]))
        for device in config["devices"]
    ]
    random.Random(int(config["run_order_seed"])).shuffle(run_order)
    results = []
    total_wall = 0.0
    total_cpu = 0.0
    torch.set_num_threads(int(config["torch_threads"]))

    for item in run_order:
        device_name = item["device"]
        replicate = int(item["replicate"])
        seed = int(config["base_seed"]) + replicate
        env = SyntheticEnv(int(config["environment"]["episode_length"]))
        env.action_space.seed(seed)
        sac = config["sac"]
        model = SAC(
            sac["policy"],
            env,
            learning_rate=float(sac["learning_rate"]),
            buffer_size=int(sac["buffer_size"]),
            learning_starts=int(sac["learning_starts"]),
            batch_size=int(sac["batch_size"]),
            train_freq=int(sac["train_frequency"]),
            gradient_steps=int(sac["gradient_steps"]),
            gamma=float(sac["gamma"]),
            tau=float(sac["tau"]),
            policy_kwargs={"net_arch": list(sac["network"])},
            seed=seed,
            device=device_name,
            verbose=0,
        )
        if device_name == "cuda":
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        model.learn(
            total_timesteps=int(config["training_steps_per_run"]),
            log_interval=None,
            progress_bar=False,
        )
        if device_name == "cuda":
            torch.cuda.synchronize(0)
        elapsed_wall = time.perf_counter() - wall_start
        elapsed_cpu = time.process_time() - cpu_start
        peak_allocated_mib = (
            torch.cuda.max_memory_allocated(0) / (1024**2) if device_name == "cuda" else None
        )
        results.append({
            "device": device_name,
            "replicate": replicate,
            "seed": seed,
            "steps": int(config["training_steps_per_run"]),
            "wall_seconds": round(elapsed_wall, 4),
            "process_cpu_seconds": round(elapsed_cpu, 4),
            "wall_seconds_per_step": round(elapsed_wall / int(config["training_steps_per_run"]), 6),
            "process_cpu_seconds_per_step": round(elapsed_cpu / int(config["training_steps_per_run"]), 6),
            "torch_cuda_peak_allocated_mib": (
                round(peak_allocated_mib, 3) if peak_allocated_mib is not None else None
            ),
        })
        total_wall += elapsed_wall
        total_cpu += elapsed_cpu
        env.close()
        del model
        assert elapsed_wall < float(config["limits"]["max_wall_seconds_total"])
        assert elapsed_cpu < float(config["limits"]["max_process_cpu_seconds_per_run"])
        assert total_wall < float(config["limits"]["max_wall_seconds_total"])
        assert total_cpu < float(config["limits"]["max_process_cpu_seconds_total"])

    print(json.dumps({
        "status": "PASS",
        "scope": config["scope"],
        "runtime": config["runtime"],
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_free_mib_before_runs": round(free_cuda_mib, 1),
        "cuda_total_mib": round(total_cuda_mib, 1),
        "torch_threads": int(config["torch_threads"]),
        "run_order": run_order,
        "results": results,
        "total_wall_seconds": round(total_wall, 4),
        "total_process_cpu_seconds": round(total_cpu, 4),
        "task_environment_imported": False,
        "task_resets_or_returns": False,
        "models_or_logs_saved": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
