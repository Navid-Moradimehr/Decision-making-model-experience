"""Check the pinned DMC -> Gymnasium -> SB3 SAC integration without scoring it."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

from consequence_lab.finger_gym import OBSERVATION_FIELDS, FingerTurnHardGymnasium


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert config["scope"] == "algorithm_integration_only_no_performance_evaluation"
    versions = config["runtime"]
    assert platform.python_version() == versions["python"]
    for package in ("dm-control", "mujoco", "gymnasium", "numpy", "stable-baselines3"):
        runtime_key = package.replace("-", "_")
        assert importlib.metadata.version(package) == versions[runtime_key]
    assert torch.__version__ == versions["torch"]
    assert not torch.cuda.is_available() and versions["device"] == "cpu"

    env = FingerTurnHardGymnasium()
    try:
        assert list(OBSERVATION_FIELDS) == config["observation"]["field_order"]
        assert list(env.observation_space.shape) == config["observation"]["flattened_shape"]
        assert list(env.action_space.shape) == config["action"]["shape"]
        np.testing.assert_array_equal(env.action_space.low, config["action"]["bounds"][0])
        np.testing.assert_array_equal(env.action_space.high, config["action"]["bounds"][1])
        env.action_space.seed(int(config["seed"]))
        check_env(env, warn=True, skip_render_check=True)
        observation_a, _ = env.reset(seed=int(config["seed"]))
        observation_b, _ = env.reset(seed=int(config["seed"]))
        np.testing.assert_array_equal(observation_a, observation_b)
        assert list(observation_a.shape) == config["observation"]["flattened_shape"]

        args_sac = config["sac"]
        model = SAC(
            args_sac["policy"],
            env,
            learning_rate=float(args_sac["learning_rate"]),
            buffer_size=int(args_sac["buffer_size"]),
            learning_starts=int(args_sac["learning_starts"]),
            batch_size=int(args_sac["batch_size"]),
            train_freq=int(args_sac["train_frequency"]),
            gradient_steps=int(args_sac["gradient_steps"]),
            gamma=float(args_sac["gamma"]),
            tau=float(args_sac["tau"]),
            policy_kwargs={"net_arch": config["sac"]["network"]},
            seed=int(config["seed"]),
            device="cpu",
            verbose=0,
        )
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        model.learn(total_timesteps=int(config["training_steps"]), log_interval=None, progress_bar=False)
        wall_elapsed = time.perf_counter() - wall_start
        cpu_elapsed = time.process_time() - cpu_start
        output = {
            "status": "PASS",
            "scope": config["scope"],
            "algorithm_steps": int(config["training_steps"]),
            "wall_seconds": round(wall_elapsed, 3),
            "process_cpu_seconds": round(cpu_elapsed, 3),
            "observation_shape": list(env.observation_space.shape),
            "action_shape": list(env.action_space.shape),
            "device": "cpu",
            "versions": {key: versions[key] for key in ("python", "dm_control", "mujoco", "gymnasium", "numpy", "torch", "stable_baselines3")},
            "task_performance_reported": False,
            "model_or_returns_saved": False,
        }
        assert cpu_elapsed < config["limits"]["max_process_cpu_seconds"]
        print(json.dumps(output, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
