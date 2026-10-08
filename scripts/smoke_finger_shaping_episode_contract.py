"""Check PBRS telescoping and DMC time-limit flags over a fixed-action smoke."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import yaml

from consequence_lab.finger_gym import FingerTurnHardGymnasium


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert cfg["scope"] == "fixed_action_wrapper_boundary_contract_only_no_performance_evaluation"
    assert cfg["artifact_policy"]["report_episode_return"] is False
    assert platform.python_version() == cfg["runtime"]["python"]
    for package, expected in cfg["runtime"].items():
        if package == "python":
            continue
        assert importlib.metadata.version(package.replace("_", "-")) == expected

    gamma = float(cfg["gamma"])
    raw_env = FingerTurnHardGymnasium()
    shaped_env = FingerTurnHardGymnasium(shaping_gamma=gamma)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    try:
        raw_obs, _ = raw_env.reset(seed=int(cfg["seed"]))
        shaped_obs, _ = shaped_env.reset(seed=int(cfg["seed"]))
        np.testing.assert_array_equal(raw_obs, shaped_obs)
        initial_phi = -float(raw_obs[-1])
        discounted_raw = 0.0
        discounted_shaped = 0.0
        discount = 1.0
        steps = 0
        terminated = truncated = False
        action = np.asarray(cfg["action"], dtype=np.float32)
        for _ in range(int(cfg["max_control_steps"])):
            raw_next, raw_reward, terminated, truncated, _ = raw_env.step(action)
            shaped_next, shaped_reward, shaped_terminated, shaped_truncated, _ = shaped_env.step(action)
            np.testing.assert_array_equal(raw_next, shaped_next)
            assert (terminated, truncated) == (shaped_terminated, shaped_truncated)
            discounted_raw += discount * raw_reward
            discounted_shaped += discount * shaped_reward
            discount *= gamma
            steps += 1
            raw_obs = raw_next
            if terminated or truncated:
                break
        final_phi = -float(raw_obs[-1])
        residual = discounted_shaped - (discounted_raw - initial_phi + discount * final_phi)
        assert abs(residual) <= 1e-6
        assert steps == int(cfg["max_control_steps"])
        assert not terminated and truncated, "DMC time limit must map to Gymnasium truncation"
    finally:
        raw_env.close()
        shaped_env.close()
    wall = time.perf_counter() - start_wall
    cpu = time.process_time() - start_cpu
    assert cpu < cfg["limits"]["max_process_cpu_seconds"]
    assert wall < cfg["limits"]["max_wall_seconds"]
    print(json.dumps({
        "status": "PASS",
        "scope": cfg["scope"],
        "seed": int(cfg["seed"]),
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "telescoping_identity_within_tolerance": True,
        "reward_or_state_values_reported": False,
        "process_cpu_seconds": round(cpu, 6),
        "wall_seconds": round(wall, 6),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
