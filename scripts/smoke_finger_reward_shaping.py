"""Verify optional Gymnasium reward shaping against a raw-reward twin."""

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
from consequence_lab.objectives import potential_shaped_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert cfg["scope"] == "one_transition_reward_wiring_only_no_performance_evaluation"
    assert cfg["artifact_policy"]["report_raw_or_shaped_rewards"] is False
    assert platform.python_version() == cfg["runtime"]["python"]
    for package, expected in cfg["runtime"].items():
        if package == "python":
            continue
        assert importlib.metadata.version(package.replace("_", "-")) == expected

    raw_env = FingerTurnHardGymnasium()
    shaped_env = FingerTurnHardGymnasium(shaping_gamma=float(cfg["gamma"]))
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    try:
        raw_observation, _ = raw_env.reset(seed=int(cfg["seed"]))
        shaped_observation, _ = shaped_env.reset(seed=int(cfg["seed"]))
        np.testing.assert_array_equal(raw_observation, shaped_observation)
        action = np.asarray(cfg["action"], dtype=np.float32)
        raw_next, raw_reward, raw_terminated, raw_truncated, _ = raw_env.step(action)
        shaped_next, shaped_reward, shaped_terminated, shaped_truncated, _ = shaped_env.step(action)
        np.testing.assert_array_equal(raw_next, shaped_next)
        expected = potential_shaped_reward(
            raw_reward, -float(raw_observation[-1]), -float(raw_next[-1]), float(cfg["gamma"])
        )
        np.testing.assert_allclose(shaped_reward, expected, rtol=0.0, atol=1e-7)
        assert (raw_terminated, raw_truncated) == (shaped_terminated, shaped_truncated)
        assert not raw_terminated and not raw_truncated
    finally:
        raw_env.close()
        shaped_env.close()
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    assert cpu < cfg["limits"]["max_process_cpu_seconds"]
    assert wall < cfg["limits"]["max_wall_seconds"]
    print(json.dumps({
        "status": "PASS",
        "scope": cfg["scope"],
        "seed": int(cfg["seed"]),
        "transitions_per_mode": 1,
        "raw_and_shaped_next_observation_equal": True,
        "shaping_formula_matches": True,
        "raw_or_shaped_rewards_reported": False,
        "process_cpu_seconds": round(cpu, 6),
        "wall_seconds": round(wall, 6),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
