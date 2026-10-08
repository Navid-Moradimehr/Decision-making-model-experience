"""Measure exact Finger CEM call cost only; do not report task performance."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml
from dm_control import suite

from consequence_lab.cem import run_bounded_cem


def _cost_function(env: Any, root_physics: Any, config: dict[str, Any], horizon: int):
    knots = horizon // int(config["knots"]["control_steps_per_knot"])
    repeat = int(config["knots"]["control_steps_per_knot"])
    gamma = float(config["gamma"])

    def cost(candidates: np.ndarray) -> np.ndarray:
        values = np.empty(len(candidates), dtype=np.float64)
        for candidate_index, sequence in enumerate(candidates):
            branch = root_physics.copy()
            current_obs = env._task.get_observation(branch)
            previous_phi = -float(current_obs["dist_to_target"])
            discounted_return = 0.0
            discount = 1.0
            transition_count = 0
            for knot_action in sequence.reshape(knots, 2):
                action = np.asarray(knot_action, dtype=env.action_spec().dtype)
                for _ in range(repeat):
                    env._task.before_step(action, branch)
                    branch.step(env._n_sub_steps)
                    env._task.after_step(branch)
                    next_obs = env._task.get_observation(branch)
                    reward = float(env._task.get_reward(branch))
                    next_phi = -float(next_obs["dist_to_target"])
                    shaped = reward + gamma * next_phi - previous_phi
                    discounted_return += discount * shaped
                    discount *= gamma
                    previous_phi = next_phi
                    transition_count += 1
                    if float(branch.data.time) >= 20.0:
                        break
                if transition_count >= horizon or float(branch.data.time) >= 20.0:
                    break
            values[candidate_index] = -discounted_return
            del branch
        return values

    return cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert config["scope"] == "planner_runtime_only_preflight_no_performance_evaluation"
    assert config["artifact_policy"]["output_timing_only"] is True
    assert float(config["initial_mean"]) == 0.0
    assert config["runtime"]["python"] == platform.python_version()
    for package, expected in config["runtime"].items():
        if package == "python":
            continue
        assert importlib.metadata.version(package.replace("_", "-")) == expected
    assert mujoco.__version__ == config["runtime"]["mujoco"]
    assert all(int(horizon) % int(config["knots"]["control_steps_per_knot"]) == 0 for horizon in config["horizons"])

    task = suite.load("finger", "turn_hard", task_kwargs={"random": int(config["seed"])})
    try:
        task.reset()
        root_physics = task.physics.copy()
        rows = []
        total_cpu = 0.0
        total_wall = 0.0
        for horizon in config["horizons"]:
            for repeat_index in range(int(config["repeats_per_horizon"])):
                seed = np.random.SeedSequence([int(config["seed"]), int(horizon), repeat_index])
                start_wall = time.perf_counter()
                start_cpu = time.process_time()
                result = run_bounded_cem(
                    _cost_function(task, root_physics, config, int(horizon)),
                    horizon_shape=(int(horizon) // int(config["knots"]["control_steps_per_knot"]), 2),
                    num_samples=int(config["samples"]),
                    iterations=int(config["iterations"]),
                    num_elites=int(config["elites"]),
                    seed=int(seed.generate_state(1, dtype=np.uint32)[0]),
                    lower_bound=float(config["action_bounds"][0]),
                    upper_bound=float(config["action_bounds"][1]),
                    initial_std=float(config["initial_std"]),
                    minimum_std=float(config["minimum_std"]),
                )
                del result
                wall = time.perf_counter() - start_wall
                cpu = time.process_time() - start_cpu
                expected_steps = int(config["samples"]) * int(config["iterations"]) * int(horizon)
                rows.append({
                    "horizon": int(horizon),
                    "repeat_index": repeat_index,
                    "rollout_transition_evaluations": expected_steps,
                    "wall_seconds": round(wall, 6),
                    "process_cpu_seconds": round(cpu, 6),
                })
                total_wall += wall
                total_cpu += cpu
                assert total_cpu < float(config["limits"]["max_process_cpu_seconds"])
                assert total_wall < float(config["limits"]["max_wall_seconds"])
        print(json.dumps({
            "status": "PASS",
            "scope": config["scope"],
            "seed": int(config["seed"]),
            "horizon_order": config["horizons"],
            "results": rows,
            "max_wall_seconds_per_call": round(max(row["wall_seconds"] for row in rows), 6),
            "max_process_cpu_seconds_per_call": round(max(row["process_cpu_seconds"] for row in rows), 6),
            "total_wall_seconds": round(total_wall, 6),
            "total_process_cpu_seconds": round(total_cpu, 6),
            "episode_return_or_success_reported": False,
            "candidate_costs_or_actions_saved": False,
        }, sort_keys=True))
    finally:
        task.close()


if __name__ == "__main__":
    main()
