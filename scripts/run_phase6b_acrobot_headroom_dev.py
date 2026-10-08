"""D052 development-only Acrobot baseline and exact shooting screen."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import time

import numpy as np
import yaml

from consequence_lab.acrobot import (
    acrobot_energy_pumping_action,
    acrobot_energy_shaping_action,
    acrobot_exact_step_batch,
    acrobot_state_from_observation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase6b_acrobot_headroom_dev_v3.yaml"
HELPER = ROOT / "src" / "consequence_lab" / "acrobot.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _height(state: np.ndarray) -> float:
    values = np.asarray(state, dtype=np.float64)
    theta1, theta2 = values[..., 0], values[..., 1]
    height = -np.cos(theta1) - np.cos(theta1 + theta2)
    return float(height) if np.ndim(height) == 0 else height


def _shooting_action(observation, seed: int, step: int, cfg: dict):
    state = acrobot_state_from_observation(observation)
    horizon = int(cfg["horizon"])
    count = int(cfg["candidate_count"])
    block = int(cfg["action_block_length"])
    rng = np.random.default_rng(seed * 1_000_003 + step * 97 + horizon * 13 + count)
    candidates = rng.integers(0, 3, size=(count, horizon), dtype=np.int8)
    # Include the constant legal-action sequences in every finite set.
    candidates[:3] = np.repeat(np.arange(3, dtype=np.int8)[:, None], horizon, axis=1)
    if block > 1:
        candidates = candidates.copy()
        for row in range(3, count):
            candidates[row] = np.repeat(candidates[row, ::block], block)[:horizon]

    branch_states = np.repeat(state[None, :], count, axis=0)
    reached = np.zeros(count, dtype=bool)
    success_cost = np.full(count, np.inf, dtype=np.float64)
    simulated_steps = 0
    for time_index in range(horizon):
        active = ~reached
        if not np.any(active):
            break
        active_indices = np.flatnonzero(active)
        branch_states[active] = acrobot_exact_step_batch(
            branch_states[active], candidates[active, time_index]
        )
        simulated_steps += len(active_indices)
        heights = -np.cos(branch_states[active, 0]) - np.cos(
            branch_states[active, 0] + branch_states[active, 1]
        )
        newly_reached = heights > 1.0
        reached_indices = active_indices[newly_reached]
        reached[reached_indices] = True
        success_cost[reached_indices] = float(time_index)
    terminal_progress = np.clip((_height(branch_states) + 2.0) / 3.0, 0.0, 1.0)
    costs = float(horizon) + float(cfg["terminal_height_penalty_scale"]) * (1.0 - terminal_progress)
    costs[reached] = success_cost[reached]
    selected = int(np.argmin(costs))
    return int(candidates[selected, 0]), bool(np.any(costs < horizon)), simulated_steps


def _evaluate_policy(env_class, seeds: list[int], policy_name: str, policy, config=None):
    episodes = []
    decision_cpu = 0.0
    branch_steps = 0
    candidate_successes = 0
    planner_decisions = 0
    for seed in seeds:
        env = env_class()
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        success = False
        steps_to_success = None
        for step in range(500):
            started = time.process_time()
            if policy_name == "exact_shooting":
                action, available, simulated = _shooting_action(observation, seed, step, config)
                candidate_successes += int(available)
                planner_decisions += 1
                branch_steps += simulated
            else:
                action = policy(observation)
            decision_cpu += time.process_time() - started
            observation, reward, terminated, _, _ = env.step(action)
            episode_return += float(reward)
            if terminated:
                success = True
                steps_to_success = step + 1
                break
        env.close()
        episodes.append(
            {
                "seed": int(seed),
                "success": success,
                "return": episode_return,
                "steps_to_success": steps_to_success,
            }
        )
    return {
        "policy": policy_name,
        "episodes": episodes,
        "successes": int(sum(item["success"] for item in episodes)),
        "episode_count": len(episodes),
        "success_rate": float(np.mean([item["success"] for item in episodes])),
        "mean_return": float(np.mean([item["return"] for item in episodes])),
        "mean_steps_to_success": float(
            np.mean([item["steps_to_success"] for item in episodes if item["steps_to_success"] is not None])
        ) if any(item["steps_to_success"] is not None for item in episodes) else None,
        "decision_process_cpu_seconds": decision_cpu,
        "decision_process_cpu_seconds_per_episode": decision_cpu / len(episodes),
        "simulator_branch_steps": branch_steps,
        "fraction_decisions_with_successful_candidate": (
            candidate_successes / planner_decisions if planner_decisions else None
        ),
    }


def _summarize(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "episodes"}


def _check_vectorized_transition_parity(env_class, cfg: dict) -> dict:
    parity_cfg = cfg["parity"]
    max_state_error = 0.0
    branches = states = 0
    for seed in range(parity_cfg["seed_start"], parity_cfg["seed_start"] + parity_cfg["seed_count"]):
        env = env_class()
        observation, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed ^ 0xACB07)
        for _ in range(parity_cfg["roll_in_steps_per_seed"]):
            public_state = acrobot_state_from_observation(observation)
            predicted = acrobot_exact_step_batch(
                np.repeat(public_state[None, :], parity_cfg["actions_per_state"], axis=0),
                np.arange(parity_cfg["actions_per_state"], dtype=np.int64),
            )
            states += 1
            for action in range(parity_cfg["actions_per_state"]):
                branch = copy.deepcopy(env)
                branch.state = public_state.astype(np.float32, copy=True)
                result = branch.step(action)
                error = float(np.max(np.abs(np.asarray(branch.state, dtype=np.float64) - predicted[action])))
                max_state_error = max(max_state_error, error)
                predicted_terminated = bool(
                    -np.cos(predicted[action, 0])
                    - np.cos(predicted[action, 0] + predicted[action, 1])
                    > 1.0
                )
                if error > parity_cfg["absolute_tolerance"]:
                    raise AssertionError(f"vectorized dynamics mismatch: {error} > tolerance")
                if result[2] != predicted_terminated or result[1] != (0.0 if predicted_terminated else -1.0):
                    raise AssertionError("vectorized dynamics changed Gymnasium reward/termination")
                branches += 1
            observation, _, terminated, _, _ = env.step(int(rng.integers(0, 3)))
            if terminated:
                break
        env.close()
    return {
        "status": "PASS",
        "states": states,
        "branches": branches,
        "max_full_state_absolute_error": max_state_error,
        "tolerance": float(parity_cfg["absolute_tolerance"]),
        "seeds": [parity_cfg["seed_start"], parity_cfg["seed_start"] + parity_cfg["seed_count"]],
    }


def run() -> dict:
    config_bytes = CONFIG.read_bytes()
    cfg = yaml.safe_load(config_bytes)
    if cfg["project"]["authorization"] != "D054_acrobot_regression_gated_development_only":
        raise RuntimeError("D054 regression-gated development-only authorization required")
    if importlib.metadata.version("gymnasium") != cfg["environment"]["version"]:
        raise RuntimeError("Gymnasium version does not match the frozen protocol")
    output = (ROOT / cfg["artifacts"]["run_directory"]).resolve()
    runs_root = (ROOT / "artifacts" / "runs").resolve()
    output.relative_to(runs_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    tuning_seeds = list(range(cfg["seeds"]["tuning_start"], cfg["seeds"]["tuning_start"] + cfg["seeds"]["tuning_count"]))
    development_seeds = list(range(cfg["seeds"]["development_start"], cfg["seeds"]["development_start"] + cfg["seeds"]["development_count"]))
    runner_path = Path(__file__)
    source_hashes = {
        "config": _sha(CONFIG.read_bytes()),
        "runner": _sha(runner_path.read_bytes()),
        "helper": _sha(HELPER.read_bytes()),
    }
    lock = {
        "protocol": cfg["project"]["protocol"],
        "authorization": cfg["project"]["authorization"],
        "tuning_seeds": tuning_seeds,
        "development_seeds": development_seeds,
        "confirmation_seeds_not_opened": [
            cfg["seeds"]["confirmation_start"],
            cfg["seeds"]["confirmation_start"] + cfg["seeds"]["confirmation_count"],
        ],
        "source_hashes": source_hashes,
        "environment_reset_at_lock": False,
        "heldout_confirmation_accessed": False,
    }
    (output / "selection_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from gymnasium.envs.classic_control.acrobot import AcrobotEnv

    started = time.process_time()
    parity = _check_vectorized_transition_parity(AcrobotEnv, cfg)
    gain_grid = cfg["classical_baselines"]["collocated_pfl_energy_shaping"]["gain_grid"]
    gains = list(itertools.product(gain_grid["k1"], gain_grid["k2"], gain_grid["k3"]))
    tuned_controllers = []
    for gain in gains:
        evaluated = _evaluate_policy(
            AcrobotEnv,
            tuning_seeds,
            "pfl_energy_shaping",
            lambda observation, gain=gain: acrobot_energy_shaping_action(observation, gain),
        )
        evaluated["gains"] = list(map(float, gain))
        tuned_controllers.append(evaluated)
    pumping_tuning = _evaluate_policy(
        AcrobotEnv, tuning_seeds, "energy_pumping", acrobot_energy_pumping_action
    )

    planner_cfg = cfg["exact_simulator_shooting"]
    planner_options = [
        {
            "horizon": int(horizon),
            "candidate_count": int(count),
            "action_block_length": int(block),
            "terminal_height_penalty_scale": float(planner_cfg["terminal_height_penalty_scale"]),
        }
        for horizon, count, block in itertools.product(
            planner_cfg["horizons"],
            planner_cfg["candidate_counts"],
            planner_cfg["action_block_lengths"],
        )
    ]
    planner_tuning = []
    for option in planner_options:
        evaluated = _evaluate_policy(
            AcrobotEnv, tuning_seeds, "exact_shooting", None, config=option
        )
        evaluated["planner_config"] = option
        planner_tuning.append(evaluated)

    def controller_key(result):
        return (result["success_rate"], result["mean_return"], -result["decision_process_cpu_seconds_per_episode"])

    best_pfl = max(tuned_controllers, key=controller_key)
    best_classical = max([best_pfl, pumping_tuning], key=controller_key)
    best_planner = max(planner_tuning, key=controller_key)

    development = {
        "energy_pumping": _evaluate_policy(
            AcrobotEnv, development_seeds, "energy_pumping", acrobot_energy_pumping_action
        ),
        "tuned_pfl": _evaluate_policy(
            AcrobotEnv,
            development_seeds,
            "pfl_energy_shaping",
            lambda observation: acrobot_energy_shaping_action(observation, tuple(best_pfl["gains"])),
        ),
        "selected_tuning_classical": _evaluate_policy(
            AcrobotEnv,
            development_seeds,
            best_classical["policy"],
            (lambda observation: acrobot_energy_shaping_action(observation, tuple(best_classical["gains"])))
            if best_classical["policy"] == "pfl_energy_shaping"
            else acrobot_energy_pumping_action,
        ),
        "selected_exact_shooting": _evaluate_policy(
            AcrobotEnv, development_seeds, "exact_shooting", None, config=best_planner["planner_config"]
        ),
    }
    elapsed = time.process_time() - started
    if elapsed > cfg["project"]["max_process_cpu_seconds"]:
        raise RuntimeError(f"development run exceeded CPU cap: {elapsed:.2f}s")

    summary = {
        "experiment": cfg["project"]["protocol"],
        "authorization": cfg["project"]["authorization"],
        "environment": cfg["environment"]["task"],
        "gymnasium_version": importlib.metadata.version("gymnasium"),
        "source_revision": cfg["environment"]["source_revision"],
        "seed_ranges": {
            "vectorized_transition_parity_half_open": [
                cfg["parity"]["seed_start"],
                cfg["parity"]["seed_start"] + cfg["parity"]["seed_count"],
            ],
            "tuning_half_open": [tuning_seeds[0], tuning_seeds[-1] + 1],
            "development_half_open": [development_seeds[0], development_seeds[-1] + 1],
            "confirmation_sealed_half_open": [
                cfg["seeds"]["confirmation_start"],
                cfg["seeds"]["confirmation_start"] + cfg["seeds"]["confirmation_count"],
            ],
        },
        "tuning": {
            "energy_pumping": _summarize(pumping_tuning),
            "best_pfl": _summarize(best_pfl),
            "best_planner": _summarize(best_planner),
            "all_pfl_gains": [_summarize(item) | {"gains": item["gains"]} for item in tuned_controllers],
            "all_planner_configs": [_summarize(item) | {"planner_config": item["planner_config"]} for item in planner_tuning],
            "selected_classical_policy": best_classical["policy"],
            "selected_classical_gains": best_classical.get("gains"),
        },
        "development": {name: _summarize(value) for name, value in development.items()},
        "vectorized_transition_parity": parity,
        "heldout_confirmation_accessed": False,
        "labels_or_student_training": False,
        "process_cpu_seconds_total": elapsed,
        "source_hashes": source_hashes,
    }
    payload = json.dumps(summary, indent=2, sort_keys=True).encode("utf-8")
    if len(payload) > cfg["project"]["max_generated_artifact_bytes"]:
        raise RuntimeError("development summary exceeds artifact cap")
    (output / "summary.json").write_bytes(payload + b"\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
