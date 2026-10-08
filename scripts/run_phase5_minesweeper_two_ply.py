"""D049 fresh training-seed exact two-ply Minesweeper screen."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import yaml

from consequence_lab.minesweeper import (
    all_mine_layouts,
    candidate_cells,
    choose_min_risk,
    choose_risk_then_information,
    posterior_layouts,
    posterior_mine_risks,
    reveal_public_cell,
)
from consequence_lab.minesweeper_planner import choose_two_ply_action

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase5_minesweeper_two_ply.yaml"
DECISION_MODULE = ROOT / "src" / "consequence_lab" / "minesweeper.py"
PLANNER_MODULE = ROOT / "src" / "consequence_lab" / "minesweeper_planner.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _choose(method: str, board: np.ndarray, layouts, config: dict, reward):
    candidates = candidate_cells(board)
    if not candidates:
        return None
    posterior = posterior_layouts(
        board,
        mine_count=config["environment"]["mine_count"],
        layout_universe=layouts,
    )
    if method == "exact_posterior_min_risk":
        return choose_min_risk(board, posterior)
    if method == "exact_posterior_min_risk_then_information_gain":
        return choose_risk_then_information(board, posterior)
    if method == "exact_two_ply_belief_lookahead":
        return choose_two_ply_action(
            board,
            posterior,
            mine_count=config["environment"]["mine_count"],
            success_reward=reward["success"],
            failure_reward=reward["failure"],
        )
    raise ValueError(f"unknown policy: {method}")


def _run_episode(env_type, method: str, seed: int, config: dict, layouts) -> dict:
    env = env_type()
    env.reset(seed=seed)
    shape = tuple(config["environment"]["board_shape"])
    board = np.full(shape, -1, dtype=np.int8)
    reward = {
        "success": float(env.success_reward_scale),
        "failure": float(env.fail_reward_scale),
    }
    configured_reward = config["environment"]["reward"]
    if not np.isclose(reward["success"], configured_reward["success"], atol=1e-12):
        raise RuntimeError("environment success reward differs from frozen protocol")
    if not np.isclose(reward["failure"], configured_reward["failure"], atol=1e-12):
        raise RuntimeError("environment failure reward differs from frozen protocol")

    safe_reveals = actions = 0
    total_return = 0.0
    decision_cpu = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        decision_start = time.process_time()
        cell = _choose(method, board, layouts, config, reward)
        decision_cpu += time.process_time() - decision_start
        if cell is None:
            raise RuntimeError("episode has no action although environment is nonterminal")
        observation, reward_value, terminated, truncated, _info = env.step(
            np.asarray(cell, dtype=np.int64)
        )
        actions += 1
        total_return += float(reward_value)
        # Outcome inspection is evaluation-only. The action selector sees only
        # the public board and the safe-cell clue returned by the environment.
        mine_hit = terminated and np.isclose(float(reward_value), reward["failure"])
        if not mine_hit:
            safe_reveals += 1
            board = reveal_public_cell(board, cell, int(observation))

    won = safe_reveals == int(np.prod(shape)) - int(config["environment"]["mine_count"])
    return {
        "seed": int(seed),
        "won": bool(won),
        "return": float(total_return),
        "safe_reveals": int(safe_reveals),
        "actions": int(actions),
        "decision_cpu_seconds": float(decision_cpu),
    }


def _metric_summary(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "episodes": count,
        "wins": sum(int(row["won"]) for row in rows),
        "win_rate": sum(int(row["won"]) for row in rows) / count,
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_safe_reveals": float(np.mean([row["safe_reveals"] for row in rows])),
        "mean_actions": float(np.mean([row["actions"] for row in rows])),
        "mean_decision_cpu_seconds": float(
            np.mean([row["decision_cpu_seconds"] for row in rows])
        ),
    }


def _paired_bootstrap(
    left: list[dict], right: list[dict], field: str, replicates: int, seed: int
) -> dict:
    if [row["seed"] for row in left] != [row["seed"] for row in right]:
        raise ValueError("paired bootstrap requires identical ordered seeds")
    differences = [float(a[field]) - float(b[field]) for a, b in zip(left, right)]
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates)
    )
    return {
        "point": float(np.mean(differences)),
        "percentile_95": [
            means[int(0.025 * (replicates - 1))],
            means[int(0.975 * (replicates - 1))],
        ],
    }


def run() -> dict:
    config_bytes = CONFIG.read_bytes()
    config = yaml.safe_load(config_bytes)
    project, env_cfg = config["project"], config["environment"]
    if project["authorization"] != "D049_training_seed_two_ply_planner_screen":
        raise RuntimeError("D049 training-seed planner-screen authorization required")
    if any(
        project[key]
        for key in (
            "validation_or_test_access_authorized",
            "teacher_label_generation_authorized",
            "student_training_authorized",
            "calibration_fit_authorized",
        )
    ):
        raise RuntimeError("D049 is a training-seed planner screen only")
    if tuple(map(int, env_cfg["python_version"].split("."))) != sys.version_info[:3]:
        raise RuntimeError(f"Python {env_cfg['python_version']} required")
    for distribution, config_key in (
        ("popgym", "version"),
        ("gymnasium", "gymnasium_version"),
        ("numpy", "numpy_version"),
    ):
        actual = importlib.metadata.version(distribution)
        if actual != env_cfg[config_key]:
            raise RuntimeError(f"{distribution} {env_cfg[config_key]} required, got {actual}")

    output = (ROOT / config["artifacts"]["run_directory"]).resolve()
    runs_root = (ROOT / "artifacts" / "runs").resolve()
    output.relative_to(runs_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    seeds = list(range(env_cfg["seed_start"], env_cfg["seed_start"] + env_cfg["seed_count"]))
    if seeds[0] < 260100:
        raise RuntimeError("D049 seed range overlaps the completed D048 screen")
    source_paths = {
        "config": CONFIG,
        "runner": Path(__file__),
        "decision_module": DECISION_MODULE,
        "planner_module": PLANNER_MODULE,
    }
    lock = {
        "protocol": project["protocol"],
        "authorization": project["authorization"],
        "source_revision": env_cfg["source_revision"],
        "task": env_cfg["task"],
        "seed_role": "exploratory_training_only",
        "seeds": seeds,
        "source_hashes": {key: _sha(path.read_bytes()) for key, path in source_paths.items()},
        "environment_reset_or_episode_run_at_lock": False,
        "student_or_teacher_labels_generated_at_lock": False,
    }
    (output / "selection_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from popgym.envs.minesweeper import MineSweeperEasy

    layouts = all_mine_layouts(*env_cfg["board_shape"], env_cfg["mine_count"])
    methods = config["baselines"]["methods"]
    all_results: dict[str, list[dict]] = {method: [] for method in methods}
    started_cpu = time.process_time()
    for method in methods:
        for seed in seeds:
            all_results[method].append(
                _run_episode(MineSweeperEasy, method, seed, config, layouts)
            )
    total_cpu = time.process_time() - started_cpu
    if total_cpu > project["max_total_cpu_seconds"]:
        raise RuntimeError(f"pilot exceeded frozen process-CPU cap: {total_cpu:.3f} s")

    if any([row["seed"] for row in rows] != seeds for rows in all_results.values()):
        raise AssertionError("paired seed sets changed")
    metrics = {method: _metric_summary(rows) for method, rows in all_results.items()}
    primary = "exact_two_ply_belief_lookahead"
    baseline = "exact_posterior_min_risk"
    win_gain = _paired_bootstrap(
        all_results[primary],
        all_results[baseline],
        "won",
        config["statistics"]["paired_bootstrap_replicates"],
        config["statistics"]["paired_bootstrap_seed"],
    )
    return_gain = _paired_bootstrap(
        all_results[primary],
        all_results[baseline],
        "return",
        config["statistics"]["paired_bootstrap_replicates"],
        config["statistics"]["paired_bootstrap_seed"] + 1,
    )
    cpu_ratio = (
        metrics[primary]["mean_decision_cpu_seconds"]
        / metrics[baseline]["mean_decision_cpu_seconds"]
    )
    gate = {
        "win_rate_gain_minimum_met": win_gain["point"]
        >= config["gate"]["candidate_advance_requires_win_rate_gain_minimum"],
        "paired_interval_lower_bound_positive": win_gain["percentile_95"][0]
        > config["gate"]["paired_bootstrap_lower_bound_strictly_greater_than"],
        "decision_cpu_multiple_minimum_met": cpu_ratio
        >= config["gate"]["planner_cpu_multiple_over_min_risk_minimum"],
        "advance": False,
    }
    gate["advance"] = all(
        gate[key]
        for key in (
            "win_rate_gain_minimum_met",
            "paired_interval_lower_bound_positive",
            "decision_cpu_multiple_minimum_met",
        )
    )

    manifest = {
        "experiment": project["protocol"],
        "authorization": project["authorization"],
        "source_revision": env_cfg["source_revision"],
        "source_package": env_cfg["package"],
        "source_version": env_cfg["version"],
        "wheel_sha256_expected": env_cfg["wheel_sha256"],
        "software_versions": {
            "python": sys.version.split()[0],
            "popgym": importlib.metadata.version("popgym"),
            "gymnasium": importlib.metadata.version("gymnasium"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "task": env_cfg["task"],
        "seed_role": "exploratory_training_only",
        "seed_count": len(seeds),
        "seed_range_half_open": [seeds[0], seeds[-1] + 1],
        "candidate_set": "all currently unrevealed cells; stable row-major IDs",
        "decision_information": "public revealed-clue board and supplied candidates only",
        "teacher_labels_generated": False,
        "student_training_run": False,
        "validation_or_test_accessed": False,
        "calibration_fitted": False,
        "hidden_board_saved": False,
        "action_traces_saved": False,
        "posterior_layout_count_initial": len(layouts),
        "summary_metrics": metrics,
        "two_ply_minus_min_risk_win_rate": win_gain,
        "two_ply_minus_min_risk_mean_return": return_gain,
        "two_ply_to_min_risk_decision_cpu_ratio": cpu_ratio,
        "gate": gate,
        "total_process_cpu_seconds": total_cpu,
        "episode_aggregates": all_results,
        "source_hashes": lock["source_hashes"],
    }
    output_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    if len(output_bytes) > project["max_generated_artifact_bytes"]:
        raise RuntimeError("run artifact exceeds frozen byte cap")
    (output / "summary.json").write_bytes(output_bytes + b"\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "episode_aggregates"}, indent=2))
    return manifest


if __name__ == "__main__":
    run()
