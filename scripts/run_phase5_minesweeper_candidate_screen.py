"""D048 training-seed heuristic screen on pinned POPGym MineSweeperEasy."""

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

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase5_minesweeper_candidate_screen.yaml"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _metric_summary(rows: list[dict]) -> dict:
    count = len(rows)
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    wins = np.asarray([row["won"] for row in rows], dtype=np.float64)
    return {
        "episodes": count,
        "wins": int(wins.sum()),
        "win_rate": float(wins.mean()),
        "mean_return": float(returns.mean()),
        "mean_safe_reveals": float(np.mean([row["safe_reveals"] for row in rows])),
        "mean_actions": float(np.mean([row["actions"] for row in rows])),
        "mean_cpu_seconds": float(np.mean([row["cpu_seconds"] for row in rows])),
        "brier_mine_risk": _brier(rows),
    }


def _brier(rows: list[dict]) -> float | None:
    pairs = [
        (action["predicted_mine_risk"], action["mine_hit"])
        for row in rows
        for action in row["action_log"]
        if action["predicted_mine_risk"] is not None
    ]
    if not pairs:
        return None
    return float(np.mean([(probability - outcome) ** 2 for probability, outcome in pairs]))


def _paired_bootstrap(left: list[dict], right: list[dict], metric: str, replicates: int, seed: int):
    differences = [float(a[metric]) - float(b[metric]) for a, b in zip(left, right)]
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates))
    return {
        "point": sum(differences) / n,
        "percentile_95": [means[int(0.025 * (replicates - 1))], means[int(0.975 * (replicates - 1))]],
    }


def _rng_for(salt: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{salt}:{seed}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _choose(method: str, board: np.ndarray, layouts, rng: random.Random):
    candidates = candidate_cells(board)
    if not candidates:
        return None, None
    if method == "random_unrevealed":
        return rng.choice(candidates), None
    if method == "row_major_first_unrevealed":
        return candidates[0], None
    posterior = posterior_layouts(board, mine_count=2, layout_universe=layouts)
    risks = posterior_mine_risks(board, posterior)
    if method == "exact_posterior_min_risk":
        return choose_min_risk(board, posterior), risks
    if method == "exact_posterior_min_risk_then_information_gain":
        return choose_risk_then_information(board, posterior), risks
    raise ValueError(f"unknown policy: {method}")


def _run_episode(env_type, method: str, seed: int, config: dict, layouts) -> dict:
    env = env_type()
    _, _ = env.reset(seed=seed)
    board = np.full(tuple(config["environment"]["board_shape"]), -1, dtype=np.int8)
    rng = _rng_for(config["baselines"]["random_seed_salt"], seed)
    action_log, total_return, safe_reveals, actions = [], 0.0, 0, 0
    started = time.perf_counter()
    terminated = truncated = False
    while not (terminated or truncated):
        cell, risks = _choose(method, board, layouts, rng)
        if cell is None:
            raise RuntimeError("episode did not terminate although no valid candidates remain")
        predicted_risk = None if risks is None else risks[cell]
        observation, reward, terminated, truncated, _info = env.step(np.asarray(cell, dtype=np.int64))
        actions += 1
        total_return += float(reward)
        # The selected cell is known to have been unrevealed. POPGym uses a
        # negative terminal reward only for a mine; no hidden state is read.
        mine_hit = bool(terminated and float(reward) < -0.25)
        action_log.append({
            "cell_id": int(cell[0] * board.shape[1] + cell[1]),
            "predicted_mine_risk": predicted_risk,
            "mine_hit": mine_hit,
            "reward": float(reward),
        })
        if mine_hit:
            continue
        safe_reveals += 1
        board = reveal_public_cell(board, cell, int(observation))
    return {
        "seed": seed,
        "won": safe_reveals == int(np.prod(board.shape)) - config["environment"]["mine_count"],
        "return": total_return,
        "safe_reveals": safe_reveals,
        "actions": actions,
        "cpu_seconds": time.perf_counter() - started,
        "action_log": action_log,
    }


def run() -> dict:
    config_bytes = CONFIG.read_bytes()
    config = yaml.safe_load(config_bytes)
    project, env_cfg = config["project"], config["environment"]
    if project["authorization"] != "D048_training_seed_baseline_screen_only":
        raise RuntimeError("D048 baseline-screen authorization is required")
    if project["validation_or_test_access_authorized"] or project["teacher_label_generation_authorized"] or project["student_training_authorized"] or project["calibration_fit_authorized"]:
        raise RuntimeError("D048 must remain training-seed baseline-only")

    expected_python = tuple(map(int, env_cfg["python_version"].split(".")))
    if sys.version_info[:3] != expected_python:
        raise RuntimeError(f"Python {env_cfg['python_version']} is required, got {sys.version_info[:3]}")
    for distribution, config_key in (("popgym", "version"), ("gymnasium", "gymnasium_version"), ("numpy", "numpy_version")):
        actual = importlib.metadata.version(distribution)
        if actual != env_cfg[config_key]:
            raise RuntimeError(f"{distribution} {env_cfg[config_key]} is required, got {actual}")

    from popgym.envs.minesweeper import MineSweeperEasy

    output = (ROOT / config["artifacts"]["run_directory"]).resolve()
    runs_root = (ROOT / "artifacts" / "runs").resolve()
    output.relative_to(runs_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    seeds = list(range(env_cfg["seed_start"], env_cfg["seed_start"] + env_cfg["seed_count"]))
    lock = {
        "protocol": project["protocol"],
        "authorization": project["authorization"],
        "source_revision": env_cfg["source_revision"],
        "task": env_cfg["task"],
        "seed_role": "exploratory_training_only",
        "seeds": seeds,
        "config_sha256": _sha(config_bytes),
        "runner_sha256": _sha(Path(__file__).read_bytes()),
        "environment_reset_or_episode_run_at_lock": False,
    }
    (output / "selection_lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    layouts = all_mine_layouts(*env_cfg["board_shape"], env_cfg["mine_count"])
    methods = config["baselines"]["methods"]
    all_results: dict[str, list[dict]] = {name: [] for name in methods}
    started_cpu = time.process_time()
    for method in methods:
        for seed in seeds:
            all_results[method].append(_run_episode(MineSweeperEasy, method, seed, config, layouts))
    total_cpu = time.process_time() - started_cpu
    if total_cpu > project["max_total_cpu_seconds"]:
        raise RuntimeError(f"pilot exceeded frozen CPU cap: {total_cpu:.3f} s")

    seed_lists = {method: [row["seed"] for row in rows] for method, rows in all_results.items()}
    if any(values != seeds for values in seed_lists.values()):
        raise AssertionError("paired episode seed sets changed")
    summary_metrics = {method: _metric_summary(rows) for method, rows in all_results.items()}
    primary = "exact_posterior_min_risk_then_information_gain"
    comparison = "exact_posterior_min_risk"
    gain_return = _paired_bootstrap(
        all_results[primary], all_results[comparison], "return",
        config["statistics"]["paired_bootstrap_replicates"],
        config["statistics"]["paired_bootstrap_seed"],
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
        "candidate_set": "all currently unrevealed cells; stable IDs row-major",
        "decision_information": "public revealed-clue board only; no hidden grid reads",
        "teacher_labels_generated": False,
        "student_training_run": False,
        "validation_or_test_accessed": False,
        "calibration_fitted": False,
        "hidden_board_saved": False,
        "posterior_layout_count_initial": len(layouts),
        "summary_metrics": summary_metrics,
        "primary_posterior_info_vs_posterior_risk_mean_return": gain_return,
        "total_cpu_seconds": total_cpu,
        "episode_results": all_results,
        "source_hashes": {
            "config": _sha(CONFIG.read_bytes()),
            "runner": _sha(Path(__file__).read_bytes()),
            "decision_module": _sha((ROOT / "src" / "consequence_lab" / "minesweeper.py").read_bytes()),
        },
    }
    output_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    if len(output_bytes) > project["max_generated_artifact_bytes"]:
        raise RuntimeError("pilot artifacts exceed frozen size cap")
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    summary_path.write_bytes(output_bytes + b"\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "episode_results"}, indent=2))
    return manifest


if __name__ == "__main__":
    run()
