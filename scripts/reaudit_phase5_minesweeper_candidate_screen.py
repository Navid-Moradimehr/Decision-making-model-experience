"""Independent trace, public-information, and metric audit for D048."""

from __future__ import annotations

from itertools import combinations
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "runs" / "phase5_minesweeper_candidate_screen_v1"
CONFIG = ROOT / "configs" / "phase5_minesweeper_candidate_screen.yaml"
RUNNER = ROOT / "scripts" / "run_phase5_minesweeper_candidate_screen.py"
DECISION_MODULE = ROOT / "src" / "consequence_lab" / "minesweeper.py"
AUDIT = RUN / "independent_audit_v1.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _neighbors(cell: tuple[int, int], shape: tuple[int, int]):
    row, col = cell
    return [
        (r, c)
        for r in range(max(0, row - 1), min(shape[0], row + 2))
        for c in range(max(0, col - 1), min(shape[1], col + 2))
        if (r, c) != cell
    ]


def _layouts(shape: tuple[int, int], mine_count: int):
    cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    return list(combinations(cells, mine_count))


def _clue(cell, layout, shape):
    mine_set = set(layout)
    return sum(neighbor in mine_set for neighbor in _neighbors(cell, shape))


def _posterior(board: np.ndarray, universe):
    shape = board.shape
    observations = [((int(r), int(c)), int(board[r, c])) for r, c in np.argwhere(board >= 0)]
    possible = []
    for layout in universe:
        mine_set = set(layout)
        if any(cell in mine_set for cell, _ in observations):
            continue
        if all(_clue(cell, layout, shape) == clue for cell, clue in observations):
            possible.append(layout)
    if not possible:
        raise AssertionError("independent posterior became empty")
    return possible


def _entropy(counts):
    total = sum(counts)
    return -sum((n / total) * math.log2(n / total) for n in counts if n)


def _reference_choice(method, board, universe, rng):
    candidates = [(int(r), int(c)) for r, c in np.argwhere(board == -1)]
    if not candidates:
        return None
    if method == "random_unrevealed":
        return rng.choice(candidates)
    if method == "row_major_first_unrevealed":
        return candidates[0]
    possible = _posterior(board, universe)
    risks = {
        cell: sum(cell in layout for layout in possible) / len(possible)
        for cell in candidates
    }
    if method == "exact_posterior_min_risk":
        return min(candidates, key=lambda cell: (risks[cell], cell))
    if method != "exact_posterior_min_risk_then_information_gain":
        raise AssertionError(f"unknown saved method {method}")
    gains = {}
    for cell in candidates:
        safe = [layout for layout in possible if cell not in layout]
        if not safe:
            gains[cell] = 0.0
            continue
        clue_counts: dict[int, int] = {}
        for layout in safe:
            value = _clue(cell, layout, board.shape)
            clue_counts[value] = clue_counts.get(value, 0) + 1
        gains[cell] = math.log2(len(safe)) - sum(
            (count / len(safe)) * math.log2(count) for count in clue_counts.values()
        )
    return min(candidates, key=lambda cell: (risks[cell], -gains[cell], cell))


def _paired_interval(left_rows, right_rows, replicates: int, seed: int):
    deltas = [float(a["return"]) - float(b["return"]) for a, b in zip(left_rows, right_rows)]
    rng = random.Random(seed)
    n = len(deltas)
    samples = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates))
    return [sum(deltas) / n, samples[int(0.025 * (replicates - 1))], samples[int(0.975 * (replicates - 1))]]


def run() -> dict:
    if AUDIT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {AUDIT}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((RUN / "selection_lock.json").read_text(encoding="utf-8"))
    env_cfg = config["environment"]
    seeds = list(range(env_cfg["seed_start"], env_cfg["seed_start"] + env_cfg["seed_count"]))
    if lock["seeds"] != seeds or lock["environment_reset_or_episode_run_at_lock"] is not False:
        raise AssertionError("seed lock mismatch or content was read before locking")
    if summary["validation_or_test_accessed"] is not False or summary["teacher_labels_generated"] is not False or summary["student_training_run"] is not False or summary["hidden_board_saved"] is not False:
        raise AssertionError("run violated D048's data/training scope")
    if importlib.metadata.version("popgym") != env_cfg["version"]:
        raise AssertionError("installed POPGym version differs from pin")
    if importlib.metadata.version("gymnasium") != env_cfg["gymnasium_version"]:
        raise AssertionError("installed Gymnasium version differs from pin")
    if importlib.metadata.version("numpy") != env_cfg["numpy_version"]:
        raise AssertionError("installed NumPy version differs from pin")
    if _sha(CONFIG.read_bytes()) != lock["config_sha256"] or _sha(RUNNER.read_bytes()) != lock["runner_sha256"]:
        raise AssertionError("locked source changed before/during run")
    source_hashes = summary["source_hashes"]
    if source_hashes != {
        "config": _sha(CONFIG.read_bytes()),
        "runner": _sha(RUNNER.read_bytes()),
        "decision_module": _sha(DECISION_MODULE.read_bytes()),
    }:
        raise AssertionError("saved code/config hashes do not match current source")

    # Audit the decision module's information boundary syntactically: it must
    # not access simulator objects or privileged state APIs.
    source = DECISION_MODULE.read_text(encoding="utf-8")
    for forbidden in ("get_state(", "hidden_grid", "mine_locations", "simulator_state"):
        if forbidden in source:
            raise AssertionError(f"decision module contains forbidden privileged field/API: {forbidden}")

    import gymnasium
    from popgym.envs.minesweeper import MineSweeperEasy

    started = time.perf_counter()
    shape = tuple(env_cfg["board_shape"])
    mine_count = int(env_cfg["mine_count"])
    universe = _layouts(shape, mine_count)
    if len(universe) != 120:
        raise AssertionError("unexpected hidden-layout prior cardinality")
    policy_rows = summary["episode_results"]
    methods = config["baselines"]["methods"]
    if set(policy_rows) != set(methods):
        raise AssertionError("saved policy set differs from frozen config")

    replayed_episodes = replayed_actions = public_risk_checks = 0
    summaries = {}
    for method in methods:
        rows = policy_rows[method]
        if [row["seed"] for row in rows] != seeds:
            raise AssertionError(f"seed list mismatch for {method}")
        wins, returns, safe_counts, action_counts, brier_terms = [], [], [], [], []
        for row in rows:
            env = MineSweeperEasy()
            env.reset(seed=row["seed"])
            board = np.full(shape, -1, dtype=np.int8)
            digest = hashlib.sha256(
                f"{config['baselines']['random_seed_salt']}:{row['seed']}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            total_return = 0.0
            safe_reveals = 0
            action_log = row["action_log"]
            terminated = truncated = False
            if len(action_log) != row["actions"]:
                raise AssertionError("action count differs from trace length")
            for action_record in action_log:
                cell_id = int(action_record["cell_id"])
                cell = divmod(cell_id, shape[1])
                if board[cell] != -1:
                    raise AssertionError("trace selected a cell outside current candidate set")
                expected_cell = _reference_choice(
                    method, board, universe, rng
                )
                if cell != expected_cell:
                    raise AssertionError(f"saved choice differs from independent policy: {method}, seed={row['seed']}")
                if method.startswith("exact_posterior"):
                    possible = _posterior(board, universe)
                    predicted_risk = sum(cell in layout for layout in possible) / len(possible)
                    if not math.isclose(predicted_risk, action_record["predicted_mine_risk"], abs_tol=1e-12):
                        raise AssertionError("logged mine risk differs from independent posterior")
                    public_risk_checks += 1
                observation, reward, terminated, truncated, _ = env.step(np.asarray(cell, dtype=np.int64))
                total_return += float(reward)
                mine_hit = bool(terminated and float(reward) < -0.25)
                if not math.isclose(float(reward), action_record["reward"], abs_tol=1e-12):
                    raise AssertionError("reward trace mismatch")
                if mine_hit != action_record["mine_hit"]:
                    raise AssertionError("mine-outcome trace mismatch")
                if action_record["predicted_mine_risk"] is not None:
                    brier_terms.append((float(action_record["predicted_mine_risk"]) - float(mine_hit)) ** 2)
                replayed_actions += 1
                if mine_hit:
                    break
                safe_reveals += 1
                board[cell] = int(observation)
            if not (terminated or truncated):
                raise AssertionError("saved episode trace did not reach a terminal/truncated state")
            won = safe_reveals == shape[0] * shape[1] - mine_count
            if won != row["won"] or safe_reveals != row["safe_reveals"] or not math.isclose(total_return, row["return"], abs_tol=1e-12):
                raise AssertionError("saved episode aggregates do not match replay")
            wins.append(int(won))
            returns.append(total_return)
            safe_counts.append(safe_reveals)
            action_counts.append(len(action_log))
            replayed_episodes += 1
        computed = {
            "episodes": len(rows),
            "wins": sum(wins),
            "win_rate": sum(wins) / len(wins),
            "mean_return": sum(returns) / len(returns),
            "mean_safe_reveals": sum(safe_counts) / len(safe_counts),
            "mean_actions": sum(action_counts) / len(action_counts),
            "brier_mine_risk": (sum(brier_terms) / len(brier_terms)) if brier_terms else None,
        }
        for metric, value in computed.items():
            saved = summary["summary_metrics"][method][metric]
            if isinstance(value, float):
                if not math.isclose(value, saved, rel_tol=0.0, abs_tol=1e-12):
                    raise AssertionError(f"metric mismatch {method}/{metric}: {value} != {saved}")
            elif value != saved:
                raise AssertionError(f"metric mismatch {method}/{metric}: {value} != {saved}")
        summaries[method] = computed

    primary = "exact_posterior_min_risk_then_information_gain"
    base = "exact_posterior_min_risk"
    gain = _paired_interval(
        policy_rows[primary], policy_rows[base],
        config["statistics"]["paired_bootstrap_replicates"],
        config["statistics"]["paired_bootstrap_seed"],
    )
    saved_gain = summary["primary_posterior_info_vs_posterior_risk_mean_return"]
    if not all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(gain, [saved_gain["point"], *saved_gain["percentile_95"]])):
        raise AssertionError("paired interval does not recompute")

    audit = {
        "audit": "phase5_minesweeper_candidate_screen_independent_reaudit",
        "summary_sha256": _sha((RUN / "summary.json").read_bytes()),
        "selection_lock_sha256": _sha((RUN / "selection_lock.json").read_bytes()),
        "source_and_dependency_pins_match": True,
        "seed_lock_matches": True,
        "decision_module_forbidden_state_scan_passed": True,
        "independent_action_policy_checks": len(methods) * len(seeds),
        "replayed_episode_count": replayed_episodes,
        "replayed_action_count": replayed_actions,
        "independent_posterior_risk_checks": public_risk_checks,
        "paired_return_interval_recomputed": gain,
        "hidden_layouts_saved_or_used_by_policy": False,
        "validation_or_test_accessed": False,
        "teacher_labels_generated": False,
        "student_training_run": False,
        "calibration_fitted": False,
        "audit_elapsed_seconds": time.perf_counter() - started,
        "audit_status": "PASS",
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    run()
