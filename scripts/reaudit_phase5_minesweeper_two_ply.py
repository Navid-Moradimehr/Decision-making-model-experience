"""Independent public-information replay/audit for D049."""

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
RUN = ROOT / "artifacts" / "runs" / "phase5_minesweeper_two_ply_v1"
CONFIG = ROOT / "configs" / "phase5_minesweeper_two_ply.yaml"
RUNNER = ROOT / "scripts" / "run_phase5_minesweeper_two_ply.py"
DECISION = ROOT / "src" / "consequence_lab" / "minesweeper.py"
PLANNER = ROOT / "src" / "consequence_lab" / "minesweeper_planner.py"
AUDIT = RUN / "independent_audit_v1.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _neighbors(cell, shape):
    row, col = cell
    return tuple(
        (r, c)
        for r in range(max(0, row - 1), min(shape[0], row + 2))
        for c in range(max(0, col - 1), min(shape[1], col + 2))
        if (r, c) != cell
    )


def _clue(cell, layout, shape):
    mines = set(layout)
    return sum(point in mines for point in _neighbors(cell, shape))


def _posterior(board, mine_count):
    shape = board.shape
    cells = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
    clues = [((int(r), int(c)), int(board[r, c])) for r, c in np.argwhere(board >= 0)]
    out = []
    for layout in combinations(cells, mine_count):
        mines = set(layout)
        if any(cell in mines for cell, _ in clues):
            continue
        if all(_clue(cell, layout, shape) == value for cell, value in clues):
            out.append(layout)
    if not out:
        raise AssertionError("public observations imply empty posterior")
    return out


def _risk_choice(board, posterior):
    candidates = [(int(r), int(c)) for r, c in np.argwhere(board == -1)]
    risks = {c: sum(c in layout for layout in posterior) / len(posterior) for c in candidates}
    return min(candidates, key=lambda c: (risks[c], c)) if candidates else None


def _info_choice(board, posterior):
    candidates = [(int(r), int(c)) for r, c in np.argwhere(board == -1)]
    if not candidates:
        return None
    risks = {c: sum(c in layout for layout in posterior) / len(posterior) for c in candidates}
    gains = {}
    for cell in candidates:
        safe = [layout for layout in posterior if cell not in layout]
        counts = {}
        for layout in safe:
            clue = _clue(cell, layout, board.shape)
            counts[clue] = counts.get(clue, 0) + 1
        gains[cell] = (
            math.log2(len(safe))
            - sum((n / len(safe)) * math.log2(n) for n in counts.values())
            if safe else 0.0
        )
    return min(candidates, key=lambda c: (risks[c], -gains[c], c))


def _one_action_value(board, posterior, mine_count, success, failure):
    candidates = [(int(r), int(c)) for r, c in np.argwhere(board == -1)]
    if not candidates:
        return 0.0
    n = len(posterior)
    return max(
        (1 - sum(cell in layout for layout in posterior) / n) * success
        + (sum(cell in layout for layout in posterior) / n) * failure
        for cell in candidates
    )


def _two_action_choice(board, posterior, mine_count, success, failure):
    shape = board.shape
    candidates = [(int(r), int(c)) for r, c in np.argwhere(board == -1)]
    if not candidates:
        return None
    n = len(posterior)
    values = {}
    for first in candidates:
        mines = sum(first in layout for layout in posterior)
        value = mines / n * failure
        branches = {}
        for layout in posterior:
            if first not in layout:
                k = _clue(first, layout, shape)
                branches.setdefault(k, []).append(layout)
        for clue, branch in branches.items():
            probability = len(branch) / n
            child = board.copy()
            child[first] = clue
            remaining = int(np.count_nonzero(child == -1))
            # With a fixed mine count, if covered cells equal mines, the
            # environment's remaining horizon is already exhausted by safe
            # reveals; it has no safe second candidate.
            if remaining <= mine_count:
                value += probability * success
            else:
                continuation = _one_action_value(
                    child, branch, mine_count, success, failure
                )
                value += probability * (success + continuation)
        values[first] = value
    return min(values, key=lambda c: (-values[c], c))


def _replay(env_type, method, seed, config):
    shape = tuple(config["environment"]["board_shape"])
    mines = int(config["environment"]["mine_count"])
    board = np.full(shape, -1, dtype=np.int8)
    env = env_type()
    env.reset(seed=seed)
    success = float(env.success_reward_scale)
    failure = float(env.fail_reward_scale)
    if not math.isclose(success, config["environment"]["reward"]["success"], abs_tol=1e-12):
        raise AssertionError("environment success reward mismatch")
    if not math.isclose(failure, config["environment"]["reward"]["failure"], abs_tol=1e-12):
        raise AssertionError("environment failure reward mismatch")
    aggregate = {"seed": seed, "won": False, "return": 0.0, "safe_reveals": 0, "actions": 0}
    terminated = truncated = False
    while not (terminated or truncated):
        posterior = _posterior(board, mines)
        if method == "exact_posterior_min_risk":
            cell = _risk_choice(board, posterior)
        elif method == "exact_posterior_min_risk_then_information_gain":
            cell = _info_choice(board, posterior)
        elif method == "exact_two_ply_belief_lookahead":
            cell = _two_action_choice(board, posterior, mines, success, failure)
        else:
            raise AssertionError(f"unknown method {method}")
        if cell is None or board[cell] != -1:
            raise AssertionError("reference policy returned invalid candidate")
        clue, reward, terminated, truncated, _ = env.step(np.asarray(cell, dtype=np.int64))
        aggregate["actions"] += 1
        aggregate["return"] += float(reward)
        if terminated and math.isclose(float(reward), failure, abs_tol=1e-12):
            continue
        aggregate["safe_reveals"] += 1
        board[cell] = int(clue)
    aggregate["won"] = aggregate["safe_reveals"] == int(np.prod(shape)) - mines
    return aggregate


def _bootstrap(left, right, field, replicates, seed):
    if [x["seed"] for x in left] != [x["seed"] for x in right]:
        raise AssertionError("paired seeds differ")
    deltas = [float(a[field]) - float(b[field]) for a, b in zip(left, right)]
    rng = random.Random(seed)
    n = len(deltas)
    samples = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(replicates)
    )
    return {
        "point": sum(deltas) / n,
        "percentile_95": [
            samples[int(0.025 * (replicates - 1))],
            samples[int(0.975 * (replicates - 1))],
        ],
    }


def run():
    if AUDIT.exists():
        raise FileExistsError(f"refusing to overwrite {AUDIT}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((RUN / "selection_lock.json").read_text(encoding="utf-8"))
    env_cfg = config["environment"]
    seeds = list(range(env_cfg["seed_start"], env_cfg["seed_start"] + env_cfg["seed_count"]))
    if lock["seeds"] != seeds or lock["environment_reset_or_episode_run_at_lock"] is not False:
        raise AssertionError("D049 frozen lock mismatch")
    if lock["student_or_teacher_labels_generated_at_lock"] is not False:
        raise AssertionError("selection lock was written after unauthorized operation")
    expected_hashes = {
        "config": _sha(CONFIG.read_bytes()),
        "runner": _sha(RUNNER.read_bytes()),
        "decision_module": _sha(DECISION.read_bytes()),
        "planner_module": _sha(PLANNER.read_bytes()),
    }
    if lock["source_hashes"] != expected_hashes or summary["source_hashes"] != expected_hashes:
        raise AssertionError("source/config hashes differ from frozen source")
    for distribution, key in (("popgym", "version"), ("gymnasium", "gymnasium_version"), ("numpy", "numpy_version")):
        if importlib.metadata.version(distribution) != env_cfg[key]:
            raise AssertionError(f"{distribution} does not match pin")
    for forbidden in ("get_state(", "hidden_grid", "mine_locations", "simulator_state"):
        if forbidden in DECISION.read_text(encoding="utf-8") or forbidden in PLANNER.read_text(encoding="utf-8"):
            raise AssertionError(f"decision implementation contains forbidden API/field: {forbidden}")
    if summary["validation_or_test_accessed"] or summary["teacher_labels_generated"] or summary["student_training_run"] or summary["calibration_fitted"] or summary["hidden_board_saved"] or summary["action_traces_saved"]:
        raise AssertionError("run exceeded frozen D049 scope")

    from popgym.envs.minesweeper import MineSweeperEasy

    methods = config["baselines"]["methods"]
    saved = summary["episode_aggregates"]
    if set(saved) != set(methods):
        raise AssertionError("method set differs from config")
    replay_count = 0
    actual = {}
    started = time.perf_counter()
    for method in methods:
        rows = saved[method]
        if [row["seed"] for row in rows] != seeds:
            raise AssertionError(f"seed range mismatch: {method}")
        replayed = []
        for row in rows:
            ref = _replay(MineSweeperEasy, method, int(row["seed"]), config)
            for key in ("won", "safe_reveals", "actions"):
                if ref[key] != row[key]:
                    raise AssertionError(f"replay mismatch {method}/{row['seed']}/{key}")
            if not math.isclose(ref["return"], row["return"], abs_tol=1e-12):
                raise AssertionError(f"return mismatch {method}/{row['seed']}")
            if row["decision_cpu_seconds"] < 0:
                raise AssertionError("negative decision CPU measurement")
            replayed.append(ref)
            replay_count += 1
        actual[method] = {
            "episodes": len(replayed),
            "wins": sum(int(x["won"]) for x in replayed),
            "win_rate": sum(int(x["won"]) for x in replayed) / len(replayed),
            "mean_return": sum(x["return"] for x in replayed) / len(replayed),
            "mean_safe_reveals": sum(x["safe_reveals"] for x in replayed) / len(replayed),
            "mean_actions": sum(x["actions"] for x in replayed) / len(replayed),
            "mean_decision_cpu_seconds": summary["summary_metrics"][method]["mean_decision_cpu_seconds"],
        }
        for metric, value in actual[method].items():
            if not math.isclose(value, summary["summary_metrics"][method][metric], abs_tol=1e-12):
                raise AssertionError(f"aggregate mismatch {method}/{metric}")

    primary, baseline = "exact_two_ply_belief_lookahead", "exact_posterior_min_risk"
    win_gain = _bootstrap(saved[primary], saved[baseline], "won", config["statistics"]["paired_bootstrap_replicates"], config["statistics"]["paired_bootstrap_seed"])
    return_gain = _bootstrap(saved[primary], saved[baseline], "return", config["statistics"]["paired_bootstrap_replicates"], config["statistics"]["paired_bootstrap_seed"] + 1)
    for saved_name, computed in (
        ("two_ply_minus_min_risk_win_rate", win_gain),
        ("two_ply_minus_min_risk_mean_return", return_gain),
    ):
        recorded = summary[saved_name]
        if not math.isclose(computed["point"], recorded["point"], abs_tol=1e-12):
            raise AssertionError(f"paired comparison point mismatch: {saved_name}")
        if any(
            not math.isclose(a, b, abs_tol=1e-12)
            for a, b in zip(computed["percentile_95"], recorded["percentile_95"])
        ):
            raise AssertionError(f"paired comparison interval mismatch: {saved_name}")
    cpu_ratio = actual[primary]["mean_decision_cpu_seconds"] / actual[baseline]["mean_decision_cpu_seconds"]
    if not math.isclose(cpu_ratio, summary["two_ply_to_min_risk_decision_cpu_ratio"], rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("decision CPU ratio mismatch")
    gate_cfg = config["gate"]
    expected_gate = {
        "win_rate_gain_minimum_met": win_gain["point"] >= gate_cfg["candidate_advance_requires_win_rate_gain_minimum"],
        "paired_interval_lower_bound_positive": win_gain["percentile_95"][0] > gate_cfg["paired_bootstrap_lower_bound_strictly_greater_than"],
        "decision_cpu_multiple_minimum_met": cpu_ratio >= gate_cfg["planner_cpu_multiple_over_min_risk_minimum"],
    }
    expected_gate["advance"] = all(expected_gate.values())
    if expected_gate != summary["gate"]:
        raise AssertionError("frozen advance gate mismatch")

    elapsed = time.perf_counter() - started
    audit_result = {
        "audit": "phase5_minesweeper_two_ply_independent_public_information_replay",
        "status": "PASS",
        "episodes_replayed": replay_count,
        "seeds_per_policy": len(seeds),
        "methods": methods,
        "two_ply_policy_reimplemented_independently": True,
        "decision_modules_scanned_for_privileged_fields": True,
        "hidden_boards_or_action_traces_saved": False,
        "teacher_labels_or_student_training": False,
        "validation_or_test_accessed": False,
        "win_rate_gain": win_gain,
        "gate": expected_gate,
        "audit_wall_seconds": elapsed,
    }
    AUDIT.write_text(json.dumps(audit_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit_result, indent=2))
    return audit_result


if __name__ == "__main__":
    run()
