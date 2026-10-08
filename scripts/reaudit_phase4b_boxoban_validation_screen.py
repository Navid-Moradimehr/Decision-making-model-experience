"""Independent provenance, exact-solution, and result-consistency audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "runs" / "phase4_boxoban_validation_screen_v1"
CONFIG = ROOT / "configs" / "phase4_boxoban_validation_screen.yaml"
AUDIT = RUN / "independent_audit_v1.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reference_transition(board, state, action):
    """Independent transcription of cardinal Sokoban state transitions."""
    delta = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}
    dx, dy = delta[action]
    x, y = state.player
    target = (x + dx, y + dy)
    boxes = set(state.boxes)
    if not board.in_bounds(target) or target in board.walls:
        return state
    if target not in boxes:
        return type(state)(target, state.boxes)
    beyond = (target[0] + dx, target[1] + dy)
    if not board.in_bounds(beyond) or beyond in board.walls or beyond in boxes:
        return state
    boxes.remove(target)
    boxes.add(beyond)
    return type(state)(target, tuple(sorted(boxes, key=lambda point: (point[1], point[0]))))


def _extract_rows(shard_text: str, wanted_index: int) -> list[str]:
    lines = shard_text.splitlines()
    cursor = 0
    while cursor < len(lines):
        header = lines[cursor]
        if not header.startswith(";"):
            if header == "":
                cursor += 1
                continue
            raise ValueError("malformed upstream shard separator")
        index = int(header[1:].strip())
        cursor += 1
        while cursor < len(lines) and not lines[cursor]:
            cursor += 1
        rows = lines[cursor : cursor + 10]
        if len(rows) != 10:
            raise ValueError(f"truncated upstream puzzle {index}")
        if index == wanted_index:
            return rows
        cursor += 10
    raise KeyError(wanted_index)


def _sample_ids_independent(config: dict) -> list[str]:
    ds = config["dataset"]
    candidates = []
    for shard in range(ds["level_count"] // ds["levels_per_shard"]):
        for level in range(ds["levels_per_shard"]):
            canonical = f"{ds['split']}/{shard:03d}.txt#{level:04d}"
            digest = hashlib.sha256(
                ds["sampler_hash_input"].replace("{canonical_level_id}", canonical).encode("ascii")
            ).hexdigest()
            candidates.append((digest, canonical))
    return [name for _, name in sorted(candidates)[: ds["selection_count"]]]


def run() -> dict:
    if AUDIT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {AUDIT}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    if summary.get("authorization") != "D044_stage1_validation_screen_only":
        raise ValueError("run is not the D044-authorized screen")
    ids = _sample_ids_independent(config)
    if ids != summary["selected_level_ids"]:
        raise AssertionError("saved selected ID list differs from independent ID sampler")

    expected_hashes = summary["source_shard_sha256"]
    shards = {}
    for shard_path, expected in sorted(expected_hashes.items()):
        url = (
            f"https://raw.githubusercontent.com/google-deepmind/boxoban-levels/"
            f"{config['dataset']['revision']}/{shard_path}"
        )
        req = Request(url, headers={"User-Agent": "consequence-lab-independent-audit/1"})
        with urlopen(req, timeout=45) as response:
            payload = response.read()
        actual = _sha(payload)
        if actual != expected:
            raise AssertionError(f"source hash mismatch for {shard_path}")
        shards[shard_path] = payload.decode("ascii")

    from consequence_lab.boxoban import (
        greedy_goal_action,
        greedy_push_action,
        is_solved,
        parse_boxoban_rows,
        solve_optimal,
        transition,
    )
    from consequence_lab.boxoban_screen import (
        deterministic_level_seed,
        paired_bootstrap_mean_interval,
        play_episode,
        random_action_policy,
    )

    checked_solutions = 0
    checked_source_levels = 0
    transition_comparisons = 0
    policy_results_match = 0
    started = time.perf_counter()
    exact_success, greedy_success = [], []
    recomputed_rows = []
    for row, level_id in zip(summary["per_level_results"], ids):
        shard_path, raw_index = level_id.rsplit("#", 1)
        index = int(raw_index)
        source_rows = _extract_rows(shards[shard_path], index)
        level_text = f"; {index}\n" + "\n".join(source_rows)
        if summary["selected_level_text"].get(level_id) != level_text:
            raise AssertionError(f"saved selected puzzle text differs from pinned source: {level_id}")
        checked_source_levels += 1
        level = parse_boxoban_rows(source_rows, index)

        result = solve_optimal(
            level.board,
            level.initial_state,
            max_expanded_states=config["solver"]["max_expanded_states_per_level"],
            timeout_seconds=config["solver"]["timeout_seconds_per_level"],
        )
        saved_exact = row["exact"]
        if (result.status, result.primitive_action_cost, result.proof_complete) != (
            saved_exact["status"],
            saved_exact["optimal_actions"],
            saved_exact["proof_complete"],
        ):
            raise AssertionError(f"exact-search result mismatch: {level_id}")
        if result.solved:
            state_a = state_b = level.initial_state
            for action in result.actions:
                next_a = transition(level.board, state_a, action)
                next_b = _reference_transition(level.board, state_b, action)
                if next_a != next_b:
                    raise AssertionError(f"transition parity mismatch along solution: {level_id}")
                state_a, state_b = next_a, next_b
                transition_comparisons += 1
            if not (is_solved(level.board, state_a) and is_solved(level.board, state_b)):
                raise AssertionError(f"recomputed solution replay failed: {level_id}")
            checked_solutions += 1
        exact_ok = bool(result.solved and len(result.actions) <= config["task"]["action_limit"])
        exact_success.append(int(exact_ok))

        seed = deterministic_level_seed(config["project"]["protocol"], level_id)
        actual_policies = {
            "random": play_episode(level.board, level.initial_state, random_action_policy(seed), action_limit=config["task"]["action_limit"]),
            "greedy_goal": play_episode(level.board, level.initial_state, greedy_goal_action, action_limit=config["task"]["action_limit"]),
            "greedy_push": play_episode(level.board, level.initial_state, greedy_push_action, action_limit=config["task"]["action_limit"]),
        }
        for policy_name, actual in actual_policies.items():
            if actual.__dict__ != row[policy_name]:
                raise AssertionError(f"policy result mismatch for {policy_name}: {level_id}")
            policy_results_match += 1
        greedy_success.append(int(row["greedy_push"]["solved"]))
        recomputed_rows.append({"level_id": level_id, "exact_status": result.status})

    point, lower, upper = paired_bootstrap_mean_interval(
        [a - b for a, b in zip(exact_success, greedy_success)],
        replicates=config["statistics"]["paired_bootstrap_replicates"],
        seed=config["statistics"]["paired_bootstrap_seed"],
    )
    if [sum(exact_success), sum(greedy_success), point, lower, upper] != [
        summary["summary_metrics"]["exact"]["successes"],
        summary["summary_metrics"]["greedy_push"]["successes"],
        summary["summary_metrics"]["exact_minus_greedy_push_success_gain"],
        summary["summary_metrics"]["paired_bootstrap_95_interval"][0],
        summary["summary_metrics"]["paired_bootstrap_95_interval"][1],
    ]:
        raise AssertionError("saved paired success summary does not recompute")

    audit = {
        "audit": "phase4b_boxoban_validation_independent_reaudit",
        "protocol": config["project"]["protocol"],
        "run_summary_sha256": _sha((RUN / "summary.json").read_bytes()),
        "independent_id_sampler_matches": True,
        "source_shards_rehashed": len(shards),
        "selected_source_levels_match": checked_source_levels,
        "exact_search_recomputations_match": len(recomputed_rows),
        "solution_replays_independently_matched": checked_solutions,
        "reference_transition_path_comparisons": transition_comparisons,
        "baseline_episode_recomputations_match": policy_results_match,
        "paired_gain_interval_recomputed": [point, lower, upper],
        "validation_levels_accessed": True,
        "official_test_accessed": False,
        "student_targets_generated": False,
        "student_training_run": False,
        "audit_wall_seconds": time.perf_counter() - started,
        "audit_status": "PASS",
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    run()
