"""Independent provenance and result audit for the D046 fresh screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "runs" / "phase4b_boxoban_fresh_validation_v1"
CONFIG = ROOT / "configs" / "phase4b_boxoban_fresh_validation.yaml"
AUDIT = RUN / "independent_audit_v1.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_rows(shard_text: str, wanted: int) -> list[str]:
    lines = shard_text.splitlines()
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor]:
            cursor += 1
            continue
        if not lines[cursor].startswith(";"):
            raise ValueError(f"malformed source at line {cursor + 1}")
        index = int(lines[cursor][1:].strip())
        cursor += 1
        while cursor < len(lines) and not lines[cursor]:
            cursor += 1
        rows = lines[cursor : cursor + 10]
        if len(rows) != 10:
            raise ValueError(f"truncated source level {index}")
        if index == wanted:
            return rows
        cursor += 10
    raise KeyError(wanted)


def _sample_ids(config: dict, excluded: set[str]) -> list[str]:
    ds = config["dataset"]
    prefix = ds["sample_hash_input"].split("{canonical_level_id}", 1)[0]
    ranked = []
    for shard in range(ds["level_count"] // ds["levels_per_shard"]):
        for level_index in range(ds["levels_per_shard"]):
            canonical = f"{ds['split']}/{shard:03d}.txt#{level_index:04d}"
            if canonical in excluded:
                continue
            rank = hashlib.sha256(f"{prefix}{canonical}".encode("ascii")).hexdigest()
            ranked.append((rank, canonical))
    return [level_id for _, level_id in sorted(ranked)[: ds["sample_count"]]]


def _reference_transition(board, state, action):
    delta = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}
    dx, dy = delta[action]
    target = (state.player[0] + dx, state.player[1] + dy)
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
    return type(state)(target, tuple(sorted(boxes, key=lambda p: (p[1], p[0]))))


def run() -> dict:
    if AUDIT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {AUDIT}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((RUN / "selection_lock.json").read_text(encoding="utf-8"))
    previous = json.loads(
        (ROOT / config["dataset"]["exclude_level_ids_from_run"]).read_text(encoding="utf-8")
    )
    if summary.get("authorization") != "D046_fresh_validation_screen_only":
        raise AssertionError("unexpected authorization")
    if summary.get("teacher_labels_generated") is not False or summary.get("student_training_run") is not False:
        raise AssertionError("unauthorized label/training flag in summary")
    if summary.get("official_test_accessed") is not False:
        raise AssertionError("official test access is not explicitly false")
    if lock.get("content_fetched_at_lock") is not False:
        raise AssertionError("selection was not locked before content fetch")
    selected = _sample_ids(config, set(previous["selected_level_ids"]))
    if selected != summary["selected_level_ids"] or selected != lock["selected_level_ids"]:
        raise AssertionError("independent fresh-ID selection does not match locked run")
    if set(selected) & set(previous["selected_level_ids"]):
        raise AssertionError("fresh sample overlaps previous validation sample")

    shards: dict[str, str] = {}
    for relative, expected in sorted(summary["source_shard_sha256"].items()):
        url = f"https://raw.githubusercontent.com/google-deepmind/boxoban-levels/{config['dataset']['revision']}/{relative}"
        request = Request(url, headers={"User-Agent": "consequence-lab-independent-audit/1"})
        with urlopen(request, timeout=45) as response:
            payload = response.read()
        if _sha(payload) != expected:
            raise AssertionError(f"pinned source shard hash mismatch: {relative}")
        shards[relative] = payload.decode("ascii")

    from consequence_lab.boxoban import (
        greedy_push_action,
        is_solved,
        parse_boxoban_rows,
        solve_optimal,
        transition,
    )
    from consequence_lab.boxoban_screen import (
        paired_bootstrap_mean_interval,
        play_episode,
    )

    started = time.perf_counter()
    exact_recomputed = 0
    bounded_recomputed = 0
    replayed = 0
    reference_transition_checks = 0
    greedy_recomputed = 0
    exact_successes, budget_successes = [], []
    per_level = {row["level_id"]: row for row in summary["per_level_results"]}
    if len(per_level) != len(selected):
        raise AssertionError("per-level result cardinality mismatch")
    for level_id in selected:
        if level_id not in per_level:
            raise AssertionError(f"missing result: {level_id}")
        row = per_level[level_id]
        shard, index_text = level_id.rsplit("#", 1)
        index = int(index_text)
        rows = _extract_rows(shards[shard], index)
        expected_text = f"; {index}\n" + "\n".join(rows)
        if summary["selected_level_text"].get(level_id) != expected_text:
            raise AssertionError(f"stored text differs from pinned shard: {level_id}")
        level = parse_boxoban_rows(rows, index)
        methods = (
            ("exact_manhattan", "manhattan_assignment", config["solver"]["exact_max_expansions_per_level"], config["solver"]["exact_timeout_seconds_per_level"]),
            ("exact_static_push", "static_push_assignment", config["solver"]["exact_max_expansions_per_level"], config["solver"]["exact_timeout_seconds_per_level"]),
        )
        recomputed = {}
        for key, mode, limit, timeout in methods:
            result = solve_optimal(level.board, level.initial_state, heuristic_mode=mode,
                                   max_expanded_states=limit, timeout_seconds=timeout)
            saved = row[key]
            replay_valid = False
            if result.solved:
                state = reference = level.initial_state
                for action in result.actions:
                    actual_next = transition(level.board, state, action)
                    reference_next = _reference_transition(level.board, reference, action)
                    if actual_next != reference_next:
                        raise AssertionError(f"transition implementation mismatch: {level_id}")
                    state, reference = actual_next, reference_next
                    reference_transition_checks += 1
                replay_valid = is_solved(level.board, state) and is_solved(level.board, reference)
                if not replay_valid:
                    raise AssertionError(f"solution replay failed: {level_id}")
                replayed += 1
            actual_tuple = (result.status, result.proof_complete, result.primitive_action_cost,
                            result.expanded_states, result.generated_states,
                            replay_valid and len(result.actions) <= config["task"]["action_limit"],
                            list(result.actions) if result.solved else [])
            saved_tuple = (saved["status"], saved["proof_complete"], saved["optimal_actions_if_solved"],
                           saved["expanded_states"], saved["generated_states"],
                           saved["solved_within_action_limit"], saved["solution_actions"])
            if actual_tuple != saved_tuple:
                raise AssertionError(f"search result mismatch for {key}: {level_id}")
            recomputed[key] = result
            exact_recomputed += 1
        if all(recomputed[k].solved for k in ("exact_manhattan", "exact_static_push")) and (
            recomputed["exact_manhattan"].primitive_action_cost != recomputed["exact_static_push"].primitive_action_cost
        ):
            raise AssertionError(f"exact heuristics disagree on optimal action cost: {level_id}")

        final_budget = str(config["solver"]["bounded_expansion_budgets"][-1])
        for budget in config["solver"]["bounded_expansion_budgets"]:
            result = solve_optimal(level.board, level.initial_state,
                                   heuristic_mode=config["solver"]["bounded_heuristic"],
                                   max_expanded_states=budget,
                                   timeout_seconds=config["solver"]["bounded_timeout_seconds_per_budget"])
            saved = row["bounded_static_push"][str(budget)]
            replay_valid = False
            if result.solved:
                state = level.initial_state
                for action in result.actions:
                    state = transition(level.board, state, action)
                replay_valid = is_solved(level.board, state)
            expected = (result.status, result.proof_complete, result.primitive_action_cost,
                        result.expanded_states, result.generated_states,
                        replay_valid and len(result.actions) <= config["task"]["action_limit"],
                        list(result.actions) if result.solved else [])
            got = (saved["status"], saved["proof_complete"], saved["optimal_actions_if_solved"],
                   saved["expanded_states"], saved["generated_states"],
                   saved["solved_within_action_limit"], saved["solution_actions"])
            if expected != got:
                raise AssertionError(f"bounded {budget} result mismatch: {level_id}")
            bounded_recomputed += 1
        greedy = play_episode(level.board, level.initial_state, greedy_push_action,
                              action_limit=config["task"]["action_limit"])
        saved_greedy = row["greedy_push"]
        if greedy.__dict__ != {key: saved_greedy[key] for key in greedy.__dict__}:
            raise AssertionError(f"greedy baseline mismatch: {level_id}")
        greedy_recomputed += 1
        exact_successes.append(int(row["exact_static_push"]["solved_within_action_limit"]))
        budget_successes.append(int(row["bounded_static_push"][final_budget]["solved_within_action_limit"]))

    point, lower, upper = paired_bootstrap_mean_interval(
        [a - b for a, b in zip(exact_successes, budget_successes)],
        replicates=config["statistics"]["paired_bootstrap_replicates"],
        seed=config["statistics"]["paired_bootstrap_seed"],
    )
    metrics = summary["summary_metrics"]
    expected_metrics = [sum(exact_successes), sum(budget_successes), point, lower, upper]
    saved_metrics = [metrics["exact_static_push"]["successes"],
                     metrics["bounded_static_push"][final_budget]["successes"],
                     metrics["exact_static_minus_budget100k_success_gain"],
                     *metrics["paired_bootstrap_95_interval"]]
    if expected_metrics != saved_metrics:
        raise AssertionError(f"paired summary mismatch: {expected_metrics} != {saved_metrics}")
    gate = config["gate"]
    gate_pass = (
        sum(row["exact_static_push"]["proof_complete"] for row in summary["per_level_results"])
        >= gate["exact_static_proofs_required"]
        and metrics["primary_unresolved"] <= gate["unresolved_exact_levels_allowed"]
        and point >= gate["exact_static_success_gain_over_budget100k_point_minimum"]
        and lower > gate["paired_bootstrap_ci95_lower_strictly_greater_than"]
    )
    expected_gate = "PASS_LABEL_PILOT_MAY_BE_CONSIDERED" if gate_pass else "FAIL_STOP_BEFORE_LABELS"
    if summary["advance_gate"] != expected_gate:
        raise AssertionError("saved advance gate does not match frozen thresholds")

    audit = {
        "audit": "phase4b_boxoban_fresh_validation_independent_reaudit",
        "protocol": config["project"]["protocol"],
        "summary_sha256": _sha((RUN / "summary.json").read_bytes()),
        "selection_lock_sha256": _sha((RUN / "selection_lock.json").read_bytes()),
        "independent_ids_match": True,
        "no_overlap_with_prior_validation": True,
        "pinned_source_shards_rehashed": len(shards),
        "selected_source_text_match": len(selected),
        "exact_search_recomputations_match": exact_recomputed,
        "bounded_search_recomputations_match": bounded_recomputed,
        "solutions_replayed": replayed,
        "reference_transition_checks": reference_transition_checks,
        "greedy_recomputations_match": greedy_recomputed,
        "paired_gain_interval_recomputed": [point, lower, upper],
        "frozen_gate_recomputed": expected_gate,
        "validation_content_accessed": True,
        "official_test_accessed": False,
        "teacher_labels_generated": False,
        "student_training_run": False,
        "audit_elapsed_seconds": time.perf_counter() - started,
        "audit_status": "PASS",
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    run()
