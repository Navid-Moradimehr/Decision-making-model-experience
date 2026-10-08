"""D046 fresh-block screen of exact versus bounded Sokoban planners."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase4b_boxoban_fresh_validation.yaml"
PINS = ROOT / "experiments" / "source_pins.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _extract_rows(text: str, wanted: int) -> list[str]:
    lines = text.splitlines()
    cursor = 0
    while cursor < len(lines):
        if lines[cursor] == "":
            cursor += 1
            continue
        if not lines[cursor].startswith(";"):
            raise ValueError(f"unexpected shard text at line {cursor + 1}")
        index = int(lines[cursor][1:].strip())
        cursor += 1
        while cursor < len(lines) and lines[cursor] == "":
            cursor += 1
        rows = lines[cursor : cursor + 10]
        if len(rows) != 10:
            raise ValueError(f"truncated source level {index}")
        if index == wanted:
            return rows
        cursor += 10
    raise KeyError(wanted)


def _quantile_nearest_rank(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _preflight() -> tuple[dict, dict, dict]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    project, dataset = config["project"], config["dataset"]
    if project["authorization"] != "D046_fresh_validation_screen_only" or not project["validation_access_authorized"]:
        raise RuntimeError("D046 fresh validation authorization is required")
    if any(project[key] for key in ("student_training_authorized", "teacher_label_generation_authorized", "calibration_access_authorized")):
        raise RuntimeError("D046 must not authorize label generation, training, or calibration")
    pin = pins["sources"]["boxoban_levels"]
    if dataset["revision"] != pin["revision"] or dataset["split"] != "medium/valid":
        raise RuntimeError("source revision/split mismatch")
    previous_path = ROOT / dataset["exclude_level_ids_from_run"]
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous.get("selected_level_count") != 60 or previous.get("split") != "medium/valid":
        raise RuntimeError("previous qualification block manifest is invalid")
    audit_path = previous_path.parent / "independent_audit_v1.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("audit_status") != "PASS" or audit.get("official_test_accessed") is not False:
        raise RuntimeError("previous block independent audit missing/failed")
    pilot_path = ROOT / "artifacts" / "runs" / "phase4b_boxoban_planner_pilot_v1" / "summary.json"
    pilot_audit_path = pilot_path.parent / "independent_audit_v1.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    pilot_audit = json.loads(pilot_audit_path.read_text(encoding="utf-8"))
    if pilot_audit.get("audit_status") != "PASS" or pilot.get("validation_or_test_content_accessed") is not False:
        raise RuntimeError("D045 training-only pilot was not independently audited")
    return config, previous, pilot


def run() -> dict:
    config, previous, pilot = _preflight()
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
        sample_level_ids,
    )

    dataset = config["dataset"]
    old_ids = previous["selected_level_ids"]
    prefix = dataset["sample_hash_input"].split("{canonical_level_id}", 1)[0]
    selected_ids = sample_level_ids(
        split=dataset["split"],
        shard_count=dataset["level_count"] // dataset["levels_per_shard"],
        levels_per_shard=dataset["levels_per_shard"],
        count=dataset["sample_count"],
        prefix=prefix,
        exclude_level_ids=old_ids,
    )
    if set(selected_ids) & set(old_ids):
        raise AssertionError("fresh validation IDs overlap D044 block")

    output = (ROOT / config["artifacts"]["run_directory"]).resolve()
    runs_root = (ROOT / "artifacts" / "runs").resolve()
    output.relative_to(runs_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    selection_lock = {
        "protocol": config["project"]["protocol"],
        "authorization": config["project"]["authorization"],
        "source_revision": dataset["revision"],
        "split": dataset["split"],
        "selected_level_ids": selected_ids,
        "excluded_previous_level_ids": old_ids,
        "config_sha256": _sha(CONFIG.read_bytes()),
        "runner_sha256": _sha(Path(__file__).read_bytes()),
        "content_fetched_at_lock": False,
    }
    (output / "selection_lock.json").write_text(
        json.dumps(selection_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    shard_paths = sorted({level_id.split("#", 1)[0] for level_id in selected_ids})
    shard_bytes, shard_hashes = {}, {}
    for shard_path in shard_paths:
        url = f"https://raw.githubusercontent.com/google-deepmind/boxoban-levels/{dataset['revision']}/{shard_path}"
        req = Request(url, headers={"User-Agent": "consequence-lab-fresh-validation/1"})
        with urlopen(req, timeout=45) as response:
            payload = response.read()
        if not payload or len(payload) > 2_000_000:
            raise ValueError(f"unexpected source size for {shard_path}")
        shard_hashes[shard_path] = _sha(payload)
        shard_bytes[shard_path] = payload
    downloaded_bytes = sum(map(len, shard_bytes.values()))
    if downloaded_bytes > config["project"]["max_generated_artifact_bytes"]:
        raise RuntimeError("source download exceeds artifact cap")

    levels, level_text = {}, {}
    for level_id in selected_ids:
        shard_path, raw_index = level_id.rsplit("#", 1)
        index = int(raw_index)
        rows = _extract_rows(shard_bytes[shard_path].decode("ascii"), index)
        level = parse_boxoban_rows(rows, index)
        if (level.board.width, level.board.height) != (10, 10):
            raise ValueError(f"unexpected board size: {level_id}")
        if len(level.initial_state.boxes) != 4 or len(level.board.goals) != 4:
            raise ValueError(f"unexpected box/goal count: {level_id}")
        levels[level_id] = level
        level_text[level_id] = f"; {index}\n" + "\n".join(rows)

    cpu_start = time.process_time()
    level_results = []
    action_limit = config["task"]["action_limit"]

    def run_solver(level, mode, max_expansions, timeout):
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        result = solve_optimal(
            level.board,
            level.initial_state,
            heuristic_mode=mode,
            max_expanded_states=max_expansions,
            timeout_seconds=timeout,
        )
        replay = level.initial_state
        for action in result.actions if result.solved else ():
            replay = transition(level.board, replay, action)
        replay_valid = bool(result.solved and is_solved(level.board, replay))
        within_limit = bool(replay_valid and len(result.actions) <= action_limit)
        return {
            "status": result.status,
            "proof_complete": result.proof_complete,
            "solved_within_action_limit": within_limit,
            "optimal_actions_if_solved": result.primitive_action_cost,
            "expanded_states": result.expanded_states,
            "generated_states": result.generated_states,
            "replay_valid": replay_valid,
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "solution_actions": list(result.actions),
        }

    for level_id in selected_ids:
        level = levels[level_id]
        manhattan = run_solver(
            level, "manhattan_assignment",
            config["solver"]["exact_max_expansions_per_level"],
            config["solver"]["exact_timeout_seconds_per_level"],
        )
        static_exact = run_solver(
            level, "static_push_assignment",
            config["solver"]["exact_max_expansions_per_level"],
            config["solver"]["exact_timeout_seconds_per_level"],
        )
        if manhattan["status"] == static_exact["status"] == "solved":
            if manhattan["optimal_actions_if_solved"] != static_exact["optimal_actions_if_solved"]:
                raise AssertionError(f"exact heuristic variants disagree on cost: {level_id}")
        bounded = {}
        for budget in config["solver"]["bounded_expansion_budgets"]:
            bounded[str(budget)] = run_solver(
                level, config["solver"]["bounded_heuristic"], budget,
                config["solver"]["bounded_timeout_seconds_per_budget"],
            )
        greedy_started = time.process_time()
        greedy = play_episode(
            level.board, level.initial_state, greedy_push_action, action_limit=action_limit
        )
        greedy_row = {**greedy.__dict__, "cpu_seconds": time.process_time() - greedy_started}
        level_results.append(
            {
                "level_id": level_id,
                "exact_manhattan": manhattan,
                "exact_static_push": static_exact,
                "bounded_static_push": bounded,
                "greedy_push": greedy_row,
            }
        )

    total_cpu = time.process_time() - cpu_start
    if total_cpu > config["project"]["max_total_cpu_seconds"]:
        raise RuntimeError(f"validation run exceeded CPU cap: {total_cpu:.2f}s")
    exact_success = [int(row["exact_static_push"]["solved_within_action_limit"]) for row in level_results]
    budget_key = str(config["solver"]["bounded_expansion_budgets"][-1])
    budget_success = [
        int(row["bounded_static_push"][budget_key]["solved_within_action_limit"])
        for row in level_results
    ]
    diff, ci_low, ci_high = paired_bootstrap_mean_interval(
        [left - right for left, right in zip(exact_success, budget_success)],
        replicates=config["statistics"]["paired_bootstrap_replicates"],
        seed=config["statistics"]["paired_bootstrap_seed"],
    )
    exact_proofs = sum(row["exact_static_push"]["proof_complete"] for row in level_results)
    unresolved = len(level_results) - exact_proofs
    gate = config["gate"]
    passed = (
        len(level_results) == dataset["sample_count"]
        and exact_proofs >= gate["exact_static_proofs_required"]
        and unresolved <= gate["unresolved_exact_levels_allowed"]
        and diff >= gate["exact_static_success_gain_over_budget100k_point_minimum"]
        and ci_low > gate["paired_bootstrap_ci95_lower_strictly_greater_than"]
    )

    def summarize_search(key: str, method_selector) -> dict:
        records = [method_selector(row) for row in level_results]
        successes = [record for record in records if record["solved_within_action_limit"]]
        return {
            "successes": len(successes),
            "success_rate": len(successes) / len(records),
            "mean_actions_when_solved": (
                sum(record["optimal_actions_if_solved"] for record in successes) / len(successes)
                if successes else None
            ),
            "proofs": sum(record["proof_complete"] for record in records),
            "unresolved": sum(not record["proof_complete"] for record in records),
            "search_cpu_p50_p95_seconds": [
                _quantile_nearest_rank([record["cpu_seconds"] for record in records], q)
                for q in config["statistics"]["latency_quantiles"]
            ],
            "mean_expansions": sum(record["expanded_states"] for record in records) / len(records),
        }

    metrics = {
        "exact_manhattan": summarize_search("exact_manhattan", lambda row: row["exact_manhattan"]),
        "exact_static_push": summarize_search("exact_static_push", lambda row: row["exact_static_push"]),
        "bounded_static_push": {
            key: summarize_search(key, lambda row, budget=key: row["bounded_static_push"][budget])
            for key in map(str, config["solver"]["bounded_expansion_budgets"])
        },
        "greedy_push": {
            "successes": sum(row["greedy_push"]["solved"] for row in level_results),
            "success_rate": sum(row["greedy_push"]["solved"] for row in level_results) / len(level_results),
            "deadlock_episodes": sum(row["greedy_push"]["deadlock_seen"] for row in level_results),
            "mean_cpu_seconds_per_episode": sum(row["greedy_push"]["cpu_seconds"] for row in level_results) / len(level_results),
        },
        "exact_static_minus_budget100k_success_gain": diff,
        "paired_bootstrap_95_interval": [ci_low, ci_high],
        "primary_exact_proofs": exact_proofs,
        "primary_unresolved": unresolved,
        "total_evaluation_cpu_seconds": total_cpu,
    }

    source_files = [
        CONFIG, PINS,
        ROOT / "configs" / "phase4b_boxoban_planner_pilot.yaml",
        ROOT / "artifacts" / "runs" / "phase4b_boxoban_planner_pilot_v1" / "summary.json",
        ROOT / "artifacts" / "runs" / "phase4b_boxoban_planner_pilot_v1" / "independent_audit_v1.json",
        ROOT / "artifacts" / "runs" / "phase4_boxoban_validation_screen_v1" / "summary.json",
        ROOT / "artifacts" / "runs" / "phase4_boxoban_validation_screen_v1" / "independent_audit_v1.json",
        ROOT / "scripts" / "run_phase4b_fresh_validation_screen.py",
        ROOT / "src" / "consequence_lab" / "boxoban.py",
        ROOT / "src" / "consequence_lab" / "boxoban_screen.py",
        ROOT / "tests" / "test_boxoban.py",
        ROOT / "tests" / "test_boxoban_screen.py",
        ROOT / "tests" / "test_config.py",
    ]
    summary = {
        "experiment": "phase4b_boxoban_fresh_budgeted_planner_screen",
        "protocol": config["project"]["protocol"],
        "authorization": config["project"]["authorization"],
        "source_revision": dataset["revision"],
        "split": dataset["split"],
        "excluded_previous_qualification_ids": old_ids,
        "selected_level_count": len(level_results),
        "selected_level_ids": selected_ids,
        "selected_level_text": level_text,
        "source_shard_sha256": shard_hashes,
        "downloaded_source_bytes": downloaded_bytes,
        "validation_content_accessed": True,
        "official_test_accessed": False,
        "teacher_labels_generated": False,
        "student_training_run": False,
        "per_level_results": level_results,
        "summary_metrics": metrics,
        "advance_gate": "PASS_LABEL_PILOT_MAY_BE_CONSIDERED" if passed else "FAIL_STOP_BEFORE_LABELS",
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_sha256": _sha(_git("status", "--porcelain").encode("utf-8")),
        "source_code_config_test_sha256": {
            str(path.relative_to(ROOT)): _sha(path.read_bytes()) for path in source_files
        },
    }
    if (output / "summary.json").exists():
        raise FileExistsError("refusing to overwrite existing Stage-4B summary")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"selected_level_text", "per_level_results", "excluded_previous_qualification_ids"}}, indent=2))
    return summary


if __name__ == "__main__":
    run()
