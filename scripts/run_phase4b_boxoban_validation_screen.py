"""Run the separately authorized, fixed-sample Boxoban validation screen."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "phase4_boxoban_validation_screen.yaml"
PINS_PATH = ROOT / "experiments" / "source_pins.json"
STAGE0_RUN = ROOT / "artifacts" / "runs" / "phase4_boxoban_qualification_v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _extract_rows(shard_text: str, wanted_index: int) -> list[str]:
    lines = shard_text.splitlines()
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor]
        if not line.startswith(";"):
            if line == "":
                cursor += 1
                continue
            raise ValueError(f"expected Boxoban header at line {cursor + 1}")
        try:
            index = int(line[1:].strip())
        except ValueError as exc:
            raise ValueError(f"malformed Boxoban header at line {cursor + 1}") from exc
        cursor += 1
        while cursor < len(lines) and lines[cursor] == "":
            cursor += 1
        rows = lines[cursor : cursor + 10]
        if len(rows) != 10 or any(row.startswith(";") for row in rows):
            raise ValueError(f"truncated level {index}")
        if index == wanted_index:
            return rows
        cursor += 10
    raise KeyError(f"level {wanted_index} not present in source shard")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _read_config() -> tuple[dict, dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    project = config["project"]
    if project["authorization"] != "D044_stage1_validation_screen_only":
        raise RuntimeError("Stage 1 requires the separately recorded D044 authorization")
    if not project["validation_screen_authorized"]:
        raise RuntimeError("validation screen is not authorized in its frozen config")
    if project["student_training_authorized"] or project["teacher_dataset_generation_authorized"]:
        raise RuntimeError("Stage 1 may not authorize student or teacher-data generation")
    dataset = config["dataset"]
    pin = pins["sources"]["boxoban_levels"]
    if dataset["revision"] != pin["revision"] or dataset["revision"] != "78011d8c86ad423bfb2e1005012393b4c3c24409":
        raise RuntimeError("Stage 1 source revision is not the independently audited pin")
    if dataset["split"] != "medium/valid" or dataset["official_test_access"] != "forbidden":
        raise RuntimeError("Stage 1 must use only medium validation and forbid official test")
    audit_path = STAGE0_RUN / "saved_data_audit_transition_v2.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("stage0_gate") != "PASS" or audit.get("validation_or_test_accessed") is not False:
        raise RuntimeError("independent Stage-0 completion audit is missing or failed")
    return config, pins


def run() -> dict:
    config, pins = _read_config()
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
        sample_level_ids,
    )

    dataset = config["dataset"]
    prefix = dataset["sampler_hash_input"].split("{canonical_level_id}", 1)[0]
    selected_ids = sample_level_ids(
        split=dataset["split"],
        shard_count=dataset["level_count"] // dataset["levels_per_shard"],
        levels_per_shard=dataset["levels_per_shard"],
        count=dataset["selection_count"],
        prefix=prefix,
    )
    output = ROOT / config["artifacts"]["run_directory"]
    runs_root = (ROOT / "artifacts" / "runs").resolve()
    output = output.resolve()
    try:
        output.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("configured run directory must be below artifacts/runs") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")

    shard_paths = sorted({level_id.split("#", 1)[0] for level_id in selected_ids})
    shard_bytes: dict[str, bytes] = {}
    source_hashes: dict[str, str] = {}
    for shard_path in shard_paths:
        url = f"https://raw.githubusercontent.com/google-deepmind/boxoban-levels/{dataset['revision']}/{shard_path}"
        request = Request(url, headers={"User-Agent": "consequence-lab-boxoban-screen/1"})
        with urlopen(request, timeout=45) as response:
            payload = response.read()
        if not payload or len(payload) > 2_000_000:
            raise ValueError(f"unexpected shard size for {shard_path}: {len(payload)}")
        # Hash the immutable-source payload before decoding or parsing it.
        source_hashes[shard_path] = _sha256(payload)
        shard_bytes[shard_path] = payload
    downloaded_bytes = sum(map(len, shard_bytes.values()))
    if downloaded_bytes > config["project"]["max_generated_artifact_bytes"]:
        raise RuntimeError("downloaded source shards exceed frozen artifact cap")

    levels = {}
    selected_level_text = {}
    for level_id in selected_ids:
        shard_path, raw_index = level_id.rsplit("#", 1)
        index = int(raw_index)
        rows = _extract_rows(shard_bytes[shard_path].decode("ascii"), index)
        parsed = parse_boxoban_rows(rows, index)
        if (parsed.board.width, parsed.board.height) != (10, 10):
            raise ValueError(f"unexpected dimensions for {level_id}")
        if len(parsed.initial_state.boxes) != 4 or len(parsed.board.goals) != 4:
            raise ValueError(f"unexpected box/goal count for {level_id}")
        levels[level_id] = parsed
        selected_level_text[level_id] = f"; {index}\n" + "\n".join(rows)

    rows = []
    total_cpu_start = time.process_time()
    solver_cfg = config["solver"]
    action_limit = config["task"]["action_limit"]
    for level_id in selected_ids:
        level = levels[level_id]
        search_cpu_start = time.process_time()
        exact = solve_optimal(
            level.board,
            level.initial_state,
            max_expanded_states=solver_cfg["max_expanded_states_per_level"],
            timeout_seconds=solver_cfg["timeout_seconds_per_level"],
        )
        exact_cpu = time.process_time() - search_cpu_start
        exact_state = level.initial_state
        for action in exact.actions if exact.solved else ():
            exact_state = transition(level.board, exact_state, action)
        exact_replay_valid = bool(exact.solved and is_solved(level.board, exact_state))
        exact_solved_within_limit = bool(
            exact_replay_valid and len(exact.actions) <= action_limit
        )

        seed = deterministic_level_seed(config["project"]["protocol"], level_id)
        random_result = play_episode(
            level.board, level.initial_state, random_action_policy(seed), action_limit=action_limit
        )
        goal_result = play_episode(
            level.board, level.initial_state, greedy_goal_action, action_limit=action_limit
        )
        greedy_result = play_episode(
            level.board, level.initial_state, greedy_push_action, action_limit=action_limit
        )
        rows.append(
            {
                "level_id": level_id,
                "exact": {
                    "status": exact.status,
                    "proof_complete": exact.proof_complete,
                    "solved_within_limit": exact_solved_within_limit,
                    "optimal_actions": exact.primitive_action_cost,
                    "expanded_states": exact.expanded_states,
                    "generated_states": exact.generated_states,
                    "cpu_seconds": exact_cpu,
                    "replay_valid": exact_replay_valid,
                },
                "random": random_result.__dict__,
                "greedy_goal": goal_result.__dict__,
                "greedy_push": greedy_result.__dict__,
            }
        )
    total_cpu = time.process_time() - total_cpu_start

    exact_success = [int(row["exact"]["solved_within_limit"]) for row in rows]
    greedy_success = [int(row["greedy_push"]["solved"]) for row in rows]
    differences = [a - b for a, b in zip(exact_success, greedy_success)]
    stats = config["statistics"]
    point, lower, upper = paired_bootstrap_mean_interval(
        differences,
        replicates=stats["paired_bootstrap_replicates"],
        seed=stats["paired_bootstrap_seed"],
    )
    exact_proofs = sum(row["exact"]["proof_complete"] for row in rows)
    unresolved = sum(not row["exact"]["proof_complete"] for row in rows)
    gate = config["gate"]
    stage1_pass = (
        len(rows) == dataset["selection_count"]
        and exact_proofs >= gate["exact_solver_proofs_required"]
        and unresolved <= gate["unresolved_levels_allowed"]
        and point >= gate["success_gain_point_minimum"]
        and lower > gate["success_gain_ci95_lower_strictly_greater_than"]
        and total_cpu <= config["project"]["max_cpu_seconds"]
    )

    def method_summary(name: str) -> dict:
        episodes = [row[name] for row in rows]
        successful = [episode for episode in episodes if episode["solved"]]
        return {
            "successes": len(successful),
            "success_rate": len(successful) / len(episodes),
            "mean_actions_when_solved": (
                sum(episode["actions"] for episode in successful) / len(successful)
                if successful else None
            ),
            "mean_pushes_when_solved": (
                sum(episode["pushes"] for episode in successful) / len(successful)
                if successful else None
            ),
            "deadlock_episodes": sum(episode["deadlock_seen"] for episode in episodes),
        }

    exact_cpu_values = sorted(row["exact"]["cpu_seconds"] for row in rows)
    source_files = [
        CONFIG_PATH,
        PINS_PATH,
        ROOT / "scripts" / "run_phase4b_boxoban_validation_screen.py",
        ROOT / "src" / "consequence_lab" / "boxoban.py",
        ROOT / "src" / "consequence_lab" / "boxoban_screen.py",
        ROOT / "tests" / "test_boxoban.py",
        ROOT / "tests" / "test_boxoban_screen.py",
    ]
    summary = {
        "experiment": "phase4b_boxoban_validation_headroom_screen",
        "protocol": config["project"]["protocol"],
        "authorization": config["project"]["authorization"],
        "source_revision": dataset["revision"],
        "split": dataset["split"],
        "selected_level_count": len(rows),
        "selected_level_ids": selected_ids,
        "selected_level_text": selected_level_text,
        "source_shard_sha256": source_hashes,
        "downloaded_source_bytes": downloaded_bytes,
        "validation_or_test_data_accessed": True,
        "official_test_data_accessed": False,
        "student_targets_generated": False,
        "student_training_run": False,
        "per_level_results": rows,
        "summary_metrics": {
            "exact": {
                "successes": sum(exact_success),
                "success_rate": sum(exact_success) / len(rows),
                "mean_optimal_actions_when_solved": (
                    sum(row["exact"]["optimal_actions"] for row in rows if row["exact"]["solved_within_limit"])
                    / sum(exact_success) if sum(exact_success) else None
                ),
                "deadlock_episodes": 0,
            },
            "random": method_summary("random"),
            "greedy_goal": method_summary("greedy_goal"),
            "greedy_push": method_summary("greedy_push"),
            "exact_minus_greedy_push_success_gain": point,
            "paired_bootstrap_95_interval": [lower, upper],
            "exact_solver_proofs": exact_proofs,
            "unresolved_solver_levels": unresolved,
            "exact_search_cpu_seconds_p50_p95": [
                exact_cpu_values[int(0.50 * (len(exact_cpu_values) - 1))],
                exact_cpu_values[int(0.95 * (len(exact_cpu_values) - 1))],
            ],
            "total_evaluation_cpu_seconds": total_cpu,
        },
        "stage1_gate": "PASS" if stage1_pass else "FAIL_STOP_BEFORE_STUDENT",
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_sha256": _sha256(_git("status", "--porcelain").encode("utf-8")),
        "code_config_test_sha256": {
            str(path.relative_to(ROOT)): _sha256(path.read_bytes()) for path in source_files
        },
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "selected_level_text" and k != "per_level_results"}, indent=2))
    return summary


if __name__ == "__main__":
    run()
