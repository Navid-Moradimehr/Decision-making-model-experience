"""D045 training-only heuristic/budget/teacher-label feasibility pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase4b_boxoban_planner_pilot.yaml"
PINS = ROOT / "experiments" / "source_pins.json"
STAGE0 = ROOT / "artifacts" / "runs" / "phase4_boxoban_qualification_v1" / "summary.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _extract_rows(shard_text: str, wanted_index: int) -> list[str]:
    lines = shard_text.splitlines()
    cursor = 0
    while cursor < len(lines):
        header = lines[cursor]
        if header == "":
            cursor += 1
            continue
        if not header.startswith(";"):
            raise ValueError(f"expected Boxoban header at source line {cursor + 1}")
        level_index = int(header[1:].strip())
        cursor += 1
        while cursor < len(lines) and lines[cursor] == "":
            cursor += 1
        rows = lines[cursor : cursor + 10]
        if len(rows) != 10:
            raise ValueError(f"truncated source level {level_index}")
        if level_index == wanted_index:
            return rows
        cursor += 10
    raise KeyError(wanted_index)


def run() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    pin = json.loads(PINS.read_text(encoding="utf-8"))["sources"]["boxoban_levels"]
    if config["project"]["authorization"] != "D045_training_split_only":
        raise RuntimeError("missing D045 training-only authorization")
    if not config["project"]["training_pilot_authorized"]:
        raise RuntimeError("training pilot is not authorized")
    if config["project"]["validation_access_authorized"] or config["project"]["student_training_authorized"]:
        raise RuntimeError("D045 forbids validation access and student training")
    if config["dataset"]["revision"] != pin["revision"] or config["dataset"]["split"] != "medium/train":
        raise RuntimeError("source pin/split mismatch")
    stage0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    if stage0["dataset"]["split"] != "medium/train":
        raise RuntimeError("Stage-0 exclusion manifest is not a training split")

    from consequence_lab.boxoban import ACTION_ORDER, is_solved, parse_boxoban_rows, solve_optimal, transition
    from consequence_lab.boxoban_screen import sample_level_ids

    ds = config["dataset"]
    selected_ids = sample_level_ids(
        split=ds["split"],
        shard_count=ds["level_count"] // ds["levels_per_shard"],
        levels_per_shard=ds["levels_per_shard"],
        count=ds["level_sample_count"],
        prefix=ds["sample_hash_prefix"],
        exclude_level_ids=stage0["dataset"]["selected_level_ids"],
    )
    output = (ROOT / config["artifacts"]["run_directory"]).resolve()
    runs_root = (ROOT / "artifacts" / "runs").resolve()
    output.relative_to(runs_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    shard_paths = sorted({level_id.split("#", 1)[0] for level_id in selected_ids})
    shard_bytes, shard_hashes = {}, {}
    for shard_path in shard_paths:
        url = f"https://raw.githubusercontent.com/google-deepmind/boxoban-levels/{ds['revision']}/{shard_path}"
        request = Request(url, headers={"User-Agent": "consequence-lab-training-pilot/1"})
        with urlopen(request, timeout=45) as response:
            payload = response.read()
        if not payload or len(payload) > 2_000_000:
            raise ValueError(f"unexpected source shard size for {shard_path}")
        shard_hashes[shard_path] = _sha(payload)  # hash bytes before decoding
        shard_bytes[shard_path] = payload
    if sum(map(len, shard_bytes.values())) > config["project"]["max_generated_artifact_bytes"]:
        raise RuntimeError("source bytes exceed artifact cap")

    levels, raw_levels = {}, {}
    for level_id in selected_ids:
        shard_path, raw_index = level_id.rsplit("#", 1)
        index = int(raw_index)
        rows = _extract_rows(shard_bytes[shard_path].decode("ascii"), index)
        level = parse_boxoban_rows(rows, index)
        if (level.board.width, level.board.height) != (10, 10):
            raise ValueError(f"unexpected dimensions for {level_id}")
        if len(level.initial_state.boxes) != 4 or len(level.board.goals) != 4:
            raise ValueError(f"unexpected box/goal count for {level_id}")
        levels[level_id] = level
        raw_levels[level_id] = f"; {index}\n" + "\n".join(rows)

    solver_cfg = config["solver"]
    budgets = solver_cfg["bounded_expansion_budgets"]
    pilot_cfg = config["teacher_cost_pilot"]
    if pilot_cfg["authorized"] or pilot_cfg["maximum_states"] != 0:
        raise RuntimeError("D045 pilot must not generate teacher action-cost labels")
    cpu_start = time.process_time()
    level_results = []
    teacher_queries = []
    for level_id in selected_ids:
        level = levels[level_id]
        entries = {}
        for mode in (solver_cfg["reference_heuristic"], solver_cfg["comparison_heuristic"]):
            started = time.perf_counter()
            cpu = time.process_time()
            result = solve_optimal(
                level.board,
                level.initial_state,
                heuristic_mode=mode,
                max_expanded_states=solver_cfg["exact_max_expansions_per_level"],
                timeout_seconds=solver_cfg["exact_timeout_seconds_per_level"],
            )
            replay = level.initial_state
            for action in result.actions if result.solved else ():
                replay = transition(level.board, replay, action)
            replay_valid = bool(result.solved and is_solved(level.board, replay))
            entries[mode] = {
                "status": result.status,
                "proof_complete": result.proof_complete,
                "optimal_actions": result.primitive_action_cost,
                "expanded_states": result.expanded_states,
                "generated_states": result.generated_states,
                "wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu,
                "solution_actions": list(result.actions),
                "replay_valid": replay_valid,
            }
        for budget in budgets:
            started = time.perf_counter()
            cpu = time.process_time()
            result = solve_optimal(
                level.board,
                level.initial_state,
                heuristic_mode=solver_cfg["comparison_heuristic"],
                max_expanded_states=budget,
                timeout_seconds=solver_cfg["bounded_timeout_seconds_per_budget"],
            )
            replay = level.initial_state
            for action in result.actions if result.solved else ():
                replay = transition(level.board, replay, action)
            replay_valid = bool(result.solved and is_solved(level.board, replay))
            entries[f"static_push_budget_{budget}"] = {
                "status": result.status,
                "solved_within_budget": result.solved,
                "optimal_actions_if_solved": result.primitive_action_cost,
                "expanded_states": result.expanded_states,
                "wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu,
                "replay_valid": replay_valid,
            }
        reference = entries[solver_cfg["reference_heuristic"]]
        static = entries[solver_cfg["comparison_heuristic"]]
        both_exact_solved = reference["status"] == static["status"] == "solved"
        exact_costs_match = bool(
            both_exact_solved and reference["optimal_actions"] == static["optimal_actions"]
        )
        level_results.append(
            {
                "level_id": level_id,
                "both_exact_solvers_solved": both_exact_solved,
                "exact_costs_match": exact_costs_match,
                "methods": entries,
                "teacher_label_status": "pending_base_solver",
            }
        )

        if static["status"] == "solved" and len(teacher_queries) < pilot_cfg["maximum_states"]:
            # Midpoint state is taken along the deterministic exact static-push path.
            path = static["solution_actions"]
            state = level.initial_state
            midpoint = len(path) // 2
            for action in path[:midpoint]:
                state = transition(level.board, state, action)
            remaining_optimal = len(path) - midpoint
            action_costs = {}
            for action in ACTION_ORDER:
                successor = transition(level.board, state, action)
                query_start = time.perf_counter()
                query_cpu = time.process_time()
                q_result = solve_optimal(
                    level.board,
                    successor,
                    heuristic_mode=solver_cfg["comparison_heuristic"],
                    max_expanded_states=pilot_cfg["successor_solver_max_expansions"],
                    timeout_seconds=pilot_cfg["successor_solver_timeout_seconds"],
                )
                if q_result.solved:
                    target = 1 + q_result.primitive_action_cost
                    status = "exact_cost"
                elif q_result.status == "proven_unsolvable":
                    target = None
                    status = "proven_unsolvable"
                else:
                    target = "unresolved"
                    status = q_result.status
                action_costs[action] = {
                    "status": status,
                    "cost_to_go_after_action": target,
                    "expanded_states": q_result.expanded_states,
                    "wall_seconds": time.perf_counter() - query_start,
                    "cpu_seconds": time.process_time() - query_cpu,
                }
            teacher_queries.append(
                {
                    "level_id": level_id,
                    "path_step_index": midpoint,
                    "remaining_optimal_actions_before_action": remaining_optimal,
                    "board": {
                        "walls": sorted(level.board.walls),
                        "goals": sorted(level.board.goals),
                        "player": state.player,
                        "boxes": state.boxes,
                    },
                    "action_costs": action_costs,
                }
            )
            level_results[-1]["teacher_label_status"] = "four_action_cost_queries_recorded"

    total_cpu = time.process_time() - cpu_start
    if total_cpu > config["project"]["max_cpu_seconds"]:
        raise RuntimeError(f"pilot exceeded total CPU cap: {total_cpu:.1f}s")
    if any(level["level_id"] in set(stage0["dataset"]["selected_level_ids"]) for level in level_results):
        raise AssertionError("pilot sample overlaps Stage-0 IDs")

    def summarize(method_name: str) -> dict:
        records = [level["methods"][method_name] for level in level_results]
        return {
            "solved": sum(item.get("solved_within_budget", item["status"] == "solved") for item in records),
            "count": len(records),
            "mean_expanded_states": sum(item["expanded_states"] for item in records) / len(records),
            "mean_cpu_seconds": sum(item["cpu_seconds"] for item in records) / len(records),
        }

    def summarize_exact(method_name: str) -> dict:
        records = [level["methods"][method_name] for level in level_results]
        solved = [item for item in records if item["status"] == "solved" and item["replay_valid"]]
        return {
            "solved_and_replayed": len(solved),
            "count": len(records),
            "mean_expanded_states": sum(item["expanded_states"] for item in records) / len(records),
            "mean_cpu_seconds": sum(item["cpu_seconds"] for item in records) / len(records),
            "optimal_cost_match_all_solved_pairs": all(level["exact_costs_match"] for level in level_results),
        }

    summary = {
        "experiment": "phase4b_boxoban_training_planner_label_pilot",
        "protocol": config["project"]["protocol"],
        "authorization": config["project"]["authorization"],
        "source_revision": ds["revision"],
        "split": ds["split"],
        "selected_level_count": len(level_results),
        "selected_level_ids": selected_ids,
        "excluded_stage0_ids": stage0["dataset"]["selected_level_ids"],
        "selected_level_text": raw_levels,
        "source_shard_sha256": shard_hashes,
        "validation_or_test_content_accessed": False,
        "student_training_run": False,
        "teacher_policy_dataset_generated": False,
        "level_results": level_results,
        "bounded_static_push_summary": {
            f"budget_{budget}": summarize(f"static_push_budget_{budget}")
            for budget in budgets
        },
        "exact_heuristic_summary": {
            solver_cfg["reference_heuristic"]: summarize_exact(solver_cfg["reference_heuristic"]),
            solver_cfg["comparison_heuristic"]: summarize_exact(solver_cfg["comparison_heuristic"]),
        },
        "teacher_cost_pilot": {
            "states_labeled": len(teacher_queries),
            "action_cost_queries": sum(len(row["action_costs"]) for row in teacher_queries),
            "queries": teacher_queries,
        },
        "resource": {"total_cpu_seconds": total_cpu},
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_sha256": _sha(_git("status", "--porcelain").encode("utf-8")),
        "hashes": {
            str(path.relative_to(ROOT)): _sha(path.read_bytes())
            for path in [
                CONFIG,
                PINS,
                ROOT / "scripts" / "run_phase4b_boxoban_training_pilot.py",
                ROOT / "src" / "consequence_lab" / "boxoban.py",
                ROOT / "src" / "consequence_lab" / "boxoban_screen.py",
                ROOT / "tests" / "test_boxoban.py",
                ROOT / "tests" / "test_boxoban_screen.py",
            ]
        },
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"selected_level_text", "level_results", "teacher_cost_pilot"}}, indent=2))
    return summary


if __name__ == "__main__":
    run()
