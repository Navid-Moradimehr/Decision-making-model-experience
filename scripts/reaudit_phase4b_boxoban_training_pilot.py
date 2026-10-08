"""Independent source/sample/result audit of the D045 training-only pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "runs" / "phase4b_boxoban_planner_pilot_v1"
CONFIG = ROOT / "configs" / "phase4b_boxoban_planner_pilot.yaml"
AUDIT = RUN / "independent_audit_v1.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_rows(text: str, index: int) -> list[str]:
    lines = text.splitlines()
    cursor = 0
    while cursor < len(lines):
        if lines[cursor] == "":
            cursor += 1
            continue
        if not lines[cursor].startswith(";"):
            raise ValueError("unexpected source shard content")
        current = int(lines[cursor][1:].strip())
        cursor += 1
        while cursor < len(lines) and not lines[cursor]:
            cursor += 1
        rows = lines[cursor : cursor + 10]
        if len(rows) != 10:
            raise ValueError(f"truncated level {current}")
        if current == index:
            return rows
        cursor += 10
    raise KeyError(index)


def run() -> dict:
    if AUDIT.exists():
        raise FileExistsError(f"refusing to overwrite {AUDIT}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    stage0 = json.loads(
        (ROOT / "artifacts" / "runs" / "phase4_boxoban_qualification_v1" / "summary.json").read_text(encoding="utf-8")
    )
    from consequence_lab.boxoban import is_solved, parse_boxoban_rows, solve_optimal, transition
    from consequence_lab.boxoban_screen import sample_level_ids

    ds = config["dataset"]
    selected = sample_level_ids(
        split=ds["split"],
        shard_count=ds["level_count"] // ds["levels_per_shard"],
        levels_per_shard=ds["levels_per_shard"],
        count=ds["level_sample_count"],
        prefix=ds["sample_hash_prefix"],
        exclude_level_ids=stage0["dataset"]["selected_level_ids"],
    )
    if selected != summary["selected_level_ids"]:
        raise AssertionError("training-only sample does not match independent ID re-derivation")

    shards = {}
    for path, expected_hash in sorted(summary["source_shard_sha256"].items()):
        url = f"https://raw.githubusercontent.com/google-deepmind/boxoban-levels/{ds['revision']}/{path}"
        req = Request(url, headers={"User-Agent": "consequence-lab-pilot-audit/1"})
        with urlopen(req, timeout=45) as response:
            payload = response.read()
        if _sha(payload) != expected_hash:
            raise AssertionError(f"pinned source hash mismatch: {path}")
        shards[path] = payload.decode("ascii")

    started = time.perf_counter()
    checked_sources = checked_exact = checked_budgets = checked_replays = 0
    for row, level_id in zip(summary["level_results"], selected):
        shard, raw_index = level_id.rsplit("#", 1)
        index = int(raw_index)
        rows = _extract_rows(shards[shard], index)
        level_text = f"; {index}\n" + "\n".join(rows)
        if summary["selected_level_text"].get(level_id) != level_text:
            raise AssertionError(f"saved training puzzle differs from pinned source: {level_id}")
        checked_sources += 1
        level = parse_boxoban_rows(rows, index)
        method_results = {}
        for mode in (config["solver"]["reference_heuristic"], config["solver"]["comparison_heuristic"]):
            result = solve_optimal(
                level.board,
                level.initial_state,
                heuristic_mode=mode,
                max_expanded_states=config["solver"]["exact_max_expansions_per_level"],
                timeout_seconds=config["solver"]["exact_timeout_seconds_per_level"],
            )
            saved = row["methods"][mode]
            if (result.status, result.primitive_action_cost, result.expanded_states) != (
                saved["status"], saved["optimal_actions"], saved["expanded_states"]
            ):
                raise AssertionError(f"exact result differs for {mode} / {level_id}")
            method_results[mode] = result
            if result.solved:
                state = level.initial_state
                for action in result.actions:
                    state = transition(level.board, state, action)
                if not is_solved(level.board, state):
                    raise AssertionError(f"saved exact path does not replay: {level_id}")
                checked_replays += 1
            checked_exact += 1
        if method_results[config["solver"]["reference_heuristic"]].solved and method_results[config["solver"]["comparison_heuristic"]].solved:
            if method_results[config["solver"]["reference_heuristic"]].primitive_action_cost != method_results[config["solver"]["comparison_heuristic"]].primitive_action_cost:
                raise AssertionError(f"exact heuristic variants disagree on optimal cost: {level_id}")
        for budget in config["solver"]["bounded_expansion_budgets"]:
            result = solve_optimal(
                level.board,
                level.initial_state,
                heuristic_mode=config["solver"]["comparison_heuristic"],
                max_expanded_states=budget,
                timeout_seconds=config["solver"]["bounded_timeout_seconds_per_budget"],
            )
            saved = row["methods"][f"static_push_budget_{budget}"]
            if (result.status, result.expanded_states, result.primitive_action_cost) != (
                saved["status"], saved["expanded_states"], saved["optimal_actions_if_solved"]
            ):
                raise AssertionError(f"bounded result differs at {budget} / {level_id}")
            if result.solved:
                state = level.initial_state
                for action in result.actions:
                    state = transition(level.board, state, action)
                if not is_solved(level.board, state):
                    raise AssertionError(f"bounded search solution replay failed: {level_id}")
                checked_replays += 1
            checked_budgets += 1

    if summary["teacher_cost_pilot"]["states_labeled"] != 0 or summary["student_training_run"]:
        raise AssertionError("D045 pilot unexpectedly generated labels or trained a student")
    audit = {
        "audit": "phase4b_boxoban_training_planner_pilot_independent_reaudit",
        "protocol": config["project"]["protocol"],
        "run_summary_sha256": _sha((RUN / "summary.json").read_bytes()),
        "independent_train_id_sample_matches": True,
        "stage0_ids_excluded": not bool(set(selected) & set(stage0["dataset"]["selected_level_ids"])),
        "source_shards_rehashed": len(shards),
        "selected_source_levels_match": checked_sources,
        "exact_heuristic_result_recomputations": checked_exact,
        "bounded_budget_result_recomputations": checked_budgets,
        "solution_replays": checked_replays,
        "teacher_action_cost_labels_generated": False,
        "validation_or_test_accessed": False,
        "student_training_run": False,
        "audit_wall_seconds": time.perf_counter() - started,
        "audit_status": "PASS",
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return audit


if __name__ == "__main__":
    run()
