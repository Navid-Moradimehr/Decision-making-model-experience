"""Independent source/selection/path re-audit of the Phase-4A training smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

import yaml

from consequence_lab.boxoban import is_solved, parse_boxoban_shard, solve_optimal, transition


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts" / "runs" / "phase4_boxoban_qualification_v1"
CONFIG_PATH = ROOT / "configs" / "phase4_boxoban_qualification.yaml"
PARITY_CONFIG_PATH = ROOT / "configs" / "phase4_boxoban_transition_parity.yaml"
PIN_PATH = ROOT / "experiments" / "source_pins.json"
HEADER = re.compile(r"^\s*;\s*(\d+)\s*$")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ids(split: str, shards: int, levels_per_shard: int, count: int, prefix: str) -> list[str]:
    ids = []
    for shard in range(shards):
        for index in range(levels_per_shard):
            level_id = f"{split}/{shard:03d}.txt#{index:04d}"
            digest = hashlib.sha256(f"{prefix}{level_id}".encode("ascii")).hexdigest()
            ids.append((digest, level_id))
    return [level_id for _, level_id in sorted(ids)[:count]]


def _extract_rows(shard_text: str, index_needed: int) -> list[str]:
    lines = shard_text.splitlines()
    position = 0
    while position < len(lines):
        header = HEADER.fullmatch(lines[position])
        if header is None:
            if lines[position] == "":
                position += 1
                continue
            raise AssertionError(f"unexpected source line {position + 1}")
        index = int(header.group(1))
        position += 1
        while position < len(lines) and lines[position] == "":
            position += 1
        rows = lines[position : position + 10]
        if len(rows) != 10:
            raise AssertionError(f"source puzzle {index} is truncated")
        if index == index_needed:
            return rows
        position += 10
    raise AssertionError(f"source puzzle {index_needed} not found")


def _reference_transition(board, state, action: str):
    """Independent test-only transcription, not calling production transition."""
    delta = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}
    if action not in delta:
        raise ValueError("invalid reference action")
    dx, dy = delta[action]

    def free(point):
        x, y = point
        return 0 <= x < board.width and 0 <= y < board.height and point not in board.walls

    next_player = (state.player[0] + dx, state.player[1] + dy)
    if not free(next_player):
        return state, "blocked_move"
    boxes = set(state.boxes)
    if next_player not in boxes:
        from consequence_lab.boxoban import BoxobanState

        return BoxobanState(next_player, state.boxes), "legal_walk"
    next_box = (next_player[0] + dx, next_player[1] + dy)
    if not free(next_box) or next_box in boxes:
        return state, "blocked_push"
    boxes.remove(next_player)
    boxes.add(next_box)
    from consequence_lab.boxoban import BoxobanState

    canonical = tuple(sorted(boxes, key=lambda point: (point[1], point[0])))
    return BoxobanState(next_player, canonical), "legal_push"


def run() -> dict:
    config_bytes = CONFIG_PATH.read_bytes()
    config = yaml.safe_load(config_bytes)
    parity_config = yaml.safe_load(PARITY_CONFIG_PATH.read_text(encoding="utf-8"))
    pins = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    summary_path = RUN_DIR / "summary.json"
    level_text_path = RUN_DIR / "selected_train_levels.txt"
    audit_relative = Path(parity_config["artifacts"]["new_audit_file"])
    audit_path = ROOT / audit_relative
    try:
        audit_path.resolve().relative_to(RUN_DIR.resolve())
    except ValueError as exc:
        raise ValueError("new audit artifact must stay in the original run directory") from exc
    if audit_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {audit_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["config_sha256"] != _hash(config_bytes):
        raise AssertionError("saved Stage-0 config hash differs from the frozen config")
    if pins["sources"]["boxoban_levels"]["revision"] != config["dataset"]["revision"]:
        raise AssertionError("source revision pin mismatch")
    if summary["dataset"]["revision"] != config["dataset"]["revision"]:
        raise AssertionError("run source revision differs from the config")
    if (
        summary["dataset"]["revision"] != parity_config["source"]["dataset_revision"]
        or parity_config["source"]["split"] != "medium/train"
        or parity_config["project"]["validation_or_test_data_access"]
    ):
        raise AssertionError("transition re-audit source/split boundary mismatch")
    ids = summary["dataset"]["selected_level_ids"]
    sampler = config["dataset"]["level_sampler"]
    prefix = sampler["hash_input"].split("{canonical_level_id}", 1)[0]
    expected_ids = _ids(
        config["dataset"]["train_directory"],
        config["dataset"]["train_level_count_reported"] // config["dataset"]["levels_per_shard"],
        config["dataset"]["levels_per_shard"],
        sampler["train_smoke_count"],
        prefix,
    )
    if ids != expected_ids or len(set(ids)) != sampler["train_smoke_count"]:
        raise AssertionError("saved level IDs differ from the independent deterministic sample")
    if any(not level_id.startswith("medium/train/") for level_id in ids):
        raise AssertionError("non-training level present in Stage-0 sample")
    if summary["validation_or_test_levels_accessed"] or summary["stage1_authorized"]:
        raise AssertionError("run summary violates the Stage-0 access boundary")

    expected_shards = sorted({level_id.split("#", 1)[0] for level_id in ids})
    stored_hashes = summary["dataset"]["downloaded_shard_sha256"]
    if sorted(stored_hashes) != expected_shards:
        raise AssertionError("saved shard hash inventory differs from selected training IDs")

    shard_texts: dict[str, str] = {}
    for shard_path in expected_shards:
        url = (
            "https://raw.githubusercontent.com/google-deepmind/boxoban-levels/"
            f"{config['dataset']['revision']}/{shard_path}"
        )
        request = Request(url, headers={"User-Agent": "consequence-lab-saved-data-audit/1"})
        with urlopen(request, timeout=30) as response:
            raw = response.read()
        if _hash(raw) != stored_hashes[shard_path]:
            raise AssertionError(f"pinned source shard hash changed: {shard_path}")
        shard_texts[shard_path] = raw.decode("ascii")

    serialized = level_text_path.read_text(encoding="ascii").strip("\n")
    blocks = [block for block in serialized.split("\n\n") if block]
    saved_levels = [parse_boxoban_shard(block)[0] for block in blocks]
    if len(saved_levels) != len(ids):
        raise AssertionError("saved selected-level artifact count differs from manifest")
    results_by_id = {row["level_id"]: row for row in summary["results"]}
    if set(results_by_id) != set(ids):
        raise AssertionError("saved solver-result IDs differ from level sample")

    solver_cfg = config["stage0"]["solver"]
    source_content_matches = 0
    replay_matches = 0
    solver_cost_matches = 0
    transition_cases = 0
    transition_mismatches = 0
    transition_coverage: set[str] = set()
    transition_case_keys: set[tuple] = set()
    re_audit_started = time.perf_counter()
    for level_id, saved_level in zip(ids, saved_levels):
        shard_path, raw_index = level_id.rsplit("#", 1)
        index = int(raw_index)
        if saved_level.level_index != index:
            raise AssertionError(f"saved level index differs for {level_id}")
        source_rows = _extract_rows(shard_texts[shard_path], index)
        # The level file is small, so compare the parsed object with the exact
        # ordered source block rather than relying on the original collector.
        if saved_level.board.width != 10 or saved_level.board.height != 10:
            raise AssertionError(f"wrong board dimensions for {level_id}")
        source_level = parse_boxoban_shard(f"; {index}\n" + "\n".join(source_rows))[0]
        if source_level != saved_level:
            raise AssertionError(f"selected level content differs from pinned source: {level_id}")
        source_content_matches += 1

        state = saved_level.initial_state
        states_to_check = [state]
        for prefix_action in parity_config["parity"]["action_prefix"]:
            state = transition(saved_level.board, state, prefix_action)
            states_to_check.append(state)
        for fixture_state in states_to_check:
            for action in parity_config["parity"]["compare_all_actions_at_each_state"]:
                expected, category = _reference_transition(
                    saved_level.board, fixture_state, action
                )
                actual = transition(saved_level.board, fixture_state, action)
                transition_cases += 1
                transition_case_keys.add((level_id, fixture_state, action))
                transition_coverage.add(category)
                if actual != expected:
                    transition_mismatches += 1

        result_row = results_by_id[level_id]
        if result_row["status"] != "solved" or not result_row["replay_valid"]:
            raise AssertionError(f"saved result is not a replay-verified solution: {level_id}")
        state = saved_level.initial_state
        actions = result_row["solution_actions"]
        for action in actions:
            state = transition(saved_level.board, state, action)
        if not is_solved(saved_level.board, state) or len(actions) != result_row[
            "optimal_primitive_action_cost"
        ]:
            raise AssertionError(f"saved action path does not reproduce the result: {level_id}")
        replay_matches += 1

        independent_result = solve_optimal(
            saved_level.board,
            saved_level.initial_state,
            max_expanded_states=solver_cfg["maximum_expanded_states_per_level"],
            timeout_seconds=solver_cfg["wall_clock_seconds_per_level"],
        )
        if (
            independent_result.status != "solved"
            or independent_result.primitive_action_cost
            != result_row["optimal_primitive_action_cost"]
        ):
            raise AssertionError(f"recomputed optimal cost differs for {level_id}")
        solver_cost_matches += 1

    if len(transition_case_keys) < parity_config["parity"]["minimum_distinct_level_state_action_cases"]:
        raise AssertionError("independent transition audit did not reach its minimum case count")
    if transition_mismatches:
        raise AssertionError(f"independent transition parity mismatches: {transition_mismatches}")
    required_coverage = parity_config["parity"]["required_coverage"]
    if "blocked_move" in required_coverage and "blocked_move" not in transition_coverage:
        raise AssertionError("transition parity sample missed blocked moves")
    if "legal_walk" in required_coverage and "legal_walk" not in transition_coverage:
        raise AssertionError("transition parity sample missed legal walking")
    if "legal_push" in required_coverage and "legal_push" not in transition_coverage:
        raise AssertionError("transition parity sample missed legal pushes")

    audit = {
        "audit": "phase4a_boxoban_saved_data_independent_reaudit",
        "config_sha256": _hash(config_bytes),
        "source_revision": config["dataset"]["revision"],
        "selected_train_level_count": len(ids),
        "validation_or_test_accessed": False,
        "recomputed_training_sample_matches": True,
        "pinned_source_shard_hash_matches": len(expected_shards),
        "source_level_content_matches": source_content_matches,
        "solution_replays_matches": replay_matches,
        "optimal_cost_recomputations_match": solver_cost_matches,
        "independent_transition_parity_cases": transition_cases,
        "independent_transition_parity_distinct_cases": len(transition_case_keys),
        "independent_transition_parity_mismatches": transition_mismatches,
        "independent_transition_case_coverage": sorted(transition_coverage),
        "reaudit_wall_seconds": time.perf_counter() - re_audit_started,
        "stage0_gate": "PASS",
        "stage1_authorized": False,
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
