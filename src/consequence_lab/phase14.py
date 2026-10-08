"""Phase 14 train-only replay and first-action continuation-value audit.

The source loader is deliberately restricted to the archived Phase 12 training
split. Counterfactual targets are outcomes under the frozen finite-sample Phase
12 shooting continuation policy; they are not optimal-MDP Q values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import subprocess
import sys
import time
from typing import Any

import numpy as np

from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.inertial_corridor import ACTION_VECTORS, CorridorState, corridor_step
from consequence_lab.phase12_teacher import make_teacher_query, state_features


CONFIG_PATH = "configs/phase14.yaml"
PROTOCOL_PATH = "reports/phase14_first_action_value_learnability_protocol.md"
SOURCE_DIR = "artifacts/runs/phase12_train_teacher_v1"
SOURCE_FILES = (
    "episodes.json",
    "manifest.json",
    "sample_metadata.npz",
    "student_inputs.npz",
    "teacher_rollouts.npz",
    "teacher_targets.npz",
)
SOURCE_CODE = (
    CONFIG_PATH,
    PROTOCOL_PATH,
    "configs/phase12_teacher_student.yaml",
    "src/consequence_lab/phase14.py",
    "scripts/audit_phase14_training_source.py",
    "scripts/freeze_phase14_source_lock.py",
    "scripts/run_phase14_teacher_audit.py",
    "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/phase12_data.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "scripts/run_phase12_teacher_data.py",
    "scripts/verify_phase12_predata.py",
    "tests/test_phase14.py",
)
PHASE12_BASELINE_COMMIT = "af650b251271477db8bf929e79ff2c3a47fe4b12"
SOURCE_AUDIT_RELATIVE = "artifacts/runs/phase14_training_source_audit_v2.json"
SOURCE_SALT = "phase14-first-action-audit-v1"
STATE_SALT = "phase14-first-action-state-v1"
OUTCOME_ORDER = {"success": 0, "timeout": 1, "collision": 2}


class Phase14IntegrityError(RuntimeError):
    """A Phase 14 source, authorization, or target integrity failure."""


@dataclass(frozen=True)
class SourceDecision:
    episode_seed: int
    layout_sha256: str
    decision_index: int
    global_query_index: int
    state: CorridorState
    scenario: CorridorScenario
    features: tuple[float, ...]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _function_ast(source: str, name: str) -> str:
    tree = ast.parse(source)
    function = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name),
        None,
    )
    if function is None:
        raise Phase14IntegrityError(f"historical source does not define {name}")
    return ast.dump(function, include_attributes=False)


def _validate_phase12_source_provenance(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check archived source hashes, allowing only the audited Phase13 append."""
    archived_config_hash = str(manifest.get("config_sha256", ""))
    current_config_hash = sha256_file(root / "configs/phase12_teacher_student.yaml")
    if archived_config_hash != current_config_hash:
        raise Phase14IntegrityError("Phase12 training config differs from archived provenance")
    archived_sources = manifest.get("source_sha256", {})
    mismatches = []
    for relative, expected in archived_sources.items():
        if not (root / relative).is_file():
            raise Phase14IntegrityError(f"archived Phase12 source is unavailable: {relative}")
        if sha256_file(root / relative) != expected:
            mismatches.append(relative)
    baseline_path = "src/consequence_lab/corridor_baselines.py"
    expected_baseline = archived_sources.get(baseline_path)
    try:
        historical_baseline = subprocess.check_output(
            ["git", "show", f"{PHASE12_BASELINE_COMMIT}:{baseline_path}"],
            cwd=root, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14IntegrityError("could not retrieve archived Phase12 baseline source") from exc
    historical_hash = hashlib.sha256(historical_baseline.encode("utf-8")).hexdigest().upper()
    current_baseline = (root / baseline_path).read_text(encoding="utf-8")
    objective_matches = _function_ast(historical_baseline, "_shooting_objective") == _function_ast(
        current_baseline, "_shooting_objective"
    )
    if (mismatches != [baseline_path] or historical_hash != expected_baseline or not objective_matches):
        raise Phase14IntegrityError(
            "Phase12 source provenance mismatch is not limited to the audited baseline append"
        )
    return {
        "phase12_config_sha256_matches": True,
        "archived_source_hash_count": len(archived_sources),
        "exact_source_hash_mismatch": baseline_path,
        "historical_baseline_commit": PHASE12_BASELINE_COMMIT,
        "historical_baseline_sha256_matches_manifest": True,
        "shooting_objective_ast_matches_historical": True,
    }


def stable_key(salt: str, seed: int, layout_hash: str, decision_index: int | None = None) -> str:
    suffix = "" if decision_index is None else f"|{decision_index}"
    payload = f"{salt}|{seed}|{layout_hash}{suffix}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_training_archive(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Hash-check and load only the Phase 12 training archive (never validation)."""
    root = Path(root).resolve()
    source = root / SOURCE_DIR
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise Phase14IntegrityError("Phase 12 training manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("experiment") != "phase12_train_teacher_v1"
            or manifest.get("status") != "completed_exact_teacher_collection"
            or manifest.get("reserve_accessed") is not False
            or manifest.get("training_performed") is not False
            or manifest.get("episode_count") != 120
            or manifest.get("decision_query_count") != 2095):
        raise Phase14IntegrityError("Phase 12 source manifest contract mismatch")
    for name in SOURCE_FILES:
        path = source / name
        if not path.is_file():
            raise Phase14IntegrityError(f"training source file is missing: {name}")
        manifest_key = Path(name).stem
        if manifest_key in manifest.get("files", {}):
            expected = manifest["files"][manifest_key]["sha256"]
            if sha256_file(path) != expected:
                raise Phase14IntegrityError(f"training source file hash mismatch: {name}")
    arrays: dict[str, np.ndarray] = {}
    for file_name, keys in (
        ("sample_metadata.npz", ("episode_seed", "decision_index", "layout_index")),
        ("student_inputs.npz", ("states", "sequences", "candidate_mask")),
        ("teacher_targets.npz", ("teacher_selected_index",)),
    ):
        with np.load(source / file_name, allow_pickle=False) as archive:
            for key in keys:
                if key not in archive:
                    raise Phase14IntegrityError(f"missing {key} in {file_name}")
                arrays[key] = archive[key].copy()
    episodes = json.loads((source / "episodes.json").read_text(encoding="utf-8"))
    return manifest, episodes, arrays


def replay_training_states(root: Path) -> list[SourceDecision]:
    """Exactly replay all training-only Phase 12 teacher episodes and states."""
    root = Path(root).resolve()
    _, episodes, arrays = _load_training_archive(root)
    expected_seeds = list(range(140000, 140120))
    if [int(item["seed"]) for item in episodes] != expected_seeds:
        raise Phase14IntegrityError("training episode seed order/range changed")
    features_array = arrays["states"]
    sequences_array = arrays["sequences"]
    candidate_mask = arrays["candidate_mask"]
    metadata_seed = arrays["episode_seed"]
    metadata_index = arrays["decision_index"]
    metadata_layout = arrays["layout_index"]
    selected = arrays["teacher_selected_index"]
    decisions: list[SourceDecision] = []
    offset = 0
    for layout_index, episode in enumerate(episodes):
        seed = int(episode["seed"])
        scenario = generate_staggered_corridor(seed)
        if scenario.layout_sha256 != episode["layout_sha256"]:
            raise Phase14IntegrityError(f"training layout hash mismatch for seed {seed}")
        state = scenario.initial_state
        decision_count = int(episode["decision_queries"])
        for decision_index in range(decision_count):
            query_index = offset + decision_index
            if state.status != "active":
                raise Phase14IntegrityError("archived state is not active")
            if (int(metadata_seed[query_index]) != seed
                    or int(metadata_index[query_index]) != decision_index
                    or int(metadata_layout[query_index]) != layout_index):
                raise Phase14IntegrityError("source metadata does not align with training episode")
            feature = state_features(scenario, state)
            if not np.array_equal(feature, features_array[query_index]):
                raise Phase14IntegrityError(
                    f"public state does not exactly replay (seed={seed}, decision={decision_index})"
                )
            decisions.append(SourceDecision(
                seed, scenario.layout_sha256, decision_index, query_index,
                state, scenario, tuple(float(value) for value in feature),
            ))
            action_index = int(selected[query_index])
            if not 0 <= action_index < int(candidate_mask[query_index].sum()):
                raise Phase14IntegrityError("archived teacher selected an invalid candidate index")
            # The selected candidate is the archived first action of the Phase 12 teacher.
            sequence = sequences_array[query_index, action_index]
            action_id = int(sequence[0])
            if not 0 <= action_id < len(ACTION_VECTORS):
                raise Phase14IntegrityError("archived teacher first-action ID is invalid")
            state = corridor_step(state, action_id, scenario.spec).state
        if (state.status != episode["outcome"]
                or state.elapsed_steps - scenario.initial_state.elapsed_steps != int(episode["steps"])):
            raise Phase14IntegrityError(f"archived outcome/length does not replay for seed {seed}")
        offset += decision_count
    if offset != len(features_array) or offset != 2095:
        raise Phase14IntegrityError("training query count does not match frozen archive contract")
    return decisions


def diagnostic_state_selection(decisions: list[SourceDecision]) -> tuple[list[SourceDecision], list[SourceDecision]]:
    """Return 32 hash-ranked one-per-layout states and all states of its first layout."""
    by_seed: dict[int, list[SourceDecision]] = {}
    for item in decisions:
        by_seed.setdefault(item.episode_seed, []).append(item)
    representatives = []
    for seed, rows in by_seed.items():
        representative = min(
            rows,
            key=lambda row: (stable_key(STATE_SALT, seed, row.layout_sha256, row.decision_index), row.decision_index),
        )
        representatives.append(representative)
    representatives.sort(key=lambda row: (stable_key(SOURCE_SALT, row.episode_seed, row.layout_sha256), row.episode_seed))
    if len(representatives) < 32:
        raise Phase14IntegrityError("fewer than 32 distinct training layouts are available")
    selected = representatives[:32]
    chosen_seed = selected[0].episode_seed
    one_layout = by_seed[chosen_seed]
    return selected, one_layout


def build_source_audit(root: Path) -> dict[str, Any]:
    """Create a provenance-only manifest; this evaluates no counterfactual action."""
    root = Path(root).resolve()
    try:
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14IntegrityError("cannot bind source audit to a Git commit") from exc
    if dirty:
        raise Phase14IntegrityError("commit Phase14 source before generating the source audit")
    manifest, episodes, _ = _load_training_archive(root)
    historical_provenance = _validate_phase12_source_provenance(root, manifest)
    decisions = replay_training_states(root)
    diagnostic, one_layout = diagnostic_state_selection(decisions)
    files = {name: sha256_file(root / SOURCE_DIR / name) for name in SOURCE_FILES}
    return {
        "schema": "phase14_training_source_audit_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exact_training_archive_replay_passed_no_phase14_labels",
        "generation_git_commit": commit,
        "generation_source_sha256": {name: sha256_file(root / name) for name in SOURCE_CODE},
        "reserve_accessed": False,
        "phase12_validation_opened": False,
        "counterfactual_transitions_evaluated": False,
        "phase12_manifest_sha256": sha256_file(root / SOURCE_DIR / "manifest.json"),
        "phase12_source_provenance": historical_provenance,
        "training_archive_sha256": files,
        "training_layout_count": len(episodes),
        "training_decision_state_count": len(decisions),
        "diagnostic_state_keys": [
            {"episode_seed": row.episode_seed, "layout_sha256": row.layout_sha256,
             "decision_index": row.decision_index, "global_query_index": row.global_query_index}
            for row in diagnostic
        ],
        "nested_state_sets": {"one": 1, "eight": 8, "thirty_two": 32},
        "one_layout_seed": one_layout[0].episode_seed,
        "one_layout_state_count": len(one_layout),
        "full_training_state_count": len(decisions),
    }


def validate_source_lock(root: Path, lock_path: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root).resolve()
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase14_source_lock_v2" or lock.get("status") != "authorized_phase14a_audit_only":
        raise Phase14IntegrityError("Phase14 source lock does not authorize only the audit stage")
    if lock.get("reserve_access") is not False or lock.get("model_training") is not False:
        raise Phase14IntegrityError("Phase14A lock must deny reserve and model training")
    if lock.get("config_sha256") != sha256_file(root / CONFIG_PATH):
        raise Phase14IntegrityError("Phase14 config hash differs from source lock")
    if lock.get("protocol_sha256") != sha256_file(root / PROTOCOL_PATH):
        raise Phase14IntegrityError("Phase14 protocol hash differs from source lock")
    if lock.get("source_audit_path") != SOURCE_AUDIT_RELATIVE:
        raise Phase14IntegrityError("Phase14 source lock points to a different train-only audit")
    if lock.get("source_code_sha256") != {name: sha256_file(root / name) for name in SOURCE_CODE}:
        raise Phase14IntegrityError("Phase14 source code hash mismatch")
    if lock.get("source_audit_sha256") != sha256_file(Path(audit_path)):
        raise Phase14IntegrityError("Phase14 train-only replay audit hash mismatch")
    if lock.get("phase12_training_archive_sha256") != audit.get("training_archive_sha256"):
        raise Phase14IntegrityError("Phase12 train-only archive hash set differs from lock")
    if audit.get("status") != "exact_training_archive_replay_passed_no_phase14_labels":
        raise Phase14IntegrityError("Phase14 source audit has not passed")
    if audit.get("reserve_accessed") is not False or audit.get("phase12_validation_opened") is not False:
        raise Phase14IntegrityError("forbidden split accessed by source audit")
    if audit.get("counterfactual_transitions_evaluated") is not False:
        raise Phase14IntegrityError("source audit unexpectedly contains Phase14 outcomes")
    if audit.get("training_archive_sha256") != {
        name: sha256_file(root / SOURCE_DIR / name) for name in SOURCE_FILES
    }:
        raise Phase14IntegrityError("Phase 12 training archive changed after audit")
    if audit.get("generation_source_sha256") != {
        name: sha256_file(root / name) for name in SOURCE_CODE
    }:
        raise Phase14IntegrityError("Phase14 source audit was generated from different code/config")
    if audit.get("generation_git_commit") != lock.get("source_commit"):
        raise Phase14IntegrityError("source audit and Phase14A lock refer to different commits")
    if lock.get("selected_state_keys") != audit.get("diagnostic_state_keys"):
        raise Phase14IntegrityError("diagnostic state selection differs from frozen source audit")
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if head != lock.get("source_commit"):
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"],
                cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14IntegrityError("could not verify Phase14 source commit/working tree") from exc
    if dirty:
        raise Phase14IntegrityError("Phase14 source tree must be clean during label generation")
    return lock, audit


def _rank_tuples(values: list[tuple[int, float, int]]) -> tuple[list[float], list[float], list[int]]:
    if len(values) != 9:
        raise ValueError("Phase14 ranking requires exactly nine primitive actions")
    ranks: list[float] = []
    for value in values:
        less = sum(other < value for other in values)
        tied = sum(other == value for other in values)
        ranks.append(less + (tied + 1.0) / 2.0)
    minimum = min(ranks)
    regrets = [(rank - minimum) / 8.0 for rank in ranks]
    optimal = [index for index, value in enumerate(values) if value == min(values)]
    return ranks, regrets, optimal


def evaluate_forced_action(
    source: SourceDecision, action_id: int
) -> dict[str, Any]:
    """Force one primitive step, then continue the frozen exact replanning teacher."""
    if not 0 <= int(action_id) < len(ACTION_VECTORS):
        raise ValueError("invalid forced action ID")
    # CorridorState and CorridorScenario are frozen dataclasses: each branch
    # begins at the exact same immutable source state, without shared mutation.
    current = source.state
    first = corridor_step(current, int(action_id), source.scenario.spec)
    current = first.state
    effort = int(action_id != 0)
    continuation_actions: list[int] = []
    while current.status == "active":
        query = make_teacher_query(source.scenario, current)
        chosen = int(query.sequences[query.selected_index, 0])
        continuation_actions.append(chosen)
        effort += int(chosen != 0)
        current = corridor_step(current, chosen, source.scenario.spec).state
        if len(continuation_actions) > source.scenario.spec.max_steps:
            raise Phase14IntegrityError("continuation exceeded task horizon")
    status = current.status
    if status not in OUTCOME_ORDER:
        raise Phase14IntegrityError(f"counterfactual did not terminate: {status}")
    total_steps = current.elapsed_steps - source.state.elapsed_steps
    terminal_distance = math.hypot(
        source.scenario.spec.goal[0] - current.x,
        source.scenario.spec.goal[1] - current.y,
    )
    primary = float(total_steps if status == "success" else terminal_distance)
    objective = (OUTCOME_ORDER[status], primary, effort)
    return {
        "forced_action_id": int(action_id),
        "student_input": list(source.features),
        "source_episode_seed": source.episode_seed,
        "source_layout_sha256": source.layout_sha256,
        "source_decision_index": source.decision_index,
        "forced_collision": bool(first.collision),
        "forced_success": bool(first.success),
        "forced_timeout": bool(first.truncated),
        "forced_successor": [float(value) for value in (
            first.state.x,
            first.state.y,
            first.state.vx,
            first.state.vy,
        )],
        "terminal_status": status,
        "terminal_state": [float(value) for value in (current.x, current.y, current.vx, current.vy)],
        "terminal_steps_from_source": total_steps,
        "terminal_goal_distance": terminal_distance,
        "total_noncoast_effort": effort,
        "objective_tuple": list(objective),
        "continuation_decision_count": len(continuation_actions),
        "continuation_action_trace": continuation_actions,
    }


def add_state_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: int(row["forced_action_id"]))
    tuples = [tuple(row["objective_tuple"]) for row in ordered]
    ranks, regrets, optimal = _rank_tuples(tuples)  # type: ignore[arg-type]
    for index, row in enumerate(ordered):
        row["average_rank"] = ranks[index]
        row["normalized_regret"] = regrets[index]
        row["optimal"] = index in optimal
    return ordered


def _is_near_primary(best: dict[str, Any], value: dict[str, Any]) -> bool:
    best_tuple = best["objective_tuple"]
    value_tuple = value["objective_tuple"]
    if best_tuple[0] != value_tuple[0]:
        return False
    tolerance = 1.0 if best_tuple[0] == OUTCOME_ORDER["success"] else 0.01
    return abs(float(best_tuple[1]) - float(value_tuple[1])) <= tolerance


def summarize_audit(states: list[list[dict[str, Any]]], repeated_count: int) -> dict[str, Any]:
    action_optimal = [0] * 9
    action_regrets: list[list[float]] = [[] for _ in range(9)]
    action_outcomes = [{"success": 0, "timeout": 0, "collision": 0} for _ in range(9)]
    nonzero_regret_states = 0
    best_action_ids: set[int] = set()
    near_tie_states = 0
    coast_near_optimal = 0
    margins: list[dict[str, Any]] = []
    for rows in states:
        ranked = add_state_ranks(rows)
        best_tuple = min(tuple(row["objective_tuple"]) for row in ranked)
        best_rows = [row for row in ranked if tuple(row["objective_tuple"]) == best_tuple]
        best_action_ids.update(int(row["forced_action_id"]) for row in best_rows)
        near_rows = [row for row in ranked if _is_near_primary(best_rows[0], row)]
        if len(near_rows) > len(best_rows):
            near_tie_states += 1
        coast_near_optimal += int(any(int(row["forced_action_id"]) == 0 for row in near_rows))
        nonzero_regret_states += int(any(float(row["normalized_regret"]) > 0.0 for row in ranked))
        ranked_by_tuple = sorted(ranked, key=lambda row: (tuple(row["objective_tuple"]), int(row["forced_action_id"])))
        second_distinct = next((row for row in ranked_by_tuple if tuple(row["objective_tuple"]) != best_tuple), None)
        if second_distinct is None:
            margin = {"component": None, "value": None}
        else:
            second_tuple = second_distinct["objective_tuple"]
            if second_tuple[0] != best_tuple[0]:
                margin = {"component": "terminal_class", "value": second_tuple[0] - best_tuple[0]}
            elif second_tuple[1] != best_tuple[1]:
                margin = {"component": "conditional_primary_cost", "value": second_tuple[1] - best_tuple[1]}
            else:
                margin = {"component": "control_effort", "value": second_tuple[2] - best_tuple[2]}
        margins.append({
            "best_action_ids": [int(row["forced_action_id"]) for row in best_rows],
            "best_tuple": best_rows[0]["objective_tuple"],
            "second_best_distinct_tuple": None if second_distinct is None else second_distinct["objective_tuple"],
            "best_to_second_margin": margin,
            "near_tie_action_ids": [int(row["forced_action_id"]) for row in near_rows],
        })
        for row in ranked:
            action = int(row["forced_action_id"])
            action_regrets[action].append(float(row["normalized_regret"]))
            action_optimal[action] += int(row["optimal"])
            action_outcomes[action][row["terminal_status"]] += 1
    state_count = len(states)
    metrics = {
        "state_count": state_count,
        "repeatability_states": repeated_count,
        "nonzero_regret_state_count": nonzero_regret_states,
        "nonzero_regret_state_fraction": nonzero_regret_states / state_count,
        "distinct_optimal_action_ids": sorted(best_action_ids),
        "optimal_action_state_fraction": [value / state_count for value in action_optimal],
        "mean_normalized_regret_by_action": [float(np.mean(values)) for values in action_regrets],
        "median_normalized_regret_by_action": [float(np.median(values)) for values in action_regrets],
        "outcome_counts_by_action": action_outcomes,
        "exact_tuple_near_tie_state_count": near_tie_states,
        "coast_near_optimal_state_fraction": coast_near_optimal / state_count,
        "state_margins": margins,
        "informativeness_gate_passed": (
            nonzero_regret_states / state_count >= 0.75 and len(best_action_ids) >= 3
        ),
    }
    return metrics


def phase14_python_environment() -> dict[str, Any]:
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}
