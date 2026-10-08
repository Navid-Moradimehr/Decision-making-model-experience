"""Phase29: bounded exact path and edit-predictability feasibility audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import numpy as np
import yaml

from consequence_lab import phase16c_prior_guided as p16c
from consequence_lab import phase16d_compute_frontier as p16d
from consequence_lab import phase18_proposal_controls as p18
from consequence_lab import phase20_exact_plan_warmstart as p20
from consequence_lab import phase24_search_budget_profile as p24
from consequence_lab import phase28_sequential_exact_search as p28
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import ACTION_VECTORS, corridor_step
from consequence_lab.phase12_teacher import state_features

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase29_iterative_refinement_v1.yaml"
SOURCE = "src/consequence_lab/phase29_iterative_refinement.py"
TESTS = "tests/test_phase29_iterative_refinement.py"
PROTOCOL = "reports/phase29_iterative_refinement_protocol.md"
AMENDMENT = "reports/phase29_parent_protocol_hash_compatibility_amendment.md"
LOCK_SCHEMA = "phase29-iterative-refinement-lock-v1"
OUTCOME = {0: "success", 1: "active", 2: "timeout", 3: "collision"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config(root: Path = ROOT) -> dict:
    result = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    if result.get("schema") != "phase29-iterative-refinement-v1":
        raise ValueError("wrong Phase29 config schema")
    if (result["planning"]["max_refinement_depth"] != 4
            or result["planning"]["maximum_horizon"] != 8
            or result["planning"]["action_count"] != len(ACTION_VECTORS)
            or result["planning"]["refinement_operator"] != "single_position_substitution"):
        raise ValueError("Phase29 planning contract changed")
    return result


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _dependencies(root: Path, cfg: dict) -> list[str]:
    paths = [CONFIG, PROTOCOL, AMENDMENT, SOURCE, TESTS, "pyproject.toml", "uv.lock",
             cfg["parent"]["phase28_config"], cfg["parent"]["phase28_source"],
             cfg["parent"]["phase28_protocol"],
             cfg["parent"]["phase28_lock"], cfg["parent"]["phase28_records"],
             cfg["parent"]["phase28_result"], cfg["parent"]["phase28_run_manifest"],
             cfg["parent"]["phase28_result_manifest"],
             "src/consequence_lab/phase12_teacher.py",
             "src/consequence_lab/phase15_d164.py",
             "src/consequence_lab/phase16c_prior_guided.py",
             "src/consequence_lab/phase16d_compute_frontier.py",
             "src/consequence_lab/phase17_proposal_guidance.py",
             "src/consequence_lab/phase18_proposal_controls.py",
             "src/consequence_lab/phase20_exact_plan_warmstart.py",
             "src/consequence_lab/phase24_search_budget_profile.py",
             "src/consequence_lab/corridor_baselines.py",
             "src/consequence_lab/corridor_tasks.py",
             "src/consequence_lab/inertial_corridor.py"]
    return sorted(set(paths))


def _verify_phase28_protocol_status_compatibility(locked_protocol: bytes,
                                                 current_protocol: bytes,
                                                 expected_hash: str) -> dict:
    if hashlib.sha256(locked_protocol).hexdigest().upper() != expected_hash:
        raise RuntimeError("Phase28 source-commit protocol does not match its frozen dependency hash")
    prior_status = b"status: protocol-frozen-pending-audit-lock"
    completed_status = b"status: protocol-frozen-complete"
    if (locked_protocol.count(prior_status) != 1 or current_protocol.count(completed_status) != 1
            or current_protocol.replace(completed_status, prior_status, 1) != locked_protocol):
        raise RuntimeError("Phase28 protocol differs from its locked version beyond the known status-only update")
    return {"locked_protocol_sha256": expected_hash,
        "current_protocol_sha256": hashlib.sha256(current_protocol).hexdigest().upper(),
        "only_difference": "frontmatter status: protocol-frozen-pending-audit-lock -> protocol-frozen-complete"}


def _phase28_locked_context(root: Path, cfg: dict) -> tuple[dict, list[dict], str, dict]:
    parent = cfg["parent"]
    p28cfg = p28.config(root)
    lock_path = root / p28cfg["outputs"]["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_hash = sha256(lock_path)
    if lock.get("schema") != p28.LOCK_SCHEMA:
        raise RuntimeError("Phase28 lock schema mismatch")
    protocol_path = parent["phase28_protocol"]
    expected_protocol_hash = lock["dependency_hashes"].get(protocol_path)
    if expected_protocol_hash is None:
        raise RuntimeError("Phase28 lock does not bind its protocol document")
    locked_protocol = subprocess.check_output(
        ["git", "show", f"{lock['source_commit']}:{protocol_path}"], cwd=root)
    current_protocol = (root / protocol_path).read_bytes()
    compatibility = _verify_phase28_protocol_status_compatibility(
        locked_protocol, current_protocol, expected_protocol_hash)
    for path, expected in lock["dependency_hashes"].items():
        if path != protocol_path and sha256(root / path) != expected:
            raise RuntimeError(f"Phase28 dependency hash mismatch: {path}")
    if p28.parent_bindings(root, p28cfg, audit_phase27b=False) != lock["parent_hashes"]:
        raise RuntimeError("Phase28 parent lineage mismatch")
    cutoff = lock["native_cutoff"].get("historical_phase27b_native_cutoff", {})
    if (cutoff.get("gap_cutoff") != p28cfg["allocation"]["native_gap_cutoff"]
            or cutoff.get("tie_hash_cutoff") != p28cfg["allocation"]["native_tie_hash_cutoff"]
            or cutoff.get("source_records") != p28cfg["parent"]["phase27b_records"]
            or cutoff.get("observed_rate") is None):
        raise RuntimeError("Phase28 native cutoff does not match its frozen config")
    audit_path = root / p28cfg["outputs"]["layout_audit"]
    schedule_path = root / p28cfg["outputs"]["execution_order"]
    if (sha256(audit_path) != lock["layout_audit_sha256"]
            or sha256(schedule_path) != lock["execution_order_sha256"]):
        raise RuntimeError("Phase28 identity/schedule hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule != p28._execution_order(audit, p28cfg):
        raise RuntimeError("Phase28 execution schedule does not reproduce")
    compatibility["locked_source_commit"] = lock["source_commit"]
    return lock, schedule, lock_hash, compatibility


def _audit_phase28_saved_records(root: Path, cfg: dict) -> dict:
    p28cfg = p28.config(root)
    lock, schedule, lock_hash, compatibility = _phase28_locked_context(root, cfg)
    records_path = root / p28cfg["outputs"]["records"]
    result_path = root / p28cfg["outputs"]["result"]
    manifest = json.loads((root / p28cfg["outputs"]["run_manifest"]).read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    score_manifest = json.loads((root / p28cfg["outputs"]["result_manifest"]).read_text(encoding="utf-8"))
    records_hash = sha256(records_path)
    if (manifest["records_sha256"] != records_hash or manifest["execution_lock_sha256"] != lock_hash
            or score_manifest["records_sha256"] != records_hash
            or score_manifest["result_sha256"] != sha256(result_path)
            or result["records_sha256"] != records_hash
            or score_manifest["planner_episode_reruns"] != 0
            or score_manifest["candidate_evaluations_during_scoring"] != 0
            or manifest.get("reserve_access") or manifest.get("test_access") or manifest.get("d168_access")
            or result.get("reserve_access") or result.get("test_access") or result.get("d168_access")):
        raise RuntimeError("Phase28 post-run hash/accounting chain failed")
    # Phase28 dependencies are checked above; this temporary lock-context patch only
    # lets its unchanged scorer recompute historical rows after the status-only note edit.
    with patch.object(p28, "_read_lock", return_value=(lock, schedule, lock_hash)):
        reproduced = p28._score_records(root, write=False)
    if p28.canonical(reproduced) != p28.canonical(result):
        raise RuntimeError("Phase28 metrics do not reproduce from its locked records")
    return {"audit_pass": True, "episodes": len(schedule), "records_sha256": records_hash,
        "result_sha256": sha256(result_path), "execution_lock_sha256": lock_hash,
        "protocol_compatibility": compatibility, "candidate_evaluations_during_audit": 0,
        "reserve_access": False, "test_access": False, "d168_access": False}


def _verify_parent(root: Path, cfg: dict) -> dict:
    parent = cfg["parent"]
    _, _, lock_hash, compatibility = _phase28_locked_context(root, cfg)
    record_path = root / parent["phase28_records"]
    result_path = root / parent["phase28_result"]
    run_manifest_path = root / parent["phase28_run_manifest"]
    result_manifest_path = root / parent["phase28_result_manifest"]
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    result_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if lock_hash != p28.sha256(root / parent["phase28_lock"]):
        raise RuntimeError("Phase28 parent lock hash is inconsistent")
    if sha256(record_path) != run_manifest["records_sha256"]:
        raise RuntimeError("Phase28 parent records differ from run manifest")
    if sha256(record_path) != result_manifest["records_sha256"]:
        raise RuntimeError("Phase28 parent records differ from result manifest")
    if sha256(result_path) != result_manifest["result_sha256"]:
        raise RuntimeError("Phase28 parent result differs from result manifest")
    if (result.get("status") != "complete" or result.get("records_sha256") != sha256(record_path)
            or result_manifest.get("execution_lock_sha256") != lock_hash
            or run_manifest.get("reserve_access") or run_manifest.get("test_access")
            or run_manifest.get("d168_access") or result.get("reserve_access")
            or result.get("test_access") or result.get("d168_access")
            or result_manifest.get("reserve_access") or result_manifest.get("test_access")
            or result_manifest.get("d168_access")):
        raise RuntimeError("Phase28 parent status or sealed-data flags are invalid")
    if sha256(record_path) != "4AAFED8AB503EF12E1DB0E09ED93902463D260B2B9DEBD8A5D955DDBC87F579A":
        raise RuntimeError("Phase28 record corpus is not the reviewed fixed input")
    if sha256(result_path) != "E818F943EA18DC63C5BB89C2357715F7C95419CD4B22F59FE57F4085DBDFE64B":
        raise RuntimeError("Phase28 result is not the reviewed fixed input")
    return {"lock_sha256": lock_hash, "records_sha256": sha256(record_path),
            "result_sha256": sha256(result_path), "run_manifest_sha256": sha256(run_manifest_path),
            "result_manifest_sha256": sha256(result_manifest_path), "episodes": result["episodes"],
            "protocol_compatibility": compatibility,
            "layouts": result["layouts"]}


def freeze(root: Path = ROOT) -> dict:
    cfg = config(root)
    lock_path = root / cfg["outputs"]["lock"]
    run_dir = root / cfg["outputs"]["run_dir"]
    if lock_path.exists() or run_dir.exists():
        raise FileExistsError("Phase29 lock or output already exists; refusing overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit the Phase29 protocol, implementation, tests, and config before lock")
    if cfg["execution"]["reserve_test_d168_access"] or cfg["execution"]["new_episodes"]:
        raise RuntimeError("Phase29 scope cannot include new episodes or sealed data")
    parent = _verify_parent(root, cfg)
    hashes = {path: sha256(root / path) for path in _dependencies(root, cfg)}
    lock = {"schema": LOCK_SCHEMA, "source_commit": _git(root, "rev-parse", "HEAD"),
            "created_utc": datetime.now(timezone.utc).isoformat(), "parent": parent,
            "state_sample_cap": cfg["planning"]["maximum_selected_states"],
            "exact_intermediate_evaluation_cap": cfg["planning"]["maximum_new_exact_sequence_evaluations"],
            "hashes": hashes, "new_episodes": False, "training": False,
            "closed_loop": False, "reserve_access": False, "test_access": False, "d168_access": False}
    p18.write_new(lock_path, lock)
    return {"lock_sha256": sha256(lock_path), "source_commit": lock["source_commit"],
            "dependencies": len(hashes), "parent": parent}


def verify(root: Path = ROOT) -> tuple[dict, dict, str]:
    cfg = config(root)
    lock_path = root / cfg["outputs"]["lock"]
    if not lock_path.exists():
        raise FileNotFoundError("Phase29 execution lock is missing")
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"], cwd=root)
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("Phase29 lock must be committed unchanged")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != LOCK_SCHEMA or lock.get("reserve_access") or lock.get("test_access") or lock.get("d168_access"):
        raise RuntimeError("Phase29 lock scope/schema mismatch")
    for path, expected in lock["hashes"].items():
        if sha256(root / path) != expected:
            raise RuntimeError(f"Phase29 locked dependency changed: {path}")
    parent = _verify_parent(root, cfg)
    if parent != lock["parent"]:
        raise RuntimeError("Phase28 parent artifact binding changed")
    return cfg, lock, sha256(lock_path)


def _candidate_row(evaluation) -> dict:
    objective = [int(evaluation.objective[0]), float(evaluation.objective[1]),
                 float(evaluation.objective[2])]
    return {"sequence": list(evaluation.sequence), "objective": objective,
            "category": OUTCOME[objective[0]], "transitions": int(evaluation.transitions),
            "success_reached": bool(evaluation.success_reached)}


def _selected_source_states(root: Path, cfg: dict) -> list[tuple[dict, dict]]:
    source = root / cfg["parent"]["phase28_records"]
    selected: list[tuple[dict, dict]] = []
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            episode = json.loads(line)
            if episode.get("method") != cfg["source"]["method"]:
                continue
            if not cfg["source"]["block"][0] <= int(episode["seed"]) < cfg["source"]["block"][1]:
                raise RuntimeError("Phase28 source episode is outside the locked block")
            eligible = [decision for decision in episode["decision_diagnostics"]
                        if decision.get("previous_plan") is not None
                        and int(decision["horizon"]) >= int(cfg["source"]["horizon_minimum"])]
            if not eligible:
                continue
            seed, replica = int(episode["seed"]), int(episode["replica"])
            decision = min(eligible, key=lambda row: (
                hashlib.sha256(f"{seed}|{replica}|{int(row['decision_index'])}|phase29-state-v1".encode()).hexdigest(),
                int(row["decision_index"])))
            selected.append((episode, decision))
    selected.sort(key=lambda pair: (int(pair[0]["seed"]), int(pair[0]["replica"])))
    if len(selected) > int(cfg["planning"]["maximum_selected_states"]):
        raise RuntimeError("selected states exceed frozen cap")
    if len(selected) < 1000:
        raise RuntimeError(f"too few eligible Phase28 states: {len(selected)}")
    return selected


def _reconstruct_state(episode: dict, selected: dict):
    scenario = generate_staggered_corridor(int(episode["seed"]))
    if scenario.layout_sha256 != episode["layout_sha256"]:
        raise RuntimeError("Phase28 layout identity failed during state replay")
    state = scenario.initial_state
    found = False
    for decision in episode["decision_diagnostics"]:
        if int(decision["decision_index"]) == int(selected["decision_index"]):
            found = True
            break
        if int(decision["decision_index"]) != state.elapsed_steps:
            raise RuntimeError("Phase28 decision index does not match replay state")
        state = corridor_step(state, int(decision["chosen_action"]), scenario.spec).state
        if state.status != "active":
            raise RuntimeError("episode terminated before selected decision")
    if not found or state.elapsed_steps != int(selected["decision_index"]):
        raise RuntimeError("selected Phase28 state could not be reconstructed")
    reconstructed_raw = state_features(scenario, state).astype(np.float64)
    logged_raw = np.asarray(selected["state_raw52"], dtype=np.float64)
    if reconstructed_raw.shape != (52,) or not np.array_equal(reconstructed_raw, logged_raw):
        raise RuntimeError("Phase28 deployable state features fail exact replay parity")
    if state.status != "active":
        raise RuntimeError("selected planning state is not active")
    return scenario, state, reconstructed_raw


def _top_three(bank: list[dict]) -> list[dict]:
    return sorted(bank, key=lambda row: (tuple(row["objective"]), tuple(row["sequence"])))[:3]


def _predictability_features(canonical_state: list[float], plan: tuple[int, ...],
                             objective: list[float], maximum_horizon: int) -> list[float]:
    """Fixed-width deployable input for states with different remaining horizons."""
    if len(plan) > maximum_horizon or not plan:
        raise ValueError("plan horizon is outside the frozen feature contract")
    vectors = np.asarray([ACTION_VECTORS[int(action)] for action in plan], dtype=np.float64)
    padded = np.zeros((maximum_horizon, vectors.shape[1]), dtype=np.float64)
    padded[:len(plan)] = vectors
    active = np.zeros(maximum_horizon, dtype=np.float64)
    active[:len(plan)] = 1.0
    return (list(map(float, canonical_state)) + padded.reshape(-1).tolist()
            + active.tolist() + [len(plan) / maximum_horizon]
            + list(map(float, objective)))


def _path_is_monotone(base: tuple[int, ...], target: tuple[int, ...], score_map: dict,
                      positions: list[int]) -> bool:
    start = frozenset()
    reachable = {start}
    for depth in range(1, len(positions) + 1):
        next_reachable = set()
        for subset in itertools.combinations(positions, depth):
            current = frozenset(subset)
            current_seq = tuple(target[i] if i in current else base[i] for i in range(len(base)))
            current_obj = tuple(score_map[current_seq]["objective"])
            for pos in subset:
                parent = current - {pos}
                if parent not in reachable:
                    continue
                parent_seq = tuple(target[i] if i in parent else base[i] for i in range(len(base)))
                if current_obj <= tuple(score_map[parent_seq]["objective"]):
                    next_reachable.add(current)
                    break
        reachable = next_reachable
        if not reachable:
            return False
    return frozenset(positions) in reachable


def _state_record(root: Path, cfg: dict, episode: dict, decision: dict,
                  eval_counter: list[int]) -> dict:
    scenario, state, raw52 = _reconstruct_state(episode, decision)
    horizon = int(decision["horizon"])
    query = int(decision["query_seed"])
    previous = tuple(int(v) for v in decision["previous_plan"])
    base = p20.shifted_plan(previous, horizon)
    if base is None or len(base) != horizon:
        raise RuntimeError("shifted starting plan is missing or has incorrect horizon")
    b32_rows = decision["candidate_bank"]
    b32 = {tuple(int(v) for v in row["sequence"]): row for row in b32_rows}
    if len(b32) != len(b32_rows):
        raise RuntimeError("duplicate plan sequence in saved B32 bank")
    b24_bank, _ = p24.shifted_bank(horizon, 24, query, previous,
                                    int(p28.config(root)["planning"]["attempt_cap"]))
    b24 = {tuple(int(v) for v in seq) for seq in b24_bank}
    b32_set = set(b32)
    if not b24.issubset(b32_set) or base not in b24 or base not in b32:
        raise RuntimeError("reconstructed B24/B32 bank does not contain shifted plan/nesting")
    # Check source-bank parity against exact Phase20-generated identities and objectives.
    b32_generated, _ = p24.shifted_bank(horizon, 32, query, previous,
                                         int(p28.config(root)["planning"]["attempt_cap"]))
    if set(tuple(int(v) for v in seq) for seq in b32_generated) != b32_set:
        raise RuntimeError("saved fixed-B32 bank differs from regenerated sequence identities")
    base_row = b32[base]
    top = _top_three(b32_rows)
    score_map = {seq: row for seq, row in b32.items()}
    newly_scored = {}
    target_info = []
    path_nodes: dict[tuple[int, ...], dict] = {base: base_row}
    positive_edit_ids: set[int] = set()
    monotone_targets = 0
    for target_row in top:
        target = tuple(int(v) for v in target_row["sequence"])
        positions = [i for i, (a, b) in enumerate(zip(base, target, strict=True)) if a != b]
        depth = len(positions)
        monotone = False
        if depth <= int(cfg["planning"]["max_refinement_depth"]):
            # Evaluate each unique nonterminal target-directed partial edit set once.
            for count in range(1, depth):
                for subset in itertools.combinations(positions, count):
                    plan = tuple(target[i] if i in subset else base[i] for i in range(horizon))
                    if plan not in score_map and plan not in newly_scored:
                        evaluation = p16d.evaluate_sequence(state, scenario, plan)
                        row = _candidate_row(evaluation)
                        newly_scored[plan] = row
                        score_map[plan] = row
                        eval_counter[0] += 1
                        if eval_counter[0] > int(cfg["planning"]["maximum_new_exact_sequence_evaluations"]):
                            raise RuntimeError("Phase29 exact intermediate evaluation cap exceeded")
                    path_nodes[plan] = score_map.get(plan, newly_scored.get(plan))
            path_nodes[target] = target_row
            monotone = _path_is_monotone(base, target, score_map, positions)
            if monotone:
                monotone_targets += 1
            for pos in positions:
                positive_edit_ids.add(pos * 9 + target[pos])
        target_info.append({"sequence": list(target), "objective": list(target_row["objective"]),
                            "category": target_row["category"], "hamming_depth": depth,
                            "reachable_within_four": depth <= int(cfg["planning"]["max_refinement_depth"]),
                            "monotone_exact_objective_path": monotone})
    best_b24 = min((b32[seq] for seq in b24),
                   key=lambda row: (tuple(row["objective"]), tuple(row["sequence"])))
    base_obj, best32_obj = tuple(base_row["objective"]), tuple(top[0]["objective"])
    meaningful = any(tuple(row["objective"])[0] < base_obj[0]
                     or (tuple(row["objective"])[0] == 0 and base_obj[0] != 0) for row in top)
    best_after = []
    for max_depth in range(int(cfg["planning"]["max_refinement_depth"]) + 1):
        eligible_nodes = [(seq, row) for seq, row in path_nodes.items()
                          if sum(a != b for a, b in zip(base, seq, strict=True)) <= max_depth]
        best_seq, best_row = min(eligible_nodes,
                                 key=lambda pair: (tuple(pair[1]["objective"]), pair[0]))
        top3_reached = any(tuple(t["sequence"]) == best_seq for t in top)
        best_after.append({"max_refinement_depth": max_depth, "sequence": list(best_seq),
                           "objective": list(best_row["objective"]), "category": best_row["category"],
                           "first_action": int(best_seq[0]), "top_three_reached": top3_reached,
                           "category_better_than_shifted": tuple(best_row["objective"])[0] < base_obj[0],
                           "scalar_improved_same_category": (
                               tuple(best_row["objective"])[0] == base_obj[0]
                               and tuple(best_row["objective"])[1] < base_obj[1])})
    shortest_depth = min((item["hamming_depth"] for item in target_info), default=None)
    all_eval_rows = list(b32_rows) + list(newly_scored.values())
    b32_winner = top[0]
    min_objective = min(tuple(row["objective"]) for row in all_eval_rows)
    canonical_state = p16c.make_state_representation(scenario, state, "canonical").astype(float).tolist()
    plan_physical = np.asarray([ACTION_VECTORS[int(action)] for action in base],
                               dtype=np.float64).reshape(-1).tolist()
    prediction_input = _predictability_features(
        canonical_state, base, list(base_obj), int(cfg["planning"]["maximum_horizon"]))
    valid_edit_ids = [pos * 9 + action for pos in range(horizon) for action in range(9) if action != base[pos]]
    base_in_top3 = base in {tuple(int(v) for v in row["sequence"]) for row in top}
    return {"schema": "phase29-state-v1", "seed": int(episode["seed"]),
        "layout_sha256": episode["layout_sha256"], "replica": int(episode["replica"]),
        "decision_index": int(decision["decision_index"]), "query_seed": query, "horizon": horizon,
        "fold": int.from_bytes(hashlib.sha256(
            f"{episode['layout_sha256']}|phase29-fold-v1".encode()).digest()[:8], "big") % 5,
        "canonical_state": canonical_state, "shifted_plan": list(base),
        "shifted_plan_physical": plan_physical, "shifted_objective": list(base_obj),
        "b24_best_objective": list(best_b24["objective"]), "b24_best_sequence": list(best_b24["sequence"]),
        "b32_best_sequence": list(b32_winner["sequence"]), "b32_best_objective": list(best32_obj),
        "b32_best_category": b32_winner["category"], "b32_top_three": target_info,
        "shifted_in_b32_top_three": base_in_top3, "shortest_b32_top_three_depth": shortest_depth,
        "reaches_b32_top_three_by_depth": [shortest_depth is not None and shortest_depth <= k for k in range(5)],
        "meaningful_improvement_state": bool(meaningful),
        "b32_changes_first_action": int(base[0]) != int(b32_winner["sequence"][0]),
        "b32_improves_primary_category": int(best32_obj[0]) < int(base_obj[0]),
        "b32_adds_safe_success_vs_shifted": int(best32_obj[0]) == 0 and int(base_obj[0]) != 0,
        "b32_best_collision_to_safe": int(base_obj[0]) == 3 and int(best32_obj[0]) < 3,
        "monotone_top_three_target_count": monotone_targets,
        "best_after_refinement_depth": best_after,
        "best_target_directed_objective_beats_b32": tuple(min_objective) < tuple(best32_obj),
        "target_path_oracle_exact_evaluations": shortest_depth,
        "unique_intermediate_exact_evaluations_new_to_b32": len(newly_scored),
        "new_exact_candidates": [{**row, "sequence": list(seq)} for seq, row in sorted(newly_scored.items())],
        "path_node_count": len(path_nodes), "positive_edit_ids": sorted(positive_edit_ids),
        "valid_edit_ids": valid_edit_ids, "predictability_input": prediction_input,
        "predictability_eligible": bool(meaningful and positive_edit_ids and not base_in_top3),
        "source_state_feature_parity": True, "reserve_access": False, "test_access": False,
        "d168_access": False}


def _load_state_rows(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(root: Path = ROOT) -> dict:
    cfg, lock, lock_hash = verify(root)
    out = root / cfg["outputs"]["run_dir"]
    if out.exists():
        raise FileExistsError("Phase29 output exists; run is locked non-resumable")
    parent_audit = _audit_phase28_saved_records(root, cfg)
    if not parent_audit.get("audit_pass") or parent_audit.get("episodes") != 9000:
        raise RuntimeError("Phase28 source audit failed before Phase29 replay")
    selected = _selected_source_states(root, cfg)
    out.mkdir(parents=True)
    started = time.perf_counter()
    eval_counter = [0]
    rows_path = root / cfg["outputs"]["states"]
    with gzip.open(rows_path, "wt", encoding="utf-8", newline="\n") as stream:
        for i, (episode, decision) in enumerate(selected, start=1):
            row = _state_record(root, cfg, episode, decision, eval_counter)
            stream.write(_canonical(row) + "\n")
            if i % 100 == 0:
                print(f"Phase29 exact refinement states {i}/{len(selected)}; "
                      f"new exact evaluations {eval_counter[0]}; {time.perf_counter()-started:.1f}s",
                      flush=True)
    manifest = {"schema": "phase29-iterative-refinement-run-manifest-v1",
        "execution_lock_sha256": lock_hash, "parent_audit": parent_audit,
        "selected_states": len(selected), "new_exact_intermediate_evaluations": eval_counter[0],
        "state_rows_sha256": sha256(rows_path), "runtime_seconds": time.perf_counter()-started,
        "python": platform.python_version(), "numpy": np.__version__,
        "source_commit": _git(root, "rev-parse", "HEAD"), "new_episodes": False,
        "training": False, "closed_loop": False, "reserve_access": False,
        "test_access": False, "d168_access": False}
    p18.write_new(root / cfg["outputs"]["manifest"], manifest)
    return manifest


def _sse(sums: np.ndarray, squares: np.ndarray, count: int) -> float:
    return float(np.maximum(squares - sums * sums / count, 0.0).sum())


def _build_tree(x: np.ndarray, y: np.ndarray, indices: np.ndarray,
                depth: int = 0, max_depth: int = 5, min_leaf: int = 20):
    target = y[indices]
    leaf = target.mean(axis=0)
    if depth >= max_depth or len(indices) < 2 * min_leaf:
        return (None, leaf)
    best_gain, best_feature, best_threshold = 0.0, None, None
    best_left = best_right = None
    parent = _sse(target.sum(axis=0), np.square(target).sum(axis=0), len(indices))
    for feature in range(x.shape[1]):
        order_local = np.argsort(x[indices, feature], kind="mergesort")
        ordered_idx = indices[order_local]
        values = x[ordered_idx, feature]
        split_points = np.flatnonzero(values[:-1] < values[1:]) + 1
        split_points = split_points[(split_points >= min_leaf) & (split_points <= len(indices)-min_leaf)]
        if not len(split_points):
            continue
        ordered_y = y[ordered_idx]
        cumulative = np.cumsum(ordered_y, axis=0)
        cumulative_sq = np.cumsum(np.square(ordered_y), axis=0)
        total_sum, total_sq = cumulative[-1], cumulative_sq[-1]
        left_sum, left_sq = cumulative[split_points - 1], cumulative_sq[split_points - 1]
        right_sum, right_sq = total_sum - left_sum, total_sq - left_sq
        right_n = len(indices) - split_points
        gains = parent - (np.maximum(left_sq - left_sum * left_sum / split_points[:, None], 0).sum(axis=1)
                          + np.maximum(right_sq - right_sum * right_sum / right_n[:, None], 0).sum(axis=1))
        local = int(np.argmax(gains))
        gain = float(gains[local])
        threshold = float((values[split_points[local]-1] + values[split_points[local]]) / 2.0)
        if gain > best_gain + 1e-12:
            best_gain, best_feature, best_threshold = gain, feature, threshold
            mask = x[indices, feature] <= threshold
            best_left, best_right = indices[mask], indices[~mask]
    if best_feature is None or best_left is None or best_right is None:
        return (None, leaf)
    left = _build_tree(x, y, best_left, depth+1, max_depth, min_leaf)
    right = _build_tree(x, y, best_right, depth+1, max_depth, min_leaf)
    return ((best_feature, best_threshold, left, right), leaf)


def _tree_predict(tree, x: np.ndarray) -> np.ndarray:
    node, leaf = tree
    if node is None:
        return np.repeat(leaf[None, :], len(x), axis=0)
    feature, threshold, left, right = node
    result = np.empty((len(x), len(leaf)), dtype=np.float64)
    mask = x[:, feature] <= threshold
    if mask.any():
        result[mask] = _tree_predict(left, x[mask])
    if (~mask).any():
        result[~mask] = _tree_predict(right, x[~mask])
    return result


def _topk_metrics(scores: np.ndarray, positives: set[int], valid: list[int]) -> tuple[float, float, float]:
    if not positives or not valid:
        return float("nan"), float("nan"), float("nan")
    ranked = sorted(valid, key=lambda idx: (-float(scores[idx]), idx))
    ranks = [rank for rank, idx in enumerate(ranked, start=1) if idx in positives]
    return float(ranks[0] == 1), float(ranks[0] <= 4), float(1.0 / ranks[0])


def _predictability(rows: list[dict], cfg: dict) -> dict:
    eligible = [row for row in rows if row["predictability_eligible"]]
    if len(eligible) < 20 or len({row["layout_sha256"] for row in eligible}) < 5:
        return {"eligible_states": len(eligible), "status": "insufficient_groups", "models": {}}
    x = np.asarray([row["predictability_input"] for row in eligible], dtype=np.float64)
    y = np.zeros((len(eligible), 72), dtype=np.float64)
    for i, row in enumerate(eligible):
        y[i, row["positive_edit_ids"]] = 1.0
    folds = np.asarray([int(row["fold"]) for row in eligible], dtype=np.int64)
    predictions = {name: [] for name in ("global_edit_prior", "nearest_neighbor", "ridge_multioutput", "cart_multioutput")}
    for fold in range(int(cfg["source"]["layout_group_folds"])):
        train = folds != fold
        test = folds == fold
        if not train.any() or not test.any():
            raise RuntimeError("Phase29 fold has an empty train or held-out partition")
        xmean = x[train].mean(axis=0)
        xstd = x[train].std(axis=0)
        xstd[xstd < 1e-8] = 1.0
        xt = (x[train] - xmean) / xstd
        xv = (x[test] - xmean) / xstd
        yt = y[train]
        prior = yt.mean(axis=0)
        # Fixed 1-NN transfers a set-valued edit target from the nearest training state.
        for local, row_index in enumerate(np.flatnonzero(test)):
            nearest = int(np.argmin(np.mean(np.square(xt - xv[local]), axis=1)))
            nearest_scores = yt[nearest]
            predictions["nearest_neighbor"].append((row_index, nearest_scores))
        # Multi-output ridge with a free intercept and fixed alpha.
        augmented = np.column_stack((np.ones(len(xt)), xt))
        penalty = np.eye(augmented.shape[1], dtype=np.float64)
        penalty[0, 0] = 0.0
        ridge_w = np.linalg.solve(augmented.T @ augmented + float(cfg["predictability"]["models"]["ridge_multioutput"]["alpha"]) * penalty,
                                  augmented.T @ yt)
        ridge_scores = np.column_stack((np.ones(len(xv)), xv)) @ ridge_w
        # Fixed deterministic CART multi-output regression.
        tree_cfg = cfg["predictability"]["models"]["decision_tree"]
        tree = _build_tree(xt, yt, np.arange(len(xt)), max_depth=int(tree_cfg["max_depth"]),
                           min_leaf=int(tree_cfg["min_samples_leaf"]))
        tree_scores = _tree_predict(tree, xv)
        for local, row_index in enumerate(np.flatnonzero(test)):
            predictions["global_edit_prior"].append((row_index, prior.copy()))
            predictions["ridge_multioutput"].append((row_index, ridge_scores[local].copy()))
            predictions["cart_multioutput"].append((row_index, tree_scores[local].copy()))
    # Make the paired layout bootstrap comparison explicit for each diagnostic model.
    # Global-prior values are computed separately as the fixed reference on the same folds.
    global_metrics = {}
    for name, items in predictions.items():
        vals = []
        for index, scores in items:
            row = eligible[index]
            vals.append({"layout_sha256": row["layout_sha256"],
                         "metrics": {name: dict(zip(("hit_at_1", "hit_at_4", "reciprocal_rank"),
                             _topk_metrics(scores, set(row["positive_edit_ids"]), row["valid_edit_ids"]), strict=True))}})
        global_metrics[name] = {metric: float(np.mean([v["metrics"][name][metric] for v in vals]))
                                for metric in ("hit_at_1", "hit_at_4", "reciprocal_rank")}
        if name == "global_edit_prior":
            continue
        paired = []
        by_layout: dict[str, list[float]] = {}
        # Recover matching global scores using row-index keyed predictions.
        prior_by_index = dict(predictions["global_edit_prior"])
        for index, scores in items:
            row = eligible[index]
            h4_model = _topk_metrics(scores, set(row["positive_edit_ids"]), row["valid_edit_ids"])[1]
            h4_prior = _topk_metrics(prior_by_index[index], set(row["positive_edit_ids"]), row["valid_edit_ids"])[1]
            by_layout.setdefault(row["layout_sha256"], []).append(float(h4_model-h4_prior))
        groups = sorted(by_layout)
        difference = np.asarray([np.mean(by_layout[group]) for group in groups], dtype=np.float64)
        analysis = cfg["analysis"]
        rng = np.random.default_rng(int(analysis["bootstrap_seed"]) + (0 if name == "ridge_multioutput" else 1))
        sampled = difference[rng.integers(0, len(difference), size=(int(analysis["bootstrap_draws"]), len(difference)))].mean(axis=1)
        paired.append({"layouts": len(groups), "mean_hit4_difference_vs_global_prior": float(difference.mean()),
                       "ci95_lower": float(np.quantile(sampled, .025)),
                       "ci95_upper": float(np.quantile(sampled, .975))})
        global_metrics[name]["paired_vs_global_prior"] = paired[0]
    primary = global_metrics["ridge_multioutput"]
    primary_ci = primary["paired_vs_global_prior"]
    gate = {"ridge_hit4_at_least_25pct": primary["hit_at_4"] >= float(cfg["analysis"]["heldout_hit_at_4_at_least"]),
            "ridge_hit4_improvement_lower_gt_zero": primary_ci["ci95_lower"] >
                float(cfg["analysis"]["hit_at_4_paired_layout_bootstrap_lower_gt"])}
    return {"eligible_states": len(eligible), "heldout_layouts": len({row["layout_sha256"] for row in eligible}),
            "models": global_metrics, "primary_ridge_gate": gate, "primary_ridge_pass": all(gate.values())}


def score_rows(rows: list[dict], cfg: dict) -> dict:
    if not rows:
        raise ValueError("Phase29 state rows cannot be empty")
    if any(not row.get("source_state_feature_parity") or row.get("reserve_access")
           or row.get("test_access") or row.get("d168_access") for row in rows):
        raise RuntimeError("Phase29 state row has failed parity or sealed-data flag")
    n = len(rows)
    meaningful = [row for row in rows if row["meaningful_improvement_state"]]
    nontrivial = [row for row in rows if not row["shifted_in_b32_top_three"]]
    reached = [row for row in meaningful if row["shortest_b32_top_three_depth"] is not None
               and row["shortest_b32_top_three_depth"] <= 4]
    category_gain_by_2 = [row for row in meaningful if any(
        item["max_refinement_depth"] <= 2 and item["category_better_than_shifted"]
        for item in row["best_after_refinement_depth"])]
    meaningful_depths = [int(row["shortest_b32_top_three_depth"]) for row in meaningful
                         if row["shortest_b32_top_three_depth"] is not None]
    all_depths = [int(row["shortest_b32_top_three_depth"]) for row in rows
                  if row["shortest_b32_top_three_depth"] is not None]
    total_new = sum(int(row["unique_intermediate_exact_evaluations_new_to_b32"]) for row in rows)
    max_k = int(cfg["planning"]["max_refinement_depth"])
    depth_summary = []
    for k in range(max_k+1):
        chosen = [row["best_after_refinement_depth"][k] for row in rows]
        top_three_count = sum(bool(item["top_three_reached"]) for item in chosen)
        meaningful_chosen = [row["best_after_refinement_depth"][k] for row in meaningful]
        meaningful_top_three_count = sum(bool(item["top_three_reached"]) for item in meaningful_chosen)
        depth_summary.append({"depth": k,
            "states_reaching_b32_top_three": top_three_count,
            "fraction_states_reaching_b32_top_three": top_three_count/n,
            "meaningful_states_reaching_b32_top_three": meaningful_top_three_count,
            "fraction_meaningful_reaching_b32_top_three": (
                meaningful_top_three_count/len(meaningful) if meaningful else 0.0),
            "states_with_category_improvement": sum(bool(item["category_better_than_shifted"]) for item in chosen),
            "states_with_same_category_scalar_improvement": sum(bool(item["scalar_improved_same_category"]) for item in chosen),
            "states_with_safe_success_best": sum(item["category"] == "success" for item in chosen),
            "collision_shifted_to_noncollision_best": sum(
                row["shifted_objective"][0] == 3 and item["objective"][0] < 3
                for row, item in zip(rows, chosen, strict=True)),
            "first_action_changes": sum(int(item["first_action"]) != int(row["shifted_plan"][0])
                                         for row, item in zip(rows, chosen, strict=True)),
            "mean_best_category": float(np.mean([item["objective"][0] for item in chosen]))})
    predictability = _predictability(rows, cfg)
    meaningful_fraction = len(meaningful)/n
    reachable_fraction = len(reached)/len(meaningful) if meaningful else 0.0
    early_category_fraction = len(category_gain_by_2)/len(meaningful) if meaningful else 0.0
    analysis = cfg["analysis"]
    gates = {"meaningful_target_reachability": reachable_fraction >= float(analysis["depth_gate_fraction"]),
        "early_category_improvement": early_category_fraction >= float(analysis["category_improvement_by_depth_two_fraction"]),
        "predictable_next_edit": bool(predictability.get("primary_ridge_pass", False))}
    quality_metrics = {
        "selected_states": n,
        "layout_groups": len({row["layout_sha256"] for row in rows}),
        "meaningful_improvement_states": len(meaningful),
        "meaningful_improvement_fraction": meaningful_fraction,
        "shifted_plan_already_in_b32_top_three": sum(row["shifted_in_b32_top_three"] for row in rows),
        "b32_best_changes_first_action": sum(row["b32_changes_first_action"] for row in rows),
        "b32_improves_primary_category": sum(row["b32_improves_primary_category"] for row in rows),
        "b32_adds_safe_success": sum(row["b32_adds_safe_success_vs_shifted"] for row in rows),
        "b32_best_collision_to_safe": sum(row["b32_best_collision_to_safe"] for row in rows),
        "nontrivial_state_count": len(nontrivial),
        "meaningful_top_three_reachable_within_four": len(reached),
        "meaningful_top_three_reachable_fraction": reachable_fraction,
        "meaningful_category_improved_by_two_edits": len(category_gain_by_2),
        "meaningful_category_improved_by_two_fraction": early_category_fraction,
        "median_minimum_edit_evaluations_meaningful": float(np.median(meaningful_depths)) if meaningful_depths else None,
        "p90_minimum_edit_evaluations_meaningful": float(np.quantile(meaningful_depths, .9)) if meaningful_depths else None,
        "median_minimum_edit_evaluations_all_selected": float(np.median(all_depths)) if all_depths else None,
        "p90_minimum_edit_evaluations_all_selected": float(np.quantile(all_depths, .9)) if all_depths else None,
        "meaningful_reachable_within_four_exact_evaluations_fraction": reachable_fraction,
        "b32_extension_candidate_reference": int(analysis["refinement_eval_reference"]),
        "exact_intermediate_evaluations_new_to_b32": total_new,
        "exact_intermediate_evaluations_mean_per_state": total_new/n,
        "depth_frontier": depth_summary,
        "monotone_top_three_target_paths": sum(row["monotone_top_three_target_count"] for row in rows),
        "mean_b24_vs_shifted_category_delta": float(np.mean([
            int(row["b24_best_objective"][0])-int(row["shifted_objective"][0]) for row in rows])),
    }
    return {"schema": "phase29-iterative-refinement-result-v1", "status": "complete",
        "decision": "feasibility_gate_pass" if all(gates.values()) else "feasibility_gate_fail",
        "interpretation": "exploratory_reused_phase28_development_data",
        "metrics": quality_metrics, "gates": gates, "all_gates_pass": all(gates.values()),
        "predictability": predictability, "selected_states": n,
        "new_exact_intermediate_evaluations": total_new,
        "new_episodes": False, "training": False, "closed_loop": False,
        "reserve_access": False, "test_access": False, "d168_access": False}


def score(root: Path = ROOT) -> dict:
    cfg, _, lock_hash = verify(root)
    out = root / cfg["outputs"]["run_dir"]
    manifest_path = root / cfg["outputs"]["manifest"]
    rows_path = root / cfg["outputs"]["states"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["execution_lock_sha256"] != lock_hash or manifest["state_rows_sha256"] != sha256(rows_path):
        raise RuntimeError("Phase29 state rows/manifest do not match lock")
    rows = _load_state_rows(rows_path)
    result = score_rows(rows, cfg)
    result.update({"execution_lock_sha256": lock_hash, "state_rows_sha256": sha256(rows_path),
                   "input_records_sha256": manifest["parent_audit"]["records_sha256"]})
    p18.write_new(root / cfg["outputs"]["result"], result)
    score_manifest = {"schema": "phase29-iterative-refinement-score-manifest-v1",
        "execution_lock_sha256": lock_hash, "state_rows_sha256": sha256(rows_path),
        "result_sha256": sha256(root / cfg["outputs"]["result"]),
        "scorer_sha256": sha256(Path(__file__).resolve()),
        "scorer_commit": _git(root, "rev-parse", "HEAD"),
        "new_exact_evaluations_during_scoring": 0, "new_episodes": False,
        "reserve_access": False, "test_access": False, "d168_access": False}
    p18.write_new(out / "score_manifest.json", score_manifest)
    return result


def audit(root: Path = ROOT) -> dict:
    cfg, _, lock_hash = verify(root)
    out = root / cfg["outputs"]["run_dir"]
    manifest = json.loads((root / cfg["outputs"]["manifest"]).read_text(encoding="utf-8"))
    score_manifest = json.loads((out / "score_manifest.json").read_text(encoding="utf-8"))
    result_path = root / cfg["outputs"]["result"]
    if sha256(root / cfg["outputs"]["states"]) != manifest["state_rows_sha256"]:
        raise RuntimeError("Phase29 state artifact hash mismatch")
    if score_manifest["execution_lock_sha256"] != lock_hash or score_manifest["result_sha256"] != sha256(result_path):
        raise RuntimeError("Phase29 result artifact provenance mismatch")
    if score_manifest["new_exact_evaluations_during_scoring"] != 0:
        raise RuntimeError("Phase29 scoring unexpectedly evaluated exact candidates")
    rows = _load_state_rows(root / cfg["outputs"]["states"])
    if len(rows) != int(manifest["selected_states"]):
        raise RuntimeError("Phase29 state row count differs from run manifest")
    recomputed = score_rows(rows, cfg)
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    for key, value in recomputed.items():
        if saved.get(key) != value:
            raise RuntimeError(f"Phase29 result does not reproduce: {key}")
    return {"audit_pass": True, "selected_states": len(rows),
        "new_exact_evaluations_during_audit": 0, "new_episodes": False,
        "reserve_access": False, "test_access": False, "d168_access": False,
        "result_sha256": sha256(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify", "run", "score", "audit"))
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command == "verify":
        cfg, lock, lock_hash = verify()
        result = {"verified": True, "lock_sha256": lock_hash,
                  "parent": lock["parent"], "state_cap": cfg["planning"]["maximum_selected_states"]}
    elif args.command == "run":
        result = run()
    elif args.command == "score":
        result = score()
    else:
        result = audit()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
