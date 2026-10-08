"""Phase 28: planner-native sequential exact-search stopping."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import yaml

from consequence_lab import phase16d_compute_frontier as p16d
from consequence_lab import phase17_proposal_guidance as p17
from consequence_lab import phase18_proposal_controls as p18
from consequence_lab import phase20_exact_plan_warmstart as p20
from consequence_lab import phase24_search_budget_profile as p24
from consequence_lab import phase26_search_feedback as p26
from consequence_lab import phase27a_search_allocation as p27a
from consequence_lab import phase27b_adaptive_exact_search as p27b
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase12_teacher import state_features

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase28_sequential_exact_search_v1.yaml"
SOURCE = "src/consequence_lab/phase28_sequential_exact_search.py"
TESTS = "tests/test_phase28_sequential_exact_search.py"
PROTOCOL = "reports/phase28_sequential_exact_search_protocol.md"
AMENDMENT = "reports/phase28_prelock_cutoff_salt_correction.md"
LOCK_SCHEMA = "phase28-sequential-exact-search-lock-v1"
ARMS = ("fixed_b24", "fixed_b26", "fixed_b32", "random_adaptive_b24_plus_8",
        "phase27_learned_adaptive_b24_plus_8", "planner_native_adaptive_b24_plus_8")
SEALED_TOKENS = ("reserve", "test", "d168")
CATEGORIES = {0: "success", 1: "active", 2: "timeout", 3: "collision"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config(root: Path = ROOT) -> dict:
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _paths(root: Path, cfg: dict) -> list[str]:
    paths = [CONFIG, PROTOCOL, AMENDMENT, SOURCE, TESTS,
             "src/consequence_lab/phase13.py", "src/consequence_lab/phase16c_prior_guided.py",
             "src/consequence_lab/phase16d_compute_frontier.py",
             "src/consequence_lab/phase17_proposal_guidance.py",
             "src/consequence_lab/phase18_proposal_controls.py",
             "src/consequence_lab/phase20_exact_plan_warmstart.py",
             "src/consequence_lab/phase24_search_budget_profile.py",
             "src/consequence_lab/phase26_search_feedback.py",
             "src/consequence_lab/phase27a_search_allocation.py",
             "src/consequence_lab/phase27b_adaptive_exact_search.py",
             "src/consequence_lab/corridor_tasks.py", "src/consequence_lab/inertial_corridor.py",
             "src/consequence_lab/phase12_teacher.py",
             "configs/phase27b_adaptive_exact_search_v2.yaml",
             cfg["parent"]["phase27b_lock"], cfg["parent"]["phase27b_records"],
             cfg["parent"]["phase27b_result"], cfg["parent"]["phase27b_run_manifest"],
             cfg["parent"]["phase27b_result_manifest"], cfg["parent"]["phase27a_model"],
             cfg["parent"]["phase27a_result"], cfg["parent"]["phase27a_manifest"],
             cfg["parent"]["phase27a_lock"]]
    return sorted(set(p for p in paths if (root / p).is_file()))


def validate_config(cfg: dict) -> None:
    if cfg.get("schema") != "phase28-sequential-exact-search-v1":
        raise ValueError("wrong Phase28 schema")
    if cfg.get("protocol") != PROTOCOL or cfg.get("amendment") != AMENDMENT:
        raise ValueError("Phase28 protocol/amendment paths changed")
    if cfg["authorization"].get("scope") != "phase28-fresh-development-evaluation-only":
        raise RuntimeError("Phase28 authorization scope is not fresh-development-only")
    block, plan, execution = cfg["block"], cfg["planning"], cfg["execution"]
    if int(block["seed_start"]) != 211000 or int(block["layout_count"]) != 500:
        raise ValueError("Phase28 fresh block changed")
    if list(block["replicas"]) != [0, 1, 2]:
        raise ValueError("Phase28 paired replica roster changed")
    if (plan["horizon"] != 8 or plan["base_budget"] != 24
            or plan["matched_average_budget"] != 26 or plan["full_budget"] != 32
            or plan["exact_rollout_scoring"] is not True or plan["exact_final_selection"] is not True):
        raise ValueError("Phase28 planner/budget contract changed")
    if tuple(cfg["arms"]) != ARMS:
        raise ValueError("Phase28 arm roster changed")
    if (execution["authorized"] is not True or execution["resume"] is not False
            or execution["interim_scoring"] is not False or execution["training"] is not False
            or execution["reserve_test_d168_access"] is not False):
        raise RuntimeError("Phase28 execution or data-scope contract invalid")
    if not 0 < float(cfg["allocation"]["random_rate"]) < 1:
        raise ValueError("invalid random extension rate")


def parent_bindings(root: Path, cfg: dict, audit_phase27b: bool = False) -> dict[str, str]:
    parent = cfg["parent"]
    keys = ("phase27b_lock", "phase27b_records", "phase27b_result", "phase27b_run_manifest",
            "phase27b_result_manifest", "phase27a_model", "phase27a_result", "phase27a_manifest",
            "phase27a_lock")
    hashes = {}
    for key in keys:
        path = parent[key]
        actual = sha256(root / path)
        if actual != parent[f"{key}_sha256"]:
            raise RuntimeError(f"Phase28 parent hash mismatch: {path}")
        hashes[path] = actual
    if audit_phase27b:
        p27b.audit(root)
    manifest = json.loads((root / parent["phase27b_run_manifest"]).read_text(encoding="utf-8"))
    if (manifest.get("records_sha256") != hashes[parent["phase27b_records"]]
            or manifest.get("execution_lock_sha256") != hashes[parent["phase27b_lock"]]
            or manifest.get("reserve_access") is not False
            or manifest.get("test_access") is not False
            or manifest.get("d168_access") is not False):
        raise RuntimeError("Phase27B parent manifest integrity/scope mismatch")
    model = json.loads((root / parent["phase27a_model"]).read_text(encoding="utf-8"))
    if not np.isclose(float(model["threshold_oof"]),
                      float(config(root)["allocation"]["phase27_learned_probability_threshold"]),
                      rtol=0, atol=1e-14):
        raise RuntimeError("Phase27A learned threshold does not match frozen Phase28 config")
    return hashes


def _native_tie(layout_sha: str, replica: int, decision: int, salt: str) -> int:
    payload = f"phase28|{salt}|{layout_sha}|{int(replica)}|{int(decision)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def planner_native_trigger(no_safe_success: bool, gap: float, tie: int, cfg: dict) -> bool:
    cutoff = (float(cfg["allocation"]["native_gap_cutoff"]),
              int(cfg["allocation"]["native_tie_hash_cutoff"]))
    return bool(no_safe_success and (float(gap), int(tie)) <= cutoff)


def _order_stat_cutoff(rows: list[dict], target_rate: float) -> dict:
    """Choose a reproducible overall-rate cutoff, breaking tied gaps by identity hash."""
    if not rows or not 0 < target_rate < 1:
        raise ValueError("cutoff selection requires rows and a rate in (0,1)")
    ranked = sorted((row for row in rows if row["no_success"]),
                    key=lambda row: (float(row["gap"]), int(row["tie"])))
    target = int(np.ceil(target_rate * len(rows)))
    if len(ranked) < target:
        raise RuntimeError("native predicate has insufficient historical support for target activation")
    selected = ranked[target - 1]
    count = sum(bool(row["no_success"])
        and (float(row["gap"]), int(row["tie"])) <= (float(selected["gap"]), int(selected["tie"]))
        for row in rows)
    return {"target_count": target, "gap_cutoff": float(selected["gap"]),
            "tie_hash_cutoff": int(selected["tie"]), "activated_count": int(count)}


def _derive_native_cutoff(root: Path, cfg: dict) -> dict:
    """Derive a fixed 25% score cutoff from Phase27B consumed fixed-B32 traces only."""
    rows = []
    path = root / cfg["parent"]["phase27b_records"]
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            episode = json.loads(line)
            if episode.get("method") != "fixed_b32":
                continue
            for d in episode["decision_diagnostics"]:
                diag = d.get("allocation_diagnostic")
                if not diag or not diag.get("extension_available"):
                    continue
                features = diag["features"]
                if len(features) != 73 or not np.isfinite(np.asarray(features, dtype=float)).all():
                    raise RuntimeError("Phase27B prefix feature contract invalid during native cutoff audit")
                gap, has_no_success = float(features[-4]), float(features[60]) == 0.0
                tie = _native_tie(episode["layout_sha256"], int(episode["replica"]),
                                  int(d["decision_index"]), cfg["allocation"]["native_tie_hash_salt"])
                rows.append({"gap": gap, "no_success": has_no_success, "tie": tie})
    if len(rows) != 30089:
        raise RuntimeError(f"expected 30,089 eligible Phase27B fixed-B32 decisions, found {len(rows)}")
    cutoff = _order_stat_cutoff(rows, float(cfg["allocation"]["native_nominal_activation_rate"]))
    return {"method": "no_b24_safe_success_then_low_same_category_gap_order_statistic",
            "source_records": cfg["parent"]["phase27b_records"],
            "source_method": "fixed_b32", "historical_decisions": len(rows),
            "eligible_no_success_decisions": sum(bool(row["no_success"]) for row in rows),
            "target_count": cutoff["target_count"],
            "nominal_rate": float(cfg["allocation"]["native_nominal_activation_rate"]),
            "observed_rate": cutoff["activated_count"] / len(rows),
            "gap_cutoff": cutoff["gap_cutoff"],
            "tie_hash_cutoff": cutoff["tie_hash_cutoff"],
            "tie_hash_salt": cfg["allocation"]["native_tie_hash_salt"]}


def _safe_layout_audit(root: Path, cfg: dict) -> dict:
    return p27b.safe_layout_identity_audit(int(cfg["block"]["seed_start"]),
                                           int(cfg["block"]["layout_count"]), root)


def _execution_order(audit: dict, cfg: dict) -> list[dict]:
    rows = []
    arms = [{"method": method, "replica": int(replica)}
            for method in ARMS for replica in cfg["block"]["replicas"]]
    for layout in audit["layouts"]:
        schedule = [dict(arm) for arm in arms]
        random.Random(int(cfg["analysis"]["schedule_seed"]) + int(layout["seed"])).shuffle(schedule)
        for local, arm in enumerate(schedule):
            rows.append({**layout, **arm, "global_schedule_index": len(rows),
                         "local_order_index": local})
    if len(rows) != int(cfg["block"]["layout_count"]) * len(ARMS) * len(cfg["block"]["replicas"]):
        raise RuntimeError("Phase28 schedule size mismatch")
    if len({(r["seed"], r["method"], r["replica"]) for r in rows}) != len(rows):
        raise RuntimeError("Phase28 schedule contains duplicate episode identities")
    return rows


def freeze(root: Path = ROOT) -> dict:
    cfg = config(root)
    validate_config(cfg)
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("commit Phase28 protocol/config/source/tests and vault updates before freeze")
    out = root / cfg["outputs"]["run_dir"]
    lock_path = root / cfg["outputs"]["execution_lock"]
    if out.exists() or lock_path.exists():
        raise FileExistsError("Phase28 is append-only; refusing to overwrite an audit, schedule, run, or lock")
    parents = parent_bindings(root, cfg, audit_phase27b=True)
    audit = _safe_layout_audit(root, cfg)
    schedule = _execution_order(audit, cfg)
    cutoff = _derive_native_cutoff(root, cfg)
    for key in ("gap_cutoff", "tie_hash_cutoff"):
        if cutoff[key] != cfg["allocation"][f"native_{key}"]:
            raise RuntimeError(f"Phase28 native {key} does not reproduce the reviewed protocol")
    if abs(cutoff["observed_rate"] - 0.25) > 1 / cutoff["historical_decisions"]:
        raise RuntimeError("Phase28 native historical activation rate is not 25%")
    out.mkdir(parents=True)
    audit_path, schedule_path = root / cfg["outputs"]["layout_audit"], root / cfg["outputs"]["execution_order"]
    p18.write_new(audit_path, audit)
    p18.write_new(schedule_path, schedule)
    deps = {path: sha256(root / path) for path in _paths(root, cfg)}
    cutoff_path = {"historical_phase27b_native_cutoff": cutoff}
    lock = {"schema": LOCK_SCHEMA, "source_commit": _git(root, "rev-parse", "HEAD"),
            "parent_hashes": parents, "dependency_hashes": deps,
            "layout_audit_sha256": sha256(audit_path),
            "execution_order_sha256": sha256(schedule_path),
            "native_cutoff": cutoff_path,
            "fresh_layout_count": len(audit["layouts"]), "scheduled_episodes": len(schedule),
            "candidate_bank_parity": "B24 subset of B26 and B32 by sequence identity; exact objectives checked online",
            "reserve_test_d168_access": False}
    p18.write_new(lock_path, lock)
    return {"audit_sha256": sha256(audit_path), "schedule_sha256": sha256(schedule_path),
            "lock_path": cfg["outputs"]["execution_lock"], "lock_sha256": sha256(lock_path),
            "layouts": len(audit["layouts"]), "episodes": len(schedule),
            "historical_native_cutoff": cutoff}


def _read_lock(root: Path, cfg: dict) -> tuple[dict, list[dict], str]:
    lock_path = root / cfg["outputs"]["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != LOCK_SCHEMA:
        raise RuntimeError("Phase28 lock schema mismatch")
    lock_hash = sha256(lock_path)
    audit_path = root / cfg["outputs"]["layout_audit"]
    schedule_path = root / cfg["outputs"]["execution_order"]
    if (sha256(audit_path) != lock["layout_audit_sha256"]
            or sha256(schedule_path) != lock["execution_order_sha256"]):
        raise RuntimeError("Phase28 identity/schedule hash mismatch")
    for path, expected in lock["dependency_hashes"].items():
        if sha256(root / path) != expected:
            raise RuntimeError(f"Phase28 dependency hash mismatch: {path}")
    if parent_bindings(root, cfg, audit_phase27b=False) != lock["parent_hashes"]:
        raise RuntimeError("Phase28 parent lineage mismatch")
    cutoff = lock["native_cutoff"].get("historical_phase27b_native_cutoff", {})
    if (cutoff.get("gap_cutoff") != cfg["allocation"]["native_gap_cutoff"]
            or cutoff.get("tie_hash_cutoff") != cfg["allocation"]["native_tie_hash_cutoff"]
            or cutoff.get("source_records") != cfg["parent"]["phase27b_records"]
            or cutoff.get("observed_rate") is None):
        raise RuntimeError("Phase28 native stopping cutoff differs from locked config")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule != _execution_order(audit, cfg):
        raise RuntimeError("Phase28 execution schedule no longer reproduces")
    return lock, schedule, lock_hash


def _record(evaluation) -> dict:
    objective = tuple(evaluation.objective)
    return {"sequence": list(evaluation.sequence), "objective": list(objective),
            "category": CATEGORIES[int(objective[0])], "transitions": int(evaluation.transitions),
            "success_reached": bool(evaluation.success_reached)}


def _winner(bank: list[dict]) -> dict:
    if not bank:
        raise ValueError("candidate bank cannot be empty")
    return min(bank, key=lambda row: (tuple(row["objective"]), tuple(row["sequence"])))


def _best_category_cost(bank: list[dict]) -> tuple[int, float]:
    obj = tuple(_winner(bank)["objective"])
    return int(obj[0]), float(obj[1])


def _decision(scenario, state, previous, replica: int, method: str, layout_sha: str,
              cfg: dict, model: dict) -> tuple[object, tuple[int, ...], dict]:
    started = time.perf_counter()
    horizon = min(int(cfg["planning"]["horizon"]), scenario.spec.max_steps - state.elapsed_steps)
    query = p17.query_seed(scenario.seed, state.elapsed_steps, replica)
    cap = int(cfg["planning"]["attempt_cap"])
    base_n, match_n, full_n = (int(cfg["planning"][key]) for key in
                               ("base_budget", "matched_average_budget", "full_budget"))
    reason, score_value = "fixed_budget", None
    if method == "fixed_b26":
        actual_bank, _ = p24.shifted_bank(horizon, match_n, query, previous, cap)
        evals = [p16d.evaluate_sequence(state, scenario, seq) for seq in actual_bank]
        candidate_count = match_n
    elif method == "fixed_b32":
        actual_bank, _ = p24.shifted_bank(horizon, full_n, query, previous, cap)
        evals = [p16d.evaluate_sequence(state, scenario, seq) for seq in actual_bank]
        candidate_count = full_n
    else:
        low_bank, _ = p24.shifted_bank(horizon, base_n, query, previous, cap)
        low_bank = tuple(tuple(int(a) for a in seq) for seq in low_bank)
        low_evals = [p16d.evaluate_sequence(state, scenario, seq) for seq in low_bank]
        low_records = [_record(ev) for ev in low_evals]
        expected_low = min(base_n, 9 ** horizon)
        if len(low_bank) != expected_low or len(set(low_bank)) != expected_low:
            raise RuntimeError("Phase28 B24 bank violated exact count/uniqueness")
        b24_features = p26._candidate_features(low_records,
             state_features(scenario, state).astype(float).tolist(),
             list(previous) if previous is not None else None)
        no_success = not any(row["category"] == "success" for row in low_records)
        gap = float(b24_features[-4])
        eligible = min(full_n, 9 ** horizon) > len(low_bank)
        trigger, reason = False, "fixed_b24"
        score_value = None
        if eligible:
            if method == "fixed_b24":
                reason = "fixed_b24"
            elif method == "random_adaptive_b24_plus_8":
                tie = _native_tie(layout_sha, replica, state.elapsed_steps,
                                  cfg["allocation"]["random_hash_salt"])
                trigger = tie < int(float(cfg["allocation"]["random_rate"]) * (2**64))
                score_value, reason = tie / 2**64, "random_extend" if trigger else "random_stop"
            elif method == "phase27_learned_adaptive_b24_plus_8":
                probability = float(p27a.predict(model, np.asarray(b24_features, dtype=float)[None, :])[0])
                trigger = probability >= float(cfg["allocation"]["phase27_learned_probability_threshold"])
                score_value, reason = probability, "phase27_learned_extend" if trigger else "phase27_learned_stop"
            elif method == "planner_native_adaptive_b24_plus_8":
                tie = _native_tie(layout_sha, replica, state.elapsed_steps,
                                  cfg["allocation"]["native_tie_hash_salt"])
                trigger = planner_native_trigger(no_success, gap, tie, cfg)
                score_value, reason = gap, "planner_native_extend" if trigger else "planner_native_stop"
            else:
                raise ValueError(f"unsupported Phase28 method: {method}")
        else:
            reason = "terminal_exhaustive_no_extension_available"
        if trigger:
            high_bank, _ = p24.shifted_bank(horizon, full_n, query, previous, cap)
            high_bank = tuple(tuple(int(a) for a in seq) for seq in high_bank)
            if not set(low_bank).issubset(set(high_bank)):
                raise RuntimeError("Phase28 B24/B32 candidate banks are not nested by identity")
            known = {tuple(ev.sequence): ev for ev in low_evals}
            additions = [seq for seq in high_bank if seq not in known]
            high_evals = [p16d.evaluate_sequence(state, scenario, seq) for seq in additions]
            evals = low_evals + high_evals
            actual_bank = tuple(ev.sequence for ev in evals)
        else:
            evals, actual_bank = low_evals, low_bank
        candidate_count = len(evals)

    chosen, tie_count = p20.choose_lexicographic_best(evals)
    records = [_record(ev) for ev in evals]
    selected = tuple(chosen.sequence)
    if selected not in {tuple(row["sequence"]) for row in records}:
        raise RuntimeError("exact selection returned a candidate outside the evaluated bank")
    decision = {"decision_index": int(state.elapsed_steps), "query_seed": int(query),
        "horizon": int(horizon), "previous_plan": list(previous) if previous is not None else None,
        "state_raw52": state_features(scenario, state).astype(float).tolist(),
        "nominal_budget": int(candidate_count), "evaluated_candidate_count": len(evals),
        "extended": len(evals) > min(base_n, 9 ** horizon), "trigger_reason": reason,
        "trigger_score": score_value, "b24_no_safe_success": bool(no_success) if method not in
            {"fixed_b26", "fixed_b32"} else None,
        "b24_same_category_gap": gap if method not in {"fixed_b26", "fixed_b32"} else None,
        "candidate_bank": records, "chosen_sequence": list(selected),
        "chosen_action": int(selected[0]), "best_objective": list(chosen.objective),
        "tie_count": int(tie_count), "exact_transitions": int(sum(ev.transitions for ev in evals)),
        "latency_seconds": float(time.perf_counter() - started)}

    # Fixed B32 supplies same-state hindsight diagnostics with no extra simulator calls.
    if method == "fixed_b32":
        low_bank, _ = p24.shifted_bank(horizon, base_n, query, previous, cap)
        by_sequence = {tuple(row["sequence"]): row for row in records}
        if not set(low_bank).issubset(by_sequence):
            raise RuntimeError("fixed B32 diagnostic failed B24 identity inclusion")
        low_records = [by_sequence[tuple(seq)] for seq in low_bank]
        low_best, high_best = _winner(low_records), _winner(records)
        low_success = any(row["category"] == "success" for row in low_records)
        first_category_change = (int(high_best["objective"][0]) < int(low_best["objective"][0]))
        decision["search_value_diagnostic"] = {
            "strict_objective_improves": tuple(high_best["objective"]) < tuple(low_best["objective"]),
            "first_action_changes": int(low_best["sequence"][0]) != int(high_best["sequence"][0]),
            "outcome_category_improves": first_category_change,
            "safe_success_candidate_newly_available": (not low_success and any(
                row["category"] == "success" and tuple(row["sequence"]) not in set(low_bank)
                for row in records)),
            "collision_best_replaced_by_safe": (low_best["category"] == "collision"
                and high_best["category"] != "collision"),
            "b24_best_objective": list(low_best["objective"]),
            "b32_best_objective": list(high_best["objective"]),
            "b24_best_action": int(low_best["sequence"][0]),
            "b32_best_action": int(high_best["sequence"][0]),
            "b24_category": low_best["category"], "b32_category": high_best["category"],
            "b24_success_candidate_count": sum(row["category"] == "success" for row in low_records),
            "b32_success_candidate_count": sum(row["category"] == "success" for row in records)}
    successor = corridor_step(state, int(selected[0]), scenario.spec).state
    return successor, selected, decision


def _episode(scenario, arm: dict, cfg: dict, model: dict, lock_hash: str) -> dict:
    state, previous, diagnostics = scenario.initial_state, None, []
    while state.status == "active":
        state, selected, row = _decision(scenario, state, previous, int(arm["replica"]),
                                         arm["method"], arm["layout_sha256"], cfg, model)
        diagnostics.append(row)
        previous = selected
    return {"seed": int(scenario.seed), "layout_sha256": scenario.layout_sha256,
            "method": arm["method"], "replica": int(arm["replica"]),
            "global_schedule_index": int(arm["global_schedule_index"]),
            "local_order_index": int(arm["local_order_index"]), "outcome": state.status,
            "steps": int(state.elapsed_steps), "decision_count": len(diagnostics),
            "candidate_sequences": sum(d["evaluated_candidate_count"] for d in diagnostics),
            "exact_transitions": sum(d["exact_transitions"] for d in diagnostics),
            "decision_latencies": [d["latency_seconds"] for d in diagnostics],
            "wall_seconds": sum(d["latency_seconds"] for d in diagnostics),
            "decision_diagnostics": diagnostics, "execution_lock_sha256": lock_hash}


def run(root: Path = ROOT) -> dict:
    cfg = config(root)
    validate_config(cfg)
    lock, schedule, lock_hash = _read_lock(root, cfg)
    out = root / cfg["outputs"]["run_dir"]
    records = root / cfg["outputs"]["records"]
    manifest_path = root / cfg["outputs"]["run_manifest"]
    if records.exists() or manifest_path.exists() or (out / "result.json").exists():
        raise FileExistsError("Phase28 is append-only/non-resumable; refusing overwrite or resume")
    model = json.loads((root / cfg["parent"]["phase27a_model"]).read_text(encoding="utf-8"))
    started = time.perf_counter()
    with gzip.open(records, "xt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        for index, arm in enumerate(schedule):
            scenario = generate_staggered_corridor(int(arm["seed"]))
            if scenario.layout_sha256 != arm["layout_sha256"]:
                raise RuntimeError("generated scenario differs from locked layout identity")
            row = _episode(scenario, arm, cfg, model, lock_hash)
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            if (index + 1) % 100 == 0:
                print(json.dumps({"completed_episodes": index + 1, "total_episodes": len(schedule),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    manifest = {"schema": "phase28-sequential-exact-search-run-manifest-v1",
        "execution_lock_sha256": lock_hash, "records_sha256": sha256(records),
        "episodes": len(schedule), "layouts": int(cfg["block"]["layout_count"]),
        "runtime_seconds": float(time.perf_counter() - started), "python": platform.python_version(),
        "numpy": np.__version__, "source_commit": _git(root, "rev-parse", "HEAD"),
        "training": False, "reserve_access": False, "test_access": False, "d168_access": False}
    p18.write_new(manifest_path, manifest)
    return manifest


def _bootstrap(reference: list[float], candidate: list[float], seed: int,
               draws: int, confidence: float = 0.95) -> dict:
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired layout bootstrap requires equal nonempty vectors")
    rng = np.random.default_rng(seed)
    values = (b - a)[rng.integers(0, len(a), size=(draws, len(a)))].mean(axis=1)
    return {"difference": float(np.mean(b - a)), "ci95_lower": float(np.quantile(values, .025)),
            "ci95_upper": float(np.quantile(values, .975)),
            "one_sided_lower": float(np.quantile(values, 1 - confidence)),
            "one_sided_upper": float(np.quantile(values, confidence)), "confidence": confidence}


def _score_records(root: Path, write: bool = True) -> dict:
    cfg = config(root)
    lock, schedule, lock_hash = _read_lock(root, cfg)
    path = root / cfg["outputs"]["records"]
    expected = {(r["seed"], r["method"], r["replica"]): r for r in schedule}
    keyed = {}
    latencies = {name: [] for name in ARMS}
    extension = {name: [] for name in ARMS}
    candidate_counts = {name: [] for name in ARMS}
    search_diag = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["seed"]), row["method"], int(row["replica"]))
            arm = expected.get(key)
            if arm is None or key in keyed or row.get("execution_lock_sha256") != lock_hash:
                raise RuntimeError("Phase28 record has duplicate/unexpected identity or bad lock binding")
            if any(row.get(field) != value for field, value in arm.items()):
                raise RuntimeError("Phase28 record differs from frozen schedule")
            if row["outcome"] not in {"success", "collision", "timeout"}:
                raise RuntimeError("invalid Phase28 terminal outcome")
            if row["steps"] != row["decision_count"] or len(row["decision_diagnostics"]) != row["steps"]:
                raise RuntimeError("Phase28 episode step/decision count mismatch")
            prev = None
            for d in row["decision_diagnostics"]:
                if d["previous_plan"] != (list(prev) if prev is not None else None):
                    raise RuntimeError("Phase28 previous-plan chain mismatch")
                if d["query_seed"] != p17.query_seed(row["seed"], d["decision_index"], row["replica"]):
                    raise RuntimeError("Phase28 query seed mismatch")
                if len(d["candidate_bank"]) != d["evaluated_candidate_count"]:
                    raise RuntimeError("Phase28 candidate record/count mismatch")
                if len({tuple(c["sequence"]) for c in d["candidate_bank"]}) != len(d["candidate_bank"]):
                    raise RuntimeError("Phase28 evaluated bank has duplicate sequence")
                if tuple(d["chosen_sequence"]) not in {tuple(c["sequence"]) for c in d["candidate_bank"]}:
                    raise RuntimeError("Phase28 selected sequence absent from exact bank")
                if tuple(d["best_objective"]) != min(tuple(c["objective"]) for c in d["candidate_bank"]):
                    raise RuntimeError("Phase28 selected objective is not the exact bank minimum")
                if d["evaluated_candidate_count"] > min(32, 9 ** d["horizon"]):
                    raise RuntimeError("Phase28 evaluated more candidates than available")
                prev = tuple(d["chosen_sequence"])
                latencies[row["method"]].append(float(d["latency_seconds"]))
                extension[row["method"]].append(bool(d["extended"]))
                candidate_counts[row["method"]].append(int(d["evaluated_candidate_count"]))
                if "search_value_diagnostic" in d:
                    search_diag.append(d["search_value_diagnostic"])
            keyed[key] = row
    if set(keyed) != set(expected):
        raise RuntimeError(f"Phase28 records incomplete: got {len(keyed)}, expected {len(expected)}")
    seeds = sorted({r["seed"] for r in schedule})
    methods, per_replica, vectors = {}, {}, {}
    for name in ARMS:
        rows = [keyed[(seed, name, rep)] for seed in seeds for rep in cfg["block"]["replicas"]]
        successes = sum(r["outcome"] == "success" for r in rows)
        collisions = sum(r["outcome"] == "collision" for r in rows)
        methods[name] = {"episodes": len(rows), "successes": successes, "collisions": collisions,
            "timeouts": sum(r["outcome"] == "timeout" for r in rows),
            "safe_success_rate": successes / len(rows), "collision_rate": collisions / len(rows),
            "timeout_rate": sum(r["outcome"] == "timeout" for r in rows) / len(rows),
            "mean_candidate_sequences_per_episode": float(np.mean([r["candidate_sequences"] for r in rows])),
            "mean_exact_transitions_per_episode": float(np.mean([r["exact_transitions"] for r in rows])),
            "median_decision_latency_seconds": float(np.median(latencies[name])),
            "extension_rate": float(np.mean(extension[name])) if name not in {"fixed_b24", "fixed_b26", "fixed_b32"} else 0.0}
        per_replica[name] = {}
        for rep in cfg["block"]["replicas"]:
            group = [keyed[(seed, name, rep)] for seed in seeds]
            per_replica[name][str(rep)] = {"safe_success_rate": float(np.mean([r["outcome"] == "success" for r in group])),
                "collision_rate": float(np.mean([r["outcome"] == "collision" for r in group])),
                "timeout_rate": float(np.mean([r["outcome"] == "timeout" for r in group]))}
        vectors[name] = {metric: [float(np.mean([
            float(keyed[(seed, name, rep)]["outcome"] == metric) if metric in {"success", "collision"}
            else keyed[(seed, name, rep)]["exact_transitions"] for rep in cfg["block"]["replicas"]]))
            for seed in seeds] for metric in ("success", "collision", "transitions")}
        vectors[name]["candidate_sequences"] = [float(np.mean([
            keyed[(seed, name, rep)]["candidate_sequences"] for rep in cfg["block"]["replicas"]]))
            for seed in seeds]

    gate = cfg["gates"]
    def absolute(name: str) -> bool:
        m = methods[name]
        return (m["safe_success_rate"] > gate["absolute_safe_success_strictly_above"]
            and m["collision_rate"] <= gate["collision_at_most"]
            and m["timeout_rate"] == gate["timeout_exactly"]
            and all(r["safe_success_rate"] >= gate["per_replica_success_at_least"]
                    and r["collision_rate"] <= gate["per_replica_collision_at_most"]
                    for r in per_replica[name].values()))
    abs_pass = {name: absolute(name) for name in ARMS}
    comp = {}
    comparisons = [(name, "fixed_b32") for name in ("fixed_b24", "fixed_b26",
        "random_adaptive_b24_plus_8", "phase27_learned_adaptive_b24_plus_8",
        "planner_native_adaptive_b24_plus_8")]
    for i, (name, ref) in enumerate(comparisons):
        succ = _bootstrap(vectors[ref]["success"], vectors[name]["success"],
            int(cfg["analysis"]["bootstrap_seed"]) + 10*i, int(cfg["analysis"]["bootstrap_draws"]),
            float(cfg["analysis"]["paired_one_sided_confidence_vs_b32"]))
        coll = _bootstrap(vectors[ref]["collision"], vectors[name]["collision"],
            int(cfg["analysis"]["bootstrap_seed"]) + 10*i + 1, int(cfg["analysis"]["bootstrap_draws"]),
            float(cfg["analysis"]["paired_one_sided_confidence_vs_b32"]))
        ratio = np.asarray(vectors[name]["transitions"]) / np.asarray(vectors[ref]["transitions"])
        trans = _bootstrap([0.0]*len(seeds), ratio.tolist(), int(cfg["analysis"]["bootstrap_seed"])+10*i+2,
            int(cfg["analysis"]["bootstrap_draws"]), .95)
        cand_ratio = np.asarray(vectors[name]["candidate_sequences"]) / np.asarray(vectors[ref]["candidate_sequences"])
        cand_boot = _bootstrap([0.0]*len(seeds), cand_ratio.tolist(), int(cfg["analysis"]["bootstrap_seed"])+10*i+3,
            int(cfg["analysis"]["bootstrap_draws"]), .95)
        success_events = sum(any(keyed[(seed, name, rep)]["outcome"] != "success"
            and keyed[(seed, ref, rep)]["outcome"] == "success" for rep in cfg["block"]["replicas"])
            for seed in seeds)
        collision_events = sum(any(keyed[(seed, name, rep)]["outcome"] == "collision"
            and keyed[(seed, ref, rep)]["outcome"] != "collision" for rep in cfg["block"]["replicas"])
            for seed in seeds)
        confidence = float(cfg["analysis"]["adverse_layout_one_sided_confidence"])
        comp[name] = {"reference": ref, "success_difference": succ, "collision_difference": coll,
            "transition_ratio": trans, "candidate_sequence_ratio": cand_boot,
            "median_latency_ratio": methods[name]["median_decision_latency_seconds"] /
                methods[ref]["median_decision_latency_seconds"],
            "success_regression_layouts": success_events,
            "success_regression_upper": p16d.binomial_upper(success_events, len(seeds), confidence),
            "added_collision_layouts": collision_events,
            "added_collision_upper": p16d.binomial_upper(collision_events, len(seeds), confidence)}
    # Attribution comparisons use the frozen one-sided 95% threshold.
    attribution = {}
    for i, (name, ref) in enumerate((
        ("random_adaptive_b24_plus_8", "fixed_b26"),
        ("planner_native_adaptive_b24_plus_8", "fixed_b26"),
        ("planner_native_adaptive_b24_plus_8", "random_adaptive_b24_plus_8"),
        ("planner_native_adaptive_b24_plus_8", "phase27_learned_adaptive_b24_plus_8"))):
        succ = _bootstrap(vectors[ref]["success"], vectors[name]["success"],
            int(cfg["analysis"]["bootstrap_seed"])+100+i*2, int(cfg["analysis"]["bootstrap_draws"]),
            float(cfg["analysis"]["attribution_one_sided_confidence"]))
        coll = _bootstrap(vectors[ref]["collision"], vectors[name]["collision"],
            int(cfg["analysis"]["bootstrap_seed"])+101+i*2, int(cfg["analysis"]["bootstrap_draws"]),
            float(cfg["analysis"]["attribution_one_sided_confidence"]))
        attribution[f"{name}_vs_{ref}"] = {"success_difference": succ, "collision_difference": coll,
            "noninferior": (succ["one_sided_lower"] >= float(gate["success_difference_lower_vs_b32"])
                and coll["one_sided_upper"] <= float(gate["collision_difference_upper_vs_b32"])),
            "superiority_on_success_or_collision": succ["one_sided_lower"] > 0 or coll["one_sided_upper"] < 0}
    native = "planner_native_adaptive_b24_plus_8"
    native_fixed = attribution[f"{native}_vs_fixed_b26"]
    native_random = attribution[f"{native}_vs_random_adaptive_b24_plus_8"]
    native_b32 = comp[native]
    reliability_gate = {"absolute_reliability": abs_pass[native],
        "success_noninferiority_vs_b32": native_b32["success_difference"]["one_sided_lower"]
            >= float(gate["success_difference_lower_vs_b32"]),
        "collision_noninferiority_vs_b32": native_b32["collision_difference"]["one_sided_upper"]
            <= float(gate["collision_difference_upper_vs_b32"]),
        "success_regression_layout_bound": native_b32["success_regression_upper"] <= gate["adverse_layout_event_upper_at_most"],
        "added_collision_layout_bound": native_b32["added_collision_upper"] <= gate["adverse_layout_event_upper_at_most"],
        "transition_mean": native_b32["transition_ratio"]["difference"] <= cfg["analysis"]["transition_ratio_mean_at_most_vs_b32"],
        "transition_upper": native_b32["transition_ratio"]["ci95_upper"] <= cfg["analysis"]["transition_ratio_95_upper_at_most_vs_b32"],
        "latency_ratio": native_b32["median_latency_ratio"] <= cfg["analysis"]["median_latency_ratio_at_most_vs_b32"]}
    # Candidate-level diagnostics are only available for decisions actually scored at B32.
    search_summary = {}
    for key in ("strict_objective_improves", "first_action_changes", "outcome_category_improves",
                "safe_success_candidate_newly_available", "collision_best_replaced_by_safe"):
        search_summary[key] = sum(bool(row[key]) for row in search_diag)
    search_summary["fixed_b32_decisions"] = len(search_diag)
    result = {"schema": "phase28-sequential-exact-search-result-v1", "status": "complete",
        "layouts": len(seeds), "episodes": len(schedule), "methods": methods,
        "per_replica": per_replica, "absolute_reliability_pass": abs_pass,
        "comparisons_vs_b32": comp, "primary_attribution_comparisons": attribution,
        "planner_native_vs_matched_fixed_average_pass": (
            native_fixed["noninferior"] and native_fixed["superiority_on_success_or_collision"]),
        "planner_native_vs_random_pass": (
            native_random["noninferior"] and native_random["superiority_on_success_or_collision"]),
        "planner_native_reliability_compute_gate": reliability_gate,
        "planner_native_reliability_compute_pass": all(reliability_gate.values()),
        "fixed_b32_same_state_search_diagnostics": search_summary,
        "observed_decision_compute": {name: {"mean_candidates_per_decision": float(np.mean(candidate_counts[name])),
            "extension_rate": float(np.mean(extension[name]))} for name in ARMS},
        "decision": "scored_by_frozen_phase28_criteria", "exact_rollout_and_selection_authoritative": True,
        "training": False, "reserve_access": False, "test_access": False, "d168_access": False,
        "records_sha256": sha256(path), "execution_lock_sha256": lock_hash}
    if write:
        result_path = root / cfg["outputs"]["result"]
        p18.write_new(result_path, result)
        score_manifest = {"schema": "phase28-sequential-exact-search-score-manifest-v1",
            "records_sha256": result["records_sha256"], "result_sha256": sha256(result_path),
            "execution_lock_sha256": lock_hash, "scorer_sha256": sha256(Path(__file__).resolve()),
            "scorer_commit": _git(root, "rev-parse", "HEAD"), "planner_episode_reruns": 0,
            "candidate_evaluations_during_scoring": 0, "reserve_access": False,
            "test_access": False, "d168_access": False}
        p18.write_new(root / cfg["outputs"]["result_manifest"], score_manifest)
    return result


def score(root: Path = ROOT) -> dict:
    return _score_records(root, write=True)


def audit(root: Path = ROOT) -> dict:
    cfg = config(root)
    lock, schedule, lock_hash = _read_lock(root, cfg)
    records_path = root / cfg["outputs"]["records"]
    result_path = root / cfg["outputs"]["result"]
    manifest = json.loads((root / cfg["outputs"]["run_manifest"]).read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    score_manifest = json.loads((root / cfg["outputs"]["result_manifest"]).read_text(encoding="utf-8"))
    records_hash = sha256(records_path)
    if (manifest["records_sha256"] != records_hash or manifest["execution_lock_sha256"] != lock_hash
            or score_manifest["records_sha256"] != records_hash or score_manifest["result_sha256"] != sha256(result_path)
            or result["records_sha256"] != records_hash or score_manifest["planner_episode_reruns"] != 0
            or score_manifest["candidate_evaluations_during_scoring"] != 0):
        raise RuntimeError("Phase28 post-run hash/accounting chain failed")
    reproduced = _score_records(root, write=False)
    if canonical(reproduced) != canonical(result):
        raise RuntimeError("Phase28 metrics do not reproduce from locked records")
    return {"audit": "pass", "episodes": len(schedule), "records_sha256": records_hash,
            "result_sha256": sha256(result_path), "execution_lock_sha256": lock_hash,
            "candidate_evaluations_during_audit": 0, "reserve_test_d168_access": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "score", "audit"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.command](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
