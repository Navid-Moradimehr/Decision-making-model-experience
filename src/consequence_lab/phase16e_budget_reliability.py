"""Phase16E: fresh-layout reliability confirmation for exact-planning budgets."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

from consequence_lab.corridor_evaluation import evaluate_corridor_episode
from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase16c_prior_guided import evaluate_sequence, layout_hash_audit
from consequence_lab.phase16d_compute_frontier import (
    binomial_upper,
    choose_lexicographic_best,
    sha256,
    uniform_prefix_bank,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase16e_budget_reliability_confirmation_v1.yaml"
PROTOCOL = ROOT / "reports/phase16e_budget_reliability_confirmation_protocol.md"
LOCK = ROOT / "experiments/phase16e_budget_reliability_confirmation_v1.lock.json"


class Phase16EError(RuntimeError):
    """Raised when Phase16E provenance, scope, or execution contracts fail."""


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _load_config() -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("schema") != "phase16e-budget-reliability-confirmation-v1":
        raise Phase16EError("unexpected Phase16E config schema")
    if cfg.get("execution_authorized") is not True:
        raise Phase16EError("Phase16E execution is not authorized")
    return cfg


def _verify_parents(root: Path, cfg: dict[str, Any]) -> None:
    for name, parent in cfg["parents"].items():
        path = root / parent["path"]
        if not path.is_file() or sha256(path) != parent["sha256"]:
            raise Phase16EError(f"Phase16D parent artifact mismatch: {name}")
    result_cfg = cfg["parents"]["phase16d_result"]
    manifest_cfg = cfg["parents"]["phase16d_manifest"]
    lock_cfg = cfg["parents"]["phase16d_lock"]
    episode_cfg = cfg["parents"]["phase16d_episodes"]
    result = json.loads((root / result_cfg["path"]).read_text(encoding="utf-8"))
    manifest = json.loads((root / manifest_cfg["path"]).read_text(encoding="utf-8"))
    if (result.get("status") != "complete"
            or result.get("gate_pass") is not False
            or result.get("layout_count") != 200
            or result.get("unique_episode_count") != 2800
            or result.get("lock_sha256") != lock_cfg["sha256"]
            or manifest.get("status") != "complete"
            or manifest.get("episode_count") != 2800
            or manifest.get("result_sha256") != result_cfg["sha256"]
            or manifest.get("episode_sha256") != episode_cfg["sha256"]
            or manifest.get("lock_sha256") != lock_cfg["sha256"]):
        raise Phase16EError("Phase16D parent result/manifest contract is inconsistent")


def _seed_block_is_disjoint(cfg: dict[str, Any]) -> None:
    start = int(cfg["environment"]["proposed_seed_start"])
    stop = start + int(cfg["environment"]["proposed_layout_count"])
    for name in ("protected_d168_block", "sealed_reserve_block"):
        low, high = map(int, cfg["environment"][name])
        if start < high and low < stop:
            raise Phase16EError(f"proposed Phase16E seeds overlap {name}")


def _source_paths() -> tuple[str, ...]:
    return (
        "AGENTS.md", "pyproject.toml", "uv.lock",
        "configs/phase16e_budget_reliability_confirmation_v1.yaml",
        "reports/phase16e_budget_reliability_confirmation_protocol.md",
        "src/consequence_lab/phase16e_budget_reliability.py",
        "scripts/phase16e_budget_reliability.py",
        "scripts/freeze_phase16e_lock.py",
        "tests/test_phase16e_protocol.py",
        "tests/test_phase16e_execution.py",
        "src/consequence_lab/phase16d_compute_frontier.py",
        "src/consequence_lab/phase16c_prior_guided.py",
        "src/consequence_lab/phase15_d164.py",
        "src/consequence_lab/phase14g.py",
        "src/consequence_lab/phase13.py",
        "src/consequence_lab/phase14.py",
        "src/consequence_lab/phase14c.py",
        "src/consequence_lab/phase12_teacher.py",
        "src/consequence_lab/corridor_baselines.py",
        "src/consequence_lab/corridor_evaluation.py",
        "src/consequence_lab/corridor_tasks.py",
        "src/consequence_lab/inertial_corridor.py",
        "configs/phase16d_planning_compute_frontier_v1.yaml",
        "reports/phase16d_planning_compute_frontier_protocol.md",
        "reports/phase16d_planning_compute_frontier_result.md",
        "experiments/phase16d_planning_compute_frontier_v1.lock.json",
    )


def build_layout_audit(root: Path = ROOT) -> dict[str, Any]:
    """Generate layout identities only; this function never steps the task."""
    root = Path(root).resolve()
    cfg = _load_config()
    _seed_block_is_disjoint(cfg)
    _verify_parents(root, cfg)
    if _git(root, "status", "--porcelain"):
        raise Phase16EError("commit Phase16E source, config, tests, and protocol before audit")
    output = root / cfg["outputs"]["proposed_run_dir"]
    if output.exists():
        raise Phase16EError("refusing to overwrite a Phase16E output directory")

    start = int(cfg["environment"]["proposed_seed_start"])
    count = int(cfg["environment"]["proposed_layout_count"])
    missing = [relative for relative in _source_paths() if not (root / relative).is_file()]
    if missing:
        raise Phase16EError(f"missing Phase16E audit sources: {missing}")
    audit = layout_hash_audit(start, count, root)
    if (audit.get("unique_hashes") != count
            or audit.get("overlap_historical") != 0
            or audit.get("overlap_outcome_bearing") != 0
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("d168_seed_block_accessed") is not False
            or audit.get("outcomes_or_transitions_computed") is not False):
        raise Phase16EError("Phase16E hash-only layout freshness audit failed")

    # Keep historical-source hashes and the explicit protected-block declaration
    # alongside the generated identities for independent lock verification.
    audit.update({
        "schema": "phase16e-layout-hash-audit-v1",
        "status": "hash_only_passed_no_outcomes",
        "generation_git_commit": _git(root, "rev-parse", "HEAD"),
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(PROTOCOL),
        "source_sha256": {relative: sha256(root / relative) for relative in _source_paths()},
        "d168_protected_seed_block": cfg["environment"]["protected_d168_block"],
        "reserve_protected_seed_block": cfg["environment"]["sealed_reserve_block"],
        "task_transitions_evaluated": False,
    })
    output.mkdir(parents=True)
    (root / cfg["outputs"]["layout_audit"]).write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    order = execution_order(audit["layouts"], int(cfg["execution"]["counterbalance_seed"]), cfg)
    (root / cfg["outputs"]["execution_order"]).write_text(
        json.dumps(order, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": audit["status"],
        "seed_start": start,
        "layout_count": count,
        "unique_hashes": audit["unique_hashes"],
        "historical_source_count": audit["historical_source_count"],
        "overlap_historical": audit["overlap_historical"],
        "overlap_outcome_bearing": audit["overlap_outcome_bearing"],
        "d168_seed_block_accessed": audit["d168_seed_block_accessed"],
        "reserve_generated_or_accessed": audit["reserve_generated_or_accessed"],
        "outcomes_or_transitions_computed": audit["outcomes_or_transitions_computed"],
        "scheduled_episodes": len(order),
    }


def execution_order(layouts: list[dict], counterbalance_seed: int,
                    cfg: dict[str, Any] | None = None) -> list[dict]:
    """Create 9 paired planner arms plus one geometry episode per layout."""
    cfg = cfg or _load_config()
    budgets = [int(value) for value in cfg["planning"]["budgets"]]
    replicas = [int(value) for value in cfg["planning"]["search_replica_ids"]]
    rows: list[dict[str, Any]] = []
    for layout in layouts:
        seed = int(layout["seed"])
        arms = [(rep, budget) for rep in replicas for budget in budgets]
        random.Random(counterbalance_seed + seed).shuffle(arms)
        stochastic = [
            {"method": "uniform_shooting", "replica": rep, "budget": budget}
            for rep, budget in arms
        ]
        geometry = {"method": "geometry_waypoint", "replica": None, "budget": None}
        schedule = ([geometry] + stochastic) if seed % 2 == 0 else (stochastic + [geometry])
        for index, arm in enumerate(schedule):
            rows.append({"seed": seed, "layout_sha256": layout["layout_sha256"],
                         "order_index": index, **arm})
    expected = len(layouts) * (len(budgets) * len(replicas) + 1)
    if len(rows) != expected:
        raise Phase16EError("Phase16E execution schedule has the wrong length")
    return rows


def _verify_audit_and_order(root: Path, cfg: dict[str, Any], lock: dict[str, Any]):
    audit_path = root / cfg["outputs"]["layout_audit"]
    order_path = root / cfg["outputs"]["execution_order"]
    if sha256(audit_path) != lock["layout_audit_sha256"]:
        raise Phase16EError("Phase16E layout audit hash differs from lock")
    if sha256(order_path) != lock["execution_order_sha256"]:
        raise Phase16EError("Phase16E execution order hash differs from lock")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("schema") != "phase16e-layout-hash-audit-v1"
            or audit.get("status") != "hash_only_passed_no_outcomes"
            or audit.get("seed_start") != cfg["environment"]["proposed_seed_start"]
            or audit.get("layout_count") != cfg["environment"]["proposed_layout_count"]
            or audit.get("unique_hashes") != audit.get("layout_count")
            or audit.get("overlap_historical") != 0
            or audit.get("overlap_outcome_bearing") != 0
            or audit.get("d168_seed_block_accessed") is not False
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("outcomes_or_transitions_computed") is not False
            or audit.get("task_transitions_evaluated") is not False):
        raise Phase16EError("Phase16E layout audit failed validation")
    for source, expected in audit["historical_source_hashes"].items():
        if sha256(root / source) != expected:
            raise Phase16EError(f"historical provenance source changed: {source}")
    for source, expected in audit["source_sha256"].items():
        if sha256(root / source) != expected:
            raise Phase16EError(f"hash-audit source changed: {source}")
    order = json.loads(order_path.read_text(encoding="utf-8"))
    expected_order = execution_order(audit["layouts"], cfg["execution"]["counterbalance_seed"], cfg)
    if order != expected_order:
        raise Phase16EError("Phase16E frozen execution schedule is inconsistent")
    return audit, order


def _verify_lock(root: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not LOCK.is_file():
        raise Phase16EError("Phase16E immutable execution lock is missing")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if (lock.get("schema") != "phase16e-execution-lock-v1"
            or lock.get("config_sha256") != sha256(CONFIG)
            or lock.get("protocol_sha256") != sha256(PROTOCOL)):
        raise Phase16EError("Phase16E lock/config/protocol mismatch")
    if _git(root, "status", "--porcelain"):
        raise Phase16EError("Phase16E requires a clean committed worktree")
    head = _git(root, "rev-parse", "HEAD")
    if subprocess.run(["git", "merge-base", "--is-ancestor", lock["source_commit"], head],
                      cwd=root).returncode != 0:
        raise Phase16EError("Phase16E implementation commit is not an ancestor of this run")
    for relative, expected in lock["source_sha256"].items():
        if sha256(root / relative) != expected:
            raise Phase16EError(f"Phase16E source changed after lock: {relative}")
    _verify_parents(root, cfg)
    audit, order = _verify_audit_and_order(root, cfg, lock)
    return lock, audit, order


def evaluate_shooting(scenario: CorridorScenario, budget: int, replica: int,
                      cfg: dict[str, Any]) -> dict[str, Any]:
    state = scenario.initial_state
    steps_start = state.elapsed_steps
    diagnostics: list[dict[str, Any]] = []
    latencies: list[float] = []
    actions: Counter[str] = Counter()
    sequences = transitions = ties = duplicates = fallbacks = available = 0
    index = 0
    started_episode = time.perf_counter()
    while state.status == "active":
        tick = time.perf_counter()
        horizon = min(int(cfg["planning"]["horizon"]),
                      scenario.spec.max_steps - state.elapsed_steps)
        query = (scenario.seed * 1_000_003 + index
                 + replica * int(cfg["planning"]["replica_seed_stride"])) % 2**32
        actual_budget = min(budget, 9**horizon)
        bank, rejected, fallback = uniform_prefix_bank(
            horizon, actual_budget, query, int(cfg["planning"]["common_attempt_cap"]))
        scores = [evaluate_sequence(state, scenario, seq) for seq in bank]
        best, tied_count = choose_lexicographic_best(scores)
        used_transitions = sum(item.transitions for item in scores)
        success_count = sum(item.success_reached for item in scores)
        state = corridor_step(state, best.sequence[0], scenario.spec).state
        latencies.append(time.perf_counter() - tick)
        actions[str(best.sequence[0])] += 1
        sequences += len(bank)
        transitions += used_transitions
        ties += tied_count > 1
        duplicates += rejected
        fallbacks += fallback
        available += success_count
        diagnostics.append({
            "decision_index": index, "query_seed": query, "horizon": horizon,
            "actual_budget": len(bank), "chosen_sequence": list(best.sequence),
            "chosen_first_action": best.sequence[0], "best_objective": list(best.objective),
            "tied_best_count": tied_count,
            "success_sequences_available": success_count,
            "exact_transitions": used_transitions,
            "duplicate_rejections": rejected, "fallback_sequences": fallback,
        })
        index += 1
    return {
        "seed": scenario.seed, "layout_sha256": scenario.layout_sha256,
        "method": "uniform_shooting", "replica": replica, "budget": budget,
        "outcome": state.status, "steps": state.elapsed_steps - steps_start,
        "decision_count": index, "exact_sequences": sequences,
        "exact_transitions": transitions, "executed_transitions": index,
        "objective_tie_decisions": int(ties), "duplicate_rejections": duplicates,
        "fallback_sequences": fallbacks, "complete_success_sequences": available,
        "selected_action_counts": dict(sorted(actions.items())),
        "latency_seconds_per_decision": latencies,
        "wall_seconds": time.perf_counter() - started_episode,
        "decision_diagnostics": diagnostics,
    }


def _geometry_episode(scenario: CorridorScenario, cfg: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_corridor_episode(
        scenario, "geometry_waypoint",
        parameters={"resolution": float(cfg["geometry"]["resolution"]),
                    "waypoint_tolerance": float(cfg["geometry"]["waypoint_tolerance"])})
    return {
        "seed": result.seed, "layout_sha256": result.layout_sha256,
        "method": "geometry_waypoint", "replica": None, "budget": None,
        "outcome": result.outcome, "steps": result.steps,
        "decision_count": result.decision_count, "exact_sequences": 0,
        "exact_transitions": 0, "executed_transitions": result.steps,
        "objective_tie_decisions": 0, "duplicate_rejections": 0,
        "fallback_sequences": 0, "complete_success_sequences": 0,
        "selected_action_counts": {}, "latency_seconds_per_decision": [],
        "wall_seconds": result.wall_seconds,
        "geometry_episode_average_decision_latency": result.wall_seconds_per_decision,
        "decision_diagnostics": [],
    }


def paired_bootstrap(reference: list[float], candidate: list[float], seed: int,
                     draws: int, *, lower_only: bool = False) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 1 or cand.shape != ref.shape or len(ref) == 0:
        raise ValueError("paired bootstrap requires aligned nonempty layout vectors")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(ref), size=(draws, len(ref)))
    diffs = (cand[samples] - ref[samples]).mean(axis=1)
    result = {"difference": float(np.mean(cand - ref))}
    result["ci_lower"] = float(np.quantile(diffs, .025))
    if not lower_only:
        result["ci_upper"] = float(np.quantile(diffs, .975))
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [x for row in rows for x in row["latency_seconds_per_decision"]]
    geometry_latencies = [r["geometry_episode_average_decision_latency"] for r in rows
                          if r.get("geometry_episode_average_decision_latency") is not None]
    n = len(rows)
    if n == 0:
        raise ValueError("cannot summarize an empty arm")
    return {
        "episodes": n,
        "safe_success_rate": sum(r["outcome"] == "success" for r in rows) / n,
        "collision_rate": sum(r["outcome"] == "collision" for r in rows) / n,
        "timeout_rate": sum(r["outcome"] == "timeout" for r in rows) / n,
        "mean_actions_to_terminal": float(np.mean([r["steps"] for r in rows])),
        "exact_sequences_per_decision": float(np.mean([
            r["exact_sequences"] / max(1, r["decision_count"]) for r in rows])),
        "exact_transitions_per_episode": float(np.mean([r["exact_transitions"] for r in rows])),
        "exact_transitions_per_decision": float(np.mean([
            r["exact_transitions"] / max(1, r["decision_count"]) for r in rows])),
        "executed_transitions_per_episode": float(np.mean([r["executed_transitions"] for r in rows])),
        "complete_success_sequences_per_episode": float(np.mean([
            r["complete_success_sequences"] for r in rows])),
        "objective_tie_decisions": sum(r["objective_tie_decisions"] for r in rows),
        "duplicate_rejections": sum(r["duplicate_rejections"] for r in rows),
        "fallback_sequences": sum(r["fallback_sequences"] for r in rows),
        "decision_latency_median_seconds": float(np.median(latencies)) if latencies else None,
        "decision_latency_p90_seconds": float(np.quantile(latencies, .90)) if latencies else None,
        "geometry_episode_average_decision_latency_median_seconds": (
            float(np.median(geometry_latencies)) if geometry_latencies else None),
        "mean_episode_wall_seconds": float(np.mean([r["wall_seconds"] for r in rows])),
    }


def _ratio_bootstrap(ratios: list[float], seed: int, draws: int) -> dict[str, float]:
    values = np.asarray(ratios, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("ratio bootstrap requires nonempty layout ratios")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {"estimate": float(np.mean(values)),
            "ci95_lower": float(np.quantile(means, .025)),
            "ci95_upper": float(np.quantile(means, .975))}


def _episode_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return row["method"], row.get("replica"), row.get("budget"), int(row["seed"])


def summarize(records: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    layouts = sorted({int(row["seed"]) for row in records})
    if len(layouts) != int(cfg["environment"]["proposed_layout_count"]):
        raise ValueError("episode rows do not cover the locked layout set")
    keyed = {_episode_key(row): row for row in records}
    if len(keyed) != len(records):
        raise ValueError("duplicate Phase16E episode key")
    budget_values = [int(x) for x in cfg["planning"]["budgets"]]
    replicas = [int(x) for x in cfg["planning"]["search_replica_ids"]]
    expected = {
        ("uniform_shooting", rep, budget, seed)
        for seed in layouts for rep in replicas for budget in budget_values
    } | {("geometry_waypoint", None, None, seed) for seed in layouts}
    if set(keyed) != expected:
        raise ValueError("episode keys do not exactly match the locked Phase16E schedule")

    methods: dict[str, Any] = {}
    per_replica: dict[str, Any] = {}
    for budget in budget_values:
        all_rows = [keyed[("uniform_shooting", rep, budget, seed)]
                    for seed in layouts for rep in replicas]
        methods[f"uniform_B{budget}"] = _summary(all_rows)
        per_replica[str(budget)] = {
            str(rep): _summary([keyed[("uniform_shooting", rep, budget, seed)] for seed in layouts])
            for rep in replicas
        }
    geometry_rows = [keyed[("geometry_waypoint", None, None, seed)] for seed in layouts]
    methods["geometry_waypoint"] = _summary(geometry_rows)

    ref_budget = 128
    ref_success = [float(np.mean([
        keyed[("uniform_shooting", rep, ref_budget, seed)]["outcome"] == "success"
        for rep in replicas])) for seed in layouts]
    geometry_success = [float(keyed[("geometry_waypoint", None, None, seed)]["outcome"] == "success")
                        for seed in layouts]
    ana = cfg["analysis"]
    gate_ref = cfg["gates"]["reference_B128"]
    ref_vs_geometry = paired_bootstrap(geometry_success, ref_success,
                                       ana["bootstrap_seed"], ana["bootstrap_draws"])
    reference_pass = (
        methods["uniform_B128"]["safe_success_rate"] > gate_ref["mean_safe_success_strictly_above"]
        and methods["uniform_B128"]["collision_rate"] <= gate_ref["mean_collision_at_most"]
        and all(per_replica["128"][str(rep)]["safe_success_rate"]
                >= gate_ref["every_replica_safe_success_at_least"] for rep in replicas)
        and all(per_replica["128"][str(rep)]["collision_rate"]
                <= gate_ref["every_replica_collision_at_most"] for rep in replicas)
        and ref_vs_geometry["difference"] >= gate_ref["geometry_success_gain_at_least"]
        and ref_vs_geometry["ci_lower"] > gate_ref["paired_geometry_gain_ci_lower_above"]
    )

    candidate_results: dict[str, Any] = {}
    candidate_gate = cfg["gates"]["candidate_B32_B64"]
    for budget in budget_values[:-1]:
        b = str(budget)
        cand_success = [float(np.mean([
            keyed[("uniform_shooting", rep, budget, seed)]["outcome"] == "success"
            for rep in replicas])) for seed in layouts]
        success_ci = paired_bootstrap(ref_success, cand_success,
                                      ana["bootstrap_seed"], ana["bootstrap_draws"], lower_only=True)
        regressions = sum(any(
            keyed[("uniform_shooting", rep, budget, seed)]["outcome"] != "success"
            and keyed[("uniform_shooting", rep, ref_budget, seed)]["outcome"] == "success"
            for rep in replicas) for seed in layouts)
        added_collisions = sum(any(
            keyed[("uniform_shooting", rep, budget, seed)]["outcome"] == "collision"
            and keyed[("uniform_shooting", rep, ref_budget, seed)]["outcome"] != "collision"
            for rep in replicas) for seed in layouts)
        alpha_confidence = .9875
        regression_upper = binomial_upper(regressions, len(layouts), alpha_confidence)
        collision_upper = binomial_upper(added_collisions, len(layouts), alpha_confidence)
        transition_ratios = []
        for seed in layouts:
            candidate_transitions = np.mean([
                keyed[("uniform_shooting", rep, budget, seed)]["exact_transitions"]
                for rep in replicas])
            reference_transitions = np.mean([
                keyed[("uniform_shooting", rep, ref_budget, seed)]["exact_transitions"]
                for rep in replicas])
            transition_ratios.append(float(candidate_transitions / max(1.0, reference_transitions)))
        transition_ci = _ratio_bootstrap(transition_ratios, ana["bootstrap_seed"],
                                         ana["bootstrap_draws"])
        candidate_latencies = [value for seed in layouts for rep in replicas
                               for value in keyed[("uniform_shooting", rep, budget, seed)]
                               ["latency_seconds_per_decision"]]
        reference_latencies = [value for seed in layouts for rep in replicas
                              for value in keyed[("uniform_shooting", rep, ref_budget, seed)]
                              ["latency_seconds_per_decision"]]
        latency_ratio = float(np.median(candidate_latencies) / np.median(reference_latencies))

        tests = {
            "mean_safe_success": methods[f"uniform_B{budget}"]["safe_success_rate"]
                > candidate_gate["mean_safe_success_strictly_above"],
            "replica_success_floors": all(per_replica[b][str(rep)]["safe_success_rate"]
                >= candidate_gate["every_replica_safe_success_at_least"] for rep in replicas),
            "mean_collision_ceiling": methods[f"uniform_B{budget}"]["collision_rate"]
                <= candidate_gate["mean_collision_at_most"],
            "replica_collision_ceilings": all(per_replica[b][str(rep)]["collision_rate"]
                <= candidate_gate["every_replica_collision_at_most"] for rep in replicas),
            "paired_success_noninferiority": success_ci["ci_lower"]
                >= candidate_gate["paired_success_difference_ci_lower_at_least"],
            "success_regression_bound": regression_upper
                <= candidate_gate["success_regression_event_upper_at_most"],
            "added_collision_bound": collision_upper
                <= candidate_gate["added_collision_event_upper_at_most"],
            "mean_transition_ratio": transition_ci["estimate"]
                <= candidate_gate["mean_rollout_transition_ratio_at_most"],
            "transition_ratio_upper_bound": transition_ci["ci95_upper"]
                <= candidate_gate["transition_ratio_ci_upper_at_most"],
            "median_latency_ratio": latency_ratio
                <= candidate_gate["median_decision_latency_ratio_at_most"],
        }
        candidate_results[b] = {
            "success_difference_vs_B128": success_ci,
            "success_regression_layout_events": regressions,
            "success_regression_upper98_75": regression_upper,
            "added_collision_layout_events": added_collisions,
            "added_collision_upper98_75": collision_upper,
            "rollout_transition_ratio": transition_ci,
            "median_decision_latency_ratio": latency_ratio,
            "gate_components": tests,
            "gate_pass": bool(reference_pass and all(tests.values())),
        }

    selected_budget = 128
    if reference_pass:
        for budget in cfg["gates"]["selection_order"][:-1]:
            if candidate_results[str(budget)]["gate_pass"]:
                selected_budget = int(budget)
                break
    return {
        "schema": "phase16e-result-v1",
        "status": "complete",
        "layout_count": len(layouts),
        "unique_episode_count": len(records),
        "methods": methods,
        "per_replica": per_replica,
        "reference": {
            "B128_minus_geometry_success": ref_vs_geometry,
            "B128_reference_gate_pass": bool(reference_pass),
        },
        "candidates": candidate_results,
        "selected_budget_under_locked_rule": selected_budget,
        "familywise_rare_event_alpha": .05,
        "layout_level_sampling_unit": "layout_sha256",
    }


def _append_episode(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def run_locked(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    cfg = _load_config()
    lock, audit, order = _verify_lock(root, cfg)
    output = root / cfg["outputs"]["proposed_run_dir"]
    episode_path = root / cfg["outputs"]["episodes"]
    allowed = {root / cfg["outputs"]["layout_audit"],
               root / cfg["outputs"]["execution_order"]}
    if output.exists():
        existing = {p.resolve() for p in output.rglob("*") if p.is_file()}
        unexpected = existing - {p.resolve() for p in allowed}
        if unexpected and unexpected != {episode_path.resolve()}:
            raise Phase16EError("refusing to overwrite or resume unexpected Phase16E artifacts")
    else:
        output.mkdir(parents=True)

    scheduled = {_episode_key(row): row for row in order}
    if len(scheduled) != len(order):
        raise Phase16EError("duplicate key in Phase16E execution schedule")
    completed: dict[tuple[Any, ...], dict[str, Any]] = {}
    if episode_path.exists():
        for line in episode_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = _episode_key(row)
            if key in completed or key not in scheduled:
                raise Phase16EError("duplicate or unexpected row in Phase16E episode file")
            if row.get("layout_sha256") != scheduled[key]["layout_sha256"]:
                raise Phase16EError("episode layout hash differs from frozen schedule")
            completed[key] = row

    hashes_by_seed = {int(row["seed"]): row["layout_sha256"] for row in audit["layouts"]}
    started = time.perf_counter()
    for arm in order:
        key = _episode_key(arm)
        if key in completed:
            continue
        scenario = generate_staggered_corridor(int(arm["seed"]))
        if scenario.layout_sha256 != hashes_by_seed[int(arm["seed"])] \
                or scenario.layout_sha256 != arm["layout_sha256"]:
            raise Phase16EError("runtime layout differs from the hash-only audit")
        if arm["method"] == "uniform_shooting":
            row = evaluate_shooting(scenario, int(arm["budget"]),
                                    int(arm["replica"]), cfg)
        elif arm["method"] == "geometry_waypoint":
            row = _geometry_episode(scenario, cfg)
        else:
            raise Phase16EError(f"unexpected scheduled method: {arm['method']}")
        if _episode_key(row) != key:
            raise Phase16EError("evaluated episode identity differs from schedule")
        _append_episode(episode_path, row)
        completed[key] = row

    expected_count = int(cfg["execution"]["expected_unique_episode_count"])
    if len(completed) != expected_count:
        raise Phase16EError(f"expected {expected_count} unique episodes, got {len(completed)}")
    result = summarize(list(completed.values()), cfg)
    result.update({
        "protocol_sha256": sha256(PROTOCOL),
        "config_sha256": sha256(CONFIG),
        "lock_sha256": sha256(LOCK),
        "layout_audit_sha256": sha256(root / cfg["outputs"]["layout_audit"]),
        "execution_order_sha256": sha256(root / cfg["outputs"]["execution_order"]),
        "source_commit": lock["source_commit"],
        "run_commit": _git(root, "rev-parse", "HEAD"),
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "hardware": {"cpu_count": __import__("os").cpu_count()},
        "run_wall_seconds": time.perf_counter() - started,
        "scope": {"training": False, "learned_prior": False,
                  "learned_world_model": False, "dagger": False,
                  "student_control": False, "adaptive_planning": False,
                  "d168_access": False, "reserve_access": False, "test_access": False},
    })
    result_path = root / cfg["outputs"]["result"]
    manifest_path = root / cfg["outputs"]["manifest"]
    if result_path.exists() or manifest_path.exists():
        raise Phase16EError("refusing to overwrite an existing Phase16E result or manifest")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "phase16e-manifest-v1", "status": result["status"],
        "lock_sha256": result["lock_sha256"], "config_sha256": result["config_sha256"],
        "protocol_sha256": result["protocol_sha256"], "layout_count": result["layout_count"],
        "episode_count": len(completed), "episode_sha256": sha256(episode_path),
        "result_sha256": sha256(result_path),
        "layout_audit_sha256": result["layout_audit_sha256"],
        "execution_order_sha256": result["execution_order_sha256"],
        "source_commit": result["source_commit"], "run_commit": result["run_commit"],
        "scope": result["scope"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return {**result, "manifest_sha256": sha256(manifest_path)}


def freeze_lock(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    cfg = _load_config()
    if LOCK.exists():
        raise Phase16EError("refusing to overwrite immutable Phase16E execution lock")
    if _git(root, "status", "--porcelain"):
        raise Phase16EError("commit Phase16E protocol, code, tests, and authorization before locking")
    _verify_parents(root, cfg)
    audit_path = root / cfg["outputs"]["layout_audit"]
    order_path = root / cfg["outputs"]["execution_order"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    order = json.loads(order_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "hash_only_passed_no_outcomes"
            or audit.get("outcomes_or_transitions_computed") is not False
            or len(order) != int(cfg["execution"]["expected_unique_episode_count"])
            or order != execution_order(audit["layouts"], cfg["execution"]["counterbalance_seed"], cfg)):
        raise Phase16EError("Phase16E audit or execution schedule failed lock preflight")
    for relative, expected in audit["historical_source_hashes"].items():
        if sha256(root / relative) != expected:
            raise Phase16EError(f"historical source changed after audit: {relative}")
    for relative, expected in audit["source_sha256"].items():
        if sha256(root / relative) != expected:
            raise Phase16EError(f"audit source changed: {relative}")
    sources = _source_paths()
    missing = [relative for relative in sources if not (root / relative).is_file()]
    if missing:
        raise Phase16EError(f"missing Phase16E lock sources: {missing}")
    lock = {
        "schema": "phase16e-execution-lock-v1",
        "status": "locked_approved_development_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "source_sha256": {relative: sha256(root / relative) for relative in sources},
        "config_sha256": sha256(CONFIG), "protocol_sha256": sha256(PROTOCOL),
        "layout_audit_sha256": sha256(audit_path),
        "execution_order_sha256": sha256(order_path),
        "parents": cfg["parents"],
        "authorized_development_block": {
            "seed_start": int(cfg["environment"]["proposed_seed_start"]),
            "layout_count": int(cfg["environment"]["proposed_layout_count"]),
            "layout_sha256": [row["layout_sha256"] for row in audit["layouts"]],
        },
        "reserve_access": False, "test_access": False, "d168_access": False,
        "training": False, "learned_prior": False, "student_control": False,
        "dagger": False, "learned_world_model": False,
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"lock_path": str(LOCK.relative_to(root)), "lock_sha256": sha256(LOCK),
            "source_commit": lock["source_commit"], "layouts": len(audit["layouts"]),
            "scheduled_episodes": len(order)}
