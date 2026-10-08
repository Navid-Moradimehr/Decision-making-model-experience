"""Train-only audit of stochastic planner-teacher and task-metric integrity."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.inertial_corridor import ACTION_VECTORS, CorridorState, corridor_step
from consequence_lab.phase12_teacher import (
    evaluate_candidate_set, phase12_query_seed, propose_sequences, state_features,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase16_contract_repair.yaml"
ROWS = ROOT / "artifacts/runs/phase15_d153_coverage_scaling_v1/fresh_training_rows.npz"
RUN_DIR = ROOT / "artifacts/runs/phase16_contract_repair_v1"
PROGRESS = RUN_DIR / "states.jsonl"
LOCK = ROOT / "experiments/phase16a_contract_integrity.lock.json"
LAYOUT_COUNT, REPLICATES, SEED_STRIDE, RNG_SEED = 64, 3, 104729, 169001
STATE_TOLERANCE = SUMMARY_TOLERANCE = 1e-6


class ContractAuditError(RuntimeError):
    """Fail-closed data, parity or run-provenance violation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _status_key(status: str) -> int:
    return {"success": 0, "active": 1, "timeout": 2, "collision": 3}[status]


def summary_from_record(record: dict[str, Any], scenario: CorridorScenario) -> np.ndarray:
    result = np.zeros(7, dtype=np.float64)
    result[{"success": 0, "active": 1, "timeout": 2, "collision": 3}[record["status"]]] = 1.0
    result[4] = float(record["steps"]) / scenario.spec.max_steps
    result[5] = float(record["distance"]) / math.hypot(scenario.spec.width, scenario.spec.height)
    result[6] = float(record["effort"]) / scenario.spec.max_steps
    return result


def _replicate_seed(scenario_seed: int, elapsed: int, replicate: int) -> int:
    if replicate == 0:
        return phase12_query_seed(scenario_seed, elapsed)
    return phase12_query_seed(scenario_seed + replicate * SEED_STRIDE, elapsed)


def evaluate_forced_action(scenario: CorridorScenario, source: CorridorState,
                           action_id: int, replicate: int) -> dict[str, Any]:
    """Force one action, then replan with shared-per-time randomized banks."""
    if source.status != "active" or action_id not in range(len(ACTION_VECTORS)):
        raise ContractAuditError("source must be active and action must be legal")
    current = corridor_step(source, int(action_id), scenario.spec).state
    effort, queries, sim_steps, continuation = int(action_id != 0), 0, 1, []
    while current.status == "active":
        bank = propose_sequences(
            current, scenario.spec.max_steps,
            seed=_replicate_seed(scenario.seed, current.elapsed_steps, replicate),
        )
        query = evaluate_candidate_set(current, scenario, bank)
        sim_steps += int(np.count_nonzero(query.rollout_state_mask) - len(bank))
        chosen = int(bank[query.selected_index, 0])
        continuation.append(chosen)
        effort += int(chosen != 0)
        current = corridor_step(current, chosen, scenario.spec).state
        queries += 1
        if queries > scenario.spec.max_steps:
            raise ContractAuditError("teacher continuation exceeded task horizon")
    if current.status not in {"success", "collision", "timeout"}:
        raise ContractAuditError("unsupported terminal state")
    distance = math.hypot(scenario.spec.goal[0] - current.x,
                          scenario.spec.goal[1] - current.y)
    steps = current.elapsed_steps - source.elapsed_steps
    return {
        "status": current.status, "steps": steps, "distance": distance, "effort": effort,
        "objective": [_status_key(current.status), steps if current.status == "success" else distance,
                      effort],
        "continuation_queries": queries,
        "simulator_transitions": sim_steps,
        "continuation_trace_sha256": hashlib.sha256(
            np.asarray(continuation, dtype=np.uint8).tobytes()).hexdigest().upper(),
        "terminal_state": [current.x, current.y, current.vx, current.vy],
    }


def _selected_training_states(rows_path: Path = ROWS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(rows_path, allow_pickle=False) as archive:
        required = {"x_R0", "summary_R4", "layouts", "seeds", "decision_index",
                    "collector_action_ids_audit_only"}
        if set(archive.files) != required:
            raise ContractAuditError("training data schema differs from frozen D153 schema")
        x_r0 = archive["x_R0"].astype(np.float64)
        summaries = archive["summary_R4"].astype(np.float64)
        layouts, seeds = archive["layouts"].astype(str), archive["seeds"].astype(np.int64)
        decisions = archive["decision_index"].astype(np.int64)
        collector_actions = archive["collector_action_ids_audit_only"].astype(np.int64)
    if x_r0.shape != (1365, 61) or summaries.shape != (1365, 9, 7):
        raise ContractAuditError("D153 train corpus dimensions changed")
    unique_seeds = np.asarray(sorted(set(seeds.tolist())), dtype=np.int64)
    if len(unique_seeds) != 140:
        raise ContractAuditError("D153 train corpus must contain 140 layouts")
    rng = np.random.default_rng(RNG_SEED)
    chosen_seeds = sorted(rng.choice(unique_seeds, size=LAYOUT_COUNT, replace=False).tolist())
    selected = []
    for seed in chosen_seeds:
        ids = np.flatnonzero(seeds == seed)
        ids = ids[np.argsort(decisions[ids], kind="stable")]
        if len(ids) < 2 or not np.array_equal(decisions[ids], np.arange(len(ids))):
            raise ContractAuditError(f"invalid replay row sequence for layout {seed}")
        split = max(1, len(ids) // 2)
        chosen_rows = (int(rng.choice(ids[:split])), int(rng.choice(ids[split:])))
        scenario = generate_staggered_corridor(int(seed))
        if set(layouts[ids].tolist()) != {scenario.layout_sha256}:
            raise ContractAuditError(f"layout hash mismatch for seed {seed}")
        states, state = {}, scenario.initial_state
        for row_id in ids:
            observed = state_features(scenario, state)
            if np.max(np.abs(observed - x_r0[row_id, :52])) > STATE_TOLERANCE:
                raise ContractAuditError(f"state replay mismatch: seed={seed}, row={row_id}")
            if int(row_id) in chosen_rows:
                states[int(row_id)] = state
            state = corridor_step(state, int(collector_actions[row_id]), scenario.spec).state
        for row_id in chosen_rows:
            selected.append({"row_id": row_id, "seed": int(seed),
                             "layout_hash": scenario.layout_sha256,
                             "decision_index": int(decisions[row_id]),
                             "state": states[row_id], "scenario": scenario,
                             "saved_summary": summaries[row_id]})
    return selected, {"selected_layout_count": len(chosen_seeds),
                      "selected_state_count": len(selected), "selected_seeds": chosen_seeds,
                      "selection_seed": RNG_SEED, "training_rows_sha256": sha256(rows_path)}


def _state_record(item: dict[str, Any]) -> dict[str, Any]:
    scenario, state = item["scenario"], item["state"]
    forced = [[evaluate_forced_action(scenario, state, action, rep)
               for action in range(9)] for rep in range(REPLICATES)]
    parity = np.stack([summary_from_record(entry, scenario) for entry in forced[0]])
    parity_error = float(np.max(np.abs(parity - item["saved_summary"])))
    if parity_error > SUMMARY_TOLERANCE:
        raise ContractAuditError(
            f"replicate-zero summary parity failed seed={item['seed']} "
            f"decision={item['decision_index']}: {parity_error}")
    return {"row_id": item["row_id"], "seed": item["seed"],
            "layout_hash": item["layout_hash"], "decision_index": item["decision_index"],
            "source_state": [state.x, state.y, state.vx, state.vy, state.elapsed_steps],
            "replicate_query_seeds": [_replicate_seed(scenario.seed, state.elapsed_steps, rep)
                                       for rep in range(REPLICATES)],
            "replicate_zero_parity_max_error": parity_error, "forced_actions": forced}


def _objective(record: dict[str, Any]) -> tuple[int, float, int]:
    return tuple(record["objective"])


def _best_indices(records: list[dict[str, Any]]) -> set[int]:
    keys = [_objective(record) for record in records]
    best = min(keys)
    return {idx for idx, key in enumerate(keys) if key == best}


def _rank_vector(records: list[dict[str, Any]]) -> np.ndarray:
    keys = [_objective(record) for record in records]
    ranks = np.asarray([sum(other < key for other in keys)
                        + (sum(other == key for other in keys) + 1) / 2
                        for key in keys], dtype=np.float64)
    return (ranks - ranks.min()) / max(1, len(keys) - 1)


def _cluster_ci(values: dict[str, list[float]], seed: int, draws: int = 10000) -> list[float]:
    names = sorted(values)
    means = np.asarray([np.mean(values[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(names), size=(draws, len(names)))
    boot = means[sampled].mean(axis=1)
    return [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    layout_values: dict[str, dict[str, list[float]]] = {}
    for row in records:
        bucket = layout_values.setdefault(row["layout_hash"], {
            "outcome_disagreement": [], "best_jaccard": [], "pair_agreement": [],
            "replicate0_collision_action_fraction": [],
            "replicate0_success_action_fraction": [], "replicate0_timeout_action_fraction": [],
            "replicate0_mean_rank_penalty": []})
        reps = row["forced_actions"]
        disagreement = [np.mean([reps[i][a]["status"] != reps[j][a]["status"]
                                 for a in range(9)])
                        for i in range(REPLICATES) for j in range(i + 1, REPLICATES)]
        best = [_best_indices(rep) for rep in reps]
        jaccard = [len(best[i] & best[j]) / len(best[i] | best[j])
                   for i in range(REPLICATES) for j in range(i + 1, REPLICATES)]
        pair_agreement = []
        for i in range(REPLICATES):
            for j in range(i + 1, REPLICATES):
                scores = []
                for a in range(9):
                    for b in range(a + 1, 9):
                        left = (_objective(reps[i][a]) > _objective(reps[i][b])) - (
                            _objective(reps[i][a]) < _objective(reps[i][b]))
                        right = (_objective(reps[j][a]) > _objective(reps[j][b])) - (
                            _objective(reps[j][a]) < _objective(reps[j][b]))
                        scores.append(float(left == right))
                pair_agreement.append(float(np.mean(scores)))
        base = reps[0]
        bucket["outcome_disagreement"].append(float(np.mean(disagreement)))
        bucket["best_jaccard"].append(float(np.mean(jaccard)))
        bucket["pair_agreement"].append(float(np.mean(pair_agreement)))
        bucket["replicate0_collision_action_fraction"].append(
            float(np.mean([x["status"] == "collision" for x in base])))
        bucket["replicate0_success_action_fraction"].append(
            float(np.mean([x["status"] == "success" for x in base])))
        bucket["replicate0_timeout_action_fraction"].append(
            float(np.mean([x["status"] == "timeout" for x in base])))
        bucket["replicate0_mean_rank_penalty"].append(float(np.mean(_rank_vector(base))))
    metrics = {}
    for name in next(iter(layout_values.values())):
        values = {layout: bucket[name] for layout, bucket in layout_values.items()}
        layout_means = np.asarray([np.mean(state_values)
                                   for state_values in values.values()], dtype=np.float64)
        metrics[name] = {"mean": float(np.mean(layout_means)),
                         "median": float(np.median(layout_means)),
                         "layout_bootstrap_95_ci": _cluster_ci(values, RNG_SEED + len(name))}
    metrics["layout_count"] = len(layout_values)
    metrics["state_count"] = len(records)
    return metrics


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def verify_lock() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase16a-lock-v1":
        raise ContractAuditError("unsupported or missing Phase 16A lock")
    for relative, expected in lock["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise ContractAuditError(f"locked Phase 16A input changed: {relative}")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise ContractAuditError("working tree must be clean before Phase 16A scoring")
    return lock


def run_smoke() -> dict[str, Any]:
    """Run the preregistered two-state, frozen-seed parity preflight."""
    selected, _ = _selected_training_states()
    errors = []
    for item in selected[:2]:
        records = [evaluate_forced_action(item["scenario"], item["state"], action, 0)
                   for action in range(9)]
        summaries = np.stack([summary_from_record(record, item["scenario"])
                              for record in records])
        errors.append(float(np.max(np.abs(summaries - item["saved_summary"]))))
    maximum = max(errors)
    if maximum > SUMMARY_TOLERANCE:
        raise ContractAuditError(f"replicate-zero parity failed: {maximum}")
    result = {"status": "passed", "states": len(errors),
              "max_summary_error": maximum, "tolerance": SUMMARY_TOLERANCE,
              "reserve_access": False, "test_access": False}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RUN_DIR / "smoke_parity.json", result)
    return result


def run_full() -> dict[str, Any]:
    lock = verify_lock()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__).resolve(),
                    ROOT / "src/consequence_lab/phase12_teacher.py",
                    ROOT / "src/consequence_lab/inertial_corridor.py",
                    ROOT / "src/consequence_lab/corridor_tasks.py"]
    manifest = {"schema": "phase16a-manifest-v1", "config_sha256": sha256(CONFIG),
                "training_rows_sha256": sha256(ROWS), "parent_commit": "350a578",
                "phase_protocol_commit": lock["protocol_commit"],
                "run_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                              check=True, capture_output=True,
                                              text=True).stdout.strip(),
                "lock_sha256": sha256(LOCK),
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path)
                                  for path in source_paths},
                "reserve_access": False, "test_access": False,
                "fresh_layout_outcomes": False, "student_closed_loop": False,
                "dagger": False, "environment_calls": 0}
    manifest_path = RUN_DIR / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ContractAuditError("existing run manifest differs from frozen config/data")
    if not manifest_path.exists():
        _write_json(manifest_path, manifest)
    selected, selection = _selected_training_states()
    known: dict[int, dict[str, Any]] = {}
    if PROGRESS.exists():
        for line in PROGRESS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["row_id"] in known:
                    raise ContractAuditError("duplicate resumable state row")
                known[row["row_id"]] = row
    start = time.time()
    for item in selected:
        if item["row_id"] in known:
            continue
        row = _state_record(item)
        with PROGRESS.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
            output.flush()
        known[row["row_id"]] = row
        if len(known) % 4 == 0:
            print(f"completed {len(known)}/{len(selected)} states; elapsed {time.time()-start:.1f}s",
                  flush=True)
    records = list(known.values())
    if len(records) != len(selected):
        raise ContractAuditError("resumable run is missing state records")
    result = {"schema": "phase16a-result-v1", "status": "complete",
              "selection": selection, "teacher_stability": summarize(records),
              "runtime_seconds_this_invocation": time.time() - start,
              "config_sha256": manifest["config_sha256"],
              "training_rows_sha256": manifest["training_rows_sha256"],
              "reserve_access": False, "test_access": False,
              "fresh_layout_outcomes": False, "student_closed_loop": False,
              "dagger": False, "environment_calls": 0,
              "simulator_transitions": int(sum(
                  entry["simulator_transitions"] for row in records
                  for replicate in row["forced_actions"] for entry in replicate)),
              "planner_queries": int(sum(
                  entry["continuation_queries"] for row in records
                  for replicate in row["forced_actions"] for entry in replicate)),
              "progress_jsonl_sha256": sha256(PROGRESS)}
    _write_json(RUN_DIR / "result.json", result)
    return result


if __name__ == "__main__":
    import sys
    result = run_smoke() if len(sys.argv) > 1 and sys.argv[1] == "smoke" else run_full()
    print(json.dumps(result, indent=2))
