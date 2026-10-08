"""Paired exact-shooting budget profile for the Inertial Corridor task."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import yaml

from consequence_lab import phase17_proposal_guidance as p17
from consequence_lab import phase18_proposal_controls as p18
from consequence_lab import phase20_exact_plan_warmstart as p20
from consequence_lab import phase23_postrun_score as p23score
from consequence_lab import phase23_structured_screen as p23
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.phase12_teacher import state_features
from consequence_lab.phase16c_prior_guided import evaluate_sequence, layout_hash_audit
from consequence_lab.phase16d_compute_frontier import binomial_upper
from consequence_lab.inertial_corridor import corridor_step

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase24_search_budget_profile_v1.yaml"
METHODS = [
    {"name": "shifted_b16", "kind": "shift_previous_winner", "budget": 16, "replicas": 3},
    {"name": "shifted_b24", "kind": "shift_previous_winner", "budget": 24, "replicas": 3},
    {"name": "shifted_b32", "kind": "shift_previous_winner", "budget": 32, "replicas": 3},
    {"name": "shifted_b48", "kind": "shift_previous_winner", "budget": 48, "replicas": 3},
]


def config(root: Path = ROOT) -> dict:
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def validate_config(cfg: dict) -> None:
    if cfg.get("schema") != "phase24-search-budget-profile-v1":
        raise ValueError("wrong Phase24 config schema")
    block, plan = cfg["block"], cfg["planning"]
    if (block["seed_start"], block["layout_count"], block["replicas"]) != (209000, 500, [0, 1, 2]):
        raise ValueError("Phase24 development block changed")
    if plan["budgets"] != [16, 24, 32, 48] or plan["horizon"] != 8:
        raise ValueError("Phase24 budget/horizon roster changed")
    if len(METHODS) * 500 * 3 != 6000:
        raise RuntimeError("Phase24 schedule contract is invalid")
    if cfg["analysis"]["success_difference_one_sided_confidence"] != 0.9875:
        raise ValueError("Phase24 multiplicity adjustment changed")
    if cfg["scope"]["reserve_test_d168_access"] or cfg["scope"]["training"]:
        raise ValueError("Phase24 cannot access sealed data or train a model")


def assert_unsealed(start: int, count: int, block: dict) -> None:
    if start < 0 or count < 1:
        raise ValueError("invalid Phase24 layout range")
    for key, interval in block.items():
        if key.endswith("_half_open"):
            lo, hi = map(int, interval)
            if start < hi and lo < start + count:
                raise ValueError(f"Phase24 layout range intersects {key}")


def execution_order(layouts: list[dict], cfg: dict) -> list[dict]:
    rows = []
    for layout in layouts:
        arms = [{**layout, "method": method["name"], "budget": method["budget"], "replica": replica}
                for method in METHODS for replica in range(3)]
        random.Random(int(cfg["analysis"]["schedule_seed"]) + int(layout["seed"])).shuffle(arms)
        for local, arm in enumerate(arms):
            rows.append({**arm, "global_schedule_index": len(rows), "local_order_index": local})
    if len(rows) != len(layouts) * 12:
        raise RuntimeError("Phase24 schedule must contain exactly 6,000 episodes")
    if len({(r["seed"], r["method"], r["replica"]) for r in rows}) != len(rows):
        raise RuntimeError("Phase24 schedule contains duplicate identities")
    return rows


def shifted_bank(horizon: int, budget: int, query: int, previous: tuple[int, ...] | None,
                 attempt_cap: int = 1_280_000):
    kind = "shift_previous_winner" if previous is not None else "uniform"
    bank, counters = p20.candidate_bank(horizon, budget, query, kind, previous, 0, 0, attempt_cap)
    return bank, counters


def assert_nested_banks(horizon: int, query: int, previous: tuple[int, ...] | None) -> None:
    banks = [shifted_bank(horizon, budget, query, previous)[0] for budget in (16, 24, 32, 48)]
    for small, large in zip(banks, banks[1:]):
        if not set(small).issubset(set(large)):
            raise RuntimeError("Phase24 fixed-state shifted candidate banks are not nested")


def _decision(scenario, state, previous, replica: int, budget: int, cfg: dict):
    started = time.perf_counter()
    horizon = min(int(cfg["planning"]["horizon"]), scenario.spec.max_steps - state.elapsed_steps)
    query = p17.query_seed(scenario.seed, state.elapsed_steps, replica)
    bank, counters = shifted_bank(horizon, budget, query, previous, int(cfg["planning"]["attempt_cap"]))
    evaluations = [evaluate_sequence(state, scenario, seq) for seq in bank]
    chosen, ties = p20.choose_lexicographic_best(evaluations)
    elapsed = time.perf_counter() - started
    successor = corridor_step(state, chosen.sequence[0], scenario.spec).state
    categories = {0: "success", 1: "active", 2: "timeout", 3: "collision"}
    candidates = [{"sequence": list(ev.sequence), "objective": list(ev.objective),
                   "transitions": int(ev.transitions), "category": categories[int(ev.objective[0])],
                   "success_reached": bool(ev.success_reached)} for ev in evaluations]
    row = {"decision_index": int(state.elapsed_steps), "query_seed": int(query),
           "horizon": int(horizon), "nominal_budget": int(budget), "effective_budget": len(bank),
           "previous_plan": list(previous) if previous is not None else None,
           "state_raw52": state_features(scenario, state).astype(float).tolist(),
           "chosen_sequence": list(chosen.sequence), "chosen_action": int(chosen.sequence[0]),
           "best_objective": list(chosen.objective), "tie_count": int(ties),
           "exact_transitions": int(sum(ev.transitions for ev in evaluations)),
           "base_duplicate_rejections": int(counters["base_duplicate_rejections"]),
           "fallback_sequences": int(counters["fallback_sequences"]),
           "candidate_bank": candidates, "latency_seconds": float(elapsed)}
    return successor, chosen.sequence, row


def evaluate_episode(scenario, method: dict, replica: int, cfg: dict) -> dict:
    state, previous = scenario.initial_state, None
    decisions = []
    while state.status == "active":
        state, selected, decision = _decision(scenario, state, previous, replica,
                                              int(method["budget"]), cfg)
        decisions.append(decision)
        previous = selected
    return {"seed": scenario.seed, "layout_sha256": scenario.layout_sha256,
            "method": method["name"], "budget": int(method["budget"]), "replica": replica,
            "outcome": state.status, "steps": int(state.elapsed_steps),
            "decision_count": len(decisions), "exact_sequences": sum(d["effective_budget"] for d in decisions),
            "exact_transitions": sum(d["exact_transitions"] for d in decisions),
            "latency_seconds_per_decision": [d["latency_seconds"] for d in decisions],
            "wall_seconds": sum(d["latency_seconds"] for d in decisions),
            "decision_diagnostics": decisions}


def _tracked_dependencies(root: Path, cfg: dict) -> list[str]:
    tracked = subprocess.check_output(["git", "ls-files", "src", "tests", "configs"],
                                      cwd=root, text=True).splitlines()
    sources = [p for p in tracked if p.endswith((".py", ".yaml"))]
    required = [CONFIG, cfg["protocol"], "second-brain/phases/Phase_24_Search_Budget_Profile.md",
                "src/consequence_lab/phase24_search_budget_profile.py",
                "reports/phase24_existing_record_audit.md",
                "reports/phase23_structured_screen_protocol.md",
                "reports/phase23_replay_result.md",
                cfg["parent"]["phase23_lock"], cfg["parent"]["phase23_result"],
                cfg["parent"]["phase23_manifest"], cfg["parent"]["phase23_records"],
                "artifacts/runs/phase23_structured_screen_v1/layout_audit.json",
                "artifacts/runs/phase23_structured_screen_v1/execution_order.json"]
    return sorted(set(sources + required + [p for p in ("pyproject.toml", "uv.lock") if (root / p).exists()]))


def freeze(root: Path = ROOT) -> dict:
    cfg = config(root)
    validate_config(cfg)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit Phase24 protocol, config, implementation, and tests before freezing")
    p23.verify(root)
    p23score.audit(root)
    start, count = int(cfg["block"]["seed_start"]), int(cfg["block"]["layout_count"])
    assert_unsealed(start, count, cfg["block"])
    out, lock_path = root / cfg["outputs"]["run_dir"], root / cfg["outputs"]["execution_lock"]
    if out.exists() or lock_path.exists():
        raise FileExistsError("Phase24 outputs or execution lock already exist")
    audit = layout_hash_audit(start, count, root)
    if audit["layout_count"] != count or audit["unique_hashes"] != count or audit["overlap_historical"] or audit["overlap_outcome_bearing"]:
        raise RuntimeError("Phase24 fresh development layout audit failed")
    schedule = execution_order(audit["layouts"], cfg)
    if len(schedule) != 6000:
        raise RuntimeError("Phase24 frozen block did not produce the declared 6,000 episodes")
    out.mkdir(parents=True)
    p18.write_new(out / "layout_audit.json", audit)
    p18.write_new(out / "execution_order.json", schedule)
    lock = {"schema": "phase24-search-budget-profile-lock-v1",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "created_utc": datetime.now(timezone.utc).isoformat(), "layout_count": count,
            "schedule_count": len(schedule), "layout_audit_sha256": p17.sha256_file(out / "layout_audit.json"),
            "schedule_sha256": p17.sha256_file(out / "execution_order.json"),
            "hashes": {p: p17.sha256_file(root / p) for p in _tracked_dependencies(root, cfg)},
            "reserve_access": False, "test_access": False, "d168_access": False,
            "training": False, "learned_model_loading": False}
    p18.write_new(lock_path, lock)
    return {"layouts": count, "episodes": len(schedule), "lock_sha256": p17.sha256_file(lock_path),
            "layout_audit_sha256": lock["layout_audit_sha256"], "schedule_sha256": lock["schedule_sha256"]}


def verify(root: Path = ROOT):
    cfg = config(root)
    validate_config(cfg)
    lock_path = root / cfg["outputs"]["execution_lock"]
    committed = subprocess.check_output(["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"], cwd=root)
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("Phase24 execution lock must be committed unchanged")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for path, digest in lock["hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"Phase24 locked dependency changed: {path}")
    p23.verify(root)
    p23score.audit(root)
    out = root / cfg["outputs"]["run_dir"]
    audit = json.loads((out / "layout_audit.json").read_text(encoding="utf-8"))
    schedule = json.loads((out / "execution_order.json").read_text(encoding="utf-8"))
    if p17.sha256_file(out / "layout_audit.json") != lock["layout_audit_sha256"]:
        raise RuntimeError("Phase24 layout audit hash differs from lock")
    if p17.sha256_file(out / "execution_order.json") != lock["schedule_sha256"]:
        raise RuntimeError("Phase24 schedule hash differs from lock")
    if schedule != execution_order(audit["layouts"], cfg):
        raise RuntimeError("Phase24 schedule differs from its deterministic reconstruction")
    return cfg, out, schedule, p17.sha256_file(lock_path)


def run(root: Path = ROOT) -> dict:
    cfg, out, schedule, lock_hash = verify(root)
    raw = out / "episodes.jsonl"
    manifest_path = out / "run_manifest.json"
    if raw.exists() or manifest_path.exists():
        raise FileExistsError("Phase24 run outputs already exist; execution is append-only and non-resumable")
    method_map = {m["name"]: m for m in METHODS}
    started = time.perf_counter()
    with raw.open("x", encoding="utf-8", newline="\n") as stream:
        for index, arm in enumerate(schedule):
            scenario = generate_staggered_corridor(int(arm["seed"]))
            row = evaluate_episode(scenario, method_map[arm["method"]], int(arm["replica"]), cfg)
            row.update({k: arm[k] for k in ("global_schedule_index", "local_order_index")})
            row["execution_lock_sha256"] = lock_hash
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            if (index + 1) % 250 == 0:
                print(json.dumps({"completed": index + 1, "total": len(schedule),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    manifest = {"schema": "phase24-search-budget-profile-manifest-v1",
                "execution_lock_sha256": lock_hash, "records_sha256": p17.sha256_file(raw),
                "episodes": len(schedule), "layouts": cfg["block"]["layout_count"],
                "runtime_seconds": time.perf_counter() - started,
                "python": platform.python_version(), "numpy": np.__version__,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "reserve_access": False, "test_access": False, "d168_access": False,
                "training": False}
    p18.write_new(manifest_path, manifest)
    return manifest


def _load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(root: Path = ROOT) -> dict:
    cfg, out, schedule, lock_hash = verify(root)
    raw, result_path = out / "episodes.jsonl", out / "result.json"
    if result_path.exists():
        raise FileExistsError("Phase24 result already exists; refusing overwrite")
    records = _load_records(raw)
    if len(records) != len(schedule):
        raise RuntimeError("Phase24 raw episodes are incomplete; do not score partial execution")
    p18.validate_prefix(records, schedule, lock_hash)
    for row, arm in zip(records, schedule, strict=True):
        if any(row.get(key) != value for key, value in arm.items()):
            raise RuntimeError("Phase24 record identity differs from frozen schedule")
        if row.get("outcome") not in {"success", "collision", "timeout"}:
            raise RuntimeError("invalid Phase24 episode outcome")
        for d in row["decision_diagnostics"]:
            bank = d["candidate_bank"]
            if len(bank) != d["effective_budget"] or len({tuple(c["sequence"]) for c in bank}) != len(bank):
                raise RuntimeError("candidate bank is incomplete or contains duplicates")
            if min(tuple(c["objective"]) for c in bank) != tuple(d["best_objective"]):
                raise RuntimeError("logged selected objective is not the exact bank minimum")
    first_decisions = {(int(r["seed"]), int(r["replica"]), int(r["budget"])):
                       r["decision_diagnostics"][0] for r in records if r["decision_diagnostics"]}
    for seed in sorted({int(r["seed"]) for r in records}):
        for replica in range(3):
            rows_by_budget = {budget: first_decisions[(seed, replica, budget)] for budget in (16, 24, 32, 48)}
            banks = {budget: {tuple(c["sequence"]): tuple(c["objective"])
                              for c in row["candidate_bank"]}
                     for budget, row in rows_by_budget.items()}
            for low, high in ((16, 24), (24, 32), (32, 48)):
                if not set(banks[low]).issubset(banks[high]):
                    raise RuntimeError("first-decision candidate banks violate the nested-budget contract")
                if any(banks[high][seq] != obj for seq, obj in banks[low].items()):
                    raise RuntimeError("shared candidate exact objective changed across budget arms")
    seeds = sorted({int(r["seed"]) for r in records})
    keyed = {(r["seed"], r["method"], r["replica"]): r for r in records}
    if len(keyed) != len(records):
        raise RuntimeError("duplicate Phase24 episode identities")
    vectors, per_replica, methods = {}, {}, {}
    for method in METHODS:
        name = method["name"]
        rows = [r for r in records if r["method"] == name]
        methods[name] = {"episodes": len(rows),
            "success_count": sum(r["outcome"] == "success" for r in rows),
            "collision_count": sum(r["outcome"] == "collision" for r in rows),
            "timeout_count": sum(r["outcome"] == "timeout" for r in rows),
            "success_rate": float(np.mean([r["outcome"] == "success" for r in rows])),
            "collision_rate": float(np.mean([r["outcome"] == "collision" for r in rows])),
            "timeout_rate": float(np.mean([r["outcome"] == "timeout" for r in rows])),
            "mean_exact_transitions_per_episode": float(np.mean([r["exact_transitions"] for r in rows])),
            "median_decision_latency_seconds": float(np.median([v for r in rows for v in r["latency_seconds_per_decision"]])),
            "p90_decision_latency_seconds": float(np.quantile([v for r in rows for v in r["latency_seconds_per_decision"]], .90))}
        per_replica[name] = {str(rep): {"success_rate": float(np.mean([keyed[(seed,name,rep)]["outcome"] == "success" for seed in seeds])),
            "collision_rate": float(np.mean([keyed[(seed,name,rep)]["outcome"] == "collision" for seed in seeds]))}
            for rep in range(3)}
        vectors[name] = {metric: [float(np.mean([float(keyed[(seed,name,rep)]["outcome"] == metric)
            if metric in {"success", "collision"} else keyed[(seed,name,rep)]["exact_transitions"] for rep in range(3)]))
            for seed in seeds] for metric in ("success", "collision", "transitions")}

    reference = "shifted_b32"
    b32_gate = cfg["gates"]["b32_absolute"]
    b32_absolute = (methods[reference]["success_rate"] > b32_gate["safe_success_strictly_above"]
        and methods[reference]["collision_rate"] <= b32_gate["collision_at_most"]
        and methods[reference]["timeout_rate"] == b32_gate["timeout_exactly"]
        and all(v["success_rate"] >= b32_gate["each_replica_success_at_least"]
                and v["collision_rate"] <= b32_gate["each_replica_collision_at_most"]
                for v in per_replica[reference].values()))
    comparisons = {}
    for name in ("shifted_b16", "shifted_b24", "shifted_b48"):
        success = p18.difference_bounds(vectors[reference]["success"], vectors[name]["success"], cfg,
            cfg["analysis"]["success_difference_one_sided_confidence"])
        collision = p18.difference_bounds(vectors[reference]["collision"], vectors[name]["collision"], cfg,
            cfg["analysis"]["collision_difference_one_sided_confidence"])
        ratio = np.asarray(vectors[name]["transitions"]) / np.asarray(vectors[reference]["transitions"])
        compute = p18.difference_bounds(np.zeros(len(seeds)), ratio.tolist(), cfg, .95)
        success_events = sum(any(keyed[(seed,name,rep)]["outcome"] != "success"
                                 and keyed[(seed,reference,rep)]["outcome"] == "success" for rep in range(3)) for seed in seeds)
        collision_events = sum(any(keyed[(seed,name,rep)]["outcome"] == "collision"
                                  and keyed[(seed,reference,rep)]["outcome"] != "collision" for rep in range(3)) for seed in seeds)
        event_conf = cfg["analysis"]["adverse_layout_exact_one_sided_confidence"]
        success_upper = binomial_upper(success_events, len(seeds), event_conf)
        collision_upper = binomial_upper(collision_events, len(seeds), event_conf)
        latency_ratio = methods[name]["median_decision_latency_seconds"] / methods[reference]["median_decision_latency_seconds"]
        checks = None
        if name in {"shifted_b16", "shifted_b24"}:
            gate = cfg["gates"]["lower_budget"]
            absolute = (methods[name]["success_rate"] > gate["safe_success_strictly_above"]
                and methods[name]["collision_rate"] <= gate["collision_at_most"]
                and methods[name]["timeout_rate"] == gate["timeout_exactly"]
                and all(v["success_rate"] >= gate["each_replica_success_at_least"] and
                        v["collision_rate"] <= gate["each_replica_collision_at_most"] for v in per_replica[name].values()))
            checks = {"absolute_reliability": absolute,
                "shifted_b32_reference_reproduced": b32_absolute,
                "success_noninferiority": success["adjusted_lower"] >= gate["adjusted_success_difference_lower_vs_b32"],
                "collision_noninferiority": collision["adjusted_upper"] <= gate["adjusted_collision_difference_upper_vs_b32"],
                "success_regression_layout_event": success_upper <= gate["success_regression_layout_event_upper_at_most"],
                "added_collision_layout_event": collision_upper <= gate["added_collision_layout_event_upper_at_most"],
                "transition_mean": compute["difference"] <= gate["mean_transition_ratio_vs_b32_at_most"],
                "transition_ci_upper": compute["ci95_upper"] <= gate["transition_ratio_95_upper_vs_b32_at_most"],
                "latency_ratio": latency_ratio <= gate["median_latency_ratio_vs_b32_at_most"]}
        comparisons[name] = {"success_difference_vs_b32": success, "collision_difference_vs_b32": collision,
            "transition_ratio_vs_b32": compute, "median_latency_ratio_vs_b32": latency_ratio,
            "success_regression_layouts": success_events, "success_regression_event_upper": success_upper,
            "added_collision_layouts": collision_events, "added_collision_event_upper": collision_upper,
            "promising_for_confirmation": bool(all(checks.values())) if checks is not None else None,
            "gate_checks": checks}

    result = {"schema": "phase24-search-budget-profile-result-v1", "layouts": len(seeds), "episodes": len(records),
        "methods": methods, "per_replica": per_replica, "comparisons": comparisons,
        "shifted_b32_reference_absolute_pass": b32_absolute,
        "profile_diagnostics": _profile_diagnostics(records),
        "decision": _phase_decision(comparisons, b32_absolute), "selection_authority": "exact_lexicographic_planner",
        "training": False, "reserve_access": False, "test_access": False, "d168_access": False,
        "records_sha256": p17.sha256_file(raw), "execution_lock_sha256": lock_hash}
    p18.write_new(result_path, result)
    return result


def _profile_diagnostics(records: list[dict]) -> dict:
    # Compare each budget's exact decision profile to the best available smaller budget
    # only within the same episode trajectory where states and incumbents are shared.
    # Cross-budget episodes diverge after their first differing executed action, so
    # only first-decision matched-bank comparisons are used for paired candidate analysis.
    first = {}
    for row in records:
        if row["decision_diagnostics"]:
            first[(row["seed"], row["replica"], int(row["budget"]))] = row["decision_diagnostics"][0]
    matched = {key[:2]: {budget: first[(key[0],key[1],budget)] for budget in (16,24,32,48)}
               for key in {(seed,rep) for seed,rep,_ in first}
               if all((key[0],key[1],b) in first for b in (16,24,32,48))}
    action_switches, category_upgrades, objective_improvements = {}, {}, {}
    for low, high in ((16,24),(24,32),(32,48)):
        switches = category = improvements = 0
        for row in matched.values():
            a, b = row[low], row[high]
            switches += int(a["chosen_action"] != b["chosen_action"])
            category += int(tuple(b["best_objective"])[0] < tuple(a["best_objective"])[0])
            improvements += int(tuple(b["best_objective"]) < tuple(a["best_objective"]))
        denom = max(1, len(matched))
        action_switches[f"{low}_to_{high}"] = switches / denom
        category_upgrades[f"{low}_to_{high}"] = category / denom
        objective_improvements[f"{low}_to_{high}"] = improvements / denom
    return {"matched_first_decision_state_count": len(matched),
            "first_decision_action_switch_fraction_by_budget_increment": action_switches,
            "first_decision_lexicographic_category_improvement_fraction": category_upgrades,
            "first_decision_any_lexicographic_improvement_fraction": objective_improvements,
            "interpretation_limit": "first-decision paired proposal diagnostic only; not counterfactual episode evidence"}


def _phase_decision(comparisons: dict, b32_absolute: bool) -> str:
    if not b32_absolute:
        return "audit_b32_reference_parity_before_interpreting_lower_budgets"
    passed = [name for name in ("shifted_b16", "shifted_b24")
              if comparisons[name]["promising_for_confirmation"]
              and comparisons[name]["gate_checks"].get("shifted_b32_reference_reproduced")]
    return f"confirm_{passed[0]}_on_a_separate_development_block" if passed else "retain_shifted_b32_and_stop_fixed_budget_reduction"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "summarize"))
    args = parser.parse_args()
    result = freeze() if args.command == "freeze" else run() if args.command == "run" else summarize()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
