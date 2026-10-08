"""Locked non-learning temporal warm-start controls for exact corridor search."""
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
from consequence_lab import phase19_uniform_confirmation as p19
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import ACTION_VECTORS, corridor_step
from consequence_lab.phase16c_prior_guided import evaluate_sequence, layout_hash_audit
from consequence_lab.phase16d_compute_frontier import binomial_upper, choose_lexicographic_best, uniform_prefix_bank

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase20_exact_plan_warmstart_v1.yaml"


def config(root=ROOT):
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def validate_config(cfg):
    expected = [("uniform_b32", "uniform", 32, 3, 0),
                ("shifted_b32", "shift_previous_winner", 32, 3, 1),
                ("mutated_b32", "shift_plus_one_action_mutations", 32, 3, 5),
                ("uniform_b48", "uniform", 48, 3, 0),
                ("geometry_waypoint", "geometry", None, 1, 0)]
    actual = [(m["name"], m["kind"], m["budget"], m["replicas"], m["warm_insertions"])
              for m in cfg["methods"]]
    if actual != expected:
        raise ValueError("Phase20 method arms/order differ from the locked study")
    if (cfg["block"]["episodes"] != cfg["block"]["proposed_layout_count"] * 13
            or cfg["block"]["replicas"] != [0, 1, 2]
            or cfg["planning"]["horizon"] != 8
            or cfg["planning"]["nominal_budgets"] != [32, 48]
            or cfg["planning"]["mandatory_constant_sequences"] != 9
            or cfg["planning"]["mutation_count"] != 4
            or cfg["planning"]["mutation_seed_xor"] != 0x51A7E20D
            or list(range(len(ACTION_VECTORS))) != cfg["actions"]["ids"]):
        raise ValueError("Phase20 frozen action, budget, or layout contract differs")
    if cfg["analysis"]["method_selection_order"] != ["uniform_b32", "shifted_b32", "mutated_b32"]:
        raise ValueError("Phase20 selection hierarchy differs")
    if cfg["analysis"]["warmstart_attribution"]["one_sided_bootstrap_confidence"] != 0.9875:
        raise ValueError("Phase20 warm-start attribution confidence differs")


def assert_unsealed_range(start: int, count: int, cfg: dict) -> None:
    if start < 0 or count < 1:
        raise ValueError("invalid development block")
    for key, bounds in cfg["block"].items():
        if key.endswith("_half_open"):
            low, high = bounds
            if start < high and low < start + count:
                raise ValueError(f"candidate block intersects {key}")


def execution_order(layouts: list[dict], cfg: dict) -> list[dict]:
    rows = []
    for layout in layouts:
        arms = [{**layout, "method": method["name"], "budget": method["budget"],
                 "replica": replica if method["replicas"] > 1 else None}
                for method in cfg["methods"] for replica in range(method["replicas"])]
        random.Random(cfg["analysis"]["order_seed"] + int(layout["seed"])).shuffle(arms)
        for local, arm in enumerate(arms):
            rows.append({**arm, "global_schedule_index": len(rows), "local_order_index": local})
    expected = int(cfg["block"]["episodes"])
    if len(rows) != expected:
        raise ValueError(f"locked schedule has {len(rows)} rows, expected {expected}")
    keys = [(r["seed"], r["layout_sha256"], r["method"], r["replica"]) for r in rows]
    if len(set(keys)) != expected:
        raise ValueError("locked schedule includes duplicate episode identity")
    return rows


def discordant_layout_count(keyed, seeds, candidate, reference, event):
    def is_event(candidate_outcome, reference_outcome):
        if event == "success_regression":
            return candidate_outcome != "success" and reference_outcome == "success"
        if event == "added_collision":
            return candidate_outcome == "collision" and reference_outcome != "collision"
        raise ValueError(f"unknown discordance event: {event}")
    return sum(any(is_event(keyed[(seed, candidate, replica)]["outcome"],
                            keyed[(seed, reference, replica)]["outcome"])
                   for replica in range(3)) for seed in seeds)


def shifted_plan(previous: tuple[int, ...] | list[int] | None, horizon: int) -> tuple[int, ...] | None:
    if previous is None:
        return None
    if horizon < 1 or not previous or any(a < 0 or a >= len(ACTION_VECTORS) for a in previous):
        raise ValueError("invalid previous plan or horizon")
    shifted = list(previous[1:1 + horizon])
    if len(shifted) < horizon:
        shifted.extend([previous[-1]] * (horizon - len(shifted)))
    return tuple(shifted)


def neighboring_action_ids(action_id: int) -> tuple[int, ...]:
    if not 0 <= action_id < len(ACTION_VECTORS):
        raise ValueError("invalid action ID")
    ax, ay = ACTION_VECTORS[action_id]
    return tuple(i for i, (bx, by) in enumerate(ACTION_VECTORS)
                 if i != action_id and max(abs(ax - bx), abs(ay - by)) == 1)


def mutation_proposals(base: tuple[int, ...], seed: int, mutation_count: int,
                       seed_xor: int) -> tuple[list[tuple[int, ...]], list[int]]:
    if not base or mutation_count < 1:
        raise ValueError("mutation needs a plan and positive candidate count")
    rng = random.Random(int(seed) ^ int(seed_xor))
    positions = rng.sample(range(len(base)), k=min(len(base), mutation_count))
    while len(positions) < mutation_count:
        positions.append(rng.randrange(len(base)))
    results, changed_positions = [], []
    for position in positions:
        candidate = list(base)
        neighbors = neighboring_action_ids(candidate[position])
        candidate[position] = neighbors[rng.randrange(len(neighbors))]
        results.append(tuple(candidate))
        changed_positions.append(position)
    return results, changed_positions


def candidate_bank(horizon: int, budget: int, seed: int, kind: str,
                   previous: tuple[int, ...] | list[int] | None,
                   mutation_count: int, seed_xor: int, attempt_cap: int = 1_280_000):
    """Insert prior-plan proposals by replacing tail residual samples, never constants."""
    base, base_duplicates, fallback = uniform_prefix_bank(horizon, budget, seed, attempt_cap)
    if kind == "uniform" or previous is None or len(base) <= 9:
        return base, {"base_duplicate_rejections": base_duplicates, "fallback_sequences": fallback,
                      "shifted_proposals": 0, "accepted_insertions": 0,
                      "proposal_overlaps": 0, "replaced_uniform_sequences": 0,
                      "uniform_sequences_retained": max(0, len(base) - 9), "mutation_positions": []}
    shifted = shifted_plan(previous, horizon)
    proposals = [shifted]
    positions = []
    if kind == "shift_plus_one_action_mutations":
        mutants, positions = mutation_proposals(shifted, seed, mutation_count, seed_xor)
        proposals.extend(mutants)
    elif kind != "shift_previous_winner":
        raise ValueError(f"unknown proposal kind: {kind}")

    bank = list(base)
    original_residual = set(base[9:])
    accepted = overlaps = replaced = 0
    for proposal in proposals[:max(0, len(base) - 9)]:
        if proposal in bank:
            overlaps += 1
            continue
        # Replace the last remaining original random candidate. The nine fixed
        # constant sequences remain present, and other uniform samples are kept.
        replace_at = next((i for i in range(len(bank) - 1, 8, -1)
                           if bank[i] in original_residual), None)
        if replace_at is None:
            break
        original_residual.remove(bank[replace_at])
        bank.pop(replace_at)
        bank.append(proposal)
        accepted += 1
        replaced += 1
    if len(bank) != len(base) or len(set(bank)) != len(bank) or tuple(bank[:9]) != base[:9]:
        raise RuntimeError("warm-start bank violated exact-budget, uniqueness, or constant coverage")
    return tuple(bank), {"base_duplicate_rejections": base_duplicates, "fallback_sequences": fallback,
                         "shifted_proposals": len(proposals), "accepted_insertions": accepted,
                         "proposal_overlaps": overlaps, "replaced_uniform_sequences": replaced,
                         "uniform_sequences_retained": len(bank) - 9 - accepted,
                         "mutation_positions": positions}


def verify_parent(root=ROOT, cfg=None):
    cfg = cfg or config(root)
    if p17.sha256_file(root / cfg["parent"]["result"]) != cfg["parent"]["result_sha256"]:
        raise RuntimeError("Phase19 result hash changed")
    if p17.sha256_file(root / cfg["parent"]["lock"]) != cfg["parent"]["lock_sha256"]:
        raise RuntimeError("Phase19 lock hash changed")
    parent = json.loads((root / cfg["parent"]["result"]).read_text(encoding="utf-8"))
    if parent["decision"] != "promote_uniform_b48_development_only" or not parent["qualification"]["pass"]:
        raise RuntimeError("Phase19 did not qualify U48")
    parent_dir = (root / cfg["parent"]["result"]).parent
    manifest = json.loads((parent_dir / "manifest.json").read_text(encoding="utf-8"))
    for key, path in (("result_sha256", root / cfg["parent"]["result"]),
                      ("records_sha256", parent_dir / "episodes.jsonl"),
                      ("lock_sha256", root / cfg["parent"]["lock"])):
        if p17.sha256_file(path) != manifest[key]:
            raise RuntimeError(f"Phase19 manifest hash mismatch for {key}")
    audited = p19.audit(root)
    if audited["episodes"] != 3500:
        raise RuntimeError("Phase19 audit did not cover the complete parent run")
    lock = json.loads((root / cfg["parent"]["lock"]).read_text(encoding="utf-8"))
    for path, digest in lock["hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"Phase19 source or identity dependency changed: {path}")
    return parent


def freeze(root=ROOT):
    cfg = config(root)
    validate_config(cfg)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit Phase20 implementation and tests before execution lock")
    verify_parent(root, cfg)
    start, count = cfg["block"]["proposed_seed_start"], cfg["block"]["proposed_layout_count"]
    assert_unsealed_range(start, count, cfg)
    out = root / cfg["outputs"]["run_dir"]
    lock_path = root / cfg["outputs"]["execution_lock"]
    if out.exists() or lock_path.exists():
        raise RuntimeError("refusing to overwrite Phase20 output or lock")
    audit = layout_hash_audit(start, count, root)
    if audit["layout_count"] != count or audit["unique_hashes"] != count:
        raise RuntimeError("Phase20 proposed layout identity audit failed")
    parent_rows = (root / cfg["parent"]["result"]).with_name("episodes.jsonl")
    consumed = {json.loads(line)["layout_sha256"] for line in parent_rows.read_text(encoding="utf-8").splitlines()}
    if consumed & {row["layout_sha256"] for row in audit["layouts"]}:
        raise RuntimeError("Phase20 identities overlap Phase19 consumed layouts")
    for path, digest in audit["historical_source_hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"historical identity source changed during audit: {path}")
    schedule = execution_order(audit["layouts"], cfg)
    out.mkdir(parents=True)
    p18.write_new(out / "layout_audit.json", audit)
    p18.write_new(out / "execution_order.json", schedule)
    tracked = subprocess.check_output(["git", "ls-files", "src", "tests", "configs"], cwd=root, text=True).splitlines()
    dependencies = {p for p in tracked if p.endswith((".py", ".yaml"))}
    dependencies.update({CONFIG, cfg["protocol"], cfg["parent"]["result"], cfg["parent"]["lock"],
        str(parent_rows.relative_to(root)),
        str(parent_rows.with_name("manifest.json").relative_to(root)),
        str((out / "layout_audit.json").relative_to(root)),
        str((out / "execution_order.json").relative_to(root))})
    dependencies.update(p for p in ("pyproject.toml", "uv.lock") if (root / p).is_file())
    dependencies.update(audit["historical_source_hashes"])
    lock = {"schema": "phase20-exact-plan-warmstart-lock-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "schedule_count": len(schedule),
        "layout_count": count, "hashes": {p: p17.sha256_file(root / p) for p in sorted(dependencies)},
        "reserve_access": False, "d168_access": False, "learned_model_loading": False,
        "training_corpus_loading": False}
    p18.write_new(lock_path, lock)
    return {"source_commit": lock["source_commit"], "layouts": count, "episodes": len(schedule),
            "layout_sha256": p17.sha256_file(out / "layout_audit.json"),
            "schedule_sha256": p17.sha256_file(out / "execution_order.json"),
            "lock_sha256": p17.sha256_file(lock_path)}


def verify(root=ROOT):
    cfg = config(root)
    validate_config(cfg)
    lock_path = root / cfg["outputs"]["execution_lock"]
    committed = subprocess.check_output(["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"], cwd=root)
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("execution lock must be committed and unchanged")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for path, digest in lock["hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"locked dependency changed: {path}")
    verify_parent(root, cfg)
    assert_unsealed_range(cfg["block"]["proposed_seed_start"], cfg["block"]["proposed_layout_count"], cfg)
    out = root / cfg["outputs"]["run_dir"]
    audit = json.loads((out / "layout_audit.json").read_text(encoding="utf-8"))
    schedule = json.loads((out / "execution_order.json").read_text(encoding="utf-8"))
    if schedule != execution_order(audit["layouts"], cfg) or len(schedule) != cfg["block"]["episodes"]:
        raise RuntimeError("schedule does not match frozen layout/arm configuration")
    return cfg, out, schedule, p17.sha256_file(lock_path)


def evaluate_decision(scenario, state, method: dict, replica: int, previous_plan, cfg):
    started = time.perf_counter()
    horizon = min(cfg["planning"]["horizon"], scenario.spec.max_steps - state.elapsed_steps)
    query = p17.query_seed(scenario.seed, state.elapsed_steps, replica)
    kind = method["kind"]
    bank, construction = candidate_bank(horizon, int(method["budget"]), query, kind, previous_plan,
        int(method.get("mutation_count", 0)), int(cfg["planning"]["mutation_seed_xor"]),
        int(cfg["planning"]["attempt_cap"]))
    evaluations = [evaluate_sequence(state, scenario, sequence) for sequence in bank]
    chosen, ties = choose_lexicographic_best(evaluations)
    successor = corridor_step(state, chosen.sequence[0], scenario.spec).state
    latency = time.perf_counter() - started
    successful = [i for i, result in enumerate(evaluations) if result.success_reached]
    return successor, chosen.sequence, latency, {
        "decision_index": state.elapsed_steps, "query_seed": query, "horizon": horizon,
        "budget": len(bank), "chosen_action": chosen.sequence[0],
        "chosen_sequence": list(chosen.sequence), "best_objective": list(chosen.objective),
        "tie_count": ties, "success_candidates": len(successful),
        "first_success_sequence_rank": successful[0] + 1 if successful else None,
        "collision_candidate_fraction": float(np.mean([r.objective[0] == 3 for r in evaluations])),
        "exact_transitions": sum(r.transitions for r in evaluations),
        "candidate_action_counts": np.bincount([seq[0] for seq in bank], minlength=9).tolist(),
        "shifted_proposals": construction["shifted_proposals"],
        "accepted_insertions": construction["accepted_insertions"],
        "proposal_overlaps": construction["proposal_overlaps"],
        "replaced_uniform_sequences": construction["replaced_uniform_sequences"],
        "uniform_sequences_retained": construction["uniform_sequences_retained"],
        "mutation_positions": construction["mutation_positions"],
        "base_duplicate_rejections": construction["base_duplicate_rejections"],
        "fallback_sequences": construction["fallback_sequences"]}


def evaluate_episode(scenario, method, replica, cfg):
    state, previous_plan = scenario.initial_state, None
    diagnostics, latencies = [], []
    action_counts = np.zeros(9, dtype=int)
    while state.status == "active":
        state, selected_plan, latency, diagnostic = evaluate_decision(
            scenario, state, method, replica, previous_plan, cfg)
        diagnostics.append(diagnostic)
        latencies.append(latency)
        action_counts[diagnostic["chosen_action"]] += 1
        previous_plan = selected_plan
    return {"seed": scenario.seed, "layout_sha256": scenario.layout_sha256,
        "method": method["name"], "replica": replica, "budget": method["budget"],
        "outcome": state.status, "steps": state.elapsed_steps, "decision_count": len(diagnostics),
        "exact_sequences": sum(d["budget"] for d in diagnostics),
        "exact_transitions": sum(d["exact_transitions"] for d in diagnostics),
        "executed_transitions": len(diagnostics),
        "duplicate_rejections": sum(d["base_duplicate_rejections"] + d["proposal_overlaps"] for d in diagnostics),
        "fallback_sequences": sum(d["fallback_sequences"] for d in diagnostics),
        "complete_success_sequences": sum(d["success_candidates"] for d in diagnostics),
        "selected_action_counts": action_counts.tolist(),
        "latency_seconds_per_decision": latencies, "wall_seconds": sum(latencies),
        "decision_diagnostics": diagnostics}


def summarize(records, schedule, cfg):
    keyed = {(r["seed"], r["method"], r["replica"]): r for r in records}
    expected = {(r["seed"], r["method"], r["replica"]) for r in schedule}
    if len(keyed) != len(records) or len(records) != len(schedule) or set(keyed) != expected:
        raise RuntimeError("records do not exactly cover the locked schedule")
    for row, arm in zip(records, schedule, strict=True):
        if any(row.get(key) != value for key, value in arm.items()):
            raise RuntimeError("record schedule identity differs")
        if row.get("outcome") not in {"success", "collision", "timeout"}:
            raise RuntimeError("invalid episode outcome")
    seeds = sorted({r["seed"] for r in schedule})
    methods, replicas, vectors = {}, {}, {}
    for method in cfg["methods"]:
        name = method["name"]
        rows = [r for r in records if r["method"] == name]
        reps = list(range(method["replicas"])) if method["replicas"] > 1 else [None]
        methods[name] = p17._rate_summary(rows)
        methods[name].update({"success_count": sum(r["outcome"] == "success" for r in rows),
                              "collision_count": sum(r["outcome"] == "collision" for r in rows),
                              "timeout_count": sum(r["outcome"] == "timeout" for r in rows)})
        replicas[name] = {str(rep): p17._rate_summary([r for r in rows if r["replica"] == rep]) for rep in reps}
        vectors[name] = {field: [float(np.mean([
            float(keyed[(seed, name, rep)]["outcome"] == field) if field in {"success", "collision"}
            else keyed[(seed, name, rep)]["exact_transitions"] for rep in reps])) for seed in seeds]
            for field in ("success", "collision", "transitions")}

    def absolute(name, gate):
        return (methods[name]["safe_success_rate"] > gate["success_strictly_above"]
            and methods[name]["collision_rate"] <= gate["collision_at_most"]
            and all(row["safe_success_rate"] >= gate["each_replica_success_at_least"]
                    and row["collision_rate"] <= gate["each_replica_collision_at_most"]
                    for row in replicas[name].values()))

    ref, ref_gate = "uniform_b48", cfg["gates"]["uniform_b48_reference"]
    geometry_gain = p18.difference_bounds(vectors["geometry_waypoint"]["success"],
                                          vectors[ref]["success"], cfg, .975)
    reference_checks = {"absolute": absolute(ref, ref_gate),
        "geometry_gain": geometry_gain["difference"] >= ref_gate["geometry_success_gain_at_least"],
        "geometry_gain_lower": geometry_gain["ci95_lower"] > ref_gate["paired_geometry_gain_95_lower_strictly_above"]}
    reference_pass = all(reference_checks.values())
    qualification = {}
    for method in cfg["methods"]:
        name = method["name"]
        if method["kind"] == "geometry":
            continue
        gate = cfg["gates"]["b32_candidate"] if method["budget"] == 32 else None
        if gate is None:
            continue
        success = p18.difference_bounds(vectors[ref]["success"], vectors[name]["success"], cfg,
                                        cfg["analysis"]["success_one_sided_bootstrap_confidence"])
        events = {}
        for event in ("success_regression", "added_collision"):
            n = discordant_layout_count(keyed, seeds, name, ref, event)
            events[event] = {"layouts": n, "upper": binomial_upper(n, len(seeds),
                cfg["analysis"]["rare_event_one_sided_exact_confidence"])}
        ratios = np.asarray(vectors[name]["transitions"]) / np.asarray(vectors[ref]["transitions"])
        compute = p18.difference_bounds(np.zeros(len(seeds)), ratios, cfg, .975)
        latency = methods[name]["median_decision_latency_seconds"] / methods[ref]["median_decision_latency_seconds"]
        checks = {"reference": reference_pass, "absolute": absolute(name, gate),
            "success_noninferiority": success["adjusted_lower"] >= gate["adjusted_success_difference_lower_vs_b48_at_least"],
            "success_rare_event": events["success_regression"]["upper"] <= gate["success_regression_layout_event_upper_at_most"],
            "collision_rare_event": events["added_collision"]["upper"] <= gate["added_collision_layout_event_upper_at_most"],
            "transitions_mean": compute["difference"] <= gate["mean_transition_ratio_vs_b48_at_most"],
            "transitions_upper": compute["ci95_upper"] <= gate["transition_ratio_95_upper_vs_b48_at_most"],
            "latency": latency <= gate["median_latency_ratio_vs_b48_at_most"]}
        qualification[name] = {"pass": all(checks.values()), "checks": checks,
            "success_difference_vs_b48": success, "events_vs_b48": events,
            "transition_ratio_vs_b48": compute, "latency_ratio_vs_b48": latency}

    attribution = {}
    for name in ("shifted_b32", "mutated_b32"):
        success = p18.difference_bounds(vectors["uniform_b32"]["success"], vectors[name]["success"],
                                        cfg, cfg["analysis"]["warmstart_attribution"]["one_sided_bootstrap_confidence"])
        collision = p18.difference_bounds(vectors["uniform_b32"]["collision"], vectors[name]["collision"],
                                          cfg, cfg["analysis"]["warmstart_attribution"]["one_sided_bootstrap_confidence"])
        attribution[name] = {"success_difference_vs_uniform_b32": success,
            "collision_difference_vs_uniform_b32": collision,
            "benefit_attributed": success["adjusted_lower"] > 0 or collision["adjusted_upper"] < 0}
    selected = next((name for name in cfg["analysis"]["method_selection_order"]
                     if qualification[name]["pass"]), None) if reference_pass else None
    return {"layouts": len(seeds), "episodes": len(records), "methods": methods,
        "per_replica": replicas, "reference_checks": reference_checks,
        "reference_pass": reference_pass, "geometry_gain": geometry_gain,
        "qualification": qualification, "warmstart_attribution": attribution,
        "selected_method": selected, "decision": selected or "retain_uniform_b48",
        "learning_claim": False, "reserve_access": False, "d168_access": False}


def run(root=ROOT):
    import torch
    cfg, out, schedule, lock_hash = verify(root)
    if (out / "result.json").exists():
        raise RuntimeError("completed Phase20 run cannot be overwritten")
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    path = out / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    p18.validate_prefix(records, schedule, lock_hash)
    resumed = len(records)
    started = time.perf_counter()
    for index in range(resumed, len(schedule)):
        arm = schedule[index]
        scenario = generate_staggered_corridor(arm["seed"])
        if scenario.layout_sha256 != arm["layout_sha256"]:
            raise RuntimeError("generated layout differs from audited identity")
        if arm["method"] == "geometry_waypoint":
            result = p17.geometry_episode(scenario, {"geometry": cfg["geometry"]})
        else:
            method = next(m for m in cfg["methods"] if m["name"] == arm["method"])
            result = evaluate_episode(scenario, method, arm["replica"], cfg)
        result.update(arm)
        result["execution_lock_sha256"] = lock_hash
        p17._jsonl_append(path, result)
        records.append(result)
        if (index + 1) % 13 == 0:
            print(f"Phase20 {index + 1}/{len(schedule)} episodes; {time.perf_counter()-started:.1f}s", flush=True)
    verify(root)
    p18.validate_prefix(records, schedule, lock_hash)
    result = summarize(records, schedule, cfg)
    result.update({"execution_lock_sha256": lock_hash, "records_sha256": p17.sha256_file(path),
                   "resumed_records": resumed, "execution_wall_seconds": time.perf_counter() - started})
    p18.write_new(out / "result.json", result)
    p18.write_new(out / "manifest.json", {"lock_sha256": lock_hash,
        "result_sha256": p17.sha256_file(out / "result.json"), "records_sha256": p17.sha256_file(path),
        "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "torch_threads": torch.get_num_threads(), "device": "cpu",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "reserve_access": False,
        "d168_access": False, "training_corpus_loading": False, "learned_model_loading": False})
    return {"episodes": len(records), "selected_method": result["selected_method"],
            "result_sha256": p17.sha256_file(out / "result.json")}


def audit(root=ROOT):
    cfg, out, schedule, lock_hash = verify(root)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    for filename, hash_name in (("episodes.jsonl", "records_sha256"), ("result.json", "result_sha256")):
        if p17.sha256_file(out / filename) != manifest[hash_name]:
            raise RuntimeError(f"Phase20 artifact digest differs: {filename}")
    if manifest["lock_sha256"] != lock_hash or result["execution_lock_sha256"] != lock_hash:
        raise RuntimeError("Phase20 lock binding differs")
    records = [json.loads(line) for line in (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    p18.validate_prefix(records, schedule, lock_hash)
    recomputed = summarize(records, schedule, cfg)
    if any(result[key] != value for key, value in recomputed.items()):
        raise RuntimeError("saved Phase20 result does not reproduce")
    return {"audit_pass": True, "episodes": len(records), "selected_method": result["selected_method"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run", "audit"])
    print(json.dumps({"freeze": freeze, "run": run, "audit": audit}[parser.parse_args().command](), indent=2))


if __name__ == "__main__":
    main()
