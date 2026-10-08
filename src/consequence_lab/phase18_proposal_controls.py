"""Locked intermediate-budget study. Learning supplies proposals, never authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import yaml

from consequence_lab import phase17_proposal_guidance as p17
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase16c_prior_guided import evaluate_sequence, layout_hash_audit
from consequence_lab.phase16d_compute_frontier import binomial_upper, choose_lexicographic_best, uniform_prefix_bank

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase18_intermediate_budget_proposal_v1.yaml"


def config(root=ROOT):
    cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    amendment = yaml.safe_load((root / "configs/phase18_execution_paths_v2.yaml").read_text(encoding="utf-8"))
    cfg["outputs"].update(amendment["outputs"])
    return cfg


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest().upper()


def verify_parents(root, cfg):
    for parent in cfg["parents"].values():
        if isinstance(parent, dict) and "sha256" in parent:
            if p17.sha256_file(root / parent["path"]) != parent["sha256"]:
                raise RuntimeError(f"parent hash mismatch: {parent['path']}")


def aggregate_global_prior(rows, expected_states, pseudocount=.5):
    """Only repeats 0..15 participate; later repeats/aggregate fields are ignored."""
    weights = np.full(9, pseudocount, dtype=np.float64)
    membership = []
    for index, row in enumerate(rows):
        if row["row"] != index or len(row["replica_records"]) < 16:
            raise ValueError("teacher membership/repeat contract differs")
        targets = np.asarray([r["tie_mass"] for r in row["replica_records"][:16]], dtype=np.float64)
        if (targets.shape != (16, 9) or not np.isfinite(targets).all()
                or np.any(targets < 0) or not np.allclose(targets.sum(axis=1), 1, atol=1e-12)):
            raise ValueError("invalid training teacher target")
        weights += targets.mean(axis=0)
        membership.append({key: row[key] for key in ("row", "seed", "layout_sha256", "decision_index")})
    if len(membership) != expected_states:
        raise ValueError("global prior source state count differs")
    return weights / weights.sum(), membership


def prepare(root=ROOT):
    cfg = config(root)
    verify_parents(root, cfg)
    data = p17.load_training_states(root)
    path = root / cfg["parents"]["teacher_queries"]["path"]
    with path.open(encoding="utf-8") as stream:
        prior, membership = aggregate_global_prior((json.loads(line) for line in stream),
                                                  cfg["proposal"]["global_prior"]["source_states"])
    for row in membership:
        i = row["row"]
        if (row["seed"] != int(data["seeds"][i]) or row["layout_sha256"] != str(data["layouts"][i])
                or row["decision_index"] != int(data["decision_index"][i])):
            raise RuntimeError("teacher/source-state membership mismatch")
    payload = {"probabilities": prior.tolist(), "probabilities_sha256": digest_value(prior.tolist()),
               "membership": membership, "membership_sha256": digest_value(membership),
               "teacher_sha256": p17.sha256_file(path), "repeats_half_open": [0, 16],
               "formula": cfg["proposal"]["global_prior"]["formula"], "dtype": "float64"}
    write_new(root / cfg["outputs"]["global_prior"], payload)
    return payload


def execution_order(layouts, cfg):
    order = []
    for layout in layouts:
        arms = [{**layout, "method": arm["name"], "budget": arm["budget"],
                 "replica": rep if arm["replicas"] > 1 else None}
                for arm in cfg["planning"]["methods"] for rep in range(arm["replicas"])]
        random.Random(cfg["stage18b"]["order_seed"] + layout["seed"]).shuffle(arms)
        for local, arm in enumerate(arms):
            order.append({**arm, "global_schedule_index": len(order), "local_order_index": local})
    return order


def assert_unsealed_range(start, count, cfg):
    if start < 0 or count < 1:
        raise ValueError("invalid development block")
    for key in ("reserve_sealed_half_open", "d168_sealed_half_open", "phase17_consumed_half_open"):
        low, high = cfg["stage18b"][key]
        if start < high and low < start + count:
            raise ValueError(f"candidate block intersects excluded range: {key}")


def validate_prefix(records, schedule, lock_hash):
    if len(records) > len(schedule):
        raise RuntimeError("resume exceeds locked schedule")
    for index, (record, expected) in enumerate(zip(records, schedule[:len(records)], strict=True)):
        if (any(record.get(key) != value for key, value in expected.items())
                or record.get("global_schedule_index") != index
                or record.get("execution_lock_sha256") != lock_hash):
            raise RuntimeError("resume is not the exact locked schedule prefix")
        if "outcome" in record and record["outcome"] not in {"success", "collision", "timeout"}:
            raise RuntimeError("invalid resumed outcome")


def freeze(stage, root=ROOT):
    cfg = config(root)
    verify_parents(root, cfg)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit source/tests before freezing")
    prior_path = root / cfg["outputs"]["global_prior"]
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if digest_value(prior["probabilities"]) != prior["probabilities_sha256"]:
        raise RuntimeError("prior vector digest differs")
    out = root / cfg["outputs"][f"stage18{stage}_run_dir"]
    dependencies = [CONFIG, cfg["protocol"], "reports/phase18_execution_authorization.md",
                    "reports/phase18_execution_abort_and_amendment_v2.md",
                    "experiments/phase18a_fixed_state_proposal_controls_v1.lock.json",
                    "artifacts/runs/phase18a_fixed_state_proposal_controls_v1/execution_order.json"]
    # Pin all tracked Python source/tests, avoiding accidental omitted transitive imports.
    tracked = subprocess.check_output(["git", "ls-files", "src", "tests", "configs"], cwd=root, text=True).splitlines()
    dependencies.extend(path for path in tracked if path.endswith((".py", ".yaml")))
    dependencies.extend([str(prior_path.relative_to(root)), *[
        v["path"] for v in cfg["parents"].values() if isinstance(v, dict) and "path" in v]])
    dependencies.extend([cfg["parents"]["phase17a_source_rows"], cfg["parents"]["phase17_training_contract"]])
    dependencies.extend([cfg["parents"]["phase17c_lock"], cfg["parents"]["phase16e_lock"]])
    if stage == "a":
        rows = json.loads((root / cfg["parents"]["phase17a_source_rows"]).read_text())["source_rows"]
        if len(rows) != 64 or len(set(rows)) != 64:
            raise RuntimeError("diagnostic source-row contract differs")
        schedule = [{"source_row": row, "replica": rep, "method": method,
                     "global_schedule_index": index}
                    for index, (row, rep, method) in enumerate(
                        (row, rep, method) for row in rows for rep in range(32)
                        for method in cfg["stage18a"]["methods"])]
        original = json.loads((root / "artifacts/runs/phase18a_fixed_state_proposal_controls_v1/execution_order.json").read_text())
        if original != schedule:
            raise RuntimeError("amendment changed diagnostic schedule")
        aborted = root / "artifacts/runs/phase18a_fixed_state_proposal_controls_v1"
        if any((aborted / name).exists() for name in ("queries.jsonl", "result.json", "manifest.json")):
            raise RuntimeError("v1 abort unexpectedly contains outcomes")
    else:
        a_result = root / cfg["outputs"]["stage18a_run_dir"] / "result.json"
        if not a_result.is_file():
            raise RuntimeError("Stage18A must complete before fresh lock")
        assert_unsealed_range(cfg["stage18b"]["proposed_seed_start"], cfg["stage18b"]["proposed_layout_count"], cfg)
        audit = layout_hash_audit(cfg["stage18b"]["proposed_seed_start"],
                                  cfg["stage18b"]["proposed_layout_count"], root)
        # Existing collector is supplemented with explicit consumed Phase17 identities.
        old_rows = root / "artifacts/runs/phase17c_exact_guided_planner_v2/episodes.jsonl"
        old_hashes = {json.loads(line)["layout_sha256"] for line in old_rows.read_text().splitlines()}
        if old_hashes & {row["layout_sha256"] for row in audit["layouts"]}:
            raise RuntimeError("fresh block overlaps Phase17 outcome identities")
        write_new(out / "layout_audit.json", audit)
        schedule = execution_order(audit["layouts"], cfg)
        if len(schedule) != cfg["stage18b"]["expected_episodes"]:
            raise RuntimeError("fresh schedule count mismatch")
        dependencies += [str(a_result.relative_to(root)), str((out / "layout_audit.json").relative_to(root)),
                         str(old_rows.relative_to(root))]
    write_new(out / "execution_order.json", schedule)
    dependencies.append(str((out / "execution_order.json").relative_to(root)))
    lock = {"schema": f"phase18{stage}-execution-lock-v1", "commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "schedule_count": len(schedule),
        "hashes": {path: p17.sha256_file(root / path) for path in sorted(set(dependencies))},
        "reserve_access": False, "d168_evaluation_access": False}
    write_new(root / cfg["outputs"][f"expected_stage18{stage}_lock"], lock)
    return {"stage": stage, "schedule_count": len(schedule), "source_commit": lock["commit"]}


def verify(stage, root):
    cfg = config(root)
    path = root / cfg["outputs"][f"expected_stage18{stage}_lock"]
    lock = json.loads(path.read_text())
    committed = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(root).as_posix()}"], cwd=root)
    if not committed_text_parity(committed, path.read_bytes()):
        raise RuntimeError("execution lock must be committed and unchanged")
    for name, sha in lock["hashes"].items():
        if p17.sha256_file(root / name) != sha:
            raise RuntimeError(f"execution dependency changed: {name}")
    verify_parents(root, cfg)
    out = root / cfg["outputs"][f"stage18{stage}_run_dir"]
    return cfg, out, json.loads((out / "execution_order.json").read_text()), p17.sha256_file(path)


def committed_text_parity(committed, working):
    """Git normalizes text on Windows; no substantive byte change is accepted."""
    return committed.replace(b"\r\n", b"\n") == working.replace(b"\r\n", b"\n")


def evaluate_decision(scenario, state, method, replica, model, prior, cfg):
    started = time.perf_counter()
    horizon = min(cfg["planning"]["horizon"], scenario.spec.max_steps - state.elapsed_steps)
    query = p17.query_seed(scenario.seed, state.elapsed_steps, replica)
    arm = next(a for a in cfg["planning"]["methods"] if a["name"] == method)
    q = p17._predict(model, scenario, state) if method == "learned_b48" else prior if method == "global_b48" else None
    if q is None:
        bank, duplicates, fallback = uniform_prefix_bank(horizon, arm["budget"], query)
    else:
        bank, duplicates, fallback = p17.guided_sequence_bank(
            horizon, arm["budget"], query, q, cfg["proposal"]["alpha"],
            max_attempts=cfg["planning"]["common_attempt_cap"])
    evaluations = [evaluate_sequence(state, scenario, sequence) for sequence in bank]
    chosen, ties = choose_lexicographic_best(evaluations)
    successor = corridor_step(state, chosen.sequence[0], scenario.spec).state
    latency = time.perf_counter() - started
    # Reporting never changes selection and is outside the decision timer.
    counts = np.bincount([seq[0] for seq in bank], minlength=9)
    residual = np.bincount([seq[0] for seq in bank[9:]], minlength=9)
    probabilities = residual / max(1, residual.sum())
    entropy = float(-sum(p * math.log(p) for p in probabilities if p > 0))
    available = sum(int(e.success_reached) for e in evaluations)
    collision = sum(int(e.objective[0] == 3) for e in evaluations)
    diagnostic = {"decision_index": state.elapsed_steps, "query_seed": query, "budget": len(bank),
                  "chosen_action": chosen.sequence[0], "best_objective": list(chosen.objective),
                  "tie_count": ties, "success_candidates": available,
                  "first_success_sequence_rank": next((i + 1 for i, e in enumerate(evaluations) if e.success_reached), None),
                  "collision_candidate_fraction": collision / len(bank),
                  "exact_transitions": sum(e.transitions for e in evaluations),
                  "duplicate_rejections": duplicates, "fallback": fallback,
                  "residual_first_action_entropy_nats": entropy, "candidate_action_diversity": entropy / math.log(9),
                  "candidate_first_action_counts": counts.tolist(), "residual_first_action_counts": residual.tolist(),
                  "proposal": q.tolist() if q is not None else None,
                  "proposal_entropy": float(-sum(p * math.log(p) for p in q if p > 0)) if q is not None else None,
                  "distinct_objective_count": len({e.objective for e in evaluations})}
    return successor, diagnostic, latency, [list(e.objective) for e in evaluations]


def evaluate_episode(scenario, method, replica, model, prior, cfg):
    state = scenario.initial_state
    diagnostics, latencies = [], []
    counts = np.zeros(9, dtype=int)
    wall = time.perf_counter()
    while state.status == "active":
        state, diagnostic, latency, _ = evaluate_decision(scenario, state, method, replica, model, prior, cfg)
        diagnostics.append(diagnostic)
        latencies.append(latency)
        counts[diagnostic["chosen_action"]] += 1
    return {"seed": scenario.seed, "layout_sha256": scenario.layout_sha256, "method": method,
            "replica": replica, "budget": 64 if method == "uniform_b64" else 48,
            "outcome": state.status, "steps": state.elapsed_steps, "decision_count": len(diagnostics),
            "exact_sequences": sum(d["budget"] for d in diagnostics),
            "exact_transitions": sum(d["exact_transitions"] for d in diagnostics),
            "executed_transitions": len(diagnostics),
            "duplicate_rejections": sum(d["duplicate_rejections"] for d in diagnostics),
            "fallback_sequences": sum(d["fallback"] for d in diagnostics),
            "complete_success_sequences": sum(d["success_candidates"] for d in diagnostics),
            "selected_action_counts": counts.tolist(), "latency_seconds_per_decision": latencies,
            "wall_seconds": sum(latencies), "total_episode_wall_seconds": time.perf_counter() - wall,
            "decision_diagnostics": diagnostics}


def difference_bounds(reference, candidate, cfg, confidence):
    ref, cand = np.asarray(reference), np.asarray(candidate)
    if ref.shape != cand.shape or ref.ndim != 1 or len(ref) == 0:
        raise ValueError("bootstrap requires aligned layout vectors")
    rng = np.random.default_rng(cfg["analysis"]["bootstrap_seed"])
    indices = rng.integers(0, len(ref), size=(cfg["analysis"]["bootstrap_draws"], len(ref)))
    samples = (cand - ref)[indices].mean(axis=1)
    return {"difference": float((cand - ref).mean()), "ci95_lower": float(np.quantile(samples, .025)),
            "ci95_upper": float(np.quantile(samples, .975)),
            "adjusted_lower": float(np.quantile(samples, 1 - confidence)),
            "adjusted_upper": float(np.quantile(samples, confidence)), "one_sided_confidence": confidence}


def event_count(keyed, seeds, candidate, event):
    def discordant(c, r):
        return c != "success" and r == "success" if event == "success_regression" else c == "collision" and r != "collision"
    return sum(any(discordant(keyed[(seed, candidate, rep)]["outcome"],
                             keyed[(seed, "uniform_b64", rep)]["outcome"]) for rep in range(3)) for seed in seeds)


def summarize_b(records, schedule, cfg):
    keyed = {(r["seed"], r["method"], r["replica"]): r for r in records}
    expected = {(r["seed"], r["method"], r["replica"]) for r in schedule}
    if set(keyed) != expected or len(keyed) != len(records):
        raise RuntimeError("results must exactly cover locked schedule")
    for arm in schedule:
        row = keyed[(arm["seed"], arm["method"], arm["replica"])]
        if any(row.get(key) != value for key, value in arm.items()):
            raise RuntimeError("result identity/order differs from schedule")
    seeds = sorted({r["seed"] for r in schedule})
    methods, replicas, vectors = {}, {}, {}
    for arm in cfg["planning"]["methods"]:
        name = arm["name"]
        rows = [r for r in records if r["method"] == name]
        methods[name] = p17._rate_summary(rows)
        reps = list(range(3)) if arm["replicas"] == 3 else [None]
        replicas[name] = {str(rep): p17._rate_summary([r for r in rows if r["replica"] == rep]) for rep in reps}
        vectors[name] = {field: [float(np.mean([
            float(keyed[(seed, name, rep)]["outcome"] == field) if field in {"success", "collision"}
            else keyed[(seed, name, rep)]["exact_transitions"] for rep in reps])) for seed in seeds]
                         for field in ("success", "collision", "transitions")}
    def absolute(name, gate):
        return (methods[name]["safe_success_rate"] > gate["mean_success_strictly_above"]
                and methods[name]["collision_rate"] <= gate["mean_collision_at_most"]
                and all(r["safe_success_rate"] >= gate["every_replica_success_at_least"]
                        and r["collision_rate"] <= gate["every_replica_collision_at_most"] for r in replicas[name].values()))
    ref = "uniform_b64"
    geometry_gain = difference_bounds(vectors["geometry_waypoint"]["success"], vectors[ref]["success"], cfg, .975)
    ref_gate = cfg["gates"]["b64_reference"]
    reference_pass = (absolute(ref, ref_gate) and geometry_gain["difference"] >= ref_gate["geometry_success_gain_at_least"]
                      and geometry_gain["ci95_lower"] > ref_gate["paired_geometry_gain_95_ci_lower_strictly_above"])
    qualification = {}
    for name in cfg["analysis"]["qualification_candidates"]:
        gate = cfg["gates"]["each_b48_qualification"]
        success = difference_bounds(vectors[ref]["success"], vectors[name]["success"], cfg,
                                    cfg["analysis"]["qualification_success_one_sided_bootstrap_confidence"])
        events = {event: {"layouts": event_count(keyed, seeds, name, event)} for event in ("success_regression", "added_collision")}
        for value in events.values():
            value["upper"] = binomial_upper(value["layouts"], len(seeds),
                cfg["analysis"]["qualification_rare_event_one_sided_exact_confidence"])
        ratios = np.asarray(vectors[name]["transitions"]) / np.asarray(vectors[ref]["transitions"])
        ratio = difference_bounds(np.zeros(len(seeds)), ratios, cfg, .975)
        latency = methods[name]["median_decision_latency_seconds"] / methods[ref]["median_decision_latency_seconds"]
        checks = {"reference": reference_pass, "absolute": absolute(name, gate),
                  "success_noninferiority": success["adjusted_lower"] >= gate["adjusted_success_difference_lower_vs_b64_at_least"],
                  "success_rare_event": events["success_regression"]["upper"] <= gate["adjusted_success_regression_upper_vs_b64_at_most"],
                  "collision_rare_event": events["added_collision"]["upper"] <= gate["adjusted_added_collision_upper_vs_b64_at_most"],
                  "transitions_mean": ratio["difference"] <= gate["mean_transition_ratio_vs_b64_at_most"],
                  "transitions_upper": ratio["ci95_upper"] <= gate["transition_ratio_95_ci_upper_vs_b64_at_most"],
                  "latency": latency <= gate["median_latency_ratio_vs_b64_at_most"]}
        qualification[name] = {"pass": all(checks.values()), "checks": checks, "success_difference": success,
                               "events": events, "transition_ratio": ratio, "latency_ratio": latency,
                               "pooled_transition_ratio": float(np.mean(vectors[name]["transitions"]) / np.mean(vectors[ref]["transitions"]))}
        common = [(keyed[(seed, name, rep)], keyed[(seed, ref, rep)]) for seed in seeds for rep in range(3)
                  if keyed[(seed, name, rep)]["outcome"] == keyed[(seed, ref, rep)]["outcome"] == "success"]
        qualification[name]["common_success_compute_descriptive"] = {
            "paired_episodes": len(common), "pooled_transition_ratio":
            sum(c["exact_transitions"] for c, _ in common) / sum(r["exact_transitions"] for _, r in common) if common else None}
    attribution, gate = {}, cfg["gates"]["learning_attribution"]
    for name in gate["comparators"]:
        success = difference_bounds(vectors[name]["success"], vectors["learned_b48"]["success"], cfg,
                                    cfg["analysis"]["learning_attribution_one_sided_bootstrap_confidence"])
        collision = difference_bounds(vectors[name]["collision"], vectors["learned_b48"]["collision"], cfg,
                                      cfg["analysis"]["learning_attribution_one_sided_bootstrap_confidence"])
        attribution[name] = {"success": success, "collision": collision,
                             "pass": success["difference"] >= gate["success_gain_at_least_each"]
                             and success["adjusted_lower"] > gate["adjusted_success_gain_lower_strictly_above_each"]
                             and collision["adjusted_upper"] <= gate["adjusted_collision_increase_upper_at_most_each"]}
    return {"methods": methods, "per_replica": replicas, "layouts": len(seeds), "episodes": len(records),
            "b64_reference_pass": reference_pass, "geometry_gain": geometry_gain,
            "qualification": qualification, "learning_attribution": attribution,
            "learning_attribution_pass": qualification["learned_b48"]["pass"] and all(a["pass"] for a in attribution.values()),
            "proposal_diagnostics": {name: p17._proposal_diagnostics([r for r in records if r["method"] == name])
                                     for name in cfg["stage18a"]["methods"]},
            "per_layout_latency_medians": {name: [float(np.median([
                latency for rep in range(3) for latency in keyed[(seed, name, rep)]["latency_seconds_per_decision"]]))
                for seed in seeds] for name in cfg["stage18a"]["methods"]}, "reserve_access": False}


def summarize_a(records):
    keys = {(r["source_row"], r["replica"], r["method"]): r for r in records}
    comparisons = {}
    for control in ("uniform_b48", "global_b48"):
        outcomes = []
        layout_means = []
        for source in sorted({r["source_row"] for r in records}):
            local = []
            for rep in range(32):
                learned = tuple(keys[(source, rep, "learned_b48")]["diagnostic"]["best_objective"])
                other = tuple(keys[(source, rep, control)]["diagnostic"]["best_objective"])
                local.append(-1 if learned < other else 1 if learned > other else 0)
            outcomes.extend(local)
            layout_means.append({"source_row": source, "win": local.count(-1) / 32,
                                 "tie": local.count(0) / 32, "loss": local.count(1) / 32})
        comparisons[control] = {"win": outcomes.count(-1) / len(outcomes), "tie": outcomes.count(0) / len(outcomes),
                                "loss": outcomes.count(1) / len(outcomes), "per_source_layout": layout_means}
    return {"queries": len(records), "comparisons": comparisons, "software_integrity_pass": len(records) == 8192,
            "methods": {method: {"success_availability": float(np.mean([r["diagnostic"]["success_candidates"] > 0 for r in records if r["method"] == method])),
                                 "collision_candidate_fraction": float(np.mean([r["diagnostic"]["collision_candidate_fraction"] for r in records if r["method"] == method])),
                                 "mean_distinct_objective_count": float(np.mean([r["diagnostic"]["distinct_objective_count"] for r in records if r["method"] == method])),
                                 "residual_first_action_entropy": float(np.mean([r["diagnostic"]["residual_first_action_entropy_nats"] for r in records if r["method"] == method])),
                                 "duplicate_rejections": sum(r["diagnostic"]["duplicate_rejections"] for r in records if r["method"] == method),
                                 "fallback_sequences": sum(r["diagnostic"]["fallback"] for r in records if r["method"] == method)}
                        for method in ("uniform_b48", "global_b48", "learned_b48", "uniform_b64")}}


def run(stage, root=ROOT):
    import torch
    cfg, out, schedule, lock_hash = verify(stage, root)
    if (out / "result.json").exists():
        raise RuntimeError("completed phase cannot be rerun")
    torch.set_num_threads(cfg["execution"]["torch_intraop_threads"])
    torch.use_deterministic_algorithms(True)
    initialization = time.perf_counter()
    model = p17._load_final_proposal(root)
    prior = np.asarray(json.loads((root / cfg["outputs"]["global_prior"]).read_text())["probabilities"])
    data = p17.load_training_states(root) if stage == "a" else None
    teacher_reference = {}
    if stage == "a":
        source_rows = {arm["source_row"] for arm in schedule}
        with (root / cfg["parents"]["teacher_queries"]["path"]).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["row"] in source_rows:
                    teacher_reference[row["row"]] = row["replica_records"]
    init_seconds = time.perf_counter() - initialization
    path = out / "queries.jsonl" if stage == "a" else out / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    validate_prefix(records, schedule, lock_hash)
    resumed = len(records)
    started = time.perf_counter()
    for index in range(resumed, len(schedule)):
        arm = schedule[index]
        if stage == "a":
            scenario, state = p17._state_for_row(data, arm["source_row"])
            _, diagnostic, latency, objectives = evaluate_decision(scenario, state, arm["method"], arm["replica"], model, prior, cfg)
            if arm["method"] == "uniform_b64":
                previous = teacher_reference[arm["source_row"]][arm["replica"]]
                if (diagnostic["best_objective"] != previous["best_objective"]
                        or diagnostic["success_candidates"] != previous["success_candidates"]):
                    raise RuntimeError("U64 fixed-state exact teacher reproduction failed")
            result = {"seed": scenario.seed, "layout_sha256": scenario.layout_sha256,
                      "diagnostic": diagnostic, "latency_seconds": latency, "candidate_objectives": objectives}
        else:
            scenario = generate_staggered_corridor(arm["seed"])
            if scenario.layout_sha256 != arm["layout_sha256"]:
                raise RuntimeError("generated layout differs from locked identity")
            result = (p17.geometry_episode(scenario, cfg) if arm["method"] == "geometry_waypoint"
                      else evaluate_episode(scenario, arm["method"], arm["replica"], model, prior, cfg))
        result.update(arm)
        result["execution_lock_sha256"] = lock_hash
        p17._jsonl_append(path, result)
        records.append(result)
        if (index + 1) % (128 if stage == "a" else 13) == 0:
            print(f"Phase18{stage}: {index + 1}/{len(schedule)} complete; {time.perf_counter()-started:.1f}s", flush=True)
    validate_prefix(records, schedule, lock_hash)
    summary = summarize_a(records) if stage == "a" else summarize_b(records, schedule, cfg)
    summary.update({"execution_lock_sha256": lock_hash, "records_sha256": p17.sha256_file(path),
                    "resumed_records": resumed, "initialization_seconds": init_seconds,
                    "execution_wall_seconds": time.perf_counter() - started})
    write_new(out / "result.json", summary)
    write_new(out / "manifest.json", {"lock_sha256": lock_hash, "result_sha256": p17.sha256_file(out / "result.json"),
                                      "records_sha256": p17.sha256_file(path), "torch": torch.__version__,
                                      "numpy": np.__version__, "python": platform.python_version(),
                                      "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
                                      "device": "cpu", "reserve_access": False})
    return {"stage": stage, "records": len(records), "result_sha256": p17.sha256_file(out / "result.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "freeze-a", "run-a", "freeze-b", "run-b"])
    command = parser.parse_args().command
    result = prepare() if command == "prepare" else freeze(command[-1]) if command.startswith("freeze") else run(command[-1])
    print(json.dumps({key: value for key, value in result.items() if key != "membership"}, indent=2))


if __name__ == "__main__":
    main()
