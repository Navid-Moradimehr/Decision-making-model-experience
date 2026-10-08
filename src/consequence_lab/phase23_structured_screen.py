"""Fresh budget-matched exact screen of structured prefix edits."""
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
from consequence_lab import phase21_shifted_b32_confirmation as p21
from consequence_lab import phase22_mutation_confirmation as p22
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.phase12_teacher import state_features
from consequence_lab.phase16c_prior_guided import evaluate_sequence
from consequence_lab.phase16d_compute_frontier import binomial_upper

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase23_structured_screen_v1.yaml"
STRUCTURED_XOR = 0x23A11


def config(root: Path = ROOT):
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def validate_config(cfg: dict):
    expected = [
        ("shifted_b32", "shift_previous_winner", 32, 3),
        ("random_mutation_b32", "phase22_random_mutation", 32, 3),
        ("structured_prefix_b32", "two_edits_each_at_positions_0_1", 32, 3),
        ("uniform_b48", "uniform", 48, 3),
        ("geometry_waypoint", "geometry", None, 1),
    ]
    actual = [(m["name"], m["kind"], m["budget"], m["replicas"])
              for m in cfg["planning"]["methods"]]
    if actual != expected:
        raise ValueError("Phase23 method roster/order changed")
    if (cfg["block"]["seed_start"] != 208000 or cfg["block"]["layout_count"] != 500
            or cfg["block"]["episodes"] != 6500 or cfg["block"]["replicas"] != [0, 1, 2]
            or cfg["planning"]["horizon"] != 8
            or cfg["planning"]["mandatory_constant_sequences"] != 9
            or cfg["planning"]["random_mutation_count"] != 4
            or cfg["planning"]["random_mutation_seed_xor"] != 0x51A7E20D
            or cfg["planning"]["structured_action_seed_xor"] != STRUCTURED_XOR
            or cfg["analysis"]["structured_attribution_confidence"] != 0.99375):
        raise ValueError("Phase23 block, planner, or attribution contract changed")


def _paths(root: Path, cfg: dict, audit: dict, schedule: list[dict]) -> list[str]:
    paths = [CONFIG, "reports/phase23_structured_screen_protocol.md",
             "src/consequence_lab/phase23_structured_screen.py",
             "src/consequence_lab/phase17_proposal_guidance.py",
             "src/consequence_lab/phase18_proposal_controls.py",
             "src/consequence_lab/phase20_exact_plan_warmstart.py",
             "src/consequence_lab/phase21_shifted_b32_confirmation.py",
             "src/consequence_lab/phase22_mutation_confirmation.py",
             "src/consequence_lab/phase23_plan_structure.py",
             "src/consequence_lab/phase12_teacher.py",
             "src/consequence_lab/phase16c_prior_guided.py",
             "src/consequence_lab/phase16d_compute_frontier.py",
             "src/consequence_lab/corridor_baselines.py",
             "src/consequence_lab/corridor_tasks.py",
             "src/consequence_lab/inertial_corridor.py", "pyproject.toml", "uv.lock"]
    paths.extend(audit["historical_source_hashes"])
    for key, path in cfg["parents"].items():
        if not key.endswith("_sha256"):
            paths.append(path)
    for block in ("phase21", "phase22"):
        run_dir = root / ("artifacts/runs/phase21_shifted_b32_confirmation_v1" if block == "phase21"
                          else "artifacts/runs/phase22_mutation_confirmation_v1")
        paths.extend(str((run_dir / name).relative_to(root)) for name in
                     ("episodes.jsonl", "manifest.json", "execution_order.json", "layout_audit.json", "result.json"))
    replay_dir = root / "artifacts/runs/phase23_structure_replay_v1"
    paths.extend(str((replay_dir / name).relative_to(root)) for name in
                 ("structural_records.jsonl", "manifest.json", "result.json"))
    paths.extend(["reports/phase23_replay_protocol.md", "configs/phase23_structure_replay_v1.yaml"])
    paths.extend([str((root / cfg["outputs"]["run_dir"] / "layout_audit.json").relative_to(root)),
                  str((root / cfg["outputs"]["run_dir"] / "execution_order.json").relative_to(root))])
    return sorted(set(paths))


def _verify_parent_artifacts(root: Path, cfg: dict):
    p21.audit(root)
    p22.audit(root)
    replay_dir = root / "artifacts/runs/phase23_structure_replay_v1"
    replay_lock = root / cfg["parents"]["phase23_replay_lock"]
    replay_manifest = json.loads((replay_dir / "manifest.json").read_text(encoding="utf-8"))
    replay_result = json.loads((replay_dir / "result.json").read_text(encoding="utf-8"))
    for field, path in (("lock_sha256", replay_lock), ("records_sha256", replay_dir / "structural_records.jsonl"),
                        ("result_sha256", replay_dir / "result.json")):
        if p17.sha256_file(path) != replay_manifest[field]:
            raise RuntimeError(f"Phase23 replay parent hash mismatch: {path.name}")
    if replay_result["status"] != "complete-exploratory-replay" or replay_result["record_count"] != 3000:
        raise RuntimeError("Phase23 structural replay did not complete its frozen parity sample")
    if p17.sha256_file(replay_lock) != cfg["parents"]["phase23_replay_lock_sha256"]:
        raise RuntimeError("Phase23 replay lock differs from screen parent binding")
    if p17.sha256_file(replay_dir / "result.json") != cfg["parents"]["phase23_replay_result_sha256"]:
        raise RuntimeError("Phase23 replay result differs from screen parent binding")
    replay_records = [json.loads(line) for line in
                      (replay_dir / "structural_records.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(replay_records) != 3000 or not all(r.get("exact_replay_parity") for r in replay_records):
        raise RuntimeError("Phase23 replay parent has incomplete parity coverage")


def _assert_unsealed(start: int, count: int, block: dict):
    if count != 500 or start < 0:
        raise ValueError("Phase23 structured screen has a fixed 500-layout block")
    for key, bounds in block.items():
        if key.endswith("_half_open"):
            low, high = bounds
            if start < high and low < start + count:
                raise ValueError(f"fresh block intersects sealed/consumed range {key}")


def _insert(base: tuple[tuple[int, ...], ...], proposals: list[tuple[int, ...]]):
    bank = list(base)
    residual = set(base[9:])
    accepted, overlaps, replaced = [], 0, 0
    for proposal in proposals:
        if proposal in bank:
            overlaps += 1
            continue
        position = next((i for i in range(len(bank) - 1, 8, -1) if bank[i] in residual), None)
        if position is None:
            break
        residual.remove(bank[position])
        bank.pop(position)
        bank.append(proposal)
        accepted.append(proposal)
        replaced += 1
    if len(bank) != len(base) or len(set(bank)) != len(bank) or bank[:9] != list(base[:9]):
        raise RuntimeError("warm proposal insertion violated budget/uniqueness/constants")
    return tuple(bank), accepted, overlaps, replaced


def _structured_proposals(shifted: tuple[int, ...], query: int, seed_xor: int):
    if len(shifted) < 2:
        return [], []
    rng = random.Random(query ^ seed_xor)
    proposals, positions = [], []
    for position in (0, 1):
        neighbors = p20.neighboring_action_ids(shifted[position])
        chosen = rng.sample(list(neighbors), k=2)
        for action in chosen:
            candidate = list(shifted)
            candidate[position] = action
            proposals.append(tuple(candidate))
            positions.append(position)
    return proposals, positions


def _make_bank(horizon: int, method: dict, query: int, previous: tuple[int, ...] | None, cfg: dict):
    budget, kind = int(method["budget"]), method["kind"]
    if kind == "uniform":
        bank, construction = p20.candidate_bank(horizon, budget, query, "uniform", None, 0,
                                                 int(cfg["planning"]["random_mutation_seed_xor"]))
        return bank, [], [], construction, set()
    if previous is None:
        bank, construction = p20.candidate_bank(horizon, budget, query, "uniform", None, 0,
                                                 int(cfg["planning"]["random_mutation_seed_xor"]))
        return bank, [], [], construction, set()
    shifted = p20.shifted_plan(previous, horizon)
    base, _ = p20.candidate_bank(horizon, budget, query, "uniform", None, 0,
                                 int(cfg["planning"]["random_mutation_seed_xor"]))
    if kind == "shift_previous_winner":
        proposals, positions = [shifted], []
    elif kind == "phase22_random_mutation":
        mutants, positions = p20.mutation_proposals(shifted, query, 4,
            int(cfg["planning"]["random_mutation_seed_xor"]))
        proposals = [shifted, *mutants]
    elif kind == "two_edits_each_at_positions_0_1":
        mutants, positions = _structured_proposals(shifted, query,
            int(cfg["planning"]["structured_action_seed_xor"]))
        proposals = [shifted, *mutants]
    else:
        raise ValueError(f"unsupported Phase23 method kind: {kind}")
    bank, accepted, overlaps, replaced = _insert(base, proposals)
    construction = {"base_duplicate_rejections": 0, "fallback_sequences": 0,
        "shifted_proposals": 1, "accepted_insertions": len(accepted),
        "proposal_overlaps": overlaps, "replaced_uniform_sequences": replaced,
        "uniform_sequences_retained": len(bank) - 9 - len(accepted),
        "mutation_positions": positions}
    return bank, proposals, positions, construction, set(accepted)


def _evaluate_decision(scenario, state, method: dict, replica: int, previous_plan, cfg: dict):
    started = time.perf_counter()
    horizon = min(cfg["planning"]["horizon"], scenario.spec.max_steps - state.elapsed_steps)
    query = p17.query_seed(scenario.seed, state.elapsed_steps, replica)
    bank, proposals, positions, construction, inserted = _make_bank(
        horizon, method, query, previous_plan, cfg)
    evaluations = {seq: evaluate_sequence(state, scenario, seq) for seq in bank}
    chosen, ties = p20.choose_lexicographic_best(list(evaluations.values()))
    successor = p20.corridor_step(state, chosen.sequence[0], scenario.spec).state
    latency = time.perf_counter() - started
    successful = sum(result.success_reached for result in evaluations.values())
    diagnostic = {
        "decision_index": state.elapsed_steps, "query_seed": query, "horizon": horizon,
        "budget": len(bank), "chosen_action": chosen.sequence[0],
        "chosen_sequence": list(chosen.sequence), "best_objective": list(chosen.objective),
        "tie_count": ties, "success_candidates": successful,
        "first_success_sequence_rank": next((i + 1 for i, seq in enumerate(bank)
            if evaluations[seq].success_reached), None),
        "collision_candidate_fraction": float(np.mean([ev.objective[0] == 3 for ev in evaluations.values()])),
        "exact_transitions": sum(ev.transitions for ev in evaluations.values()),
        "candidate_action_counts": np.bincount([seq[0] for seq in bank], minlength=9).tolist(),
        **construction,
    }
    if method["kind"] in {"phase22_random_mutation", "two_edits_each_at_positions_0_1"} and previous_plan is not None:
        shifted = p20.shifted_plan(previous_plan, horizon)
        mutations = proposals[1:]
        diagnostic["edit_context"] = {
            "raw52": state_features(scenario, state).astype(float).tolist(),
            "shifted_previous_plan": list(shifted),
            "proposal_rule": method["kind"],
            "edits": [{"sequence": list(seq),
                "changed_positions": [i for i, (old, new) in enumerate(zip(shifted, seq, strict=True)) if old != new],
                "inserted_into_bank": seq in inserted,
                "exact_objective": list(evaluations[seq].objective),
                "safe_success": evaluations[seq].objective[0] == 0,
                "transitions": evaluations[seq].transitions} for seq in mutations],
            "shifted_plan_exact_objective": list(evaluations[shifted].objective),
            "shifted_plan_safe_success": evaluations[shifted].objective[0] == 0,
        }
    return successor, chosen.sequence, latency, diagnostic


def _evaluate_episode(scenario, method: dict, replica: int, cfg: dict):
    state, previous_plan = scenario.initial_state, None
    diagnostics, latencies = [], []
    actions = np.zeros(9, dtype=int)
    while state.status == "active":
        state, selected, latency, diagnostic = _evaluate_decision(
            scenario, state, method, replica, previous_plan, cfg)
        diagnostics.append(diagnostic)
        latencies.append(latency)
        actions[diagnostic["chosen_action"]] += 1
        previous_plan = selected
    return {"seed": scenario.seed, "layout_sha256": scenario.layout_sha256,
        "method": method["name"], "replica": replica, "budget": method["budget"],
        "outcome": state.status, "steps": state.elapsed_steps, "decision_count": len(diagnostics),
        "exact_sequences": sum(d["budget"] for d in diagnostics),
        "exact_transitions": sum(d["exact_transitions"] for d in diagnostics),
        "executed_transitions": len(diagnostics),
        "complete_success_sequences": sum(d["success_candidates"] for d in diagnostics),
        "selected_action_counts": actions.tolist(), "latency_seconds_per_decision": latencies,
        "wall_seconds": sum(latencies), "decision_diagnostics": diagnostics}


def _summarize(records: list[dict], schedule: list[dict], cfg: dict):
    keyed = {(r["seed"], r["method"], r["replica"]): r for r in records}
    expected = {(r["seed"], r["method"], r["replica"]) for r in schedule}
    if len(keyed) != len(records) or set(keyed) != expected or len(records) != len(schedule):
        raise RuntimeError("Phase23 structured screen records do not cover the frozen schedule")
    for row, arm in zip(records, schedule, strict=True):
        if any(row.get(k) != v for k, v in arm.items()):
            raise RuntimeError("episode identity differs from frozen schedule")
        if row.get("outcome") not in {"success", "collision", "timeout"}:
            raise RuntimeError("invalid episode outcome")
    seeds = sorted({r["seed"] for r in schedule})
    methods, replicas, vectors = {}, {}, {}
    for method in cfg["planning"]["methods"]:
        name = method["name"]
        rows = [r for r in records if r["method"] == name]
        methods[name] = p17._rate_summary(rows)
        methods[name].update({"success_count": sum(r["outcome"] == "success" for r in rows),
                              "collision_count": sum(r["outcome"] == "collision" for r in rows),
                              "timeout_count": sum(r["outcome"] == "timeout" for r in rows)})
        reps = list(range(method["replicas"])) if method["replicas"] > 1 else [None]
        replicas[name] = {str(rep): p17._rate_summary([r for r in rows if r["replica"] == rep])
                          for rep in reps}
        vectors[name] = {field: [float(np.mean([
            float(keyed[(seed, name, rep)]["outcome"] == field)
            if field in {"success", "collision"}
            else keyed[(seed, name, rep)]["exact_transitions"] for rep in reps]))
            for seed in seeds] for field in ("success", "collision", "transitions")}

    ref, geom = "uniform_b48", "geometry_waypoint"
    geometry_gain = p18.difference_bounds(vectors[geom]["success"], vectors[ref]["success"],
                                          cfg, .975)
    ref_gate = cfg["gates"]["uniform_b48_reference"]
    ref_abs = (methods[ref]["safe_success_rate"] > ref_gate["success_strictly_above"]
        and methods[ref]["collision_rate"] <= ref_gate["collision_at_most"]
        and methods[ref]["timeout_rate"] == 0
        and all(v["safe_success_rate"] >= ref_gate["each_replica_success_at_least"]
                and v["collision_rate"] <= ref_gate["each_replica_collision_at_most"]
                for v in replicas[ref].values()))
    reference_checks = {"absolute": ref_abs,
        "geometry_gain": geometry_gain["difference"] >= ref_gate["geometry_success_gain_at_least"],
        "geometry_gain_lower": geometry_gain["ci95_lower"] > ref_gate["paired_geometry_gain_95_lower_strictly_above"]}
    reference_pass = all(reference_checks.values())

    def qualify(name):
        gate = cfg["gates"]["b32_candidate"]
        success = p18.difference_bounds(vectors[ref]["success"], vectors[name]["success"], cfg,
                                        cfg["analysis"]["success_one_sided_bootstrap_confidence"])
        events = {}
        for event in ("success_regression", "added_collision"):
            n = p22._discordant_count(keyed, seeds, name, ref, event)
            events[event] = {"layouts": n, "upper": binomial_upper(n, len(seeds),
                cfg["analysis"]["rare_event_one_sided_exact_confidence"])}
        ratios = np.asarray(vectors[name]["transitions"]) / np.asarray(vectors[ref]["transitions"])
        transition = p18.difference_bounds(np.zeros(len(seeds)), ratios, cfg, .975)
        latency = methods[name]["median_decision_latency_seconds"] / methods[ref]["median_decision_latency_seconds"]
        checks = {"reference": reference_pass,
            "absolute": methods[name]["safe_success_rate"] > gate["success_strictly_above"]
                and methods[name]["collision_rate"] <= gate["collision_at_most"]
                and methods[name]["timeout_rate"] == 0
                and all(v["safe_success_rate"] >= gate["each_replica_success_at_least"]
                        and v["collision_rate"] <= gate["each_replica_collision_at_most"]
                        for v in replicas[name].values()),
            "noninferiority": success["adjusted_lower"] >= gate["adjusted_success_difference_lower_vs_b48_at_least"],
            "success_event": events["success_regression"]["upper"] <= gate["success_regression_layout_event_upper_at_most"],
            "collision_event": events["added_collision"]["upper"] <= gate["added_collision_layout_event_upper_at_most"],
            "transitions_mean": transition["difference"] <= gate["mean_transition_ratio_vs_b48_at_most"],
            "transitions_upper": transition["ci95_upper"] <= gate["transition_ratio_95_upper_vs_b48_at_most"],
            "latency": latency <= gate["median_latency_ratio_vs_b48_at_most"]}
        return {"pass": all(checks.values()), "checks": checks,
            "success_vs_b48": success, "events_vs_b48": events,
            "transition_ratio_vs_b48": transition, "latency_ratio_vs_b48": latency}

    candidates = ("shifted_b32", "random_mutation_b32", "structured_prefix_b32")
    qualification = {name: qualify(name) for name in candidates}

    def attribution(candidate: str, reference: str):
        conf = cfg["analysis"]["structured_attribution_confidence"]
        success = p18.difference_bounds(vectors[reference]["success"], vectors[candidate]["success"], cfg, conf)
        collision = p18.difference_bounds(vectors[reference]["collision"], vectors[candidate]["collision"], cfg, conf)
        return {"success_difference": success, "collision_difference": collision,
            "benefit_attributed": success["adjusted_lower"] > 0 or collision["adjusted_upper"] < 0}

    struct_vs_shift = attribution("structured_prefix_b32", "shifted_b32")
    struct_vs_random = attribution("structured_prefix_b32", "random_mutation_b32")
    random_vs_shift = attribution("random_mutation_b32", "shifted_b32")
    if (reference_pass and qualification["structured_prefix_b32"]["pass"]
            and struct_vs_shift["benefit_attributed"] and struct_vs_random["benefit_attributed"]):
        decision = "structured_prefix_b32"
    elif reference_pass and qualification["shifted_b32"]["pass"]:
        decision = "shifted_b32"
    else:
        decision = "uniform_b48"
    edit_records = sum(len(d.get("edit_context", {}).get("edits", []))
                       for r in records for d in r.get("decision_diagnostics", []))
    return {"layouts": len(seeds), "episodes": len(records), "methods": methods,
        "per_replica": replicas, "reference_checks": reference_checks,
        "reference_pass": reference_pass, "geometry_gain": geometry_gain,
        "qualification": qualification,
        "structured_attribution_vs_shifted": struct_vs_shift,
        "structured_attribution_vs_random": struct_vs_random,
        "random_attribution_vs_shifted": random_vs_shift,
        "candidate_edit_records": edit_records,
        "selected_method": decision, "decision": decision,
        "learning_claim": False, "reserve_access": False, "d168_access": False,
        "test_access": False}


def freeze(root: Path = ROOT):
    cfg = config(root)
    validate_config(cfg)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit Phase23 structured screen implementation/protocol before freeze")
    _verify_parent_artifacts(root, cfg)
    start, count = cfg["block"]["seed_start"], cfg["block"]["layout_count"]
    _assert_unsealed(start, count, cfg["block"])
    out, lock_path = root / cfg["outputs"]["run_dir"], root / cfg["outputs"]["execution_lock"]
    if out.exists() or lock_path.exists():
        raise FileExistsError("refusing to overwrite Phase23 structured screen output/lock")
    audit = p22.layout_hash_audit(start, count, root)
    if (audit["layout_count"] != count or audit["unique_hashes"] != count
            or audit["overlap_historical"] or audit["overlap_outcome_bearing"]):
        raise RuntimeError("Phase23 structured screen identity audit failed")
    schedule = p22.execution_order(audit["layouts"], cfg)
    if len(schedule) != cfg["block"]["episodes"]:
        raise RuntimeError("Phase23 schedule count differs")
    out.mkdir(parents=True)
    p18.write_new(out / "layout_audit.json", audit)
    p18.write_new(out / "execution_order.json", schedule)
    dependencies = {p for p in subprocess.check_output(
        ["git", "ls-files", "src", "tests", "configs"], cwd=root, text=True).splitlines()
        if p.endswith((".py", ".yaml"))}
    dependencies.update(_paths(root, cfg, audit, schedule))
    lock = {"schema": "phase23-structured-screen-lock-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "layout_count": count,
        "schedule_count": len(schedule), "hashes": {p: p17.sha256_file(root / p) for p in sorted(dependencies)},
        "reserve_access": False, "test_access": False, "d168_access": False,
        "training": False, "learned_control": False}
    p18.write_new(lock_path, lock)
    return {"layouts": count, "episodes": len(schedule), "lock_sha256": p17.sha256_file(lock_path),
            "source_commit": lock["source_commit"]}


def verify(root: Path = ROOT):
    cfg = config(root)
    validate_config(cfg)
    lock_path = root / cfg["outputs"]["execution_lock"]
    committed = subprocess.check_output(["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"], cwd=root)
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("Phase23 structured screen lock must be committed unchanged")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for path, digest in lock["hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"Phase23 structured screen dependency changed: {path}")
    _verify_parent_artifacts(root, cfg)
    out = root / cfg["outputs"]["run_dir"]
    audit = json.loads((out / "layout_audit.json").read_text(encoding="utf-8"))
    schedule = json.loads((out / "execution_order.json").read_text(encoding="utf-8"))
    if schedule != p22.execution_order(audit["layouts"], cfg) or len(schedule) != cfg["block"]["episodes"]:
        raise RuntimeError("Phase23 schedule differs from lock/config")
    return cfg, out, schedule, p17.sha256_file(lock_path)


def run(root: Path = ROOT):
    import torch
    cfg, out, schedule, lock_hash = verify(root)
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    path = out / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    p18.validate_prefix(records, schedule, lock_hash)
    resumed, started = len(records), time.perf_counter()
    for index in range(resumed, len(schedule)):
        arm = schedule[index]
        scenario = generate_staggered_corridor(arm["seed"])
        if scenario.layout_sha256 != arm["layout_sha256"]:
            raise RuntimeError("generated layout hash differs from locked audit")
        method = next(m for m in cfg["planning"]["methods"] if m["name"] == arm["method"])
        if method["kind"] == "geometry":
            result = p17.geometry_episode(scenario, {"geometry": cfg["geometry"]})
        else:
            result = _evaluate_episode(scenario, method, int(arm["replica"]), cfg)
        result.update(arm)
        result["execution_lock_sha256"] = lock_hash
        p17._jsonl_append(path, result)
        records.append(result)
        if (index + 1) % 25 == 0:
            print(f"Phase23 structured screen {index + 1}/{len(schedule)}; "
                  f"{time.perf_counter()-started:.1f}s", flush=True)
    p18.validate_prefix(records, schedule, lock_hash)
    result = _summarize(records, schedule, cfg)
    result.update({"execution_lock_sha256": lock_hash, "records_sha256": p17.sha256_file(path),
                   "resumed_records": resumed, "execution_wall_seconds": time.perf_counter()-started})
    p18.write_new(out / "result.json", result)
    p18.write_new(out / "manifest.json", {"lock_sha256": lock_hash,
        "records_sha256": p17.sha256_file(path), "result_sha256": p17.sha256_file(out / "result.json"),
        "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "torch_threads": torch.get_num_threads(), "device": "cpu",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "reserve_access": False,
        "test_access": False, "d168_access": False, "training_corpus_loading": False,
        "learned_model_loading": False})
    return result


def audit(root: Path = ROOT):
    cfg, out, schedule, lock_hash = verify(root)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    for filename, key in (("episodes.jsonl", "records_sha256"), ("result.json", "result_sha256")):
        if p17.sha256_file(out / filename) != manifest[key]:
            raise RuntimeError(f"Phase23 structured artifact hash mismatch: {filename}")
    if manifest["lock_sha256"] != lock_hash or result["execution_lock_sha256"] != lock_hash:
        raise RuntimeError("Phase23 structured output does not bind the committed lock")
    records = [json.loads(line) for line in (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    p18.validate_prefix(records, schedule, lock_hash)
    recomputed = _summarize(records, schedule, cfg)
    for key, value in recomputed.items():
        if result.get(key) != value:
            raise RuntimeError(f"Phase23 structured result does not reproduce: {key}")
    edit_count = sum(len(d.get("edit_context", {}).get("edits", []))
                     for episode in records for d in episode.get("decision_diagnostics", []))
    if edit_count != result["candidate_edit_records"]:
        raise RuntimeError("Phase23 candidate-edit data count does not reproduce")
    return {"audit_pass": True, "episodes": len(records), "decision": result["decision"],
            "candidate_edit_records": edit_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify", "run", "audit"))
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(), indent=2))
    elif args.command == "verify":
        cfg, _, _, lock_hash = verify()
        print(json.dumps({"verified": True, "lock_sha256": lock_hash,
                          "episodes": cfg["block"]["episodes"]}, indent=2))
    elif args.command == "run":
        print(json.dumps(run(), indent=2))
    else:
        print(json.dumps(audit(), indent=2))


if __name__ == "__main__":
    main()
