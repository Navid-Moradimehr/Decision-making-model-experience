from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import yaml

from consequence_lab import phase17_proposal_guidance as p17
from consequence_lab import phase18_proposal_controls as p18
from consequence_lab import phase20_exact_plan_warmstart as p20
from consequence_lab import phase21_shifted_b32_confirmation as p21
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.phase16c_prior_guided import layout_hash_audit
from consequence_lab.phase16d_compute_frontier import binomial_upper

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase22_mutation_confirmation_v1.yaml"


def config(root=ROOT):
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def validate_config(cfg):
    expected = [
        ("uniform_b32", "uniform", 32, 3, 0, 0),
        ("shifted_b32", "shift_previous_winner", 32, 3, 1, 0),
        ("mutated_b32", "shift_plus_one_action_mutations", 32, 3, 5, 4),
        ("uniform_b48", "uniform", 48, 3, 0, 0),
        ("geometry_waypoint", "geometry", None, 1, 0, 0),
    ]
    actual = [(m["name"], m["kind"], m["budget"], m["replicas"],
               m["warm_insertions"], m["mutation_count"])
              for m in cfg["planning"]["methods"]]
    if actual != expected:
        raise ValueError("Phase22 methods differ from the frozen mutation confirmation")
    if (cfg["block"]["episodes"] != cfg["block"]["proposed_layout_count"] * 13
            or cfg["block"]["replicas"] != [0, 1, 2]
            or cfg["planning"]["horizon"] != 8
            or cfg["planning"]["nominal_budgets"] != [32, 48]
            or cfg["planning"]["mandatory_constant_sequences"] != 9
            or cfg["planning"]["mutation_count"] != 4
            or cfg["planning"]["mutation_seed_xor"] != 0x51A7E20D
            or cfg["analysis"]["attribution_one_sided_confidence"] != 0.9875):
        raise ValueError("Phase22 budget, mutation, or statistics contract differs")
    if list(range(9)) != cfg["actions"]["ids"]:
        raise ValueError("Phase22 action-ID contract differs")
    for key in ("phase20", "phase21"):
        if key not in cfg["parents"]:
            raise ValueError(f"Phase22 is missing parent binding: {key}")


def assert_unsealed_range(start, count, cfg):
    p21.assert_unsealed_range(start, count, cfg)


def execution_order(layouts, cfg):
    return p21.execution_order(layouts, cfg)


def verify_parents(root, cfg):
    p20_result = root / cfg["parents"]["phase20"]["result"]
    p20_lock = root / cfg["parents"]["phase20"]["lock"]
    p21_result = root / cfg["parents"]["phase21"]["result"]
    p21_lock = root / cfg["parents"]["phase21"]["lock"]
    for key, path in (("phase20", p20_result), ("phase20 lock", p20_lock),
                       ("phase21", p21_result), ("phase21 lock", p21_lock)):
        result_key = "result_sha256" if key in {"phase20", "phase21"} else "lock_sha256"
        parent_key = key if key in {"phase20", "phase21"} else key.removesuffix(" lock")
        if p17.sha256_file(path) != cfg["parents"][parent_key][result_key]:
            raise RuntimeError(f"Phase22 parent hash mismatch: {key}")
    p21.audit(root)
    result20 = json.loads(p20_result.read_text(encoding="utf-8"))
    result21 = json.loads(p21_result.read_text(encoding="utf-8"))
    if result20.get("decision") != "shifted_b32" or not result20.get("qualification", {}).get(
            "shifted_b32", {}).get("pass"):
        raise RuntimeError("Phase20 did not qualify shifted B32")
    if result21.get("decision") != "shifted_b32" or not result21.get("qualification", {}).get("pass"):
        raise RuntimeError("Phase21 did not independently confirm shifted B32")
    for result_path, lock_path in ((p20_result, p20_lock), (p21_result, p21_lock)):
        run_dir = result_path.parent
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        for field, path in (("result_sha256", result_path),
                            ("records_sha256", run_dir / "episodes.jsonl"),
                            ("lock_sha256", lock_path)):
            if p17.sha256_file(path) != manifest[field]:
                raise RuntimeError(f"Phase22 parent manifest mismatch: {path.name}")
    return {"phase20": result20, "phase21": result21}


def freeze(root=ROOT):
    cfg = config(root)
    validate_config(cfg)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit Phase22 source/config/tests and vault before identity freeze")
    verify_parents(root, cfg)
    start, count = cfg["block"]["proposed_seed_start"], cfg["block"]["proposed_layout_count"]
    assert_unsealed_range(start, count, cfg)
    out, lock_path = root / cfg["outputs"]["run_dir"], root / cfg["outputs"]["execution_lock"]
    if out.exists() or lock_path.exists():
        raise RuntimeError("refusing to overwrite Phase22 output or lock")
    audit = layout_hash_audit(start, count, root)
    if audit["layout_count"] != count or audit["unique_hashes"] != count:
        raise RuntimeError("Phase22 identity audit failed")
    if audit["overlap_historical"] or audit["overlap_outcome_bearing"]:
        raise RuntimeError("Phase22 proposed identities overlap historical data")
    parent_layouts = set()
    for parent in cfg["parents"].values():
        records_path = root / parent["result"].replace("result.json", "episodes.jsonl")
        parent_layouts |= {json.loads(line)["layout_sha256"]
                           for line in records_path.read_text(encoding="utf-8").splitlines()}
    if parent_layouts & {row["layout_sha256"] for row in audit["layouts"]}:
        raise RuntimeError("Phase22 identities overlap a consumed parent block")
    for path, digest in audit["historical_source_hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"historical identity source changed: {path}")
    schedule = execution_order(audit["layouts"], cfg)
    out.mkdir(parents=True)
    p18.write_new(out / "layout_audit.json", audit)
    p18.write_new(out / "execution_order.json", schedule)
    dependencies = {p for p in subprocess.check_output(
        ["git", "ls-files", "src", "tests", "configs"], cwd=root, text=True
    ).splitlines() if p.endswith((".py", ".yaml"))}
    dependencies.update({CONFIG, cfg["protocol"], *audit["historical_source_hashes"].keys()})
    for parent in cfg["parents"].values():
        result_path = root / parent["result"]
        run_dir = result_path.parent
        dependencies.update({parent["result"], parent["lock"],
            str((run_dir / "episodes.jsonl").relative_to(root)),
            str((run_dir / "manifest.json").relative_to(root))})
    dependencies.update({str((out / "layout_audit.json").relative_to(root)),
                         str((out / "execution_order.json").relative_to(root))})
    dependencies.update(p for p in ("pyproject.toml", "uv.lock") if (root / p).is_file())
    lock = {"schema": "phase22-mutation-confirmation-lock-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                  text=True).strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "schedule_count": len(schedule),
        "layout_count": count,
        "hashes": {p: p17.sha256_file(root / p) for p in sorted(dependencies)},
        "reserve_access": False, "d168_access": False, "training_corpus_loading": False,
        "learned_model_loading": False}
    p18.write_new(lock_path, lock)
    return {"source_commit": lock["source_commit"], "layouts": count, "episodes": len(schedule),
            "layout_audit_sha256": p17.sha256_file(out / "layout_audit.json"),
            "schedule_sha256": p17.sha256_file(out / "execution_order.json"),
            "lock_sha256": p17.sha256_file(lock_path)}


def verify(root=ROOT):
    cfg = config(root)
    validate_config(cfg)
    lock_path = root / cfg["outputs"]["execution_lock"]
    committed = subprocess.check_output(["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"],
                                        cwd=root)
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("Phase22 execution lock must be committed unchanged")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for path, digest in lock["hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"Phase22 dependency changed: {path}")
    verify_parents(root, cfg)
    assert_unsealed_range(cfg["block"]["proposed_seed_start"],
                          cfg["block"]["proposed_layout_count"], cfg)
    out = root / cfg["outputs"]["run_dir"]
    audit = json.loads((out / "layout_audit.json").read_text(encoding="utf-8"))
    schedule = json.loads((out / "execution_order.json").read_text(encoding="utf-8"))
    if schedule != execution_order(audit["layouts"], cfg) or len(schedule) != cfg["block"]["episodes"]:
        raise RuntimeError("Phase22 schedule differs from identity/config contract")
    return cfg, out, schedule, p17.sha256_file(lock_path)


def _discordant_count(keyed, seeds, method, reference, event):
    def flagged(a, b):
        if event == "success_regression":
            return a != "success" and b == "success"
        if event == "added_collision":
            return a == "collision" and b != "collision"
        raise ValueError(f"unknown event: {event}")
    return sum(any(flagged(keyed[(seed, method, rep)]["outcome"],
                           keyed[(seed, reference, rep)]["outcome"]) for rep in range(3))
               for seed in seeds)


def summarize(records, schedule, cfg):
    keyed = {(r["seed"], r["method"], r["replica"]): r for r in records}
    expected = {(r["seed"], r["method"], r["replica"]) for r in schedule}
    if len(keyed) != len(records) or set(keyed) != expected or len(records) != len(schedule):
        raise RuntimeError("Phase22 records do not exactly cover frozen schedule")
    for row, arm in zip(records, schedule, strict=True):
        if any(row.get(k) != v for k, v in arm.items()):
            raise RuntimeError("Phase22 record identity differs from schedule")
        if row.get("outcome") not in {"success", "collision", "timeout"}:
            raise RuntimeError("invalid Phase22 outcome")
    seeds = sorted({r["seed"] for r in schedule})
    methods, replicas, vectors = {}, {}, {}
    for method in cfg["planning"]["methods"]:
        name = method["name"]
        rows = [r for r in records if r["method"] == name]
        reps = list(range(method["replicas"])) if method["replicas"] > 1 else [None]
        methods[name] = p17._rate_summary(rows)
        methods[name].update({"success_count": sum(r["outcome"] == "success" for r in rows),
            "collision_count": sum(r["outcome"] == "collision" for r in rows),
            "timeout_count": sum(r["outcome"] == "timeout" for r in rows)})
        replicas[name] = {str(rep): p17._rate_summary([r for r in rows if r["replica"] == rep])
                          for rep in reps}
        vectors[name] = {field: [float(np.mean([
            float(keyed[(seed, name, rep)]["outcome"] == field)
            if field in {"success", "collision"}
            else keyed[(seed, name, rep)]["exact_transitions"] for rep in reps])) for seed in seeds]
            for field in ("success", "collision", "transitions")}

    def absolute(name, gate):
        return (methods[name]["safe_success_rate"] > gate["success_strictly_above"]
            and methods[name]["collision_rate"] <= gate["collision_at_most"]
            and all(rep["safe_success_rate"] >= gate["each_replica_success_at_least"]
                    and rep["collision_rate"] <= gate["each_replica_collision_at_most"]
                    for rep in replicas[name].values())
            and methods[name]["timeout_rate"] == 0)

    ref, uniform, shifted, mutated, geom = (
        "uniform_b48", "uniform_b32", "shifted_b32", "mutated_b32", "geometry_waypoint")
    ref_gate = cfg["gates"]["uniform_b48_reference"]
    geometry_gain = p18.difference_bounds(vectors[geom]["success"], vectors[ref]["success"], cfg, .975)
    ref_checks = {"absolute": absolute(ref, ref_gate),
        "geometry_gain": geometry_gain["difference"] >= ref_gate["geometry_success_gain_at_least"],
        "geometry_gain_lower": geometry_gain["ci95_lower"] > ref_gate["paired_geometry_gain_95_lower_strictly_above"]}
    ref_pass = all(ref_checks.values())

    def qualify(candidate):
        gate = cfg["gates"]["b32_candidate"]
        success = p18.difference_bounds(vectors[ref]["success"], vectors[candidate]["success"], cfg,
                                        cfg["analysis"]["success_one_sided_bootstrap_confidence"])
        events = {}
        for event in ("success_regression", "added_collision"):
            n = _discordant_count(keyed, seeds, candidate, ref, event)
            events[event] = {"layouts": n, "upper": binomial_upper(
                n, len(seeds), cfg["analysis"]["rare_event_one_sided_exact_confidence"])}
        ratios = np.asarray(vectors[candidate]["transitions"]) / np.asarray(vectors[ref]["transitions"])
        transition = p18.difference_bounds(np.zeros(len(seeds)), ratios, cfg, .975)
        latency = methods[candidate]["median_decision_latency_seconds"] / methods[ref]["median_decision_latency_seconds"]
        checks = {"reference": ref_pass, "absolute": absolute(candidate, gate),
            "success_noninferiority": success["adjusted_lower"] >= gate["adjusted_success_difference_lower_vs_b48_at_least"],
            "success_rare_event": events["success_regression"]["upper"] <= gate["success_regression_layout_event_upper_at_most"],
            "collision_rare_event": events["added_collision"]["upper"] <= gate["added_collision_layout_event_upper_at_most"],
            "transitions_mean": transition["difference"] <= gate["mean_transition_ratio_vs_b48_at_most"],
            "transitions_upper": transition["ci95_upper"] <= gate["transition_ratio_95_upper_vs_b48_at_most"],
            "latency": latency <= gate["median_latency_ratio_vs_b48_at_most"]}
        return {"pass": all(checks.values()), "checks": checks,
            "success_difference_vs_b48": success, "events_vs_b48": events,
            "transition_ratio_vs_b48": transition, "latency_ratio_vs_b48": latency}

    qualifications = {name: qualify(name) for name in (shifted, mutated)}

    def attribution(candidate, reference):
        conf = cfg["analysis"]["attribution_one_sided_confidence"]
        success = p18.difference_bounds(vectors[reference]["success"], vectors[candidate]["success"], cfg, conf)
        collision = p18.difference_bounds(vectors[reference]["collision"], vectors[candidate]["collision"], cfg, conf)
        return {"success_difference_candidate_minus_reference": success,
            "collision_difference_candidate_minus_reference": collision,
            "benefit_attributed": success["adjusted_lower"] > 0 or collision["adjusted_upper"] < 0}

    attribution_u32 = {name: attribution(name, uniform) for name in (shifted, mutated)}
    mutation_vs_shift = attribution(mutated, shifted)
    if ref_pass and qualifications[mutated]["pass"] and mutation_vs_shift["benefit_attributed"]:
        decision = mutated
    elif ref_pass and qualifications[shifted]["pass"]:
        decision = shifted
    else:
        decision = ref
    return {"layouts": len(seeds), "episodes": len(records), "methods": methods,
        "per_replica": replicas, "reference_checks": ref_checks, "reference_pass": ref_pass,
        "geometry_gain": geometry_gain, "qualification": qualifications,
        "attribution_vs_uniform_b32": attribution_u32,
        "mutation_incremental_attribution_vs_shifted": mutation_vs_shift,
        "selected_method": decision, "decision": decision, "learning_claim": False,
        "reserve_access": False, "d168_access": False}


def run(root=ROOT):
    import torch
    cfg, out, schedule, lock_hash = verify(root)
    if (out / "result.json").exists():
        raise RuntimeError("completed Phase22 run cannot be overwritten")
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
            raise RuntimeError("generated layout identity differs from audit")
        if arm["method"] == "geometry_waypoint":
            result = p17.geometry_episode(scenario, {"geometry": cfg["geometry"]})
        else:
            method = next(m for m in cfg["planning"]["methods"] if m["name"] == arm["method"])
            result = p20.evaluate_episode(scenario, method, arm["replica"], cfg)
        result.update(arm)
        result["execution_lock_sha256"] = lock_hash
        p17._jsonl_append(path, result)
        records.append(result)
        if (index + 1) % 10 == 0:
            print(f"Phase22 {index + 1}/{len(schedule)} episodes; {time.perf_counter()-started:.1f}s", flush=True)
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
    for filename, key in (("episodes.jsonl", "records_sha256"), ("result.json", "result_sha256")):
        if p17.sha256_file(out / filename) != manifest[key]:
            raise RuntimeError(f"Phase22 artifact digest differs: {filename}")
    if manifest["lock_sha256"] != lock_hash or result["execution_lock_sha256"] != lock_hash:
        raise RuntimeError("Phase22 result/manifest lock binding differs")
    records = [json.loads(line) for line in (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    p18.validate_prefix(records, schedule, lock_hash)
    recomputed = summarize(records, schedule, cfg)
    if any(result[key] != value for key, value in recomputed.items()):
        raise RuntimeError("Phase22 result does not reproduce from saved records")
    return {"audit_pass": True, "episodes": len(records), "decision": result["decision"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run", "audit"])
    command = parser.parse_args().command
    print(json.dumps({"freeze": freeze, "run": run, "audit": audit}[command](), indent=2))


if __name__ == "__main__":
    main()
