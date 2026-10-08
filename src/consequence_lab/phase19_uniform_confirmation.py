"""Independent, locked Uniform-B48 confirmation; no learned proposal loading."""
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
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.phase16c_prior_guided import layout_hash_audit
from consequence_lab.phase16d_compute_frontier import binomial_upper

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase19_uniform_b48_confirmation_v1.yaml"


def config(root=ROOT):
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def runtime_config(cfg):
    """Adapt only names to the frozen Phase18 exact evaluator interface."""
    expected = [("uniform_b48", 48, 3), ("uniform_b64", 64, 3),
                ("geometry_waypoint", None, 1)]
    actual = [(a["name"], a["budget"], a["replicas"]) for a in cfg["methods"]]
    if actual != expected or cfg["planning"]["horizon"] != 8 or cfg["planning"]["attempt_cap"] != 1280000:
        raise ValueError("confirmation does not implement modified arms or sampler")
    if cfg["block"]["replicas"] != [0, 1, 2]:
        raise ValueError("replica contract changed")
    return {"planning": {"methods": cfg["methods"], "horizon": 8,
                         "common_attempt_cap": 1280000},
            "geometry": cfg["geometry"], "analysis": cfg["analysis"],
            "stage18b": {"order_seed": cfg["analysis"]["order_seed"]}}


def assert_unsealed_range(start, count, cfg):
    if start < 0 or count < 1:
        raise ValueError("invalid development range")
    for key, bounds in cfg["block"].items():
        if key.endswith("_half_open"):
            low, high = bounds
            if start < high and low < start + count:
                raise ValueError(f"excluded seed block: {key}")


def verify_parent(root, cfg):
    for name in ("result", "lock"):
        if p17.sha256_file(root / cfg["parent"][name]) != cfg["parent"][name + "_sha256"]:
            raise RuntimeError(f"Phase18 {name} changed")
    parent = json.loads((root / cfg["parent"]["result"]).read_text())
    if not parent["qualification"]["uniform_b48"]["pass"] or parent["learning_attribution_pass"]:
        raise RuntimeError("Phase18 simplicity-rule premises differ")
    # Verify frozen transitive source/data hashes without deserializing training data or models.
    parent_lock = json.loads((root / cfg["parent"]["lock"]).read_text())
    for path, digest in parent_lock["hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"parent dependency changed: {path}")
    manifest_path = (root / cfg["parent"]["result"]).with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest["result_sha256"] != cfg["parent"]["result_sha256"]:
        raise RuntimeError("parent manifest/result mismatch")
    if manifest["lock_sha256"] != cfg["parent"]["lock_sha256"]:
        raise RuntimeError("parent manifest/lock mismatch")
    episodes = manifest_path.with_name("episodes.jsonl")
    if p17.sha256_file(episodes) != manifest["records_sha256"]:
        raise RuntimeError("parent episode hash mismatch")


def freeze(root=ROOT):
    cfg = config(root)
    runtime_config(cfg)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit implementation/tests before freezing")
    out = root / cfg["outputs"]["run_dir"]
    lock_path = root / cfg["outputs"]["execution_lock"]
    if out.exists() or lock_path.exists():
        raise RuntimeError("refusing to overwrite a previous lock or run directory")
    verify_parent(root, cfg)
    start, count = cfg["block"]["proposed_seed_start"], cfg["block"]["proposed_layout_count"]
    assert_unsealed_range(start, count, cfg)
    audit = layout_hash_audit(start, count, root)
    parent_episodes = (root / cfg["parent"]["result"]).with_name("episodes.jsonl")
    consumed = {json.loads(line)["layout_sha256"] for line in parent_episodes.read_text().splitlines()}
    if consumed & {r["layout_sha256"] for r in audit["layouts"]}:
        raise RuntimeError("Phase18 consumed-layout overlap")
    schedule = p18.execution_order(audit["layouts"], runtime_config(cfg))
    if len(schedule) != cfg["block"]["episodes"] or len(schedule) != 7 * count:
        raise RuntimeError("schedule count differs")
    p18.write_new(out / "layout_audit.json", audit)
    p18.write_new(out / "execution_order.json", schedule)
    tracked = subprocess.check_output(["git", "ls-files", "src", "tests", "configs"], cwd=root, text=True).splitlines()
    paths = {p for p in tracked if p.endswith((".py", ".yaml"))}
    paths.update({CONFIG, cfg["protocol"], cfg["parent"]["result"], cfg["parent"]["lock"],
                  str(parent_episodes.relative_to(root)),
                  str(parent_episodes.with_name("manifest.json").relative_to(root)),
                  str((out / "layout_audit.json").relative_to(root)),
                  str((out / "execution_order.json").relative_to(root))})
    paths.update(p for p in ("pyproject.toml", "uv.lock") if (root / p).is_file())
    paths.update(audit["historical_source_hashes"])
    for path, digest in audit["historical_source_hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"historical identity evidence changed during audit: {path}")
    lock = {"schema": "phase19-uniform-confirmation-lock-v1",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "schedule_count": len(schedule), "layout_count": count,
            "hashes": {p: p17.sha256_file(root / p) for p in sorted(paths)},
            "reserve_access": False, "d168_access": False,
            "proposal_checkpoint_loading": False, "training_corpus_loading": False}
    p18.write_new(lock_path, lock)
    return {"schedule_count": len(schedule), "layout_count": count,
            "source_commit": lock["source_commit"], "lock_sha256": p17.sha256_file(lock_path)}


def verify(root=ROOT):
    cfg = config(root)
    runtime_config(cfg)
    lock_path = root / cfg["outputs"]["execution_lock"]
    committed = subprocess.check_output(["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"], cwd=root)
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("lock must be committed and unchanged")
    lock = json.loads(lock_path.read_text())
    for path, digest in lock["hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"locked dependency changed: {path}")
    verify_parent(root, cfg)
    assert_unsealed_range(cfg["block"]["proposed_seed_start"], cfg["block"]["proposed_layout_count"], cfg)
    out = root / cfg["outputs"]["run_dir"]
    audit = json.loads((out / "layout_audit.json").read_text())
    schedule = json.loads((out / "execution_order.json").read_text())
    if schedule != p18.execution_order(audit["layouts"], runtime_config(cfg)) or len(schedule) != cfg["block"]["episodes"]:
        raise RuntimeError("locked schedule differs")
    return cfg, out, schedule, p17.sha256_file(lock_path)


def summarize(records, schedule, cfg):
    """No pooling of prior phases; layout is the primary sampling unit."""
    keyed = {(r["seed"], r["method"], r["replica"]): r for r in records}
    expected = {(r["seed"], r["method"], r["replica"]) for r in schedule}
    if len(records) != len(schedule) or len(keyed) != len(records) or set(keyed) != expected:
        raise RuntimeError("incomplete/duplicate confirmation records")
    for row, arm in zip(records, schedule, strict=True):
        if any(row.get(k) != v for k, v in arm.items()) or row["outcome"] not in {"success", "collision", "timeout"}:
            raise RuntimeError("record differs from schedule")
    seeds = sorted({r["seed"] for r in schedule})
    methods, replicas, vectors = {}, {}, {}
    for arm in cfg["methods"]:
        name = arm["name"]
        rows = [r for r in records if r["method"] == name]
        reps = list(range(arm["replicas"])) if arm["replicas"] > 1 else [None]
        methods[name] = p17._rate_summary(rows)
        methods[name]["success_count"] = sum(r["outcome"] == "success" for r in rows)
        methods[name]["collision_count"] = sum(r["outcome"] == "collision" for r in rows)
        methods[name]["timeout_count"] = sum(r["outcome"] == "timeout" for r in rows)
        replicas[name] = {str(rep): p17._rate_summary([r for r in rows if r["replica"] == rep]) for rep in reps}
        vectors[name] = {field: [float(np.mean([
            float(keyed[(seed, name, rep)]["outcome"] == field) if field in {"success", "collision"}
            else keyed[(seed, name, rep)]["exact_transitions"] for rep in reps])) for seed in seeds]
                         for field in ("success", "collision", "transitions")}
    def absolute(name, gate):
        return (methods[name]["safe_success_rate"] > gate["success_strictly_above"]
                and methods[name]["collision_rate"] <= gate["collision_at_most"]
                and all(r["safe_success_rate"] >= gate["each_replica_success_at_least"]
                        and r["collision_rate"] <= gate["each_replica_collision_at_most"] for r in replicas[name].values()))
    ref, candidate = "uniform_b64", "uniform_b48"
    geometry_gain = p18.difference_bounds(vectors["geometry_waypoint"]["success"], vectors[ref]["success"], cfg, .975)
    ref_gate, gate = cfg["gates"]["b64_reference"], cfg["gates"][candidate]
    ref_checks = {"absolute": absolute(ref, ref_gate),
                  "geometry_gain": geometry_gain["difference"] >= ref_gate["geometry_success_gain_at_least"],
                  "geometry_gain_lower": geometry_gain["ci95_lower"] > ref_gate["paired_geometry_gain_95_lower_strictly_above"]}
    success = p18.difference_bounds(vectors[ref]["success"], vectors[candidate]["success"], cfg,
                                   cfg["analysis"]["success_one_sided_bootstrap_confidence"])
    events = {}
    for event in ("success_regression", "added_collision"):
        count = p18.event_count(keyed, seeds, candidate, event)
        events[event] = {"layouts": count, "upper": binomial_upper(count, len(seeds),
            cfg["analysis"]["rare_event_one_sided_exact_confidence"])}
    if np.any(np.asarray(vectors[ref]["transitions"]) <= 0):
        raise RuntimeError("invalid B64 compute denominator")
    ratios = np.asarray(vectors[candidate]["transitions"]) / np.asarray(vectors[ref]["transitions"])
    ratio = p18.difference_bounds(np.zeros(len(seeds)), ratios, cfg, .975)
    latency = methods[candidate]["median_decision_latency_seconds"] / methods[ref]["median_decision_latency_seconds"]
    checks = {"reference": all(ref_checks.values()), "absolute": absolute(candidate, gate),
              "success_noninferiority": success["adjusted_lower"] >= gate["adjusted_success_difference_lower_at_least"],
              "success_rare_event": events["success_regression"]["upper"] <= gate["success_regression_layout_event_upper_at_most"],
              "collision_rare_event": events["added_collision"]["upper"] <= gate["added_collision_layout_event_upper_at_most"],
              "transitions_mean": ratio["difference"] <= gate["mean_transition_ratio_at_most"],
              "transitions_upper": ratio["ci95_upper"] <= gate["transition_ratio_95_upper_at_most"],
              "latency": latency <= gate["median_latency_ratio_at_most"]}
    common = [(keyed[(s, candidate, rep)], keyed[(s, ref, rep)]) for s in seeds for rep in range(3)
              if keyed[(s, candidate, rep)]["outcome"] == keyed[(s, ref, rep)]["outcome"] == "success"]
    qualification = {"pass": all(checks.values()), "checks": checks, "success_difference": success,
                     "events": events, "transition_ratio": ratio, "latency_ratio": latency,
                     "pooled_transition_ratio": float(np.mean(vectors[candidate]["transitions"]) / np.mean(vectors[ref]["transitions"])),
                     "common_success_compute_descriptive": {"paired_episodes": len(common), "pooled_transition_ratio":
                         sum(c["exact_transitions"] for c, _ in common) / sum(r["exact_transitions"] for _, r in common) if common else None}}
    return {"layouts": len(seeds), "episodes": len(records), "methods": methods, "per_replica": replicas,
            "reference_checks": ref_checks, "b64_reference_pass": all(ref_checks.values()),
            "geometry_gain": geometry_gain, "qualification": qualification,
            "decision": "promote_uniform_b48_development_only" if qualification["pass"] else "retain_uniform_b64",
            "reserve_access": False, "d168_access": False, "learning_claim": False}


def run(root=ROOT):
    import torch
    cfg, out, schedule, lock_hash = verify(root)
    if (out / "result.json").exists():
        raise RuntimeError("completed confirmation cannot be rerun")
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    runtime = runtime_config(cfg)
    path = out / "episodes.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    p18.validate_prefix(records, schedule, lock_hash)
    resumed = len(records)
    started = time.perf_counter()
    for index in range(resumed, len(schedule)):
        arm = schedule[index]
        scenario = generate_staggered_corridor(arm["seed"])
        if scenario.layout_sha256 != arm["layout_sha256"]:
            raise RuntimeError("generated identity differs from locked layout")
        result = (p17.geometry_episode(scenario, runtime) if arm["method"] == "geometry_waypoint"
                  else p18.evaluate_episode(scenario, arm["method"], arm["replica"], None, None, runtime))
        result.update(arm)
        result["execution_lock_sha256"] = lock_hash
        p17._jsonl_append(path, result)
        records.append(result)
        if (index + 1) % 7 == 0:
            print(f"Phase19: {index + 1}/{len(schedule)} episodes; {time.perf_counter()-started:.1f}s", flush=True)
    verify(root)
    p18.validate_prefix(records, schedule, lock_hash)
    result = summarize(records, schedule, cfg)
    result.update({"execution_lock_sha256": lock_hash, "records_sha256": p17.sha256_file(path),
                   "resumed_records": resumed, "execution_wall_seconds": time.perf_counter() - started})
    p18.write_new(out / "result.json", result)
    p18.write_new(out / "manifest.json", {"lock_sha256": lock_hash, "result_sha256": p17.sha256_file(out / "result.json"),
        "records_sha256": p17.sha256_file(path), "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
        "device": "cpu", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "reserve_access": False, "d168_access": False, "training_corpus_loading": False,
        "proposal_checkpoint_loading": False})
    return {"episodes": len(records), "decision": result["decision"], "result_sha256": p17.sha256_file(out / "result.json")}


def audit(root=ROOT):
    cfg, out, schedule, lock_hash = verify(root)
    manifest = json.loads((out / "manifest.json").read_text())
    result = json.loads((out / "result.json").read_text())
    for file, key in (("episodes.jsonl", "records_sha256"), ("result.json", "result_sha256")):
        if p17.sha256_file(out / file) != manifest[key]:
            raise RuntimeError(f"result integrity failure: {file}")
    if manifest["lock_sha256"] != lock_hash or result["execution_lock_sha256"] != lock_hash:
        raise RuntimeError("result/manifest lock differs")
    records = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    p18.validate_prefix(records, schedule, lock_hash)
    recomputed = summarize(records, schedule, cfg)
    if any(result[k] != v for k, v in recomputed.items()):
        raise RuntimeError("saved result does not reproduce from episode records")
    return {"audit_pass": True, "episodes": len(records), "decision": result["decision"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run", "audit"])
    print(json.dumps({"freeze": freeze, "run": run, "audit": audit}[parser.parse_args().command](), indent=2))


if __name__ == "__main__":
    main()
