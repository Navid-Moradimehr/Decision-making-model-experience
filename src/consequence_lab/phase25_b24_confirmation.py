"""Independent fresh-layout confirmation of exact shifted B24 vs B32."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from consequence_lab import phase24_postrun_score as p24score
from consequence_lab import phase24_search_budget_profile as p24
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.phase16c_prior_guided import layout_hash_audit
from consequence_lab.phase16d_compute_frontier import binomial_upper

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase25_b24_confirmation_v1.yaml"
METHODS = [
    {"name": "shifted_b24", "budget": 24, "replicas": 3},
    {"name": "shifted_b32", "budget": 32, "replicas": 3},
]


def config(root: Path = ROOT) -> dict:
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def validate_config(cfg: dict) -> None:
    if cfg.get("schema") != "phase25-b24-confirmation-v1":
        raise ValueError("wrong Phase25 config schema")
    block, plan, analysis = cfg["block"], cfg["planning"], cfg["analysis"]
    if (block["seed_start"], block["layout_count"], block["replicas"]) != (209500, 500, [0, 1, 2]):
        raise ValueError("Phase25 fresh development block changed")
    if plan["budgets"] != [24, 32] or plan["horizon"] != 8:
        raise ValueError("Phase25 budget/horizon roster changed")
    if len(METHODS) * block["layout_count"] * len(block["replicas"]) != 3000:
        raise RuntimeError("Phase25 schedule contract is invalid")
    if analysis["bootstrap_draws"] != 10000 or analysis["bootstrap_seed"] != 25003:
        raise ValueError("Phase25 bootstrap procedure changed")
    if cfg["scope"]["reserve_test_d168_access"] or cfg["scope"]["training"]:
        raise ValueError("Phase25 cannot access sealed data or train")


def assert_unsealed(start: int, count: int, block: dict) -> None:
    if start < 0 or count < 1:
        raise ValueError("invalid Phase25 layout range")
    for key, interval in block.items():
        if key.endswith("_half_open"):
            lo, hi = map(int, interval)
            if start < hi and lo < start + count:
                raise ValueError(f"Phase25 layout range intersects {key}")


def execution_order(layouts: list[dict], cfg: dict) -> list[dict]:
    rows = []
    for layout in layouts:
        arms = [{**layout, "method": method["name"], "budget": method["budget"],
                 "replica": replica} for method in METHODS for replica in cfg["block"]["replicas"]]
        random.Random(int(cfg["analysis"]["schedule_seed"]) + int(layout["seed"])).shuffle(arms)
        for local, arm in enumerate(arms):
            rows.append({**arm, "global_schedule_index": len(rows), "local_order_index": local})
    expected = len(layouts) * len(METHODS) * len(cfg["block"]["replicas"])
    if len(rows) != expected:
        raise RuntimeError(f"Phase25 schedule has {len(rows)} rows; expected {expected}")
    keys = [(row["seed"], row["method"], row["replica"]) for row in rows]
    if len(set(keys)) != len(rows):
        raise RuntimeError("Phase25 schedule contains duplicate episode identities")
    return rows


def _parent_paths(cfg: dict) -> dict[str, str]:
    return {name: value["path"] for name, value in cfg["parent"].items()}


def _verify_parents(root: Path, cfg: dict) -> dict:
    documentation_drift = cfg["parent"]["phase24_documentation_drift"]
    digest_bindings = {name: value for name, value in cfg["parent"].items()
                       if name != "phase24_documentation_drift"}
    for name, value in digest_bindings.items():
        actual = p17.sha256_file(root / value["path"])
        if actual != value["sha256"]:
            raise RuntimeError(f"Phase24 parent hash mismatch ({name})")
    lock_path = root / digest_bindings["phase24_lock"]["path"]
    lock_bytes = lock_path.read_bytes()
    committed_lock = subprocess.check_output(
        ["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"], cwd=root)
    if not p18.committed_text_parity(committed_lock, lock_bytes):
        raise RuntimeError("Phase24 immutable lock changed after commit")
    phase24_lock = json.loads(lock_bytes)
    if phase24_lock.get("source_commit") != "34b323d3eca71da1c4f8dc0dbf70e4ae0a90a010":
        raise RuntimeError("Phase24 source commit differs from the audited parent")

    drift_path = documentation_drift["path"]
    locked_note_hash = documentation_drift["lock_expected_sha256"]
    if phase24_lock["hashes"].get(drift_path) != locked_note_hash:
        raise RuntimeError("Phase24 lock does not contain the documented note hash")
    if p17.sha256_file(root / drift_path) != documentation_drift["postrun_sha256"]:
        raise RuntimeError("Phase24 post-run documentation note differs from pinned bytes")
    old_blob = subprocess.check_output(
        ["git", "show", f"{phase24_lock['source_commit']}:{drift_path}"], cwd=root)
    new_blob = subprocess.check_output(
        ["git", "show", f"{documentation_drift['postrun_documentation_commit']}:{drift_path}"], cwd=root)
    if (hashlib.sha256(old_blob).hexdigest().upper() != locked_note_hash
            or hashlib.sha256(new_blob).hexdigest().upper() != documentation_drift["postrun_sha256"]):
        raise RuntimeError("Phase24 documentation-drift commit blobs do not match pinned hashes")
    changed_paths = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r",
         documentation_drift["postrun_documentation_commit"]], cwd=root, text=True).splitlines()
    if not changed_paths or any(not (path.startswith("reports/") or path.startswith("second-brain/"))
                                for path in changed_paths):
        raise RuntimeError("pinned Phase24 post-run commit includes non-documentation changes")

    mismatches = []
    for path, expected in phase24_lock["hashes"].items():
        actual = p17.sha256_file(root / path)
        if actual != expected:
            mismatches.append(path)
            if path != drift_path:
                raise RuntimeError(f"Phase24 locked dependency changed: {path}")
    if mismatches != [drift_path]:
        raise RuntimeError("Phase24 lock drift set differs from the one documented note update")

    phase24_cfg = p24.config(root)
    p24.validate_config(phase24_cfg)
    phase24_out = root / phase24_cfg["outputs"]["run_dir"]
    audit_path, order_path = phase24_out / "layout_audit.json", phase24_out / "execution_order.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    order = json.loads(order_path.read_text(encoding="utf-8"))
    if (p17.sha256_file(audit_path) != phase24_lock["layout_audit_sha256"]
            or p17.sha256_file(order_path) != phase24_lock["schedule_sha256"]
            or order != p24.execution_order(audit["layouts"], phase24_cfg)):
        raise RuntimeError("Phase24 saved layout audit/schedule fails locked parity")
    if (audit.get("overlap_historical") != 0 or audit.get("overlap_outcome_bearing") != 0
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("d168_seed_block_accessed") is not False):
        raise RuntimeError("Phase24 saved layout audit violates sealed-data scope")
    for path, digest in audit["historical_source_hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"Phase24 historical identity source changed: {path}")

    phase24_lock_hash = p17.sha256_file(lock_path)
    raw_path = phase24_out / "episodes.jsonl"
    raw_hash = p17.sha256_file(raw_path)
    raw_manifest = json.loads((phase24_out / "run_manifest.json").read_text(encoding="utf-8"))
    if (raw_manifest.get("execution_lock_sha256") != phase24_lock_hash
            or raw_manifest.get("records_sha256") != raw_hash
            or raw_manifest.get("episodes") != len(order)):
        raise RuntimeError("Phase24 run manifest does not bind the raw records")
    records = p24._load_records(raw_path)
    result_path = root / digest_bindings["phase24_result"]["path"]
    score_manifest_path = root / digest_bindings["phase24_manifest"]["path"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    if (score_manifest.get("execution_lock_sha256") != phase24_lock_hash
            or score_manifest.get("records_sha256") != raw_hash
            or score_manifest.get("result_sha256") != p17.sha256_file(result_path)):
        raise RuntimeError("Phase24 rescored manifest does not bind the raw/result artifacts")
    if p24score._score_records(records, order, phase24_cfg, phase24_lock_hash, raw_hash) != result:
        raise RuntimeError("Phase24 corrected result does not reproduce from its raw episodes")
    return {name: value["sha256"] for name, value in digest_bindings.items()}


def _tracked_dependencies(root: Path, cfg: dict) -> list[str]:
    tracked = subprocess.check_output(["git", "ls-files", "src", "tests", "configs"],
                                      cwd=root, text=True).splitlines()
    sources = [path for path in tracked if path.endswith((".py", ".yaml"))]
    required = [CONFIG, cfg["protocol"], "reports/phase24_search_budget_profile_protocol.md",
                "reports/phase24_search_budget_profile_result.md",
                "reports/phase24_scoring_abort_amendment.md",
                "second-brain/phases/Phase_24_Search_Budget_Profile.md",
                "src/consequence_lab/phase24_search_budget_profile.py",
                "src/consequence_lab/phase24_postrun_score.py",
                *list(_parent_paths(cfg).values())]
    return sorted(set(sources + required + [path for path in ("pyproject.toml", "uv.lock")
                                              if (root / path).exists()]))


def freeze(root: Path = ROOT) -> dict:
    cfg = config(root)
    validate_config(cfg)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit Phase25 protocol, config, implementation, tests, and vault note before freeze")
    parents = _verify_parents(root, cfg)
    start, count = int(cfg["block"]["seed_start"]), int(cfg["block"]["layout_count"])
    assert_unsealed(start, count, cfg["block"])
    out, lock_path = root / cfg["outputs"]["run_dir"], root / cfg["outputs"]["execution_lock"]
    if out.exists() or lock_path.exists():
        raise FileExistsError("Phase25 output or execution lock already exists")
    audit = layout_hash_audit(start, count, root)
    if (audit["layout_count"] != count or audit["unique_hashes"] != count
            or audit["overlap_historical"] or audit["overlap_outcome_bearing"]):
        raise RuntimeError("Phase25 fresh development layout audit failed")
    if set(audit["historical_source_hashes"]) & {"artifacts/runs/phase25_b24_confirmation_v1/layout_audit.json"}:
        raise RuntimeError("Phase25 audit unexpectedly included itself")
    schedule = execution_order(audit["layouts"], cfg)
    out.mkdir(parents=True)
    p18.write_new(out / "layout_audit.json", audit)
    p18.write_new(out / "execution_order.json", schedule)
    lock = {"schema": "phase25-b24-confirmation-lock-v1",
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "created_utc": datetime.now(timezone.utc).isoformat(), "layout_count": count,
            "schedule_count": len(schedule), "parent_hashes": parents,
            "layout_audit_sha256": p17.sha256_file(out / "layout_audit.json"),
            "schedule_sha256": p17.sha256_file(out / "execution_order.json"),
            "hashes": {path: p17.sha256_file(root / path) for path in _tracked_dependencies(root, cfg)},
            "reserve_access": False, "test_access": False, "d168_access": False,
            "training": False, "learned_model_loading": False}
    p18.write_new(lock_path, lock)
    return {"layouts": count, "episodes": len(schedule),
            "layout_audit_sha256": lock["layout_audit_sha256"],
            "schedule_sha256": lock["schedule_sha256"],
            "execution_lock_sha256": p17.sha256_file(lock_path),
            "historical_sources": audit["historical_source_count"],
            "overlap_historical": audit["overlap_historical"],
            "overlap_outcome_bearing": audit["overlap_outcome_bearing"]}


def verify(root: Path = ROOT) -> tuple[dict, Path, list[dict], str]:
    cfg = config(root)
    validate_config(cfg)
    lock_path = root / cfg["outputs"]["execution_lock"]
    try:
        committed = subprocess.check_output(["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"],
                                            cwd=root)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Phase25 execution lock must be committed") from exc
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("Phase25 lock changed after commit")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for path, digest in lock["hashes"].items():
        if p17.sha256_file(root / path) != digest:
            raise RuntimeError(f"Phase25 locked dependency changed: {path}")
    parents = _verify_parents(root, cfg)
    if lock["parent_hashes"] != parents:
        raise RuntimeError("Phase25 parent bindings differ from lock")
    out = root / cfg["outputs"]["run_dir"]
    audit = json.loads((out / "layout_audit.json").read_text(encoding="utf-8"))
    schedule = json.loads((out / "execution_order.json").read_text(encoding="utf-8"))
    if p17.sha256_file(out / "layout_audit.json") != lock["layout_audit_sha256"]:
        raise RuntimeError("Phase25 audit hash differs from immutable lock")
    if p17.sha256_file(out / "execution_order.json") != lock["schedule_sha256"]:
        raise RuntimeError("Phase25 schedule hash differs from immutable lock")
    if schedule != execution_order(audit["layouts"], cfg):
        raise RuntimeError("Phase25 schedule differs from deterministic reconstruction")
    if (audit.get("overlap_historical") != 0 or audit.get("overlap_outcome_bearing") != 0
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("d168_seed_block_accessed") is not False):
        raise RuntimeError("Phase25 saved layout audit violates freshness/scope")
    candidates = {row["layout_sha256"] for row in audit["layouts"]}
    if len(candidates) != cfg["block"]["layout_count"]:
        raise RuntimeError("Phase25 audit has duplicate layout identities")
    for relative, digest in audit["historical_source_hashes"].items():
        if p17.sha256_file(root / relative) != digest:
            raise RuntimeError(f"Phase25 historical identity source changed: {relative}")
    for row in audit["layouts"]:
        if generate_staggered_corridor(int(row["seed"])).layout_sha256 != row["layout_sha256"]:
            raise RuntimeError("Phase25 layout identity does not reproduce")
    return cfg, out, schedule, p17.sha256_file(lock_path)


def run(root: Path = ROOT) -> dict:
    cfg, out, schedule, lock_hash = verify(root)
    raw, manifest_path = out / "episodes.jsonl", out / "run_manifest.json"
    if raw.exists() or manifest_path.exists():
        raise FileExistsError("Phase25 outputs already exist; execution is append-only and non-resumable")
    method_map = {method["name"]: method for method in METHODS}
    started = time.perf_counter()
    with raw.open("x", encoding="utf-8", newline="\n") as stream:
        for index, arm in enumerate(schedule):
            scenario = generate_staggered_corridor(int(arm["seed"]))
            row = p24.evaluate_episode(scenario, method_map[arm["method"]], int(arm["replica"]), cfg)
            row.update({key: arm[key] for key in ("global_schedule_index", "local_order_index")})
            row["execution_lock_sha256"] = lock_hash
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            if (index + 1) % 250 == 0:
                print(json.dumps({"completed": index + 1, "total": len(schedule),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    raw_hash = p17.sha256_file(raw)
    manifest = {"schema": "phase25-b24-confirmation-manifest-v1",
                "execution_lock_sha256": lock_hash, "records_sha256": raw_hash,
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


def _aggregate(records: list[dict], schedule: list[dict], cfg: dict, lock_hash: str,
               records_sha256: str) -> dict:
    if len(records) != len(schedule) or len(records) != 3000:
        raise RuntimeError("Phase25 records do not cover the complete frozen schedule")
    p18.validate_prefix(records, schedule, lock_hash)
    for row, arm in zip(records, schedule, strict=True):
        if any(row.get(key) != value for key, value in arm.items()):
            raise RuntimeError("Phase25 record identity differs from frozen schedule")
        if row.get("outcome") not in {"success", "collision", "timeout"}:
            raise RuntimeError("invalid Phase25 terminal outcome")
        for decision in row["decision_diagnostics"]:
            bank = decision["candidate_bank"]
            if (len(bank) != decision["effective_budget"]
                    or len({tuple(candidate["sequence"]) for candidate in bank}) != len(bank)):
                raise RuntimeError("Phase25 candidate bank incomplete or duplicated")
            if min(tuple(candidate["objective"]) for candidate in bank) != tuple(decision["best_objective"]):
                raise RuntimeError("Phase25 selected plan is not exact candidate-bank minimum")
    keyed = {(row["seed"], row["method"], row["replica"]): row for row in records}
    if len(keyed) != len(records):
        raise RuntimeError("Phase25 duplicate episode identity")
    seeds = sorted({int(row["seed"]) for row in records})
    if len(seeds) != 500:
        raise RuntimeError("Phase25 layout count differs from lock")
    methods, per_replica, vectors = {}, {}, {}
    for method in METHODS:
        name = method["name"]
        rows = [row for row in records if row["method"] == name]
        if len(rows) != 1500:
            raise RuntimeError("Phase25 method episode count mismatch")
        latencies = [value for row in rows for value in row["latency_seconds_per_decision"]]
        methods[name] = {"episodes": len(rows),
            "success_count": sum(row["outcome"] == "success" for row in rows),
            "collision_count": sum(row["outcome"] == "collision" for row in rows),
            "timeout_count": sum(row["outcome"] == "timeout" for row in rows),
            "success_rate": float(np.mean([row["outcome"] == "success" for row in rows])),
            "collision_rate": float(np.mean([row["outcome"] == "collision" for row in rows])),
            "timeout_rate": float(np.mean([row["outcome"] == "timeout" for row in rows])),
            "mean_exact_transitions_per_episode": float(np.mean([row["exact_transitions"] for row in rows])),
            "median_decision_latency_seconds": float(np.median(latencies)),
            "p90_decision_latency_seconds": float(np.quantile(latencies, .90))}
        per_replica[name] = {str(replica): {
            "success_rate": float(np.mean([keyed[(seed, name, replica)]["outcome"] == "success" for seed in seeds])),
            "collision_rate": float(np.mean([keyed[(seed, name, replica)]["outcome"] == "collision" for seed in seeds]))}
            for replica in cfg["block"]["replicas"]}
        vectors[name] = {
            metric: [float(np.mean([
                float(keyed[(seed, name, replica)]["outcome"] == metric)
                if metric in {"success", "collision"}
                else keyed[(seed, name, replica)]["exact_transitions"]
                for replica in cfg["block"]["replicas"]])) for seed in seeds]
            for metric in ("success", "collision", "transitions")}

    reference, candidate = "shifted_b32", "shifted_b24"
    absolute = cfg["gates"]["absolute"]
    b32_pass = _absolute_pass(methods[reference], per_replica[reference], absolute)
    b24_pass = _absolute_pass(methods[candidate], per_replica[candidate], absolute)
    success = p18.difference_bounds(vectors[reference]["success"], vectors[candidate]["success"], cfg,
                                    cfg["analysis"]["success_difference_one_sided_confidence"])
    collision = p18.difference_bounds(vectors[reference]["collision"], vectors[candidate]["collision"], cfg,
                                      cfg["analysis"]["collision_difference_one_sided_confidence"])
    ratio = np.asarray(vectors[candidate]["transitions"]) / np.asarray(vectors[reference]["transitions"])
    ratio_bounds = p18.difference_bounds(np.zeros(len(seeds)), ratio.tolist(), cfg, .95)
    success_events = sum(any(keyed[(seed, candidate, rep)]["outcome"] != "success"
                             and keyed[(seed, reference, rep)]["outcome"] == "success"
                             for rep in cfg["block"]["replicas"]) for seed in seeds)
    collision_events = sum(any(keyed[(seed, candidate, rep)]["outcome"] == "collision"
                               and keyed[(seed, reference, rep)]["outcome"] != "collision"
                               for rep in cfg["block"]["replicas"]) for seed in seeds)
    event_confidence = cfg["analysis"]["adverse_layout_exact_one_sided_confidence"]
    success_upper = binomial_upper(success_events, len(seeds), event_confidence)
    collision_upper = binomial_upper(collision_events, len(seeds), event_confidence)
    latency_ratio = (methods[candidate]["median_decision_latency_seconds"]
                     / methods[reference]["median_decision_latency_seconds"])
    gate = cfg["gates"]["b24_vs_b32"]
    checks = {
        "b32_absolute_reference": b32_pass,
        "b24_absolute_reliability": b24_pass,
        "success_noninferiority": success["adjusted_lower"] >= gate["success_difference_lower_at_least"],
        "collision_noninferiority": collision["adjusted_upper"] <= gate["collision_difference_upper_at_most"],
        "success_regression_layout_event": success_upper <= gate["success_regression_layout_event_upper_at_most"],
        "added_collision_layout_event": collision_upper <= gate["added_collision_layout_event_upper_at_most"],
        "transition_ratio_point": ratio_bounds["difference"] <= gate["transition_ratio_point_at_most"],
        "transition_ratio_95_upper": ratio_bounds["ci95_upper"] <= gate["transition_ratio_95_upper_at_most"],
        "median_latency_ratio": latency_ratio <= gate["median_latency_ratio_at_most"]}
    if not b32_pass:
        decision = "stop_reference_discrepancy_b32_absolute_gate_failed"
    elif all(checks.values()):
        decision = "phase25_b24_independent_development_confirmation_passed"
    else:
        decision = "retain_shifted_b32_phase25_confirmation_gate_not_passed"
    first = {(int(row["seed"]), int(row["replica"]), int(row["budget"])): row["decision_diagnostics"][0]
             for row in records if row["decision_diagnostics"]}
    nested_count = 0
    action_switch_count = 0
    shared_score_mismatches = 0
    for seed in seeds:
        for replica in cfg["block"]["replicas"]:
            low = first[(seed, replica, 24)]
            high = first[(seed, replica, 32)]
            low_bank = {tuple(item["sequence"]): tuple(item["objective"]) for item in low["candidate_bank"]}
            high_bank = {tuple(item["sequence"]): tuple(item["objective"]) for item in high["candidate_bank"]}
            if not set(low_bank).issubset(high_bank):
                raise RuntimeError("Phase25 first-decision banks violate B24 subset-of-B32 contract")
            nested_count += 1
            action_switch_count += int(low["chosen_action"] != high["chosen_action"])
            shared_score_mismatches += sum(high_bank[sequence] != objective for sequence, objective in low_bank.items())
    if shared_score_mismatches:
        raise RuntimeError("shared candidate exact scores differ between budgets")
    return {"schema": "phase25-b24-confirmation-result-v1", "layouts": len(seeds), "episodes": len(records),
        "methods": methods, "per_replica": per_replica,
        "comparison": {"success_difference_b24_minus_b32": success,
            "collision_difference_b24_minus_b32": collision,
            "per_layout_transition_ratio_b24_over_b32": ratio_bounds,
            "median_decision_latency_ratio_b24_over_b32": latency_ratio,
            "success_regression_layouts": success_events, "success_regression_upper": success_upper,
            "added_collision_layouts": collision_events, "added_collision_upper": collision_upper},
        "first_decision": {"paired_state_replica_count": nested_count,
            "b24_chosen_first_action_differs_from_b32_fraction": action_switch_count / max(1, nested_count),
            "shared_candidate_score_mismatches": shared_score_mismatches,
            "interpretation_limit": "matched first-decision bank diagnostic, not counterfactual episode effect"},
        "gate_checks": checks, "decision": decision, "selection_authority": "exact_lexicographic_planner",
        "training": False, "reserve_access": False, "test_access": False, "d168_access": False,
        "records_sha256": records_sha256,
        "execution_lock_sha256": lock_hash}


def _absolute_pass(method: dict, replicas: dict, gate: dict) -> bool:
    return (method["success_rate"] > gate["safe_success_strictly_above"]
            and method["collision_rate"] <= gate["collision_at_most"]
            and method["timeout_rate"] == gate["timeout_exactly"]
            and all(row["success_rate"] >= gate["each_replica_success_at_least"]
                    and row["collision_rate"] <= gate["each_replica_collision_at_most"]
                    for row in replicas.values()))


def score(root: Path = ROOT) -> dict:
    cfg, out, schedule, lock_hash = verify(root)
    raw, run_manifest = out / "episodes.jsonl", out / "run_manifest.json"
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    raw_hash = p17.sha256_file(raw)
    if manifest.get("execution_lock_sha256") != lock_hash or manifest.get("records_sha256") != raw_hash:
        raise RuntimeError("Phase25 run manifest does not bind raw episodes")
    result_path, result_manifest = out / "result.json", out / "result_manifest.json"
    if result_path.exists() or result_manifest.exists():
        raise FileExistsError("Phase25 result exists; refusing overwrite")
    result = _aggregate(_load_records(raw), schedule, cfg, lock_hash, raw_hash)
    p18.write_new(result_path, result)
    score_manifest = {"schema": "phase25-b24-confirmation-score-manifest-v1",
        "execution_lock_sha256": lock_hash, "records_sha256": raw_hash,
        "result_sha256": p17.sha256_file(result_path),
        "scorer_sha256": p17.sha256_file(Path(__file__).resolve()),
        "scorer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "episodes": len(schedule), "scored_utc": datetime.now(timezone.utc).isoformat(),
        "reserve_access": False, "test_access": False, "d168_access": False,
        "planner_episode_reruns": 0, "candidate_evaluations_during_scoring": 0}
    p18.write_new(result_manifest, score_manifest)
    return result


def audit(root: Path = ROOT) -> dict:
    cfg, out, schedule, lock_hash = verify(root)
    raw, result_path, result_manifest_path = out / "episodes.jsonl", out / "result.json", out / "result_manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    score_manifest = json.loads(result_manifest_path.read_text(encoding="utf-8"))
    raw_hash = p17.sha256_file(raw)
    if (score_manifest["execution_lock_sha256"] != lock_hash
            or score_manifest["records_sha256"] != raw_hash
            or score_manifest["result_sha256"] != p17.sha256_file(result_path)):
        raise RuntimeError("Phase25 score artifact hashes do not reproduce")
    reproduced = _aggregate(_load_records(raw), schedule, cfg, lock_hash, raw_hash)
    if reproduced != result:
        raise RuntimeError("Phase25 result does not reproduce from raw records")
    return {"audit_pass": True, "episodes": len(schedule),
            "records_sha256": raw_hash, "result_sha256": score_manifest["result_sha256"],
            "decision": result["decision"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "score", "audit"))
    args = parser.parse_args()
    result = {"freeze": freeze, "run": run, "score": score, "audit": audit}[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
