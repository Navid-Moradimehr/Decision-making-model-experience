"""Separate no-simulation scoring correction for a completed Phase24 run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from consequence_lab import phase17_proposal_guidance as p17
from consequence_lab import phase18_proposal_controls as p18
from consequence_lab import phase24_search_budget_profile as p24
from consequence_lab.phase16d_compute_frontier import binomial_upper

ROOT = Path(__file__).resolve().parents[2]
RESULT_NAME = "rescored_result.json"
MANIFEST_NAME = "rescored_manifest.json"


def _bounds(reference, candidate, cfg, confidence):
    """Call the frozen paired bootstrap with the configured Phase24 seed."""
    amended = dict(cfg)
    amended["analysis"] = {**cfg["analysis"], "bootstrap_seed": cfg["analysis"]["paired_bootstrap_seed"]}
    return p18.difference_bounds(reference, candidate, amended, confidence)


def _score_records(records: list[dict], schedule: list[dict], cfg: dict,
                   lock_hash: str, raw_hash: str) -> dict:
    if len(records) != len(schedule) or len(records) != 6000:
        raise RuntimeError("Phase24 records do not cover the complete frozen schedule")
    p18.validate_prefix(records, schedule, lock_hash)
    for row, arm in zip(records, schedule, strict=True):
        if any(row.get(key) != value for key, value in arm.items()):
            raise RuntimeError("Phase24 record identity differs from frozen schedule")
        if row.get("outcome") not in {"success", "collision", "timeout"}:
            raise RuntimeError("invalid Phase24 episode outcome")
        for decision in row["decision_diagnostics"]:
            bank = decision["candidate_bank"]
            if (len(bank) != decision["effective_budget"]
                    or len({tuple(c["sequence"]) for c in bank}) != len(bank)):
                raise RuntimeError("candidate bank is incomplete or contains duplicate plans")
            if min(tuple(c["objective"]) for c in bank) != tuple(decision["best_objective"]):
                raise RuntimeError("selected objective is not the exact candidate-bank minimum")

    keyed = {(r["seed"], r["method"], r["replica"]): r for r in records}
    if len(keyed) != len(records):
        raise RuntimeError("duplicate Phase24 episode identities")
    seeds = sorted({int(r["seed"]) for r in records})
    first = {(int(r["seed"]), int(r["replica"]), int(r["budget"])): r["decision_diagnostics"][0]
             for r in records if r["decision_diagnostics"]}
    for seed in seeds:
        for rep in range(3):
            banks = {budget: {tuple(c["sequence"]): tuple(c["objective"])
                              for c in first[(seed, rep, budget)]["candidate_bank"]}
                     for budget in (16, 24, 32, 48)}
            for low, high in ((16, 24), (24, 32), (32, 48)):
                if not set(banks[low]).issubset(banks[high]):
                    raise RuntimeError("first-decision candidate banks are not nested")
                if any(banks[high][seq] != obj for seq, obj in banks[low].items()):
                    raise RuntimeError("shared first-decision candidate has inconsistent exact score")

    methods, per_replica, vectors = {}, {}, {}
    for method in p24.METHODS:
        name = method["name"]
        rows = [r for r in records if r["method"] == name]
        latencies = [v for row in rows for v in row["latency_seconds_per_decision"]]
        methods[name] = {"episodes": len(rows),
            "success_count": sum(r["outcome"] == "success" for r in rows),
            "collision_count": sum(r["outcome"] == "collision" for r in rows),
            "timeout_count": sum(r["outcome"] == "timeout" for r in rows),
            "success_rate": float(np.mean([r["outcome"] == "success" for r in rows])),
            "collision_rate": float(np.mean([r["outcome"] == "collision" for r in rows])),
            "timeout_rate": float(np.mean([r["outcome"] == "timeout" for r in rows])),
            "mean_exact_transitions_per_episode": float(np.mean([r["exact_transitions"] for r in rows])),
            "median_decision_latency_seconds": float(np.median(latencies)),
            "p90_decision_latency_seconds": float(np.quantile(latencies, .90))}
        per_replica[name] = {str(rep): {"success_rate": float(np.mean([
            keyed[(seed, name, rep)]["outcome"] == "success" for seed in seeds])),
            "collision_rate": float(np.mean([keyed[(seed, name, rep)]["outcome"] == "collision"
                                              for seed in seeds]))} for rep in range(3)}
        vectors[name] = {metric: [float(np.mean([
            float(keyed[(seed, name, rep)]["outcome"] == metric)
            if metric in {"success", "collision"} else keyed[(seed, name, rep)]["exact_transitions"]
            for rep in range(3)])) for seed in seeds]
            for metric in ("success", "collision", "transitions")}

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
        success = _bounds(vectors[reference]["success"], vectors[name]["success"], cfg,
                          cfg["analysis"]["success_difference_one_sided_confidence"])
        collision = _bounds(vectors[reference]["collision"], vectors[name]["collision"], cfg,
                            cfg["analysis"]["collision_difference_one_sided_confidence"])
        ratios = np.asarray(vectors[name]["transitions"]) / np.asarray(vectors[reference]["transitions"])
        compute = _bounds(np.zeros(len(seeds)), ratios.tolist(), cfg, .95)
        success_events = sum(any(keyed[(seed, name, rep)]["outcome"] != "success"
            and keyed[(seed, reference, rep)]["outcome"] == "success" for rep in range(3)) for seed in seeds)
        collision_events = sum(any(keyed[(seed, name, rep)]["outcome"] == "collision"
            and keyed[(seed, reference, rep)]["outcome"] != "collision" for rep in range(3)) for seed in seeds)
        confidence = cfg["analysis"]["adverse_layout_exact_one_sided_confidence"]
        success_upper = binomial_upper(success_events, len(seeds), confidence)
        collision_upper = binomial_upper(collision_events, len(seeds), confidence)
        latency_ratio = methods[name]["median_decision_latency_seconds"] / methods[reference]["median_decision_latency_seconds"]
        checks = None
        if name in {"shifted_b16", "shifted_b24"}:
            gate = cfg["gates"]["lower_budget"]
            absolute = (methods[name]["success_rate"] > gate["safe_success_strictly_above"]
                and methods[name]["collision_rate"] <= gate["collision_at_most"]
                and methods[name]["timeout_rate"] == gate["timeout_exactly"]
                and all(v["success_rate"] >= gate["each_replica_success_at_least"] and
                        v["collision_rate"] <= gate["each_replica_collision_at_most"]
                        for v in per_replica[name].values()))
            checks = {"absolute_reliability": absolute,
                "shifted_b32_reference_reproduced": b32_absolute,
                "success_noninferiority": success["adjusted_lower"] >= gate["adjusted_success_difference_lower_vs_b32"],
                "collision_noninferiority": collision["adjusted_upper"] <= gate["adjusted_collision_difference_upper_vs_b32"],
                "success_regression_layout_event": success_upper <= gate["success_regression_layout_event_upper_at_most"],
                "added_collision_layout_event": collision_upper <= gate["added_collision_layout_event_upper_at_most"],
                "transition_mean": compute["difference"] <= gate["mean_transition_ratio_vs_b32_at_most"],
                "transition_ci_upper": compute["ci95_upper"] <= gate["transition_ratio_95_upper_vs_b32_at_most"],
                "latency_ratio": latency_ratio <= gate["median_latency_ratio_vs_b32_at_most"]}
        comparisons[name] = {"success_difference_vs_b32": success,
            "collision_difference_vs_b32": collision, "transition_ratio_vs_b32": compute,
            "median_latency_ratio_vs_b32": latency_ratio, "success_regression_layouts": success_events,
            "success_regression_event_upper": success_upper, "added_collision_layouts": collision_events,
            "added_collision_event_upper": collision_upper,
            "promising_for_confirmation": bool(all(checks.values())) if checks is not None else None,
            "gate_checks": checks}

    diagnostics = p24._profile_diagnostics(records)
    decision = p24._phase_decision(comparisons, b32_absolute)
    return {"schema": "phase24-search-budget-profile-result-v1",
        "scoring_amendment": "phase24-score-amendment-v1-bootstrap-config-alias",
        "scoring_failure": "locked runner requested bootstrap_seed; frozen config field is paired_bootstrap_seed",
        "layouts": len(seeds), "episodes": len(records), "methods": methods, "per_replica": per_replica,
        "comparisons": comparisons, "shifted_b32_reference_absolute_pass": b32_absolute,
        "profile_diagnostics": diagnostics, "decision": decision,
        "selection_authority": "exact_lexicographic_planner", "training": False,
        "reserve_access": False, "test_access": False, "d168_access": False,
        "planner_episode_reruns": 0, "candidate_evaluations_during_scoring": 0,
        "records_sha256": raw_hash, "execution_lock_sha256": lock_hash}


def score(root: Path = ROOT) -> dict:
    cfg, out, schedule, lock_hash = p24.verify(root)
    raw, run_manifest = out / "episodes.jsonl", out / "run_manifest.json"
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    raw_hash = p17.sha256_file(raw)
    if manifest.get("execution_lock_sha256") != lock_hash or manifest.get("records_sha256") != raw_hash:
        raise RuntimeError("Phase24 run manifest does not bind the completed raw records")
    result_path, manifest_path = out / RESULT_NAME, out / MANIFEST_NAME
    if result_path.exists() or manifest_path.exists():
        raise FileExistsError("Phase24 amendment output exists; refusing overwrite")
    records = p24._load_records(raw)
    result = _score_records(records, schedule, cfg, lock_hash, raw_hash)
    p18.write_new(result_path, result)
    scorer_path = Path(__file__).resolve()
    scoring_manifest = {"schema": "phase24-postrun-score-manifest-v1",
        "execution_lock_sha256": lock_hash, "records_sha256": raw_hash,
        "result_sha256": p17.sha256_file(result_path), "scorer_sha256": p17.sha256_file(scorer_path),
        "scorer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "episodes": len(records), "scored_utc": datetime.now(timezone.utc).isoformat(),
        "reserve_access": False, "test_access": False, "d168_access": False,
        "planner_episode_reruns": 0, "candidate_evaluations_during_scoring": 0}
    p18.write_new(manifest_path, scoring_manifest)
    return result


def audit(root: Path = ROOT) -> dict:
    cfg, out, schedule, lock_hash = p24.verify(root)
    raw = out / "episodes.jsonl"
    result_path, manifest_path = out / RESULT_NAME, out / MANIFEST_NAME
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_hash = p17.sha256_file(raw)
    if (manifest["execution_lock_sha256"] != lock_hash or manifest["records_sha256"] != raw_hash
            or manifest["result_sha256"] != p17.sha256_file(result_path)):
        raise RuntimeError("Phase24 score amendment artifact hashes do not reproduce")
    recomputed = _score_records(p24._load_records(raw), schedule, cfg, lock_hash, raw_hash)
    if recomputed != result:
        raise RuntimeError("Phase24 amendment result does not reproduce")
    return {"audit_pass": True, "episodes": len(schedule), "result_sha256": manifest["result_sha256"],
            "records_sha256": raw_hash, "decision": result["decision"]}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score", "audit"))
    args = parser.parse_args()
    result = score() if args.command == "score" else audit()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
