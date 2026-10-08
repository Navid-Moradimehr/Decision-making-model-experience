"""Score the completed Phase23 raw episodes after a post-run aggregation bug.

This module never calls the environment, evaluates a candidate sequence, or
generates planner episodes. It validates the committed execution lock and raw
episode schedule, reconstructs deterministic candidate-bank bookkeeping from
logged query seeds, and applies the original frozen Phase23 summary/gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from consequence_lab import phase17_proposal_guidance as p17
from consequence_lab import phase18_proposal_controls as p18
from consequence_lab import phase20_exact_plan_warmstart as p20
from consequence_lab import phase23_structured_screen as p23

ROOT = Path(__file__).resolve().parents[2]
RESULT_NAME = "rescored_result.json"
MANIFEST_NAME = "rescored_manifest.json"


def _load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _reconstruct_episode_counters(row: dict, method: dict, cfg: dict) -> dict:
    """Recover omitted base sampler counters without running environment dynamics."""
    if row["method"] == "geometry_waypoint":
        if row.get("duplicate_rejections") != 0 or row.get("fallback_sequences") != 0:
            raise RuntimeError("geometry non-planner counters differ from the frozen record")
        return {"duplicate_rejections": 0, "fallback_sequences": 0}
    diagnostics = row.get("decision_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != row["decision_count"]:
        raise RuntimeError("planner decision diagnostics do not match decision_count")

    previous = None
    duplicates = fallbacks = 0
    xor = int(cfg["planning"]["random_mutation_seed_xor"])
    for diagnostic in diagnostics:
        horizon = int(diagnostic["horizon"])
        query = int(diagnostic["query_seed"])
        if query != p17.query_seed(int(row["seed"]), int(diagnostic["decision_index"]),
                                   int(row["replica"])):
            raise RuntimeError("logged planner query seed differs from the frozen seed rule")
        bank, _, _, construction, _ = p23._make_bank(
            horizon, method, query, previous, cfg)
        if len(bank) != int(diagnostic["budget"]) or len(set(bank)) != len(bank):
            raise RuntimeError("reconstructed candidate bank differs in size/uniqueness")
        action_counts = np.bincount([sequence[0] for sequence in bank], minlength=9).tolist()
        if action_counts != diagnostic.get("candidate_action_counts"):
            raise RuntimeError("reconstructed first-action counts differ from the raw record")
        chosen = tuple(int(action) for action in diagnostic["chosen_sequence"])
        if (chosen not in bank or not chosen
                or int(diagnostic["chosen_action"]) != chosen[0]):
            raise RuntimeError("logged exact winner is inconsistent with reconstructed bank")

        # The Phase23 shifted/mutation branch retained zero placeholders for
        # these episode aggregates. Recompute the base uniform sampler's exact
        # deterministic counters using the same locked query/horizon/budget.
        _, base = p20.candidate_bank(
            horizon, int(method["budget"]), query, "uniform", None, 0, xor)
        if method["kind"] == "uniform":
            # Uniform diagnostics already preserve these fields; assert parity.
            for field in ("base_duplicate_rejections", "fallback_sequences"):
                if int(diagnostic.get(field, -1)) != int(base[field]):
                    raise RuntimeError(f"logged uniform {field} differs from deterministic reconstruction")
        duplicates += int(base["base_duplicate_rejections"])
        fallbacks += int(base["fallback_sequences"])
        previous = chosen

    if int(row["exact_sequences"]) != sum(int(d["budget"]) for d in diagnostics):
        raise RuntimeError("episode exact-sequence total differs from decision records")
    if int(row["exact_transitions"]) != sum(int(d["exact_transitions"]) for d in diagnostics):
        raise RuntimeError("episode exact-transition total differs from decision records")
    if int(row["steps"]) != int(row["decision_count"]):
        raise RuntimeError("planner episode must execute exactly one transition per decision")
    return {"duplicate_rejections": duplicates, "fallback_sequences": fallbacks}


def prepare_records(records: list[dict], schedule: list[dict], cfg: dict,
                    lock_hash: str) -> tuple[list[dict], dict[str, dict[str, int]]]:
    if len(records) != len(schedule) or len(records) != cfg["block"]["episodes"]:
        raise RuntimeError("raw Phase23 episodes are not a complete locked schedule")
    p18.validate_prefix(records, schedule, lock_hash)
    method_map = {m["name"]: m for m in cfg["planning"]["methods"]}
    repaired, counters = [], {}
    for record in records:
        method = method_map[record["method"]]
        derived = _reconstruct_episode_counters(record, method, cfg)
        copy = dict(record)
        for key, value in derived.items():
            # Do not overwrite a genuine field if present; the older runner's
            # planner schema omits episode-level values, so only fill omissions.
            if key not in copy:
                copy[key] = value
        repaired.append(copy)
        totals = counters.setdefault(record["method"],
                                     {"duplicate_rejections": 0, "fallback_sequences": 0})
        for key, value in derived.items():
            totals[key] += value
    return repaired, counters


def score(root: Path = ROOT) -> dict:
    cfg, out, schedule, lock_hash = p23.verify(root)
    lock_path = root / cfg["outputs"]["execution_lock"]
    committed_lock = subprocess.check_output(
        ["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"], cwd=root)
    if not p18.committed_text_parity(committed_lock, lock_path.read_bytes()):
        raise RuntimeError("execution lock differs from committed lock")
    raw_path = out / "episodes.jsonl"
    records = _load_records(raw_path)
    prepared, counters = prepare_records(records, schedule, cfg, lock_hash)
    result = p23._summarize(prepared, schedule, cfg)
    result.update({
        "schema": "phase23-postrun-score-amendment-v1",
        "execution_lock_sha256": lock_hash,
        "records_sha256": p17.sha256_file(raw_path),
        "rescore_method": "frozen deterministic candidate-bank counter reconstruction + original Phase23 summary",
        "planner_episode_reruns": 0,
        "candidate_sequence_evaluations_during_rescore": 0,
        "derived_candidate_bank_counters": counters,
        "driver_status": "episodes-complete-summary-driver-failed-schema-mismatch",
        "reserve_access": False,
        "test_access": False,
        "d168_access": False,
    })
    result_path, manifest_path = out / RESULT_NAME, out / MANIFEST_NAME
    if result_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite existing Phase23 rescore artifacts")
    p18.write_new(result_path, result)
    manifest = {
        "schema": "phase23-postrun-score-manifest-v1",
        "execution_lock_sha256": lock_hash,
        "records_sha256": p17.sha256_file(raw_path),
        "result_sha256": p17.sha256_file(result_path),
        "scorer_sha256": p17.sha256_file(Path(__file__).resolve()),
        "scorer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "episodes": len(records),
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "reserve_access": False,
        "test_access": False,
        "d168_access": False,
        "planner_episode_reruns": 0,
        "candidate_sequence_evaluations_during_rescore": 0,
    }
    p18.write_new(manifest_path, manifest)
    return result


def audit(root: Path = ROOT) -> dict:
    cfg, out, schedule, lock_hash = p23.verify(root)
    raw_path = out / "episodes.jsonl"
    result_path, manifest_path = out / RESULT_NAME, out / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest["execution_lock_sha256"] != lock_hash or result["execution_lock_sha256"] != lock_hash:
        raise RuntimeError("rescore artifacts are not bound to the frozen execution lock")
    if manifest["records_sha256"] != p17.sha256_file(raw_path):
        raise RuntimeError("raw Phase23 episode hash differs from rescore manifest")
    if manifest["result_sha256"] != p17.sha256_file(result_path):
        raise RuntimeError("rescored result hash differs from manifest")
    records = _load_records(raw_path)
    prepared, counters = prepare_records(records, schedule, cfg, lock_hash)
    recomputed = p23._summarize(prepared, schedule, cfg)
    for key, value in recomputed.items():
        if result.get(key) != value:
            raise RuntimeError(f"rescored Phase23 result does not reproduce: {key}")
    if result.get("derived_candidate_bank_counters") != counters:
        raise RuntimeError("derived candidate-bank counters do not reproduce")
    return {"audit_pass": True, "episodes": len(records),
            "selected_method": result["selected_method"],
            "records_sha256": manifest["records_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("score", "audit"))
    args = parser.parse_args()
    result = score() if args.command == "score" else audit()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
