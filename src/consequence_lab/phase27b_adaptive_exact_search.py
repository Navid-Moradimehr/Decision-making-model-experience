from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import yaml

from consequence_lab import phase13 as p13
from consequence_lab import phase16c_prior_guided as p16c
from consequence_lab import phase16d_compute_frontier as p16d
from consequence_lab import phase17_proposal_guidance as p17
from consequence_lab import phase18_proposal_controls as p18
from consequence_lab import phase20_exact_plan_warmstart as p20
from consequence_lab import phase24_search_budget_profile as p24
from consequence_lab import phase26_search_feedback as p26
from consequence_lab import phase27a_search_allocation as p27a
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase12_teacher import state_features

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase27b_adaptive_exact_search_v2.yaml"
SOURCE = "src/consequence_lab/phase27b_adaptive_exact_search.py"
TESTS = "tests/test_phase27b_adaptive_exact_search.py"
LOCK_SCHEMA = "phase27b-adaptive-exact-search-lock-v2"
ARMS = ("fixed_b24", "fixed_b32", "learned_adaptive_b24_plus_8",
        "random_rate_matched_b24_plus_8", "prefix_ambiguity_rate_matched_b24_plus_8")
SEALED_PATH_TOKENS = ("d168", "reserve", "test")
CATEGORY_NAMES = {0: "success", 1: "active", 2: "timeout", 3: "collision"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config(root: Path = ROOT) -> dict:
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def validate_config(cfg: dict) -> None:
    if cfg.get("schema") != "phase27b-adaptive-exact-search-v2":
        raise ValueError("wrong Phase27B config schema")
    if cfg.get("status") != "execution-authorized-pending-lock":
        raise RuntimeError("Phase27B config is not in the frozen pre-lock state")
    if cfg.get("authorization", {}).get("scope") != "phase27b-fresh-development-episodes-only":
        raise RuntimeError("Phase27B authorization scope mismatch")
    block, plan, execution = cfg["block"], cfg["planning"], cfg["execution"]
    if (int(block["seed_start"],) != 210000 or int(block["layout_count"]) != 500
            or block["replicas"] != [0, 1, 2]):
        raise ValueError("Phase27B fresh block differs from the authorized protocol")
    for interval in block["sealed_or_prior_ranges_half_open"].values():
        lo, hi = map(int, interval)
        if int(block["seed_start"]) < hi and lo < int(block["seed_start"]) + int(block["layout_count"]):
            raise ValueError("Phase27B block intersects a sealed or consumed seed range")
    if (plan["horizon"] != 8 or plan["base_budget"] != 24 or plan["full_budget"] != 32
            or plan["exact_rollout_scoring"] is not True or plan["exact_final_selection"] is not True):
        raise ValueError("Phase27B exact planner contract changed")
    if tuple(cfg["arms"]) != ARMS:
        raise ValueError("Phase27B arm roster changed")
    if (execution["authorized"] is not True or execution["resume"] is not False
            or execution["interim_scoring"] is not False or execution["training"] is not False
            or execution["reserve_test_d168_access"] is not False):
        raise RuntimeError("Phase27B execution/scope contract invalid")
    if cfg["parent"]["phase27a_model_sha256"] != cfg["allocation"]["model_artifact_sha256"]:
        raise RuntimeError("Phase27B allocator model hash differs from parent binding")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _parent_bindings(root: Path, cfg: dict) -> dict[str, str]:
    parent = cfg["parent"]
    paths = ("phase27a_lock", "phase27a_result", "phase27a_model", "phase27a_manifest",
             "phase24_records", "phase25_records")
    hashes = {}
    for key in paths:
        path, expected = parent[key], parent[f"{key}_sha256"]
        actual = sha256(root / path)
        if actual != expected:
            raise RuntimeError(f"Phase27B parent digest mismatch: {path}")
        hashes[path] = actual
    result = json.loads((root / parent["phase27a_result"]).read_text(encoding="utf-8"))
    model = json.loads((root / parent["phase27a_model"]).read_text(encoding="utf-8"))
    manifest = json.loads((root / parent["phase27a_manifest"]).read_text(encoding="utf-8"))
    if (result.get("status") != "pass" or result.get("final_model") is None
            or canonical_json(result["final_model"]) != canonical_json(model)
            or manifest.get("result_sha256") != hashes[parent["phase27a_result"]]
            or manifest.get("allocation_model_sha256") != hashes[parent["phase27a_model"]]
            or manifest.get("execution_lock_sha256") != hashes[parent["phase27a_lock"]]
            or manifest.get("reserve_test_d168_access") is not False):
        raise RuntimeError("Phase27A result/model/manifest lineage is invalid")
    if not np.isclose(float(model["threshold_oof"]),
                      float(cfg["allocation"]["learned_probability_threshold"]), rtol=0, atol=1e-14):
        raise RuntimeError("Phase27A frozen OOF allocation threshold changed")
    return hashes


def _dependency_paths(root: Path, cfg: dict) -> list[str]:
    fixed = [CONFIG, cfg["protocol"], cfg["amendment"], SOURCE, TESTS,
             "configs/phase27_adaptive_search_allocation_v1.yaml",
             "configs/phase27a_search_allocation_feasibility_v1.yaml",
             "src/consequence_lab/phase27a_search_allocation.py",
             "src/consequence_lab/phase26_search_feedback.py",
             "src/consequence_lab/phase24_search_budget_profile.py",
             "src/consequence_lab/phase20_exact_plan_warmstart.py",
             "src/consequence_lab/phase17_proposal_guidance.py",
             "src/consequence_lab/phase18_proposal_controls.py",
             "src/consequence_lab/phase16c_prior_guided.py",
             "src/consequence_lab/phase16d_compute_frontier.py",
             "src/consequence_lab/phase13.py",
             "src/consequence_lab/corridor_tasks.py",
             "src/consequence_lab/inertial_corridor.py",
             "src/consequence_lab/phase12_teacher.py",
             cfg["parent"]["phase27a_lock"], cfg["parent"]["phase27a_result"],
             cfg["parent"]["phase27a_model"], cfg["parent"]["phase27a_manifest"],
             cfg["parent"]["phase24_records"], cfg["parent"]["phase25_records"],
             "reports/phase27a_historical_feasibility_result.md",
             "reports/phase27_adaptive_search_allocation_protocol.md",
             "reports/phase27_candidate_set_nesting_correction.md"]
    return sorted(set(path for path in fixed if (root / path).is_file()))


def _path_is_sealed(path: Path) -> bool:
    return any(token in part.lower() for part in path.parts for token in SEALED_PATH_TOKENS)


def _safe_read_layout_hashes(path: Path) -> set[str]:
    """Extract identity fields from one allowed artifact without retaining full JSONL payloads."""
    found: set[str] = set()
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        found.update(p13._layout_hashes(json.loads(line)))
        elif path.suffix.lower() == ".json":
            found.update(p13._layout_hashes(json.loads(path.read_text(encoding="utf-8"))))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unreadable allowed historical identity artifact {path}") from exc
    return found


def safe_layout_identity_audit(start: int, count: int, root: Path) -> dict:
    """Hash-only freshness audit; never opens paths marked D168/reserve/test."""
    if start < 0 or count <= 0:
        raise ValueError("invalid fresh layout block")
    scenarios = [generate_staggered_corridor(seed) for seed in range(start, start + count)]
    rows = [{"seed": int(scenario.seed), "layout_sha256": scenario.layout_sha256}
            for scenario in scenarios]
    candidates = {row["layout_sha256"] for row in rows}
    if len(candidates) != count:
        raise RuntimeError("fresh block contains duplicate layout identities")

    all_prior: set[str] = set()
    outcome_prior: set[str] = set()
    source_hashes: dict[str, str] = {}
    skipped: list[str] = []
    run_root = root / "artifacts" / "runs"
    if not run_root.is_dir():
        raise RuntimeError("historical run artifact root is unavailable")
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if _path_is_sealed(path.relative_to(root)):
            skipped.append(path.relative_to(root).as_posix())
            continue
        hashes = _safe_read_layout_hashes(path)
        if not hashes:
            continue
        relative = path.relative_to(root).as_posix()
        all_prior.update(hashes)
        source_hashes[relative] = sha256(path)
        if path.name.lower() == "manifest.json":
            value = json.loads(path.read_text(encoding="utf-8"))
            is_outcome = p13._is_outcome_artifact(path, value)
        else:
            is_outcome = p13._is_outcome_artifact(path, {})
        if is_outcome:
            outcome_prior.update(hashes)
    overlap = candidates & all_prior
    outcome_overlap = candidates & outcome_prior
    if overlap or outcome_overlap:
        raise RuntimeError("fresh layout hash overlaps an allowed prior identity/outcome source")
    return {"schema": "phase27b-safe-layout-identity-audit-v1", "seed_start": int(start),
            "layout_count": int(count), "layouts": rows, "unique_hashes": len(candidates),
            "historical_source_count": len(source_hashes),
            "historical_source_hashes": source_hashes,
            "historical_source_hashes_sha256": hashlib.sha256(
                canonical_json(source_hashes).encode()).hexdigest().upper(),
            "sealed_path_count_skipped_without_opening": len(skipped),
            "sealed_path_names_skipped": skipped,
            "overlap_historical": len(overlap), "overlap_outcome_bearing": len(outcome_overlap),
            "reserve_generated_or_accessed": False, "test_generated_or_accessed": False,
            "d168_generated_or_accessed": False, "outcomes_or_transitions_computed": False}


def _identity_tiebreak(layout_sha256: str, replica: int, decision_index: int, salt: str) -> int:
    payload = f"phase27b|{salt}|{layout_sha256}|{int(replica)}|{int(decision_index)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _derive_ambiguity_cutoff(root: Path, cfg: dict) -> dict:
    """Use consumed Phase24/25 B32 traces only; reconstruct B24 summaries, no rollouts."""
    p27_cfg = p27a.config(root)
    rows = []
    for cohort in ("phase24", "phase25"):
        cohort_rows, stats = p27a._decision_rows(root, p27_cfg, cohort)
        if stats["simulator_calls"] != 0 or stats["candidate_rollout_evaluations"] != 0:
            raise RuntimeError("ambiguity cutoff derivation unexpectedly evaluated candidates")
        for row in cohort_rows:
            gap = float(row["ambiguity_gap"])
            tie = _identity_tiebreak(row["layout_sha256"], row["replica"],
                                     row["decision_index"], "ambiguity")
            rows.append((gap, tie))
    if not rows:
        raise RuntimeError("no historical B24 prefix gaps available")
    rows.sort()
    nominal = float(cfg["allocation"]["ambiguity_rate"])
    selected = max(1, min(len(rows), int(np.ceil(nominal * len(rows)))))
    cutoff_gap, cutoff_hash = rows[selected - 1]
    observed = sum(pair <= (cutoff_gap, cutoff_hash) for pair in rows) / len(rows)
    return {"method": "pooled_consumed_phase24_phase25_order_statistic_with_identity_hash_ties",
            "historical_decisions": len(rows), "nominal_fraction": nominal,
            "gap_cutoff": float(cutoff_gap), "tie_hash_cutoff": int(cutoff_hash),
            "historical_trigger_fraction": float(observed),
            "source_records": [cfg["parent"]["phase24_records"], cfg["parent"]["phase25_records"]]}


def execution_order(layouts: list[dict], cfg: dict) -> list[dict]:
    rows = []
    for layout in layouts:
        arms = [{"seed": int(layout["seed"]), "layout_sha256": layout["layout_sha256"],
                 "method": method, "replica": int(replica)}
                for method in cfg["arms"] for replica in cfg["block"]["replicas"]]
        random.Random(int(cfg["analysis"]["schedule_seed"]) + int(layout["seed"])).shuffle(arms)
        for local_order, arm in enumerate(arms):
            rows.append({**arm, "global_schedule_index": len(rows), "local_order_index": local_order})
    expected = int(cfg["block"]["layout_count"]) * len(ARMS) * len(cfg["block"]["replicas"])
    keys = [(row["seed"], row["method"], row["replica"]) for row in rows]
    if len(rows) != expected or len(set(keys)) != expected:
        raise RuntimeError("Phase27B schedule count or identity uniqueness failure")
    return rows


def _locked_dependencies(cfg: dict, root: Path) -> dict[str, str]:
    return {path: sha256(root / path) for path in _dependency_paths(root, cfg)}


def _iso_date(value: object) -> str:
    """YAML may parse an unquoted ISO date into datetime.date; locks store text only."""
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _verify_reusable_prelock(cfg: dict, root: Path) -> tuple[dict, list[dict]]:
    """Validate and reuse only the v1 identity audit/schedule after the pre-episode abort."""
    recovery = cfg["prelock_recovery"]
    source_dir = root / recovery["source_run_dir"]
    source_lock = root / recovery["source_lock"]
    expected_entries = sorted(recovery["expected_source_directory_entries"])
    if not source_dir.is_dir() or sorted(path.name for path in source_dir.iterdir()) != expected_entries:
        raise RuntimeError("Phase27B v1 source directory is not identity/schedule-only")
    if not source_lock.is_file() or sha256(source_lock) != recovery["source_lock_sha256"]:
        raise RuntimeError("Phase27B preserved v1 partial-lock hash mismatch")
    audit_path, order_path = source_dir / "layout_audit.json", source_dir / "execution_order.json"
    if (sha256(audit_path) != recovery["layout_audit_sha256"]
            or sha256(order_path) != recovery["execution_order_sha256"]):
        raise RuntimeError("Phase27B v1 identity/schedule recovery hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    schedule = json.loads(order_path.read_text(encoding="utf-8"))
    block = cfg["block"]
    if (audit.get("schema") != "phase27b-safe-layout-identity-audit-v1"
            or audit.get("seed_start") != int(block["seed_start"])
            or audit.get("layout_count") != int(block["layout_count"])
            or audit.get("unique_hashes") != int(block["layout_count"])
            or audit.get("overlap_historical") != 0
            or audit.get("overlap_outcome_bearing") != 0
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("test_generated_or_accessed") is not False
            or audit.get("d168_generated_or_accessed") is not False
            or audit.get("outcomes_or_transitions_computed") is not False):
        raise RuntimeError("Phase27B recovered identity audit violates its frozen contract")
    rows = audit.get("layouts", [])
    expected_seeds = list(range(int(block["seed_start"]),
                                int(block["seed_start"]) + int(block["layout_count"])))
    if [int(row["seed"]) for row in rows] != expected_seeds:
        raise RuntimeError("Phase27B recovered layout identities do not match the frozen seed block")
    for row in rows:
        if generate_staggered_corridor(int(row["seed"])).layout_sha256 != row["layout_sha256"]:
            raise RuntimeError("Phase27B recovered layout identity does not regenerate")
    for relative, digest in audit["historical_source_hashes"].items():
        source = root / relative
        if _path_is_sealed(Path(relative)) or not source.is_file() or sha256(source) != digest:
            raise RuntimeError(f"Phase27B historical identity source changed or is sealed: {relative}")
    if schedule != execution_order(rows, cfg):
        raise RuntimeError("Phase27B recovered execution schedule does not reproduce")
    return audit, schedule


def _copy_new_bytes(source: Path, destination: Path) -> None:
    """Create an immutable recovery copy without normalizing JSON serialization."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        writer.write(reader.read())


def freeze(root: Path = ROOT) -> dict:
    cfg = config(root)
    validate_config(cfg)
    block = cfg["block"]
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("commit Phase27B protocol/config/source/tests before identity audit and lock")
    if _parent_bindings(root, cfg) != {
            path: cfg["parent"][key + "_sha256"] for key, path in (
                ("phase27a_lock", cfg["parent"]["phase27a_lock"]),
                ("phase27a_result", cfg["parent"]["phase27a_result"]),
                ("phase27a_model", cfg["parent"]["phase27a_model"]),
                ("phase27a_manifest", cfg["parent"]["phase27a_manifest"]),
                ("phase24_records", cfg["parent"]["phase24_records"]),
                ("phase25_records", cfg["parent"]["phase25_records"]))}:
        raise RuntimeError("Phase27B parent bindings failed")
    # Reproduce the Stage27A analysis once before relying on its fitted allocator.
    p27a.audit(root)
    output = root / cfg["outputs"]["run_dir"]
    lock_path = root / cfg["outputs"]["execution_lock"]
    if output.exists() or lock_path.exists():
        raise FileExistsError("refusing to overwrite Phase27B audit, schedule, or lock")
    audit, schedule = _verify_reusable_prelock(cfg, root)
    ambiguity = _derive_ambiguity_cutoff(root, cfg)
    if canonical_json(ambiguity) != canonical_json(cfg["prelock_recovery"]["ambiguity_cutoff"]):
        raise RuntimeError("Phase27B historical ambiguity cutoff did not reproduce exactly")
    output.mkdir(parents=True)
    source_dir = root / cfg["prelock_recovery"]["source_run_dir"]
    _copy_new_bytes(source_dir / "layout_audit.json", output / "layout_audit.json")
    _copy_new_bytes(source_dir / "execution_order.json", output / "execution_order.json")
    model = json.loads((root / cfg["parent"]["phase27a_model"]).read_text(encoding="utf-8"))
    result = json.loads((root / cfg["parent"]["phase27a_result"]).read_text(encoding="utf-8"))
    lock = {"schema": LOCK_SCHEMA, "source_commit": _git(root, "rev-parse", "HEAD"),
            "parent_hashes": _parent_bindings(root, cfg),
            "dependency_hashes": _locked_dependencies(cfg, root),
            "recovery_source_hashes": {
                cfg["prelock_recovery"]["source_lock"]: cfg["prelock_recovery"]["source_lock_sha256"],
                f"{cfg['prelock_recovery']['source_run_dir']}/layout_audit.json":
                    cfg["prelock_recovery"]["layout_audit_sha256"],
                f"{cfg['prelock_recovery']['source_run_dir']}/execution_order.json":
                    cfg["prelock_recovery"]["execution_order_sha256"],
            },
            "layout_audit_sha256": sha256(output / "layout_audit.json"),
            "schedule_sha256": sha256(output / "execution_order.json"),
            "layout_count": int(block["layout_count"]), "episode_count": len(schedule),
            "learned_threshold": float(result["threshold"]["probability_threshold"]),
            "ambiguity_cutoff": ambiguity,
            "model_schema": model["feature_contract"],
            "execution_authorized": True,
            "authorization_date": _iso_date(cfg["authorization"]["date"]),
            "exact_candidate_evaluations": True, "exact_final_selection": True,
            "training": False, "reserve_test_d168_access": False}
    p18.write_new(lock_path, lock)
    return {"lock_path": lock_path.relative_to(root).as_posix(),
            "execution_lock_sha256": sha256(lock_path),
            "layouts": int(block["layout_count"]), "episodes": len(schedule),
            "historical_identity_sources": audit["historical_source_count"],
            "sealed_paths_skipped_without_opening": audit["sealed_path_count_skipped_without_opening"],
            "ambiguity_cutoff": ambiguity, "reserve_test_d168_access": False}


def _verify_lock(root: Path, cfg: dict) -> tuple[dict, list[dict], str]:
    lock_path = root / cfg["outputs"]["execution_lock"]
    try:
        committed = subprocess.check_output(["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"], cwd=root)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Phase27B execution lock must be committed unchanged") from exc
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("Phase27B lock bytes differ from committed bytes")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != LOCK_SCHEMA:
        raise RuntimeError("invalid Phase27B lock schema")
    if subprocess.run(["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"],
                      cwd=root, capture_output=True, check=False).returncode != 0:
        raise RuntimeError("Phase27B source commit is not an ancestor of lock commit")
    for path, digest in lock["dependency_hashes"].items():
        if sha256(root / path) != digest:
            raise RuntimeError(f"locked Phase27B dependency changed: {path}")
    for path, digest in lock["recovery_source_hashes"].items():
        if sha256(root / path) != digest:
            raise RuntimeError(f"Phase27B preserved v1 recovery source changed: {path}")
    if _parent_bindings(root, cfg) != lock["parent_hashes"]:
        raise RuntimeError("Phase27B parent artifact binding changed")
    out = root / cfg["outputs"]["run_dir"]
    audit_path, order_path = out / "layout_audit.json", out / "execution_order.json"
    if (sha256(audit_path) != lock["layout_audit_sha256"]
            or sha256(order_path) != lock["schedule_sha256"]):
        raise RuntimeError("Phase27B identity audit/schedule hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    schedule = json.loads(order_path.read_text(encoding="utf-8"))
    if (audit.get("reserve_generated_or_accessed") is not False
            or audit.get("test_generated_or_accessed") is not False
            or audit.get("d168_generated_or_accessed") is not False
            or audit.get("overlap_historical") != 0 or audit.get("overlap_outcome_bearing") != 0):
        raise RuntimeError("Phase27B saved identity audit violates freshness/sealing contract")
    for relative, digest in audit["historical_source_hashes"].items():
        source = root / relative
        if _path_is_sealed(Path(relative)) or sha256(source) != digest:
            raise RuntimeError(f"Phase27B allowed historical identity source changed or is sealed: {relative}")
    if schedule != execution_order(audit["layouts"], cfg) or len(schedule) != lock["episode_count"]:
        raise RuntimeError("Phase27B frozen schedule does not reproduce")
    for row in audit["layouts"]:
        if generate_staggered_corridor(int(row["seed"])).layout_sha256 != row["layout_sha256"]:
            raise RuntimeError("Phase27B layout identity does not regenerate")
    return lock, schedule, sha256(lock_path)


def _model_probability(features: list[float], model: dict) -> float:
    p = float(p27a.predict(model, np.asarray(features, dtype=float)[None, :])[0])
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise RuntimeError("Phase27B model emitted an invalid allocation score")
    return p


def _gap_trigger(gap: float, tie_hash: int, cutoff: dict) -> bool:
    return (float(gap), int(tie_hash)) <= (float(cutoff["gap_cutoff"]), int(cutoff["tie_hash_cutoff"]))


def _candidate_record(evaluation) -> dict:
    objective = tuple(evaluation.objective)
    return {"sequence": list(evaluation.sequence), "objective": list(objective),
            "category": CATEGORY_NAMES[int(objective[0])],
            "transitions": int(evaluation.transitions),
            "success_reached": bool(evaluation.success_reached)}


def _feature_and_best(bank: list[dict], state: list[float], previous: list[int] | None) -> tuple[list[float], dict]:
    features = p26._candidate_features(bank, state, previous)
    if len(features) != 73 or not np.all(np.isfinite(np.asarray(features, dtype=float))):
        raise RuntimeError("B24 allocation features violate the frozen Phase27A feature contract")
    return features, p26._winner(bank)


def _primary_target(low: list[dict], high: list[dict]) -> dict:
    low_best, high_best = p26._winner(low), p26._winner(high)
    return {"strict_objective_improves": int(tuple(high_best["objective"]) < tuple(low_best["objective"])),
            "first_action_changes": int(int(high_best["sequence"][0]) != int(low_best["sequence"][0])),
            "primary_decision_impact": int(tuple(high_best["objective"]) < tuple(low_best["objective"])
                                            and int(high_best["sequence"][0]) != int(low_best["sequence"][0])),
            "success_candidate_newly_available": int(
                not any(item["category"] == "success" for item in low)
                and any(item["category"] == "success" for item in high if tuple(item["sequence"]) not
                        in {tuple(c["sequence"]) for c in low}))}


def _evaluate_decision(scenario, state, previous, replica: int, method: str,
                       layout_sha256: str, lock: dict, cfg: dict,
                       model: dict) -> tuple[object, tuple[int, ...], dict]:
    started = time.perf_counter()
    horizon = min(int(cfg["planning"]["horizon"]), scenario.spec.max_steps - state.elapsed_steps)
    query = p17.query_seed(scenario.seed, state.elapsed_steps, replica)
    attempt_cap = int(cfg["planning"]["attempt_cap"])
    b24 = None
    low_evals = []
    low_records = []
    features = None
    trigger, score_value, ambiguity_gap, trigger_reason = False, None, None, "fixed_b24"
    if method == "fixed_b32":
        b32, _ = p24.shifted_bank(horizon, 32, query, previous, attempt_cap)
        b32 = tuple(tuple(int(a) for a in seq) for seq in b32)
        expected_full = min(32, 9 ** horizon)
        if len(b32) != expected_full or len(set(b32)) != expected_full:
            raise RuntimeError("fixed B32 generator violated the frozen terminal-budget rule")
        evaluations = [p16c.evaluate_sequence(state, scenario, seq) for seq in b32]
        trigger_reason = "fixed_b32"
    else:
        b24, _ = p24.shifted_bank(horizon, 24, query, previous, attempt_cap)
        b24 = tuple(tuple(int(a) for a in seq) for seq in b24)
        expected_base = min(24, 9 ** horizon)
        if len(b24) != expected_base or len(set(b24)) != expected_base:
            raise RuntimeError("B24 generator violated the frozen terminal-budget rule")
        low_evals = [p16c.evaluate_sequence(state, scenario, seq) for seq in b24]
        low_records = [_candidate_record(ev) for ev in low_evals]
    if method == "fixed_b24":
        evaluations = low_evals
    elif method not in {"fixed_b32"}:
        expected_full = min(32, 9 ** horizon)
        if expected_full == len(b24):
            trigger_reason = "terminal_exhaustive_no_extension_available"
        elif method == "learned_adaptive_b24_plus_8":
            features, _ = _feature_and_best(low_records, state_features(scenario, state).astype(float).tolist(),
                                             list(previous) if previous is not None else None)
            score_value = _model_probability(features, model)
            trigger = score_value >= float(lock["learned_threshold"])
            trigger_reason = "learned_trigger" if trigger else "learned_no_trigger"
        elif method == "random_rate_matched_b24_plus_8":
            value = _identity_tiebreak(layout_sha256, replica, state.elapsed_steps, "random-extension")
            trigger = value < int(float(cfg["allocation"]["random_rate"]) * (2**64))
            score_value = value / 2**64
            trigger_reason = "random_trigger" if trigger else "random_no_trigger"
        elif method == "prefix_ambiguity_rate_matched_b24_plus_8":
            features, _ = _feature_and_best(low_records, state_features(scenario, state).astype(float).tolist(),
                                             list(previous) if previous is not None else None)
            ambiguity_gap = float(features[-4])
            tie = _identity_tiebreak(layout_sha256, replica, state.elapsed_steps, "ambiguity")
            trigger = _gap_trigger(ambiguity_gap, tie, lock["ambiguity_cutoff"])
            score_value = ambiguity_gap
            trigger_reason = "ambiguity_trigger" if trigger else "ambiguity_no_trigger"
        else:
            raise ValueError(f"unknown Phase27B method {method}")
        if trigger:
            b32, _ = p24.shifted_bank(horizon, 32, query, previous, attempt_cap)
            b32 = tuple(tuple(int(a) for a in seq) for seq in b32)
            expected_full = min(32, 9 ** horizon)
            if (len(set(b24)) != len(b24) or not set(b24) <= set(b32)
                    or len(b32) != expected_full or len(set(b32) - set(b24)) != len(b32) - len(b24)):
                raise RuntimeError("adaptive extension failed exact nested-bank identity contract")
            known = {tuple(ev.sequence): ev for ev in low_evals}
            additions = [seq for seq in b32 if seq not in known]
            high_evals = [p16c.evaluate_sequence(state, scenario, seq) for seq in additions]
            evaluations = low_evals + high_evals
        else:
            evaluations = low_evals

    chosen, ties = p20.choose_lexicographic_best(evaluations)
    latency = time.perf_counter() - started
    chosen_sequence = tuple(chosen.sequence)
    selected_set = {tuple(ev.sequence) for ev in evaluations}
    if chosen_sequence not in selected_set or chosen.objective != min(ev.objective for ev in evaluations):
        raise RuntimeError("selected sequence is not the exact minimum over evaluated candidates")

    if method == "fixed_b32":
        b24, _ = p24.shifted_bank(horizon, 24, query, previous, attempt_cap)
        b24 = tuple(tuple(int(a) for a in seq) for seq in b24)
        b32_set = {tuple(ev.sequence) for ev in evaluations}
        if (len(b24) != min(24, 9 ** horizon) or not set(b24) <= b32_set
                or len(b32_set) != min(32, 9 ** horizon)):
            raise RuntimeError("fixed B32 failed exact nested-bank identity contract")

    state_raw = state_features(scenario, state).astype(float).tolist()
    decision = {"decision_index": int(state.elapsed_steps), "query_seed": int(query),
                "horizon": int(horizon), "previous_plan": list(previous) if previous is not None else None,
                "state_raw52": state_raw, "base_candidate_count": len(b24),
                "evaluated_candidate_count": len(evaluations),
                "extended": bool(len(evaluations) > min(24, 9 ** horizon)),
                "trigger_reason": trigger_reason, "trigger_score": score_value,
                "ambiguity_gap": ambiguity_gap,
                "allocation_features": features,
                "candidate_bank": [_candidate_record(ev) for ev in evaluations],
                "chosen_sequence": list(chosen_sequence), "chosen_action": int(chosen_sequence[0]),
                "best_objective": list(chosen.objective), "tie_count": int(ties),
                "exact_transitions": int(sum(ev.transitions for ev in evaluations)),
                "latency_seconds": float(latency)}

    # Fixed B32 supports same-state, fully observed hindsight allocation diagnostics.
    if method == "fixed_b32":
        full_records = decision["candidate_bank"]
        by_sequence = {tuple(item["sequence"]): item for item in full_records}
        expected_full = min(32, 9 ** horizon)
        if len(by_sequence) != expected_full or not set(b24) <= set(by_sequence):
            raise RuntimeError("fixed-B32 diagnostic candidate identity mismatch")
        low_records_ordered = [by_sequence[seq] for seq in b24]
        features, _ = _feature_and_best(low_records_ordered, state_raw, decision["previous_plan"])
        suffix_records = [item for item in full_records if tuple(item["sequence"]) not in set(b24)]
        targets = _primary_target(low_records_ordered, full_records)
        extension_available = expected_full > len(b24)
        prob = _model_probability(features, model) if extension_available else None
        ambiguity_gap = float(features[-4])
        tie = _identity_tiebreak(layout_sha256, replica, state.elapsed_steps, "ambiguity")
        random_value = _identity_tiebreak(layout_sha256, replica, state.elapsed_steps, "random-extension")
        decision["allocation_diagnostic"] = {"extension_available": extension_available,
            "features": features, "learned_probability": prob,
            "learned_trigger": bool(prob >= float(lock["learned_threshold"])) if extension_available else False,
            "ambiguity_gap": ambiguity_gap,
            "ambiguity_trigger": _gap_trigger(ambiguity_gap, tie, lock["ambiguity_cutoff"])
                if extension_available else False,
            "random_trigger": random_value < int(float(cfg["allocation"]["random_rate"]) * (2**64))
                if extension_available else False,
            "random_hash_value": int(random_value),
            "targets": targets, "low_best_objective": list(p26._winner(low_records_ordered)["objective"]),
            "high_best_objective": list(p26._winner(full_records)["objective"]),
            "suffix_success_candidate": any(item["category"] == "success" for item in suffix_records),
            "b24_success_candidate": any(item["category"] == "success" for item in low_records_ordered),
            "success_candidate_newly_available": targets["success_candidate_newly_available"]}

    successor = corridor_step(state, int(chosen_sequence[0]), scenario.spec).state
    return successor, chosen_sequence, decision


def evaluate_episode(scenario, arm: dict, lock: dict, cfg: dict, model: dict) -> dict:
    state, previous = scenario.initial_state, None
    decisions = []
    while state.status == "active":
        state, selected, row = _evaluate_decision(scenario, state, previous, int(arm["replica"]),
            arm["method"], arm["layout_sha256"], lock, cfg, model)
        decisions.append(row)
        previous = selected
    return {"seed": int(scenario.seed), "layout_sha256": scenario.layout_sha256,
            "method": arm["method"], "replica": int(arm["replica"]),
            "global_schedule_index": int(arm["global_schedule_index"]),
            "local_order_index": int(arm["local_order_index"]),
            "outcome": state.status, "steps": int(state.elapsed_steps),
            "decision_count": len(decisions),
            "exact_sequences": sum(int(row["evaluated_candidate_count"]) for row in decisions),
            "exact_transitions": sum(int(row["exact_transitions"]) for row in decisions),
            "latency_seconds_per_decision": [float(row["latency_seconds"]) for row in decisions],
            "wall_seconds": float(sum(row["latency_seconds"] for row in decisions)),
            "decision_diagnostics": decisions}


def run(root: Path = ROOT) -> dict:
    cfg = config(root)
    validate_config(cfg)
    lock, schedule, lock_hash = _verify_lock(root, cfg)
    out = root / cfg["outputs"]["run_dir"]
    records_path, manifest_path = root / cfg["outputs"]["records"], root / cfg["outputs"]["run_manifest"]
    if records_path.exists() or manifest_path.exists() or (out / "result.json").exists():
        raise FileExistsError("Phase27B is append-only and non-resumable; refusing overwrite")
    started = time.perf_counter()
    model = json.loads((root / cfg["parent"]["phase27a_model"]).read_text(encoding="utf-8"))
    with gzip.open(records_path, "xt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        for index, arm in enumerate(schedule):
            scenario = generate_staggered_corridor(int(arm["seed"]))
            if scenario.layout_sha256 != arm["layout_sha256"]:
                raise RuntimeError("generated layout identity differs from locked schedule")
            row = evaluate_episode(scenario, arm, lock, cfg, model=model)
            row["execution_lock_sha256"] = lock_hash
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            if (index + 1) % 100 == 0:
                print(json.dumps({"completed_episodes": index + 1, "total_episodes": len(schedule),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    manifest = {"schema": "phase27b-adaptive-exact-search-run-manifest-v1",
        "execution_lock_sha256": lock_hash, "records_sha256": sha256(records_path),
        "episodes": len(schedule), "layouts": int(cfg["block"]["layout_count"]),
        "runtime_seconds": float(time.perf_counter() - started),
        "python": platform.python_version(), "numpy": np.__version__,
        "source_commit": _git(root, "rev-parse", "HEAD"), "training": False,
        "reserve_access": False, "test_access": False, "d168_access": False}
    p18.write_new(manifest_path, manifest)
    return manifest


def _bootstrap(reference: list[float], candidate: list[float], seed: int,
               draws: int, confidence: float) -> dict:
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) == 0:
        raise ValueError("paired bootstrap needs aligned nonempty layout vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(draws, len(a)))
    diff = (b - a)[indices].mean(axis=1)
    return {"difference": float(np.mean(b - a)), "ci95_lower": float(np.quantile(diff, .025)),
            "ci95_upper": float(np.quantile(diff, .975)),
            "one_sided_lower": float(np.quantile(diff, 1 - confidence)),
            "one_sided_upper": float(np.quantile(diff, confidence)), "confidence": float(confidence)}


def _absolute_pass(method: dict, replicas: dict, gate: dict) -> bool:
    return (method["safe_success_rate"] > gate["absolute_safe_success_strictly_above"]
            and method["collision_rate"] <= gate["collision_at_most"]
            and method["timeout_rate"] == gate["timeout_exactly"]
            and all(row["safe_success_rate"] >= gate["per_replica_success_at_least"]
                    and row["collision_rate"] <= gate["per_replica_collision_at_most"]
                    for row in replicas.values()))


def _consume_records(path: Path, schedule: list[dict], lock_hash: str, cfg: dict):
    expected = {(row["seed"], row["method"], row["replica"]): row for row in schedule}
    keyed: dict[tuple[int, str, int], dict] = {}
    latencies: dict[str, list[float]] = {name: [] for name in ARMS}
    selected_actions: dict[str, np.ndarray] = {name: np.zeros(9, dtype=np.int64) for name in ARMS}
    extensions: dict[str, list[int]] = {name: [] for name in ARMS}
    diagnostics: dict[str, dict[str, list]] = {name: {} for name in ("learned", "ambiguity", "random")}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["seed"]), str(row["method"]), int(row["replica"]))
            if key not in expected or key in keyed:
                raise RuntimeError("Phase27B output has unexpected/duplicate episode identity")
            arm = expected[key]
            if any(row.get(field) != value for field, value in arm.items()):
                raise RuntimeError("Phase27B episode differs from frozen schedule")
            if row.get("execution_lock_sha256") != lock_hash:
                raise RuntimeError("Phase27B episode lock binding mismatch")
            if row.get("outcome") not in {"success", "collision", "timeout"}:
                raise RuntimeError("invalid Phase27B episode terminal outcome")
            if int(row["decision_count"]) != int(row["steps"]) or len(row["decision_diagnostics"]) != int(row["steps"]):
                raise RuntimeError("episode action/decision count differs from task steps")
            previous = None
            for decision in row["decision_diagnostics"]:
                if decision.get("previous_plan") != previous:
                    raise RuntimeError("Phase27B prior-plan chain is inconsistent")
                if int(decision["query_seed"]) != p17.query_seed(int(row["seed"]),
                    int(decision["decision_index"]), int(row["replica"])):
                    raise RuntimeError("Phase27B decision query seed mismatch")
                bank = decision["candidate_bank"]
                seqs = [tuple(item["sequence"]) for item in bank]
                horizon = int(decision["horizon"])
                expected_base = min(24, 9 ** horizon)
                expected_full = min(32, 9 ** horizon)
                if (len(bank) != int(decision["evaluated_candidate_count"])
                        or len(set(seqs)) != len(seqs)
                        or int(decision["base_candidate_count"]) != expected_base
                        or len(bank) not in {expected_base, expected_full}
                        or bool(decision["extended"]) != (len(bank) > expected_base)):
                    raise RuntimeError("Phase27B evaluated candidate count/uniqueness failure")
                winner = min(bank, key=lambda item: (tuple(item["objective"]), tuple(item["sequence"])))
                if (tuple(winner["sequence"]) != tuple(decision["chosen_sequence"])
                        or int(decision["chosen_action"]) != int(winner["sequence"][0])
                        or tuple(winner["objective"]) != tuple(decision["best_objective"])):
                    raise RuntimeError("Phase27B final selection is not exact minimum over evaluated candidates")
                if sum(int(item["transitions"]) for item in bank) != int(decision["exact_transitions"]):
                    raise RuntimeError("Phase27B exact transition total does not match candidate rows")
                previous = decision["chosen_sequence"]
                selected_actions[row["method"]][int(decision["chosen_action"])] += 1
                extensions[row["method"]].append(int(bool(decision["extended"])))
                latencies[row["method"]].append(float(decision["latency_seconds"]))
                if row["method"] == "fixed_b32":
                    item = decision.get("allocation_diagnostic")
                    if not item:
                        raise RuntimeError("fixed-B32 decision lacks frozen allocation diagnostics")
                    if not item["extension_available"]:
                        continue
                    for name, probability, trigger, gap, target in (
                        ("learned", item["learned_probability"], item["learned_trigger"],
                         None, item["targets"]["primary_decision_impact"]),
                        ("ambiguity", None, item["ambiguity_trigger"], item["ambiguity_gap"],
                         item["targets"]["primary_decision_impact"]),
                        ("random", None, item["random_trigger"], item["random_hash_value"],
                         item["targets"]["primary_decision_impact"])):
                        bucket = diagnostics[name].setdefault(row["layout_sha256"], [])
                        bucket.append({"decision_index": int(decision["decision_index"]),
                            "replica": int(row["replica"]), "probability": probability,
                            "trigger": bool(trigger), "gap": gap, "random_hash": gap if name == "random" else None,
                            "target": int(target),
                            "success_candidate_newly_available": int(item["targets"].get(
                                "success_candidate_newly_available", 0))})
            if sum(float(x) for x in row["latency_seconds_per_decision"]) < 0:
                raise RuntimeError("negative episode latency")
            if int(row["exact_sequences"]) != sum(int(d["evaluated_candidate_count"])
                                                   for d in row["decision_diagnostics"]):
                raise RuntimeError("episode exact-candidate count mismatch")
            if int(row["exact_transitions"]) != sum(int(d["exact_transitions"])
                                                     for d in row["decision_diagnostics"]):
                raise RuntimeError("episode exact-transition count mismatch")
            keyed[key] = {"outcome": row["outcome"], "transitions": int(row["exact_transitions"]),
                          "steps": int(row["steps"]), "layout_sha256": row["layout_sha256"],
                          "wall_seconds": float(row["wall_seconds"])}
            if (index + 1) % 1000 == 0:
                print(json.dumps({"audited_episodes": index + 1, "expected": len(schedule)}), flush=True)
    if set(keyed) != set(expected):
        raise RuntimeError(f"Phase27B output is incomplete: {len(keyed)} / {len(expected)} episodes")
    return keyed, latencies, selected_actions, extensions, diagnostics


def _allocation_diagnostic(diags: dict[str, dict[str, list]], layouts: list[str], fraction: float = .25) -> dict:
    report = {}
    for name, by_layout in diags.items():
        all_rows = [row for layout in layouts for row in by_layout.get(layout, [])]
        labels = np.asarray([row["target"] for row in all_rows], dtype=float)
        triggers = np.asarray([row["trigger"] for row in all_rows], dtype=bool)
        if len(all_rows) == 0:
            raise RuntimeError("no fixed-B32 allocation diagnostic rows")
        per_layout_top, prevalence, top_rate = [], [], []
        for layout in layouts:
            rows = by_layout.get(layout, [])
            if not rows:
                raise RuntimeError("fixed-B32 diagnostic missing a layout")
            if name == "learned":
                ordered = sorted(rows, key=lambda r: (-float(r["probability"]), r["replica"], r["decision_index"]))
            elif name == "ambiguity":
                ordered = sorted(rows, key=lambda r: (float(r["gap"]), r["replica"], r["decision_index"]))
            else:
                ordered = sorted(rows, key=lambda r: (int(r["random_hash"]), r["replica"], r["decision_index"]))
            selected = ordered[:max(1, int(np.ceil(fraction * len(ordered))))]
            per_layout_top.append(float(np.mean([r["target"] for r in selected])))
            prevalence.append(float(np.mean([r["target"] for r in rows])))
            top_rate.append(float(np.mean([r["target"] for r in selected])))
        triggered = labels[triggers]
        report[name] = {"decisions": int(len(labels)), "target_prevalence": float(labels.mean()),
            "actual_frozen_rule_trigger_fraction": float(triggers.mean()),
            "trigger_precision": float(triggered.mean()) if len(triggered) else None,
            "trigger_capture_recall": float(triggered.sum() / labels.sum()) if labels.sum() else None,
            "top_quartile_macro_event_rate": float(np.mean(top_rate)),
            "all_decision_macro_event_rate": float(np.mean(prevalence)),
            "top_quartile_enrichment": float(np.mean(top_rate) / np.mean(prevalence))
                if np.mean(prevalence) > 0 else None,
            "positive_labels": int(labels.sum()),
            "success_candidate_newly_available_count": int(sum(
                row.get("success_candidate_newly_available", 0) for rows in by_layout.values() for row in rows)),
            "interpretation": "same-state fixed-B32 retrospective diagnostic, not adaptive episode effect"}
    return report


def _score(root: Path = ROOT, write_artifacts: bool = True) -> dict:
    cfg = config(root)
    validate_config(cfg)
    lock, schedule, lock_hash = _verify_lock(root, cfg)
    records = root / cfg["outputs"]["records"]
    run_manifest = json.loads((root / cfg["outputs"]["run_manifest"]).read_text(encoding="utf-8"))
    records_hash = sha256(records)
    if (run_manifest.get("execution_lock_sha256") != lock_hash
            or run_manifest.get("records_sha256") != records_hash
            or run_manifest.get("episodes") != len(schedule)
            or run_manifest.get("reserve_access") is not False
            or run_manifest.get("d168_access") is not False):
        raise RuntimeError("Phase27B run manifest or compressed-record hash mismatch")
    result_path = root / cfg["outputs"]["result"]
    result_manifest_path = root / cfg["outputs"]["result_manifest"]
    if write_artifacts and (result_path.exists() or result_manifest_path.exists()):
        raise FileExistsError("Phase27B result exists; refusing overwrite")
    keyed, latencies, actions, extension_rows, diagnostics = _consume_records(records, schedule, lock_hash, cfg)
    seeds = sorted({int(row["seed"]) for row in schedule})
    reps = cfg["block"]["replicas"]
    methods, per_replica, vectors = {}, {}, {}
    for name in ARMS:
        rows = [row for (seed, method, replica), row in keyed.items() if method == name]
        decision_latencies = latencies[name]
        methods[name] = {"episodes": len(rows),
            "success_count": sum(row["outcome"] == "success" for row in rows),
            "collision_count": sum(row["outcome"] == "collision" for row in rows),
            "timeout_count": sum(row["outcome"] == "timeout" for row in rows),
            "safe_success_rate": float(np.mean([row["outcome"] == "success" for row in rows])),
            "collision_rate": float(np.mean([row["outcome"] == "collision" for row in rows])),
            "timeout_rate": float(np.mean([row["outcome"] == "timeout" for row in rows])),
            "mean_exact_transitions_per_episode": float(np.mean([r["transitions"] for r in rows])),
            "median_decision_latency_seconds": float(np.median(decision_latencies)),
            "p90_decision_latency_seconds": float(np.quantile(decision_latencies, .90)),
            "mean_decisions_per_episode": float(np.mean([r["steps"] for r in rows])),
            "extended_decision_fraction": float(np.mean(extension_rows[name])),
            "selected_action_counts": actions[name].tolist()}
        per_replica[name] = {}
        for replica in reps:
            group = [keyed[(seed, name, replica)] for seed in seeds]
            per_replica[name][str(replica)] = {
                "safe_success_rate": float(np.mean([r["outcome"] == "success" for r in group])),
                "collision_rate": float(np.mean([r["outcome"] == "collision" for r in group])),
                "timeout_rate": float(np.mean([r["outcome"] == "timeout" for r in group]))}
        vectors[name] = {metric: [float(np.mean([
            float(keyed[(seed, name, replica)]["outcome"] == metric)
            if metric in {"success", "collision"} else keyed[(seed, name, replica)]["transitions"]
            for replica in reps])) for seed in seeds]
            for metric in ("success", "collision", "transitions")}

    ref = "fixed_b32"
    gate_cfg = cfg["gates"]
    absolute = {name: _absolute_pass(methods[name], per_replica[name], gate_cfg) for name in ARMS}
    comparisons = {}
    for offset, name in enumerate(("fixed_b24", "learned_adaptive_b24_plus_8",
                                   "random_rate_matched_b24_plus_8",
                                   "prefix_ambiguity_rate_matched_b24_plus_8")):
        succ = _bootstrap(vectors[ref]["success"], vectors[name]["success"],
                          int(cfg["analysis"]["bootstrap_seed"]) + offset,
                          int(cfg["analysis"]["bootstrap_draws"]),
                          float(cfg["analysis"]["success_difference_one_sided_confidence"]))
        coll = _bootstrap(vectors[ref]["collision"], vectors[name]["collision"],
                          int(cfg["analysis"]["bootstrap_seed"]) + 10 + offset,
                          int(cfg["analysis"]["bootstrap_draws"]),
                          float(cfg["analysis"]["collision_difference_one_sided_confidence"]))
        ratio = np.asarray(vectors[name]["transitions"]) / np.asarray(vectors[ref]["transitions"])
        trans = _bootstrap([0.0] * len(seeds), ratio.tolist(),
                           int(cfg["analysis"]["bootstrap_seed"]) + 20 + offset,
                           int(cfg["analysis"]["bootstrap_draws"]), .95)
        success_events = sum(any(keyed[(seed, name, rep)]["outcome"] != "success"
                                 and keyed[(seed, ref, rep)]["outcome"] == "success" for rep in reps)
                             for seed in seeds)
        collision_events = sum(any(keyed[(seed, name, rep)]["outcome"] == "collision"
                                   and keyed[(seed, ref, rep)]["outcome"] != "collision" for rep in reps)
                               for seed in seeds)
        confidence = float(cfg["analysis"]["adverse_layout_one_sided_confidence"])
        succ_upper = p16d.binomial_upper(success_events, len(seeds), confidence)
        coll_upper = p16d.binomial_upper(collision_events, len(seeds), confidence)
        latency_ratio = methods[name]["median_decision_latency_seconds"] / methods[ref]["median_decision_latency_seconds"]
        comparisons[name] = {"safe_success_difference_vs_b32": succ,
            "collision_difference_vs_b32": coll,
            "per_layout_exact_transition_ratio_vs_b32": trans,
            "median_decision_latency_ratio_vs_b32": latency_ratio,
            "success_regression_layouts": int(success_events), "success_regression_upper": succ_upper,
            "added_collision_layouts": int(collision_events), "added_collision_upper": coll_upper}
    learned = "learned_adaptive_b24_plus_8"
    random_name = "random_rate_matched_b24_plus_8"
    attr_succ = _bootstrap(vectors[random_name]["success"], vectors[learned]["success"],
        int(cfg["analysis"]["bootstrap_seed"]) + 40, int(cfg["analysis"]["bootstrap_draws"]),
        float(cfg["analysis"]["attribution_one_sided_confidence"]))
    attr_coll = _bootstrap(vectors[random_name]["collision"], vectors[learned]["collision"],
        int(cfg["analysis"]["bootstrap_seed"]) + 41, int(cfg["analysis"]["bootstrap_draws"]),
        float(cfg["analysis"]["attribution_one_sided_confidence"]))
    learned_cmp = comparisons[learned]
    reliability_gate = {
        "b32_absolute_reference": absolute[ref],
        "learned_absolute_reliability": absolute[learned],
        "success_noninferiority_vs_b32": learned_cmp["safe_success_difference_vs_b32"]["one_sided_lower"]
            >= float(gate_cfg["success_difference_lower_vs_b32"]),
        "collision_noninferiority_vs_b32": learned_cmp["collision_difference_vs_b32"]["one_sided_upper"]
            <= float(gate_cfg["collision_difference_upper_vs_b32"]),
        "success_regression_layout_event": learned_cmp["success_regression_upper"]
            <= float(gate_cfg["adverse_layout_event_upper_at_most"]),
        "added_collision_layout_event": learned_cmp["added_collision_upper"]
            <= float(gate_cfg["adverse_layout_event_upper_at_most"]),
        "transition_ratio_mean": learned_cmp["per_layout_exact_transition_ratio_vs_b32"]["difference"]
            <= float(cfg["analysis"]["transition_ratio_mean_at_most_vs_b32"]),
        "transition_ratio_95_upper": learned_cmp["per_layout_exact_transition_ratio_vs_b32"]["ci95_upper"]
            <= float(cfg["analysis"]["transition_ratio_95_upper_at_most_vs_b32"]),
        "median_decision_latency_ratio": learned_cmp["median_decision_latency_ratio_vs_b32"]
            <= float(cfg["analysis"]["median_decision_latency_ratio_at_most_vs_b32"])}
    attribution_gate = {
        "collision_better_than_random": attr_coll["one_sided_upper"]
            < float(gate_cfg["learned_attribution_collision_upper_vs_random_strictly_below"]),
        "safe_success_noninferior_to_random": attr_succ["one_sided_lower"]
            >= float(gate_cfg["learned_attribution_success_lower_vs_random_at_least"])}
    attribution = {"learned_minus_random_success": attr_succ,
                   "learned_minus_random_collision": attr_coll,
                   "checks": attribution_gate,
                   "pass": all(attribution_gate.values())}
    learned_pass = all(reliability_gate.values())
    layout_ids = [row["layout_sha256"] for row in json.loads(
        (root / cfg["outputs"]["layout_audit"]).read_text(encoding="utf-8"))["layouts"]]
    allocation_diag = _allocation_diagnostic(diagnostics, layout_ids)
    result = {"schema": "phase27b-adaptive-exact-search-result-v1",
        "status": "complete", "layouts": len(seeds), "episodes": len(schedule),
        "methods": methods, "per_replica": per_replica, "comparisons": comparisons,
        "absolute_reliability_pass": absolute, "learned_reliability_gate": reliability_gate,
        "learned_reliability_pass": learned_pass, "learning_attribution": attribution,
        "learned_attribution_pass": attribution["pass"],
        "allocation_diagnostics_on_fixed_b32_states": allocation_diag,
        "decision": ("strong_learned_allocation_success" if learned_pass and attribution["pass"]
            else "adaptive_reliability_without_learning_attribution" if learned_pass
            else "stage27b_reliability_compute_gate_not_passed"),
        "selection_authority": "exact_frozen_lexicographic_planner",
        "training": False, "reserve_access": False, "test_access": False, "d168_access": False,
        "records_sha256": records_hash, "execution_lock_sha256": lock_hash}
    if write_artifacts:
        p18.write_new(result_path, result)
    score_manifest = {"schema": "phase27b-adaptive-exact-search-score-manifest-v1",
        "execution_lock_sha256": lock_hash, "records_sha256": records_hash,
        "result_sha256": sha256(result_path), "scorer_sha256": sha256(Path(__file__).resolve()),
        "scorer_commit": _git(root, "rev-parse", "HEAD"), "episodes": len(schedule),
        "reserve_access": False, "test_access": False, "d168_access": False,
        "planner_episode_reruns": 0, "candidate_evaluations_during_scoring": 0}
    if write_artifacts:
        p18.write_new(result_manifest_path, score_manifest)
    return result


def score(root: Path = ROOT) -> dict:
    return _score(root, write_artifacts=True)


def audit(root: Path = ROOT) -> dict:
    cfg = config(root)
    lock, schedule, lock_hash = _verify_lock(root, cfg)
    records = root / cfg["outputs"]["records"]
    run_manifest = json.loads((root / cfg["outputs"]["run_manifest"]).read_text(encoding="utf-8"))
    result_path = root / cfg["outputs"]["result"]
    score_manifest_path = root / cfg["outputs"]["result_manifest"]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
    records_hash = sha256(records)
    if (run_manifest["records_sha256"] != records_hash or run_manifest["execution_lock_sha256"] != lock_hash
            or score_manifest["records_sha256"] != records_hash
            or score_manifest["execution_lock_sha256"] != lock_hash
            or score_manifest["result_sha256"] != sha256(result_path)
            or result["records_sha256"] != records_hash):
        raise RuntimeError("Phase27B artifact hash chain failed")
    if score_manifest["planner_episode_reruns"] != 0 or score_manifest["candidate_evaluations_during_scoring"] != 0:
        raise RuntimeError("Phase27B audit must not rerun planner episodes")
    # Recompute metrics from immutable episode records; never rerun environment rollouts.
    reproduced = _score(root, write_artifacts=False)
    if canonical_json(reproduced) != canonical_json(result):
        raise RuntimeError("Phase27B result does not reproduce from locked episode records")
    return {"audit": "pass", "episodes": len(schedule), "records_sha256": records_hash,
            "result_sha256": sha256(result_path), "execution_lock_sha256": lock_hash,
            "reserve_test_d168_access": False,
            "independent_metric_recomputation": "record-structure-complete; primary metrics hash-bound"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "score", "audit"))
    args = parser.parse_args()
    result = {"freeze": freeze, "run": run, "score": score, "audit": audit}[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
