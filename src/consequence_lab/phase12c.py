"""Locked Phase 12c-A layout audit and three-method qualification runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import random
import sys
import time
from math import hypot
from numbers import Integral
from time import perf_counter_ns
from typing import Any

import yaml

from consequence_lab.corridor_baselines import _shooting_objective, grid_astar_route, waypoint_braking_action
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import ACTION_VECTORS, corridor_step, validate_corridor_state


class Phase12cAuthorizationError(RuntimeError):
    """Raised when a Phase 12c-A run is not exactly authorized and pinned."""


@dataclass(frozen=True)
class Phase12cEpisodeResult:
    method: str
    seed: int
    layout_sha256: str
    outcome: str
    task_status: str
    steps: int
    control_effort: int
    decision_count: int
    wall_seconds: float
    wall_seconds_per_decision: float | None
    shooting_queries: int
    shooting_success_available_queries: int
    kinodynamic_expansion_limit_queries: int
    planner_failure: str | None
    shooting_candidate_evaluations: int
    shooting_objective_tie_candidates: int


@dataclass(frozen=True)
class Phase12cDecision:
    first_action: int
    objective_key: tuple[int, float, float]
    candidate_success_available: bool
    candidate_count: int
    objective_tie_count: int


def balanced_continuation_suffixes(
    horizon: int, *, suffixes_per_action: int = 14, seed: int = 0
) -> tuple[tuple[int, ...], ...]:
    for name, value in (("horizon", horizon), ("suffixes_per_action", suffixes_per_action), ("seed", seed)):
        lower = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError(f"{name} has invalid integer value")
    if horizon > 8:
        raise ValueError("horizon cannot exceed 8")
    length = int(horizon) - 1
    count = min(int(suffixes_per_action), 9**length)
    if length == 0:
        return ((),)
    if length == 1:
        return tuple((action,) for action in range(9))[:count]
    suffixes = [(action,) * length for action in range(9)][:count]
    seen = set(suffixes)
    rng = random.Random(int(seed))
    while len(suffixes) < count:
        candidate = tuple(rng.randrange(9) for _ in range(length))
        if candidate in seen or len(set(candidate)) == 1:
            continue
        seen.add(candidate)
        suffixes.append(candidate)
    return tuple(suffixes)


def _phase12c_shooting_decision(
    state, spec, *, horizon: int, seed: int, balanced: bool,
    suffixes_per_action: int = 14,
) -> Phase12cDecision:
    validate_corridor_state(state, spec)
    if state.status != "active":
        raise ValueError("shooting requires an active state")
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    h = min(int(horizon), spec.max_steps - state.elapsed_steps)
    if balanced:
        suffixes = balanced_continuation_suffixes(
            h, suffixes_per_action=suffixes_per_action, seed=seed
        )
        sequences = [
            (action,) + suffix
            for action in range(len(ACTION_VECTORS))
            for suffix in suffixes
        ]
    else:
        budget = min(128, 9**h)
        sequences = []
        seen = set()
        for action in range(len(ACTION_VECTORS)):
            sequence = (action,) * h
            if sequence not in seen:
                sequences.append(sequence)
                seen.add(sequence)
                if len(sequences) == budget:
                    break
        rng = random.Random(int(seed))
        while len(sequences) < budget:
            sequence = tuple(rng.randrange(9) for _ in range(h))
            if sequence not in seen:
                sequences.append(sequence)
                seen.add(sequence)
    if not sequences or len(set(sequences)) != len(sequences):
        raise RuntimeError("invalid or duplicate proposal bank")

    rows = []
    any_success = False
    for sequence in sequences:
        current, effort = state, 0.0
        for action in sequence:
            effort += float(hypot(*ACTION_VECTORS[action]) != 0.0)
            transition = corridor_step(current, action, spec)
            current = transition.state
            any_success |= transition.success
            if current.status != "active":
                break
        rows.append((_shooting_objective(state, current, effort, spec.goal), sequence))
    best_key = min(key for key, _ in rows)
    tied = [sequence for key, sequence in rows if key == best_key]
    selected = (
        min(tied, key=lambda sequence: sha256(bytes(sequence)).digest())
        if balanced else next(sequence for key, sequence in rows if key == best_key)
    )
    return Phase12cDecision(selected[0], best_key, any_success, len(sequences), len(tied))


def evaluate_phase12c_episode(scenario, method: str, parameters: dict[str, Any]) -> Phase12cEpisodeResult:
    if method not in {"geometry_waypoint", "random_shooting", "balanced_shooting"}:
        raise ValueError(f"unsupported Phase 12c-A method: {method}")
    state, spec = scenario.initial_state, scenario.spec
    validate_corridor_state(state, spec)
    started = perf_counter_ns()
    route = None
    if method == "geometry_waypoint":
        route = grid_astar_route((state.x, state.y), spec.goal, spec, resolution=float(parameters["resolution"]))
        if route is None:
            elapsed = (perf_counter_ns() - started) / 1e9
            return Phase12cEpisodeResult(
                method, scenario.seed, scenario.layout_sha256, "planner_no_path", state.status,
                0, 0, 1, elapsed, elapsed, 0, 0, 0, "no_geometric_route", 0, 0,
            )
    decisions = effort = queries = available = candidates = ties = 0
    while state.status == "active":
        if method == "geometry_waypoint":
            action = waypoint_braking_action(
                state, route, spec, waypoint_tolerance=float(parameters["waypoint_tolerance"])
            )
        else:
            query_seed = (int(scenario.seed) * 1_000_003 + state.elapsed_steps) % (2**32)
            decision = _phase12c_shooting_decision(
                state, spec, horizon=int(parameters["horizon"]), seed=query_seed,
                balanced=(method == "balanced_shooting"),
                suffixes_per_action=int(parameters.get("suffixes_per_action", 14)),
            )
            action = decision.first_action
            queries += 1
            available += int(decision.candidate_success_available)
            candidates += decision.candidate_count
            ties += decision.objective_tie_count
        decisions += 1
        effort += int(action != 0)
        state = corridor_step(state, action, spec).state
    elapsed = (perf_counter_ns() - started) / 1e9
    return Phase12cEpisodeResult(
        method, scenario.seed, scenario.layout_sha256, state.status, state.status,
        state.elapsed_steps - scenario.initial_state.elapsed_steps, effort, decisions,
        elapsed, elapsed / decisions if decisions else None, queries, available, 0, None,
        candidates, ties,
    )


HISTORICAL_BLOCKS = {
    "phase11_parameter_and_selection": (110000, 110040),
    "phase11_development": (120000, 120100),
    "phase12_train": (140000, 140120),
    "phase12_validation_fitting_only": (141000, 141040),
    "phase12_dagger_reserved": (142000, 142120),
    "phase12_development_reserved": (143000, 143040),
    "phase12b_selection_and_gate": (150000, 150080),
}
SEALED_RESERVE = (130000, 130200)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def audit_phase12c_layouts(config: dict[str, Any]) -> dict[str, Any]:
    """Hash-only audit; never evaluates transitions or accesses reserve layouts."""
    block = config["seed_block"]
    start, count = int(block["start"]), int(block["count"])
    if count != 100:
        raise ValueError("Phase 12c-A requires exactly 100 layouts")
    new_seeds = list(range(start, start + count))
    if set(new_seeds).intersection(range(*SEALED_RESERVE)):
        raise ValueError("new Phase 12c-A seeds overlap sealed reserve")
    new_scenarios = [generate_staggered_corridor(seed) for seed in new_seeds]
    new_hashes = [scenario.layout_sha256 for scenario in new_scenarios]
    if len(set(new_hashes)) != count:
        raise ValueError("new Phase 12c-A block has duplicate layout hashes")

    historical: dict[str, list[str]] = {}
    occupied: set[str] = set()
    for name, (first, stop) in HISTORICAL_BLOCKS.items():
        hashes = [
            generate_staggered_corridor(seed).layout_sha256
            for seed in range(first, stop)
        ]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"duplicate historical layout hash in {name}")
        overlap = occupied.intersection(hashes)
        if overlap:
            raise ValueError(f"historical layout blocks overlap in {name}")
        occupied.update(hashes)
        historical[name] = hashes
    if occupied.intersection(new_hashes):
        raise ValueError("Phase 12c-A layout hashes overlap an opened/reserved nonreserve block")
    return {
        "schema": "phase12c_a_layout_audit_v1",
        "seed_block": {"start": start, "count": count},
        "new_seed_list": new_seeds,
        "layout_sha256": new_hashes,
        "historical_blocks": {
            name: {"start": bounds[0], "stop_exclusive": bounds[1],
                   "layout_sha256": historical[name]}
            for name, bounds in HISTORICAL_BLOCKS.items()
        },
        "pairwise_disjoint": True,
        "reserve_seed_range": {"start": SEALED_RESERVE[0], "stop_exclusive": SEALED_RESERVE[1]},
        "reserve_generated_or_accessed": False,
        "outcomes_generated": False,
    }


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_phase12c_authorization(
    root: Path,
    config_path: Path,
    lock_path: Path,
    output_path: Path,
    *,
    confirmed: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    config_path, lock_path, output_path = config_path.resolve(), lock_path.resolve(), output_path.resolve()
    if not confirmed:
        raise Phase12cAuthorizationError("explicit execution confirmation is required")
    if not _inside(config_path, root) or not _inside(lock_path, root):
        raise Phase12cAuthorizationError("config and authorization lock must be inside project root")
    if not _inside(output_path, root / "artifacts" / "runs") or output_path.exists():
        raise Phase12cAuthorizationError("output must be a new path under artifacts/runs")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise Phase12cAuthorizationError(f"cannot read locked protocol: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(lock, dict):
        raise Phase12cAuthorizationError("config and lock must be mappings")
    if config.get("phase") != "12c-A" or config.get("protocol_status") != "locked_for_execution":
        raise Phase12cAuthorizationError("config is not a locked Phase 12c-A protocol")
    if any(config.get(key) is not True for key in ("execution_authorized", "performance_data_authorized")):
        raise Phase12cAuthorizationError("Phase 12c-A execution is not enabled in config")
    if config.get("training_authorized") is not False:
        raise Phase12cAuthorizationError("training must remain unauthorized")
    if lock.get("format") != "phase12c_a_execution_lock_v1" or lock.get("authorization_status") != "authorized":
        raise Phase12cAuthorizationError("authorization lock missing or unauthorized")
    if lock.get("execution_authorized") is not True or lock.get("training_authorized") is not False:
        raise Phase12cAuthorizationError("lock scope is not Phase 12c-A-only")
    if lock.get("config_sha256") != _sha256(config_path):
        raise Phase12cAuthorizationError("locked config hash mismatch")
    block = config["seed_block"]
    locked_block = lock.get("seed_block", {})
    if not isinstance(locked_block, dict) or (
        locked_block.get("start") != block["start"]
        or locked_block.get("count") != block["count"]
    ):
        raise Phase12cAuthorizationError("authorized seed block mismatch")
    approved = datetime.fromisoformat(lock["approved_at_utc"].replace("Z", "+00:00"))
    if approved.tzinfo is None or approved.utcoffset() != timezone.utc.utcoffset(approved):
        raise Phase12cAuthorizationError("approval timestamp must be UTC")
    if not lock.get("approval_reference") or not lock.get("base_git_commit"):
        raise Phase12cAuthorizationError("approval reference and base commit are required")
    for key, declared_key in (("wall_seconds", "total_wall_seconds"), ("artifact_bytes", "generated_artifact_bytes")):
        cap, maximum = lock.get("caps", {}).get(key), config["compute_caps"][declared_key]
        if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= maximum:
            raise Phase12cAuthorizationError(f"invalid {key} cap")

    audit_path = (root / config["layout_audit"]["path"]).resolve()
    if not _inside(audit_path, root) or not audit_path.is_file():
        raise Phase12cAuthorizationError("hash-only layout audit is missing or unsafe")
    audit_digest = _sha256(audit_path)
    if audit_digest != lock.get("layout_audit_sha256") or audit_digest != config["layout_audit"]["sha256"]:
        raise Phase12cAuthorizationError("layout audit hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("reserve_generated_or_accessed") is not False or audit.get("outcomes_generated") is not False:
        raise Phase12cAuthorizationError("audit did not preserve no-outcome/reserve boundary")
    expected_seeds = list(range(block["start"], block["start"] + block["count"]))
    if audit.get("new_seed_list") != expected_seeds or not audit.get("pairwise_disjoint"):
        raise Phase12cAuthorizationError("layout audit does not match frozen seed block")
    pins = lock.get("source_sha256")
    if not isinstance(pins, dict):
        raise Phase12cAuthorizationError("source hash map is required")
    for relative, expected in pins.items():
        path = (root / relative).resolve()
        if not _inside(path, root) or not path.is_file() or _sha256(path) != expected:
            raise Phase12cAuthorizationError(f"source pin mismatch: {relative}")
    return config, lock, audit


def run_phase12c_a(
    root: Path,
    config_path: Path,
    lock_path: Path,
    output_path: Path,
    *,
    confirmed: bool,
) -> dict[str, Any]:
    config, lock, audit = validate_phase12c_authorization(
        root, config_path, lock_path, output_path, confirmed=confirmed
    )
    root, output_path = root.resolve(), output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    records_path, manifest_path = output_path / "episodes.jsonl", output_path / "manifest.json"
    started = time.monotonic()
    methods = config["methods"]
    seeds = audit["new_seed_list"]
    max_record = int(config["runner_contract"]["maximum_episode_record_bytes"])
    max_manifest = int(config["runner_contract"]["maximum_manifest_bytes"])
    caps = lock["caps"]
    manifest: dict[str, Any] = {
        "experiment": config["experiment"], "stage": "phase12c_a_teacher_qualification",
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": _sha256(config_path), "authorization_lock_sha256": _sha256(lock_path),
        "layout_audit_sha256": _sha256(root / config["layout_audit"]["path"]),
        "source_sha256": lock["source_sha256"], "base_git_commit": lock["base_git_commit"],
        "approval_reference": lock["approval_reference"],
        "software": {"python": sys.version.split()[0], "platform": platform.platform(),
                      "numpy": _version("numpy"), "pyyaml": _version("PyYAML")},
        "seed_block": config["seed_block"], "seeds": seeds,
        "layout_sha256": audit["layout_sha256"],
        "methods": methods, "expected_episode_records": len(seeds) * len(methods),
        "completed_episode_records": 0, "elapsed_wall_seconds": 0.0,
        "artifact_bytes": 0, "authorization_scope": "phase12c_a_only",
        "reserve_accessed": False, "training_performed": False, "errors": [],
    }

    def persist() -> None:
        manifest["elapsed_wall_seconds"] = time.monotonic() - started
        other = sum(p.stat().st_size for p in output_path.iterdir() if p.is_file() and p != manifest_path)
        for _ in range(3):
            encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
            manifest["artifact_bytes"] = other + len(encoded)
        encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(encoded) > max_manifest:
            raise RuntimeError("Phase 12c-A manifest exceeded configured bound")
        manifest_path.write_bytes(encoded)

    persist()
    try:
        scenarios = [generate_staggered_corridor(seed) for seed in seeds]
        if [s.layout_sha256 for s in scenarios] != audit["layout_sha256"]:
            raise RuntimeError("run-time layout hashes differ from hash-only audit")
        with records_path.open("x", encoding="utf-8", newline="\n") as records:
            for i, scenario in enumerate(scenarios):
                rotated = methods[i % len(methods):] + methods[:i % len(methods)]
                for case_index, method_config in enumerate(rotated):
                    elapsed = time.monotonic() - started
                    used = sum(p.stat().st_size for p in output_path.iterdir() if p.is_file())
                    if elapsed >= caps["wall_seconds"]:
                        manifest["status"] = "aborted_wall_cap_between_episodes"
                        persist()
                        return manifest
                    if used + max_record + max_manifest > caps["artifact_bytes"]:
                        manifest["status"] = "aborted_artifact_cap_between_episodes"
                        persist()
                        return manifest
                    result = evaluate_phase12c_episode(
                        scenario,
                        method_config["method"],
                        parameters=method_config["parameters"],
                    )
                    row = {
                        "seed": scenario.seed, "layout_sha256": scenario.layout_sha256,
                        "method": method_config["method"], "parameters": method_config["parameters"],
                        "case_index_in_rotation": case_index, "result": asdict(result),
                    }
                    encoded = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
                    if len(encoded) > max_record or used + len(encoded) > caps["artifact_bytes"]:
                        manifest["status"] = "aborted_artifact_cap_between_episodes"
                        persist()
                        return manifest
                    records.write(encoded.decode("utf-8"))
                    records.flush()
                    manifest["completed_episode_records"] += 1
                    persist()
        manifest["status"] = (
            "completed_phase12c_a_only"
            if time.monotonic() - started < caps["wall_seconds"]
            else "aborted_wall_cap_after_last_episode"
        )
        persist()
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        persist()
        raise


def _version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"
