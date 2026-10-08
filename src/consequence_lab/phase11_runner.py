"""Fail-closed Phase 11 parameter-screen runner.

Only the first 8-layout parameter-screen stage is supported. Real execution
requires a separately frozen config, matching stage authorization lock, and an
explicit command-line confirmation. This module never advances to later seed
blocks or trains a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping

import yaml

from consequence_lab.corridor_evaluation import evaluate_corridor_episode
from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor


class Phase11AuthorizationError(RuntimeError):
    """Raised when a requested Phase 11 stage is not explicitly authorized."""


REQUIRED_SOURCE_PATHS = (
    "configs/phase11_corridor_development_draft.yaml",
    "src/consequence_lab/phase11_runner.py",
    "scripts/run_phase11_parameter_screen.py",
    "src/consequence_lab/corridor_evaluation.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
)


@dataclass(frozen=True)
class ParameterScreenCase:
    method: str
    parameters: Mapping[str, float | int]
    role: str = "parameter_screen_grid"


def enumerate_parameter_screen_cases(config: Mapping[str, Any]) -> tuple[ParameterScreenCase, ...]:
    """Expand the declared grid deterministically, plus the fine diagnostic level."""
    grids = config["baseline_tuning_ranges"]
    cases: list[ParameterScreenCase] = []
    for obstacle, clearance in product(
        grids["reactive"]["obstacle_influence"],
        grids["reactive"]["clearance_weight"],
    ):
        cases.append(ParameterScreenCase("reactive", {
            "obstacle_influence": obstacle,
            "clearance_weight": clearance,
        }))
    for resolution, tolerance in product(
        grids["geometry_astar_waypoint"]["resolution"],
        grids["geometry_astar_waypoint"]["waypoint_tolerance"],
    ):
        cases.append(ParameterScreenCase("geometry_waypoint", {
            "resolution": resolution,
            "waypoint_tolerance": tolerance,
        }))
    for position, velocity, expansions in product(
        grids["quantized_kinodynamic"]["position_resolution"],
        grids["quantized_kinodynamic"]["velocity_resolution"],
        grids["quantized_kinodynamic"]["max_expansions"],
    ):
        cases.append(ParameterScreenCase("kinodynamic", {
            "position_resolution": position,
            "velocity_resolution": velocity,
            "max_expansions": expansions,
        }))
    shooting = grids["exact_dynamics_random_shooting"]
    for horizon, count in product(shooting["horizon"], shooting["num_sequences"]):
        cases.append(ParameterScreenCase("random_shooting", {
            "horizon": horizon,
            "num_sequences": count,
        }))

    diagnostic = config["kinodynamic_resolution_diagnostic"]
    fine = diagnostic["resolutions"][-1]
    cases.append(ParameterScreenCase("kinodynamic", {
        "position_resolution": fine["position_resolution"],
        "velocity_resolution": fine["velocity_resolution"],
        "max_expansions": diagnostic["matched_max_expansions"],
    }, role="resolution_diagnostic_fine"))
    return tuple(cases)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_parameter_screen_authorization(
    root: Path,
    config_path: Path,
    authorization_path: Path,
    output_path: Path,
    *,
    confirmed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate all gates before task generation or output creation."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    output_path = Path(output_path).resolve()
    if not confirmed:
        raise Phase11AuthorizationError("explicit --execute confirmation is required")
    if not _inside(config_path, root) or not _inside(authorization_path, root):
        raise Phase11AuthorizationError("config and authorization lock must be inside project root")
    draft_config_path = root / "configs" / "phase11_corridor_development_draft.yaml"
    if config_path == draft_config_path.resolve():
        raise Phase11AuthorizationError("the review draft cannot be used as an execution config")
    output_root = (root / "artifacts" / "runs").resolve()
    if not _inside(output_path, output_root):
        raise Phase11AuthorizationError("output path must be inside artifacts/runs")
    if output_path.exists():
        raise Phase11AuthorizationError("output path already exists; refusing to overwrite")

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        lock = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise Phase11AuthorizationError(f"could not read config/authorization lock: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(lock, dict):
        raise Phase11AuthorizationError("config and authorization lock must be mappings")

    if lock.get("format") != "phase11_execution_lock_v1":
        raise Phase11AuthorizationError("missing/unsupported execution-lock format")
    if lock.get("authorization_status") != "authorized":
        raise Phase11AuthorizationError("execution lock is not marked authorized")
    runner_policy = config.get("runner_contract", {})
    try:
        config_relative = config_path.relative_to(root).as_posix()
        authorization_relative = authorization_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise Phase11AuthorizationError("config/authorization path is outside project root") from exc
    if config_relative != runner_policy.get("locked_config_path"):
        raise Phase11AuthorizationError("execution config path differs from runner contract")
    if authorization_relative != runner_policy.get("authorization_lock_path"):
        raise Phase11AuthorizationError("authorization lock path differs from runner contract")
    if lock.get("stage") != "parameter_screen":
        raise Phase11AuthorizationError("execution lock does not authorize parameter_screen")
    if config.get("protocol_status") != "locked_for_execution":
        raise Phase11AuthorizationError("config is not frozen as locked_for_execution")
    if any(config.get(field) is not True for field in (
        "execution_authorized", "performance_data_authorized"
    )) or config.get("training_authorized") is not False:
        raise Phase11AuthorizationError("config authorization flags are not valid for this stage")
    if lock.get("execution_authorized") is not True or lock.get("performance_data_authorized") is not True:
        raise Phase11AuthorizationError("execution lock does not explicitly authorize performance evaluation")
    if lock.get("training_authorized") is not False:
        raise Phase11AuthorizationError("training must remain unauthorized")
    if not isinstance(lock.get("approval_reference"), str) or not lock["approval_reference"].strip():
        raise Phase11AuthorizationError("execution lock must record an approval reference")
    if not isinstance(lock.get("approved_at_utc"), str) or not lock["approved_at_utc"].strip():
        raise Phase11AuthorizationError("execution lock must record an approval timestamp")
    try:
        approved_at = datetime.fromisoformat(lock["approved_at_utc"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise Phase11AuthorizationError("execution approval timestamp is invalid") from exc
    if approved_at.tzinfo is None or approved_at.utcoffset() != timezone.utc.utcoffset(approved_at):
        raise Phase11AuthorizationError("execution approval timestamp must be UTC")
    if not isinstance(lock.get("base_git_commit"), str) or not lock["base_git_commit"].strip():
        raise Phase11AuthorizationError("execution lock must record the base Git commit")

    draft_config = yaml.safe_load(draft_config_path.read_text(encoding="utf-8"))
    normalized_locked_config = deepcopy(config)
    for key in ("protocol_status", "execution_authorized", "performance_data_authorized", "training_authorized"):
        normalized_locked_config[key] = draft_config[key]
    if normalized_locked_config != draft_config:
        raise Phase11AuthorizationError("locked config differs from the reviewed draft beyond authorization fields")

    expected_config_hash = _sha256(config_path)
    if lock.get("config_sha256") != expected_config_hash:
        raise Phase11AuthorizationError("locked config SHA-256 mismatch")
    task = config.get("task", {})
    generator_path = root / "src" / "consequence_lab" / "corridor_tasks.py"
    if task.get("generator_sha256") != _sha256(generator_path):
        raise Phase11AuthorizationError("config task-generator source pin mismatch")

    source_pins = lock.get("source_sha256")
    if not isinstance(source_pins, dict):
        raise Phase11AuthorizationError("execution lock has no source hash map")
    for relative in REQUIRED_SOURCE_PATHS:
        expected = source_pins.get(relative)
        source_path = root / relative
        if not isinstance(expected, str) or not source_path.is_file() or _sha256(source_path) != expected:
            raise Phase11AuthorizationError(f"execution source pin missing or mismatched: {relative}")

    block = config["seed_partitions"]["design_tuning"]["parameter_screen"]
    if lock.get("seed_block") != {"start": block["start"], "count": block["count"]}:
        raise Phase11AuthorizationError("execution lock seed block differs from parameter_screen config")
    caps = config["compute_caps"]
    limits = lock.get("caps")
    if not isinstance(limits, dict):
        raise Phase11AuthorizationError("execution lock has no resource caps")
    if (
        isinstance(limits.get("wall_seconds"), bool)
        or not isinstance(limits.get("wall_seconds"), int)
        or limits["wall_seconds"] <= 0
        or limits["wall_seconds"] > caps["total_wall_seconds"]
        or isinstance(limits.get("artifact_bytes"), bool)
        or not isinstance(limits.get("artifact_bytes"), int)
        or limits["artifact_bytes"] <= 0
        or limits["artifact_bytes"] > caps["generated_artifact_bytes"]
    ):
        raise Phase11AuthorizationError("execution lock resource caps exceed or violate draft caps")
    runner_policy = config["runner_contract"]
    if runner_policy.get("first_executable_stage") != "parameter_screen":
        raise Phase11AuthorizationError("config runner contract does not match requested stage")
    if (
        limits["artifact_bytes"]
        < runner_policy["maximum_episode_record_bytes"]
        + runner_policy["maximum_manifest_bytes"]
    ):
        raise Phase11AuthorizationError("artifact cap is too small for a safe record/manifest reserve")
    return config, lock


def run_parameter_screen(
    root: Path,
    config_path: Path,
    authorization_path: Path,
    output_path: Path,
    *,
    confirmed: bool,
    evaluator: Callable[..., Any] = evaluate_corridor_episode,
    scenario_factory: Callable[[int], CorridorScenario] = generate_staggered_corridor,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run only the authorized 8-layout parameter screen, preserving compact records.

    The stage checks wall/artifact caps between episodes and records an abort
    status if a cap is reached. It never executes selection/development/reserve
    blocks and never trains.
    """
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    output_path = Path(output_path).resolve()
    config, lock = validate_parameter_screen_authorization(
        root, config_path, authorization_path, output_path, confirmed=confirmed
    )
    block = config["seed_partitions"]["design_tuning"]["parameter_screen"]
    seeds = list(range(block["start"], block["start"] + block["count"]))
    caps = lock["caps"]
    max_record_bytes = config["runner_contract"]["maximum_episode_record_bytes"]
    max_manifest_bytes = config["runner_contract"]["maximum_manifest_bytes"]
    cases = enumerate_parameter_screen_cases(config)
    output_path.mkdir(parents=True, exist_ok=False)
    records_path = output_path / "episodes.jsonl"
    started = monotonic()
    layout_hashes: list[str] = []
    manifest: dict[str, Any] = {
        "experiment": config["experiment"],
        "stage": "parameter_screen",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": _sha256(config_path),
        "authorization_lock_sha256": _sha256(authorization_path),
        "base_git_commit": lock["base_git_commit"],
        "approval_reference": lock["approval_reference"],
        "source_sha256": lock["source_sha256"],
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _distribution_version("numpy"),
            "pyyaml": _distribution_version("PyYAML"),
        },
        "seeds": seeds,
        "expected_cases_per_seed": len(cases),
        "expected_episode_records": len(seeds) * len(cases),
        "layout_sha256": [],
        "completed_episode_records": 0,
        "elapsed_wall_seconds": 0.0,
        "artifact_bytes": 0,
        "authorization_scope": "parameter_screen_only",
        "training_performed": False,
        "errors": [],
    }
    manifest_path = output_path / "manifest.json"

    def persist_manifest() -> None:
        manifest["elapsed_wall_seconds"] = max(0.0, monotonic() - started)
        other_files = sum(
            item.stat().st_size
            for item in output_path.iterdir()
            if item.is_file() and item != manifest_path
        )
        for _ in range(4):
            encoded_manifest = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            total = other_files + len(encoded_manifest)
            if manifest["artifact_bytes"] == total:
                break
            manifest["artifact_bytes"] = total
        encoded_manifest = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        if len(encoded_manifest) > max_manifest_bytes:
            raise RuntimeError("manifest exceeded configured maximum size")
        manifest_path.write_bytes(encoded_manifest)

    persist_manifest()
    try:
        scenarios = [scenario_factory(seed) for seed in seeds]
        if [scenario.seed for scenario in scenarios] != seeds:
            raise RuntimeError("scenario factory returned unexpected seed identities")
        layout_hashes = [scenario.layout_sha256 for scenario in scenarios]
        if len(set(layout_hashes)) != len(layout_hashes):
            raise RuntimeError("parameter-screen layouts are not geometry-hash unique")
        manifest["layout_sha256"] = layout_hashes
        persist_manifest()

        with records_path.open("x", encoding="utf-8", newline="\n") as records:
            for scenario_index, scenario in enumerate(scenarios):
                rotated_cases = cases[scenario_index:] + cases[:scenario_index]
                for case_index, case in enumerate(rotated_cases):
                    elapsed = monotonic() - started
                    if elapsed >= caps["wall_seconds"]:
                        manifest["status"] = "aborted_wall_cap_between_episodes"
                        persist_manifest()
                        return manifest
                    current_bytes = sum(
                        item.stat().st_size for item in output_path.iterdir() if item.is_file()
                    )
                    if current_bytes + max_record_bytes + max_manifest_bytes > caps["artifact_bytes"]:
                        manifest["status"] = "aborted_artifact_cap_between_episodes"
                        persist_manifest()
                        return manifest

                    result = evaluator(
                        scenario,
                        case.method,
                        parameters=dict(case.parameters),
                    )
                    record = {
                        "seed": scenario.seed,
                        "layout_sha256": scenario.layout_sha256,
                        "configuration_id": sha256(json.dumps({
                            "method": case.method,
                            "parameters": dict(case.parameters),
                            "role": case.role,
                        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16],
                        "case_role": case.role,
                        "case_index_in_rotation": case_index,
                        "method": case.method,
                        "parameters": dict(case.parameters),
                        "result": asdict(result),
                    }
                    encoded = (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
                    if len(encoded) > max_record_bytes:
                        raise RuntimeError("episode record exceeded configured maximum record size")
                    if current_bytes + len(encoded) > caps["artifact_bytes"]:
                        manifest["status"] = "aborted_artifact_cap_between_episodes"
                        persist_manifest()
                        return manifest
                    records.write(encoded.decode("utf-8"))
                    records.flush()
                    manifest["completed_episode_records"] += 1
                    persist_manifest()
        if monotonic() - started >= caps["wall_seconds"]:
            manifest["status"] = "aborted_wall_cap_after_last_episode"
        else:
            manifest["status"] = "completed_parameter_screen_only"
        persist_manifest()
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
        persist_manifest()
        raise


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"
