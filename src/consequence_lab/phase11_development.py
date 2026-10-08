"""Fail-closed runner for Phase 11's independent 100-layout development screen.

The runner evaluates only frozen random-shooting and geometry-waypoint
configurations. It does not tune, access reserve seeds, or train models.
"""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import yaml

from consequence_lab.corridor_evaluation import (
    CorridorEpisodeResult,
    evaluate_corridor_episode,
    summarize_corridor_results,
)
from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.phase11_runner import Phase11AuthorizationError


REQUIRED_SOURCE_PATHS = (
    "configs/phase11_development_draft.yaml",
    "src/consequence_lab/phase11_development.py",
    "scripts/run_phase11_development.py",
    "src/consequence_lab/phase11_selection.py",
    "src/consequence_lab/phase11_runner.py",
    "scripts/run_phase11_frozen_selection.py",
    "src/consequence_lab/corridor_evaluation.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "reports/phase11_frozen_configuration_selection_110008_110039.md",
    "configs/phase11_frozen_selection_locked.yaml",
    "experiments/phase11_frozen_selection_authorization.lock.json",
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase11AuthorizationError(f"invalid selection episode data: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise Phase11AuthorizationError("selection episode rows must be JSON objects")
    return rows


def _validate_selection_provenance(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    provenance = config["selection_provenance"]
    path_keys = {
        "selection_report": "selection_report_sha256",
        "selection_episodes": "selection_episodes_sha256",
        "selection_manifest": "selection_manifest_sha256",
        "selection_locked_config": "selection_locked_config_sha256",
        "selection_authorization_lock": "selection_authorization_lock_sha256",
    }
    files: dict[str, Path] = {}
    for path_key, digest_key in path_keys.items():
        path = (root / provenance[path_key]).resolve()
        if not _inside(path, root) or not path.is_file():
            raise Phase11AuthorizationError(f"selection provenance file missing or unsafe: {provenance[path_key]}")
        if _sha256(path) != provenance[digest_key]:
            raise Phase11AuthorizationError(f"selection provenance hash mismatch: {provenance[path_key]}")
        files[path_key] = path

    try:
        selection_config = yaml.safe_load(files["selection_locked_config"].read_text(encoding="utf-8"))
        selection_lock = json.loads(files["selection_authorization_lock"].read_text(encoding="utf-8"))
        selection_manifest = json.loads(files["selection_manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise Phase11AuthorizationError(f"could not read selection provenance metadata: {exc}") from exc
    if (
        selection_lock.get("stage") != "frozen_configuration_selection"
        or selection_lock.get("authorization_status") != "authorized"
        or selection_lock.get("config_sha256") != _sha256(files["selection_locked_config"])
        or selection_manifest.get("status") != "completed_frozen_configuration_selection_only"
        or selection_manifest.get("authorization_scope") != "frozen_configuration_selection_only"
        or selection_manifest.get("completed_episode_records") != 128
        or selection_manifest.get("seeds") != list(range(110008, 110040))
        or selection_manifest.get("training_performed") is not False
    ):
        raise Phase11AuthorizationError("frozen-selection run is not the expected completed stage")
    if selection_config.get("stage") != "frozen_configuration_selection":
        raise Phase11AuthorizationError("selection config stage is inconsistent")

    # Recheck the source pins of the selection lock, including its dependency
    # chain to the earlier parameter screen, before trusting its chosen baseline.
    for relative, expected in selection_lock.get("source_sha256", {}).items():
        source = root / relative
        if not source.is_file() or _sha256(source) != expected:
            raise Phase11AuthorizationError(f"upstream selection source pin mismatch: {relative}")

    rows = _load_jsonl(files["selection_episodes"])
    if len(rows) != 128:
        raise Phase11AuthorizationError("selection stage must contain exactly 128 episode records")
    seeds = list(range(110008, 110040))
    expected_methods = {"reactive", "geometry_waypoint", "kinodynamic", "random_shooting"}
    for seed in seeds:
        per_seed = [row for row in rows if row.get("seed") == seed]
        if len(per_seed) != 4 or {row.get("method") for row in per_seed} != expected_methods:
            raise Phase11AuthorizationError(f"selection stage is incomplete or duplicated for seed {seed}")
    hashes = selection_manifest.get("layout_sha256", [])
    if len(hashes) != 32 or len(set(hashes)) != 32:
        raise Phase11AuthorizationError("selection layout hashes are incomplete or duplicated")
    if set(hashes).intersection(selection_manifest.get("previous_parameter_screen_layout_sha256", [])):
        raise Phase11AuthorizationError("selection block overlaps parameter-screen layouts")

    results = [CorridorEpisodeResult(**row["result"]) for row in rows]
    summaries = {item.method: item for item in summarize_corridor_results(results)}
    cheap_methods = ("reactive", "geometry_waypoint", "kinodynamic")
    strongest = sorted(
        (summaries[name] for name in cheap_methods),
        key=lambda item: (
            -item.safe_success_rate,
            item.collision_rate,
            item.median_steps_on_success if item.median_steps_on_success is not None else float("inf"),
            item.median_wall_seconds_per_decision
            if item.median_wall_seconds_per_decision is not None else float("inf"),
        ),
    )[0]
    if strongest.method != config["frozen_comparison"]["comparator_method"]:
        raise Phase11AuthorizationError("development comparator is not strongest under frozen selection rule")
    primary = config["frozen_comparison"]
    frozen_parameters = {
        "random_shooting": dict(primary["primary_parameters"]),
        "geometry_waypoint": dict(primary["comparator_parameters"]),
    }
    frozen_ids = {
        "random_shooting": primary["primary_configuration_id"],
        "geometry_waypoint": primary["comparator_configuration_id"],
    }
    for method, parameters in frozen_parameters.items():
        matches = [
            row for row in rows
            if row.get("method") == method
            and row.get("parameters") == parameters
            and row.get("configuration_id") == frozen_ids[method]
        ]
        if len(matches) != 32:
            raise Phase11AuthorizationError(f"frozen {method} configuration differs from selection records")
    return selection_manifest


def validate_development_authorization(
    root: Path,
    config_path: Path,
    authorization_path: Path,
    output_path: Path,
    *,
    confirmed: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    output_path = Path(output_path).resolve()
    if not confirmed:
        raise Phase11AuthorizationError("explicit --execute confirmation is required")
    if not _inside(config_path, root) or not _inside(authorization_path, root):
        raise Phase11AuthorizationError("config and authorization lock must be inside project root")
    draft_path = root / "configs" / "phase11_development_draft.yaml"
    if config_path == draft_path.resolve():
        raise Phase11AuthorizationError("development review draft cannot be executed")
    if not config_path.is_file() or not authorization_path.is_file():
        raise Phase11AuthorizationError("locked config and authorization lock must exist")
    if not _inside(output_path, root / "artifacts" / "runs"):
        raise Phase11AuthorizationError("output path must be inside artifacts/runs")
    if output_path.exists():
        raise Phase11AuthorizationError("output path already exists; refusing to overwrite")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        lock = json.loads(authorization_path.read_text(encoding="utf-8"))
        draft = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise Phase11AuthorizationError(f"could not read development config/lock/draft: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(lock, dict) or not isinstance(draft, dict):
        raise Phase11AuthorizationError("development config, lock, and draft must be mappings")
    if lock.get("format") != "phase11_execution_lock_v1" or lock.get("authorization_status") != "authorized":
        raise Phase11AuthorizationError("development execution lock is missing or unauthorized")
    if lock.get("stage") != "development_screen":
        raise Phase11AuthorizationError("lock does not authorize development_screen")
    policy = config.get("runner_contract", {})
    try:
        config_relative = config_path.relative_to(root).as_posix()
        lock_relative = authorization_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise Phase11AuthorizationError("config or lock escapes project root") from exc
    if config_relative != policy.get("locked_config_path") or lock_relative != policy.get("authorization_lock_path"):
        raise Phase11AuthorizationError("config/lock paths differ from the development runner contract")
    if config.get("stage") != "development_screen" or config.get("protocol_status") != "locked_for_execution":
        raise Phase11AuthorizationError("development config stage/status is not locked")
    if any(config.get(key) is not True for key in ("execution_authorized", "performance_data_authorized")):
        raise Phase11AuthorizationError("development config does not authorize performance data")
    if config.get("training_authorized") is not False or lock.get("training_authorized") is not False:
        raise Phase11AuthorizationError("training must remain unauthorized")
    if lock.get("execution_authorized") is not True or lock.get("performance_data_authorized") is not True:
        raise Phase11AuthorizationError("lock does not authorize development performance data")
    if not isinstance(lock.get("approval_reference"), str) or not lock["approval_reference"].strip():
        raise Phase11AuthorizationError("approval reference is required")
    try:
        approved_at = datetime.fromisoformat(str(lock["approved_at_utc"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise Phase11AuthorizationError("approval timestamp must be valid UTC") from exc
    if approved_at.tzinfo is None or approved_at.utcoffset() != timezone.utc.utcoffset(approved_at):
        raise Phase11AuthorizationError("approval timestamp must be UTC")
    normalized = deepcopy(config)
    for key in ("protocol_status", "execution_authorized", "performance_data_authorized", "training_authorized"):
        normalized[key] = draft[key]
    if normalized != draft:
        raise Phase11AuthorizationError("locked config differs from reviewed draft beyond authorization fields")
    if lock.get("config_path") != config_relative or lock.get("config_sha256") != _sha256(config_path):
        raise Phase11AuthorizationError("locked config path/hash mismatch")
    if not isinstance(lock.get("base_git_commit"), str) or not lock["base_git_commit"].strip():
        raise Phase11AuthorizationError("base Git commit must be recorded")

    pins = lock.get("source_sha256")
    if not isinstance(pins, dict):
        raise Phase11AuthorizationError("source hash map is required")
    for relative in REQUIRED_SOURCE_PATHS:
        source = root / relative
        if not source.is_file() or not isinstance(pins.get(relative), str) or _sha256(source) != pins[relative]:
            raise Phase11AuthorizationError(f"development source pin missing or mismatched: {relative}")
    block = config.get("seed_block", {})
    expected_block = {"start": 120000, "count": 100}
    if {"start": block.get("start"), "count": block.get("count")} != expected_block:
        raise Phase11AuthorizationError("development block must be exactly seeds 120000-120099")
    if lock.get("seed_block") != expected_block:
        raise Phase11AuthorizationError("authorized seed block differs from the development block")
    if config["task"].get("generator_sha256") != _sha256(root / "src/consequence_lab/corridor_tasks.py"):
        raise Phase11AuthorizationError("development task-generator source hash mismatch")
    limits = lock.get("caps")
    declared = config["compute_caps"]
    if not isinstance(limits, dict):
        raise Phase11AuthorizationError("development resource caps are required")
    for key, declared_key in (("wall_seconds", "total_wall_seconds"), ("artifact_bytes", "generated_artifact_bytes")):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > declared[declared_key]:
            raise Phase11AuthorizationError(f"development {key} cap exceeds or violates the draft")
    if policy.get("first_executable_stage") != "development_screen":
        raise Phase11AuthorizationError("runner stage differs from development config")
    if limits["artifact_bytes"] < policy["maximum_episode_record_bytes"] + policy["maximum_manifest_bytes"]:
        raise Phase11AuthorizationError("artifact cap is too small to reserve bounded records and manifest")

    selection_manifest = _validate_selection_provenance(root, config)
    return config, lock, selection_manifest


def run_development_screen(
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
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    output_path = Path(output_path).resolve()
    config, lock, selection_manifest = validate_development_authorization(
        root, config_path, authorization_path, output_path, confirmed=confirmed
    )
    block = config["seed_block"]
    seeds = list(range(block["start"], block["start"] + block["count"]))
    frozen = config["frozen_comparison"]
    cases = (
        (frozen["comparator_method"], dict(frozen["comparator_parameters"]), "cheap_comparator"),
        (frozen["primary_method"], dict(frozen["primary_parameters"]), "planner"),
    )
    caps = lock["caps"]
    max_record = config["runner_contract"]["maximum_episode_record_bytes"]
    max_manifest = config["runner_contract"]["maximum_manifest_bytes"]

    output_path.mkdir(parents=True, exist_ok=False)
    records_path = output_path / "episodes.jsonl"
    started = monotonic()
    manifest: dict[str, Any] = {
        "experiment": config["experiment"],
        "stage": "development_screen",
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
        "methods": [
            {"method": method, "role": role, "parameters": parameters}
            for method, parameters, role in cases
        ],
        "expected_cases_per_seed": len(cases),
        "expected_episode_records": len(seeds) * len(cases),
        "layout_sha256": [],
        "frozen_selection_layout_sha256": selection_manifest["layout_sha256"],
        "parameter_screen_layout_sha256": selection_manifest["previous_parameter_screen_layout_sha256"],
        "completed_episode_records": 0,
        "elapsed_wall_seconds": 0.0,
        "artifact_bytes": 0,
        "authorization_scope": "development_screen_only",
        "reserve_accessed": False,
        "training_performed": False,
        "errors": [],
    }
    manifest_path = output_path / "manifest.json"

    def persist_manifest() -> None:
        manifest["elapsed_wall_seconds"] = max(0.0, monotonic() - started)
        other_size = sum(
            path.stat().st_size for path in output_path.iterdir()
            if path.is_file() and path != manifest_path
        )
        for _ in range(4):
            encoded = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            manifest["artifact_bytes"] = other_size + len(encoded)
        encoded = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        if len(encoded) > max_manifest:
            raise RuntimeError("development manifest exceeded configured maximum size")
        manifest_path.write_bytes(encoded)

    persist_manifest()
    try:
        scenarios = [scenario_factory(seed) for seed in seeds]
        if [scenario.seed for scenario in scenarios] != seeds:
            raise RuntimeError("scenario factory returned unexpected development seed identities")
        hashes = [scenario.layout_sha256 for scenario in scenarios]
        if len(set(hashes)) != len(hashes):
            raise RuntimeError("development layout hashes are not unique")
        prior_hashes = set(selection_manifest["layout_sha256"])
        prior_hashes.update(selection_manifest["previous_parameter_screen_layout_sha256"])
        if set(hashes).intersection(prior_hashes):
            raise RuntimeError("development layout overlaps an opened design layout")
        manifest["layout_sha256"] = hashes
        persist_manifest()

        with records_path.open("x", encoding="utf-8", newline="\n") as records:
            for scenario_index, scenario in enumerate(scenarios):
                rotated = cases[scenario_index % len(cases):] + cases[:scenario_index % len(cases)]
                for case_index, (method, parameters, role) in enumerate(rotated):
                    if monotonic() - started >= caps["wall_seconds"]:
                        manifest["status"] = "aborted_wall_cap_between_episodes"
                        persist_manifest()
                        return manifest
                    current = sum(path.stat().st_size for path in output_path.iterdir() if path.is_file())
                    if current + max_record + max_manifest > caps["artifact_bytes"]:
                        manifest["status"] = "aborted_artifact_cap_between_episodes"
                        persist_manifest()
                        return manifest
                    result = evaluator(scenario, method, parameters=parameters)
                    record = {
                        "seed": scenario.seed,
                        "layout_sha256": scenario.layout_sha256,
                        "configuration_id": (
                            frozen["primary_configuration_id"]
                            if method == frozen["primary_method"]
                            else frozen["comparator_configuration_id"]
                        ),
                        "case_role": "development_screen",
                        "method_role": role,
                        "case_index_in_rotation": case_index,
                        "method": method,
                        "parameters": parameters,
                        "result": asdict(result),
                    }
                    encoded = (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
                    if len(encoded) > max_record:
                        raise RuntimeError("development episode record exceeded configured maximum size")
                    if current + len(encoded) > caps["artifact_bytes"]:
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
            manifest["status"] = "completed_development_screen_only"
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
