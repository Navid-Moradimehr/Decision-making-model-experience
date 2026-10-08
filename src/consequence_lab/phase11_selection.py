"""Fail-closed runner for Phase 11's frozen-configuration selection block.

Default use is plan-only. Executing seeds 110008--110039 requires a separate
locked config and stage-specific authorization record. This module never
retunes configurations, opens development/reserve seeds, or trains a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
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
from consequence_lab.phase11_runner import Phase11AuthorizationError


REQUIRED_SOURCE_PATHS = (
    "configs/phase11_frozen_selection_draft.yaml",
    "src/consequence_lab/phase11_selection.py",
    "src/consequence_lab/phase11_runner.py",
    "scripts/run_phase11_frozen_selection.py",
    "src/consequence_lab/corridor_evaluation.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "reports/phase11_parameter_screen_110000_110007.md",
)


@dataclass(frozen=True)
class FrozenSelectionCase:
    method: str
    role: str
    configuration_id: str
    parameters: Mapping[str, float | int]


def enumerate_frozen_selection_cases(config: Mapping[str, Any]) -> tuple[FrozenSelectionCase, ...]:
    return tuple(
        FrozenSelectionCase(
            method=str(item["method"]),
            role=str(item["role"]),
            configuration_id=str(item["configuration_id"]),
            parameters=dict(item["parameters"]),
        )
        for item in config["frozen_configurations"]
    )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_provenance(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    provenance = config["provenance"]
    paths = {
        "parameter_screen_report": provenance["parameter_screen_report"],
        "parameter_screen_episodes": provenance["parameter_screen_episodes"],
        "parameter_screen_manifest": provenance["parameter_screen_manifest"],
    }
    hashes = {
        "parameter_screen_report": provenance["parameter_screen_report_sha256"],
        "parameter_screen_episodes": provenance["parameter_screen_episodes_sha256"],
        "parameter_screen_manifest": provenance["parameter_screen_manifest_sha256"],
    }
    resolved: dict[str, Path] = {}
    for name, relative in paths.items():
        path = (root / relative).resolve()
        if not _inside(path, root) or not path.is_file():
            raise Phase11AuthorizationError(f"previous-stage provenance file missing or unsafe: {relative}")
        if _sha256(path) != hashes[name]:
            raise Phase11AuthorizationError(f"previous-stage provenance hash mismatch: {relative}")
        resolved[name] = path

    prior_manifest = json.loads(resolved["parameter_screen_manifest"].read_text(encoding="utf-8"))
    expected_seeds = list(range(110000, 110008))
    if (
        prior_manifest.get("status") != "completed_parameter_screen_only"
        or prior_manifest.get("seeds") != expected_seeds
        or prior_manifest.get("completed_episode_records") != 264
        or prior_manifest.get("training_performed") is not False
    ):
        raise Phase11AuthorizationError("previous parameter-screen manifest is not the expected completed run")

    rows = [
        json.loads(line)
        for line in resolved["parameter_screen_episodes"].read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != 264 or {row.get("seed") for row in rows} != set(expected_seeds):
        raise Phase11AuthorizationError("previous parameter-screen episode data is incomplete or out of scope")
    for case in enumerate_frozen_selection_cases(config):
        matching = [
            row for row in rows
            if row.get("method") == case.method
            and row.get("parameters") == dict(case.parameters)
            and row.get("case_role") == "parameter_screen_grid"
            and row.get("configuration_id") == case.configuration_id
        ]
        if len(matching) != 8 or {row["seed"] for row in matching} != set(expected_seeds):
            raise Phase11AuthorizationError(
                f"frozen configuration does not match the completed parameter screen: {case.method}"
            )
    return prior_manifest


def validate_frozen_selection_authorization(
    root: Path,
    config_path: Path,
    authorization_path: Path,
    output_path: Path,
    *,
    confirmed: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate stage, provenance, source pins, scope, and caps before generation."""
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    authorization_path = Path(authorization_path).resolve()
    output_path = Path(output_path).resolve()
    if not confirmed:
        raise Phase11AuthorizationError("explicit --execute confirmation is required")
    if not _inside(config_path, root) or not _inside(authorization_path, root):
        raise Phase11AuthorizationError("config and authorization lock must be inside project root")
    draft_path = root / "configs" / "phase11_frozen_selection_draft.yaml"
    if config_path == draft_path.resolve():
        raise Phase11AuthorizationError("the frozen-selection review draft cannot be executed")
    if not config_path.is_file() or not authorization_path.is_file():
        raise Phase11AuthorizationError("locked config and authorization lock must exist")
    output_root = (root / "artifacts" / "runs").resolve()
    if not _inside(output_path, output_root):
        raise Phase11AuthorizationError("output path must be inside artifacts/runs")
    if output_path.exists():
        raise Phase11AuthorizationError("output path already exists; refusing to overwrite")

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        lock = json.loads(authorization_path.read_text(encoding="utf-8"))
        draft = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise Phase11AuthorizationError(f"could not read config/authorization/draft: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(lock, dict) or not isinstance(draft, dict):
        raise Phase11AuthorizationError("config, lock, and draft must be mappings")

    if lock.get("format") != "phase11_execution_lock_v1" or lock.get("authorization_status") != "authorized":
        raise Phase11AuthorizationError("execution lock is missing or not authorized")
    if lock.get("stage") != "frozen_configuration_selection":
        raise Phase11AuthorizationError("execution lock does not authorize frozen_configuration_selection")
    policy = config.get("runner_contract", {})
    try:
        config_relative = config_path.relative_to(root).as_posix()
        lock_relative = authorization_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise Phase11AuthorizationError("execution config or lock escapes project root") from exc
    if config_relative != policy.get("locked_config_path"):
        raise Phase11AuthorizationError("config path differs from runner contract")
    if lock_relative != policy.get("authorization_lock_path"):
        raise Phase11AuthorizationError("authorization path differs from runner contract")

    if config.get("protocol_status") != "locked_for_execution":
        raise Phase11AuthorizationError("config is not locked for execution")
    if any(config.get(key) is not True for key in ("execution_authorized", "performance_data_authorized")):
        raise Phase11AuthorizationError("locked config does not authorize performance execution")
    if config.get("training_authorized") is not False:
        raise Phase11AuthorizationError("training must remain unauthorized")
    if lock.get("execution_authorized") is not True or lock.get("performance_data_authorized") is not True:
        raise Phase11AuthorizationError("execution lock does not authorize performance data")
    if lock.get("training_authorized") is not False:
        raise Phase11AuthorizationError("authorization lock must leave training unauthorized")
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
        raise Phase11AuthorizationError("locked config differs from the reviewed draft beyond authorization fields")
    if lock.get("config_path") != config_relative or lock.get("config_sha256") != _sha256(config_path):
        raise Phase11AuthorizationError("locked config path or hash differs from authorization lock")

    commit = lock.get("base_git_commit")
    if not isinstance(commit, str) or not commit.strip():
        raise Phase11AuthorizationError("base Git commit must be recorded")
    source_hashes = lock.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise Phase11AuthorizationError("source hash map is required")
    for relative in REQUIRED_SOURCE_PATHS:
        source = root / relative
        expected = source_hashes.get(relative)
        if not source.is_file() or not isinstance(expected, str) or _sha256(source) != expected:
            raise Phase11AuthorizationError(f"source pin missing or mismatched: {relative}")

    block = config["seed_block"]
    expected_block = {"start": 110008, "count": 32}
    if {"start": block.get("start"), "count": block.get("count")} != expected_block:
        raise Phase11AuthorizationError("draft must be frozen to seeds 110008--110039")
    if lock.get("seed_block") != expected_block:
        raise Phase11AuthorizationError("authorization seed block differs from the reviewed selection block")
    if config["task"].get("generator_sha256") != _sha256(root / "src/consequence_lab/corridor_tasks.py"):
        raise Phase11AuthorizationError("task generator source hash mismatch")
    limits = lock.get("caps")
    declared = config["compute_caps"]
    if not isinstance(limits, dict):
        raise Phase11AuthorizationError("resource caps are required in the execution lock")
    for key, config_key in (("wall_seconds", "total_wall_seconds"), ("artifact_bytes", "generated_artifact_bytes")):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > declared[config_key]:
            raise Phase11AuthorizationError(f"execution {key} cap exceeds or violates the reviewed draft")
    if policy.get("first_executable_stage") != "frozen_configuration_selection":
        raise Phase11AuthorizationError("runner stage does not match config")
    if limits["artifact_bytes"] < policy["maximum_episode_record_bytes"] + policy["maximum_manifest_bytes"]:
        raise Phase11AuthorizationError("artifact cap cannot safely reserve record and manifest limits")

    prior_manifest = _validate_provenance(root, config)
    return config, lock, prior_manifest


def run_frozen_configuration_selection(
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
    config, lock, prior_manifest = validate_frozen_selection_authorization(
        root, config_path, authorization_path, output_path, confirmed=confirmed
    )
    block = config["seed_block"]
    seeds = list(range(block["start"], block["start"] + block["count"]))
    cases = enumerate_frozen_selection_cases(config)
    caps = lock["caps"]
    max_record = config["runner_contract"]["maximum_episode_record_bytes"]
    max_manifest = config["runner_contract"]["maximum_manifest_bytes"]

    output_path.mkdir(parents=True, exist_ok=False)
    records_path = output_path / "episodes.jsonl"
    started = monotonic()
    manifest: dict[str, Any] = {
        "experiment": config["experiment"],
        "stage": "frozen_configuration_selection",
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
        "previous_parameter_screen_layout_sha256": prior_manifest["layout_sha256"],
        "completed_episode_records": 0,
        "elapsed_wall_seconds": 0.0,
        "artifact_bytes": 0,
        "authorization_scope": "frozen_configuration_selection_only",
        "training_performed": False,
        "errors": [],
    }
    manifest_path = output_path / "manifest.json"

    def persist_manifest() -> None:
        manifest["elapsed_wall_seconds"] = max(0.0, monotonic() - started)
        other_size = sum(
            item.stat().st_size for item in output_path.iterdir()
            if item.is_file() and item != manifest_path
        )
        for _ in range(4):
            encoded = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            manifest["artifact_bytes"] = other_size + len(encoded)
        encoded = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        if len(encoded) > max_manifest:
            raise RuntimeError("manifest exceeded configured maximum size")
        manifest_path.write_bytes(encoded)

    persist_manifest()
    try:
        scenarios = [scenario_factory(seed) for seed in seeds]
        if [scenario.seed for scenario in scenarios] != seeds:
            raise RuntimeError("scenario factory returned unexpected seed identities")
        hashes = [scenario.layout_sha256 for scenario in scenarios]
        if len(set(hashes)) != len(hashes):
            raise RuntimeError("frozen-selection layouts are not geometry-hash unique")
        if set(hashes).intersection(prior_manifest["layout_sha256"]):
            raise RuntimeError("frozen-selection layout overlaps a parameter-screen layout")
        manifest["layout_sha256"] = hashes
        persist_manifest()

        with records_path.open("x", encoding="utf-8", newline="\n") as records:
            for scenario_index, scenario in enumerate(scenarios):
                rotated = cases[scenario_index % len(cases):] + cases[:scenario_index % len(cases)]
                for case_index, case in enumerate(rotated):
                    elapsed = monotonic() - started
                    if elapsed >= caps["wall_seconds"]:
                        manifest["status"] = "aborted_wall_cap_between_episodes"
                        persist_manifest()
                        return manifest
                    current = sum(item.stat().st_size for item in output_path.iterdir() if item.is_file())
                    if current + max_record + max_manifest > caps["artifact_bytes"]:
                        manifest["status"] = "aborted_artifact_cap_between_episodes"
                        persist_manifest()
                        return manifest
                    result = evaluator(scenario, case.method, parameters=dict(case.parameters))
                    record = {
                        "seed": scenario.seed,
                        "layout_sha256": scenario.layout_sha256,
                        "configuration_id": case.configuration_id,
                        "case_role": "frozen_configuration_selection",
                        "case_index_in_rotation": case_index,
                        "method": case.method,
                        "method_role": case.role,
                        "parameters": dict(case.parameters),
                        "result": asdict(result),
                    }
                    encoded = (json.dumps(record, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")
                    if len(encoded) > max_record:
                        raise RuntimeError("episode record exceeded configured maximum size")
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
            manifest["status"] = "completed_frozen_configuration_selection_only"
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
