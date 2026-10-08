"""Hash audits and gated execution for Phase 13A only.

Layout identity audits read no task transitions or outcomes. The run stage is
deliberately limited to Phase 13A; later labels/training require a different
runner and the gates in the frozen protocol.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

import yaml

from consequence_lab.corridor_evaluation import (
    CorridorEpisodeResult,
    evaluate_corridor_episode,
    paired_outcome_rate_interval,
    summarize_corridor_results,
)
from consequence_lab.corridor_tasks import generate_staggered_corridor


class Phase13ProtocolError(RuntimeError):
    """Raised when a Phase 13A execution prerequisite is not met."""


ROOT_RELATIVE = (
    "configs/phase13.yaml",
    "reports/phase13_counterfactual_action_value_protocol.md",
    "src/consequence_lab/phase13.py",
    "scripts/audit_phase13_layouts.py",
    "scripts/freeze_phase13a_authorization.py",
    "scripts/run_phase13a.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_evaluation.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "tests/test_phase13.py",
    "tests/test_corridor_baselines.py",
    "tests/test_corridor_evaluation.py",
)
BLOCKS = {
    "phase13A_development": (143000, 40),
    "phase13A_independent": (190000, 100),
    "phase13E_model_selection": (191000, 40),
    "phase13E_offline_gate": (191040, 40),
    "phase13F_closed_loop_development": (192000, 100),
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
PHASE13_AUDIT_NAME = "phase13_layout_hash_audit_v2.json"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest().upper()


def _layout_hashes(value: Any, key_context: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key).lower()
            if "layout" in name or "geometry" in name:
                found.update(_hash_strings(child))
            else:
                found.update(_layout_hashes(child, name))
    elif isinstance(value, list):
        for child in value:
            found.update(_layout_hashes(child, key_context))
    return found


def _hash_strings(value: Any) -> set[str]:
    if isinstance(value, str):
        candidate = value.lower()
        return {candidate} if HASH_RE.fullmatch(candidate) else set()
    if isinstance(value, list):
        return set().union(*(_hash_strings(item) for item in value)) if value else set()
    if isinstance(value, dict):
        return set().union(*(_hash_strings(item) for item in value.values())) if value else set()
    return set()


def _read_layout_hashes(path: Path) -> set[str]:
    try:
        if path.suffix.lower() == ".jsonl":
            values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix.lower() == ".json":
            values = [json.loads(path.read_text(encoding="utf-8"))]
        else:
            return set()
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase13ProtocolError(f"unreadable historical layout artifact {path}: {exc}") from exc
    return set().union(*(_layout_hashes(value) for value in values)) if values else set()


def _is_outcome_artifact(path: Path, value: Any) -> bool:
    name = path.name.lower()
    if "audit" in name or "replay" in name or "provenance" in name:
        return False
    if name in {"episodes.json", "episodes.jsonl", "results.json", "evaluation.json", "bank_records.jsonl"}:
        return True
    if name == "manifest.json" and isinstance(value, dict):
        status = str(value.get("status", ""))
        return (
            status.startswith("completed_")
            and (value.get("episode_count", 0) or value.get("decision_query_count", 0) or value.get("completed_episode_records", 0))
        )
    return False


def _collect_historical_layouts(root: Path) -> tuple[set[str], set[str], list[str]]:
    all_hashes: set[str] = set()
    spent_hashes: set[str] = set()
    sources: list[str] = []
    run_root = root / "artifacts" / "runs"
    if not run_root.is_dir():
        raise Phase13ProtocolError("artifacts/runs is unavailable for the required history audit")
    for path in sorted(run_root.rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith("phase13_layout_hash_audit_v")
        ):
            continue
        if path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        hashes = _read_layout_hashes(path)
        if not hashes:
            continue
        all_hashes.update(hashes)
        sources.append(path.relative_to(root).as_posix())
        if path.name.lower() == "manifest.json":
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            if _is_outcome_artifact(path, metadata):
                spent_hashes.update(hashes)
        elif _is_outcome_artifact(path, {}):
            spent_hashes.update(hashes)
    return all_hashes, spent_hashes, sources


def build_layout_audit(root: Path) -> dict[str, Any]:
    """Generate candidate geometry hashes only; never execute a transition."""
    root = Path(root).resolve()
    prior_all, prior_spent, sources = _collect_historical_layouts(root)
    blocks: dict[str, list[dict[str, Any]]] = {}
    block_sets: dict[str, set[str]] = {}
    for name, (start, count) in BLOCKS.items():
        records = []
        hashes = set()
        for seed in range(start, start + count):
            scenario = generate_staggered_corridor(seed)
            records.append({"seed": seed, "layout_sha256": scenario.layout_sha256})
            hashes.add(scenario.layout_sha256)
        if len(hashes) != count:
            raise Phase13ProtocolError(f"{name} contains repeated geometry layouts")
        block_sets[name] = hashes
        blocks[name] = records

    # The 143k block was hash-audited previously but had no outcomes. It may be
    # reused only if none of its geometry appears in outcome-bearing artifacts.
    development_spent = block_sets["phase13A_development"] & prior_spent
    if development_spent:
        raise Phase13ProtocolError(
            f"Phase 13A development overlaps {len(development_spent)} prior outcome layouts"
        )
    for name in ("phase13A_independent", "phase13E_model_selection", "phase13E_offline_gate", "phase13F_closed_loop_development"):
        overlap = block_sets[name] & prior_all
        if overlap:
            raise Phase13ProtocolError(f"{name} overlaps {len(overlap)} previously audited layouts")
    names = list(block_sets)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            overlap = block_sets[name] & block_sets[other]
            if overlap:
                raise Phase13ProtocolError(f"new layout blocks {name} and {other} overlap")

    return {
        "schema": "phase13_layout_hash_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "hash_only_passed_no_outcomes",
        "task_transitions_evaluated": False,
        "episode_outcomes_read": False,
        "reserve_performance_outcomes_read": False,
        "previous_layout_hash_count": len(prior_all),
        "previous_outcome_layout_hash_count": len(prior_spent),
        "historical_hash_source_files": sources,
        "blocks": blocks,
        "all_new_blocks_pairwise_disjoint": True,
        "phase13A_development_reused_only_from_hash_audit": True,
    }


def validate_authorization(
    root: Path, lock_path: Path, audit_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(root).resolve()
    lock_path = Path(lock_path).resolve()
    audit_path = Path(audit_path).resolve()
    try:
        config_for_paths = yaml.safe_load((root / "configs/phase13.yaml").read_text(encoding="utf-8"))
        expected_lock = root / config_for_paths["phase13A"]["provenance"]["authorization_lock"]
        expected_audit = root / config_for_paths["phase13A"]["provenance"]["layout_audit"]
    except (OSError, KeyError, yaml.YAMLError) as exc:
        raise Phase13ProtocolError(f"invalid Phase 13 provenance paths: {exc}") from exc
    if lock_path != expected_lock.resolve() or audit_path != expected_audit.resolve():
        raise Phase13ProtocolError("Phase 13A lock/audit path differs from the frozen protocol")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        config = yaml.safe_load((root / "configs/phase13.yaml").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise Phase13ProtocolError(f"could not read source lock/config/audit: {exc}") from exc
    if lock.get("authorization_status") != "authorized_with_preregistered_gates":
        raise Phase13ProtocolError("authorization lock does not carry the approved scope")
    if lock.get("schema") != "phase13a_authorization_lock_v2":
        raise Phase13ProtocolError("authorization lock schema is not the v2 audited lock")
    if any(lock.get(field) is not False for field in (
        "reserve_access_authorized", "student_training_authorized",
        "dagger_authorized", "world_model_or_jepa_authorized",
    )):
        raise Phase13ProtocolError("reserve, training, DAgger, and world-model work must remain unauthorized")
    if lock.get("phase13a_execution_authorized") is not True:
        raise Phase13ProtocolError("Phase 13A outcomes are not authorized")
    if lock.get("config_sha256") != file_sha256(root / "configs/phase13.yaml"):
        raise Phase13ProtocolError("Phase 13 config hash mismatch")
    if lock.get("layout_audit_path") != config_for_paths["phase13A"]["provenance"]["layout_audit"]:
        raise Phase13ProtocolError("authorization lock points to a different layout audit")
    if lock.get("layout_audit_sha256") != file_sha256(audit_path):
        raise Phase13ProtocolError("layout audit hash mismatch")
    if not isinstance(lock.get("approval_reference"), str) or not lock["approval_reference"].strip():
        raise Phase13ProtocolError("user authorization reference is missing")
    if audit.get("status") != "hash_only_passed_no_outcomes" or audit.get("task_transitions_evaluated") is not False:
        raise Phase13ProtocolError("layout audit is not a valid hash-only pass")
    if audit.get("generation_git_commit") != lock.get("base_git_commit"):
        raise Phase13ProtocolError("layout audit and source lock do not share a source commit")
    audit_sources = {
        relative: file_sha256(root / relative)
        for relative in (
            "configs/phase13.yaml",
            "scripts/audit_phase13_layouts.py",
            "src/consequence_lab/phase13.py",
            "src/consequence_lab/corridor_tasks.py",
            "src/consequence_lab/inertial_corridor.py",
        )
    }
    if audit.get("generation_source_sha256") != audit_sources:
        raise Phase13ProtocolError("layout audit was generated from different source/config")
    if audit.get("blocks") is None or not audit.get("all_new_blocks_pairwise_disjoint"):
        raise Phase13ProtocolError("layout audit is incomplete")
    for name, (start, count) in BLOCKS.items():
        records = audit["blocks"].get(name, [])
        if (
            len(records) != count
            or [record.get("seed") for record in records] != list(range(start, start + count))
            or len({record.get("layout_sha256") for record in records}) != count
        ):
            raise Phase13ProtocolError(f"layout audit block is incomplete or reordered: {name}")
    expected_stages = {
        "phase13A_development": {"start": 143000, "count": 40},
        "phase13A_independent_conditional": {"start": 190000, "count": 100},
    }
    if lock.get("authorized_stages") != expected_stages:
        raise Phase13ProtocolError("authorization seed scope differs from the Phase 13A protocol")
    expected_caps = {
        "wall_seconds": config["integrity"]["cpu_wall_cap_seconds_per_stage"],
        "generated_bytes": config["integrity"]["generated_artifact_cap_bytes_per_stage"],
        "device": "cpu",
    }
    if lock.get("caps_per_stage") != expected_caps:
        raise Phase13ProtocolError("resource caps differ from the frozen protocol")
    if lock.get("source_sha256") != {
        relative: file_sha256(root / relative) for relative in ROOT_RELATIVE
    }:
        raise Phase13ProtocolError("one or more locked source hashes differ")
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase13ProtocolError("could not verify Git commit") from exc
    if head != lock.get("base_git_commit"):
        try:
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", lock["base_git_commit"], "HEAD"],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Phase13ProtocolError("locked source commit is not an ancestor of current HEAD") from exc
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase13ProtocolError("could not verify clean Git state") from exc
    if dirty:
        raise Phase13ProtocolError("working tree must be clean after committing the source lock")
    return lock, config, audit


def _within_limits(started: float, root: Path, max_seconds: int, max_bytes: int, output: Path) -> None:
    if time.monotonic() - started > max_seconds:
        raise Phase13ProtocolError("Phase 13A wall-time cap reached; stage stopped")
    current_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if current_bytes > max_bytes:
        raise Phase13ProtocolError("Phase 13A output cap reached; stage stopped")


def _result_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = CorridorEpisodeResult.__dataclass_fields__
    summaries = summarize_corridor_results(
        [CorridorEpisodeResult(**{key: row[key] for key in fields}) for row in rows]
    )
    return {item.method: asdict(item) for item in summaries}


def run_phase13a(
    root: Path,
    *,
    stage: str,
    output_path: Path,
    lock_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_path = Path(output_path).resolve()
    if stage not in {"development", "independent"}:
        raise Phase13ProtocolError("unknown Phase 13A stage")
    if output_path.exists():
        raise Phase13ProtocolError("refusing to overwrite an existing Phase 13A output")
    lock, config, audit = validate_authorization(root, lock_path, audit_path)
    expected_output = root / config["phase13A"]["outputs"][stage]
    if output_path != expected_output.resolve():
        raise Phase13ProtocolError("output path differs from the frozen Phase 13A config")
    if stage == "independent":
        dev_manifest = root / "artifacts/runs/phase13a_development_v1/manifest.json"
        if not dev_manifest.is_file():
            raise Phase13ProtocolError("independent comparison requires completed development selection")
        dev = json.loads(dev_manifest.read_text(encoding="utf-8"))
        if dev.get("status") != "completed_development_selection_only" or dev.get("reserve_accessed") is not False:
            raise Phase13ProtocolError("development output is not a valid completed tuning stage")
        margin = dev.get("selected_brake_safety_margin")
        if margin not in config["phase13A"]["methods"]["phase_space"]["brake_safety_margin_candidates"]:
            raise Phase13ProtocolError("development-selected braking margin is not preregistered")
        if dev.get("config_sha256") != lock["config_sha256"] or dev.get("source_lock_sha256") != file_sha256(lock_path):
            raise Phase13ProtocolError("development result is not bound to the current frozen lock")
        expected = {
            "config_sha256": lock["config_sha256"],
            "source_lock_sha256": file_sha256(lock_path),
            "selected_margin": margin,
        }
        method_parameters = {
            "phase_space": {
                "resolution": 0.04,
                "corner_angle_degrees": 30.0,
                "brake_safety_margin": margin,
            },
            "geometry_waypoint": {"resolution": 0.04, "waypoint_tolerance": 0.03},
            "random_shooting": {"horizon": 8, "num_sequences": 128},
        }
        seeds = range(190000, 190100)
        methods = ("random_shooting", "geometry_waypoint", "phase_space")
    else:
        expected = {
            "config_sha256": lock["config_sha256"],
            "source_lock_sha256": file_sha256(lock_path),
        }
        method_parameters = {}
        seeds = range(143000, 143040)
        methods = ("phase_space",)

    audit_by_seed = {
        name: {int(record["seed"]): record["layout_sha256"] for record in records}
        for name, records in audit["blocks"].items()
    }
    block_name = "phase13A_development" if stage == "development" else "phase13A_independent"
    expected_layouts = audit_by_seed[block_name]
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise Phase13ProtocolError("output parent is not a directory")
    output_path.mkdir(parents=True)
    episodes_path = output_path / "episodes.jsonl"
    episodes_path.touch(exist_ok=False)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    cap_seconds = config["integrity"]["cpu_wall_cap_seconds_per_stage"]
    cap_bytes = config["integrity"]["generated_artifact_cap_bytes_per_stage"]

    def evaluate_seed(seed: int, method: str, parameters: dict[str, Any], config_id: str) -> None:
        scenario = generate_staggered_corridor(seed)
        if scenario.layout_sha256 != expected_layouts[seed]:
            raise Phase13ProtocolError(f"layout hash changed after audit at seed {seed}")
        result = evaluate_corridor_episode(scenario, method, parameters=parameters)
        row = {
            **asdict(result),
            "configuration_id": config_id,
            "parameters": parameters,
        }
        rows.append(row)
        with episodes_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
        _within_limits(started, root, cap_seconds, cap_bytes, output_path)

    if stage == "development":
        for margin in config["phase13A"]["methods"]["phase_space"]["brake_safety_margin_candidates"]:
            for seed in seeds:
                phase_parameters = {
                    "resolution": 0.04,
                    "corner_angle_degrees": 30.0,
                    "brake_safety_margin": margin,
                }
                evaluate_seed(
                    seed, "phase_space", phase_parameters,
                    f"phase_space_margin_{margin:.2f}",
                )
                method_parameters["phase_space"] = {
                    "resolution": 0.04,
                    "corner_angle_degrees": 30.0,
                    "brake_safety_margin": margin,
                }
    else:
        for seed in seeds:
            # Rotate method order across layouts using the frozen common runner
            # convention; pair all methods on the exact same generated layout.
            rotation = (seed - 190000) % len(methods)
            ordered = methods[rotation:] + methods[:rotation]
            for method in ordered:
                evaluate_seed(seed, method, method_parameters[method], method)

    summaries = {}
    gate: dict[str, Any] | None = None
    selected_margin = None
    if stage == "development":
        scores = []
        for margin in config["phase13A"]["methods"]["phase_space"]["brake_safety_margin_candidates"]:
            group = [row for row in rows if row["configuration_id"] == f"phase_space_margin_{margin:.2f}"]
            summary = _result_summary(group)["phase_space"]
            scores.append((summary, margin))
        scores.sort(key=lambda item: (
            -item[0]["safe_success_rate"],
            item[0]["collision_rate"],
            item[0]["median_steps_on_success"] if item[0]["median_steps_on_success"] is not None else float("inf"),
            item[0]["median_wall_seconds_per_decision"] if item[0]["median_wall_seconds_per_decision"] is not None else float("inf"),
            item[1],
        ))
        selected_margin = scores[0][1]
        summaries = {f"margin_{margin:.2f}": summary for summary, margin in scores}
        status = "completed_development_selection_only"
    else:
        summaries = _result_summary(rows)
        shooting_gain = paired_outcome_rate_interval(
            [CorridorEpisodeResult(**{key: row[key] for key in CorridorEpisodeResult.__dataclass_fields__}) for row in rows],
            "random_shooting", "phase_space", event="success",
            bootstrap_seed=190011, resamples=10000,
        )
        collision_difference = paired_outcome_rate_interval(
            [CorridorEpisodeResult(**{key: row[key] for key in CorridorEpisodeResult.__dataclass_fields__}) for row in rows],
            "random_shooting", "phase_space", event="collision",
            bootstrap_seed=190011, resamples=10000,
        )
        planner_latency = summaries["random_shooting"]["median_wall_seconds_per_decision"]
        phase_latency = summaries["phase_space"]["median_wall_seconds_per_decision"]
        latency_ratio = planner_latency / phase_latency if phase_latency else float("inf")
        rules = config["phase13A"]["headroom_gate"]
        gate = {
            "success_gain": asdict(shooting_gain),
            "collision_difference": asdict(collision_difference),
            "latency_ratio": latency_ratio,
            "passed": (
                shooting_gain.paired_rate_difference >= rules["planner_minus_phase_space_success_point_gain_at_least"]
                and shooting_gain.lower_95 > rules["paired_layout_bootstrap_95_lower_bound_above"]
                and collision_difference.upper_95 <= rules["planner_minus_phase_space_collision_rate_95_upper_bound_at_most"]
                and latency_ratio >= rules["planner_to_phase_space_median_decision_latency_ratio_at_least"]
            ),
        }
        selected_margin = expected["selected_margin"]
        status = "completed_independent_headroom_comparison"

    manifest = {
        "experiment": f"phase13A_{stage}",
        "status": status,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        **expected,
        "base_git_commit": lock["base_git_commit"],
        "authorization_reference": lock["approval_reference"],
        "source_sha256": lock["source_sha256"],
        "layout_audit_sha256": lock["layout_audit_sha256"],
        "stage": stage,
        "seed_start": 143000 if stage == "development" else 190000,
        "seed_count": 40 if stage == "development" else 100,
        "episode_records": len(rows),
        "episode_artifact": "episodes.jsonl",
        "summaries": summaries,
        "selected_brake_safety_margin": selected_margin,
        "headroom_gate": gate,
        "reserve_accessed": False,
        "student_training_performed": False,
        "dagger_performed": False,
        "world_model_used": False,
        "elapsed_wall_seconds": time.monotonic() - started,
        "artifact_bytes_excluding_manifest": episodes_path.stat().st_size,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pyyaml": yaml.__version__,
        },
    }
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    actual_bytes = sum(path.stat().st_size for path in output_path.rglob("*") if path.is_file())
    if actual_bytes > cap_bytes or manifest["elapsed_wall_seconds"] > cap_seconds:
        raise Phase13ProtocolError("Phase 13A cap exceeded during finalization; output retained and disclosed")
    manifest["artifact_bytes_total"] = actual_bytes
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
