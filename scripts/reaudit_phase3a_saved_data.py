"""Recheck a completed Phase-3A dataset without collecting or overwriting data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from consequence_lab.environment import TwoRoomAdapter
from consequence_lab.realizability import dependency_audit_report, neighborhood_consistency
from run_phase3a_tworoom_realizability import (
    CONFIG_PATH,
    _alias_control,
    _cost,
    _geometry_signature,
    _load_upstream_environment,
    _reset_observable_state,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def reaudit(config: dict, env_type, source_run: Path, output_run: Path) -> dict:
    if output_run.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {output_run}")
    source_run = source_run.resolve()
    source_data = source_run / "teacher_diagnostic.npz"
    source_summary = source_run / "summary.json"
    if not source_data.is_file() or not source_summary.is_file():
        raise FileNotFoundError("source run must contain teacher_diagnostic.npz and summary.json")

    output_run.mkdir(parents=True)
    begin = time.perf_counter()
    budget = config["project"]["max_cpu_minutes"] * 60
    previous = json.loads(source_summary.read_text(encoding="utf-8"))
    with np.load(source_data, allow_pickle=False) as saved:
        observations = saved["observations"]
        costs = saved["candidate_costs"]
        reachability = saved["candidate_reachability"]
        seeds = saved["episode_seeds"]
        behavior = saved["behavior"]
        candidate_sequences = saved["candidate_sequences"]

    if observations.ndim != 2 or observations.shape[1] != 10:
        raise ValueError("saved observations violate the audited runtime shape")
    if costs.shape != (len(observations), len(candidate_sequences)):
        raise ValueError("saved cost matrix does not align with observations/candidates")
    if len(seeds) != len(observations) or len(behavior) != len(observations):
        raise ValueError("saved row provenance does not align with observations")

    env_cfg = config["environment"]
    reconstruction_mismatches = 0
    max_observation_error = 0.0
    max_cost_error = 0.0
    nuisance_mismatches = 0
    for index, observation in enumerate(observations):
        if time.perf_counter() - begin > budget:
            raise TimeoutError("Phase-3A re-audit exceeded its CPU budget")
        env, reconstructed_observation = _reset_observable_state(env_type, observation)
        observation_error = float(np.max(np.abs(reconstructed_observation - observation)))
        max_observation_error = max(max_observation_error, observation_error)
        if observation_error > config["realizability"]["observation_reconstruction_tolerance"]:
            reconstruction_mismatches += 1
        adapter = TwoRoomAdapter(env, success_distance=env_cfg["success_distance_px"])
        reconstructed_costs, reconstructed_reachability = _cost(
            adapter, adapter.snapshot(), candidate_sequences, observation[2:4], config
        )
        cost_error = float(np.max(np.abs(reconstructed_costs - costs[index])))
        max_cost_error = max(max_cost_error, cost_error)
        if cost_error > config["realizability"]["exact_reconstruction_cost_tolerance"]:
            reconstruction_mismatches += 1
        if not np.array_equal(reconstructed_reachability, reachability[index]):
            reconstruction_mismatches += 1
        env.close()

        if index < min(16, len(observations)):
            nuisance_env, nuisance_observation = _reset_observable_state(
                env_type, observation, nuisance=True
            )
            if not np.allclose(nuisance_observation, observation, atol=1e-6, rtol=0.0):
                nuisance_mismatches += 1
            nuisance_adapter = TwoRoomAdapter(
                nuisance_env, success_distance=env_cfg["success_distance_px"]
            )
            nuisance_costs, nuisance_reachability = _cost(
                nuisance_adapter,
                nuisance_adapter.snapshot(),
                candidate_sequences,
                observation[2:4],
                config,
            )
            if not np.allclose(nuisance_costs, costs[index], atol=1e-6, rtol=0.0):
                nuisance_mismatches += 1
            if not np.array_equal(nuisance_reachability, reachability[index]):
                nuisance_mismatches += 1
            nuisance_env.close()

    alias = _alias_control(env_type, candidate_sequences, config)
    neighborhood = neighborhood_consistency(
        observations,
        costs,
        candidate_sequences,
        seeds,
        config["realizability"]["nearest_neighbor_radii_px"],
        config["realizability"]["near_optimal_regret_epsilon"],
    )
    dependency_audit = dependency_audit_report()
    geometry_signatures = previous["fixed_geometry_signatures"]
    geometry_signatures = {
        json.dumps(row, sort_keys=True): row for row in geometry_signatures
    }.values()
    geometry_signatures = list(geometry_signatures)
    expected_geometry = config["environment"]["fixed_geometry_values"]
    geometry_ok = bool(geometry_signatures) and all(
        all(signature.get(key) == value for key, value in expected_geometry.items())
        for signature in geometry_signatures
    )

    gate_results = {
        "fixed_geometry_matches_declared": geometry_ok,
        "unexplained_transition_or_objective_dependencies": dependency_audit[
            "unexplained_dependency_count"
        ],
        "reconstruction_mismatches": reconstruction_mismatches,
        "missed_intentional_alias_controls": 0 if alias["detected"] else 1,
        "nuisance_control_cost_mismatches": nuisance_mismatches,
    }
    gates = config["realizability"]["hard_gates"]
    passed = geometry_ok and all(
        gate_results[key] <= int(value) for key, value in gates.items()
    )

    output_data = output_run / "teacher_diagnostic.npz"
    shutil.copy2(source_data, output_data)
    source_paths = [
        CONFIG_PATH,
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("run_phase3a_tworoom_realizability.py"),
        ROOT / "src/consequence_lab/environment.py",
        ROOT / "src/consequence_lab/objectives.py",
        ROOT / "src/consequence_lab/candidates.py",
        ROOT / "src/consequence_lab/realizability.py",
        ROOT / "experiments/source_pins.json",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    source_hashes.update(getattr(env_type, "_phase3_source_hashes", {}))
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    summary = {
        "experiment": "Phase 3A saved-data realizability re-audit",
        "status": "completed_reaudit",
        "interpretation": "Rechecks the original development dataset; no trajectories were recollected and no model was trained.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(source_run.relative_to(ROOT)),
        "source_data_sha256": sha256(source_data),
        "episode_count": int(len(np.unique(seeds))),
        "state_count": int(len(observations)),
        "states_by_behavior": {
            str(name): int(np.sum(behavior == name))
            for name in np.unique(behavior)
        },
        "fixed_geometry_signatures": geometry_signatures,
        "fixed_geometry_matches_declared": geometry_ok,
        "dependency_audit": dependency_audit,
        "reconstruction": {
            "observation_max_abs_error": max_observation_error,
            "candidate_cost_max_abs_error": max_cost_error,
            "mismatch_count": reconstruction_mismatches,
        },
        "nuisance_control_mismatch_count": nuisance_mismatches,
        "intentional_goal_mask_alias_control": alias,
        "cross_episode_neighborhoods": neighborhood,
        "gates": gate_results,
        "go_no_go": "go" if passed else "no_go_diagnose_before_student_training",
        "wall_seconds": round(time.perf_counter() - begin, 3),
        "data_sha256": sha256(output_data),
        "data_bytes": output_data.stat().st_size,
        "source_sha256": source_hashes,
        "git": {"commit": commit, "dirty": bool(status), "tracked_and_untracked_status": status},
    }
    summary_path = output_run / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_id": output_run.name,
        "phase": "phase_3a_reaudit",
        "status": "completed",
        "created_utc": summary["created_utc"],
        "git": summary["git"],
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG_PATH),
        "environment": {
            "id": env_cfg["id"],
            "revision": env_cfg["revision"],
            "source_seed_range": [int(seeds.min()), int(seeds.max()) + 1],
            "source_episode_count": summary["episode_count"],
        },
        "method": "read_only_reconstruction_and_dependency_audit_on_saved_phase3a_data",
        "summary_path": str(summary_path.relative_to(ROOT)),
        "resources": {
            "device": "cpu",
            "wall_seconds": summary["wall_seconds"],
            "peak_vram_bytes": 0,
            "artifact_bytes": int(output_data.stat().st_size + summary_path.stat().st_size),
        },
        "sources_sha256": source_hashes,
        "source_run_manifest_sha256": sha256(source_run / "manifest.json"),
    }
    (output_run / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument(
        "--source-run",
        type=Path,
        default=ROOT / "artifacts/runs/phase3a_tworoom_realizability_240000_240063",
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=ROOT / "artifacts/runs/phase3a_tworoom_realizability_reaudit_240000_240063",
    )
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    env_type = _load_upstream_environment(args.upstream_root.resolve())
    reaudit(config, env_type, args.source_run, args.output_run)


if __name__ == "__main__":
    main()
