"""Audit observable sufficiency for the pinned TwoRoom exact-simulator teacher.

This experiment creates labels solely for the diagnostic dataset. It does not
train a student. ``--upstream-root`` must point to a local checkout of the exact
stable-worldmodel revision recorded in the YAML protocol.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.candidates import primitive_actions, two_segment_sequences
from consequence_lab.environment import TwoRoomAdapter
from consequence_lab.objectives import goal_cost
from consequence_lab.realizability import (
    dependency_audit_report,
    neighborhood_consistency,
)


CONFIG_PATH = ROOT / "configs" / "phase3_tworoom_realizability.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _load_upstream_environment(upstream_root: Path):
    """Load only the two pinned source modules, bypassing unused package imports.

    This matches the project's earlier TwoRoom smoke procedure. The pinned
    ``spaces.py`` and ``env.py`` are loaded unchanged; the tiny ``utils.get_in``
    shim is the exact nested-index operation used by the pinned space classes.
    """
    import gymnasium  # noqa: F401

    package_dir = upstream_root / "stable_worldmodel"
    spaces_path = package_dir / "spaces.py"
    env_path = package_dir / "envs" / "two_room" / "env.py"
    if not spaces_path.is_file() or not env_path.is_file():
        raise FileNotFoundError(
            "upstream root must contain stable_worldmodel/spaces.py and "
            "stable_worldmodel/envs/two_room/env.py"
        )

    package = types.ModuleType("stable_worldmodel")
    package.__path__ = [str(package_dir)]
    package.__package__ = "stable_worldmodel"

    def get_in(mapping, path):
        current = mapping
        for key in list(path):
            current = current[key]
        return current

    package.utils = types.SimpleNamespace(get_in=get_in)
    sys.modules["stable_worldmodel"] = package

    spaces_spec = importlib.util.spec_from_file_location(
        "stable_worldmodel.spaces", spaces_path
    )
    if spaces_spec is None or spaces_spec.loader is None:
        raise ImportError(f"cannot load pinned spaces module: {spaces_path}")
    spaces_module = importlib.util.module_from_spec(spaces_spec)
    sys.modules[spaces_spec.name] = spaces_module
    spaces_spec.loader.exec_module(spaces_module)
    package.spaces = spaces_module

    env_spec = importlib.util.spec_from_file_location(
        "phase3_pinned_tworoom_env", env_path
    )
    if env_spec is None or env_spec.loader is None:
        raise ImportError(f"cannot load pinned TwoRoom module: {env_path}")
    env_module = importlib.util.module_from_spec(env_spec)
    sys.modules[env_spec.name] = env_module
    env_spec.loader.exec_module(env_module)
    env_module.TwoRoomEnv._phase3_source_hashes = {
        "stable_worldmodel/spaces.py": _sha256(spaces_path),
        "stable_worldmodel/envs/two_room/env.py": _sha256(env_path),
    }
    return env_module.TwoRoomEnv


def _cost(adapter: TwoRoomAdapter, snapshot, sequences, goal, config):
    positions = adapter.rollout_candidates(snapshot, sequences)
    objective = config["teacher"]["objective"]
    costs = goal_cost(
        positions,
        goal,
        arena_diagonal=objective["distance_scale_px"],
        terminal_distance_weight=objective["terminal_distance_weight"],
        mean_distance_weight=objective["mean_distance_weight"],
        success_distance=config["environment"]["success_distance_px"],
    )
    distances = np.linalg.norm(positions - goal[None, None, :], axis=-1)
    reached = (distances < config["environment"]["success_distance_px"]).any(axis=1)
    return costs.astype(np.float32), reached


def _geometry_signature(env) -> dict:
    values = env.variation_space.value
    return {
        "agent_speed": float(values["agent"]["speed"].item()),
        "agent_radius": float(values["agent"]["radius"].item()),
        "wall_axis": int(values["wall"]["axis"]),
        "wall_thickness": int(values["wall"]["thickness"]),
        "door_count": int(values["door"]["number"]),
        "door_centers": np.asarray(values["door"]["position"]).tolist(),
        "door_half_extents": np.asarray(values["door"]["size"]).tolist(),
        "wall_center": float(env.wall_pos),
        "boundary_px": int(env.BORDER_SIZE),
        "image_size_px": int(env.IMG_SIZE),
    }


def _reset_observable_state(env_type, observation, seed: int = 0, nuisance=False):
    env = env_type(render_mode="rgb_array")
    options = {
        "variation": [],
        "state": np.asarray(observation[:2], dtype=np.float32),
        "target_state": np.asarray(observation[2:4], dtype=np.float32),
    }
    if nuisance:
        options["variation_values"] = {
            "agent.color": np.asarray([0, 0, 255], dtype=np.uint8),
            "target.color": np.asarray([255, 255, 0], dtype=np.uint8),
            "background.color": np.asarray([0, 0, 0], dtype=np.uint8),
            "rendering.render_target": 1,
        }
    out, _ = env.reset(seed=seed, options=options)
    return env, np.asarray(out.detach().cpu().numpy(), dtype=np.float32)


def _alias_control(env_type, sequences, config):
    start = np.asarray([60.0, 60.0], dtype=np.float32)
    goals = [
        np.asarray([164.0, 48.0], dtype=np.float32),
        np.asarray([164.0, 176.0], dtype=np.float32),
    ]
    observations, costs, actions = [], [], []
    for goal in goals:
        env = env_type(render_mode="rgb_array")
        obs, _ = env.reset(
            seed=0,
            options={"variation": [], "state": start, "target_state": goal},
        )
        obs = np.asarray(obs.detach().cpu().numpy(), dtype=np.float32)
        adapter = TwoRoomAdapter(env)
        row_costs, _ = _cost(adapter, adapter.snapshot(), sequences, goal, config)
        observations.append(obs)
        costs.append(row_costs)
        actions.append(sequences[np.flatnonzero(row_costs <= row_costs.min() + 0.01), 0])
    masked_equal = bool(
        np.allclose(observations[0][:2], observations[1][:2], atol=1e-6)
        and np.allclose(observations[0][4:], observations[1][4:], atol=1e-6)
    )
    costs_differ = not np.allclose(costs[0], costs[1], atol=1e-6, rtol=0.0)
    first_sets_disjoint = not bool(
        set(map(tuple, np.round(actions[0], 6)))
        & set(map(tuple, np.round(actions[1], 6)))
    )
    return {
        "goal_masked_observations_equal": masked_equal,
        "candidate_costs_differ": costs_differ,
        "near_optimal_first_actions_disjoint": first_sets_disjoint,
        "detected": bool(masked_equal and (costs_differ or first_sets_disjoint)),
    }


def run(config: dict, env_type, run_dir: Path) -> dict:
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    env_cfg = config["environment"]
    data_cfg = config["data"]
    sequences = two_segment_sequences(env_cfg["candidate_horizon_steps"])
    if len(sequences) != env_cfg["candidate_count"]:
        raise AssertionError("candidate bank differs from preregistered count")

    start, end = data_cfg["development_episode_seeds"]
    behaviors = []
    for name, count in data_cfg["behavior_assignment"].items():
        behaviors.extend([name] * int(count))
    if len(behaviors) != end - start:
        raise AssertionError("behavior allocation must cover the seed range exactly")
    assignments = [(seed, behaviors[(seed - start) % len(behaviors)]) for seed in range(start, end)]

    rows_obs, rows_cost, rows_reach, rows_seed, rows_behavior, rows_step = [], [], [], [], [], []
    geometry_signatures = []
    progress_path = run_dir / "progress.jsonl"
    begin = time.perf_counter()
    budget_seconds = config["project"]["max_cpu_minutes"] * 60
    sample_steps = set(data_cfg["sample_step_indices"])

    with progress_path.open("w", encoding="utf-8") as progress:
        for episode_number, (seed, behavior) in enumerate(assignments, start=1):
            env = env_type(render_mode="rgb_array")
            adapter = TwoRoomAdapter(env, success_distance=env_cfg["success_distance_px"])
            observation, _ = adapter.reset(seed=seed)
            geometry_signatures.append(_geometry_signature(env))
            rng = np.random.default_rng(seed + 170000)
            captured_steps = []
            for step_index in range(env_cfg["max_episode_steps"]):
                snapshot = None
                costs = reachable = None
                if behavior == "simulator_truth_shooting" or step_index in sample_steps:
                    snapshot = adapter.snapshot()
                    costs, reachable = _cost(
                        adapter, snapshot, sequences, observation[2:4], config
                    )
                if step_index in sample_steps:
                    assert costs is not None and reachable is not None and snapshot is not None
                    rows_obs.append(observation.copy())
                    rows_cost.append(costs.copy())
                    rows_reach.append(reachable.copy())
                    rows_seed.append(seed)
                    rows_behavior.append(behavior)
                    rows_step.append(step_index)
                    captured_steps.append(step_index)

                if behavior == "simulator_truth_shooting":
                    action = sequences[int(np.argmin(costs)), 0]
                elif behavior == "greedy":
                    direction = observation[2:4] - observation[:2]
                    norm = float(np.linalg.norm(direction))
                    action = np.zeros(2, dtype=np.float32) if norm == 0 else direction / norm
                else:
                    action = primitive_actions()[int(rng.integers(0, 9))]
                observation, _, terminated, truncated, _ = adapter.step(action)
                if terminated or truncated:
                    break
            env.close()
            progress.write(
                json.dumps(
                    {"episode": episode_number, "seed": seed, "behavior": behavior,
                     "captured_steps": captured_steps},
                    sort_keys=True,
                ) + "\n"
            )
            progress.flush()
            if episode_number % 4 == 0 or episode_number == len(assignments):
                print(f"completed {episode_number}/{len(assignments)} episodes", flush=True)
            if time.perf_counter() - begin > budget_seconds:
                raise TimeoutError("Phase-3A CPU-time budget exceeded; run is incomplete")

    obs_array = np.asarray(rows_obs, dtype=np.float32)
    costs_array = np.asarray(rows_cost, dtype=np.float32)
    reach_array = np.asarray(rows_reach, dtype=bool)
    seed_array = np.asarray(rows_seed, dtype=np.int64)
    behavior_array = np.asarray(rows_behavior, dtype="U32")
    step_array = np.asarray(rows_step, dtype=np.int16)

    exact_mismatches = 0
    max_obs_error = 0.0
    max_cost_error = 0.0
    nuisance_mismatches = 0
    for index, observation in enumerate(obs_array):
        if time.perf_counter() - begin > budget_seconds:
            raise TimeoutError("Phase-3A CPU-time budget exceeded during reconstruction; run is incomplete")
        reconstructed_env, reconstructed_obs = _reset_observable_state(env_type, observation)
        obs_error = float(np.max(np.abs(reconstructed_obs - observation)))
        max_obs_error = max(max_obs_error, obs_error)
        if obs_error > config["realizability"]["observation_reconstruction_tolerance"]:
            exact_mismatches += 1
        reconstructed_adapter = TwoRoomAdapter(
            reconstructed_env, success_distance=env_cfg["success_distance_px"]
        )
        rec_costs, rec_reach = _cost(
            reconstructed_adapter,
            reconstructed_adapter.snapshot(),
            sequences,
            observation[2:4],
            config,
        )
        cost_error = float(np.max(np.abs(rec_costs - costs_array[index])))
        max_cost_error = max(max_cost_error, cost_error)
        if cost_error > config["realizability"]["exact_reconstruction_cost_tolerance"]:
            exact_mismatches += 1
        if not np.array_equal(rec_reach, reach_array[index]):
            exact_mismatches += 1
        reconstructed_env.close()

        if index < min(16, len(obs_array)):
            nuisance_env, nuisance_obs = _reset_observable_state(
                env_type, observation, nuisance=True
            )
            if not np.allclose(nuisance_obs, observation, atol=1e-6, rtol=0.0):
                nuisance_mismatches += 1
            nuisance_adapter = TwoRoomAdapter(
                nuisance_env, success_distance=env_cfg["success_distance_px"]
            )
            nuisance_costs, nuisance_reach = _cost(
                nuisance_adapter,
                nuisance_adapter.snapshot(),
                sequences,
                observation[2:4],
                config,
            )
            if not np.allclose(nuisance_costs, costs_array[index], atol=1e-6, rtol=0.0):
                nuisance_mismatches += 1
            if not np.array_equal(nuisance_reach, reach_array[index]):
                nuisance_mismatches += 1
            nuisance_env.close()

    alias = _alias_control(env_type, sequences, config)
    neighborhood = neighborhood_consistency(
        obs_array,
        costs_array,
        sequences,
        seed_array,
        config["realizability"]["nearest_neighbor_radii_px"],
        config["realizability"]["near_optimal_regret_epsilon"],
    )
    fixed_geometry = all(signature == geometry_signatures[0] for signature in geometry_signatures)
    expected_geometries = config["environment"].get("fixed_geometry_values", {})
    geometry_matches = all(
        geometry_signatures
        and all(signature.get(key) == value for key, value in expected_geometries.items())
        for signature in geometry_signatures
    )

    data_path = run_dir / "teacher_diagnostic.npz"
    np.savez_compressed(
        data_path,
        observations=obs_array,
        candidate_costs=costs_array,
        candidate_reachability=reach_array,
        episode_seeds=seed_array,
        behavior=behavior_array,
        sampled_step_indices=step_array,
        candidate_sequences=sequences,
    )

    source_paths = [
        CONFIG_PATH,
        Path(__file__).resolve(),
        ROOT / "src/consequence_lab/environment.py",
        ROOT / "src/consequence_lab/objectives.py",
        ROOT / "src/consequence_lab/candidates.py",
        ROOT / "src/consequence_lab/realizability.py",
        ROOT / "experiments/source_pins.json",
    ]
    sources = {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths}
    sources.update(getattr(env_type, "_phase3_source_hashes", {}))
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None

    gates = config["realizability"]["hard_gates"]
    dependency_audit = dependency_audit_report()
    gate_results = {
        "fixed_geometry_matches_declared": bool(fixed_geometry and geometry_matches),
        "unexplained_transition_or_objective_dependencies": dependency_audit[
            "unexplained_dependency_count"
        ],
        "reconstruction_mismatches": int(exact_mismatches),
        "missed_intentional_alias_controls": 0 if alias["detected"] else 1,
        "nuisance_control_cost_mismatches": int(nuisance_mismatches),
    }
    passed = all(
        gate_results[key] <= int(value)
        for key, value in gates.items()
        if key in gate_results
    ) and gate_results["fixed_geometry_matches_declared"]

    summary = {
        "experiment": "Phase 3A teacher realizability diagnostic",
        "status": "completed",
        "interpretation": "Finite-sample information audit only; no student training or irreducible-error claim.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment_revision": config["environment"]["revision"],
        "device": "cpu",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "episode_count": len(assignments),
        "state_count": int(len(obs_array)),
        "states_by_behavior": {
            behavior: int(np.sum(behavior_array == behavior))
            for behavior in config["data"]["behavior_assignment"]
        },
        "fixed_geometry_signatures": list(
            {
                json.dumps(signature, sort_keys=True): signature
                for signature in geometry_signatures
            }.values()
        ),
        "dependency_audit": dependency_audit,
        "fixed_geometry_matches_declared": bool(fixed_geometry and geometry_matches),
        "reconstruction": {
            "observation_max_abs_error": max_obs_error,
            "candidate_cost_max_abs_error": max_cost_error,
            "mismatch_count": exact_mismatches,
        },
        "nuisance_control_mismatch_count": nuisance_mismatches,
        "intentional_goal_mask_alias_control": alias,
        "cross_episode_neighborhoods": neighborhood,
        "gates": gate_results,
        "go_no_go": "go" if passed else "no_go_diagnose_before_student_training",
        "wall_seconds": round(time.perf_counter() - begin, 3),
        "data_sha256": _sha256(data_path),
        "data_bytes": data_path.stat().st_size,
        "source_sha256": sources,
        "git": {"commit": commit, "dirty": dirty, "tracked_and_untracked_status": _git("status", "--porcelain") if commit else None},
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "run_id": config["artifacts"]["run_id"],
        "phase": "phase_3a",
        "status": "completed",
        "created_utc": summary["created_utc"],
        "git": summary["git"],
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": _sha256(CONFIG_PATH),
        "environment": {
            "id": env_cfg["id"],
            "revision": env_cfg["revision"],
            "episode_seed_start": start,
            "episode_count": len(assignments),
        },
        "method": "observable_state_reconstruction_and_deliberate_goal_mask_alias_controls",
        "seed": start,
        "summary_path": str(summary_path.relative_to(ROOT)),
        "resources": {
            "device": "cpu",
            "wall_seconds": summary["wall_seconds"],
            "peak_vram_bytes": 0,
            "artifact_bytes": int(data_path.stat().st_size + summary_path.stat().st_size),
        },
        "sources_sha256": sources,
        "notes": [
            "Only pinned upstream spaces.py and TwoRoom env.py were loaded; no student was trained.",
            "Episode and teacher-only fields exist only in the diagnostic artifact and are blocked by the runtime allowlist.",
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    env_type = _load_upstream_environment(args.upstream_root.resolve())
    run_dir = ROOT / config["artifacts"]["output_directory"]
    run(config, env_type, run_dir)


if __name__ == "__main__":
    main()
