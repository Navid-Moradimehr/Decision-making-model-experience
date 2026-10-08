"""Independent D051 reset, observation, and branch-equivalence re-audit."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "runs" / "phase6_acrobot_branch_smoke_v1"
CONFIG = ROOT / "configs" / "phase6_acrobot_branch_smoke.yaml"
RUNNER = ROOT / "scripts" / "smoke_phase6_acrobot_branch.py"
HELPER = ROOT / "src" / "consequence_lab" / "acrobot.py"
AUDIT = RUN / "independent_audit_v1.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _state_from_public_obs(observation):
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise AssertionError("reference received invalid public observation")
    return np.array(
        [math.atan2(values[1], values[0]), math.atan2(values[3], values[2]), values[4], values[5]],
        dtype=np.float64,
    )


def _compare_step(expected, actual, tolerance):
    obs_a, reward_a, term_a, trunc_a, _ = expected
    obs_b, reward_b, term_b, trunc_b, _ = actual
    if reward_a != reward_b or term_a != term_b or trunc_a != trunc_b:
        raise AssertionError("independent one-step branch mismatch in reward/end flags")
    error = float(np.max(np.abs(np.asarray(obs_a) - np.asarray(obs_b))))
    if error > tolerance:
        raise AssertionError(f"independent branch observation error {error} > {tolerance}")
    return error


def run():
    if AUDIT.exists():
        raise FileExistsError(f"refusing to overwrite {AUDIT}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    lock = json.loads((RUN / "selection_lock.json").read_text(encoding="utf-8"))
    env_cfg = config["environment"]
    seeds = list(range(env_cfg["smoke_seed_start"], env_cfg["smoke_seed_start"] + env_cfg["smoke_seed_count"]))
    expected_sources = {
        "config": _sha(CONFIG.read_bytes()),
        "runner": _sha(RUNNER.read_bytes()),
        "state_helper": _sha(HELPER.read_bytes()),
    }
    if lock["seeds"] != seeds or lock["environment_reset_at_lock"] is not False:
        raise AssertionError("seed lock mismatch or written after reset")
    if lock["source_hashes"] != expected_sources or summary["source_hashes"] != expected_sources:
        raise AssertionError("locked source hashes mismatch")
    if lock["performance_screen_or_training_started_at_lock"] is not False:
        raise AssertionError("lock indicates unauthorized work began")
    if importlib.metadata.version("gymnasium") != env_cfg["version"]:
        raise AssertionError("Gymnasium version mismatch")
    if importlib.metadata.version("numpy") != env_cfg["numpy_version"]:
        raise AssertionError("NumPy version mismatch")
    if summary["pypi_wheel_sha256_verified"] is not True:
        raise AssertionError("frozen run did not verify the PyPI wheel SHA-256")
    for flag in (
        "training_or_baseline_screen",
        "teacher_labels_generated",
        "student_training",
        "validation_or_test_accessed",
        "action_traces_or_state_saved",
    ):
        if summary[flag] is not False:
            raise AssertionError(f"unauthorized D051 scope flagged: {flag}")

    from gymnasium.envs.classic_control.acrobot import AcrobotEnv

    tolerance = float(env_cfg["absolute_tolerance"])
    actions_checked = transitions_checked = states_checked = 0
    max_state_error = max_branch_error = 0.0
    started = time.perf_counter()
    for seed in seeds:
        actual = AcrobotEnv()
        replica = AcrobotEnv()
        observation, _ = actual.reset(seed=seed)
        replica_obs, _ = replica.reset(seed=seed)
        if not np.array_equal(observation, replica_obs):
            raise AssertionError(f"same-seed reset mismatch: {seed}")
        random = np.random.default_rng(seed ^ 0xA6C0B07)
        for _ in range(env_cfg["random_roll_in_steps"]):
            state = _state_from_public_obs(observation)
            error = float(np.max(np.abs(state - np.asarray(actual.state))))
            if error > tolerance:
                raise AssertionError(f"independent state inversion error on seed {seed}: {error}")
            max_state_error = max(max_state_error, error)
            states_checked += 1
            for action in range(env_cfg["branch_action_count"]):
                source = copy.deepcopy(actual)
                candidate = copy.deepcopy(actual)
                candidate.state = state.copy()
                mismatch = _compare_step(source.step(action), candidate.step(action), tolerance)
                max_branch_error = max(max_branch_error, mismatch)
                actions_checked += 1
            action = int(random.integers(0, env_cfg["branch_action_count"]))
            out_a, out_b = actual.step(action), replica.step(action)
            _compare_step(out_a, out_b, 0.0)
            observation = out_a[0]
            transitions_checked += 1
            if out_a[2] or out_a[3]:
                break
    elapsed = time.perf_counter() - started

    expected_branches = summary["branch_checks"]
    if (actions_checked != expected_branches
            or states_checked != summary["observation_state_checks"]
            or transitions_checked != summary["replay_transitions"]):
        raise AssertionError("independent replay counts differ from smoke summary")
    if not math.isclose(max_state_error, summary["max_internal_state_reconstruction_error"], abs_tol=1e-12):
        raise AssertionError("state reconstruction diagnostic mismatch")
    if not math.isclose(max_branch_error, summary["max_one_step_observation_branch_error"], abs_tol=1e-12):
        raise AssertionError("branch error diagnostic mismatch")
    result = {
        "audit": "phase6_acrobot_observation_branch_independent_replay",
        "status": "PASS",
        "seeds_replayed": len(seeds),
        "roll_in_transitions_replayed": transitions_checked,
        "public_observation_state_checks": states_checked,
        "all_three_action_branches_replayed": actions_checked,
        "pypi_wheel_sha256_verified_by_frozen_smoke": True,
        "source_hashes_match_lock": True,
        "max_state_reconstruction_error": max_state_error,
        "max_branch_observation_error": max_branch_error,
        "training_or_performance_screen": False,
        "audit_wall_seconds": elapsed,
    }
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
