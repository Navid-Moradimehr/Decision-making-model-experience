"""D051 CPU-only Acrobot observation/state branch-equivalence smoke."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import numpy as np
import yaml

from consequence_lab.acrobot import acrobot_state_from_observation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase6_acrobot_branch_smoke.yaml"
HELPER = ROOT / "src" / "consequence_lab" / "acrobot.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _close_step(left, right, atol: float) -> float:
    obs_l, reward_l, term_l, trunc_l, _ = left
    obs_r, reward_r, term_r, trunc_r, _ = right
    if term_l != term_r or trunc_l != trunc_r or reward_l != reward_r:
        raise AssertionError("reconstructed observation changed branch reward/end flags")
    difference = float(np.max(np.abs(np.asarray(obs_l) - np.asarray(obs_r))))
    if difference > atol:
        raise AssertionError(f"branch observation error {difference} exceeds {atol}")
    return difference


def _verify_pypi_wheel(version: str, expected_sha256: str) -> bool:
    with urlopen(f"https://pypi.org/pypi/gymnasium/{version}/json", timeout=20) as response:
        release = json.loads(response.read())
    wheels = [
        item for item in release["urls"]
        if item["filename"] == f"gymnasium-{version}-py3-none-any.whl"
    ]
    if len(wheels) != 1 or wheels[0]["digests"]["sha256"] != expected_sha256:
        raise RuntimeError("PyPI release metadata does not match frozen Gymnasium wheel hash")
    with urlopen(wheels[0]["url"], timeout=60) as response:
        wheel = response.read()
    if _sha(wheel) != expected_sha256:
        raise RuntimeError("downloaded Gymnasium wheel does not match frozen SHA-256")
    return True


def run() -> dict:
    config_bytes = CONFIG.read_bytes()
    config = yaml.safe_load(config_bytes)
    project, env_cfg = config["project"], config["environment"]
    if project["authorization"] != "D051_acrobot_branch_equivalence_smoke_only":
        raise RuntimeError("D051 preflight authorization required")
    if project["baseline_screen_authorized"] or project["teacher_label_generation_authorized"] or project["student_training_authorized"] or project["validation_or_test_access_authorized"]:
        raise RuntimeError("D051 authorizes only branch-equivalence smoke")
    if tuple(map(int, env_cfg["python_version"].split("."))) != sys.version_info[:3]:
        raise RuntimeError(f"Python {env_cfg['python_version']} required")
    if importlib.metadata.version("gymnasium") != env_cfg["version"]:
        raise RuntimeError(f"Gymnasium {env_cfg['version']} required")
    if importlib.metadata.version("numpy") != env_cfg["numpy_version"]:
        raise RuntimeError(f"NumPy {env_cfg['numpy_version']} required")
    wheel_hash_verified = _verify_pypi_wheel(env_cfg["version"], env_cfg["wheel_sha256"])

    output = (ROOT / config["artifacts"]["run_directory"]).resolve()
    runs_root = (ROOT / "artifacts" / "runs").resolve()
    output.relative_to(runs_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    seeds = list(range(env_cfg["smoke_seed_start"], env_cfg["smoke_seed_start"] + env_cfg["smoke_seed_count"]))
    source_hashes = {
        "config": _sha(CONFIG.read_bytes()),
        "runner": _sha(Path(__file__).read_bytes()),
        "state_helper": _sha(HELPER.read_bytes()),
    }
    lock = {
        "protocol": project["protocol"],
        "authorization": project["authorization"],
        "source_revision": env_cfg["source_revision"],
        "seeds": seeds,
        "seed_role": "compatibility_smoke_only",
        "source_hashes": source_hashes,
        "environment_reset_at_lock": False,
        "performance_screen_or_training_started_at_lock": False,
    }
    (output / "selection_lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from gymnasium.envs.classic_control.acrobot import AcrobotEnv

    started = time.process_time()
    branch_checks = replay_transitions = states_checked = 0
    max_state_error = max_branch_error = 0.0
    for seed in seeds:
        live = AcrobotEnv()
        replay = AcrobotEnv()
        obs, _ = live.reset(seed=seed)
        replay_obs, _ = replay.reset(seed=seed)
        if not np.array_equal(obs, replay_obs):
            raise AssertionError(f"seeded reset is not reproducible: {seed}")
        rng = np.random.default_rng(seed ^ 0xA6C0B07)
        for _step in range(env_cfg["random_roll_in_steps"]):
            obs = np.asarray(obs)
            state = acrobot_state_from_observation(obs)
            state_error = float(np.max(np.abs(state - np.asarray(live.state))))
            max_state_error = max(max_state_error, state_error)
            if state_error > env_cfg["absolute_tolerance"]:
                raise AssertionError(f"observation state inversion failed on seed {seed}")
            states_checked += 1
            for action in range(env_cfg["branch_action_count"]):
                reference_branch = copy.deepcopy(live)
                observation_branch = copy.deepcopy(live)
                # The candidate-facing branch starts from a state reconstructed
                # only from the public observation. The private state is read
                # solely above for this equivalence assertion.
                observation_branch.state = state.copy()
                diff = _close_step(
                    reference_branch.step(action),
                    observation_branch.step(action),
                    env_cfg["absolute_tolerance"],
                )
                max_branch_error = max(max_branch_error, diff)
                branch_checks += 1

            action = int(rng.integers(0, env_cfg["branch_action_count"]))
            live_result = live.step(action)
            replay_result = replay.step(action)
            _close_step(live_result, replay_result, 0.0)
            obs = live_result[0]
            replay_transitions += 1
            if live_result[2] or live_result[3]:
                break

    total_cpu = time.process_time() - started
    if total_cpu > project["max_process_cpu_seconds"]:
        raise RuntimeError(f"smoke exceeded CPU limit: {total_cpu:.3f}s")
    summary = {
        "experiment": project["protocol"],
        "authorization": project["authorization"],
        "environment": env_cfg["task"],
        "gymnasium_version": importlib.metadata.version("gymnasium"),
        "expected_wheel_sha256": env_cfg["wheel_sha256"],
        "pypi_wheel_sha256_verified": wheel_hash_verified,
        "source_revision": env_cfg["source_revision"],
        "seed_range_half_open": [seeds[0], seeds[-1] + 1],
        "seeded_reset_replay_passed": True,
        "replay_transitions": replay_transitions,
        "observation_state_checks": states_checked,
        "all_three_actions_branched_per_state": True,
        "branch_checks": branch_checks,
        "max_internal_state_reconstruction_error": max_state_error,
        "max_one_step_observation_branch_error": max_branch_error,
        "private_state_used_only_for_equivalence_assertion": True,
        "training_or_baseline_screen": False,
        "teacher_labels_generated": False,
        "student_training": False,
        "validation_or_test_accessed": False,
        "action_traces_or_state_saved": False,
        "process_cpu_seconds": total_cpu,
        "source_hashes": source_hashes,
    }
    payload = json.dumps(summary, indent=2, sort_keys=True).encode("utf-8")
    if len(payload) > project["max_generated_artifact_bytes"]:
        raise RuntimeError("smoke summary exceeds artifact cap")
    (output / "summary.json").write_bytes(payload + b"\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
