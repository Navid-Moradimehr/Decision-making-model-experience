"""Source-pinned contract and physics-copy smoke for dm_control Finger.

Run inside the isolated DMC environment:
    python scripts/smoke_finger_contract.py --config configs/phase9_finger_contract_smoke.yaml

This is a feasibility check only. It does not compare controllers, score task
performance, or save episode observations/physics states.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dm_control import suite
import mujoco


def _flatten_observation(observation: dict[str, Any]) -> np.ndarray:
    return np.concatenate([np.asarray(value, dtype=np.float64).reshape(-1) for value in observation.values()])


def _step_physics(env: Any, physics: Any, action: np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    """Apply one source-equivalent control step to a physics branch."""
    env._task.before_step(action, physics)
    physics.step(env._n_sub_steps)
    env._task.after_step(physics)
    observation = env._task.get_observation(physics)
    reward = float(env._task.get_reward(physics))
    return {key: np.asarray(value).copy() for key, value in observation.items()}, reward


def _assert_obs_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray], atol: float) -> float:
    assert tuple(left) == tuple(right)
    max_error = 0.0
    for key in left:
        assert left[key].shape == right[key].shape, key
        error = float(np.max(np.abs(left[key] - right[key]))) if left[key].size else 0.0
        max_error = max(max_error, error)
        np.testing.assert_allclose(left[key], right[key], rtol=0.0, atol=atol, err_msg=key)
    return max_error


def _one_seed(config: dict[str, Any], seed: int) -> float:
    env = suite.load("finger", "turn_hard", task_kwargs={"random": seed})
    replay = suite.load("finger", "turn_hard", task_kwargs={"random": seed})
    try:
        timestep = env.reset()
        replay_timestep = replay.reset()
        expected = config["checks"]["observation_shapes"]
        assert {key: list(np.asarray(value).shape) for key, value in timestep.observation.items()} == expected
        _assert_obs_equal(
            {key: np.asarray(value) for key, value in timestep.observation.items()},
            {key: np.asarray(value) for key, value in replay_timestep.observation.items()},
            atol=0.0,
        )

        action_spec = env.action_spec()
        assert list(action_spec.shape) == config["checks"]["action_shape"]
        np.testing.assert_array_equal(action_spec.minimum, config["checks"]["action_bounds"][0])
        np.testing.assert_array_equal(action_spec.maximum, config["checks"]["action_bounds"][1])

        physics = env.physics
        obs = timestep.observation
        state = physics.get_state()
        assert state.ndim == 1 and np.isfinite(state).all()
        assert physics.data.qpos.shape == (3,)
        assert physics.data.qvel.shape == (3,)
        assert physics.data.act.size == 0
        # The observation exposes the two arm joint positions and all velocities.
        np.testing.assert_allclose(obs["position"][:2], physics.data.qpos[:2], atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(obs["velocity"], physics.data.qvel, atol=1e-12, rtol=0.0)
        # `tip` is a site fixed 0.13 m from the spinner hinge. Its observed
        # vector therefore encodes hinge angle modulo 2*pi; the disk is
        # rotationally symmetric, so winding count is not task-relevant.
        tip = physics.tip_position()
        spinner_angle = float(physics.data.qpos[2])
        tip_offset = physics.named.model.site_pos["tip"]
        tip_radius = float(np.linalg.norm(tip_offset[[0, 2]]))
        expected_tip = tip_radius * np.asarray([-np.sin(spinner_angle), np.cos(spinner_angle)])
        np.testing.assert_allclose(tip, expected_tip, atol=1e-12, rtol=0.0)
        assert np.isfinite(physics.target_position()).all()
        assert np.isfinite(_flatten_observation(obs)).all()

        # Move to a non-reset state, copy full physics/model state, then compare
        # identical future actions on the original and copied physics branches.
        for raw_action in config["prefix_actions"]:
            action = np.asarray(raw_action, dtype=action_spec.dtype)
            _step_physics(env, physics, action)
        branch = physics.copy()
        max_error = 0.0
        for raw_action in config["branch_actions"]:
            action = np.asarray(raw_action, dtype=action_spec.dtype)
            obs_a, reward_a = _step_physics(env, physics, action)
            obs_b, reward_b = _step_physics(env, branch, action)
            max_error = max(max_error, _assert_obs_equal(obs_a, obs_b, config["checks"]["branch_absolute_tolerance"]))
            assert reward_a == reward_b
            assert np.isfinite(_flatten_observation(obs_a)).all()
        return max_error
    finally:
        env.close()
        replay.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert config["scope"] == "feasibility_only_no_performance_evaluation"
    for package, expected in config["runtime"].items():
        actual = platform.python_version() if package == "python" else importlib.metadata.version(package.replace("_", "-"))
        assert actual == expected, f"{package}: expected {expected}, got {actual}"
    assert mujoco.__version__ == config["runtime"]["mujoco"]

    errors = [_one_seed(config, int(seed)) for seed in config["seeds"]]
    result = {
        "status": "PASS",
        "scope": config["scope"],
        "environment": config["source"],
        "seeds_checked": len(errors),
        "branch_comparisons": len(errors) * len(config["branch_actions"]),
        "max_observation_absolute_error": max(errors, default=0.0),
        "versions": {
            "python": platform.python_version(),
            "dm_control": importlib.metadata.version("dm-control"),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "artifacts": "aggregate smoke result only; no observations or physics states",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
