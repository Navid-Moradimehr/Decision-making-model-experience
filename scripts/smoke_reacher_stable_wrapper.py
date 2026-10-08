"""Compatibility smoke for the pinned stable-worldmodel Reacher wrapper.

Run in the isolated simulator environment with the pinned source checkout on
PYTHONPATH. This tests the qpos_match task/render/action-block interface only;
it does not load LeWM or evaluate planner performance.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import numpy as np
import gymnasium as gym
import mujoco

# The pinned package's dmcontrol/__init__.py eagerly imports an optional
# expert-policy class, which imports Torch even for simulator-only use. Match
# the project's earlier TwoRoom smoke: retain the exact package search path
# while skipping only that unrelated namespace initializer.
source_root = Path(os.environ["STABLE_WORLD_MODEL_SOURCE"]).resolve()
sys.path.insert(0, str(source_root))
dmcontrol_path = source_root / "stable_worldmodel" / "envs" / "dmcontrol"
dmcontrol_namespace = types.ModuleType("stable_worldmodel.envs.dmcontrol")
dmcontrol_namespace.__path__ = [str(dmcontrol_path)]
sys.modules.setdefault("stable_worldmodel.envs.dmcontrol", dmcontrol_namespace)

import stable_worldmodel  # noqa: F401 - registers swm environment IDs


ENV_ID = "swm/ReacherDMControl-v0"
SEED = 61002


def main() -> None:
    env = gym.make(ENV_ID, task="qpos_match")
    try:
        obs_a, info_a = env.reset(seed=SEED)
        frame_a = env.unwrapped.render(width=224, height=224)

        # The official evaluator supplies target configurations separately
        # from the current configuration. First use a far target to observe
        # both wrapper repeats without triggering early success.
        qpos = np.asarray(info_a["qpos"][:2]).copy()
        far_goal = qpos + np.asarray([1.0, 1.0])
        env.unwrapped.set_target_qpos(far_goal)
        start_time = env.unwrapped.dmc_env.physics.data.time
        obs_next, reward, terminated, truncated, info_next = env.step(
            np.zeros(2, dtype=env.action_space.dtype)
        )
        elapsed = env.unwrapped.dmc_env.physics.data.time - start_time
        control_timestep = env.unwrapped.dmc_env.control_timestep()
        expected_elapsed = env.unwrapped.action_repeat * control_timestep
        assert obs_a.shape == env.observation_space.shape
        assert env.action_space.shape == (2,)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)
        assert np.isclose(elapsed, expected_elapsed)
        assert not terminated
        assert not truncated

        # Matching the current qpos must trigger qpos_match early termination
        # within the next action block (normally after its first repeat).
        env.unwrapped.set_target_qpos(np.asarray(info_next["qpos"]).copy())
        _, _, goal_terminated, _, _ = env.step(
            np.zeros(2, dtype=env.action_space.dtype)
        )
        assert goal_terminated

        # The official planner's goal is a separately rendered target state.
        env.unwrapped.set_state(far_goal, np.zeros_like(info_next["qvel"]))
        goal_frame = env.unwrapped.render(width=224, height=224)
        assert frame_a.shape == (224, 224, 3) and frame_a.dtype == np.uint8
        assert goal_frame.shape == (224, 224, 3)
        assert not np.array_equal(frame_a, goal_frame)
        assert obs_next.shape == obs_a.shape
        assert np.isfinite(reward)
        assert info_next["qpos"].shape == (2,)

        obs_b, info_b = env.reset(seed=SEED)
        frame_b = env.unwrapped.render(width=224, height=224)
        assert np.array_equal(obs_a, obs_b)
        assert np.array_equal(info_a["qpos"], info_b["qpos"])
        assert np.array_equal(frame_a, frame_b)

        print(f"Environment: {ENV_ID}, task=qpos_match")
        print(f"Observation: {obs_a.shape}; action: {env.action_space.shape}")
        print(
            "Action repeat: "
            f"{env.unwrapped.action_repeat} x "
            f"{control_timestep}s control timestep "
            f"= {elapsed:.5f}s"
        )
        print(f"qpos-match near-target early termination: {goal_terminated}")
        print(f"Current/goal images: {frame_a.shape}/{goal_frame.shape}")
        print(f"MuJoCo: {mujoco.__version__}")
        print("Pinned wrapper seeded reset/step/render: PASS")
        print("LeWM action-block/context alignment: NOT TESTED")
    finally:
        env.close()


if __name__ == "__main__":
    main()
