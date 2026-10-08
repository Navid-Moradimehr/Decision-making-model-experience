"""Tiny CPU physics/render smoke for the pinned Reacher phase.

Run from the isolated environment:
    MUJOCO_GL=egl python scripts/smoke_reacher_dm_control.py

This tests DeepMind Control Suite directly, not the stable-worldmodel wrapper
or the LeWM checkpoint. Passing is only dependency feasibility evidence.
"""

from __future__ import annotations

import numpy as np
from dm_control import suite
import mujoco


def rollout(seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    env = suite.load("reacher", "hard", task_kwargs={"random": seed})
    try:
        timestep = env.reset()
        first = {key: value.copy() for key, value in timestep.observation.items()}
        action_spec = env.action_spec()
        action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        timestep = env.step(action)
        second = {key: value.copy() for key, value in timestep.observation.items()}
        frame = env.physics.render(height=224, width=224, camera_id=0)
        return first, second, frame
    finally:
        env.close()


def main() -> None:
    first_a, second_a, frame_a = rollout(61001)
    first_b, second_b, frame_b = rollout(61001)
    assert first_a.keys() == first_b.keys()
    assert second_a.keys() == second_b.keys()
    assert all(np.array_equal(first_a[k], first_b[k]) for k in first_a)
    assert all(np.array_equal(second_a[k], second_b[k]) for k in second_a)
    assert frame_a.shape == (224, 224, 3) and frame_a.dtype == np.uint8
    assert np.array_equal(frame_a, frame_b)

    print(f"MuJoCo: {mujoco.__version__}")
    print(f"Observation: { {key: value.shape for key, value in first_a.items()} }")
    print(f"Action: shape={rollout_action_shape()}, bounds=[-1, 1]")
    print(f"Render: shape={frame_a.shape}, dtype={frame_a.dtype}")
    print("Deterministic seeded reset/step/render: PASS")
    print("Scope: direct dm_control smoke only; stable-worldmodel wrapper and LeWM untested")


def rollout_action_shape() -> tuple[int, ...]:
    env = suite.load("reacher", "hard", task_kwargs={"random": 61001})
    try:
        return env.action_spec().shape
    finally:
        env.close()


if __name__ == "__main__":
    main()
