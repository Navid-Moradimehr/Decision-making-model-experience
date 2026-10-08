"""Minimal, observation-only Gymnasium adapter for DMC Finger Turn Hard.

Optional runtime dependencies: dm-control==1.0.47 and gymnasium==1.2.3.
This adapter exposes only the documented task observation, flattening fields in
the explicit order configured by the research protocol.
"""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np
from dm_control import suite
from gymnasium import spaces

from consequence_lab.objectives import potential_shaped_reward


OBSERVATION_FIELDS = ("position", "velocity", "touch", "target_position", "dist_to_target")
OBSERVATION_SIZE = 12


def flatten_finger_observation(observation: Mapping[str, np.ndarray]) -> np.ndarray:
    if tuple(observation) != OBSERVATION_FIELDS:
        raise ValueError(f"Unexpected DMC observation fields/order: {tuple(observation)}")
    flat = np.concatenate([np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in OBSERVATION_FIELDS])
    if flat.shape != (OBSERVATION_SIZE,) or not np.isfinite(flat).all():
        raise ValueError(f"Invalid flattened observation shape/values: {flat.shape}")
    return flat


class FingerTurnHardGymnasium(gym.Env[np.ndarray, np.ndarray]):
    """Expose the 12 public DMC state features and two normalized controls."""

    metadata = {"render_modes": []}

    def __init__(self, *, shaping_gamma: float | None = None) -> None:
        super().__init__()
        if shaping_gamma is not None and (not np.isfinite(shaping_gamma) or not 0.0 <= shaping_gamma <= 1.0):
            raise ValueError("shaping_gamma must be None or a finite value in [0, 1]")
        self._shaping_gamma = shaping_gamma
        self._previous_distance: float | None = None
        self._env = suite.load("finger", "turn_hard")
        action_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=np.asarray(action_spec.minimum, dtype=np.float32),
            high=np.asarray(action_spec.maximum, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        del options
        super().reset(seed=seed)
        if seed is not None:
            # DMC Task owns a RandomState; reseed it before its reset samples.
            self._env.task.random.seed(int(seed))
        timestep = self._env.reset()
        self._previous_distance = float(timestep.observation["dist_to_target"])
        return flatten_finger_observation(timestep.observation), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Invalid action shape/values: {action.shape}")
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            raise ValueError("Action outside declared [-1, 1] bounds")
        timestep = self._env.step(action.astype(self._env.action_spec().dtype, copy=False))
        observation = flatten_finger_observation(timestep.observation)
        raw_reward = float(timestep.reward or 0.0)
        if self._shaping_gamma is None:
            reward = raw_reward
        else:
            if self._previous_distance is None:
                raise RuntimeError("reset() must be called before shaped step()")
            next_distance = float(timestep.observation["dist_to_target"])
            reward = potential_shaped_reward(
                raw_reward,
                -self._previous_distance,
                -next_distance,
                self._shaping_gamma,
            )
            self._previous_distance = next_distance
        is_last = bool(timestep.last())
        terminated = is_last and timestep.discount == 0.0
        truncated = is_last and not terminated
        return observation, reward, terminated, truncated, {}

    def close(self) -> None:
        self._env.close()
