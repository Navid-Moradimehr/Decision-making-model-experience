"""Thin adapter for the pinned TwoRoom Gymnasium environment contract."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy-compatible values and CPU/GPU tensors to owned arrays."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


class TwoRoomAdapter:
    """Normalize observations and make exact candidate rollouts from snapshots.

    ``env`` may be a Gymnasium wrapper or the underlying TwoRoom environment.
    The adapter deliberately imports neither Gymnasium nor stable-worldmodel;
    callers install and construct the optional upstream environment separately.
    """

    def __init__(self, env: Any, success_distance: float = 16.0):
        self.env = getattr(env, "unwrapped", env)
        if not callable(getattr(self.env, "reset", None)) or not callable(
            getattr(self.env, "step", None)
        ):
            raise TypeError("env must expose reset() and step()")
        if not np.isfinite(success_distance) or success_distance < 0:
            raise ValueError("success_distance must be non-negative and finite")
        self.success_distance = float(success_distance)

    def reset(self, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and return a NumPy observation and copied info mapping."""
        observation, info = self.env.reset(seed=seed)
        if not isinstance(info, dict):
            raise TypeError("TwoRoom reset info must be a mapping")
        return _as_numpy(observation).astype(np.float32, copy=False), copy.deepcopy(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one legal normalized velocity action with NumPy outputs."""
        velocity = np.asarray(action, dtype=np.float32)
        if velocity.shape != (2,) or not np.isfinite(velocity).all():
            raise ValueError("action must be a finite two-vector")
        if np.any(np.abs(velocity) > 1.0):
            raise ValueError("action components must lie in [-1, 1]")
        observation, reward, terminated, truncated, info = self.env.step(velocity)
        if not isinstance(info, dict):
            raise TypeError("TwoRoom step info must be a mapping")
        return (
            _as_numpy(observation).astype(np.float32, copy=False),
            float(reward),
            bool(terminated),
            bool(truncated),
            copy.deepcopy(info),
        )

    def snapshot(self) -> Any:
        """Copy the complete underlying simulator state for fair branching."""
        if not hasattr(self.env, "agent_position"):
            raise TypeError("TwoRoom snapshot requires an agent_position state")
        return _TwoRoomSnapshot(
            environment=copy.deepcopy(self.env),
            agent_position=copy.deepcopy(self.env.agent_position),
        )

    @staticmethod
    def rollout_candidates(snapshot: Any, sequences: np.ndarray) -> np.ndarray:
        """Return true simulator positions after every candidate action.

        Each sequence starts from the same saved agent position on a private
        copy of the static simulator. This fast path is valid for the pinned
        TwoRoom transition, whose only mutable step state is agent_position.
        If an episode terminates early, later positions repeat the terminal
        position, matching the absorbing-success convention in the objective.
        Output shape is ``[candidate, time, 2]`` in environment pixel units.
        """
        actions = np.asarray(sequences, dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] == 0 or actions.shape[1] == 0 or actions.shape[2] != 2:
            raise ValueError("sequences must have shape [candidate, time, 2]")
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.0):
            raise ValueError("candidate actions must be finite and within [-1, 1]")

        if not isinstance(snapshot, _TwoRoomSnapshot):
            raise TypeError("snapshot must come from TwoRoomAdapter.snapshot()")
        positions = np.empty(actions.shape, dtype=np.float32)
        # The pinned TwoRoom transition (stable-worldmodel 63988116) mutates
        # only agent_position. Keep one branch copy and reset that tensor before
        # each candidate; copying static 224x224 geometry per candidate is slow.
        branch = copy.deepcopy(snapshot.environment)
        for candidate_index, sequence in enumerate(actions):
            branch.agent_position = copy.deepcopy(snapshot.agent_position)
            terminated = False
            truncated = False
            last_position = _as_numpy(branch.agent_position).astype(np.float32, copy=False)
            for time_index, action in enumerate(sequence):
                if not (terminated or truncated):
                    observation, _, terminated, truncated, _ = branch.step(action)
                    last_position = _as_numpy(observation)[:2].astype(np.float32, copy=False)
                positions[candidate_index, time_index] = last_position
        return positions


@dataclass(frozen=True)
class _TwoRoomSnapshot:
    environment: Any
    agent_position: Any
