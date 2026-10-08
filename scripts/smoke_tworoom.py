"""Exercise the pinned TwoRoom environment without downloading a dataset."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers the swm/ Gymnasium environments

from consequence_lab.environment import TwoRoomAdapter


def main() -> None:
    env = gym.make("swm/TwoRoom-v1")
    adapter = TwoRoomAdapter(env)
    observation, info = adapter.reset(seed=601)
    snapshot = adapter.snapshot()
    candidates = np.asarray(
        [
            [[0.70710677, 0.70710677], [0.70710677, 0.70710677]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    outcomes = adapter.rollout_candidates(snapshot, candidates)

    adapter.reset(seed=601)
    repeated = adapter.rollout_candidates(adapter.snapshot(), candidates)
    env.close()

    result = {
        "environment": "swm/TwoRoom-v1",
        "stable_worldmodel_version": _package_version("stable-worldmodel"),
        "observation_shape": list(observation.shape),
        "observation_dtype": str(observation.dtype),
        "info_keys": sorted(info),
        "candidate_rollout_shape": list(outcomes.shape),
        "same_seed_replay_exact": bool(np.array_equal(outcomes, repeated)),
        "candidate_branches_differ": bool(not np.array_equal(outcomes[0], outcomes[1])),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if observation.shape != (10,):
        raise AssertionError(f"expected the pinned 10-value observation, got {observation.shape}")
    if outcomes.shape != (2, 2, 2):
        raise AssertionError(f"unexpected candidate rollout shape: {outcomes.shape}")
    if not result["same_seed_replay_exact"] or not result["candidate_branches_differ"]:
        raise AssertionError("deterministic branch smoke failed")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
