"""Check exact seeded Reacher branch replay required by Phase 2B."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types

import gymnasium as gym
import numpy as np


def _namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def main() -> None:
    source_root = Path(os.environ["STABLE_WORLD_MODEL_SOURCE"]).resolve()
    sys.path.insert(0, str(source_root))
    dmcontrol_path = source_root / "stable_worldmodel" / "envs" / "dmcontrol"
    _namespace("stable_worldmodel.envs.dmcontrol", dmcontrol_path)
    import stable_worldmodel  # noqa: F401 - register pinned environment IDs

    env = gym.make("swm/ReacherDMControl-v0", task="qpos_match")
    seed = 221000
    rng = np.random.default_rng(seed + 1)
    sequences = rng.uniform(-1.0, 1.0, size=(3, 25, 2)).astype(np.float32)
    try:
        _, initial_info = env.reset(seed=seed)
        start_qpos = np.asarray(initial_info["qpos"]).copy()
        start_qvel = np.asarray(initial_info["qvel"]).copy()
        unreachable_target = start_qpos + np.asarray([100.0, 100.0])

        results = []
        for sequence in sequences:
            branch_results = []
            for _ in range(2):
                env.reset(seed=seed)
                env.unwrapped.set_state(start_qpos, start_qvel)
                env.unwrapped.set_target_qpos(unreachable_target)
                terminated = False
                steps = 0
                for action in sequence:
                    _, _, terminated, truncated, info = env.step(action)
                    steps += 1
                    if terminated or truncated:
                        break
                frame = env.unwrapped.render(width=224, height=224)
                branch_results.append(
                    (
                        np.asarray(info["qpos"]).copy(),
                        np.asarray(info["qvel"]).copy(),
                        frame.copy(),
                        steps,
                        bool(terminated),
                        bool(truncated),
                    )
                )
            a, b = branch_results
            assert np.array_equal(a[0], b[0]), "qpos branch replay diverged"
            assert np.array_equal(a[1], b[1]), "qvel branch replay diverged"
            assert np.array_equal(a[2], b[2]), "rendered branch replay diverged"
            assert a[3:] == b[3:], "termination branch replay diverged"
            results.append(a)

        assert all(not row[4] and not row[5] for row in results)
        print(f"Pinned Reacher deterministic branch replay: PASS ({len(sequences)} sequences x 25 actions x 2 replays)")
        print("Compared qpos, qvel, rendered frame, step count, termination/truncation")
        print("No model ranking or task-performance claim")
    finally:
        env.close()


if __name__ == "__main__":
    main()
