"""Deterministic task-layout generator for Phase 11 qualification design.

Generating scenarios is not a performance evaluation. The layouts are a
single, explicitly named topology family intended for later baseline screening.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from numbers import Integral

from consequence_lab.inertial_corridor import (
    AxisAlignedObstacle,
    CorridorState,
    CorridorTaskSpec,
    validate_corridor_state,
)


@dataclass(frozen=True)
class CorridorScenario:
    seed: int
    layout_family: str
    layout_sha256: str
    spec: CorridorTaskSpec
    initial_state: CorridorState


def generate_staggered_corridor(seed: int) -> CorridorScenario:
    """Generate one public three-baffle, two-opening layout and initial state.

    Each baffle spans the workspace except for two offset openings. Opening
    centers, widths, start/goal heights, and initial velocity are seeded. The
    fixed x locations and topology are intentionally disclosed; validation
    uses layout-disjoint seeds within this family, not hidden map parameters.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = random.Random(seed)
    obstacles = []
    for x_center in (0.26, 0.50, 0.74):
        thickness = rng.uniform(0.028, 0.040)
        x_min, x_max = x_center - thickness / 2.0, x_center + thickness / 2.0
        low_center = rng.uniform(0.30, 0.40)
        high_center = rng.uniform(0.60, 0.70)
        low_width = rng.uniform(0.14, 0.19)
        high_width = rng.uniform(0.14, 0.19)
        openings = (
            (low_center - low_width / 2.0, low_center + low_width / 2.0),
            (high_center - high_width / 2.0, high_center + high_width / 2.0),
        )
        lower, upper = 0.0, 1.0
        for gap_lo, gap_hi in openings:
            if lower < gap_lo:
                obstacles.append(AxisAlignedObstacle(x_min, lower, x_max, gap_lo))
            lower = gap_hi
        if lower < upper:
            obstacles.append(AxisAlignedObstacle(x_min, lower, x_max, upper))

    spec = CorridorTaskSpec(
        width=1.0,
        height=1.0,
        obstacles=tuple(obstacles),
        goal=(0.93, rng.uniform(0.18, 0.82)),
        goal_radius=0.04,
        safe_speed=0.035,
        agent_radius=0.018,
        acceleration=0.024,
        damping=0.97,
        max_speed=0.12,
        dt=1.0,
        max_steps=160,
    )
    initial_state = CorridorState(
        x=0.07,
        y=rng.uniform(0.18, 0.82),
        vx=rng.uniform(-0.035, 0.035),
        vy=rng.uniform(-0.035, 0.035),
    )
    validate_corridor_state(initial_state, spec)
    geometry = [
        [format(obstacle.x_min, ".17g"), format(obstacle.y_min, ".17g"),
         format(obstacle.x_max, ".17g"), format(obstacle.y_max, ".17g")]
        for obstacle in spec.obstacles
    ]
    layout_sha256 = hashlib.sha256(
        json.dumps(geometry, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return CorridorScenario(
        seed=seed,
        layout_family="three_baffle_two_openings",
        layout_sha256=layout_sha256,
        spec=spec,
        initial_state=initial_state,
    )
