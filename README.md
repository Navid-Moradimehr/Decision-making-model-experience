# Decision-Making Model Experience

An engineering project exploring how to make repeated, exact-planning decisions more efficient in a deterministic inertial-navigation task.

The work builds an auditable simulator and receding-horizon shooting planner, then evaluates planning budgets, plan reuse, and learning-assisted search with exact rollout scoring. The strongest practical gains came from reusing the previous winning plan as a warm start and selecting search budgets against explicit reliability and compute criteria.

## What the project builds

- **Inertial Corridor simulator:** nine discrete physical controls, velocity-dependent motion, staggered obstacle layouts, and explicit collision and goal conditions.
- **Exact receding-horizon planner:** samples finite action sequences, evaluates every candidate in the simulator, and executes only the first action before replanning.
- **Temporal warm start:** shifts the previous winning sequence forward and includes it in the next exact search; it proposes a candidate but does not bypass exact evaluation.
- **Budget and reliability evaluation:** paired layout comparisons, rollout-transition and latency accounting, and explicit rare-event checks.
- **Learning-assisted planning experiments:** controlled implementations for policy, consequence, proposal, and compute-allocation studies.

## Measured engineering results

| Evaluation | Exact-search method | Outcome | Compute observation |
|---|---|---:|---:|
| Phase 11, fresh development layouts | Exact shooting vs. geometry waypoint | 100/100 vs. 64/100 successes | Exact shooting established planning headroom on the paired block. |
| Phase 16E, fresh development layouts | Uniform B64 vs. B128 | 99.53% vs. 99.67% success | B64 used 57.27% of B128 rollout transitions and 50.87% of its median decision latency; B64 passed the frozen reliability gate. |
| Phase 21, fresh development layouts | Shifted B32 vs. uniform B32 | 99.53% vs. 98.13% success | Paired success advantage: 1.40 percentage points. Shifted B32 also passed the frozen B48-relative reliability and compute checks. |
| Phase 25, separate fresh block | Shifted B24 vs. shifted B32 | 99.33% vs. 99.53% success | B24 used 17.89% fewer transitions and 24.93% lower median latency, passing its separately preregistered development gate. |

The Phase 16E, 21, and 25 measurements come from separate evaluation blocks; the reported paired comparisons are only within each phase. These are simulator development results, not a physical-system certification.

## Engineering takeaways

1. **Reuse the plan where successive decisions overlap.** Shifting a previously successful sequence gave the next exact search a useful candidate at negligible proposal cost.
2. **Choose budgets from reliability and cost together.** Average success alone can hide layout-level failure tails, so operating points were checked with paired outcomes and rare-event bounds.
3. **Keep verification authoritative.** Candidate generation can be improved independently from candidate evaluation: exact rollout scoring and final planner selection remained in place.
4. **Measure the whole control loop.** Candidate availability, prediction error, or lower scalar cost did not by themselves imply better episode outcomes.

## Getting started

Requires Python 3.11 or newer and uv.

    uv sync --extra test
    uv run pytest tests/test_inertial_corridor.py tests/test_corridor_tasks.py tests/test_corridor_baselines.py tests/test_corridor_evaluation.py

The versioned configs/, scripts/, src/, and focused test modules contain the implementation and experiment tooling. Generated runs, datasets, and model checkpoints are kept outside the tracked source tree.
