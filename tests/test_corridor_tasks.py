import pytest

from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import validate_corridor_state


def test_generator_is_deterministic_and_layout_hash_is_geometry_only():
    first = generate_staggered_corridor(12007)
    again = generate_staggered_corridor(12007)
    assert first == again
    assert len(first.layout_sha256) == 64
    assert first.layout_family == "three_baffle_two_openings"
    assert len(first.spec.obstacles) == 9
    assert first.spec.obstacles == again.spec.obstacles


def test_layouts_vary_and_all_generated_initial_states_obey_public_contract():
    scenarios = [generate_staggered_corridor(seed) for seed in range(4100, 4140)]
    assert len({scenario.layout_sha256 for scenario in scenarios}) == len(scenarios)
    assert len({scenario.spec.goal[1] for scenario in scenarios}) > 30
    assert len({scenario.initial_state.y for scenario in scenarios}) > 30
    assert len({(scenario.initial_state.vx, scenario.initial_state.vy) for scenario in scenarios}) > 35
    for scenario in scenarios:
        validate_corridor_state(scenario.initial_state, scenario.spec)
        assert scenario.spec.width == scenario.spec.height == 1.0
        assert scenario.spec.max_steps == 160
        assert scenario.spec.goal[0] == 0.93
        assert all(0.0 <= obs.x_min < obs.x_max <= 1.0 for obs in scenario.spec.obstacles)
        assert all(0.0 <= obs.y_min < obs.y_max <= 1.0 for obs in scenario.spec.obstacles)


@pytest.mark.parametrize("seed", [-1, 1.5, True, "10"])
def test_generator_rejects_invalid_seed(seed):
    with pytest.raises(ValueError):
        generate_staggered_corridor(seed)
