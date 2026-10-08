from copy import deepcopy

import pytest

from consequence_lab import phase21_shifted_b32_confirmation as p21
from consequence_lab.inertial_corridor import ACTION_VECTORS
from consequence_lab.phase16d_compute_frontier import uniform_prefix_bank


def test_frozen_phase21_design_and_parent_binding():
    cfg = p21.config()
    p21.validate_config(cfg)
    assert cfg["block"]["episodes"] == 5000
    assert [m["name"] for m in cfg["planning"]["methods"]] == [
        "uniform_b32", "shifted_b32", "uniform_b48", "geometry_waypoint"]
    assert cfg["parent"]["result_sha256"] == "F0340FC52AB7297BF662221591A5BB1D9FBEBD6AA6513BBD3F7C848DDB4C98E3"
    bad = deepcopy(cfg)
    bad["planning"]["methods"].pop(1)
    with pytest.raises(ValueError):
        p21.validate_config(bad)


def test_phase21_exclusion_ranges_fail_closed():
    cfg = p21.config()
    for start, count in ((130199, 2), (197000, 1), (201000, 1), (202499, 2),
                         (203000, 1), (205499, 2)):
        with pytest.raises(ValueError):
            p21.assert_unsealed_range(start, count, cfg)


def test_shifted_candidate_bank_uses_phase20_contract_without_mutation():
    cfg = p21.config()
    horizon, budget, seed = 8, 32, 7381
    uniform, _, _ = uniform_prefix_bank(horizon, budget, seed)
    first, _ = p21.p20.candidate_bank(horizon, budget, seed, "shift_previous_winner", None, 0,
                                      cfg["planning"]["mutation_seed_xor"])
    shifted, stats = p21.p20.candidate_bank(horizon, budget, seed, "shift_previous_winner",
        (0, 1, 2, 3, 4, 5, 6, 7), 0, cfg["planning"]["mutation_seed_xor"])
    assert first == uniform
    assert len(shifted) == len(set(shifted)) == 32
    assert shifted[:9] == uniform[:9]
    assert stats["accepted_insertions"] == 1
    assert stats["uniform_sequences_retained"] == 22
    assert all(0 <= action < len(ACTION_VECTORS) for sequence in shifted for action in sequence)
    assert stats["mutation_positions"] == []


def test_schedule_has_ten_arms_per_layout_and_stable_counterbalanced_order():
    cfg = deepcopy(p21.config())
    cfg["block"]["episodes"] = 20
    cfg["block"]["proposed_layout_count"] = 2
    layouts = [{"seed": 1, "layout_sha256": "A"}, {"seed": 2, "layout_sha256": "B"}]
    a = p21.execution_order(layouts, cfg)
    b = p21.execution_order(layouts, cfg)
    assert a == b
    assert len(a) == 20
    for seed in (1, 2):
        rows = [r for r in a if r["seed"] == seed]
        assert len(rows) == 10
        assert {r["method"] for r in rows} == {m["name"] for m in cfg["planning"]["methods"]}
        assert [r["local_order_index"] for r in rows] == list(range(10))
