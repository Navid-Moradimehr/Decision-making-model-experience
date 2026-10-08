from __future__ import annotations

import yaml

from consequence_lab import phase24_search_budget_profile as p24
from consequence_lab import phase28_sequential_exact_search as p28


def test_phase28_config_is_exact_and_reserve_free():
    cfg = yaml.safe_load((p28.ROOT / p28.CONFIG).read_text(encoding="utf-8"))
    p28.validate_config(cfg)
    assert cfg["arms"] == list(p28.ARMS)
    assert cfg["amendment"] == p28.AMENDMENT
    assert cfg["planning"]["matched_average_budget"] == 26
    assert cfg["execution"]["reserve_test_d168_access"] is False


def test_phase28_shifted_candidate_banks_nest_by_identity_for_no_previous_and_shifted():
    for previous in (None, (0, 1, 2, 3, 4, 5, 6, 7)):
        banks = [set(p24.shifted_bank(8, budget, 7123, previous)[0]) for budget in (24, 26, 32)]
        assert len(banks[0]) == 24 and len(banks[1]) == 26 and len(banks[2]) == 32
        assert banks[0] < banks[1] < banks[2]


def test_planner_native_trigger_requires_no_safe_success_and_uses_margin_cutoff():
    cfg = p28.config()
    tie_cut = cfg["allocation"]["native_tie_hash_cutoff"]
    assert p28.planner_native_trigger(True, .01, tie_cut, cfg)
    assert not p28.planner_native_trigger(False, .01, tie_cut, cfg)
    assert not p28.planner_native_trigger(True, .05, tie_cut, cfg)
    gap_cut = cfg["allocation"]["native_gap_cutoff"]
    assert not p28.planner_native_trigger(True, gap_cut, tie_cut + 1, cfg)


def test_phase28_identity_hash_is_deterministic_and_salt_separated():
    a = p28._native_tie("a" * 64, 1, 20, "random")
    assert a == p28._native_tie("a" * 64, 1, 20, "random")
    assert a != p28._native_tie("a" * 64, 1, 20, "native")


def test_phase28_cutoff_uses_stable_tie_order_and_hits_target_rate():
    rows = [
        {"gap": 0.0, "tie": 5, "no_success": True},
        {"gap": 0.0, "tie": 2, "no_success": True},
        {"gap": 0.0, "tie": 8, "no_success": True},
        {"gap": 0.1, "tie": 1, "no_success": True},
        {"gap": 0.0, "tie": 0, "no_success": False},
        {"gap": 0.0, "tie": 3, "no_success": False},
        {"gap": 0.2, "tie": 9, "no_success": True},
        {"gap": 0.3, "tie": 4, "no_success": True},
    ]
    cutoff = p28._order_stat_cutoff(rows, .25)
    assert cutoff == {"target_count": 2, "gap_cutoff": 0.0,
                      "tie_hash_cutoff": 5, "activated_count": 2}


def test_phase28_candidate_selection_is_exact_lexicographic_minimum():
    from types import SimpleNamespace
    from consequence_lab import phase20_exact_plan_warmstart as p20

    candidates = [SimpleNamespace(sequence=(2, 0), objective=(1, .2, 2.0)),
                  SimpleNamespace(sequence=(1, 0), objective=(1, .2, 1.0)),
                  SimpleNamespace(sequence=(0, 0), objective=(0, .9, 8.0))]
    selected, ties = p20.choose_lexicographic_best(candidates)
    assert selected.sequence == (0, 0)
    assert ties == 1


def test_phase28_primary_comparison_config_is_frozen():
    cfg = p28.config()
    assert cfg["analysis"]["success_noninferiority_margin"] == -.015
    assert cfg["analysis"]["collision_noninferiority_margin"] == .015
    assert cfg["gates"]["absolute_safe_success_strictly_above"] == .90
    assert cfg["gates"]["timeout_exactly"] == 0
