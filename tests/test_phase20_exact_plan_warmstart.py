from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from consequence_lab import phase18_proposal_controls as p18
from consequence_lab import phase20_exact_plan_warmstart as p20
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import ACTION_VECTORS
from consequence_lab.phase16d_compute_frontier import uniform_prefix_bank


def test_locked_config_methods_budgets_and_familywise_confidence():
    cfg = p20.config()
    p20.validate_config(cfg)
    assert sum(m["replicas"] for m in cfg["methods"]) == 13
    assert cfg["block"]["episodes"] == 6500
    assert cfg["analysis"]["rare_event_one_sided_exact_confidence"] == pytest.approx(1 - .05 / 6)
    assert cfg["analysis"]["success_one_sided_bootstrap_confidence"] == pytest.approx(1 - .05 / 3)
    broken = deepcopy(cfg)
    broken["methods"][1]["budget"] = 48
    with pytest.raises(ValueError):
        p20.validate_config(broken)


def test_uniform_controls_are_exact_nested_prefix_and_first_query_has_no_warm_bias():
    for horizon in range(1, 9):
        b32, _, _ = uniform_prefix_bank(horizon, 32, 931)
        b48, _, _ = uniform_prefix_bank(horizon, 48, 931)
        p32, _ = p20.candidate_bank(horizon, 32, 931, "shift_previous_winner", None, 0,
                                   p20.config()["planning"]["mutation_seed_xor"])
        assert b32 == b48[:len(b32)] == p32


def test_shift_and_mutation_keep_exact_budget_unique_and_nine_constants():
    cfg = p20.config()
    previous = (0, 1, 2, 3, 4, 5, 6, 7)
    uniform, _, _ = uniform_prefix_bank(8, 32, 7101)
    shift, shift_stats = p20.candidate_bank(8, 32, 7101, "shift_previous_winner", previous, 0,
                                            cfg["planning"]["mutation_seed_xor"])
    mutated, mutation_stats = p20.candidate_bank(8, 32, 7101, "shift_plus_one_action_mutations", previous, 4,
        cfg["planning"]["mutation_seed_xor"])
    for bank, stats, max_proposals in ((shift, shift_stats, 1), (mutated, mutation_stats, 5)):
        assert len(bank) == len(set(bank)) == 32
        assert bank[:9] == uniform[:9]
        assert stats["accepted_insertions"] <= max_proposals
        assert stats["replaced_uniform_sequences"] == stats["accepted_insertions"]
        assert stats["uniform_sequences_retained"] + 9 + stats["accepted_insertions"] == 32
    assert shift_stats["accepted_insertions"] == 1
    assert mutation_stats["accepted_insertions"] == 5
    assert p20.shifted_plan((2, 4, 6), 2) == (4, 6)
    assert p20.shifted_plan((2, 4), 4) == (4, 4, 4, 4)


def test_mutations_change_one_action_to_a_physical_grid_neighbor_and_repeat():
    plan = (0, 1, 2, 3, 4, 5, 6, 7)
    first, positions = p20.mutation_proposals(plan, 12345, 4, 0x51A7E20D)
    second, positions2 = p20.mutation_proposals(plan, 12345, 4, 0x51A7E20D)
    assert first == second and positions == positions2
    assert len(set(positions)) == 4
    for candidate, position in zip(first, positions, strict=True):
        changed = [i for i, pair in enumerate(zip(plan, candidate, strict=True)) if pair[0] != pair[1]]
        assert changed == [position]
        a, b = ACTION_VECTORS[plan[position]], ACTION_VECTORS[candidate[position]]
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
        assert candidate[position] in p20.neighboring_action_ids(plan[position])


def test_duplicate_proposal_keeps_uniform_member_without_breaking_budget():
    base, _, _ = uniform_prefix_bank(8, 32, 17)
    bank, stats = p20.candidate_bank(8, 32, 17, "shift_previous_winner", (0,) * 8, 0,
                                     p20.config()["planning"]["mutation_seed_xor"])
    assert bank == base
    assert stats["accepted_insertions"] == 0
    assert stats["proposal_overlaps"] == 1
    assert stats["uniform_sequences_retained"] == 23


def test_sealed_ranges_and_schedule_prefix_contract():
    cfg = p20.config()
    for start, count in ((130000, 1), (197099, 3), (201000, 1), (202499, 2), (203499, 2)):
        with pytest.raises(ValueError):
            p20.assert_unsealed_range(start, count, cfg)
    local_cfg = deepcopy(cfg)
    local_cfg["block"]["episodes"] = 26
    layouts = [{"seed": 1, "layout_sha256": "L1"}, {"seed": 2, "layout_sha256": "L2"}]
    schedule = p20.execution_order(layouts, local_cfg)
    assert len(schedule) == 26
    records = [{**row, "outcome": "success", "execution_lock_sha256": "hash"} for row in schedule[:14]]
    p18.validate_prefix(records, schedule, "hash")
    assert schedule[13]["local_order_index"] == 0
    for index_name in ("global_schedule_index", "local_order_index", "execution_lock_sha256"):
        bad = deepcopy(records)
        bad[-1][index_name] = "corrupt"
        with pytest.raises(RuntimeError):
            p18.validate_prefix(bad, schedule, "hash")


def fixture_record(arm, outcome="success", transitions=None):
    geometry = arm["method"] == "geometry_waypoint"
    if geometry:
        outcome = "collision"
    transitions = transitions if transitions is not None else (
        0 if geometry else 650 if arm["budget"] == 32 else 1000)
    latency = .001 if geometry else .040 if arm["budget"] == 32 else .060
    return {**arm, "outcome": outcome, "steps": 20, "decision_count": 20,
        "exact_sequences": 0 if geometry else arm["budget"] * 20,
        "exact_transitions": transitions, "executed_transitions": 20,
        "duplicate_rejections": 0, "fallback_sequences": 0,
        "complete_success_sequences": 20 if not geometry else 0,
        "selected_action_counts": [0] * 9,
        "latency_seconds_per_decision": [latency] * 20,
        "geometry_episode_average_decision_latency": latency if geometry else None,
        "wall_seconds": latency * 20, "decision_diagnostics": []}


def synthetic_study(count=500):
    cfg = deepcopy(p20.config())
    cfg["analysis"]["bootstrap_draws"] = 100
    cfg["block"]["episodes"] = 13 * count
    cfg["block"]["proposed_layout_count"] = count
    layouts = [{"seed": i, "layout_sha256": str(i)} for i in range(count)]
    schedule = p20.execution_order(layouts, cfg)
    return cfg, schedule, [fixture_record(arm) for arm in schedule]


def test_selection_chooses_simplest_passing_method_and_metrics_recompute():
    cfg, schedule, records = synthetic_study()
    result = p20.summarize(records, schedule, cfg)
    assert result["reference_pass"]
    assert all(result["qualification"][name]["pass"]
               for name in ("uniform_b32", "shifted_b32", "mutated_b32"))
    assert result["selected_method"] == "uniform_b32"
    assert result["qualification"]["shifted_b32"]["transition_ratio_vs_b48"]["difference"] == pytest.approx(.65)
    assert result["qualification"]["mutated_b32"]["latency_ratio_vs_b48"] == pytest.approx(2 / 3)
    assert result["warmstart_attribution"]["shifted_b32"]["benefit_attributed"] is False


def test_warmstart_does_not_pass_when_layout_rare_event_gate_fails():
    cfg, schedule, records = synthetic_study()
    for row in records:
        if row["method"] == "shifted_b32" and row["seed"] < 18:
            row["outcome"] = "collision"
    result = p20.summarize(records, schedule, cfg)
    assert result["qualification"]["shifted_b32"]["events_vs_b48"]["added_collision"]["layouts"] == 18
    assert result["qualification"]["shifted_b32"]["events_vs_b48"]["added_collision"]["upper"] > .05
    assert result["qualification"]["shifted_b32"]["pass"] is False
    assert result["selected_method"] == "uniform_b32"


def test_no_b32_arm_selected_when_reliability_or_compute_gate_fails():
    cfg, schedule, records = synthetic_study()
    for row in records:
        if row["method"] in {"uniform_b32", "shifted_b32", "mutated_b32"} and row["seed"] < 18:
            row["outcome"] = "collision"
        if row["method"] in {"uniform_b32", "shifted_b32", "mutated_b32"}:
            row["exact_transitions"] = 900
    result = p20.summarize(records, schedule, cfg)
    assert not any(g["pass"] for g in result["qualification"].values())
    assert result["selected_method"] is None
    assert result["decision"] == "retain_uniform_b48"


def test_locked_sampler_choices_feed_exact_objective_selection(monkeypatch):
    scenario = generate_staggered_corridor(201000)  # Previously consumed software fixture.
    scenario = replace(scenario, initial_state=replace(
        scenario.initial_state, elapsed_steps=scenario.spec.max_steps - 1))
    cfg = p20.config()
    method = next(m for m in cfg["methods"] if m["name"] == "uniform_b32")
    observed = []

    def exact_score(state, scene, sequence):
        observed.append(sequence)
        return SimpleNamespace(sequence=sequence, objective=(0 if sequence[0] == 8 else 1, sequence),
                               success_reached=False, transitions=1)

    monkeypatch.setattr(p20, "evaluate_sequence", exact_score)
    successor, chosen, _, record = p20.evaluate_decision(scenario, scenario.initial_state, method, 0, None, cfg)
    assert len(observed) == len(set(observed)) == 9  # min(32, 9**1)
    assert observed[:9] == [(action,) for action in range(9)]
    assert chosen == (8,)
    assert record["chosen_action"] == 8
    assert successor.elapsed_steps == scenario.spec.max_steps


def test_episode_keeps_previous_winning_plan_local_to_same_method_and_replica(monkeypatch):
    scenario = generate_staggered_corridor(201000)  # Previously consumed software fixture.
    scenario = replace(scenario, initial_state=replace(
        scenario.initial_state, elapsed_steps=scenario.spec.max_steps - 2))
    method = next(m for m in p20.config()["methods"] if m["name"] == "shifted_b32")
    seen = []

    def fake_decision(scene, state, arm, replica, previous_plan, cfg):
        seen.append(previous_plan)
        next_state = replace(state, elapsed_steps=state.elapsed_steps + 1,
                             status="active" if len(seen) == 1 else "timeout")
        plan = (0, 1, 2, 3, 4, 5, 6, 7) if len(seen) == 1 else (8,)
        return next_state, plan, .001, {"chosen_action": plan[0], "budget": 32,
            "exact_transitions": 32, "success_candidates": 0, "base_duplicate_rejections": 0,
            "proposal_overlaps": 0, "fallback_sequences": 0}

    monkeypatch.setattr(p20, "evaluate_decision", fake_decision)
    result = p20.evaluate_episode(scenario, method, 1, p20.config())
    assert seen == [None, (0, 1, 2, 3, 4, 5, 6, 7)]
    assert result["outcome"] == "timeout" and result["decision_count"] == 2
