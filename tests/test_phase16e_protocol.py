from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_phase16e_protocol_is_proposal_only_and_keeps_protected_blocks_closed():
    config = yaml.safe_load(
        (ROOT / "configs/phase16e_budget_reliability_confirmation_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["status"] == "approved_awaiting_hash_audit"
    assert config["execution_authorized"] is True
    assert config["environment"]["freshness_audit_required_before_lock"] is True
    assert config["environment"]["proposed_seed_start"] == 200000
    assert config["environment"]["proposed_seed_start"] + config["environment"][
        "proposed_layout_count"
    ] == 200500
    assert config["environment"]["proposed_seed_start"] >= 130200
    assert config["environment"]["proposed_seed_start"] >= 197100
    assert config["scope"]["d168_access"] is False
    assert config["scope"]["reserve_access"] is False
    assert config["scope"]["test_access"] is False


def test_phase16e_schedule_and_reliability_gates_are_fixed():
    config = yaml.safe_load(
        (ROOT / "configs/phase16e_budget_reliability_confirmation_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["planning"]["budgets"] == [32, 64, 128]
    assert config["planning"]["search_replica_ids"] == [0, 1, 2]
    assert config["execution"]["expected_unique_episode_count"] == 5000
    reference = config["gates"]["reference_B128"]
    candidates = config["gates"]["candidate_B32_B64"]
    assert reference["mean_safe_success_strictly_above"] == 0.90
    assert candidates["mean_safe_success_strictly_above"] == 0.90
    assert candidates["success_regression_event_upper_at_most"] == 0.05
    assert candidates["added_collision_event_upper_at_most"] == 0.05
    assert candidates["mean_rollout_transition_ratio_at_most"] == 0.70
    assert config["gates"]["selection_order"] == [32, 64, 128]
