from consequence_lab import phase25_b24_confirmation as p25


def test_phase25_config_freezes_fresh_block_and_practical_compute_gate():
    cfg = p25.config()
    p25.validate_config(cfg)
    assert cfg["block"]["seed_start"] == 209500
    assert cfg["block"]["layout_count"] == 500
    assert cfg["planning"]["budgets"] == [24, 32]
    assert cfg["gates"]["b24_vs_b32"]["transition_ratio_point_at_most"] == 0.90


def test_phase25_schedule_pairs_only_b24_and_b32_with_three_replicas():
    cfg = p25.config()
    rows = p25.execution_order([{"seed": 209500, "layout_sha256": "a" * 64}], cfg)
    assert len(rows) == 6
    assert {(row["method"], row["replica"]) for row in rows} == {
        (f"shifted_b{budget}", replica) for budget in (24, 32) for replica in (0, 1, 2)
    }
    assert len({row["global_schedule_index"] for row in rows}) == 6


def test_phase25_rejects_overlap_with_phase24_and_sealed_blocks():
    cfg = p25.config()
    p25.assert_unsealed(209500, 500, cfg["block"])
    for start, count in ((209499, 2), (196999, 2), (130199, 2)):
        try:
            p25.assert_unsealed(start, count, cfg["block"])
        except ValueError:
            pass
        else:
            raise AssertionError(f"overlapping seed range was accepted: {(start, count)}")


def test_phase25_absolute_gate_requires_all_outcome_and_replica_conditions():
    gate = p25.config()["gates"]["absolute"]
    method = {"success_rate": .995, "collision_rate": .005, "timeout_rate": 0.0}
    replicas = {str(i): {"success_rate": .99, "collision_rate": .01} for i in range(3)}
    assert p25._absolute_pass(method, replicas, gate)
    replicas["2"]["collision_rate"] = .11
    assert not p25._absolute_pass(method, replicas, gate)
