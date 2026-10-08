from __future__ import annotations

import math
import hashlib
import numpy as np

from consequence_lab import phase29_iterative_refinement as p29


def test_phase29_config_is_bounded_and_keeps_sealed_data_closed():
    cfg = p29.config()
    assert cfg["planning"]["max_refinement_depth"] == 4
    assert cfg["planning"]["maximum_horizon"] == 8
    assert cfg["planning"]["maximum_selected_states"] == 1500
    assert cfg["planning"]["maximum_new_exact_sequence_evaluations"] == 63000
    assert cfg["execution"]["new_episodes"] is False
    assert cfg["execution"]["model_training"] is False
    assert cfg["execution"]["offline_predictability_fit_only"] is True
    assert cfg["execution"]["reserve_test_d168_access"] is False
    assert cfg["amendment"] == "reports/phase29_parent_protocol_hash_compatibility_amendment.md"


def test_predictability_plan_features_are_fixed_width_across_horizons():
    cfg = p29.config()
    maximum_horizon = cfg["planning"]["maximum_horizon"]
    canonical = [0.25, -0.5, 1.0]
    objective = [1, 0.125, 3.0]
    short = p29._predictability_features(canonical, (1, 2, 3, 4), objective, maximum_horizon)
    long = p29._predictability_features(canonical, (1, 2, 3, 4, 5, 6, 7, 8), objective, maximum_horizon)
    assert len(short) == len(long)
    assert short[:3] == canonical
    # Three features per active primitive: two physical action coordinates, then the mask.
    assert short[3 + 2 * maximum_horizon:3 + 3 * maximum_horizon] == [1.0] * 4 + [0.0] * 4
    assert short[-4] == 0.5
    assert short[-3:] == objective


def test_plan_feature_builder_rejects_invalid_horizons():
    try:
        p29._predictability_features([0.0], (), [0, 0, 0], 8)
    except ValueError as exc:
        assert "horizon" in str(exc)
    else:
        raise AssertionError("empty plan was accepted")
    try:
        p29._predictability_features([0.0], tuple(range(9)), [0, 0, 0], 8)
    except ValueError as exc:
        assert "horizon" in str(exc)
    else:
        raise AssertionError("overlong plan was accepted")


def test_top_three_uses_objective_then_sequence_tiebreak():
    bank = [
        {"objective": [1, 0.4, 1.0], "sequence": [2, 0]},
        {"objective": [1, 0.4, 1.0], "sequence": [1, 2]},
        {"objective": [0, 9.0, 5.0], "sequence": [8, 8]},
        {"objective": [2, 0.0, 0.0], "sequence": [0, 0]},
    ]
    assert [row["sequence"] for row in p29._top_three(bank)] == [[8, 8], [1, 2], [2, 0]]


def test_phase28_protocol_compatibility_accepts_only_the_known_status_change():
    locked = b"---\nstatus: protocol-frozen-pending-audit-lock\n---\nbody\n"
    current = b"---\nstatus: protocol-frozen-complete\n---\nbody\n"
    expected = p29.hashlib.sha256(locked).hexdigest().upper()
    checked = p29._verify_phase28_protocol_status_compatibility(locked, current, expected)
    assert checked["locked_protocol_sha256"] == expected
    try:
        p29._verify_phase28_protocol_status_compatibility(locked, current.replace(b"body", b"changed"), expected)
    except RuntimeError as exc:
        assert "status-only" in str(exc)
    else:
        raise AssertionError("unrelated Phase28 protocol change was accepted")


def test_monotone_path_requires_every_step_to_be_nonworsening():
    base, target = (0, 0), (1, 1)
    improving = {
        base: {"objective": [2, 0.0, 0.0]},
        (1, 0): {"objective": [1, 0.0, 0.0]},
        (0, 1): {"objective": [2, 1.0, 0.0]},
        target: {"objective": [0, 0.0, 0.0]},
    }
    assert p29._path_is_monotone(base, target, improving, [0, 1])
    worsening_first = dict(improving)
    worsening_first[(1, 0)] = {"objective": [3, 0.0, 0.0]}
    worsening_first[(0, 1)] = {"objective": [3, 0.0, 0.0]}
    assert not p29._path_is_monotone(base, target, worsening_first, [0, 1])


def test_multilabel_topk_uses_only_valid_edits_and_stable_id_ties():
    scores = np.zeros(8, dtype=np.float64)
    assert p29._topk_metrics(scores, {3}, [3, 1, 4, 6]) == (0.0, 1.0, 0.5)
    assert all(math.isnan(value) for value in p29._topk_metrics(scores, set(), [1, 2]))


def test_small_cart_can_recover_deterministic_multilabel_rule():
    x = np.arange(40, dtype=np.float64).reshape(-1, 1)
    y = np.zeros((40, 2), dtype=np.float64)
    y[:20, 0] = 1.0
    y[20:, 1] = 1.0
    tree = p29._build_tree(x, y, np.arange(len(x)), max_depth=1, min_leaf=5)
    prediction = p29._tree_predict(tree, np.asarray([[1.0], [38.0]]))
    assert prediction[0, 0] > prediction[0, 1]
    assert prediction[1, 1] > prediction[1, 0]


def test_predictability_audit_uses_grouped_folds_and_returns_paired_reference():
    cfg = p29.config()
    rows = []
    fold_layouts: dict[int, str] = {}
    candidate = 0
    while len(fold_layouts) < 5:
        layout = hashlib.sha256(f"layout-{candidate}".encode()).hexdigest()
        fold = int.from_bytes(hashlib.sha256(
            f"{layout}|phase29-fold-v1".encode()).digest()[:8], "big") % 5
        fold_layouts.setdefault(fold, layout)
        candidate += 1
    for fold, layout in sorted(fold_layouts.items()):
        for replica in range(8):
            value = float(replica - 4)
            positive = 0 if value < 0 else 1
            rows.append({"predictability_eligible": True, "predictability_input": [value, float(fold)],
                         "positive_edit_ids": [positive], "valid_edit_ids": [0, 1, 2, 3],
                         "layout_sha256": layout, "fold": fold})
    result = p29._predictability(rows, cfg)
    assert result["heldout_layouts"] == 5
    assert set(result["models"]) == {
        "global_edit_prior", "nearest_neighbor", "ridge_multioutput", "cart_multioutput"}
    paired = result["models"]["ridge_multioutput"]["paired_vs_global_prior"]
    assert paired["layouts"] == 5
    assert -1.0 <= paired["ci95_lower"] <= paired["ci95_upper"] <= 1.0
