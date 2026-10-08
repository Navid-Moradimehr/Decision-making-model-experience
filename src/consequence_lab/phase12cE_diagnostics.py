"""Detailed, read-only diagnostics for Phase 12c-E candidate-order checks.

This module does not train models or decide whether a model passes the frozen
permutation gate. It records enough detail to distinguish restored-probability
drift from semantic choice flips, including near-tie margins.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from consequence_lab.phase12c_c_training import ACTION_FEATURES, _model_probabilities

FROZEN_MAX_PROBABILITY_DEVIATION = 1e-6
FROZEN_MAX_SEMANTIC_CHOICE_FLIPS = 0


def detailed_permutation_diagnostic(
    model: torch.nn.Module,
    *,
    variant: str,
    state_rows: np.ndarray,
    device: torch.device,
    seed: int = 171002,
    fixtures: int = 16,
    permutations_per_fixture: int = 100,
) -> dict[str, Any]:
    """Return per-fixture and per-permutation evidence using the v2 test semantics.

    Candidate IDs are the canonical action IDs 0..8. A tie is resolved by the
    smallest semantic action ID, independent of the presented row order.
    """
    if fixtures < 1 or permutations_per_fixture < 1:
        raise ValueError("fixtures and permutations_per_fixture must be positive")
    rows = np.asarray(state_rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 52 or len(rows) < fixtures:
        raise ValueError("state_rows must contain at least fixtures rows of shape [N,52]")
    if not np.isfinite(rows[:fixtures]).all():
        raise ValueError("state_rows contain non-finite values")

    states = torch.as_tensor(rows[:fixtures], dtype=torch.float32, device=device)
    candidates = torch.as_tensor(ACTION_FEATURES, dtype=torch.float32, device=device)
    mask = torch.ones(9, dtype=torch.bool, device=device)
    rng = np.random.default_rng(seed)
    fixture_reports: list[dict[str, Any]] = []
    global_max = -1.0
    global_max_location: dict[str, int] | None = None
    semantic_flips = 0

    model.eval()
    with torch.no_grad():
        for fixture_index, state in enumerate(states):
            canonical_scores = model(state[None], candidates[None], mask[None])[0]
            canonical_probabilities = _model_probabilities(canonical_scores[None], variant)[0]
            canonical_array = canonical_probabilities.detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(canonical_array).all():
                raise FloatingPointError(f"non-finite canonical probabilities at fixture {fixture_index}")
            canonical_choice = int(np.argmax(canonical_array))
            ordered = np.sort(canonical_array)
            canonical_margin = float(ordered[-1] - ordered[-2])
            comparisons: list[dict[str, Any]] = []
            fixture_max = 0.0
            fixture_flips = 0
            for permutation_index in range(permutations_per_fixture):
                permutation = rng.permutation(9)
                perm = torch.as_tensor(permutation, dtype=torch.long, device=device)
                permuted_scores = model(state[None], candidates[perm][None], mask[perm][None])[0]
                permuted_probabilities = _model_probabilities(permuted_scores[None], variant)[0]
                permuted_array = permuted_probabilities.detach().cpu().numpy().astype(np.float64)
                if not np.isfinite(permuted_array).all():
                    raise FloatingPointError(
                        f"non-finite probabilities at fixture {fixture_index}, permutation {permutation_index}"
                    )
                restored = np.empty_like(permuted_array)
                restored[permutation] = permuted_array
                deviation = float(np.max(np.abs(restored - canonical_array)))
                choice = int(min(permutation[np.flatnonzero(permuted_array == permuted_array.max())]))
                flipped = choice != canonical_choice
                semantic_flips += int(flipped)
                fixture_flips += int(flipped)
                fixture_max = max(fixture_max, deviation)
                if deviation > global_max:
                    global_max = deviation
                    global_max_location = {
                        "fixture_index": fixture_index,
                        "permutation_index": permutation_index,
                    }
                sorted_permuted = np.sort(permuted_array)
                comparisons.append({
                    "permutation_index": permutation_index,
                    "candidate_ids_in_order": permutation.astype(int).tolist(),
                    "restored_max_probability_deviation": deviation,
                    "canonical_choice_id": canonical_choice,
                    "permuted_choice_id": choice,
                    "semantic_choice_flip": flipped,
                    "permuted_top_two_probability_margin": float(sorted_permuted[-1] - sorted_permuted[-2]),
                })
            fixture_reports.append({
                "fixture_index": fixture_index,
                "canonical_choice_id": canonical_choice,
                "canonical_top_two_probability_margin": canonical_margin,
                "max_probability_deviation": fixture_max,
                "semantic_choice_flips": fixture_flips,
                "comparisons": comparisons,
            })

    comparison_count = fixtures * permutations_per_fixture
    return {
        "schema": "phase12cE_detailed_permutation_diagnostic_v1",
        "seed": seed,
        "fixtures": fixtures,
        "permutations_per_fixture": permutations_per_fixture,
        "comparisons": comparison_count,
        "max_probability_deviation": max(global_max, 0.0),
        "max_deviation_location": global_max_location,
        "semantic_choice_flips": semantic_flips,
        "frozen_gate": {
            "max_probability_deviation_threshold": FROZEN_MAX_PROBABILITY_DEVIATION,
            "max_semantic_choice_flips": FROZEN_MAX_SEMANTIC_CHOICE_FLIPS,
            "passed": (
                max(global_max, 0.0) <= FROZEN_MAX_PROBABILITY_DEVIATION
                and semantic_flips <= FROZEN_MAX_SEMANTIC_CHOICE_FLIPS
            ),
        },
        "fixtures_detail": fixture_reports,
    }
