"""Audit the saved Phase-2F Reacher populations for decision-local repairability.

This script only reads the existing planner development cache. It performs no
simulation, predictor inference, fitting, or mutation of the source artifacts.
"""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/runs/phase2_reacher_planner_dev/progress.jsonl"
RUN = ROOT / "artifacts/runs/phase10a_reacher_decision_gap"
REPORT = ROOT / "reports/phase10a_reacher_decision_gap.md"
GENERATORS = ("mlp_cem", "native_lewm_cem", "uniform_random")
SCORERS = ("supervised_mlp", "native_lewm")
SUCCESS_DISTANCE_TOLERANCE = 0.05
PHYSICAL_COST_TIE_TOLERANCE = 0.0025


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_vector(value, name: str, expected: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or (expected is not None and len(result) != expected):
        raise ValueError(f"{name} must be a vector of length {expected}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (zero-based), deterministic under exact ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def pairwise_accuracy(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, int]:
    i, j = np.triu_indices(len(truth), k=1)
    informative = np.abs(truth[i] - truth[j]) > PHYSICAL_COST_TIE_TOLERANCE
    count = int(informative.sum())
    if count == 0:
        return float("nan"), 0
    pdiff = predicted[i[informative]] - predicted[j[informative]]
    tdiff = truth[i[informative]] - truth[j[informative]]
    credit = np.where(pdiff == 0, 0.5, (pdiff * tdiff > 0).astype(float))
    return float(credit.mean()), count


def action_key(sequence: np.ndarray) -> tuple[float, ...]:
    """Stable tie-breaker based only on candidate content, never row position."""
    return tuple(np.asarray(sequence, dtype=np.float64).reshape(-1).tolist())


def select_index(scores: np.ndarray, actions: np.ndarray) -> tuple[int, int]:
    minimum = float(scores.min())
    tied = np.flatnonzero(scores == minimum)
    selected = min(tied.tolist(), key=lambda index: action_key(actions[index]))
    return selected, len(tied)


def candidate_order(scores: np.ndarray, actions: np.ndarray) -> np.ndarray:
    # Python's stable content sort handles score ties without depending on the
    # stored candidate row; identical action duplicates are outcome-equivalent.
    return np.asarray(
        sorted(range(len(scores)), key=lambda i: (float(scores[i]), action_key(actions[i]))),
        dtype=np.int64,
    )


def nearest_action_rms(actions: np.ndarray) -> dict:
    """Per-candidate nearest-neighbour RMS distance in flattened action space."""
    flat = actions.reshape(len(actions), -1)
    squared_norm = np.einsum("ij,ij->i", flat, flat)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (flat @ flat.T)
    np.maximum(squared_distance, 0.0, out=squared_distance)
    squared_distance /= flat.shape[1]
    np.fill_diagonal(squared_distance, np.inf)
    nearest = np.sqrt(squared_distance.min(axis=1))
    return {
        "minimum": float(nearest.min()),
        "median": float(np.median(nearest)),
        "p10": float(np.percentile(nearest, 10)),
        "definition": "RMS Euclidean distance across all 50 bounded action scalars to the nearest other candidate",
    }


def population_audit(actions_value, costs_value, success_value, scorer_values) -> dict:
    actions = np.asarray(actions_value, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[1:] != (25, 2) or len(actions) != 300:
        raise ValueError(f"candidate actions must have shape [300,25,2], got {actions.shape}")
    if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.0 + 1e-7):
        raise ValueError("candidate actions violate finite [-1,1] contract")
    truth = validate_vector(costs_value, "simulator costs", len(actions))
    success = np.asarray(success_value, dtype=bool)
    if success.shape != (len(actions),):
        raise ValueError("success bits do not align with candidate count")
    if not np.isfinite(truth).all() or np.any(truth < 0):
        raise ValueError("simulator costs must be finite and non-negative")

    unique_actions = np.unique(actions.reshape(len(actions), -1), axis=0)
    duplicate_count = len(actions) - len(unique_actions)
    result = {
        "candidate_count": len(actions),
        "exact_duplicate_count": int(duplicate_count),
        "nearest_action_rms": nearest_action_rms(actions),
        "candidate_available": bool(success.any()),
        "success_count": int(success.sum()),
        "cost_min": float(truth.min()),
        "cost_max": float(truth.max()),
        "cost_range": float(np.ptp(truth)),
        "scorers": {},
    }
    for scorer in SCORERS:
        predicted = validate_vector(scorer_values[scorer]["model_costs"], scorer, len(actions))
        selected, tie_count = select_index(predicted, actions)
        ordered = candidate_order(predicted, actions)
        pair_acc, pair_count = pairwise_accuracy(predicted, truth)
        predicted_rank = average_ranks(predicted)
        rank_gap = predicted_rank - predicted_rank.min()
        top = {}
        for k in (4, 8, 16, 120):
            top_indices = ordered[:k]
            top[str(k)] = {
                "success_present": bool(success[top_indices].any()),
                "success_count": int(success[top_indices].sum()),
                "candidate_indices": top_indices.tolist() if k <= 16 else None,
            }
        selected_true_cost = float(truth[selected])
        oracle_cost = float(truth.min())
        result["scorers"][scorer] = {
            "selected_index_content_tiebreak": selected,
            "selected_score_tie_count": tie_count,
            "selected_success": bool(success[selected]),
            "selected_regret": max(0.0, selected_true_cost - oracle_cost),
            "oracle_available_success": bool(success.any()),
            "pairwise_accuracy": pair_acc,
            "informative_pair_count": pair_count,
            "spearman_rank_correlation": float(
                np.corrcoef(average_ranks(predicted), average_ranks(truth))[0, 1]
            ) if np.ptp(predicted) > 0 and np.ptp(truth) > 0 else None,
            "top_k": top,
            # For s = base + delta and |delta| < .2, candidates more than
            # .4 normalized rank units behind the best cannot overtake it.
            "bounded_correction_success_ceiling": bool(
                np.any(success & (rank_gap < 0.4 * (len(actions) - 1)))
            ),
            "successful_candidates_within_base_rank_gap_0_4": int(
                np.sum(success & (rank_gap < 0.4 * (len(actions) - 1)))
            ),
        }
    return result


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 20:
        raise ValueError(f"expected 20 independent state contexts, found {len(rows)}")
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != 20 or set(seeds) != set(range(232000, 232020)):
        raise ValueError("state seeds do not match the historical [232000,232020) range")
    return sorted(rows, key=lambda row: int(row["seed"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN.with_name("phase10a_reacher_decision_gap_v2"),
        help="new run-artifact directory (must not already exist)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT.with_name("phase10a_reacher_decision_gap_v2.md"),
        help="new Markdown report path (must not already exist)",
    )
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    report_path = args.report_path.resolve()
    try:
        output_dir.relative_to(ROOT)
        report_path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("audit outputs must remain inside the project root") from exc
    if output_dir.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite a Phase-10A audit artifact")
    rows = load_rows(INPUT)
    seed_results = []
    for row in rows:
        groups = {}
        for generator in GENERATORS:
            group = row["groups"][generator]
            groups[generator] = population_audit(
                group["candidate_sequences"],
                group["simulator_costs"],
                group["success_bits"],
                group,
            )
        seed_results.append({"seed": int(row["seed"]), "groups": groups})

    aggregates = {}
    for generator in GENERATORS:
        aggregates[generator] = {}
        for scorer in SCORERS:
            cells = [item["groups"][generator]["scorers"][scorer] for item in seed_results]
            aggregates[generator][scorer] = {
                "availability_count": sum(item["candidate_available"] for item in [
                    result["groups"][generator] for result in seed_results
                ]),
                "selected_success_count": sum(item["selected_success"] for item in cells),
                "mean_regret_equal_state": float(np.mean([item["selected_regret"] for item in cells])),
                "mean_pairwise_accuracy_equal_state": float(np.nanmean([item["pairwise_accuracy"] for item in cells])),
                "top_k_success_availability": {
                    str(k): int(sum(item["top_k"][str(k)]["success_present"] for item in cells))
                    for k in (4, 8, 16, 120)
                },
                "bounded_correction_success_ceiling_count": sum(
                    item["bounded_correction_success_ceiling"] for item in cells
                ),
            }

    action_redundancy = {}
    for generator in GENERATORS:
        distances = []
        exact_duplicates = 0
        for row in rows:
            actions = np.asarray(row["groups"][generator]["candidate_sequences"], dtype=np.float64)
            exact_duplicates += len(actions) - len(np.unique(actions.reshape(len(actions), -1), axis=0))
            flat = actions.reshape(len(actions), -1)
            squared_norm = np.einsum("ij,ij->i", flat, flat)
            squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (flat @ flat.T)
            np.maximum(squared_distance, 0.0, out=squared_distance)
            squared_distance /= flat.shape[1]
            np.fill_diagonal(squared_distance, np.inf)
            distances.append(np.sqrt(squared_distance.min(axis=1)))
        pooled = np.concatenate(distances)
        action_redundancy[generator] = {
            "candidate_count": len(pooled),
            "exact_duplicate_count": int(exact_duplicates),
            "nearest_action_rms_pooled": {
                "minimum": float(pooled.min()),
                "median": float(np.median(pooled)),
                "p10": float(np.percentile(pooled, 10)),
            },
            "interpretation": "action-sequence similarity only; not dynamics- or outcome-equivalence",
        }

    result = {
        "audit_schema_version": 2,
        "status": "phase10a_cached_audit_complete",
        "exploratory_reuse": True,
        "independent_contexts": len(seed_results),
        "candidate_populations": len(seed_results) * len(GENERATORS),
        "candidate_outcomes": len(seed_results) * len(GENERATORS) * 300,
        "input_sha256": sha256(INPUT),
        "aggregation_unit": "independent state/goal seed; candidates are nested observations",
        "objective": {
            "simulator_cost": "saved phase2f terminal mean absolute qpos error divided by 2pi",
            "success": "saved phase2f qpos match at every joint within 0.05 radians",
            "physical_cost_tie_tolerance": PHYSICAL_COST_TIE_TOLERANCE,
        },
        "aggregates": aggregates,
        "action_redundancy": action_redundancy,
        "seed_results": seed_results,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = [
        "# Phase 10A — Reacher decision-gap cache audit",
        "",
        "## Status",
        "",
        "Exploratory audit of existing Phase-2F artifacts. No simulator, predictor, fitting, or new labels were used.",
        "",
        f"- Input: `artifacts/runs/phase2_reacher_planner_dev/progress.jsonl` (SHA-256 `{result['input_sha256']}`).",
        f"- Independent state/goal contexts: {result['independent_contexts']}; candidate populations: {result['candidate_populations']}; saved candidate outcomes: {result['candidate_outcomes']:,}.",
        "- The independent unit is the state/goal seed, not each candidate and not each candidate pair.",
        "- Objective is the historical terminal qpos cost and 0.05-radian success contract.",
        "",
        "## Results by generator and frozen scorer",
        "",
        "| Generator | Scorer | Available | Selected success | Mean regret | Mean pairwise | Top-4 / 8 / 16 / 120 availability | Bounded-repair ceiling |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for generator in GENERATORS:
        for scorer in SCORERS:
            cell = aggregates[generator][scorer]
            top = cell["top_k_success_availability"]
            md.append(
                f"| {generator} | {scorer} | {cell['availability_count']}/20 | "
                f"{cell['selected_success_count']}/20 | {cell['mean_regret_equal_state']:.5f} | "
                f"{cell['mean_pairwise_accuracy_equal_state']:.3f} | "
                f"{top['4']}/20 / {top['8']}/20 / {top['16']}/20 / {top['120']}/20 | "
                f"{cell['bounded_correction_success_ceiling_count']}/20 |"
            )
    md += [
        "",
        "## Interpretation",
        "",
        "The 18,000 candidate outcomes are nested within only 20 state/goal contexts. All confidence calculations and future splits must group by context. These data are already inspected; they are exploratory and cannot provide fresh confirmation.",
        "",
        "Candidate availability and ranking/selection are reported separately. The bounded-correction ceiling is only a necessary reachability screen under the specified rank scale; it does not establish that a learned correction can find or generalize to those candidates.",
        "",
        "The final CEM cache contains candidate actions, predicted costs, simulator costs, endpoint qpos and success, but no predicted/realized latent features. It supports score-level ordinal correction, not dense-future D-JEPA reproduction. CEM populations were themselves model-generated, so no-score-feature models are scoring baselines, not end-to-end model-free planners.",
        "",
    ]
    md += ["", "## Candidate action redundancy", "", "Nearest-neighbor distance is computed over all other candidates as RMS Euclidean distance across the 50 action scalars (25 steps x 2 controls). These are action-sequence proximity statistics only; they do not establish dynamical or outcome equivalence.", "", "| Generator | Exact duplicates | Pooled nearest-action RMS: minimum / p10 / median |", "|---|---:|---:|"]
    for generator in GENERATORS:
        redundancy = action_redundancy[generator]
        metric = redundancy["nearest_action_rms_pooled"]
        md.append(f"| {generator} | {redundancy['exact_duplicate_count']} | {metric['minimum']:.6f} / {metric['p10']:.6f} / {metric['median']:.6f} |")
    md += [
        "",
        "## Reproduction",
        "",
        f"- Audit: `python scripts/audit_phase10_reacher_decision_gap.py --output-dir {output_dir.relative_to(ROOT).as_posix()} --report-path {report_path.relative_to(ROOT).as_posix()}`.",
        f"- Machine-readable results: `{(output_dir / 'audit.json').relative_to(ROOT).as_posix()}`.",
        "- No source data was overwritten.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(report_path)
    print(json.dumps(aggregates, indent=2))


if __name__ == "__main__":
    main()
