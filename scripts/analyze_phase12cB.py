"""Analyze Phase 12c-B repeated banks with layouts as the bootstrap unit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_layout_mean(
    values: dict[str, list[float]], *, resamples: int, seed: int
) -> dict[str, float | int]:
    layout_ids = sorted(values)
    per_layout = np.asarray([np.mean(values[key]) for key in layout_ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(per_layout), size=(resamples, len(per_layout)))
    bootstrap = per_layout[indexes].mean(axis=1)
    return {
        "layouts": len(per_layout),
        "layout_macro_mean": float(per_layout.mean()),
        "ci95_lower": float(np.quantile(bootstrap, 0.025)),
        "ci95_upper": float(np.quantile(bootstrap, 0.975)),
    }


def _preference_distribution(records: list[dict[str, Any]], banks: range) -> tuple[np.ndarray, np.ndarray]:
    mass = np.zeros(9, dtype=np.float64)
    regret = np.zeros(9, dtype=np.float64)
    for index in banks:
        record = records[index]
        optimal = record["optimal_action_ids"]
        if not optimal:
            raise ValueError("bank record lacks an optimal action")
        mass[np.asarray(optimal, dtype=np.int64)] += 1.0 / (len(optimal) * len(banks))
        regret += np.asarray(record["group_normalized_regrets"], dtype=np.float64) / len(banks)
    return mass, regret


def _modal_action(mass: np.ndarray, regret: np.ndarray) -> int:
    best_mass = float(np.max(mass))
    tied = np.flatnonzero(np.isclose(mass, best_mass, rtol=0.0, atol=1e-12))
    best_regret = float(np.min(regret[tied]))
    tied = tied[np.isclose(regret[tied], best_regret, rtol=0.0, atol=1e-12)]
    return int(tied[0])


def _normalized_entropy(probability: np.ndarray) -> float:
    positive = probability[probability > 0]
    return float(-np.sum(positive * np.log(positive)) / math.log(9.0))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    middle = 0.5 * (p + q)
    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log(left[mask] / right[mask])))
    return float((0.5 * kl(p, middle) + 0.5 * kl(q, middle)) / math.log(2.0))


def _pairwise_flip_rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("near", "medium", "wide")
    limits = (0.125, 0.375)
    counts = {name: [0, 0] for name in names}
    for left_index in range(len(records)):
        left_ranks = records[left_index]["group_average_ranks"]
        for right_index in range(left_index + 1, len(records)):
            right_ranks = records[right_index]["group_average_ranks"]
            for first in range(9):
                for second in range(first + 1, 9):
                    left_gap = float(left_ranks[first]) - float(left_ranks[second])
                    right_gap = float(right_ranks[first]) - float(right_ranks[second])
                    if left_gap == 0 or right_gap == 0:
                        continue
                    margin = min(abs(left_gap), abs(right_gap)) / 8.0
                    bin_name = "near" if margin <= limits[0] else "medium" if margin <= limits[1] else "wide"
                    counts[bin_name][1] += 1
                    counts[bin_name][0] += int((left_gap < 0) != (right_gap < 0))
    return {
        name: {
            "reversals": counts[name][0],
            "comparisons": counts[name][1],
            "flip_rate": counts[name][0] / counts[name][1] if counts[name][1] else None,
        }
        for name in names
    }


def analyze(run_dir: Path, sample_manifest_path: Path, config_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_phase12c_b_only":
        raise ValueError("run manifest is not complete")
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    samples = json.loads(sample_manifest_path.read_text(encoding="utf-8"))["samples"]
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    records_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (run_dir / "bank_records.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                sample_id = record["sample_id"]
                if sample_id not in sample_by_id:
                    raise ValueError("bank record refers to a sample outside the frozen manifest")
                if record["layout_sha256"] != sample_by_id[sample_id]["layout_sha256"]:
                    raise ValueError("bank record layout hash differs from sample manifest")
                records_by_sample[sample_id].append(record)
    if set(records_by_sample) != set(sample_by_id):
        raise ValueError("one or more sampled states are missing bank records")
    for sample_id, rows in records_by_sample.items():
        rows.sort(key=lambda row: row["bank_index"])
        if [row["bank_index"] for row in rows] != list(range(12)):
            raise ValueError(f"bank indices incomplete or duplicated for {sample_id}")
        if len({row["proposal_seed"] for row in rows}) != 12:
            raise ValueError(f"proposal bank seeds are not unique for {sample_id}")
        if any(len(row["group_objective_keys"]) != 9 for row in rows):
            raise ValueError("every bank must contain all nine action-group keys")

    layout_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    state_details: list[dict[str, Any]] = []
    train_entropies: list[float] = []
    holdout_inclusions: list[float] = []
    preference_js: list[float] = []
    availability: list[float] = []
    objective_ranges = np.zeros(3, dtype=np.float64)
    pairwise_per_state: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for sample in samples:
        rows = records_by_sample[sample["sample_id"]]
        train_mass, train_regret = _preference_distribution(rows, range(6))
        modal = _modal_action(train_mass, train_regret)
        held_mass, held_regret = _preference_distribution(rows, range(6, 12))
        regret = float(held_regret[modal])
        inclusion = float(np.mean([
            modal in rows[index]["optimal_action_ids"] for index in range(6, 12)
        ]))
        entropy = _normalized_entropy(train_mass)
        jsd = _js_divergence(train_mass, held_mass)
        flips = _pairwise_flip_rates(rows)
        layout = sample["layout_sha256"]
        for key, value in (
            ("heldout_modal_regret", regret),
            ("heldout_optimal_set_inclusion", inclusion),
            ("train_preference_entropy", entropy),
            ("train_heldout_preference_js", jsd),
            ("bank_success_candidate_availability", float(np.mean([row["candidate_success_available"] for row in rows]))),
        ):
            layout_values[layout][key].append(value)
        for name in ("near", "medium", "wide"):
            rate = flips[name]["flip_rate"]
            if rate is not None:
                pairwise_per_state[name][layout].append(rate)
        keys = np.asarray([row["group_objective_keys"] for row in rows], dtype=np.float64)
        objective_ranges += np.ptp(keys, axis=0).mean(axis=0) / len(samples)

        swapped_modal = _modal_action(held_mass, held_regret)
        swapped_regret = float(train_regret[swapped_modal])
        state_details.append({
            **sample,
            "training_preference_probabilities": train_mass.tolist(),
            "heldout_preference_probabilities": held_mass.tolist(),
            "training_modal_action": modal,
            "training_action_normalized_regrets": train_regret.tolist(),
            "heldout_action_normalized_regrets": held_regret.tolist(),
            "heldout_modal_action_normalized_regret": regret,
            "heldout_optimal_set_inclusion": inclusion,
            "training_normalized_preference_entropy": entropy,
            "train_heldout_preference_js_over_log2": jsd,
            "bank_success_candidate_availability": float(np.mean([row["candidate_success_available"] for row in rows])),
            "pairwise_flip_rates": flips,
            "reverse_half_modal_action": swapped_modal,
            "reverse_half_heldout_normalized_regret": swapped_regret,
        })
        train_entropies.append(entropy)
        holdout_inclusions.append(inclusion)
        preference_js.append(jsd)
        availability.append(float(np.mean([row["candidate_success_available"] for row in rows])))

    stats = config["statistics"]
    bootstrap = stats["bootstrap"]
    metric_stats = {
        key: _bootstrap_layout_mean(values, resamples=int(bootstrap["resamples"]), seed=int(bootstrap["seed"]))
        for key, values in (
            ("heldout_modal_regret", {k: v["heldout_modal_regret"] for k, v in layout_values.items()}),
            ("heldout_optimal_set_inclusion", {k: v["heldout_optimal_set_inclusion"] for k, v in layout_values.items()}),
            ("training_preference_entropy", {k: v["train_preference_entropy"] for k, v in layout_values.items()}),
            ("train_heldout_preference_js", {k: v["train_heldout_preference_js"] for k, v in layout_values.items()}),
            ("success_candidate_availability", {k: v["bank_success_candidate_availability"] for k, v in layout_values.items()}),
        )
    }
    reverse_values: dict[str, list[float]] = defaultdict(list)
    for state in state_details:
        reverse_values[state["layout_sha256"]].append(state["reverse_half_heldout_normalized_regret"])
    reverse_by_layout = {key: float(np.mean(value)) for key, value in reverse_values.items()}
    reverse_stats = _bootstrap_layout_mean(
        {key: [value] for key, value in reverse_by_layout.items()},
        resamples=int(bootstrap["resamples"]), seed=int(bootstrap["seed"]) + 1,
    )
    flip_stats: dict[str, Any] = {}
    for name in ("near", "medium", "wide"):
        by_layout = pairwise_per_state[name]
        reduced = {key: [float(np.mean(rates))] for key, rates in by_layout.items() if rates}
        flip_stats[name] = _bootstrap_layout_mean(
            reduced, resamples=int(bootstrap["resamples"]), seed=int(bootstrap["seed"]) + 2,
        ) if reduced else {"layouts": 0, "layout_macro_mean": None, "ci95_lower": None, "ci95_upper": None}

    primary = metric_stats["heldout_modal_regret"]
    gate = (
        primary["layout_macro_mean"] <= float(stats["primary_gate"]["layout_macro_heldout_modal_action_regret_maximum"])
        and primary["ci95_upper"] <= float(stats["primary_gate"]["two_sided_95_percent_interval_upper_maximum"])
    )
    return {
        "schema": "phase12c_b_repeated_bank_analysis_v1",
        "status": "analyzed_phase12c_b_only",
        "sample_count": len(samples),
        "layout_count": len(layout_values),
        "bank_records": sum(len(rows) for rows in records_by_sample.values()),
        "primary_gate_pass": bool(gate),
        "primary_metric": metric_stats["heldout_modal_regret"],
        "gate_thresholds": stats["primary_gate"],
        "metrics_layout_cluster_bootstrap_95": metric_stats,
        "reverse_half_sensitivity_heldout_regret": reverse_stats,
        "pairwise_order_flip_rates_by_rank_margin_bin": flip_stats,
        "mean_objective_component_range_across_banks": {
            "outcome_class": float(objective_ranges[0]),
            "conditional_progress_metric": float(objective_ranges[1]),
            "noncoast_action_effort": float(objective_ranges[2]),
        },
        "interpretation": (
            "state_only_modal_preferences_pass_operational_realizability_gate"
            if gate else "state_only_modal_preferences_fail_operational_realizability_gate"
        ),
        "teacher_bank_action_agreement_is_not_treated_as_bayes_error": True,
        "reserve_accessed": False,
        "student_training_performed": False,
        "dagger_performed": False,
        "states": state_details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--sample-manifest", type=Path, default=ROOT / "artifacts/runs/phase12cB_state_sample_manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase12cB_locked.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    sample_path = args.sample_manifest if args.sample_manifest.is_absolute() else ROOT / args.sample_manifest
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    result = analyze(run_dir, sample_path, config_path)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output = output.resolve()
        if output.exists() or ROOT not in output.parents:
            raise SystemExit("refusing overwrite or output outside repository")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in (
            "status", "primary_gate_pass", "primary_metric", "gate_thresholds",
            "reverse_half_sensitivity_heldout_regret", "interpretation",
            "reserve_accessed", "student_training_performed",
        )
    }, indent=2))
    return 0 if result["primary_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
