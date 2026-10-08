"""Run diagnosis-only Phase 12 analyses on already-frozen artifacts.

This script performs no fitting, simulation, new seed generation, or reserve
access. Its action-equivalence metrics are post-hoc and do not change the
preregistered Phase 12 sequence-regret gate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_evaluation import load_student
from consequence_lab.phase12_postmortem import summarize_first_action_decision
from consequence_lab.phase12_training import evaluate_offline, load_phase12_split


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _dataset_summary(data: dict[str, np.ndarray]) -> dict[str, object]:
    mask = data["candidate_mask"].astype(bool)
    regrets = data["target_normalized_regret"]
    optimal = data["target_optimal_mask"].astype(bool)
    valid_n = mask.sum(axis=1)
    optimal_n = optimal.sum(axis=1)
    first = data["sequences"][:, :, 0]
    distinct_opt_actions = np.asarray([
        len(set(int(x) for x in first[row, optimal[row] & mask[row]]))
        for row in range(len(mask))
    ])
    optimal_candidate_fraction = optimal_n / valid_n
    return {
        "queries": int(len(mask)),
        "layouts": int(len(np.unique(data["meta_layout_index"]))),
        "valid_candidates_per_query_min_max": [int(valid_n.min()), int(valid_n.max())],
        "optimal_candidate_count_median": float(np.median(optimal_n)),
        "optimal_candidate_count_mean": float(np.mean(optimal_n)),
        "optimal_candidate_fraction_mean": float(np.mean(optimal_candidate_fraction)),
        "multiple_optimal_first_action_fraction": float(np.mean(distinct_opt_actions > 1)),
        "distinct_optimal_first_actions_mean": float(np.mean(distinct_opt_actions)),
        "success_sequence_available_fraction": float(np.mean(data["target_success_candidate_available"])),
        "decision_index_min_max": [int(data["meta_decision_index"].min()), int(data["meta_decision_index"].max())],
        "normalized_regret_quantiles": {
            "q0": float(np.quantile(regrets[mask], 0.0)),
            "q25": float(np.quantile(regrets[mask], 0.25)),
            "q50": float(np.quantile(regrets[mask], 0.50)),
            "q75": float(np.quantile(regrets[mask], 0.75)),
            "q100": float(np.quantile(regrets[mask], 1.0)),
        },
    }


def _per_layout(selected_regret: np.ndarray, action_regret: np.ndarray, layout: np.ndarray) -> dict[str, object]:
    seq_values = []
    action_values = []
    rows = []
    for item in np.unique(layout):
        active = layout == item
        seq_value = float(np.mean(selected_regret[active]))
        action_value = float(np.mean(action_regret[active]))
        seq_values.append(seq_value)
        action_values.append(action_value)
        rows.append({
            "layout_index": int(item),
            "queries": int(active.sum()),
            "sequence_regret": seq_value,
            "teacher_action_group_normalized_regret": action_value,
        })
    seq_values = np.asarray(seq_values)
    action_values = np.asarray(action_values)
    rng = np.random.default_rng(1212)
    indices = rng.integers(0, len(rows), size=(10000, len(rows)))
    seq_boot = seq_values[indices].mean(axis=1)
    action_boot = action_values[indices].mean(axis=1)
    quantiles = lambda values: [float(v) for v in np.quantile(values, [0.025, 0.5, 0.975])]
    return {
        "layout_count": len(rows),
        "sequence_regret_macro_mean": float(seq_values.mean()),
        "sequence_regret_layout_q25_median_q75": [float(v) for v in np.quantile(seq_values, [0.25, 0.5, 0.75])],
        "teacher_action_group_regret_macro_mean": float(action_values.mean()),
        "teacher_action_group_regret_layout_q25_median_q75": [float(v) for v in np.quantile(action_values, [0.25, 0.5, 0.75])],
        "bootstrap_95pct_interval_sequence_regret_macro": quantiles(seq_boot),
        "bootstrap_95pct_interval_teacher_action_group_regret_macro": quantiles(action_boot),
        "highest_teacher_action_group_regret_layouts": sorted(rows, key=lambda row: row["teacher_action_group_normalized_regret"], reverse=True)[:5],
    }


def _compact_action(result: dict[str, object], layout: np.ndarray) -> dict[str, object]:
    seq_r = np.asarray(result["per_query_sequence_regret"])
    act_r = np.asarray(result["per_query_teacher_action_group_regret"])
    actions = np.asarray(result["per_query_student_first_action"])
    teacher_actions = np.asarray(result["per_query_teacher_first_action"])
    return {
        key: value for key, value in result.items()
        if not key.startswith("per_query_")
    } | {
        "macro_layout_sequence_regret": float(np.mean([seq_r[layout == i].mean() for i in np.unique(layout)])),
        "macro_layout_teacher_action_group_normalized_regret": float(np.mean([act_r[layout == i].mean() for i in np.unique(layout)])),
        "student_first_action_histogram": {str(int(a)): int(np.sum(actions == a)) for a in range(9)},
        "teacher_first_action_histogram": {str(int(a)): int(np.sum(teacher_actions == a)) for a in range(9)},
        "majority_teacher_action_sample_agreement": float(max(np.mean(teacher_actions == a) for a in range(9))),
        "per_layout_summary": _per_layout(seq_r, act_r, layout),
    }


def main() -> int:
    run_dir = ROOT / "artifacts/runs/phase12_training_v1"
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    selected = results["selected"]
    model, payload = load_student(Path(selected["checkpoint"]), device="cpu")
    reports: dict[str, object] = {}
    for name, folder in (
        ("train", "phase12_train_teacher_v1"),
        ("validation", "phase12_validation_teacher_v1"),
    ):
        data_dir = ROOT / "artifacts/runs" / folder
        data = load_phase12_split(data_dir)
        offline = evaluate_offline(model, data, kind=payload["kind"], device=torch.device("cpu"))
        action = summarize_first_action_decision(
            data["sequences"], data["candidate_mask"], data["target_normalized_regret"],
            data["target_optimal_mask"], data["target_objective_key"],
            data["target_teacher_selected_index"], offline["selected_indices"],
        )
        reports[name] = {
            "input_sha256": sha(data_dir / "student_inputs.npz"),
            "targets_sha256": sha(data_dir / "teacher_targets.npz"),
            "metadata_sha256": sha(data_dir / "sample_metadata.npz"),
            "target_structure": _dataset_summary(data),
            "frozen_sequence_metrics": {
                "layout_macro_mean_selected_normalized_rank_regret": offline["layout_macro_mean_selected_normalized_rank_regret"],
                "sample_mean_selected_normalized_rank_regret": offline["sample_mean_selected_normalized_rank_regret"],
                "tie_aware_pairwise_rank_accuracy": offline["tie_aware_pairwise_rank_accuracy"],
                "optimal_set_top1_agreement": offline["optimal_set_top1_agreement"],
            },
            "posthoc_first_action_analysis": _compact_action(action, data["meta_layout_index"]),
        }
    report = {
        "schema": "phase12_postmortem_diagnostic_v1",
        "status": "diagnosis_only_no_refit_no_new_data_no_gate_change",
        "generated_utc_date": "2026-09-28",
        "question": "Does sequence-level ordinal regret measure errors in the first action actually executed before replanning?",
        "selected_variant": selected["variant"],
        "fit_seed": selected["fit_seed"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "training_results_sha256": sha(run_dir / "results.json"),
        "diagnostic_script_sha256": sha(Path(__file__)),
        "diagnostic_module_sha256": sha(ROOT / "src/consequence_lab/phase12_postmortem.py"),
        "training_config_sha256": sha(ROOT / "configs/phase12_teacher_student.yaml"),
        "teacher_inputs_are_unchanged": True,
        "reserve_accessed": False,
        "closed_loop_or_dagger_run": False,
        "datasets": reports,
    }
    output = ROOT / "reports/phase12_postmortem_diagnostic.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
