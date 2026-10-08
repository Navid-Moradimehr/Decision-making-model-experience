"""Post-hoc, non-selective audit of unequal action-group proposal counts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_training import load_phase12_split
from consequence_lab.phase12b import derive_action_group_targets

SPLITS = {
    "model_selection": ROOT / "artifacts" / "runs" / "phase12b_model_selection_v1",
    "offline_gate": ROOT / "artifacts" / "runs" / "phase12b_offline_gate_v1",
}
OUTPUT_JSON = ROOT / "reports" / "phase12b_proposal_count_bias.json"
OUTPUT_MD = ROOT / "reports" / "phase12b_proposal_count_bias.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def analyze(root: Path) -> dict:
    data = load_phase12_split(root)
    mask = data["candidate_mask"].astype(bool)
    sequences = data["sequences"]
    targets = derive_action_group_targets(sequences, mask, data["target_objective_key"])
    counts = np.stack([
        ((sequences[:, :, 0] == action) & mask).sum(axis=1)
        for action in range(9)
    ], axis=1)
    regret = targets["action_group_regret"]
    optimal = targets["optimal_action_mask"]
    count_regret_correlation = float(np.corrcoef(counts.ravel(), regret.ravel())[0, 1])
    return {
        "layouts": int(len(np.unique(data["meta_layout_index"]))),
        "queries": int(len(sequences)),
        "action_groups": 9,
        "candidate_count_min": int(counts.min()),
        "candidate_count_max": int(counts.max()),
        "candidate_count_mean": float(counts.mean()),
        "candidate_count_std_population": float(counts.std()),
        "queries_with_unequal_action_group_counts": int(np.any(counts != counts[:, :1], axis=1).sum()),
        "per_action_mean_counts": counts.mean(axis=0).tolist(),
        "pooled_group_count_vs_normalized_rank_regret_pearson": count_regret_correlation,
        "optimal_group_mean_count": float(counts[optimal].mean()),
        "nonoptimal_group_mean_count": float(counts[~optimal].mean()),
        "optimal_group_fraction": float(optimal.mean()),
        "per_action_optimal_frequency": optimal.mean(axis=0).tolist(),
        "interpretation": "descriptive pooled association only; action groups within a query are dependent and this analysis does not establish causal bias or explain student failure",
    }


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise SystemExit("refusing to overwrite proposal-count diagnostic outputs")
    rows = {name: analyze(path) for name, path in SPLITS.items()}
    data_hashes = {}
    for name, root in SPLITS.items():
        data_hashes[name] = {
            filename: digest(root / filename)
            for filename in ("student_inputs.npz", "sample_metadata.npz", "teacher_targets.npz")
        }
    report = {
        "schema": "phase12b_proposal_count_bias_diagnostic_v1",
        "status": "post_hoc_descriptive_not_used_for_model_selection_or_gate",
        "analysis_script_sha256": digest(Path(__file__).resolve()),
        "phase12b_source_sha256": digest(ROOT / "src" / "consequence_lab" / "phase12b.py"),
        "data_sha256": data_hashes,
        "splits": rows,
        "closed_loop_development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
    }
    OUTPUT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selection, gate = rows["model_selection"], rows["offline_gate"]
    OUTPUT_MD.write_text(
        "---\ntype: diagnostic-report\nstatus: completed-posthoc-descriptive\nupdated: 2026-09-28\n---\n\n"
        "# Phase 12b proposal-count bias diagnostic\n\n"
        "This analysis was run after the frozen offline gate. It did not change model selection, the gate, or the stop decision. "
        "The action-group target takes a minimum objective within each first-action group, but the random sequence bank gives groups unequal numbers of suffix proposals.\n\n"
        "| Split | Queries | Per-action proposals (min–max; mean ± SD) | Corr(count, rank regret) | Mean count: optimal vs nonoptimal |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {row['queries']} | {row['candidate_count_min']}–{row['candidate_count_max']}; "
            f"{row['candidate_count_mean']:.2f} ± {row['candidate_count_std_population']:.2f} | "
            f"{row['pooled_group_count_vs_normalized_rank_regret_pearson']:.3f} | "
            f"{row['optimal_group_mean_count']:.2f} vs {row['nonoptimal_group_mean_count']:.2f} |"
            for name, row in rows.items()
        )
        + "\n\nThe negative pooled correlations are consistent with an extreme-value/proposal-count effect: groups receiving more sampled suffixes have more opportunities to contain a low-cost plan. The analysis is descriptive; action groups within a query are dependent, and it does not show that count imbalance caused the student's poor generalization. A future protocol should consider balanced per-action continuation proposals or a count-corrected estimator, compared on fresh layouts. This does not reopen Phase 12b.\n\n"
        "The full metrics and file hashes are in `phase12b_proposal_count_bias.json`. No development, DAgger, or reserve data were accessed.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "splits": rows, "reserve_accessed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
