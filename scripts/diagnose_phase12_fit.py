"""Post-selection train/validation fit-gap diagnostic; never retrains or changes choice."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_evaluation import load_student
from consequence_lab.phase12_training import evaluate_offline, load_phase12_split


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    results_path = ROOT / "artifacts" / "runs" / "phase12_training_v1" / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selected = results["selected"]
    model, payload = load_student(Path(selected["checkpoint"]), device="cpu")
    train = load_phase12_split(ROOT / "artifacts" / "runs" / "phase12_train_teacher_v1")
    val = load_phase12_split(ROOT / "artifacts" / "runs" / "phase12_validation_teacher_v1")
    train_metrics = evaluate_offline(model, train, kind=payload["kind"], device=torch.device("cpu"))
    val_metrics = evaluate_offline(model, val, kind=payload["kind"], device=torch.device("cpu"))
    keep = {
        "layout_macro_mean_selected_normalized_rank_regret",
        "sample_mean_selected_normalized_rank_regret",
        "expected_normalized_rank_regret",
        "optimal_set_top1_agreement",
        "tie_aware_pairwise_rank_accuracy",
        "kendall_tau_b",
    }
    report = {
        "schema": "phase12_selected_student_fit_gap_v1",
        "status": "post_selection_diagnostic_only_not_used_for_model_selection",
        "variant": selected["variant"],
        "fit_seed": selected["fit_seed"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "train_input_sha256": sha(ROOT / "artifacts" / "runs" / "phase12_train_teacher_v1" / "student_inputs.npz"),
        "validation_input_sha256": sha(ROOT / "artifacts" / "runs" / "phase12_validation_teacher_v1" / "student_inputs.npz"),
        "train": {key: value for key, value in train_metrics.items() if key in keep},
        "validation": {key: value for key, value in val_metrics.items() if key in keep},
        "reserve_accessed": False,
        "closed_loop_evaluation_performed": False,
    }
    output = ROOT / "reports" / "phase12_fit_diagnostic.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
