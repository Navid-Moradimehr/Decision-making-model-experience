"""Create a source/checkpoint-pinned development lock only after offline gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    results_path = ROOT / "artifacts" / "runs" / "phase12_training_v1" / "results.json"
    layout_path = ROOT / "artifacts" / "runs" / "phase12_layout_audit.json"
    data_audit = ROOT / "reports" / "phase12_teacher_data_audit.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not results["offline_competence_gate"]:
        print(json.dumps({"status": "closed_loop_blocked_offline_competence_gate_failed", "reserve_accessed": False}, indent=2))
        return 2
    selected = results["selected"]
    checkpoint = Path(selected["checkpoint"])
    if not checkpoint.is_file() or sha(checkpoint) != selected["checkpoint_sha256"]:
        raise SystemExit("selected model checkpoint hash mismatch")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    dev = layout["blocks"]["closed_loop_development"]
    lock = {
        "schema": "phase12_development_evaluation_lock_v1",
        "stage": "closed_loop_development_only",
        "authorization_reference": "Development block execution was included in the user's Phase 12 execute-through-gates request; final reserve excluded.",
        "config_sha256": sha(ROOT / "configs" / "phase12_teacher_student.yaml"),
        "training_lock_sha256": sha(ROOT / "experiments" / "phase12_training.lock.json"),
        "training_results_sha256": sha(results_path),
        "teacher_data_audit_sha256": sha(data_audit),
        "layout_manifest_sha256": sha(layout_path),
        "selected_checkpoint": str(checkpoint.resolve()),
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_variant": selected["variant"],
        "selected_fit_seed": selected["fit_seed"],
        "selected_validation_regret": selected["validation"]["layout_macro_mean_selected_normalized_rank_regret"],
        "development_seeds": {"start": 143000, "count": 40},
        "development_layout_sha256": dev["layout_sha256"],
        "reserve_accessed": False,
        "training_performed": False,
        "maximum_bootstrap_resamples": 10000,
        "bootstrap_seed_success": 143011,
        "bootstrap_seed_collision": 143012,
        "evaluation_source_sha256": {
            "src/consequence_lab/phase12_evaluation.py": sha(ROOT / "src" / "consequence_lab" / "phase12_evaluation.py"),
            "scripts/run_phase12_development.py": sha(ROOT / "scripts" / "run_phase12_development.py"),
            "scripts/freeze_phase12_development.py": sha(Path(__file__).resolve()),
        },
    }
    destination = ROOT / "experiments" / "phase12_development.lock.json"
    if destination.exists():
        raise SystemExit("development lock already exists; refusing to overwrite")
    destination.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "development_lock_frozen", "output": str(destination), **lock}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
