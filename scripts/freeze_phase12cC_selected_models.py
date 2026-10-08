"""Pin both chosen model-family checkpoints before any offline-gate labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--fit-output", default="artifacts/runs/phase12cC_selected_models_v1")
    parser.add_argument("--training-lock", default="experiments/phase12cC_training.lock.json")
    parser.add_argument("--output", default="experiments/phase12cC_selected_models.lock.json")
    args = parser.parse_args()
    if args.confirmation != "FREEZE_PHASE12C_C_SELECTED_MODELS":
        raise SystemExit("checkpoint lock requires --confirmation FREEZE_PHASE12C_C_SELECTED_MODELS")
    fit_root = ROOT / args.fit_output
    fit_manifest_path = fit_root / "fit_manifest.json"
    results_path = fit_root / "selection_results.json"
    if not fit_manifest_path.is_file() or not results_path.is_file():
        raise SystemExit("Phase 12c-C fit artifacts are incomplete")
    fit_manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    training_lock_path = ROOT / args.training_lock
    training_lock = json.loads(training_lock_path.read_text(encoding="utf-8"))
    if fit_manifest.get("offline_gate_read") is not False or fit_manifest.get("reserve_accessed") is not False:
        raise SystemExit("fit manifest indicates forbidden split access")
    if fit_manifest.get("selection_results_sha256") != digest(results_path):
        raise SystemExit("selected-model results hash differs from fit manifest")
    if fit_manifest.get("training_lock_sha256") != digest(training_lock_path):
        raise SystemExit("fit was not conducted under the Phase 12c-C training lock")
    if set(results.get("selected_models", {})) != {"direct_candidate_value", "bounded_preference_policy"}:
        raise SystemExit("both predeclared model families must be selected")
    checkpoints = {}
    for relative, expected in results.get("checkpoint_sha256", {}).items():
        path = (fit_root / relative).resolve()
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"selected checkpoint hash mismatch: {relative}")
        checkpoints[str(path.relative_to(ROOT))] = digest(path)
    if len(checkpoints) != 2:
        raise SystemExit("exactly two family checkpoints must be pinned")

    lock = {
        "schema": "phase12cC_selected_models_lock_v1",
        "phase": "12c-C",
        "training_lock_sha256": digest(training_lock_path),
        "fit_manifest_sha256": digest(fit_manifest_path),
        "selection_results_sha256": digest(results_path),
        "training_data_manifest_sha256": fit_manifest["training_manifest_sha256"],
        "model_selection_manifest_sha256": fit_manifest["selection_manifest_sha256"],
        "selected_models": results["selected_models"],
        "checkpoint_sha256": checkpoints,
        "offline_gate_labels_collected": False,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
        "gate_stage_authorized_after_this_lock": True,
    }
    output_path = (ROOT / args.output).resolve()
    if ROOT / "artifacts/runs/phase12cC_offline_gate_v1" in output_path.parents:
        raise SystemExit("selected checkpoint lock must not be stored inside evaluation artifacts")
    if output_path.exists() or ROOT / "experiments" not in output_path.parents:
        raise SystemExit("selected lock must be a new file under experiments/")
    output_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "phase12cC_selected_models_locked_before_gate",
        "lock": str(output_path),
        "lock_sha256": digest(output_path),
        "checkpoint_sha256": checkpoints,
        "offline_gate_labels_collected": False,
        "reserve_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
