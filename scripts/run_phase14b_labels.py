"""Generate only exact first-action values for the permitted training split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import yaml

from consequence_lab.phase14 import (
    Phase14IntegrityError,
    phase14_python_environment,
    replay_training_states,
    sha256_file,
    summarize_audit,
)
from consequence_lab.phase14b import (
    PHASE14B_SOURCE_CODE,
    Phase14BLockError,
    _read_phase14a,
    make_compact_state_record,
    projected_full_label_seconds,
    validate_training_lock,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/phase14b.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / CONFIG_PATH)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    if config_path != root / CONFIG_PATH:
        raise SystemExit("Phase14B config must be the frozen repository config")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = root / config["outputs"]["run_dir"]
    output_path = root / config["outputs"]["records"]
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing Phase14B labels: {output_dir}")
    lock = validate_training_lock(root, config)
    phase14a_manifest, _, _ = _read_phase14a(root, config)
    decisions = replay_training_states(root)
    if len(decisions) != int(config["source"]["expected_states"]):
        raise Phase14IntegrityError("replayed full training state count changed")
    estimate = projected_full_label_seconds(phase14a_manifest, len(decisions))
    if estimate != float(lock["full_label_wall_projection_seconds"]):
        raise Phase14BLockError("full-label runtime projection differs from source lock")

    wall_cap = int(config["resources"]["wall_seconds"])
    byte_cap = int(config["resources"]["generated_bytes"])
    if estimate >= wall_cap:
        raise Phase14BLockError("measured full-label projection no longer fits the frozen wall cap")
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    state_targets: list[list[dict]] = []
    completed = 0
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            for source in decisions:
                record, compact_actions = make_compact_state_record(source)
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                state_targets.append(compact_actions)
                completed += 1
                elapsed = time.monotonic() - started
                size = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
                if elapsed > wall_cap:
                    raise TimeoutError("Phase14B full-label wall-time cap exceeded")
                if size > byte_cap:
                    raise OSError("Phase14B generated-artifact cap exceeded")
                if completed % 25 == 0 or completed == len(decisions):
                    print(
                        f"Phase14B labels {completed}/{len(decisions)}; "
                        f"elapsed={elapsed:.1f}s; bytes={size}", flush=True,
                    )
        metrics = summarize_audit(state_targets, repeated_count=0)
        gate_pass = bool(metrics["informativeness_gate_passed"])
        manifest = {
            "schema": "phase14b_full_training_values_v1",
            "status": "completed_teacher_values_gate_passed" if gate_pass else "completed_teacher_values_gate_failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": lock["source_commit"],
            "phase14b_lock_sha256": sha256_file(root / config["outputs"]["lock"]),
            "phase14a_manifest_sha256": lock["phase14a_manifest_sha256"],
            "phase14a_records_sha256": lock["phase14a_records_sha256"],
            "phase14a_source_lock_sha256": lock["phase14a_lock_sha256"],
            "phase12_training_archive_sha256": lock["phase12_training_archive_sha256"],
            "phase14b_source_code_sha256": lock["source_code_sha256"],
            "records_sha256": sha256_file(output_path),
            "records_bytes": output_path.stat().st_size,
            "wall_seconds": time.monotonic() - started,
            "full_label_wall_projection_seconds": estimate,
            "python_environment": phase14_python_environment(),
            "source_layouts": 120,
            "source_decision_states": len(decisions),
            "forced_action_records": len(decisions) * 9,
            "student_training": False,
            "heldout_access": False,
            "dagger": False,
            "world_model": False,
            "reserve_accessed": False,
            "phase12_validation_opened": False,
            "metric_summary": metrics,
            "informativeness_gate_passed_on_full_train": gate_pass,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        total = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
        if total > byte_cap:
            raise OSError("Phase14B total output cap exceeded")
        print(json.dumps({
            "status": manifest["status"], "states": len(decisions),
            "forced_action_records": len(decisions) * 9,
            "nonzero_regret_fraction": metrics["nonzero_regret_state_fraction"],
            "optimal_action_ids": metrics["distinct_optimal_action_ids"],
            "coast_optimal_fraction": metrics["optimal_action_state_fraction"][0],
            "coast_mean_regret": metrics["mean_normalized_regret_by_action"][0],
            "wall_seconds": manifest["wall_seconds"], "bytes": total,
        }, sort_keys=True))
        if not gate_pass:
            raise SystemExit("full training teacher values failed informativeness; stop before training")
    except BaseException as exc:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            partial = {
                "schema": "phase14b_full_training_values_v1",
                "status": "aborted_or_incomplete",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source_commit": lock["source_commit"],
                "phase14b_lock_sha256": sha256_file(root / config["outputs"]["lock"]),
                "completed_state_rows": completed,
                "student_training": False,
                "heldout_access": False,
                "dagger": False,
                "world_model": False,
                "reserve_accessed": False,
                "wall_seconds": time.monotonic() - started,
            }
            manifest_path.write_text(json.dumps(partial, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
