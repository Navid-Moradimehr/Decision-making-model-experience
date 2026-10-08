"""Generate only the missing train-only Phase14B targets and verify the full merge."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

from consequence_lab.phase14 import phase14_python_environment, sha256_file, summarize_audit
from consequence_lab.phase14b_resume import (
    CONFIG_PATH,
    Phase14BResumeError,
    iter_remaining_rows,
    read_config,
    read_jsonl,
    validate_continuation_lock,
    validate_record_range,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_config(root)
    if (root / CONFIG_PATH).resolve() != root / "configs/phase14b_resume.yaml":
        raise SystemExit("continuation config path differs from frozen config")
    output_dir = root / config["outputs"]["run_dir"]
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite continuation output: {output_dir}")

    lock, sources, prefix = validate_continuation_lock(root, config)
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    suffix_path = root / config["outputs"]["remaining_records"]
    merged_path = root / config["outputs"]["merged_records"]
    manifest_path = root / config["outputs"]["manifest"]
    generated_count = 0
    try:
        suffix_rows: list[dict] = []
        with suffix_path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in iter_remaining_rows(sources, len(prefix)):
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                suffix_rows.append(record)
                generated_count += 1
                if generated_count % 10 == 0 or generated_count == len(sources) - len(prefix):
                    print(
                        f"Phase14B continuation {generated_count}/{len(sources) - len(prefix)}; "
                        f"total_states={len(prefix) + generated_count}", flush=True,
                    )
        suffix = suffix_rows

        combined = prefix + suffix
        validate_record_range(combined, sources, 0, len(sources))
        expected_rows = int(config["source"]["total_training_rows"])
        expected_actions = expected_rows * int(config["source"]["actions_per_state"])
        action_count = sum(len(row["actions"]) for row in combined)
        if len(combined) != expected_rows or action_count != expected_actions:
            raise Phase14BResumeError("complete training label cardinality mismatch")

        with merged_path.open("xb") as merged:
            merged.write((root / config["source"]["partial_records"]).read_bytes())
            merged.write(suffix_path.read_bytes())
            merged.flush()
        merged_rows = read_jsonl(merged_path)
        merged_targets = validate_record_range(merged_rows, sources, 0, len(sources))
        if merged_rows != combined:
            raise Phase14BResumeError("serialized merged file differs from validated prefix+suffix")

        metrics = summarize_audit(merged_targets, repeated_count=0)
        gate_pass = bool(metrics["informativeness_gate_passed"])
        manifest = {
            "schema": "phase14b_completed_training_values_v2",
            "status": "completed_teacher_values_gate_passed" if gate_pass else "completed_teacher_values_gate_failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": lock["source_commit"],
            "continuation_lock_sha256": sha256_file(root / config["outputs"]["lock"]),
            "base_phase14b_lock_sha256": lock["base_phase14b_lock_sha256"],
            "base_phase14b_manifest_sha256": lock["failed_partial_manifest_sha256"],
            "base_phase14b_partial_records_sha256": lock["failed_partial_records_sha256"],
            "phase12_training_seed_range": config["source"]["train_seed_range"],
            "source_decision_states": len(merged_rows),
            "prefix_state_rows_reused": len(prefix),
            "continuation_state_rows_generated": generated_count,
            "forced_action_records": action_count,
            "records_sha256": sha256_file(merged_path),
            "records_bytes": merged_path.stat().st_size,
            "continuation_records_sha256": sha256_file(suffix_path),
            "continuation_records_bytes": suffix_path.stat().st_size,
            "wall_seconds": time.monotonic() - started,
            "python_environment": phase14_python_environment(),
            "student_training": False,
            "heldout_access": False,
            "phase12_validation_opened": False,
            "dagger": False,
            "world_model": False,
            "reserve_accessed": False,
            "integrity_passed": True,
            "informativeness_gate_passed_on_full_train": gate_pass,
            "metric_summary": metrics,
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": manifest["status"],
            "states": len(merged_rows),
            "continuation_states": generated_count,
            "forced_action_records": action_count,
            "nonzero_regret_fraction": metrics["nonzero_regret_state_fraction"],
            "optimal_action_ids": metrics["distinct_optimal_action_ids"],
            "coast_optimal_fraction": metrics["optimal_action_state_fraction"][0],
            "coast_mean_regret": metrics["mean_normalized_regret_by_action"][0],
            "wall_seconds": manifest["wall_seconds"],
            "records_sha256": manifest["records_sha256"],
        }, sort_keys=True))
        if not gate_pass:
            raise SystemExit("full training teacher values failed informativeness; stop before student fitting")
    except BaseException as exc:
        if not manifest_path.exists():
            partial = {
                "schema": "phase14b_completed_training_values_v2",
                "status": "aborted_or_incomplete",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source_commit": lock["source_commit"],
                "continuation_lock_sha256": sha256_file(root / config["outputs"]["lock"]),
                "completed_continuation_state_rows": generated_count,
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
