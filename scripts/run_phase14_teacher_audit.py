"""Run only the frozen 32-state Phase14A first-action teacher audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import yaml

from consequence_lab.phase14 import (
    Phase14IntegrityError,
    add_state_ranks,
    diagnostic_state_selection,
    evaluate_forced_action,
    phase14_python_environment,
    replay_training_states,
    sha256_file,
    summarize_audit,
    validate_source_lock,
)


ROOT = Path(__file__).resolve().parents[1]


def _comparison_payload(row: dict) -> tuple:
    return (
        row["terminal_status"], row["terminal_steps_from_source"],
        row["terminal_goal_distance"], row["total_noncoast_effort"],
        tuple(row["objective_tuple"]), row["continuation_decision_count"],
        tuple(row["continuation_action_trace"]), tuple(row["forced_successor"]),
        row["forced_collision"], row["forced_success"], row["forced_timeout"],
        tuple(row["terminal_state"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase14.yaml")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config_path != root / "configs/phase14.yaml":
        raise SystemExit("Phase14 config must be the frozen repository config")
    audit_path = root / config["provenance"]["source_audit"]
    lock_path = root / config["provenance"]["source_lock"]
    output_dir = root / config["provenance"]["audit_output"]
    output_path = root / config["provenance"]["diagnostic_output"]
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite Phase14A output: {output_dir}")
    lock, source_audit = validate_source_lock(root, lock_path, audit_path)
    if lock.get("authorized_stage") != "phase14A_32_state_teacher_audit_only":
        raise Phase14IntegrityError("source lock does not authorize Phase14A diagnostic only")
    all_decisions = replay_training_states(root)
    selected, _ = diagnostic_state_selection(all_decisions)
    if len(selected) != 32:
        raise Phase14IntegrityError("diagnostic selection did not yield 32 states")
    expected_keys = source_audit["diagnostic_state_keys"]
    observed_keys = [
        {"episode_seed": row.episode_seed, "layout_sha256": row.layout_sha256,
         "decision_index": row.decision_index, "global_query_index": row.global_query_index}
        for row in selected
    ]
    if observed_keys != expected_keys:
        raise Phase14IntegrityError("selected diagnostic states differ from audited state keys")

    wall_cap = int(config["resources"]["max_wall_seconds_per_stage"])
    byte_cap = int(config["resources"]["max_generated_bytes_per_stage"])
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=False)
    state_rows: list[list[dict]] = []
    repeat_count = 0
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            for state_index, source in enumerate(selected):
                rows = [evaluate_forced_action(source, action_id) for action_id in range(9)]
                if state_index < 8:
                    repeat = [evaluate_forced_action(source, action_id) for action_id in range(9)]
                    for first, second in zip(rows, repeat, strict=True):
                        if _comparison_payload(first) != _comparison_payload(second):
                            raise Phase14IntegrityError(
                                f"counterfactual replay mismatch state={state_index} action={first['forced_action_id']}"
                            )
                    repeat_count += 1
                ranked = add_state_ranks(rows)
                record = {
                    "state_index": state_index,
                    "source_episode_seed": source.episode_seed,
                    "source_layout_sha256": source.layout_sha256,
                    "source_decision_index": source.decision_index,
                    "student_input_52": list(source.features),
                    "repeat_exact_passed": state_index < 8,
                    "actions": ranked,
                }
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                stream.flush()
                state_rows.append(ranked)
                elapsed = time.monotonic() - started
                current_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
                if elapsed > wall_cap:
                    raise TimeoutError("Phase14A wall-time cap exceeded")
                if current_bytes > byte_cap:
                    raise OSError("Phase14A generated-artifact cap exceeded")
                print(f"Phase14A diagnostic state {state_index + 1}/32 complete; elapsed={elapsed:.1f}s; bytes={current_bytes}", flush=True)
        metrics = summarize_audit(state_rows, repeat_count)
        gate_repeat = repeat_count == 8
        gate_pass = gate_repeat and bool(metrics["informativeness_gate_passed"])
        manifest = {
            "schema": "phase14a_teacher_audit_v2",
            "status": "completed_gate_passed" if gate_pass else "completed_gate_failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": lock["source_commit"],
            "source_lock_sha256": sha256_file(lock_path),
            "source_audit_sha256": sha256_file(audit_path),
            "phase12_training_archive_sha256": source_audit["training_archive_sha256"],
            "output_jsonl_sha256": sha256_file(output_path),
            "output_jsonl_bytes": output_path.stat().st_size,
            "wall_seconds": time.monotonic() - started,
            "python_environment": phase14_python_environment(),
            "reserve_accessed": False,
            "phase12_validation_opened": False,
            "model_training": False,
            "metric_summary": metrics,
            "repeatability_gate_passed": gate_repeat,
            "informativeness_gate_passed": bool(metrics["informativeness_gate_passed"]),
            "phase14a_gate_passed": gate_pass,
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        total_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
        if total_bytes > byte_cap:
            raise OSError("Phase14A total artifact cap exceeded")
        print(json.dumps({
            "status": manifest["status"], "state_count": metrics["state_count"],
            "nonzero_regret_fraction": metrics["nonzero_regret_state_fraction"],
            "optimal_action_ids": metrics["distinct_optimal_action_ids"],
            "coast_optimal_fraction": metrics["optimal_action_state_fraction"][0],
            "coast_mean_regret": metrics["mean_normalized_regret_by_action"][0],
            "wall_seconds": manifest["wall_seconds"], "bytes": total_bytes,
        }, sort_keys=True))
        if not gate_pass:
            raise SystemExit("Phase14A teacher audit gate failed; stop before full labels/training")
    except BaseException as exc:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            partial = {
                "schema": "phase14a_teacher_audit_v2",
                "status": "aborted_or_incomplete",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source_commit": lock["source_commit"],
                "source_lock_sha256": sha256_file(lock_path),
                "completed_state_rows": len(state_rows),
                "reserve_accessed": False,
                "model_training": False,
                "wall_seconds": time.monotonic() - started,
            }
            manifest_path.write_text(json.dumps(partial, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
