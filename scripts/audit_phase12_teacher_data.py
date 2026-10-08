"""Audit Phase-12 teacher datasets and enforce the runtime information firewall."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit_split(split_name: str, expected_start: int, expected_count: int) -> dict:
    split_dir = ROOT / "artifacts" / "runs" / f"phase12_{split_name}_v1"
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["reserve_accessed"] or manifest["training_performed"]:
        raise ValueError(f"unexpected reserve/training flag in {split_name} manifest")
    if manifest["episode_count"] != expected_count:
        raise ValueError(f"wrong episode count for {split_name}")
    expected_seeds = set(range(expected_start, expected_start + expected_count))
    rows = manifest["episodes"]
    if {row["seed"] for row in rows} != expected_seeds:
        raise ValueError(f"seed mismatch for {split_name}")
    layout_hashes = [row["layout_sha256"] for row in rows]
    if len(set(layout_hashes)) != expected_count or layout_hashes != manifest["layout_sha256"]:
        raise ValueError(f"layout hash mismatch for {split_name}")
    with np.load(split_dir / "student_inputs.npz", allow_pickle=False) as z:
        if set(z.files) != {"states", "sequences", "candidate_mask"}:
            raise ValueError(f"student input firewall violation in {split_name}")
        states, sequences, masks = z["states"], z["sequences"], z["candidate_mask"]
    with np.load(split_dir / "teacher_targets.npz", allow_pickle=False) as z:
        targets = {key: z[key] for key in z.files}
    with np.load(split_dir / "sample_metadata.npz", allow_pickle=False) as z:
        metadata = {key: z[key] for key in z.files}
    with np.load(split_dir / "teacher_rollouts.npz", allow_pickle=False) as z:
        rollouts = {key: z[key] for key in z.files}
    count = len(states)
    if states.shape != (count, 52) or not np.isfinite(states).all():
        raise ValueError(f"invalid student state features in {split_name}")
    if sequences.shape != (count, 128, 8) or masks.shape != (count, 128):
        raise ValueError(f"invalid candidate bank shape in {split_name}")
    if not masks.all() or np.any(sequences > 8):
        raise ValueError(f"unexpected padded candidate in full-horizon {split_name}")
    if targets["normalized_regret"].shape != (count, 128):
        raise ValueError(f"invalid regret targets in {split_name}")
    if not np.isfinite(targets["normalized_regret"]).all() or np.any(targets["normalized_regret"] < 0) or np.any(targets["normalized_regret"] > 1):
        raise ValueError(f"rank regret outside [0,1] in {split_name}")
    chosen = targets["teacher_selected_index"].astype(int)
    rows_idx = np.arange(count)
    if not targets["optimal_mask"][rows_idx, chosen].all():
        raise ValueError(f"historical selected candidate is not an exact optimum in {split_name}")
    if not np.allclose(targets["normalized_regret"].min(axis=1), 0.0):
        raise ValueError(f"candidate set does not retain zero-regret teacher alternatives in {split_name}")
    if metadata["episode_seed"].shape != (count,) or metadata["layout_index"].shape != (count,):
        raise ValueError(f"metadata shape mismatch in {split_name}")
    if rollouts["states"].shape != (count, 128, 9, 4) or rollouts["state_mask"].shape != (count, 128, 9):
        raise ValueError(f"teacher rollout shape mismatch in {split_name}")
    hashes = {}
    for filename in ("student_inputs.npz", "sample_metadata.npz", "teacher_targets.npz", "teacher_rollouts.npz", "episodes.json"):
        path = split_dir / filename
        hashes[filename] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    for key, filename in (
        ("student_inputs", "student_inputs.npz"),
        ("sample_metadata", "sample_metadata.npz"),
        ("teacher_targets", "teacher_targets.npz"),
        ("teacher_rollouts", "teacher_rollouts.npz"),
        ("episodes", "episodes.json"),
    ):
        if manifest["files"][key]["sha256"] != hashes[filename]["sha256"]:
            raise ValueError(f"artifact hash mismatch in {split_name}: {filename}")
    for index, row in enumerate(rows):
        sample_mask = metadata["layout_index"] == index
        if not sample_mask.any() or not np.all(metadata["episode_seed"][sample_mask] == row["seed"]):
            raise ValueError(f"query-to-layout provenance mismatch in {split_name}")
    return {
        "split": split_name,
        "episodes": expected_count,
        "decision_queries": count,
        "layouts_unique": len(set(layout_hashes)),
        "teacher_success_rate": sum(row["outcome"] == "success" for row in rows) / len(rows),
        "candidate_success_availability_fraction": manifest["candidate_availability_fraction"],
        "student_input_allowlist_exact": True,
        "teacher_labels_separate": True,
        "teacher_rollouts_separate": True,
        "files": hashes,
        "artifact_bytes": sum(item["bytes"] for item in hashes.values()),
    }


def main() -> int:
    layout_path = ROOT / "artifacts" / "runs" / "phase12_layout_audit.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if layout["reserve_generated_or_accessed"] or not layout["pairwise_disjoint"]:
        raise ValueError("Phase 12 split layout audit failed or touched reserve")
    train = audit_split("train_teacher", 140000, 120)
    validation = audit_split("validation_teacher", 141000, 40)
    train_hashes = set(layout["blocks"]["train_teacher"]["layout_sha256"])
    val_hashes = set(layout["blocks"]["validation_teacher"]["layout_sha256"])
    if train_hashes & val_hashes:
        raise ValueError("train and validation layouts overlap")
    report = {
        "schema": "phase12_teacher_data_audit_v1",
        "status": "passed",
        "layout_audit_sha256": sha256(layout_path),
        "train": train,
        "validation": validation,
        "train_validation_layouts_disjoint": True,
        "reserve_accessed": False,
        "student_information_firewall_passed": True,
    }
    out = ROOT / "reports" / "phase12_teacher_data_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
