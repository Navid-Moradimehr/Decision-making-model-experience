"""Collect/evaluate the sealed Phase-12b offline gate after model pinning."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_data import collect_teacher_split
from consequence_lab.phase12_model import CorridorCandidateScorer
from consequence_lab.phase12_training import load_phase12_split
from consequence_lab.phase12b import evaluate_phase12b

CONFIG = ROOT / "configs" / "phase12b_action_aligned.yaml"
COLLECTION_LOCK = ROOT / "experiments" / "phase12b_offline_gate_collection.lock.json"
EVALUATION_LOCK = ROOT / "experiments" / "phase12b_offline_gate_evaluation.lock.json"
SELECTED_LOCK = ROOT / "experiments" / "phase12b_selected_model.lock.json"
AUDIT = ROOT / "artifacts" / "runs" / "phase12b_layout_audit_v2.json"
DATA_DIR = ROOT / "artifacts" / "runs" / "phase12b_offline_gate_v1"
REPORT = ROOT / "reports" / "phase12b_offline_gate.md"
REPORT_JSON = ROOT / "reports" / "phase12b_offline_gate.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_collection_lock() -> tuple[dict, dict]:
    if not COLLECTION_LOCK.is_file() or not SELECTED_LOCK.is_file():
        raise SystemExit("offline-gate collection lock and selected-model lock are required")
    lock = json.loads(COLLECTION_LOCK.read_text(encoding="utf-8"))
    selected_lock = json.loads(SELECTED_LOCK.read_text(encoding="utf-8"))
    if lock.get("gate_authorized") is not True or lock.get("reserve_accessed") is not False:
        raise SystemExit("invalid offline-gate authorization lock")
    if selected_lock.get("offline_gate_collection_authorized") is not True:
        raise SystemExit("selected checkpoint was not frozen for offline-gate collection")
    if selected_lock.get("selected_checkpoint_sha256") != lock.get("selected_checkpoint_sha256"):
        raise SystemExit("selected checkpoint hash differs between locks")
    for relpath, expected in lock["source_sha256"].items():
        path = (ROOT / relpath).resolve()
        if ROOT not in path.parents or not path.is_file() or digest(path) != expected:
            raise SystemExit(f"offline-gate source changed: {relpath}")
    if digest(SELECTED_LOCK) != lock["selected_lock_sha256"]:
        raise SystemExit("selected-model lock changed after the gate was frozen")
    if digest(AUDIT) != lock["layout_audit_sha256"]:
        raise SystemExit("layout audit changed after the gate was frozen")
    return lock, selected_lock


def load_evaluation_lock() -> tuple[dict, dict, dict]:
    if not EVALUATION_LOCK.is_file() or not COLLECTION_LOCK.is_file() or not SELECTED_LOCK.is_file():
        raise SystemExit("offline-gate evaluation/collection and selected-model locks are required")
    lock = json.loads(EVALUATION_LOCK.read_text(encoding="utf-8"))
    collection_lock = json.loads(COLLECTION_LOCK.read_text(encoding="utf-8"))
    selected_lock = json.loads(SELECTED_LOCK.read_text(encoding="utf-8"))
    if lock.get("gate_evaluation_authorized") is not True or lock.get("reserve_accessed") is not False:
        raise SystemExit("invalid offline-gate evaluation lock")
    if digest(COLLECTION_LOCK) != lock.get("collection_lock_sha256"):
        raise SystemExit("offline-gate collection lock changed")
    if digest(SELECTED_LOCK) != lock.get("selected_lock_sha256"):
        raise SystemExit("selected-model lock changed after the gate was frozen")
    if digest(AUDIT) != lock.get("layout_audit_sha256"):
        raise SystemExit("layout audit changed after the gate was frozen")
    for relpath, expected in lock["source_sha256"].items():
        path = (ROOT / relpath).resolve()
        if ROOT not in path.parents or not path.is_file() or digest(path) != expected:
            raise SystemExit(f"offline-gate evaluation source changed: {relpath}")
    return lock, selected_lock, collection_lock


def collect() -> int:
    lock, _ = load_collection_lock()
    if DATA_DIR.exists():
        raise SystemExit(f"refusing to overwrite {DATA_DIR}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    block = config["seed_splits"]["offline_gate"]
    seeds = list(range(int(block["start"]), int(block["start"]) + int(block["count"])))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    expected = audit["new_blocks"]["offline_gate"]
    if seeds != expected["seeds"]:
        raise SystemExit("offline-gate seeds differ from locked layout audit")
    start = time.monotonic()
    result = collect_teacher_split(
        seeds,
        DATA_DIR,
        artifact_cap_bytes=int(config["resource_caps"]["teacher_collection_artifact_bytes"]),
        wall_cap_seconds=int(config["resource_caps"]["teacher_collection_wall_seconds"]),
    )
    if result["layout_sha256"] != expected["layout_sha256"]:
        raise RuntimeError("offline-gate layouts differ from the pre-audited hashes")
    manifest = {
        "schema": "phase12b_offline_gate_collection_v1",
        "config_sha256": digest(CONFIG),
        "execution_lock_sha256": digest(COLLECTION_LOCK),
        "selected_lock_sha256": digest(SELECTED_LOCK),
        "layout_audit_sha256": digest(AUDIT),
        "authorized_split": "offline_gate",
        "model_was_pinned_before_collection": True,
        "closed_loop_development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
        **result,
        "runner_wall_seconds": time.monotonic() - start,
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "offline_gate_teacher_data_collected",
        "layouts": result["episode_count"],
        "queries": result["decision_query_count"],
        "artifact_bytes": result["artifact_bytes"],
        "manifest_sha256": digest(DATA_DIR / "manifest.json"),
        "model_pinned_before_collection": True,
        "reserve_accessed": False,
    }, indent=2))
    return 0


def evaluate() -> int:
    lock, selected_lock, _ = load_evaluation_lock()
    if not DATA_DIR.is_dir() or not (DATA_DIR / "manifest.json").is_file():
        raise SystemExit("offline-gate teacher data have not been collected")
    if REPORT.exists() or REPORT_JSON.exists():
        raise SystemExit("refusing to overwrite an existing offline-gate report")
    manifest_path = DATA_DIR / "manifest.json"
    manifest_hash = digest(manifest_path)
    if lock.get("offline_gate_manifest_sha256") != manifest_hash:
        raise SystemExit("offline-gate data manifest does not match the frozen lock")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("authorized_split") != "offline_gate" or manifest.get("model_was_pinned_before_collection") is not True:
        raise SystemExit("offline-gate data manifest has invalid provenance")
    if manifest.get("execution_lock_sha256") != digest(COLLECTION_LOCK):
        raise SystemExit("offline-gate manifest was not collected under the pinned collection lock")
    if manifest.get("selected_lock_sha256") != digest(SELECTED_LOCK) or manifest.get("layout_audit_sha256") != digest(AUDIT):
        raise SystemExit("offline-gate manifest selected-model/layout provenance mismatch")
    if manifest.get("layout_sha256") != json.loads(AUDIT.read_text(encoding="utf-8"))["new_blocks"]["offline_gate"]["layout_sha256"]:
        raise SystemExit("offline-gate manifest layout hashes differ from the locked audit")
    for filename, expected in lock["offline_data_sha256"].items():
        path = DATA_DIR / filename
        if digest(path) != expected:
            raise SystemExit(f"offline-gate dataset hash mismatch: {filename}")
    checkpoint = (ROOT / selected_lock["selected_checkpoint"]).resolve()
    if digest(checkpoint) != selected_lock["selected_checkpoint_sha256"]:
        raise SystemExit("selected checkpoint changed")
    checkpoint_blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_blob.get("variant") != "direct_action_value" or checkpoint_blob.get("fit_seed") != 1211:
        raise SystemExit("checkpoint contents do not match the frozen model selection")
    model = CorridorCandidateScorer("value")
    model.load_state_dict(checkpoint_blob["state_dict"])
    model.eval()
    data = load_phase12_split(DATA_DIR)
    metrics = evaluate_phase12b(model, data, variant="direct_action_value", device=torch.device("cpu"))
    layout_ids = data["meta_layout_index"]
    per_layout = np.asarray([
        metrics["selected_action_regret"][layout_ids == layout].mean()
        for layout in np.unique(layout_ids)
    ], dtype=np.float64)
    rng = np.random.default_rng(int(lock["bootstrap"]["seed"]))
    draws = rng.integers(0, len(per_layout), size=(int(lock["bootstrap"]["resamples"]), len(per_layout)))
    bootstrap_means = per_layout[draws].mean(axis=1)
    ci = np.quantile(bootstrap_means, [0.025, 0.975])
    threshold = float(lock["offline_gate_threshold"])
    gate_pass = metrics["layout_macro_mean_normalized_action_group_regret"] <= threshold
    report = {
        "schema": "phase12b_offline_gate_evaluation_v1",
        "selected_variant": selected_lock["selected_variant"],
        "selected_fit_seed": selected_lock["selected_fit_seed"],
        "checkpoint_sha256": digest(checkpoint),
        "config_sha256": digest(CONFIG),
        "training_lock_sha256": lock["training_lock_sha256"],
        "offline_gate_evaluation_lock_sha256": digest(EVALUATION_LOCK),
        "offline_gate_manifest_sha256": manifest_hash,
        "layout_hash_audit_sha256": digest(AUDIT),
        "device": "cpu",
        "torch_version": torch.__version__,
        "queries": metrics["queries"],
        "layouts": metrics["layouts"],
        "layout_macro_mean_normalized_action_group_regret": metrics["layout_macro_mean_normalized_action_group_regret"],
        "layout_bootstrap_95_percent_ci": [float(ci[0]), float(ci[1])],
        "sample_mean_normalized_action_group_regret": metrics["sample_mean_normalized_action_group_regret"],
        "expected_normalized_action_group_regret": metrics["expected_normalized_action_group_regret"],
        "teacher_first_action_exact_agreement": metrics["teacher_first_action_exact_agreement"],
        "optimal_action_set_top1_inclusion": metrics["optimal_action_set_top1_inclusion"],
        "top3_optimal_action_set_recall": metrics["top3_optimal_action_set_recall"],
        "action_set_nll": metrics["action_set_nll"],
        "action_set_brier": metrics["action_set_brier"],
        "offline_gate_threshold": threshold,
        "offline_gate_pass": bool(gate_pass),
        "closed_loop_development_authorized_by_result": bool(gate_pass),
        "dagger_or_reserve_accessed": False,
        "interpretation": (
            "Offline action-level regret gate passed; this permits consideration of the separately frozen closed-loop development protocol only."
            if gate_pass else
            "Offline action-level regret gate failed; stop before closed-loop development, DAgger, and world-model mechanisms."
        ),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "---\n" +
        "type: experiment-report\nstatus: completed\nupdated: 2026-09-28\n" +
        "---\n\n# Phase 12b offline gate\n\n" +
        f"Selected model: `{report['selected_variant']}` seed `{report['selected_fit_seed']}`.\n\n" +
        f"- Layout-macro action-group regret: **{report['layout_macro_mean_normalized_action_group_regret']:.4f}** "
        f"(layout bootstrap 95% CI {report['layout_bootstrap_95_percent_ci'][0]:.4f}–{report['layout_bootstrap_95_percent_ci'][1]:.4f})\n" +
        f"- Frozen threshold: **≤ {threshold:.2f}**; gate: **{'PASS' if gate_pass else 'FAIL'}**\n" +
        f"- Exact first-action agreement: {report['teacher_first_action_exact_agreement']:.3f}; "
        f"optimal-action-set inclusion: {report['optimal_action_set_top1_inclusion']:.3f}\n" +
        f"- Closed-loop development eligible: **{str(gate_pass).lower()}**; DAgger/reserve accessed: **false**\n\n" +
        "This is an offline, execution-relevant ranking test. It is not closed-loop control evidence. "
        "The result and full provenance are in `phase12b_offline_gate.json`.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("collect", "evaluate"), required=True)
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    expected = "COLLECT_PHASE12B_OFFLINE_GATE" if args.mode == "collect" else "EVALUATE_PHASE12B_OFFLINE_GATE"
    if args.confirmation != expected:
        raise SystemExit(f"operation requires --confirmation {expected}")
    return collect() if args.mode == "collect" else evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
