"""Collect the frozen Phase 12c-C offline gate after both models are locked."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_model import CorridorCandidateScorer, model_parameter_count
from consequence_lab.phase12c_c import collect_balanced_teacher_dataset, persist_teacher_dataset
from consequence_lab.phase12c_c_training import (
    evaluate_model,
    load_dataset,
    measure_cpu_decision_latency,
    permutation_consistency,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--config", default="configs/phase12cC_locked.yaml")
    parser.add_argument("--training-lock", default="experiments/phase12cC_training.lock.json")
    parser.add_argument("--selected-lock", default="experiments/phase12cC_selected_models.lock.json")
    parser.add_argument("--fit-output", default="artifacts/runs/phase12cC_selected_models_v1")
    parser.add_argument("--output", default="artifacts/runs/phase12cC_offline_gate_v1")
    args = parser.parse_args()
    if args.confirmation != "RUN_PHASE12C_C_OFFLINE_GATE_ONLY":
        raise SystemExit("offline gate requires --confirmation RUN_PHASE12C_C_OFFLINE_GATE_ONLY")
    config_path, training_lock_path = ROOT / args.config, ROOT / args.training_lock
    selected_lock_path, fit_output = ROOT / args.selected_lock, ROOT / args.fit_output
    output = ROOT / args.output
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training_lock = json.loads(training_lock_path.read_text(encoding="utf-8"))
    selected_lock = json.loads(selected_lock_path.read_text(encoding="utf-8"))
    runtime = training_lock.get("runtime", {})
    if runtime.get("torch") != torch.__version__ or runtime.get("cuda_available") != torch.cuda.is_available():
        raise SystemExit("offline evaluation runtime differs from frozen training runtime")
    if training_lock.get("training_authorized") is not True or training_lock.get("offline_gate_labels_collected") is not False:
        raise SystemExit("Phase 12c-C execution lock does not preserve training/gate boundary")
    if digest(config_path) != training_lock.get("config_sha256"):
        raise SystemExit("Phase 12c-C config hash mismatch")
    if selected_lock.get("schema") != "phase12cC_selected_models_lock_v1":
        raise SystemExit("selected model lock is missing/invalid")
    if selected_lock.get("training_lock_sha256") != digest(training_lock_path):
        raise SystemExit("selected checkpoints were not frozen under current training lock")
    if selected_lock.get("offline_gate_labels_collected") is not False or selected_lock.get("reserve_accessed") is not False:
        raise SystemExit("selected model lock indicates forbidden data access")
    if output.exists() or not output.resolve().is_relative_to((ROOT / "artifacts/runs").resolve()):
        raise SystemExit("offline-gate output must be a new path under artifacts/runs")
    for relative, expected in training_lock["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"locked source changed: {relative}")
    audit_path = ROOT / config["seed_splits"]["hash_audit"]["output"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if digest(audit_path) != training_lock.get("phase12cC_layout_audit_sha256"):
        raise SystemExit("fresh split audit hash mismatch")
    if audit.get("reserve_accessed") is not False:
        raise SystemExit("layout audit does not attest sealed reserve")
    for relative, expected in selected_lock["checkpoint_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"selected checkpoint changed: {relative}")
    results_path = fit_output / "selection_results.json"
    fit_manifest_path = fit_output / "fit_manifest.json"
    if not results_path.is_file() or digest(results_path) != selected_lock.get("selection_results_sha256"):
        raise SystemExit("selected-model results are not the ones frozen in the lock")
    if digest(fit_manifest_path) != selected_lock.get("fit_manifest_sha256"):
        raise SystemExit("fit manifest is not the one frozen in the selected lock")
    training_data_path = ROOT / config["data_generation"]["outputs"]["archived_training"]
    selection_data_path = ROOT / config["data_generation"]["outputs"]["model_selection"]
    training_data_manifest = training_data_path / "manifest.json"
    selection_data_manifest = selection_data_path / "manifest.json"
    if digest(training_data_manifest) != selected_lock.get("training_data_manifest_sha256"):
        raise SystemExit("archived training dataset differs from selected-model lock")
    if digest(selection_data_manifest) != selected_lock.get("model_selection_manifest_sha256"):
        raise SystemExit("model-selection dataset differs from selected-model lock")
    for manifest_path in (training_data_manifest, selection_data_manifest):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("offline_gate_labels_collected") is not False or manifest.get("reserve_accessed") is not False:
            raise SystemExit("pre-gate dataset manifest violates the frozen split boundary")

    block = audit["new_blocks"]["offline_gate"]
    started = time.monotonic()
    gate_dataset = collect_balanced_teacher_dataset(
        block["seeds"], block["layout_sha256"], split="offline_gate",
        master_seed=int(config["data_generation"]["master_seed"]), banks=12,
        suffixes_per_action=int(config["teacher"]["suffixes_per_first_action"]),
        time_cap_seconds=int(config["resource_caps"]["max_total_wall_seconds"]),
        geometry_resolution=float(config["evaluation"]["geometry_parameters"]["resolution"]),
        waypoint_tolerance=float(config["evaluation"]["geometry_parameters"]["waypoint_tolerance"]),
    )
    dataset_manifest = persist_teacher_dataset(
        output, gate_dataset, split="offline_gate", lock_sha256=digest(selected_lock_path),
        config_sha256=digest(config_path), max_bytes=int(config["resource_caps"]["max_generated_artifact_bytes"]),
    )
    gate = load_dataset(output)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    fit_results = json.loads(results_path.read_text(encoding="utf-8"))
    variants: dict[str, dict] = {}
    for variant, selected in selected_lock["selected_models"].items():
        model = CorridorCandidateScorer(
            "policy" if variant == "bounded_preference_policy" else "value",
            hidden_state=int(config["models"]["common"]["hidden_state"]),
            embedding=int(config["models"]["common"]["embedding"]),
        ).to(device)
        if model_parameter_count(model) > int(config["models"]["common"]["maximum_parameters"]):
            raise SystemExit("pinned model exceeds parameter cap")
        ckpt_rel = next(path for path in selected_lock["checkpoint_sha256"]
                        if Path(path).name == f"{variant}_seed{selected['fit_seed']}.pt")
        checkpoint = torch.load(ROOT / ckpt_rel, map_location=device, weights_only=False)
        if checkpoint.get("variant") != variant or int(checkpoint.get("fit_seed", -1)) != int(selected["fit_seed"]):
            raise SystemExit("checkpoint metadata does not match selected-model lock")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        metrics = evaluate_model(model, gate, variant=variant, device=device)
        permutation = permutation_consistency(
            model, variant=variant, state_rows=gate["inputs"]["states"], device=device
        )
        latency = measure_cpu_decision_latency(
            model, variant=variant, state_row=gate["inputs"]["states"][0]
        )
        promotion = config["evaluation"]["promotion_gate"]
        passed = bool(
            metrics["layout_macro_argmax_heldout_regret"] <= promotion["regret_point_mean_maximum"]
            and metrics["regret_one_sided_98_75_upper"] <= promotion["one_sided_regret_upper_bound_maximum"]
            and metrics["mean_improvement_over_geometry"] >= promotion["minimum_mean_improvement_over_geometry"]
            and metrics["improvement_one_sided_98_75_lower"] > promotion["one_sided_improvement_lower_bound_minimum"]
            and permutation["max_probability_deviation"] <= promotion["required_permutation_max_probability_deviation"]
            and permutation["semantic_choice_flips"] == promotion["required_semantic_choice_flips"]
        )
        variants[variant] = {
            "fit_seed": selected["fit_seed"],
            "parameter_count": model_parameter_count(model),
            "offline_gate_metrics": metrics,
            "permutation": permutation,
            "decision_latency": latency,
            "promotion_gate_pass": passed,
        }

    result = {
        "schema": "phase12cC_offline_gate_result_v1",
        "phase12cC_training_lock_sha256": digest(training_lock_path),
        "selected_models_lock_sha256": digest(selected_lock_path),
        "offline_gate_data_manifest_sha256": digest(output / "manifest.json"),
        "training_selection_results_sha256": digest(results_path),
        "training_fit_manifest_sha256": digest(fit_manifest_path),
        "offline_gate_layout_count": dataset_manifest["layout_count"],
        "offline_gate_query_count": dataset_manifest["query_count"],
        "variants": variants,
        "phase12cC_promotion_pass": any(row["promotion_gate_pass"] for row in variants.values()),
        "interpretation": "offline_generalization_only_no_closed_loop_or_planner_amortization_claim",
        "offline_gate_read": True,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    result_path = output / "evaluation.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluation_sha256"] = digest(result_path)
    manifest["selected_models_lock_sha256"] = digest(selected_lock_path)
    manifest["elapsed_wall_seconds"] = time.monotonic() - started
    manifest["files"]["evaluation.json"] = {"bytes": result_path.stat().st_size, "sha256": digest(result_path)}
    for _ in range(3):
        manifest["artifact_bytes"] = sum(item["bytes"] for item in manifest["files"].values()) + manifest_path.stat().st_size
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["artifact_bytes"] = sum(item["bytes"] for item in manifest["files"].values()) + manifest_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if manifest["artifact_bytes"] > int(config["resource_caps"]["max_generated_artifact_bytes"]):
        raise OSError("Phase 12c-C offline-gate artifact cap exceeded")
    generated_roots = (training_data_path, selection_data_path, fit_output, output)
    total_generated = sum(
        file_path.stat().st_size
        for base in generated_roots for file_path in base.rglob("*") if file_path.is_file()
    )
    if total_generated > int(config["resource_caps"]["max_generated_artifact_bytes"]):
        raise OSError("combined Phase 12c-C artifact cap exceeded")
    print(json.dumps({
        "status": "phase12cC_offline_gate_complete",
        "promotion_pass": result["phase12cC_promotion_pass"],
        "variants": {name: {
            "promotion_gate_pass": value["promotion_gate_pass"],
            "regret": value["offline_gate_metrics"]["layout_macro_argmax_heldout_regret"],
            "regret_ci95": [value["offline_gate_metrics"]["regret_ci95_lower"],
                            value["offline_gate_metrics"]["regret_ci95_upper"]],
            "improvement_over_geometry": value["offline_gate_metrics"]["mean_improvement_over_geometry"],
        } for name, value in variants.items()},
        "reserve_accessed": False,
        "total_generated_artifact_bytes": total_generated,
        "closed_loop_run": False,
        "dagger_run": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
