"""Fit matched absolute and agent-relative bounded policies; lock before gate collection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from consequence_lab.phase12cE import (
    agent_relative_state_features,
    geometry_waypoint_actions,
    phase12cE_artifact_bytes,
    verify_phase12cE_dataset_files,
)
from consequence_lab.phase12c_c_training import (
    _fit_one,
    evaluate_model,
    load_dataset,
    model_parameter_count,
    permutation_consistency,
)
from audit_phase12cE_layouts import verify_lock
from freeze_phase12cE_lock import CONFIG, LOCK, digest


def transformed_copy(data: dict, representation: str, floor: float,
                     mean: np.ndarray | None = None, scale: np.ndarray | None = None):
    result = copy.deepcopy(data)
    states = result["inputs"]["states"].astype(np.float32, copy=True)
    if representation == "agent_relative":
        states = agent_relative_state_features(states)
    elif representation != "absolute":
        raise ValueError("unknown state representation")
    if mean is None:
        mean = states.mean(axis=0, dtype=np.float64)
        scale = states.std(axis=0, dtype=np.float64)
        scale = np.where(scale < floor, 1.0, scale)
    assert scale is not None
    result["inputs"]["states"] = ((states - mean) / scale).astype(np.float32)
    return result, mean, scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    if args.confirmation != "FIT_PHASE12CE_STUDENTS":
        raise SystemExit("student fitting requires --confirmation FIT_PHASE12CE_STUDENTS")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lock, lock_path = verify_lock(config)
    audit_path = ROOT / config["seed_splits"]["audit_output"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != "phase12cE_layout_audit_v1" or audit.get("reserve_layouts_generated_or_accessed") is not False:
        raise SystemExit("Phase 12c-E hash audit is missing or invalid")
    selection_dir = ROOT / config["seed_splits"]["selection_output"]
    selection = load_dataset(selection_dir)
    selection_manifest = verify_phase12cE_dataset_files(selection_dir)
    if selection_manifest.get("split") != "model_selection" or selection_manifest.get("layout_count") != 40:
        raise SystemExit("selection dataset is not the approved 40-layout block")
    if selection_manifest.get("execution_lock_sha256") != digest(lock_path):
        raise SystemExit("selection dataset was generated under another source lock")
    if selection_manifest.get("layout_audit_sha256") != digest(audit_path):
        raise SystemExit("selection dataset was generated under another layout audit")
    expected_selection_seeds = set(range(
        int(config["seed_splits"]["model_selection"]["start"]),
        int(config["seed_splits"]["model_selection"]["start"])
        + int(config["seed_splits"]["model_selection"]["count"]),
    ))
    if ({int(row["episode_seed"]) for row in selection["metadata"]} != expected_selection_seeds
            or {row["layout_sha256"] for row in selection["metadata"]}
            != set(audit["blocks"]["model_selection"]["layout_sha256"])):
        raise SystemExit("selection query rows do not match the frozen layout block")

    training_dir = ROOT / config["training_source"]["archived_dataset"]
    if digest(training_dir / "manifest.json") != lock["training_manifest_sha256"]:
        raise SystemExit("archived fitting source manifest changed")
    train = load_dataset(training_dir)
    if len(train["metadata"]) != int(config["training_source"]["query_count"]):
        raise SystemExit("archived fitting query count mismatch")
    if len({row["layout_sha256"] for row in train["metadata"]}) != int(config["training_source"]["layout_count"]):
        raise SystemExit("archived fitting layout count mismatch")
    train_actions = geometry_waypoint_actions(train["inputs"]["states"], train["metadata"])
    for row, action in zip(train["metadata"], train_actions):
        row["geometry_action"] = action

    device = torch.device("cpu")
    torch.set_num_threads(1)
    training_started = time.monotonic()
    student_config = config["student"]
    fit_config = {"models": {
        "common": {
            "hidden_state": int(student_config["hidden_state"]),
            "embedding": int(student_config["embedding"]),
            "maximum_parameters": int(student_config["max_parameters"]),
        },
        "fitting": {
            "learning_rate": float(student_config["learning_rate"]),
            "weight_decay": float(student_config["weight_decay"]),
            "batch_decision_sets": int(student_config["batch_decision_sets"]),
            "max_epochs": int(student_config["max_epochs"]),
            "early_stopping_patience": int(student_config["early_stopping_patience"]),
            "gradient_norm_clip": float(student_config["gradient_norm_clip"]),
            "seeds": [int(value) for value in student_config["fit_seeds"]],
        },
    }}
    output = ROOT / config["seed_splits"]["training_output"]
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    selected: dict[str, dict] = {}
    run_records: dict[str, list[dict]] = {}
    normalizers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for representation in ("absolute", "agent_relative"):
        train_view, mean, scale = transformed_copy(
            train, representation, float(student_config["std_floor"])
        )
        selection_view, _, _ = transformed_copy(
            selection, representation, float(student_config["std_floor"]), mean, scale
        )
        normalizers[representation] = (mean, scale)
        candidates = []
        for seed in fit_config["models"]["fitting"]["seeds"]:
            model, summary = _fit_one(
                train_view, selection_view, variant="bounded_preference_policy",
                seed=seed, config=fit_config, device=device, started=time.monotonic(),
                time_cap=2**31 - 1,
            )
            metrics = evaluate_model(
                model, selection_view, variant="bounded_preference_policy", device=device
            )
            summary["selection_metrics"] = metrics
            candidates.append((model, summary))
        candidates.sort(key=lambda item: (
            float(item[1]["best_selection_heldout_regret"]),
            int(item[1]["fit_seed"]),
            int(item[1]["best_epoch"]),
        ))
        model, summary = candidates[0]
        if model_parameter_count(model) != 32705 or model_parameter_count(model) > int(student_config["max_parameters"]):
            raise SystemExit("student architecture parameter count differs from frozen protocol")
        permutation = permutation_consistency(
            model, variant="bounded_preference_policy",
            state_rows=selection_view["inputs"]["states"], device=device,
        )
        if (permutation["max_probability_deviation"] > float(student_config["candidate_permutation"]["probability_deviation_max"])
                or permutation["semantic_choice_flips"] > int(student_config["candidate_permutation"]["semantic_flips_max"])):
            raise SystemExit(f"{representation} model failed the candidate permutation gate")
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{representation}_seed{summary['fit_seed']}.pt"
        torch.save({
            "phase": "12c-E",
            "representation": representation,
            "variant": "bounded_preference_policy",
            "fit_seed": int(summary["fit_seed"]),
            "state_dict": model.state_dict(),
            "input_mean": torch.as_tensor(normalizers[representation][0], dtype=torch.float64),
            "input_scale": torch.as_tensor(normalizers[representation][1], dtype=torch.float64),
            "config_sha256": digest(CONFIG),
            "execution_lock_sha256": digest(lock_path),
            "selection_manifest_sha256": digest(selection_dir / "manifest.json"),
        }, checkpoint_path)
        normalizer_path = output / f"{representation}_normalizer.npz"
        np.savez_compressed(normalizer_path, mean=normalizers[representation][0],
                            scale=normalizers[representation][1])
        selected[representation] = {
            "fit_seed": int(summary["fit_seed"]),
            "best_epoch": int(summary["best_epoch"]),
            "selection_regret": float(summary["best_selection_heldout_regret"]),
            "selection_metrics": summary["selection_metrics"],
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": digest(checkpoint_path),
            "normalizer_path": str(normalizer_path.relative_to(ROOT)),
            "normalizer_sha256": digest(normalizer_path),
            "parameter_count": model_parameter_count(model),
            "permutation": permutation,
        }
        run_records[representation] = [
            {"fit_seed": int(summary["fit_seed"]), "best_epoch": int(summary["best_epoch"]),
             "best_selection_regret": float(summary["best_selection_heldout_regret"]),
             "history": summary["history"]}
            for _, summary in candidates
        ]

    result = {
        "schema": "phase12cE_training_result_v1",
        "config_sha256": digest(CONFIG),
        "execution_lock_sha256": digest(lock_path),
        "selection_manifest_sha256": digest(selection_dir / "manifest.json"),
        "training_manifest_sha256": digest(training_dir / "manifest.json"),
        "training_layouts": len({row["layout_sha256"] for row in train["metadata"]}),
        "training_queries": len(train["metadata"]),
        "wall_seconds": time.monotonic() - training_started,
        "device": str(device),
        "cuda_used": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "selected": selected,
        "fit_runs": run_records,
        "offline_gate_outcomes_accessed": False,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    selected_lock = {
        "schema": "phase12cE_selected_models_lock_v1",
        "status": "frozen_before_gate",
        "config_sha256": digest(CONFIG),
        "execution_lock_sha256": digest(lock_path),
        "selection_manifest_sha256": digest(selection_dir / "manifest.json"),
        "training_manifest_sha256": digest(training_dir / "manifest.json"),
        "training_result_sha256": digest(result_path),
        "models": selected,
        "gate_outcomes_accessed": False,
        "reserve_accessed": False,
    }
    selected_lock_path = ROOT / config["seed_splits"]["selected_model_lock"]
    selected_lock_path.parent.mkdir(parents=True, exist_ok=True)
    selected_lock_path.write_text(json.dumps(selected_lock, indent=2) + "\n", encoding="utf-8")
    if phase12cE_artifact_bytes(ROOT) > int(config["resources"]["generated_artifact_cap_bytes"]):
        raise SystemExit("Phase 12c-E total generated-artifact cap exceeded")
    print(json.dumps({"status": "phase12cE_models_frozen_before_gate", "selected": {
        key: {"fit_seed": value["fit_seed"], "best_epoch": value["best_epoch"],
              "selection_regret": value["selection_regret"],
              "checkpoint_sha256": value["checkpoint_sha256"]}
        for key, value in selected.items()}, "selected_model_lock": str(selected_lock_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
