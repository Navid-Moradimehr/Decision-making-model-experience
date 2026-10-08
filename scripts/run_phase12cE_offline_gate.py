"""Evaluate the frozen Phase 12c-E models on the once-collected offline gate."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from consequence_lab.phase12_model import CorridorCandidateScorer
from consequence_lab.phase12cE import (
    agent_relative_state_features,
    phase12cE_artifact_bytes,
    verify_phase12cE_dataset_files,
)
from consequence_lab.phase12c_c_training import (
    ACTION_FEATURES,
    _layout_arrays,
    _layout_macro,
    evaluate_model,
    load_dataset,
    measure_cpu_decision_latency,
    model_parameter_count,
    permutation_consistency,
)
from audit_phase12cE_layouts import verify_lock
from freeze_phase12cE_lock import CONFIG, digest


def paired_bootstrap_interval(values: np.ndarray, *, seed: int, resamples: int,
                              confidence: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def load_frozen_model(representation: str, selected: dict, gate: dict,
                      device: torch.device) -> tuple[torch.nn.Module, dict, dict]:
    checkpoint_path = ROOT / selected["checkpoint_path"]
    if digest(checkpoint_path) != selected["checkpoint_sha256"]:
        raise SystemExit(f"frozen checkpoint hash mismatch for {representation}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("representation") != representation
            or checkpoint.get("execution_lock_sha256") != selected["execution_lock_sha256"]
            or checkpoint.get("selection_manifest_sha256") != selected["selection_manifest_sha256"]):
        raise SystemExit(f"frozen checkpoint provenance mismatch for {representation}")
    model = CorridorCandidateScorer("policy", hidden_state=128, embedding=64).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    states = gate["inputs"]["states"].astype(np.float32, copy=True)
    if representation == "agent_relative":
        states = agent_relative_state_features(states)
    normalizer_path = ROOT / selected["normalizer_path"]
    if digest(normalizer_path) != selected["normalizer_sha256"]:
        raise SystemExit(f"frozen normalizer hash mismatch for {representation}")
    with np.load(normalizer_path, allow_pickle=False) as archive:
        mean, scale = archive["mean"], archive["scale"]
    if mean.shape != (52,) or scale.shape != (52,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise SystemExit("invalid frozen normalization parameters")
    prepared = {
        "inputs": dict(gate["inputs"], states=((states - mean) / scale).astype(np.float32)),
        "targets": gate["targets"],
        "metadata": gate["metadata"],
        "manifest": gate["manifest"],
    }
    return model, prepared, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", default="")
    args = parser.parse_args()
    if args.confirmation != "EVALUATE_PHASE12CE_OFFLINE_GATE":
        raise SystemExit("offline gate requires --confirmation EVALUATE_PHASE12CE_OFFLINE_GATE")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    lock, lock_path = verify_lock(config)
    seed_config = config["seed_splits"]
    selected_lock_path = ROOT / seed_config["selected_model_lock"]
    selected_lock = json.loads(selected_lock_path.read_text(encoding="utf-8"))
    if (selected_lock.get("schema") != "phase12cE_selected_models_lock_v1"
            or selected_lock.get("status") != "frozen_before_gate"
            or selected_lock.get("gate_outcomes_accessed") is not False
            or selected_lock.get("reserve_accessed") is not False
            or selected_lock.get("execution_lock_sha256") != digest(lock_path)):
        raise SystemExit("selected models were not frozen before gate access")
    gate_dir = ROOT / seed_config["gate_output"]
    gate_manifest_path = gate_dir / "manifest.json"
    if not gate_manifest_path.is_file():
        raise SystemExit("offline gate data has not been collected")
    gate_manifest = verify_phase12cE_dataset_files(gate_dir)
    if (gate_manifest.get("split") != "offline_gate" or gate_manifest.get("layout_count") != 40
            or gate_manifest.get("execution_lock_sha256") != digest(lock_path)
            or gate_manifest.get("selected_model_lock_sha256") != digest(selected_lock_path)
            or gate_manifest.get("reserve_accessed") is not False):
        raise SystemExit("offline gate provenance/scope mismatch")
    audit = json.loads((ROOT / seed_config["audit_output"]).read_text(encoding="utf-8"))
    if selected_lock.get("selection_manifest_sha256") != digest(
        ROOT / seed_config["selection_output"] / "manifest.json"
    ):
        raise SystemExit("selected models are not bound to the approved selection split")
    output_path = ROOT / "artifacts/runs/phase12cE_offline_gate_result_v1.json"
    if output_path.exists():
        raise SystemExit("refusing to evaluate or overwrite the offline gate more than once")

    gate = load_dataset(gate_dir)
    layout_ids, layout_names = _layout_arrays(gate["metadata"])
    expected_gate_seeds = set(range(
        int(seed_config["offline_gate"]["start"]),
        int(seed_config["offline_gate"]["start"]) + int(seed_config["offline_gate"]["count"]),
    ))
    if (len(layout_names) != 40
            or {int(row["episode_seed"]) for row in gate["metadata"]} != expected_gate_seeds
            or set(layout_names) != set(audit["blocks"]["offline_gate"]["layout_sha256"])):
        raise SystemExit("offline gate does not contain exactly 40 independent layouts")
    for model_meta in selected_lock["models"].values():
        if model_meta["parameter_count"] > int(config["student"]["max_parameters"]):
            raise SystemExit("frozen model exceeds parameter cap")

    device = torch.device("cpu")
    torch.set_num_threads(1)
    started = time.monotonic()
    results: dict[str, dict] = {}
    per_layout_regret: dict[str, np.ndarray] = {}
    for representation in ("absolute", "agent_relative"):
        model, prepared, checkpoint = load_frozen_model(
            representation, selected_lock["models"][representation] | {
                "execution_lock_sha256": selected_lock["execution_lock_sha256"],
                "selection_manifest_sha256": selected_lock["selection_manifest_sha256"],
            }, gate, device,
        )
        metrics = evaluate_model(model, prepared, variant="bounded_preference_policy", device=device)
        states = prepared["inputs"]["states"]
        permutation = permutation_consistency(
            model, variant="bounded_preference_policy", state_rows=states, device=device,
        )
        if (permutation["max_probability_deviation"] > float(config["student"]["candidate_permutation"]["probability_deviation_max"])
                or permutation["semantic_choice_flips"] > int(config["student"]["candidate_permutation"]["semantic_flips_max"])):
            raise SystemExit(f"{representation} model failed gate-time permutation integrity")
        logits = model(
            torch.as_tensor(states[:1], dtype=torch.float32),
            torch.as_tensor(ACTION_FEATURES, dtype=torch.float32)[None],
            torch.ones((1, 9), dtype=torch.bool),
        )[0].detach().cpu().numpy()
        selected_action = int(np.argmax(logits))
        bank_regret = gate["targets"]["per_bank_normalized_regret"][:, 6:12, :].mean(axis=1)
        actions = np.argmax(model(
            torch.as_tensor(states, dtype=torch.float32),
            torch.as_tensor(ACTION_FEATURES, dtype=torch.float32)[None].expand(len(states), -1, -1),
            torch.ones((len(states), 9), dtype=torch.bool),
        ).detach().cpu().numpy(), axis=1)
        selected_regret = bank_regret[np.arange(len(actions)), actions]
        _, macro = _layout_macro(selected_regret, layout_ids)
        per_layout_regret[representation] = macro
        modal_teacher_actions = np.argmax(
            gate["targets"]["per_bank_tie_split_optimal_mass"][:, 0:6, :].mean(axis=1), axis=1
        )
        top1_agreement, _ = _layout_macro(
            (actions == modal_teacher_actions).astype(np.float64), layout_ids
        )
        results[representation] = {
            "metrics": metrics,
            "top1_agreement_with_training_bank_modal_teacher": top1_agreement,
            "permutation": permutation,
            "parameter_count": model_parameter_count(model),
            "checkpoint_sha256": selected_lock["models"][representation]["checkpoint_sha256"],
            "latency": measure_cpu_decision_latency(
                model, variant="bounded_preference_policy", state_row=states[0]
            ),
            "action_counts": np.bincount(actions, minlength=9).tolist(),
        }

    # Geometry and coast use exactly the same observed states and held-out bank costs.
    bank_regret = gate["targets"]["per_bank_normalized_regret"][:, 6:12, :].mean(axis=1)
    row_indices = np.arange(len(gate["metadata"]))
    geometry_actions = np.asarray([row["geometry_action"] for row in gate["metadata"]], dtype=np.int64)
    coast_actions = np.zeros(len(row_indices), dtype=np.int64)
    geometry_rows = bank_regret[row_indices, geometry_actions]
    coast_rows = bank_regret[row_indices, coast_actions]
    geometry_mean, geometry_macro = _layout_macro(geometry_rows, layout_ids)
    coast_mean, coast_macro = _layout_macro(coast_rows, layout_ids)
    results["geometry_waypoint"] = {"layout_macro_mean_regret": geometry_mean,
                                    "per_layout_regret": geometry_macro.tolist()}
    results["constant_coast"] = {"layout_macro_mean_regret": coast_mean,
                                 "per_layout_regret": coast_macro.tolist()}

    bootstrap = config["evaluation"]["bootstrap"]
    relative_gain_vs_absolute = per_layout_regret["absolute"] - per_layout_regret["agent_relative"]
    relative_gain_vs_geometry = geometry_macro - per_layout_regret["agent_relative"]
    ci_abs = paired_bootstrap_interval(
        relative_gain_vs_absolute, seed=int(bootstrap["seed"]),
        resamples=int(bootstrap["resamples"]), confidence=float(bootstrap["confidence"]),
    )
    ci_geo = paired_bootstrap_interval(
        relative_gain_vs_geometry, seed=int(bootstrap["seed"]) + 1,
        resamples=int(bootstrap["resamples"]), confidence=float(bootstrap["confidence"]),
    )
    mean_gain_abs = float(relative_gain_vs_absolute.mean())
    mean_gain_geo = float(relative_gain_vs_geometry.mean())
    gates = config["evaluation"]["advancement"]
    passed = bool(
        mean_gain_abs >= float(gates["minimum_mean_regret_improvement_relative_over_absolute"])
        and ci_abs[0] > float(gates["paired_95_lower_relative_over_absolute"])
        and mean_gain_geo > float(gates["mean_regret_improvement_relative_over_geometry"])
        and ci_geo[0] > float(gates["paired_95_lower_relative_over_geometry"])
    )
    result = {
        "schema": "phase12cE_offline_gate_result_v1",
        "status": "passed_relative_representation_gate" if passed else "failed_relative_representation_gate",
        "config_sha256": digest(CONFIG),
        "execution_lock_sha256": digest(lock_path),
        "selected_model_lock_sha256": digest(selected_lock_path),
        "gate_manifest_sha256": digest(gate_manifest_path),
        "gate_layout_count": len(layout_names),
        "gate_query_count": len(gate["metadata"]),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "models": results,
        "paired_comparison": {
            "relative_regret_improvement_vs_absolute": mean_gain_abs,
            "relative_vs_absolute_paired_95_ci": list(ci_abs),
            "relative_regret_improvement_vs_geometry": mean_gain_geo,
            "relative_vs_geometry_paired_95_ci": list(ci_geo),
        },
        "advancement_gates": gates,
        "wall_seconds": time.monotonic() - started,
        "reserve_accessed": False,
        "closed_loop_development_authorized_or_accessed": False,
        "dagger_accessed": False,
        "world_model_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if phase12cE_artifact_bytes(ROOT) > int(config["resources"]["generated_artifact_cap_bytes"]):
        raise SystemExit("Phase 12c-E total generated-artifact cap exceeded")
    print(json.dumps({"status": result["status"], "relative_vs_absolute": {
        "mean_gain": mean_gain_abs, "ci95": ci_abs}, "relative_vs_geometry": {
        "mean_gain": mean_gain_geo, "ci95": ci_geo},
        "output": str(output_path.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
