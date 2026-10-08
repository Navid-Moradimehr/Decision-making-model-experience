"""Fit/evaluate compact Phase 12c-C students with layout-aware selection."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from consequence_lab.phase12_model import CorridorCandidateScorer, model_parameter_count
from consequence_lab.phase12_teacher import candidate_features


INPUT_ALLOWLIST = {"states", "candidate_action_ids", "candidate_mask"}
ACTION_SEQUENCES = np.full((9, 8), 9, dtype=np.uint8)
ACTION_SEQUENCES[:, 0] = np.arange(9, dtype=np.uint8)
ACTION_FEATURES = candidate_features(ACTION_SEQUENCES)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_config_sha256(config: dict[str, Any]) -> str:
    """Hash YAML-loaded config values deterministically, including date scalars."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def load_dataset(directory: Path) -> dict[str, Any]:
    with np.load(directory / "student_inputs.npz", allow_pickle=False) as archive:
        if set(archive.files) != INPUT_ALLOWLIST:
            raise ValueError("Phase 12c-C student input archive violates its allowlist")
        inputs = {key: archive[key] for key in archive.files}
    with np.load(directory / "teacher_targets.npz", allow_pickle=False) as archive:
        targets = {key: archive[key] for key in archive.files}
    metadata = json.loads((directory / "sample_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    n = len(inputs["states"])
    if inputs["states"].shape != (n, 52) or inputs["candidate_action_ids"].shape != (n, 9):
        raise ValueError("Phase 12c-C student input shapes are invalid")
    if inputs["candidate_mask"].shape != (n, 9) or not np.all(inputs["candidate_mask"]):
        raise ValueError("Phase 12c-C candidate masks must expose all nine legal actions")
    if not np.array_equal(inputs["candidate_action_ids"], np.broadcast_to(np.arange(9), (n, 9))):
        raise ValueError("runtime action candidates are not exactly actions 0 through 8")
    if len(metadata) != n or manifest.get("query_count") != n:
        raise ValueError("Phase 12c-C data/metadata sample count mismatch")
    for name in ("mean_regret_target", "preference_target", "per_bank_normalized_regret",
                 "per_bank_tie_split_optimal_mass", "candidate_objective_keys",
                 "candidate_objective_mask", "suffix_count", "proposal_seed"):
        if name not in targets or targets[name].shape[0] != n:
            raise ValueError(f"missing or malformed Phase 12c-C target {name}")
    objective_keys = targets["candidate_objective_keys"]
    objective_mask = targets["candidate_objective_mask"].astype(bool)
    suffix_count = targets["suffix_count"].astype(np.int64)
    if objective_keys.shape != (n, 12, 126, 3) or objective_mask.shape != (n, 12, 126):
        raise ValueError("Phase 12c-C complete candidate-cost landscape shape mismatch")
    if suffix_count.shape != (n, 12) or np.any(suffix_count < 1) or np.any(suffix_count > 14):
        raise ValueError("Phase 12c-C suffix-count metadata invalid")
    if not np.array_equal(objective_mask.sum(axis=2), 9 * suffix_count):
        raise ValueError("candidate mask does not match each bank's available balanced suffixes")
    if np.any(~np.isfinite(objective_keys[objective_mask])) or np.any(np.isfinite(objective_keys[~objective_mask])):
        raise ValueError("candidate objective keys have malformed valid/padded entries")
    proposal_seeds = targets["proposal_seed"]
    if proposal_seeds.shape != (n, 12) or np.unique(proposal_seeds).size != proposal_seeds.size:
        raise ValueError("proposal-bank seeds must be unique within a split")
    if manifest.get("unique_bank_seeds") is not True or manifest.get("unique_query_layout_decisions") is not True:
        raise ValueError("dataset manifest failed bank/query uniqueness checks")
    if not np.isfinite(inputs["states"]).all() or not np.isfinite(targets["mean_regret_target"]).all():
        raise ValueError("non-finite input or teacher target")
    if np.any(targets["mean_regret_target"] < -1e-7) or np.any(targets["mean_regret_target"] > 1 + 1e-7):
        raise ValueError("normalized regret target outside [0,1]")
    policy_targets = targets["preference_target"]
    if not np.allclose(policy_targets.sum(axis=1), 1.0, atol=1e-6) or np.any(policy_targets < 0):
        raise ValueError("tie-split preference targets must be distributions over nine actions")
    return {"inputs": inputs, "targets": targets, "metadata": metadata, "manifest": manifest}


def _layout_arrays(metadata: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    names = sorted({str(row["layout_sha256"]) for row in metadata})
    lookup = {name: index for index, name in enumerate(names)}
    ids = np.asarray([lookup[str(row["layout_sha256"])] for row in metadata], dtype=np.int64)
    return ids, names


def _layout_macro(values: np.ndarray, layout_ids: np.ndarray) -> tuple[float, np.ndarray]:
    per_layout = np.asarray([values[layout_ids == item].mean() for item in np.unique(layout_ids)], dtype=np.float64)
    return float(per_layout.mean()), per_layout


def _one_sided_cluster_bounds(
    per_layout: np.ndarray, *, seed: int, confidence: float = 0.9875, resamples: int = 10000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(per_layout), size=(resamples, len(per_layout)))
    means = per_layout[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence
    return float(np.quantile(means, alpha)), float(np.quantile(means, confidence))


def _two_sided_cluster_interval(per_layout: np.ndarray, *, seed: int, confidence: float = 0.95,
                                resamples: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(per_layout), size=(resamples, len(per_layout)))
    means = per_layout[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _model_probabilities(scores: torch.Tensor, variant: str, temperature: float = 0.15) -> torch.Tensor:
    logits = scores if variant == "bounded_preference_policy" else -scores / temperature
    return torch.softmax(logits, dim=-1)


def score_model(model: torch.nn.Module, data: dict[str, Any], *, variant: str,
                device: torch.device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    states = torch.as_tensor(data["inputs"]["states"], dtype=torch.float32, device=device)
    candidates_one = torch.as_tensor(ACTION_FEATURES, dtype=torch.float32, device=device)
    mask_one = torch.ones((1, 9), dtype=torch.bool, device=device)
    outputs: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            stop = min(start + batch_size, len(states))
            count = stop - start
            candidates = candidates_one.unsqueeze(0).expand(count, -1, -1)
            masks = mask_one.expand(count, -1)
            scores = model(states[start:stop], candidates, masks)
            probs = _model_probabilities(scores, variant)
            outputs.append(scores.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(probabilities)


def evaluate_model(model: torch.nn.Module, data: dict[str, Any], *, variant: str,
                   device: torch.device, bootstrap_seed: int = 171001) -> dict[str, Any]:
    scores, probabilities = score_model(model, data, variant=variant, device=device)
    selected = (np.argmax(scores, axis=1) if variant == "bounded_preference_policy"
                else np.argmin(scores, axis=1))
    targets = data["targets"]
    bank_regret = targets["per_bank_normalized_regret"]
    heldout_regret = bank_regret[:, 6:12, :].mean(axis=1)
    rows = np.arange(len(selected))
    selected_regret = heldout_regret[rows, selected]
    value, per_layout = _layout_macro(selected_regret, _layout_arrays(data["metadata"])[0])
    lower, upper = _one_sided_cluster_bounds(per_layout, seed=bootstrap_seed)
    ci95_lower, ci95_upper = _two_sided_cluster_interval(per_layout, seed=bootstrap_seed)

    geometry_action = np.asarray([row["geometry_action"] for row in data["metadata"]], dtype=np.int64)
    geometry_regret = heldout_regret[rows, geometry_action]
    layout_ids, layout_names = _layout_arrays(data["metadata"])
    geometry_mean, geometry_layout = _layout_macro(geometry_regret, layout_ids)
    difference = geometry_layout - per_layout
    improvement_lower, improvement_upper = _one_sided_cluster_bounds(
        difference, seed=bootstrap_seed + 1
    )
    improvement_ci95_lower, improvement_ci95_upper = _two_sided_cluster_interval(
        difference, seed=bootstrap_seed + 1
    )

    train_preferences = targets["per_bank_tie_split_optimal_mass"][:, 0:6, :].mean(axis=1)
    modal_action = np.argmax(train_preferences, axis=1)
    modal_regret = heldout_regret[rows, modal_action]
    modal_regret_mean, _ = _layout_macro(modal_regret, layout_ids)
    selected_in_optimal = (targets["per_bank_tie_split_optimal_mass"][:, 6:12, :].sum(axis=1) > 0)
    inclusion = selected_in_optimal[rows, selected].astype(np.float64)
    inclusion_mean, _ = _layout_macro(inclusion, layout_ids)
    nll = -np.sum(train_preferences * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    brier = np.sum((probabilities - train_preferences) ** 2, axis=1)
    nll_mean, _ = _layout_macro(nll, layout_ids)
    brier_mean, _ = _layout_macro(brier, layout_ids)
    expected_regret = np.sum(probabilities * heldout_regret, axis=1)
    expected_mean, _ = _layout_macro(expected_regret, layout_ids)

    return {
        "variant": variant,
        "queries": len(selected),
        "layouts": len(layout_names),
        "layout_macro_argmax_heldout_regret": value,
        "regret_one_sided_98_75_lower": lower,
        "regret_one_sided_98_75_upper": upper,
        "regret_ci95_lower": ci95_lower,
        "regret_ci95_upper": ci95_upper,
        "geometry_layout_macro_heldout_regret": geometry_mean,
        "training_half_empirical_modal_action_heldout_regret": modal_regret_mean,
        "mean_improvement_over_geometry": geometry_mean - value,
        "improvement_one_sided_98_75_lower": improvement_lower,
        "improvement_one_sided_98_75_upper": improvement_upper,
        "improvement_ci95_lower": improvement_ci95_lower,
        "improvement_ci95_upper": improvement_ci95_upper,
        "heldout_optimal_set_inclusion": inclusion_mean,
        "nll_against_training_half_empirical_preference": nll_mean,
        "brier_against_training_half_empirical_preference": brier_mean,
        "expected_regret_under_predicted_distribution": expected_mean,
        "per_layout_regret": per_layout.tolist(),
        "per_layout_geometry_regret": geometry_layout.tolist(),
        "per_layout_improvement": difference.tolist(),
    }


def permutation_consistency(model: torch.nn.Module, *, variant: str, state_rows: np.ndarray,
                            device: torch.device, seed: int = 171002,
                            fixtures: int = 16, permutations_per_fixture: int = 100) -> dict[str, Any]:
    states = torch.as_tensor(state_rows[:fixtures], dtype=torch.float32, device=device)
    candidate = torch.as_tensor(ACTION_FEATURES, dtype=torch.float32, device=device)
    mask = torch.ones(9, dtype=torch.bool, device=device)
    rng = np.random.default_rng(seed)
    max_deviation = 0.0
    flips = 0
    comparisons = 0
    model.eval()
    with torch.no_grad():
        for state in states:
            canonical_scores = model(state[None], candidate[None], mask[None])[0]
            canonical_probability = _model_probabilities(canonical_scores[None], variant)[0]
            canonical_choice = int(torch.argmax(canonical_probability).item())
            for _ in range(permutations_per_fixture):
                permutation = rng.permutation(9)
                perm = torch.as_tensor(permutation, dtype=torch.long, device=device)
                permuted_scores = model(state[None], candidate[perm][None], mask[perm][None])[0]
                permuted_probability = _model_probabilities(permuted_scores[None], variant)[0]
                restored = torch.empty_like(permuted_probability)
                restored[perm] = permuted_probability
                max_deviation = max(max_deviation, float(torch.max(torch.abs(restored - canonical_probability)).item()))
                maximum = float(permuted_probability.max().item())
                tied_candidates = torch.nonzero(permuted_probability == maximum).flatten().tolist()
                permuted_choice = min(int(permutation[index]) for index in tied_candidates)
                flips += int(permuted_choice != canonical_choice)
                comparisons += 1
    return {"fixtures": len(states), "permutations_per_fixture": permutations_per_fixture,
            "comparisons": comparisons, "max_probability_deviation": max_deviation,
            "semantic_choice_flips": flips}


def measure_cpu_decision_latency(model: torch.nn.Module, *, variant: str, state_row: np.ndarray,
                                 warmup: int = 50, repetitions: int = 500) -> dict[str, float | int | str]:
    """Measure state-tensor-to-nine-candidate scoring/choice latency, not simulator time."""
    device = torch.device("cpu")
    model.eval()
    candidate = torch.as_tensor(ACTION_FEATURES, dtype=torch.float32, device=device)[None]
    mask = torch.ones((1, 9), dtype=torch.bool, device=device)
    state_array = np.asarray(state_row, dtype=np.float32).reshape(1, 52)
    durations: list[float] = []
    with torch.no_grad():
        for index in range(warmup + repetitions):
            started = time.perf_counter_ns()
            state = torch.as_tensor(state_array, dtype=torch.float32, device=device)
            scores = model(state, candidate, mask)
            probabilities = _model_probabilities(scores, variant)
            if variant == "bounded_preference_policy":
                int(torch.argmax(probabilities, dim=-1).item())
            else:
                int(torch.argmin(scores, dim=-1).item())
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            if index >= warmup:
                durations.append(elapsed_ms)
    values = np.asarray(durations, dtype=np.float64)
    return {
        "scope": "state_tensor_to_candidate_scoring_distribution_and_choice_cpu_only",
        "repetitions": repetitions,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
    }


def _training_weights(metadata: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    ids, _ = _layout_arrays(metadata)
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float32)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _fit_one(
    train: dict[str, Any], selection: dict[str, Any], *, variant: str, seed: int,
    config: dict[str, Any], device: torch.device, started: float, time_cap: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    model = CorridorCandidateScorer(
        "policy" if variant == "bounded_preference_policy" else "value",
        hidden_state=int(config["models"]["common"]["hidden_state"]),
        embedding=int(config["models"]["common"]["embedding"]),
    ).to(device)
    if model_parameter_count(model) > int(config["models"]["common"]["maximum_parameters"]):
        raise ValueError("Phase 12c-C model exceeds its parameter cap")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["models"]["fitting"]["learning_rate"]),
        weight_decay=float(config["models"]["fitting"]["weight_decay"]),
    )
    train_states = torch.as_tensor(train["inputs"]["states"], dtype=torch.float32, device=device)
    selection_states = torch.as_tensor(selection["inputs"]["states"], dtype=torch.float32, device=device)
    candidate_features_tensor = torch.as_tensor(ACTION_FEATURES, dtype=torch.float32, device=device)
    train_candidates = candidate_features_tensor.unsqueeze(0).expand(len(train_states), -1, -1)
    train_mask = torch.ones((len(train_states), 9), dtype=torch.bool, device=device)
    target_key = "preference_target" if variant == "bounded_preference_policy" else "mean_regret_target"
    target = torch.as_tensor(train["targets"][target_key], dtype=torch.float32, device=device)
    weights = _training_weights(train["metadata"], device)
    selection_candidates = candidate_features_tensor.unsqueeze(0).expand(len(selection_states), -1, -1)
    selection_mask = torch.ones((len(selection_states), 9), dtype=torch.bool, device=device)
    batch = int(config["models"]["fitting"]["batch_decision_sets"])
    epochs = int(config["models"]["fitting"]["max_epochs"])
    patience = int(config["models"]["fitting"]["early_stopping_patience"])
    best_metric, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history: list[dict[str, float | int]] = []
    best_model = None

    for epoch in range(1, epochs + 1):
        if time.monotonic() - started > time_cap:
            raise TimeoutError("Phase 12c-C fit wall-time cap reached")
        model.train()
        order = np_rng.permutation(len(train_states))
        losses: list[float] = []
        for offset in range(0, len(order), batch):
            indices = torch.as_tensor(order[offset:offset + batch], dtype=torch.long, device=device)
            scores = model(train_states[indices], train_candidates[indices], train_mask[indices])
            if variant == "direct_candidate_value":
                row_loss = ((scores - target[indices]) ** 2).mean(dim=1)
            else:
                row_loss = -(target[indices] * F.log_softmax(scores, dim=-1)).sum(dim=1)
            loss = (row_loss * weights[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["models"]["fitting"]["gradient_norm_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        selection_metrics = evaluate_model(model, selection, variant=variant, device=device,
                                           bootstrap_seed=171001)
        metric = float(selection_metrics["layout_macro_argmax_heldout_regret"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "selection_heldout_regret": metric})
        if metric < best_metric - 1e-12:
            best_metric, best_epoch, best_state, stale = metric, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError("Phase 12c-C fit did not produce a selected checkpoint")
    model.load_state_dict(best_state)
    summary = {
        "variant": variant,
        "fit_seed": seed,
        "parameter_count": model_parameter_count(model),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_selection_heldout_regret": best_metric,
        "selection_metrics": evaluate_model(model, selection, variant=variant, device=device),
        "history": history,
    }
    return model, summary


def train_phase12c_c(
    train: dict[str, Any], selection: dict[str, Any], *, config: dict[str, Any],
    output_dir: Path, time_cap_seconds: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if train["manifest"].get("split") != "archived_training" or selection["manifest"].get("split") != "model_selection":
        raise ValueError("training/selection split manifests have unexpected stage identities")
    if selection["manifest"].get("offline_gate_labels_collected") is not False:
        raise ValueError("model-selection manifest indicates offline-gate access")
    torch.set_num_threads(1)
    device = torch.device("cpu")
    started = time.monotonic()
    all_fits: list[dict[str, Any]] = []
    selected_models: dict[str, tuple[torch.nn.Module, dict[str, Any]]] = {}
    for variant in ("direct_candidate_value", "bounded_preference_policy"):
        best: tuple[float, torch.nn.Module, dict[str, Any]] | None = None
        for seed in config["models"]["fitting"]["seeds"]:
            model, result = _fit_one(
                train, selection, variant=variant, seed=int(seed), config=config,
                device=device, started=started, time_cap=time_cap_seconds,
            )
            all_fits.append(result)
            score = float(result["best_selection_heldout_regret"])
            if best is None or score < best[0]:
                best = score, model, result
        assert best is not None
        selected_models[variant] = best[1], best[2]

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    selected_results: dict[str, Any] = {}
    checkpoint_hashes: dict[str, str] = {}
    for variant, (model, result) in selected_models.items():
        path = checkpoint_dir / f"{variant}_seed{result['fit_seed']}.pt"
        torch.save({"variant": variant, "fit_seed": result["fit_seed"],
                    "state_dict": model.state_dict(),
                    "config_sha256": canonical_config_sha256(config)}, path)
        checkpoint_hashes[str(path.relative_to(output_dir))] = sha256(path)
        selected_results[variant] = {
            key: value for key, value in result.items() if key != "history"
        }
        selected_results[variant]["permutation"] = permutation_consistency(
            model, variant=variant, state_rows=selection["inputs"]["states"], device=device
        )

    result = {
        "schema": "phase12cC_student_model_selection_v1",
        "training_layouts": train["manifest"]["layout_count"],
        "training_queries": train["manifest"]["query_count"],
        "selection_layouts": selection["manifest"]["layout_count"],
        "selection_queries": selection["manifest"]["query_count"],
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "elapsed_wall_seconds": time.monotonic() - started,
        "all_fit_runs": all_fits,
        "selected_models": selected_results,
        "checkpoint_sha256": checkpoint_hashes,
        "offline_gate_read": False,
        "development_accessed": False,
        "dagger_accessed": False,
        "reserve_accessed": False,
    }
    results_path = output_dir / "selection_results.json"
    results_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["selection_results_sha256"] = sha256(results_path)
    return result
