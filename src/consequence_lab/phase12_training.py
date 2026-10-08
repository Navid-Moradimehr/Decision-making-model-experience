"""Deterministic small-model fitting and offline evaluation for Phase 12."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from consequence_lab.phase12_model import CorridorCandidateScorer, model_parameter_count


VARIANTS = ("value", "hard", "soft_0.05", "soft_0.15", "soft_0.30", "expected_regret")


def load_phase12_split(root: Path) -> dict[str, np.ndarray]:
    """Load student tensors and separate targets; never merge labels into model input."""
    with np.load(root / "student_inputs.npz", allow_pickle=False) as data:
        inputs = {key: data[key] for key in ("states", "sequences", "candidate_mask")}
        if set(data.files) != {"states", "sequences", "candidate_mask"}:
            raise ValueError("student input archive violates the runtime allowlist")
    with np.load(root / "teacher_targets.npz", allow_pickle=False) as data:
        targets = {key: data[key] for key in data.files}
    with np.load(root / "sample_metadata.npz", allow_pickle=False) as data:
        metadata = {key: data[key] for key in data.files}
    n = inputs["states"].shape[0]
    if inputs["states"].shape != (n, 52) or inputs["sequences"].shape != (n, 128, 8):
        raise ValueError("Phase 12 student input tensor shape mismatch")
    if inputs["candidate_mask"].shape != (n, 128):
        raise ValueError("Phase 12 candidate mask shape mismatch")
    if any(value.shape[0] != n for value in targets.values()) or any(value.shape[0] != n for value in metadata.values()):
        raise ValueError("student inputs, targets, metadata have different sample counts")
    return {**inputs, **{f"target_{k}": v for k, v in targets.items()}, **{f"meta_{k}": v for k, v in metadata.items()}}


def _tensors(data: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    seq = torch.as_tensor(data["sequences"].astype(np.int64), device=device)
    return {
        "states": torch.as_tensor(data["states"], dtype=torch.float32, device=device),
        "candidates": F.one_hot(seq, num_classes=10).to(torch.float32).flatten(start_dim=-2),
        "mask": torch.as_tensor(data["candidate_mask"], dtype=torch.bool, device=device),
        "regret": torch.as_tensor(data["target_normalized_regret"], dtype=torch.float32, device=device),
        "optimal": torch.as_tensor(data["target_optimal_mask"], dtype=torch.bool, device=device),
        "selected": torch.as_tensor(data["target_teacher_selected_index"], dtype=torch.long, device=device),
        "layout": torch.as_tensor(data["meta_layout_index"], dtype=torch.long, device=device),
    }


def _macro_weights(layout_ids: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(layout_ids, return_counts=True)
    count_by_layout = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_layout[int(item)] for item in layout_ids], dtype=np.float32)
    # Normalize to unit mean so configured optimizer scale remains stable.
    return weights / weights.mean()


def _probabilities(scores: torch.Tensor, mask: torch.Tensor, kind: str) -> torch.Tensor:
    logits = -scores if kind == "value" else scores
    logits = logits.masked_fill(~mask, -torch.inf)
    return torch.softmax(logits, dim=-1)


def _stable_choice(scores: np.ndarray, sequences: np.ndarray, mask: np.ndarray, kind: str) -> np.ndarray:
    chosen = np.empty(len(scores), dtype=np.int64)
    for row in range(len(scores)):
        indices = np.flatnonzero(mask[row])
        values = scores[row, indices]
        target = np.min(values) if kind == "value" else np.max(values)
        tied = indices[values == target]
        # Candidate-content tie break, independent of row/proposal order.
        chosen[row] = min(tied.tolist(), key=lambda index: tuple(int(x) for x in sequences[row, index]))
    return chosen


def _ranking_accuracy(scores: np.ndarray, targets: np.ndarray, mask: np.ndarray, kind: str) -> tuple[float, float]:
    concordant = discordant = predicted_ties = target_ties = comparable = total_pairs = 0
    predicted_ties_on_untied_targets = 0
    for row in range(len(scores)):
        valid = np.flatnonzero(mask[row])
        for offset, left in enumerate(valid[:-1]):
            right = valid[offset + 1:]
            true_delta = targets[row, right] - targets[row, left]
            pred_delta = scores[row, right] - scores[row, left]
            if kind == "policy":
                pred_delta = -pred_delta
            active = true_delta != 0
            target_ties += int((~active).sum())
            predicted_ties += int((pred_delta == 0).sum())
            total_pairs += len(right)
            if not active.any():
                continue
            comparable += int(active.sum())
            predicted_ties_on_untied_targets += int((pred_delta[active] == 0).sum())
            product = true_delta[active] * pred_delta[active]
            concordant += int((product > 0).sum())
            discordant += int((product < 0).sum())
    # For pairwise accuracy, predicted ties on true non-tied pairs earn half.
    accuracy = (concordant + 0.5 * predicted_ties_on_untied_targets) / comparable if comparable else float("nan")
    denom = (total_pairs - target_ties) * (total_pairs - predicted_ties)
    tau_b = (concordant - discordant) / np.sqrt(denom) if denom > 0 else float("nan")
    return float(accuracy), float(tau_b)


def evaluate_offline(
    model: CorridorCandidateScorer,
    data: dict[str, np.ndarray],
    *, kind: str,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, Any]:
    tensors = _tensors(data, device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(data["states"]), batch_size):
            stop = min(start + batch_size, len(data["states"]))
            chunks.append(model(tensors["states"][start:stop], tensors["candidates"][start:stop], tensors["mask"][start:stop]).cpu().numpy())
    scores = np.concatenate(chunks)
    mask = data["candidate_mask"].astype(bool)
    regret = data["target_normalized_regret"]
    optimal = data["target_optimal_mask"].astype(bool)
    chosen = _stable_choice(scores, data["sequences"], mask, kind)
    row_ids = np.arange(len(chosen))
    selected_regret = regret[row_ids, chosen]
    if kind == "value":
        probabilities = np.zeros_like(scores, dtype=np.float64)
        safe = np.where(mask, -scores, -np.inf)
        safe -= np.max(safe, axis=1, keepdims=True)
        exp = np.where(mask, np.exp(safe), 0.0)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
    else:
        safe = np.where(mask, scores, -np.inf)
        safe -= np.max(safe, axis=1, keepdims=True)
        exp = np.where(mask, np.exp(safe), 0.0)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
    uniform_optimal = optimal / optimal.sum(axis=1, keepdims=True)
    nll = -np.sum(uniform_optimal * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1)
    brier = np.sum((probabilities - uniform_optimal) ** 2, axis=1)
    topk = min(5, mask.shape[1])
    ranked = np.argsort(np.where(mask, -scores if kind == "policy" else scores, np.inf), axis=1)[:, :topk]
    topk_recall = np.mean([optimal[row, ranked[row]].any() for row in range(len(chosen))])
    pair_acc, tau_b = _ranking_accuracy(scores, regret, mask, kind)
    layout = data["meta_layout_index"]
    per_layout = [float(selected_regret[layout == item].mean()) for item in np.unique(layout)]
    expected_regret = np.sum(probabilities * regret, axis=1)
    return {
        "samples": len(chosen),
        "layouts": len(per_layout),
        "layout_macro_mean_selected_normalized_rank_regret": float(np.mean(per_layout)),
        "sample_mean_selected_normalized_rank_regret": float(np.mean(selected_regret)),
        "expected_normalized_rank_regret": float(np.mean(expected_regret)),
        "optimal_set_top1_agreement": float(np.mean(optimal[row_ids, chosen])),
        "exact_teacher_selected_candidate_agreement": float(np.mean(chosen == data["target_teacher_selected_index"])),
        "top_5_optimal_set_recall": float(topk_recall),
        "tie_aware_pairwise_rank_accuracy": pair_acc,
        "kendall_tau_b": tau_b,
        "nll_against_uniform_optimal_set": float(np.mean(nll)),
        "brier_against_uniform_optimal_set": float(np.mean(brier)),
        "selected_indices": chosen,
        "scores": scores,
        "probabilities": probabilities,
        "selected_regret": selected_regret,
    }


def _batch_loss(variant: str, scores: torch.Tensor, data: dict[str, torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    mask = data["mask"]
    regrets = data["regret"]
    row_weights = weights / weights.sum().clamp_min(1e-8)
    if variant == "value":
        safe_scores = torch.where(mask, scores, regrets)
        element = (safe_scores - regrets) ** 2
        per_row = (element * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    elif variant == "hard":
        logits = scores.masked_fill(~mask, -torch.inf)
        probs = data["optimal"].float()
        probs = probs / probs.sum(dim=1, keepdim=True)
        log_probs = torch.where(mask, F.log_softmax(logits, dim=1), 0.0)
        per_row = -(probs * log_probs).sum(dim=1)
    elif variant.startswith("soft_"):
        tau = float(variant.split("_")[1])
        target_logits = (-regrets / tau).masked_fill(~mask, -torch.inf)
        target = torch.softmax(target_logits, dim=1)
        logits = scores.masked_fill(~mask, -torch.inf)
        log_probs = torch.where(mask, F.log_softmax(logits, dim=1), 0.0)
        per_row = -(target * log_probs).sum(dim=1)
    elif variant == "expected_regret":
        probs = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=1)
        per_row = (probs * regrets).sum(dim=1)
    else:
        raise ValueError(f"unknown training variant {variant}")
    return (per_row * row_weights).sum()


def fit_variant(
    train: dict[str, np.ndarray], val: dict[str, np.ndarray], *, variant: str, fit_seed: int,
    config: dict[str, Any], output_dir: Path, device: torch.device,
) -> dict[str, Any]:
    random.seed(fit_seed)
    np.random.seed(fit_seed)
    torch.manual_seed(fit_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fit_seed)
    torch.use_deterministic_algorithms(True)
    model_kind = "value" if variant == "value" else "policy"
    model = CorridorCandidateScorer(model_kind).to(device)
    if model_parameter_count(model) >= config["models"]["common"]["parameters_max"]:
        raise ValueError("model exceeds frozen parameter-count limit")
    train_t = _tensors(train, device)
    val_inputs = _tensors(val, device)
    weights_np = _macro_weights(train["meta_layout_index"])
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    fit_cfg = config["models"]["fit"]
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=fit_cfg["learning_rate"],
        weight_decay=fit_cfg["weight_decay"],
    )
    batch_size = int(fit_cfg["batch_decision_sets"])
    best_score = float("inf")
    best_nll = float("inf")
    best_state = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    started = time.monotonic()
    for epoch in range(int(fit_cfg["max_epochs"])):
        model.train()
        order = np.random.permutation(len(train["states"]))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            indices_np = order[start:start + batch_size]
            indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
            batch_data = {key: value[indices] for key, value in train_t.items()}
            batch_weights = weights[indices]
            optim.zero_grad(set_to_none=True)
            output = model(batch_data["states"], batch_data["candidates"], batch_data["mask"])
            loss = _batch_loss(variant, output, batch_data, batch_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(fit_cfg["gradient_norm_clip"]))
            optim.step()
            epoch_losses.append(float(loss.detach().cpu()))
        validation = evaluate_offline(model, val, kind=model_kind, device=device, batch_size=batch_size)
        score = validation["layout_macro_mean_selected_normalized_rank_regret"]
        nll = validation["nll_against_uniform_optimal_set"]
        history.append({"epoch": epoch + 1, "training_loss": float(np.mean(epoch_losses)), "validation_regret": score, "validation_nll": nll})
        if score < best_score - 1e-10 or (abs(score - best_score) <= 1e-10 and nll < best_nll):
            best_score, best_nll = score, nll
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(fit_cfg["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)
    final_metrics = evaluate_offline(model, val, kind=model_kind, device=device, batch_size=batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{variant}_seed{fit_seed}.pt"
    torch.save({"kind": model_kind, "variant": variant, "fit_seed": fit_seed, "state_dict": best_state}, checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    history_path = output_dir / f"{variant}_seed{fit_seed}_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {
        "variant": variant,
        "fit_seed": fit_seed,
        "kind": model_kind,
        "parameters": model_parameter_count(model),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation": {key: value for key, value in final_metrics.items() if key not in {"selected_indices", "scores", "probabilities", "selected_regret"}},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "history": str(history_path),
        "wall_seconds": time.monotonic() - started,
    }


def train_phase12(
    train_dir: Path, val_dir: Path, output_dir: Path, *, config: dict[str, Any], device_name: str = "auto",
) -> dict[str, Any]:
    train = load_phase12_split(train_dir)
    val = load_phase12_split(val_dir)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment")
    fit_seeds = config["models"]["fit"]["fit_seeds"]
    rows = []
    started = time.monotonic()
    for variant in VARIANTS:
        for seed in fit_seeds:
            rows.append(fit_variant(
                train, val, variant=variant, fit_seed=int(seed), config=config,
                output_dir=output_dir / "checkpoints", device=device,
            ))
            if time.monotonic() - started > 2 * 60 * 60:
                raise TimeoutError("Phase 12 student-training 2-hour cap reached")
    # Frozen selector: val macro regret, then NLL, then simple loss order, then seed.
    variant_order = {name: index for index, name in enumerate(VARIANTS)}
    chosen = min(rows, key=lambda row: (
        row["validation"]["layout_macro_mean_selected_normalized_rank_regret"],
        row["validation"]["nll_against_uniform_optimal_set"],
        variant_order[row["variant"]], row["fit_seed"],
    ))
    result = {
        "schema": "phase12_training_results_v1",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "train_samples": len(train["states"]),
        "train_layouts": int(len(np.unique(train["meta_layout_index"]))),
        "validation_samples": len(val["states"]),
        "validation_layouts": int(len(np.unique(val["meta_layout_index"]))),
        "all_runs": rows,
        "selected": chosen,
        "offline_competence_gate": chosen["validation"]["layout_macro_mean_selected_normalized_rank_regret"] <= 0.15,
        "training_wall_seconds": time.monotonic() - started,
        "reserve_accessed": False,
    }
    result_path = output_dir / "results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
