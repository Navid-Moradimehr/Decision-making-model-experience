"""Action-grouped teacher targets and compact Phase 12b students."""

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
from consequence_lab.phase12_teacher import candidate_features
from consequence_lab.phase12_training import _macro_weights, load_phase12_split


VARIANTS = ("sequence_policy_control", "action_group_policy", "direct_action_value")
ACTION_COUNT = 9
PADDING_ACTION = 9


def derive_action_group_targets(
    sequences: np.ndarray, candidate_mask: np.ndarray, objective_keys: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct exact action-group ranks from best lexicographic plan per first action."""
    sequences = np.asarray(sequences)
    mask = np.asarray(candidate_mask, dtype=bool)
    keys = np.asarray(objective_keys, dtype=np.float64)
    n, count = mask.shape
    if sequences.shape != (n, count, 8) or keys.shape != (n, count, 3):
        raise ValueError("Phase 12b sequence/objective shapes do not match")
    # Sequence suffixes are PAD-filled by design (especially on late episode
    # queries with a shortened planning horizon). Only t=0 defines the group.
    first_actions = sequences[:, :, 0].astype(np.int64)
    if np.any(first_actions[mask] >= ACTION_COUNT):
        raise ValueError("valid candidate has padding/invalid first action")
    if np.any(sequences > PADDING_ACTION):
        raise ValueError("candidate sequence contains an invalid action token")

    group_keys = np.empty((n, ACTION_COUNT, 3), dtype=np.float64)
    action_mask = np.zeros((n, ACTION_COUNT), dtype=bool)
    regrets = np.zeros((n, ACTION_COUNT), dtype=np.float32)
    optimal = np.zeros((n, ACTION_COUNT), dtype=bool)
    ranks = np.zeros((n, ACTION_COUNT), dtype=np.float32)
    for row in range(n):
        actions = first_actions[row]
        valid_actions = sorted(set(int(a) for a in actions[mask[row]]))
        if valid_actions != list(range(ACTION_COUNT)):
            raise ValueError("each teacher candidate set must cover all nine first actions")
        for action in valid_actions:
            group = np.flatnonzero(mask[row] & (actions == action))
            best = min(tuple(float(v) for v in keys[row, item]) for item in group)
            group_keys[row, action] = best
            action_mask[row, action] = True
        values = [tuple(float(v) for v in group_keys[row, a]) for a in valid_actions]
        for action in valid_actions:
            value = tuple(float(v) for v in group_keys[row, action])
            strictly_better = sum(other < value for other in values)
            tied = sum(other == value for other in values)
            ranks[row, action] = strictly_better + (tied + 1) / 2.0
        best_rank = float(ranks[row, action_mask[row]].min())
        regrets[row, action_mask[row]] = (ranks[row, action_mask[row]] - best_rank) / 8.0
        best_key = min(values)
        optimal[row] = np.asarray([
            action_mask[row, action] and tuple(group_keys[row, action]) == best_key
            for action in range(ACTION_COUNT)
        ])
    return {
        "action_group_objective_key": group_keys,
        "action_mask": action_mask,
        "action_group_rank": ranks,
        "action_group_regret": regrets,
        "optimal_action_mask": optimal,
    }


def first_action_ids(sequences: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    sequences = np.asarray(sequences)
    mask = np.asarray(candidate_mask, dtype=bool)
    actions = sequences[:, :, 0].astype(np.int64)
    if actions.shape != mask.shape or np.any(actions[mask] >= ACTION_COUNT):
        raise ValueError("first-action IDs/mask invalid")
    return np.where(mask, actions, 0)


def action_candidate_sequences(n: int) -> np.ndarray:
    """Canonical set of nine immediate actions, with no imagined suffix."""
    if n <= 0:
        raise ValueError("query count must be positive")
    result = np.full((n, ACTION_COUNT, 8), PADDING_ACTION, dtype=np.uint8)
    result[:, :, 0] = np.arange(ACTION_COUNT, dtype=np.uint8)[None, :]
    return result


def merge_training_sources(train_dir: Path, former_validation_dir: Path) -> dict[str, np.ndarray]:
    """Use both previously opened Phase 12 splits as training-only in 12b."""
    train = load_phase12_split(train_dir)
    old_val = load_phase12_split(former_validation_dir)
    merged: dict[str, np.ndarray] = {}
    for key in train:
        left, right = train[key], old_val[key]
        if key == "meta_layout_index":
            right = right + int(train[key].max()) + 1
        if left.ndim == 0 or right.ndim == 0 or left.shape[1:] != right.shape[1:]:
            raise ValueError(f"incompatible Phase 12 training source field {key}")
        merged[key] = np.concatenate((left, right), axis=0)
    if len(np.unique(merged["meta_layout_index"])) != 160:
        raise ValueError("Phase 12b fitting pool must contain exactly 160 distinct layouts")
    return merged


def _tensors(data: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    sequence_ids = torch.as_tensor(data["sequences"].astype(np.int64), device=device)
    action_sequences = action_candidate_sequences(len(data["states"]))
    action_ids = torch.as_tensor(action_sequences.astype(np.int64), device=device)
    keys = derive_action_group_targets(
        data["sequences"], data["candidate_mask"], data["target_objective_key"],
    )
    return {
        "states": torch.as_tensor(data["states"], dtype=torch.float32, device=device),
        "sequences": F.one_hot(sequence_ids, num_classes=10).to(torch.float32).flatten(start_dim=-2),
        "sequence_mask": torch.as_tensor(data["candidate_mask"], dtype=torch.bool, device=device),
        "sequence_regret": torch.as_tensor(data["target_normalized_regret"], dtype=torch.float32, device=device),
        "action_features": torch.as_tensor(
            candidate_features(action_sequences.reshape(-1, 8)).reshape(len(data["states"]), ACTION_COUNT, 80),
            dtype=torch.float32, device=device,
        ),
        "action_mask": torch.as_tensor(keys["action_mask"], dtype=torch.bool, device=device),
        "action_regret": torch.as_tensor(keys["action_group_regret"], dtype=torch.float32, device=device),
        "optimal_action": torch.as_tensor(keys["optimal_action_mask"], dtype=torch.bool, device=device),
        "first_action": torch.as_tensor(first_action_ids(data["sequences"], data["candidate_mask"]), dtype=torch.long, device=device),
        "teacher_first_action": torch.as_tensor(
            data["sequences"][np.arange(len(data["states"])), data["target_teacher_selected_index"], 0].astype(np.int64),
            dtype=torch.long, device=device,
        ),
        "layout": torch.as_tensor(data["meta_layout_index"], dtype=torch.long, device=device),
    }


def _group_probabilities(sequence_probabilities: torch.Tensor, first_actions: torch.Tensor) -> torch.Tensor:
    n, _ = first_actions.shape
    result = sequence_probabilities.new_zeros((n, ACTION_COUNT))
    return result.scatter_add(1, first_actions, sequence_probabilities)


def _content_tie_action(probabilities: np.ndarray) -> np.ndarray:
    # Canonical action sequence content is [action, PAD, ..., PAD], so exact
    # ties resolve independently of proposal/candidate order by action ID.
    return np.argmax(probabilities, axis=1)


def evaluate_phase12b(model: CorridorCandidateScorer, data: dict[str, np.ndarray], *, variant: str,
                      device: torch.device, batch_size: int = 32) -> dict[str, Any]:
    tensors = _tensors(data, device)
    model.eval()
    score_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data["states"]), batch_size):
            stop = min(start + batch_size, len(data["states"]))
            state = tensors["states"][start:stop]
            if variant == "direct_action_value":
                scores = model(state, tensors["action_features"][start:stop], tensors["action_mask"][start:stop])
            else:
                scores = model(state, tensors["sequences"][start:stop], tensors["sequence_mask"][start:stop])
            score_rows.append(scores.cpu().numpy())
    scores = np.concatenate(score_rows)
    action_regret = derive_action_group_targets(
        data["sequences"], data["candidate_mask"], data["target_objective_key"],
    )["action_group_regret"]
    optimal = derive_action_group_targets(
        data["sequences"], data["candidate_mask"], data["target_objective_key"],
    )["optimal_action_mask"]
    first = first_action_ids(data["sequences"], data["candidate_mask"])
    if variant == "direct_action_value":
        logits = -scores
        action_probabilities = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
        selected_action = np.argmin(scores, axis=1)
    elif variant == "action_group_policy":
        safe = scores - np.max(scores, axis=1, keepdims=True)
        seq_prob = np.exp(safe)
        seq_prob /= seq_prob.sum(axis=1, keepdims=True)
        action_probabilities = np.zeros((len(scores), ACTION_COUNT), dtype=np.float64)
        np.add.at(action_probabilities, (np.arange(len(scores))[:, None], first), seq_prob)
        selected_action = _content_tie_action(action_probabilities)
    elif variant == "sequence_policy_control":
        safe = scores - np.max(scores, axis=1, keepdims=True)
        seq_prob = np.exp(safe)
        seq_prob /= seq_prob.sum(axis=1, keepdims=True)
        action_probabilities = np.zeros((len(scores), ACTION_COUNT), dtype=np.float64)
        np.add.at(action_probabilities, (np.arange(len(scores))[:, None], first), seq_prob)
        selected_sequence = np.argmax(scores, axis=1)
        selected_action = first[np.arange(len(scores)), selected_sequence]
    else:
        raise ValueError(f"unknown Phase 12b variant: {variant}")

    rows = np.arange(len(scores))
    selected_regret = action_regret[rows, selected_action]
    optimal_count = optimal.sum(axis=1)
    target = optimal / optimal_count[:, None]
    nll = -np.sum(target * np.log(np.clip(action_probabilities, 1e-12, 1.0)), axis=1)
    brier = np.sum((action_probabilities - target) ** 2, axis=1)
    top3 = np.argpartition(-action_probabilities, kth=2, axis=1)[:, :3]
    top3_recall = np.mean([optimal[i, top3[i]].any() for i in range(len(rows))])
    layout = data["meta_layout_index"]
    per_layout = [float(selected_regret[layout == item].mean()) for item in np.unique(layout)]
    expected_regret = np.sum(action_probabilities * action_regret, axis=1)
    teacher_action = tensors["teacher_first_action"].cpu().numpy()
    return {
        "variant": variant,
        "queries": len(rows),
        "layouts": len(per_layout),
        "layout_macro_mean_normalized_action_group_regret": float(np.mean(per_layout)),
        "sample_mean_normalized_action_group_regret": float(np.mean(selected_regret)),
        "expected_normalized_action_group_regret": float(np.mean(expected_regret)),
        "teacher_first_action_exact_agreement": float(np.mean(selected_action == teacher_action)),
        "optimal_action_set_top1_inclusion": float(np.mean(optimal[rows, selected_action])),
        "top3_optimal_action_set_recall": float(top3_recall),
        "action_set_nll": float(np.mean(nll)),
        "action_set_brier": float(np.mean(brier)),
        "selected_action": selected_action,
        "action_probabilities": action_probabilities,
        "selected_action_regret": selected_regret,
    }


def _loss(variant: str, scores: torch.Tensor, batch: dict[str, torch.Tensor], row_weights: torch.Tensor,
          *, temperature: float) -> torch.Tensor:
    if variant == "sequence_policy_control":
        mask = batch["sequence_mask"]
        target = torch.softmax((-batch["sequence_regret"] / temperature).masked_fill(~mask, -torch.inf), dim=-1)
        log_prob = F.log_softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
        per_query = -(target * torch.where(mask, log_prob, 0.0)).sum(dim=-1)
    elif variant == "action_group_policy":
        seq_mask = batch["sequence_mask"]
        seq_prob = torch.softmax(scores.masked_fill(~seq_mask, -torch.inf), dim=-1)
        action_prob = _group_probabilities(seq_prob, batch["first_action"])
        target_logits = (-batch["action_regret"] / temperature).masked_fill(~batch["action_mask"], -torch.inf)
        target = torch.softmax(target_logits, dim=-1)
        per_query = -(target * torch.log(action_prob.clamp_min(1e-12))).sum(dim=-1)
    elif variant == "direct_action_value":
        mask = batch["action_mask"]
        residual = torch.where(mask, scores - batch["action_regret"], 0.0)
        per_query = residual.square().sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    else:
        raise ValueError(f"unknown Phase 12b variant: {variant}")
    return (per_query * row_weights).sum() / row_weights.sum().clamp_min(1e-8)


def fit_phase12b_variant(train: dict[str, np.ndarray], validation: dict[str, np.ndarray], *, variant: str,
                         fit_seed: int, config: dict[str, Any], output_dir: Path,
                         device: torch.device) -> dict[str, Any]:
    random.seed(fit_seed)
    np.random.seed(fit_seed)
    torch.manual_seed(fit_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fit_seed)
    torch.use_deterministic_algorithms(True)
    kind = "value" if variant == "direct_action_value" else "policy"
    model = CorridorCandidateScorer(kind).to(device)
    if model_parameter_count(model) > int(config["models"]["common"]["parameters_max"]):
        raise ValueError("model exceeds Phase 12b parameter cap")
    train_t, val_t = _tensors(train, device), _tensors(validation, device)
    weights = torch.as_tensor(_macro_weights(train["meta_layout_index"]), dtype=torch.float32, device=device)
    fit = config["models"]["fit"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=fit["learning_rate"], weight_decay=fit["weight_decay"])
    batch_size = int(fit["batch_decision_sets"])
    best_state, best_score, best_nll, best_epoch, stale = None, float("inf"), float("inf"), -1, 0
    history: list[dict[str, float]] = []
    start_time = time.monotonic()
    for epoch in range(int(fit["max_epochs"])):
        model.train()
        order = np.random.permutation(len(train["states"]))
        losses = []
        for offset in range(0, len(order), batch_size):
            idx_np = order[offset:offset + batch_size]
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=device)
            batch = {name: tensor[idx] for name, tensor in train_t.items()}
            if variant == "direct_action_value":
                scores = model(batch["states"], batch["action_features"], batch["action_mask"])
            else:
                scores = model(batch["states"], batch["sequences"], batch["sequence_mask"])
            loss = _loss(variant, scores, batch, weights[idx], temperature=float(config["models"]["fit"]["temperature"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(fit["gradient_norm_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_metrics = evaluate_phase12b(model, validation, variant=variant, device=device, batch_size=batch_size)
        score, nll = validation_metrics["layout_macro_mean_normalized_action_group_regret"], validation_metrics["action_set_nll"]
        history.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses)), "validation_action_regret": score, "validation_action_nll": nll})
        if score < best_score - 1e-10 or (abs(score - best_score) <= 1e-10 and nll < best_nll):
            best_score, best_nll, best_epoch = score, nll, epoch + 1
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= int(fit["early_stopping_patience"]):
            break
    if best_state is None:
        raise RuntimeError("Phase 12b produced no checkpoint")
    model.load_state_dict(best_state)
    final_metrics = evaluate_phase12b(model, validation, variant=variant, device=device, batch_size=batch_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{variant}_seed{fit_seed}.pt"
    torch.save({"variant": variant, "kind": kind, "fit_seed": fit_seed, "state_dict": best_state}, checkpoint)
    history_file = output_dir / f"{variant}_seed{fit_seed}_history.json"
    history_file.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {
        "variant": variant,
        "fit_seed": fit_seed,
        "kind": kind,
        "parameter_count": model_parameter_count(model),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation": {k: v for k, v in final_metrics.items() if k not in {"selected_action", "action_probabilities", "selected_action_regret"}},
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper(),
        "history": str(history_file),
        "wall_seconds": time.monotonic() - start_time,
    }


def train_phase12b(train: dict[str, np.ndarray], validation: dict[str, np.ndarray], *, config: dict[str, Any],
                   output_dir: Path, device_name: str = "auto") -> dict[str, Any]:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this pinned Python")
    start_time = time.monotonic()
    runs = []
    for variant in VARIANTS:
        for seed in config["models"]["fit"]["seeds"]:
            runs.append(fit_phase12b_variant(
                train, validation, variant=variant, fit_seed=int(seed), config=config,
                output_dir=output_dir / "checkpoints", device=device,
            ))
            if time.monotonic() - start_time > float(config["resource_caps"]["student_fit_wall_seconds"]):
                raise TimeoutError("Phase 12b fit wall-time cap reached")
    variant_order = {variant: index for index, variant in enumerate(VARIANTS)}
    selected = min(runs, key=lambda run: (
        run["validation"]["layout_macro_mean_normalized_action_group_regret"],
        run["validation"]["action_set_nll"], variant_order[run["variant"]], run["fit_seed"],
    ))
    result = {
        "schema": "phase12b_training_selection_v1",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "training_layouts": int(len(np.unique(train["meta_layout_index"]))),
        "training_queries": int(len(train["states"])),
        "selection_layouts": int(len(np.unique(validation["meta_layout_index"]))),
        "selection_queries": int(len(validation["states"])),
        "runs": runs,
        "selected": selected,
        "offline_gate_pass": None,
        "offline_gate_split_used_for_selection": False,
        "reserve_accessed": False,
        "wall_seconds": time.monotonic() - start_time,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
