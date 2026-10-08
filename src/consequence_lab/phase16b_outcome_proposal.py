"""Train-only, layout-cross-fitted outcome proposal feasibility screen."""
from __future__ import annotations

import hashlib
import json
import copy
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import yaml

from consequence_lab.phase15_d164 import canonical_agent_relative, normalized_action_vectors

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase16b_outcome_proposal_v2.yaml"
PROTOCOL = ROOT / "reports/phase16b_protocol.md"
DATA = ROOT / "artifacts/runs/phase15_d153_coverage_scaling_v1/fresh_training_rows.npz"
PHASE16A = ROOT / "artifacts/runs/phase16_contract_repair_v1/result.json"
PHASE16A_PROGRESS = ROOT / "artifacts/runs/phase16_contract_repair_v1/states.jsonl"
LOCK = ROOT / "experiments/phase16b_outcome_proposal_v2.lock.json"
MODEL_SOURCE = "src/consequence_lab/phase16b_outcome_proposal.py"
CANONICAL_SOURCE = "src/consequence_lab/phase15_d164.py"
ACTION_SOURCE = "src/consequence_lab/inertial_corridor.py"


class Phase16BError(RuntimeError):
    """Data, lock, or experiment contract violation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _cfg() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def load_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the 128 selected D153 train states and all three saved outcome replicates."""
    cfg = _cfg()
    if cfg["model"]["activation"] != "relu" or float(cfg["model"]["dropout"]) != 0.0:
        raise Phase16BError("runner only implements the frozen ReLU/no-dropout architecture")
    if cfg["training"]["optimizer"] != "AdamW" or cfg["training"]["batch"] != "full_fold":
        raise Phase16BError("optimizer or batch mode differs from the implemented protocol")
    if cfg["folds"]["method"] != "sort_unique_layouts_by_sha256_then_round_robin":
        raise Phase16BError("layout fold method differs from the implemented protocol")
    if (ROOT / cfg["data"]["training_rows"]).resolve() != DATA.resolve():
        raise Phase16BError("configured training-row path differs from the frozen runner input")
    if (ROOT / cfg["data"]["phase16a_result"]).resolve() != PHASE16A.resolve():
        raise Phase16BError("configured Phase16A result path differs from the frozen input")
    if (ROOT / cfg["data"]["phase16a_progress"]).resolve() != PHASE16A_PROGRESS.resolve():
        raise Phase16BError("configured Phase16A progress path differs from the frozen input")
    expected_states = int(cfg["data"]["source_state_count"])
    expected_layouts = int(cfg["data"]["layout_count"])
    replicate_count = int(cfg["data"]["search_replicates_per_state_action"])
    with np.load(DATA, allow_pickle=False) as archive:
        if set(archive.files) != {"x_R0", "summary_R4", "layouts", "seeds",
                                  "decision_index", "collector_action_ids_audit_only"}:
            raise Phase16BError("unexpected D153 training schema")
        all_r0 = archive["x_R0"].astype(np.float64)
        all_layouts = archive["layouts"].astype(str)
        all_seeds = archive["seeds"].astype(np.int64)
    if all_r0.shape != (1365, 61) or all_layouts.shape != (1365,):
        raise Phase16BError("unexpected D153 train dimensions")
    records = [json.loads(line) for line in PHASE16A_PROGRESS.read_text(encoding="utf-8").splitlines()]
    if len(records) != expected_states or len({r["row_id"] for r in records}) != expected_states:
        raise Phase16BError("Phase16A selected-state count/schema mismatch")
    row_ids = np.asarray([int(r["row_id"]) for r in records], dtype=np.int64)
    layouts = all_layouts[row_ids]
    if len(np.unique(layouts)) != expected_layouts:
        raise Phase16BError("selected data layout count differs from protocol")
    success = np.zeros((replicate_count, expected_states, 9), dtype=np.int64)
    for state_i, record in enumerate(records):
        if record["seed"] != int(all_seeds[row_ids[state_i]]) or record["layout_hash"] != layouts[state_i]:
            raise Phase16BError("Phase16A record does not map to its D153 training row")
        if record["replicate_zero_parity_max_error"] > 1e-6:
            raise Phase16BError("Phase16A replicate-zero parity failure")
        reps = record["forced_actions"]
        if len(reps) != replicate_count or any(len(rep) != 9 for rep in reps):
            raise Phase16BError("each selected state must have three nine-action outcome replicates")
        for rep_i, rep in enumerate(reps):
            for action_i, branch in enumerate(rep):
                if branch["status"] not in {"success", "collision"}:
                    raise Phase16BError("only success/collision outcomes are allowed in Phase16B")
                success[rep_i, state_i, action_i] = int(branch["status"] == "success")
    r0 = all_r0[row_ids]
    if not np.isfinite(r0).all():
        raise Phase16BError("non-finite deployable state feature")
    canonical = canonical_agent_relative(r0[:, :52], r0[:, 52:])
    if canonical.shape != (expected_states, 63):
        raise Phase16BError("D164 canonical representation shape changed")
    return r0, canonical, success, layouts, row_ids


def action_inputs(state_features: np.ndarray, actions: np.ndarray | None = None) -> np.ndarray:
    """Pointwise shared inputs; candidate order is deliberately absent."""
    states = np.asarray(state_features, dtype=np.float64)
    action_vectors = normalized_action_vectors() if actions is None else np.asarray(actions)
    if states.ndim != 2 or action_vectors.shape != (9, 2):
        raise ValueError("expected [N,D] states and nine two-dimensional actions")
    return np.concatenate((np.repeat(states, 9, axis=0),
                           np.tile(action_vectors, (len(states), 1))), axis=1)


def layout_fold_map(layouts: np.ndarray, count: int = 5) -> dict[str, int]:
    """Deterministic, balanced layout-grouped folds based only on identity hashes."""
    if count < 2:
        raise ValueError("at least two layout folds required")
    unique = sorted(set(np.asarray(layouts).astype(str).tolist()),
                    key=lambda value: (hashlib.sha256(
                        f"phase16b-fold-v1:{value}".encode()).hexdigest(), value))
    return {layout: i % count for i, layout in enumerate(unique)}


def equivariance_error(predict_fn, state: np.ndarray, actions: np.ndarray,
                       permutation: np.ndarray) -> float:
    """Check that changing query order only permutes action scores."""
    before = np.asarray(predict_fn(action_inputs(state[None, :], actions)))
    after = np.asarray(predict_fn(action_inputs(state[None, :], actions[permutation])))
    return float(np.max(np.abs(before[permutation] - after)))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    y, p = labels[order], scores[order]
    tp, fp, previous_recall, ap = 0, 0, 0.0, 0.0
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and p[j] == p[i]:
            j += 1
        tp += int(y[i:j].sum())
        fp += (j - i) - int(y[i:j].sum())
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        i = j
    return float(ap)


def proposal_metrics(success: np.ndarray, collision_score: np.ndarray,
                     threshold: float = 0.5, collision_k: int = 2,
                     success_hit_k: int = 3, tie_tolerance: float = 1e-12,
                     probability_epsilon: float = 1e-7) -> dict[str, float]:
    """State-balanced physical proposal metrics; ties share selection mass."""
    y = np.asarray(success, dtype=np.int64)
    risk = np.asarray(collision_score, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 9 or risk.shape != y.shape:
        raise ValueError("metrics require aligned [states,9] arrays")
    if not 1 <= collision_k <= 9 or not 1 <= success_hit_k <= 9:
        raise ValueError("proposal k must be within the nine-action bank")
    states = len(y)
    top1_success, top3_hit = [], []
    collision_selected, collision_total = [], []
    for sy, sr in zip(y, risk, strict=True):
        # Lowest collision-risk actions are proposals; average over exact ties.
        best = sr.min()
        ids = np.flatnonzero(np.isclose(sr, best, rtol=0.0, atol=tie_tolerance))
        top1_success.append(float(sy[ids].mean()))
        two_cutoff = np.sort(sr)[collision_k - 1]
        two_above = np.flatnonzero(sr < two_cutoff - tie_tolerance)
        two_tied = np.flatnonzero(np.abs(sr - two_cutoff) <= tie_tolerance)
        two_slots = collision_k - len(two_above)
        collision_total.append(float(np.sum(1 - sy)))
        collision_selected.append(float(np.sum(1 - sy[two_above])
                                        + two_slots * np.mean(1 - sy[two_tied])))
        if int(sy.sum()) == 0:
            continue
        # Probability of at least one success when ties cross the top-three cut.
        cutoff = np.sort(sr)[success_hit_k - 1]
        above = np.flatnonzero(sr < cutoff - tie_tolerance)
        tied = np.flatnonzero(np.abs(sr - cutoff) <= tie_tolerance)
        slots = success_hit_k - len(above)
        if np.any(sy[above] == 1):
            hit = 1.0
        else:
            successes_in_tie = int(sy[tied].sum())
            if successes_in_tie == 0:
                hit = 0.0
            else:
                # Exact probability a uniform subset of tied candidates misses all positives.
                n, k, negatives = len(tied), slots, len(tied) - successes_in_tie
                miss = 0.0 if k > negatives else float(math.comb(negatives, k) / math.comb(n, k))
                hit = 1.0 - miss
        top3_hit.append(hit)
    labels = 1 - y
    predicted_collision = risk >= threshold
    tp = int(np.logical_and(predicted_collision, labels == 1).sum())
    fn = int(np.logical_and(~predicted_collision, labels == 1).sum())
    tn = int(np.logical_and(~predicted_collision, labels == 0).sum())
    fp = int(np.logical_and(predicted_collision, labels == 0).sum())
    prob = np.clip(risk, probability_epsilon, 1 - probability_epsilon)
    brier = float(np.mean((prob - labels) ** 2))
    logloss = float(-np.mean(labels * np.log(prob) + (1 - labels) * np.log(1 - prob)))
    macro_recall = 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))
    return {
        "collision_average_precision": _average_precision(labels, risk),
        f"collision_recall_at_{collision_k}": float(
            sum(collision_selected) / max(1.0, sum(collision_total))),
        "top_one_success_rate": float(np.mean(top1_success)),
        f"success_set_hit_at_{success_hit_k}": float(np.mean(top3_hit)),
        f"collision_recall_threshold_{threshold:g}": float(tp / max(1, tp + fn)),
        f"macro_recall_threshold_{threshold:g}": float(macro_recall),
        "brier_score": brier,
        "log_loss": logloss,
        "uniform_top_one_success": float(np.mean(y) / 9.0),
        f"uniform_collision_recall_at_{collision_k}": collision_k / 9.0,
        "states_with_any_success": float(np.mean(np.any(y == 1, axis=1))),
    }


def _layout_proposal_statistics(success: np.ndarray, risk: np.ndarray,
                                layouts: np.ndarray, collision_k: int,
                                tie_tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-layout macro top-one success, selected collisions, and all collisions."""
    top1, selected, total = [], [], []
    for layout in np.unique(layouts):
        rows = layouts == layout
        y, p = success[rows], risk[rows]
        state_top1, state_selected, state_total = [], [], []
        for sy, sr in zip(y, p, strict=True):
            best = sr.min()
            ids = np.flatnonzero(np.isclose(sr, best, rtol=0.0, atol=tie_tolerance))
            state_top1.append(float(sy[ids].mean()))
            cutoff = np.sort(sr)[collision_k - 1]
            above = np.flatnonzero(sr < cutoff - tie_tolerance)
            tied = np.flatnonzero(np.abs(sr - cutoff) <= tie_tolerance)
            slots = collision_k - len(above)
            state_selected.append(float(np.sum(1 - sy[above])
                                        + slots * np.mean(1 - sy[tied])))
            state_total.append(float(np.sum(1 - sy)))
        top1.append(float(np.mean(state_top1)))
        selected.append(float(np.sum(state_selected)))
        total.append(float(np.sum(state_total)))
    return np.asarray(top1), np.asarray(selected), np.asarray(total)


def _bootstrap_advancement(success: np.ndarray, model_risk: np.ndarray,
                           prior_risk: np.ndarray, layouts: np.ndarray,
                           seed: int, draws: int = 10000,
                           collision_k: int = 2,
                           tie_tolerance: float = 1e-12) -> dict[str, list[float]]:
    model_top, model_sel, model_total = _layout_proposal_statistics(
        success, model_risk, layouts, collision_k, tie_tolerance)
    prior_top, _, _ = _layout_proposal_statistics(
        success, prior_risk, layouts, collision_k, tie_tolerance)
    unique_layouts = np.unique(layouts)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(unique_layouts), size=(draws, len(unique_layouts)))
    top_gain = (model_top - prior_top)[sample].mean(axis=1)
    selected_sum = model_sel[sample].sum(axis=1)
    total_sum = model_total[sample].sum(axis=1)
    collision_gain = selected_sum / np.maximum(total_sum, 1.0) - collision_k / 9.0
    return {
        "top_one_success_gain_vs_action_frequency_95_ci": [
            float(np.quantile(top_gain, .025)), float(np.quantile(top_gain, .975))],
        f"collision_recall_at_{collision_k}_gain_vs_uniform_95_ci": [
            float(np.quantile(collision_gain, .025)), float(np.quantile(collision_gain, .975))],
        "top_one_success_gain_point": float(model_top.mean() - prior_top.mean()),
        f"collision_recall_at_{collision_k}_gain_vs_uniform_point": float(
            model_sel.sum() / max(1.0, model_total.sum()) - collision_k / 9.0),
    }


def _lock_files() -> list[str]:
    return ["configs/phase16b_outcome_proposal_v2.yaml", "reports/phase16b_protocol.md",
            "reports/phase16b_equivariance_abort_v1.md",
            "experiments/phase16b_outcome_proposal.lock.json",
            MODEL_SOURCE, CANONICAL_SOURCE, ACTION_SOURCE,
            "tests/test_phase16b_outcome_proposal.py",
            "artifacts/runs/phase15_d153_coverage_scaling_v1/fresh_training_rows.npz",
            "artifacts/runs/phase16_contract_repair_v1/result.json",
            "artifacts/runs/phase16_contract_repair_v1/states.jsonl"]


def freeze_lock() -> dict[str, Any]:
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise Phase16BError("commit protocol/source/tests before freezing the data lock")
    if LOCK.exists():
        raise Phase16BError("refusing to overwrite an existing Phase16B lock")
    files = {name: sha256(ROOT / name) for name in _lock_files()}
    payload = {"schema": "phase16b-lock-v2", "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "files": files,
        "data_policy": {"allowed": ["D153 train only", "Phase16A audit result for provenance"],
                        "validation": False, "fresh_development": False,
                        "reserve": False, "test": False, "closed_loop": False}}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_lock() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase16b-lock-v2":
        raise Phase16BError("unsupported Phase16B lock")
    for name, digest in lock["files"].items():
        if sha256(ROOT / name) != digest:
            raise Phase16BError(f"locked Phase16B input changed: {name}")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise Phase16BError("working tree must be clean before Phase16B scoring")
    return lock


def run() -> dict[str, Any]:
    lock = verify_lock()
    started = time.perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch import nn

    cfg = _cfg()
    if sha256(DATA) != cfg["data"]["training_sha256"]:
        raise Phase16BError("training-data hash disagrees with the protocol")
    torch.manual_seed(int(cfg["folds"]["seeds"][0]))
    torch.use_deterministic_algorithms(True)
    device_mode = cfg["training"]["device"]
    if device_mode == "cuda_if_available":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_mode == "cpu":
        device = torch.device("cpu")
    else:
        raise Phase16BError(f"unsupported device mode: {device_mode}")
    r0, canonical, success_draws, layouts, row_ids = load_training_data()
    if sha256(PHASE16A) != lock["files"]["artifacts/runs/phase16_contract_repair_v1/result.json"]:
        raise Phase16BError("Phase16A provenance result changed")
    phase16a = json.loads(PHASE16A.read_text(encoding="utf-8"))
    if phase16a.get("schema") != "phase16a-result-v1" or phase16a.get("status") != "complete":
        raise Phase16BError("Phase16A prerequisite result is not complete")
    actions = normalized_action_vectors().astype(np.float32)
    fold_map = layout_fold_map(layouts)
    fold_ids = np.asarray([fold_map[x] for x in layouts], dtype=np.int64)
    unique_layouts = np.unique(layouts)
    fold_count = int(cfg["folds"]["count"])
    if len(unique_layouts) != int(cfg["data"]["layout_count"]) \
            or set(fold_ids.tolist()) != set(range(fold_count)):
        raise Phase16BError("layout folds do not cover the frozen 64-layout audit sample")

    class Net(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            hidden = tuple(int(value) for value in cfg["model"]["hidden"])
            self.layers = nn.Sequential(nn.Linear(dim + 2, hidden[0]), nn.ReLU(),
                                        nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                                        nn.Linear(hidden[1], 1))

        def forward(self, x):
            return self.layers(x).squeeze(-1)

    # Software-only integrity check: same architecture must fit an easy known boundary.
    torch.manual_seed(int(cfg["integrity"]["synthetic_seed"]))
    synthetic_x = torch.randn(512, 65, device=device)
    synthetic_y = (synthetic_x[:, 0] + 0.5 * synthetic_x[:, -2] > 0).float()
    smoke = Net(63).to(device)
    smoke_optimizer = torch.optim.AdamW(
        smoke.parameters(), lr=float(cfg["integrity"]["synthetic_learning_rate"]))
    for _ in range(int(cfg["integrity"]["synthetic_epochs"])):
        smoke_optimizer.zero_grad(set_to_none=True)
        smoke_loss = nn.functional.binary_cross_entropy_with_logits(smoke(synthetic_x), synthetic_y)
        smoke_loss.backward()
        smoke_optimizer.step()
    with torch.no_grad():
        synthetic_accuracy = float(((smoke(synthetic_x) > 0) == (synthetic_y > .5)).float().mean().cpu())
    if synthetic_accuracy < float(cfg["integrity"]["synthetic_min_accuracy"]):
        raise Phase16BError(f"synthetic known-rule fit failed: accuracy={synthetic_accuracy:.4f}")

    out_dir = ROOT / cfg["data"]["run_dir"]
    out_dir.mkdir(parents=True, exist_ok=False)
    replicate_count = int(cfg["data"]["search_replicates_per_state_action"])
    if success_draws.shape[0] != replicate_count:
        raise Phase16BError("saved outcome replicate count differs from the protocol")
    initialization_seeds = [int(value) for value in cfg["folds"]["seeds"]]
    predictions = {name: np.full((len(initialization_seeds), len(r0), 9), np.nan, dtype=np.float32)
                   for name in ("r0", "canonical")}
    prior_predictions = np.full((len(r0), 9), np.nan, dtype=np.float32)
    training_metrics = []
    histories = []
    for rep_name, states in (("r0", r0), ("canonical", canonical)):
        for fold in range(fold_count):
            train_rows, held_rows = fold_ids != fold, fold_ids == fold
            if not train_rows.any() or not held_rows.any():
                raise Phase16BError("empty layout fold")
            mean, scale = states[train_rows].mean(0), states[train_rows].std(0)
            scale[scale < 1e-8] = 1.0
            tr_state = ((states[train_rows] - mean) / scale).astype(np.float32)
            va_state = ((states[held_rows] - mean) / scale).astype(np.float32)
            tr_x = np.concatenate((np.repeat(tr_state, 9, axis=0),
                                   np.tile(actions, (len(tr_state), 1))), axis=1)
            va_x = np.concatenate((np.repeat(va_state, 9, axis=0),
                                   np.tile(actions, (len(va_state), 1))), axis=1)
            tr_y = np.concatenate([1 - success_draws[rep_i, train_rows].reshape(-1)
                                   for rep_i in range(replicate_count)]).astype(np.float32)
            action_collision_prior = 1.0 - success_draws[:, train_rows, :].mean(axis=(0, 1))
            prior_predictions[held_rows] = np.tile(action_collision_prior, (int(held_rows.sum()), 1))
            tr_x = np.tile(tr_x, (replicate_count, 1))
            tx = torch.as_tensor(tr_x, device=device)
            ty = torch.as_tensor(tr_y, device=device)
            vx = torch.as_tensor(va_x, device=device)
            for seed in initialization_seeds:
                # Seed each fit independently; reinitialize model/optimizer per preregistered seed.
                torch.manual_seed(int(seed) + fold)
                model = Net(states.shape[1]).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                    weight_decay=float(cfg["training"]["weight_decay"]))
                best, best_state, best_optim, history = float("inf"), None, None, []
                for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(tx)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, ty)
                    loss.backward()
                    optimizer.step()
                    value = float(loss.detach().cpu())
                    history.append(value)
                    if value < best:
                        best = value
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        best_optim = copy.deepcopy(optimizer.state_dict())
                final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                final_optim = copy.deepcopy(optimizer.state_dict())
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    train_risk = torch.sigmoid(model(tx)).cpu().numpy().reshape(replicate_count, -1, 9)
                    held_risk = torch.sigmoid(model(vx)).cpu().numpy().reshape(-1, 9)
                    one_state = va_state[:1]
                    perm = np.random.default_rng(int(seed) + fold).permutation(9)
                    perm_x = np.concatenate((np.repeat(one_state, 9, axis=0),
                                             actions[perm]), axis=1)
                    perm_risk = torch.sigmoid(model(torch.as_tensor(perm_x, device=device)))
                    equivariance = float(np.max(np.abs(
                        held_risk[0][perm] - perm_risk.cpu().numpy().reshape(-1))))
                if equivariance > float(cfg["integrity"]["permutation_max_error"]):
                    raise Phase16BError(f"candidate-order equivariance failed: {equivariance}")
                seed_i = cfg["folds"]["seeds"].index(seed)
                predictions[rep_name][seed_i, held_rows] = held_risk
                train_metric = proposal_metrics(
                    success_draws[:, train_rows, :].reshape(-1, 9),
                    np.tile(train_risk[0], (replicate_count, 1)),
                    float(cfg["metrics"]["collision_score_threshold"]),
                    int(cfg["metrics"]["collision_recall_k"]),
                    int(cfg["metrics"]["success_set_hit_k"]),
                    float(cfg["metrics"]["score_tie_abs_tolerance"]),
                    float(cfg["metrics"]["probability_clip_epsilon"]))
                training_metrics.append({"representation": rep_name, "fold": fold,
                                        "seed": seed, "best_train_bce": best,
                                        "train_metrics": train_metric})
                stem = f"{rep_name}_fold{fold}_seed{seed}"
                torch.save({"state_dict": best_state, "optimizer_state": best_optim,
                            "final_state_dict": final_state,
                            "final_optimizer_state": final_optim,
                            "epoch": int(np.argmin(history) + 1), "mean": mean,
                            "scale": scale, "history": history}, out_dir / f"{stem}.pt")
                histories.append({"representation": rep_name, "fold": fold, "seed": seed,
                                  "best_train_bce": best, "final_train_bce": history[-1],
                                  "epochs": len(history), "checkpoint": f"{stem}.pt",
                                  "permutation_max_deviation": equivariance})
    # Every row must have one held-out prediction for every initialization seed.
    metrics = {}
    score_success = success_draws.reshape(-1, 9)
    score_layouts = np.tile(layouts, replicate_count)
    for rep_name in ("r0", "canonical"):
        by_seed = []
        for seed_i, seed in enumerate(initialization_seeds):
            risk = predictions[rep_name][seed_i]
            if not np.isfinite(risk).all():
                raise Phase16BError("missing out-of-fold predictions")
            prior = prior_predictions
            repeated_risk, repeated_prior = (np.tile(risk, (replicate_count, 1)),
                                             np.tile(prior, (replicate_count, 1)))
            metric_args = (float(cfg["metrics"]["collision_score_threshold"]),
                           int(cfg["metrics"]["collision_recall_k"]),
                           int(cfg["metrics"]["success_set_hit_k"]),
                           float(cfg["metrics"]["score_tie_abs_tolerance"]),
                           float(cfg["metrics"]["probability_clip_epsilon"]))
            by_seed.append({"seed": seed,
                            "metrics": proposal_metrics(score_success, repeated_risk, *metric_args),
                            "action_frequency_baseline": proposal_metrics(
                                score_success, repeated_prior, *metric_args),
                            "advancement_bootstrap": _bootstrap_advancement(
                                score_success, repeated_risk, repeated_prior,
                                score_layouts,
                                int(cfg["metrics"]["bootstrap"]["seed"]) + seed_i,
                                int(cfg["metrics"]["bootstrap"]["draws"]),
                                int(cfg["metrics"]["collision_recall_k"]),
                                float(cfg["metrics"]["score_tie_abs_tolerance"]))})
        gate_cfg = cfg["gates"]["advance_to_fresh_development_protocol_only_if"]
        gate_rows = []
        for row in by_seed:
            boot = row["advancement_bootstrap"]
            passed = (
                boot["top_one_success_gain_point"] >= float(
                    gate_cfg["top_one_success_gain_over_action_frequency_at_least"])
                and boot["top_one_success_gain_vs_action_frequency_95_ci"][0] > float(
                    gate_cfg["top_one_success_gain_ci_lower_above"])
                and boot["collision_recall_at_2_gain_vs_uniform_point"] >= float(
                    gate_cfg["collision_recall_at_two_gain_over_uniform_at_least"])
                and boot["collision_recall_at_2_gain_vs_uniform_95_ci"][0] > float(
                    gate_cfg["collision_recall_at_two_gain_ci_lower_above"]))
            gate_rows.append({"seed": row["seed"], "pass": bool(passed)})
        metrics[rep_name] = {"seed_results": by_seed, "gate_by_seed": gate_rows,
                             "gate_pass": sum(item["pass"] for item in gate_rows)
                             >= int(cfg["metrics"][
                                 "advancement_minimum_seeds_passing_joint_gate"])}
    result = {"schema": "phase16b-result-v1", "status": "complete",
              "protocol_commit": lock["source_commit"],
              "run_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "config_sha256": sha256(CONFIG), "data_sha256": sha256(DATA),
              "phase16a_result_sha256": sha256(PHASE16A),
              "phase16a_progress_sha256": sha256(PHASE16A_PROGRESS),
              "synthetic_fit_accuracy": synthetic_accuracy,
              "layout_count": int(len(unique_layouts)), "state_count": int(len(r0)),
              "search_replicates_per_state_action": replicate_count,
              "training_outcome_draw_count": int(success_draws.size),
              "layout_fold_map_sha256": hashlib.sha256(
                  json.dumps(sorted(fold_map.items()),
                             separators=(",", ":")).encode()).hexdigest().upper(),
              "folds": fold_count, "seeds": initialization_seeds, "device": str(device),
              "python_version": __import__("sys").version,
              "numpy_version": np.__version__, "torch_version": torch.__version__,
              "torch_cuda_version": torch.version.cuda,
              "parameter_count_per_model": sum(p.numel() for p in Net(canonical.shape[1]).parameters()),
              "out_of_fold_metrics": metrics, "training_fit": training_metrics,
              "fold_histories": histories, "reserve_access": False, "test_access": False,
              "fresh_layout_outcomes": False, "simulator_transitions": 0, "planner_queries": 0,
              "closed_loop": False, "dagger": False,
              "runtime_seconds": time.perf_counter() - started}
    np.savez_compressed(out_dir / "oof_predictions.npz", r0=predictions["r0"],
                        canonical=predictions["canonical"], action_frequency=prior_predictions,
                        success_draws=success_draws, layouts=layouts, row_ids=row_ids,
                        fold_ids=fold_ids)
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema": "phase16b-manifest-v1", "lock_sha256": sha256(LOCK),
                "result_sha256": sha256(result_path),
                "oof_predictions_sha256": sha256(out_dir / "oof_predictions.npz"),
                "checkpoint_hashes": {p.name: sha256(p) for p in out_dir.glob("*.pt")}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "run"))
    args = parser.parse_args()
    print(json.dumps(freeze_lock() if args.mode == "freeze" else run(), indent=2))
