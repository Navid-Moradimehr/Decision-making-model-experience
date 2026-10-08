"""D168 trajectory reconstruction, frozen latent targets, and decision models.

The first implementation stage intentionally exposes replay/parity helpers
separately from fitting and fresh-layout code. Reserve layouts are never loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import ACTION_VECTORS, AxisAlignedObstacle, CorridorState, corridor_step
from consequence_lab.phase12_teacher import state_features, make_teacher_query
from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14c import build_feature_matrix
from consequence_lab.phase14g import forced_first_action_values
from consequence_lab.phase15_d164 import canonical_agent_relative
from consequence_lab.phase15_d157 import ACTION_COUNT
from consequence_lab.phase15_decision_aligned import regrets_from_exact_summaries


ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d168_latent_intervention.yaml"
STATUS_CODE = {"active": 0, "success": 1, "collision": 2, "timeout": 3}
OUTCOME_ORDER = {"success": 0, "timeout": 1, "collision": 2}
OFFSETS = np.asarray((1, 2, 4, 8, 16, 32, 64, 128, 160), dtype=np.int64)
TRAJECTORY_CHANNELS = 9
SIGNATURE_DIM = len(OFFSETS) * TRAJECTORY_CHANNELS


class D168Error(RuntimeError):
    """D168 provenance, replay, target, or training failure."""


@dataclass(frozen=True)
class BranchTrajectory:
    states: np.ndarray  # [T, 5]: x, y, vx, vy, status code; first row is t=1
    summary: np.ndarray  # [7], exact D153 R4 schema
    terminal_status: str
    terminal_steps: int
    action_trace: tuple[int, ...]


def normalized_physical_actions(acceleration: float, workspace_diagonal: float = 1.0) -> np.ndarray:
    """Return actual environment acceleration vectors, including diagonal scale."""
    if (not np.isfinite(acceleration) or acceleration <= 0
            or not np.isfinite(workspace_diagonal) or workspace_diagonal <= 0):
        raise ValueError("acceleration and workspace diagonal must be finite and positive")
    vectors = np.asarray(ACTION_VECTORS, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    return unit * (float(acceleration) / float(workspace_diagonal))


def _signed_aabb_distance(x: float, y: float, obstacle: AxisAlignedObstacle) -> float:
    cx = (obstacle.x_min + obstacle.x_max) / 2.0
    cy = (obstacle.y_min + obstacle.y_max) / 2.0
    hx = (obstacle.x_max - obstacle.x_min) / 2.0
    hy = (obstacle.y_max - obstacle.y_min) / 2.0
    qx, qy = abs(x - cx) - hx, abs(y - cy) - hy
    return math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0)


def physical_clearance(state: CorridorState, scenario: Any) -> float:
    """Signed clearance from the agent disc to obstacles and workspace boundary."""
    spec = scenario.spec
    radius = float(spec.agent_radius)
    values = [state.x - radius, spec.width - radius - state.x,
              state.y - radius, spec.height - radius - state.y]
    values.extend(_signed_aabb_distance(state.x, state.y, obstacle) - radius
                  for obstacle in spec.obstacles)
    return float(min(values))


def capture_replanned_branch(scenario: Any, source: CorridorState, action_id: int) -> BranchTrajectory:
    """Force one primitive action then exactly resume the frozen planner."""
    if source.status != "active" or action_id not in range(ACTION_COUNT):
        raise ValueError("branch requires an active source and a legal primitive action")
    current = corridor_step(source, int(action_id), scenario.spec).state
    future = [current]
    trace: list[int] = []
    effort = int(action_id != 0)
    while current.status == "active":
        query = make_teacher_query(scenario, current)
        chosen = int(query.sequences[query.selected_index, 0])
        trace.append(chosen)
        effort += int(chosen != 0)
        current = corridor_step(current, chosen, scenario.spec).state
        future.append(current)
        if len(trace) > scenario.spec.max_steps:
            raise D168Error("planner continuation exceeded task horizon")
    if current.status not in OUTCOME_ORDER:
        raise D168Error(f"unsupported terminal status {current.status}")
    total_steps = current.elapsed_steps - source.elapsed_steps
    distance = math.hypot(scenario.spec.goal[0] - current.x,
                          scenario.spec.goal[1] - current.y)
    summary_status_index = {"success": 0, "timeout": 2, "collision": 3}[current.status]
    onehot = np.eye(4, dtype=np.float64)[summary_status_index]
    diagonal = float(math.hypot(scenario.spec.width, scenario.spec.height))
    summary = np.concatenate((onehot, (total_steps / scenario.spec.max_steps,
                                       distance / diagonal,
                                       effort / scenario.spec.max_steps)))
    states = np.asarray([(s.x, s.y, s.vx, s.vy, STATUS_CODE[s.status]) for s in future], dtype=np.float64)
    return BranchTrajectory(states, summary, current.status, total_steps, tuple(trace))


def trajectory_signature(branch: BranchTrajectory, scenario: Any) -> np.ndarray:
    """Sample fixed future anchors into 81 deployable-physics target channels."""
    diagonal = float(math.hypot(scenario.spec.width, scenario.spec.height))
    output = np.zeros((len(OFFSETS), TRAJECTORY_CHANNELS), dtype=np.float64)
    for row, offset in enumerate(OFFSETS):
        state_index = min(int(offset), len(branch.states)) - 1
        x, y, vx, vy, status_code = branch.states[state_index]
        status = int(status_code)
        output[row] = (
            (scenario.spec.goal[0] - x) / diagonal,
            (scenario.spec.goal[1] - y) / diagonal,
            vx / scenario.spec.max_speed,
            vy / scenario.spec.max_speed,
            physical_clearance(CorridorState(float(x), float(y), float(vx), float(vy),
                                             int(offset), tuple(STATUS_CODE)[status]), scenario) / diagonal,
            float(status == 0), float(status == 1), float(status == 2), float(status == 3),
        )
    result = output.reshape(SIGNATURE_DIM)
    if not np.isfinite(result).all():
        raise D168Error("trajectory signature contains non-finite values")
    return result


def exact_regrets(summary: np.ndarray) -> np.ndarray:
    """Recover frozen D153 normalized within-state teacher regrets."""
    values = np.asarray(summary, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (ACTION_COUNT, 7):
        raise ValueError("exact summaries must have shape [states,9,7]")
    return regrets_from_exact_summaries(values).astype(np.float32)


def replay_training_rows(rows_path: Path, expected_sha256: str,
                         state_tolerance: float = 1e-6,
                         summary_tolerance: float = 1e-6) -> dict[str, Any]:
    """Reconstruct all D153 training branch trajectories and enforce exact parity."""
    if sha256_file(rows_path) != expected_sha256:
        raise D168Error("D153 training rows hash differs from the frozen parent")
    with np.load(rows_path, allow_pickle=False) as archive:
        required = {"x_R0", "summary_R4", "layouts", "seeds", "decision_index",
                    "collector_action_ids_audit_only"}
        if set(archive.files) != required:
            raise D168Error("D153 source archive keys differ from the audited schema")
        x_r0 = np.asarray(archive["x_R0"], dtype=np.float64)
        saved_summary = np.asarray(archive["summary_R4"], dtype=np.float64)
        layouts = archive["layouts"].astype(str)
        seeds = np.asarray(archive["seeds"], dtype=np.int64)
        decisions = np.asarray(archive["decision_index"], dtype=np.int64)
        collector_actions = np.asarray(archive["collector_action_ids_audit_only"], dtype=np.int64)
    count = len(x_r0)
    if (x_r0.shape != (1365, 61) or saved_summary.shape != (count, 9, 7)
            or any(a.shape != (count,) for a in (layouts, seeds, decisions, collector_actions))
            or not np.isfinite(x_r0).all() or not np.isfinite(saved_summary).all()):
        raise D168Error("D153 archive shape/finite-value contract failed")
    unique_seeds = sorted(set(seeds.tolist()))
    if len(unique_seeds) != 140:
        raise D168Error("D153 training layout count is not 140")

    state_rows: list[np.ndarray] = []
    canonical_rows: list[np.ndarray] = []
    layout_rows: list[str] = []
    seed_rows: list[int] = []
    decision_rows: list[int] = []
    signatures: list[np.ndarray] = []
    summaries: list[np.ndarray] = []
    flat_states: list[np.ndarray] = []
    offsets = [0]
    branch_query_counts: list[int] = []
    replayed_state_count = 0
    branch_count = 0
    max_state_error = 0.0
    max_summary_error = 0.0
    branch_hashes: list[str] = []
    replayed_actions: list[int] = []
    physical_actions: np.ndarray | None = None

    for seed in unique_seeds:
        ids = np.flatnonzero(seeds == seed)
        ids = ids[np.argsort(decisions[ids], kind="stable")]
        if not np.array_equal(decisions[ids], np.arange(len(ids))):
            raise D168Error(f"D153 decision indices are not contiguous for seed {seed}")
        scenario = generate_staggered_corridor(int(seed))
        row_actions = normalized_physical_actions(
            scenario.spec.acceleration, math.hypot(scenario.spec.width, scenario.spec.height)
        )
        if physical_actions is None:
            physical_actions = row_actions
        elif not np.array_equal(physical_actions, row_actions):
            raise D168Error("D153 training layouts do not share the frozen action/dynamics contract")
        state = scenario.initial_state
        layout_hashes = set(layouts[ids].tolist())
        if layout_hashes != {scenario.layout_sha256}:
            raise D168Error(f"D153 layout identity mismatch for seed {seed}")
        for row_id in ids:
            if state.status != "active":
                raise D168Error("saved collector state follows terminal replay state")
            raw52 = state_features(scenario, state).astype(np.float64)
            r0 = build_feature_matrix(raw52[None], "raw52_plus_physics9")[0].astype(np.float64)
            error = float(np.max(np.abs(r0 - x_r0[row_id])))
            max_state_error = max(max_state_error, error)
            if error > state_tolerance:
                raise D168Error(f"D153 state replay mismatch at seed={seed}, decision={decisions[row_id]}: {error}")
            canonical = canonical_agent_relative(raw52[None], r0[None, 52:])[0]
            state_rows.append(r0)
            canonical_rows.append(canonical)
            layout_rows.append(scenario.layout_sha256)
            seed_rows.append(int(seed))
            decision_rows.append(int(decisions[row_id]))
            per_action_signature, per_action_summary = [], []
            audit_rows = None
            if replayed_state_count == 0:
                audit_rows = forced_first_action_values(
                    scenario, state, decision_index=int(decisions[row_id]),
                    expected_student_features=raw52,
                )
            for action_id in range(ACTION_COUNT):
                branch = capture_replanned_branch(scenario, state, action_id)
                # One independent frozen evaluator parity check validates the
                # trajectory recorder; all rows additionally pass saved R4 parity.
                if audit_rows is not None:
                    record = audit_rows[action_id]
                    if (int(record["forced_action_id"]) != action_id
                            or tuple(record["continuation_action_trace"]) != branch.action_trace
                            or record["terminal_status"] != branch.terminal_status
                            or int(record["terminal_steps_from_source"]) != branch.terminal_steps):
                        raise D168Error("trajectory recorder differs from frozen forced-action evaluator")
                if replayed_state_count == 0 and action_id == 0:
                    repeated = capture_replanned_branch(scenario, state, action_id)
                    round_trip = np.asarray(branch.states, dtype=np.float32).tobytes()
                    if round_trip != np.asarray(repeated.states, dtype=np.float32).tobytes():
                        raise D168Error("repeated training branch trajectory is not byte-identical")
                signature = trajectory_signature(branch, scenario)
                per_action_signature.append(signature)
                per_action_summary.append(branch.summary)
                flat_states.extend(branch.states)
                offsets.append(len(flat_states))
                branch_count += 1
                branch_query_counts.append(len(branch.action_trace))
                branch_hashes.append(hashlib.sha256(branch.states.tobytes()).hexdigest())
            summary_array = np.stack(per_action_summary)
            summary_error = float(np.max(np.abs(summary_array - saved_summary[row_id])))
            max_summary_error = max(max_summary_error, summary_error)
            if summary_error > summary_tolerance:
                raise D168Error(f"D153 branch summary mismatch at seed={seed}, decision={decisions[row_id]}: {summary_error}")
            signatures.append(np.stack(per_action_signature))
            summaries.append(summary_array)
            action = int(collector_actions[row_id])
            if action not in range(ACTION_COUNT):
                raise D168Error("saved collector action ID outside legal range")
            state = corridor_step(state, action, scenario.spec).state
            replayed_actions.append(action)
            replayed_state_count += 1

    if replayed_state_count != count or branch_count != count * ACTION_COUNT:
        raise D168Error("D153 replay row or branch count mismatch")
    result = {
        "x_R0": np.stack(state_rows).astype(np.float32),
        "x_canonical": np.stack(canonical_rows).astype(np.float32),
        "signatures": np.stack(signatures).astype(np.float32),
        "summary_R4": np.stack(summaries).astype(np.float32),
        "layouts": np.asarray(layout_rows, dtype="U64"),
        "seeds": np.asarray(seed_rows, dtype=np.int64),
        "decision_index": np.asarray(decision_rows, dtype=np.int64),
        "collector_action_ids_audit_only": np.asarray(replayed_actions, dtype=np.int8),
        "branch_offsets": np.asarray(offsets, dtype=np.int64),
        "branch_states": np.asarray(flat_states, dtype=np.float32),
        "branch_continuation_query_counts": np.asarray(branch_query_counts, dtype=np.int16),
        "branch_state_sha256": np.asarray(branch_hashes, dtype="U64"),
        "teacher_regret": regrets_from_exact_summaries(np.stack(summaries)).astype(np.float32),
        "physical_actions": physical_actions.astype(np.float32),
        "integrity": {"source_sha256": expected_sha256, "rows": count,
                      "layouts": len(unique_seeds), "branches": branch_count,
                      "max_state_feature_abs_error": max_state_error,
                      "max_summary_abs_error": max_summary_error,
                      "state_tolerance": state_tolerance,
                      "summary_tolerance": summary_tolerance,
                      "reserve_access": False, "test_rows_accessed": 0},
    }
    return result


def save_reconstructed_training(data: dict[str, Any], destination: Path) -> str:
    """Persist immutable G0 arrays and return their content hash."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise D168Error("refusing to overwrite D168 reconstructed training data")
    np.savez_compressed(destination, **{k: v for k, v in data.items() if isinstance(v, np.ndarray)})
    return sha256_file(destination)


def grouped_layout_folds(layouts: np.ndarray, folds: int = 5) -> np.ndarray:
    values = np.asarray(layouts).astype(str)
    if values.ndim != 1 or folds < 2:
        raise ValueError("layout fold input must be one-dimensional with at least two folds")
    return np.asarray([int(hashlib.sha256(v.encode()).hexdigest()[:8], 16) % folds
                       for v in values], dtype=np.int64)


@dataclass(frozen=True)
class FrozenLatentEncoder:
    signature_mean: np.ndarray
    signature_scale: np.ndarray
    components: np.ndarray
    latent_mean: np.ndarray
    latent_scale: np.ndarray
    explained_variance_ratio: np.ndarray

    def encode(self, signatures: np.ndarray) -> np.ndarray:
        x = np.asarray(signatures, dtype=np.float64)
        if x.shape[-1] != SIGNATURE_DIM:
            raise ValueError("future signatures must have 81 components")
        score = ((x - self.signature_mean) / self.signature_scale) @ self.components.T
        z = (score - self.latent_mean) / self.latent_scale
        if not np.isfinite(z).all():
            raise D168Error("frozen latent encoder produced non-finite targets")
        return z.astype(np.float32)


def fit_frozen_latent_encoder(signatures: np.ndarray, layouts: np.ndarray,
                              latent_dim: int = 16, folds: int = 5) -> tuple[FrozenLatentEncoder, np.ndarray, np.ndarray]:
    """Fit deterministic SVD/PCA target mapping on D153 train folds 1..4 only."""
    sig = np.asarray(signatures, dtype=np.float64)
    layout = np.asarray(layouts).astype(str)
    if sig.ndim != 3 or sig.shape[1:] != (ACTION_COUNT, SIGNATURE_DIM) or len(layout) != len(sig):
        raise ValueError("signature/layout arrays violate D168 shape contract")
    fold = grouped_layout_folds(layout, folds)
    fit_rows = fold != 0
    if np.count_nonzero(fit_rows) < latent_dim or len(set(layout[fit_rows])) < 2:
        raise D168Error("too few D153 training layouts to fit future target encoder")
    fit = sig[fit_rows].reshape(-1, SIGNATURE_DIM)
    mean = fit.mean(axis=0)
    scale = fit.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (fit - mean) / scale
    _, singular, vt = np.linalg.svd(standardized, full_matrices=False)
    if latent_dim > vt.shape[0]:
        raise D168Error("requested PCA dimension exceeds available rank")
    components = vt[:latent_dim].copy()
    # Fix SVD sign ambiguity deterministically by making each largest-magnitude
    # loading positive, so repeated lock/run computations hash identically.
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1.0
    score_fit = standardized @ components.T
    latent_mean = score_fit.mean(axis=0)
    latent_scale = score_fit.std(axis=0)
    latent_scale[latent_scale < 1e-8] = 1.0
    variance = singular ** 2
    ratio = variance[:latent_dim] / max(float(variance.sum()), 1e-12)
    encoder = FrozenLatentEncoder(mean, scale, components, latent_mean, latent_scale, ratio)
    encoded = encoder.encode(sig)
    return encoder, encoded, fold


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise D168Error("D168 model fitting requires the project's optional training extra") from exc

    class Predictor(nn.Module):
        def __init__(self, input_dim: int, hidden: int, latent: int):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent))

        def forward(self, x):
            return self.net(x)

    class CandidateModel(nn.Module):
        def __init__(self, input_dim: int, hidden: int, relational: bool, correction_bound: float):
            super().__init__()
            self.relational = relational
            self.correction_bound = float(correction_bound)
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(),
                                         nn.Linear(hidden, hidden), nn.GELU())
            if relational:
                self.attention = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
                self.relation = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                              nn.Linear(hidden, hidden))
                self.base = nn.Linear(hidden, 1)
                self.correction = nn.Linear(hidden, 1)
                nn.init.zeros_(self.correction.weight)
                nn.init.zeros_(self.correction.bias)
            else:
                self.score = nn.Linear(hidden, 1)

        def forward(self, candidate_features):
            h = self.encoder(candidate_features)
            if self.relational:
                attended, _ = self.attention(h, h, h, need_weights=False)
                contextual = h + attended + self.relation(h + attended)
                base = self.base(h).squeeze(-1)
                correction = self.correction(contextual).squeeze(-1).tanh() * self.correction_bound
                return base + correction, correction, base
            score = self.score(h).squeeze(-1)
            zero = torch.zeros_like(score)
            return score, zero, score

    return torch, nn, Predictor, CandidateModel


def _standardize_train(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    mean, scale = train.mean(axis=(0, 1)), train.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.0
    return ((train - mean) / scale).astype(np.float32), ((other - mean) / scale).astype(np.float32), {
        "mean": mean.tolist(), "scale": scale.tolist()}


def _candidate_tensor(state: np.ndarray, actions: np.ndarray, latent: np.ndarray | None,
                      stats: dict[str, dict[str, list[float]]]) -> np.ndarray:
    n = len(state)
    state_set = np.broadcast_to(state[:, None, :], (n, ACTION_COUNT, state.shape[1]))
    action_set = np.broadcast_to(actions[None, :, :], (n, ACTION_COUNT, actions.shape[1]))
    pieces = [state_set, action_set]
    if latent is not None:
        pieces.append(latent)
    features = np.concatenate(pieces, axis=-1).astype(np.float32)
    return ((features - np.asarray(stats["mean"], dtype=np.float32))
            / np.asarray(stats["scale"], dtype=np.float32)).astype(np.float32)


def decision_metrics(scores: np.ndarray, regrets: np.ndarray, summaries: np.ndarray,
                     layouts: np.ndarray, epsilon: float = 0.125,
                     catastrophic_threshold: float = 0.5) -> dict[str, Any]:
    score = np.asarray(scores, dtype=np.float64)
    regret = np.asarray(regrets, dtype=np.float64)
    summary = np.asarray(summaries, dtype=np.float64)
    group = np.asarray(layouts).astype(str)
    if score.shape != regret.shape or score.ndim != 2 or score.shape[1] != ACTION_COUNT:
        raise ValueError("scores/regrets must have shape [states,9]")
    chosen = np.argmin(score, axis=1)
    rows = np.arange(len(score))
    selected = regret[rows, chosen]
    best_match = np.isclose(regret[rows, chosen], 0.0, atol=1e-12)
    layout_means = [float(selected[group == key].mean()) for key in sorted(set(group.tolist()))]
    upper_i, upper_j = np.triu_indices(ACTION_COUNT, 1)
    target_gap = regret[:, upper_i] - regret[:, upper_j]
    pred_gap = score[:, upper_i] - score[:, upper_j]
    high = np.abs(target_gap) >= epsilon
    safe = (summary[:, upper_i, 3] < 0.5) != (summary[:, upper_j, 3] < 0.5)
    safe_correct = ((pred_gap[safe] < 0) == (target_gap[safe] < 0)) if np.any(safe) else np.asarray([])
    high_correct = ((pred_gap[high] < 0) == (target_gap[high] < 0)) if np.any(high) else np.asarray([])
    return {
        "state_count": int(len(score)), "layout_count": int(len(set(group.tolist()))),
        "layout_macro_normalized_regret": float(np.mean(layout_means)),
        "catastrophic_fraction_regret_ge_0_5": float(np.mean(selected >= catastrophic_threshold)),
        "epsilon_optimal_rate_regret_le_0_125": float(np.mean(selected <= epsilon)),
        "best_action_set_agreement": float(np.mean(best_match)),
        "safe_vs_collision_pair_accuracy": float(np.mean(safe_correct)) if len(safe_correct) else None,
        "high_margin_pairwise_accuracy": float(np.mean(high_correct)) if len(high_correct) else None,
        "collision_action_selection_rate": float(np.mean(summary[rows, chosen, 3] > 0.5)),
        "action_distribution": np.bincount(chosen, minlength=ACTION_COUNT).tolist(),
        "mean_selected_regret": float(selected.mean()),
    }


def _pair_objective(torch: Any, scores: Any, regrets: Any, gap_floor: float,
                    gap_scale: float, weight_cap: float):
    import torch.nn.functional as functional
    i, j = torch.triu_indices(ACTION_COUNT, ACTION_COUNT, offset=1, device=scores.device)
    gap = regrets[:, i] - regrets[:, j]
    sign = torch.where(gap < 0, torch.ones_like(gap), -torch.ones_like(gap))
    weights = torch.clamp(torch.abs(gap) / gap_scale, max=weight_cap)
    mask = torch.abs(gap) >= gap_floor
    losses = functional.softplus(-sign * (scores[:, j] - scores[:, i]))
    if not torch.any(mask):
        return scores.new_zeros(())
    return (losses[mask] * weights[mask]).sum() / weights[mask].sum().clamp_min(1e-8)


def _decision_objective(torch: Any, scores: Any, correction: Any, base: Any,
                        regrets: Any, arm: str, cfg: dict[str, Any]):
    import torch.nn.functional as functional
    obj = cfg["objectives"]
    target = torch.softmax(-regrets / float(obj["listwise_temperature"]), dim=1)
    listwise = -(target * torch.log_softmax(-scores / float(obj["listwise_temperature"]), dim=1)).sum(dim=1).mean()
    pair = _pair_objective(torch, scores, regrets, float(obj["pairwise_ignore_regret_gap_below"]),
                           float(obj["pairwise_weight_gap_scale"]), float(obj["pairwise_weight_cap"]))
    preserve = scores.new_zeros(())
    if arm in {"A0", "A1", "A4", "A5", "A6"}:
        i, j = torch.triu_indices(ACTION_COUNT, ACTION_COUNT, offset=1, device=scores.device)
        base_delta = base[:, i] - base[:, j]
        score_delta = scores[:, i] - scores[:, j]
        outside = torch.abs(base_delta) > (2.0 * float(cfg["models"]["relational_correction_bound"]))
        preserve_rank = functional.relu(-torch.sign(base_delta) * score_delta)
        preserve_rank = preserve_rank[outside].mean() if torch.any(outside) else scores.new_zeros(())
        preserve = correction.square().mean() + preserve_rank
    total = (float(obj["listwise_coefficient"]) * listwise
             + float(obj["pairwise_coefficient"]) * pair
             + float(obj["preserve_coefficient"]) * preserve)
    return total, {"listwise": listwise, "pairwise": pair, "preserve": preserve}


def _full_features(state: np.ndarray, actions: np.ndarray, latent: np.ndarray | None) -> np.ndarray:
    state_set = np.broadcast_to(state[:, None, :], (len(state), ACTION_COUNT, state.shape[1]))
    action_set = np.broadcast_to(actions[None], (len(state), ACTION_COUNT, actions.shape[1]))
    parts = [state_set, action_set]
    if latent is not None:
        parts.append(np.asarray(latent, dtype=np.float32))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def _fit_neural_model(torch: Any, model: Any, train_x: np.ndarray, train_y: np.ndarray,
                      train_regret: np.ndarray | None, val_x: np.ndarray, val_y: np.ndarray,
                      val_regret: np.ndarray | None, arm: str, cfg: dict[str, Any],
                      checkpoint_path: Path,
                      extra_checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full-state minibatches; save model and optimizer states at best validation objective."""
    import torch.nn.functional as functional
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["models"]["learning_rate"]),
                            weight_decay=float(cfg["models"]["weight_decay"]))
    patience = int(cfg["models"]["validation_patience"])
    max_epochs = int(cfg["models"]["max_epochs"])
    batch_states = 128
    xtr = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    xva = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    yva = torch.as_tensor(val_y, dtype=torch.float32, device=device)
    rtr = None if train_regret is None else torch.as_tensor(train_regret, dtype=torch.float32, device=device)
    rva = None if val_regret is None else torch.as_tensor(val_regret, dtype=torch.float32, device=device)
    best_value, best_epoch, best_state, best_opt, stale = float("inf"), -1, None, None, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(xtr))
        train_loss_total = 0.0
        for ids in order.split(batch_states):
            xb, yb = xtr[ids], ytr[ids]
            opt.zero_grad(set_to_none=True)
            if train_regret is None:
                pred = model(xb.reshape(-1, xb.shape[-1])).reshape(len(ids), ACTION_COUNT, -1)
                loss = functional.mse_loss(pred, yb)
            else:
                scores, correction, base = model(xb)
                loss, _ = _decision_objective(torch, scores, correction, base, rtr[ids], arm, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            train_loss_total += float(loss.detach()) * len(ids) / len(xtr)
        model.eval()
        with torch.no_grad():
            if val_regret is None:
                val_pred = model(xva.reshape(-1, xva.shape[-1])).reshape(len(xva), ACTION_COUNT, -1)
                val_loss = functional.mse_loss(val_pred, yva)
            else:
                val_score, val_corr, val_base = model(xva)
                val_loss, _ = _decision_objective(torch, val_score, val_corr, val_base, rva, arm, cfg)
            value = float(val_loss)
        history.append({"epoch": epoch + 1, "train_objective": train_loss_total, "validation_objective": value})
        if value < best_value - 1e-10:
            best_value, best_epoch, stale = value, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = copy.deepcopy(opt.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None or best_epoch < 0:
        raise D168Error("model fit did not produce a finite best checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": best_epoch, "model_state_dict": best_state,
                "optimizer_state_dict": best_opt, "best_validation_objective": best_value,
                "history": history, "parameter_count": sum(p.numel() for p in model.parameters())}
    payload.update(extra_checkpoint or {})
    torch.save(payload, checkpoint_path)
    model.to("cpu")
    return {"best_epoch": best_epoch, "epochs_run": len(history),
            "best_validation_objective": best_value,
            "final_train_objective": history[-1]["train_objective"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "optimizer_state_saved": True, "history": history}


def _train_predictor(torch: Any, Predictor: Any, state: np.ndarray, actions: np.ndarray,
                     latent: np.ndarray, fold: np.ndarray, cfg: dict[str, Any],
                     effect: bool, output_dir: Path, tag: str) -> tuple[Any, np.ndarray, dict[str, Any]]:
    import torch.nn.functional as functional
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, val_ids = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    if not len(val_ids):
        raise D168Error("D168 internal validation fold is empty")
    train_raw = _full_features(state[train_ids], actions, None)
    val_raw = _full_features(state[val_ids], actions, None)
    # Predictor feature normalizers are fit only on D153 training layouts.
    mean, scale = train_raw.reshape(-1, train_raw.shape[-1]).mean(0), train_raw.reshape(-1, train_raw.shape[-1]).std(0)
    scale[scale < 1e-8] = 1.0
    train_x = ((train_raw - mean) / scale).astype(np.float32)
    val_x = ((val_raw - mean) / scale).astype(np.float32)
    ytr, yva = latent[train_ids], latent[val_ids]
    xtr = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    ttr = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    xva = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    tva = torch.as_tensor(yva, dtype=torch.float32, device=device)
    model = Predictor(train_x.shape[-1], int(cfg["models"]["hidden_dimension"]), latent.shape[-1])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["models"]["learning_rate"]),
                            weight_decay=float(cfg["models"]["weight_decay"]))
    best, best_epoch, best_state, best_opt, stale, history = float("inf"), -1, None, None, 0, []
    pair_i, pair_j = torch.triu_indices(ACTION_COUNT, ACTION_COUNT, offset=1)
    for epoch in range(int(cfg["models"]["max_epochs"])):
        model.train()
        order = torch.randperm(len(xtr))
        total = 0.0
        for ids in order.split(128):
            pred = model(xtr[ids].reshape(-1, train_x.shape[-1])).reshape(len(ids), ACTION_COUNT, -1)
            pred_loss = functional.mse_loss(pred, ttr[ids])
            effect_loss = pred_loss.new_zeros(())
            if effect:
                pd = pred[:, pair_i] - pred[:, pair_j]
                td = ttr[ids][:, pair_i] - ttr[ids][:, pair_j]
                effect_loss = functional.huber_loss(pd, td, delta=float(cfg["objectives"]["effect_huber_delta"]))
            loss = pred_loss + (float(cfg["objectives"]["effect_coefficient"]) * effect_loss if effect else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total += float(loss.detach()) * len(ids) / len(xtr)
        model.eval()
        with torch.no_grad():
            pred = model(xva.reshape(-1, train_x.shape[-1])).reshape(len(xva), ACTION_COUNT, -1)
            pred_loss = functional.mse_loss(pred, tva)
            effect_loss = pred_loss.new_zeros(())
            if effect:
                effect_loss = functional.huber_loss(pred[:, pair_i] - pred[:, pair_j],
                                                    tva[:, pair_i] - tva[:, pair_j],
                                                    delta=float(cfg["objectives"]["effect_huber_delta"]))
            val_loss = pred_loss + (float(cfg["objectives"]["effect_coefficient"]) * effect_loss if effect else 0.0)
            value = float(val_loss)
        history.append({"epoch": epoch + 1, "train_objective": total,
                        "validation_objective": value, "validation_latent_mse": float(pred_loss),
                        "validation_effect_huber": float(effect_loss)})
        if value < best - 1e-10:
            best, best_epoch, stale = value, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = copy.deepcopy(opt.state_dict())
        else:
            stale += 1
        if stale >= int(cfg["models"]["validation_patience"]):
            break
    if best_state is None:
        raise D168Error("future predictor did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    path = output_dir / f"predictor_{tag}.pt"
    torch.save({"epoch": best_epoch, "model_state_dict": best_state,
                "optimizer_state_dict": best_opt, "best_validation_objective": best,
                "history": history, "input_mean": mean, "input_scale": scale}, path)
    model.eval()
    all_raw = _full_features(state, actions, None)
    all_x = torch.as_tensor(((all_raw - mean) / scale).astype(np.float32), device=device)
    with torch.no_grad():
        predicted = model(all_x.reshape(-1, all_x.shape[-1])).reshape(len(state), ACTION_COUNT, -1).cpu().numpy()
    model.to("cpu")
    pair_i_np, pair_j_np = np.triu_indices(ACTION_COUNT, 1)
    pred_effect = predicted[:, pair_i_np] - predicted[:, pair_j_np]
    true_effect = latent[:, pair_i_np] - latent[:, pair_j_np]
    return model, predicted.astype(np.float32), {
        "best_epoch": best_epoch, "epochs_run": len(history),
        "best_validation_objective": best,
        "train_latent_mse": float(np.mean((predicted[train_ids] - latent[train_ids]) ** 2)),
        "validation_latent_mse": float(np.mean((predicted[val_ids] - latent[val_ids]) ** 2)),
        "train_intervention_effect_mse": float(np.mean((pred_effect[train_ids] - true_effect[train_ids]) ** 2)),
        "validation_intervention_effect_mse": float(np.mean((pred_effect[val_ids] - true_effect[val_ids]) ** 2)),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer_state_saved": True, "checkpoint_sha256": sha256_file(path),
        "history": history,
    }


def _permutation_check(model: Any, x: np.ndarray, action_ids: np.ndarray,
                      trials: int, seed: int) -> dict[str, Any]:
    torch, _, _, _ = _torch_modules()
    model.eval()
    rng = np.random.default_rng(seed)
    max_deviation, mean_deviation, top_flips = 0.0, 0.0, 0
    deviations = []
    with torch.no_grad():
        for row in range(len(x)):
            base = torch.as_tensor(x[row:row + 1], dtype=torch.float32)
            base_scores = model(base)[0].cpu().numpy()[0]
            for _ in range(trials):
                permutation = rng.permutation(ACTION_COUNT)
                permuted = torch.as_tensor(x[row:row + 1, permutation], dtype=torch.float32)
                perm_scores = model(permuted)[0].cpu().numpy()[0]
                restored = np.empty(ACTION_COUNT, dtype=np.float64)
                restored[permutation] = perm_scores
                difference = np.abs(base_scores - restored)
                deviations.extend(difference.tolist())
                max_deviation = max(max_deviation, float(difference.max()))
                # Stable semantic tie-break is action identity, never array position.
                base_choice = int(min(action_ids[np.flatnonzero(base_scores == base_scores.min())]))
                perm_choice = int(min(action_ids[permutation[np.flatnonzero(perm_scores == perm_scores.min())]]))
                top_flips += int(base_choice != perm_choice)
    mean_deviation = float(np.mean(deviations)) if deviations else 0.0
    return {"max_score_deviation": max_deviation, "mean_score_deviation": mean_deviation,
            "top_choice_flips": top_flips, "trials_per_state": trials}


def _fit_eight_state_oracle_gate(torch: Any, CandidateModel: Any, canonical: np.ndarray,
                                 actions: np.ndarray, latent: np.ndarray, regret: np.ndarray,
                                 summaries: np.ndarray, layouts: np.ndarray, cfg: dict[str, Any],
                                 output_dir: Path) -> dict[str, Any]:
    """Prove exact-latent scorer can memorize eight deterministic training states."""
    state_ids = np.arange(min(8, len(canonical)))
    raw = _full_features(canonical[state_ids], actions, latent[state_ids])
    mean, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(((raw - mean) / scale).astype(np.float32), device=device)
    target = torch.as_tensor(regret[state_ids], dtype=torch.float32, device=device)
    model = CandidateModel(x.shape[-1], int(cfg["models"]["hidden_dimension"]), True,
                           float(cfg["models"]["relational_correction_bound"]))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["models"]["learning_rate"]),
                            weight_decay=float(cfg["models"]["weight_decay"]))
    best, best_state, best_opt, best_epoch = float("inf"), None, None, -1
    epochs = min(800, int(cfg["models"]["max_epochs"]))
    for epoch in range(epochs):
        model.train()
        score, correction, base = model(x)
        loss, _ = _decision_objective(torch, score, correction, base, target, "A6", cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            score = model(x)[0].cpu().numpy()
        metric = decision_metrics(score, regret[state_ids], summaries[state_ids], layouts[state_ids])
        value = metric["layout_macro_normalized_regret"]
        if value < best:
            best, best_epoch = value, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_opt = copy.deepcopy(opt.state_dict())
        if best <= 1e-8:
            break
    passed = best_state is not None and best <= 0.05
    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": best_state, "epoch": best_epoch,
                "optimizer_state_dict": best_opt, "training_states": len(state_ids)},
               output_dir / "eight_state_oracle_memorization.pt")
    model.to("cpu")
    return {"status": "passed" if passed else "failed", "states": len(state_ids),
            "best_training_regret": best, "best_epoch": best_epoch,
            "fit_gate": 0.05, "optimizer_state_saved": True}


def _synthetic_training_loop_check(torch: Any, CandidateModel: Any, output_dir: Path,
                                  cfg: dict[str, Any]) -> dict[str, Any]:
    """Exercise the actual listwise/pairwise optimizer on a known state-action rule."""
    rng = np.random.default_rng(int(cfg["models"]["seed"]) + 99)
    n = 48
    state = rng.choice((-1.0, 1.0), size=(n, 1)).astype(np.float32)
    actions = np.asarray(((-1.0, 0.0), (-0.7, -0.7), (-0.7, 0.7), (0.0, -1.0),
                          (0.0, 0.0), (0.0, 1.0), (0.7, -0.7), (0.7, 0.7), (1.0, 0.0)),
                         dtype=np.float32)
    raw = _full_features(state, actions, None)
    mean, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.0
    x = ((raw - mean) / scale).astype(np.float32)
    costs = (actions[None, :, 0] - state[:, None, 0]) ** 2
    regret = costs - costs.min(axis=1, keepdims=True)
    regret /= max(float(regret.max()), 1e-8)
    model = CandidateModel(x.shape[-1], int(cfg["models"]["hidden_dimension"]), True,
                           float(cfg["models"]["relational_correction_bound"]))
    train_cfg = {**cfg, "models": {**cfg["models"], "max_epochs": 400, "validation_patience": 80}}
    metrics = _fit_neural_model(torch, model, x, regret, regret, x, regret, regret,
                                "A6", train_cfg, output_dir / "synthetic_pipeline.pt")
    model.eval()
    with torch.no_grad():
        scores = model(torch.as_tensor(x))[0].numpy()
    selected = np.argmin(scores, axis=1)
    expected = np.argmin(costs, axis=1)
    accuracy = float(np.mean(selected == expected))
    return {"best_epoch": metrics["best_epoch"], "epochs_run": metrics["epochs_run"],
            "optimal_action_accuracy": accuracy, "states": n,
            "passed": accuracy >= 0.98, "parameter_count": metrics["parameter_count"]}


def fit_training_feasibility(data_path: Path, cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run D168 G1 only; never reads or scores the fresh-layout block."""
    configure_deterministic_runtime(cfg["runtime"])
    torch, _, Predictor, CandidateModel = _torch_modules()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(cfg["models"]["seed"]))
    np.random.seed(int(cfg["models"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["models"]["seed"]))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    with np.load(data_path, allow_pickle=False) as archive:
        required = {"x_R0", "x_canonical", "signatures", "summary_R4", "layouts",
                    "teacher_regret", "physical_actions"}
        if not required.issubset(archive.files):
            raise D168Error("G1 input archive is missing audited G0 arrays")
        r0 = np.asarray(archive["x_R0"], dtype=np.float32)
        canonical = np.asarray(archive["x_canonical"], dtype=np.float32)
        signatures = np.asarray(archive["signatures"], dtype=np.float32)
        summary = np.asarray(archive["summary_R4"], dtype=np.float32)
        layouts = archive["layouts"].astype(str)
        regret = np.asarray(archive["teacher_regret"], dtype=np.float32)
        actions = np.asarray(archive["physical_actions"], dtype=np.float32)
    if (r0.shape != (1365, 61) or canonical.shape != (1365, 63)
            or signatures.shape != (1365, 9, 81) or summary.shape != (1365, 9, 7)
            or regret.shape != (1365, 9) or actions.shape != (9, 2)):
        raise D168Error("G1 data shapes do not match frozen protocol")
    latent_encoder, exact_latent, fold = fit_frozen_latent_encoder(
        signatures, layouts, int(cfg["representation"]["latent_dimension"]),
        int(cfg["data"]["grouped_split"]["folds"]),
    )
    train_ids, val_ids = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    if not len(val_ids):
        raise D168Error("frozen D153 internal-validation fold is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = _synthetic_training_loop_check(torch, CandidateModel, output_dir, cfg)
    if not synthetic["passed"]:
        return {"status": "stopped_synthetic_training_loop_failed", "synthetic_check": synthetic,
                "reserve_access": False, "fresh_layout_outcomes_read": False}
    np.savez_compressed(output_dir / "frozen_future_encoder.npz",
                        signature_mean=latent_encoder.signature_mean,
                        signature_scale=latent_encoder.signature_scale,
                        components=latent_encoder.components,
                        latent_mean=latent_encoder.latent_mean,
                        latent_scale=latent_encoder.latent_scale,
                        explained_variance_ratio=latent_encoder.explained_variance_ratio)
    fit_signatures = signatures[train_ids].reshape(-1, SIGNATURE_DIM).astype(np.float64)
    standardized = (fit_signatures - latent_encoder.signature_mean) / latent_encoder.signature_scale
    reconstructed = (standardized @ latent_encoder.components.T) @ latent_encoder.components
    pca_error = (standardized - reconstructed).reshape(-1, len(OFFSETS), TRAJECTORY_CHANNELS)
    pca_anchor_rmse = np.sqrt(np.mean(pca_error ** 2, axis=(0, 2))).tolist()
    predictors: dict[str, np.ndarray] = {}
    predictor_meta = {}
    for name, use_effect in (("pred", False), ("effect", True)):
        _, pred, meta = _train_predictor(torch, Predictor, canonical, actions, exact_latent, fold,
                                         cfg, use_effect, output_dir, name)
        predictors[name] = pred
        predictor_meta[name] = meta

    sanity = _fit_eight_state_oracle_gate(torch, CandidateModel, canonical, actions, exact_latent,
                                          regret, summary, layouts, cfg, output_dir)
    if sanity["status"] != "passed":
        return {"status": "stopped_eight_state_memorization_failed", "synthetic_check": synthetic,
                "sanity": sanity,
                "predictors": predictor_meta, "reserve_access": False,
                "fresh_layout_outcomes_read": False}

    arm_inputs = {
        "A0": (r0, None, True),
        "A1": (canonical, None, True),
        "A2": (canonical, predictors["pred"], False),
        "A3": (canonical, predictors["effect"], False),
        "A4": (canonical, predictors["pred"], True),
        "A5": (canonical, predictors["effect"], True),
        "A6": (canonical, exact_latent, True),
    }
    arm_results: dict[str, Any] = {}
    checkpoint_paths: dict[str, str] = {}
    ordered_arms = list(arm_inputs.items())
    for arm_index, (arm, (state, latent_features, relational)) in enumerate(ordered_arms):
        raw = _full_features(state[train_ids], actions, None if latent_features is None else latent_features[train_ids])
        val_raw = _full_features(state[val_ids], actions, None if latent_features is None else latent_features[val_ids])
        mean, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
        scale[scale < 1e-8] = 1.0
        x_train = ((raw - mean) / scale).astype(np.float32)
        x_val = ((val_raw - mean) / scale).astype(np.float32)
        torch.manual_seed(int(cfg["models"]["seed"]) + arm_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg["models"]["seed"]) + arm_index)
        model = CandidateModel(x_train.shape[-1], int(cfg["models"]["hidden_dimension"]), relational,
                               float(cfg["models"]["relational_correction_bound"]))
        checkpoint = output_dir / f"{arm.lower()}_best.pt"
        meta = _fit_neural_model(torch, model, x_train, regret[train_ids], regret[train_ids],
                                 x_val, regret[val_ids], regret[val_ids], arm, cfg, checkpoint,
                                 {"input_mean": mean, "input_scale": scale, "arm": arm,
                                  "state_dimension": int(state.shape[1]),
                                  "latent_dimension": 0 if latent_features is None else int(latent_features.shape[-1])})
        model.eval()
        with torch.no_grad():
            train_scores = model(torch.as_tensor(x_train))[0].cpu().numpy()
            val_scores = model(torch.as_tensor(x_val))[0].cpu().numpy()
            latency_samples = []
            for idx in range(min(100, len(x_val))):
                tic = time.perf_counter_ns()
                model(torch.as_tensor(x_val[idx:idx + 1]))
                latency_samples.append((time.perf_counter_ns() - tic) / 1e6)
        train_metric = decision_metrics(train_scores, regret[train_ids], summary[train_ids], layouts[train_ids],
                                        float(cfg["metrics"]["epsilon_optimal_regret_threshold"]),
                                        float(cfg["metrics"]["catastrophic_regret_threshold"]))
        val_metric = decision_metrics(val_scores, regret[val_ids], summary[val_ids], layouts[val_ids],
                                      float(cfg["metrics"]["epsilon_optimal_regret_threshold"]),
                                      float(cfg["metrics"]["catastrophic_regret_threshold"]))
        perm = _permutation_check(model, x_val[:min(32, len(x_val))], np.arange(ACTION_COUNT),
                                  int(cfg["gates"]["permutation_trials"]), int(cfg["models"]["seed"]) + 1)
        meta.update({"train": train_metric, "internal_validation": val_metric,
                     "permutation": perm, "relational": relational,
                     "feature_dimension": int(x_train.shape[-1]),
                     "decision_latency_median_ms": float(np.median(latency_samples)),
                     "decision_latency_p90_ms": float(np.percentile(latency_samples, 90)),
                     "parameter_count": sum(p.numel() for p in model.parameters()),
                     "checkpoint_sha256": sha256_file(checkpoint)})
        arm_results[arm] = meta
        checkpoint_paths[arm] = str(checkpoint)
    # Preregistered A5 objective-only comparisons keep inputs, architecture,
    # predictor, parameterization, optimizer and epoch/patience limits fixed.
    a5_raw = _full_features(canonical[train_ids], actions, predictors["effect"][train_ids])
    a5_val_raw = _full_features(canonical[val_ids], actions, predictors["effect"][val_ids])
    a5_mean, a5_scale = a5_raw.mean(axis=(0, 1)), a5_raw.std(axis=(0, 1))
    a5_scale[a5_scale < 1e-8] = 1.0
    a5_train_x = ((a5_raw - a5_mean) / a5_scale).astype(np.float32)
    a5_val_x = ((a5_val_raw - a5_mean) / a5_scale).astype(np.float32)
    loss_ablations = {}
    for ablation_index, (name, listwise_coefficient, pairwise_coefficient) in enumerate(
            (("A5_listwise_only", 1.0, 0.0), ("A5_pairwise_only", 0.0, 0.25))):
        ablation_cfg = {**cfg, "objectives": {**cfg["objectives"],
                            "listwise_coefficient": listwise_coefficient,
                            "pairwise_coefficient": pairwise_coefficient}}
        torch.manual_seed(int(cfg["models"]["seed"]) + 50 + ablation_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg["models"]["seed"]) + 50 + ablation_index)
        ablation_model = CandidateModel(a5_train_x.shape[-1], int(cfg["models"]["hidden_dimension"]), True,
                                        float(cfg["models"]["relational_correction_bound"]))
        checkpoint = output_dir / f"{name.lower()}_best.pt"
        meta = _fit_neural_model(torch, ablation_model, a5_train_x, regret[train_ids], regret[train_ids],
                                 a5_val_x, regret[val_ids], regret[val_ids], "A5", ablation_cfg,
                                 checkpoint, {"input_mean": a5_mean, "input_scale": a5_scale,
                                              "arm": name, "loss_variant": name})
        with torch.no_grad():
            train_score = ablation_model(torch.as_tensor(a5_train_x))[0].cpu().numpy()
            val_score = ablation_model(torch.as_tensor(a5_val_x))[0].cpu().numpy()
        meta["train"] = decision_metrics(train_score, regret[train_ids], summary[train_ids], layouts[train_ids])
        meta["internal_validation"] = decision_metrics(val_score, regret[val_ids], summary[val_ids], layouts[val_ids])
        meta["checkpoint_sha256"] = sha256_file(checkpoint)
        meta["parameter_count"] = sum(p.numel() for p in ablation_model.parameters())
        loss_ablations[name] = meta
    gate = cfg["gates"]
    a5 = arm_results["A5"]["train"]
    a6 = arm_results["A6"]["train"]
    eq = arm_results["A5"]["permutation"]
    passed = (
        a5["layout_macro_normalized_regret"] <= float(gate["full_A5_train_macro_regret_max"])
        and a5["catastrophic_fraction_regret_ge_0_5"] <= float(gate["full_A5_train_catastrophic_fraction_max"])
        and a6["layout_macro_normalized_regret"] <= float(gate["oracle_A6_train_macro_regret_max"])
        and a6["catastrophic_fraction_regret_ge_0_5"] <= float(gate["oracle_A6_train_catastrophic_fraction_max"])
        and eq["max_score_deviation"] <= float(gate["permutation_max_score_deviation"])
        and eq["top_choice_flips"] == 0
    )
    return {
        "status": "G1_passed_training_feasibility" if passed else "G1_failed_training_feasibility",
        "synthetic_training_loop": synthetic,
        "latent_target": {"dimension": int(exact_latent.shape[-1]),
                          "explained_variance_ratio": latent_encoder.explained_variance_ratio.tolist(),
                          "explained_variance_sum": float(latent_encoder.explained_variance_ratio.sum()),
                          "training_reconstruction_rmse_by_anchor": pca_anchor_rmse,
                          "train_layout_count": int(len(set(layouts[train_ids].tolist()))),
                          "internal_validation_layout_count": int(len(set(layouts[val_ids].tolist())))},
        "predictors": predictor_meta, "eight_state_memorization": sanity,
        "arms": arm_results, "objective_ablations": loss_ablations,
        "checkpoint_paths": checkpoint_paths,
        "gates": {"A5_pass": bool(a5["layout_macro_normalized_regret"] <= 0.05
                                   and a5["catastrophic_fraction_regret_ge_0_5"] <= 0.05),
                  "A6_pass": bool(a6["layout_macro_normalized_regret"] <= 0.05
                                   and a6["catastrophic_fraction_regret_ge_0_5"] <= 0.05),
                  "A5_permutation_pass": bool(eq["max_score_deviation"] <= 1e-6
                                               and eq["top_choice_flips"] == 0)},
        "reserve_access": False, "fresh_layout_outcomes_read": False,
        "closed_loop": False, "dagger": False,
    }


def configure_deterministic_runtime(runtime: dict[str, Any], environ: Any = None) -> str:
    """Set the preregistered CUDA workspace before importing/using Torch CUDA ops."""
    env = os.environ if environ is None else environ
    expected = str(runtime["cublas_workspace_config"])
    if expected not in (":4096:8", ":16:8"):
        raise D168Error("unsupported deterministic cuBLAS workspace configuration")
    current = env.get("CUBLAS_WORKSPACE_CONFIG")
    if current is not None and current != expected:
        raise D168Error("CUBLAS_WORKSPACE_CONFIG conflicts with the frozen D168 runtime")
    env["CUBLAS_WORKSPACE_CONFIG"] = expected
    return expected
