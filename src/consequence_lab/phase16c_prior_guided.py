"""Phase 16C equal-sequence-budget learned proposal experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Sequence

import numpy as np

from consequence_lab.corridor_baselines import _shooting_objective
from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.inertial_corridor import CorridorState, corridor_step
from consequence_lab.phase12_teacher import state_features
from consequence_lab.phase14c import build_feature_matrix
from consequence_lab.phase15_d164 import canonical_agent_relative, normalized_action_vectors
from consequence_lab.corridor_evaluation import evaluate_corridor_episode


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase16c_prior_guided_planner_v1.yaml"
PROTOCOL = ROOT / "reports/phase16c_prior_guided_planner_protocol.md"
LOCK = ROOT / "experiments/phase16c_prior_guided_planner_v3.lock.json"


@dataclass(frozen=True)
class Episode:
    seed: int
    layout_sha256: str
    method: str
    outcome: str
    steps: int
    decision_count: int
    exact_sequences: int
    exact_transitions: int
    objective_tie_decisions: int
    duplicate_rejections: int
    fallback_sequences: int
    latency_seconds_per_decision: tuple[float, ...]


def inverse_rank_probabilities(risk: Sequence[float]) -> np.ndarray:
    """Return normalized inverse average-rank masses; low risk ranks first."""
    values = np.asarray(risk, dtype=np.float64)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("risk must be a finite nine-action vector")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(9, dtype=np.float64)
    i = 0
    while i < 9:
        j = i + 1
        while j < 9 and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    weights = 1.0 / (1.0 + ranks)
    return weights / weights.sum()


def _categorical_from_quantile(probabilities: np.ndarray, q: float) -> int:
    if not 0.0 <= q < 1.0:
        raise ValueError("quantile must lie in [0,1)")
    return min(8, int(np.searchsorted(np.cumsum(probabilities), q, side="right")))


def proposal_bank(
    horizon: int,
    budget: int,
    seed: int,
    first_action_probabilities: Sequence[float] | None = None,
    *,
    max_attempt_multiplier: int = 10_000,
) -> tuple[tuple[tuple[int, ...], ...], int, int]:
    """Make a stable unique sequence bank; return sequences, duplicates, fallbacks."""
    if horizon <= 0 or budget <= 0 or seed < 0 or max_attempt_multiplier <= 0:
        raise ValueError("horizon, budget, seed, and attempt multiplier are invalid")
    total = min(int(budget), 9**int(horizon))
    probs = None
    if first_action_probabilities is not None:
        probs = np.asarray(first_action_probabilities, dtype=np.float64)
        if probs.shape != (9,) or not np.isfinite(probs).all() or np.any(probs < 0):
            raise ValueError("first-action distribution must be a finite nonnegative 9-vector")
        if not np.isclose(probs.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("first-action distribution must sum to one")
    proposals: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for action_id in range(9):
        seq = (action_id,) * horizon
        if seq not in seen:
            proposals.append(seq)
            seen.add(seq)
            if len(proposals) == total:
                return tuple(proposals), 0, 0
    rng = random.Random(seed)
    duplicates = 0
    attempts = 0
    attempt_cap = total * max_attempt_multiplier
    while len(proposals) < total and attempts < attempt_cap:
        attempts += 1
        if probs is None:
            first = min(8, int(rng.random() * 9.0))
        else:
            first = _categorical_from_quantile(probs, rng.random())
        seq = (first,) + tuple(rng.randrange(9) for _ in range(horizon - 1))
        if seq in seen:
            duplicates += 1
            continue
        proposals.append(seq)
        seen.add(seq)
    fallback = 0
    if len(proposals) < total:
        fill_rng = random.Random(seed ^ 0x9E3779B9)
        while len(proposals) < total:
            seq = tuple(fill_rng.randrange(9) for _ in range(horizon))
            if seq in seen:
                continue
            proposals.append(seq)
            seen.add(seq)
            fallback += 1
    if len(proposals) != total or len(seen) != total:
        raise RuntimeError("could not construct the frozen unique candidate budget")
    return tuple(proposals), duplicates, fallback


@dataclass(frozen=True)
class SequenceEvaluation:
    sequence: tuple[int, ...]
    objective: tuple[int, float, float]
    transitions: int
    success_reached: bool


def evaluate_sequence(
    initial_state: CorridorState,
    scenario: CorridorScenario,
    sequence: Sequence[int],
) -> SequenceEvaluation:
    state = initial_state
    effort = 0.0
    transitions = 0
    success = False
    for action in sequence:
        if state.status != "active":
            break
        effort += float(action != 0)
        step = corridor_step(state, int(action), scenario.spec)
        state = step.state
        transitions += 1
        success = success or step.success
    return SequenceEvaluation(tuple(int(a) for a in sequence),
                              _shooting_objective(initial_state, state, effort, scenario.spec.goal),
                              transitions, success)


def choose_sequence(
    state: CorridorState,
    scenario: CorridorScenario,
    bank: Sequence[Sequence[int]],
) -> tuple[SequenceEvaluation, int, int]:
    evaluations = [evaluate_sequence(state, scenario, seq) for seq in bank]
    if len(evaluations) != len(bank):
        raise RuntimeError("sequence evaluation count mismatch")
    best_key = min(item.objective for item in evaluations)
    tied = [item for item in evaluations if item.objective == best_key]
    chosen = min(tied, key=lambda item: item.sequence)
    return chosen, sum(item.transitions for item in evaluations), int(len(tied) > 1)


def make_state_representation(scenario: CorridorScenario, state: CorridorState,
                              representation: str) -> np.ndarray:
    raw52 = state_features(scenario, state)[None, :].astype(np.float64)
    raw61 = build_feature_matrix(raw52, "raw52_plus_physics9")
    if representation == "r0":
        return raw61[0]
    if representation == "canonical":
        return canonical_agent_relative(raw52, raw61[:, 52:])[0]
    raise ValueError(f"unknown representation: {representation}")


def paired_bootstrap(
    baseline: Sequence[float], treatment: Sequence[float], seed: int,
    draws: int = 10_000,
) -> dict[str, float]:
    a, b = np.asarray(baseline, dtype=np.float64), np.asarray(treatment, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or len(a) == 0:
        raise ValueError("paired bootstrap requires aligned nonempty layout vectors")
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(a), size=(draws, len(a)))
    diff = (b[sample] - a[sample]).mean(axis=1)
    return {"difference": float(np.mean(b - a)),
            "ci95_lower": float(np.quantile(diff, .025)),
            "ci95_upper": float(np.quantile(diff, .975))}


def layout_hash_audit(start: int, count: int, root: Path) -> dict:
    """Read-only audit of layout identities; it never steps the simulator."""
    from consequence_lab.phase14g import _collect_prior_layout_hashes

    if start < 0 or count <= 0 or (start < 130200 and 130000 < start + count):
        raise ValueError("candidate seed block invalid or intersects reserve")
    records = [{"seed": seed, "layout_sha256": generate_staggered_corridor(seed).layout_sha256}
               for seed in range(start, start + count)]
    candidates = {row["layout_sha256"] for row in records}
    prior, spent, sources, source_hashes = _collect_prior_layout_hashes(
        root, root / "artifacts/runs/phase16c_layout_audit_exclusion.json")
    if len(candidates) != count or candidates & prior or candidates & spent:
        raise ValueError("candidate layout block duplicates or overlaps historical layouts")
    return {"schema": "phase16c-layout-audit-v1", "seed_start": start,
            "layout_count": count, "layouts": records,
            "unique_hashes": len(candidates), "historical_source_count": len(sources),
            "historical_source_hashes": source_hashes,
            "historical_source_hashes_sha256": hashlib.sha256(json.dumps(
                source_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper(),
            "overlap_historical": 0, "overlap_outcome_bearing": 0,
            "d168_seed_block_accessed": False,
            "reserve_generated_or_accessed": False,
            "outcomes_or_transitions_computed": False}


def prepare_locked_output_directory(output_dir: Path, audit_path: Path) -> None:
    """Allow only the already-locked audit file in a pre-existing output dir."""
    output_dir, audit_path = Path(output_dir), Path(audit_path)
    if output_dir.exists():
        files = {path.resolve() for path in output_dir.rglob("*") if path.is_file()}
        if files != {audit_path.resolve()} or not audit_path.is_file():
            raise RuntimeError("refusing to use a nonempty or unexpected existing output directory")
        return
    output_dir.mkdir(parents=True)


def repeat_features_for_replicates(features: np.ndarray, replicate_count: int) -> np.ndarray:
    """Match replicate-major labels while sharing each state's action features."""
    features = np.asarray(features)
    if features.ndim != 2 or replicate_count < 1:
        raise ValueError("features must be a matrix and replicate_count must be positive")
    return np.tile(features, (int(replicate_count), 1))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _train_prior(output_path: Path, cfg: dict) -> tuple[dict, float]:
    """Fit the frozen final Phase16B canonical model on D153 train labels only."""
    import os
    prior_cfg, data_cfg = cfg["prior"], cfg["training_data"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", str(prior_cfg["cublas_workspace_config"]))
    import torch
    from torch import nn

    from consequence_lab.phase16b_outcome_proposal import load_training_data

    _, canonical, success_draws, layouts, _ = load_training_data()
    reference_cfg = __import__("yaml").safe_load(
        (ROOT / "configs/phase16b_outcome_proposal_v2.yaml").read_text(encoding="utf-8"))
    reference = reference_cfg["model"], reference_cfg["training"]
    if (prior_cfg["architecture"] != "shared_action_conditioned_mlp"
            or prior_cfg["activation"] != "relu" or float(prior_cfg["dropout"]) != 0.0
            or prior_cfg["optimizer"] != "AdamW"
            or prior_cfg["hidden"] != reference[0]["hidden"]
            or prior_cfg["activation"] != reference[0]["activation"]
            or prior_cfg["dropout"] != reference[0]["dropout"]
            or prior_cfg["optimizer"] != reference[1]["optimizer"]
            or float(prior_cfg["learning_rate"]) != float(reference[1]["learning_rate"])
            or float(prior_cfg["weight_decay"]) != float(reference[1]["weight_decay"])
            or int(prior_cfg["epochs"]) != int(reference[1]["epochs"])
            or int(prior_cfg["initialization_seed"]) != int(reference_cfg["folds"]["seeds"][0])):
        raise ValueError("Phase16C runner does not implement the configured prior architecture")
    if (len(canonical) != int(data_cfg["source_states"])
            or len(np.unique(layouts)) != int(data_cfg["source_layouts"])
            or success_draws.shape != (int(data_cfg["search_replicates_per_state_action"]),
                                       int(data_cfg["source_states"]),
                                       int(data_cfg["actions_per_state"]))):
        raise ValueError("Phase16C training target dimensions differ from the locked config")
    mean = canonical.mean(axis=0)
    scale = canonical.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = ((canonical - mean) / scale).astype(np.float32)
    actions = normalized_action_vectors().astype(np.float32)
    action_count = int(data_cfg["actions_per_state"])
    if actions.shape != (action_count, 2):
        raise ValueError("physical action vocabulary differs from the locked training data")
    features = np.concatenate((np.repeat(normalized, action_count, axis=0),
                               np.tile(actions, (len(normalized), 1))), axis=1)
    collision = (1 - success_draws).reshape(-1).astype(np.float32)
    expected_targets = (int(data_cfg["search_replicates_per_state_action"])
                        * int(data_cfg["source_states"]) * int(data_cfg["actions_per_state"]))
    input_dim = canonical.shape[1] + actions.shape[1]
    features = repeat_features_for_replicates(
        features, int(data_cfg["search_replicates_per_state_action"]))
    if features.shape != (expected_targets, input_dim) \
            or collision.shape != (expected_targets,):
        raise ValueError("Phase 16B training matrix dimensions differ from the frozen contract")
    torch.manual_seed(int(prior_cfg["initialization_seed"]))
    if bool(prior_cfg["deterministic_algorithms"]):
        torch.use_deterministic_algorithms(True)
    if prior_cfg["device"] == "cuda_if_available":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif prior_cfg["device"] == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("unsupported Phase16C training device")
    if prior_cfg["loss"] != "binary_cross_entropy_with_logits":
        raise ValueError("unsupported Phase16C prior loss")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = [int(width) for width in prior_cfg["hidden"]]
            if len(hidden) != 2:
                raise ValueError("Phase16C shared MLP requires exactly two hidden layers")
            self.layers = nn.Sequential(nn.Linear(input_dim, hidden[0]), nn.ReLU(),
                                        nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], 1))

        def forward(self, x):
            return self.layers(x).squeeze(-1)

    x = torch.as_tensor(features, device=device)
    y = torch.as_tensor(collision, device=device)
    model = Net().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(prior_cfg["learning_rate"]),
                                  weight_decay=float(prior_cfg["weight_decay"]))
    best_loss, best_state, losses = float("inf"), None, []
    started = time.perf_counter()
    for _ in range(int(prior_cfg["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        losses.append(value)
        if value < best_loss:
            best_loss = value
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("prior training produced no checkpoint")
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "mean": mean, "scale": scale,
                "train_bce": best_loss, "epochs": len(losses),
                "history": losses, "seed": int(prior_cfg["initialization_seed"]),
                "data_sha256": _sha(ROOT / "artifacts/runs/phase15_d153_coverage_scaling_v1/fresh_training_rows.npz"),
                "phase16a_sha256": _sha(ROOT / "artifacts/runs/phase16_contract_repair_v1/result.json")},
               output_path)
    metadata = {"device": str(device), "parameter_count": sum(p.numel() for p in model.parameters()),
                "training_rows": len(features), "best_train_bce": best_loss,
                "final_train_bce": losses[-1], "epochs": len(losses)}
    return {"model": model, "mean": mean, "scale": scale, "device": device,
            "metadata": metadata}, time.perf_counter() - started


def _risk_scores(prior: dict, scenario: CorridorScenario, state: CorridorState) -> np.ndarray:
    import torch

    canonical = make_state_representation(scenario, state, "canonical")
    normalized = ((canonical - prior["mean"]) / prior["scale"]).astype(np.float32)
    actions = normalized_action_vectors().astype(np.float32)
    x = np.concatenate((np.repeat(normalized[None, :], len(actions), axis=0), actions), axis=1)
    with torch.no_grad():
        return torch.sigmoid(prior["model"](
            torch.as_tensor(x, device=prior["device"]))).detach().cpu().numpy().astype(np.float64)


def _evaluate_shooting_episode(
    scenario: CorridorScenario,
    method: str,
    prior: dict | None,
    *,
    horizon: int,
    budget: int,
    attempt_multiplier: int,
    query_seed_multiplier: int,
) -> Episode:
    if method not in {"uniform_exact_shooting", "canonical_rank_prior_exact_shooting"}:
        raise ValueError("unsupported Phase 16C shooting method")
    state = scenario.initial_state
    steps_start = state.elapsed_steps
    decision = 0
    exact_sequences = exact_transitions = ties = duplicates = fallbacks = 0
    latencies: list[float] = []
    while state.status == "active":
        started = time.perf_counter()
        rollout_horizon = min(horizon, scenario.spec.max_steps - state.elapsed_steps)
        query_seed = (scenario.seed * query_seed_multiplier + decision) % (2**32)
        probabilities = None
        if method == "canonical_rank_prior_exact_shooting":
            if prior is None:
                raise ValueError("prior-guided method requires the frozen prior")
            probabilities = inverse_rank_probabilities(_risk_scores(prior, scenario, state))
        bank, rejected, fallback = proposal_bank(
            rollout_horizon, min(budget, 9**rollout_horizon), query_seed, probabilities,
            max_attempt_multiplier=attempt_multiplier)
        chosen, transitions, tied = choose_sequence(state, scenario, bank)
        step = corridor_step(state, chosen.sequence[0], scenario.spec)
        state = step.state
        decision += 1
        exact_sequences += len(bank)
        exact_transitions += transitions
        ties += tied
        duplicates += rejected
        fallbacks += fallback
        latencies.append((time.perf_counter() - started) / 1.0)
    return Episode(scenario.seed, scenario.layout_sha256, method, state.status,
                   state.elapsed_steps - steps_start, decision, exact_sequences,
                   exact_transitions, ties, duplicates, fallbacks, tuple(latencies))


def _summarize(rows: Sequence[Episode]) -> dict:
    episodes = len(rows)
    flat_latency = np.asarray([x for row in rows for x in row.latency_seconds_per_decision], dtype=float)
    return {"episodes": episodes,
            "safe_success_rate": float(np.mean([row.outcome == "success" for row in rows])),
            "collision_rate": float(np.mean([row.outcome == "collision" for row in rows])),
            "timeout_rate": float(np.mean([row.outcome == "timeout" for row in rows])),
            "mean_actions_to_terminal": float(np.mean([row.steps for row in rows])),
            "mean_sequences_per_decision": float(np.mean([
                row.exact_sequences / max(1, row.decision_count) for row in rows])),
            "mean_exact_transitions_per_decision": float(np.mean([
                row.exact_transitions / max(1, row.decision_count) for row in rows])),
            "objective_tie_decisions": int(sum(row.objective_tie_decisions for row in rows)),
            "objective_tie_decision_rate": float(sum(row.objective_tie_decisions for row in rows)
                / max(1, sum(row.decision_count for row in rows))),
            "duplicate_rejections": int(sum(row.duplicate_rejections for row in rows)),
            "fallback_sequences": int(sum(row.fallback_sequences for row in rows)),
            "decision_latency_seconds_median": float(np.median(flat_latency)),
            "decision_latency_seconds_p90": float(np.quantile(flat_latency, .9))}


def run_locked(root: Path = ROOT) -> dict:
    """Train the locked prior and score only the audited development block."""
    root = Path(root).resolve()
    run_started = time.perf_counter()
    config_path = root / "configs/phase16c_prior_guided_planner_v1.yaml"
    protocol_path = root / "reports/phase16c_prior_guided_planner_protocol.md"
    lock_path = root / "experiments/phase16c_prior_guided_planner_v3.lock.json"
    if not lock_path.is_file():
        raise RuntimeError("Phase 16C immutable lock is missing")
    cfg, lock = json.loads(json.dumps(__import__("yaml").safe_load(config_path.read_text(encoding="utf-8")))), \
        json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase16c-prior-guided-lock-v3" or lock.get("config_sha256") != _sha(config_path):
        raise RuntimeError("Phase 16C lock/config mismatch")
    if subprocess_status := __import__("subprocess").check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise RuntimeError(f"Phase16C runner requires a clean locked worktree: {subprocess_status}")
    current_head = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    ancestor = __import__("subprocess").run(
        ["git", "merge-base", "--is-ancestor", lock["source_commit"], current_head], cwd=root)
    if ancestor.returncode != 0:
        raise RuntimeError("Phase16C protocol commit is not an ancestor of the execution commit")
    for rel, digest in lock["source_sha256"].items():
        if _sha(root / rel) != digest:
            raise RuntimeError(f"Phase 16C source hash mismatch: {rel}")
    for name, path in cfg["parents"].items():
        if _sha(root / path) != lock["parent_sha256"][name]:
            raise RuntimeError(f"Phase16C parent hash mismatch: {name}")
    audit = json.loads((root / cfg["outputs"]["layout_audit"]).read_text(encoding="utf-8"))
    if _sha(root / cfg["outputs"]["layout_audit"]) != lock.get("layout_audit_sha256"):
        raise RuntimeError("Phase16C hash-only layout audit differs from the immutable lock")
    locked_layout_hash = hashlib.sha256(json.dumps(
        audit["layouts"], sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper()
    if locked_layout_hash != lock.get("layout_hashes_sha256"):
        raise RuntimeError("Phase16C audited layout list differs from the immutable lock")
    if (audit.get("outcomes_or_transitions_computed") is not False
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("d168_seed_block_accessed") is not False
            or audit.get("overlap_historical") != 0
            or audit.get("seed_start") != cfg["environment"]["seed_start"]
            or audit.get("layout_count") != cfg["environment"]["layout_count"]):
        raise RuntimeError("Phase16C layout audit failed")
    for rel, expected in audit["historical_source_hashes"].items():
        if _sha(root / rel) != expected:
            raise RuntimeError(f"historical layout audit source changed: {rel}")
    out_dir = root / cfg["outputs"]["run_dir"]
    prepare_locked_output_directory(out_dir, root / cfg["outputs"]["layout_audit"])
    prior, train_seconds = _train_prior(root / cfg["outputs"]["checkpoint"], cfg)
    evaluation_started = time.perf_counter()
    # A single frozen training seed is used; no validation-based model selection.
    with np.load(root / cfg["parents"]["d153_rows"], allow_pickle=False) as d153_rows:
        if len(d153_rows["x_R0"]) != int(cfg["training_data"]["expected_d153_rows"]):
            raise RuntimeError("unexpected D153 corpus row count")
    prior["model"].eval()
    scenarios = [generate_staggered_corridor(seed) for seed in range(
        int(cfg["environment"]["seed_start"]),
        int(cfg["environment"]["seed_start"]) + int(cfg["environment"]["layout_count"]))]
    expected_layouts = {int(row["seed"]): row["layout_sha256"] for row in audit["layouts"]}
    if any(expected_layouts.get(scenario.seed) != scenario.layout_sha256 for scenario in scenarios):
        raise RuntimeError("executed development layouts differ from the hash-only lock audit")
    rows: dict[str, list[Episode]] = {name: [] for name in (
        "uniform_exact_shooting", "canonical_rank_prior_exact_shooting", "geometry_waypoint")}
    for scenario in scenarios:
        rows["uniform_exact_shooting"].append(_evaluate_shooting_episode(scenario,
            "uniform_exact_shooting", None,
            horizon=int(cfg["planning"]["horizon"]),
            budget=int(cfg["planning"]["sequences_per_decision"]),
            attempt_multiplier=int(cfg["planning"]["maximum_attempt_multiplier"]),
            query_seed_multiplier=int(cfg["planning"]["query_seed_multiplier"])))
        rows["canonical_rank_prior_exact_shooting"].append(_evaluate_shooting_episode(scenario,
            "canonical_rank_prior_exact_shooting", prior,
            horizon=int(cfg["planning"]["horizon"]),
            budget=int(cfg["planning"]["sequences_per_decision"]),
            attempt_multiplier=int(cfg["planning"]["maximum_attempt_multiplier"]),
            query_seed_multiplier=int(cfg["planning"]["query_seed_multiplier"])))
        waypoint = evaluate_corridor_episode(scenario, "geometry_waypoint",
            parameters={"resolution": float(cfg["evaluation"]["waypoint"]["resolution"]),
                        "waypoint_tolerance": float(cfg["evaluation"]["waypoint"]["tolerance"])})
        rows["geometry_waypoint"].append(Episode(scenario.seed, scenario.layout_sha256,
            "geometry_waypoint", waypoint.outcome, waypoint.steps, waypoint.decision_count,
            0, 0, 0, 0, 0, (waypoint.wall_seconds_per_decision or 0.0,)))
    indexed = {method: {row.seed: row for row in values} for method, values in rows.items()}
    uniform, guided = indexed["uniform_exact_shooting"], indexed["canonical_rank_prior_exact_shooting"]
    if set(uniform) != set(guided) or any(
            uniform[seed].layout_sha256 != guided[seed].layout_sha256 for seed in uniform):
        raise RuntimeError("paired method layout identities do not align")
    def outcome_vector(method: str, event: str) -> list[float]:
        return [float(row.outcome == event) for row in indexed[method].values()]

    bootstrap_cfg = cfg["evaluation"]["paired_bootstrap"]
    success_ci = paired_bootstrap(outcome_vector("uniform_exact_shooting", "success"),
                                  outcome_vector("canonical_rank_prior_exact_shooting", "success"),
                                  int(bootstrap_cfg["seed"]), int(bootstrap_cfg["draws"]))
    collision_ci = paired_bootstrap(outcome_vector("uniform_exact_shooting", "collision"),
                                    outcome_vector("canonical_rank_prior_exact_shooting", "collision"),
                                    int(bootstrap_cfg["seed"]), int(bootstrap_cfg["draws"]))
    gate_cfg = cfg["gates"]["useful_prior_guidance"]
    result = {"schema": "phase16c-result-v1", "status": "completed_fresh_development_test",
        "protocol_sha256": _sha(protocol_path), "config_sha256": _sha(config_path),
        "source_commit": lock["source_commit"],
        "run_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "lock_sha256": _sha(lock_path), "layout_audit_sha256": _sha(root / cfg["outputs"]["layout_audit"]),
        "prior_checkpoint_sha256": _sha(root / cfg["outputs"]["checkpoint"]),
        "prior_training_wall_seconds": train_seconds,
        "prior_training": {"train_rows": int(cfg["training_data"]["source_states"])
                                      * int(cfg["training_data"]["actions_per_state"])
                                      * int(cfg["training_data"]["search_replicates_per_state_action"]),
                           "source_states": int(cfg["training_data"]["source_states"]),
                           "source_layouts": int(cfg["training_data"]["source_layouts"]),
                           "epochs": int(cfg["prior"]["epochs"]),
                           "seed": int(cfg["prior"]["initialization_seed"]),
                           "device": prior["metadata"]["device"],
                           "parameter_count": prior["metadata"]["parameter_count"],
                           "best_train_bce": float(__import__("torch").load(
                               root / cfg["outputs"]["checkpoint"], map_location="cpu",
                               weights_only=False)["train_bce"]),
                           "final_train_bce": prior["metadata"]["final_train_bce"]},
        "layouts": len(scenarios), "methods": {key: _summarize(value) for key, value in rows.items()},
        "paired_primary": {"success_gain_prior_minus_uniform": success_ci,
                           "collision_increase_prior_minus_uniform": collision_ci},
        "gate_pass": bool(success_ci["difference"] >= float(gate_cfg["success_gain_at_least"])
                           and success_ci["ci95_lower"] > float(gate_cfg["paired_95_ci_lower_above"])
                           and collision_ci["ci95_upper"] <= float(
                               gate_cfg["collision_increase_95_ci_upper_at_most"])),
        "scope": {"reserve_access": False, "test_access": False, "d168_block_access": False,
                  "validation_access": False, "student_control": False, "dagger": False,
                  "learned_world_model": False},
        "software": {"python": sys.version, "platform": platform.platform(),
                     "numpy": np.__version__, "torch": importlib.metadata.version("torch"),
                     "cuda": __import__("torch").version.cuda},
        "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
        "elapsed_wall_seconds": time.perf_counter() - run_started}
    episodes_path = root / cfg["outputs"]["episodes"]
    with episodes_path.open("x", encoding="utf-8") as stream:
        for method, values in rows.items():
            for row in values:
                stream.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    result_path = root / cfg["outputs"]["result"]
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema": "phase16c-manifest-v1", "status": result["status"],
        "lock_sha256": result["lock_sha256"], "config_sha256": result["config_sha256"],
        "result_sha256": _sha(result_path), "episodes_sha256": _sha(episodes_path),
        "prior_checkpoint_sha256": result["prior_checkpoint_sha256"],
        "layout_audit_sha256": result["layout_audit_sha256"], "layout_count": len(scenarios),
        "source_commit": result["source_commit"], "run_commit": result["run_commit"],
        "software": result["software"],
        "reserve_access": False, "test_access": False, "d168_block_access": False,
        "validation_access": False, "training_source_only": "D153 train + Phase16A train labels"}
    (root / cfg["outputs"]["manifest"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
