"""Phase 17: learn a proposal prior while exact shooting retains authority."""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Sequence

import numpy as np
import yaml

from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.corridor_evaluation import evaluate_corridor_episode
from consequence_lab.inertial_corridor import CorridorState, corridor_step
from consequence_lab.phase12_teacher import state_features
from consequence_lab.phase15_d164 import canonical_agent_relative
from consequence_lab.phase16c_prior_guided import evaluate_sequence
from consequence_lab.phase16d_compute_frontier import uniform_prefix_bank, choose_lexicographic_best

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase17_learned_proposal_guidance_v1.yaml"
PILOT_LOCK = ROOT / "experiments/phase17a_teacher_pilot_v1.lock.json"
TEACHER_LOCK = ROOT / "experiments/phase17b_full_teacher_v1.lock.json"
FIT_LOCK = ROOT / "experiments/phase17b_proposal_fit_v1.lock.json"
ALPHA_LOCK = ROOT / "experiments/phase17b_alpha_screen_v3.lock.json"
FINAL_FIT_LOCK = ROOT / "experiments/phase17b_final_proposal_fit_v1.lock.json"
FRESH_AUDIT_LOCK = ROOT / "experiments/phase17c_fresh_layout_audit_v1.lock.json"
FRESH_EXECUTION_LOCK_V1 = ROOT / "experiments/phase17c_guided_planner_execution_v1.lock.json"
FRESH_EXECUTION_LOCK = ROOT / "experiments/phase17c_guided_planner_execution_v2.lock.json"
FRESH_V1_ORDER = ROOT / "artifacts/runs/phase17c_exact_guided_planner_v1/execution_order.json"
RUN_DIR = ROOT / "artifacts/runs/phase17_learned_proposal_guidance_v1"
ACTION_COUNT = 9
REPLICA_STRIDE = 2_654_435_769


class Phase17Error(RuntimeError):
    """Phase17 data, target, proposal, or integrity contract failure."""


def _build_proposal_model(input_dim: int, hidden: Sequence[int]):
    """Rebuild the exact state-dict module shape used by the fit stage."""
    from torch import nn

    class ProposalNet(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            previous = input_dim
            for width in hidden:
                layers.extend((nn.Linear(previous, int(width)), nn.ReLU()))
                previous = int(width)
            layers.append(nn.Linear(previous, ACTION_COUNT))
            self.net = nn.Sequential(*layers)

        def forward(self, values):
            return self.net(values)

    return ProposalNet()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def verify_pilot_lock(root: Path = ROOT) -> dict[str, Any]:
    """Fail closed unless the pilot's committed sources and tree are exactly locked."""
    root = Path(root).resolve()
    lock_path = root / "experiments/phase17a_teacher_pilot_v1.lock.json"
    if not lock_path.is_file():
        raise Phase17Error("Phase17A immutable pilot lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if parent != lock.get("source_commit") or status:
        raise Phase17Error("Phase17A pilot requires a clean tree at the child of its locked source commit")
    for relative, expected in lock.get("source_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"Phase17A parent/source hash mismatch: {relative}")
    for relative in lock.get("outputs_must_not_exist_before_run", []):
        if (root / relative).exists():
            raise Phase17Error(f"Phase17A output already exists; refusing overwrite: {relative}")
    return lock


def verify_teacher_collection_lock(root: Path = ROOT) -> dict[str, Any]:
    """Verify the separately locked full-target collection and pilot gate."""
    root = Path(root).resolve()
    lock_path = root / "experiments/phase17b_full_teacher_v1.lock.json"
    if not lock_path.is_file():
        raise Phase17Error("Phase17B immutable teacher-collection lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if parent != lock.get("source_commit") or status:
        raise Phase17Error("Phase17B collection requires a clean tree at the child of its locked source commit")
    for relative, expected in lock.get("source_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"Phase17B parent/source hash mismatch: {relative}")
    for relative, expected in lock.get("pilot_inputs_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"Phase17B pilot input hash mismatch: {relative}")
    pilot = json.loads((root / RUN_DIR.relative_to(ROOT) / "pilot_result.json").read_text(encoding="utf-8"))
    if pilot.get("status") != "complete" or pilot.get("teacher_signal_gate_pass") is not True:
        raise Phase17Error("Phase17B collection requires a passed Phase17A pilot gate")
    for relative in lock.get("outputs_must_not_exist_before_run", []):
        if (root / relative).exists():
            raise Phase17Error(f"Phase17B output already exists; refusing overwrite: {relative}")
    return lock


def verify_proposal_fit_lock(root: Path = ROOT) -> dict[str, Any]:
    """Verify exact teacher data, audit and frozen code before model fitting."""
    root = Path(root).resolve()
    lock_path = root / "experiments/phase17b_proposal_fit_v1.lock.json"
    if not lock_path.is_file():
        raise Phase17Error("Phase17B proposal-fit immutable lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if parent != lock.get("source_commit") or status:
        raise Phase17Error("proposal fitting requires a clean tree at the child of its locked source commit")
    for relative, expected in lock.get("source_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"proposal-fit source hash mismatch: {relative}")
    for relative, expected in lock.get("input_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"proposal-fit input hash mismatch: {relative}")
    for relative in lock.get("outputs_must_not_exist_before_run", []):
        if (root / relative).exists():
            raise Phase17Error(f"proposal-fit output already exists; refusing overwrite: {relative}")
    return lock


def verify_alpha_screen_lock(root: Path = ROOT) -> dict[str, Any]:
    """Verify frozen internal-layout exact-bank alpha screening inputs."""
    root = Path(root).resolve()
    lock_path = root / "experiments/phase17b_alpha_screen_v3.lock.json"
    if not lock_path.is_file():
        raise Phase17Error("Phase17B alpha-screen immutable lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if parent != lock.get("source_commit") or status:
        raise Phase17Error("alpha screening requires a clean tree at the child of its locked source commit")
    for relative, expected in lock.get("source_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"alpha-screen source hash mismatch: {relative}")
    for relative, expected in lock.get("input_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"alpha-screen input hash mismatch: {relative}")
    for relative in lock.get("outputs_must_not_exist_before_run", []):
        if (root / relative).exists():
            raise Phase17Error(f"alpha-screen output exists; refusing overwrite: {relative}")
    return lock


def verify_final_proposal_fit_lock(root: Path = ROOT) -> dict[str, Any]:
    """Verify selected alpha, frozen training data and final-refit source lock."""
    root = Path(root).resolve()
    lock_path = root / "experiments/phase17b_final_proposal_fit_v1.lock.json"
    if not lock_path.is_file():
        raise Phase17Error("Phase17B final proposal-refit immutable lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if parent != lock.get("source_commit") or status:
        raise Phase17Error("final proposal fit requires a clean tree at the child of its locked source commit")
    for relative, expected in lock.get("source_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"final proposal-fit source hash mismatch: {relative}")
    for relative, expected in lock.get("input_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"final proposal-fit input hash mismatch: {relative}")
    for relative in lock.get("outputs_must_not_exist_before_run", []):
        if (root / relative).exists():
            raise Phase17Error(f"final proposal-fit output exists; refusing overwrite: {relative}")
    alpha = json.loads((root / "artifacts/runs/phase17_learned_proposal_guidance_v1/alpha_screen.json").read_text(encoding="utf-8"))
    if alpha.get("selected_alpha") != lock.get("selected_alpha") or alpha.get("advancement_gate", {}).get("pass") is not True:
        raise Phase17Error("final proposal fit requires the locked alpha-screen selection and passed gate")
    return lock


def verify_fresh_layout_audit_lock(root: Path = ROOT) -> dict[str, Any]:
    """Verify source, final checkpoint and output contract before hash-only audit."""
    root = Path(root).resolve()
    lock_path = root / "experiments/phase17c_fresh_layout_audit_v1.lock.json"
    if not lock_path.is_file():
        raise Phase17Error("Phase17C hash-only layout-audit immutable lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if parent != lock.get("source_commit") or status:
        raise Phase17Error("hash-only audit requires a clean tree at the child of its locked source commit")
    for relative, expected in lock.get("source_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"fresh layout audit source hash mismatch: {relative}")
    for relative, expected in lock.get("input_files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase17Error(f"fresh layout audit input hash mismatch: {relative}")
    for relative in lock.get("outputs_must_not_exist_before_run", []):
        if (root / relative).exists():
            raise Phase17Error(f"fresh layout audit output exists; refusing overwrite: {relative}")
    return lock


def query_seed(seed: int, decision_index: int, replica: int) -> int:
    if min(seed, decision_index, replica) < 0:
        raise ValueError("seed, decision index, and replica must be nonnegative")
    return (int(seed) * 1_000_003 + int(decision_index) + int(replica) * REPLICA_STRIDE) % 2**32


def state_from_cached_row(scenario: CorridorScenario, x_r0: np.ndarray,
                          decision_index: int, tolerance: float = 1e-6) -> CorridorState:
    """Decode the four public physical state values and enforce feature parity."""
    row = np.asarray(x_r0, dtype=np.float64)
    if row.shape != (61,) or not np.isfinite(row).all() or decision_index < 0:
        raise ValueError("cached row must be a finite 61-vector and decision index nonnegative")
    spec = scenario.spec
    vx, vy = float(row[2] * spec.max_speed), float(row[3] * spec.max_speed)
    speed = math.hypot(vx, vy)
    # D153 stores normalized R0 as float32. At the speed boundary this can
    # reconstruct a physically legal vector a few ulps outside the simulator's
    # strict bound. Project only that quantization-sized excess back to the
    # closed feasible ball; reject anything larger rather than hiding a bad row.
    if speed > spec.max_speed:
        relative_excess = (speed - spec.max_speed) / spec.max_speed
        if relative_excess > tolerance:
            raise Phase17Error(f"source velocity exceeds the physical bound: {relative_excess}")
        factor = spec.max_speed / speed
        vx, vy = vx * factor, vy * factor
    state = CorridorState(float(row[0] * spec.width), float(row[1] * spec.height),
                          vx, vy,
                          int(decision_index), "active")
    computed = state_features(scenario, state).astype(np.float64)
    error = float(np.max(np.abs(computed - row[:52])))
    if error > tolerance:
        raise Phase17Error(f"source-state reconstruction parity failed: {error}")
    if state.elapsed_steps >= spec.max_steps:
        raise Phase17Error("active source state is at or beyond the task horizon")
    return state


def load_training_states(root: Path = ROOT) -> dict[str, np.ndarray]:
    """Load only cached D153 training rows and parity-check each state."""
    cfg = load_config()
    path = Path(root) / cfg["training_data"]["source_archive"]
    if sha256_file(path) != cfg["training_data"]["source_sha256"]:
        raise Phase17Error("D168 replay archive hash differs from frozen Phase17 parent")
    d153_path = Path(root) / cfg["training_data"]["d153_parent_archive"]
    if sha256_file(d153_path) != cfg["training_data"]["d153_parent_sha256"]:
        raise Phase17Error("D153 parent archive hash differs from frozen Phase17 parent")
    integrity_path = Path(root) / cfg["training_data"]["d168_integrity_result"]
    if sha256_file(integrity_path) != cfg["training_data"]["d168_integrity_result_sha256"]:
        raise Phase17Error("D168 integrity result differs from frozen Phase17 parent")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if (integrity.get("status") != "G0_passed_training_replay_parity"
            or integrity.get("reserve_access") is not False
            or integrity.get("test_access") is not False
            or integrity.get("training_started") is not False
            or integrity.get("training_data_sha256") != cfg["training_data"]["source_sha256"]
            or integrity.get("integrity", {}).get("source_sha256") != cfg["training_data"]["d153_parent_sha256"]):
        raise Phase17Error("D168 G0 provenance contract failed")
    with np.load(path, allow_pickle=False) as archive:
        required = {"x_R0", "x_canonical", "signatures", "layouts", "seeds", "decision_index"}
        if set(archive.files) != required and set(archive.files) != required | {
                "summary_R4", "collector_action_ids_audit_only", "branch_offsets",
                "branch_states", "branch_continuation_query_counts", "branch_state_sha256",
                "teacher_regret", "physical_actions"}:
            raise Phase17Error("cached D168 training replay schema changed")
        raw = np.asarray(archive["x_R0"], dtype=np.float64)
        canonical = np.asarray(archive["x_canonical"], dtype=np.float64)
        layouts = archive["layouts"].astype(str)
        seeds = np.asarray(archive["seeds"], dtype=np.int64)
        decisions = np.asarray(archive["decision_index"], dtype=np.int64)
    n = int(cfg["training_data"]["source_states"])
    if (raw.shape != (n, 61) or canonical.shape != (n, 63)
            or layouts.shape != (n,) or seeds.shape != (n,) or decisions.shape != (n,)
            or not np.isfinite(raw).all() or not np.isfinite(canonical).all()):
        raise Phase17Error("cached training row dimensions/finite contract failed")
    if len(set(layouts.tolist())) != int(cfg["training_data"]["source_layouts"]):
        raise Phase17Error("cached training layout count differs from locked contract")
    scenarios: dict[int, CorridorScenario] = {}
    for i in range(n):
        seed = int(seeds[i])
        if seed not in scenarios:
            scenarios[seed] = generate_staggered_corridor(seed)
        scenario = scenarios[seed]
        if layouts[i] != scenario.layout_sha256:
            raise Phase17Error(f"layout hash mismatch at source row {i}")
        state = state_from_cached_row(scenario, raw[i], int(decisions[i]))
        derived = canonical_agent_relative(raw[i:i + 1, :52], raw[i:i + 1, 52:])[0]
        if not np.allclose(derived, canonical[i], atol=1e-6, rtol=0.0):
            raise Phase17Error(f"canonical feature parity failed at source row {i}")
        if state.status != "active":
            raise Phase17Error(f"non-active source state at row {i}")
    return {"x_R0": raw.astype(np.float32), "x_canonical": canonical.astype(np.float32),
            "layouts": layouts, "seeds": seeds, "decision_index": decisions}


def layout_split(layouts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(str(x) for x in layouts))
    validation = set(unique[::5])
    if len(unique) != 140 or len(validation) != 28:
        raise ValueError("Phase17 split expects 140 unique layouts and exactly 28 validation layouts")
    row_layouts = np.asarray(layouts).astype(str)
    val = np.asarray([x in validation for x in row_layouts], dtype=bool)
    return np.flatnonzero(~val), np.flatnonzero(val)


def pilot_row_indices(data: dict[str, np.ndarray], count: int = 64,
                      seed: int = 17001) -> np.ndarray:
    """Choose one source row in each of count distinct training layouts."""
    unique = sorted(set(data["layouts"].tolist()))
    if count <= 0 or count > len(unique):
        raise ValueError("pilot count must fit the available layouts")
    rng = np.random.default_rng(seed)
    selected_layouts = sorted(rng.choice(unique, size=count, replace=False).tolist())
    rows = []
    for layout in selected_layouts:
        options = np.flatnonzero(data["layouts"] == layout)
        rows.append(int(options[int(rng.integers(0, len(options)))]))
    return np.asarray(rows, dtype=np.int64)


def tie_mass_target(evaluations: Sequence[Any]) -> tuple[np.ndarray, int, tuple[int, ...], tuple[float, float, float]]:
    """Equal mass over first actions among exact lexicographic best sequences."""
    if not evaluations:
        raise ValueError("at least one exact candidate evaluation is required")
    best_key = min(item.objective for item in evaluations)
    tied = [item for item in evaluations if item.objective == best_key]
    if not tied:
        raise Phase17Error("exact best-set construction unexpectedly produced an empty set")
    probabilities = np.zeros(ACTION_COUNT, dtype=np.float64)
    for item in tied:
        action = int(item.sequence[0])
        if action not in range(ACTION_COUNT):
            raise ValueError("candidate sequence has illegal first action")
        probabilities[action] += 1.0 / len(tied)
    return probabilities, len(tied), tuple(int(x.sequence[0]) for x in tied), best_key


def evaluate_teacher_search(scenario: CorridorScenario, state: CorridorState,
                            replica: int) -> dict[str, Any]:
    horizon = min(8, scenario.spec.max_steps - state.elapsed_steps)
    seed = query_seed(scenario.seed, state.elapsed_steps, replica)
    bank, duplicates, fallback = uniform_prefix_bank(horizon, 64, seed)
    evaluations = [evaluate_sequence(state, scenario, sequence) for sequence in bank]
    target, tie_count, tied_actions, best_key = tie_mass_target(evaluations)
    chosen, _ = choose_lexicographic_best(evaluations)
    transitions = sum(int(item.transitions) for item in evaluations)
    return {
        "replica": int(replica), "query_seed": int(seed),
        "tie_mass": target.tolist(), "tie_count": int(tie_count),
        "tied_first_actions": list(tied_actions),
        "tied_best_sequences": [list(item.sequence) for item in evaluations
                                if item.objective == best_key],
        "candidate_objective_keys": [list(item.objective) for item in evaluations],
        "lexicographic_winner": int(chosen.sequence[0]),
        "best_objective": list(best_key),
        "winner_status": {"success": int(best_key[0] == 0),
                          "active_horizon": int(best_key[0] == 1),
                          "timeout": int(best_key[0] == 2),
                          "collision": int(best_key[0] == 3)},
        "steps_if_success": float(best_key[1]) if best_key[0] == 0 else None,
        "effort": float(best_key[2]),
        "success_candidates": int(sum(item.success_reached for item in evaluations)),
        "exact_sequences": len(bank), "exact_transitions": transitions,
        "objective_tie": int(tie_count > 1),
        "duplicate_rejections": int(duplicates), "fallback_sequences": int(fallback),
    }


def _state_for_row(data: dict[str, np.ndarray], row: int) -> tuple[CorridorScenario, CorridorState]:
    scenario = generate_staggered_corridor(int(data["seeds"][row]))
    state = state_from_cached_row(scenario, data["x_R0"][row], int(data["decision_index"][row]))
    return scenario, state


def collect_state_teacher(data: dict[str, np.ndarray], row: int,
                          replicas: int = 32) -> dict[str, Any]:
    scenario, state = _state_for_row(data, int(row))
    if int(data["decision_index"][row]) != state.elapsed_steps:
        raise Phase17Error("saved decision index differs from reconstructed elapsed step")
    records = [evaluate_teacher_search(scenario, state, replica) for replica in range(replicas)]
    tie_targets = np.asarray([item["tie_mass"] for item in records], dtype=np.float64)
    winner_ids = np.asarray([item["lexicographic_winner"] for item in records], dtype=np.int64)
    frequencies = tie_targets.mean(axis=0)
    winner_frequency = np.bincount(winner_ids, minlength=ACTION_COUNT) / replicas
    records_by_replica = {k: [record[k] for record in records]
                          for k in ("steps_if_success", "effort", "success_candidates", "exact_transitions")}
    cumulative = {}
    for count in (1, 2, 4, 8, 16, 32):
        cumulative[str(count)] = tie_targets[:count].mean(axis=0).tolist()
    return {
        "row": int(row), "seed": int(data["seeds"][row]),
        "decision_index": int(data["decision_index"][row]),
        "layout_sha256": str(data["layouts"][row]),
        "tie_mass_mean": frequencies.tolist(),
        "lexicographic_winner_frequency": winner_frequency.tolist(),
        "entropy_normalized": float(-np.sum(frequencies[frequencies > 0]
                                             * np.log(frequencies[frequencies > 0])) / math.log(9)),
        "top1_mass": float(np.max(frequencies)),
        "top2_mass": float(np.sort(frequencies)[-2:].sum()),
        "replica_records": records,
        "replica_summary": records_by_replica,
        "cumulative_tie_mass": cumulative,
        "replicates": int(replicas),
    }


def pilot_distribution_metrics(rows: Sequence[dict[str, Any]], bootstrap_draws: int = 10_000,
                               seed: int = 17001) -> dict[str, Any]:
    """Test whether half-sample teacher distributions predict independent repeats."""
    if not rows:
        raise ValueError("pilot rows are required")
    per_layout: dict[str, list[float]] = defaultdict(list)
    all_entropy, top1, top2, tie_rates = [], [], [], []
    outcome_rates, collision_rates, success_rates, cost_samples = [], [], [], []
    outcome_entropies, success_step_stds, effort_stds = [], [], []
    for row in rows:
        repeats = row["replica_records"]
        if len(repeats) != 32:
            raise ValueError("pilot teacher rows must contain exactly 32 repeats")
        targets = np.asarray([record["tie_mass"] for record in repeats], dtype=np.float64)
        first, second = targets[:16].mean(axis=0), targets[16:].mean(axis=0)
        smoothed = (first * 16.0 + 0.5) / (16.0 + ACTION_COUNT * 0.5)
        second_ce = float(-np.sum(second * np.log(np.clip(smoothed, 1e-12, 1))))
        reverse_smoothed = (second * 16.0 + 0.5) / (16.0 + ACTION_COUNT * 0.5)
        first_ce = float(-np.sum(first * np.log(np.clip(reverse_smoothed, 1e-12, 1))))
        gain = math.log(ACTION_COUNT) - 0.5 * (first_ce + second_ce)
        per_layout[row["layout_sha256"]].append(gain)
        p = np.asarray(row["tie_mass_mean"], dtype=np.float64)
        all_entropy.append(float(-np.sum(p[p > 0] * np.log(p[p > 0])) / math.log(9)))
        top1.append(float(np.max(p)))
        top2.append(float(np.sort(p)[-2:].sum()))
        tie_rates.append(float(np.mean([x["tie_count"] > 1 for x in repeats])))
        outcome_rates.append(float(np.mean([x["winner_status"]["success"] for x in repeats])))
        collision_rates.append(float(np.mean([x["winner_status"]["collision"] for x in repeats])))
        success_rates.append(float(np.mean([x["success_candidates"] > 0 for x in repeats])))
        cost_samples.extend([tuple(x["best_objective"]) for x in repeats])
        class_counts = np.bincount([int(x["best_objective"][0]) for x in repeats], minlength=4) / len(repeats)
        nonzero = class_counts[class_counts > 0]
        outcome_entropies.append(float(-np.sum(nonzero * np.log(nonzero)) / math.log(4)))
        success_steps = [float(x["steps_if_success"]) for x in repeats
                         if x["steps_if_success"] is not None]
        success_step_stds.append(float(np.std(success_steps)) if len(success_steps) > 1 else 0.0)
        effort_stds.append(float(np.std([float(x["effort"]) for x in repeats])))
    layouts = sorted(per_layout)
    values = np.asarray([np.mean(per_layout[key]) for key in layouts], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(bootstrap_draws, len(values)))
    boot = values[sampled].mean(axis=1)
    half_tv = []
    pair_agreement = []
    for row in rows:
        target = np.asarray([record["tie_mass"] for record in row["replica_records"]], dtype=float)
        p1, p2 = target[:16].mean(axis=0), target[16:].mean(axis=0)
        half_tv.append(float(0.5 * np.abs(p1 - p2).sum()))
        agree = []
        for i in range(9):
            for j in range(i + 1, 9):
                d1, d2 = p1[i] - p1[j], p2[i] - p2[j]
                if abs(d1) >= 0.10 and abs(d2) >= 0.10:
                    agree.append(float(np.sign(d1) == np.sign(d2)))
        if agree:
            pair_agreement.extend(agree)
    return {
        "states": len(rows), "layouts": len(layouts),
        "heldout_nll_gain_vs_uniform": float(values.mean()),
        "heldout_nll_gain_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "teacher_signal_gate_pass": bool(np.quantile(boot, .025) > 0.0),
        "entropy_normalized_mean": float(np.mean(all_entropy)),
        "top1_mass_mean": float(np.mean(top1)), "top2_mass_mean": float(np.mean(top2)),
        "split_half_tv_mean": float(np.mean(half_tv)),
        "split_half_tv_p90": float(np.quantile(half_tv, .90)),
        "high_margin_pair_sign_agreement": float(np.mean(pair_agreement)) if pair_agreement else None,
        "exact_objective_tie_rate": float(np.mean(tie_rates)),
        "selected_winner_success_rate": float(np.mean(outcome_rates)),
        "selected_winner_collision_rate": float(np.mean(collision_rates)),
        "any_success_candidate_rate": float(np.mean(success_rates)),
        "selected_outcome_normalized_entropy_mean": float(np.mean(outcome_entropies)),
        "success_steps_within_state_std_mean": float(np.mean(success_step_stds)),
        "effort_within_state_std_mean": float(np.mean(effort_stds)),
        "selected_objective_class_counts": {
            str(k): sum(1 for outcome, _, _ in cost_samples if outcome == k)
            for k in sorted({x[0] for x in cost_samples})},
        "prefix_distance_to_32_repeat_tie_mass": {
            str(k): float(np.mean([0.5 * np.abs(
                np.asarray(row["cumulative_tie_mass"][str(k)]) -
                np.asarray(row["tie_mass_mean"])
            ).sum() for row in rows])) for k in (1, 2, 4, 8, 16)},
        "bootstrap_draws": int(bootstrap_draws), "bootstrap_seed": int(seed),
    }


def mixture_probabilities(q: Sequence[float], alpha: float) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64)
    if values.shape != (ACTION_COUNT,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("proposal q must be a finite nonnegative nine-vector")
    total = float(values.sum())
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError("proposal q must sum to one")
    # Softmax probabilities may miss unit mass by a few float32 ulps.
    values = values / total
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    result = float(alpha) * values + (1.0 - float(alpha)) / ACTION_COUNT
    return result / result.sum()


def guided_sequence_bank(horizon: int, budget: int, seed: int,
                         q: Sequence[float] | None = None, alpha: float = 0.0,
                         max_attempts: int = 1_280_000):
    """Unique bank with mandatory constants; only residual first actions may be guided."""
    if min(horizon, budget, max_attempts) <= 0 or seed < 0:
        raise ValueError("horizon, budget, attempts must be positive and seed nonnegative")
    if q is None:
        if alpha != 0.0:
            raise ValueError("nonzero learned mass requires q")
        p = np.full(ACTION_COUNT, 1 / ACTION_COUNT, dtype=np.float64)
    else:
        p = mixture_probabilities(q, alpha)
    total = min(int(budget), ACTION_COUNT ** int(horizon))
    proposals = [(a,) * int(horizon) for a in range(ACTION_COUNT)][:total]
    seen = set(proposals)
    if len(proposals) == total:
        return tuple(proposals), 0, 0
    rng = random.Random(int(seed))
    duplicates = attempts = 0
    cumulative = np.cumsum(p)
    while len(proposals) < total and attempts < max_attempts:
        attempts += 1
        first = min(ACTION_COUNT - 1, int(np.searchsorted(cumulative, rng.random(), side="right")))
        sequence = (first,) + tuple(rng.randrange(ACTION_COUNT) for _ in range(horizon - 1))
        if sequence in seen:
            duplicates += 1
            continue
        proposals.append(sequence)
        seen.add(sequence)
    fallback = 0
    if len(proposals) < total:
        fill = random.Random(int(seed) ^ 0x9E3779B9)
        while len(proposals) < total:
            sequence = tuple(fill.randrange(ACTION_COUNT) for _ in range(horizon))
            if sequence not in seen:
                proposals.append(sequence)
                seen.add(sequence)
                fallback += 1
    if len(proposals) != total or len(seen) != total:
        raise Phase17Error("proposal sampler failed exact unique budget")
    return tuple(proposals), duplicates, fallback


def _jsonl_append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def run_pilot(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    lock = verify_pilot_lock(root)
    data = load_training_states(root)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    pilot_path = RUN_DIR / "pilot_queries.jsonl"
    if pilot_path.exists():
        raise Phase17Error("pilot output already exists; refuse overwrite")
    indices = pilot_row_indices(data)
    rows = []
    for position, row_id in enumerate(indices, start=1):
        row = collect_state_teacher(data, int(row_id), 32)
        _jsonl_append(pilot_path, row)
        rows.append(row)
        print(f"Phase17A pilot: {position}/{len(indices)} states complete", flush=True)
    result = pilot_distribution_metrics(rows)
    result.update({"schema": "phase17-teacher-pilot-v1", "status": "complete",
                   "pilot_lock_sha256": sha256_file(root / "experiments/phase17a_teacher_pilot_v1.lock.json"),
                   "source_commit": lock["source_commit"],
                   "source_archive_sha256": sha256_file(root / load_config()["training_data"]["source_archive"]),
                   "pilot_rows_sha256": sha256_file(pilot_path),
                   "source_rows": indices.tolist(), "reserve_access": False,
                   "fresh_layout_outcomes": False})
    result_path = RUN_DIR / "pilot_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_full_teacher(root: Path = ROOT) -> dict[str, Any]:
    """Collect 32 repeated exact B64 best-set targets for each D153 training state."""
    root = Path(root).resolve()
    lock = verify_teacher_collection_lock(root)
    data = load_training_states(root)
    pilot_path = RUN_DIR / "pilot_queries.jsonl"
    pilot_result_path = RUN_DIR / "pilot_result.json"
    if not pilot_path.is_file() or not pilot_result_path.is_file():
        raise Phase17Error("teacher pilot is incomplete")
    pilot_result = json.loads(pilot_result_path.read_text(encoding="utf-8"))
    if pilot_result.get("teacher_signal_gate_pass") is not True:
        raise Phase17Error("teacher-signal feasibility gate failed; stop before full target collection")
    teacher_path = RUN_DIR / "teacher_queries.jsonl"
    if teacher_path.exists():
        raise Phase17Error("full teacher file already exists; refuse overwrite")
    cached = {}
    with pilot_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            cached[int(item["row"])] = item
    if len(cached) != 64:
        raise Phase17Error("pilot cache row count differs from frozen 64-state design")
    start = time.perf_counter()
    for row_id in range(len(data["x_R0"])):
        row = cached.get(row_id)
        if row is None:
            row = collect_state_teacher(data, row_id, 32)
        _jsonl_append(teacher_path, row)
    result = {"schema": "phase17-full-teacher-v1", "rows": len(data["x_R0"]),
              "layouts": len(set(data["layouts"].tolist())),
              "repeats_per_state": 32, "fit_replicates": [0, 16],
              "holdout_replicates": [16, 32],
              "teacher_sha256": sha256_file(teacher_path),
              "collection_lock_sha256": sha256_file(root / "experiments/phase17b_full_teacher_v1.lock.json"),
              "source_commit": lock["source_commit"],
              "pilot_sha256": sha256_file(pilot_path),
              "elapsed_seconds": time.perf_counter() - start,
              "reserve_access": False, "test_access": False}
    (RUN_DIR / "teacher_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def load_teacher_targets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    if len(rows) != 1365 or [int(x["row"]) for x in rows] != list(range(1365)):
        raise Phase17Error("full teacher rows missing, duplicated, or reordered")
    targets = np.empty((len(rows), ACTION_COUNT), dtype=np.float32)
    heldout = np.empty_like(targets)
    for i, row in enumerate(rows):
        records = row["replica_records"]
        if len(records) != 32:
            raise Phase17Error("teacher repeat count mismatch")
        target_rows = np.asarray([record["tie_mass"] for record in records], dtype=np.float64)
        if target_rows.shape != (32, 9) or not np.isfinite(target_rows).all():
            raise Phase17Error("malformed tie-mass target")
        if not np.allclose(target_rows.sum(axis=1), 1.0, atol=1e-10, rtol=0):
            raise Phase17Error("teacher tie-mass distributions must sum to one")
        targets[i] = target_rows[:16].mean(axis=0)
        heldout[i] = target_rows[16:].mean(axis=0)
    return targets, heldout


def audit_full_teacher_dataset(root: Path = ROOT) -> dict[str, Any]:
    """Audit every stored best set against its deterministic B64 bank and source row."""
    root = Path(root).resolve()
    data = load_training_states(root)
    run_dir = root / RUN_DIR.relative_to(ROOT)
    teacher_path = run_dir / "teacher_queries.jsonl"
    summary_path = run_dir / "teacher_summary.json"
    if not teacher_path.is_file() or not summary_path.is_file():
        raise Phase17Error("full teacher collection is incomplete")
    expected_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (sha256_file(teacher_path) != expected_summary.get("teacher_sha256")
            or expected_summary.get("collection_lock_sha256") !=
            sha256_file(root / "experiments/phase17b_full_teacher_v1.lock.json")
            or expected_summary.get("source_commit") !=
            json.loads((root / "experiments/phase17b_full_teacher_v1.lock.json").read_text(
                encoding="utf-8")).get("source_commit")):
        raise Phase17Error("full teacher artifact hash/provenance mismatch")
    pilot_path = run_dir / "pilot_queries.jsonl"
    pilot_rows = {}
    with pilot_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            pilot_rows[int(item["row"])] = item
    seen_rows, total_queries, total_sequences = 0, 0, 0
    total_transitions, total_duplicates = 0, 0
    objective_classes = np.zeros(4, dtype=np.int64)
    with teacher_path.open("r", encoding="utf-8") as stream:
        for expected_row, line in enumerate(stream):
            item = json.loads(line)
            row = int(item.get("row", -1))
            if row != expected_row or row >= len(data["x_R0"]):
                raise Phase17Error("teacher row order/coverage is invalid")
            if (int(item["seed"]) != int(data["seeds"][row])
                    or int(item["decision_index"]) != int(data["decision_index"][row])
                    or item["layout_sha256"] != str(data["layouts"][row])):
                raise Phase17Error(f"teacher provenance mismatch at row {row}")
            scenario = generate_staggered_corridor(int(item["seed"]))
            if row in pilot_rows and item != pilot_rows[row]:
                raise Phase17Error(f"pilot cache row differs from full teacher row {row}")
            records = item.get("replica_records")
            if not isinstance(records, list) or len(records) != 32:
                raise Phase17Error(f"repetition count mismatch at row {row}")
            for replica, record in enumerate(records):
                seed = query_seed(int(item["seed"]), int(item["decision_index"]), replica)
                if int(record.get("replica", -1)) != replica or int(record.get("query_seed", -1)) != seed:
                    raise Phase17Error(f"teacher query seed/replica mismatch at row {row}")
                objectives = record.get("candidate_objective_keys")
                if not isinstance(objectives, list) or len(objectives) != 64:
                    raise Phase17Error(f"candidate objective count mismatch at row {row}")
                normalized_keys = []
                for key in objectives:
                    if (not isinstance(key, list) or len(key) != 3
                            or int(key[0]) not in (0, 1, 2, 3)
                            or not all(np.isfinite(float(value)) for value in key[1:])):
                        raise Phase17Error(f"malformed exact objective at row {row}")
                    normalized_keys.append((int(key[0]), float(key[1]), float(key[2])))
                best_key = min(normalized_keys)
                if list(best_key) != record.get("best_objective"):
                    raise Phase17Error(f"stored best objective is not the candidate minimum at row {row}")
                horizon = min(8, scenario.spec.max_steps - int(item["decision_index"]))
                if horizon < 1:
                    raise Phase17Error(f"teacher query begins beyond task horizon at row {row}")
                bank, _, _ = uniform_prefix_bank(horizon, 64, seed)
                tied = [sequence for sequence, key in zip(bank, normalized_keys) if key == best_key]
                stored_tied = [tuple(int(a) for a in sequence)
                               for sequence in record.get("tied_best_sequences", [])]
                if stored_tied != tied or len(tied) != int(record.get("tie_count", -1)):
                    raise Phase17Error(f"exact tied best set mismatch at row {row}")
                chosen = min(tied)
                if int(record.get("lexicographic_winner", -1)) != chosen[0]:
                    raise Phase17Error(f"lexicographic exact winner mismatch at row {row}")
                expected_target = np.bincount([sequence[0] for sequence in tied], minlength=9) / len(tied)
                target = np.asarray(record.get("tie_mass"), dtype=np.float64)
                if target.shape != (9,) or not np.allclose(target, expected_target, atol=1e-12, rtol=0):
                    raise Phase17Error(f"tie-mass target mismatch at row {row}")
                total_queries += 1
                total_sequences += 64
                total_transitions += int(record["exact_transitions"])
                total_duplicates += int(record["duplicate_rejections"])
                objective_classes[best_key[0]] += 1
            seen_rows += 1
    if seen_rows != 1365 or len(pilot_rows) != 64:
        raise Phase17Error("full teacher dataset or pilot coverage is incomplete")
    result = {
        "schema": "phase17-full-teacher-audit-v1", "status": "passed",
        "rows": seen_rows, "layouts": len(set(data["layouts"].tolist())),
        "repeats_per_state": 32, "queries": total_queries,
        "candidate_sequences": total_sequences,
        "exact_rollout_transitions": total_transitions,
        "mean_transitions_per_query": total_transitions / total_queries,
        "duplicate_rejections": total_duplicates,
        "selected_best_objective_class_counts": objective_classes.tolist(),
        "teacher_queries_sha256": sha256_file(teacher_path),
        "teacher_summary_sha256": sha256_file(summary_path),
        "pilot_sha256": sha256_file(pilot_path),
        "reserve_access": False, "test_access": False,
    }
    output = run_dir / "teacher_audit.json"
    if output.exists():
        raise Phase17Error("teacher audit output exists; refusing overwrite")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def train_proposal_models(root: Path = ROOT) -> dict[str, Any]:
    """Fit the three frozen small proposal controls and save internal metrics."""
    import torch

    root = Path(root).resolve()
    lock = verify_proposal_fit_lock(root)
    cfg = load_config()
    model_cfg = cfg["proposal_model"]
    data = load_training_states(root)
    target_path = RUN_DIR / "teacher_queries.jsonl"
    audit_path = RUN_DIR / "teacher_audit.json"
    if not audit_path.is_file() or json.loads(audit_path.read_text(encoding="utf-8")).get("status") != "passed":
        raise Phase17Error("proposal fitting requires a passed exact teacher dataset audit")
    if any((RUN_DIR / name).exists() for name in ("proposal_models.pt", "proposal_fit.json")):
        raise Phase17Error("proposal-fit output exists; refusing overwrite")
    targets, heldout = load_teacher_targets(target_path)
    train_ids, val_ids = layout_split(data["layouts"])
    training_seed = int(model_cfg["random_seed"])
    torch.manual_seed(training_seed)
    torch.use_deterministic_algorithms(bool(model_cfg["deterministic"]))
    outputs = {}
    models: dict[str, dict[str, Any]] = {}
    for name, key, hard in (("canonical_soft", "x_canonical", False),
                            ("raw_r0_soft", "x_R0", False),
                            ("canonical_hard", "x_canonical", True)):
        x = data[key].astype(np.float32)
        mean = x[train_ids].mean(axis=0)
        scale = x[train_ids].std(axis=0)
        scale[scale < 1e-8] = 1.0
        x = ((x - mean) / scale).astype(np.float32)
        y = targets.copy()
        if hard:
            maxima = y.max(axis=1, keepdims=True)
            y = (y == maxima).astype(np.float32)
            y /= y.sum(axis=1, keepdims=True)
        input_dim = x.shape[1]

        hidden = [int(width) for width in model_cfg["hidden"]]

        torch.manual_seed(training_seed + len(outputs))
        model = _build_proposal_model(input_dim, hidden)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=float(model_cfg["learning_rate"]),
                                      weight_decay=float(model_cfg["weight_decay"]))
        tx = torch.as_tensor(x[train_ids])
        ty = torch.as_tensor(y[train_ids])
        vx = torch.as_tensor(x[val_ids])
        vy = torch.as_tensor(heldout[val_ids])
        best_loss, best_state, best_optimizer, best_epoch = float("inf"), None, None, None
        train_losses, val_losses = [], []
        for epoch in range(1, int(model_cfg["max_epochs"]) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logp = torch.log_softmax(model(tx), dim=1)
            loss = -(ty * logp).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                train_nll = float((-(ty * torch.log_softmax(model(tx), dim=1)).sum(dim=1).mean()).item())
                val_nll = float((-(vy * torch.log_softmax(model(vx), dim=1)).sum(dim=1).mean()).item())
            train_losses.append(train_nll)
            val_losses.append(val_nll)
            if val_nll < best_loss:
                best_loss, best_epoch = val_nll, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_optimizer = copy.deepcopy(optimizer.state_dict())
        if best_state is None:
            raise Phase17Error(f"{name} produced no checkpoint")
        model.load_state_dict(best_state)
        with torch.no_grad():
            train_p = torch.softmax(model(tx), dim=1).cpu().numpy()
            validation_p = torch.softmax(model(vx), dim=1).cpu().numpy()
        global_p = (targets[train_ids].sum(axis=0) + .5)
        global_p = global_p / global_p.sum()
        val_y = heldout[val_ids]
        nll = float(-np.sum(val_y * np.log(np.clip(validation_p, 1e-12, 1))) / len(val_ids))
        train_nll_selected = float(-np.sum(ty.numpy() * np.log(np.clip(train_p, 1e-12, 1))) / len(train_ids))
        global_nll = float(-np.sum(val_y * np.log(global_p[None, :])) / len(val_ids))
        brier = float(np.mean(np.sum((validation_p - val_y) ** 2, axis=1)))
        predicted_entropy = -np.sum(validation_p * np.log(np.clip(validation_p, 1e-12, 1)), axis=1)
        teacher_entropy = -np.sum(val_y * np.log(np.clip(val_y, 1e-12, 1)), axis=1)
        confidence = np.max(validation_p, axis=1)
        selected = np.argmax(validation_p, axis=1)
        selected_target_mass = val_y[np.arange(len(val_y)), selected]
        ece = 0.0
        for lower in np.linspace(0.0, 0.9, 10):
            upper = lower + 0.1
            mask = ((confidence >= lower) & (confidence <= upper if lower >= 0.9
                                               else confidence < upper))
            if np.any(mask):
                ece += float(mask.mean()) * abs(float(confidence[mask].mean()) -
                                                float(selected_target_mass[mask].mean()))
        outputs[name] = {"best_epoch": int(best_epoch), "best_validation_nll": best_loss,
                         "validation_nll_recomputed": nll, "global_prior_nll": global_nll,
                         "uniform_nll": float(math.log(9)),
                         "training_nll_at_checkpoint": train_nll_selected,
                         "validation_brier": brier,
                         "validation_ece_soft_top1": float(ece),
                         "validation_predicted_entropy_mean": float(predicted_entropy.mean()),
                         "validation_teacher_entropy_mean": float(teacher_entropy.mean()),
                         "validation_top1_mass_mean": float(confidence.mean()),
                         "validation_top2_mass_mean": float(np.sort(validation_p, axis=1)[:, -2:].sum(axis=1).mean()),
                         "training_rows": int(len(train_ids)),
                         "validation_rows": int(len(val_ids)),
                         "parameter_count": int(sum(p.numel() for p in model.parameters())),
                         "training_curve_nll": train_losses,
                         "validation_curve_nll": val_losses}
        models[name] = {"model": model, "mean": mean, "scale": scale,
                        "features": key, "metadata": outputs[name],
                        "optimizer_state": best_optimizer}
    torch.save({name: {"state_dict": pack["model"].state_dict(),
                       "mean": pack["mean"], "scale": pack["scale"],
                       "features": pack["features"], "metadata": pack["metadata"],
                       "optimizer_state": pack["optimizer_state"]}
                for name, pack in models.items()}, RUN_DIR / "proposal_models.pt")
    report = {"schema": "phase17-proposal-fit-v1", "layout_split": {
                  "fit_layouts": len(set(data["layouts"][train_ids].tolist())),
                  "validation_layouts": len(set(data["layouts"][val_ids].tolist()))},
              "models": outputs,
              "fit_lock_sha256": sha256_file(root / "experiments/phase17b_proposal_fit_v1.lock.json"),
              "source_commit": lock["source_commit"],
              "target_sha256": sha256_file(target_path),
              "teacher_audit_sha256": sha256_file(audit_path),
              "training_seed": training_seed, "max_epochs": int(model_cfg["max_epochs"]),
              "checkpoint_rule": "minimum heldout-layout NLL using repeats 16-31",
              "reserve_access": False, "fresh_outcomes": False}
    (RUN_DIR / "proposal_fit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    return report


def screen_alpha(root: Path = ROOT) -> dict[str, Any]:
    """Choose alpha using exact B32 candidate banks on held-out D153 layouts only."""
    import torch

    root = Path(root).resolve()
    lock = verify_alpha_screen_lock(root)
    cfg = load_config()
    run_dir = root / cfg["outputs"]["run_dir"]
    data = load_training_states(root)
    _, val_ids = layout_split(data["layouts"])
    saved = torch.load(run_dir / "proposal_models.pt", map_location="cpu", weights_only=False)
    item = saved["canonical_soft"]
    input_dim = int(np.asarray(item["mean"]).shape[0])
    if input_dim != 63:
        raise Phase17Error("canonical proposal model input dimension differs from frozen contract")
    model = _build_proposal_model(input_dim, cfg["proposal_model"]["hidden"])
    model.load_state_dict(item["state_dict"])
    model.eval()
    feature_rows = data["x_canonical"]
    normalized = (feature_rows - item["mean"]) / item["scale"]
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.as_tensor(normalized)), dim=1).numpy()
    global_p = (np.asarray(load_teacher_targets(run_dir / "teacher_queries.jsonl")[0])[layout_split(
        data["layouts"])[0]].sum(axis=0) + .5)
    global_p = global_p / global_p.sum()
    metrics: dict[str, list[dict[str, float]]] = {"uniform": [], "global": []}
    alphas = [float(x) for x in cfg["planning"]["learned_mass_alpha_candidates"]]
    metrics.update({f"alpha_{a:.2f}": [] for a in alphas})
    for row_id in val_ids:
        scenario, state = _state_for_row(data, int(row_id))
        query = query_seed(int(data["seeds"][row_id]), int(data["decision_index"][row_id]), 500)
        arms = [("uniform", np.full(9, 1 / 9))]
        arms.append(("global", global_p))
        arms.extend((f"alpha_{a:.2f}", mixture_probabilities(probabilities[row_id], a)) for a in alphas)
        for arm, p in arms:
            bank, duplicates, fallback = guided_sequence_bank(8, 32, query, p, 1.0)
            evaluated = [evaluate_sequence(state, scenario, seq) for seq in bank]
            exact_success = sum(x.success_reached for x in evaluated)
            metrics[arm].append({"layout": str(data["layouts"][row_id]),
                                 "success_available": float(exact_success > 0),
                                 "collision_candidates": float(sum(x.objective[0] == 3 for x in evaluated) / len(evaluated)),
                                 "best_class": float(min(x.objective[0] for x in evaluated)),
                                 "duplicates": float(duplicates), "fallback": float(fallback)})
    grouped = {}
    for arm, rows in metrics.items():
        by_layout: dict[str, list[dict[str, float]]] = defaultdict(list)
        for row in rows:
            by_layout[row["layout"]].append(row)
        grouped[arm] = {
            "states": len(rows), "layouts": len(by_layout),
            "layout_macro_success_available": float(np.mean([np.mean([x["success_available"] for x in v])
                                                               for v in by_layout.values()])),
            "layout_macro_collision_candidate_fraction": float(np.mean([np.mean([x["collision_candidates"] for x in v])
                                                                          for v in by_layout.values()])),
            "layout_macro_best_objective_class": float(np.mean([np.mean([x["best_class"] for x in v])
                                                                  for v in by_layout.values()])),
        }
    chosen = min(alphas, key=lambda a: (-grouped[f"alpha_{a:.2f}"]["layout_macro_success_available"],
                                        grouped[f"alpha_{a:.2f}"]["layout_macro_collision_candidate_fraction"], a))
    advancement_gate = alpha_screen_advancement_gate(grouped[f"alpha_{chosen:.2f}"], grouped["uniform"])
    result = {"schema": "phase17-alpha-screen-v1", "arms": grouped,
              "selected_alpha": float(chosen), "selection_rule": "max success availability; min collision candidate fraction; min alpha",
              "advancement_gate": advancement_gate,
              "reserve_access": False, "fresh_outcomes": False}
    result["alpha_lock_sha256"] = sha256_file(root / "experiments/phase17b_alpha_screen_v3.lock.json")
    result["source_commit"] = lock["source_commit"]
    output = run_dir / "alpha_screen.json"
    if output.exists():
        raise Phase17Error("alpha-screen output exists; refusing overwrite")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def fit_final_proposal(root: Path = ROOT) -> dict[str, Any]:
    """Fixed-epoch refit of the selected soft proposal on all training layouts."""
    import torch

    root = Path(root).resolve()
    lock = verify_final_proposal_fit_lock(root)
    cfg = load_config()
    run_dir = root / cfg["outputs"]["run_dir"]
    checkpoint_path = root / cfg["outputs"]["proposal_checkpoint"]
    manifest_path = root / cfg["outputs"]["final_fit_manifest"]
    if checkpoint_path.exists() or manifest_path.exists():
        raise Phase17Error("final proposal-fit output exists; refusing overwrite")
    data = load_training_states(root)
    targets, _ = load_teacher_targets(run_dir / "teacher_queries.jsonl")
    saved = torch.load(run_dir / "proposal_models.pt", map_location="cpu", weights_only=False)
    selection = cfg["proposal_model"]["final_refit"]
    if selection["architecture"] != "canonical_soft_unchanged":
        raise Phase17Error("final proposal refit architecture differs from locked primary")
    base = saved["canonical_soft"]
    x = ((data["x_canonical"].astype(np.float32) - base["mean"]) / base["scale"]).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if x.shape != (1365, 63) or y.shape != (1365, ACTION_COUNT):
        raise Phase17Error("final refit data dimensions differ from frozen contract")
    torch.manual_seed(int(selection["random_seed"]))
    torch.use_deterministic_algorithms(bool(cfg["proposal_model"]["deterministic"]))
    model = _build_proposal_model(63, cfg["proposal_model"]["hidden"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg["proposal_model"]["learning_rate"]),
                                  weight_decay=float(cfg["proposal_model"]["weight_decay"]))
    tx = torch.as_tensor(x)
    ty = torch.as_tensor(y)
    epochs = int(selection["epochs"])
    train_curve = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = -(ty * torch.log_softmax(model(tx), dim=1)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_curve.append(float((-(ty * torch.log_softmax(model(tx), dim=1))
                                      .sum(dim=1).mean()).item()))
    torch.save({"state_dict": model.state_dict(), "mean": base["mean"], "scale": base["scale"],
                "features": "x_canonical", "selected_alpha": float(lock["selected_alpha"]),
                "optimizer_state": optimizer.state_dict(), "epoch": epochs,
                "architecture": "input->64-relu->64-relu->9"}, checkpoint_path)
    result = {
        "schema": "phase17-final-proposal-fit-v1", "status": "complete",
        "source_commit": lock["source_commit"],
        "final_fit_lock_sha256": sha256_file(root / "experiments/phase17b_final_proposal_fit_v1.lock.json"),
        "alpha_screen_lock_sha256": sha256_file(root / "experiments/phase17b_alpha_screen_v3.lock.json"),
        "alpha_screen_result_sha256": sha256_file(run_dir / "alpha_screen.json"),
        "training_target_sha256": sha256_file(run_dir / "teacher_queries.jsonl"),
        "training_states": int(len(x)), "training_layouts": int(len(set(data["layouts"].tolist()))),
        "epochs": epochs, "random_seed": int(selection["random_seed"]),
        "normalization_source": "frozen 112-fit-layout canonical_soft checkpoint",
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "training_nll_final": float(train_curve[-1]), "training_curve_nll": train_curve,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "reserve_access": False, "test_access": False, "fresh_outcomes": False,
    }
    if manifest_path.exists():
        raise Phase17Error("final proposal-fit manifest exists; refusing overwrite")
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def audit_fresh_layouts(root: Path = ROOT) -> dict[str, Any]:
    """Generate only proposed fresh layout identities and compare historical hashes."""
    from consequence_lab.phase16c_prior_guided import layout_hash_audit

    root = Path(root).resolve()
    lock = verify_fresh_layout_audit_lock(root)
    cfg = load_config()
    start = int(cfg["fresh_evaluation"]["proposed_seed_start"])
    count = int(cfg["fresh_evaluation"]["proposed_layout_count"])
    audit = layout_hash_audit(start, count, root)
    if (audit["unique_hashes"] != count or audit["overlap_historical"] != 0
            or audit["overlap_outcome_bearing"] != 0
            or audit["reserve_generated_or_accessed"] is not False
            or audit["d168_seed_block_accessed"] is not False
            or audit["outcomes_or_transitions_computed"] is not False):
        raise Phase17Error("Phase17C proposed fresh block failed hash-only freshness audit")
    audit["schema"] = "phase17-fresh-layout-audit-v1"
    audit["audit_lock_sha256"] = sha256_file(root / "experiments/phase17c_fresh_layout_audit_v1.lock.json")
    audit["source_commit"] = lock["source_commit"]
    audit["scope"] = {"hashes_only": True, "simulator_transitions": False,
                      "reserve_outcomes_read": False, "test_outcomes_read": False,
                      "fresh_outcomes": False}
    output = root / cfg["outputs"]["fresh_layout_audit"]
    if output.exists():
        raise Phase17Error("Phase17C layout-audit output exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "passed", "seed_start": start, "layout_count": count,
            "unique_hashes": audit["unique_hashes"],
            "historical_source_count": audit["historical_source_count"],
            "overlap_historical": audit["overlap_historical"],
            "overlap_outcome_bearing": audit["overlap_outcome_bearing"],
            "outcomes_or_transitions_computed": audit["outcomes_or_transitions_computed"],
            "audit_sha256": sha256_file(output), "audit_lock_sha256": audit["audit_lock_sha256"]}


def alpha_screen_advancement_gate(selected: dict[str, float],
                                  uniform: dict[str, float]) -> dict[str, Any]:
    """Descriptive internal gate: improve complete-success availability without more collisions."""
    success_improved = (selected["layout_macro_success_available"]
                        > uniform["layout_macro_success_available"])
    collision_not_higher = (selected["layout_macro_collision_candidate_fraction"]
                            <= uniform["layout_macro_collision_candidate_fraction"])
    return {
        "strictly_improves_success_availability_vs_uniform": bool(success_improved),
        "collision_candidate_fraction_no_higher_than_uniform": bool(collision_not_higher),
        "pass": bool(success_improved and collision_not_higher),
        "interpretation": "descriptive_internal_screen_only; fresh confirmation required",
    }


def _predict(model: Any, scenario: CorridorScenario, state: CorridorState,
             representation: str = "canonical") -> np.ndarray:
    import torch
    feature = _state_decision_feature(scenario, state, representation)
    x = ((feature - model["mean"]) / model["scale"]).astype(np.float32)
    with torch.no_grad():
        return torch.softmax(model["net"](torch.as_tensor(x[None, :])), dim=1).numpy()[0].astype(np.float64)


def _state_decision_feature(scenario: CorridorScenario, state: CorridorState,
                            representation: str = "canonical") -> np.ndarray:
    from consequence_lab.phase12_teacher import state_features as state_features_fn
    from consequence_lab.phase14c import build_feature_matrix
    from consequence_lab.phase15_d164 import canonical_agent_relative

    raw52 = state_features_fn(scenario, state)[None, :].astype(np.float64)
    r0 = build_feature_matrix(raw52, "raw52_plus_physics9")
    if representation == "r0":
        return r0[0].astype(np.float32)
    if representation != "canonical":
        raise ValueError(f"unknown decision-state representation: {representation}")
    return canonical_agent_relative(raw52, r0[:, 52:])[0].astype(np.float32)


def evaluate_episode(scenario: CorridorScenario, method: str, replica: int | None,
                     model: dict[str, Any] | None = None, alpha: float = 0.0,
                     support_states: np.ndarray | None = None) -> dict[str, Any]:
    if method not in {"uniform_b32", "guided_b32", "uniform_b64"}:
        raise ValueError(f"unknown Phase17 shooting method: {method}")
    state = scenario.initial_state
    diagnostics, latencies, action_counts = [], [], np.zeros(9, dtype=np.int64)
    exact_sequences = exact_transitions = ties = duplicates = fallbacks = success_available = 0
    diagnostic_overhead = 0.0
    while state.status == "active":
        novelty_distance = None
        if support_states is not None:
            diagnostic_tick = time.perf_counter()
            feature = _state_decision_feature(scenario, state)
            x = ((feature - model["mean"]) / model["scale"]).astype(np.float32)
            novelty_distance = float(np.sqrt(np.min(np.sum((support_states - x[None, :]) ** 2, axis=1))))
            diagnostic_overhead += time.perf_counter() - diagnostic_tick
        started = time.perf_counter()
        horizon = min(8, scenario.spec.max_steps - state.elapsed_steps)
        query = query_seed(scenario.seed, state.elapsed_steps, int(replica or 0))
        q = None
        if method == "guided_b32":
            if model is None:
                raise ValueError("guided method requires the frozen proposal model")
            q = _predict(model, scenario, state)
        budget = 64 if method == "uniform_b64" else 32
        if method == "guided_b32":
            bank, rejected, fallback = guided_sequence_bank(horizon, budget, query, q, alpha)
        else:
            # Preserve Phase16D/E's exact uniform-prefix generator, including B32 ⊂ B64.
            bank, rejected, fallback = uniform_prefix_bank(horizon, budget, query)
        evaluations = [evaluate_sequence(state, scenario, seq) for seq in bank]
        chosen, tie_count = choose_lexicographic_best(evaluations)
        used_transitions = sum(int(item.transitions) for item in evaluations)
        available = sum(int(item.success_reached) for item in evaluations)
        collision_candidates = sum(int(item.objective[0] == 3) for item in evaluations)
        first_success_rank = next((index + 1 for index, item in enumerate(evaluations)
                                   if item.success_reached), None)
        state = corridor_step(state, chosen.sequence[0], scenario.spec).state
        latencies.append(time.perf_counter() - started)
        action_counts[chosen.sequence[0]] += 1
        exact_sequences += len(bank)
        exact_transitions += used_transitions
        ties += int(tie_count > 1)
        duplicates += rejected
        fallbacks += fallback
        success_available += available
        sampled_first_actions = np.bincount([sequence[0] for sequence in bank], minlength=ACTION_COUNT)
        residual_first_actions = np.bincount(
            [sequence[0] for sequence in bank if len(set(sequence)) > 1], minlength=ACTION_COUNT)
        proposal_entropy = (float(-np.sum(q[q > 0] * np.log(q[q > 0]))) if q is not None else None)
        residual_support = int(np.count_nonzero(residual_first_actions))
        residual_total = int(residual_first_actions.sum())
        residual_probabilities = (residual_first_actions / residual_total if residual_total else
                                  np.zeros(ACTION_COUNT, dtype=np.float64))
        residual_entropy = (float(-np.sum(residual_probabilities[residual_probabilities > 0]
                                           * np.log(residual_probabilities[residual_probabilities > 0])))
                            if residual_total else 0.0)
        diagnostics.append({"decision_index": len(diagnostics), "query_seed": query,
                            "budget": len(bank), "chosen_action": chosen.sequence[0],
                            "best_objective": list(chosen.objective), "tie_count": int(tie_count),
                            "success_candidates": int(available),
                            "first_success_sequence_rank": first_success_rank,
                            "collision_candidates": int(collision_candidates),
                            "collision_candidate_fraction": collision_candidates / len(evaluations),
                            "exact_transitions": int(used_transitions),
                            "duplicate_rejections": int(rejected), "fallback": int(fallback),
                            "residual_first_action_support": residual_support,
                            "residual_first_action_entropy_nats": residual_entropy,
                            "candidate_action_diversity": residual_entropy / math.log(ACTION_COUNT),
                            "candidate_first_action_counts": sampled_first_actions.tolist(),
                            "residual_first_action_counts": residual_first_actions.tolist(),
                            "proposal_entropy": proposal_entropy,
                            "proposal": (q.tolist() if q is not None else None),
                            "canonical_support_distance": novelty_distance,
                            "resulting_status": state.status})
    return {"seed": int(scenario.seed), "layout_sha256": scenario.layout_sha256,
            "method": method, "replica": replica, "budget": (32 if method != "uniform_b64" else 64),
            "outcome": state.status, "steps": int(state.elapsed_steps - scenario.initial_state.elapsed_steps),
            "decision_count": len(diagnostics), "exact_sequences": int(exact_sequences),
            "exact_transitions": int(exact_transitions), "executed_transitions": len(diagnostics),
            "objective_tie_decisions": int(ties), "duplicate_rejections": int(duplicates),
            "fallback_sequences": int(fallbacks), "complete_success_sequences": int(success_available),
            "selected_action_counts": action_counts.tolist(),
            "latency_seconds_per_decision": latencies, "wall_seconds": float(sum(latencies)),
            "diagnostic_overhead_seconds": float(diagnostic_overhead),
            "decision_diagnostics": diagnostics}


def geometry_episode(scenario: CorridorScenario, config: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_corridor_episode(
        scenario, "geometry_waypoint",
        parameters={"resolution": float(config["geometry"]["resolution"]),
                    "waypoint_tolerance": float(config["geometry"]["waypoint_tolerance"])})
    return {"seed": result.seed, "layout_sha256": result.layout_sha256,
            "method": "geometry_waypoint", "replica": None, "budget": None,
            "outcome": result.outcome, "steps": result.steps, "decision_count": result.decision_count,
            "exact_sequences": 0, "exact_transitions": 0, "executed_transitions": result.steps,
            "objective_tie_decisions": 0, "duplicate_rejections": 0, "fallback_sequences": 0,
            "complete_success_sequences": 0, "selected_action_counts": [],
            "latency_seconds_per_decision": ([result.wall_seconds_per_decision]
                                                if result.wall_seconds_per_decision is not None else []),
            "wall_seconds": result.wall_seconds,
            "geometry_episode_average_decision_latency": result.wall_seconds_per_decision,
            "decision_diagnostics": []}


def fresh_execution_order(layouts: Sequence[dict[str, Any]], cfg: dict[str, Any] | None = None
                          ) -> list[dict[str, Any]]:
    """Counterbalance all paired Phase17C arms within each audited layout."""
    cfg = cfg or load_config()
    fresh = cfg["fresh_evaluation"]
    rows: list[dict[str, Any]] = []
    for layout in layouts:
        seed = int(layout["seed"])
        arms = []
        for method in fresh["methods"]:
            name = str(method["name"])
            replica_ids = (range(int(method["replicas"]))
                           if name != "geometry_waypoint" else [None])
            for replica in replica_ids:
                arms.append({"method": name, "replica": replica,
                             "budget": method["budget"]})
        random.Random(int(fresh["order_counterbalance_seed"]) + seed).shuffle(arms)
        for order_index, arm in enumerate(arms):
            rows.append({"seed": seed, "layout_sha256": layout["layout_sha256"],
                         "order_index": order_index, **arm})
    if len(rows) != int(fresh["total_episodes"]):
        raise Phase17Error("Phase17C schedule has unexpected episode count")
    keys = [(r["method"], r["replica"], r["budget"], r["seed"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise Phase17Error("Phase17C schedule contains duplicate episode keys")
    return rows


def _episode_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return row["method"], row.get("replica"), row.get("budget"), int(row["seed"])


def _rate_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty Phase17C arm")
    is_geometry = rows[0]["method"] == "geometry_waypoint"
    latencies = ([float(row["geometry_episode_average_decision_latency"]) for row in rows
                  if row.get("geometry_episode_average_decision_latency") is not None]
                 if is_geometry else
                 [float(x) for row in rows for x in row["latency_seconds_per_decision"]])
    decision_rows = [d for row in rows for d in row.get("decision_diagnostics", [])]
    successful_candidate_decisions = [d for d in decision_rows if d.get("success_candidates", 0) > 0]
    success_ranks = [d["first_success_sequence_rank"] for d in decision_rows
                     if d.get("first_success_sequence_rank") is not None]
    return {
        "episodes": len(rows),
        "safe_success_rate": float(np.mean([row["outcome"] == "success" for row in rows])),
        "collision_rate": float(np.mean([row["outcome"] == "collision" for row in rows])),
        "timeout_rate": float(np.mean([row["outcome"] == "timeout" for row in rows])),
        "mean_actions_to_terminal": float(np.mean([row["steps"] for row in rows])),
        "mean_exact_sequences_per_decision": float(np.mean([
            row["exact_sequences"] / max(1, row["decision_count"]) for row in rows])),
        "mean_exact_transitions_per_episode": float(np.mean([row["exact_transitions"] for row in rows])),
        "median_decision_latency_seconds": float(np.median(latencies)) if latencies else None,
        "p90_decision_latency_seconds": float(np.quantile(latencies, .90)) if latencies else None,
        "latency_aggregation": ("median_of_episode_average_decision_time" if is_geometry
                                else "pooled_per_decision_samples"),
        "mean_episode_wall_seconds": float(np.mean([row["wall_seconds"] for row in rows])),
        "candidate_success_sequences_per_episode": float(np.mean([
            row["complete_success_sequences"] for row in rows])),
        "candidate_success_availability_per_decision": (
            len(successful_candidate_decisions) / len(decision_rows) if decision_rows else None),
        "candidate_success_sequences_per_decision": (
            float(np.mean([d["success_candidates"] for d in decision_rows]))
            if decision_rows else None),
        "median_first_success_sequence_rank": (float(np.median(success_ranks))
                                               if success_ranks else None),
        "collision_terminal_candidate_fraction": (
            float(np.mean([d["collision_candidate_fraction"] for d in decision_rows]))
            if decision_rows else None),
        "duplicate_rejections": int(sum(row["duplicate_rejections"] for row in rows)),
        "fallback_sequences": int(sum(row["fallback_sequences"] for row in rows)),
    }


def _proposal_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate sampled proposal use without the mandatory constant-action bank."""
    decision_rows = [d for row in rows for d in row.get("decision_diagnostics", [])]
    if not decision_rows:
        return {"decisions": 0, "proposal_action_probability_mean": None,
                "candidate_first_action_counts": [0] * ACTION_COUNT,
                "residual_first_action_counts": [0] * ACTION_COUNT,
                "selected_action_counts": [0] * ACTION_COUNT,
                "mean_candidate_action_diversity": None,
                "mean_residual_first_action_entropy_nats": None,
                "mean_proposal_entropy_nats": None}
    proposal_rows = [np.asarray(d["proposal"], dtype=np.float64) for d in decision_rows
                     if d.get("proposal") is not None]
    candidate_counts = np.sum([np.asarray(d["candidate_first_action_counts"], dtype=np.int64)
                               for d in decision_rows], axis=0)
    residual_counts = np.sum([np.asarray(d["residual_first_action_counts"], dtype=np.int64)
                              for d in decision_rows], axis=0)
    selected_counts = np.sum([np.asarray(row["selected_action_counts"], dtype=np.int64)
                              for row in rows], axis=0)
    return {"decisions": len(decision_rows),
            "proposal_action_probability_mean": (np.mean(proposal_rows, axis=0).tolist()
                                                 if proposal_rows else None),
            "candidate_first_action_counts": candidate_counts.tolist(),
            "residual_first_action_counts": residual_counts.tolist(),
            "selected_action_counts": selected_counts.tolist(),
            "mean_candidate_action_diversity": float(np.mean(
                [d["candidate_action_diversity"] for d in decision_rows])),
            "mean_residual_first_action_entropy_nats": float(np.mean(
                [d["residual_first_action_entropy_nats"] for d in decision_rows])),
            "mean_proposal_entropy_nats": (float(np.mean([d["proposal_entropy"]
                                                          for d in decision_rows
                                                          if d.get("proposal_entropy") is not None]))
                                           if proposal_rows else None)}


def _validated_resume_records(lines: Sequence[str], order: Sequence[dict[str, Any]],
                              execution_lock_sha256: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Accept only a valid append-only prefix of the locked episode schedule."""
    if len(lines) > len(order):
        raise Phase17Error("Phase17C resume file is longer than the frozen schedule")
    completed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row_index, (line, expected_arm) in enumerate(
            zip(lines, order[:len(lines)], strict=True)):
        row = json.loads(line)
        key = _episode_key(row)
        if (key in completed or key != _episode_key(expected_arm)
                or row.get("layout_sha256") != expected_arm["layout_sha256"]
                or row.get("schedule_order_index") != row_index
                or row.get("execution_lock_sha256") != execution_lock_sha256):
            raise Phase17Error("Phase17C resume rows do not match the locked schedule prefix")
        if row.get("outcome") not in {"success", "collision", "timeout"}:
            raise Phase17Error("Phase17C resumed episode has an invalid terminal outcome")
        completed[key] = row
    return completed


def _paired_layout_bootstrap(reference: Sequence[float], candidate: Sequence[float],
                             seed: int, draws: int, confidence: float = .95) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 1 or cand.shape != ref.shape or not len(ref):
        raise ValueError("paired Phase17C bootstrap requires aligned layout vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ref), size=(draws, len(ref)))
    diff = cand[indices].mean(axis=1) - ref[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    return {"difference_candidate_minus_reference": float(np.mean(cand - ref)),
            "ci_lower": float(np.quantile(diff, tail)),
            "ci_upper": float(np.quantile(diff, 1 - tail)),
            "confidence": float(confidence), "layouts": int(len(ref)),
            "bootstrap_draws": int(draws), "bootstrap_seed": int(seed)}


def _one_sided_lower_bootstrap(reference: Sequence[float], candidate: Sequence[float],
                               seed: int, draws: int) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 1 or cand.shape != ref.shape or not len(ref):
        raise ValueError("paired Phase17C bootstrap requires aligned layout vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ref), size=(draws, len(ref)))
    diff = cand[indices].mean(axis=1) - ref[indices].mean(axis=1)
    return {"difference_candidate_minus_reference": float(np.mean(cand - ref)),
            "one_sided_97_5_lower": float(np.quantile(diff, .025)),
            "layouts": int(len(ref)), "bootstrap_draws": int(draws),
            "bootstrap_seed": int(seed)}


def _layout_vectors(keyed: dict[tuple[Any, ...], dict[str, Any]], layouts: Sequence[int],
                    method: str) -> dict[str, list[float]]:
    reps = [0, 1, 2]
    if method == "geometry_waypoint":
        rows = [keyed[(method, None, None, seed)] for seed in layouts]
        return {"success": [float(row["outcome"] == "success") for row in rows],
                "collision": [float(row["outcome"] == "collision") for row in rows],
                "transitions": [float(row["exact_transitions"]) for row in rows]}
    return {
        "success": [float(np.mean([keyed[(method, rep, 32 if method != "uniform_b64" else 64,
                                          seed)]["outcome"] == "success" for rep in reps]))
                    for seed in layouts],
        "collision": [float(np.mean([keyed[(method, rep, 32 if method != "uniform_b64" else 64,
                                            seed)]["outcome"] == "collision" for rep in reps]))
                      for seed in layouts],
        "transitions": [float(np.mean([keyed[(method, rep, 32 if method != "uniform_b64" else 64,
                                              seed)]["exact_transitions"] for rep in reps]))
                        for seed in layouts],
    }


def _layout_event_count(keyed: dict[tuple[Any, ...], dict[str, Any]], layouts: Sequence[int],
                        candidate: str, reference: str, event: str) -> int:
    budget = {"uniform_b32": 32, "guided_b32": 32, "uniform_b64": 64}
    count = 0
    for seed in layouts:
        for replica in range(3):
            cand = keyed[(candidate, replica, budget[candidate], seed)]["outcome"]
            ref = keyed[(reference, replica, budget[reference], seed)]["outcome"]
            if event == "success_regression" and cand != "success" and ref == "success":
                count += 1
                break
            if event == "added_collision" and cand == "collision" and ref != "collision":
                count += 1
                break
    return count


def summarize_fresh_evaluation(records: Sequence[dict[str, Any]],
                               order: Sequence[dict[str, Any]],
                               cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Summarize all outcomes at layout level and apply the frozen Phase17C gates."""
    from consequence_lab.phase16d_compute_frontier import binomial_upper

    cfg = cfg or load_config()
    expected = {_episode_key(row) for row in order}
    keyed = {_episode_key(row): row for row in records}
    if len(keyed) != len(records) or set(keyed) != expected:
        raise Phase17Error("Phase17C results do not exactly cover the frozen schedule")
    layout_hashes = {int(row["seed"]): row["layout_sha256"] for row in order}
    if len(layout_hashes) != int(cfg["fresh_evaluation"]["proposed_layout_count"]):
        raise Phase17Error("Phase17C layout count differs from protocol")
    for key, row in keyed.items():
        if row["layout_sha256"] != layout_hashes[key[-1]]:
            raise Phase17Error("Phase17C episode layout hash differs from schedule")
    layouts = sorted(layout_hashes)
    methods: dict[str, Any] = {}
    per_replica: dict[str, Any] = {}
    for method in ("uniform_b32", "guided_b32", "uniform_b64", "geometry_waypoint"):
        if method == "geometry_waypoint":
            rows = [keyed[(method, None, None, seed)] for seed in layouts]
            methods[method] = _rate_summary(rows)
            continue
        rows = [keyed[(method, rep, 32 if method != "uniform_b64" else 64, seed)]
                for seed in layouts for rep in range(3)]
        methods[method] = _rate_summary(rows)
        per_replica[method] = {
            str(rep): _rate_summary([keyed[(method, rep, 32 if method != "uniform_b64" else 64, seed)]
                                     for seed in layouts]) for rep in range(3)}

    vectors = {method: _layout_vectors(keyed, layouts, method)
               for method in ("uniform_b32", "guided_b32", "uniform_b64", "geometry_waypoint")}
    analysis = cfg["analysis"]
    draws, seed = int(analysis["bootstrap_draws"]), int(analysis["bootstrap_seed"])
    ref_success = vectors["uniform_b64"]["success"]
    geometry_gain = _paired_layout_bootstrap(vectors["geometry_waypoint"]["success"], ref_success,
                                             seed, draws)
    ref_gate = cfg["gates"]["b64_reference"]
    b64_pass = (
        methods["uniform_b64"]["safe_success_rate"] > ref_gate["mean_success_strictly_above"]
        and methods["uniform_b64"]["collision_rate"] <= ref_gate["mean_collision_at_most"]
        and all(per_replica["uniform_b64"][str(r)]["safe_success_rate"]
                >= ref_gate["every_replica_success_at_least"] for r in range(3))
        and all(per_replica["uniform_b64"][str(r)]["collision_rate"]
                <= ref_gate["every_replica_collision_at_most"] for r in range(3))
        and geometry_gain["difference_candidate_minus_reference"]
                >= ref_gate["geometry_success_gain_at_least"]
        and geometry_gain["ci_lower"] > ref_gate["paired_geometry_gain_ci_lower_above"])

    def candidate_gate(method: str) -> dict[str, Any]:
        candidate = vectors[method]
        success_diff = _one_sided_lower_bootstrap(ref_success, candidate["success"], seed, draws)
        regression_count = _layout_event_count(keyed, layouts, method, "uniform_b64", "success_regression")
        collision_count = _layout_event_count(keyed, layouts, method, "uniform_b64", "added_collision")
        regression_upper = binomial_upper(regression_count, len(layouts), .9875)
        collision_upper = binomial_upper(collision_count, len(layouts), .9875)
        ratios = [c / max(1.0, r) for c, r in zip(candidate["transitions"],
                                                   vectors["uniform_b64"]["transitions"], strict=True)]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(ratios), size=(draws, len(ratios)))
        ratio_samples = np.asarray(ratios)[idx].mean(axis=1)
        transition_summary = {"mean_ratio": float(np.mean(ratios)),
                              "ci95_lower": float(np.quantile(ratio_samples, .025)),
                              "ci95_upper": float(np.quantile(ratio_samples, .975))}
        candidate_latency = [lat for layout in layouts for rep in range(3)
                             for lat in keyed[(method, rep, 32, layout)]["latency_seconds_per_decision"]]
        reference_latency = [lat for layout in layouts for rep in range(3)
                            for lat in keyed[("uniform_b64", rep, 64, layout)]["latency_seconds_per_decision"]]
        latency_ratio = float(np.median(candidate_latency) / np.median(reference_latency))
        gate = cfg["gates"]["guided_b32"]
        components = {
            "mean_success": methods[method]["safe_success_rate"] > gate["mean_success_strictly_above"],
            "replica_success_floors": all(per_replica[method][str(r)]["safe_success_rate"]
                >= gate["every_replica_success_at_least"] for r in range(3)),
            "mean_collision_ceiling": methods[method]["collision_rate"] <= gate["mean_collision_at_most"],
            "replica_collision_ceilings": all(per_replica[method][str(r)]["collision_rate"]
                <= gate["every_replica_collision_at_most"] for r in range(3)),
            "paired_success_noninferiority": success_diff["one_sided_97_5_lower"]
                >= gate["paired_success_difference_lower_vs_b64_at_least"],
            "success_regression_rare_event_bound": regression_upper
                <= gate["success_regression_upper_vs_b64_at_most"],
            "added_collision_rare_event_bound": collision_upper
                <= gate["added_collision_upper_vs_b64_at_most"],
            "mean_transition_ratio": transition_summary["mean_ratio"]
                <= gate["mean_transition_ratio_vs_b64_at_most"],
            "transition_ratio_upper_bound": transition_summary["ci95_upper"]
                <= gate["transition_ratio_ci_upper_vs_b64_at_most"],
            "median_latency_ratio": latency_ratio <= gate["median_latency_ratio_vs_b64_at_most"],
        }
        return {"success_difference_vs_b64": success_diff,
                "success_regression_layout_events": regression_count,
                "success_regression_upper98_75": regression_upper,
                "added_collision_layout_events": collision_count,
                "added_collision_upper98_75": collision_upper,
                "transition_ratio_vs_b64": transition_summary,
                "median_decision_latency_ratio_vs_b64": latency_ratio,
                "gate_components": components, "gate_pass": bool(b64_pass and all(components.values()))}

    uniform_gate = candidate_gate("uniform_b32")
    guided_gate = candidate_gate("guided_b32")
    rescue = bool(not uniform_gate["gate_components"]["success_regression_rare_event_bound"]
                  or not uniform_gate["gate_components"]["added_collision_rare_event_bound"])
    guided_rescue = bool(rescue and guided_gate["gate_pass"])
    guided_rows = [row for row in records if row["method"] == "guided_b32"]
    decision_rows = [d for row in guided_rows for d in row["decision_diagnostics"]
                     if d.get("canonical_support_distance") is not None]
    novelty = {}
    if decision_rows:
        dist = np.asarray([x["canonical_support_distance"] for x in decision_rows], dtype=np.float64)
        bins = np.quantile(dist, [0, .25, .5, .75, 1])
        for i in range(4):
            selected = [x for x in decision_rows if bins[i] <= x["canonical_support_distance"]
                        <= bins[i + 1] and (i == 3 or x["canonical_support_distance"] < bins[i + 1])]
            novelty[str(i + 1)] = {"decisions": len(selected),
                "collision_after_decision_fraction": float(np.mean([x["resulting_status"] == "collision"
                                                                       for x in selected])) if selected else None,
                "success_after_decision_fraction": float(np.mean([x["resulting_status"] == "success"
                                                                    for x in selected])) if selected else None,
                "distance_low": float(bins[i]), "distance_high": float(bins[i + 1])}
    proposal_diagnostics = _proposal_diagnostics(guided_rows)
    return {"schema": "phase17c-fresh-evaluation-result-v1", "status": "complete",
            "layout_count": len(layouts), "unique_episode_count": len(records),
            "methods": methods, "per_replica": per_replica,
            "paired": {"b64_minus_geometry_success": geometry_gain,
                       "guided_b32_vs_b64": guided_gate,
                       "uniform_b32_vs_b64": uniform_gate},
            "b64_reference_gate_pass": bool(b64_pass),
            "guided_b32_gate_pass": bool(guided_gate["gate_pass"]),
            "uniform_b32_rare_event_gate_pass": bool(
                uniform_gate["gate_components"]["success_regression_rare_event_bound"]
                and uniform_gate["gate_components"]["added_collision_rare_event_bound"]),
            "guided_b32_rescue_claim_pass": guided_rescue,
            "guided_proposal_diagnostics": proposal_diagnostics,
            "guided_decision_novelty_quartiles": novelty,
            "layout_level_sampling_unit": "layout_sha256",
            "scope": {"reserve_access": False, "test_access": False, "d168_access": False,
                      "training": False, "dagger": False, "student_control": False,
                      "learned_world_model": False, "learned_action_execution": False}}


def _fresh_execution_source_paths() -> tuple[str, ...]:
    return (
        "AGENTS.md", "pyproject.toml", "uv.lock",
        "configs/phase17_learned_proposal_guidance_v1.yaml",
        "reports/phase17_learned_proposal_guidance_protocol.md",
        "reports/phase17c_fresh_layout_audit_protocol.md",
        "reports/phase17c_fresh_evaluation_protocol.md",
        "src/consequence_lab/phase17_proposal_guidance.py",
        "src/consequence_lab/phase16e_budget_reliability.py",
        "src/consequence_lab/phase16d_compute_frontier.py",
        "src/consequence_lab/phase16c_prior_guided.py",
        "src/consequence_lab/phase15_d164.py",
        "src/consequence_lab/phase14g.py",
        "src/consequence_lab/phase12_teacher.py",
        "src/consequence_lab/corridor_baselines.py",
        "src/consequence_lab/corridor_evaluation.py",
        "src/consequence_lab/corridor_tasks.py",
        "src/consequence_lab/inertial_corridor.py",
        "tests/test_phase17_proposal_guidance.py",
        "tests/test_phase17c_fresh_evaluation.py",
        "scripts/phase17_fresh_evaluation.py",
        "scripts/freeze_phase17c_execution_lock.py",
        "reports/phase17c_execution_abort_v1.md",
        "reports/phase17c_execution_amendment_v2.md",
        "experiments/phase17c_guided_planner_execution_v1.lock.json",
        "experiments/phase16e_budget_reliability_confirmation_v1.lock.json",
        "artifacts/runs/phase16e_budget_reliability_confirmation_v1/result.json",
        "reports/phase16e_budget_reliability_confirmation_result.md",
    )


def _git_blob_sha256(root: Path, commit: str, relative: str) -> str:
    blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
    return hashlib.sha256(blob).hexdigest().upper()


def _safe_load_checkpoint(path: Path) -> dict[str, Any]:
    """Load NumPy metadata in a project checkpoint using a narrow safe allowlist."""
    import torch

    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:  # NumPy < 2.0
        numpy_core = np.core
    allowed_numpy_globals = [numpy_core.multiarray._reconstruct, np.ndarray, np.dtype,
                             type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed_numpy_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def _load_final_proposal(root: Path) -> dict[str, Any]:
    cfg = load_config()
    ckpt_path = root / cfg["outputs"]["proposal_checkpoint"]
    checkpoint = _safe_load_checkpoint(ckpt_path)
    net = _build_proposal_model(63, cfg["proposal_model"]["hidden"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    if (checkpoint.get("features") != "x_canonical"
            or checkpoint.get("selected_alpha")
            != float(cfg["fresh_evaluation"]["selected_alpha"])):
        raise Phase17Error("final proposal checkpoint feature/alpha metadata mismatch")
    return {"net": net, "mean": np.asarray(checkpoint["mean"], dtype=np.float32),
            "scale": np.asarray(checkpoint["scale"], dtype=np.float32)}


def freeze_fresh_execution_lock(root: Path = ROOT) -> dict[str, Any]:
    """Freeze the final fresh-layout comparison after all code/tests are committed."""
    root = Path(root).resolve()
    if FRESH_EXECUTION_LOCK.exists():
        raise Phase17Error("refusing to overwrite immutable Phase17C execution lock")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if status:
        raise Phase17Error("commit Phase17C protocol, runner, tests, and vault before lock")
    cfg = load_config()
    audit_path = root / cfg["outputs"]["fresh_layout_audit"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    count = int(cfg["fresh_evaluation"]["proposed_layout_count"])
    seed_start = int(cfg["fresh_evaluation"]["proposed_seed_start"])
    if (audit.get("schema") != "phase17-fresh-layout-audit-v1"
            or audit.get("unique_hashes") != count
            or [int(row["seed"]) for row in audit.get("layouts", [])]
               != list(range(seed_start, seed_start + count))
            or len({row["layout_sha256"] for row in audit.get("layouts", [])}) != count
            or audit.get("overlap_historical") != 0
            or audit.get("overlap_outcome_bearing") != 0
            or audit.get("outcomes_or_transitions_computed") is not False
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("d168_seed_block_accessed") is not False):
        raise Phase17Error("fresh-layout identity audit does not satisfy execution prerequisite")
    audit_lock_path = root / "experiments/phase17c_fresh_layout_audit_v1.lock.json"
    audit_lock = json.loads(audit_lock_path.read_text(encoding="utf-8"))
    if (audit.get("audit_lock_sha256") != sha256_file(audit_lock_path)
            or audit.get("source_commit") != audit_lock.get("source_commit")):
        raise Phase17Error("audit result does not bind to the committed hash-only audit lock")
    for source, digest in audit.get("historical_source_hashes", {}).items():
        if sha256_file(root / source) != digest:
            raise Phase17Error(f"historical identity source changed after hash audit: {source}")
    for source, digest in audit_lock.get("source_files_sha256", {}).items():
        if _git_blob_sha256(root, audit_lock["source_commit"], source) != digest:
            raise Phase17Error(f"hash audit source changed after audit: {source}")
    checkpoint_path = root / cfg["outputs"]["proposal_checkpoint"]
    fit_manifest_path = root / cfg["outputs"]["final_fit_manifest"]
    phase16e_lock_path = root / "experiments/phase16e_budget_reliability_confirmation_v1.lock.json"
    phase16e_result_path = root / "artifacts/runs/phase16e_budget_reliability_confirmation_v1/result.json"
    phase16e_lock = json.loads(phase16e_lock_path.read_text(encoding="utf-8"))
    phase16e_result = json.loads(phase16e_result_path.read_text(encoding="utf-8"))
    phase16e_scope = phase16e_result.get("scope", {})
    if (phase16e_result.get("status") != "complete"
            or phase16e_result.get("selected_budget_under_locked_rule") != 64
            or phase16e_result.get("lock_sha256") != sha256_file(phase16e_lock_path)
            or phase16e_lock.get("status") != "locked_approved_development_run"
            or phase16e_scope.get("reserve_access") is not False
            or phase16e_scope.get("test_access") is not False
            or phase16e_scope.get("learned_prior") is not False
            or phase16e_scope.get("learned_world_model") is not False):
        raise Phase17Error("Phase16E B64 reference provenance/selection check failed")
    manifest = json.loads(fit_manifest_path.read_text(encoding="utf-8"))
    alpha_result_path = root / cfg["outputs"]["run_dir"] / "alpha_screen.json"
    alpha_result = json.loads(alpha_result_path.read_text(encoding="utf-8"))
    final_fit_lock_path = root / "experiments/phase17b_final_proposal_fit_v1.lock.json"
    if (manifest.get("final_fit_lock_sha256") != sha256_file(final_fit_lock_path)
            or manifest.get("checkpoint_sha256") != sha256_file(checkpoint_path)
            or manifest.get("training_target_sha256") != sha256_file(
                root / cfg["outputs"]["teacher_queries"])
            or manifest.get("alpha_screen_result_sha256") != sha256_file(alpha_result_path)
            or alpha_result.get("selected_alpha") != cfg["fresh_evaluation"]["selected_alpha"]
            or alpha_result.get("advancement_gate", {}).get("pass") is not True):
        raise Phase17Error("final fit / alpha-selection parent chain failed integrity checks")
    if (sha256_file(checkpoint_path) != manifest.get("checkpoint_sha256")
            or manifest.get("epochs") != 69
            or manifest.get("training_layouts") != 140
            or manifest.get("reserve_access") is not False
            or manifest.get("fresh_outcomes") is not False):
        raise Phase17Error("final proposal checkpoint/manifest integrity check failed")
    sources = _fresh_execution_source_paths()
    missing = [relative for relative in sources if not (root / relative).is_file()]
    if missing:
        raise Phase17Error(f"missing Phase17C lock sources: {missing}")
    output_dir = root / cfg["outputs"]["fresh_run_dir"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Phase17Error("fresh Phase17C output directory is not empty")
    order_path = root / cfg["outputs"]["fresh_execution_order"]
    if order_path.exists():
        raise Phase17Error("refusing to overwrite Phase17C execution order")
    order = fresh_execution_order(audit["layouts"], cfg)
    v1_lock = json.loads(FRESH_EXECUTION_LOCK_V1.read_text(encoding="utf-8"))
    v1_episodes = root / "artifacts/runs/phase17c_exact_guided_planner_v1/episodes.jsonl"
    v1_result = root / "artifacts/runs/phase17c_exact_guided_planner_v1/result.json"
    v1_manifest = root / "artifacts/runs/phase17c_exact_guided_planner_v1/manifest.json"
    if (not FRESH_V1_ORDER.is_file()
            or sha256_file(FRESH_V1_ORDER) != v1_lock.get("execution_order_sha256")
            or json.loads(FRESH_V1_ORDER.read_text(encoding="utf-8")) != order
            or any(path.exists() for path in (v1_episodes, v1_result, v1_manifest))):
        raise Phase17Error("v2 requires the identical v1 schedule and a preserved pre-outcome v1 abort")
    output_dir.mkdir(parents=True, exist_ok=True)
    order_path.write_text(json.dumps(order, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lock = {
        "schema": "phase17c-fresh-execution-lock-v2",
        "status": "locked_authorized_fresh_development_evaluation",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                   text=True).strip(),
        "source_sha256": {relative: sha256_file(root / relative) for relative in sources},
        "config_sha256": sha256_file(CONFIG),
        "protocol_sha256": sha256_file(root / "reports/phase17c_fresh_evaluation_protocol.md"),
        "audit_lock_sha256": sha256_file(audit_lock_path),
        "audit_result_sha256": sha256_file(audit_path),
        "execution_order_sha256": sha256_file(order_path),
        "final_checkpoint_sha256": sha256_file(checkpoint_path),
        "final_fit_manifest_sha256": sha256_file(fit_manifest_path),
        "training_support_path": cfg["training_data"]["d153_parent_archive"],
        "training_support_sha256": sha256_file(root / cfg["training_data"]["d153_parent_archive"]),
        "supersedes_failed_execution_lock_sha256": sha256_file(FRESH_EXECUTION_LOCK_V1),
        "superseded_schedule_sha256": sha256_file(FRESH_V1_ORDER),
        "alpha": float(cfg["fresh_evaluation"]["selected_alpha"]),
        "layout_count": count,
        "episode_count": len(order),
        "layout_sha256": [row["layout_sha256"] for row in audit["layouts"]],
        "scope": {"reserve_access": False, "test_access": False, "d168_access": False,
                  "training": False, "dagger": False, "student_control": False,
                  "learned_world_model": False, "learned_action_execution": False},
    }
    FRESH_EXECUTION_LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8")
    return {"lock_path": str(FRESH_EXECUTION_LOCK.relative_to(root)),
            "lock_sha256": sha256_file(FRESH_EXECUTION_LOCK),
            "source_commit": lock["source_commit"], "layout_count": count,
            "episode_count": len(order), "execution_order_sha256": lock["execution_order_sha256"]}


def _verify_fresh_execution_lock(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not FRESH_EXECUTION_LOCK.is_file():
        raise Phase17Error("immutable Phase17C fresh-evaluation lock is missing")
    lock = json.loads(FRESH_EXECUTION_LOCK.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase17c-fresh-execution-lock-v2":
        raise Phase17Error("unexpected Phase17C execution-lock version")
    cfg = load_config()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if status:
        raise Phase17Error("Phase17C execution requires a clean committed worktree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.run(["git", "merge-base", "--is-ancestor", lock["source_commit"], head],
                      cwd=root).returncode != 0:
        raise Phase17Error("Phase17C source commit is not an ancestor of execution commit")
    if lock.get("config_sha256") != sha256_file(CONFIG):
        raise Phase17Error("Phase17C config differs from execution lock")
    protocol_path = root / "reports/phase17c_fresh_evaluation_protocol.md"
    if lock.get("protocol_sha256") != sha256_file(protocol_path):
        raise Phase17Error("Phase17C protocol differs from execution lock")
    for relative, digest in lock.get("source_sha256", {}).items():
        if sha256_file(root / relative) != digest:
            raise Phase17Error(f"Phase17C locked source changed: {relative}")
    audit_path = root / cfg["outputs"]["fresh_layout_audit"]
    order_path = root / cfg["outputs"]["fresh_execution_order"]
    checkpoint_path = root / cfg["outputs"]["proposal_checkpoint"]
    fit_manifest_path = root / cfg["outputs"]["final_fit_manifest"]
    if (sha256_file(audit_path) != lock.get("audit_result_sha256")
            or sha256_file(order_path) != lock.get("execution_order_sha256")
            or sha256_file(checkpoint_path) != lock.get("final_checkpoint_sha256")
            or sha256_file(fit_manifest_path) != lock.get("final_fit_manifest_sha256")):
        raise Phase17Error("Phase17C audited layout, schedule or proposal parent hash mismatch")
    v1_episodes = root / "artifacts/runs/phase17c_exact_guided_planner_v1/episodes.jsonl"
    v1_result = root / "artifacts/runs/phase17c_exact_guided_planner_v1/result.json"
    v1_manifest = root / "artifacts/runs/phase17c_exact_guided_planner_v1/manifest.json"
    if (sha256_file(FRESH_EXECUTION_LOCK_V1)
            != lock.get("supersedes_failed_execution_lock_sha256")
            or sha256_file(FRESH_V1_ORDER) != lock.get("superseded_schedule_sha256")
            or json.loads(FRESH_V1_ORDER.read_text(encoding="utf-8"))
            != json.loads(order_path.read_text(encoding="utf-8"))
            or any(path.exists() for path in (v1_episodes, v1_result, v1_manifest))):
        raise Phase17Error("Phase17C v2 does not preserve the pre-outcome v1 attempt and schedule")
    if sha256_file(root / lock["training_support_path"]) != lock.get("training_support_sha256"):
        raise Phase17Error("Phase17C novelty-support dataset hash mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    order = json.loads(order_path.read_text(encoding="utf-8"))
    if order != fresh_execution_order(audit["layouts"], cfg):
        raise Phase17Error("Phase17C execution order does not match its audited layouts")
    expected_layout_hashes = [row["layout_sha256"] for row in audit["layouts"]]
    if (len(order) != int(cfg["execution"]["expected_unique_episode_count"])
            or lock.get("layout_sha256") != expected_layout_hashes
            or lock.get("episode_count") != len(order)
            or lock.get("alpha") != float(cfg["fresh_evaluation"]["selected_alpha"])
            or lock.get("scope", {}).get("reserve_access") is not False
            or lock.get("scope", {}).get("test_access") is not False):
        raise Phase17Error("Phase17C lock scope or schedule cardinality is invalid")
    return lock, order


def run_fresh_evaluation(root: Path = ROOT) -> dict[str, Any]:
    """Run or safely resume the immutable 5,000-episode Phase17C evaluation."""
    import platform
    import sys

    root = Path(root).resolve()
    lock, order = _verify_fresh_execution_lock(root)
    cfg = load_config()
    output = root / cfg["outputs"]["fresh_run_dir"]
    episode_path = root / cfg["outputs"]["fresh_episodes"]
    result_path = root / cfg["outputs"]["fresh_result"]
    manifest_path = root / cfg["outputs"]["fresh_manifest"]
    if result_path.exists() or manifest_path.exists():
        raise Phase17Error("Phase17C final outputs already exist; refusing overwrite")
    allowed = {root / cfg["outputs"]["fresh_execution_order"], episode_path}
    if output.exists():
        unexpected = {p.resolve() for p in output.rglob("*") if p.is_file()} - {
            p.resolve() for p in allowed if p.exists()}
        if unexpected:
            raise Phase17Error("unexpected Phase17C output exists; refusing unsafe resume")
    else:
        output.mkdir(parents=True)

    existing_lines = (episode_path.read_text(encoding="utf-8").splitlines()
                      if episode_path.exists() else [])
    completed = _validated_resume_records(existing_lines, order,
                                          sha256_file(FRESH_EXECUTION_LOCK))
    resumed_episode_count = len(completed)

    audit = json.loads((root / cfg["outputs"]["fresh_layout_audit"]).read_text(encoding="utf-8"))
    expected_hashes = {int(row["seed"]): row["layout_sha256"] for row in audit["layouts"]}
    model = _load_final_proposal(root)
    train = load_training_states(root)
    support = ((train["x_canonical"].astype(np.float32) - model["mean"]) / model["scale"]).astype(np.float32)
    alpha = float(lock["alpha"])
    started = time.perf_counter()
    for arm in order:
        key = _episode_key(arm)
        if key in completed:
            continue
        scenario = generate_staggered_corridor(int(arm["seed"]))
        if scenario.layout_sha256 != expected_hashes[int(arm["seed"])]:
            raise Phase17Error("runtime-generated layout differs from hash-only audit")
        if arm["method"] == "geometry_waypoint":
            row = geometry_episode(scenario, cfg)
        else:
            row = evaluate_episode(scenario, arm["method"], arm["replica"], model,
                                   alpha if arm["method"] == "guided_b32" else 0.0,
                                   support_states=support if arm["method"] == "guided_b32" else None)
        if _episode_key(row) != key:
            raise Phase17Error("Phase17C evaluated episode does not match frozen schedule key")
        row["schedule_order_index"] = int(arm["order_index"])
        row["execution_lock_sha256"] = sha256_file(FRESH_EXECUTION_LOCK)
        _jsonl_append(episode_path, row)
        completed[key] = row

    if len(completed) != int(cfg["execution"]["expected_unique_episode_count"]):
        raise Phase17Error("Phase17C run did not complete all scheduled episodes")
    result = summarize_fresh_evaluation(list(completed.values()), order, cfg)
    import torch

    result.update({"execution_lock_sha256": sha256_file(FRESH_EXECUTION_LOCK),
                   "config_sha256": sha256_file(CONFIG),
                   "protocol_sha256": sha256_file(root / "reports/phase17c_fresh_evaluation_protocol.md"),
                   "layout_audit_sha256": sha256_file(root / cfg["outputs"]["fresh_layout_audit"]),
                   "execution_order_sha256": sha256_file(root / cfg["outputs"]["fresh_execution_order"]),
                   "checkpoint_sha256": sha256_file(root / cfg["outputs"]["proposal_checkpoint"]),
                   "source_commit": lock["source_commit"],
                   "run_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                          text=True).strip(),
                   "software": {"python": sys.version, "platform": platform.platform(),
                                "numpy": np.__version__, "torch": torch.__version__},
                   "hardware": {"cpu_count": __import__("os").cpu_count(),
                                "torch_intraop_threads": torch.get_num_threads()},
                   "resumed_episode_count": resumed_episode_count,
                   "measured_episode_wall_seconds": float(sum(
                       row["wall_seconds"] for row in completed.values())),
                   "final_invocation_wall_seconds": time.perf_counter() - started})
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema": "phase17c-fresh-evaluation-manifest-v1", "status": "complete",
                "execution_lock_sha256": result["execution_lock_sha256"],
                "config_sha256": result["config_sha256"], "protocol_sha256": result["protocol_sha256"],
                "episode_count": len(completed), "episode_sha256": sha256_file(episode_path),
                "result_sha256": sha256_file(result_path),
                "layout_audit_sha256": result["layout_audit_sha256"],
                "execution_order_sha256": result["execution_order_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "source_commit": result["source_commit"], "run_commit": result["run_commit"],
                "scope": result["scope"]}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "manifest_sha256": sha256_file(manifest_path)}
