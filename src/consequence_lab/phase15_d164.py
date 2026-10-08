"""D164 representation, action geometry, and symmetry audit on saved data."""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import permutations, product
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import yaml

from consequence_lab.corridor_tasks import CorridorScenario
from consequence_lab.inertial_corridor import (
    ACTION_VECTORS, AxisAlignedObstacle, CorridorState, CorridorTaskSpec,
)
from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14c import build_feature_matrix
from consequence_lab.phase15_d157 import ACTION_COUNT, _strip_metrics
from consequence_lab.phase15_decision_aligned import regrets_from_exact_summaries
from consequence_lab.phase15_d158 import _candidate_metrics

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d164_representation_symmetry_audit.yaml"
SOURCES = (
    CONFIG, "reports/phase15_d164_protocol.md", "src/consequence_lab/phase15_d164.py",
    "scripts/freeze_phase15_d164_lock.py", "scripts/run_phase15_d164.py", "tests/test_phase15_d164.py",
    "src/consequence_lab/inertial_corridor.py", "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/phase12_teacher.py", "src/consequence_lab/phase14c.py",
    "src/consequence_lab/phase15_decision_aligned.py", "src/consequence_lab/phase15_d157.py",
    "src/consequence_lab/phase15_d158.py", "experiments/phase15_d153_coverage_scaling_v3.lock.json",
    "configs/phase15_d155_stage_c.yaml",
    "experiments/phase15_d162_fresh_layout_relational.lock.json",
    "reports/phase15_d162_result.md", "second-brain/00_Project_Home.md",
    "second-brain/14_Decisions.md", "second-brain/16_Experiment_Log.md",
    "second-brain/phases/Phase_15_Decision_Aligned_Target_Study.md",
)
D4_MATRICES = tuple(
    np.asarray(((sx if p[0] == 0 else 0, sx if p[0] == 1 else 0),
                (sy if p[1] == 0 else 0, sy if p[1] == 1 else 0)), dtype=np.float64)
    for p in permutations((0, 1)) for sx, sy in product((-1, 1), repeat=2)
)


class D164Error(RuntimeError):
    """D164 provenance or data-contract failure."""


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _read(path: str | Path) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def normalized_action_vectors() -> np.ndarray:
    vectors = np.asarray(ACTION_VECTORS, dtype=np.float64)
    diagonal = np.all(vectors != 0.0, axis=1)
    vectors[diagonal] /= np.sqrt(2.0)
    return vectors


def transform_point(point: tuple[float, float], matrix: np.ndarray,
                    width: float = 1.0, height: float = 1.0) -> tuple[float, float]:
    if not np.isclose(width, height):
        raise ValueError("D164 D4 transforms require the frozen square workspace")
    center = np.asarray((width / 2.0, height / 2.0), dtype=np.float64)
    out = center + np.asarray(matrix, dtype=np.float64) @ (np.asarray(point, dtype=np.float64) - center)
    return float(out[0]), float(out[1])


def action_permutation(matrix: np.ndarray) -> tuple[int, ...]:
    vectors = normalized_action_vectors()
    transformed = vectors @ np.asarray(matrix, dtype=np.float64).T
    result = []
    for vector in transformed:
        matches = np.flatnonzero(np.all(np.isclose(vectors, vector[None, :], atol=1e-12), axis=1))
        if len(matches) != 1:
            raise D164Error("D4 transform does not map each primitive action uniquely")
        result.append(int(matches[0]))
    return tuple(result)


def transform_scenario_state(scenario: CorridorScenario, state: CorridorState,
                              matrix: np.ndarray) -> tuple[CorridorTaskSpec, CorridorState]:
    spec = scenario.spec
    if not np.isclose(spec.width, spec.height):
        raise ValueError("only the current square task is D4 invariant")
    def point(x: float, y: float) -> tuple[float, float]:
        return transform_point((x, y), matrix, spec.width, spec.height)
    obstacles = []
    for obstacle in spec.obstacles:
        corners = [point(x, y) for x, y in ((obstacle.x_min, obstacle.y_min),
                                            (obstacle.x_min, obstacle.y_max),
                                            (obstacle.x_max, obstacle.y_min),
                                            (obstacle.x_max, obstacle.y_max))]
        obstacles.append(AxisAlignedObstacle(min(x for x, _ in corners), min(y for _, y in corners),
                                             max(x for x, _ in corners), max(y for _, y in corners)))
    goal = point(*spec.goal)
    transformed_spec = CorridorTaskSpec(
        width=spec.width, height=spec.height, obstacles=tuple(obstacles), goal=goal,
        goal_radius=spec.goal_radius, safe_speed=spec.safe_speed, agent_radius=spec.agent_radius,
        acceleration=spec.acceleration, damping=spec.damping, max_speed=spec.max_speed,
        dt=spec.dt, max_steps=spec.max_steps,
    )
    px, py = point(state.x, state.y)
    velocity = np.asarray(matrix, dtype=np.float64) @ np.asarray((state.vx, state.vy), dtype=np.float64)
    transformed_state = CorridorState(px, py, float(velocity[0]), float(velocity[1]),
                                      state.elapsed_steps, state.status)
    return transformed_spec, transformed_state


def canonical_agent_relative(raw52: np.ndarray, r0_physics9: np.ndarray) -> np.ndarray:
    """Deployable origin-relative coordinates plus full local geometry and physics features."""
    raw = np.asarray(raw52, dtype=np.float64)
    physics = np.asarray(r0_physics9, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 52 or physics.shape != (len(raw), 9):
        raise ValueError("canonical features require aligned [N,52] raw and [N,9] physics arrays")
    if not np.isfinite(raw).all() or not np.isfinite(physics).all():
        raise ValueError("canonical inputs must be finite")
    output = []
    for row, derived in zip(raw, physics, strict=True):
        width, height = row[4], row[5]
        diagonal = float(np.hypot(width, height))
        if width <= 0 or height <= 0 or diagonal <= 0:
            raise ValueError("invalid task dimensions")
        x, y = row[0] * width, row[1] * height
        gx, gy = row[6] * width, row[7] * height
        goal_delta = np.asarray(((gx - x) / diagonal, (gy - y) / diagonal))
        boundary = np.asarray((x, width - x, y, height - y), dtype=np.float64) / diagonal
        boxes = []
        for xmin, ymin, xmax, ymax in row[16:52].reshape(9, 4):
            box = np.asarray(((xmin * width - x) / diagonal, (ymin * height - y) / diagonal,
                              (xmax * width - x) / diagonal, (ymax * height - y) / diagonal))
            center_x = (box[0] + box[2]) / 2.0
            center_y = (box[1] + box[3]) / 2.0
            boxes.append((center_x, center_y, *box.tolist()))
        boxes.sort()
        sorted_boxes = np.asarray([box[2:] for box in boxes], dtype=np.float64).reshape(36)
        constants = np.concatenate((np.asarray((width, height), dtype=np.float64) / diagonal, row[8:16]))
        feature = np.concatenate((goal_delta, row[2:4], constants, boundary,
                                  sorted_boxes, derived))
        if feature.shape != (63,) or not np.isfinite(feature).all():
            raise D164Error("canonical deployable representation contract failed")
        output.append(feature)
    return np.asarray(output, dtype=np.float64)


def _nearest(train: np.ndarray, query: np.ndarray, train_layouts: np.ndarray,
             query_layouts: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    mean, std = train.mean(axis=0), train.std(axis=0)
    std[std < 1e-8] = 1.0
    x, q = (train - mean) / std, (query - mean) / std
    indices, distances = np.empty(len(q), dtype=np.int64), np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), 64):
        stop = min(start + 64, len(q))
        squared = np.mean((q[start:stop, None, :] - x[None, :, :]) ** 2, axis=2)
        if query_layouts is not None:
            forbidden = query_layouts[start:stop, None] == train_layouts[None, :]
            squared[forbidden] = np.inf
        nearest = np.argmin(squared, axis=1)
        indices[start:stop] = nearest
        distances[start:stop] = np.sqrt(squared[np.arange(stop - start), nearest])
    if not np.isfinite(distances).all():
        raise D164Error("nearest-neighbor search found no eligible distinct-layout source")
    return indices, distances


def _neighbor_metrics(train: dict[str, np.ndarray], query: dict[str, np.ndarray],
                      rep: str, metrics_cfg: dict[str, Any]) -> dict[str, Any]:
    xtrain, xquery = train[rep], query[rep]
    same_layout_exclusion = query["layouts"] if query is train else None
    indices, distance = _nearest(xtrain, xquery, train["layouts"], same_layout_exclusion)
    source_regret = train["regret"][indices]
    metrics = _candidate_metrics(query["regret"], source_regret, query["summary"],
                                 query["layouts"], metrics_cfg)
    return {"nearest_distance_rms_standardized_per_feature": {
                "median": float(np.median(distance)), "p90": float(np.quantile(distance, .9)),
                "mean": float(np.mean(distance))},
            "neighbor_transferred_decisions": _strip_metrics(metrics),
            "source_indices": indices}


def _load_data(cfg: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    p = cfg["parents"]
    d153_lock = _read(p["d153_lock"])
    d153_result = _read(p["d153_result"])
    d162_lock, d162_manifest, d162_result = (_read(p[k]) for k in ("d162_lock", "d162_manifest", "d162_result"))
    d153_path, d162_path = ROOT / p["d153_rows"], ROOT / p["d162_rows"]
    if (d153_lock.get("schema") != "phase15_d153_coverage_scaling_lock_v3"
            or d153_lock.get("reused_corpus", {}).get("sha256") != sha256_file(d153_path)
            or d153_result.get("reserve_access") is not False
            or d153_result.get("test_rows_indexed_or_scored") != 0):
        raise D164Error("D153 training parent hash/status/scope failed")
    if (d162_lock.get("schema") != "phase15_d162_fresh_layout_relational_lock_v1"
            or d162_manifest.get("dataset_sha256") != sha256_file(d162_path)
            or d162_manifest.get("accounting_sha256") != sha256_file(ROOT / p["d162_accounting"])
            or d162_manifest.get("reserve_access") is not False
            or d162_manifest.get("test_access") is not False
            or d162_result.get("status") != "completed_offline_fresh_layout_relational_evaluation"):
        raise D164Error("D162 parent hash/status/scope failed")
    with np.load(d153_path, allow_pickle=False) as d:
        tr0, tsum, tlayout = d["x_R0"].astype(float), d["summary_R4"].astype(float), d["layouts"].astype(str)
    with np.load(d162_path, allow_pickle=False) as d:
        vr0, vsum, vlayout = d["x_R0"].astype(float), d["summary_R4"].astype(float), d["layouts"].astype(str)
    for states, summary, label in ((tr0, tsum, "D153"), (vr0, vsum, "D162")):
        if (states.ndim != 2 or states.shape[1] != 61 or summary.shape != (len(states), 9, 7)
                or not np.isfinite(states).all() or not np.isfinite(summary).all()):
            raise D164Error(f"{label} saved state/summary contract invalid")
        rebuilt = build_feature_matrix(states[:, :52], "raw52_plus_physics9")
        if not np.allclose(rebuilt, states, atol=1e-6, rtol=0.0):
            raise D164Error(f"{label} R0 does not reproduce from deployable raw state and physics features")
    if set(tlayout.tolist()) & set(vlayout.tolist()):
        raise D164Error("D153/D162 layout hashes overlap")
    train = {"r0": tr0, "canonical": canonical_agent_relative(tr0[:, :52], tr0[:, 52:]),
             "regret": regrets_from_exact_summaries(tsum), "summary": tsum, "layouts": tlayout}
    query = {"r0": vr0, "canonical": canonical_agent_relative(vr0[:, :52], vr0[:, 52:]),
             "regret": regrets_from_exact_summaries(vsum), "summary": vsum, "layouts": vlayout}
    parents = {key: sha256_file(ROOT / path) for key, path in p.items() if isinstance(path, str)}
    parents["d153_training_rows"] = sha256_file(d153_path)
    parents["d162_rows"] = sha256_file(d162_path)
    return train, query, parents


def analyze(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or _cfg()
    train, query, parents = _load_data(cfg)
    metrics_cfg = yaml.safe_load((ROOT / cfg["parents"]["d155_metrics_config"]).read_text(encoding="utf-8"))["metrics"]
    train_eval = {key: _neighbor_metrics(train, train, key, metrics_cfg) for key in ("r0", "canonical")}
    fresh_eval = {key: _neighbor_metrics(train, query, key, metrics_cfg) for key in ("r0", "canonical")}
    # Source indices are diagnostics only; avoid inflating result with index vectors.
    for section in (train_eval, fresh_eval):
        for record in section.values():
            record.pop("source_indices", None)
    raw_median = fresh_eval["r0"]["nearest_distance_rms_standardized_per_feature"]["median"]
    canonical_median = fresh_eval["canonical"]["nearest_distance_rms_standardized_per_feature"]["median"]
    raw_regret = fresh_eval["r0"]["neighbor_transferred_decisions"]["layout_macro_normalized_selected_regret"]
    canonical_regret = fresh_eval["canonical"]["neighbor_transferred_decisions"]["layout_macro_normalized_selected_regret"]
    raw_agreement = fresh_eval["r0"]["neighbor_transferred_decisions"]["best_action_set_agreement"]
    canonical_agreement = fresh_eval["canonical"]["neighbor_transferred_decisions"]["best_action_set_agreement"]
    thresholds = cfg["gates"]["canonical_representation_supported_only_if"]
    distance_reduction = 1.0 - canonical_median / raw_median
    regret_improvement = raw_regret - canonical_regret
    agreement_change = canonical_agreement - raw_agreement
    gate = (distance_reduction >= float(thresholds["nearest_distance_ratio_reduction_at_least"])
            and regret_improvement >= float(thresholds["target_macro_regret_improvement_at_least"])
            and agreement_change >= -float(thresholds["best_action_set_agreement_noninferiority_margin"]))
    action_vectors = normalized_action_vectors()
    dists = np.linalg.norm(action_vectors[:, None, :] - action_vectors[None, :, :], axis=2)
    offdiag = dists[np.triu_indices(ACTION_COUNT, k=1)]
    onehot_dist = np.sqrt(2.0)
    return {"schema": "phase15_d164_representation_symmetry_result_v1",
            "status": "completed_exploratory_representation_audit",
            "source_commit": _git("rev-parse", "HEAD"), "parent_sha256": parents,
            "data": {"d153_training_rows": len(train["r0"]), "d153_layouts": len(set(train["layouts"].tolist())),
                     "d162_query_rows": len(query["r0"]), "d162_layouts": len(set(query["layouts"].tolist())),
                     "d153_d162_layout_overlap": 0, "r0_dimensions": 61, "canonical_dimensions": 63},
            "symmetry_audit": {"valid_mechanics_group": "D4 signed permutations for the square frozen mechanics when the entire scene, boundaries, velocity, and action vectors transform",
                               "generator_distribution_has_D4_symmetry": False,
                               "translation_scope": "coordinate reparameterization only if workspace boundary and all geometry transform together",
                               "transformed_teacher_rankings_in_existing_data": False,
                               "action_vectors_after_diagonal_normalization": action_vectors.tolist(),
                               "onehot_pairwise_distance": float(onehot_dist),
                               "physical_action_pairwise_distance_min": float(offdiag.min()),
                               "physical_action_pairwise_distance_max": float(offdiag.max()),
                               "physical_action_pairwise_unique_distances": sorted(float(v) for v in np.unique(np.round(offdiag, 12)))},
            "train_leave_layout_out": train_eval,
            "d162_query_against_d153_training": fresh_eval,
            "canonical_vs_r0": {"median_distance_reduction_fraction": float(distance_reduction),
                                "macro_regret_improvement": float(regret_improvement),
                                "best_action_set_agreement_change": float(agreement_change),
                                "gate_passed": bool(gate), "gate_thresholds": thresholds},
            "scope": {"new_task_labels": False, "planner_calls": 0, "simulator_calls": 0,
                      "model_training": False, "new_layouts": False,
                      "reserve_access": False, "test_access": False,
                      "closed_loop": False, "dagger": False}}


def _source_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in SOURCES}


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D164Error("commit D164 protocol/source/tests/vault before locking")
    if cfg.get("schema") != "phase15_d164_representation_symmetry_audit_v1":
        raise D164Error("D164 config schema mismatch")
    for key in ("lock", "run_dir", "report"):
        if (ROOT / cfg["outputs"][key]).exists():
            raise D164Error(f"D164 output already exists: {key}")
    train, query, parents = _load_data(cfg)
    return {"schema": "phase15_d164_representation_symmetry_lock_v1",
            "status": "locked_one_shot_saved_data_audit",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": _git("rev-parse", "HEAD"), "source_sha256": _source_hashes(),
            "config_sha256": sha256_file(ROOT / CONFIG), "parent_sha256": parents,
            "shapes": {"train_r0": list(train["r0"].shape), "train_summary": list(train["summary"].shape),
                       "query_r0": list(query["r0"].shape), "query_summary": list(query["summary"].shape)},
            "scope": _cfg()["scope"], "reserve_access": False, "test_access": False}


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    lock = _read(cfg["outputs"]["lock"])
    if lock.get("schema") != "phase15_d164_representation_symmetry_lock_v1":
        raise D164Error("D164 lock schema mismatch")
    if _git("status", "--porcelain"):
        raise D164Error("D164 execution requires a clean committed tree")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    if (lock["source_sha256"] != _source_hashes() or lock["config_sha256"] != sha256_file(ROOT / CONFIG)
            or lock["scope"] != cfg["scope"] or lock.get("reserve_access") is not False
            or lock.get("test_access") is not False):
        raise D164Error("D164 lock/source/config/scope mismatch")
    _, _, parents = _load_data(cfg)
    if parents != lock["parent_sha256"]:
        raise D164Error("D164 parent artifacts changed after lock")
    for key in ("run_dir", "report"):
        if (ROOT / cfg["outputs"][key]).exists():
            raise D164Error(f"D164 output already exists: {key}")
    return cfg, lock


def render_report(result: dict[str, Any], lock_sha: str) -> str:
    lines = ["# D164 — Representation and symmetry audit", "",
             "**Status:** completed exploratory, existing-artifact-only analysis. No models were trained and no task labels were generated.", "",
             f"D153: {result['data']['d153_training_rows']} rows / {result['data']['d153_layouts']} layouts. D162: {result['data']['d162_query_rows']} rows / {result['data']['d162_layouts']} layouts. Layout overlap: zero.", "",
             "## Nearest-neighbor transfer", "",
             "| Split | Representation | Median standardized RMS distance | p90 | Macro regret | Catastrophic | Best-set agreement |", "|---|---|---:|---:|---:|---:|---:|"]
    for label, section in (("D153 leave-layout-out", result["train_leave_layout_out"]),
                           ("D162 query → D153 train", result["d162_query_against_d153_training"])):
        for representation, record in section.items():
            d = record["nearest_distance_rms_standardized_per_feature"]
            m = record["neighbor_transferred_decisions"]
            lines.append(f"| {label} | {representation} | {d['median']:.5f} | {d['p90']:.5f} | {m['layout_macro_normalized_selected_regret']:.5f} | {m['catastrophic_fraction_layout_macro']:.5f} | {m['best_action_set_agreement']:.5f} |")
    g = result["canonical_vs_r0"]
    lines += ["", "## Frozen representation gate", "",
              f"Canonical distance reduction: {g['median_distance_reduction_fraction']:.3%}; macro regret improvement: {g['macro_regret_improvement']:.5f}; best-set agreement change: {g['best_action_set_agreement_change']:.5f}; gate: **{'PASS' if g['gate_passed'] else 'FAIL'}**.", "",
              "## Symmetry/action audit", "",
              f"Mechanically valid transforms: {result['symmetry_audit']['valid_mechanics_group']}. The generated layout distribution itself is D4-asymmetric: {result['symmetry_audit']['generator_distribution_has_D4_symmetry']}. Translation requires transforming boundaries too. Existing data contain no paired transformed teacher rankings.",
              f"One-hot action IDs give a constant pairwise distance of {result['symmetry_audit']['onehot_pairwise_distance']:.5f}; actual normalized action-vector distances range from {result['symmetry_audit']['physical_action_pairwise_distance_min']:.5f} to {result['symmetry_audit']['physical_action_pairwise_distance_max']:.5f}.", "",
              "## Interpretation limits", "",
              "This is nearest-neighbor teacher-regret transfer, not a learned policy or controlled generalization experiment. D162 is a reused development block. D4 mechanics symmetry is structural; no transformed teacher rankings were generated or empirically compared. See protocol for the frozen advancement criterion.", "",
              f"Lock SHA-256: {lock_sha}.", ""]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    cfg, lock = validate_lock()
    out = ROOT / cfg["outputs"]["run_dir"]
    out.mkdir(parents=True, exist_ok=False)
    result = analyze(cfg)
    result["lock_sha256"] = sha256_file(ROOT / cfg["outputs"]["lock"])
    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    report_path = ROOT / cfg["outputs"]["report"]
    report_path.write_text(render_report(result, result["lock_sha256"]), encoding="utf-8")
    manifest = {"schema": "phase15_d164_representation_symmetry_manifest_v1", "status": result["status"],
                "source_commit": lock["source_commit"], "lock_sha256": result["lock_sha256"],
                "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
                "parent_sha256": result["parent_sha256"], "planner_calls": 0, "simulator_calls": 0,
                "model_training": False, "reserve_access": False, "test_access": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
