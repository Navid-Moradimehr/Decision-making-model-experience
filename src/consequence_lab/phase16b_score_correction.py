"""Score-only correction for Phase 16B's reversed collision-recall selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase16b_score_correction_v1.yaml"
PROTOCOL = ROOT / "reports/phase16b_score_correction_protocol.md"
ABORT_REPORT = ROOT / "reports/phase16b_metric_contract_abort_v2.md"
PARENT_SOURCE = ROOT / "src/consequence_lab/phase16b_outcome_proposal.py"
PARENT_LOCK = ROOT / "experiments/phase16b_outcome_proposal_v2.lock.json"
RUN_DIR = ROOT / "artifacts/runs/phase16b_outcome_proposal_v2"
LOCK = ROOT / "experiments/phase16b_score_correction_v1.lock.json"
SOURCE = "src/consequence_lab/phase16b_score_correction.py"


class ScoreCorrectionError(RuntimeError):
    """Correction contract or provenance failure."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _cfg() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _files() -> list[str]:
    return ["configs/phase16b_score_correction_v1.yaml",
            "reports/phase16b_score_correction_protocol.md",
            "reports/phase16b_metric_contract_abort_v2.md",
            "src/consequence_lab/phase16b_score_correction.py",
            "src/consequence_lab/phase16b_outcome_proposal.py",
            "tests/test_phase16b_score_correction.py",
            "experiments/phase16b_outcome_proposal.lock.json",
            "experiments/phase16b_outcome_proposal_v2.lock.json",
            "artifacts/runs/phase16b_outcome_proposal_v2/result.json",
            "artifacts/runs/phase16b_outcome_proposal_v2/manifest.json",
            "artifacts/runs/phase16b_outcome_proposal_v2/oof_predictions.npz"]


def freeze_lock() -> dict[str, Any]:
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise ScoreCorrectionError("commit protocol/source/tests before freezing correction lock")
    if LOCK.exists():
        raise ScoreCorrectionError("refusing to overwrite the score-correction lock")
    payload = {"schema": "phase16b-score-correction-lock-v1",
               "source_commit": subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "files": {name: sha256(ROOT / name) for name in _files()},
               "data_policy": {"saved_oof_only": True, "training": False,
                               "simulator": False, "planner": False,
                               "validation": False, "fresh_development": False,
                               "reserve": False, "test": False,
                               "closed_loop": False, "dagger": False}}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def verify_lock() -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase16b-score-correction-lock-v1":
        raise ScoreCorrectionError("unsupported score-correction lock")
    for name, digest in lock["files"].items():
        if sha256(ROOT / name) != digest:
            raise ScoreCorrectionError(f"locked correction input changed: {name}")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise ScoreCorrectionError("working tree must be clean before score correction")
    return lock


def _selection_weights(scores: np.ndarray, k: int, *, highest: bool,
                       tolerance: float) -> np.ndarray:
    """Fractional membership weights for a tie-aware top/bottom-k selection."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not 1 <= k <= len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a finite vector and k must be in range")
    ordered = np.sort(values)
    cutoff = ordered[-k] if highest else ordered[k - 1]
    strict = values > cutoff + tolerance if highest else values < cutoff - tolerance
    tied = np.abs(values - cutoff) <= tolerance
    slots = k - int(strict.sum())
    if slots < 0 or int(tied.sum()) == 0:
        raise ValueError("invalid tie partition")
    weights = strict.astype(np.float64)
    weights[tied] = slots / int(tied.sum())
    return weights


def _state_stats(success: np.ndarray, risk: np.ndarray, *, collision_k: int,
                 tolerance: float) -> tuple[float, float, float, float]:
    """Return top1 success, high-risk collision recall numerator/denominator,
    and low-risk top1 collision selection rate for aligned [R,N,9] outcomes.
    """
    y = np.asarray(success, dtype=np.int64)
    p = np.asarray(risk, dtype=np.float64)
    if (y.ndim != 3 or y.shape[2] != 9 or p.shape != y.shape[1:]
            or not set(np.unique(y)).issubset({0, 1})):
        raise ValueError("expected outcomes [replicates,states,9] and aligned risks [states,9]")
    top_success: list[float] = []
    high_selected = total_collisions = 0.0
    low_collisions: list[float] = []
    for state_i in range(y.shape[1]):
        scores = p[state_i]
        low = _selection_weights(scores, 1, highest=False, tolerance=tolerance)
        high = _selection_weights(scores, collision_k, highest=True, tolerance=tolerance)
        for rep_i in range(y.shape[0]):
            outcome = y[rep_i, state_i]
            collision = 1 - outcome
            top_success.append(float(np.dot(low, outcome)))
            low_collisions.append(float(np.dot(low, collision)))
            high_selected += float(np.dot(high, collision))
            total_collisions += float(collision.sum())
    return (float(np.mean(top_success)), high_selected, total_collisions,
            float(np.mean(low_collisions)))


def decision_metrics(success: np.ndarray, risk: np.ndarray, *, collision_k: int = 2,
                     tolerance: float = 1e-12) -> dict[str, float]:
    top, selected, collisions, low_collision = _state_stats(
        success, risk, collision_k=collision_k, tolerance=tolerance)
    recall = selected / max(1.0, collisions)
    return {"top_one_low_risk_success": top,
            f"high_risk_collision_recall_at_{collision_k}": recall,
            "low_risk_collision_selection_rate_at_1": low_collision,
            "uniform_high_risk_recall": collision_k / 9.0}


def _layout_components(success: np.ndarray, risk: np.ndarray, layouts: np.ndarray,
                       collision_k: int, tolerance: float) -> tuple[np.ndarray, ...]:
    unique = np.unique(layouts)
    top = np.zeros(len(unique), dtype=np.float64)
    selected = np.zeros(len(unique), dtype=np.float64)
    collisions = np.zeros(len(unique), dtype=np.float64)
    low_collision = np.zeros(len(unique), dtype=np.float64)
    for i, layout in enumerate(unique):
        rows = np.flatnonzero(layouts == layout)
        y = success[:, rows, :]
        p = risk[rows]
        top[i], selected[i], collisions[i], low_collision[i] = _state_stats(
            y, p, collision_k=collision_k, tolerance=tolerance)
        # Convert the per-decision low-risk collision rate to its additive
        # fractional selected-collision count for cluster bootstrap.
        low_collision[i] *= y.shape[0] * y.shape[1]
    return top, selected, collisions, low_collision


def _bootstrap(success: np.ndarray, model: np.ndarray, baseline: np.ndarray,
               layouts: np.ndarray, *, draws: int, seed: int, collision_k: int,
               tolerance: float) -> dict[str, Any]:
    model_parts = _layout_components(success, model, layouts, collision_k, tolerance)
    base_parts = _layout_components(success, baseline, layouts, collision_k, tolerance)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(model_parts[0]), size=(draws, len(model_parts[0])))
    top_gain = (model_parts[0] - base_parts[0])[sample].mean(axis=1)
    selected_sum = model_parts[1][sample].sum(axis=1)
    collision_sum = model_parts[2][sample].sum(axis=1)
    risk_gain = selected_sum / np.maximum(1.0, collision_sum) - collision_k / 9.0
    return {"top_one_success_gain_point": float(model_parts[0].mean() - base_parts[0].mean()),
            "top_one_success_gain_95_ci": [float(np.quantile(top_gain, .025)),
                                           float(np.quantile(top_gain, .975))],
            "high_risk_recall_gain_point": float(
                model_parts[1].sum() / max(1.0, model_parts[2].sum()) - collision_k / 9.0),
            "high_risk_recall_gain_95_ci": [float(np.quantile(risk_gain, .025)),
                                             float(np.quantile(risk_gain, .975))]}


def score() -> dict[str, Any]:
    lock = verify_lock()
    cfg = _cfg()
    started = time.perf_counter()
    run_dir = ROOT / cfg["parent_run"]
    expected = cfg["inputs"]
    parent_lock_sha = sha256(PARENT_LOCK)
    if parent_lock_sha != expected["source_lock_sha256"]:
        raise ScoreCorrectionError("parent source-lock SHA mismatch")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, key in (("result.json", "result_sha256"),
                      ("oof_predictions.npz", "oof_predictions_sha256")):
        if sha256(run_dir / name) != expected[key] or sha256(run_dir / name) != manifest[key]:
            raise ScoreCorrectionError(f"parent {name} SHA mismatch")
    if sha256(run_dir / "manifest.json") != expected["manifest_sha256"]:
        raise ScoreCorrectionError("parent manifest SHA mismatch")
    if manifest.get("lock_sha256") != parent_lock_sha:
        raise ScoreCorrectionError("parent manifest does not reference the v2 lock")
    parent_result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    if parent_result.get("status") != "complete" or any((
            parent_result.get("reserve_access"), parent_result.get("test_access"),
            parent_result.get("fresh_layout_outcomes"), parent_result.get("simulator_transitions"),
            parent_result.get("planner_queries"), parent_result.get("closed_loop"),
            parent_result.get("dagger"))):
        raise ScoreCorrectionError("parent run violates score-only source policy")
    with np.load(run_dir / "oof_predictions.npz", allow_pickle=False) as archive:
        required = {"r0", "canonical", "action_frequency", "success_draws", "layouts", "row_ids", "fold_ids"}
        if set(archive.files) != required:
            raise ScoreCorrectionError("unexpected saved OOF schema")
        r0, canonical = archive["r0"].copy(), archive["canonical"].copy()
        prior = archive["action_frequency"].copy()
        success = archive["success_draws"].copy()
        layouts = archive["layouts"].astype(str)
        row_ids, fold_ids = archive["row_ids"].copy(), archive["fold_ids"].copy()
    if (success.shape != (3, 128, 9) or r0.shape != (3, 128, 9)
            or canonical.shape != r0.shape or prior.shape != (128, 9)
            or layouts.shape != (128,) or len(np.unique(layouts)) != 64
            or row_ids.shape != (128,) or fold_ids.shape != (128,)
            or not np.isfinite(r0).all() or not np.isfinite(canonical).all()
            or not np.isfinite(prior).all()
            or (r0 < 0).any() or (r0 > 1).any()
            or (canonical < 0).any() or (canonical > 1).any()
            or (prior < 0).any() or (prior > 1).any()):
        raise ScoreCorrectionError("parent arrays violate the frozen Phase16B dimensions")
    if not set(np.unique(success)).issubset({0, 1}) or set(np.unique(fold_ids)) != set(range(5)):
        raise ScoreCorrectionError("invalid outcome labels or grouped folds")
    tolerance = float(cfg["scoring"]["tie_abs_tolerance"])
    collision_k = int(cfg["scoring"]["high_risk_collision_recall_k"])
    draws = int(cfg["scoring"]["bootstrap"]["draws"])
    bootstrap_seed = int(cfg["scoring"]["bootstrap"]["seed"])
    results = {}
    for rep_name, prediction in (("r0", r0), ("canonical", canonical)):
        seed_rows = []
        for seed_i, saved in enumerate(parent_result["out_of_fold_metrics"][rep_name]["seed_results"]):
            risk = prediction[seed_i]
            metric = decision_metrics(success, risk, collision_k=collision_k, tolerance=tolerance)
            baseline_metric = decision_metrics(success, prior, collision_k=collision_k, tolerance=tolerance)
            boot = _bootstrap(success, risk, prior, layouts, draws=draws,
                              seed=bootstrap_seed + seed_i, collision_k=collision_k,
                              tolerance=tolerance)
            # Contract check: top-one metric was not affected by risk-ranking direction.
            if abs(metric["top_one_low_risk_success"]
                   - float(saved["metrics"]["top_one_success_rate"])) > 1e-12:
                raise ScoreCorrectionError("corrected top-one metric failed exact parity")
            seed_rows.append({"seed": saved["seed"], "metrics": metric,
                              "action_frequency_metrics": baseline_metric,
                              "paired_bootstrap": boot})
        results[rep_name] = {"seed_results": seed_rows}
    gate = cfg["advancement"]
    gate_summary = {}
    passing_representations = []
    for rep_name, rep_result in results.items():
        passing = 0
        for row in rep_result["seed_results"]:
            boot = row["paired_bootstrap"]
            passes = (boot["top_one_success_gain_point"] >= float(
                gate["top_one_success_gain_over_action_frequency_at_least"])
                and boot["top_one_success_gain_95_ci"][0] > float(
                    gate["top_one_success_gain_ci_lower_above"])
                and boot["high_risk_recall_gain_point"] >= float(
                    gate["high_risk_collision_recall_at_two_gain_over_uniform_at_least"])
                and boot["high_risk_recall_gain_95_ci"][0] > float(
                    gate["high_risk_collision_recall_gain_ci_lower_above"]))
            passing += int(passes)
        gate_summary[rep_name] = {"seeds_passing_joint_gate": passing,
                                  "required": int(gate["minimum_seeds_passing_joint_gate"]),
                                  "passes": passing >= int(gate["minimum_seeds_passing_joint_gate"])}
        if gate_summary[rep_name]["passes"]:
            passing_representations.append(rep_name)
    output = {"schema": "phase16b-score-correction-result-v1", "status": "complete",
              "gate_status": "pass" if passing_representations else "fail",
              "passing_representations": passing_representations,
              "parent_result_sha256": sha256(run_dir / "result.json"),
              "parent_manifest_sha256": sha256(run_dir / "manifest.json"),
              "parent_oof_predictions_sha256": sha256(run_dir / "oof_predictions.npz"),
              "parent_v2_lock_sha256": parent_lock_sha,
              "correction_lock_sha256": sha256(LOCK), "results": results,
              "gate_summary": gate_summary, "data_policy": lock["data_policy"],
              "model_fitting": False, "simulator_transitions": 0, "planner_queries": 0,
              "reserve_access": False, "test_access": False,
              "fresh_layout_outcomes": False, "closed_loop": False, "dagger": False,
              "runtime_seconds": time.perf_counter() - started}
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=False)
    result_path = out_dir / "result.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    manifest_out = {"schema": "phase16b-score-correction-manifest-v1",
                    "lock_sha256": sha256(LOCK), "result_sha256": sha256(result_path)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n",
                                           encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score"))
    args = parser.parse_args()
    output = freeze_lock() if args.mode == "freeze" else score()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
