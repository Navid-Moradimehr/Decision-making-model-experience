"""Offline audit of the value signal in nested exact-search candidate prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase26_search_feedback_v4.yaml"
PROTOCOL = "reports/phase26_search_feedback_protocol.md"
AMENDMENT = "reports/phase26_v4_canonical_audit_amendment.md"
SOURCE = "src/consequence_lab/phase26_search_feedback.py"
TESTS = "tests/test_phase26_search_feedback.py"
CATEGORIES = ("success", "active", "timeout", "collision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_json(value: object) -> str:
    """Normalize tuples/lists through JSON and reject non-finite results."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config(root: Path = ROOT) -> dict:
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def _all_bound_paths(cfg: dict) -> dict[str, str]:
    paths = {"config": CONFIG, "protocol": cfg["protocol"], "amendment": AMENDMENT,
             "source": SOURCE, "tests": TESTS}
    for cohort, bindings in cfg["parents"].items():
        paths.update({f"{cohort}_{key}": value for key, value in bindings.items()
                      if key not in {"lock_sha256", "records_sha256", "result_sha256", "manifest_sha256",
                                     "run_manifest_sha256", "score_manifest_sha256"}})
    return paths


def _dependency_hashes(root: Path, cfg: dict) -> dict[str, str]:
    return {path: sha256(root / path) for path in sorted(set(_all_bound_paths(cfg).values()))}


def verify_parents(root: Path = ROOT, cfg: dict | None = None) -> dict:
    cfg = cfg or config(root)
    expected = {
        "phase24": {"lock": "lock_sha256", "records": "records_sha256", "result": "result_sha256",
                    "run_manifest": "run_manifest_sha256", "score_manifest": "score_manifest_sha256"},
        "phase25": {"lock": "lock_sha256", "records": "records_sha256", "result": "result_sha256",
                    "run_manifest": "run_manifest_sha256", "score_manifest": "score_manifest_sha256"},
    }
    verified = {}
    for cohort, keys in expected.items():
        bindings = cfg["parents"][cohort]
        for path_key, digest_key in keys.items():
            actual = sha256(root / bindings[path_key])
            if actual != bindings[digest_key].upper():
                raise RuntimeError(f"{cohort} {path_key} hash mismatch")
        run_manifest = json.loads((root / bindings["run_manifest"]).read_text(encoding="utf-8"))
        score_manifest = json.loads((root / bindings["score_manifest"]).read_text(encoding="utf-8"))
        if (run_manifest.get("records_sha256", "").upper() != bindings["records_sha256"].upper()
                or score_manifest.get("records_sha256", "").upper() != bindings["records_sha256"].upper()
                or score_manifest.get("result_sha256", "").upper() != bindings["result_sha256"].upper()):
            raise RuntimeError(f"{cohort} manifests do not bind the configured records/result")
        if run_manifest.get("reserve_access") or run_manifest.get("test_access") or run_manifest.get("d168_access"):
            raise RuntimeError(f"{cohort} manifest reports sealed-data access")
        verified[cohort] = {key: bindings[digest_key] for key, digest_key in keys.items()}
    return verified


def freeze(root: Path = ROOT) -> dict:
    cfg = config(root)
    if cfg["schema"] != "phase26-search-feedback-audit-v1":
        raise ValueError("wrong Phase26 schema")
    if cfg["scope"]["reserve_test_d168_access"] or cfg["scope"]["simulator_calls"] or cfg["scope"]["candidate_evaluations"]:
        raise ValueError("Phase26 scope must remain offline and sealed-data-free")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit Phase26 protocol, config, source, and tests before freezing")
    parents = verify_parents(root, cfg)
    # Identity and nested-bank audit is part of freezing, not outcome analysis.
    _cohort_pairs(root, cfg, "phase24")
    _cohort_pairs(root, cfg, "phase25")
    lock_path = root / cfg["outputs"]["execution_lock"]
    if lock_path.exists():
        raise FileExistsError(f"refusing to overwrite {lock_path}")
    content = _dependency_hashes(root, cfg)
    lock = {"schema": "phase26-search-feedback-lock-v1", "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "hashes": content,
        "parent_hashes": parents, "historical_inputs_only": True,
        "reserve_test_d168_access": False, "simulator_calls": 0, "candidate_evaluations": 0}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"lock_path": lock_path.relative_to(root).as_posix(), "lock_sha256": sha256(lock_path),
            "source_commit": lock["source_commit"], "bound_inputs": len(content)}


def _load_first_decision_budgets(root: Path, path: str) -> dict[tuple[int, int, int], dict]:
    """Stream full historical episodes and retain only matched first-decision B24/B32 banks."""
    by_key = {}
    with (root / path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["method"] not in {"shifted_b24", "shifted_b32"}:
                continue
            decisions = row.get("decision_diagnostics") or []
            if not decisions:
                raise RuntimeError("historical episode has no first-decision diagnostics")
            d = decisions[0]
            compact = {key: d[key] for key in ("state_raw52", "previous_plan", "candidate_bank", "query_seed", "horizon")}
            key = (int(row["seed"]), int(row["replica"]), int(row["budget"]))
            if key in by_key:
                raise RuntimeError("duplicate historical layout/replica/budget row")
            by_key[key] = {"seed": key[0], "replica": key[1], "budget": key[2],
                           "layout_sha256": row["layout_sha256"], "decision_diagnostics": [compact]}
    return by_key


def _best(bank: list[dict]) -> tuple:
    if not bank:
        raise ValueError("empty candidate prefix")
    return min(tuple(candidate["objective"]) for candidate in bank)


def _winner(bank: list[dict]) -> dict:
    return min(bank, key=lambda candidate: (tuple(candidate["objective"]), tuple(candidate["sequence"])))


def _candidate_features(bank: list[dict], state: list[float], previous: list[int] | None) -> list[float]:
    if len(state) != 52:
        raise ValueError(f"expected deployable state_raw52, got {len(state)} values")
    counts = [sum(candidate["category"] == category for candidate in bank) for category in CATEGORIES]
    best = _best(bank)
    tied = sum(tuple(candidate["objective"]) == best for candidate in bank)
    same_class = [tuple(candidate["objective"])[1] for candidate in bank
                  if tuple(candidate["objective"])[0] == best[0]]
    same_class = sorted(same_class)
    gap = same_class[1] - same_class[0] if len(same_class) > 1 else 0.0
    first_actions = [int(candidate["sequence"][0]) for candidate in bank]
    action_counts = np.bincount(first_actions, minlength=9).astype(float)
    probs = action_counts[action_counts > 0] / len(first_actions)
    entropy = float(-np.sum(probs * np.log(probs)))
    prior = [-1.0] * 8 if previous is None else [float(x) / 8.0 for x in previous]
    return ([float(x) for x in state] + prior + [float(x) for x in counts]
            + [float(best[0] == c) for c in range(4)]
            + [float(best[1]), float(gap), float(tied), float(len(set(first_actions))), entropy])


def _pair(low_row: dict, high_row: dict, cohort: str, cfg: dict) -> dict:
    low, high = low_row["decision_diagnostics"][0], high_row["decision_diagnostics"][0]
    k, extension = int(cfg["analysis"]["budget_prefix"]), int(cfg["analysis"]["extended_budget"])
    if (low_row["seed"], low_row["replica"]) != (high_row["seed"], high_row["replica"]):
        raise RuntimeError("candidate banks do not refer to the same layout/replica")
    if (low["state_raw52"] != high["state_raw52"] or low.get("previous_plan") != high.get("previous_plan")
            or low["query_seed"] != high["query_seed"] or low["horizon"] != high["horizon"]):
        raise RuntimeError("matched first-decision deployable context differs")
    low_bank, high_bank = low["candidate_bank"], high["candidate_bank"]
    if len(low_bank) != k or len(high_bank) != extension:
        raise RuntimeError("unexpected effective candidate budget")
    for index, candidate in enumerate(low_bank):
        other = high_bank[index]
        if candidate["sequence"] != other["sequence"] or candidate["objective"] != other["objective"]:
            raise RuntimeError("nested bank prefix/parity mismatch")
    best_low, best_high = _best(low_bank), _best(high_bank)
    added = high_bank[k:extension]
    low_success = any(candidate["category"] == "success" for candidate in low_bank)
    newly_success = (not low_success and any(candidate["category"] == "success" for candidate in added))
    low_first, high_first = _winner(low_bank), _winner(high_bank)
    features = _candidate_features(low_bank, low["state_raw52"], low.get("previous_plan"))
    # Marginal objective cost is logged exact rollout transitions for the eight added candidates.
    return {"cohort": cohort, "seed": int(low_row["seed"]), "replica": int(low_row["replica"]),
            "features": features, "strict_objective_improvement": int(best_high < best_low),
            "outcome_category_improvement": int(best_high[0] < best_low[0]),
            "newly_available_success_candidate": int(newly_success),
            "best_first_action_changed": int(low_first["sequence"][0] != high_first["sequence"][0]),
            "added_transitions": int(sum(candidate["transitions"] for candidate in added)),
            "best_low": best_low, "best_high": best_high,
            "success_candidates_low": sum(c["category"] == "success" for c in low_bank),
            "success_candidates_high": sum(c["category"] == "success" for c in high_bank)}


def _cohort_pairs(root: Path, cfg: dict, cohort: str) -> list[dict]:
    bindings = cfg["parents"][cohort]
    by_key = _load_first_decision_budgets(root, bindings["records"])
    expected_layouts = 500
    seeds = sorted({key[0] for key in by_key})
    if len(seeds) != expected_layouts or len(by_key) != expected_layouts * 3 * 2:
        raise RuntimeError(f"{cohort} must provide 500 layouts × 3 replicas × B24/B32")
    output = []
    for seed in seeds:
        for replica in range(3):
            low, high = by_key[(seed, replica, 24)], by_key[(seed, replica, 32)]
            if low["layout_sha256"] != high["layout_sha256"]:
                raise RuntimeError("paired budget rows have different layout identity")
            output.append(_pair(low, high, cohort, cfg))
    return output


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, lam: float,
                       clip: float) -> tuple[np.ndarray, np.ndarray]:
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    a = (x_train - mean) / scale
    b = (x_test - mean) / scale
    a = np.column_stack([np.ones(len(a)), a])
    b = np.column_stack([np.ones(len(b)), b])
    penalty = np.eye(a.shape[1]) * lam
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(a.T @ a + penalty, a.T @ y_train)
    return np.clip(b @ weights, clip, 1.0 - clip), np.full(len(x_test), float(y_train.mean()))


def _fold_id(seed: int, cohort: str, cfg: dict) -> int:
    start = 209000 if cohort == "phase24" else 209500
    return ((int(seed) - start) % 500) % int(cfg["analysis"]["folds"])


def _brier_skill(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> dict:
    brier, base = float(np.mean((pred - y) ** 2)), float(np.mean((baseline - y) ** 2))
    clipped = np.clip(pred, 1e-6, 1 - 1e-6)
    logloss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    return {"brier": brier, "prevalence_baseline_brier": base,
            "brier_skill": float(1 - brier / base) if base > 0 else None,
            "log_loss": logloss, "prevalence": float(y.mean()), "n": int(len(y))}


def _grouped_cv(rows: list[dict], cohort: str, target: str, cfg: dict) -> dict:
    x = np.asarray([row["features"] for row in rows], dtype=float)
    y = np.asarray([row[target] for row in rows], dtype=float)
    folds = np.asarray([_fold_id(row["seed"], cohort, cfg) for row in rows])
    pred, baseline = np.zeros(len(y)), np.zeros(len(y))
    for fold in range(int(cfg["analysis"]["folds"])):
        train, test = folds != fold, folds == fold
        if not np.any(test):
            raise RuntimeError("grouped fold has no held-out rows")
        pred[test], baseline[test] = _ridge_fit_predict(x[train], y[train], x[test],
            float(cfg["analysis"]["ridge_l2"]), float(cfg["analysis"]["probability_clip"]))
    return {"metrics": _brier_skill(y, pred, baseline), "predictions": pred.tolist(),
            "baseline_predictions": baseline.tolist()}


def _cross_cohort(train_rows: list[dict], test_rows: list[dict], target: str, cfg: dict) -> dict:
    xtr, ytr = np.asarray([r["features"] for r in train_rows]), np.asarray([r[target] for r in train_rows], dtype=float)
    xte, yte = np.asarray([r["features"] for r in test_rows]), np.asarray([r[target] for r in test_rows], dtype=float)
    pred, baseline = _ridge_fit_predict(xtr, ytr, xte, float(cfg["analysis"]["ridge_l2"]),
                                         float(cfg["analysis"]["probability_clip"]))
    return {"metrics": _brier_skill(yte, pred, baseline), "predictions": pred.tolist(),
            "baseline_predictions": baseline.tolist()}


def _descriptive(rows: list[dict]) -> dict:
    return {"matched_first_decisions": len(rows),
            "strict_objective_improvement_rate": float(np.mean([r["strict_objective_improvement"] for r in rows])),
            "outcome_category_improvement_rate": float(np.mean([r["outcome_category_improvement"] for r in rows])),
            "new_success_candidate_rate": float(np.mean([r["newly_available_success_candidate"] for r in rows])),
            "best_first_action_change_rate": float(np.mean([r["best_first_action_changed"] for r in rows])),
            "mean_added_exact_transitions": float(np.mean([r["added_transitions"] for r in rows])),
            "median_added_exact_transitions": float(np.median([r["added_transitions"] for r in rows])),
            "mean_success_candidates_b24": float(np.mean([r["success_candidates_low"] for r in rows])),
            "mean_success_candidates_b32": float(np.mean([r["success_candidates_high"] for r in rows]))}


def analyze(root: Path = ROOT, cfg: dict | None = None) -> dict:
    cfg = cfg or config(root)
    verify_parents(root, cfg)
    p24, p25 = _cohort_pairs(root, cfg, "phase24"), _cohort_pairs(root, cfg, "phase25")
    targets = cfg["analysis"]["targets"]
    models = {}
    for target in targets:
        models[target] = {
            "phase24_grouped_cv": _grouped_cv(p24, "phase24", target, cfg),
            "phase25_grouped_cv": _grouped_cv(p25, "phase25", target, cfg),
            "phase24_to_phase25": _cross_cohort(p24, p25, target, cfg),
            "phase25_to_phase24": _cross_cohort(p25, p24, target, cfg),
        }
    return {"schema": "phase26-search-feedback-result-v1", "analysis_only": True,
            "cohorts": {"phase24": _descriptive(p24), "phase25": _descriptive(p25)},
            "predictability": {target: {split: fit["metrics"] for split, fit in fits.items()}
                                for target, fits in models.items()},
            "rows": {"phase24": p24, "phase25": p25},
            "oof_predictions": {target: {split: fit["predictions"] for split, fit in fits.items()}
                                for target, fits in models.items()},
            "baseline_predictions": {target: {split: fit["baseline_predictions"] for split, fit in fits.items()}
                                     for target, fits in models.items()},
            "simulator_calls": 0, "candidate_evaluations": 0, "reserve_test_d168_access": False}


def _lock(root: Path, cfg: dict) -> dict:
    path = root / cfg["outputs"]["execution_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase26-search-feedback-lock-v1":
        raise RuntimeError("invalid Phase26 lock schema")
    for name, digest in lock["hashes"].items():
        if sha256(root / name) != digest:
            raise RuntimeError(f"Phase26 locked dependency changed: {name}")
    verify_parents(root, cfg)
    lock["lock_sha256"] = sha256(path)
    return lock


def run(root: Path = ROOT) -> dict:
    cfg = config(root)
    lock = _lock(root, cfg)
    out = root / cfg["outputs"]["run_dir"]
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    result = analyze(root, cfg)
    out.mkdir(parents=True)
    result_path = out / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema": "phase26-search-feedback-manifest-v1", "execution_lock_sha256": lock["lock_sha256"],
                "result_sha256": sha256(result_path), "parents": lock["parent_hashes"],
                "simulator_calls": 0, "candidate_evaluations": 0, "reserve_test_d168_access": False,
                "diagnostic_ridge_fits": 36}
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def audit(root: Path = ROOT) -> dict:
    cfg = config(root)
    lock = _lock(root, cfg)
    out = root / cfg["outputs"]["run_dir"]
    result_path, manifest_path = out / "result.json", out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("execution_lock_sha256") != lock["lock_sha256"] or manifest.get("result_sha256") != sha256(result_path):
        raise RuntimeError("Phase26 result manifest mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    # Remove provenance/array duplication from equality checking by recomputing the deterministic report.
    reproduced = analyze(root, cfg)
    if _canonical_json(result) != _canonical_json(reproduced):
        raise RuntimeError("Phase26 result does not reproduce from the locked historical inputs")
    return {"audit": "pass", "result_sha256": sha256(result_path), "manifest_sha256": sha256(manifest_path),
            "simulator_calls": 0, "candidate_evaluations": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "audit"))
    args = parser.parse_args()
    cfg = config()
    if args.command == "freeze":
        result = freeze()
        result["lock_sha256"] = sha256(ROOT / cfg["outputs"]["execution_lock"])
    elif args.command == "run":
        result = run()
    else:
        result = audit()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
