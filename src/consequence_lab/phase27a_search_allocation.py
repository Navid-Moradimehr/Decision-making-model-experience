"""Historical-only feasibility analysis for Phase27 adaptive search allocation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import yaml

from consequence_lab import phase17_proposal_guidance as p17
from consequence_lab import phase18_proposal_controls as p18
from consequence_lab import phase20_exact_plan_warmstart as p20
from consequence_lab import phase26_search_feedback as p26

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase27a_search_allocation_feasibility_v1.yaml"
PHASE27_CONFIG = "configs/phase27_adaptive_search_allocation_v1.yaml"
PROTOCOL = "reports/phase27_adaptive_search_allocation_protocol.md"
CORRECTION = "reports/phase27_candidate_set_nesting_correction.md"
LOCK_AMENDMENT = "reports/phase27a_lock_lifecycle_amendment.md"
SOURCE = "src/consequence_lab/phase27a_search_allocation.py"
TESTS = "tests/test_phase27a_search_allocation.py"
PARENT_CONFIG = "configs/phase26_search_feedback_v4.yaml"
PARENT_SOURCE = "src/consequence_lab/phase26_search_feedback.py"
GENERATOR_SOURCES = (
    "src/consequence_lab/phase20_exact_plan_warmstart.py",
    "src/consequence_lab/phase17_proposal_guidance.py",
    "src/consequence_lab/phase16d_compute_frontier.py",
)
TARGETS = (
    "primary_decision_impact",
    "best_outcome_category_improves",
    "strict_objective_improves",
    "first_action_changes",
    "success_candidate_newly_available",
)
CATEGORY_CODES = {"success": 0, "active": 1, "timeout": 2, "collision": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config(root: Path = ROOT) -> dict:
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _dependency_paths(cfg: dict) -> list[str]:
    paths = [CONFIG, PHASE27_CONFIG, cfg["protocol"], CORRECTION, LOCK_AMENDMENT, SOURCE, TESTS,
             PARENT_CONFIG, PARENT_SOURCE, "reports/phase26_search_feedback_protocol.md",
             "reports/phase26_v4_canonical_audit_amendment.md",
             "tests/test_phase26_search_feedback.py", *GENERATOR_SOURCES]
    paths.extend(cfg["parents"][key] for key in (
        "phase26_lock", "phase26_result", "phase24_records", "phase25_records"))
    return sorted(set(paths))


def _verify_inputs(root: Path, cfg: dict) -> dict:
    if cfg.get("schema") != "phase27a-search-allocation-feasibility-v1":
        raise ValueError("wrong Stage27A config schema")
    if cfg["scope"] != {
        "historical_inputs_only": True,
        "simulator_calls": 0,
        "candidate_rollout_evaluations": 0,
        "fresh_layout_generation": False,
        "reserve_test_d168_access": False,
        "stage27b_episode_evaluation": False,
    }:
        raise RuntimeError("Stage27A scope must remain historical-only and sealed-data-free")
    if cfg["analysis"]["primary_target"] != "strict_objective_improvement_and_selected_first_action_changes":
        raise RuntimeError("primary target differs from the corrected protocol")
    phase27_cfg = yaml.safe_load((root / PHASE27_CONFIG).read_text(encoding="utf-8"))
    if (phase27_cfg["execution"]["stage27a_historical_analysis_authorized"] is not True
            or phase27_cfg["execution"]["stage27b_episode_run_authorized"] is not False):
        raise RuntimeError("authorization scope mismatch")

    p26_cfg = p26.config(root)
    p26_lock = p26._lock(root, p26_cfg)
    expected_phase26_lock = cfg["parents"]["phase26_lock_sha256"]
    if (expected_phase26_lock != phase27_cfg["parent"]["phase26_lock_sha256"]
            or p26_lock["lock_sha256"] != expected_phase26_lock):
        raise RuntimeError("Phase26 lock hash differs from the Stage27A parent pin")
    p26.audit(root)

    expected_parent = {
        "phase24_records": "8C3610927622B68FF1C0FEECB992F4344C3C1D863B6BE085CDA08A12528E32CE",
        "phase25_records": "EA3A641D000760DD158115077EA9A5BF2252599C27E30D466D0A506EEADB6461",
    }
    for key, expected in expected_parent.items():
        if cfg["parents"][key + "_sha256"].upper() != expected:
            raise RuntimeError(f"configured {key} digest differs from the audited parent")
        if sha256(root / cfg["parents"][key]) != expected:
            raise RuntimeError(f"{key} source hash mismatch")
    return {path: sha256(root / path) for path in _dependency_paths(cfg)}


def _generated_bank(horizon: int, budget: int, query_seed: int,
                    previous: list[int] | None, attempt_cap: int) -> tuple[tuple[int, ...], ...]:
    kind = "shift_previous_winner" if previous is not None else "uniform"
    bank, _ = p20.candidate_bank(horizon, budget, query_seed, kind, previous, 0, 0, attempt_cap)
    return tuple(tuple(int(action) for action in sequence) for sequence in bank)


def _check_candidate(candidate: dict) -> tuple[tuple[int, ...], tuple]:
    sequence = tuple(int(action) for action in candidate["sequence"])
    objective = tuple(candidate["objective"])
    category = candidate["category"]
    if len(objective) != 3 or category not in CATEGORY_CODES:
        raise RuntimeError("invalid exact candidate objective/category schema")
    if int(objective[0]) != CATEGORY_CODES[category]:
        raise RuntimeError("candidate category disagrees with the frozen objective category")
    if not sequence:
        raise RuntimeError("empty action sequence in candidate bank")
    return sequence, objective


def _compare_candidate_sets(saved_b32: list[dict], generated_b24: tuple[tuple[int, ...], ...],
                            generated_b32: tuple[tuple[int, ...], ...], state: list[float],
                            previous: list[int] | None) -> dict:
    """Join bank outcomes by sequence identity; budgets are not positional prefixes."""
    saved_sequences = [_check_candidate(candidate)[0] for candidate in saved_b32]
    if tuple(saved_sequences) != generated_b32:
        raise RuntimeError("generated B32 bank does not exactly reproduce saved candidate order")
    saved_set, low_set = set(saved_sequences), set(generated_b24)
    if len(saved_set) != 32 or len(low_set) != 24 or len(saved_set - low_set) != 8 or not low_set <= saved_set:
        raise RuntimeError("candidate banks violate frozen B24 subset-of-B32 contract")
    record_by_sequence = {seq: candidate for seq, candidate in zip(saved_sequences, saved_b32, strict=True)}
    low_bank = [record_by_sequence[seq] for seq in generated_b24]
    extension = [record_by_sequence[seq] for seq in generated_b32 if seq not in low_set]
    low_best = p26._winner(low_bank)
    full_best = p26._winner(saved_b32)
    strict = tuple(full_best["objective"]) < tuple(low_best["objective"])
    category_improves = int(tuple(full_best["objective"])[0] < tuple(low_best["objective"])[0])
    action_changes = int(int(full_best["sequence"][0]) != int(low_best["sequence"][0]))
    low_has_success = any(candidate["category"] == "success" for candidate in low_bank)
    extension_has_success = any(candidate["category"] == "success" for candidate in extension)
    features = p26._candidate_features(low_bank, state, previous)
    if not np.all(np.isfinite(np.asarray(features, dtype=float))):
        raise RuntimeError("non-finite model feature")
    return {"low_bank": low_bank, "extension": extension, "features": features,
            "ambiguity_gap": float(features[-4]),
            "primary_decision_impact": int(strict and action_changes),
            "best_outcome_category_improves": category_improves,
            "strict_objective_improves": int(strict),
            "first_action_changes": action_changes,
            "success_candidate_newly_available": int(not low_has_success and extension_has_success),
            "added_transitions": int(sum(int(candidate["transitions"]) for candidate in extension))}


def _decision_rows(root: Path, cfg: dict, cohort: str,
                   structural_only: bool = False) -> tuple[list[dict], dict]:
    analysis, block = cfg["analysis"], cfg["analysis"]["cohorts"][cohort]
    expected_start, layout_count = int(block["seed_start"]), int(block["layout_count"])
    records_path = root / cfg["parents"][f"{cohort}_records"]
    replicas = tuple(int(x) for x in analysis["expected_replicas"])
    episodes: dict[tuple[int, int], dict] = {}
    layouts_by_seed: dict[int, str] = {}
    with records_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            episode = json.loads(line)
            if episode.get("method") != "shifted_b32" or int(episode.get("budget", -1)) != 32:
                continue
            seed, replica = int(episode["seed"]), int(episode["replica"])
            if not expected_start <= seed < expected_start + layout_count or replica not in replicas:
                raise RuntimeError(f"{cohort} B32 episode identity outside frozen cohort: {(seed, replica)}")
            key = seed, replica
            if key in episodes:
                raise RuntimeError(f"duplicate B32 episode identity: {key}")
            old_layout = layouts_by_seed.setdefault(seed, episode["layout_sha256"])
            if old_layout != episode["layout_sha256"]:
                raise RuntimeError(f"replicas disagree on layout hash for seed {seed}")
            episodes[key] = episode

    expected_episodes = layout_count * len(replicas)
    if len(episodes) != expected_episodes or set(layouts_by_seed) != set(range(expected_start, expected_start + layout_count)):
        raise RuntimeError(f"{cohort} expected exactly {expected_episodes} complete B32 episodes")
    if len(set(layouts_by_seed.values())) != layout_count:
        raise RuntimeError(f"{cohort} layout identities are not one-to-one with layout seeds")

    rows: list[dict] = []
    for (seed, replica), episode in sorted(episodes.items()):
        decisions = episode.get("decision_diagnostics") or []
        if not decisions:
            raise RuntimeError(f"missing decision diagnostics for {(seed, replica)}")
        previous_expected = None
        for decision_index, decision in enumerate(decisions):
            actual_index = int(decision["decision_index"])
            query = int(decision["query_seed"])
            horizon = int(decision["horizon"])
            if actual_index < 0 or query != p17.query_seed(seed, actual_index, replica):
                raise RuntimeError("decision query seed does not match the frozen Phase17 derivation")
            if horizon != int(analysis["expected_horizon"]):
                raise RuntimeError("decision horizon differs from the frozen Phase27A horizon")
            if int(decision["nominal_budget"]) != int(analysis["extension_budget"]):
                raise RuntimeError("diagnostic row is not from the B32 arm")
            if int(decision["effective_budget"]) != int(analysis["extension_budget"]):
                raise RuntimeError("B32 decision has an unexpected effective candidate count")
            if decision.get("previous_plan") != previous_expected:
                raise RuntimeError("saved prior plan is inconsistent with the preceding B32 winner")

            state = decision["state_raw52"]
            if len(state) != 52 or not np.all(np.isfinite(np.asarray(state, dtype=float))):
                raise RuntimeError("invalid or non-finite deployable state_raw52")
            saved_bank = decision["candidate_bank"]
            if len(saved_bank) != int(analysis["extension_budget"]):
                raise RuntimeError("saved B32 candidate bank has incorrect length")
            saved_sequences = [_check_candidate(candidate)[0] for candidate in saved_bank]
            if len(set(saved_sequences)) != len(saved_sequences):
                raise RuntimeError("duplicate sequence in saved B32 bank")
            query_attempt_cap = 1_280_000
            generated32 = _generated_bank(horizon, 32, query, decision.get("previous_plan"), query_attempt_cap)
            generated24 = _generated_bank(horizon, 24, query, decision.get("previous_plan"), query_attempt_cap)
            compared = _compare_candidate_sets(saved_bank, generated24, generated32,
                state, decision.get("previous_plan"))
            if structural_only:
                pass
            else:
                rows.append({
                    "cohort": cohort,
                    "seed": seed,
                    "replica": replica,
                    "decision_index": actual_index,
                    "layout_sha256": episode["layout_sha256"],
                    "features": compared["features"],
                    "ambiguity_gap": compared["ambiguity_gap"],
                    **{key: compared[key] for key in ("primary_decision_impact",
                       "best_outcome_category_improves", "strict_objective_improves",
                       "first_action_changes", "success_candidate_newly_available",
                       "added_transitions")},
                })
            previous_expected = list(p26._winner(saved_bank)["sequence"])

    stats = {"episodes": len(episodes), "layouts": len(layouts_by_seed),
             "decision_rows": sum(len(ep["decision_diagnostics"]) for ep in episodes.values()),
             "layout_sha256": sorted(set(layouts_by_seed.values())),
             "all_b32_banks_reproduced": True, "all_b24_sets_subset_of_b32": True,
             "simulator_calls": 0, "candidate_rollout_evaluations": 0}
    return rows, stats


def _all_rows(root: Path, cfg: dict, structural_only: bool = False):
    rows, stats = {}, {}
    for cohort in ("phase24", "phase25"):
        rows[cohort], stats[cohort] = _decision_rows(root, cfg, cohort, structural_only)
    return rows, stats


def _layout_weights(rows: list[dict]) -> np.ndarray:
    counts: dict[str, int] = {}
    for row in rows:
        key = row["layout_sha256"]
        counts[key] = counts.get(key, 0) + 1
    n, groups = len(rows), len(counts)
    if not n or not groups:
        raise ValueError("empty training/evaluation rows")
    return np.asarray([n / (groups * counts[row["layout_sha256"]]) for row in rows], dtype=float)


def fit_ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray, l2: float,
              clip: float) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(weights):
        raise ValueError("ridge data dimensions disagree")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(weights)):
        raise ValueError("ridge data contain non-finite values")
    total_weight = float(weights.sum())
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1]) * float(l2)
    penalty[0, 0] = 0.0
    lhs = design.T @ (weights[:, None] * design) + penalty
    rhs = design.T @ (weights * y)
    beta = np.linalg.solve(lhs, rhs)
    return {"intercept": float(beta[0]), "coefficients": beta[1:].tolist(),
            "mean": mean.tolist(), "scale": scale.tolist(),
            "clip": float(clip), "l2": float(l2), "weight_sum": total_weight}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    raw = float(model["intercept"]) + ((np.asarray(x, dtype=float) - np.asarray(model["mean"]))
        / np.asarray(model["scale"])) @ np.asarray(model["coefficients"])
    return np.clip(raw, float(model["clip"]), 1.0 - float(model["clip"]))


def _fold(seed: int, cohort: str, cfg: dict) -> int:
    start = int(cfg["analysis"]["cohorts"][cohort]["seed_start"])
    count = int(cfg["analysis"]["cohorts"][cohort]["layout_count"])
    offset = int(seed) - start
    if not 0 <= offset < count:
        raise ValueError("layout seed outside configured cohort")
    return offset % int(cfg["analysis"]["folds"])


def _metrics(rows: list[dict], target: str, probabilities: np.ndarray,
             baseline_probability: float | np.ndarray) -> dict:
    by_layout: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_layout.setdefault(row["layout_sha256"], []).append(i)
    y = np.asarray([row[target] for row in rows], dtype=float)
    p = np.asarray(probabilities, dtype=float)
    base = (np.full(len(y), float(baseline_probability)) if np.ndim(baseline_probability) == 0
            else np.asarray(baseline_probability, dtype=float))
    if len(base) != len(y):
        raise ValueError("baseline prediction length disagrees with observations")
    clip = 1e-6
    row_brier = (p - y) ** 2
    base_brier = (base - y) ** 2
    row_logloss = -(y * np.log(np.clip(p, clip, 1 - clip))
                    + (1 - y) * np.log(np.clip(1 - p, clip, 1 - clip)))
    layout_brier, layout_base, layout_logloss = [], [], []
    layout_prevalence = []
    for indices in by_layout.values():
        layout_brier.append(float(np.mean(row_brier[indices])))
        layout_base.append(float(np.mean(base_brier[indices])))
        layout_logloss.append(float(np.mean(row_logloss[indices])))
        layout_prevalence.append(float(np.mean(y[indices])))
    brier, base_score = float(np.mean(layout_brier)), float(np.mean(layout_base))
    return {"layouts": len(by_layout), "decision_rows": len(rows),
            "positive_rows": int(y.sum()), "macro_prevalence": float(np.mean(layout_prevalence)),
            "macro_brier": brier, "macro_prevalence_baseline_brier": base_score,
            "macro_brier_skill": float(1 - brier / base_score) if base_score > 0 else None,
            "macro_log_loss": float(np.mean(layout_logloss))}


def _top_quartile_by_layout(rows: list[dict], target: str, probabilities: np.ndarray,
                            fraction: float) -> dict:
    by_layout: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_layout.setdefault(row["layout_sha256"], []).append(i)
    top_rates, base_rates = [], []
    for indices in by_layout.values():
        ranked = sorted(indices, key=lambda i: (-float(probabilities[i]),
            int(rows[i]["replica"]), int(rows[i]["decision_index"])))
        k = max(1, int(np.ceil(float(fraction) * len(ranked))))
        top_rates.append(float(np.mean([rows[i][target] for i in ranked[:k]])))
        base_rates.append(float(np.mean([rows[i][target] for i in indices])))
    return {"layout_hashes": sorted(by_layout), "top_rate_by_layout": top_rates,
            "base_rate_by_layout": base_rates,
            "macro_top_quartile_event_rate": float(np.mean(top_rates)),
            "macro_all_decision_event_rate": float(np.mean(base_rates)),
            "enrichment": (float(np.mean(top_rates) / np.mean(base_rates))
                           if np.mean(base_rates) > 0 else None)}


def _bootstrap_enrichment(top: dict, draws: int, seed: int) -> dict:
    top_rates = np.asarray(top["top_rate_by_layout"], dtype=float)
    base_rates = np.asarray(top["base_rate_by_layout"], dtype=float)
    if len(top_rates) == 0:
        raise ValueError("no layouts for enrichment bootstrap")
    if float(base_rates.mean()) <= 0:
        return {"lower_95": None, "median": None, "upper_95": None, "draws": int(draws)}
    rng = np.random.default_rng(seed)
    n = len(top_rates)
    indices = rng.integers(0, n, size=(int(draws), n))
    sampled_top = top_rates[indices].mean(axis=1)
    sampled_base = base_rates[indices].mean(axis=1)
    ratios = np.divide(sampled_top, sampled_base, out=np.zeros_like(sampled_top), where=sampled_base > 0)
    return {"lower_95": float(np.quantile(ratios, 0.05)),
            "median": float(np.quantile(ratios, 0.5)),
            "upper_95": float(np.quantile(ratios, 0.95)), "draws": int(draws)}


def _ambiguity_enrichment(rows: list[dict], target: str, fraction: float) -> dict:
    by_layout: dict[str, list[dict]] = {}
    for row in rows:
        by_layout.setdefault(row["layout_sha256"], []).append(row)
    top_rates, base_rates = [], []
    for layout_rows in by_layout.values():
        ranked = sorted(layout_rows, key=lambda r: (float(r["ambiguity_gap"]),
            int(r["replica"]), int(r["decision_index"])))
        k = max(1, int(np.ceil(float(fraction) * len(ranked))))
        top_rates.append(float(np.mean([r[target] for r in ranked[:k]])))
        base_rates.append(float(np.mean([r[target] for r in layout_rows])))
    base, top = float(np.mean(base_rates)), float(np.mean(top_rates))
    return {"macro_low_gap_quartile_event_rate": top,
            "macro_all_decision_event_rate": base,
            "enrichment": top / base if base > 0 else None}


def _cohort_analysis(rows: list[dict], cohort: str, cfg: dict, targets: tuple[str, ...]) -> dict:
    x = np.asarray([row["features"] for row in rows], dtype=float)
    folds = np.asarray([_fold(row["seed"], cohort, cfg) for row in rows], dtype=int)
    predictions = {target: np.zeros(len(rows), dtype=float) for target in targets}
    baseline_predictions = {target: np.zeros(len(rows), dtype=float) for target in targets}
    fold_details = []
    for fold_id in range(int(cfg["analysis"]["folds"])):
        train, test = folds != fold_id, folds == fold_id
        if not np.any(test) or not np.any(train):
            raise RuntimeError("empty grouped fold")
        train_rows = [row for row, keep in zip(rows, train, strict=True) if keep]
        weights = _layout_weights(train_rows)
        for target in targets:
            y_train = np.asarray([row[target] for row, keep in zip(rows, train, strict=True) if keep], dtype=float)
            prevalence = float(np.average(y_train, weights=weights))
            model = fit_ridge(x[train], y_train, weights,
                float(cfg["analysis"]["ridge_l2"]), float(cfg["analysis"]["probability_clip"]))
            predictions[target][test] = predict(model, x[test])
            baseline_predictions[target][test] = prevalence
        fold_details.append({"fold": fold_id, "train_layouts": len({r["layout_sha256"] for r, k in zip(rows, train, strict=True) if k}),
                             "test_layouts": len({r["layout_sha256"] for r, k in zip(rows, test, strict=True) if k})})
    metrics, top, ambiguity = {}, {}, {}
    for target in targets:
        metrics[target] = _metrics(rows, target, predictions[target], baseline_predictions[target])
        top[target] = _top_quartile_by_layout(rows, target, predictions[target],
            float(cfg["analysis"]["top_quartile_fraction"]))
        ambiguity[target] = _ambiguity_enrichment(rows, target,
            float(cfg["analysis"]["top_quartile_fraction"]))
    return {"metrics": metrics, "top_quartile": top, "ambiguity_baseline": ambiguity,
            "predictions": predictions, "folds": fold_details}


def _cross_cohort(train_rows: list[dict], test_rows: list[dict], targets: tuple[str, ...],
                  cfg: dict, seed: int) -> dict:
    x_train = np.asarray([row["features"] for row in train_rows], dtype=float)
    x_test = np.asarray([row["features"] for row in test_rows], dtype=float)
    w_train = _layout_weights(train_rows)
    output = {}
    for index, target in enumerate(targets):
        y_train = np.asarray([row[target] for row in train_rows], dtype=float)
        model = fit_ridge(x_train, y_train, w_train,
            float(cfg["analysis"]["ridge_l2"]), float(cfg["analysis"]["probability_clip"]))
        probabilities = predict(model, x_test)
        prevalence = float(np.average(y_train, weights=w_train))
        target_metrics = _metrics(test_rows, target, probabilities, prevalence)
        top = _top_quartile_by_layout(test_rows, target, probabilities,
            float(cfg["analysis"]["top_quartile_fraction"]))
        top["bootstrap"] = _bootstrap_enrichment(top,
            int(cfg["analysis"]["bootstrap_draws"]), int(seed + index))
        output[target] = {"metrics": target_metrics, "top_quartile": top,
                          "ambiguity_baseline": _ambiguity_enrichment(test_rows, target,
                              float(cfg["analysis"]["top_quartile_fraction"])),
                          "predictions": probabilities.tolist(), "model": model}
    return output


def analyze(root: Path = ROOT, cfg: dict | None = None) -> dict:
    cfg = cfg or config(root)
    _verify_inputs(root, cfg)
    rows, structural = _all_rows(root, cfg)
    if ({row["layout_sha256"] for row in rows["phase24"]}
            & {row["layout_sha256"] for row in rows["phase25"]}):
        raise RuntimeError("Phase24 and Phase25 layout identities overlap")
    targets = tuple(TARGETS)
    by_cohort = {cohort: _cohort_analysis(rows[cohort], cohort, cfg, targets)
                 for cohort in ("phase24", "phase25")}
    p24_to25 = _cross_cohort(rows["phase24"], rows["phase25"], targets, cfg,
                            int(cfg["analysis"]["bootstrap_seed"]))
    p25_to24 = _cross_cohort(rows["phase25"], rows["phase24"], targets, cfg,
                            int(cfg["analysis"]["bootstrap_seed"]) + 100)
    all_oof_scores = np.concatenate([
        by_cohort[c]["predictions"]["primary_decision_impact"] for c in ("phase24", "phase25")])
    threshold = float(np.quantile(all_oof_scores, 1.0 - float(cfg["analysis"]["extension_fraction"])))
    realized_trigger = float(np.mean(all_oof_scores >= threshold))

    gate = {}
    min_positive = int(cfg["analysis"]["minimum_primary_positives_per_cohort"])
    required_skill = float(cfg["analysis"]["minimum_primary_brier_skill_both_transfer_directions"])
    for direction, result in (("phase24_to_phase25", p24_to25), ("phase25_to_phase24", p25_to24)):
        metric = result["primary_decision_impact"]["metrics"]
        enrichment = result["primary_decision_impact"]["top_quartile"]["bootstrap"]["lower_95"]
        test_cohort = "phase25" if direction == "phase24_to_phase25" else "phase24"
        positives = sum(r["primary_decision_impact"] for r in rows[test_cohort])
        gate[direction] = {
            "positive_count_pass": positives >= min_positive,
            "brier_skill_pass": metric["macro_brier_skill"] is not None and metric["macro_brier_skill"] > required_skill,
            "top_quartile_enrichment_ci_pass": enrichment is not None and enrichment > 1.0,
        }
    passed = all(all(item.values()) for item in gate.values())

    final_model = None
    if passed:
        # Fit the deployment candidate only after the frozen feasibility gate passes.
        pooled = rows["phase24"] + rows["phase25"]
        x = np.asarray([row["features"] for row in pooled], dtype=float)
        y = np.asarray([row["primary_decision_impact"] for row in pooled], dtype=float)
        final_model = fit_ridge(x, y, _layout_weights(pooled),
            float(cfg["analysis"]["ridge_l2"]), float(cfg["analysis"]["probability_clip"]))
        final_model["feature_contract"] = {
            "state_raw52": 52, "previous_plan": 8,
            "candidate_features": 13,
            "feature_count": int(x.shape[1]),
            "excluded": ["seed", "replica", "layout_sha256", "decision_index",
                         "candidate 25-32 outcomes", "episode outcome", "future state"],
        }
        final_model["threshold_oof"] = threshold
        final_model["oof_trigger_fraction"] = realized_trigger

    prediction_rows = {}
    for cohort in ("phase24", "phase25"):
        prediction_rows[cohort] = []
        for index, row in enumerate(rows[cohort]):
            prediction_rows[cohort].append({
                "seed": row["seed"], "replica": row["replica"],
                "decision_index": row["decision_index"],
                "layout_sha256": row["layout_sha256"],
                "targets": {target: int(row[target]) for target in targets},
                "oof_probability": {target: float(by_cohort[cohort]["predictions"][target][index])
                                    for target in targets},
            })
    compact_metrics = {
        "within_cohort_grouped_oof": {c: {"metrics": by_cohort[c]["metrics"],
            "top_quartile": by_cohort[c]["top_quartile"],
            "ambiguity_baseline": by_cohort[c]["ambiguity_baseline"],
            "folds": by_cohort[c]["folds"]} for c in ("phase24", "phase25")},
        "cross_cohort": {"phase24_to_phase25": {k: {"metrics": v["metrics"],
            "top_quartile": v["top_quartile"], "ambiguity_baseline": v["ambiguity_baseline"]}
            for k, v in p24_to25.items()},
            "phase25_to_phase24": {k: {"metrics": v["metrics"],
            "top_quartile": v["top_quartile"], "ambiguity_baseline": v["ambiguity_baseline"]}
            for k, v in p25_to24.items()}},
    }
    return {
        "schema": "phase27a-search-allocation-result-v1",
        "status": "pass" if passed else "fail",
        "advancement_gate": gate,
        "structural_audit": structural,
        "target_positive_counts": {c: int(sum(r["primary_decision_impact"] for r in rows[c]))
                                   for c in ("phase24", "phase25")},
        "metrics": compact_metrics,
        "threshold": {"method": "pooled_within_cohort_grouped_oof_quantile",
                      "quantile": 1.0 - float(cfg["analysis"]["extension_fraction"]),
                      "probability_threshold": threshold,
                      "oof_trigger_fraction_ge_threshold": realized_trigger},
        "final_model": final_model,
        "oof_predictions": prediction_rows,
        "scope": {"simulator_calls": 0, "candidate_rollout_evaluations": 0,
                  "fresh_layout_generation": False, "reserve_test_d168_access": False,
                  "closed_loop": False},
    }


def _structural_audit(root: Path, cfg: dict) -> dict:
    _, structural = _all_rows(root, cfg, structural_only=True)
    phase24_layouts = set(structural["phase24"]["layout_sha256"])
    phase25_layouts = set(structural["phase25"]["layout_sha256"])
    if phase24_layouts & phase25_layouts:
        raise RuntimeError("Phase24 and Phase25 layout identities overlap")
    structural["cross_cohort_layout_overlap"] = 0
    for cohort in ("phase24", "phase25"):
        structural[cohort].pop("layout_sha256")
    return structural


def freeze(root: Path = ROOT) -> dict:
    cfg = config(root)
    if cfg["status"] != "frozen-historical-only-pending-lock":
        raise RuntimeError("Stage27A config must be frozen before creating its lock")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit the Stage27A protocol, config, runner, tests, and correction before lock")
    hashes = _verify_inputs(root, cfg)
    structure = _structural_audit(root, cfg)
    lock_path = root / cfg["outputs"]["execution_lock"]
    run_path = root / cfg["outputs"]["run_dir"]
    if lock_path.exists() or run_path.exists():
        raise FileExistsError("refusing to overwrite a Stage27A lock or result directory")
    lock = {
        "schema": "phase27a-search-allocation-lock-v1",
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "hashes": hashes,
        "parent_phase26_lock_sha256": cfg["parents"]["phase26_lock_sha256"],
        "input_records": {c: cfg["parents"][f"{c}_records_sha256"] for c in ("phase24", "phase25")},
        "structural_audit": structure,
        "simulator_calls": 0,
        "candidate_rollout_evaluations": 0,
        "fresh_layout_generation": False,
        "reserve_test_d168_access": False,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"lock_path": lock_path.relative_to(root).as_posix(), "lock_sha256": sha256(lock_path),
            "source_commit": lock["source_commit"], "structural_audit": structure}


def _verify_committed_lock(root: Path, lock_path: Path, source_commit: str) -> None:
    relative_lock = lock_path.relative_to(root).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{relative_lock}"], cwd=root)
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("Stage27A execution lock must be committed unchanged")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=root,
        capture_output=True, check=False)
    if ancestor.returncode != 0:
        raise RuntimeError("Stage27A source commit is not an ancestor of the committed lock")


def _locked(root: Path, cfg: dict) -> dict:
    lock_path = root / cfg["outputs"]["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase27a-search-allocation-lock-v1":
        raise RuntimeError("invalid Stage27A lock schema")
    _verify_committed_lock(root, lock_path, lock["source_commit"])
    for path, digest in lock["hashes"].items():
        if sha256(root / path) != digest:
            raise RuntimeError(f"locked Stage27A dependency changed: {path}")
    if _verify_inputs(root, cfg) != lock["hashes"]:
        raise RuntimeError("Stage27A dependency hash set differs from lock")
    return lock


def run(root: Path = ROOT) -> dict:
    cfg = config(root)
    _locked(root, cfg)
    run_dir = root / cfg["outputs"]["run_dir"]
    if run_dir.exists():
        raise FileExistsError("refusing to overwrite Stage27A outputs")
    started = time.perf_counter()
    result = analyze(root, cfg)
    run_dir.mkdir(parents=True)
    result_path = run_dir / "result.json"
    model_path = run_dir / "allocation_model.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["final_model"] is not None:
        model_path.write_text(json.dumps(result["final_model"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "phase27a-search-allocation-manifest-v1",
        "execution_lock_sha256": sha256(root / cfg["outputs"]["execution_lock"]),
        "result_sha256": sha256(result_path),
        "allocation_model_sha256": sha256(model_path) if model_path.exists() else None,
        "runtime_seconds": float(time.perf_counter() - started),
        "python": platform.python_version(), "numpy": np.__version__,
        "simulator_calls": 0, "candidate_rollout_evaluations": 0,
        "fresh_layout_generation": False, "reserve_test_d168_access": False,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def audit(root: Path = ROOT) -> dict:
    cfg = config(root)
    lock = _locked(root, cfg)
    run_dir = root / cfg["outputs"]["run_dir"]
    result_path, model_path, manifest_path = (run_dir / "result.json", run_dir / "allocation_model.json",
                                               run_dir / "manifest.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["execution_lock_sha256"] != sha256(root / cfg["outputs"]["execution_lock"]):
        raise RuntimeError("Stage27A manifest lock mismatch")
    if manifest["result_sha256"] != sha256(result_path):
        raise RuntimeError("Stage27A output hash mismatch")
    if result["final_model"] is not None:
        if not model_path.exists() or manifest["allocation_model_sha256"] != sha256(model_path):
            raise RuntimeError("Stage27A model artifact hash mismatch")
    elif manifest["allocation_model_sha256"] is not None or model_path.exists():
        raise RuntimeError("failed feasibility gate must not emit a deployment model")
    reproduced = analyze(root, cfg)
    if canonical_json(result) != canonical_json(reproduced):
        raise RuntimeError("Stage27A result does not reproduce from locked historical inputs")
    if result["final_model"] is not None and canonical_json(
            json.loads(model_path.read_text(encoding="utf-8"))) != canonical_json(result["final_model"]):
        raise RuntimeError("Stage27A model artifact differs from audited result")
    if lock["simulator_calls"] or lock["candidate_rollout_evaluations"]:
        raise RuntimeError("Stage27A lock must record zero simulator/candidate evaluations")
    return {"audit": "pass", "result_sha256": sha256(result_path),
            "model_sha256": sha256(model_path) if model_path.exists() else None,
            "manifest_sha256": sha256(manifest_path),
            "simulator_calls": 0, "candidate_rollout_evaluations": 0,
            "reserve_test_d168_access": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run", "audit"))
    args = parser.parse_args()
    if args.command == "freeze":
        value = freeze()
    elif args.command == "run":
        value = run()
    else:
        value = audit()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
