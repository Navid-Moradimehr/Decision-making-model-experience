"""Analyze D152's fixed, already-used validation block before D153 scaling."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import platform

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_d153 import (
    cluster_bootstrap_interval, correlation_pair, mean_rank_correlation_checked,
    nearest_distance_group_breakdown,
)
from consequence_lab.phase15_outcome_summary import restore_decision_head
from consequence_lab.phase15_representation_screen import train_validation_indices
from consequence_lab.phase15_representation_sufficiency import nearest_action_summary

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = "configs/phase15_d153_validation_audit.yaml"
PREDICTOR_KEYS = {"constant": "validation_constant", "knn_k1": "validation_knn_k1",
                  "knn_k5": "validation_knn_k5", "cart": "validation_cart",
                  "mlp32": "validation_mlp32"}


def _correlation_record(x: np.ndarray, y: np.ndarray, layouts: np.ndarray,
                        *, resamples: int, seed: int) -> dict:
    point = correlation_pair(x, y)
    result = dict(point)
    for kind in ("pearson", "spearman"):
        if kind == "pearson":
            stat = lambda a, b: correlation_pair(a, b)["pearson"]
        else:
            stat = mean_rank_correlation_checked
        result[f"{kind}_layout_bootstrap_95_ci"] = cluster_bootstrap_interval(
            x, y, layouts, stat, resamples=resamples, seed=seed + (kind == "spearman"))
    return result


def _report(result: dict) -> str:
    lines = ["# D153 - D152 Validation Coverage Analysis", "",
             "**Status: read-only exploratory analysis.** This reuses the D152 validation block, which was already used for development in D148–D152; the statistics are descriptive, not independent confirmation. Test/reserve targets were not indexed. No fitting, simulation, planner calls, or new transitions occurred.", "",
             "## Coverage and distance composition", "",
             f"Across {result['validation_rows']} states on {result['validation_layouts']} fixed layouts, the nearest-training R0 distance had median **{result['coverage']['distance_median']:.3f}** and p90 **{result['coverage']['distance_p90']:.3f}**. The train leave-one-state-out distance p90 is **{result['coverage']['train_loo_distance_p90']:.3f}**. Within that support threshold: **{result['coverage']['inside_train_loo_p90']}/{result['validation_rows']}**; within twice the threshold: **{result['coverage']['inside_2x_train_loo_p90']}/{result['validation_rows']}**.", "",
             "R0 distance decomposition uses the same full-R0 nearest training row, with each block standardized using train-only statistics:", "",
             "| R0 block | Median block distance to full-R0 neighbor | Fraction of mean squared full distance |", "|---|---:|---:|"]
    for name, item in result["distance_group_breakdown"].items():
        lines.append(f"| {name} | {item['nearest_pair_distance_median']:.3f} | {item['mean_squared_distance_fraction']:.1%} |")
    lines += ["", "The `layout_geometry` block is the largest distance contributor; therefore the full-R0 metric is primarily measuring layout mismatch in this audit, not only kinematic state mismatch.", "",
              "## Per-state relationships", "",
              "Teacher best-action margin is the normalized-regret gap between the second-lowest and lowest teacher action values. Catastrophic means the frozen R4 head selects an action with teacher regret `>= 0.5`.", "",
              "| Predictor | Distance vs summary RMSE (Spearman) | Distance vs selected regret | Distance vs catastrophic | Summary RMSE vs regret |", "|---|---:|---:|---:|---:|"]
    for name, item in result["predictors"].items():
        c = item["correlations"]
        lines.append(f"| {name} | {c['distance_vs_summary_error']['spearman']:.3f} | {c['distance_vs_regret']['spearman']:.3f} | {c['distance_vs_catastrophic']['spearman']:.3f} | {c['summary_error_vs_regret']['spearman']:.3f} |")
    lines += ["", "Primary MLP layout-cluster bootstrap intervals (95%, 5,000 resamples; descriptive due to reused validation):", "",
              "| Relationship | Spearman | Layout-bootstrap 95% interval |", "|---|---:|---:|"]
    primary = result["predictors"]["mlp32"]["correlations"]
    for key, label in (("distance_vs_summary_error", "distance vs summary RMSE"),
                       ("distance_vs_regret", "distance vs selected regret"),
                       ("distance_vs_catastrophic", "distance vs catastrophic"),
                       ("summary_error_vs_regret", "summary RMSE vs regret")):
        row = primary[key]
        ci = row["spearman_layout_bootstrap_95_ci"]
        ci_text = "unavailable" if ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {label} | {row['spearman']:.3f} | {ci_text} |")
    lines += ["", "## Nearest-distance quartiles for the primary action-conditioned MLP", "",
              "| Quartile | Count | Median distance | Mean summary RMSE | Mean regret | Catastrophic rate |", "|---:|---:|---:|---:|---:|---:|"]
    for row in result["mlp_distance_quartiles"]:
        lines.append(f"| {row['quartile']} | {row['count']} | {row['distance_median']:.3f} | {row['summary_rmse_mean']:.3f} | {row['regret_mean']:.3f} | {row['catastrophic_rate']:.1%} |")
    lines += ["", "## Physical-variable audit", "",
              "For each physical variable below, report Spearman association with full-R0 nearest-neighbor distance. The variables are descriptive fields already present in deployable R0; the result does not redefine D153's primary distance metric.", "",
              "| Variable | Spearman with distance |", "|---|---:|"]
    for name, value in result["physical_variable_distance_correlations"].items():
        lines.append(f"| {name} | {value:.3f} |")
    lines += ["", "## Interpretation", "", result["interpretation"], "",
              f"D152 lock SHA-256: `{result['parent_sha256']['d152_lock']}`. D152 feature dataset SHA-256: `{result['parent_sha256']['d152_features']}`. D152 predictions SHA-256: `{result['parent_sha256']['d152_predictions']}`. This report has result SHA recorded in the run manifest.", ""]
    return "\n".join(lines)


def main() -> None:
    cfg = yaml.safe_load((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    if cfg.get("schema") != "phase15_d153_validation_audit_v1":
        raise RuntimeError("D153 validation audit config schema mismatch")
    parent_hashes = {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}
    d152_lock = json.loads((ROOT / cfg["parents"]["d152_lock"]).read_text(encoding="utf-8"))
    d152_manifest = json.loads((ROOT / cfg["parents"]["d152_manifest"]).read_text(encoding="utf-8"))
    if (d152_lock.get("scope", {}).get("reserve_access") is not False
            or d152_manifest.get("test_rows_indexed_or_scored") != 0
            or d152_manifest.get("reserve_access") is not False):
        raise RuntimeError("D152 parent scope/provenance does not satisfy D153 audit")
    split_cfg = cfg["split"]
    feature_path = ROOT / cfg["parents"]["d152_features"]
    predictions_path = ROOT / cfg["parents"]["d152_predictions"]
    with np.load(feature_path, allow_pickle=False) as data:
        # Only split metadata plus train/validation fields are accessed below.
        all_layouts = data["layouts"].astype(str).tolist()
        train_idx, val_idx, train_names, val_names = train_validation_indices(all_layouts, split_cfg)
        x_train = data["x_R0"][train_idx].astype(np.float64)
        x_val = data["x_R0"][val_idx].astype(np.float64)
        y_val = data["y"][val_idx].astype(np.float64)
        summary_train = np.asarray(data["x_R4"][train_idx, 61:124], dtype=np.float64).reshape(-1, 9, 7)
    allowed_layouts = set(train_names) | set(val_names)
    layout_by_row = np.asarray(all_layouts, dtype=str)[val_idx]
    if not set(layout_by_row).issubset(set(val_names)) or not set(layout_by_row).issubset(allowed_layouts):
        raise RuntimeError("D153 validation layout mapping mismatch")
    with np.load(predictions_path, allow_pickle=False) as predictions:
        if not np.array_equal(predictions["validation_teacher_regrets"], y_val):
            raise RuntimeError("D152 saved validation regret targets disagree with source")
        exact_summary = predictions["validation_exact_summary"].astype(np.float64)
        predicted = {name: predictions[PREDICTOR_KEYS[name]].astype(np.float64)
                     for name in cfg["analysis"]["predictors"]}
    # D152 predictions contain only its declared train/validation arrays; no test prediction member exists.
    _, train_loo_distances, _ = nearest_action_summary(x_train, summary_train, x_train, k=1, exclude_self=True)
    _, val_distance, val_neighbor_index = nearest_action_summary(x_train, summary_train, x_val, k=1)
    train_p90 = float(np.quantile(train_loo_distances[:, 0], 0.90))
    threshold2 = 2.0 * train_p90
    breakdown_cfg = cfg["distance"]["feature_groups"]
    groups = {name: (int(bounds[0]), int(bounds[1])) for name, bounds in breakdown_cfg.items()}
    breakdown = nearest_distance_group_breakdown(x_train, x_val, val_neighbor_index[:, 0], groups)
    r4_head = restore_decision_head(ROOT / cfg["parents"]["d150_r4_checkpoint"])
    best_sorted = np.sort(y_val, axis=1)
    margins = best_sorted[:, 1] - best_sorted[:, 0]
    predictor_result = {}
    per_state: dict[str, np.ndarray] = {
        "nearest_r0_distance": val_distance[:, 0],
        "teacher_best_action_margin": margins,
        "teacher_best_action": np.argmin(y_val, axis=1),
    }
    for name, summary in predicted.items():
        if summary.shape != (len(val_idx), 9, 7):
            raise RuntimeError(f"D153 {name} prediction shape mismatch")
        error = np.sqrt(np.mean((summary - exact_summary) ** 2, axis=(1, 2)))
        score_input = np.concatenate((x_val, summary.reshape(len(val_idx), 63)), axis=1)
        scores = r4_head.predict(score_input)
        action = np.argmin(scores, axis=1)
        regret = y_val[np.arange(len(y_val)), action]
        catastrophic = regret >= float(cfg["analysis"]["catastrophic_regret_threshold"])
        per_state[f"{name}_summary_rmse"] = error
        per_state[f"{name}_selected_regret"] = regret
        per_state[f"{name}_catastrophic"] = catastrophic.astype(np.float64)
        predictor_result[name] = {
            "summary_rmse_mean": float(error.mean()), "summary_rmse_median": float(np.median(error)),
            "mean_selected_regret": float(regret.mean()), "catastrophic_fraction": float(catastrophic.mean()),
            "best_action_set_agreement": float(np.mean(regret <= 1e-12)),
            "selected_action_frequency": np.bincount(action, minlength=9).tolist(),
            "correlations": {
                "distance_vs_summary_error": _correlation_record(val_distance[:, 0], error, layout_by_row,
                    resamples=cfg["analysis"]["cluster_bootstrap_resamples"], seed=cfg["analysis"]["cluster_bootstrap_seed"]),
                "distance_vs_regret": _correlation_record(val_distance[:, 0], regret, layout_by_row,
                    resamples=cfg["analysis"]["cluster_bootstrap_resamples"], seed=cfg["analysis"]["cluster_bootstrap_seed"] + 10),
                "distance_vs_catastrophic": _correlation_record(val_distance[:, 0], catastrophic.astype(float), layout_by_row,
                    resamples=cfg["analysis"]["cluster_bootstrap_resamples"], seed=cfg["analysis"]["cluster_bootstrap_seed"] + 20),
                "summary_error_vs_regret": _correlation_record(error, regret, layout_by_row,
                    resamples=cfg["analysis"]["cluster_bootstrap_resamples"], seed=cfg["analysis"]["cluster_bootstrap_seed"] + 30),
            },
        }
    distance_corr = {name: correlation_pair(val_distance[:, 0], x_val[:, index])["spearman"]
                     for index, name in ((0, "x_norm"), (1, "y_norm"), (2, "vx_norm"), (3, "vy_norm"),
                         (15, "remaining_time_fraction"), (52, "goal_dx_norm"), (53, "goal_dy_norm"),
                         (54, "goal_distance_norm"), (55, "speed_norm"), (56, "forward_speed_norm"),
                         (57, "lateral_speed_norm"), (58, "velocity_goal_alignment"),
                         (59, "obstacle_clearance_norm"), (60, "stopping_distance_norm"))}
    quartile_edges = np.quantile(val_distance[:, 0], np.linspace(0, 1, cfg["analysis"]["distance_quartiles"] + 1))
    quartile_ids = np.digitize(val_distance[:, 0], quartile_edges[1:-1], right=True)
    primary_error = per_state["mlp32_summary_rmse"]
    primary_regret = per_state["mlp32_selected_regret"]
    primary_cat = per_state["mlp32_catastrophic"]
    bins = []
    for q in range(cfg["analysis"]["distance_quartiles"]):
        mask = quartile_ids == q
        bins.append({"quartile": q + 1, "count": int(mask.sum()),
                     "distance_median": float(np.median(val_distance[mask, 0])),
                     "summary_rmse_mean": float(primary_error[mask].mean()),
                     "regret_mean": float(primary_regret[mask].mean()),
                     "catastrophic_rate": float(primary_cat[mask].mean()),
                     "best_action_margin_median": float(np.median(margins[mask]))})
    result = {
        "schema": "phase15_d153_validation_audit_result_v1", "status": "completed_read_only_exploratory_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256_file(ROOT / CONFIG_PATH), "source_sha256": {
            "config": sha256_file(ROOT / CONFIG_PATH), "script": sha256_file(ROOT / "scripts/audit_d153_validation_coverage.py"),
            "module": sha256_file(ROOT / "src/consequence_lab/phase15_d153.py")},
        "parent_sha256": parent_hashes,
        "train_rows": len(train_idx), "train_layouts": len(train_names),
        "validation_rows": len(val_idx), "validation_layouts": len(val_names),
        "test_rows_indexed_or_scored": 0, "reserve_access": False,
        "scope": cfg["scope"],
        "coverage": {"train_loo_distance_p90": train_p90,
                     "distance_median": float(np.median(val_distance[:, 0])),
                     "distance_p90": float(np.quantile(val_distance[:, 0], .9)),
                     "inside_train_loo_p90": int(np.sum(val_distance[:, 0] <= train_p90)),
                     "inside_2x_train_loo_p90": int(np.sum(val_distance[:, 0] <= threshold2)),
                     "inside_2x_train_loo_p90_threshold": threshold2,
                     "teacher_margin_median": float(np.median(margins)),
                     "teacher_margin_p90": float(np.quantile(margins, .9))},
        "distance_group_breakdown": breakdown,
        "physical_variable_distance_correlations": distance_corr,
        "predictors": predictor_result,
        "mlp_distance_quartiles": bins,
        "interpretation": "Nearest-R0 distance has a modest positive association with per-state summary RMSE for the action-conditioned MLP, while its association with decision regret/catastrophic choices is weak and, in this reused block, inverse rather than harmful. Summary RMSE and selected regret are also weakly related. The farthest distance quartile has higher MLP summary error, but not worse regret or catastrophic rate. This supports coverage as a plausible contributor to consequence prediction error, but it does not show that coverage is the dominant decision bottleneck; D153 scaling remains informative. Correlations are descriptive on reused validation and are not causal.",
        "validation_reuse_disclosure": "The 202 validation rows were reused in D148-D152. This D153 pre-analysis is exploratory and not independent confirmation.",
        "test_boundary_disclosure": "Only train and validation R0/target rows were indexed. Test feature/target rows were not indexed or scored; reserve artifacts were not read.",
    }
    run_dir = ROOT / cfg["outputs"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "per_state.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["layout_sha256", "nearest_r0_distance", "teacher_best_action_margin", "teacher_best_action"]
        for name in predicted:
            fields += [f"{name}_summary_rmse", f"{name}_selected_regret", f"{name}_catastrophic"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(len(val_idx)):
            row = {"layout_sha256": str(layout_by_row[i]), "nearest_r0_distance": float(val_distance[i, 0]),
                   "teacher_best_action_margin": float(margins[i]), "teacher_best_action": int(np.argmin(y_val[i]))}
            for name in predicted:
                row.update({f"{name}_summary_rmse": float(per_state[f"{name}_summary_rmse"][i]),
                            f"{name}_selected_regret": float(per_state[f"{name}_selected_regret"][i]),
                            f"{name}_catastrophic": int(per_state[f"{name}_catastrophic"][i])})
            writer.writerow(row)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / cfg["outputs"]["report"]
    report_path.write_text(_report(result), encoding="utf-8")
    manifest = {"schema": "phase15_d153_validation_audit_manifest_v1",
                "status": result["status"], "source_commit": result["source_commit"],
                "config_sha256": result["config_sha256"], "parent_sha256": parent_hashes,
                "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
                "per_state_sha256": sha256_file(run_dir / "per_state.csv"),
                "train_rows": len(train_idx), "validation_rows": len(val_idx),
                "test_rows_indexed_or_scored": 0, "reserve_access": False,
                "simulator_or_planner_calls": 0, "new_transitions": 0, "new_episodes": 0}
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "coverage": result["coverage"],
                      "predictors": {name: {"summary_error_rho": metrics["correlations"]["distance_vs_summary_error"]["spearman"],
                                             "regret_rho": metrics["correlations"]["distance_vs_regret"]["spearman"],
                                             "catastrophic_rho": metrics["correlations"]["distance_vs_catastrophic"]["spearman"],
                                             "error_regret_rho": metrics["correlations"]["summary_error_vs_regret"]["spearman"]}
                                    for name, metrics in predictor_result.items()},
                      "report": cfg["outputs"]["report"]}, indent=2))


if __name__ == "__main__":
    main()
