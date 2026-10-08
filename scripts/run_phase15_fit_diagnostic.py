"""Replay exactly the failed Phase15 v2 32-state R0/regression fit cell."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_data import layout_split
from consequence_lab.phase15_learning import TinyRegretMLP, decision_metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/phase15_offline_v2_fit_diagnostic.yaml"
LOCK = "experiments/phase15_offline_v2_fit_diagnostic.lock.json"


def _validate(config_path: str, lock_path: str) -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    lock = json.loads((ROOT / lock_path).read_text(encoding="utf-8"))
    if lock.get("schema") != "phase15_v2_failed_cell_replay_lock_v1" or lock.get("status") != "authorized_single_diagnostic_replay":
        raise RuntimeError("diagnostic lock does not authorize this replay")
    if lock.get("config_path") != config_path or lock.get("config_sha256") != sha256_file(ROOT / config_path):
        raise RuntimeError("diagnostic config differs from lock")
    if lock.get("source_sha256") != {p: sha256_file(ROOT / p) for p in lock["source_sha256"]}:
        raise RuntimeError("locked diagnostic source changed")
    if lock.get("parent_sha256") != {p: sha256_file(ROOT / config["parent"][p]) for p in lock["parent_sha256"]}:
        raise RuntimeError("Phase15 v2 parent artifact changed")
    if sha256_file(ROOT / config["parent"]["feature_dataset"]) != lock["feature_dataset_sha256"]:
        raise RuntimeError("feature dataset hash mismatch")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != lock["source_commit"]:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"], cwd=ROOT)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("replay requires a clean tree after lock commit")
    scope = lock["scope"]
    for key in ("compute_heldout_metrics", "heldout_rows_for_model_selection", "new_transitions", "new_episodes",
                "exact_planner_calls", "closed_loop", "dagger", "world_model", "reserve_access"):
        if scope.get(key) is not False:
            raise RuntimeError(f"scope violation: {key}")
    if scope.get("exact_parent_cell_only") is not True:
        raise RuntimeError("not authorized for a single-cell replay")
    return config, lock


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--lock", default=LOCK)
    args = parser.parse_args()
    config, lock = _validate(args.config, args.lock)
    run_dir = ROOT / config["outputs"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source_npz = ROOT / config["parent"]["feature_dataset"]
    with np.load(source_npz, allow_pickle=False) as data:
        # The archive contains previously materialized other split rows. This replay indexes only the
        # frozen 32 train rows and computes no held-out metric.
        layouts_all = data["layouts"].astype(str)
        layout_ids = layout_split(layouts_all.tolist())
        train_layouts = set(layout_ids["train"])
        reproduced_ids = [i for i, layout in enumerate(layouts_all) if layout in train_layouts][:32]
        ids = config["cell"]["row_indices"]
        if reproduced_ids != ids:
            raise RuntimeError("frozen row indices no longer match original v2 ladder ordering")
        x = np.asarray(data["x_R0"][ids], dtype=np.float64)
        y = np.asarray(data["y"][ids], dtype=np.float64)
        layouts = layouts_all[ids]
    x_hash = hashlib.sha256(x.tobytes()).hexdigest().upper()
    y_hash = hashlib.sha256(y.tobytes()).hexdigest().upper()
    if x_hash != config["cell"]["input_sha256"] or y_hash != config["cell"]["target_sha256"]:
        raise RuntimeError("replay fitting rows differ from frozen hashes")
    training = config["training"]
    model = TinyRegretMLP(
        seed=training["seed"], objective=config["cell"]["objective"],
        max_epochs=training["max_epochs"], patience=training["patience"],
        learning_rate=training["learning_rate"], l2=training["l2"], record_history=True,
    ).fit(x, y, layouts, x, y, layouts)
    predictions = model.predict(x)
    metrics = decision_metrics(y, predictions, layouts)
    if len(model.history) != model.epochs_run:
        raise RuntimeError("training history length differs from epochs run")
    sorted_targets = np.sort(y, axis=1)
    row_details = []
    for i, row_id in enumerate(ids):
        pred_action = int(np.argmin(predictions[i]))
        best_set = np.flatnonzero(np.isclose(y[i], np.min(y[i]), rtol=0.0, atol=1e-12)).tolist()
        row_details.append({
            "feature_row_index": int(row_id), "layout_hash": str(layouts[i]),
            "target_normalized_regret": y[i].tolist(),
            "predicted_normalized_regret": predictions[i].tolist(),
            "teacher_best_action_set": [int(action) for action in best_set],
            "predicted_action": pred_action,
            "selected_teacher_regret": float(y[i, pred_action]),
            "best_vs_second_best_margin": float(sorted_targets[i, 1] - sorted_targets[i, 0]),
            "best_action_set_match": bool(pred_action in best_set),
        })
    expected = config["expected_original"]
    reproduced = (
        model.best_epoch == expected["best_epoch"]
        and metrics["best_action_set_agreement"] == expected["best_action_set_agreement"]
        and metrics["mean_normalized_regret_state_micro"] == expected["mean_normalized_regret_state_micro"]
    )
    np.savez_compressed(run_dir / "fit_arrays.npz", row_indices=np.asarray(ids), layouts=layouts,
                        targets=y, predictions=predictions,
                        selected_actions=np.asarray(metrics["selected_actions"]),
                        selected_regret=np.asarray(metrics["selected_regret"]))
    (run_dir / "training_history.json").write_text(json.dumps(model.history, indent=2) + "\n", encoding="utf-8")
    details = {
        "schema": "phase15_v2_failed_cell_fit_diagnostic_v1",
        "status": "reproduced" if reproduced else "non_reproducing_stop",
        "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / args.lock),
        "parent": {key: sha256_file(ROOT / config["parent"][key]) for key in config["parent"]},
        "dataset_sha256": x_hash,
        "target_sha256": y_hash,
        "cell": config["cell"], "training": training,
        "row_details": row_details,
        "metrics": metrics,
        "optimization": {
            "best_epoch": model.best_epoch, "epochs_run": model.epochs_run,
            "best_validation_loss": model.validation_loss,
            "best_validation_decision_regret": model.validation_decision_regret,
            "parameter_count": model.parameter_count,
            "history_rows": len(model.history),
            "last_history_row": model.history[-1],
        },
        "reproduction_comparison": {
            "expected": expected,
            "matches_all_frozen_summaries": reproduced,
            "note": "A mismatch invalidates causal interpretation; stop after reporting non-reproduction.",
        },
        "scope": {"heldout_metrics_computed": False, "heldout_for_selection": False,
                  "new_transitions": 0, "new_episodes": 0, "exact_planner_calls": 0,
                  "closed_loop": False, "dagger": False, "world_model": False, "reserve_access": False},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "wall_seconds": time.perf_counter() - start,
    }
    (run_dir / "diagnostic.json").write_text(json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / config["outputs"]["report"]).write_text(_render(details), encoding="utf-8")
    output_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    if details["wall_seconds"] > config["resource_caps"]["wall_seconds"]:
        raise RuntimeError("replay exceeded the authorized 3-hour wall-time cap")
    if output_bytes > config["resource_caps"]["output_bytes"]:
        raise RuntimeError("replay artifacts exceeded the authorized 20-MiB output cap")
    print(json.dumps({"status": details["status"], "metrics": metrics,
                      "run_dir": str(run_dir), "output_bytes": output_bytes,
                      "report": config["outputs"]["report"]}, indent=2))
    if not reproduced:
        raise SystemExit("replay did not reproduce the frozen v2 summary; no further analysis authorized")


def _render(result: dict) -> str:
    m = result["metrics"]
    lines = [
        "# Phase15 v2 failed-cell diagnostic replay result", "",
        f"**Status: {result['status']}**. This is one deterministic replay of the failed 32-row R0/regression fitting cell.", "",
        "## Scope", "",
        "No held-out metric or selection, no reserve access, no new transition/episode/planner call, no closed-loop evaluation, no DAgger, and no world model were run. The feature NPZ contains previously materialized split rows; the runner indexed only the locked 32 training rows.", "",
        "## Replay result", "",
        f"- Best epoch: {result['optimization']['best_epoch']} (frozen original: {result['reproduction_comparison']['expected']['best_epoch']})",
        f"- Best-action-set agreement: {m['best_action_set_agreement']:.8f} (frozen original: {result['reproduction_comparison']['expected']['best_action_set_agreement']:.8f})",
        f"- Mean normalized fitting regret: {m['mean_normalized_regret_state_micro']:.8f} (frozen original: {result['reproduction_comparison']['expected']['mean_normalized_regret_state_micro']:.8f})",
        f"- Mean pairwise ranking accuracy: {m['pairwise_ranking_accuracy_macro']:.6f}",
        f"- Training epochs recorded: {result['optimization']['history_rows']}",
        f"- Training parameter count: {result['optimization']['parameter_count']}",
        f"- Runtime: {result['wall_seconds']:.3f}s", "",
        "## Diagnostic artifacts", "",
        "`artifacts/runs/phase15_offline_v2_fit_diagnostic_v1/fit_arrays.npz` contains the row-aligned targets/predictions and selected actions; `training_history.json` contains per-epoch losses, gradient norms, checkpoint updates, and staleness; `diagnostic.json` contains provenance and per-row decision details.", "",
        "## Interpretation boundary", "",
        "This replay can localize which training row(s) account for the pre-existing one-row best-action mismatch and whether optimization steadily approached that boundary. It cannot establish whether representation, objective, generalization, or closed-loop distribution shift is causal. No follow-on experiment is initiated by this result.", "",
        f"Feature rows SHA-256: `{result['dataset_sha256']}`; target rows SHA-256: `{result['target_sha256']}`; replay lock SHA-256: `{result['lock_sha256']}`.", "",
    ]
    for row in result["row_details"]:
        if not row["best_action_set_match"]:
            lines.extend(["## Fitting disagreement row", "", f"- Row: `{row['feature_row_index']}`; layout `{row['layout_hash']}`.",
                          f"- Teacher best set: `{row['teacher_best_action_set']}`; selected action: `{row['predicted_action']}`.",
                          f"- Selected teacher regret: `{row['selected_teacher_regret']:.8f}`; best/second margin: `{row['best_vs_second_best_margin']:.8f}`.",
                          f"- Teacher regret vector: `{row['target_normalized_regret']}`.",
                          f"- Predicted output vector: `{row['predicted_normalized_regret']}`.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
