"""Re-score preserved D154 v2 fits after two post-fit serialization aborts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_decision_aligned import (
    D154Error, ListwiseRegretMLP, _load_data, _model_metrics, _read_json,
    _reference_metrics, validate_d153_parent_manifest,
)
from consequence_lab.phase15_learning import TinyRegretMLP
from consequence_lab.phase15_outcome_summary import restore_decision_head
from consequence_lab.phase15_representation_sufficiency import component_ablation

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d154_finalize_v3.yaml"
LOCK = "experiments/phase15_d154_finalize_v3.lock.json"
V2_CONFIG = "configs/phase15_d154_decision_aligned_v2.yaml"
SOURCE_FILES = (
    CONFIG, "reports/phase15_d154_finalize_amendment_v3.md",
    "src/consequence_lab/phase15_d154_finalize.py",
    "scripts/freeze_phase15_d154_finalize_lock.py",
    "scripts/finalize_phase15_d154_v3.py",
    "tests/test_phase15_d154_finalize.py",
)
WEIGHT_FIELDS = ("x_mean", "x_std", "w1", "b1", "w2", "b2", "w3", "b3")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _v2_cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / V2_CONFIG).read_text(encoding="utf-8"))


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D154Error("commit D154 finalizer, amendment, tests, and config before locking")
    if cfg.get("schema") != "phase15_d154_finalize_v3":
        raise D154Error("D154 finalizer config schema mismatch")
    if cfg["scope"].get("reuse_existing_student_checkpoints_only") is not True:
        raise D154Error("D154 finalizer must be limited to existing checkpoints")
    if any(value is not False for key, value in cfg["scope"].items()
           if key != "reuse_existing_student_checkpoints_only"):
        raise D154Error("D154 finalizer scope must prohibit retraining and all new interaction")
    v2_lock_path = ROOT / cfg["parents"]["d154_v2_lock"]
    v2_lock = _read_json(v2_lock_path)
    v2_cfg = _v2_cfg()
    if (v2_lock.get("schema") != "phase15_d154_decision_aligned_lock_v2"
            or v2_lock.get("config_sha256") != sha256_file(ROOT / V2_CONFIG)):
        raise D154Error("D154 v2 lock/config identity failed")
    if v2_lock.get("source_sha256") != {
            path: sha256_file(ROOT / path) for path in v2_lock.get("source_sha256", {})}:
        raise D154Error("D154 v2 locked sources have changed")
    if v2_lock.get("parent_sha256") != {
            name: sha256_file(ROOT / path) for name, path in v2_cfg["parents"].items()}:
        raise D154Error("D154 v2 parent artifacts have changed")
    d153_manifest = _read_json(ROOT / v2_cfg["parents"]["d153_manifest"])
    validate_d153_parent_manifest(d153_manifest)
    run_dir = ROOT / v2_cfg["outputs"]["run_dir"]
    expected_names = {
        "memorization.json", "T1_regret_regression.npz", "T2_margin_weighted_pairwise.npz",
        "T3_listwise_tau_0.125.npz", "T3_listwise_tau_0.25.npz", "T3_listwise_tau_0.5.npz",
    }
    if not run_dir.is_dir() or {path.name for path in run_dir.iterdir()} != expected_names:
        raise D154Error("preserved D154 v2 partial run has unexpected/missing files")
    aborted_finalize_dir = ROOT / "artifacts/runs/phase15_d154_finalize_v2"
    if not aborted_finalize_dir.is_dir() or any(aborted_finalize_dir.iterdir()):
        raise D154Error("preserved D154 v2 finalizer output is not the expected empty directory")
    mem = _read_json(run_dir / "memorization.json")
    if not mem.get("all_passed") or mem.get("states") != 8:
        raise D154Error("preserved D154 v2 memorization gate is not passing")
    for path in (ROOT / cfg["outputs"]["run_dir"], ROOT / cfg["outputs"]["report"]):
        if path.exists():
            raise D154Error(f"refusing to overwrite D154 finalization output: {path.relative_to(ROOT)}")
    parents = {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}
    return {
        "schema": "phase15_d154_finalize_lock_v2",
        "status": "locked_postfit_recompute_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "parent_sha256": parents,
        "scope": cfg["scope"],
        "outputs": cfg["outputs"],
        "d154_v2_lock_sha256": sha256_file(v2_lock_path),
        "d154_v2_source_commit": v2_lock["source_commit"],
        "d154_v2_finalize_abort_output_empty": True,
        "preserved_checkpoint_sha256": {
            path.name: sha256_file(path) for path in sorted(run_dir.glob("*.npz"))},
        "memorization_sha256": sha256_file(run_dir / "memorization.json"),
        "validation_reuse": "D152 validation reused through D153/D154; exploratory only",
        "test_or_reserve_used": False,
    }


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    lock = _read_json(ROOT / LOCK)
    if lock.get("schema") != "phase15_d154_finalize_lock_v2":
        raise D154Error("D154 finalizer lock schema mismatch")
    if lock.get("source_sha256") != {path: sha256_file(ROOT / path) for path in SOURCE_FILES}:
        raise D154Error("D154 finalizer source differs from its lock")
    if lock.get("parent_sha256") != {name: sha256_file(ROOT / path)
                                      for name, path in cfg["parents"].items()}:
        raise D154Error("D154 finalizer parent changed")
    if lock.get("d154_v2_lock_sha256") != sha256_file(ROOT / cfg["parents"]["d154_v2_lock"]):
        raise D154Error("D154 v2 lock changed")
    if lock.get("scope") != cfg["scope"] or lock.get("test_or_reserve_used") is not False:
        raise D154Error("D154 finalizer scope mismatch")
    if _git("status", "--porcelain"):
        raise D154Error("D154 finalization requires a clean committed tree")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    run_dir = ROOT / cfg["outputs"]["run_dir"]
    if run_dir.exists() or (ROOT / cfg["outputs"]["report"]).exists():
        raise D154Error("D154 finalization output already exists")
    v2_cfg = _v2_cfg()
    v2_run = ROOT / v2_cfg["outputs"]["run_dir"]
    actual = {path.name: sha256_file(path) for path in sorted(v2_run.glob("*.npz"))}
    if actual != lock["preserved_checkpoint_sha256"]:
        raise D154Error("preserved D154 v2 checkpoint set/hash differs from finalizer lock")
    return cfg, lock


def _load_model(path: Path, name: str) -> tuple[Any, bool, int]:
    with np.load(path, allow_pickle=False) as weights:
        if set(weights.files) != set(WEIGHT_FIELDS):
            raise D154Error(f"checkpoint fields invalid: {path.name}")
        fields = {key: np.asarray(weights[key], dtype=np.float64) for key in WEIGHT_FIELDS}
    expected_shapes = {
        "x_mean": (61,), "x_std": (61,), "w1": (61, 32), "b1": (32,),
        "w2": (32, 32), "b2": (32,), "w3": (32, 9), "b3": (9,),
    }
    if any(fields[key].shape != shape or not np.isfinite(fields[key]).all()
           for key, shape in expected_shapes.items()) or np.any(fields["x_std"] <= 0):
        raise D154Error(f"checkpoint tensor invalid: {path.name}")
    tau_text = name.removeprefix("T3_listwise_tau_") if name.startswith("T3_") else ""
    if tau_text:
        model = ListwiseRegretMLP.__new__(ListwiseRegretMLP)
        model.temperature = float(tau_text)
        listwise = True
    else:
        model = TinyRegretMLP.__new__(TinyRegretMLP)
        model.objective = "regression" if name.startswith("T1") else "weighted_pairwise"
        listwise = False
    for key, value in fields.items():
        setattr(model, key, value)
    params = int(sum(fields[key].size for key in ("w1", "b1", "w2", "b2", "w3", "b3")))
    return model, listwise, params


def _parity(references: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    v2_cfg = _v2_cfg()
    d153 = _read_json(ROOT / v2_cfg["parents"]["d153_result"])
    expected = {
        "direct_D150_R0": d153["fixed_references"]["direct_r0_decisions"],
        "exact_R4_then_D150_head_oracle": d153["fixed_references"]["exact_r4_decisions"],
        "D153_predicted_R4_then_D150_head": next(
            row["validation_downstream_decisions"] for row in d153["scale_results"]
            if int(row["layout_count"]) == 200),
    }
    pairs = (("layout_macro_normalized_selected_regret", "layout_macro_normalized_regret"),
             ("catastrophic_fraction_state_micro", "state_micro_catastrophic_fraction_regret_ge_0_5"),
             ("best_action_set_agreement", "best_action_set_agreement"))
    parity = {
        name: {metric: abs(float(record["metrics"][metric]) - float(expected[name][saved]))
               for metric, saved in pairs}
        for name, record in references.items()
    }
    if max(value for rows in parity.values() for value in rows.values()) > 1e-10:
        raise D154Error(f"recomputed frozen-reference parity failed: {parity}")
    return parity


def _reference_result_records(references: dict[str, Any]) -> dict[str, Any]:
    """Select serializable fields from the scorer's nested reference records."""
    records = {}
    for name, row in references.items():
        if "metrics" not in row or "parameter_count" not in row["metrics"]:
            raise D154Error(f"reference metric record is incomplete: {name}")
        records[name] = {"metrics": row["metrics"],
                         "parameter_count": row["metrics"]["parameter_count"]}
    return records


def run_finalization() -> dict[str, Any]:
    cfg, lock = validate_lock()
    v2_cfg = _v2_cfg()
    data = _load_data(v2_cfg)
    output = ROOT / cfg["outputs"]["run_dir"]
    output.mkdir(parents=True, exist_ok=False)
    mem = _read_json(ROOT / v2_cfg["outputs"]["run_dir"] / "memorization.json")
    refs = _reference_metrics(v2_cfg, data["x_val"], data["y_val"], data["val_layouts"],
                              data["exact_r4"], data["predicted_summary"])
    reference_parity = _parity(refs, v2_cfg)
    models: dict[str, Any] = {}
    v2_run = ROOT / v2_cfg["outputs"]["run_dir"]
    for name in (
        "T1_regret_regression", "T2_margin_weighted_pairwise",
        "T3_listwise_tau_0.125", "T3_listwise_tau_0.25", "T3_listwise_tau_0.5",
    ):
        model, listwise, parameter_count = _load_model(v2_run / f"{name}.npz", name)
        models[name] = {
            "objective": "listwise_cross_entropy" if listwise else model.objective,
            "temperature": model.temperature if listwise else None,
            "best_epoch": None,
            "fit_seconds": None,
            "training_metadata_note": "not serialized by the v2 runner; not reconstructed or retrained",
            "parameter_count": parameter_count,
            "checkpoint_sha256": sha256_file(v2_run / f"{name}.npz"),
            "train": _model_metrics(model, data["x_train"], data["y_train"],
                                    data["train_layouts"], v2_cfg, listwise),
            "validation": _model_metrics(model, data["x_val"], data["y_val"],
                                         data["val_layouts"], v2_cfg, listwise),
        }
    component = component_ablation(
        data["exact_r4"], data["predicted_summary"].reshape(-1, 63), data["x_val"],
        data["y_val"], data["val_layouts"],
        restore_decision_head(ROOT / v2_cfg["parents"]["d150_r4_checkpoint"]))
    component_metrics = {
        name: {key: value for key, value in row.items()
               if key not in {"selected_regret", "per_layout_mean_regret"}}
        for name, row in component.items()
    }
    baseline_regret = refs["D153_predicted_R4_then_D150_head"]["metrics"][
        "layout_macro_normalized_selected_regret"]
    best_direct = min((row["validation"]["layout_macro_normalized_selected_regret"], name)
                      for name, row in models.items())
    result = {
        "schema": "phase15_d154_v2_postfit_finalization",
        "status": "exploratory_recomputed_from_preserved_fits",
        "source_commit": lock["source_commit"],
        "finalizer_lock_sha256": sha256_file(ROOT / LOCK),
        "v2_lock_sha256": lock["d154_v2_lock_sha256"],
        "preserved_checkpoint_sha256": lock["preserved_checkpoint_sha256"],
        "target_reconstruction_audit": _read_json(ROOT / v2_cfg["outputs"]["lock"])[
            "target_reconstruction_audit"],
        "split": {"train_rows": len(data["x_train"]), "train_layouts": len(set(data["train_layouts"].tolist())),
                  "validation_rows": len(data["x_val"]), "validation_layouts": len(set(data["val_layouts"].tolist())),
                  "validation_reused": True, "test_or_reserve_used": False},
        "memorization": mem,
        "frozen_reference_reproduction_max_abs_differences": reference_parity,
        "references": _reference_result_records(refs),
        "models": models,
        "component_ablation": component_metrics,
        "decision_summary": {
            "best_direct_formulation": best_direct[1],
            "best_direct_validation_macro_regret": best_direct[0],
            "d153_predicted_r4_validation_macro_regret": baseline_regret,
            "best_direct_minus_predicted_r4_regret": best_direct[0] - baseline_regret,
            "direct_improves_regret_vs_predicted_r4": best_direct[0] < baseline_regret,
            "validation_reuse_precludes_independent_confirmation": True,
        },
        "training_repeated": False,
        "missing_v2_metadata": ["best_epoch", "fit_seconds"],
        "scope": {"simulator_calls": 0, "planner_calls": 0, "new_transitions": 0,
                  "new_episodes": 0, "closed_loop": False, "dagger": False,
                  "world_model": False, "jepa": False, "test_rows_scored": 0,
                  "reserve_access": False},
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = ROOT / cfg["outputs"]["report"]
    report_path.write_text(render_report(result), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "schema": "phase15_d154_finalize_manifest_v2",
        "status": result["status"],
        "lock_sha256": result["finalizer_lock_sha256"],
        "result_sha256": sha256_file(output / "result.json"),
        "report_sha256": sha256_file(report_path),
        "parent_sha256": lock["parent_sha256"],
        "checkpoint_sha256": lock["preserved_checkpoint_sha256"],
        "training_repeated": False,
        "test_rows_scored": 0,
        "reserve_access": False,
        "simulator_calls": 0,
        "planner_calls": 0,
        "closed_loop": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# D154 — Decision-Aligned Target Study (v2 post-fit finalization)", "",
        "**Status: exploratory offline mechanism study using the reused D152 validation block.** This finalization recomputes metrics from preserved v2 checkpoints; it does not retrain. No independent confirmation is claimed; reserve/test remain sealed.", "",
        "## Runner history and integrity", "",
        "The v1 run stopped before fitting due exact-R4 reference input construction. The v2 amendment corrected that shape error; v2 passed reference parity and the 8-state memorization gate, saved all five full-data model checkpoints, then stopped after fitting on a `NameError` in final-result assembly. The original v2 directory is preserved. This report is recomputed from its hash-pinned checkpoints in a separate output namespace.", "",
        f"Full training: {result['split']['train_rows']} states / {result['split']['train_layouts']} layouts. Validation: {result['split']['validation_rows']} states / {result['split']['validation_layouts']} previously reused layouts. Exact R4-to-regret reconstruction remains bit-exact on D152 train/validation. Training repeated: `{result['training_repeated']}`. Best epoch and fit time were not serialized in v2 and are therefore unavailable.", "",
        "## Frozen validation references", "",
        "| Method | Macro regret | Catastrophic | Best-action agreement | Epsilon-optimal | High-margin ranking | Safe/catastrophic ordering | Latency scope |", "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    rows = [(name, value["metrics"]) for name, value in result["references"].items()]
    rows += [(name, value["validation"]) for name, value in result["models"].items()]
    for name, m in rows:
        latency = m.get("inference_median_seconds_per_state")
        if latency is None and "r4_head_only_median_seconds_per_state" in m:
            scope = f"scorer only ({m['latency_scope']})"
        else:
            scope = "end-to-end state scorer"
        lines.append(f"| {name} | {m['layout_macro_normalized_selected_regret']:.6f} | {m['catastrophic_fraction_state_micro']:.4f} | {m['best_action_set_agreement']:.4f} | {m['epsilon_optimal_rate']:.4f} | {m['high_margin_pairwise_accuracy_macro'] if m['high_margin_pairwise_accuracy_macro'] is not None else float('nan'):.4f} | {m['safe_vs_catastrophic_order_accuracy_macro'] if m['safe_vs_catastrophic_order_accuracy_macro'] is not None else float('nan'):.4f} | {scope} |")
    lines.extend(["", "## Preserved student checkpoint metrics", "", "| Formulation | Train macro regret | Train catastrophic | Validation macro regret | Validation catastrophic | Regret RMSE (validation) | Latency ms/state | Parameters |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name, record in result["models"].items():
        tr, va = record["train"], record["validation"]
        latency = va.get("inference_median_seconds_per_state")
        rmse = va.get("regret_score_rmse")
        lines.append(f"| {name} | {tr['layout_macro_normalized_selected_regret']:.6f} | {tr['catastrophic_fraction_state_micro']:.4f} | {va['layout_macro_normalized_selected_regret']:.6f} | {va['catastrophic_fraction_state_micro']:.4f} | {rmse if rmse is not None else float('nan'):.6f} | {1000*latency if latency is not None else float('nan'):.5f} | {record['parameter_count']} |")
    lines.extend(["", "## Frozen R4 component substitution", "", "This diagnoses sensitivity of the fixed D150 R4 head; it is not causal component importance.", "", "| Substitution | Macro regret | Catastrophic | Best-action agreement |", "|---|---:|---:|---:|"])
    for name, m in result["component_ablation"].items():
        lines.append(f"| {name} | {m['mean_normalized_regret_layout_macro']:.6f} | {m['catastrophic_fraction_state_micro']:.4f} | {m['best_action_set_agreement']:.4f} |")
    summary = result["decision_summary"]
    lines.extend(["", "## Interpretation boundary", "", f"Best direct checkpoint by reused-validation macro regret: `{summary['best_direct_formulation']}` ({summary['best_direct_validation_macro_regret']:.6f}); D153 predicted-R4 reference: {summary['d153_predicted_r4_validation_macro_regret']:.6f}. Difference: {summary['best_direct_minus_predicted_r4_regret']:+.6f}. This remains exploratory because validation was repeatedly reused.", "", "No new transition, simulator/planner call, episode, closed-loop evaluation, DAgger, world-model/JEPA training, or test/reserve scoring occurred. See structured result for complete metrics, hashes, and scope.", ""])
    return "\n".join(lines)
