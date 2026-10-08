"""D158 permutation-equivariant, decision-aligned candidate selection."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase15_d155_stage_c import (
    CARTOutcomeSteps, _load_stage_a_head, _weights_per_action, action_conditioned_inputs, encode_targets,
)
from consequence_lab.phase15_d156 import _read_json, load_predictors
from consequence_lab.phase15_d157 import (
    ACTION_COUNT, TRAIN_SPLIT, _paired_layout_bootstrap,
    _predictor_output, _strip_metrics,
)
from consequence_lab.phase15_decision_aligned import (
    decision_detail_metrics, regrets_from_exact_summaries, sha256_array, validate_d153_parent_manifest,
)
from consequence_lab.phase15_learning import layout_weights
from consequence_lab.phase15_outcome_summary import restore_decision_head
from consequence_lab.phase15_representation_screen import train_validation_indices

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase15_d158_relational_selection.yaml"
PROTOCOL = "reports/phase15_d158_amendment_v1.md"
LOCK = "experiments/phase15_d158_relational_selection_v2.lock.json"
SOURCE_FILES = (
    CONFIG, "reports/phase15_d158_protocol.md", PROTOCOL,
    "reports/phase15_d158_v1_gate_abort.md", "src/consequence_lab/phase15_d158.py",
    "scripts/freeze_phase15_d158_lock.py", "scripts/run_phase15_d158.py",
    "tests/test_phase15_d158.py", "src/consequence_lab/phase14.py",
    "second-brain/00_Project_Home.md", "second-brain/14_Decisions.md",
    "second-brain/16_Experiment_Log.md",
    "second-brain/phases/Phase_15_Decision_Aligned_Target_Study.md",
    "src/consequence_lab/phase15_d154_finalize.py", "src/consequence_lab/phase15_d155.py",
    "src/consequence_lab/phase15_d155_stage_c.py",
    "src/consequence_lab/phase15_d156.py", "src/consequence_lab/phase15_d157.py",
    "src/consequence_lab/phase15_decision_aligned.py", "src/consequence_lab/phase15_learning.py",
    "src/consequence_lab/phase15_outcome_summary.py", "src/consequence_lab/phase15_representation_screen.py",
    "src/consequence_lab/phase15_data.py", "src/consequence_lab/phase15_generalization.py",
)
REPRESENTATION_IDS = (
    "r0_action_only", "r0_action_predicted_outcome", "r0_action_predicted_steps",
    "r0_action_predicted_outcome_steps", "r0_action_exact_outcome_steps",
)
OBJECTIVE_IDS = ("listwise_tau_025", "margin_pairwise", "listwise_plus_pairwise")


class D158Error(RuntimeError):
    """D158 lock, data-contract, or execution failure."""


def _cfg() -> dict[str, Any]:
    return yaml.safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise D158Error(f"git {' '.join(args)} failed") from exc


def _parent_hashes(cfg: dict[str, Any]) -> dict[str, str]:
    return {name: sha256_file(ROOT / path) for name, path in cfg["parents"].items()}


def _validate_parent_lineage(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["parents"]
    d157_lock = _read_json(ROOT / p["d157_correction_lock"])
    d157_manifest = _read_json(ROOT / p["d157_correction_manifest"])
    d157_result = _read_json(ROOT / p["d157_correction_result"])
    if (d157_lock.get("schema") != "phase15_d157_score_correction_lock_v1"
            or d157_result.get("status") != "completed_score_only_finalization_from_preserved_d157_dataset"
            or d157_manifest.get("status") != d157_result.get("status")
            or d157_manifest.get("lock_sha256") != sha256_file(ROOT / p["d157_correction_lock"])
            or d157_manifest.get("result_sha256") != sha256_file(ROOT / p["d157_correction_result"])
            or d157_manifest.get("reserve_access") is not False
            or d157_manifest.get("test_rows_dereferenced") != 0
            or d157_result.get("reserve_access") is not False
            or d157_result.get("scope", {}).get("test_rows_dereferenced") != 0):
        raise D158Error("D157 score-only parent status/hash/scope audit failed")
    if (d157_manifest.get("dataset_sha256") != sha256_file(ROOT / p["d157_rows"])
            or d157_manifest.get("accounting_sha256") != sha256_file(ROOT / p["d157_accounting"])):
        raise D158Error("D157 fresh evaluation dataset/accounting hash mismatch")

    d157_original = _read_json(ROOT / p["d157_original_lock"])
    original_abort = _read_json(ROOT / p["d157_original_abort"])
    if (d157_original.get("schema") != "phase15_d157_fresh_confirmation_lock_v1"
            or d157_original.get("scope", {}).get("reserve_access") is not False
            or original_abort.get("status") != "aborted_or_incomplete"
            or original_abort.get("lock_sha256") != sha256_file(ROOT / p["d157_original_lock"])):
        raise D158Error("original D157 abort/lock preservation audit failed")

    d158_v1_lock = _read_json(ROOT / p["d158_v1_lock"])
    d158_v1_abort = _read_json(ROOT / p["d158_v1_abort"])
    if (d158_v1_lock.get("schema") != "phase15_d158_relational_selection_lock_v1"
            or d158_v1_abort.get("status") != "aborted_or_incomplete"
            or d158_v1_abort.get("lock_sha256") != sha256_file(ROOT / p["d158_v1_lock"])
            or d158_v1_abort.get("source_commit") != d158_v1_lock.get("source_commit")
            or "tiny memorization gate failed" not in d158_v1_abort.get("error", "")):
        raise D158Error("D158 v1 gate abort/lock preservation audit failed")

    d154_lock = _read_json(ROOT / p["d154_v2_lock"])
    d154_result = _read_json(ROOT / p["d154_result"])
    d154_manifest = _read_json(ROOT / p["d154_manifest"])
    d154_finalize_lock = _read_json(ROOT / p["d154_finalize_lock"])
    if (d154_lock.get("schema") != "phase15_d154_decision_aligned_lock_v2"
            or d154_result.get("status") != "exploratory_recomputed_from_preserved_fits"
            or d154_finalize_lock.get("schema") != "phase15_d154_finalize_lock_v2"
            or d154_manifest.get("reserve_access") is not False
            or d154_manifest.get("test_rows_scored") != 0
            or d154_manifest.get("status") != d154_result.get("status")
            or d154_manifest.get("lock_sha256") != sha256_file(ROOT / p["d154_finalize_lock"])):
        raise D158Error("D154 direct-listwise baseline provenance/scope audit failed")
    d154_hash = sha256_file(ROOT / p["d154_listwise_r0"])
    manifest_hashes = d154_manifest.get("checkpoint_sha256", {})
    if manifest_hashes.get("T3_listwise_tau_0.25.npz") != d154_hash:
        raise D158Error("D154 direct-listwise checkpoint hash mismatch")

    d155_lock = _read_json(ROOT / p["d155_stage_c_lock"])
    d155_manifest = _read_json(ROOT / "artifacts/runs/phase15_d155_stage_c_v1/manifest.json")
    for filename, parent_key in (("per_action_constant.npz", "d155_stage_c_constant"),
                                 ("bounded_cart.json", "d155_stage_c_cart"),
                                 ("action_conditioned_mlp.npz", "d155_stage_c_mlp")):
        if d155_manifest.get("predictor_checkpoint_sha256", {}).get(filename) != sha256_file(ROOT / p[parent_key]):
            raise D158Error(f"D155 frozen consequence checkpoint hash mismatch: {filename}")
    if (d155_lock.get("schema") != "phase15_d155_stage_c_lock_v1"
            or d155_manifest.get("reserve_access") is not False
            or d155_manifest.get("test_rows_dereferenced") != 0
            or d155_manifest.get("lock_sha256") != sha256_file(ROOT / p["d155_stage_c_lock"])):
        raise D158Error("D155 compact predictor provenance/scope audit failed")

    d153_lock = _read_json(ROOT / p["d153_lock"])
    if (d153_lock.get("schema") != "phase15_d153_coverage_scaling_lock_v3"
            or d153_lock.get("reused_corpus", {}).get("sha256") != sha256_file(ROOT / p["d153_train_rows"])):
        raise D158Error("D153 training corpus hash/lock mismatch")
    d153_manifest = _read_json(ROOT / "artifacts/runs/phase15_d153_coverage_scaling_v3/manifest.json")
    validate_d153_parent_manifest(d153_manifest)

    d152_lock = _read_json(ROOT / p["d152_lock"])
    if d152_lock.get("schema") != "phase15_representation_sufficiency_lock_v1":
        raise D158Error("D152 split parent lock schema mismatch")
    return {
        "d157_status": d157_result["status"],
        "d157_abort_preserved": True,
        "d154_checkpoint_sha256": d154_hash,
        "d153_status": d153_manifest["status"],
        "reserve_access": False,
        "test_rows_dereferenced": 0,
    }


def _load_data(cfg: dict[str, Any]) -> dict[str, Any]:
    p = cfg["parents"]
    with np.load(ROOT / p["d152_features"], allow_pickle=False) as d152:
        d152_layouts_all = d152["layouts"].astype(str).tolist()
        train_idx, val_idx, train_layouts, val_layouts = train_validation_indices(d152_layouts_all, TRAIN_SPLIT)
        train_r0_d152 = np.asarray(d152["x_R0"][train_idx], dtype=np.float64)
        train_sum_d152 = np.asarray(d152["x_R4"][train_idx, 61:].reshape(-1, 9, 7), dtype=np.float64)
        val_r0 = np.asarray(d152["x_R0"][val_idx], dtype=np.float64)
        val_sum = np.asarray(d152["x_R4"][val_idx, 61:].reshape(-1, 9, 7), dtype=np.float64)
        val_layouts_rows = np.asarray(d152_layouts_all, dtype=str)[val_idx]
        d152_train_layout_rows = np.asarray(d152_layouts_all, dtype=str)[train_idx]

    with np.load(ROOT / p["d153_train_rows"], allow_pickle=False) as d153:
        fresh_r0 = np.asarray(d153["x_R0"], dtype=np.float64)
        fresh_sum = np.asarray(d153["summary_R4"], dtype=np.float64)
        fresh_layouts = d153["layouts"].astype(str)
    train_r0 = np.concatenate((train_r0_d152, fresh_r0), axis=0)
    train_sum = np.concatenate((train_sum_d152, fresh_sum), axis=0)
    train_layout_rows = np.concatenate((d152_train_layout_rows, fresh_layouts), axis=0)

    with np.load(ROOT / p["d157_rows"], allow_pickle=False) as d157:
        eval_r0 = np.asarray(d157["x_R0"], dtype=np.float64)
        eval_sum = np.asarray(d157["summary_R4"], dtype=np.float64)
        eval_layouts = d157["layouts"].astype(str)
        eval_seeds = np.asarray(d157["seeds"], dtype=np.int64)

    expected = ((train_r0, (1936, 61)), (train_sum, (1936, 9, 7)),
                (val_r0, (202, 61)), (val_sum, (202, 9, 7)),
                (eval_r0, (794, 61)), (eval_sum, (794, 9, 7)))
    if any(value.shape != shape or not np.isfinite(value).all() for value, shape in expected):
        raise D158Error("D158 split feature/target shape or finite-value contract failed")
    train_layout_set, val_layout_set, eval_layout_set = (
        set(train_layout_rows.tolist()), set(val_layouts_rows.tolist()), set(eval_layouts.tolist()))
    if (len(train_layout_set) != 200 or len(val_layout_set) != 20 or len(eval_layout_set) != 100
            or train_layout_set & val_layout_set or train_layout_set & eval_layout_set
            or val_layout_set & eval_layout_set or len(set(eval_seeds.tolist())) != 100):
        raise D158Error("D158 train/validation/heldout layout independence contract failed")
    if (not np.allclose(train_sum[:, :, :4].sum(axis=2), 1.0, atol=0.0, rtol=0.0)
            or not np.allclose(val_sum[:, :, :4].sum(axis=2), 1.0, atol=0.0, rtol=0.0)
            or not np.allclose(eval_sum[:, :, :4].sum(axis=2), 1.0, atol=0.0, rtol=0.0)
            or np.any(train_sum[:, :, 1] != 0.0) or np.any(val_sum[:, :, 1] != 0.0)
            or np.any(eval_sum[:, :, 1] != 0.0)):
        raise D158Error("D158 requires terminal one-hot outcomes with no active-zero branches")
    if (np.any(train_sum < 0.0) or np.any(train_sum > 1.0)
            or np.any(val_sum < 0.0) or np.any(val_sum > 1.0)
            or np.any(eval_sum < 0.0) or np.any(eval_sum > 1.0)):
        raise D158Error("D158 exact summaries must be normalized to [0,1]")

    y_train = regrets_from_exact_summaries(train_sum)
    y_val = regrets_from_exact_summaries(val_sum)
    y_eval = regrets_from_exact_summaries(eval_sum)
    d154_lock = _read_json(ROOT / p["d154_v2_lock"])
    expected_target = d154_lock["target_reconstruction_audit"]
    if (sha256_array(y_train) != expected_target["train_regret_sha256"]
            or sha256_array(y_val) != expected_target["validation_regret_sha256"]):
        raise D158Error("D158 exact regrets fail frozen D154 train/validation parity")

    return {
        "train_r0": train_r0, "train_summary": train_sum, "train_regret": y_train,
        "train_layouts": train_layout_rows,
        "val_r0": val_r0, "val_summary": val_sum, "val_regret": y_val,
        "val_layouts": val_layouts_rows,
        "eval_r0": eval_r0, "eval_summary": eval_sum, "eval_regret": y_eval,
        "eval_layouts": eval_layouts, "eval_seeds": eval_seeds,
        "split": {
            "train_rows": len(train_r0), "train_layout_count": len(train_layout_set),
            "train_layout_sha256": _hash_layouts(train_layout_set),
            "d152_validation_rows": len(val_r0), "d152_validation_layout_count": len(val_layout_set),
            "d152_validation_layout_sha256": _hash_layouts(val_layout_set),
            "d157_heldout_rows": len(eval_r0), "d157_heldout_layout_count": len(eval_layout_set),
            "d157_heldout_layout_sha256": _hash_layouts(eval_layout_set),
            "d157_seeds": [int(value) for value in sorted(set(eval_seeds.tolist()))],
            "train_regret_sha256": sha256_array(y_train),
            "validation_regret_sha256": sha256_array(y_val),
            "heldout_regret_sha256": sha256_array(y_eval),
            "validation_exploratory_reused": True,
            "heldout_exploratory_previously_used_for_frozen_reference_scoring": True,
        },
    }


def _hash_layouts(layouts: set[str]) -> str:
    payload = json.dumps(sorted(layouts), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def build_lock() -> dict[str, Any]:
    cfg = _cfg()
    if _git("status", "--porcelain"):
        raise D158Error("commit D158 protocol/config/source/tests/vault before freezing its lock")
    if cfg.get("schema") != "phase15_d158_relational_selection_v2":
        raise D158Error("D158 config schema mismatch")
    if _validate_parent_lineage(cfg).get("reserve_access") is not False:
        raise D158Error("D158 parent lineage unexpectedly touched reserve")
    data = _load_data(cfg)
    for output_name, output in cfg["outputs"].items():
        if (ROOT / output).exists():
            raise D158Error(f"refusing to overwrite D158 output {output_name}: {output}")
    return {
        "schema": "phase15_d158_relational_selection_lock_v2",
        "status": "locked_exploratory_offline_relational_candidate_study_tie_gate_amendment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "config_sha256": sha256_file(ROOT / CONFIG),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "parent_sha256": _parent_hashes(cfg),
        "split": data["split"],
        "lineage": _validate_parent_lineage(cfg),
        "model": cfg["model"], "representations": cfg["representations"],
        "objectives": cfg["model"]["objective_comparison"],
        "risk_policy": cfg["risk_policy"], "metrics": cfg["metrics"],
        "advancement_gate": cfg["advancement_gate"],
        "scope": cfg["scope"], "outputs": cfg["outputs"],
    }


def validate_lock() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _cfg()
    lock = _read_json(ROOT / cfg["outputs"]["lock"])
    if (lock.get("schema") != "phase15_d158_relational_selection_lock_v2"
            or lock.get("status") != "locked_exploratory_offline_relational_candidate_study_tie_gate_amendment"):
        raise D158Error("D158 lock schema/status mismatch")
    if (lock.get("config_sha256") != sha256_file(ROOT / CONFIG)
            or lock.get("source_sha256") != {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
            or lock.get("parent_sha256") != _parent_hashes(cfg)
            or lock.get("model") != cfg["model"]
            or lock.get("representations") != cfg["representations"]
            or lock.get("objectives") != cfg["model"]["objective_comparison"]
            or lock.get("risk_policy") != cfg["risk_policy"]
            or lock.get("metrics") != cfg["metrics"]
            or lock.get("advancement_gate") != cfg["advancement_gate"]
            or lock.get("scope") != cfg["scope"]
            or lock.get("outputs") != cfg["outputs"]):
        raise D158Error("D158 source/config/parent/protocol lock mismatch")
    if _git("status", "--porcelain"):
        raise D158Error("D158 one-shot training/evaluation requires a clean committed tree")
    _git("merge-base", "--is-ancestor", lock["source_commit"], "HEAD")
    _validate_parent_lineage(cfg)
    if _load_data(cfg)["split"] != lock["split"]:
        raise D158Error("D158 split/data fingerprint changed after lock")
    for output_name, output in cfg["outputs"].items():
        if output_name == "lock":
            continue
        if (ROOT / output).exists():
            raise D158Error(f"refusing second D158 run: {output}")
    return cfg, lock


def _fold_index(layout: str, fold_count: int) -> int:
    digest = hashlib.sha256(str(layout).encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % fold_count


def _action_inputs(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(states)
    actions = np.broadcast_to(np.eye(ACTION_COUNT, dtype=np.float64),
                              (count, ACTION_COUNT, ACTION_COUNT)).copy()
    state_action = action_conditioned_inputs(states)
    return actions, state_action


def _predict_cart(predictor: Any, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _predictor_output(predictor, states)


def _crossfit_train_consequences(data: dict[str, Any], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    fold_count = 5
    train_states, train_summary = data["train_r0"], data["train_summary"]
    train_layouts = data["train_layouts"]
    inputs = action_conditioned_inputs(train_states)
    classes, steps = encode_targets(train_summary)
    classes = classes.reshape(len(train_states), ACTION_COUNT)
    steps = steps.reshape(len(train_states), ACTION_COUNT)
    fold_ids = np.asarray([_fold_index(layout, fold_count) for layout in train_layouts], dtype=np.int64)
    predictions = np.empty((len(train_states), ACTION_COUNT, 3), dtype=np.float64)
    step_predictions = np.empty((len(train_states), ACTION_COUNT), dtype=np.float64)
    stage_cfg = yaml.safe_load((ROOT / cfg["parents"]["d155_stage_c_config"]).read_text(encoding="utf-8"))
    model_cfg = stage_cfg["predictors"]["bounded_cart"]
    audit = []
    for fold in range(fold_count):
        validation_rows = np.flatnonzero(fold_ids == fold)
        training_rows = np.flatnonzero(fold_ids != fold)
        if not len(validation_rows) or not len(training_rows):
            raise D158Error("layout-grouped consequence cross-fit produced an empty fold")
        train_mask = np.repeat(np.isin(np.arange(len(train_states)), training_rows), ACTION_COUNT)
        validation_mask = np.repeat(np.isin(np.arange(len(train_states)), validation_rows), ACTION_COUNT)
        predictor = CARTOutcomeSteps(model_cfg).fit(
            inputs[train_mask], classes.reshape(-1)[train_mask], steps.reshape(-1)[train_mask],
            _weights_per_action(train_layouts[training_rows]),
        )
        probs, predicted_steps = predictor.predict(inputs[validation_mask])
        predictions[validation_rows] = np.maximum(probs.reshape(-1, ACTION_COUNT, 3), 1e-12)
        predictions[validation_rows] /= predictions[validation_rows].sum(axis=2, keepdims=True)
        step_predictions[validation_rows] = np.clip(predicted_steps.reshape(-1, ACTION_COUNT), 0.0, 1.0)
        audit.append({
            "fold": fold, "train_layout_count": len(set(train_layouts[training_rows].tolist())),
            "heldout_layout_count": len(set(train_layouts[validation_rows].tolist())),
            "train_row_count": len(training_rows), "heldout_row_count": len(validation_rows),
            "prediction_model_nodes": int(predictor.node_count),
            "heldout_layout_sha256": _hash_layouts(set(train_layouts[validation_rows].tolist())),
        })
    if not np.isfinite(predictions).all() or not np.isfinite(step_predictions).all():
        raise D158Error("cross-fitted frozen-family consequence predictions are non-finite")
    return predictions, step_predictions, audit


def _probabilities_from_exact(summary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(summary, dtype=np.float64)
    probabilities = values[:, :, [0, 2, 3]].copy()
    steps = np.asarray(values[:, :, 4], dtype=np.float64).copy()
    return probabilities, steps


def _feature_set(probabilities: np.ndarray, steps: np.ndarray, rep_id: str) -> tuple[np.ndarray, tuple[float, float]]:
    probs = np.asarray(probabilities, dtype=np.float64).copy()
    step_values = np.asarray(steps, dtype=np.float64).copy()
    use_outcome = "outcome" in rep_id
    use_steps = "steps" in rep_id
    if rep_id == "r0_action_only":
        use_outcome = use_steps = False
    if rep_id == "r0_action_exact_outcome_steps":
        use_outcome = use_steps = True
    if not use_outcome:
        probs.fill(0.0)
    if not use_steps:
        step_values.fill(0.0)
    consequence = np.concatenate((probs, step_values[:, :, None]), axis=2)
    if consequence.shape[1:] != (ACTION_COUNT, 4) or not np.isfinite(consequence).all():
        raise D158Error("D158 consequence candidate feature contract failed")
    return consequence, (float(use_outcome), float(use_steps))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = np.asarray(x, dtype=np.float64) - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class RelationalCandidateModel:
    """A small permutation-equivariant Deep Sets scorer with a collision-risk head."""

    def __init__(self, width: int, seed: int, lr: float, l2: float,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        self.width, self.seed, self.lr, self.l2 = int(width), int(seed), float(lr), float(l2)
        self.beta1, self.beta2, self.epsilon = float(beta1), float(beta2), float(epsilon)

    def initialize(self, states: np.ndarray) -> None:
        x = np.asarray(states, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 61 or not np.isfinite(x).all():
            raise ValueError("D158 model expects finite deployable state [N,61]")
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        rng = np.random.default_rng(self.seed)
        h = self.width
        self.ws = rng.normal(0.0, np.sqrt(2.0 / 61), (61, h))
        self.bs = np.zeros(h)
        self.wc = rng.normal(0.0, np.sqrt(2.0 / (h + 15)), (h + 15, h))
        self.bc = np.zeros(h)
        self.wctx = rng.normal(0.0, np.sqrt(2.0 / (3 * h)), (3 * h, h))
        self.bctx = np.zeros(h)
        self.wscore = rng.normal(0.0, np.sqrt(1.0 / h), h)
        self.bscore = np.asarray(0.0)
        self.wrisk = rng.normal(0.0, np.sqrt(1.0 / h), h)
        self.brisk = np.asarray(0.0)
        self.parameter_count = int(sum(v.size for v in self.parameters()))

    def parameters(self) -> list[np.ndarray]:
        return [self.ws, self.bs, self.wc, self.bc, self.wctx, self.bctx,
                self.wscore, self.bscore, self.wrisk, self.brisk]

    def _forward(self, states: np.ndarray, actions: np.ndarray, consequences: np.ndarray,
                 mask: tuple[float, float], *, cache: bool = False):
        state = (np.asarray(states, dtype=np.float64) - self.x_mean) / self.x_std
        action = np.asarray(actions, dtype=np.float64)
        consequence = np.asarray(consequences, dtype=np.float64)
        n, candidates = action.shape[:2]
        if (state.shape != (n, 61) or action.shape != (n, 9, 9)
                or consequence.shape != (n, 9, 4) or not np.isfinite(state).all()
                or not np.isfinite(action).all() or not np.isfinite(consequence).all()):
            raise ValueError("invalid D158 candidate-set input dimensions or values")
        state_pre = state @ self.ws + self.bs
        state_h = np.maximum(state_pre, 0.0)
        state_rep = np.broadcast_to(state_h[:, None, :], (n, candidates, self.width))
        mask_values = np.broadcast_to(np.asarray(mask, dtype=np.float64), (n, candidates, 2))
        candidate_input = np.concatenate((state_rep, action, consequence, mask_values), axis=2)
        cand_pre = candidate_input @ self.wc + self.bc
        cand_h = np.maximum(cand_pre, 0.0)
        pooled = np.mean(cand_h, axis=1, keepdims=True)
        pool_rep = np.broadcast_to(pooled, cand_h.shape)
        state_rep2 = np.broadcast_to(state_h[:, None, :], cand_h.shape)
        context_input = np.concatenate((cand_h, pool_rep, state_rep2), axis=2)
        ctx_pre = context_input @ self.wctx + self.bctx
        ctx = np.maximum(ctx_pre, 0.0)
        utility = ctx @ self.wscore + self.bscore
        risk_logit = ctx @ self.wrisk + self.brisk
        if not cache:
            return utility, risk_logit
        return utility, risk_logit, (state, state_pre, state_h, candidate_input, cand_pre,
                                     cand_h, pooled, context_input, ctx_pre, ctx)

    def predict(self, states: np.ndarray, actions: np.ndarray, consequences: np.ndarray,
                mask: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        utility, risk_logit = self._forward(states, actions, consequences, mask)
        return utility, _sigmoid(risk_logit)

    def fit(self, states: np.ndarray, actions: np.ndarray, consequences: np.ndarray,
            mask: tuple[float, float], regret: np.ndarray, collision: np.ndarray,
            layouts: np.ndarray, val_states: np.ndarray, val_actions: np.ndarray,
            val_consequences: np.ndarray, val_mask: tuple[float, float], val_regret: np.ndarray,
            val_collision: np.ndarray, val_layouts: np.ndarray, objective: str,
            settings: dict[str, Any], metrics_cfg: dict[str, Any]) -> dict[str, Any]:
        self.initialize(states)
        wt = layout_weights(layouts)
        candidate_wt = wt[:, None] / ACTION_COUNT
        val_w = layout_weights(val_layouts)
        pair_i, pair_j = np.triu_indices(ACTION_COUNT, 1)
        train_pair_weights = _pair_weights(regret, collision, pair_i, pair_j, settings)
        val_pair_weights = _pair_weights(val_regret, val_collision, pair_i, pair_j, settings)
        teacher = _softmax(-np.asarray(regret) / float(settings["listwise_temperature"]))
        val_teacher = _softmax(-np.asarray(val_regret) / float(settings["listwise_temperature"]))
        params = self.parameters()
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        best = None
        best_criterion = (float("inf"), float("inf"))
        stale = 0
        max_epochs, patience = int(settings["max_epochs"]), int(settings["patience"])
        pair_lambda = float(settings["pairwise_weight"])
        risk_lambda = float(settings["collision_loss_weight"])
        for epoch in range(max_epochs):
            utility, risk_logit, cache = self._forward(states, actions, consequences, mask, cache=True)
            pref_loss, d_utility = _preference_loss_grad(
                utility, regret, teacher, wt, train_pair_weights, pair_i, pair_j,
                objective, pair_lambda,
            )
            risk_prob = _sigmoid(risk_logit)
            col = np.asarray(collision, dtype=np.float64)
            risk_loss = -float(np.sum(candidate_wt * (col * np.log(np.maximum(risk_prob, 1e-12))
                                                    + (1.0 - col) * np.log(np.maximum(1.0 - risk_prob, 1e-12)))))
            d_risk = risk_lambda * candidate_wt * (risk_prob - col)
            gradients = self._backward(d_utility, d_risk, cache)
            for idx, (parameter, gradient) in enumerate(zip(params, gradients, strict=True)):
                if parameter.ndim > 0 and parameter is not self.bs and parameter is not self.bc and parameter is not self.bctx:
                    gradient = gradient + self.l2 * parameter
                first[idx] = self.beta1 * first[idx] + (1.0 - self.beta1) * gradient
                second[idx] = self.beta2 * second[idx] + (1.0 - self.beta2) * gradient * gradient
                step = epoch + 1
                parameter -= self.lr * (first[idx] / (1.0 - self.beta1**step)) / (
                    np.sqrt(second[idx] / (1.0 - self.beta2**step)) + self.epsilon)
            val_utility, val_risk_logit = self._forward(val_states, val_actions, val_consequences, val_mask)
            val_pref, _ = _preference_loss_grad(
                val_utility, val_regret, val_teacher, val_w, val_pair_weights, pair_i, pair_j,
                objective, pair_lambda,
            )
            val_risk = _sigmoid(val_risk_logit)
            val_risk_loss = -float(np.sum((val_w[:, None] / ACTION_COUNT)
                                          * (val_collision * np.log(np.maximum(val_risk, 1e-12))
                                             + (1.0 - val_collision) * np.log(np.maximum(1.0 - val_risk, 1e-12)))))
            val_metrics = decision_detail_metrics(val_regret, -val_utility, val_layouts, metrics_cfg)
            criterion = (float(val_metrics["layout_macro_normalized_selected_regret"]),
                         float(val_pref + risk_lambda * val_risk_loss))
            improved = (criterion[0] < best_criterion[0] - 1e-9
                        or (abs(criterion[0] - best_criterion[0]) <= 1e-9
                            and criterion[1] < best_criterion[1] - 1e-7))
            if improved:
                best_criterion, stale = criterion, 0
                best = [p.copy() for p in params]
                best_epoch = epoch + 1
            else:
                stale += 1
            if stale >= patience:
                break
        if best is None:
            raise D158Error("D158 model failed to produce a validation checkpoint")
        for param, value in zip(params, best, strict=True):
            param[...] = value
        self.epochs_run = epoch + 1
        self.best_epoch = best_epoch
        self.validation_macro_regret = best_criterion[0]
        self.validation_objective_loss = best_criterion[1]
        self.parameter_count = int(sum(p.size for p in self.parameters()))
        return {"epochs_run": self.epochs_run, "best_epoch": self.best_epoch,
                "validation_macro_regret": self.validation_macro_regret,
                "validation_objective_loss": self.validation_objective_loss,
                "parameter_count": self.parameter_count,
                "training_preference_loss_last": float(pref_loss),
                "training_collision_bce_last": float(risk_loss)}

    def _backward(self, d_utility: np.ndarray, d_risk: np.ndarray,
                  cache: tuple[np.ndarray, ...]) -> list[np.ndarray]:
        state, state_pre, state_h, candidate_input, cand_pre, cand_h, pooled, context_input, ctx_pre, ctx = cache
        n, candidates, h = ctx.shape
        grad_score = np.asarray(d_utility, dtype=np.float64)
        grad_risk = np.asarray(d_risk, dtype=np.float64)
        grad_wscore = np.einsum("nch,nc->h", ctx, grad_score)
        grad_bscore = np.asarray(grad_score.sum())
        grad_wrisk = np.einsum("nch,nc->h", ctx, grad_risk)
        grad_brisk = np.asarray(grad_risk.sum())
        d_ctx = grad_score[:, :, None] * self.wscore + grad_risk[:, :, None] * self.wrisk
        d_ctx_pre = d_ctx * (ctx_pre > 0.0)
        flat_ctx_pre = d_ctx_pre.reshape(-1, h)
        flat_context = context_input.reshape(-1, 3 * h)
        grad_wctx = flat_context.T @ flat_ctx_pre
        grad_bctx = flat_ctx_pre.sum(axis=0)
        d_rel = (flat_ctx_pre @ self.wctx.T).reshape(n, candidates, 3 * h)
        d_cand = d_rel[:, :, :h] + np.sum(d_rel[:, :, h:2 * h], axis=1, keepdims=True) / candidates
        d_cand = np.broadcast_to(d_cand, cand_h.shape).copy()
        d_stateh = d_rel[:, :, 2 * h:].sum(axis=1)
        d_cand_pre = d_cand * (cand_pre > 0.0)
        flat_cand_pre = d_cand_pre.reshape(-1, h)
        grad_wc = candidate_input.reshape(-1, candidate_input.shape[-1]).T @ flat_cand_pre
        grad_bc = flat_cand_pre.sum(axis=0)
        d_cand_input = (flat_cand_pre @ self.wc.T).reshape(n, candidates, -1)
        d_stateh += d_cand_input[:, :, :h].sum(axis=1)
        d_statepre = d_stateh * (state_pre > 0.0)
        grad_ws = state.T @ d_statepre
        grad_bs = d_statepre.sum(axis=0)
        return [grad_ws, grad_bs, grad_wc, grad_bc, grad_wctx, grad_bctx,
                grad_wscore, grad_bscore, grad_wrisk, grad_brisk]


def _pair_weights(regret: np.ndarray, collision: np.ndarray, pair_i: np.ndarray,
                  pair_j: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    r = np.asarray(regret, dtype=np.float64)
    col = np.asarray(collision, dtype=bool)
    gaps = np.abs(r[:, pair_i] - r[:, pair_j])
    keep = gaps >= float(settings["pairwise_minimum_regret_gap"])
    safe = r <= float(settings["safe_regret_at_most"])
    catastrophic = r >= float(settings["catastrophic_regret_threshold"])
    collision_boost = float(settings["pairwise_safe_collision_weight_multiplier"])
    safe_cat_boost = float(settings["pairwise_safe_catastrophic_weight_multiplier"])
    collision_crossing = col[:, pair_i] != col[:, pair_j]
    safe_cat_crossing = ((safe[:, pair_i] & catastrophic[:, pair_j])
                         | (catastrophic[:, pair_i] & safe[:, pair_j]))
    boosted = collision_crossing | safe_cat_crossing
    multiplier = np.where(
        collision_crossing & safe_cat_crossing, max(collision_boost, safe_cat_boost),
        np.where(collision_crossing, collision_boost,
                 np.where(safe_cat_crossing, safe_cat_boost, 1.0)),
    )
    weights = gaps * np.where(boosted, multiplier, 1.0)
    weights *= keep
    sums = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0.0)


def _preference_loss_grad(utility: np.ndarray, regret: np.ndarray, teacher: np.ndarray,
                          state_weights: np.ndarray, pair_weights: np.ndarray,
                          pair_i: np.ndarray, pair_j: np.ndarray, objective: str,
                          pair_lambda: float) -> tuple[float, np.ndarray]:
    logits = np.asarray(utility, dtype=np.float64)
    r = np.asarray(regret, dtype=np.float64)
    sw = np.asarray(state_weights, dtype=np.float64)
    is_listwise = objective in {"listwise_tau_025", "listwise_plus_pairwise"}
    is_pairwise = objective in {"margin_pairwise", "listwise_plus_pairwise"}
    loss, grad = 0.0, np.zeros_like(logits)
    if is_listwise:
        probability = _softmax(logits)
        loss -= float(np.sum(sw[:, None] * teacher * np.log(np.maximum(probability, 1e-12))))
        grad += sw[:, None] * (probability - teacher)
    if is_pairwise:
        sign = np.sign(r[:, pair_j] - r[:, pair_i])
        diff = logits[:, pair_i] - logits[:, pair_j]
        z = -sign * diff
        pair_loss = np.logaddexp(0.0, z)
        weighted = sw[:, None] * pair_weights
        pair_scale = pair_lambda if objective == "listwise_plus_pairwise" else 1.0
        loss += pair_scale * float(np.sum(weighted * pair_loss))
        pair_grad = -sign / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        pair_grad *= pair_scale * weighted
        for idx in range(len(pair_i)):
            grad[:, pair_i[idx]] += pair_grad[:, idx]
            grad[:, pair_j[idx]] -= pair_grad[:, idx]
    return loss, grad


def _candidate_metrics(regret: np.ndarray, scores: np.ndarray, summary: np.ndarray,
                       layouts: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    target = np.asarray(regret, dtype=np.float64)
    lower_score = np.asarray(scores, dtype=np.float64)
    outcome = np.argmax(np.asarray(summary)[:, :, :4], axis=2)
    collision = outcome == 3
    safe_collision_acc, safe_cat_acc, success_timeout_acc = [], [], []
    near_safe_pairs = 0
    for row, row_score, row_collision, row_outcome, layout in zip(
            target, lower_score, collision, outcome, layouts, strict=True):
        left, right = np.triu_indices(ACTION_COUNT, 1)
        def correct(i: int, j: int) -> float:
            if row_score[i] < row_score[j]:
                return 1.0
            return 0.5 if row_score[i] == row_score[j] else 0.0
        for i, j in zip(left, right, strict=True):
            if row_collision[i] != row_collision[j]:
                safe_i = j if row_collision[i] else i
                coll_i = i if row_collision[i] else j
                safe_collision_acc.append((str(layout), correct(safe_i, coll_i)))
            if ((row[i] <= cfg["safe_action_regret_at_most"] and row[j] >= cfg["catastrophic_action_regret_at_least"])
                    or (row[j] <= cfg["safe_action_regret_at_most"] and row[i] >= cfg["catastrophic_action_regret_at_least"])):
                safe_i = i if row[i] <= cfg["safe_action_regret_at_most"] else j
                cat_i = j if safe_i == i else i
                safe_cat_acc.append((str(layout), correct(safe_i, cat_i)))
            if {int(row_outcome[i]), int(row_outcome[j])} == {0, 2}:
                success_i = i if row_outcome[i] == 0 else j
                timeout_i = j if success_i == i else i
                success_timeout_acc.append((str(layout), correct(success_i, timeout_i)))
            if (not row_collision[i] and not row_collision[j]
                    and row[i] <= cfg["safe_action_regret_at_most"]
                    and row[j] <= cfg["safe_action_regret_at_most"]
                    and abs(row[i] - row[j]) < cfg.get("pairwise_minimum_regret_gap", 0.125)):
                near_safe_pairs += 1
    # State-micro candidate-pair metrics; counts are included to expose class support.
    details = decision_detail_metrics(target, lower_score, layouts, cfg)
    selected = np.argmin(lower_score, axis=1)
    selected_collision = collision[np.arange(len(target)), selected]

    def layout_macro(pairs: list[tuple[str, float]]) -> float | None:
        if not pairs:
            return None
        groups: dict[str, list[float]] = {}
        for group, value in pairs:
            groups.setdefault(group, []).append(value)
        return float(np.mean([np.mean(values) for values in groups.values()]))

    return {
        **_strip_metrics(details),
        "safe_vs_collision_pair_accuracy": layout_macro(safe_collision_acc),
        "safe_vs_collision_pair_count": len(safe_collision_acc),
        "low_regret_vs_catastrophic_pair_accuracy": layout_macro(safe_cat_acc),
        "low_regret_vs_catastrophic_pair_count": len(safe_cat_acc),
        "success_vs_timeout_pair_accuracy": layout_macro(success_timeout_acc),
        "success_vs_timeout_pair_count": len(success_timeout_acc),
        "near_tied_safe_pair_count": near_safe_pairs,
        "collision_action_selection_rate": float(np.mean(selected_collision)),
        "selected_action_frequency": np.bincount(selected, minlength=ACTION_COUNT).tolist(),
    }


def _risk_metrics(collision_probability: np.ndarray, collision_target: np.ndarray,
                  layouts: np.ndarray, threshold: float) -> dict[str, Any]:
    probability_array = np.asarray(collision_probability, dtype=np.float64)
    target_array = np.asarray(collision_target, dtype=bool)
    if probability_array.shape != target_array.shape or probability_array.ndim != 2:
        raise ValueError("collision risk metrics expect aligned [states,candidates] arrays")
    p = probability_array.reshape(-1)
    y = target_array.reshape(-1)
    candidate_weights = np.repeat(layout_weights(layouts), probability_array.shape[1]) / probability_array.shape[1]
    predicted = p >= float(threshold)
    tp = float(candidate_weights[predicted & y].sum())
    fp = float(candidate_weights[predicted & ~y].sum())
    positive = float(candidate_weights[y].sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / positive if positive > 0 else None
    ap = _weighted_average_precision(p, y, candidate_weights)
    return {"threshold": float(threshold), "collision_recall": recall, "precision": precision,
            "false_negative_rate": (1.0 - recall) if recall is not None else None,
            "weighted_candidate_average_precision": ap,
            "predicted_collision_rate": float(candidate_weights[predicted].sum()),
            "true_collision_rate": positive, "collision_candidate_count": int(y.sum()),
            "candidate_count": int(len(y))}


def _weighted_average_precision(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float | None:
    s, y, w = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool), np.asarray(weights, dtype=np.float64)
    total_positive = float(w[y].sum())
    if total_positive <= 0.0:
        return None
    order = np.argsort(-s, kind="stable")
    y_sorted, w_sorted = y[order], w[order]
    true_positive = np.cumsum(w_sorted * y_sorted)
    total = np.cumsum(w_sorted)
    precision = true_positive / np.maximum(total, 1e-15)
    return float(np.sum(precision * w_sorted * y_sorted) / total_positive)


def _select_risk_threshold(probability: np.ndarray, target: np.ndarray, layouts: np.ndarray,
                           settings: dict[str, Any]) -> dict[str, Any]:
    records = [_risk_metrics(probability, target, layouts, float(threshold))
               for threshold in settings["thresholds"]]
    target_recall = float(settings["validation_collision_recall_target"])
    eligible = [record for record in records
                if record["collision_recall"] is not None and record["collision_recall"] >= target_recall]
    if eligible:
        selected = max(eligible, key=lambda item: item["threshold"])
        met = True
    else:
        selected = max(records, key=lambda item: (item["collision_recall"] or 0.0, item["threshold"]))
        met = False
    return {"selected_threshold": selected["threshold"], "validation_recall": selected["collision_recall"],
            "recall_target": target_recall, "recall_target_met": met,
            "selection_source": "D152_reused_validation_only", "threshold_sweep": records}


def _apply_risk_veto(utility: np.ndarray, collision_probability: np.ndarray,
                     threshold: float) -> tuple[np.ndarray, np.ndarray]:
    utility = np.asarray(utility, dtype=np.float64)
    risk = np.asarray(collision_probability, dtype=np.float64)
    if utility.shape != risk.shape or utility.ndim != 2 or utility.shape[1] < 1:
        raise ValueError("risk veto expects aligned [N,candidates] utility and risk matrices")
    rejected = risk >= float(threshold)
    selected = np.empty(len(utility), dtype=np.int64)
    for row in range(len(utility)):
        viable = np.flatnonzero(~rejected[row])
        if len(viable):
            selected[row] = viable[np.argmax(utility[row, viable])]
        else:
            minimum = np.min(risk[row])
            safest = np.flatnonzero(risk[row] == minimum)
            selected[row] = safest[np.argmax(utility[row, safest])]
    return selected, rejected


def _selected_policy_metrics(selected: np.ndarray, regret: np.ndarray, summary: np.ndarray,
                             layouts: np.ndarray, cfg: dict[str, Any],
                             risk: np.ndarray, rejected: np.ndarray) -> dict[str, Any]:
    idx = np.asarray(selected, dtype=np.int64)
    target = np.asarray(regret, dtype=np.float64)
    chosen_regret = target[np.arange(len(target)), idx]
    collision_labels = np.asarray(summary)[:, :, 3] > 0.5
    groups = np.asarray(layouts).astype(str)
    per_layout = [float(np.mean(chosen_regret[groups == layout])) for layout in sorted(set(groups.tolist()))]
    return {
        "layout_macro_normalized_selected_regret": float(np.mean(per_layout)),
        "state_micro_normalized_selected_regret": float(np.mean(chosen_regret)),
        "catastrophic_fraction_state_micro": float(np.mean(chosen_regret >= cfg["catastrophic_regret_threshold"])),
        "epsilon_optimal_rate": float(np.mean(chosen_regret <= cfg["epsilon_optimal_regret_threshold"] + 1e-12)),
        "best_action_set_agreement": float(np.mean(chosen_regret <= 1e-12)),
        "collision_action_selection_rate": float(np.mean(collision_labels[np.arange(len(idx)), idx])),
        "selected_action_frequency": np.bincount(idx, minlength=ACTION_COUNT).tolist(),
        "mean_predicted_risk_selected": float(np.mean(np.asarray(risk)[np.arange(len(idx)), idx])),
        "mean_vetoed_candidates_per_state": float(np.mean(np.sum(rejected, axis=1))),
        "all_candidates_vetoed_state_count": int(np.sum(np.all(rejected, axis=1))),
    }


def _load_d154_logits(path: Path, states: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        mean, std = np.asarray(saved["x_mean"], float), np.asarray(saved["x_std"], float)
        w1, b1 = np.asarray(saved["w1"], float), np.asarray(saved["b1"], float)
        w2, b2 = np.asarray(saved["w2"], float), np.asarray(saved["b2"], float)
        w3, b3 = np.asarray(saved["w3"], float), np.asarray(saved["b3"], float)
    if (mean.shape != (61,) or std.shape != (61,) or w1.shape != (61, 32)
            or w2.shape != (32, 32) or w3.shape != (32, 9)):
        raise D158Error("frozen D154 listwise checkpoint dimensions are invalid")
    h1 = np.maximum((states - mean) / std @ w1 + b1, 0.0)
    h2 = np.maximum(h1 @ w2 + b2, 0.0)
    return h2 @ w3 + b3


def _baseline_decisions(data: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    p = cfg["parents"]
    states, summary, layouts, target = (data["eval_r0"], data["eval_summary"],
                                        data["eval_layouts"], data["eval_regret"])
    stage_cfg = yaml.safe_load((ROOT / p["d155_stage_c_config"]).read_text(encoding="utf-8"))
    metrics_cfg = stage_cfg["metrics"]
    d150 = restore_decision_head(ROOT / p["d150_r0_head"])
    stage_a = _load_stage_a_head(ROOT / p["d155_stage_a_head"])
    predictors = load_predictors({"parents": {"stage_c_constant": p["d155_stage_c_constant"],
                                               "stage_c_cart": p["d155_stage_c_cart"],
                                               "stage_c_mlp": p["d155_stage_c_mlp"]}})
    action_prob, action_steps = _predict_cart(predictors["bounded_cart"], states)
    stage_feature_cases = _stage_a_features(states, summary, action_prob, action_steps)
    # Frozen D157 Stage-A feature assembly is independently asserted through stored metrics below.
    direct_score = d150.predict(states)
    baseline_scores = {
        "B0_direct_D150_R0": direct_score,
        "B1_D157_CART_predicted_then_frozen_StageA": -stage_a.predict_logits(stage_feature_cases["predicted"]),
        "B5_exact_compact_then_frozen_StageA": -stage_a.predict_logits(stage_feature_cases["exact"]),
        "B2_D154_direct_listwise_R0": -_load_d154_logits(ROOT / p["d154_listwise_r0"], states),
    }
    metrics = {name: _candidate_metrics(target, scores, summary, layouts, metrics_cfg)
               for name, scores in baseline_scores.items()}
    frozen_result = _read_json(ROOT / p["d157_correction_result"])
    parity = {
        "B0_direct_D150_R0": "direct_D150_R0",
        "B1_D157_CART_predicted_then_frozen_StageA": "bounded_cart_predicted_predicted",
        "B5_exact_compact_then_frozen_StageA": "exact_exact",
    }
    for ours, previous in parity.items():
        checks = ("layout_macro_normalized_selected_regret", "catastrophic_fraction_state_micro",
                  "best_action_set_agreement", "epsilon_optimal_rate")
        for key in checks:
            if abs(metrics[ours][key] - frozen_result["decision_metrics"][previous][key]) > 1e-10:
                raise D158Error(f"D158 baseline parity failed: {ours} {key}")
    return {"metrics": metrics, "scores": baseline_scores,
            "predicted_probabilities": action_prob, "predicted_steps": action_steps,
            "stage_a_feature_cases": stage_feature_cases, "baseline_parity": parity}


def _stage_a_features(states: np.ndarray, summary: np.ndarray,
                      predicted_prob: np.ndarray, predicted_steps: np.ndarray) -> dict[str, np.ndarray]:
    exact_prob, exact_steps = _probabilities_from_exact(summary)
    def assemble(prob: np.ndarray, steps: np.ndarray) -> np.ndarray:
        outcome4 = np.zeros((len(states), ACTION_COUNT, 4), dtype=np.float64)
        outcome4[:, :, 0] = prob[:, :, 0]
        outcome4[:, :, 2] = prob[:, :, 1]
        outcome4[:, :, 3] = prob[:, :, 2]
        return np.concatenate((states, outcome4.reshape(len(states), 36), steps.reshape(len(states), 9)), axis=1)
    return {"exact": assemble(exact_prob, exact_steps),
            "predicted": assemble(predicted_prob, predicted_steps)}


def _measure_latency(model: RelationalCandidateModel, states: np.ndarray, actions: np.ndarray,
                     consequences: np.ndarray, mask: tuple[float, float]) -> float:
    count = min(20, len(states))
    for idx in range(count):
        model.predict(states[idx:idx + 1], actions[idx:idx + 1], consequences[idx:idx + 1], mask)
    times = []
    for idx in range(count):
        start = time.perf_counter_ns()
        model.predict(states[idx:idx + 1], actions[idx:idx + 1], consequences[idx:idx + 1], mask)
        times.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(times))


def run_d158() -> dict[str, Any]:
    cfg, lock = validate_lock()
    started = time.perf_counter()
    out = ROOT / cfg["outputs"]["run_dir"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        data = _load_data(cfg)
        train_pred, train_steps, crossfit_audit = _crossfit_train_consequences(data, cfg)
        predictor = load_predictors({"parents": {"stage_c_constant": cfg["parents"]["d155_stage_c_constant"],
                                                  "stage_c_cart": cfg["parents"]["d155_stage_c_cart"],
                                                  "stage_c_mlp": cfg["parents"]["d155_stage_c_mlp"]}})["bounded_cart"]
        val_pred, val_steps = _predict_cart(predictor, data["val_r0"])
        eval_pred, eval_steps = _predict_cart(predictor, data["eval_r0"])
        train_actions, _ = _action_inputs(data["train_r0"])
        val_actions, _ = _action_inputs(data["val_r0"])
        eval_actions, _ = _action_inputs(data["eval_r0"])
        train_collision = data["train_summary"][:, :, 3]
        val_collision = data["val_summary"][:, :, 3]
        eval_targets = data["eval_summary"][:, :, 3]
        exact_val_prob, exact_val_steps = _probabilities_from_exact(data["val_summary"])
        exact_eval_prob, exact_eval_steps = _probabilities_from_exact(data["eval_summary"])

        baseline = _baseline_decisions(data, cfg)
        metrics_cfg = yaml.safe_load((ROOT / cfg["parents"]["d155_stage_c_config"]).read_text(encoding="utf-8"))["metrics"]
        data["val_selected_consequence"] = np.concatenate((val_pred, val_steps[:, :, None]), axis=2)
        data["val_selected_mask"] = (1.0, 1.0)
        predictor_inputs: dict[str, dict[str, Any]] = {}
        for rep_id in REPRESENTATION_IDS:
            if rep_id == "r0_action_exact_outcome_steps":
                t_prob, t_steps, v_prob, v_steps, e_prob, e_steps = (
                    _probabilities_from_exact(data["train_summary"])[0],
                    _probabilities_from_exact(data["train_summary"])[1],
                    exact_val_prob, exact_val_steps, exact_eval_prob, exact_eval_steps,
                )
            else:
                t_prob, t_steps, v_prob, v_steps, e_prob, e_steps = (
                    train_pred, train_steps, val_pred, val_steps, eval_pred, eval_steps)
            train_c, mask = _feature_set(t_prob, t_steps, rep_id)
            val_c, val_mask = _feature_set(v_prob, v_steps, rep_id)
            eval_c, eval_mask = _feature_set(e_prob, e_steps, rep_id)
            if mask != val_mask or mask != eval_mask:
                raise D158Error("D158 feature mask mismatch between split roles")
            predictor_inputs[rep_id] = {"train": train_c, "val": val_c, "eval": eval_c, "mask": mask}

        memorization = _memorization_gate(
            data, train_actions, predictor_inputs["r0_action_predicted_outcome_steps"],
            cfg["model"], metrics_cfg,
        )
        memorization_path = out / "memorization_gate.json"
        memorization_path.write_text(json.dumps(memorization, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
        if not memorization["passed"]:
            raise D158Error("D158 primary tiny memorization gate failed; full fit was not started")

        model_settings = cfg["model"]
        output_models: dict[str, RelationalCandidateModel] = {}
        fit_records: dict[str, Any] = {}
        all_specs = [(rep_id, "listwise_tau_025") for rep_id in REPRESENTATION_IDS]
        all_specs.extend(("r0_action_predicted_outcome_steps", obj)
                         for obj in ("margin_pairwise", "listwise_plus_pairwise"))
        for model_index, (rep_id, objective_id) in enumerate(all_specs):
            inp = predictor_inputs[rep_id]
            objective_cfg = next(item for item in model_settings["objective_comparison"]
                                 if item["id"] == objective_id)
            settings = _fit_settings(model_settings, cfg["targets"], objective_cfg)
            model = RelationalCandidateModel(
                model_settings["state_hidden"], int(model_settings["base_seed"]) + model_index,
                model_settings["learning_rate"], model_settings["l2"],
                model_settings["beta1"], model_settings["beta2"], model_settings["epsilon"],
            )
            fit = model.fit(
                data["train_r0"], train_actions, inp["train"], inp["mask"], data["train_regret"],
                train_collision, data["train_layouts"], data["val_r0"], val_actions, inp["val"],
                inp["mask"], data["val_regret"], val_collision, data["val_layouts"],
                objective_id, settings, metrics_cfg,
            )
            label = f"{rep_id}__{objective_id}"
            output_models[label] = model
            fit_records[label] = fit

        risk_settings = cfg["risk_policy"]
        result_models: dict[str, Any] = {}
        model_checkpoints: dict[str, str] = {}
        for label, model in output_models.items():
            rep_id, objective_id = label.split("__", 1)
            inp = predictor_inputs[rep_id]
            utility_train, risk_train = model.predict(data["train_r0"], train_actions, inp["train"], inp["mask"])
            utility_val, risk_val = model.predict(data["val_r0"], val_actions, inp["val"], inp["mask"])
            utility_eval, risk_eval = model.predict(data["eval_r0"], eval_actions, inp["eval"], inp["mask"])
            train_metrics = _candidate_metrics(data["train_regret"], -utility_train, data["train_summary"],
                                               data["train_layouts"], metrics_cfg)
            val_metrics = _candidate_metrics(data["val_regret"], -utility_val, data["val_summary"],
                                             data["val_layouts"], metrics_cfg)
            eval_metrics = _candidate_metrics(data["eval_regret"], -utility_eval, data["eval_summary"],
                                              data["eval_layouts"], metrics_cfg)
            threshold = _select_risk_threshold(risk_val, val_collision > 0.5, data["val_layouts"], risk_settings)
            eval_risk_metrics = _risk_metrics(risk_eval, eval_targets > 0.5, data["eval_layouts"],
                                              threshold["selected_threshold"])
            veto_choice, rejected = _apply_risk_veto(utility_eval, risk_eval, threshold["selected_threshold"])
            risk_policy_metrics = _selected_policy_metrics(
                veto_choice, data["eval_regret"], data["eval_summary"], data["eval_layouts"],
                cfg["metrics"], risk_eval, rejected,
            )
            latency = _measure_latency(model, data["eval_r0"], eval_actions, inp["eval"], inp["mask"])
            model_checkpoints[label] = _save_model(out / f"{label}.npz", model)
            result_models[label] = {
                "representation": rep_id, "objective": objective_id,
                "fit": fit_records[label], "parameter_count": model.parameter_count,
                "latency_ms_median_per_state": latency,
                "training": train_metrics, "validation": val_metrics,
                "heldout_D157_relational_only": eval_metrics,
                "heldout_D157_risk_aware_veto": risk_policy_metrics,
                "heldout_D157_collision_risk": eval_risk_metrics,
                "risk_threshold_selection": threshold,
            }
        gate_cfg = cfg["advancement_gate"]
        primary_label = gate_cfg["primary_model"]
        comparisons = {}
        # _candidate_metrics strips row-wise regrets; recompute aligned per-state vectors for paired bootstrap.
        primary_utility, _ = output_models[primary_label].predict(
            data["eval_r0"], eval_actions,
            predictor_inputs["r0_action_predicted_outcome_steps"]["eval"],
            predictor_inputs["r0_action_predicted_outcome_steps"]["mask"],
        )
        primary_chosen = np.argmax(primary_utility, axis=1)
        primary_regret = data["eval_regret"][np.arange(len(primary_chosen)), primary_chosen]
        primary_cat = (primary_regret >= cfg["metrics"]["catastrophic_regret_threshold"]).astype(float)
        baseline_pair_values = {}
        for baseline_name in ("B0_direct_D150_R0", "B1_D157_CART_predicted_then_frozen_StageA",
                              "B2_D154_direct_listwise_R0"):
            scores = baseline["scores"][baseline_name]
            chosen = np.argmin(scores, axis=1)
            regret_values = data["eval_regret"][np.arange(len(chosen)), chosen]
            cat_values = (regret_values >= cfg["metrics"]["catastrophic_regret_threshold"]).astype(float)
            baseline_pair_values[baseline_name] = (regret_values, cat_values)
            comparisons[f"primary_relational_vs_{baseline_name}"] = {
                "regret": _paired_layout_bootstrap(cat_values * 0 + regret_values, primary_regret,
                                                    data["eval_layouts"], statistic="layout_macro",
                                                    resamples=cfg["metrics"]["bootstrap_resamples"],
                                                    seed=cfg["metrics"]["bootstrap_seed"] + len(comparisons)),
                "catastrophic_fraction": _paired_layout_bootstrap(cat_values, primary_cat,
                                                                   data["eval_layouts"], statistic="state_micro",
                                                                   resamples=cfg["metrics"]["bootstrap_resamples"],
                                                                   seed=cfg["metrics"]["bootstrap_seed"] + len(comparisons) + 10),
            }
        primary_risk_policy = result_models[primary_label]["heldout_D157_risk_aware_veto"]
        risk_policy_gate = (
            result_models[primary_label]["heldout_D157_relational_only"]["catastrophic_fraction_state_micro"]
            - primary_risk_policy["catastrophic_fraction_state_micro"] >= cfg["metrics"]["risk_policy_minimum_catastrophic_reduction"]
            and primary_risk_policy["layout_macro_normalized_selected_regret"]
            <= result_models[primary_label]["heldout_D157_relational_only"]["layout_macro_normalized_selected_regret"]
               + cfg["metrics"]["risk_policy_maximum_regret_increase"]
        )
        compare_against = gate_cfg["compare_against"]
        baseline_better = max(
            baseline["metrics"][name]["safe_vs_collision_pair_accuracy"] or 0.0
            for name in compare_against
        )
        primary_eval = result_models[primary_label]["heldout_D157_relational_only"]
        gate = {
            "relational_materially_beats_R0_and_frozen_predicted_head": all(
                comparisons[f"primary_relational_vs_{baseline_name}"]["regret"]["point_improvement_left_minus_right"]
                    >= gate_cfg["minimum_macro_regret_improvement"]
                and comparisons[f"primary_relational_vs_{baseline_name}"]["catastrophic_fraction"]["point_improvement_left_minus_right"]
                    >= gate_cfg["minimum_catastrophic_fraction_improvement"]
                and comparisons[f"primary_relational_vs_{baseline_name}"]["regret"]["ci95_lower"] > 0.0
                and comparisons[f"primary_relational_vs_{baseline_name}"]["catastrophic_fraction"]["ci95_lower"] > 0.0
                for baseline_name in compare_against),
            "safe_collision_pair_gate": ((primary_eval["safe_vs_collision_pair_accuracy"] or 0.0)
                                         >= baseline_better + gate_cfg["safe_collision_pair_accuracy_minimum_gain_over_best_baseline"]),
            "risk_aware_policy_gate": bool(risk_policy_gate),
        }
        result = {
            "schema": "phase15_d158_relational_selection_result_v2",
            "status": "completed_exploratory_offline_candidate_selection_study",
            "source_commit": lock["source_commit"],
            "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
            "parent_sha256": lock["parent_sha256"], "split": data["split"],
            "crossfit_predictor_audit": crossfit_audit,
            "tiny_memorization": memorization,
            "baseline_metrics": baseline["metrics"],
            "models": result_models,
            "paired_layout_bootstrap": comparisons,
            "advancement_gates": gate,
            "scope": {"new_simulator_calls": 0, "new_planner_calls": 0,
                      "new_transitions_or_episodes": 0, "reserve_access": False,
                      "test_rows_dereferenced": 0, "student_closed_loop": False,
                      "dagger": False, "world_model_training": False, "jepa": False,
                      "iterative_refinement": False},
            "time_seconds": time.perf_counter() - started,
            "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        }
        for label, model in output_models.items():
            result_models[label]["checkpoint_sha256"] = model_checkpoints[label]
        result_path = out / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report_path = ROOT / cfg["outputs"]["report"]
        report_path.write_text(render_report(result), encoding="utf-8")
        manifest = {
            "schema": "phase15_d158_relational_selection_manifest_v2",
            "status": result["status"], "source_commit": result["source_commit"],
            "lock_sha256": result["lock_sha256"], "parent_sha256": result["parent_sha256"],
            "train_regret_sha256": data["split"]["train_regret_sha256"],
            "validation_regret_sha256": data["split"]["validation_regret_sha256"],
            "heldout_regret_sha256": data["split"]["heldout_regret_sha256"],
            "checkpoint_sha256": model_checkpoints,
            "result_sha256": sha256_file(result_path), "report_sha256": sha256_file(report_path),
            "new_simulator_calls": 0, "new_planner_calls": 0,
            "new_transitions_or_episodes": 0, "test_rows_dereferenced": 0,
            "reserve_access": False, "student_closed_loop": False,
            "model_training": True, "cross_fitted_predictor_training": True,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        abort = {"schema": "phase15_d158_abort_v2", "status": "aborted_or_incomplete",
                 "source_commit": lock["source_commit"], "lock_sha256": sha256_file(ROOT / cfg["outputs"]["lock"]),
                 "error_type": type(exc).__name__, "error": str(exc),
                 "new_simulator_calls": 0, "new_planner_calls": 0,
                 "reserve_access": False, "test_rows_dereferenced": 0,
                 "memorization_gate_sha256": (sha256_file(out / "memorization_gate.json")
                                              if (out / "memorization_gate.json").exists() else None),
                 "note": "Preserve partial output; do not rerun under this one-shot lock."}
        (out / "abort.json").write_text(json.dumps(abort, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


def _fit_settings(model_cfg: dict[str, Any], target_cfg: dict[str, Any], objective: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_epochs": model_cfg["max_epochs"], "patience": model_cfg["patience"],
        "listwise_temperature": target_cfg["listwise_temperature"],
        "pairwise_minimum_regret_gap": target_cfg["pairwise_minimum_regret_gap"],
        "safe_regret_at_most": target_cfg["safe_regret_at_most"],
        "catastrophic_regret_threshold": target_cfg["catastrophic_regret_threshold"],
        "pairwise_safe_collision_weight_multiplier": target_cfg["pairwise_safe_collision_weight_multiplier"],
        "pairwise_safe_catastrophic_weight_multiplier": target_cfg["pairwise_safe_catastrophic_weight_multiplier"],
        "pairwise_weight": objective["pairwise_weight"],
        "collision_loss_weight": model_cfg["collision_loss_weight"],
    }


def _memorization_gate(data: dict[str, Any], actions: np.ndarray,
                       input_record: dict[str, Any], model_cfg: dict[str, Any],
                       metrics_cfg: dict[str, Any]) -> dict[str, Any]:
    # Fixed first eight train rows; this is a software/fit sanity check, not a result-selection split.
    subset = np.arange(min(8, len(data["train_r0"])))
    if len(subset) != 8:
        return {"passed": False, "state_count": len(subset), "reason": "fewer than eight training states"}
    model = RelationalCandidateModel(model_cfg["state_hidden"], model_cfg["base_seed"] + 999,
                                     model_cfg["learning_rate"], model_cfg["l2"], model_cfg["beta1"],
                                     model_cfg["beta2"], model_cfg["epsilon"])
    inp = input_record["train"][subset]
    objective = model_cfg["objective_comparison"][0]
    settings = _fit_settings(model_cfg, _cfg()["targets"], objective)
    settings["max_epochs"] = 1600
    settings["patience"] = 220
    fit = model.fit(data["train_r0"][subset], actions[subset], inp, input_record["mask"],
                    data["train_regret"][subset], data["train_summary"][subset, :, 3],
                    data["train_layouts"][subset], data["train_r0"][subset], actions[subset],
                    inp, input_record["mask"], data["train_regret"][subset],
                    data["train_summary"][subset, :, 3], data["train_layouts"][subset],
                    "listwise_tau_025", settings, metrics_cfg)
    utility, risk = model.predict(data["train_r0"][subset], actions[subset], inp, input_record["mask"])
    pred_action = np.argmax(utility, axis=1)
    true_action = np.argmin(data["train_regret"][subset], axis=1)
    regret = data["train_regret"][subset][np.arange(len(subset)), pred_action]
    best_agreement = _optimal_set_agreement(data["train_regret"][subset], pred_action)
    mean_regret = float(np.mean(regret))
    risk_label = data["train_summary"][subset, :, 3]
    risk_accuracy = float(np.mean((risk >= 0.5) == (risk_label > 0.5)))
    return {"passed": bool(best_agreement >= 0.99 and mean_regret <= 0.01),
            "state_count": len(subset), "training_optimal_set_agreement": best_agreement,
            "training_first_argmin_action_agreement_diagnostic": float(np.mean(pred_action == true_action)),
            "training_mean_selected_regret": mean_regret, "risk_binary_accuracy_diagnostic": risk_accuracy,
            "fit": fit, "used_eval_rows": False}


def _optimal_set_agreement(regret: np.ndarray, selected: np.ndarray, tolerance: float = 1e-12) -> float:
    values = np.asarray(regret, dtype=np.float64)
    actions = np.asarray(selected, dtype=np.int64)
    if values.ndim != 2 or actions.shape != (len(values),):
        raise ValueError("optimal-set agreement expects regrets [N,K] and selected actions [N]")
    minima = np.min(values, axis=1, keepdims=True)
    optimal = values <= minima + float(tolerance)
    return float(np.mean(optimal[np.arange(len(values)), actions]))


def _save_model(path: Path, model: RelationalCandidateModel) -> str:
    np.savez_compressed(path, **{name: getattr(model, name) for name in
                                 ("x_mean", "x_std", "ws", "bs", "wc", "bc", "wctx", "bctx",
                                  "wscore", "bscore", "wrisk", "brisk")})
    return sha256_file(path)


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# D158 — Decision-aligned relational candidate selection", "",
        "Exploratory offline mechanism study. D157 remains closed. D157 labels are reused only as this phase's held-out development evaluation; they are not a final test. No new simulator/planner calls, transitions, episodes, DAgger, closed-loop control, reserve/test access, or world-model training occurred.", "",
        f"Train: {result['split']['train_rows']} states / {result['split']['train_layout_count']} layouts. "
        f"D152 reused validation: {result['split']['d152_validation_rows']} / {result['split']['d152_validation_layout_count']}. "
        f"D157 exploratory held-out: {result['split']['d157_heldout_rows']} / {result['split']['d157_heldout_layout_count']}.", "",
        f"Tiny memorization gate: **{result['tiny_memorization']['passed']}**; optimal-set agreement={result['tiny_memorization'].get('training_optimal_set_agreement')}, mean regret={result['tiny_memorization'].get('training_mean_selected_regret')}.", "",
        "## Frozen and learned decision metrics", "",
        "| Method | Macro regret ↓ | Catastrophic ↓ | Epsilon-optimal ↑ | Best-action agreement ↑ | Safe-vs-collision ↑ | Collision action rate ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result["baseline_metrics"].items():
        lines.append(_metric_row(name, metrics))
    for key, model in result["models"].items():
        lines.append(_metric_row(key + " [heldout]", model["heldout_D157_relational_only"]))
    lines.extend(["", "## Risk heads and veto policy", "",
                  "| Model | Threshold (validation-selected) | Recall | Precision | FNR | Weighted AP | Veto regret | Veto catastrophic | Selected collision rate |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for key, model in result["models"].items():
        risk = model["heldout_D157_collision_risk"]
        veto = model["heldout_D157_risk_aware_veto"]
        lines.append(f"| {key} | {risk['threshold']:.3f} | {risk['collision_recall']:.4f} | {risk['precision']:.4f} | "
                     f"{risk['false_negative_rate']:.4f} | {risk['weighted_candidate_average_precision']:.4f} | "
                     f"{veto['layout_macro_normalized_selected_regret']:.5f} | "
                     f"{veto['catastrophic_fraction_state_micro']:.4f} | {veto['collision_action_selection_rate']:.4f} |")
    lines.extend(["", "## Pairwise hard negatives", "",
                  "Counts report available within-state action pairs; near-tied safe pairs are not used as hard ranking targets.", "",
                  "| Model | Safe-vs-collision accuracy (pairs) | Low-regret-vs-catastrophic accuracy (pairs) | Success-vs-timeout accuracy (pairs) | Near-tied safe pairs |",
                  "|---|---:|---:|---:|---:|"])
    for name, metrics in result["baseline_metrics"].items():
        lines.append(_pair_row(name, metrics))
    for name, model in result["models"].items():
        lines.append(_pair_row(name, model["heldout_D157_relational_only"]))
    lines.extend(["", "## Component and objective comparisons", "",
                  "The four ablations use listwise training; exact outcome+steps is an oracle-information diagnostic. Pairwise and combined losses are additional predicted-both runs.", "",
                  "| Representation / objective | Macro regret | Catastrophic | Safe-vs-collision | Risk AP | Latency ms/state | Parameters |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name, model in result["models"].items():
        metric, risk = model["heldout_D157_relational_only"], model["heldout_D157_collision_risk"]
        lines.append(f"| {name} | {metric['layout_macro_normalized_selected_regret']:.5f} | "
                     f"{metric['catastrophic_fraction_state_micro']:.4f} | {metric['safe_vs_collision_pair_accuracy']} | "
                     f"{risk['weighted_candidate_average_precision']:.4f} | "
                     f"{model['latency_ms_median_per_state']:.4f} | {model['parameter_count']} |")
    lines.extend(["", "## Paired layout bootstrap", "",
                  "Positive baseline-minus-model improvement means the learned candidate policy is better.", "",
                  "| Comparison | Metric | Improvement | 95% CI |", "|---|---|---:|---:|"])
    for comparison, fields in result["paired_layout_bootstrap"].items():
        for metric, value in fields.items():
            lines.append(f"| {comparison} | {metric} | {value['point_improvement_left_minus_right']:.5f} | "
                         f"[{value['ci95_lower']:.5f}, {value['ci95_upper']:.5f}] |")
    lines.extend(["", "## Preregistered gates", "",
                  f"Relational primary gate: **{result['advancement_gates']['relational_materially_beats_R0_and_frozen_predicted_head']}**. "
                  f"Safe-vs-collision gate: **{result['advancement_gates']['safe_collision_pair_gate']}**. "
                  f"Risk-veto gate: **{result['advancement_gates']['risk_aware_policy_gate']}**.", "",
                  "## Interpretation and limits", "",
                  "This study cannot establish a faithful reproduction of D-JEPA/MCR, causal component importance, or final generalization: D152 validation is reused and D157 has prior frozen-reference scoring. Full checkpoint metrics, fit records, cross-fit fold identities, bootstrap results, and input hashes are in the structured run result/manifest.", "",
                  "No DAgger, closed-loop student control, reserve/test evaluation, world-model/JEPA training, or iterative refinement is started by this report.", ""])
    return "\n".join(lines)


def _metric_row(name: str, m: dict[str, Any]) -> str:
    return (f"| {name} | {m['layout_macro_normalized_selected_regret']:.5f} | "
            f"{m['catastrophic_fraction_state_micro']:.4f} | {m['epsilon_optimal_rate']:.4f} | "
            f"{m['best_action_set_agreement']:.4f} | {m['safe_vs_collision_pair_accuracy']} | "
            f"{m['collision_action_selection_rate']:.4f} |")


def _pair_row(name: str, m: dict[str, Any]) -> str:
    return (f"| {name} | {m['safe_vs_collision_pair_accuracy']} ({m['safe_vs_collision_pair_count']}) | "
            f"{m['low_regret_vs_catastrophic_pair_accuracy']} ({m['low_regret_vs_catastrophic_pair_count']}) | "
            f"{m['success_vs_timeout_pair_accuracy']} ({m['success_vs_timeout_pair_count']}) | "
            f"{m['near_tied_safe_pair_count']} |")
