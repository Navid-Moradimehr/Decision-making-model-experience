"""Training-only Phase14C integrity, feature, and memorization study."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from math import hypot, sqrt
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import yaml

from consequence_lab.inertial_corridor import ACTION_VECTORS
from consequence_lab.phase14 import replay_training_states, sha256_file
from consequence_lab.phase14b_resume import (
    read_jsonl,
    validate_continuation_lock,
    validate_record_range,
)


CONFIG_PATH = "configs/phase14c.yaml"
LOCK_PATH = "experiments/phase14c_memorization_v2.lock.json"
SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "configs/phase14.yaml",
    "configs/phase14b.yaml",
    "configs/phase14b_resume.yaml",
    CONFIG_PATH,
    "experiments/phase14b_training_labels_v1.lock.json",
    "experiments/phase14b_training_labels_resume_v1.lock.json",
    "reports/phase14_first_action_value_learnability_protocol.md",
    "reports/phase14b_full_training_labels_protocol.md",
    "reports/phase14b_full_training_labels_result.md",
    "reports/phase14b_resume_protocol.md",
    "reports/phase14b_resume_result.md",
    "reports/phase14c_memorization_protocol.md",
    "reports/phase14c_v1_pre_fit_runner_failure.md",
    "experiments/phase14c_memorization_v1.lock.json",
    "artifacts/runs/phase14c_memorization_v1/manifest.json",
    "artifacts/runs/phase14_training_source_audit_v2.json",
    "src/consequence_lab/inertial_corridor.py",
    "src/consequence_lab/phase14.py",
    "src/consequence_lab/phase14b.py",
    "src/consequence_lab/phase14b_resume.py",
    "src/consequence_lab/phase14c.py",
    "scripts/freeze_phase14b_resume_lock.py",
    "scripts/run_phase14b_resume.py",
    "scripts/freeze_phase14c_lock.py",
    "scripts/run_phase14c_memorization.py",
    "tests/test_phase14b_resume.py",
    "tests/test_phase14c.py",
)


class Phase14CError(RuntimeError):
    """Phase14C source, split, input, or fitting contract failure."""


def _validate_config(root: Path, config: dict[str, Any]) -> None:
    if config.get("schema") != "phase14c_config_v2":
        raise Phase14CError("unsupported Phase14C config schema")
    authorization = config.get("authorization", {})
    if (authorization.get("stage") != "synthetic_integrity_and_same_data_memorization_only"
            or authorization.get("student_training") is not True
            or authorization.get("training_split_only") is not True
            or any(authorization.get(key) is not False for key in (
                "heldout_access", "dagger", "world_model", "reserve_access"
            ))):
        raise Phase14CError("Phase14C config opens an unauthorized stage or data split")
    resources = config.get("resources", {})
    if resources.get("wall_time_cap") is not None or resources.get("generated_storage_cap") is not None:
        raise Phase14CError("Phase14C must preserve the user's uncapped time/storage authorization")
    if resources.get("device") != "cpu" or resources.get("overwrite_forbidden") is not True:
        raise Phase14CError("Phase14C resource/output policy differs from its reviewed protocol")


def _git_state(root: Path) -> tuple[str, str]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Phase14CError("cannot verify Phase14C Git state") from exc
    return head, dirty


def _validate_dataset(root: Path, config: dict[str, Any]) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    _validate_config(root, config)
    lock, sources, _ = validate_continuation_lock(
        root, yaml.safe_load((root / "configs/phase14b_resume.yaml").read_text(encoding="utf-8"))
    )
    src = config["source"]
    records_path = root / src["full_training_records"]
    manifest_path = root / src["full_training_manifest"]
    continuation_lock_path = root / src["continuation_lock"]
    if lock["source_commit"] != "41cc8dff0d45e7116ee47d553d94c2991a849469":
        raise Phase14CError("Phase14B continuation source commit changed")
    if sha256_file(continuation_lock_path) != src["continuation_lock_sha256"]:
        raise Phase14CError("continuation lock hash differs from Phase14C config")
    if sha256_file(root / src["failed_phase14c_v1_lock"]) != src["failed_phase14c_v1_lock_sha256"]:
        raise Phase14CError("preserved Phase14C v1 lock hash differs")
    failed_manifest_path = root / src["failed_phase14c_v1_manifest"]
    if sha256_file(failed_manifest_path) != src["failed_phase14c_v1_manifest_sha256"]:
        raise Phase14CError("preserved Phase14C v1 pre-fit failure manifest hash differs")
    failed_manifest = json.loads(failed_manifest_path.read_text(encoding="utf-8"))
    if (failed_manifest.get("status") != "aborted_or_incomplete"
            or failed_manifest.get("error_type") != "KeyError"
            or failed_manifest.get("error") != "'synthetic_seed'"):
        raise Phase14CError("Phase14C v1 attempt no longer matches its pre-fit software failure")
    if sha256_file(records_path) != src["full_training_records_sha256"]:
        raise Phase14CError("completed training records hash differs from Phase14C config")
    if sha256_file(manifest_path) != src["full_training_manifest_sha256"]:
        raise Phase14CError("completed training manifest hash differs from Phase14C config")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema": "phase14b_completed_training_values_v2",
        "status": "completed_teacher_values_gate_passed",
        "integrity_passed": True,
        "informativeness_gate_passed_on_full_train": True,
        "source_decision_states": int(src["expected_states"]),
        "forced_action_records": int(src["expected_action_values"]),
        "student_training": False,
        "heldout_access": False,
        "dagger": False,
        "world_model": False,
        "reserve_accessed": False,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise Phase14CError("Phase14B continuation manifest does not satisfy train-data gate")
    if (manifest.get("records_sha256") != src["full_training_records_sha256"]
            or manifest.get("records_bytes") != records_path.stat().st_size):
        raise Phase14CError("completed label manifest does not describe the frozen records artifact")
    rows = read_jsonl(records_path)
    if len(rows) != int(src["expected_states"]):
        raise Phase14CError("completed label rows differ from expected training state count")
    targets = validate_record_range(rows, sources, 0, len(sources))
    if sum(len(action_set) for action_set in targets) != int(src["expected_action_values"]):
        raise Phase14CError("completed action-value cardinality differs from frozen contract")
    return sources, rows, manifest


def create_training_lock(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    _validate_config(root, config)
    lock_path = root / config["outputs"]["lock"]
    if lock_path.exists():
        raise Phase14CError("refusing to overwrite Phase14C training lock")
    head, dirty = _git_state(root)
    if dirty:
        raise Phase14CError("commit Phase14C protocol/source/tests/dependencies before freezing lock")
    _, _, manifest = _validate_dataset(root, config)
    packages = {"numpy": importlib.metadata.version("numpy")}
    lock = {
        "schema": "phase14c_memorization_lock_v2",
        "status": "authorized_phase14c_train_split_memorization_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approval_reference": "Phase14 authorization to proceed through evidence gates; Phase14B full-label and informativeness gates passed; user removed time/storage caps 2026-09-29",
        "source_commit": head,
        "source_code_sha256": {name: sha256_file(root / name) for name in SOURCE_FILES},
        "config_sha256": sha256_file(root / CONFIG_PATH),
        "phase14b_continuation_manifest_sha256": sha256_file(root / config["source"]["full_training_manifest"]),
        "phase14b_records_sha256": sha256_file(root / config["source"]["full_training_records"]),
        "phase14b_continuation_lock_sha256": sha256_file(root / config["source"]["continuation_lock"]),
        "failed_phase14c_v1_lock_sha256": sha256_file(root / config["source"]["failed_phase14c_v1_lock"]),
        "failed_phase14c_v1_manifest_sha256": sha256_file(root / config["source"]["failed_phase14c_v1_manifest"]),
        "source_decision_states": int(manifest["source_decision_states"]),
        "action_values": int(manifest["forced_action_records"]),
        "dependencies": packages,
        "authorized_stage": "synthetic_integrity_and_same_data_memorization_only",
        "student_training": True,
        "training_split_only": True,
        "heldout_access": False,
        "dagger": False,
        "world_model": False,
        "reserve_access": False,
        "resources": config["resources"],
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return lock


def validate_training_lock(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]], dict[str, Any]]:
    root = Path(root).resolve()
    _validate_config(root, config)
    lock_path = root / config["outputs"]["lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase14c_memorization_lock_v2" or lock.get("status") != "authorized_phase14c_train_split_memorization_only":
        raise Phase14CError("Phase14C lock does not authorize its training-only stage")
    for name, expected in {
        "student_training": True,
        "training_split_only": True,
        "heldout_access": False,
        "dagger": False,
        "world_model": False,
        "reserve_access": False,
    }.items():
        if lock.get(name) is not expected:
            raise Phase14CError(f"Phase14C scope flag mismatch: {name}")
    if lock.get("config_sha256") != sha256_file(root / CONFIG_PATH):
        raise Phase14CError("Phase14C config hash mismatch")
    if lock.get("source_code_sha256") != {name: sha256_file(root / name) for name in SOURCE_FILES}:
        raise Phase14CError("Phase14C source code hash mismatch")
    if lock.get("resources") != config["resources"]:
        raise Phase14CError("Phase14C resource declaration differs from config")
    if lock.get("dependencies") != {"numpy": importlib.metadata.version("numpy")}:
        raise Phase14CError("Phase14C dependency versions changed after lock")
    head, dirty = _git_state(root)
    if head != lock.get("source_commit"):
        try:
            subprocess.check_call(
                ["git", "merge-base", "--is-ancestor", lock["source_commit"], "HEAD"],
                cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Phase14CError("Phase14C source commit is not an ancestor") from exc
    if dirty:
        raise Phase14CError("working tree must be clean after committing Phase14C lock")
    sources, rows, manifest = _validate_dataset(root, config)
    if (lock.get("phase14b_continuation_manifest_sha256") != sha256_file(root / config["source"]["full_training_manifest"])
            or lock.get("phase14b_records_sha256") != sha256_file(root / config["source"]["full_training_records"])
            or lock.get("phase14b_continuation_lock_sha256") != sha256_file(root / config["source"]["continuation_lock"])
            or lock.get("failed_phase14c_v1_lock_sha256") != sha256_file(root / config["source"]["failed_phase14c_v1_lock"])
            or lock.get("failed_phase14c_v1_manifest_sha256") != sha256_file(root / config["source"]["failed_phase14c_v1_manifest"])):
        raise Phase14CError("Phase14B teacher dataset provenance differs from Phase14C lock")
    return lock, sources, rows, manifest


def physics_features(raw: np.ndarray) -> np.ndarray:
    """Derive nine deterministic control features from the 52 public values only."""
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (52,) or not np.isfinite(values).all():
        raise ValueError("raw deployable feature row must contain 52 finite values")
    width, height = values[4], values[5]
    if min(width, height, values[13], values[14]) <= 0:
        raise ValueError("invalid public dimensions or dynamics parameters")
    px, py = values[0] * width, values[1] * height
    gx, gy = values[6] * width, values[7] * height
    dx, dy = gx - px, gy - py
    distance = hypot(dx, dy)
    diagonal = hypot(width, height)
    max_speed = values[13]
    vx, vy = values[2] * max_speed, values[3] * max_speed
    speed = hypot(vx, vy)
    goal_x, goal_y = (dx / distance, dy / distance) if distance > 1e-12 else (0.0, 0.0)
    forward = vx * goal_x + vy * goal_y
    lateral = -vx * goal_y + vy * goal_x
    alignment = forward / speed if speed > 1e-12 else 0.0

    clearances = []
    radius = values[10]
    for obstacle in values[16:].reshape(9, 4):
        xmin, ymin, xmax, ymax = obstacle
        qx, qy = px / width, py / height
        ox = max(xmin - qx, 0.0, qx - xmax) * width
        oy = max(ymin - qy, 0.0, qy - ymax) * height
        clearances.append(hypot(ox, oy) - radius)
    nearest_clearance = min(clearances) / diagonal

    acceleration = values[11] * max_speed
    damping = values[12]
    safe_speed = values[9] * max_speed
    dt = values[14]
    stop_distance = 0.0
    brake_vx, brake_vy = vx, vy
    for _ in range(128):
        if hypot(brake_vx, brake_vy) <= safe_speed:
            break
        best: tuple[float, int, float, float] | None = None
        for action_id, (ax, ay) in enumerate(ACTION_VECTORS):
            if ax and ay:
                ax /= sqrt(2.0)
                ay /= sqrt(2.0)
            nvx = damping * brake_vx + acceleration * ax
            nvy = damping * brake_vy + acceleration * ay
            next_speed = hypot(nvx, nvy)
            if next_speed > max_speed:
                scale = max_speed / next_speed
                nvx, nvy = nvx * scale, nvy * scale
                next_speed = max_speed
            item = (next_speed * next_speed, action_id, nvx, nvy)
            if best is None or item[:2] < best[:2]:
                best = item
        assert best is not None
        brake_vx, brake_vy = best[2], best[3]
        stop_distance += hypot(brake_vx, brake_vy) * dt
    derived = np.asarray((
        dx / width,
        dy / height,
        distance / diagonal,
        speed / max_speed,
        forward / max_speed,
        lateral / max_speed,
        alignment,
        nearest_clearance,
        stop_distance / diagonal,
    ), dtype=np.float64)
    if derived.shape != (9,) or not np.isfinite(derived).all():
        raise ValueError("physics-derived feature contract violated")
    return derived


def build_feature_matrix(raw: np.ndarray, variant: str) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 52 or not np.isfinite(raw).all():
        raise ValueError("raw feature matrix must have shape [N,52] and be finite")
    if variant == "raw52":
        return raw.copy()
    if variant == "raw52_plus_physics9":
        extras = np.stack([physics_features(row) for row in raw])
        return np.concatenate((raw, extras), axis=1)
    raise ValueError(f"unknown Phase14C feature variant: {variant}")


def rung_indices(rows: list[dict[str, Any]], source_audit: dict[str, Any]) -> dict[str, list[int]]:
    keys = source_audit["diagnostic_state_keys"]
    indices = [int(row["global_query_index"]) for row in keys]
    if len(indices) != 32 or len(set(indices)) != 32:
        raise Phase14CError("source audit does not provide 32 distinct diagnostic query keys")
    one_seed = int(source_audit["one_layout_seed"])
    one_layout = [index for index, row in enumerate(rows) if int(row["source_episode_seed"]) == one_seed]
    if not one_layout:
        raise Phase14CError("frozen one-layout group absent from full training targets")
    return {
        "diagnostic_1": indices[:1],
        "diagnostic_8": indices[:8],
        "diagnostic_32": indices,
        "one_layout": one_layout,
        "full_training": list(range(len(rows))),
    }


def extract_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray([row["student_input_52"] for row in rows], dtype=np.float64)
    regrets = np.asarray([
        [float(action["normalized_regret"]) for action in row["actions"]]
        for row in rows
    ], dtype=np.float64)
    objective = np.asarray([
        [action["objective_tuple"] for action in row["actions"]]
        for row in rows
    ], dtype=np.float64)
    if raw.shape != (len(rows), 52) or regrets.shape != (len(rows), 9) or objective.shape != (len(rows), 9, 3):
        raise Phase14CError("training input/target matrix shape mismatch")
    if not (np.isfinite(raw).all() and np.isfinite(regrets).all() and np.isfinite(objective).all()):
        raise Phase14CError("training input/target contains non-finite values")
    return raw, regrets, objective


class _OneNearestNeighbor:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "_OneNearestNeighbor":
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.y = np.asarray(y, dtype=np.float64).copy()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        query = np.asarray(x, dtype=np.float64)
        distances = np.sum((query[:, None, :] - self.x[None, :, :]) ** 2, axis=2)
        return self.y[np.argmin(distances, axis=1)]


class _TreeNode:
    __slots__ = ("value", "feature", "threshold", "left", "right")

    def __init__(self, value: np.ndarray, feature: int = -1, threshold: float = 0.0,
                 left: "_TreeNode | None" = None, right: "_TreeNode | None" = None) -> None:
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class _SquaredErrorTree:
    """Small deterministic multi-output CART with exhaustive binary thresholds."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_SquaredErrorTree":
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.node_count = 1
        all_rows = np.arange(len(self.x), dtype=np.int64)
        self.root = _TreeNode(self.y.mean(axis=0))
        stack: list[tuple[_TreeNode, np.ndarray]] = [(self.root, all_rows)]
        while stack:
            node, indices = stack.pop()
            split = self._best_split(indices)
            if split is None:
                continue
            feature, threshold, left_ids, right_ids = split
            node.feature = feature
            node.threshold = threshold
            left_node = _TreeNode(self.y[left_ids].mean(axis=0))
            right_node = _TreeNode(self.y[right_ids].mean(axis=0))
            node.left = left_node
            node.right = right_node
            self.node_count += 2
            stack.append((right_node, right_ids))
            stack.append((left_node, left_ids))
        return self

    def _best_split(self, indices: np.ndarray) -> tuple[int, float, np.ndarray, np.ndarray] | None:
        target = self.y[indices]
        mean = target.mean(axis=0)
        if len(indices) <= 1 or np.all(target == target[0]):
            return None
        total_sse = float(np.sum((target - mean) ** 2))
        if total_sse <= 1e-14:
            return None
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        for feature in range(self.x.shape[1]):
            order = np.argsort(self.x[indices, feature], kind="mergesort")
            sorted_indices = indices[order]
            x_sorted = self.x[sorted_indices, feature]
            boundaries = np.flatnonzero(x_sorted[:-1] < x_sorted[1:]) + 1
            if len(boundaries) == 0:
                continue
            sorted_y = self.y[sorted_indices]
            prefix_sum = np.cumsum(sorted_y, axis=0)
            prefix_sq = np.cumsum(sorted_y * sorted_y, axis=0)
            left_sum = prefix_sum[boundaries - 1]
            left_sq = prefix_sq[boundaries - 1]
            right_sum = prefix_sum[-1] - left_sum
            right_sq = prefix_sq[-1] - left_sq
            left_count = boundaries[:, None]
            right_count = (len(indices) - boundaries)[:, None]
            left_sse = np.sum(left_sq - left_sum * left_sum / left_count, axis=1)
            right_sse = np.sum(right_sq - right_sum * right_sum / right_count, axis=1)
            losses = left_sse + right_sse
            best_offset = int(np.argmin(losses))
            split = int(boundaries[best_offset])
            loss = float(losses[best_offset])
            threshold = float(x_sorted[split - 1] + (x_sorted[split] - x_sorted[split - 1]) / 2.0)
            candidate = (loss, feature, threshold, sorted_indices[:split], sorted_indices[split:])
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is None or best[0] >= total_sse - 1e-14:
            return None
        _, feature, threshold, left_ids, right_ids = best
        return feature, threshold, left_ids, right_ids

    def predict(self, x: np.ndarray) -> np.ndarray:
        rows = np.asarray(x, dtype=np.float64)
        result = np.empty((len(rows), self.y.shape[1]), dtype=np.float64)
        for row_index, row in enumerate(rows):
            node = self.root
            while node.feature >= 0:
                assert node.left is not None and node.right is not None
                node = node.left if row[node.feature] <= node.threshold else node.right
            result[row_index] = node.value
        return result


class _TinyMLP:
    def __init__(self, seed: int, hidden: tuple[int, int] = (64, 64), learning_rate: float = 1e-3,
                 alpha: float = 1e-4, batch_size: int = 64, max_epochs: int = 3000,
                 tolerance: float = 1e-6, patience: int = 200) -> None:
        self.seed = seed
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.patience = patience

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_TinyMLP":
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        dimensions = (self.x.shape[1], *self.hidden, self.y.shape[1])
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / left), size=(left, right))
            for left, right in zip(dimensions[:-1], dimensions[1:], strict=True)
        ]
        self.biases = [np.zeros((1, width), dtype=np.float64) for width in dimensions[1:]]
        mw = [np.zeros_like(value) for value in self.weights]
        vw = [np.zeros_like(value) for value in self.weights]
        mb = [np.zeros_like(value) for value in self.biases]
        vb = [np.zeros_like(value) for value in self.biases]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        step, stale, best_loss = 0, 0, float("inf")
        self.n_iter_ = 0
        n = len(self.x)
        for epoch in range(self.max_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                ids = order[start:start + self.batch_size]
                xb, yb = self.x[ids], self.y[ids]
                activations = [xb]
                preactivations = []
                for layer in range(len(self.weights) - 1):
                    z = activations[-1] @ self.weights[layer] + self.biases[layer]
                    preactivations.append(z)
                    activations.append(np.maximum(z, 0.0))
                prediction = activations[-1] @ self.weights[-1] + self.biases[-1]
                delta = 2.0 * (prediction - yb) / max(1, len(ids))
                grad_w = [np.empty_like(value) for value in self.weights]
                grad_b = [np.empty_like(value) for value in self.biases]
                grad_w[-1] = activations[-1].T @ delta + self.alpha * self.weights[-1] / n
                grad_b[-1] = delta.sum(axis=0, keepdims=True)
                for layer in range(len(self.weights) - 2, -1, -1):
                    delta = (delta @ self.weights[layer + 1].T) * (preactivations[layer] > 0.0)
                    grad_w[layer] = activations[layer].T @ delta + self.alpha * self.weights[layer] / n
                    grad_b[layer] = delta.sum(axis=0, keepdims=True)
                step += 1
                for layer in range(len(self.weights)):
                    mw[layer] = beta1 * mw[layer] + (1.0 - beta1) * grad_w[layer]
                    vw[layer] = beta2 * vw[layer] + (1.0 - beta2) * grad_w[layer] ** 2
                    mb[layer] = beta1 * mb[layer] + (1.0 - beta1) * grad_b[layer]
                    vb[layer] = beta2 * vb[layer] + (1.0 - beta2) * grad_b[layer] ** 2
                    corrected_mw = mw[layer] / (1.0 - beta1**step)
                    corrected_vw = vw[layer] / (1.0 - beta2**step)
                    corrected_mb = mb[layer] / (1.0 - beta1**step)
                    corrected_vb = vb[layer] / (1.0 - beta2**step)
                    self.weights[layer] -= self.learning_rate * corrected_mw / (np.sqrt(corrected_vw) + epsilon)
                    self.biases[layer] -= self.learning_rate * corrected_mb / (np.sqrt(corrected_vb) + epsilon)
            self.n_iter_ = epoch + 1
            loss = float(np.mean((self.predict(self.x) - self.y) ** 2))
            if best_loss - loss > self.tolerance:
                best_loss, stale = loss, 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.loss_ = best_loss
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        activation = np.asarray(x, dtype=np.float64)
        for layer in range(len(self.weights) - 1):
            activation = np.maximum(activation @ self.weights[layer] + self.biases[layer], 0.0)
        return activation @ self.weights[-1] + self.biases[-1]


def create_estimator(family: str, random_state: int):
    if family == "nearest_neighbor":
        return _OneNearestNeighbor()
    if family == "decision_tree":
        return _SquaredErrorTree()
    if family == "tiny_mlp":
        return _TinyMLP(seed=random_state)
    raise ValueError(f"unknown learner family {family}")


def model_parameter_count(family: str, estimator: Any, input_dim: int, output_dim: int) -> int:
    if family == "tiny_mlp" and hasattr(estimator, "weights"):
        return int(sum(value.size for value in estimator.weights) + sum(value.size for value in estimator.biases))
    if family == "decision_tree" and hasattr(estimator, "node_count"):
        return int(estimator.node_count)
    if family == "nearest_neighbor":
        return 0
    return int(input_dim * output_dim)


def _q_target_scale(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pooled = q.reshape(-1, 3)
    mean = pooled.mean(axis=0)
    std = np.maximum(pooled.std(axis=0), 1e-8)
    return ((q - mean[None, None, :]) / std[None, None, :]).reshape(len(q), 27), mean, std


def clear_pair_order_accuracy(
    target_tuples: np.ndarray, predicted: np.ndarray, is_q: bool,
) -> tuple[float, int]:
    correct = 0
    total = 0
    for state_index in range(len(target_tuples)):
        target = target_tuples[state_index]
        for left in range(9):
            for right in range(left + 1, 9):
                lt, rt = target[left], target[right]
                if lt[0] != rt[0]:
                    expected = -1 if lt[0] < rt[0] else 1
                else:
                    tolerance = 1.0 if lt[0] == 0 else 0.01
                    if abs(lt[1] - rt[1]) <= tolerance:
                        continue
                    expected = -1 if lt[1] < rt[1] else 1
                if is_q:
                    lp, rp = predicted[state_index, left], predicted[state_index, right]
                    observed = -1 if tuple(lp) < tuple(rp) else (1 if tuple(lp) > tuple(rp) else 0)
                else:
                    lp, rp = predicted[state_index, left], predicted[state_index, right]
                    observed = -1 if lp < rp else (1 if lp > rp else 0)
                total += 1
                correct += int(observed == expected)
    return (correct / total if total else 1.0), total


def evaluate_predictions(
    family: str,
    predicted_y: np.ndarray,
    true_regret: np.ndarray,
    true_tuples: np.ndarray,
    q_mean: np.ndarray | None,
    q_std: np.ndarray | None,
) -> dict[str, Any]:
    n = len(true_regret)
    if q_mean is None:
        predicted = predicted_y.reshape(n, 9)
        choices = np.argmin(predicted, axis=1)
        q_flag = False
    else:
        predicted = predicted_y.reshape(n, 9, 3) * q_std[None, None, :] + q_mean[None, None, :]
        choices = np.asarray([
            min(range(9), key=lambda action: (tuple(row[action]), action))
            for row in predicted
        ])
        q_flag = True
    chosen_regret = true_regret[np.arange(n), choices]
    best_masks = np.asarray([
        np.isclose(row, np.min(row), atol=1e-12, rtol=0.0) for row in true_regret
    ])
    best_agreement = float(np.mean(best_masks[np.arange(n), choices]))
    pair_accuracy, pair_count = clear_pair_order_accuracy(true_tuples, predicted, q_flag)
    frequencies = np.bincount(choices, minlength=9).tolist()
    return {
        "mean_normalized_decision_regret": float(chosen_regret.mean()),
        "median_normalized_decision_regret": float(np.median(chosen_regret)),
        "clear_pair_order_accuracy": pair_accuracy,
        "clear_pair_count": pair_count,
        "best_action_set_agreement": best_agreement,
        "catastrophic_regret_ge_0_5_fraction": float(np.mean(chosen_regret >= 0.5)),
        "predicted_action_frequency": frequencies,
        "selected_action_ids": choices.astype(int).tolist(),
    }


def fit_one_cell(
    family: str,
    x: np.ndarray,
    regrets: np.ndarray,
    objective: np.ndarray,
    target_variant: str,
    random_state: int,
) -> dict[str, Any]:
    import time

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std[x_std < 1e-12] = 1.0
    x_scaled = (x - x_mean) / x_std
    if target_variant == "within_state_normalized_regret9":
        y = regrets
        q_mean = q_std = None
    elif target_variant == "standardized_structured_objective27":
        y, q_mean, q_std = _q_target_scale(objective)
    else:
        raise ValueError(f"unknown target variant {target_variant}")
    estimator = create_estimator(family, random_state)
    start = time.perf_counter()
    estimator.fit(x_scaled, y)
    fit_seconds = time.perf_counter() - start
    predicted = np.asarray(estimator.predict(x_scaled), dtype=np.float64)
    if predicted.shape != y.shape or not np.isfinite(predicted).all():
        raise Phase14CError("learner returned a malformed or non-finite training prediction")
    metrics = evaluate_predictions(family, predicted, regrets, objective, q_mean, q_std)
    metrics.update({
        "learner": family,
        "target_variant": target_variant,
        "sample_count": int(len(x)),
        "input_dim": int(x.shape[1]),
        "parameter_count": model_parameter_count(family, estimator, x.shape[1], y.shape[1]),
        "fit_seconds": fit_seconds,
        "iterations": int(getattr(estimator, "n_iter_", 0)),
        "warnings": [],
        "training_loss_final": float(estimator.loss_) if hasattr(estimator, "loss_") else None,
        "passes_memorization_gate": (
            metrics["mean_normalized_decision_regret"] <= 0.01
            and metrics["clear_pair_order_accuracy"] >= 0.99
        ),
    })
    if family == "tiny_mlp" and metrics["parameter_count"] > 10000:
        raise Phase14CError("tiny MLP exceeds its frozen 10,000-parameter limit")
    return metrics


def synthetic_integrity_data(seed: int, n: int = 201) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create valid public-state rows and a deterministic action-value mapping."""
    if n < 3 or n % 2 == 0:
        raise ValueError("synthetic integrity grid must have an odd count >= 3")
    vx = np.linspace(-1.0, 1.0, n)
    rng = np.random.default_rng(seed)
    vx = vx[rng.permutation(n)]
    x = np.zeros((n, 52), dtype=np.float64)
    x[:, 0:2] = (0.15, 0.15)
    x[:, 2] = vx
    x[:, 3] = 0.0
    x[:, 4:6] = 1.0
    x[:, 6:8] = (0.8, 0.8)
    x[:, 8:16] = (0.025, 0.25, 0.02, 0.20, 0.98, 0.12, 1.0, 1.0)
    # Nine valid, separated public obstacle boxes. Their exact layout is immaterial
    # to the synthetic vx rule but exercises the production feature transformer.
    for obstacle_id in range(9):
        left = 0.30 + obstacle_id * 0.065
        x[:, 16 + obstacle_id * 4:20 + obstacle_id * 4] = (left, 0.05, left + 0.03, 0.08)
    regrets = np.ones((n, 9), dtype=np.float64)
    objectives = np.zeros((n, 9, 3), dtype=np.float64)
    for index, value in enumerate(vx):
        best_action = 1 if value > 0.2 else 2
        regrets[index, best_action] = 0.0
        objectives[index, :, 0] = 0.0
        objectives[index, :, 1] = 1.0
        objectives[index, best_action, 1] = 0.0
        objectives[index, :, 2] = np.arange(9, dtype=np.float64)
        objectives[index, best_action, 2] = 0.0
    return x, regrets, objectives


def run_synthetic_integrity(seed: int, feature_variants: list[str]) -> dict[str, Any]:
    """Exercise the production scaler, estimator factory, and evaluator on a known map."""
    raw, regrets, objectives = synthetic_integrity_data(seed)
    results: list[dict[str, Any]] = []
    requirements_met = True
    for feature_variant in feature_variants:
        features = build_feature_matrix(raw, feature_variant)
        for family in ("nearest_neighbor", "decision_tree", "tiny_mlp"):
            print(
                f"Phase14C synthetic integrity features={feature_variant} learner={family} n={len(raw)}",
                flush=True,
            )
            result = fit_one_cell(
                family, features, regrets, objectives,
                "within_state_normalized_regret9", seed,
            )
            result["feature_variant"] = feature_variant
            results.append(result)
            if family in {"nearest_neighbor", "decision_tree"}:
                passed = (result["mean_normalized_decision_regret"] == 0.0
                          and result["clear_pair_order_accuracy"] == 1.0)
            else:
                passed = result["clear_pair_order_accuracy"] == 1.0
            requirements_met &= passed
    return {
        "status": "passed" if requirements_met else "failed",
        "seed": int(seed),
        "sample_count": len(raw),
        "rule": "action 1 is optimal iff public normalized vx > 0.2; action 2 otherwise",
        "results": results,
    }


def run_memorization_ladder(
    root: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    """Fit only on each frozen training rung; stop immediately on a failed rung."""
    raw, regrets, objectives = extract_arrays(rows)
    rungs = rung_indices(rows, source_audit)
    results: list[dict[str, Any]] = []
    status = "passed"
    for rung_name in config["memorization"]["ladder"]:
        indices = np.asarray(rungs[rung_name], dtype=np.int64)
        if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(rows)):
            raise Phase14CError(f"invalid/empty memorization rung: {rung_name}")
        rung_results: list[dict[str, Any]] = []
        cell_passes: dict[str, bool] = {}
        for feature_variant in config["features"]["variants"]:
            x = build_feature_matrix(raw[indices], feature_variant)
            for target_variant in config["targets"]["variants"]:
                cell_key = f"{feature_variant}__{target_variant}"
                cell_passes[cell_key] = False
                for family in config["learner_families"]:
                    print(
                        f"Phase14C fitting rung={rung_name} features={feature_variant} "
                        f"target={target_variant} learner={family} n={len(indices)}",
                        flush=True,
                    )
                    metrics = fit_one_cell(
                        family, x, regrets[indices], objectives[indices],
                        target_variant, int(config["source"]["memorization_seed"]),
                    )
                    metrics.update({
                        "rung": rung_name,
                        "feature_variant": feature_variant,
                        "row_count": int(len(indices)),
                        "source_query_indices": [int(rows[index]["global_query_index"]) for index in indices],
                    })
                    rung_results.append(metrics)
                    results.append(metrics)
                    cell_passes[cell_key] |= bool(metrics["passes_memorization_gate"])
        passed = all(cell_passes.values())
        results[-len(rung_results):] = rung_results
        if not passed:
            status = "stopped_memorization_gate_failed"
            break
    return {
        "status": status,
        "rungs_completed": list(dict.fromkeys(row["rung"] for row in results)),
        "results": results,
        "rung_gates": {
            rung: {
                key: any(row["rung"] == rung and f'{row["feature_variant"]}__{row["target_variant"]}' == key
                         and row["passes_memorization_gate"] for row in results)
                for key in (
                    f"{feature}__{target}"
                    for feature in config["features"]["variants"]
                    for target in config["targets"]["variants"]
                )
            }
            for rung in dict.fromkeys(row["rung"] for row in results)
        },
    }


def write_phase14c_report(path: Path, lock: dict[str, Any], synthetic: dict[str, Any], ladder: dict[str, Any] | None) -> None:
    lines = [
        "# Phase 14C — Synthetic integrity and same-data memorization result",
        "",
        f"**Status:** {synthetic['status']}; same-data ladder: {ladder['status'] if ladder else 'not run' }.",
        "",
        "## Provenance and scope",
        "",
        f"- Source commit: `{lock['source_commit']}`.",
        f"- Source states/action values: {lock['source_decision_states']} / {lock['action_values']}.",
        f"- Phase14B record SHA-256: `{lock['phase14b_records_sha256']}`.",
        f"- Phase14C lock scope: `{lock['authorized_stage']}`; held-out, closed-loop, DAgger, world model, and reserve remain prohibited.",
        "- This is fitting-set diagnostics only; it is not a generalization or control result.",
        "",
        "## Synthetic integrity",
        "",
        "| Features | Learner | Fit regret | Clear-pair accuracy | Pass | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in synthetic["results"]:
        lines.append(
            f"| {row['feature_variant']} | {row['learner']} | {row['mean_normalized_decision_regret']:.6f} | "
            f"{row['clear_pair_order_accuracy']:.6f} | {row['passes_memorization_gate']} | {row['fit_seconds']:.3f} |"
        )
    if ladder is not None:
        lines.extend(("", "## Same-data ladder", "", "| Rung | Features | Target | Learner | N | Mean regret | Clear-pair accuracy | Gate |", "|---|---|---|---|---:|---:|---:|---|"))
        for row in ladder["results"]:
            lines.append(
                f"| {row['rung']} | {row['feature_variant']} | {row['target_variant']} | {row['learner']} | "
                f"{row['row_count']} | {row['mean_normalized_decision_regret']:.6f} | "
                f"{row['clear_pair_order_accuracy']:.6f} | {row['passes_memorization_gate']} |"
            )
        lines.extend(("", "### Rung gate matrix", ""))
        for rung, cells in ladder["rung_gates"].items():
            lines.append(f"- `{rung}`: " + ", ".join(f"{key}={value}" for key, value in cells.items()))
    lines.extend((
        "",
        "## Interpretation / next decision",
        "",
        "A failed synthetic test is a pipeline failure; no teacher-label model fitting follows. A failed 1/8/32/one-layout rung stops the ladder at that rung. Passing the full ladder permits only a separately locked held-out protocol; this report itself does not open held-out or closed-loop evaluation.",
        "",
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
