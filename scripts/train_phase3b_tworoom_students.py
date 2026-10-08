"""Train small Phase-3B policies/scorers using train and validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.candidates import primitive_actions, two_segment_sequences
from consequence_lab.student import (
    CandidateCostRegressor,
    CandidateScorer,
    DirectActionPolicy,
    candidate_features,
    parameter_count,
)


CONFIG_PATH = ROOT / "configs" / "phase3_tworoom_amortization.yaml"
ACTION_BANK = primitive_actions()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _load_npz(path: Path, split: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"required {split} dataset is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    required = {
        "observations", "candidate_costs", "candidate_reachability", "episode_seeds",
        "sampled_step_indices", "behavior", "candidate_sequences",
    }
    if set(data) != required:
        raise ValueError(f"{split} dataset fields differ from frozen schema")
    state_count = len(data["observations"])
    if data["observations"].shape != (state_count, 10):
        raise ValueError(f"{split} observations must have shape [states,10]")
    if data["candidate_costs"].shape != (state_count, 65):
        raise ValueError(f"{split} cost matrix must have shape [states,65]")
    if data["candidate_reachability"].shape != (state_count, 65):
        raise ValueError(f"{split} candidate reachability must have shape [states,65]")
    if not np.isfinite(data["observations"]).all() or not np.isfinite(data["candidate_costs"]).all():
        raise ValueError(f"{split} data contains non-finite values")
    expected = two_segment_sequences(64)
    if not np.array_equal(data["candidate_sequences"], expected):
        raise ValueError(f"{split} candidate bank differs from frozen generator")
    if not len(data["episode_seeds"]):
        raise ValueError(f"{split} contains no states")
    return data


def _first_action_class(sequences: np.ndarray) -> np.ndarray:
    first_actions = sequences[:, 0]
    distances = np.linalg.norm(first_actions[:, None, :] - ACTION_BANK[None, :, :], axis=-1)
    classes = distances.argmin(axis=1)
    if np.any(distances[np.arange(len(classes)), classes] > 1e-6):
        raise ValueError("candidate first action is not in the primitive action bank")
    return classes.astype(np.int64)


def _episode_weights(seeds: np.ndarray) -> torch.Tensor:
    _, inverse, counts = np.unique(seeds, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights /= weights.sum()
    return torch.as_tensor(weights, dtype=torch.float32)


def _episode_mean(values: np.ndarray, seeds: np.ndarray) -> float:
    return float(np.mean([values[seeds == seed].mean() for seed in np.unique(seeds)]))


def _teacher_arrays(costs: np.ndarray, action_classes: np.ndarray, temperature: float):
    regrets = costs - costs.min(axis=1, keepdims=True)
    soft = torch.softmax(torch.as_tensor(-regrets / temperature, dtype=torch.float32), dim=-1)
    action_costs = np.full((len(costs), len(ACTION_BANK)), np.inf, dtype=np.float32)
    for action_index in range(len(ACTION_BANK)):
        chosen = action_classes == action_index
        if chosen.any():
            action_costs[:, action_index] = costs[:, chosen].min(axis=1)
    if not np.isfinite(action_costs).all():
        raise ValueError("every primitive action must be represented by at least one candidate")
    action_labels = action_costs.argmin(axis=1)
    return regrets.astype(np.float32), soft, action_costs, action_labels


def _min_positive_regret(costs: np.ndarray) -> float:
    regrets = costs - costs.min(axis=1, keepdims=True)
    positive = regrets[regrets > 1e-8]
    if not len(positive):
        raise ValueError("training teacher cost matrix has no positive candidate regrets")
    return float(np.median(positive))


def _batch_inputs(data: dict[str, np.ndarray], indices: torch.Tensor, feat_tensor: torch.Tensor):
    obs_all = torch.as_tensor(data["observations"], dtype=torch.float32)
    obs = obs_all[indices]
    feature_batch = feat_tensor.unsqueeze(0).expand(len(indices), -1, -1)
    mask = torch.ones(feature_batch.shape[:2], dtype=torch.bool)
    return obs, feature_batch, mask


def _predict(model, kind: str, data: dict[str, np.ndarray], feats: torch.Tensor, batch_size=256):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(data["observations"]), batch_size):
            stop = min(start + batch_size, len(data["observations"]))
            ix = torch.arange(start, stop)
            obs, feature_batch, mask = _batch_inputs(data, ix, feats)
            if kind == "direct_action_policy":
                output = model(obs)
            elif kind == "candidate_cost_regressor":
                output = model(obs, feature_batch, mask)
            else:
                output = model(obs, feature_batch, mask)
            outputs.append(output.cpu())
    return torch.cat(outputs, dim=0).numpy()


def _validation_metrics(
    output: np.ndarray,
    kind: str,
    costs: np.ndarray,
    action_classes: np.ndarray,
    seeds: np.ndarray,
    tau: float,
    reachability: np.ndarray,
) -> dict[str, float]:
    normalized_costs = costs / 1.1
    regrets = normalized_costs - normalized_costs.min(axis=1, keepdims=True)
    action_costs = np.full((len(costs), len(ACTION_BANK)), np.inf, dtype=np.float32)
    for action_index in range(len(ACTION_BANK)):
        action_costs[:, action_index] = costs[:, action_classes == action_index].min(axis=1)
    optimal_plan = costs.argmin(axis=1)
    optimal_action = action_costs.argmin(axis=1)

    action_regrets = (action_costs - action_costs.min(axis=1, keepdims=True)) / 1.1
    action_teacher_probability = torch.softmax(
        torch.as_tensor(-action_regrets / (tau / 1.1), dtype=torch.float32), dim=-1
    ).numpy()
    candidate_reach = np.asarray(reachability, dtype=bool)
    if candidate_reach.shape != costs.shape:
        raise ValueError("candidate reachability must match validation costs")
    if kind in {"direct_action_policy"}:
        probability = torch.softmax(torch.as_tensor(output, dtype=torch.float32), dim=-1).numpy()
        chosen_action = output.argmax(axis=1)
        selected_available = np.asarray([
            candidate_reach[row, action_classes == chosen_action[row]].any()
            for row in range(len(costs))
        ], dtype=np.float32)
        selected_regret = (
            action_costs[np.arange(len(costs)), chosen_action] - costs.min(axis=1)
        ) / 1.1
        expected_regret = (probability * action_regrets).sum(axis=1)
        nll = -(
            action_teacher_probability * np.log(np.maximum(probability, 1e-12))
        ).sum(axis=1)
        brier = ((probability - action_teacher_probability) ** 2).sum(axis=1)
    elif kind == "candidate_cost_regressor":
        chosen = output.argmin(axis=1)
        probability = torch.softmax(-torch.as_tensor(output) / (tau / 1.1), dim=-1).numpy()
        selected_regret = regrets[np.arange(len(costs)), chosen]
        chosen_action = action_classes[chosen]
        selected_available = candidate_reach[np.arange(len(costs)), chosen].astype(np.float32)
    else:
        probability = torch.softmax(torch.as_tensor(output, dtype=torch.float32), dim=-1).numpy()
        chosen = output.argmax(axis=1)
        selected_regret = regrets[np.arange(len(costs)), chosen]
        chosen_action = action_classes[chosen]
        selected_available = candidate_reach[np.arange(len(costs)), chosen].astype(np.float32)
    if kind != "direct_action_policy":
        expected_regret = (probability * regrets).sum(axis=1)
    teacher_probability = torch.softmax(
        torch.as_tensor(-(costs - costs.min(axis=1, keepdims=True)) / tau, dtype=torch.float32),
        dim=-1,
    ).numpy()
    if kind != "direct_action_policy":
        nll = -(teacher_probability * np.log(np.maximum(probability, 1e-12))).sum(axis=1)
        brier = ((probability - teacher_probability) ** 2).sum(axis=1)
    plan_agreement = (probability.argmax(axis=1) == optimal_plan) if kind != "direct_action_policy" else np.zeros(len(costs), dtype=bool)
    return {
        "selected_regret": _episode_mean(selected_regret, seeds),
        "expected_regret": _episode_mean(expected_regret, seeds),
        "first_action_regret": _episode_mean(
            ((action_costs[np.arange(len(costs)), chosen_action] - costs.min(axis=1)) / 1.1), seeds
        ),
        "teacher_first_action_agreement": _episode_mean((chosen_action == optimal_action).astype(np.float32), seeds),
        "teacher_plan_top1_agreement": _episode_mean(plan_agreement.astype(np.float32), seeds),
        "teacher_preference_nll": _episode_mean(nll, seeds),
        "teacher_preference_brier": _episode_mean(brier, seeds),
        "oracle_candidate_availability": _episode_mean(candidate_reach.any(axis=1).astype(np.float32), seeds),
        "selected_candidate_reachability": _episode_mean(selected_available, seeds),
    }


def _fit_one(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    sequences: np.ndarray,
    kind: str,
    loss_name: str,
    init_seed: int,
    temperature: float,
    regret_weight: float,
    settings: dict,
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    torch.set_num_threads(1)
    feats_np = candidate_features(sequences)
    feats = torch.as_tensor(feats_np, dtype=torch.float32)
    train_costs = train["candidate_costs"].astype(np.float32)
    val_costs = validation["candidate_costs"].astype(np.float32)
    action_classes = _first_action_class(sequences)
    train_regret, train_soft, train_action_costs, train_action_labels = _teacher_arrays(
        train_costs, action_classes, temperature
    )
    val_regret, val_soft, val_action_costs, val_action_labels = _teacher_arrays(
        val_costs, action_classes, temperature
    )
    train_weights = _episode_weights(train["episode_seeds"])
    train_regret_t = torch.as_tensor(train_regret / 1.1, dtype=torch.float32)
    train_soft_t = train_soft
    train_cost_t = torch.as_tensor(train_costs / 1.1, dtype=torch.float32)
    train_action_labels_t = torch.as_tensor(train_action_labels, dtype=torch.long)
    train_action_costs_t = torch.as_tensor(train_action_costs / 1.1, dtype=torch.float32)

    if kind == "direct_action_policy":
        model = DirectActionPolicy()
        loss_name = "hard_action_cross_entropy"
    elif kind == "candidate_cost_regressor":
        model = CandidateCostRegressor()
        loss_name = "smooth_l1_cost_regression"
    else:
        model = CandidateScorer(set_context=(kind == "set_equivariant_candidate_scorer"))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    best_state = None
    best_metric = float("inf")
    best_epoch = 0
    patience = int(settings["early_stopping_patience"])
    stale = 0
    rows = []
    batch_size = int(settings["batch_size_states"])
    n = len(train_costs)

    for epoch in range(1, int(settings["max_epochs"]) + 1):
        model.train()
        batch_count = int(np.ceil(n / batch_size))
        for _ in range(batch_count):
            indices = torch.multinomial(train_weights, batch_size, replacement=True)
            obs, feature_batch, mask = _batch_inputs(train, indices, feats)
            optimizer.zero_grad(set_to_none=True)
            if kind == "direct_action_policy":
                logits = model(obs)
                loss = F.cross_entropy(logits, train_action_labels_t[indices])
            elif kind == "candidate_cost_regressor":
                predicted_cost = model(obs, feature_batch, mask)
                loss = F.smooth_l1_loss(predicted_cost, train_cost_t[indices])
            else:
                logits = model(obs, feature_batch, mask)
                if loss_name == "hard_cross_entropy":
                    loss = F.cross_entropy(logits, train_cost_t[indices].argmin(dim=1))
                else:
                    log_probability = F.log_softmax(logits, dim=-1)
                    cross_entropy = -(train_soft_t[indices] * log_probability).sum(dim=-1).mean()
                    if loss_name == "soft_kl_plus_expected_regret":
                        kl = F.kl_div(
                            log_probability,
                            train_soft_t[indices],
                            reduction="none",
                        ).sum(dim=-1).mean()
                        probability = torch.softmax(logits, dim=-1)
                        expected_regret = (probability * train_regret_t[indices]).sum(dim=-1).mean()
                        loss = kl + regret_weight * expected_regret
                    else:
                        loss = cross_entropy
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite {kind}/{loss_name} loss at epoch {epoch}")
            loss.backward()
            optimizer.step()

        validation_output = _predict(model, kind, validation, feats)
        metrics = _validation_metrics(
            validation_output,
            kind,
            val_costs,
            action_classes,
            validation["episode_seeds"],
            temperature,
            validation["candidate_reachability"],
        )
        selected_regret = metrics["selected_regret"]
        rows.append({"epoch": epoch, "validation": metrics})
        if selected_regret < best_metric - 1e-8:
            best_metric = selected_regret
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_output = _predict(model, kind, validation, feats)
    best_metrics = _validation_metrics(
        final_output,
        kind,
        val_costs,
        action_classes,
        validation["episode_seeds"],
        temperature,
        validation["candidate_reachability"],
    )
    return model, {
        "kind": kind,
        "loss": loss_name,
        "init_seed": init_seed,
        "temperature": temperature,
        "regret_weight": regret_weight,
        "best_epoch": best_epoch,
        "epochs_run": len(rows),
        "parameter_count": parameter_count(model),
        "validation": best_metrics,
        "curve": rows,
    }


def _variant_specs(train: dict[str, np.ndarray], config: dict):
    med_gap = _min_positive_regret(train["candidate_costs"])
    specs = []
    for seed in config["models"]["initialization_seeds"]:
        specs.append(("direct_action_policy", "hard_action_cross_entropy", seed, med_gap, 0.0))
        specs.append(("candidate_cost_regressor", "smooth_l1_cost_regression", seed, med_gap, 0.0))
        for kind in ("independent_candidate_scorer", "set_equivariant_candidate_scorer"):
            specs.append((kind, "hard_cross_entropy", seed, med_gap, 0.0))
            for multiplier in config["teacher"]["temperature"]["multipliers"]:
                tau = med_gap * multiplier
                specs.append((kind, "soft_cross_entropy", seed, tau, 0.0))
                for weight in config["teacher"]["joint_loss"]["regret_weights"]:
                    if weight > 0:
                        specs.append((kind, "soft_kl_plus_expected_regret", seed, tau, weight))
    return specs


def train_phase3b(config: dict, dataset_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite training directory {output_dir}")
    train = _load_npz(dataset_dir / "train.npz", "train")
    validation = _load_npz(dataset_dir / "validation.npz", "validation")
    train_seeds = set(map(int, np.unique(train["episode_seeds"])))
    validation_seeds = set(map(int, np.unique(validation["episode_seeds"])))
    if train_seeds & validation_seeds:
        raise ValueError("train and validation episode seeds overlap")
    locked_start, locked_end = config["data"]["seed_ranges"]["locked_test"]
    if any(locked_start <= seed < locked_end for seed in train_seeds | validation_seeds):
        raise ValueError("locked final-test seed found in train/validation files")

    # Check all episode seeds against the registered ranges: this catches accidental
    # row slicing or misnamed split files before any training begins.
    for split, actual in (("train", train_seeds), ("validation", validation_seeds)):
        low, high = config["data"]["seed_ranges"][split]
        if not actual or any(seed < low or seed >= high for seed in actual):
            raise ValueError(f"{split} artifact contains seeds outside its frozen interval")

    settings = config["models"]
    sequences = train["candidate_sequences"]
    feats = torch.as_tensor(candidate_features(sequences), dtype=torch.float32)
    output_dir.mkdir(parents=True)
    rows = []
    begin = time.perf_counter()
    for index, (kind, loss_name, seed, tau, weight) in enumerate(_variant_specs(train, config), start=1):
        variant_id = f"{kind}__{loss_name}__seed{seed}__tau{tau:.8g}__lambda{weight:.3g}"
        print(f"training {index}: {variant_id}", flush=True)
        model, result = _fit_one(
            train,
            validation,
            sequences,
            kind,
            loss_name,
            int(seed),
            float(tau),
            float(weight),
            settings,
        )
        result["variant_id"] = variant_id
        rows.append(result)
        checkpoint_path = output_dir / f"{variant_id}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "kind": kind,
                "loss": result["loss"],
                "init_seed": int(seed),
                "temperature": float(tau),
                "regret_weight": float(weight),
                "best_epoch": result["best_epoch"],
                "validation_selected_regret": result["validation"]["selected_regret"],
                "candidate_features": feats,
            },
            checkpoint_path,
        )
        result["checkpoint_sha256"] = _sha256(checkpoint_path)
        result["checkpoint_bytes"] = checkpoint_path.stat().st_size
        print(
            f"  epoch={result['best_epoch']} val regret="
            f"{result['validation']['selected_regret']:.4f}"
            f" expected={result['validation']['expected_regret']:.4f}",
            flush=True,
        )

    # Rank hyperparameter configurations by the mean validation selected regret
    # across the three predeclared initialization seeds. Do not inspect locked test.
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = "|".join(
            [row["kind"], row["loss"], f"tau={row['temperature']:.12g}", f"lambda={row['regret_weight']:.12g}"]
        )
        groups.setdefault(key, []).append(row)
    ranking = []
    for key, members in groups.items():
        ranking.append(
            {
                "configuration": key,
                "seed_count": len(members),
                "mean_validation_selected_regret": float(np.mean([m["validation"]["selected_regret"] for m in members])),
                "seed_selected_regret": [m["validation"]["selected_regret"] for m in members],
                "mean_validation_expected_regret": float(np.mean([m["validation"]["expected_regret"] for m in members])),
            }
        )
    ranking.sort(key=lambda row: (row["mean_validation_selected_regret"], row["configuration"]))
    elapsed = time.perf_counter() - begin
    summary = {
        "experiment": "Phase 3B train/validation amortized decision screen",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment_revision": config["environment"]["revision"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "train_seed_count": len(train_seeds),
        "validation_seed_count": len(validation_seeds),
        "train_state_count": len(train["observations"]),
        "validation_state_count": len(validation["observations"]),
        "train_data_sha256": _sha256(dataset_dir / "train.npz"),
        "validation_data_sha256": _sha256(dataset_dir / "validation.npz"),
        "config_sha256": _sha256(CONFIG_PATH),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "student_module_sha256": _sha256(ROOT / "src/consequence_lab/student.py"),
        "variant_count": len(rows),
        "ranking_by_configuration": ranking,
        "variant_results": rows,
        "wall_seconds": round(elapsed, 3),
        "locked_test_opened": False,
        "git": {"commit": _git("rev-parse", "HEAD"), "status": _git("status", "--porcelain")},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "run_id": config["artifacts"]["phase3b_run_id"],
        "phase": "phase_3b_training",
        "status": "completed",
        "created_utc": summary["created_utc"],
        "data_splits_used": ["train", "validation"],
        "data_sha256": {
            "train": summary["train_data_sha256"],
            "validation": summary["validation_data_sha256"],
        },
        "locked_test_opened": False,
        "config_sha256": summary["config_sha256"],
        "script_sha256": summary["script_sha256"],
        "student_module_sha256": summary["student_module_sha256"],
        "git": summary["git"],
        "summary_path": str(summary_path.relative_to(ROOT)),
        "resources": {"device": "cpu", "wall_seconds": summary["wall_seconds"], "gpu_hours": 0},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("variant_count", "ranking_by_configuration", "wall_seconds")}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset_dir = args.dataset_dir or (ROOT / config["artifacts"]["dataset_directory"])
    output_dir = args.output_dir or (ROOT / config["artifacts"]["training_directory"])
    result = train_phase3b(config, dataset_dir, output_dir)
    if not result["ranking_by_configuration"]:
        raise RuntimeError("no student configurations were trained")


if __name__ == "__main__":
    main()
