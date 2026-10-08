"""Closed-loop validation screen; this script rejects the locked test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from consequence_lab.candidates import candidate_subset, primitive_actions, two_segment_sequences
from consequence_lab.environment import TwoRoomAdapter
from consequence_lab.realizability import validate_runtime_payload
from consequence_lab.student import CandidateCostRegressor, CandidateScorer, DirectActionPolicy, candidate_features
from run_phase3a_tworoom_realizability import _cost, _git, _load_upstream_environment, _sha256
from train_phase3b_tworoom_students import _first_action_class, _load_npz


CONFIG_PATH = ROOT / "configs" / "phase3_tworoom_amortization.yaml"


def _make_model(kind: str, set_context: bool = False):
    if kind == "direct_action_policy":
        return DirectActionPolicy()
    if kind == "candidate_cost_regressor":
        return CandidateCostRegressor()
    return CandidateScorer(set_context=set_context)


def _load_checkpoint(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older pinned Torch builds
        payload = torch.load(path, map_location="cpu")
    model = _make_model(payload["kind"], payload["kind"] == "set_equivariant_candidate_scorer")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _paired_bootstrap_difference(a: np.ndarray, b: np.ndarray, seed: int = 501091):
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(difference), size=(10000, len(difference)))
    means = difference[draws].mean(axis=1)
    return {
        "mean": float(difference.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
    }


def _rollout_method(env_type, method: str, seeds, config, sequences, model=None):
    env_cfg = config["environment"]
    actions = primitive_actions()
    action_classes = _first_action_class(sequences)
    features = torch.as_tensor(candidate_features(sequences), dtype=torch.float32).unsqueeze(0)
    mask = torch.ones((1, len(sequences)), dtype=torch.bool)
    outcomes = []
    latency = []
    teacher_cost_evaluations = 0
    random_generators = {int(seed): np.random.default_rng(int(seed) + 599) for seed in seeds}
    for episode_seed in seeds:
        env = env_type(render_mode="rgb_array")
        adapter = TwoRoomAdapter(env, success_distance=env_cfg["success_distance_px"])
        observation, _ = adapter.reset(seed=int(episode_seed))
        success = False
        elapsed_steps = 0
        try:
            for step in range(env_cfg["max_episode_steps"]):
                if method == "greedy":
                    delta = observation[2:4] - observation[:2]
                    norm = float(np.linalg.norm(delta))
                    action = np.zeros(2, dtype=np.float32) if norm == 0 else delta / norm
                elif method == "doorway_waypoint":
                    agent = observation[:2]
                    goal = observation[2:4]
                    wall_x = float(observation[4])  # first observed door lies on the wall centerline
                    crosses_wall = (agent[0] - wall_x) * (goal[0] - wall_x) < 0.0
                    if crosses_wall:
                        waypoint = np.asarray([wall_x, observation[5]], dtype=np.float32)
                        delta = waypoint - agent
                        if np.linalg.norm(delta) < 8.0:
                            delta = goal - agent
                    else:
                        delta = goal - agent
                    norm = float(np.linalg.norm(delta))
                    action = np.zeros(2, dtype=np.float32) if norm == 0 else delta / norm
                elif method == "random":
                    action = actions[int(random_generators[int(episode_seed)].integers(0, len(actions)))]
                elif method == "exact_planner":
                    started = time.perf_counter()
                    costs, _ = _cost(
                        adapter,
                        adapter.snapshot(),
                        sequences,
                        observation[2:4],
                        {"environment": env_cfg, "teacher": {"objective": env_cfg["objective"]}},
                    )
                    latency.append(time.perf_counter() - started)
                    teacher_cost_evaluations += 1
                    action = sequences[int(np.argmin(costs)), 0]
                else:
                    started = time.perf_counter()
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
                    runtime_payload = {
                        "observation": observation,
                        "candidate_sequences": sequences,
                        "valid_candidate_mask": np.ones(len(sequences), dtype=np.bool_),
                    }
                    validate_runtime_payload(runtime_payload)
                    with torch.no_grad():
                        if method == "direct_action_policy":
                            output = model(obs_tensor)
                            action = actions[int(output.argmax(dim=-1).item())]
                        elif method == "candidate_cost_regressor":
                            output = model(obs_tensor, features, mask)
                            candidate = int(output.argmin(dim=-1).item())
                            action = sequences[candidate, 0]
                        else:
                            output = model(obs_tensor, features, mask)
                            candidate = int(output.argmax(dim=-1).item())
                            action = sequences[candidate, 0]
                    latency.append(time.perf_counter() - started)
                observation, _, terminated, truncated, _ = adapter.step(action)
                elapsed_steps = step + 1
                if terminated or truncated:
                    success = bool(np.linalg.norm(observation[:2] - observation[2:4]) < env_cfg["success_distance_px"])
                    break
        finally:
            env.close()
        outcomes.append({"seed": int(episode_seed), "success": bool(success), "steps": int(elapsed_steps)})
        if len(outcomes) % 10 == 0 or len(outcomes) == len(seeds):
            print(f"  {method}: {len(outcomes)}/{len(seeds)} validation episodes", flush=True)
    result = {
        "method": method,
        "episode_count": len(outcomes),
        "success_count": int(sum(row["success"] for row in outcomes)),
        "success_rate": float(np.mean([row["success"] for row in outcomes])),
        "mean_steps": float(np.mean([row["steps"] for row in outcomes])),
        "episode_results": outcomes,
        "teacher_cost_evaluations": int(teacher_cost_evaluations),
        "decision_latency_median_ms": float(np.median(latency) * 1000) if latency else None,
        "decision_latency_p95_ms": float(np.quantile(latency, 0.95) * 1000) if latency else None,
    }
    return result


def _permutation_equivariance_check(model, sequences, repeats=20):
    errors = {}
    rng = np.random.default_rng(77319)
    model.eval()
    for count in (9, 17, 33, 65):
        subset = candidate_subset(sequences, count)
        feat = torch.as_tensor(candidate_features(subset), dtype=torch.float32).unsqueeze(0)
        mask = torch.ones((1, count), dtype=torch.bool)
        obs = torch.as_tensor(
            [112.0, 50.0, 160.0, 170.0, 112.0, 49.0, 112.0, 49.0, 112.0, 49.0],
            dtype=torch.float32,
        ).unsqueeze(0)
        def probabilities(features, valid_mask):
            values = model(obs, features, valid_mask)
            if isinstance(model, CandidateCostRegressor):
                values = -values
            return torch.softmax(values, dim=-1)

        with torch.no_grad():
            baseline = probabilities(feat, mask).cpu().numpy()[0]
            maximum = 0.0
            for _ in range(repeats):
                permutation = torch.as_tensor(rng.permutation(count), dtype=torch.long)
                probability = probabilities(feat[:, permutation], mask[:, permutation]).cpu().numpy()[0]
                restored = probability[np.argsort(permutation.cpu().numpy())]
                maximum = max(maximum, float(np.max(np.abs(restored - baseline))))
        errors[str(count)] = maximum
    return errors


def _select_best_variants(training_summary: dict):
    by_kind: dict[str, list[dict]] = {}
    for row in training_summary["variant_results"]:
        by_kind.setdefault(row["kind"], []).append(row)
    chosen = {}
    for kind, rows in by_kind.items():
        if kind == "direct_action_policy" or kind == "candidate_cost_regressor":
            family_rows = rows
        else:
            # Select the loss/temperature from aggregate validation results,
            # then select its best initialization seed. The locked test is absent.
            groups: dict[str, list[dict]] = {}
            for row in rows:
                key = (row["loss"], row["temperature"], row["regret_weight"])
                groups.setdefault(str(key), []).append(row)
            selected_group = min(
                groups.values(),
                key=lambda group: float(np.mean([r["validation"]["selected_regret"] for r in group])),
            )
            family_rows = selected_group
        selected = min(family_rows, key=lambda row: row["validation"]["selected_regret"])
        chosen[kind] = {"variant_id": selected["variant_id"], "checkpoint_sha256": selected["checkpoint_sha256"]}
    return chosen


def evaluate(
    config: dict,
    env_type,
    data_dir: Path,
    training_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
):
    torch.set_num_threads(1)
    begin = time.perf_counter()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite validation-evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True)
    locked_low, locked_high = config["data"]["seed_ranges"]["locked_test"]
    val_low, val_high = config["data"]["seed_ranges"]["validation"]
    if max(val_low, locked_low) < min(val_high, locked_high):
        raise ValueError("validation seeds overlap locked test seeds")
    validation = _load_npz(data_dir / "validation.npz", "validation")
    seeds = np.arange(val_low, val_high, dtype=np.int64)
    actual_seeds = np.unique(validation["episode_seeds"])
    if not np.array_equal(actual_seeds, seeds):
        raise ValueError("validation data episode seeds differ from frozen validation range")
    summary_path = training_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Phase-3B training summary is absent")
    training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline_path = baseline_dir / "summary.json"
    if not baseline_path.is_file():
        raise FileNotFoundError("validation-only geometric baseline screen must pass before student rollout evaluation")
    baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline_summary.get("status") != "completed":
        raise ValueError("geometric baseline screen is not complete")
    if baseline_summary.get("validation_data_sha256") != _sha256(data_dir / "validation.npz"):
        raise ValueError("geometric baseline screen was run on different validation data")
    if baseline_summary.get("validation_seed_range") != [val_low, val_high]:
        raise ValueError("geometric baseline seed range differs from frozen validation range")
    if baseline_summary.get("locked_test_opened") is not False:
        raise ValueError("invalid baseline artifact: locked test was marked open")
    if set(baseline_summary.get("baselines", {})) != {
        "greedy", "doorway_waypoint", "random", "exact_planner"
    }:
        raise ValueError("geometric baseline artifact is missing a required method")
    for method, baseline_result in baseline_summary["baselines"].items():
        baseline_seeds = [row["seed"] for row in baseline_result["episode_results"]]
        if baseline_seeds != list(map(int, seeds)):
            raise ValueError(f"{method} baseline episode seeds differ from validation split")
    selected = _select_best_variants(training_summary)
    sequences = two_segment_sequences(config["environment"]["candidate_horizon_steps"])
    methods = []
    student_results = {}
    permutation_errors = {}
    baselines = baseline_summary["baselines"]
    for kind, pick in selected.items():
        checkpoint_path = training_dir / f"{pick['variant_id']}.pt"
        if _sha256(checkpoint_path) != pick["checkpoint_sha256"]:
            raise ValueError(f"selected checkpoint hash mismatch: {pick['variant_id']}")
        model, payload = _load_checkpoint(checkpoint_path)
        if payload["kind"] != kind:
            raise ValueError("checkpoint model kind differs from validation selection")
        methods.append(kind)
        if kind in {
            "independent_candidate_scorer",
            "set_equivariant_candidate_scorer",
            "candidate_cost_regressor",
        }:
            permutation_errors[kind] = _permutation_equivariance_check(model, sequences)
        result = _rollout_method(env_type, kind, seeds, config, sequences, model)
        result["selected_variant_id"] = pick["variant_id"]
        result["selected_checkpoint_sha256"] = pick["checkpoint_sha256"]
        result["parameter_count"] = int(sum(p.numel() for p in model.parameters()))
        student_results[kind] = result
        print(
            f"{kind}: success {result['success_count']}/{result['episode_count']}, "
            f"mean steps {result['mean_steps']:.1f}",
            flush=True,
        )
    greedy_success = np.asarray([row["success"] for row in baselines["greedy"]["episode_results"]])
    paired = {}
    for kind, result in student_results.items():
        student_success = np.asarray([row["success"] for row in result["episode_results"]])
        waypoint_success = np.asarray(
            [row["success"] for row in baselines["doorway_waypoint"]["episode_results"]]
        )
        paired[kind] = {
            "vs_greedy": _paired_bootstrap_difference(student_success, greedy_success),
            "vs_doorway_waypoint": _paired_bootstrap_difference(student_success, waypoint_success),
        }

    random_cost_regret = (
        validation["candidate_costs"] - validation["candidate_costs"].min(axis=1, keepdims=True)
    ).mean(axis=1) / 1.1
    random_offline_regret = float(
        np.mean([random_cost_regret[validation["episode_seeds"] == seed].mean() for seed in actual_seeds])
    )
    offline_best = float(
        training_summary["ranking_by_configuration"][0]["mean_validation_selected_regret"]
    )
    summary = {
        "experiment": "Phase 3B closed-loop validation screen",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_seed_range": [val_low, val_high],
        "locked_test_seed_range": [locked_low, locked_high],
        "locked_test_opened": False,
        "validation_data_sha256": _sha256(data_dir / "validation.npz"),
        "training_summary_sha256": _sha256(summary_path),
        "geometry_baseline_summary_sha256": _sha256(baseline_path),
        "config_sha256": _sha256(CONFIG_PATH),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "torch": torch.__version__,
        "random_expected_offline_regret": random_offline_regret,
        "best_validation_offline_selected_regret": float(offline_best),
        "best_validation_configuration": training_summary["ranking_by_configuration"][0]["configuration"],
        "offline_regret_ratio_vs_random": float(offline_best / random_offline_regret) if random_offline_regret else None,
        "offline_gate_pass": bool(random_offline_regret and offline_best <= 0.5 * random_offline_regret),
        "baselines": baselines,
        "students": student_results,
        "permutation_probability_max_abs_error": permutation_errors,
        "permutation_gate_pass": all(
            error <= config["gates"]["permutation_equivariance"]["max_absolute_probability_error"]
            for model_errors in permutation_errors.values()
            for error in model_errors.values()
        ),
        "paired_success_difference_vs_greedy": paired,
        "paired_bootstrap": {"unit": "episode_seed", "replicates": 10000, "seed": 501091},
        "validation_closed_loop_gate_pass": any(
            row["success_rate"] > baselines["greedy"]["success_rate"]
            for row in student_results.values()
        ),
        "wall_seconds": round(time.perf_counter() - begin, 3),
        "device": "cpu",
        "gpu_hours": 0,
        "candidate_features_cached_for_static_bank": True,
        "wall_clock_note": "CPU validation; candidate descriptors are cached because this frozen candidate bank is static. Results are for selection/screening, not final confirmation.",
        "git": {"commit": _git("rev-parse", "HEAD"), "status": _git("status", "--porcelain")},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "phase": "phase_3b_validation_closed_loop",
        "status": "completed",
        "data_split": "validation",
        "locked_test_opened": False,
        "validation_data_sha256": summary["validation_data_sha256"],
        "training_summary_sha256": summary["training_summary_sha256"],
        "geometry_baseline_summary_sha256": summary["geometry_baseline_summary_sha256"],
        "config_sha256": summary["config_sha256"],
        "script_sha256": summary["script_sha256"],
        "git": summary["git"],
        "summary_path": str(summary_path.relative_to(ROOT)),
        "seed_range": [val_low, val_high],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_dir = args.data_dir or (ROOT / config["artifacts"]["dataset_directory"])
    training_dir = args.training_dir or (ROOT / config["artifacts"]["training_directory"])
    baseline_dir = args.baseline_dir or (ROOT / "artifacts/runs/phase3b_tworoom_geometry_baselines_v1")
    output_dir = args.output_dir or (ROOT / "artifacts/runs/phase3b_tworoom_validation_eval_v1")
    env_type = _load_upstream_environment(args.upstream_root.resolve())
    summary = evaluate(config, env_type, data_dir, training_dir, baseline_dir, output_dir)
    print(json.dumps({
        "offline_gate_pass": summary["offline_gate_pass"],
        "validation_closed_loop_gate_pass": summary["validation_closed_loop_gate_pass"],
        "students": {kind: row["success_rate"] for kind, row in summary["students"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
