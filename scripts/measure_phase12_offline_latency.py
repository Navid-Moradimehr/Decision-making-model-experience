"""Measure selected Phase-12 scorer latency on validation inputs (no task rollout)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12_evaluation import load_student
from consequence_lab.phase12_teacher import candidate_features


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    results_path = ROOT / "artifacts" / "runs" / "phase12_training_v1" / "results.json"
    validation_path = ROOT / "artifacts" / "runs" / "phase12_validation_teacher_v1" / "student_inputs.npz"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    selected = results["selected"]
    model, payload = load_student(Path(selected["checkpoint"]), device="cpu")
    with np.load(validation_path, allow_pickle=False) as data:
        states, sequences, masks = data["states"], data["sequences"], data["candidate_mask"]
    times_ms = []
    with torch.no_grad():
        for index in range(len(states)):
            features = candidate_features(sequences[index])
            state_tensor = torch.from_numpy(states[index]).unsqueeze(0)
            candidate_tensor = torch.from_numpy(features).unsqueeze(0)
            mask_tensor = torch.from_numpy(masks[index]).unsqueeze(0)
            started = perf_counter_ns()
            score = model(state_tensor, candidate_tensor, mask_tensor)
            _ = score[0].argmax().item()
            times_ms.append((perf_counter_ns() - started) / 1_000_000.0)
    measured = np.asarray(times_ms[20:], dtype=np.float64)  # first 20 are warm-up only
    report = {
        "schema": "phase12_offline_scorer_latency_v1",
        "status": "measurement_only_not_used_for_selection",
        "selected_variant": selected["variant"],
        "selected_fit_seed": selected["fit_seed"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "validation_input_sha256": sha(validation_path),
        "torch_version": torch.__version__,
        "device": "CPU",
        "torch_cpu_threads": torch.get_num_threads(),
        "parameter_count": selected["parameters"],
        "candidate_set_count": int(masks.shape[1]),
        "queries_measured": int(measured.size),
        "warmup_queries": 20,
        "latency_scope": "one state/candidate tensor to scorer output plus argmax; excludes scenario proposal generation, feature encoding, simulator transition, teacher audit",
        "median_forward_ms": float(np.median(measured)),
        "p95_forward_ms": float(np.percentile(measured, 95)),
        "mean_forward_ms": float(np.mean(measured)),
        "baseline_phase11_exact_shooting_median_wall_ms": 25.409,
        "scorer_only_speedup_vs_phase11_reference": float(25.409 / np.median(measured)),
        "reserve_accessed": False,
        "closed_loop_evaluation_performed": False,
    }
    out = ROOT / "reports" / "phase12_offline_latency.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
