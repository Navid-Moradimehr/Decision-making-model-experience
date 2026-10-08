"""Exploratory grouped-CV models on the saved Reacher planner cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CFG = ROOT / "configs/phase10b_reacher_cached_ranking.yaml"
DATA = ROOT / "artifacts/runs/phase2_reacher_planner_dev/progress.jsonl"
DEFAULT_OUT = ROOT / "artifacts/runs/phase10b_reacher_cached_ranking"
GENERATORS = ("mlp_cem", "native_lewm_cem", "uniform_random")
MODELS = ("direct_value", "direct_soft_policy", "direct_hard_policy",
          "shared_rank_correction", "relational_rank_correction")
BASELINES = ("supervised_mlp", "native_lewm", "rank_fusion", *MODELS[:-1])
TIE = 0.0025


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def average_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), dtype=np.float64)
    a = 0
    while a < len(order):
        b = a + 1
        while b < len(order) and x[order[b]] == x[order[a]]:
            b += 1
        out[order[a:b]] = (a + b - 1) / 2
        a = b
    return out / max(1, len(x) - 1)


def robust_z(x: np.ndarray) -> np.ndarray:
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return np.clip((x - np.median(x)) / max(float(q3 - q1), 1e-6), -10, 10)


def load_data(config: dict) -> dict:
    if sha256(DATA) != config["source"]["data_sha256"]:
        raise ValueError("Phase-2F input hash differs from the frozen config")
    rows = [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != 20:
        raise ValueError("Expected exactly 20 independent state/goal contexts")
    rows.sort(key=lambda r: int(r["seed"]))
    if [r["seed"] for r in rows] != list(range(232000, 232020)):
        raise ValueError("Unexpected state/goal seed range")
    state, actions, truth, success, mlp, lewm, aux, base = [], [], [], [], [], [], [], []
    for row in rows:
        s = np.r_[np.asarray(row["start_qpos"], np.float32) / np.pi,
                  np.asarray(row["start_qvel"], np.float32) / 10,
                  np.asarray(row["goal_qpos"], np.float32) / np.pi]
        if s.shape != (6,) or not np.isfinite(s).all():
            raise ValueError("Invalid state feature contract")
        state.append(s)
        parts = [[] for _ in range(7)]
        for gen in GENERATORS:
            g = row["groups"][gen]
            a = np.asarray(g["candidate_sequences"], np.float32)
            c = np.asarray(g["simulator_costs"], np.float32)
            y = np.asarray(g["success_bits"], bool)
            pm = np.asarray(g["supervised_mlp"]["model_costs"], np.float32)
            pl = np.asarray(g["native_lewm"]["model_costs"], np.float32)
            if a.shape != (300, 25, 2) or any(v.shape != (300,) for v in (c, y, pm, pl)):
                raise ValueError("Candidate, cost, or success shape mismatch")
            if not np.isfinite(a).all() or np.any(np.abs(a) > 1.000001):
                raise ValueError("Action candidate outside normalized bounds")
            if not all(np.isfinite(v).all() for v in (c, pm, pl)):
                raise ValueError("Nonfinite candidate outcomes")
            rm, rl = average_rank(pm), average_rank(pl)
            af = a.reshape(300, 50)
            xaux = np.stack([rm, rl, robust_z(pm), robust_z(pl)], -1).astype(np.float32)
            vals = (af, c, y, pm, pl, np.concatenate([af, xaux], -1), rm.astype(np.float32))
            for target, value in zip(parts, vals):
                target.append(value)
        for target, value in zip((actions, truth, success, mlp, lewm, aux, base), parts):
            target.append(value)
    return {"seeds": np.arange(232000, 232020), "state": np.asarray(state, np.float32),
            "actions": np.asarray(actions, np.float32), "truth": np.asarray(truth, np.float32),
            "success": np.asarray(success, bool), "mlp": np.asarray(mlp, np.float32),
            "lewm": np.asarray(lewm, np.float32), "aux": np.asarray(aux, np.float32),
            "base": np.asarray(base, np.float32)}


def build_model(name: str):
    from consequence_lab.decision_transfer import (
        DirectCandidateModel, RelationalRankCorrection, SharedRankCorrection)
    if name == "direct_value":
        return DirectCandidateModel("cost")
    if name in ("direct_soft_policy", "direct_hard_policy"):
        return DirectCandidateModel("logit")
    if name == "shared_rank_correction":
        return SharedRankCorrection(64, 0.2)
    if name == "relational_rank_correction":
        return RelationalRankCorrection(64, 4, 2, 0.2)
    raise ValueError(name)


def train_model(name, data, train_idx, seed, config, device, updates):
    from consequence_lab.decision_transfer import ordinal_correction_loss, student_target_distribution
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                            weight_decay=config["training"]["weight_decay"])
    arrays = [data["state"], data["actions"][:, 0], data["aux"][:, 0],
              data["base"][:, 0], data["truth"][:, 0], data["success"][:, 0]]
    ts = [torch.as_tensor(x, device=device) for x in arrays]
    mask = torch.ones(ts[4].shape, dtype=torch.bool, device=device)
    history, start = [], time.perf_counter()
    model.train()
    for step in range(updates):
        chosen = rng.choice(train_idx, min(config["training"]["batch_contexts"], len(train_idx)), replace=False)
        ix = torch.as_tensor(chosen, dtype=torch.long, device=device)
        s, a, x, r, c, y = [v.index_select(0, ix) for v in ts]
        m = mask.index_select(0, ix)
        opt.zero_grad(set_to_none=True)
        if name == "direct_value":
            lo = c.min(1, keepdim=True).values
            scale = (c.max(1, keepdim=True).values - lo).clamp_min(1e-6)
            loss = F.mse_loss(model(s, a, m), (c - lo) / scale)
        elif name in ("direct_soft_policy", "direct_hard_policy"):
            logits = model(s, a, m)
            if name == "direct_soft_policy":
                target = student_target_distribution(c, config["training"]["target_soft_temperature"])
            else:
                lo = c.min(1, keepdim=True).values
                target = (c <= lo + config["training"]["hard_target_cost_tie_tolerance"]).float()
                target = target / target.sum(1, keepdim=True)
            loss = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
        else:
            scores, delta = model(s, x, r, m)
            o = config["training"]["ordinal_correction"]
            loss = ordinal_correction_loss(scores, delta, y, r,
                    temperature=o["success_mass_temperature"], margin=o["pair_margin"],
                    pair_weight=o["pair_weight"], regularization_weight=o["correction_l2_weight"],
                    local_top_k=o["local_candidates"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss for {name} at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
        opt.step()
        if step in (0, updates - 1) or (step + 1) % 250 == 0:
            history.append({"update": step + 1, "loss": float(loss.detach().cpu())})
    model.eval()
    return model, {"training_seconds": time.perf_counter() - start,
                   "parameters": sum(p.numel() for p in model.parameters()), "loss_history": history}


def infer(model, name, state, actions, aux, base):
    mask = torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
    with torch.inference_mode():
        if name == "direct_value":
            return model(state, actions, mask)
        if name in ("direct_soft_policy", "direct_hard_policy"):
            return -model(state, actions, mask)
        return model(state, aux, base, mask)[0]


def scorer_latency_ms(model, name, data, device):
    state = torch.as_tensor(data["state"][:1], device=device)
    actions = torch.as_tensor(data["actions"][:1, 0], device=device)
    aux = torch.as_tensor(data["aux"][:1, 0], device=device)
    base = torch.as_tensor(data["base"][:1, 0], device=device)
    for _ in range(5):
        infer(model, name, state, actions, aux, base)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(25):
        infer(model, name, state, actions, aux, base)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000 * (time.perf_counter() - start) / 25


def metric(scores, costs, success, actions):
    chosen = min(range(len(scores)), key=lambda j: (float(scores[j]), tuple(actions[j].tolist())))
    i, j = np.triu_indices(len(costs), 1)
    informative = np.abs(costs[i] - costs[j]) > TIE
    pd, cd = scores[i[informative]] - scores[j[informative]], costs[i[informative]] - costs[j[informative]]
    pair = None if not informative.any() else float(np.mean(np.where(pd == 0, 0.5, pd * cd > 0)))
    order = sorted(range(len(scores)), key=lambda k: (float(scores[k]), tuple(actions[k].tolist())))
    return {"success": int(success[chosen]), "availability": int(success.any()),
            "regret": float(max(0, costs[chosen] - costs.min())), "pairwise": pair,
            "topk": {str(k): int(success[order[:k]].any()) for k in (4, 8, 16, 120)}}


def ci(values, seed):
    x = np.asarray(values, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(x), size=(10000, len(x)))
    means = x[sample].mean(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="single three-update interface smoke, not evidence")
    parser.add_argument("--profile", action="store_true", help="one 100-update fit per model; compute profile only")
    parser.add_argument("--evaluate-existing", action="store_true",
                        help="evaluate all saved fold checkpoints without repeating training")
    args = parser.parse_args()
    config = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    out = (ROOT / args.run_dir if not args.run_dir.is_absolute() else args.run_dir).resolve()
    if out.exists() and not args.evaluate_existing:
        raise FileExistsError(f"Refusing to overwrite {out}")
    data = load_data(config)
    device = torch.device(args.device or config["resources"]["device_default"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but unavailable")
    torch.set_num_threads(4)
    if args.evaluate_existing:
        if args.smoke or args.profile:
            raise ValueError("--evaluate-existing cannot be combined with smoke/profile")
        required = [out / "checkpoints" / f"{name}_fold{fold}_seed{seed}.pt"
                    for fold in range(5) for seed in config["training"]["seeds"] for name in MODELS]
        if not out.is_dir() or not (out / "progress.jsonl").is_file() or not all(p.is_file() for p in required):
            raise FileNotFoundError("Existing run must contain all 75 fold checkpoints and progress records")
        records = [json.loads(line) for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(records) != 75:
            raise ValueError(f"Expected 75 completed fit records, found {len(records)}")
        record_by_key = {(r["model"], r["fold"], r["seed"]): r for r in records}
    else:
        out.mkdir(parents=True)
        (out / "checkpoints").mkdir()
    if sum((args.smoke, args.profile, args.evaluate_existing)) > 1:
        raise ValueError("Choose either --smoke or --profile")
    folds = 1 if (args.smoke or args.profile) else 5
    seeds = [0] if (args.smoke or args.profile) else config["training"]["seeds"]
    updates = 3 if args.smoke else (100 if args.profile else config["training"]["updates"])
    preds = {name: np.full((len(seeds), 20, 3, 300), np.nan, np.float32) for name in MODELS}
    preds.update({"supervised_mlp": data["mlp"], "native_lewm": data["lewm"]})
    preds["rank_fusion"] = np.stack([[0.5 * (average_rank(data["mlp"][i, g]) +
                                           average_rank(data["lewm"][i, g]))
                                     for g in range(3)] for i in range(20)]).astype(np.float32)
    start = time.perf_counter()
    fit_count, incomplete = 0, False
    latency_by_model = {name: [] for name in MODELS}
    progress = out / "progress.jsonl"
    with progress.open("a" if args.evaluate_existing else "x", encoding="utf-8") as log:
        for fold in range(folds):
            test = np.flatnonzero((data["seeds"] - 232000) % 5 == fold)
            train = np.flatnonzero((data["seeds"] - 232000) % 5 != fold)
            for si, train_seed in enumerate(seeds):
                for name in MODELS:
                    if time.perf_counter() - start >= config["resources"]["max_wall_hours"] * 3600:
                        incomplete = True
                        break
                    if args.evaluate_existing:
                        model = build_model(name).to(device)
                        checkpoint = out / "checkpoints" / f"{name}_fold{fold}_seed{train_seed}.pt"
                        payload = torch.load(checkpoint, map_location=device, weights_only=True)
                        model.load_state_dict(payload["state_dict"], strict=True)
                        model.eval()
                        stats = record_by_key[(name, fold, int(train_seed))]
                    else:
                        model, stats = train_model(name, data, train, int(train_seed), config, device, updates)
                    fit_count += 1
                    latency = stats.get("scorer_latency_ms_batch1_k300")
                    if latency is not None:
                        latency_by_model[name].append(latency)
                    for state_i in test:
                        s = torch.as_tensor(data["state"][state_i:state_i + 1], device=device)
                        for gi in range(3):
                            a = torch.as_tensor(data["actions"][state_i:state_i + 1, gi], device=device)
                            x = torch.as_tensor(data["aux"][state_i:state_i + 1, gi], device=device)
                            b = torch.as_tensor(data["base"][state_i:state_i + 1, gi], device=device)
                            preds[name][si, state_i, gi] = infer(model, name, s, a, x, b).squeeze(0).float().cpu().numpy()
                    if not args.evaluate_existing:
                        ckpt = out / "checkpoints" / f"{name}_fold{fold}_seed{train_seed}.pt"
                        torch.save({"state_dict": model.cpu().state_dict(), "name": name,
                                    "fold": fold, "seed": int(train_seed)}, ckpt)
                        log.write(json.dumps({"model": name, "fold": fold, "seed": int(train_seed),
                                              "train_states": data["seeds"][train].tolist(), **stats,
                                              "checkpoint": ckpt.name}) + "\n")
                        log.flush()
                    if sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) > config["resources"]["max_artifact_bytes"]:
                        incomplete = True
                        break
                if incomplete:
                    break
            if incomplete:
                break

    if args.smoke or args.profile:
        elapsed = time.perf_counter() - start
        summary = {"status": "smoke_only_no_scientific_result" if args.smoke else "profile_only_no_scientific_result",
            "exploratory_reuse": True, "fit_count": fit_count, "elapsed_seconds": elapsed,
            "estimated_full_matrix_seconds": elapsed * 150 if args.profile else None, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "data_sha256": sha256(DATA), "config_sha256": sha256(CFG),
            "runner_sha256": sha256(Path(__file__)),
            "model_sha256": sha256(ROOT / "src/consequence_lab/decision_transfer.py")}
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (out / "report.md").write_text(
            "# Phase 10B feasibility run\n\nThis limited run verifies data loading, model forward/backward, checkpoint writing and device/runtime feasibility. It is not a scientific result and is excluded from Phase-10B metrics.\n",
            encoding="utf-8")
        print(json.dumps({"status": summary["status"], "fit_count": fit_count,
                          "elapsed_seconds": elapsed, "output": str(out), "gpu": summary["gpu"]}, indent=2))
        return

    # Save per-state outcomes; contexts, not candidates, are the sampling unit.
    state_metrics = {}
    for name, scores in preds.items():
        if name not in MODELS:
            scores = scores[None]
        state_metrics[name] = {}
        for gi, gen in enumerate(GENERATORS):
            state_metrics[name][gen] = []
            for si in range(scores.shape[0]):
                state_metrics[name][gen].append([
                    metric(scores[si, i, gi], data["truth"][i, gi],
                           data["success"][i, gi], data["actions"][i, gi])
                    if np.isfinite(scores[si, i, gi]).all() else None for i in range(20)])
    rows = []
    for name in state_metrics:
        for gen in GENERATORS:
            per_state = []
            for i in range(20):
                context_rows = [rep[i] for rep in state_metrics[name][gen] if rep[i] is not None]
                if not context_rows:
                    continue
                per_state.append({
                    "success": float(np.mean([m["success"] for m in context_rows])),
                    "availability": float(np.mean([m["availability"] for m in context_rows])),
                    "regret": float(np.mean([m["regret"] for m in context_rows])),
                    "pairwise": float(np.mean([m["pairwise"] for m in context_rows if m["pairwise"] is not None])),
                    "topk": {str(k): float(np.mean([m["topk"][str(k)] for m in context_rows])) for k in (4, 8, 16, 120)},
                })
            rows.append({"model": name, "generator": gen, "independent_contexts": len(per_state),
                "selected_success": float(np.mean([m["success"] for m in per_state])) if per_state else None,
                "availability": float(np.mean([m["availability"] for m in per_state])) if per_state else None,
                "selected_regret": float(np.mean([m["regret"] for m in per_state])) if per_state else None,
                "regret_ci95_state_bootstrap": ci([m["regret"] for m in per_state], 905),
                "scorer_latency_ms_batch1_k300": float(np.mean(latency_by_model[name])) if latency_by_model.get(name) else None,
                "pairwise_accuracy": float(np.mean([m["pairwise"] for m in per_state])) if per_state else None,
                "topk_availability_rate_by_k": {str(k): float(np.mean([m["topk"][str(k)] for m in per_state])) for k in (4, 8, 16, 120)},
            })

    paired = {}
    relational = "relational_rank_correction"
    for gen in GENERATORS:
        for baseline in BASELINES:
            rr, rs, br, bs = [], [], [], []
            for i in range(20):
                rv = [rep[i] for rep in state_metrics[relational][gen] if rep[i] is not None]
                bv = [rep[i] for rep in state_metrics[baseline][gen] if rep[i] is not None]
                rr.append(np.mean([v["regret"] for v in rv]) if rv else np.nan)
                rs.append(np.mean([v["success"] for v in rv]) if rv else np.nan)
                br.append(np.mean([v["regret"] for v in bv]) if bv else np.nan)
                bs.append(np.mean([v["success"] for v in bv]) if bv else np.nan)
            dr, ds = np.asarray(br) - rr, np.asarray(rs) - bs
            paired[f"{gen}:{baseline}"] = {
                "regret_improvement": float(np.nanmean(dr)), "regret_ci95": ci(dr, 1005),
                "success_improvement": float(np.nanmean(ds)), "success_ci95": ci(ds, 1105)}

    elapsed = time.perf_counter() - start
    expected = 1 if args.smoke else 75
    status = "completed" if fit_count == expected and not incomplete else "incomplete_resource_stop"
    checkpoint_files = sorted((out / "checkpoints").glob("*.pt"))
    summary = {"status": "smoke_only_no_scientific_result" if args.smoke else status,
        "exploratory_reuse": True, "interpretation": "offline scoring only; not closed-loop control",
        "data_sha256": sha256(DATA), "config_sha256": sha256(CFG),
        "runner_sha256": sha256(Path(__file__)),
        "model_sha256": sha256(ROOT / "src/consequence_lab/decision_transfer.py"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "numpy": np.__version__,
        "contexts": 20, "candidate_outcomes": int(data["truth"].size), "fits_completed": fit_count,
        "elapsed_seconds": elapsed, "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "original_training_seconds": float(sum(r.get("training_seconds", 0.0) for r in records)) if args.evaluate_existing else elapsed,
        "evaluation_reused_saved_checkpoints": bool(args.evaluate_existing),
        "training_script_hash_pre_run_captured": False,
        "progress_sha256": sha256(progress),
        "checkpoints": {p.name: sha256(p) for p in checkpoint_files},
        "rows": rows, "paired_relational_vs_baselines": paired}
    np.savez_compressed(out / "predictions.npz", **{f"scores_{k}": v for k, v in preds.items()})
    summary["predictions_sha256"] = sha256(out / "predictions.npz")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = ["# Phase 10B — Cached Reacher ranking repair", "",
        f"Status: {summary['status']}. Exploratory reuse of previously inspected Phase-2F contexts.", "",
        f"- Device: {device} ({summary['gpu'] or 'CPU'}); elapsed {elapsed / 60:.1f} min; fits {fit_count}.",
        "- No simulator labels or world-model inference were run; this is offline evaluation, not closed-loop control.",
        "- The CEM banks were model-generated, so direct scorers without cached model features are not end-to-end model-free planners.", "",
        "| Model | Candidate population | Contexts | Selected success | Availability | Regret [95% state bootstrap CI] | Pairwise ranking | Top-4 / 8 / 16 / 120 availability rate | Scorer ms (K=300) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        k = r["topk_availability_rate_by_k"]
        report.append(f"| {r['model']} | {r['generator']} | {r['independent_contexts']} | {r['selected_success']} | {r['availability']} | {r['selected_regret']} {r['regret_ci95_state_bootstrap']} | {r['pairwise_accuracy']} | {k['4']:.3f} / {k['8']:.3f} / {k['16']:.3f} / {k['120']:.3f} | {r['scorer_latency_ms_batch1_k300']} |")
    report += ["", "Positive paired improvement favors relational correction:", "",
        "| Population | Comparator | Regret improvement (95% CI) | Success improvement (95% CI) |",
        "|---|---|---:|---:|"]
    for key, value in paired.items():
        gen, comparator = key.split(":", 1)
        report.append(f"| {gen} | {comparator} | {value['regret_improvement']} {value['regret_ci95']} | {value['success_improvement']} {value['success_ci95']} |")
    report += ["", "Only 20 state/goal contexts are independent. Each metric first averages the three training-seed models within a state; the bootstrap then resamples those 20 states. Grouped folds prevent candidate leakage but cannot create fresh confirmation data. Cached score features incur upstream world-model runtime cost at deployment; scorer latency excludes candidate generation and predictor inference. The training script hash was not captured before fitting; the model, config and data hashes are recorded post hoc, and this run remains exploratory. A positive screen would support a fresh confirmation protocol only. It does not establish closed-loop performance, calibrated task-success probabilities, or full D-JEPA reproduction.", "",
        "Artifacts: summary.json, predictions.npz, progress.jsonl and saved fold checkpoints."]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "fits": fit_count, "seconds": elapsed, "output": str(out), "gpu": summary["gpu"]}, indent=2))


if __name__ == "__main__":
    main()
