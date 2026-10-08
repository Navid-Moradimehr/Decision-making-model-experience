"""Phase 16D: frozen uniform exact-planning budget frontier."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np

from consequence_lab.corridor_evaluation import evaluate_corridor_episode
from consequence_lab.corridor_tasks import CorridorScenario, generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase16c_prior_guided import evaluate_sequence

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase16d_planning_compute_frontier_v1.yaml"
PROTOCOL = ROOT / "reports/phase16d_planning_compute_frontier_protocol.md"
LOCK = ROOT / "experiments/phase16d_planning_compute_frontier_v1.lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def uniform_prefix_bank(horizon: int, budget: int, seed: int,
                        max_attempts: int = 1_280_000):
    """Stable unique uniform bank; smaller budgets are exact prefixes."""
    if horizon < 1 or budget < 1 or seed < 0 or max_attempts < 1:
        raise ValueError("invalid horizon, budget, seed, or attempt limit")
    total = min(budget, 9**horizon)
    proposals = [(a,) * horizon for a in range(9)][:total]
    seen = set(proposals)
    if len(proposals) == total:
        return tuple(proposals), 0, 0
    rng = random.Random(seed)
    duplicates = attempts = 0
    while len(proposals) < total and attempts < max_attempts:
        attempts += 1
        first = min(8, int(rng.random() * 9.0))
        seq = (first,) + tuple(rng.randrange(9) for _ in range(horizon - 1))
        if seq in seen:
            duplicates += 1
        else:
            seen.add(seq)
            proposals.append(seq)
    fallback = 0
    if len(proposals) < total:
        fill = random.Random(seed ^ 0x9E3779B9)
        while len(proposals) < total:
            seq = tuple(fill.randrange(9) for _ in range(horizon))
            if seq not in seen:
                seen.add(seq)
                proposals.append(seq)
                fallback += 1
    return tuple(proposals), duplicates, fallback


def choose_lexicographic_best(evaluations):
    if not evaluations:
        raise ValueError("at least one sequence evaluation is required")
    objective = min(item.objective for item in evaluations)
    tied = [item for item in evaluations if item.objective == objective]
    return min(tied, key=lambda item: item.sequence), len(tied)


def execution_order(layouts: list[dict], seed: int) -> list[dict]:
    rows = []
    budgets = [9, 16, 32, 64, 128]
    for layout in layouts:
        arms = [(rep, budget) for rep in range(3) for budget in budgets[1:]]
        random.Random(seed + int(layout["seed"])).shuffle(arms)
        stochastic = [{"method": "uniform_shooting", "replica": rep, "budget": budget}
                      for rep, budget in arms]
        deterministic = [
            {"method": "constant_action_shooting", "replica": None, "budget": 9},
            {"method": "geometry_waypoint", "replica": None, "budget": None},
        ]
        if int(layout["seed"]) % 2:
            deterministic.reverse()
            schedule = stochastic + deterministic
        else:
            schedule = deterministic + stochastic
        for i, arm in enumerate(schedule):
            rows.append({"seed": int(layout["seed"]), "layout_sha256": layout["layout_sha256"],
                         "order_index": i, **arm})
    if len(rows) != len(layouts) * 14:
        raise RuntimeError("execution schedule has the wrong number of arms")
    return rows


def evaluate_shooting(scenario: CorridorScenario, budget: int, replica: int, config: dict):
    state = scenario.initial_state
    steps_start = state.elapsed_steps
    diagnostics, latencies = [], []
    actions = Counter()
    sequences = transitions = ties = duplicates = fallbacks = available = 0
    index = 0
    started_episode = time.perf_counter()
    while state.status == "active":
        tick = time.perf_counter()
        horizon = min(int(config["planning"]["horizon"]),
                      scenario.spec.max_steps - state.elapsed_steps)
        query = (scenario.seed * 1_000_003 + index
                 + replica * int(config["planning"]["replica_seed_stride"])) % 2**32
        actual_budget = min(budget, 9**horizon)
        bank, rejected, fallback = uniform_prefix_bank(
            horizon, actual_budget, query, int(config["planning"]["common_attempt_cap"]))
        scores = [evaluate_sequence(state, scenario, seq) for seq in bank]
        best, tied_count = choose_lexicographic_best(scores)
        used_transitions = sum(item.transitions for item in scores)
        success_count = sum(item.success_reached for item in scores)
        state = corridor_step(state, best.sequence[0], scenario.spec).state
        latencies.append(time.perf_counter() - tick)
        actions[str(best.sequence[0])] += 1
        sequences += len(bank)
        transitions += used_transitions
        ties += tied_count > 1
        duplicates += rejected
        fallbacks += fallback
        available += success_count
        diagnostics.append({
            "decision_index": index, "query_seed": query, "horizon": horizon,
            "actual_budget": len(bank), "chosen_sequence": list(best.sequence),
            "chosen_first_action": best.sequence[0], "best_objective": list(best.objective),
            "tied_best_count": tied_count,
            "success_sequences_available": success_count, "exact_transitions": used_transitions,
            "duplicate_rejections": rejected, "fallback_sequences": fallback,
        })
        index += 1
    return {
        "seed": scenario.seed, "layout_sha256": scenario.layout_sha256,
        "method": "uniform_shooting", "replica": replica, "budget": budget,
        "outcome": state.status, "steps": state.elapsed_steps - steps_start,
        "decision_count": index, "exact_sequences": sequences,
        "exact_transitions": transitions, "executed_transitions": index,
        "objective_tie_decisions": int(ties), "duplicate_rejections": duplicates,
        "fallback_sequences": fallbacks, "complete_success_sequences": available,
        "selected_action_counts": dict(sorted(actions.items())),
        "latency_seconds_per_decision": latencies,
        "wall_seconds": time.perf_counter() - started_episode,
        "decision_diagnostics": diagnostics,
    }


def paired_bootstrap(a: list[float], b: list[float], seed: int, draws: int) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("paired bootstrap requires equal nonempty vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(draws, len(a)))
    diffs = (b[indices] - a[indices]).mean(axis=1)
    return {"difference": float(np.mean(b-a)), "ci95_lower": float(np.quantile(diffs, .025)),
            "ci95_upper": float(np.quantile(diffs, .975))}


def binomial_upper(k: int, n: int, confidence: float = .95) -> float:
    """One-sided Clopper-Pearson upper bound via binomial-CDF inversion."""
    if n < 1 or not 0 <= k <= n or not 0 < confidence < 1:
        raise ValueError("invalid exact-binomial arguments")
    if k == n:
        return 1.0
    alpha = 1 - confidence

    def cdf(p):
        if p <= 0:
            return 1.0
        if p >= 1:
            return 0.0
        return math.fsum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k+1))

    low, high = 0.0, 1.0
    for _ in range(80):
        mid = (low + high) / 2
        if cdf(mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _avg(values):
    return float(np.mean(values))


def _summary(rows: list[dict]) -> dict:
    lat = [x for row in rows for x in row.get("latency_seconds_per_decision", [])]
    count = len(rows)
    return {
        "episodes": count,
        "safe_success_rate": sum(r["outcome"] == "success" for r in rows) / count,
        "collision_rate": sum(r["outcome"] == "collision" for r in rows) / count,
        "timeout_rate": sum(r["outcome"] == "timeout" for r in rows) / count,
        "mean_actions_to_terminal": _avg([r["steps"] for r in rows]),
        "exact_sequences_per_decision": _avg([
            r["exact_sequences"] / max(1, r["decision_count"]) for r in rows]),
        "exact_transitions_per_episode": _avg([r["exact_transitions"] for r in rows]),
        "exact_transitions_per_decision": _avg([
            r["exact_transitions"] / max(1, r["decision_count"]) for r in rows]),
        "executed_transitions_per_episode": _avg([r["executed_transitions"] for r in rows]),
        "complete_success_sequences_per_episode": _avg([
            r["complete_success_sequences"] for r in rows]),
        "objective_tie_decisions": sum(r["objective_tie_decisions"] for r in rows),
        "duplicate_rejections": sum(r["duplicate_rejections"] for r in rows),
        "fallback_sequences": sum(r["fallback_sequences"] for r in rows),
        "decision_latency_median_seconds": float(np.median(lat)) if lat else None,
        "decision_latency_p90_seconds": float(np.quantile(lat, .90)) if lat else None,
        "mean_episode_wall_seconds": _avg([r["wall_seconds"] for r in rows]),
    }


def _ratio_ci(values: list[float], seed: int, draws: int) -> dict:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(v), size=(draws, len(v)))
    sample = v[ids].mean(axis=1)
    return {"estimate": float(v.mean()), "ci95_lower": float(np.quantile(sample, .025)),
            "ci95_upper": float(np.quantile(sample, .975))}


def summarize(records: list[dict], config: dict) -> dict:
    layouts = sorted({row["layout_sha256"] for row in records})
    keyed = {(r["method"], r.get("replica"), r.get("budget"), r["layout_sha256"]): r
             for r in records}
    if len(keyed) != len(records):
        raise ValueError("duplicate episode key")
    budgets = config["planning"]["budgets"]
    constant_rows = [keyed[("constant_action_shooting", None, 9, layout)] for layout in layouts]
    methods = {}
    for b in budgets:
        rows = (constant_rows if b == 9 else [
            keyed[("uniform_shooting", rep, b, layout)] for layout in layouts for rep in range(3)])
        methods[f"uniform_B{b}"] = _summary(rows)
    methods["constant_action_B9"] = methods["uniform_B9"]
    geometry = [keyed[("geometry_waypoint", None, None, layout)] for layout in layouts]
    methods["geometry_waypoint"] = _summary(geometry)
    replica_summary = {}
    for b in budgets:
        if b == 9:
            summary = _summary(constant_rows)
            replica_summary[str(b)] = {"0": summary, "1_reused": summary, "2_reused": summary}
        else:
            replica_summary[str(b)] = {str(rep): _summary([
                keyed[("uniform_shooting", rep, b, layout)] for layout in layouts])
                for rep in range(3)}
    primary, reference = config["planning"]["primary_budget"], config["planning"]["reference_budget"]
    rref = [[keyed[("uniform_shooting", rep, reference, layout)] for rep in range(3)]
            for layout in layouts]
    rprim = [[keyed[("uniform_shooting", rep, primary, layout)] for rep in range(3)]
             for layout in layouts]
    ref_success = [_avg([float(r["outcome"] == "success") for r in group]) for group in rref]
    p_success = [_avg([float(r["outcome"] == "success") for r in group]) for group in rprim]
    geo_success = [float(r["outcome"] == "success") for r in geometry]
    ana = config["analysis"]
    delta = paired_bootstrap(ref_success, p_success, ana["bootstrap_seed"], ana["bootstrap_draws"])
    geo_gain = paired_bootstrap(geo_success, ref_success, ana["bootstrap_seed"], ana["bootstrap_draws"])
    regressions = sum(any(a["outcome"] != "success" and b["outcome"] == "success"
        for a,b in zip(p,r)) for p,r in zip(rprim,rref))
    added_collisions = sum(any(a["outcome"] == "collision" and b["outcome"] != "collision"
        for a,b in zip(p,r)) for p,r in zip(rprim,rref))
    trans_ratios, wall_ratios = [], []
    for p, r in zip(rprim,rref):
        trans_ratios.append(_avg([x["exact_transitions"] for x in p]) /
                            max(1, _avg([x["exact_transitions"] for x in r])))
        wall_ratios.append(_avg([x["wall_seconds"] for x in p]) /
                           max(1e-12, _avg([x["wall_seconds"] for x in r])))
    trans_ci = _ratio_ci(trans_ratios, ana["bootstrap_seed"], ana["bootstrap_draws"])
    wall_ci = _ratio_ci(wall_ratios, ana["bootstrap_seed"], ana["bootstrap_draws"])
    latency_prim = [x for row in methods[f"uniform_B{primary}"] and
                    [keyed[("uniform_shooting", rep, primary, layout)]
                     for layout in layouts for rep in range(3)] for x in row["latency_seconds_per_decision"]]
    latency_ref = [x for row in [keyed[("uniform_shooting", rep, reference, layout)]
                    for layout in layouts for rep in range(3)] for x in row["latency_seconds_per_decision"]]
    latency_ratio = float(np.median(latency_prim) / np.median(latency_ref))
    ref_gate, gate = config["gates"]["reference"], config["gates"]["primary_B32"]
    geometry_pass = (geo_gain["difference"] >= ref_gate["success_gain_over_geometry_at_least"]
                     and geo_gain["ci95_lower"] > ref_gate["paired_geometry_success_gain_ci_lower_above"])
    ref_pass = (
        methods[f"uniform_B{reference}"]["safe_success_rate"] > ref_gate["mean_safe_success_strictly_above"]
        and methods[f"uniform_B{reference}"]["collision_rate"] <= ref_gate["mean_collision_at_most"]
        and all(replica_summary[str(reference)][str(r)]["safe_success_rate"]
                >= ref_gate["every_replica_safe_success_at_least"] for r in range(3))
        and all(replica_summary[str(reference)][str(r)]["collision_rate"]
                <= ref_gate["every_replica_collision_at_most"] for r in range(3))
        and geometry_pass)
    reg_upper = binomial_upper(regressions, len(layouts))
    collision_upper = binomial_upper(added_collisions, len(layouts))
    primary_pass = (
        ref_pass
        and methods[f"uniform_B{primary}"]["safe_success_rate"] > gate["mean_safe_success_strictly_above"]
        and all(replica_summary[str(primary)][str(r)]["safe_success_rate"]
                >= gate["every_replica_safe_success_at_least"] for r in range(3))
        and methods[f"uniform_B{primary}"]["collision_rate"] <= gate["mean_collision_at_most"]
        and all(replica_summary[str(primary)][str(r)]["collision_rate"]
                <= gate["every_replica_collision_at_most"] for r in range(3))
        and delta["ci95_lower"] >= gate["paired_success_difference_ci_lower_at_least"]
        and reg_upper <= gate["success_regression_layout_event_upper_at_most"]
        and collision_upper <= gate["added_collision_layout_event_upper_at_most"]
        and methods[f"uniform_B{primary}"]["exact_transitions_per_episode"]
            / max(1e-12, methods[f"uniform_B{reference}"]["exact_transitions_per_episode"])
            <= gate["mean_episode_rollout_transition_ratio_at_most"]
        and trans_ci["ci95_upper"] <= gate["episode_transition_ratio_ci_upper_at_most"]
        and latency_ratio <= gate["median_decision_latency_ratio_at_most"])
    return {
        "schema": "phase16d-result-v1", "status": "complete",
        "layout_count": len(layouts), "unique_episode_count": len(records),
        "methods": methods, "per_replica": replica_summary,
        "primary": {"budget": primary, "reference_budget": reference,
            "B32_minus_B128_safe_success": delta, "B128_minus_geometry_success": geo_gain,
            "success_regression_layout_events": regressions, "success_regression_upper95": reg_upper,
            "added_collision_layout_events": added_collisions, "added_collision_upper95": collision_upper,
            "mean_rollout_transition_ratio": trans_ci, "mean_episode_wall_ratio": wall_ci,
            "median_decision_latency_ratio": latency_ratio,
            "reference_gate_pass": bool(ref_pass), "B32_gate_pass": bool(primary_pass)},
        "gate_pass": bool(primary_pass), "constant_action_arm_reused_once_per_layout": True,
    }


def _append_episode(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def _geometry_episode(scenario: CorridorScenario, config: dict) -> dict:
    result = evaluate_corridor_episode(
        scenario, "geometry_waypoint",
        parameters={"resolution": float(config["geometry"]["resolution"]),
                    "waypoint_tolerance": float(config["geometry"]["waypoint_tolerance"])})
    return {
        "seed": result.seed, "layout_sha256": result.layout_sha256,
        "method": "geometry_waypoint", "replica": None, "budget": None,
        "outcome": result.outcome, "steps": result.steps,
        "decision_count": result.decision_count, "exact_sequences": 0,
        "exact_transitions": 0, "executed_transitions": result.steps,
        "objective_tie_decisions": 0, "duplicate_rejections": 0,
        "fallback_sequences": 0, "complete_success_sequences": 0,
        "selected_action_counts": {}, "latency_seconds_per_decision": [],
        "wall_seconds": result.wall_seconds,
        "geometry_episode_average_decision_latency": result.wall_seconds_per_decision,
        "decision_diagnostics": [],
    }


def _verify_lock(root: Path, config: dict) -> tuple[dict, dict]:
    if not LOCK.is_file():
        raise RuntimeError("Phase16D immutable execution lock is missing")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("schema") != "phase16d-execution-lock-v1" or lock.get("config_sha256") != sha256(CONFIG):
        raise RuntimeError("Phase16D lock/config mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise RuntimeError("Phase16D requires a clean locked worktree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor",
                               lock["source_commit"], head], cwd=root)
    if ancestor.returncode != 0:
        raise RuntimeError("Phase16D implementation commit is not an ancestor of this run")
    for name, expected in lock["source_sha256"].items():
        if sha256(root / name) != expected:
            raise RuntimeError(f"Phase16D source changed: {name}")
    for name, expected in lock["parents"].items():
        if sha256(root / expected["path"]) != expected["sha256"]:
            raise RuntimeError(f"Phase16D parent changed: {name}")
    for field, hash_field in (("layout_audit", "layout_audit_sha256"),
                              ("execution_order", "execution_order_sha256")):
        if sha256(root / config["outputs"][field]) != lock[hash_field]:
            raise RuntimeError(f"Phase16D {field} hash mismatch")
    audit = json.loads((root / config["outputs"]["layout_audit"]).read_text(encoding="utf-8"))
    if (audit.get("seed_start") != config["environment"]["proposed_seed_start"]
            or audit.get("layout_count") != config["environment"]["proposed_layout_count"]
            or audit.get("unique_hashes") != audit.get("layout_count")
            or audit.get("overlap_historical") != 0 or audit.get("overlap_outcome_bearing") != 0
            or audit.get("d168_seed_block_accessed") is not False
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("outcomes_or_transitions_computed") is not False):
        raise RuntimeError("Phase16D layout hash audit failed")
    for source, expected in audit["historical_source_hashes"].items():
        if sha256(root / source) != expected:
            raise RuntimeError(f"Phase16D historical source changed: {source}")
    return lock, audit


def run_locked(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    config = __import__("yaml").safe_load(CONFIG.read_text(encoding="utf-8"))
    lock, audit = _verify_lock(root, config)
    output = root / config["outputs"]["proposed_run_dir"]
    allowed = {root / config["outputs"][key] for key in ("layout_audit", "execution_order")}
    episode_path = root / config["outputs"]["episodes"]
    if output.exists():
        existing = {p.resolve() for p in output.rglob("*") if p.is_file()}
        unexpected = existing - {p.resolve() for p in allowed}
        if unexpected and unexpected != {episode_path.resolve()}:
            raise RuntimeError("refusing to overwrite or resume unexpected Phase16D artifacts")
    else:
        output.mkdir(parents=True)

    order = json.loads((root / config["outputs"]["execution_order"]).read_text(encoding="utf-8"))
    scheduled = {(r["method"], r.get("replica"), r.get("budget"), r["seed"]) for r in order}
    completed = {}
    if episode_path.exists():
        for line in episode_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["method"], row.get("replica"), row.get("budget"), row["seed"])
            if key in completed or key not in scheduled:
                raise RuntimeError("invalid or duplicate row in Phase16D append-only episode file")
            completed[key] = row
    layout_hashes = {r["seed"]: r["layout_sha256"] for r in audit["layouts"]}
    started = time.perf_counter()
    for arm in order:
        key = (arm["method"], arm.get("replica"), arm.get("budget"), arm["seed"])
        if key in completed:
            continue
        scenario = generate_staggered_corridor(int(arm["seed"]))
        if scenario.layout_sha256 != layout_hashes[arm["seed"]] or scenario.layout_sha256 != arm["layout_sha256"]:
            raise RuntimeError("runtime layout differs from locked layout identity")
        if arm["method"] == "uniform_shooting":
            row = evaluate_shooting(scenario, int(arm["budget"]), int(arm["replica"]), config)
        elif arm["method"] == "constant_action_shooting":
            row = evaluate_shooting(scenario, 9, 0, config)
            row.update(method="constant_action_shooting", replica=None)
        elif arm["method"] == "geometry_waypoint":
            row = _geometry_episode(scenario, config)
        else:
            raise RuntimeError(f"unknown Phase16D method {arm['method']}")
        _append_episode(episode_path, row)
        completed[key] = row
    if len(completed) != int(config["outputs"]["expected_unique_episode_count"]):
        raise RuntimeError(f"expected {config['outputs']['expected_unique_episode_count']} episodes, got {len(completed)}")

    result = summarize(list(completed.values()), config)
    result.update({
        "protocol_sha256": sha256(PROTOCOL), "config_sha256": sha256(CONFIG),
        "lock_sha256": sha256(LOCK),
        "layout_audit_sha256": sha256(root / config["outputs"]["layout_audit"]),
        "execution_order_sha256": sha256(root / config["outputs"]["execution_order"]),
        "source_commit": lock["source_commit"],
        "run_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "hardware": {"cpu_count": __import__("os").cpu_count()},
        "run_wall_seconds": time.perf_counter() - started,
        "scope": {"training": False, "learned_prior": False, "learned_world_model": False,
                  "dagger": False, "student_control": False, "adaptive_planning": False,
                  "d168_block_access": False, "reserve_access": False, "test_access": False,
                  "validation_access": False},
    })
    plot_path = root / config["outputs"]["frontier_plot"]
    _write_frontier_svg(result, plot_path)
    result_path = root / config["outputs"]["result"]
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "phase16d-manifest-v1", "status": result["status"],
        "lock_sha256": result["lock_sha256"], "config_sha256": result["config_sha256"],
        "protocol_sha256": result["protocol_sha256"], "layout_count": result["layout_count"],
        "episode_count": len(completed), "episode_sha256": sha256(episode_path),
        "result_sha256": sha256(result_path), "plot_sha256": sha256(plot_path),
        "layout_audit_sha256": result["layout_audit_sha256"],
        "execution_order_sha256": result["execution_order_sha256"],
        "source_commit": result["source_commit"], "run_commit": result["run_commit"],
        "scope": result["scope"],
    }
    (root / config["outputs"]["manifest"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "manifest_sha256": sha256(root / config["outputs"]["manifest"])}


def _write_frontier_svg(result: dict, path: Path) -> None:
    budgets = [9, 16, 32, 64, 128]
    rates = [result["methods"][f"uniform_B{b}"]["safe_success_rate"] for b in budgets]
    w, h, left, right, top, bottom = 720, 430, 76, 24, 32, 68
    xs = [left + i * (w-left-right)/(len(budgets)-1) for i in range(len(budgets))]
    ys = [top + (1-rate)*(h-top-bottom) for rate in rates]
    points = " ".join(f"{x:.1f},{y:.1f}" for x,y in zip(xs,ys))
    circles = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb"/>'
        f'<text x="{x:.1f}" y="{y-12:.1f}" text-anchor="middle">{v:.1%}</text>'
        for x,y,v in zip(xs,ys,rates))
    labels = "".join(f'<text x="{x:.1f}" y="{h-34}" text-anchor="middle">{b}</text>'
                     for x,b in zip(xs,budgets))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<g font-family="sans-serif" font-size="14" fill="#111827">'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{h-bottom}" stroke="#374151"/>'
        f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#374151"/>'
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="3"/>'
        f'{circles}{labels}<text x="{w/2}" y="{h-7}" text-anchor="middle">Sequences per decision</text>'
        f'<text transform="translate(20 {h/2}) rotate(-90)" text-anchor="middle">Safe success</text>'
        '</g></svg>')
    path.write_text(svg, encoding="utf-8")
