"""Bounded, outcome-exposed exact replay for Phase23 plan-structure diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import yaml

from consequence_lab import phase17_proposal_guidance as p17
from consequence_lab import phase20_exact_plan_warmstart as p20
from consequence_lab import phase18_proposal_controls as p18
from consequence_lab.corridor_tasks import generate_staggered_corridor
from consequence_lab.inertial_corridor import corridor_step
from consequence_lab.phase16c_prior_guided import evaluate_sequence

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/phase23_structure_replay_v1.yaml"


def cfg_load(root: Path = ROOT):
    return yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))


def _dependencies(root: Path, cfg: dict) -> list[str]:
    paths = [CONFIG, "reports/phase23_replay_protocol.md",
             "src/consequence_lab/phase23_plan_structure.py",
             "src/consequence_lab/phase17_proposal_guidance.py",
             "src/consequence_lab/phase18_proposal_controls.py",
             "src/consequence_lab/phase20_exact_plan_warmstart.py",
             "src/consequence_lab/phase16c_prior_guided.py",
             "src/consequence_lab/phase16d_compute_frontier.py",
             "src/consequence_lab/corridor_baselines.py",
             "src/consequence_lab/corridor_tasks.py",
             "src/consequence_lab/inertial_corridor.py", "pyproject.toml", "uv.lock"]
    for block in cfg["inputs"]:
        run_dir = block["run_dir"]
        paths.extend([f"{run_dir}/episodes.jsonl", f"{run_dir}/manifest.json",
                      f"{run_dir}/execution_order.json", f"{run_dir}/layout_audit.json",
                      f"{run_dir}/result.json"])
        lock = ("experiments/phase21_shifted_b32_confirmation_v1.lock.json" if block["block"] == "phase21"
                else "experiments/phase22_mutation_confirmation_v1.lock.json")
        paths.append(lock)
    return sorted(set(paths))


def freeze(root: Path = ROOT):
    cfg = cfg_load(root)
    lock_path = root / cfg["outputs"]["lock"]
    out = root / cfg["outputs"]["run_dir"]
    if lock_path.exists() or out.exists():
        raise FileExistsError("Phase23 replay lock/output already exists; refusing overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        raise RuntimeError("commit Phase23 implementation/tests/protocol before freezing")
    for block in cfg["inputs"]:
        run_dir = root / block["run_dir"]
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        lock_name = ("phase21_shifted_b32_confirmation_v1.lock.json" if block["block"] == "phase21"
                     else "phase22_mutation_confirmation_v1.lock.json")
        old_lock = root / "experiments" / lock_name
        for field, path in (("records_sha256", run_dir / "episodes.jsonl"),
                            ("result_sha256", run_dir / "result.json"),
                            ("lock_sha256", old_lock)):
            if p17.sha256_file(path) != manifest[field]:
                raise RuntimeError(f"Phase23 input provenance mismatch: {path}")
        shifted_count = sum(json.loads(line)["method"] == "shifted_b32"
                            for line in (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines())
        if shifted_count != 1500:
            raise RuntimeError(f"unexpected shifted-B32 input count for {block['block']}: {shifted_count}")
    deps = _dependencies(root, cfg)
    hashes = {name: p17.sha256_file(root / name) for name in deps}
    lock = {
        "schema": cfg["schema"] + "-execution-lock-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                   text=True).strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_rule": cfg["sample"]["index_formula"],
        "expected_records": 3000,
        "hashes": hashes,
        "reserve_access": False, "test_access": False, "d168_access": False,
        "training": False, "control_integration": False,
    }
    p18.write_new(lock_path, lock)
    return {"source_commit": lock["source_commit"], "dependencies": len(hashes),
            "lock_sha256": p17.sha256_file(lock_path)}


def verify(root: Path = ROOT):
    cfg = cfg_load(root)
    lock_path = root / cfg["outputs"]["lock"]
    committed = subprocess.check_output(
        ["git", "show", f"HEAD:{lock_path.relative_to(root).as_posix()}"], cwd=root)
    if not p18.committed_text_parity(committed, lock_path.read_bytes()):
        raise RuntimeError("Phase23 replay lock must be committed unchanged")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for name, digest in lock["hashes"].items():
        if p17.sha256_file(root / name) != digest:
            raise RuntimeError(f"Phase23 replay dependency changed: {name}")
    if lock["expected_records"] != 3000 or lock["reserve_access"] or lock["test_access"] or lock["d168_access"]:
        raise RuntimeError("Phase23 replay scope differs from frozen contract")
    return cfg, lock


def hamming(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if len(left) != len(right):
        raise ValueError("sequences must have equal horizon")
    return sum(a != b for a, b in zip(left, right, strict=True))


def shell_sample(base: tuple[int, ...], radius: int, count: int, seed_material: str):
    """Sample distinct plans uniformly from a fixed-Hamming-radius shell."""
    if radius not in (2, 3) or not 0 < count <= 64:
        raise ValueError("only frozen radius-2/3 bounded samples are supported")
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big"))
    positions = tuple(combinations(range(len(base)), radius))
    seen: set[tuple[int, ...]] = set()
    while len(seen) < count:
        chosen_positions = rng.choice(positions)
        candidate = list(base)
        for pos in chosen_positions:
            choices = [a for a in range(9) if a != base[pos]]
            candidate[pos] = rng.choice(choices)
        seen.add(tuple(candidate))
    return tuple(sorted(seen))


def neighborhood(base: tuple[int, ...], radius: int, sample_seed: str):
    if radius == 0:
        return (base,)
    if radius == 1:
        out = []
        for pos in range(len(base)):
            for action in range(9):
                if action != base[pos]:
                    candidate = list(base)
                    candidate[pos] = action
                    out.append(tuple(candidate))
        return tuple(out)
    return shell_sample(base, radius, 64, sample_seed)


def selected_index(row: dict, block_offset: int) -> int:
    count = len(row["decision_diagnostics"])
    span = min(8, count - 3)
    if span < 1:
        raise ValueError("episode too short for frozen sample rule")
    return 1 + ((int(row["seed"]) * 1000003 + int(row["replica"]) * 2654435769
                 + block_offset) % span)


def replay_state(scenario, diagnostics: list[dict], index: int):
    state = scenario.initial_state
    for diagnostic in diagnostics[:index]:
        if state.elapsed_steps != diagnostic["decision_index"]:
            raise RuntimeError("saved decision index does not match reconstructed state")
        action = int(diagnostic["chosen_action"])
        if action != int(diagnostic["chosen_sequence"][0]):
            raise RuntimeError("saved executed action differs from winner first action")
        state = corridor_step(state, action, scenario.spec).state
    if state.elapsed_steps != diagnostics[index]["decision_index"] or state.status != "active":
        raise RuntimeError("selected state reconstruction is not active/aligned")
    return state


def replay_decision(row: dict, block: str, block_offset: int):
    diagnostics = row["decision_diagnostics"]
    index = selected_index(row, block_offset)
    if index + 2 >= len(diagnostics):
        raise RuntimeError("selected state does not retain the required later decisions")
    scenario = generate_staggered_corridor(int(row["seed"]))
    state = replay_state(scenario, diagnostics, index)
    previous = tuple(diagnostics[index - 1]["chosen_sequence"])
    saved = diagnostics[index]
    horizon = min(8, scenario.spec.max_steps - state.elapsed_steps)
    if horizon != saved["horizon"]:
        raise RuntimeError("reconstructed horizon differs from archive")
    query = p17.query_seed(scenario.seed, state.elapsed_steps, int(row["replica"]))
    if query != saved["query_seed"]:
        raise RuntimeError("reconstructed query seed differs from archive")
    shifted = p20.shifted_plan(previous, horizon)
    bank, _ = p20.candidate_bank(horizon, 32, query, "shift_previous_winner", previous,
                                 0, 0x51A7E20D)
    results = {tuple(seq): evaluate_sequence(state, scenario, seq) for seq in bank}
    chosen = min(results.values(), key=lambda ev: (ev.objective, ev.sequence))
    success_count = sum(ev.success_reached for ev in results.values())
    transition_count = sum(ev.transitions for ev in results.values())
    if (list(chosen.sequence) != saved["chosen_sequence"]
            or list(chosen.objective) != saved["best_objective"]
            or success_count != saved["success_candidates"]
            or transition_count != saved["exact_transitions"]
            or len(bank) != saved["budget"] or len(set(bank)) != len(bank)
            or shifted not in bank):
        raise RuntimeError("archived decision failed exact replay parity")

    ordered = sorted(results.values(), key=lambda ev: (ev.objective, ev.sequence))
    top3 = ordered[:3]
    safe_success = [ev for ev in results.values() if ev.objective[0] == 0]
    top3_cutoff = top3[-1].objective
    all_local: dict[tuple[int, ...], object] = {}
    cumulative: set[tuple[int, ...]] = set()
    radii = {}
    for radius in range(4):
        shell = neighborhood(shifted, radius,
            f"{block}|{row['seed']}|{row['replica']}|{saved['decision_index']}|r{radius}")
        new = [seq for seq in shell if seq not in cumulative]
        for seq in new:
            all_local[seq] = results.get(seq) or evaluate_sequence(state, scenario, seq)
        cumulative.update(shell)
        evaluated = [all_local[s] for s in cumulative]
        radii[str(radius)] = {
            "shell_candidates": len(shell), "new_exact_evaluations": len(new),
            "cumulative_unique_candidates": len(cumulative),
            "safe_success_available": any(ev.objective[0] == 0 for ev in evaluated),
            "top3_cutoff_matched": any(ev.objective <= top3_cutoff for ev in evaluated),
            "best_objective": list(min(ev.objective for ev in evaluated)),
        }
    distances = [hamming(shifted, tuple(ev.sequence)) for ev in top3]
    success_distances = [hamming(shifted, tuple(ev.sequence)) for ev in safe_success]
    return {
        "block": block, "seed": row["seed"], "replica": row["replica"],
        "layout_sha256": row["layout_sha256"], "decision_index": saved["decision_index"],
        "episode_outcome": row["outcome"], "archived_success_candidates": success_count,
        "shifted_sequence": list(shifted), "winner_sequence": list(chosen.sequence),
        "winner_distance": hamming(shifted, chosen.sequence),
        "distance_to_top3": min(distances),
        "distance_to_safe_success": min(success_distances) if success_distances else None,
        "bank_top3": [{"sequence": list(ev.sequence), "objective": list(ev.objective),
                       "distance": hamming(shifted, ev.sequence)} for ev in top3],
        "bank_safe_success_count": len(safe_success), "radii": radii,
        "exact_replay_parity": True,
    }


def run(root: Path, records_by_block: dict[str, Path], output: Path):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    result_path = output / "structural_records.jsonl"
    counts = {"phase21": 0, "phase22": 0}
    with result_path.open("x", encoding="utf-8", newline="\n") as handle:
        for block, path in records_by_block.items():
            offset = 0 if block == "phase21" else 1
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["method"] != "shifted_b32":
                    continue
                diagnostic = replay_decision(row, block, offset)
                handle.write(json.dumps(diagnostic, sort_keys=True) + "\n")
                counts[block] += 1
    expected = 3000
    if sum(counts.values()) != expected:
        raise RuntimeError(f"sample coverage mismatch: {counts}")
    records = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    summary = {}
    for block in ("phase21", "phase22"):
        rows = [r for r in records if r["block"] == block]
        summary[block] = {
            "records": len(rows),
            "winner_distance_histogram": {
                str(d): sum(r["winner_distance"] == d for r in rows) for d in range(9)},
            "top3_distance_le_1": sum(r["distance_to_top3"] <= 1 for r in rows),
            "safe_success_distance_defined": sum(r["distance_to_safe_success"] is not None for r in rows),
            "radius_coverage": {
                str(radius): {
                    "safe_success": sum(r["radii"][str(radius)]["safe_success_available"] for r in rows),
                    "top3_cutoff": sum(r["radii"][str(radius)]["top3_cutoff_matched"] for r in rows),
                    "exact_evaluations_added": sum(r["radii"][str(radius)]["new_exact_evaluations"] for r in rows),
                } for radius in range(4)},
            "by_archived_success_availability": {
                key: {"n": sum((r["archived_success_candidates"] > 0) == flag for r in rows),
                      "radius1_safe_success": sum(r["radii"]["1"]["safe_success_available"]
                          and ((r["archived_success_candidates"] > 0) == flag) for r in rows)}
                for key, flag in (("zero", False), ("positive", True))},
        }
    result = {"status": "complete-exploratory-replay",
              "sample_counts": counts, "record_count": sum(counts.values()),
              "summary": summary}
    result_path_json = output / "result.json"
    result_path_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify", "run"))
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(), indent=2))
        return
    cfg, lock = verify()
    if args.command == "verify":
        print(json.dumps({"verified": True, "lock_sha256": p17.sha256_file(ROOT / cfg["outputs"]["lock"]),
                          "source_commit": lock["source_commit"]}, indent=2))
        return
    out = ROOT / cfg["outputs"]["run_dir"]
    if out.exists():
        raise FileExistsError("refusing to overwrite Phase23 replay output")
    started = time.perf_counter()
    records = {block["block"]: ROOT / block["run_dir"] / "episodes.jsonl"
               for block in cfg["inputs"]}
    result = run(ROOT, records, out)
    manifest = {
        "schema": "phase23-structure-replay-manifest-v1",
        "status": result["status"], "source_commit": lock["source_commit"],
        "lock_sha256": p17.sha256_file(ROOT / cfg["outputs"]["lock"]),
        "records_sha256": p17.sha256_file(out / "structural_records.jsonl"),
        "result_sha256": p17.sha256_file(out / "result.json"),
        "record_count": result["record_count"],
        "wall_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "reserve_access": False,
        "test_access": False, "d168_access": False, "training": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
