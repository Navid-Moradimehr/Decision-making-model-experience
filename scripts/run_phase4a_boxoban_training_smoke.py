"""Run the D043-authorized Boxoban medium training-only parser/solver smoke."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "phase4_boxoban_qualification.yaml"
PINS_PATH = ROOT / "experiments" / "source_pins.json"
_HEADER = re.compile(r"^\s*;\s*(\d+)\s*$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sample_ids(split_directory: str, shard_count: int, levels_per_shard: int, count: int, prefix: str) -> list[str]:
    ids = (
        f"{split_directory}/{shard:03d}.txt#{level_index:04d}"
        for shard in range(shard_count)
        for level_index in range(levels_per_shard)
    )
    return sorted(
        ids,
        key=lambda level_id: (
            hashlib.sha256(f"{prefix}{level_id}".encode("ascii")).hexdigest(),
            level_id,
        ),
    )[:count]


def _extract_level_rows(shard_text: str, wanted_index: int) -> list[str]:
    """Locate one 10-row puzzle, without parsing other puzzle contents."""
    lines = shard_text.splitlines()
    cursor = 0
    while cursor < len(lines):
        match = _HEADER.fullmatch(lines[cursor])
        if match is None:
            if lines[cursor] == "":
                cursor += 1
                continue
            raise ValueError(f"expected shard header at line {cursor + 1}")
        level_index = int(match.group(1))
        cursor += 1
        while cursor < len(lines) and lines[cursor] == "":
            cursor += 1
        rows = lines[cursor : cursor + 10]
        if len(rows) != 10:
            raise ValueError(f"truncated level {level_index}")
        if any(_HEADER.fullmatch(row) for row in rows):
            raise ValueError(f"truncated level {level_index}")
        if level_index == wanted_index:
            return rows
        cursor += 10
    raise KeyError(f"level index {wanted_index} not present in shard")


def _source_revision(config: dict, pins: dict) -> str:
    configured = config["dataset"]["revision"]
    pinned = pins["sources"]["boxoban_levels"]
    if configured != pinned["revision"]:
        raise ValueError("Boxoban source revision differs from experiments/source_pins.json")
    if pinned["variant"] != config["dataset"]["variant"]:
        raise ValueError("Boxoban variant differs from the source pin")
    return configured


def _git_capture(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _peak_process_memory_bytes() -> int:
    """Return process peak RSS using only platform standard libraries."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        size_t = ctypes.c_size_t

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", size_t),
                ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t),
                ("QuotaPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t),
                ("PeakPagefileUsage", size_t),
                ("PrivateUsage", size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCountersEx),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def run() -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    if config["project"]["authorized_stage"] != "stage0_training_split_smoke":
        raise RuntimeError("config does not authorize the Stage-0 training smoke")
    if config["project"]["validation_screen_authorized"]:
        raise RuntimeError("Stage 0 must not authorize validation evaluation")
    if config["project"]["student_training_authorized"]:
        raise RuntimeError("Stage 0 must not authorize student training")
    if config["dataset"]["test_data_access"] != "forbidden":
        raise RuntimeError("Stage 0 must keep upstream test data forbidden")

    revision = _source_revision(config, pins)
    dataset = config["dataset"]
    sampler = dataset["level_sampler"]
    prefix = sampler["hash_input"].split("{canonical_level_id}", 1)[0]
    level_count = dataset["train_level_count_reported"]
    levels_per_shard = dataset["levels_per_shard"]
    if level_count % levels_per_shard:
        raise ValueError("reported level count is not an integer number of shards")
    selected_ids = _sample_ids(
        dataset["train_directory"],
        level_count // levels_per_shard,
        levels_per_shard,
        sampler["train_smoke_count"],
        prefix,
    )

    shard_ids = sorted({level_id.split("#", 1)[0] for level_id in selected_ids})
    max_shards = sampler["train_smoke_count"]
    if len(shard_ids) > max_shards:
        raise AssertionError("selected shard count exceeded the number of sampled levels")

    shard_bytes: dict[str, bytes] = {}
    shard_hashes: dict[str, str] = {}
    for shard_path in shard_ids:
        url = f"https://raw.githubusercontent.com/google-deepmind/boxoban-levels/{revision}/{shard_path}"
        request = Request(url, headers={"User-Agent": "consequence-lab-research-smoke/1"})
        with urlopen(request, timeout=30) as response:
            payload = response.read()
        if not payload or len(payload) > 2_000_000:
            raise ValueError(f"unexpected source shard size for {shard_path}: {len(payload)}")
        # Capture the immutable-source byte hash before decoding/parsing any level.
        shard_hashes[shard_path] = _sha256(payload)
        shard_bytes[shard_path] = payload

    selected_text: dict[str, str] = {}
    parsed = {}
    for level_id in selected_ids:
        shard_path, raw_index = level_id.rsplit("#", 1)
        level_index = int(raw_index)
        shard_text = shard_bytes[shard_path].decode("ascii")
        rows = _extract_level_rows(shard_text, level_index)
        selected_text[level_id] = f"; {level_index}\n" + "\n".join(rows)
        from consequence_lab.boxoban import parse_boxoban_rows

        level = parse_boxoban_rows(rows, level_index)
        if level.board.width != 10 or level.board.height != 10:
            raise ValueError(f"unexpected board dimensions for {level_id}")
        if len(level.initial_state.boxes) != 4 or len(level.board.goals) != 4:
            raise ValueError(f"unexpected box/goal count for {level_id}")
        parsed[level_id] = level

    solver_cfg = config["stage0"]["solver"]
    results: list[dict] = []
    cpu_started = time.process_time()
    from consequence_lab.boxoban import is_solved, solve_optimal, transition

    for level_id in selected_ids:
        level = parsed[level_id]
        level_started = time.perf_counter()
        result = solve_optimal(
            level.board,
            level.initial_state,
            max_expanded_states=solver_cfg["maximum_expanded_states_per_level"],
            timeout_seconds=solver_cfg["wall_clock_seconds_per_level"],
        )
        elapsed = time.perf_counter() - level_started
        replay = level.initial_state
        replay_valid = result.solved
        if result.solved:
            for action in result.actions:
                replay = transition(level.board, replay, action)
            replay_valid = is_solved(level.board, replay)
            if not replay_valid:
                raise AssertionError(f"solver path did not solve {level_id}")
        results.append(
            {
                "level_id": level_id,
                "status": result.status,
                "proof_complete": result.proof_complete,
                "optimal_primitive_action_cost": result.primitive_action_cost,
                "expanded_states": result.expanded_states,
                "generated_states": result.generated_states,
                "wall_seconds": elapsed,
                "replay_valid": replay_valid,
                "solution_actions": list(result.actions),
            }
        )

    cpu_seconds = time.process_time() - cpu_started
    peak_memory_bytes = _peak_process_memory_bytes()
    passed = (
        len(results) == sampler["train_smoke_count"]
        and all(row["status"] == "solved" and row["replay_valid"] for row in results)
        and cpu_seconds <= config["project"]["max_cpu_hours"] * 3600
        and peak_memory_bytes <= config["stage0"]["resource_caps"]["peak_process_memory_bytes"]
    )
    total_source_bytes = sum(map(len, shard_bytes.values()))
    if total_source_bytes > config["project"]["generated_artifact_stop_bytes"]:
        raise RuntimeError("source shard download exceeds the generated-artifact cap")

    output = ROOT / config["artifacts"]["run_directory"]
    runs_root = (ROOT / "artifacts" / "runs").resolve()
    output = output.resolve()
    try:
        output.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("configured artifact directory must remain under artifacts/runs") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")

    git_head = _git_capture("rev-parse", "HEAD")
    git_status = _git_capture("status", "--porcelain")
    source_files = [
        CONFIG_PATH,
        PINS_PATH,
        Path(__file__),
        ROOT / "src" / "consequence_lab" / "boxoban.py",
        ROOT / "tests" / "test_boxoban.py",
        ROOT / "tests" / "test_config.py",
    ]
    manifest = {
        "experiment": "phase4a_boxoban_training_split_smoke",
        "authorization": "D043_stage0_only",
        "validation_or_test_levels_accessed": False,
        "student_targets_generated": False,
        "student_training_run": False,
        "git_head": git_head,
        "git_worktree_status_sha256": _sha256(git_status.encode("utf-8")),
        "config_sha256": _sha256(CONFIG_PATH.read_bytes()),
        "source_file_sha256": {
            path.relative_to(ROOT).as_posix(): _sha256(path.read_bytes())
            for path in source_files
        },
        "dataset": {
            "repository": dataset["repository"],
            "revision": revision,
            "split": dataset["train_directory"],
            "license": dataset["license"],
            "selected_level_ids": selected_ids,
            "downloaded_shard_sha256": shard_hashes,
            "downloaded_shard_bytes": total_source_bytes,
            "selected_puzzle_content_only_saved": True,
        },
        "runtime": {
            "python": sys.version,
            "platform": sys.platform,
            "cpu_seconds_solver": cpu_seconds,
            "source_download_cpu_excluded": True,
            "wall_clock_seconds_solver_sum": sum(row["wall_seconds"] for row in results),
            "peak_process_memory_bytes": peak_memory_bytes,
        },
        "solver": {
            **solver_cfg,
            "objective": "minimum_primitive_actions",
            "heuristic": "minimum_box_goal_manhattan_assignment",
            "deadlock_pruning": "static_non_goal_corner_only",
        },
        "results": results,
        "stage0_pass": passed,
        "stage1_authorized": False,
    }

    output.mkdir(parents=True, exist_ok=False)
    (output / "selected_train_levels.txt").write_text(
        "\n\n".join(selected_text[level_id] for level_id in selected_ids) + "\n",
        encoding="ascii",
    )
    (output / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    manifest = run()
    print(json.dumps({
        "stage0_pass": manifest["stage0_pass"],
        "selected_levels": manifest["dataset"]["selected_level_ids"],
        "solved": sum(row["status"] == "solved" for row in manifest["results"]),
        "unresolved": sum(row["status"] == "limit" for row in manifest["results"]),
        "proven_unsolvable": sum(row["status"] == "proven_unsolvable" for row in manifest["results"]),
        "cpu_seconds_solver": manifest["runtime"]["cpu_seconds_solver"],
        "peak_process_memory_bytes": manifest["runtime"]["peak_process_memory_bytes"],
        "stage1_authorized": manifest["stage1_authorized"],
    }, indent=2))
    if not manifest["stage0_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
