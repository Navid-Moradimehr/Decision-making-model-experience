from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase16d_compute_frontier import CONFIG, PROTOCOL, sha256  # noqa: E402

LOCK = ROOT / "experiments/phase16d_planning_compute_frontier_v1.lock.json"
RUN = ROOT / "artifacts/runs/phase16d_planning_compute_frontier_v1"
SOURCES = (
    "AGENTS.md", "pyproject.toml", "uv.lock",
    "configs/phase16d_planning_compute_frontier_v1.yaml",
    "reports/phase16d_planning_compute_frontier_protocol.md",
    "src/consequence_lab/phase16d_compute_frontier.py",
    "scripts/phase16d_compute_frontier.py", "scripts/freeze_phase16d_lock.py",
    "tests/test_phase16d_compute_frontier.py",
    "src/consequence_lab/phase16c_prior_guided.py",
    "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_evaluation.py",
    "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py",
    "reports/phase16c_prior_guided_planner_result.md",
    "experiments/phase16c_prior_guided_planner_v3.lock.json",
)


def main() -> None:
    if LOCK.exists():
        raise SystemExit("refusing to overwrite the Phase16D execution lock")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("commit the amended protocol, implementation, and tests before locking")
    cfg = __import__("yaml").safe_load(CONFIG.read_text(encoding="utf-8"))
    audit_path = ROOT / cfg["outputs"]["layout_audit"]
    order_path = ROOT / cfg["outputs"]["execution_order"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    schedule = json.loads(order_path.read_text(encoding="utf-8"))
    count = int(cfg["environment"]["proposed_layout_count"])
    if (audit.get("seed_start") != cfg["environment"]["proposed_seed_start"]
            or audit.get("layout_count") != count or audit.get("unique_hashes") != count
            or audit.get("overlap_historical") != 0 or audit.get("overlap_outcome_bearing") != 0
            or audit.get("d168_seed_block_accessed") is not False
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("outcomes_or_transitions_computed") is not False
            or len(schedule) != int(cfg["outputs"]["expected_unique_episode_count"])):
        raise SystemExit("Phase16D layout audit or frozen schedule failed")
    if any(sha256(ROOT / name) != digest
           for name, digest in audit["historical_source_hashes"].items()):
        raise SystemExit("historical layout-bearing source changed after hash audit")
    parents = cfg["parents"]
    for name, parent in parents.items():
        if sha256(ROOT / parent["path"]) != parent["sha256"]:
            raise SystemExit(f"Phase16D parent hash mismatch: {name}")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "schema": "phase16d-execution-lock-v1",
        "status": "locked_approved_development_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
        "config_sha256": sha256(CONFIG), "protocol_sha256": sha256(PROTOCOL),
        "layout_audit_sha256": sha256(audit_path),
        "execution_order_sha256": sha256(order_path),
        "parents": parents,
        "authorized_development_block": {
            "seed_start": int(cfg["environment"]["proposed_seed_start"]),
            "layout_count": count,
            "layout_sha256": [row["layout_sha256"] for row in audit["layouts"]],
        },
        "reserve_access": False, "test_access": False, "validation_access": False,
        "d168_block_access": False, "training": False, "learned_prior": False,
        "student_control": False, "dagger": False, "learned_world_model": False,
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": str(LOCK.relative_to(ROOT)), "sha256": sha256(LOCK),
                      "source_commit": source_commit, "layouts": count,
                      "scheduled_episodes": len(schedule)}, indent=2))


if __name__ == "__main__":
    main()
