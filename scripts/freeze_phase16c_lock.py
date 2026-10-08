from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase16c_prior_guided import _sha  # noqa: E402

CONFIG = "configs/phase16c_prior_guided_planner_v1.yaml"
PROTOCOL = "reports/phase16c_prior_guided_planner_protocol.md"
LOCK = "experiments/phase16c_prior_guided_planner_v3.lock.json"
AUDIT = "artifacts/runs/phase16c_prior_guided_planner_v1/layout_audit.json"
SOURCES = (
    "AGENTS.md", "pyproject.toml", "uv.lock", CONFIG, PROTOCOL,
    "src/consequence_lab/phase16c_prior_guided.py",
    "scripts/phase16c_prior_guided.py", "scripts/freeze_phase16c_lock.py",
    "tests/test_phase16c_prior_guided.py", "src/consequence_lab/corridor_baselines.py",
    "src/consequence_lab/corridor_evaluation.py", "src/consequence_lab/corridor_tasks.py",
    "src/consequence_lab/inertial_corridor.py", "src/consequence_lab/phase12_teacher.py",
    "src/consequence_lab/phase14c.py", "src/consequence_lab/phase14g.py",
    "src/consequence_lab/phase15_d164.py", "src/consequence_lab/phase16b_outcome_proposal.py",
    "configs/phase16b_outcome_proposal_v2.yaml", "experiments/phase16b_outcome_proposal_v2.lock.json",
    "reports/phase16b_protocol.md", "reports/phase16b_score_correction_protocol.md",
    "experiments/phase16c_prior_guided_planner_v1.lock.json",
    "experiments/phase16c_prior_guided_planner_v2.lock.json",
    "reports/phase16c_runner_abort_v1.md",
    "reports/phase16c_runner_abort_v2.md",
)


def main() -> None:
    lock_path = ROOT / LOCK
    if lock_path.exists():
        raise SystemExit("refusing to overwrite Phase 16C lock")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise SystemExit("commit Phase 16C protocol, source, tests, and second-brain update first")
    audit = json.loads((ROOT / AUDIT).read_text(encoding="utf-8"))
    if (audit.get("schema") != "phase16c-layout-audit-v1" or audit.get("seed_start") != 198000
            or audit.get("layout_count") != 100 or audit.get("unique_hashes") != 100
            or audit.get("overlap_historical") != 0 or audit.get("overlap_outcome_bearing") != 0
            or audit.get("d168_seed_block_accessed") is not False
            or audit.get("reserve_generated_or_accessed") is not False
            or audit.get("outcomes_or_transitions_computed") is not False):
        raise SystemExit("Phase 16C hash-only layout audit did not pass")
    if any(_sha(ROOT / name) != digest for name, digest in audit["historical_source_hashes"].items()):
        raise SystemExit("historical layout source changed after Phase 16C audit")
    prior_result = json.loads((ROOT / "artifacts/runs/phase16b_score_correction_v1/result.json").read_text())
    if (prior_result.get("gate_status") != "pass"
            or prior_result.get("passing_representations") != ["canonical"]
            or prior_result.get("fresh_layout_outcomes") is not False
            or prior_result.get("reserve_access") is not False):
        raise SystemExit("Phase 16B corrected prerequisite gate/scope is invalid")
    cfg = __import__("yaml").safe_load((ROOT / CONFIG).read_text(encoding="utf-8"))
    parent_paths = {key: path for key, path in cfg["parents"].items()}
    parent_hashes = {key: _sha(ROOT / path) for key, path in parent_paths.items()}
    if parent_hashes["phase16b_score_lock"] != "2D3AFC49A032AB91DD8E82D383F20A201FA67912B2FD5AD2471512024B3B461D":
        raise SystemExit("unexpected Phase 16B score lock hash")
    payload = {
        "schema": "phase16c-prior-guided-lock-v3",
        "status": "locked_retry_after_v2_training_shape_runner_abort",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: _sha(ROOT / name) for name in SOURCES},
        "config_sha256": _sha(ROOT / CONFIG),
        "protocol_sha256": _sha(ROOT / PROTOCOL),
        "layout_audit_sha256": _sha(ROOT / AUDIT),
        "layout_hashes_sha256": hashlib.sha256(json.dumps(
            audit["layouts"], sort_keys=True, separators=(",", ":")).encode()).hexdigest().upper(),
        "parent_sha256": parent_hashes,
        "authorized_development_block": {"seed_start": 198000, "layout_count": 100,
                                          "layout_sha256": [x["layout_sha256"] for x in audit["layouts"]]},
        "reassigned_prior_block": {"seed_start": 197000, "seed_stop_exclusive": 197100,
                                   "status": "remains_unopened_reserved_for_closed_D168_G2"},
        "reserve_access": False, "test_access": False, "validation_access": False,
        "dagger": False, "student_control": False, "learned_world_model": False,
        "outcome_generation_authorized": True,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"lock": LOCK, "sha256": _sha(lock_path),
                      "source_commit": payload["source_commit"],
                      "layout_count": len(payload["authorized_development_block"]["layout_sha256"])}, indent=2))


if __name__ == "__main__":
    main()
