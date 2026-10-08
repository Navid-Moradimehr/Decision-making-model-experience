"""Hash-only Phase 12b split audit; no task rollout or reserve generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12b_data import audit_phase12b_layouts

CONFIG = ROOT / "configs" / "phase12b_action_aligned.yaml"
HISTORICAL = ROOT / "artifacts" / "runs" / "phase12_layout_audit.json"
DATA_AUDIT = ROOT / "reports" / "phase12_teacher_data_audit.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_phase12_training_data() -> None:
    audit = json.loads(DATA_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("reserve_accessed") is not False:
        raise SystemExit("historical Phase 12 teacher data audit failed or touched reserve")
    for split, dirname in (("train", "phase12_train_teacher_v1"), ("validation", "phase12_validation_teacher_v1")):
        root = ROOT / "artifacts" / "runs" / dirname
        for name, item in audit[split]["files"].items():
            path = root / name
            if not path.is_file() or digest(path) != item["sha256"]:
                raise SystemExit(f"historical Phase 12 dataset hash mismatch: {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "audit"), default="plan")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output", default="artifacts/runs/phase12b_layout_audit.json")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.mode == "plan":
        print(json.dumps({
            "status": "plan_only_no_layout_generation_or_outcomes",
            "only_new_hash_blocks": {name: config["seed_splits"][name] for name in ("model_selection", "offline_gate")},
            "reserve_generated_or_accessed": False,
        }, indent=2))
        return 0
    if args.confirmation != "AUDIT_PHASE12B_HASHES":
        raise SystemExit("hash-only audit requires --confirmation AUDIT_PHASE12B_HASHES")
    if not HISTORICAL.is_file() or not DATA_AUDIT.is_file():
        raise SystemExit("historical Phase 12 layout/data audits are required")
    verify_phase12_training_data()
    old_manifest = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    report = audit_phase12b_layouts(config, old_manifest)
    report.update({
        "config_sha256": digest(CONFIG),
        "historical_layout_manifest_sha256": digest(HISTORICAL),
        "phase12_teacher_data_audit_sha256": digest(DATA_AUDIT),
        "audit_runner_sha256": digest(Path(__file__).resolve()),
        "audit_module_sha256": digest(ROOT / "src" / "consequence_lab" / "phase12b_data.py"),
        "reserve_generated_or_accessed": False,
    })
    output = (ROOT / args.output).resolve()
    artifacts_root = (ROOT / "artifacts" / "runs").resolve()
    if artifacts_root not in output.parents:
        raise SystemExit("output must remain under artifacts/runs")
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "phase12b_hash_audit_passed",
        "output": str(output),
        "new_blocks": {name: {"count": len(row["seeds"]), "layout_list_sha256": row["layout_list_sha256"]}
                       for name, row in report["new_blocks"].items()},
        "historical_blocks_checked": report["historical_blocks_checked"],
        "reserve_generated_or_accessed": False,
        "closed_loop_development_or_dagger_generated": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
