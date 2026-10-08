"""Hash-only Phase 12c-C split audit; does not evaluate the task or labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12c_c import PHASE11_12CA_AUDIT, audit_phase12c_c_layouts


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase12cC_locked.yaml")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--output")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    blocks = {name: config["seed_splits"][name] for name in ("model_selection", "offline_gate")}
    if not args.audit:
        print(json.dumps({
            "status": "plan_only_no_layout_or_outcome_generation",
            "blocks": blocks,
            "reserve_accessed": False,
        }, indent=2))
        return 0
    if args.confirmation != "HASH_AUDIT_PHASE12C_C_ONLY":
        raise SystemExit("hash-only split audit requires --confirmation HASH_AUDIT_PHASE12C_C_ONLY")

    prior_audit_rel = config["seed_splits"]["hash_audit"]["prior_phase12cA_layout_audit"]
    prior_lock_rel = config["seed_splits"]["hash_audit"]["prior_phase12cA_authorization_lock"]
    supplemental_rel = config["seed_splits"]["hash_audit"]["supplemental_phase12b_hash_audit"]
    prior_audit_path, prior_lock_path = ROOT / prior_audit_rel, ROOT / prior_lock_rel
    supplemental_path = ROOT / supplemental_rel
    if not all(path.is_file() for path in (prior_audit_path, prior_lock_path, supplemental_path)):
        raise SystemExit("required Phase 12b/12c-A split evidence is missing")
    prior_lock = json.loads(prior_lock_path.read_text(encoding="utf-8"))
    prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
    supplemental = json.loads(supplemental_path.read_text(encoding="utf-8"))
    if prior_lock.get("authorization_status") != "authorized" or prior_lock.get("reserve_accessed") is not False:
        raise SystemExit("Phase 12c-A authorization lock does not attest its bounded scope")
    if digest(prior_audit_path) != prior_lock.get("layout_audit_sha256"):
        raise SystemExit("Phase 12c-A audit does not match its execution lock")
    if supplemental.get("new_hashes_disjoint_from_all_historical_blocks") is not True:
        raise SystemExit("supplemental Phase 12b hash audit did not pass")
    if supplemental.get("final_reserve_generated_or_accessed") is not False:
        raise SystemExit("supplemental Phase 12b audit does not attest sealed reserve")
    generator = ROOT / "src/consequence_lab/corridor_tasks.py"
    if digest(generator) != prior_audit.get("task_generator_sha256"):
        raise SystemExit("corridor layout generator changed since Phase 12c-A audit")

    report = audit_phase12c_c_layouts(config, prior_audit)
    report.update({
        "config_sha256": digest(config_path),
        "phase12cA_audit_sha256": digest(prior_audit_path),
        "phase12cA_lock_sha256": digest(prior_lock_path),
        "phase12b_supplemental_audit_sha256": digest(supplemental_path),
        "layout_generator_sha256": digest(generator),
        "audit_runner_sha256": digest(Path(__file__).resolve()),
        "audit_module_sha256": digest(ROOT / "src/consequence_lab/phase12c_c.py"),
    })
    output_path = (ROOT / (args.output or config["seed_splits"]["hash_audit"]["output"])).resolve()
    if not inside(output_path, ROOT / "artifacts/runs") or output_path.exists():
        raise SystemExit("audit output must be a new file under artifacts/runs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "phase12cC_hash_audit_passed",
        "output": str(output_path),
        "blocks": {
            name: {
                "layouts": item["unique_layouts"],
                "layout_list_sha256": item["layout_list_sha256"],
            }
            for name, item in report["new_blocks"].items()
        },
        "layout_generation_only_no_outcomes": True,
        "reserve_accessed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
