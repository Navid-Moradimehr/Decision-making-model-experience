"""Run synthetic integrity and same-training-data Phase14C memorization only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import yaml

from consequence_lab.phase14 import sha256_file
from consequence_lab.phase14c import (
    CONFIG_PATH,
    Phase14CError,
    run_memorization_ladder,
    run_synthetic_integrity,
    validate_training_lock,
    write_phase14c_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    run_dir = root / config["outputs"]["run_dir"]
    report_path = root / config["outputs"]["report"]
    if run_dir.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite Phase14C run artifacts")
    lock, _, rows, _ = validate_training_lock(root, config)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "schema": "phase14c_memorization_result_v2",
        "status": "running_synthetic_integrity",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": lock["source_commit"],
        "lock_sha256": sha256_file(root / config["outputs"]["lock"]),
        "dataset_sha256": lock["phase14b_records_sha256"],
        "state_count": len(rows),
        "action_value_count": int(lock["action_values"]),
        "scope": {
            "training_split_only": True,
            "heldout_access": False,
            "closed_loop": False,
            "dagger": False,
            "world_model": False,
            "reserve_access": False,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    metrics: dict[str, object] = {"synthetic": None, "memorization_ladder": None}
    source_audit = json.loads((root / config["source"]["source_audit"]).read_text(encoding="utf-8"))
    try:
        synthetic = run_synthetic_integrity(
            int(config["source"]["synthetic_seed"]), list(config["features"]["variants"])
        )
        metrics["synthetic"] = synthetic
        _write_json(run_dir / "metrics.json", metrics)
        if synthetic["status"] != "passed":
            manifest["status"] = "failed_synthetic_integrity"
            manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["failure_gate"] = "synthetic_integrity"
            _write_json(run_dir / "manifest.json", manifest)
            write_phase14c_report(report_path, lock, synthetic, None)
            return

        manifest["status"] = "running_same_data_memorization"
        _write_json(run_dir / "manifest.json", manifest)
        ladder = run_memorization_ladder(root, config, rows, source_audit)
        metrics["memorization_ladder"] = ladder
        _write_json(run_dir / "metrics.json", metrics)
        manifest["status"] = ladder["status"]
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_rungs"] = ladder["rungs_completed"]
        manifest["training_split_only"] = True
        _write_json(run_dir / "manifest.json", manifest)
        write_phase14c_report(report_path, lock, synthetic, ladder)
    except BaseException as exc:
        manifest["status"] = "aborted_or_incomplete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        _write_json(run_dir / "manifest.json", manifest)
        raise
    print(json.dumps({
        "status": manifest["status"],
        "synthetic": synthetic["status"],
        "completed_rungs": manifest.get("completed_rungs", []),
        "metrics": str(run_dir / "metrics.json"),
        "report": str(report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
