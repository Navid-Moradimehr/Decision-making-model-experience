"""Hash-only audit for the approved Phase 12c-A layout block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase12c import audit_phase12c_layouts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase12cA_locked.yaml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-only", action="store_true")
    args = parser.parse_args()
    if not args.hash_only:
        parser.error("--hash-only is required; this script never evaluates outcomes")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    report = audit_phase12c_layouts(config)
    report["task_generator_sha256"] = hashlib.sha256(
        (ROOT / "src/consequence_lab/corridor_tasks.py").read_bytes()
    ).hexdigest().upper()
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output = output.resolve()
        runs = (ROOT / "artifacts/runs").resolve()
        try:
            output.relative_to(runs)
        except ValueError:
            parser.error("audit output must be under artifacts/runs")
        if output.exists():
            parser.error("audit output already exists; refusing overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
    print(encoded.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
