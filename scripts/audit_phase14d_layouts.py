"""Generate a Phase14D hash-only split audit; never evaluate outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from consequence_lab.phase14d import CONFIG_PATH, build_layout_audit


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    result = build_layout_audit(root)
    config = yaml.safe_load((root / CONFIG_PATH).read_text(encoding="utf-8"))
    out = root / config["outputs"]["hash_audit"]
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(out), "blocks": {k: len(v) for k, v in result["blocks"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
