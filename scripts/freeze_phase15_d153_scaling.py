"""Freeze the one-shot D153 coverage-scaling lock without running transitions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase15_d153_scaling import build_run_lock, D153ScalingError  # noqa: E402


def main() -> None:
    try:
        lock = build_run_lock()
    except D153ScalingError as exc:
        raise SystemExit(f"D153 lock freeze stopped: {exc}") from exc
    path = ROOT / lock["outputs"]["lock"]
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": lock["status"], "lock": str(path.relative_to(ROOT)),
                      "fresh_layouts": lock["layout_audit"]["fresh_layout_count"],
                      "nested_sizes": sorted(map(int, lock["layout_audit"]["nested_layout_hashes"]))}, indent=2))


if __name__ == "__main__":
    main()
