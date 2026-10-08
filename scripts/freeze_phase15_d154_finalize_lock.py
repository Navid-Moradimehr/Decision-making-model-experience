"""Freeze the no-retraining D154 v3 artifact finalization."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consequence_lab.phase15_d154_finalize import LOCK, ROOT as PROJECT_ROOT, build_lock  # noqa: E402

path = PROJECT_ROOT / LOCK
path.write_text(json.dumps(build_lock(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"D154 finalizer lock created: {LOCK}")
