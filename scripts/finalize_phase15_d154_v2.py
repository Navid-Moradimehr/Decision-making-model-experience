"""Recompute D154 metrics from the existing v2 checkpoints, without training."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consequence_lab.phase15_d154_finalize import run_finalization  # noqa: E402

result = run_finalization()
print("D154 v2 finalization completed:", result["decision_summary"])
