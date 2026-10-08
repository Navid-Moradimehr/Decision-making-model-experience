"""Finalize metrics from the saved D154 v2 checkpoints without retraining."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from consequence_lab.phase15_d154_finalize import run_finalization  # noqa: E402

result = run_finalization()
print("D154 v3 checkpoint-only finalization completed:", result["decision_summary"])
