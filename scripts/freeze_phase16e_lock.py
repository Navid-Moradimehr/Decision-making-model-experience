from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consequence_lab.phase16e_budget_reliability import freeze_lock  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(freeze_lock(ROOT), indent=2, sort_keys=True))
