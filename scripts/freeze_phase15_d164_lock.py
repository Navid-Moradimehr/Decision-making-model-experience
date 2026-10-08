from __future__ import annotations

import json

from consequence_lab.phase15_d164 import ROOT, _cfg, build_lock


def main() -> None:
    cfg = _cfg()
    lock = build_lock()
    path = ROOT / cfg["outputs"]["lock"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"D164 lock written: {path}")


if __name__ == "__main__":
    main()
