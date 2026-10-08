import json

from consequence_lab.phase15_d162 import ROOT, build_lock


if __name__ == "__main__":
    lock = build_lock()
    path = ROOT / "experiments/phase15_d162_fresh_layout_relational.lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
