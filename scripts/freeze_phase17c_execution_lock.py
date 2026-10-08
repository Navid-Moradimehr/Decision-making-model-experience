"""Create the append-only Phase17C fresh-evaluation execution lock."""
from __future__ import annotations

import json

from consequence_lab.phase17_proposal_guidance import freeze_fresh_execution_lock


if __name__ == "__main__":
    print(json.dumps(freeze_fresh_execution_lock(), indent=2, sort_keys=True))
