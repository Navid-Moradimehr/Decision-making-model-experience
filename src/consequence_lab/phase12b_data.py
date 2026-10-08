"""Fresh Phase-12b layout-hash audit; never generates task outcomes."""

from __future__ import annotations

import hashlib
from typing import Any

from consequence_lab.corridor_tasks import generate_staggered_corridor


def audit_phase12b_layouts(config: dict[str, Any], historical_manifest: dict[str, Any]) -> dict[str, Any]:
    if historical_manifest.get("reserve_generated_or_accessed") is not False:
        raise ValueError("historical manifest does not attest sealed reserve")
    prior: dict[str, set[str]] = {}
    for name, block in historical_manifest.get("blocks", {}).items():
        hashes = set(block["layout_sha256"])
        if len(hashes) != len(block["layout_sha256"]):
            raise ValueError(f"historical block {name} contains duplicate layout hashes")
        prior[name] = hashes
    if not prior:
        raise ValueError("historical split manifest has no audited blocks")
    prior_union = set().union(*prior.values())
    phase11_scope = historical_manifest.get("historical_phase11_seeds_audited")
    if phase11_scope != [110000, 110039, 120000, 120099]:
        raise ValueError("historical manifest does not pin the expected spent Phase 11 seed ranges")
    phase11_seeds = list(range(110000, 110040)) + list(range(120000, 120100))
    phase11_hashes = [generate_staggered_corridor(seed).layout_sha256 for seed in phase11_seeds]
    if len(phase11_hashes) != len(set(phase11_hashes)):
        raise ValueError("spent Phase 11 layout hashes contain duplicates")
    phase11_set = set(phase11_hashes)
    overlap_phase11_historical = phase11_set.intersection(prior_union)
    if overlap_phase11_historical:
        raise ValueError("Phase 11 seed-range hashes disagree/overlap stored Phase 12 split hashes")
    prior_union.update(phase11_set)

    blocks: dict[str, dict[str, Any]] = {}
    new_union: set[str] = set()
    for name in ("model_selection", "offline_gate"):
        spec = config["seed_splits"][name]
        seeds = list(range(int(spec["start"]), int(spec["start"]) + int(spec["count"])))
        hashes = [generate_staggered_corridor(seed).layout_sha256 for seed in seeds]
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"duplicate layout within Phase 12b block {name}")
        overlap_prior = prior_union.intersection(hashes)
        overlap_new = new_union.intersection(hashes)
        if overlap_prior or overlap_new:
            raise ValueError(f"layout overlap in {name}: prior={len(overlap_prior)}, new={len(overlap_new)}")
        new_union.update(hashes)
        blocks[name] = {
            "seeds": seeds,
            "layout_sha256": hashes,
            "unique_layouts": len(hashes),
            "layout_list_sha256": hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest().upper(),
        }
    return {
        "schema": "phase12b_layout_hash_audit_v1",
        "historical_manifest_schema": historical_manifest.get("schema"),
        "historical_blocks_checked": {name: len(values) for name, values in prior.items()},
        "phase11_seed_ranges_rehashed_without_outcomes": {
            "design_selection": {"start": 110000, "count": 40},
            "planner_development": {"start": 120000, "count": 100},
            "unique_layouts": len(phase11_set),
            "layout_list_sha256": hashlib.sha256("\n".join(phase11_hashes).encode("ascii")).hexdigest().upper(),
        },
        "new_blocks": blocks,
        "new_hashes_disjoint_from_all_historical_blocks": True,
        "new_blocks_pairwise_disjoint": True,
        "final_reserve_generated_or_accessed": False,
        "closed_loop_development_generated_or_accessed": False,
        "dagger_generated_or_accessed": False,
    }
