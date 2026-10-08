"""Pure layout-split hashing and paired-bootstrap helpers for Phase15 evals."""

from __future__ import annotations

import hashlib
import json

import numpy as np


def layout_list_sha256(layouts: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(layouts), separators=(",", ":")).encode()).hexdigest().upper()


def paired_layout_bootstrap(pairwise: dict[str, float], regression: dict[str, float],
                            seed: int, resamples: int) -> dict:
    common = sorted(set(pairwise) & set(regression))
    if not common:
        raise ValueError("paired layout bootstrap needs common layouts")
    delta = np.asarray([pairwise[key] - regression[key] for key in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    return {"layout_count": len(common), "pairwise_minus_regression_mean": float(delta.mean()),
            "ci_95_lower": float(np.quantile(samples, 0.025)),
            "ci_95_upper": float(np.quantile(samples, 0.975)),
            "resamples": int(resamples), "seed": int(seed), "samples": samples.tolist(),
            "per_layout_differences": {key: float(value) for key, value in zip(common, delta, strict=True)}}
