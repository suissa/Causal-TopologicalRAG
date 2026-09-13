"""Deterministic paired inference for CT-RAG benchmark observations."""
from __future__ import annotations
import math
import random
import statistics
from collections import defaultdict
from typing import Iterable

def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a quantile of no values")
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)

def paired_summary(left: Iterable[float | None], right: Iterable[float | None], *, seed: int = 20260913, resamples: int = 2000) -> dict:
    """Return bootstrap CIs and a two-sided paired sign-flip p-value."""
    raw = list(zip(left, right))
    pairs = [(float(a), float(b)) for a, b in raw if a is not None and b is not None]
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not pairs:
        return {"n_pairs": 0, "n_non_applicable": len(raw), "effect": None, "ci95": None, "randomization_p": None, "resampling": {"seed": seed, "bootstrap_resamples": resamples}}
    diffs = [a - b for a, b in pairs]
    effect = statistics.mean(diffs)
    rng = random.Random(seed)
    boot = [statistics.mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(resamples)]
    observed, extreme = abs(sum(diffs)), 0
    for _ in range(resamples):
        if abs(sum(diff if rng.random() < .5 else -diff for diff in diffs)) >= observed:
            extreme += 1
    return {"n_pairs": len(pairs), "n_non_applicable": len(raw) - len(pairs), "effect": effect,
            "ci95": [_quantile(boot, .025), _quantile(boot, .975)], "randomization_p": (extreme + 1) / (resamples + 1),
            "resampling": {"seed": seed, "bootstrap_resamples": resamples, "randomization_resamples": resamples, "test": "two_sided_paired_sign_flip"}}

def holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    valid = sorted(((name, p) for name, p in p_values.items() if p is not None), key=lambda item: item[1])
    output, previous, total = {name: None for name in p_values}, 0.0, len(valid)
    for index, (name, p) in enumerate(valid):
        previous = max(previous, min(1.0, (total - index) * p))
        output[name] = previous
    return output

def paired_comparisons(rows: list[dict], *, candidate: str, baselines: Iterable[str], metrics: Iterable[str], seed: int = 20260913, resamples: int = 2000) -> list[dict]:
    """Compare a candidate to baselines in exactly matched experiment cells."""
    keys = ("dataset", "seed", "query_id", "mode", "k")
    indexed: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        indexed[tuple(row.get(k) for k in keys)][str(row["baseline"])] = row
    output = []
    for metric in metrics:
        current = []
        for baseline in baselines:
            pairs = [(arms[candidate].get(metric), arms[baseline].get(metric)) for arms in indexed.values() if candidate in arms and baseline in arms]
            current.append({"candidate": candidate, "baseline": baseline, "metric": metric, **paired_summary((a for a, _ in pairs), (b for _, b in pairs), seed=seed, resamples=resamples)})
        adjusted = holm_adjust({item["baseline"]: item["randomization_p"] for item in current})
        for item in current:
            item["holm_adjusted_p_within_metric"] = adjusted[item["baseline"]]
        output.extend(current)
    return output
