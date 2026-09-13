from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

LABELS = {
    "relevance": (0, 1, 2),
    "causal_support": (0, 1, 2),
    "recovery_usefulness": (0, 1, 2),
    "unsupported_claim": (0, 1),
}


def _blind_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def make_blinded_sample(results: list[dict[str, Any]], *, seed: int, max_items: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in results:
        retrieved = row.get("retrieved_ids") or []
        for rank, node_id in enumerate(retrieved, 1):
            rows.append({
                "item_id": _blind_id(str(row.get("dataset")), str(row.get("query_id")), str(node_id), str(rank)),
                "dataset": row.get("dataset"),
                "query_id": row.get("query_id"),
                "candidate_id": node_id,
                "rank": rank,
            })
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = rows[:max_items]
    for index, row in enumerate(selected):
        row["presentation_order"] = index + 1
    return selected


def validate_annotation(row: dict[str, Any]) -> None:
    if not row.get("item_id") or not row.get("annotator_id"):
        raise ValueError("annotation requires item_id and annotator_id")
    for field, allowed in LABELS.items():
        value = row.get(field)
        if value not in allowed:
            raise ValueError(f"{field} must be one of {allowed}")


def cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float | None:
    if len(labels_a) != len(labels_b):
        raise ValueError("paired labels must have equal length")
    if not labels_a:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    categories = sorted(set(labels_a) | set(labels_b))
    expected = 0.0
    for category in categories:
        pa = sum(value == category for value in labels_a) / len(labels_a)
        pb = sum(value == category for value in labels_b) / len(labels_b)
        expected += pa * pb
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def agreement(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    for row in annotations:
        validate_annotation(row)
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        by_item[str(row["item_id"])].append(row)
    paired = [rows for rows in by_item.values() if len({row["annotator_id"] for row in rows}) >= 2]
    result: dict[str, Any] = {"paired_items": len(paired), "agreement_before_adjudication": {}}
    for field in LABELS:
        left: list[int] = []
        right: list[int] = []
        for rows in paired:
            unique = {}
            for row in rows:
                unique.setdefault(str(row["annotator_id"]), row)
            pair = [unique[key] for key in sorted(unique)[:2]]
            left.append(int(pair[0][field]))
            right.append(int(pair[1][field]))
        result["agreement_before_adjudication"][field] = cohen_kappa(left, right)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CT-RAG blinded human annotation utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("results", type=Path)
    sample.add_argument("output", type=Path)
    sample.add_argument("--seed", type=int, default=20260912)
    sample.add_argument("--max-items", type=int, default=100)
    score = sub.add_parser("agreement")
    score.add_argument("annotations", type=Path)
    args = parser.parse_args()
    if args.command == "sample":
        raw = json.loads(args.results.read_text(encoding="utf-8"))
        rows = raw.get("results", raw if isinstance(raw, list) else [])
        payload = {"schema_version": 1, "seed": args.seed, "system_identity_hidden": True,
                   "items": make_blinded_sample(rows, seed=args.seed, max_items=args.max_items)}
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        raw = json.loads(args.annotations.read_text(encoding="utf-8"))
        rows = raw.get("annotations", raw if isinstance(raw, list) else [])
        print(json.dumps(agreement(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
