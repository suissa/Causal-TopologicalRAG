"""Metrics on retrieved context. None means no applicable ground-truth label."""
from __future__ import annotations

import math
from collections import deque

from ctrag.embedding import tokenize
from ctrag.models import EdgeKind, QueryMode
from ctrag.topology import CausalTopology
from .datasets import Query


def ranking_metrics(ranked: list[str], relevance: dict[str, int], k: int) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    if len(ranked) != len(set(ranked)):
        raise ValueError("ranking must contain unique node IDs")
    ranked = ranked[:k]
    relevant = {node for node, grade in relevance.items() if grade > 0}
    gains = [relevance.get(node, 0) for node in ranked]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum((2**g - 1) / math.log2(i + 2)
                for i, g in enumerate(sorted((relevance[n] for n in relevant), reverse=True)[:k]))
    return {
        "recall_at_k": len(set(ranked) & relevant) / len(relevant) if relevant else None,
        "precision_at_k": len(set(ranked) & relevant) / k,
        "mrr": next((1 / (i + 1) for i, n in enumerate(ranked) if n in relevant), 0.0) if relevant else None,
        "ndcg": dcg / ideal if ideal else None,
    }


def _context_distances(topology, anchor, context, direction):
    distances = {anchor: 0}
    queue = deque([anchor])
    while queue:
        node = queue.popleft()
        edges = topology.incoming(node, {EdgeKind.CAUSAL}) if direction == "in" else topology.outgoing(node, {EdgeKind.CAUSAL})
        for edge in edges:
            neighbor = edge.source if direction == "in" else edge.target
            if neighbor in context and neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def evaluate(ranked: list[str], query: Query, topology: CausalTopology, k: int) -> dict:
    metrics = ranking_metrics(ranked, query.relevance, k)
    ranked = ranked[:k]
    selected = set(ranked)
    context = selected | {query.anchor}
    causal = set(query.causal_nodes)
    metrics["causal_recall_at_k"] = len(selected & causal) / len(causal) if causal else None
    metrics["causal_path_recall"] = (
        sum(set(path) <= context and all(
            any(e.target == b for e in topology.outgoing(a, {EdgeKind.CAUSAL}))
            for a, b in zip(path, path[1:])) for path in query.causal_paths) / len(query.causal_paths)
        if query.causal_paths else None
    )
    direction = "in" if query.mode is QueryMode.WHY else "out"
    observed = _context_distances(topology, query.anchor, context, direction)
    # Penalize missing/disconnected evidence instead of silently dropping it.
    penalty = len(topology.nodes)
    metrics["causal_distance_error"] = (
        sum(abs(observed[node] - gold) if node in observed else penalty
            for node, gold in query.causal_distances.items()) / len(query.causal_distances)
        if query.causal_distances else None
    )
    # Reconstruction adapter uses timestamps; similarity ranking is not a trajectory.
    if query.trajectory is not None:
        predicted = sorted(context, key=lambda n: (topology.nodes[n].timestamp, n))
        gold = query.trajectory
        row = [0] * (len(gold) + 1)
        for node in predicted:
            previous = row[:]
            for j, target in enumerate(gold, 1):
                row[j] = previous[j - 1] + 1 if node == target else max(previous[j], row[j - 1])
        metrics["trajectory_reconstruction_accuracy"] = row[-1] / max(len(predicted), len(gold))
    else:
        metrics["trajectory_reconstruction_accuracy"] = None
    for metric, labels in (("basin_purity", query.basin_nodes), ("recovery_path_precision", query.recovery_nodes)):
        metrics[metric] = len(selected & set(labels)) / len(selected) if labels is not None and selected else (0.0 if labels is not None else None)
    tokens = {node: len(tokenize(topology.nodes[node].text)) for node in ranked}
    total = sum(tokens.values())
    metrics["context_token_efficiency"] = sum(count for node, count in tokens.items() if query.relevance.get(node, 0) > 0) / total if total else 0.0
    metrics["context_tokens"] = total
    return metrics
