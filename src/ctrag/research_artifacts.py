from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


def _dot_quote(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _edge_value(edge: dict[str, Any], key: str) -> str | None:
    value = edge.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        raw = value.get("value")
        return None if raw is None else str(raw)
    return None if value is None else str(value)


def topology_dot(dataset: dict[str, Any]) -> str:
    """Generate a deterministic Graphviz DOT figure for one dataset topology."""
    lines = [
        "digraph CTRAGTopology {",
        '  graph [rankdir="LR", label="CT-RAG causal/topological terrain", labelloc="t"];',
        '  node [shape="ellipse"];',
    ]
    for node in sorted(dataset["nodes"], key=lambda item: item["id"]):
        label = f'{node["id"]}\\n{node["text"]}'
        shape = "doublecircle" if node.get("is_attractor") else "ellipse"
        lines.append(f"  {_dot_quote(node['id'])} [label={_dot_quote(label)}, shape={_dot_quote(shape)}];")
    for edge in sorted(
        dataset["edges"],
        key=lambda item: (
            str(item.get("source")), str(item.get("target")), str(_edge_value(item, "kind"))
        ),
    ):
        kind = _edge_value(edge, "kind") or "unknown"
        provenance = _edge_value(edge, "provenance")
        style = "solid" if kind == "causal" else "dashed"
        label = kind if provenance is None else f"{kind}:{provenance}"
        lines.append(
            f"  {_dot_quote(edge['source'])} -> {_dot_quote(edge['target'])} "
            f"[label={_dot_quote(label)}, style={_dot_quote(style)}];"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def causal_path_dot(dataset: dict[str, Any]) -> str:
    """Generate a deterministic figure from the first ground-truth causal path."""
    query = next(
        (item for item in dataset["queries"] if item.get("causal_paths")),
        None,
    )
    if query is None:
        return "digraph CausalPath {}\n"
    path = query["causal_paths"][0]
    text_by_id = {node["id"]: node["text"] for node in dataset["nodes"]}
    lines = [
        "digraph CausalPath {",
        '  graph [rankdir="LR", label="Ground-truth causal path", labelloc="t"];',
    ]
    for node_id in path:
        label = f"{node_id}\\n{text_by_id.get(node_id, '')}"
        shape = "box" if node_id == query.get("anchor") else "ellipse"
        lines.append(f"  {_dot_quote(node_id)} [label={_dot_quote(label)}, shape={_dot_quote(shape)}];")
    for source, target in zip(path, path[1:]):
        lines.append(f"  {_dot_quote(source)} -> {_dot_quote(target)} [label=\"causal\"];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def benchmark_table(rows: list[dict[str, str]], *, k: int = 3) -> str:
    selected = [row for row in rows if int(row["k"]) == k]
    selected.sort(key=lambda row: (row["dataset"], row["mode"], row["baseline"]))
    metrics = [
        "recall_at_k",
        "mrr",
        "ndcg",
        "causal_path_recall",
        "causal_distance_error",
        "context_token_efficiency",
    ]
    header = ["Dataset", "Mode", "System", *metrics]
    lines = [
        f"# Benchmark table (K={k})",
        "",
        "Generated from `benchmark-results/table.csv`.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in selected:
        values = [row["dataset"], row["mode"], row["baseline"]]
        for metric in metrics:
            raw = row.get(metric, "")
            if raw in {"", "None"}:
                values.append("—")
            else:
                values.append(f"{float(raw):.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def generate(benchmark_dir: Path, output_dir: Path, *, k: int = 3) -> dict[str, Any]:
    benchmark_dir = benchmark_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "config": benchmark_dir / "config.json",
        "datasets": benchmark_dir / "datasets.json",
        "table": benchmark_dir / "table.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing benchmark artifacts: {missing}")

    config = _read_json(required["config"])
    datasets = _read_json(required["datasets"])
    with required["table"].open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not datasets:
        raise ValueError("datasets.json contains no datasets")

    outputs = {
        "topology": output_dir / "topology.dot",
        "causal_path": output_dir / "causal-path.dot",
        "benchmark_table": output_dir / f"benchmark-k{k}.md",
        "readme": output_dir / "README.md",
    }
    _write(outputs["topology"], topology_dot(datasets[0]))
    _write(outputs["causal_path"], causal_path_dot(datasets[0]))
    _write(outputs["benchmark_table"], benchmark_table(rows, k=k))
    _write(
        outputs["readme"],
        "# Generated CT-RAG research artifacts\n\n"
        "These files are generated; do not hand-edit reported values.\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        f"python -m ctrag.research_artifacts --benchmark-dir benchmark-results --out research-artifacts --k {k}\n"
        "```\n\n"
        "Render DOT figures with any Graphviz-compatible renderer, for example `dot -Tsvg topology.dot -o topology.svg`.\n",
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_command": "python -m ctrag.benchmarks",
        "artifact_command": (
            f"python -m ctrag.research_artifacts --benchmark-dir benchmark-results "
            f"--out research-artifacts --k {k}"
        ),
        "k": k,
        "benchmark_config": config,
        "inputs_sha256": {name: _sha256(path) for name, path in required.items()},
        "outputs_sha256": {name: _sha256(path) for name, path in outputs.items()},
    }
    manifest_path = output_dir / "manifest.json"
    _write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate reproducible CT-RAG research artifacts")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--out", type=Path, default=Path("research-artifacts"))
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("--k must be positive")
    manifest = generate(args.benchmark_dir, args.out, k=args.k)
    print(f"Wrote research artifacts with {len(manifest['outputs_sha256'])} generated files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
