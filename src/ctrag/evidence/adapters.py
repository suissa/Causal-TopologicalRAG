from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .models import (
    EvidenceRoute,
    EvidenceShape,
    LocalRelation,
    LocalRelationKind,
    RawEvidence,
    TypedEvidenceAnchor,
)


class EvidenceAdapter(Protocol):
    adapter_id: str
    version: str
    shape: EvidenceShape

    def adapt(self, evidence: RawEvidence, route: EvidenceRoute) -> list[TypedEvidenceAnchor]: ...


def _lineage(evidence: RawEvidence, route: EvidenceRoute, adapter_id: str, version: str) -> dict[str, Any]:
    return {
        "source_id": evidence.source_id,
        "source_uri": evidence.source_uri,
        "route_method": route.method_id,
        "classifier_version": route.classifier_version,
        "adapter_id": adapter_id,
        "adapter_version": version,
        "source_metadata": dict(route.source_metadata),
    }


def _assert_route(route: EvidenceRoute, expected: EvidenceShape) -> None:
    if route.shape is not expected:
        raise ValueError(f"adapter requires route {expected.value!r}, got {route.shape!r}")


def _summary(value: Any, limit: int = 180) -> str:
    rendered = json.dumps(value, sort_keys=True, default=str) if not isinstance(value, str) else value
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


class ConfigTreeAdapter:
    adapter_id = "config-tree-v1"
    version = "v1"
    shape = EvidenceShape.CONFIG_TREE

    @staticmethod
    def _parse(evidence: RawEvidence) -> Any:
        if not isinstance(evidence.content, str):
            return evidence.content
        if not evidence.filename:
            return evidence.content
        suffix = Path(evidence.filename).suffix.lower()
        try:
            if suffix == ".json":
                return json.loads(evidence.content)
            if suffix == ".toml":
                return tomllib.loads(evidence.content)
        except (ValueError, tomllib.TOMLDecodeError):
            return evidence.content
        # YAML parsing is intentionally dependency-free in the skeleton.
        # The raw text remains preserved rather than heuristically rewritten.
        return evidence.content

    @staticmethod
    def _flatten(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
        if isinstance(value, Mapping):
            items: list[tuple[tuple[str, ...], Any]] = []
            for key in sorted(value, key=str):
                items.extend(ConfigTreeAdapter._flatten(value[key], path + (str(key),)))
            return items or [(path, {})]
        if isinstance(value, list):
            items = []
            for index, child in enumerate(value):
                items.extend(ConfigTreeAdapter._flatten(child, path + (str(index),)))
            return items or [(path, [])]
        return [(path, value)]

    @staticmethod
    def _resolution_chain(evidence: RawEvidence, dotted_path: str) -> Any | None:
        chains = evidence.metadata.get("resolution_chain")
        if isinstance(chains, Mapping):
            return chains.get(dotted_path)
        return chains if isinstance(chains, list) else None

    def adapt(self, evidence: RawEvidence, route: EvidenceRoute) -> list[TypedEvidenceAnchor]:
        _assert_route(route, self.shape)
        parsed = self._parse(evidence)
        lineage = _lineage(evidence, route, self.adapter_id, self.version)

        if isinstance(parsed, str):
            preserved = {"raw_text": parsed}
            missing = [field for field in route.preserve if field not in {"path", "value"} and field not in preserved]
            if missing:
                raise ValueError(f"cannot preserve required config fields: {missing}")
            return [
                TypedEvidenceAnchor(
                    id=f"{evidence.source_id}:config:root",
                    shape=self.shape,
                    source_id=evidence.source_id,
                    source_uri=evidence.source_uri,
                    observed_at=evidence.observed_at,
                    content_summary=_summary(parsed),
                    structural_locator={"path": "$"},
                    preserved_fields=preserved,
                    retrieval_features={"path_tokens": []},
                    provenance=lineage,
                    confidence=route.confidence,
                    classifier_version=route.classifier_version,
                    adapter_version=self.version,
                )
            ]

        anchors: list[TypedEvidenceAnchor] = []
        for path, value in self._flatten(parsed):
            dotted = ".".join(path) if path else "$"
            preserved: dict[str, Any] = {"path": dotted, "value": value}
            chain = self._resolution_chain(evidence, dotted)
            if chain is not None:
                preserved["resolution_chain"] = chain
            missing = [field for field in route.preserve if field not in preserved]
            if missing:
                raise ValueError(f"cannot preserve required config fields at {dotted!r}: {missing}")
            anchors.append(
                TypedEvidenceAnchor(
                    id=f"{evidence.source_id}:config:{dotted}",
                    shape=self.shape,
                    source_id=evidence.source_id,
                    source_uri=evidence.source_uri,
                    observed_at=evidence.observed_at,
                    content_summary=f"{dotted}={_summary(value)}",
                    structural_locator={"path": dotted, "segments": list(path)},
                    preserved_fields=preserved,
                    retrieval_features={"path_tokens": list(path), "value_text": _summary(value)},
                    provenance=lineage,
                    confidence=route.confidence,
                    classifier_version=route.classifier_version,
                    adapter_version=self.version,
                )
            )
        return anchors


class TraceAdapter:
    adapter_id = "trace-v1"
    version = "v1"
    shape = EvidenceShape.TRACE

    @staticmethod
    def _normalize_span(span: Mapping[str, Any]) -> dict[str, Any]:
        aliases = {
            "traceId": "trace_id",
            "spanId": "span_id",
            "parentSpanId": "parent_span_id",
            "startTime": "start_time",
            "endTime": "end_time",
        }
        result = dict(span)
        for source, target in aliases.items():
            if source in result and target not in result:
                result[target] = result[source]
        return result

    def adapt(self, evidence: RawEvidence, route: EvidenceRoute) -> list[TypedEvidenceAnchor]:
        _assert_route(route, self.shape)
        if not isinstance(evidence.content, Mapping):
            raise TypeError("trace evidence must be a mapping")
        raw_spans = evidence.content.get("spans")
        spans = raw_spans if isinstance(raw_spans, list) else [evidence.content]
        normalized = [self._normalize_span(span) for span in spans if isinstance(span, Mapping)]
        if not normalized:
            raise ValueError("trace evidence contains no spans")

        lineage = _lineage(evidence, route, self.adapter_id, self.version)
        span_ids = {str(span.get("span_id")) for span in normalized if span.get("span_id") is not None}
        anchors: list[TypedEvidenceAnchor] = []

        for index, span in enumerate(normalized):
            span_id = str(span.get("span_id") or f"span-{index}")
            anchor_id = f"{evidence.source_id}:span:{span_id}"
            parent = span.get("parent_span_id")
            relations: tuple[LocalRelation, ...] = ()
            if parent is not None and str(parent) in span_ids:
                relations = (
                    LocalRelation(
                        kind=LocalRelationKind.TRACE_PARENT,
                        source_anchor_id=f"{evidence.source_id}:span:{parent}",
                        target_anchor_id=anchor_id,
                        metadata={"authority": "artifact-local", "causal": False},
                    ),
                )

            core_fields = (
                "trace_id", "span_id", "parent_span_id", "name",
                "start_time", "end_time", "attributes", "status",
            )
            preserved = {field: span[field] for field in core_fields if field in span}
            anchors.append(
                TypedEvidenceAnchor(
                    id=anchor_id,
                    shape=self.shape,
                    source_id=evidence.source_id,
                    source_uri=evidence.source_uri,
                    observed_at=evidence.observed_at,
                    content_summary=_summary({
                        "name": span.get("name"),
                        "status": span.get("status"),
                        "attributes": span.get("attributes", {}),
                    }),
                    structural_locator={
                        "trace_id": span.get("trace_id"),
                        "span_id": span_id,
                        "parent_span_id": parent,
                    },
                    preserved_fields=preserved,
                    retrieval_features={
                        "operation": span.get("name"),
                        "status": span.get("status"),
                        "attributes": span.get("attributes", {}),
                    },
                    provenance=lineage,
                    confidence=route.confidence,
                    classifier_version=route.classifier_version,
                    adapter_version=self.version,
                    local_relations=relations,
                )
            )
        return anchors


class MetricSeriesAdapter:
    adapter_id = "metric-series-v1"
    version = "v1"
    shape = EvidenceShape.METRIC_SERIES

    @staticmethod
    def _samples(content: Mapping[str, Any]) -> list[Any]:
        for key in ("samples", "points", "series", "values"):
            value = content.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return list(value)
        return []

    def adapt(self, evidence: RawEvidence, route: EvidenceRoute) -> list[TypedEvidenceAnchor]:
        _assert_route(route, self.shape)
        if not isinstance(evidence.content, Mapping):
            raise TypeError("metric-series evidence must be a mapping")

        samples = self._samples(evidence.content)
        metric_name = evidence.content.get("metric") or evidence.content.get("name") or "metric"
        labels = evidence.content.get("labels", {})
        unit = evidence.content.get("unit")
        lineage = _lineage(evidence, route, self.adapter_id, self.version)
        anchors: list[TypedEvidenceAnchor] = []
        previous_id: str | None = None

        for index, raw in enumerate(samples):
            if isinstance(raw, Mapping):
                timestamp = raw.get("timestamp", raw.get("time", raw.get("ts", raw.get("at"))))
                value = raw.get("value", raw.get("v"))
            else:
                timestamp = index
                value = raw
            anchor_id = f"{evidence.source_id}:metric:{index}"
            relations: tuple[LocalRelation, ...] = ()
            if previous_id is not None:
                relations = (
                    LocalRelation(
                        kind=LocalRelationKind.SERIES_ADJACENT,
                        source_anchor_id=previous_id,
                        target_anchor_id=anchor_id,
                        metadata={"authority": "artifact-local", "causal": False},
                    ),
                )
            preserved = {
                "metric": metric_name,
                "labels": labels,
                "unit": unit,
                "timestamp": timestamp,
                "value": value,
            }
            anchors.append(
                TypedEvidenceAnchor(
                    id=anchor_id,
                    shape=self.shape,
                    source_id=evidence.source_id,
                    source_uri=evidence.source_uri,
                    observed_at=evidence.observed_at,
                    content_summary=f"{metric_name} {timestamp}={value}",
                    structural_locator={"metric": metric_name, "index": index, "timestamp": timestamp},
                    preserved_fields=preserved,
                    retrieval_features={"metric": metric_name, "labels": labels, "value": value},
                    provenance=lineage,
                    confidence=route.confidence,
                    classifier_version=route.classifier_version,
                    adapter_version=self.version,
                    local_relations=relations,
                )
            )
            previous_id = anchor_id
        return anchors


_ADAPTERS: dict[EvidenceShape, EvidenceAdapter] = {
    EvidenceShape.CONFIG_TREE: ConfigTreeAdapter(),
    EvidenceShape.TRACE: TraceAdapter(),
    EvidenceShape.METRIC_SERIES: MetricSeriesAdapter(),
}


def adapter_for(route: EvidenceRoute) -> EvidenceAdapter:
    if route.shape is None:
        raise ValueError("cannot select an adapter for an undetermined route")
    try:
        return _ADAPTERS[route.shape]
    except KeyError as exc:
        raise KeyError(f"no executable adapter registered for {route.shape.value!r}") from exc
