from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .models import EvidenceRoute, EvidenceShape, RawEvidence


class SemanticShapeClassifier(Protocol):
    """Optional ambiguity resolver. Deterministic routing always runs first."""

    def classify(self, evidence: RawEvidence) -> EvidenceRoute | None: ...


class EvidenceRouter:
    VERSION = "evidence-router-deterministic-v1"

    _ADAPTERS = {
        EvidenceShape.CONFIG_TREE: "config-tree-v1",
        EvidenceShape.TRACE: "trace-v1",
        EvidenceShape.METRIC_SERIES: "metric-series-v1",
    }

    _CODE_EXTENSIONS = {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".zig",
        ".java", ".kt", ".kts", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs",
        ".hs", ".lhs", ".pl", ".pro", ".rb", ".php", ".sh", ".ps1",
    }
    _CONFIG_FILENAMES = re.compile(r"(config|settings|manifest|core|values|application)", re.I)

    def __init__(self, semantic_fallback: SemanticShapeClassifier | None = None) -> None:
        self.semantic_fallback = semantic_fallback

    @staticmethod
    def _mapping(content: Any) -> Mapping[str, Any] | None:
        return content if isinstance(content, Mapping) else None

    @staticmethod
    def _trace_like(mapping: Mapping[str, Any]) -> bool:
        keys = set(mapping)
        return (
            {"trace_id", "span_id"} <= keys
            or "spans" in keys
            or {"traceId", "spanId"} <= keys
        )

    @staticmethod
    def _event_like(mapping: Mapping[str, Any]) -> bool:
        keys = set(mapping)
        return (
            {"event_id", "event_type", "timestamp"} <= keys
            or {"id", "type", "timestamp", "causation_id"} <= keys
        )

    @staticmethod
    def _log_like(mapping: Mapping[str, Any]) -> bool:
        keys = set(mapping)
        return "message" in keys and bool(keys.intersection({"level", "severity", "logger"}))

    @staticmethod
    def _sample_like(item: Any) -> bool:
        if isinstance(item, Mapping):
            keys = set(item)
            return bool(keys.intersection({"value", "v"})) and bool(
                keys.intersection({"timestamp", "time", "ts", "at"})
            )
        return isinstance(item, (int, float))

    @classmethod
    def _metric_like(cls, mapping: Mapping[str, Any]) -> bool:
        for key in ("samples", "points", "values", "series"):
            value = mapping.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
                if all(cls._sample_like(item) for item in value):
                    return bool(set(mapping).intersection({"metric", "name", "labels", "unit"}))
        return False

    @staticmethod
    def _table_like(content: Any) -> bool:
        if not isinstance(content, list) or len(content) < 2:
            return False
        if not all(isinstance(row, Mapping) for row in content):
            return False
        keys = [set(row) for row in content]
        return bool(keys[0]) and all(item == keys[0] for item in keys[1:])

    @classmethod
    def _config_by_filename(cls, evidence: RawEvidence) -> bool:
        if evidence.metadata.get("source_kind") == "config":
            return True
        if not evidence.filename:
            return False
        path = Path(evidence.filename)
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml", ".toml"}:
            return True
        return suffix == ".json" and bool(cls._CONFIG_FILENAMES.search(path.stem))

    @staticmethod
    def _parse_structured_text(evidence: RawEvidence) -> Any | None:
        if not isinstance(evidence.content, str) or not evidence.filename:
            return None
        suffix = Path(evidence.filename).suffix.lower()
        try:
            if suffix == ".json":
                return json.loads(evidence.content)
            if suffix == ".toml":
                return tomllib.loads(evidence.content)
        except (ValueError, tomllib.TOMLDecodeError):
            return None
        return None

    @staticmethod
    def _route(
        evidence: RawEvidence,
        shape: EvidenceShape,
        *,
        confidence: float,
        method: str,
        preserve: tuple[str, ...],
    ) -> EvidenceRoute:
        return EvidenceRoute(
            shape=shape,
            confidence=confidence,
            method_id=method,
            adapter_id=EvidenceRouter._ADAPTERS.get(shape),
            preserve=preserve,
            source_metadata={
                "source_id": evidence.source_id,
                "source_uri": evidence.source_uri,
                "filename": evidence.filename,
                "media_type": evidence.media_type,
                **dict(evidence.metadata),
            },
            classifier_version=EvidenceRouter.VERSION,
        )

    @staticmethod
    def _config_preserve(evidence: RawEvidence) -> tuple[str, ...]:
        fields = ["path", "value"]
        if "resolution_chain" in evidence.metadata:
            fields.append("resolution_chain")
        return tuple(fields)

    def route(self, evidence: RawEvidence) -> EvidenceRoute:
        explicit = evidence.metadata.get("shape")
        if explicit is not None:
            try:
                shape = explicit if isinstance(explicit, EvidenceShape) else EvidenceShape(str(explicit).lower())
            except ValueError:
                shape = None
            if shape is not None:
                return self._route(
                    evidence,
                    shape,
                    confidence=1.0,
                    method="metadata:shape",
                    preserve=("content",),
                )

        content = evidence.content
        parsed = self._parse_structured_text(evidence)
        mapping = self._mapping(content) or self._mapping(parsed)

        if mapping is not None and self._trace_like(mapping):
            preserve = tuple(
                field
                for field in (
                    "trace_id", "traceId", "span_id", "spanId", "parent_span_id",
                    "parentSpanId", "name", "start_time", "end_time", "attributes", "status"
                )
                if field in mapping or field in {"trace_id", "span_id", "parent_span_id"}
            )
            return self._route(
                evidence, EvidenceShape.TRACE, confidence=0.99,
                method="schema:trace", preserve=preserve,
            )

        if mapping is not None and self._metric_like(mapping):
            return self._route(
                evidence, EvidenceShape.METRIC_SERIES, confidence=0.98,
                method="schema:metric-series",
                preserve=("metric", "name", "labels", "unit", "timestamp", "value"),
            )

        if mapping is not None and self._event_like(mapping):
            return self._route(
                evidence, EvidenceShape.EVENT, confidence=0.99,
                method="schema:event",
                preserve=("event_id", "event_type", "timestamp", "causation_id", "correlation_id"),
            )

        if mapping is not None and self._log_like(mapping):
            return self._route(
                evidence, EvidenceShape.LOG, confidence=0.96,
                method="schema:log",
                preserve=("timestamp", "level", "severity", "message", "logger"),
            )

        if self._config_by_filename(evidence):
            return self._route(
                evidence, EvidenceShape.CONFIG_TREE, confidence=0.98,
                method="filename:config", preserve=self._config_preserve(evidence),
            )

        if self._table_like(content):
            return self._route(
                evidence, EvidenceShape.TABLE, confidence=0.95,
                method="schema:table", preserve=("columns", "rows"),
            )

        if evidence.filename and Path(evidence.filename).suffix.lower() in self._CODE_EXTENSIONS:
            return self._route(
                evidence, EvidenceShape.CODE, confidence=0.98,
                method="filename:code", preserve=("language", "symbol", "line_range", "content"),
            )

        if isinstance(content, str):
            words = re.findall(r"\w+", content)
            if len(words) >= 8 and "\n" not in content[:120]:
                return self._route(
                    evidence, EvidenceShape.NARRATIVE, confidence=0.85,
                    method="text:narrative", preserve=("content",),
                )

        if evidence.metadata.get("structured_payload") is True:
            return self._route(
                evidence, EvidenceShape.STRUCTURED_PAYLOAD, confidence=0.90,
                method="metadata:structured-payload", preserve=("content",),
            )

        if self.semantic_fallback is not None:
            fallback = self.semantic_fallback.classify(evidence)
            if fallback is not None:
                return fallback

        return EvidenceRoute(
            shape=None,
            confidence=0.0,
            method_id="deterministic:undetermined",
            adapter_id=None,
            preserve=(),
            source_metadata={
                "source_id": evidence.source_id,
                "source_uri": evidence.source_uri,
                "filename": evidence.filename,
                "media_type": evidence.media_type,
                **dict(evidence.metadata),
            },
            fallback_required=True,
            classifier_version=self.VERSION,
        )
