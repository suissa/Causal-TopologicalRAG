from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol, runtime_checkable
from urllib.request import Request, urlopen

from .embedding import tokenize


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal provider contract used by CT-RAG dense retrieval."""

    def embed(self, text: str) -> tuple[float, ...]: ...


@runtime_checkable
class LexicalRetriever(Protocol):
    """Return normalized lexical scores keyed by document id."""

    def score(self, query: str, documents: Mapping[str, str]) -> dict[str, float]: ...


class IdfOverlapRetriever:
    """Dependency-free lexical proxy used by the original CT-RAG report."""

    def score(self, query: str, documents: Mapping[str, str]) -> dict[str, float]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return {document_id: 0.0 for document_id in documents}

        tokenized = {document_id: set(tokenize(text)) for document_id, text in documents.items()}
        document_frequency = Counter(
            term for terms in tokenized.values() for term in query_terms.intersection(terms)
        )
        total = max(1, len(tokenized))
        idf = {
            term: math.log((total + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
            for term in query_terms
        }
        denominator = sum(idf.values()) or 1.0
        return {
            document_id: sum(idf[term] for term in query_terms.intersection(terms)) / denominator
            for document_id, terms in tokenized.items()
        }


@dataclass(slots=True, frozen=True)
class BM25Retriever:
    """Dependency-free Okapi BM25 adapter kept for deterministic regression tests."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("BM25 k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0.0 <= self.b <= 1.0:
            raise ValueError("BM25 b must be finite and between 0 and 1")

    def score(self, query: str, documents: Mapping[str, str]) -> dict[str, float]:
        if not documents:
            return {}
        query_terms = tokenize(query)
        if not query_terms:
            return {document_id: 0.0 for document_id in documents}

        tokenized = {document_id: tokenize(text) for document_id, text in documents.items()}
        frequencies = {document_id: Counter(tokens) for document_id, tokens in tokenized.items()}
        lengths = {document_id: len(tokens) for document_id, tokens in tokenized.items()}
        average_length = sum(lengths.values()) / max(1, len(lengths))
        total = len(documents)
        document_frequency = Counter(
            term
            for term in set(query_terms)
            for tokens in tokenized.values()
            if term in tokens
        )
        raw: dict[str, float] = {}
        for document_id, term_frequency in frequencies.items():
            score = 0.0
            length = lengths[document_id]
            for term in query_terms:
                tf = term_frequency.get(term, 0)
                if not tf:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
                norm = self.k1 * (1.0 - self.b + self.b * length / max(1.0, average_length))
                score += idf * (tf * (self.k1 + 1.0)) / (tf + norm)
            raw[document_id] = score
        maximum = max(raw.values(), default=0.0)
        if maximum <= 0.0:
            return {document_id: 0.0 for document_id in documents}
        return {document_id: score / maximum for document_id, score in raw.items()}


@dataclass(slots=True, frozen=True)
class RankBM25Retriever:
    """Production comparison adapter backed by ``rank-bm25`` BM25Okapi."""

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("BM25 k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0.0 <= self.b <= 1.0:
            raise ValueError("BM25 b must be finite and between 0 and 1")
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("BM25 epsilon must be finite and non-negative")

    def score(self, query: str, documents: Mapping[str, str]) -> dict[str, float]:
        if not documents:
            return {}
        query_terms = tokenize(query)
        if not query_terms:
            return {document_id: 0.0 for document_id in documents}
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover - optional science dependency
            raise ImportError(
                "RankBM25Retriever requires the optional `rank-bm25` package"
            ) from exc

        ids = list(documents)
        corpus = [tokenize(documents[document_id]) for document_id in ids]
        engine = BM25Okapi(corpus, k1=self.k1, b=self.b, epsilon=self.epsilon)
        raw_values = [float(value) for value in engine.get_scores(query_terms)]
        if not raw_values:
            return {}
        low = min(raw_values)
        high = max(raw_values)
        if math.isclose(low, high):
            return {document_id: 0.0 for document_id in ids}
        scale = high - low
        return {
            document_id: (value - low) / scale
            for document_id, value in zip(ids, raw_values, strict=True)
        }

    def descriptor(self) -> dict[str, Any]:
        try:
            package_version = version("rank-bm25")
        except PackageNotFoundError:
            package_version = None
        return {
            "name": "BM25Okapi",
            "implementation": "rank-bm25",
            "package_version": package_version,
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
        }


class SentenceTransformersEmbedder:
    """Pinned local adapter loaded lazily from ``sentence-transformers``."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        revision: str | None = None,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        expected_dimensions: int | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional science dependency
            raise ImportError(
                "SentenceTransformersEmbedder requires the optional `sentence-transformers` package"
            ) from exc
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            trust_remote_code=False,
        )
        dimension = self._model.get_sentence_embedding_dimension()
        self.dimensions = int(dimension) if dimension is not None else None
        if expected_dimensions is not None and self.dimensions != expected_dimensions:
            raise ValueError(
                f"embedding dimension mismatch for {model_name}: "
                f"expected {expected_dimensions}, got {self.dimensions}"
            )

    def embed(self, text: str) -> tuple[float, ...]:
        vector = self._model.encode(text, normalize_embeddings=self.normalize_embeddings)
        result = tuple(float(value) for value in vector)
        if not result or any(not math.isfinite(value) for value in result):
            raise ValueError("sentence-transformers returned an invalid embedding")
        return result

    def descriptor(self) -> dict[str, Any]:
        try:
            package_version = version("sentence-transformers")
        except PackageNotFoundError:  # pragma: no cover - constructor requires the package
            package_version = None
        return {
            "name": self.model_name,
            "revision": self.revision,
            "dimensions": self.dimensions,
            "normalize_embeddings": self.normalize_embeddings,
            "device": self.device,
            "package_version": package_version,
        }


class OpenAICompatibleEmbedder:
    """OpenAI-compatible `/embeddings` adapter using only the Python stdlib."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("embedding endpoint must be non-empty")
        if not model.strip():
            raise ValueError("embedding model must be non-empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, text: str) -> tuple[float, ...]:
        payload = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:  # nosec B310 - caller-controlled endpoint
            raw: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        try:
            vector = raw["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("OpenAI-compatible embedding response is missing data[0].embedding") from exc
        result = tuple(float(value) for value in vector)
        if not result or any(not math.isfinite(value) for value in result):
            raise ValueError("embedding response must contain finite values")
        return result


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rank_constant: int = 60,
) -> list[tuple[str, float]]:
    """Fuse rankings deterministically using Reciprocal Rank Fusion."""

    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, document_id in enumerate(ranking, start=1):
            if document_id in seen:
                continue
            seen.add(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))