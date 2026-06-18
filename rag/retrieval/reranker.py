"""
rag/retrieval/reranker.py

Cross-encoder reranker for RAG retrieval.

After vector retrieval returns top-K candidates, the reranker scores each
(query, chunk) pair with a cross-encoder model for precise relevance ranking.
This significantly improves answer quality when multiple runbooks match.

Supports:
  - Cohere Rerank API (cloud, high quality)
  - Local cross-encoder via sentence-transformers (no API key needed)
  - No-op passthrough (fallback when neither is available)
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base reranker interface."""

    @abstractmethod
    def rerank(self, query: str, nodes: list[Any], top_n: int = 3) -> list[Any]:
        """Return top_n nodes sorted by relevance to query."""
        ...


class CohereReranker(BaseReranker):
    """Reranks using the Cohere Rerank API."""

    def __init__(self, model: str = "rerank-english-v3.0", top_n: int = 3) -> None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not set")
        import cohere
        self.client  = cohere.Client(api_key)
        self.model   = model
        self.default_top_n = top_n

    def rerank(self, query: str, nodes: list[Any], top_n: int | None = None) -> list[Any]:
        k = top_n or self.default_top_n
        if not nodes:
            return nodes

        documents = [node.get_content() for node in nodes]
        try:
            results = self.client.rerank(
                query=query,
                documents=documents,
                model=self.model,
                top_n=min(k, len(nodes)),
            )
            return [nodes[r.index] for r in results.results]
        except Exception as e:
            logger.warning("Cohere rerank failed (%s) — using original order", e)
            return nodes[:k]


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker via sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 3,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.default_top_n = top_n
            logger.info("Loaded cross-encoder: %s", model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers required for local reranking.\n"
                "Run: pip install sentence-transformers"
            )

    def rerank(self, query: str, nodes: list[Any], top_n: int | None = None) -> list[Any]:
        k = top_n or self.default_top_n
        if not nodes:
            return nodes

        pairs  = [(query, node.get_content()) for node in nodes]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, nodes), key=lambda t: t[0], reverse=True)
        return [node for _, node in ranked[:k]]


class PassthroughReranker(BaseReranker):
    """No-op reranker — returns nodes in original retrieval order."""

    def __init__(self, top_n: int = 5) -> None:
        self.default_top_n = top_n

    def rerank(self, query: str, nodes: list[Any], top_n: int | None = None) -> list[Any]:
        k = top_n or self.default_top_n
        return nodes[:k]


def get_reranker(top_n: int = 3) -> BaseReranker:
    """
    Factory: return the best available reranker.

    Priority:
        1. Cohere API (if COHERE_API_KEY is set)
        2. Local cross-encoder (if sentence-transformers installed)
        3. Passthrough (always available)
    """
    if os.getenv("COHERE_API_KEY"):
        try:
            return CohereReranker(top_n=top_n)
        except Exception as e:
            logger.warning("Cohere reranker unavailable: %s", e)

    try:
        return CrossEncoderReranker(top_n=top_n)
    except ImportError:
        pass

    logger.info("Using passthrough reranker (no reranking model available)")
    return PassthroughReranker(top_n=top_n)
