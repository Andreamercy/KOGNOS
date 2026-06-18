"""
rag/retrieval/query_engine.py

LlamaIndex query engine that combines:
  1. Vector retrieval from Qdrant (runbooks + incidents)
  2. Live anomaly context injection (from GNN inference engine)
  3. Claude / Ollama LLM for grounded, actionable answers

The engine produces answers that cite their sources (runbook names, incident IDs)
and always recommend explicit kubectl commands when suggesting remediation.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from rag.ingestion.live_context import get_live_context
from rag.agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEMO_MODE   = os.getenv("KOGNOS_DEMO_MODE", "false").lower() == "true"
LLM_BACKEND = os.getenv("LLM_BACKEND", "anthropic").lower()


def build_query_engine(
    live_alerts: list[dict] | None = None,
) -> Any:
    """
    Build and return a LlamaIndex query engine backed by Qdrant + Claude/Ollama.

    Args:
        live_alerts: Current active anomaly alerts from the inference engine.

    Returns:
        A LlamaIndex QueryEngine instance.
    """
    if DEMO_MODE:
        return _build_mock_engine(live_alerts or [])

    try:
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from qdrant_client import QdrantClient
    except ImportError as e:
        logger.warning("LlamaIndex/Qdrant not available (%s) — using mock engine", e)
        return _build_mock_engine(live_alerts or [])

    # ── Embedding model ──────────────────────────────────────────────────
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.embed_model = embed_model

    # ── LLM backend ──────────────────────────────────────────────────────
    if LLM_BACKEND == "anthropic":
        from rag.llm.claude_backend import build_claude_llm
        llm = build_claude_llm()
    else:
        from rag.llm.ollama_backend import build_ollama_llm
        llm = build_ollama_llm()

    Settings.llm = llm

    # ── Vector store ─────────────────────────────────────────────────────
    host       = os.getenv("QDRANT_HOST", "localhost")
    port       = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("QDRANT_COLLECTION", "kognos_knowledge")

    try:
        client       = QdrantClient(host=host, port=port, timeout=5)
        vector_store = QdrantVectorStore(client=client, collection_name=collection)
        index        = VectorStoreIndex.from_vector_store(vector_store)
    except Exception as e:
        logger.warning("Cannot connect to Qdrant (%s) — using mock engine", e)
        return _build_mock_engine(live_alerts or [])

    # ── Inject live context into system prompt ────────────────────────────
    live_ctx = get_live_context(live_alerts or [])
    system_prompt = SYSTEM_PROMPT.format(live_context=live_ctx)

    engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        response_mode="compact",
        system_prompt=system_prompt,
    )
    logger.info("Query engine ready (backend=%s, collection=%s)", LLM_BACKEND, collection)
    return engine


class MockQueryEngine:
    """
    Deterministic mock query engine for demo and testing.
    Returns canned responses based on keyword matching.
    """

    def __init__(self, live_alerts: list[dict]) -> None:
        self.live_alerts = live_alerts
        self._live_ctx   = get_live_context(live_alerts)

    def query(self, question: str) -> "MockResponse":
        question_lower = question.lower()
        answer, sources, score, cmd = _mock_answer(question_lower, self.live_alerts)
        return MockResponse(
            response=answer,
            sources=sources,
            anomaly_score=score,
            suggested_command=cmd,
        )


class MockResponse:
    """Mimics the interface of a LlamaIndex Response."""
    def __init__(self, response: str, sources: list[str],
                 anomaly_score: float, suggested_command: str | None) -> None:
        self.response          = response
        self.source_nodes      = [_MockNode(s) for s in sources]
        self.anomaly_score     = anomaly_score
        self.suggested_command = suggested_command

    def __str__(self) -> str:
        return self.response


class _MockNode:
    def __init__(self, path: str) -> None:
        self.metadata = {"source_path": path}


def _build_mock_engine(live_alerts: list[dict]) -> MockQueryEngine:
    logger.info("Using mock query engine (demo mode)")
    return MockQueryEngine(live_alerts)


def _mock_answer(question: str, alerts: list[dict]) -> tuple[str, list[str], float, str | None]:
    """Return canned answers for common question patterns."""
    top_alert = max(alerts, key=lambda a: a["anomaly_score"]) if alerts else None
    top_pod   = top_alert["pod"] if top_alert else "payment-svc"
    top_score = top_alert["anomaly_score"] if top_alert else 0.91

    if "payment" in question or "500" in question:
        return (
            f"The {top_pod} pod has an anomaly score of {top_score:.2f}. "
            "Retrieved runbook 'payment-svc-high-error-rate' indicates this pattern "
            "matches an OOMKill loop — the pod is being killed before completing requests. "
            "Recommended action: scale up replicas to absorb load.",
            ["docs/runbooks/payment-svc.md", "docs/incidents/2024-11-incident-03.md"],
            top_score,
            f"kubectl scale deployment/{top_pod} --replicas=5 -n production",
        )
    if "blast radius" in question or "depends" in question:
        return (
            "If auth-svc goes down, the following services will be affected: "
            "api-gateway (direct dependency), frontend (indirect via api-gateway). "
            "The blast radius covers ~80% of user-facing traffic.",
            ["docs/architecture.md"],
            0.0,
            None,
        )
    if "rollback" in question or "previous" in question:
        return (
            "Rolling back the checkout service to the previous revision. "
            "This will revert to the last known-good deployment.",
            [],
            0.0,
            "kubectl rollout undo deployment/checkout-svc -n production",
        )
    if "fail" in question or "predict" in question:
        pods = [a["pod"] for a in alerts if a["anomaly_score"] > 0.7]
        pods_str = ", ".join(pods[:3]) if pods else "no pods above threshold"
        return (
            f"Based on current GNN anomaly scores, pods most at risk: {pods_str}. "
            "Recommendation: monitor closely and consider pre-emptive scaling.",
            [],
            max((a["anomaly_score"] for a in alerts), default=0.0),
            None,
        )

    # Generic fallback
    ctx_summary = f"There are currently {len(alerts)} active alerts." if alerts else "No active alerts."
    return (
        f"KOGNOS analysis: {ctx_summary} "
        "For detailed cluster health, check GET /alerts and GET /graph. "
        "Ask a more specific question about a service for targeted runbook retrieval.",
        [],
        0.0,
        None,
    )
