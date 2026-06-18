"""
api/routes/query.py

POST /query — Natural language question answering about cluster health.

Request:  {"question": "Why is the payment service throwing 500 errors?"}
Response: {"answer": "...", "sources": [...], "anomaly_score": 0.91, ...}
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about cluster health.",
        example="Why is the payment service throwing 500 errors?",
    )
    use_agent: bool = Field(
        default=False,
        description="If true, use the ReAct agent with kubectl tools instead of the RAG engine.",
    )


class QueryResponse(BaseModel):
    answer:            str
    sources:           list[str] = []
    anomaly_score:     float     = 0.0
    suggested_command: Optional[str] = None
    latency_ms:        float     = 0.0
    used_agent:        bool      = False


@router.post("", response_model=QueryResponse, summary="Ask KOGNOS a question")
async def query_cluster(body: QueryRequest, request: Request) -> QueryResponse:
    """
    Ask KOGNOS a natural language question about your Kubernetes cluster.

    KOGNOS retrieves grounded context from runbooks and past incidents,
    injects live anomaly data from the GNN engine, and returns an
    actionable answer with source citations.

    Set `use_agent=true` for multi-step ReAct reasoning with kubectl access.
    """
    state    = request.app.state.kognos
    question = body.question
    t0       = time.time()

    logger.info("Query: %s", question)

    try:
        if body.use_agent and state.agent:
            # ── ReAct agent path ──────────────────────────────────────────
            response      = state.agent.chat(question)
            answer        = str(response.response)
            sources       = []
            anomaly_score = 0.0
            suggested_cmd = None
            used_agent    = True

        elif state.query_engine:
            # ── RAG query engine path ─────────────────────────────────────
            # Re-inject fresh live context before each query
            from rag.retrieval.query_engine import build_query_engine
            engine   = build_query_engine(live_alerts=state.live_alerts)
            response = engine.query(question)

            answer = str(response.response)

            # Extract source paths from retrieved nodes
            sources = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    path = node.metadata.get("source_path", "")
                    if path and path not in sources:
                        sources.append(path)

            # Extract anomaly score and command if available (mock engine)
            anomaly_score = getattr(response, "anomaly_score", 0.0)
            suggested_cmd = getattr(response, "suggested_command", None)
            used_agent    = False

        else:
            raise HTTPException(
                status_code=503,
                detail="Query engine not initialised. Try again in a moment.",
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Query error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.time() - t0) * 1000
    logger.info("Query answered in %.1fms", latency_ms)

    return QueryResponse(
        answer=answer,
        sources=sources,
        anomaly_score=anomaly_score,
        suggested_command=suggested_cmd,
        latency_ms=round(latency_ms, 1),
        used_agent=used_agent,
    )
