"""
api/main.py

KOGNOS FastAPI application entry point.

Startup sequence:
  1. Load environment / settings
  2. Start GNN inference engine (background asyncio task)
  3. Build RAG query engine (connects to Qdrant)
  4. Mount all routes and WebSocket handler

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.routes import alerts, graph, query, heal
from api.websocket.stream import router as ws_router

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

DEMO_MODE = os.getenv("KOGNOS_DEMO_MODE", "false").lower() == "true"


# ── App state shared across request handlers ──────────────────────────────────

class AppState:
    """Mutable application state passed to all route handlers via request.app.state."""
    live_alerts: list[dict] = []
    query_engine = None
    agent        = None
    inference_task: asyncio.Task | None = None


# ── Lifespan context manager ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan: starts background inference and initialises RAG.
    Called once at startup; cleanup runs on shutdown.
    """
    state = AppState()
    app.state.kognos = state

    logger.info("🚀 KOGNOS starting (demo_mode=%s)", DEMO_MODE)

    # ── Start GNN inference engine in the background ──────────────────────
    async def run_inference() -> None:
        from graph.inference.engine import InferenceEngine
        engine = InferenceEngine()
        async for alerts in engine.run(interval_s=10.0):
            state.live_alerts = alerts
            if alerts:
                logger.info("Inference update: %d alerts", len(alerts))

    state.inference_task = asyncio.create_task(run_inference())
    logger.info("✅ GNN inference engine started")

    # ── Build RAG query engine ────────────────────────────────────────────
    try:
        from rag.retrieval.query_engine import build_query_engine
        state.query_engine = build_query_engine(live_alerts=state.live_alerts)
        logger.info("✅ RAG query engine ready")
    except Exception as exc:
        logger.warning("RAG query engine failed to initialise: %s", exc)
        state.query_engine = None

    # ── Build ReAct agent ─────────────────────────────────────────────────
    try:
        from rag.agent.kognos_agent import build_agent
        state.agent = build_agent()
        logger.info("✅ ReAct agent ready")
    except Exception as exc:
        logger.warning("Agent failed to initialise: %s", exc)
        state.agent = None

    logger.info("✅ KOGNOS API ready on http://0.0.0.0:%s", os.getenv("API_PORT", "8000"))
    logger.info("   Docs → http://localhost:%s/docs", os.getenv("API_PORT", "8000"))

    yield   # <-- application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down KOGNOS...")
    if state.inference_task and not state.inference_task.done():
        state.inference_task.cancel()
        try:
            await state.inference_task
        except asyncio.CancelledError:
            pass
    logger.info("KOGNOS shutdown complete.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="KOGNOS",
    description=(
        "Multi-Agent Kubernetes Observability Platform.\n\n"
        "Real-time anomaly detection and conversational intelligence "
        "powered by eBPF telemetry, Graph Neural Networks, and RAG-based AI agents."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(query.router,  prefix="/query",  tags=["Query"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(graph.router,  prefix="/graph",  tags=["Graph"])
app.include_router(heal.router,   prefix="/heal",   tags=["Self-Healing"])
app.include_router(ws_router,     prefix="/stream", tags=["WebSocket"])

# ── Static dashboard (mounted BEFORE catch-all) ──────────────────────────────
import os as _os
from pathlib import Path as _Path
_dashboard_dir = str((_Path(__file__).parent.parent / "dashboard").resolve())


@app.get("/", include_in_schema=False)
async def root():
    """Serve the KOGNOS live dashboard."""
    from fastapi.responses import HTMLResponse
    html_path = _Path(_dashboard_dir) / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return RedirectResponse(url="/docs")


@app.get("/api", tags=["Health"])
async def api_info():
    return {
        "service":   "KOGNOS",
        "version":   "0.1.0",
        "demo_mode": DEMO_MODE,
        "status":    "running",
        "endpoints": {
            "dashboard": "GET /dashboard",
            "query":     "POST /query",
            "alerts":    "GET /alerts",
            "graph":     "GET /graph",
            "heal":      "POST /heal",
            "stream":    "WS /stream",
            "docs":      "GET /docs",
        },
    }


@app.get("/health", tags=["Health"])
async def health():
    """Kubernetes liveness / readiness probe."""
    return {"status": "ok"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )
