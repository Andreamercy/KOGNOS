"""
api/websocket/stream.py

WS /stream — Real-time alert streaming via WebSocket.

Clients connect and receive JSON alert updates as they arrive from the
GNN inference engine. Each message is a JSON array of alert dicts.

Example client (JavaScript):
    const ws = new WebSocket("ws://localhost:8000/stream");
    ws.onmessage = (e) => console.log(JSON.parse(e.data));
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Connection manager ────────────────────────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections and broadcasts to all."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info("WS client connected. Total: %d", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        self.active = [w for w in self.active if w is not ws]
        logger.info("WS client disconnected. Total: %d", len(self.active))

    async def broadcast(self, data: str) -> None:
        """Send data to all connected clients, removing dead connections."""
        dead = []
        for ws in self.active:
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("")
async def stream_alerts(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time alert streaming.

    Connect at: ws://localhost:8000/stream

    Each message is a JSON object:
    {
        "type": "alerts",
        "data": [...alert dicts...],
        "as_of": <unix timestamp>,
        "total": <count>
    }

    Send {"type": "ping"} to check connection liveness.
    """
    await manager.connect(websocket)

    try:
        # Send an initial snapshot immediately on connect
        app   = websocket.app
        state = getattr(app.state, "kognos", None)

        if state:
            initial = _make_message(state.live_alerts or [])
            await websocket.send_text(json.dumps(initial))

        # Then stream updates every 5 seconds (or whenever alerts change)
        last_alerts = None
        while True:
            # Check for client messages (e.g., ping)
            try:
                msg = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=5.0,
                )
                data = json.loads(msg)
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except asyncio.TimeoutError:
                pass  # No client message — proceed to push update

            # Push alert update if changed
            if state:
                current = state.live_alerts or []
                if current != last_alerts:
                    message = _make_message(current)
                    await websocket.send_text(json.dumps(message))
                    last_alerts = list(current)

    except WebSocketDisconnect:
        logger.info("WS client disconnected cleanly")
    except Exception as exc:
        logger.error("WS error: %s", exc)
    finally:
        manager.disconnect(websocket)


def _make_message(alerts: list[dict]) -> dict:
    return {
        "type":  "alerts",
        "data":  alerts,
        "as_of": time.time(),
        "total": len(alerts),
    }
