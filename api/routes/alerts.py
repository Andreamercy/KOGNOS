"""
api/routes/alerts.py

GET /alerts — Current live anomaly list from the GNN inference engine.

Returns the current snapshot of anomalous pods with their scores, severities,
and suggested remediation actions. This list is updated every inference cycle
(default 10 seconds).
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter()


class AlertItem(BaseModel):
    pod:              str
    namespace:        str       = "unknown"
    anomaly_score:    float
    severity:         str       # "critical" | "warning" | "info"
    auto_healable:    bool      = False
    detected_at:      float
    suggested_action: Optional[str] = None


class AlertsResponse(BaseModel):
    alerts:      list[AlertItem]
    total:       int
    critical:    int
    warning:     int
    as_of:       float   # Unix timestamp of this snapshot
    demo_mode:   bool    = False


@router.get(
    "",
    response_model=AlertsResponse,
    summary="Get live anomaly alerts",
)
async def get_alerts(
    request: Request,
    severity: Optional[str] = Query(
        default=None,
        description="Filter by severity: critical | warning | info",
        pattern="^(critical|warning|info)$",
    ),
    min_score: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum anomaly score to include.",
    ),
) -> AlertsResponse:
    """
    Get the current live anomaly alert list from the GNN inference engine.

    Alerts are refreshed every inference cycle (~10 seconds).
    Filter by severity or minimum anomaly score as needed.
    """
    state  = request.app.state.kognos
    raw    = state.live_alerts or []

    # Apply filters
    filtered = [
        a for a in raw
        if a.get("anomaly_score", 0.0) >= min_score
        and (severity is None or a.get("severity") == severity)
    ]

    items = [AlertItem(**_normalise_alert(a)) for a in filtered]

    return AlertsResponse(
        alerts=items,
        total=len(items),
        critical=sum(1 for i in items if i.severity == "critical"),
        warning=sum(1 for i in items if i.severity == "warning"),
        as_of=time.time(),
        demo_mode=bool(getattr(state, "demo_mode", True)),
    )


def _normalise_alert(a: dict) -> dict:
    """Fill in any missing fields with safe defaults."""
    return {
        "pod":              a.get("pod", "unknown"),
        "namespace":        a.get("namespace", "production"),
        "anomaly_score":    a.get("anomaly_score", 0.0),
        "severity":         a.get("severity", "info"),
        "auto_healable":    a.get("auto_healable", False),
        "detected_at":      a.get("detected_at", time.time()),
        "suggested_action": a.get("suggested_action"),
    }
