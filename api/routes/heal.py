"""
api/routes/heal.py

POST /heal — Trigger autonomous remediation for an anomalous pod.

Safety gates:
  1. Anomaly score must be >= AUTO_HEAL_THRESHOLD (default 0.9)
  2. DRY_RUN=true by default — always simulates unless explicitly disabled
  3. Returns the kubectl command that WOULD be run for review
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

DRY_RUN            = os.getenv("DRY_RUN", "true").lower() == "true"
AUTO_HEAL_THRESHOLD = float(os.getenv("AUTO_HEAL_THRESHOLD", "0.9"))


class HealRequest(BaseModel):
    pod:       str  = Field(..., description="Pod name to remediate")
    namespace: str  = Field(default="production", description="Kubernetes namespace")
    action:    str  = Field(
        default="auto",
        description="Remediation action: 'auto' | 'scale' | 'rollback' | 'restart'",
        pattern="^(auto|scale|rollback|restart)$",
    )
    replicas:  Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Target replica count (only used with action='scale')",
    )
    dry_run:   bool = Field(
        default=DRY_RUN,
        description="If true, simulate the action without executing.",
    )


class HealResponse(BaseModel):
    pod:           str
    namespace:     str
    action_taken:  str
    command:       str
    dry_run:       bool
    anomaly_score: float
    success:       bool
    message:       str
    executed_at:   float


@router.post("", response_model=HealResponse, summary="Trigger autonomous remediation")
async def heal(body: HealRequest, request: Request) -> HealResponse:
    """
    Trigger autonomous pod remediation.

    KOGNOS will:
    1. Check the current anomaly score for the pod.
    2. Select the appropriate remediation action (scale / rollback / restart).
    3. Execute the action (or simulate in dry-run mode).

    **Safety**: Requires anomaly score ≥ AUTO_HEAL_THRESHOLD.
    **Default**: dry_run=true — always verify before executing in production.
    """
    state = request.app.state.kognos
    pod   = body.pod
    ns    = body.namespace

    # ── Check current anomaly score for this pod ─────────────────────────
    alerts    = state.live_alerts or []
    pod_alert = next((a for a in alerts if a.get("pod") == pod), None)
    score     = pod_alert["anomaly_score"] if pod_alert else 0.0

    if score < AUTO_HEAL_THRESHOLD and not body.dry_run:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Anomaly score {score:.3f} is below the auto-heal threshold "
                f"{AUTO_HEAL_THRESHOLD}. KOGNOS will not auto-heal healthy pods. "
                "Use dry_run=true to inspect what would happen."
            ),
        )

    # ── Select action ─────────────────────────────────────────────────────
    action, command = _select_action(pod, ns, body)

    # ── Execute or simulate ───────────────────────────────────────────────
    if body.dry_run:
        message = f"DRY RUN: Would execute '{command}'. Set dry_run=false to apply."
        success = True
    else:
        from rag.agent.tools import _run_kubectl
        output  = _run_kubectl(command.replace("kubectl ", "").split())
        success = "Error" not in output
        message = output[:500]

    logger.info(
        "Heal: pod=%s action=%s dry_run=%s score=%.3f",
        pod, action, body.dry_run, score,
    )

    return HealResponse(
        pod=pod,
        namespace=ns,
        action_taken=action,
        command=command,
        dry_run=body.dry_run,
        anomaly_score=score,
        success=success,
        message=message,
        executed_at=time.time(),
    )


def _select_action(pod: str, ns: str, body: HealRequest) -> tuple[str, str]:
    """Choose the remediation action and generate the kubectl command."""
    action = body.action

    if action == "auto":
        # Heuristic: infer action from pod name
        if any(k in pod for k in ["postgres", "redis", "kafka"]):
            action = "restart"
        elif any(k in pod for k in ["payment", "cart", "frontend"]):
            action = "scale"
        else:
            action = "rollback"

    match action:
        case "scale":
            replicas = body.replicas or 5
            cmd = f"kubectl scale deployment/{pod} --replicas={replicas} -n {ns}"
        case "rollback":
            cmd = f"kubectl rollout undo deployment/{pod} -n {ns}"
        case "restart":
            cmd = f"kubectl rollout restart deployment/{pod} -n {ns}"
        case _:
            cmd = f"kubectl rollout restart deployment/{pod} -n {ns}"

    return action, cmd
