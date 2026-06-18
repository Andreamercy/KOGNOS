"""
graph/inference/scorer.py

Converts raw anomaly score tensors from the GNN into structured alert dicts.

Provides:
  - AlertScorer: main scoring class with configurable thresholds
  - Alert dataclass: typed alert representation
  - Deduplication: suppresses repeated alerts within a cooldown window
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from graph.model.anomaly_head import ThresholdConfig, Severity


@dataclass
class Alert:
    """Structured alert produced by the inference engine."""
    pod:           str
    namespace:     str
    anomaly_score: float
    severity:      Severity
    auto_healable: bool
    detected_at:   float  = field(default_factory=time.time)
    node_index:    int    = 0
    suggested_action: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "pod":             self.pod,
            "namespace":       self.namespace,
            "anomaly_score":   round(self.anomaly_score, 4),
            "severity":        self.severity.value,
            "auto_healable":   self.auto_healable,
            "detected_at":     self.detected_at,
            "suggested_action": self.suggested_action,
        }


class AlertScorer:
    """
    Converts GNN output scores to structured alerts with deduplication.

    Deduplication: an alert for pod X is suppressed if the same pod
    fired a CRITICAL/WARNING alert within the last `cooldown_s` seconds.
    This prevents alert storms during rolling restarts.
    """

    def __init__(
        self,
        thresholds: ThresholdConfig | None = None,
        cooldown_s: float = 120.0,
    ) -> None:
        self.cfg       = thresholds or ThresholdConfig.from_env()
        self.cooldown  = cooldown_s
        self._last_seen: dict[str, float] = {}  # pod → last alert timestamp

    def score_to_alerts(
        self,
        scores: torch.Tensor,
        pod_names: list[str],
        namespaces: list[str] | None = None,
    ) -> list[dict]:
        """
        Convert score tensor to deduplicated alert dicts.

        Args:
            scores:     [N, 1] or [N] anomaly probabilities.
            pod_names:  Pod name strings (parallel to scores).
            namespaces: Optional namespace strings (parallel to scores).

        Returns:
            Sorted list of alert dicts (most critical first).
        """
        flat = scores.squeeze(-1).tolist()
        ns   = namespaces or ["unknown"] * len(pod_names)
        now  = time.time()

        alerts: list[dict] = []
        for idx, (score, pod, namespace) in enumerate(zip(flat, pod_names, ns)):
            severity = self.cfg.classify(score)
            if severity == Severity.NORMAL:
                continue

            # Deduplication: skip if same pod alerted recently
            last = self._last_seen.get(pod, 0.0)
            if now - last < self.cooldown and severity != Severity.CRITICAL:
                continue

            self._last_seen[pod] = now

            alert = Alert(
                pod=pod,
                namespace=namespace,
                anomaly_score=score,
                severity=severity,
                auto_healable=score >= self.cfg.auto_heal,
                detected_at=now,
                node_index=idx,
                suggested_action=_suggest_action(pod, score, severity),
            )
            alerts.append(alert.to_dict())

        return sorted(alerts, key=lambda a: a["anomaly_score"], reverse=True)

    def reset_cooldowns(self) -> None:
        """Clear all cooldown state (useful for testing)."""
        self._last_seen.clear()


# ── Action suggestion heuristics ───────────────────────────────────────────────

_REMEDIATION_HINTS: dict[str, str] = {
    "payment":   "kubectl scale deployment/{pod} --replicas=5 -n {ns}",
    "auth":      "kubectl rollout restart deployment/{pod} -n {ns}",
    "postgres":  "kubectl exec -n {ns} {pod} -- pg_ctl reload",
    "redis":     "kubectl delete pod {pod} -n {ns}",
    "kafka":     "kubectl rollout restart statefulset/kafka -n {ns}",
    "frontend":  "kubectl scale deployment/{pod} --replicas=3 -n {ns}",
    "ml":        "kubectl rollout restart deployment/{pod} -n {ns}",
}


def _suggest_action(pod: str, score: float, severity: Severity) -> str | None:
    """Return a kubectl command hint for well-known service patterns."""
    if severity == Severity.INFO:
        return None
    for keyword, template in _REMEDIATION_HINTS.items():
        if keyword in pod:
            return template.format(pod=pod, ns="production")
    if score >= 0.9:
        return f"kubectl describe pod {pod} -n production"
    return None
