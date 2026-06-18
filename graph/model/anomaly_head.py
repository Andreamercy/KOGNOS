"""
graph/model/anomaly_head.py

Standalone anomaly classification head and threshold configuration.

Separating the head from the backbone (graphsage.py) allows:
  - Swapping backbone without retraining the anomaly head.
  - Fine-tuning the head on real incident data without retraining the GNN.
  - A/B testing different scoring strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class Severity(str, Enum):
    """Alert severity levels aligned with PagerDuty / OpsGenie conventions."""
    CRITICAL = "critical"
    WARNING  = "warning"
    INFO     = "info"
    NORMAL   = "normal"


@dataclass(frozen=True)
class ThresholdConfig:
    """Configurable thresholds for converting anomaly scores → severities."""
    critical:  float = 0.90   # score ≥ 0.90 → CRITICAL (page on-call)
    warning:   float = 0.75   # score ≥ 0.75 → WARNING  (Slack alert)
    info:      float = 0.50   # score ≥ 0.50 → INFO     (dashboard only)
    auto_heal: float = 0.90   # score ≥ this → trigger agentic remediation

    @classmethod
    def from_env(cls) -> "ThresholdConfig":
        """Read thresholds from environment variables with safe defaults."""
        import os
        return cls(
            critical=float(os.getenv("THRESHOLD_CRITICAL", "0.90")),
            warning=float(os.getenv("THRESHOLD_WARNING",  "0.75")),
            info=float(os.getenv("THRESHOLD_INFO",        "0.50")),
            auto_heal=float(os.getenv("AUTO_HEAL_THRESHOLD", "0.90")),
        )

    def classify(self, score: float) -> Severity:
        """Map a raw anomaly score to a Severity enum value."""
        if score >= self.critical:
            return Severity.CRITICAL
        if score >= self.warning:
            return Severity.WARNING
        if score >= self.info:
            return Severity.INFO
        return Severity.NORMAL


class AnomalyHead(nn.Module):
    """
    Standalone anomaly scoring head that operates on GNN node embeddings.

    Can be attached to any backbone that produces per-node embeddings.

    Args:
        in_channels: Dimensionality of input embeddings (e.g. 64 from SAGE).
        hidden:      Hidden layer width (default 32).
    """

    def __init__(self, in_channels: int = 64, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [N, in_channels] node embeddings
        Returns:
            scores: [N, 1] anomaly probabilities in [0, 1]
        """
        return self.net(embeddings)

    @staticmethod
    def score_to_alerts(
        scores: torch.Tensor,
        pod_names: list[str],
        thresholds: ThresholdConfig | None = None,
    ) -> list[dict]:
        """
        Convert raw score tensor to a list of alert dicts.

        Args:
            scores:    [N, 1] or [N] tensor of anomaly probabilities.
            pod_names: Parallel list of pod name strings.
            thresholds: Threshold config (uses defaults if None).

        Returns:
            List of alert dicts for pods above the info threshold.
        """
        cfg = thresholds or ThresholdConfig()
        flat = scores.squeeze(-1).tolist()

        alerts = []
        for idx, (score, pod) in enumerate(zip(flat, pod_names)):
            severity = cfg.classify(score)
            if severity == Severity.NORMAL:
                continue
            alerts.append({
                "pod":           pod,
                "anomaly_score": round(score, 4),
                "severity":      severity.value,
                "auto_healable": score >= cfg.auto_heal,
                "node_index":    idx,
            })

        # Sort: highest score first
        return sorted(alerts, key=lambda a: a["anomaly_score"], reverse=True)
