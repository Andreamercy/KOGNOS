"""
rag/ingestion/live_context.py

Formats live anomaly alerts from the GNN inference engine into a structured
context string that gets injected into every LLM query prompt.

This gives the LLM real-time situational awareness — it knows *right now*
which pods are anomalous — without needing to retrieve that data via RAG.
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone


def get_live_context(alerts: list[dict]) -> str:
    """
    Format a list of alert dicts into a concise context block for LLM injection.

    Args:
        alerts: List of alert dicts from AlertScorer.score_to_alerts().

    Returns:
        Formatted string describing current cluster anomaly state.
    """
    if not alerts:
        return "No active anomalies detected. Cluster appears healthy."

    lines = ["Active anomalies (sorted by severity):"]
    for alert in sorted(alerts, key=lambda a: a["anomaly_score"], reverse=True):
        ts = _fmt_time(alert.get("detected_at"))
        action = alert.get("suggested_action") or "investigate with kubectl describe"
        lines.append(
            f"  • [{alert['severity'].upper()}] {alert['pod']}"
            f" (score={alert['anomaly_score']:.3f})"
            f" — detected {ts}"
            f" — hint: {action}"
        )

    # Add a brief summary count
    critical = sum(1 for a in alerts if a["severity"] == "critical")
    warning  = sum(1 for a in alerts if a["severity"] == "warning")
    lines.append(
        f"\nSummary: {len(alerts)} total alerts "
        f"({critical} critical, {warning} warning)"
    )

    return "\n".join(lines)


def format_graph_context(graph_snapshot: dict) -> str:
    """
    Format a cluster graph snapshot into a concise topology summary for
    injection into LLM prompts.

    Args:
        graph_snapshot: Dict with "nodes" and "edges" keys.

    Returns:
        Formatted topology string.
    """
    nodes = graph_snapshot.get("nodes", [])
    edges = graph_snapshot.get("edges", [])

    anomalous = [n for n in nodes if n.get("anomaly_score", 0) > 0.5]
    healthy   = len(nodes) - len(anomalous)

    lines = [
        f"Cluster topology: {len(nodes)} pods, {len(edges)} observed flows.",
        f"  {healthy} pods healthy, {len(anomalous)} pods anomalous.",
    ]

    if anomalous:
        lines.append("Anomalous pods:")
        for node in sorted(anomalous, key=lambda n: n["anomaly_score"], reverse=True):
            lines.append(
                f"  • {node['name']} (score={node['anomaly_score']:.3f},"
                f" severity={node.get('severity', 'unknown')})"
            )

    return "\n".join(lines)


def _fmt_time(ts: float | None) -> str:
    """Format a Unix timestamp as a human-readable relative time."""
    if ts is None:
        return "unknown"
    try:
        import time
        elapsed = time.time() - ts
        if elapsed < 60:
            return f"{elapsed:.0f}s ago"
        if elapsed < 3600:
            return f"{elapsed/60:.0f}m ago"
        return f"{elapsed/3600:.1f}h ago"
    except Exception:
        return "recently"
