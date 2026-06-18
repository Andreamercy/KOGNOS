"""
rag/ingestion/incident_loader.py

Loads past incident reports from JSON or Markdown files into the Qdrant
vector store. Incident reports are tagged with:
  - service affected
  - root cause category
  - resolution time
  - remediation steps taken

These are prime retrieval candidates for "What happened last time we saw X?"
type queries.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_incidents(
    incidents_dir: str,
    vector_store: Any,  # llama_index VectorStore
    settings: Any,      # llama_index Settings
) -> int:
    """
    Load all incident reports from a directory into the vector store.

    Supports:
      - .md files: treated as free-form incident narratives
      - .json files: parsed as structured incident objects (see format below)

    JSON incident format:
    {
        "id": "INC-2024-11-03",
        "title": "Payment service OOMKill cascade",
        "service": "payment-svc",
        "severity": "P1",
        "start_time": "2024-11-03T14:22:00Z",
        "resolution_time": "2024-11-03T15:07:00Z",
        "root_cause": "Memory leak in payment processor under high load",
        "symptoms": ["OOMKill events", "500 error rate >40%", "cart abandonment spike"],
        "resolution": "Scaled replicas from 3→8, patched memory leak in v2.3.1",
        "kubectl_commands": [
            "kubectl scale deployment/payment-svc --replicas=8 -n production",
            "kubectl rollout restart deployment/payment-svc -n production"
        ]
    }

    Returns:
        Number of incident documents indexed.
    """
    from llama_index.core import Document

    p = Path(incidents_dir)
    if not p.exists():
        logger.warning("Incidents directory not found: %s", incidents_dir)
        return 0

    docs = []

    for path in p.rglob("*"):
        if path.suffix == ".json":
            doc = _load_json_incident(path)
            if doc:
                docs.append(doc)

        elif path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            docs.append(Document(
                text=text,
                metadata={
                    "source_type": "incident",
                    "source_path": str(path),
                    "format": "markdown",
                },
            ))

    logger.info("Loaded %d incident documents from %s", len(docs), incidents_dir)
    return len(docs)


def _load_json_incident(path: Path):
    """Parse a structured JSON incident report into a LlamaIndex Document."""
    try:
        from llama_index.core import Document

        data: dict = json.loads(path.read_text(encoding="utf-8"))

        # Render the JSON incident as a readable narrative for embedding
        text = _render_incident(data)

        return Document(
            text=text,
            metadata={
                "source_type":   "incident",
                "source_path":   str(path),
                "incident_id":   data.get("id", ""),
                "service":       data.get("service", ""),
                "severity":      data.get("severity", ""),
                "root_cause":    data.get("root_cause", ""),
                "format":        "json",
            },
        )
    except Exception as e:
        logger.error("Failed to parse incident %s: %s", path, e)
        return None


def _render_incident(data: dict) -> str:
    """Convert a structured incident dict to a readable narrative string."""
    parts = [
        f"# Incident Report: {data.get('id', 'Unknown')}",
        f"**Title**: {data.get('title', '')}",
        f"**Service**: {data.get('service', '')}",
        f"**Severity**: {data.get('severity', '')}",
        f"**Duration**: {data.get('start_time', '')} → {data.get('resolution_time', '')}",
        "",
        f"## Root Cause\n{data.get('root_cause', 'Unknown')}",
        "",
        "## Symptoms",
    ]
    for symptom in data.get("symptoms", []):
        parts.append(f"- {symptom}")

    parts.extend([
        "",
        f"## Resolution\n{data.get('resolution', '')}",
        "",
        "## kubectl Commands Used",
    ])
    for cmd in data.get("kubectl_commands", []):
        parts.append(f"```\n{cmd}\n```")

    return "\n".join(parts)
