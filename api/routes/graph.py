"""
api/routes/graph.py

GET /graph — Cluster graph snapshot as JSON.

Returns nodes (pods) and edges (network flows) with anomaly scores
for visualization and further analysis (e.g., blast radius calculation).
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class GraphNode(BaseModel):
    id:            str
    name:          str
    namespace:     str
    anomaly_score: float
    severity:      str
    cpu_usage:     float
    mem_usage:     float
    restart_count: int
    is_ready:      bool
    labels:        dict = {}


class GraphEdge(BaseModel):
    source:       str   # pod name
    target:       str   # pod name
    bytes_per_sec: float = 0.0
    latency_ms:   float = 0.0
    error_rate:   float = 0.0


class GraphSnapshot(BaseModel):
    nodes:       list[GraphNode]
    edges:       list[GraphEdge]
    num_nodes:   int
    num_edges:   int
    snapshot_at: float


@router.get(
    "",
    response_model=GraphSnapshot,
    summary="Get cluster graph snapshot",
)
async def get_graph(request: Request) -> GraphSnapshot:
    """
    Returns the current cluster topology as a graph.

    Nodes represent pods with their feature values and anomaly scores.
    Edges represent observed network flows between pods.

    Use this data to:
    - Visualize cluster topology
    - Calculate blast radius for a failing pod
    - Identify dependency chains
    """
    from graph.data.synthetic_gen import generate_labelled_window
    from graph.model.anomaly_head import ThresholdConfig

    cfg = ThresholdConfig()
    lw  = generate_labelled_window()
    pods  = lw.window.pods
    flows = lw.window.flows
    labels = lw.labels  # pre-computed ground truth labels

    # Try to use the real GNN model if torch_geometric is available
    scores: list[float] = []
    try:
        import torch
        from graph.builder.cluster_graph import build_cluster_graph, HAS_PYG
        from graph.model.graphsage import KOGNOSGraphSAGE

        if HAS_PYG:
            graph = build_cluster_graph(lw.window)
            model = KOGNOSGraphSAGE(in_channels=6)
            model.eval()
            with torch.no_grad():
                scores = model(graph.x, graph.edge_index).squeeze(-1).tolist()
        else:
            scores = labels
    except Exception:
        scores = labels

    nodes = []
    for pod, score in zip(pods, scores):
        nodes.append(GraphNode(
            id=pod.name,
            name=pod.name,
            namespace=pod.namespace,
            anomaly_score=round(score, 4),
            severity=cfg.classify(score).value,
            cpu_usage=round(pod.cpu_usage, 3),
            mem_usage=round(pod.mem_usage, 3),
            restart_count=pod.restart_count,
            is_ready=pod.is_ready,
        ))

    edges = [
        GraphEdge(
            source=f.src_pod,
            target=f.dst_pod,
            bytes_per_sec=round(f.bytes_per_sec, 2),
            latency_ms=round(f.latency_ms, 2),
            error_rate=round(f.error_rate, 4),
        )
        for f in flows
    ]

    return GraphSnapshot(
        nodes=nodes,
        edges=edges,
        num_nodes=len(nodes),
        num_edges=len(edges),
        snapshot_at=time.time(),
    )
