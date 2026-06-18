"""
graph/builder/cluster_graph.py

Builds a PyTorch Geometric graph from live Kubernetes cluster state + eBPF
telemetry. Each node is a pod; edges represent observed network flows.

In DEMO_MODE (KOGNOS_DEMO_MODE=true) the k8s API is bypassed and a synthetic
cluster is generated instead, allowing the full pipeline to run without a live
cluster.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = None  # type: ignore[assignment,misc]

from graph.builder.feature_encoder import PodFeatureEncoder, FlowFeatureEncoder

logger = logging.getLogger(__name__)

DEMO_MODE = os.getenv("KOGNOS_DEMO_MODE", "false").lower() == "true"


@dataclass
class PodNode:
    """Lightweight pod representation for graph construction."""
    name: str
    namespace: str
    cpu_usage: float       # 0.0–1.0 (fraction of requested)
    mem_usage: float       # 0.0–1.0 (fraction of limit)
    restart_count: int
    error_rate: float      # requests/s returning 5xx
    latency_p99: float     # milliseconds
    is_ready: bool
    labels: dict = field(default_factory=dict)


@dataclass
class FlowEdge:
    """Directed network flow observed by eBPF between two pods."""
    src_pod: str
    dst_pod: str
    bytes_per_sec: float
    latency_ms: float
    error_rate: float      # fraction of requests with errors


@dataclass
class TelemetryWindow:
    """A snapshot of cluster telemetry for a given time window."""
    pods: list[PodNode]
    flows: list[FlowEdge]
    window_start_ns: int
    window_end_ns: int


def build_cluster_graph(telemetry: TelemetryWindow):  # -> Data
    """
    Build a PyTorch Geometric Data object from a telemetry window.

    Node features (per pod):
        [cpu_usage, mem_usage, restart_count_norm, error_rate, latency_p99_norm, is_ready]

    Edge features (per flow):
        [bytes_per_sec_norm, latency_ms_norm, error_rate]

    Returns:
        torch_geometric.data.Data with fields:
            .x          — node feature matrix  [N, 6]
            .edge_index — COO edge list         [2, E]
            .edge_attr  — edge feature matrix   [E, 3]
            .pod_names  — list[str] for alert labeling
    """
    pods   = telemetry.pods
    flows  = telemetry.flows
    N      = len(pods)

    if not HAS_PYG:
        raise ImportError(
            "torch_geometric is required for graph building.\n"
            "Install: pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cpu.html"
        )

    if N == 0:
        raise ValueError("Telemetry window contains no pods")

    pod_to_idx: dict[str, int] = {p.name: i for i, p in enumerate(pods)}

    # ── Node features ──────────────────────────────────────────────────────
    node_encoder = PodFeatureEncoder()
    node_features = [node_encoder.encode(pod) for pod in pods]
    x = torch.tensor(node_features, dtype=torch.float)  # [N, 6]

    # ── Edge index + features ──────────────────────────────────────────────
    edge_encoder  = FlowFeatureEncoder()
    edge_index_list: list[list[int]] = []
    edge_attr_list:  list[list[float]] = []

    for flow in flows:
        src = pod_to_idx.get(flow.src_pod)
        dst = pod_to_idx.get(flow.dst_pod)
        if src is None or dst is None:
            logger.debug("Unknown pod in flow: %s → %s", flow.src_pod, flow.dst_pod)
            continue
        edge_index_list.append([src, dst])
        edge_attr_list.append(edge_encoder.encode(flow))

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr_list,  dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 3), dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.pod_names = [p.name for p in pods]  # type: ignore[attr-defined]
    graph.num_nodes = N

    logger.info("Built cluster graph: %d nodes, %d edges", N, edge_index.size(1))
    return graph


def load_telemetry_window(source: str = "k8s") -> TelemetryWindow:
    """
    Load a telemetry window from Kubernetes API + Kafka/Redis, or generate
    synthetic data in demo mode.

    Args:
        source: "k8s" for live data, "synthetic" or auto-selected in demo mode.
    """
    if DEMO_MODE or source == "synthetic":
        from graph.data.synthetic_gen import generate_window
        return generate_window()

    return _load_from_k8s()


def _load_from_k8s() -> TelemetryWindow:
    """Load live pod state from the Kubernetes API server."""
    try:
        from kubernetes import client, config as k8s_config
    except ImportError:
        raise RuntimeError("kubernetes Python client not installed. "
                           "Run: pip install kubernetes")

    try:
        k8s_config.load_incluster_config()
    except Exception:
        k8s_config.load_kube_config()

    v1    = client.CoreV1Api()
    pods  = v1.list_pod_for_all_namespaces(watch=False).items

    pod_nodes = []
    for pod in pods:
        pod_nodes.append(PodNode(
            name=pod.metadata.name,
            namespace=pod.metadata.namespace,
            cpu_usage=0.0,       # populated from Prometheus / metrics-server
            mem_usage=0.0,
            restart_count=sum(
                cs.restart_count
                for cs in (pod.status.container_statuses or [])
            ),
            error_rate=0.0,
            latency_p99=0.0,
            is_ready=all(
                c.status for c in (pod.status.conditions or [])
                if c.type == "Ready"
            ),
            labels=pod.metadata.labels or {},
        ))

    import time
    now = int(time.time_ns())
    return TelemetryWindow(
        pods=pod_nodes,
        flows=[],           # populated by Kafka consumer in production
        window_start_ns=now - 30_000_000_000,
        window_end_ns=now,
    )
