"""
graph/builder/feature_encoder.py

Node and edge feature encoders for the KOGNOS cluster graph.

Normalises raw telemetry metrics to [0, 1] ranges so the GNN model
trains stably. Uses empirical maxima from production Kubernetes clusters
(cpu=100%, mem=100%, latency_p99=10s, bytes=1GB/s).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph.builder.cluster_graph import PodNode, FlowEdge

# Feature indices for documentation clarity
_NODE_FEATURES = ["cpu_usage", "mem_usage", "restart_count_norm",
                  "error_rate", "latency_p99_norm", "is_ready"]
_EDGE_FEATURES = ["bytes_per_sec_norm", "latency_ms_norm", "error_rate"]


class PodFeatureEncoder:
    """
    Encodes a PodNode into a 6-dimensional feature vector.

    Feature vector layout:
        [0] cpu_usage        — fraction [0, 1]
        [1] mem_usage        — fraction [0, 1]
        [2] restart_count    — log-normalised to [0, 1] (log base 100)
        [3] error_rate       — fraction of requests with 5xx [0, 1]
        [4] latency_p99      — normalised by 10 000 ms cap → [0, 1]
        [5] is_ready         — binary 0/1
    """

    MAX_LATENCY_MS: float = 10_000.0   # 10 seconds
    MAX_RESTARTS: float   = 100.0      # log-scale normalisation base

    def encode(self, pod: "PodNode") -> list[float]:
        cpu   = _clamp(pod.cpu_usage, 0.0, 1.0)
        mem   = _clamp(pod.mem_usage, 0.0, 1.0)
        rst   = _log_norm(pod.restart_count, self.MAX_RESTARTS)
        err   = _clamp(pod.error_rate, 0.0, 1.0)
        lat   = _clamp(pod.latency_p99 / self.MAX_LATENCY_MS, 0.0, 1.0)
        ready = 1.0 if pod.is_ready else 0.0
        return [cpu, mem, rst, err, lat, ready]

    @property
    def num_features(self) -> int:
        return 6

    @property
    def feature_names(self) -> list[str]:
        return _NODE_FEATURES


class FlowFeatureEncoder:
    """
    Encodes a FlowEdge into a 3-dimensional feature vector.

    Feature vector layout:
        [0] bytes_per_sec — normalised by 1 GB/s cap → [0, 1]
        [1] latency_ms    — normalised by 5 000 ms cap → [0, 1]
        [2] error_rate    — fraction of requests with errors [0, 1]
    """

    MAX_BYTES_PER_SEC: float = 1_000_000_000.0  # 1 GB/s
    MAX_LATENCY_MS: float    = 5_000.0           # 5 seconds

    def encode(self, flow: "FlowEdge") -> list[float]:
        bps = _clamp(flow.bytes_per_sec / self.MAX_BYTES_PER_SEC, 0.0, 1.0)
        lat = _clamp(flow.latency_ms / self.MAX_LATENCY_MS, 0.0, 1.0)
        err = _clamp(flow.error_rate, 0.0, 1.0)
        return [bps, lat, err]

    @property
    def num_features(self) -> int:
        return 3

    @property
    def feature_names(self) -> list[str]:
        return _EDGE_FEATURES


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _log_norm(v: float, base: float) -> float:
    """Logarithmic normalisation: log(1 + v) / log(1 + base) → [0, 1]."""
    if v <= 0:
        return 0.0
    return math.log1p(v) / math.log1p(base)
