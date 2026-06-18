"""
graph/data/synthetic_gen.py

Synthetic Kubernetes cluster failure scenario generator.

Generates realistic telemetry windows for:
  - Normal operation
  - OOMKill cascade (memory pressure propagates via retries)
  - Network partition (one service becomes unreachable)
  - CPU throttling storm (latency spike across downstream services)
  - Rolling restart (sequential pod restarts)

Used for:
  1. Training the GraphSAGE model on labelled failure scenarios.
  2. Demo mode — running the full KOGNOS pipeline without a live cluster.
  3. Unit / integration tests.
"""

from __future__ import annotations

import random
import time
from enum import Enum
from typing import NamedTuple

from graph.builder.cluster_graph import PodNode, FlowEdge, TelemetryWindow

# ── Cluster topology fixture ───────────────────────────────────────────────────
# Represents a realistic microservices cluster (e-commerce domain)

SERVICES = [
    # (name, namespace, depends_on)
    ("frontend",        "production", ["api-gateway"]),
    ("api-gateway",     "production", ["auth-svc", "product-svc", "cart-svc"]),
    ("auth-svc",        "production", ["postgres-auth"]),
    ("product-svc",     "production", ["postgres-product", "redis-cache"]),
    ("cart-svc",        "production", ["redis-cart", "payment-svc"]),
    ("payment-svc",     "production", ["postgres-payment", "fraud-svc"]),
    ("fraud-svc",       "production", ["ml-inference"]),
    ("ml-inference",    "production", []),
    ("notification-svc","production", ["kafka"]),
    ("postgres-auth",   "data",       []),
    ("postgres-product","data",       []),
    ("postgres-payment","data",       []),
    ("redis-cache",     "data",       []),
    ("redis-cart",      "data",       []),
    ("kafka",           "data",       []),
]


class FailureScenario(str, Enum):
    NORMAL              = "normal"
    OOMKILL             = "oomkill"
    NETWORK_PARTITION   = "network_partition"
    CPU_THROTTLE        = "cpu_throttle"
    ROLLING_RESTART     = "rolling_restart"
    CASCADE             = "cascade"


class LabelledWindow(NamedTuple):
    """A telemetry window with ground-truth anomaly labels for training."""
    window:   TelemetryWindow
    labels:   list[float]        # Per-pod anomaly probability [0, 1]
    scenario: FailureScenario
    affected: list[str]          # Names of pods that are truly anomalous


def generate_window(
    scenario: FailureScenario | None = None,
    seed: int | None = None,
) -> TelemetryWindow:
    """
    Generate a single telemetry window (for demo / inference).
    Randomly picks a scenario if none specified.
    """
    lw = generate_labelled_window(scenario=scenario, seed=seed)
    return lw.window


def generate_labelled_window(
    scenario: FailureScenario | None = None,
    seed: int | None = None,
) -> LabelledWindow:
    """Generate a labelled telemetry window for training."""
    rng = random.Random(seed)

    if scenario is None:
        # Bias toward normal (70%) to reflect real cluster distributions
        weights = [0.70, 0.06, 0.06, 0.06, 0.06, 0.06]
        scenario = rng.choices(list(FailureScenario), weights=weights)[0]

    pods   = _build_pods(scenario, rng)
    flows  = _build_flows(pods, scenario, rng)
    labels = _build_labels(pods, scenario)

    now = int(time.time_ns())
    return LabelledWindow(
        window=TelemetryWindow(
            pods=pods,
            flows=flows,
            window_start_ns=now - 30_000_000_000,
            window_end_ns=now,
        ),
        labels=labels,
        scenario=scenario,
        affected=[p.name for p, lbl in zip(pods, labels) if lbl > 0.5],
    )


def generate_dataset(
    n_windows: int = 1000,
    seed: int = 42,
) -> list[LabelledWindow]:
    """Generate a full training dataset of labelled windows."""
    rng = random.Random(seed)
    return [
        generate_labelled_window(seed=rng.randint(0, 2**31))
        for _ in range(n_windows)
    ]


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_pods(scenario: FailureScenario, rng: random.Random) -> list[PodNode]:
    pods = []
    affected = _affected_pods(scenario, rng)

    for name, ns, _ in SERVICES:
        base_cpu  = rng.uniform(0.10, 0.40)
        base_mem  = rng.uniform(0.20, 0.60)
        restarts  = rng.randint(0, 2)
        err_rate  = rng.uniform(0.0, 0.02)
        latency   = rng.uniform(5.0, 80.0)
        is_ready  = True

        if name in affected:
            match scenario:
                case FailureScenario.OOMKILL:
                    base_mem = rng.uniform(0.92, 1.0)
                    restarts = rng.randint(3, 15)
                    err_rate = rng.uniform(0.3, 0.8)
                    latency  = rng.uniform(500.0, 3000.0)
                    is_ready = rng.random() < 0.4

                case FailureScenario.NETWORK_PARTITION:
                    err_rate = rng.uniform(0.5, 1.0)
                    latency  = rng.uniform(5000.0, 10000.0)

                case FailureScenario.CPU_THROTTLE:
                    base_cpu = rng.uniform(0.85, 1.0)
                    latency  = rng.uniform(800.0, 5000.0)
                    err_rate = rng.uniform(0.05, 0.3)

                case FailureScenario.ROLLING_RESTART:
                    restarts = rng.randint(5, 30)
                    is_ready = rng.random() < 0.5

                case FailureScenario.CASCADE:
                    base_cpu = rng.uniform(0.7, 1.0)
                    base_mem = rng.uniform(0.7, 1.0)
                    err_rate = rng.uniform(0.2, 0.9)
                    latency  = rng.uniform(300.0, 8000.0)
                    restarts = rng.randint(2, 10)
                    is_ready = rng.random() < 0.3

        pods.append(PodNode(
            name=name,
            namespace=ns,
            cpu_usage=base_cpu,
            mem_usage=base_mem,
            restart_count=restarts,
            error_rate=err_rate,
            latency_p99=latency,
            is_ready=is_ready,
        ))

    return pods


def _build_flows(
    pods: list[PodNode],
    scenario: FailureScenario,
    rng: random.Random,
) -> list[FlowEdge]:
    """Build network flows from the topology adjacency list."""
    pod_map = {p.name: p for p in pods}
    flows   = []

    for src_name, _, deps in SERVICES:
        src = pod_map.get(src_name)
        if src is None:
            continue
        for dst_name in deps:
            dst = pod_map.get(dst_name)
            if dst is None:
                continue

            # Base traffic
            bps     = rng.uniform(1e5, 5e7)
            lat_ms  = (src.latency_p99 + dst.latency_p99) / 2.0
            err_r   = max(src.error_rate, dst.error_rate)

            # Network partition: zero bytes for affected flows
            if scenario == FailureScenario.NETWORK_PARTITION:
                if not src.is_ready or not dst.is_ready:
                    bps = 0.0
                    err_r = 1.0

            flows.append(FlowEdge(
                src_pod=src_name,
                dst_pod=dst_name,
                bytes_per_sec=bps,
                latency_ms=lat_ms,
                error_rate=err_r,
            ))

    return flows


def _build_labels(pods: list[PodNode], scenario: FailureScenario) -> list[float]:
    """Assign ground-truth anomaly labels based on pod state."""
    labels = []
    for pod in pods:
        score = 0.0
        # Heuristic label from observed signals
        score += pod.cpu_usage * 0.2
        score += pod.mem_usage * 0.2
        score += min(pod.restart_count / 10.0, 1.0) * 0.25
        score += pod.error_rate * 0.25
        score += min(pod.latency_p99 / 5000.0, 1.0) * 0.10
        if not pod.is_ready:
            score = max(score, 0.85)
        labels.append(min(score, 1.0))
    return labels


def _affected_pods(scenario: FailureScenario, rng: random.Random) -> set[str]:
    """Pick which pods are directly affected by the failure scenario."""
    all_pods = [s[0] for s in SERVICES]

    match scenario:
        case FailureScenario.NORMAL:
            return set()
        case FailureScenario.OOMKILL:
            primary = rng.choice(["payment-svc", "ml-inference", "postgres-payment"])
            return {primary}
        case FailureScenario.NETWORK_PARTITION:
            # Partition isolates the data tier
            return {"postgres-payment", "payment-svc", "cart-svc"}
        case FailureScenario.CPU_THROTTLE:
            return {"ml-inference", "fraud-svc", "payment-svc"}
        case FailureScenario.ROLLING_RESTART:
            return set(rng.sample(all_pods, k=rng.randint(2, 5)))
        case FailureScenario.CASCADE:
            # Start with one root cause, cascade through dependencies
            root = rng.choice(["auth-svc", "postgres-auth", "redis-cache"])
            cascaded = {root, "api-gateway", "frontend"}
            return cascaded
        case _:
            return set()
