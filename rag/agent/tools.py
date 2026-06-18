"""
rag/agent/tools.py

kubectl tool definitions for the KOGNOS ReAct agent.

Each tool wraps a kubectl subprocess call with:
  - Dry-run mode (simulates the action, returns what would happen)
  - Timeout protection (no hanging on unreachable clusters)
  - Structured output (returns the stdout as a string for LLM consumption)
  - Safe defaults (namespace defaults to "production" for k8s clusters)

In DEMO_MODE, all tools return realistic canned responses.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Callable

logger = logging.getLogger(__name__)

DEMO_MODE = os.getenv("KOGNOS_DEMO_MODE", "false").lower() == "true"
DRY_RUN   = os.getenv("DRY_RUN", "true").lower() == "true"
TIMEOUT_S = int(os.getenv("KUBECTL_TIMEOUT_S", "15"))


# ── kubectl tool functions ─────────────────────────────────────────────────────

def kubectl_describe(pod_name: str, namespace: str = "production") -> str:
    """
    Describe a Kubernetes pod to get its current status, conditions, and events.

    Args:
        pod_name:  Name of the pod to describe.
        namespace: Kubernetes namespace (default: production).

    Returns:
        kubectl describe output as a string.
    """
    if DEMO_MODE:
        return _mock_describe(pod_name, namespace)

    return _run_kubectl(["describe", "pod", pod_name, "-n", namespace])


def kubectl_logs(
    pod_name: str,
    namespace: str = "production",
    tail: int = 50,
    container: str | None = None,
) -> str:
    """
    Fetch the last N lines of logs from a pod container.

    Args:
        pod_name:   Name of the pod.
        namespace:  Kubernetes namespace.
        tail:       Number of log lines to retrieve (default: 50).
        container:  Specific container name (optional, auto-selected if omitted).

    Returns:
        Log lines as a string.
    """
    if DEMO_MODE:
        return _mock_logs(pod_name, namespace, tail)

    cmd = ["logs", pod_name, "-n", namespace, f"--tail={tail}"]
    if container:
        cmd += ["-c", container]
    return _run_kubectl(cmd)


def kubectl_rollback(deployment: str, namespace: str = "production") -> str:
    """
    Roll back a deployment to its previous stable revision.

    Args:
        deployment: Deployment name (without "deployment/" prefix).
        namespace:  Kubernetes namespace.

    Returns:
        kubectl rollout undo output.
    """
    if DEMO_MODE:
        return f"[DRY RUN] Would execute: kubectl rollout undo deployment/{deployment} -n {namespace}"

    if DRY_RUN:
        return (
            f"DRY RUN: kubectl rollout undo deployment/{deployment} -n {namespace}\n"
            "Set DRY_RUN=false to execute."
        )
    return _run_kubectl(["rollout", "undo", f"deployment/{deployment}", "-n", namespace])


def kubectl_scale(
    deployment: str,
    replicas: int,
    namespace: str = "production",
) -> str:
    """
    Scale a deployment to the specified number of replicas.

    Args:
        deployment: Deployment name.
        replicas:   Desired replica count.
        namespace:  Kubernetes namespace.

    Returns:
        kubectl scale output.
    """
    if DEMO_MODE:
        return (
            f"[DEMO] Scaled deployment/{deployment} to {replicas} replicas "
            f"in namespace {namespace}. (simulated)"
        )

    if DRY_RUN:
        return (
            f"DRY RUN: kubectl scale deployment/{deployment} "
            f"--replicas={replicas} -n {namespace}\n"
            "Set DRY_RUN=false to execute."
        )
    return _run_kubectl([
        "scale", f"deployment/{deployment}",
        f"--replicas={replicas}", "-n", namespace,
    ])


def kubectl_get_events(namespace: str = "production", pod: str | None = None) -> str:
    """
    Get Kubernetes events, optionally filtered by pod name.

    Args:
        namespace: Kubernetes namespace.
        pod:       Filter events related to this pod name.

    Returns:
        kubectl get events output.
    """
    if DEMO_MODE:
        return _mock_events(namespace, pod)

    cmd = ["get", "events", "-n", namespace, "--sort-by=.lastTimestamp"]
    if pod:
        cmd += [f"--field-selector=involvedObject.name={pod}"]
    return _run_kubectl(cmd)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _run_kubectl(args: list[str]) -> str:
    """Execute a kubectl command and return its stdout."""
    cmd = ["kubectl"] + args
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        if result.returncode != 0:
            return f"Error (exit {result.returncode}):\n{result.stderr}"
        return result.stdout or "(no output)"
    except subprocess.TimeoutExpired:
        return f"kubectl timed out after {TIMEOUT_S}s"
    except FileNotFoundError:
        return "kubectl not found. Is it installed and on PATH?"
    except Exception as e:
        return f"kubectl error: {e}"


# ── Demo responses ─────────────────────────────────────────────────────────────

def _mock_describe(pod: str, ns: str) -> str:
    return f"""\
Name:         {pod}
Namespace:    {ns}
Status:       Running
Conditions:
  Ready:      True
  Available:  True
Events:
  Warning  OOMKilling  2m    kubelet  Memory cgroup out of memory: \
Kill process 12345 ({pod}) score 1200 or sacrifice child
  Normal   Pulled      1m    kubelet  Successfully pulled image
  Normal   Started     1m    kubelet  Started container {pod}
"""


def _mock_logs(pod: str, ns: str, tail: int) -> str:
    return f"""\
[2024-11-03T14:22:01Z] ERROR payment-processor: java.lang.OutOfMemoryError: Java heap space
[2024-11-03T14:22:02Z] ERROR payment-processor: Request failed: timeout after 30000ms
[2024-11-03T14:22:03Z] WARN  payment-processor: Retry 3/3 for transaction tx-98765
[2024-11-03T14:22:05Z] ERROR payment-processor: Connection pool exhausted (max=20)
[2024-11-03T14:22:06Z] FATAL payment-processor: JVM OOMKill imminent, dumping heap...
(showing last {tail} lines of {pod} in {ns})
"""


def _mock_events(ns: str, pod: str | None) -> str:
    subject = pod or "payment-svc"
    return f"""\
LAST SEEN   TYPE      REASON      OBJECT                 MESSAGE
2m          Warning   OOMKilling  pod/{subject}          Memory limit exceeded
5m          Normal    Scheduled   pod/{subject}          Assigned to node worker-3
7m          Normal    Pulling     pod/{subject}          Pulling image "payment:v2.3.0"
10m         Normal    Started     pod/{subject}          Started container payment
"""
