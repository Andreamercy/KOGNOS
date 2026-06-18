# KOGNOS Architecture

## Overview

KOGNOS is a three-layer observability platform for Kubernetes clusters.
Each layer feeds the next, creating a pipeline from raw kernel signals
to natural language answers.

```
                         Engineer
                            │
                    "Why is payment-svc down?"
                            │
              ┌─────────────▼──────────────┐
              │   Layer 3 — RAG + Agent    │
              │   LlamaIndex · Qdrant      │
              │   Claude / Ollama LLM      │
              │   ReAct agent (kubectl)    │
              └─────────────┬──────────────┘
                            │ live anomaly context
              ┌─────────────▼──────────────┐
              │   Layer 2 — GNN Reasoning  │
              │   GraphSAGE (PyTorch)      │
              │   Pod graph inference      │
              │   Anomaly score per pod    │
              └─────────────┬──────────────┘
                            │ telemetry windows
              ┌─────────────▼──────────────┐
              │   Layer 1 — eBPF Telemetry │
              │   kprobe/tcp_sendmsg       │
              │   tracepoint/sys_enter     │
              │   cgroupv2 lifecycle       │
              └────────────────────────────┘
                   Kubernetes Cluster
```

## Layer 1: eBPF Telemetry

eBPF programs run inside the Linux kernel with near-zero overhead.
Three probes are attached:

| Probe                  | Hook                          | Data Captured |
|------------------------|-------------------------------|---------------|
| `network_flow.c`       | `kprobe/tcp_sendmsg`          | src/dst IP, port, bytes, latency |
| `syscall_trace.c`      | `tracepoint/raw_syscalls/sys_enter` | syscall frequency per PID |
| `pod_lifecycle.c`      | `kprobe/oom_kill_process`     | OOMKill timestamps |

Events are streamed via **perf ring buffer** → Go loader → **Kafka**.

## Layer 2: GNN Reasoning

The **GraphSAGE** model operates on a dynamic graph where:
- **Nodes** = pods (features: cpu, mem, restarts, error_rate, latency_p99, is_ready)
- **Edges** = observed network flows (features: bytes/s, latency_ms, error_rate)

GraphSAGE is **inductive** — it generalises to pods not seen during training.
The model outputs an anomaly probability [0, 1] per pod.

Inference runs every 10 seconds on a rolling telemetry window.

## Layer 3: RAG Conversational AI

The RAG pipeline combines:
1. **Vector retrieval** from Qdrant (runbooks + past incidents)
2. **Live context injection** (current GNN anomaly scores)
3. **LLM generation** (Claude or Ollama)

The **ReAct agent** can take autonomous actions:
- `kubectl describe` / `kubectl logs` for investigation
- `kubectl scale` / `kubectl rollout undo` for remediation
- All gated by confidence threshold + dry-run mode

## API Surface

| Method | Endpoint  | Description |
|--------|-----------|-------------|
| GET    | `/`       | Health + endpoint discovery |
| GET    | `/health` | Kubernetes liveness probe |
| POST   | `/query`  | Natural language question |
| GET    | `/alerts` | Live anomaly list |
| GET    | `/graph`  | Cluster graph snapshot |
| POST   | `/heal`   | Trigger remediation |
| WS     | `/stream` | Real-time alert stream |

## Data Flow

```
eBPF probes (kernel)
    └─→ Go loader
          └─→ Kafka (ebpf-flows topic)
                └─→ GNN Inference Engine
                      ├─→ Redis (latest alerts cache)
                      └─→ FastAPI /alerts + /stream
                              └─→ RAG query engine
                                    ├─→ Qdrant (runbook retrieval)
                                    └─→ Claude / Ollama (answer generation)
```
