# KOGNOS 🧠
### Multi-Agent Kubernetes Observability Platform

> Real-time anomaly detection and conversational intelligence for Kubernetes clusters — powered by eBPF telemetry, Graph Neural Networks, and RAG-based AI agents.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch-geometric.readthedocs.io)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.28+-326CE5?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat&logo=go&logoColor=white)](https://go.dev)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## What is KOGNOS?

Enterprise Kubernetes clusters generate thousands of telemetry signals per second. Existing tools like Grafana or Datadog surface dashboards — but they can't *reason* about **why** something is failing or **what to do next**.

KOGNOS is a three-layer observability platform that:

1. **Collects** kernel-level signals invisibly via eBPF (no code instrumentation needed)
2. **Reasons** over the cluster topology using a Graph Neural Network that understands pod dependencies
3. **Converses** — engineers ask natural language questions and get grounded, actionable answers backed by live cluster context

Targeting the problem of cascading failure detection and intelligent incident response at enterprise scale.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        KOGNOS Platform                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Layer 3 — Conversational AI (RAG + Agentic)            │   │
│  │  LlamaIndex · Qdrant · Claude API / Ollama              │   │
│  │  "Why is payment-svc degraded?" → grounded answer       │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  Layer 2 — Graph Neural Network Reasoning               │   │
│  │  PyTorch Geometric · GraphSAGE · Anomaly Scoring        │   │
│  │  Pods as nodes · Network flows as edges                 │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  Layer 1 — eBPF Telemetry Collection                    │   │
│  │  Cilium · libbpf · Kernel-level signals                 │   │
│  │  Network flows · Syscalls · Pod-to-pod traffic          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Kubernetes Cluster (target environment)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Project Structure](#project-structure)
3. [Layer 1 — eBPF Telemetry](#layer-1--ebpf-telemetry)
4. [Layer 2 — GNN Reasoning](#layer-2--gnn-reasoning)
5. [Layer 3 — RAG Conversational AI](#layer-3--rag-conversational-ai)
6. [Agentic Self-Healing](#agentic-self-healing)
7. [Setup & Installation](#setup--installation)
8. [Running KOGNOS](#running-kognos)
9. [API Reference](#api-reference)
10. [Example Queries](#example-queries)
11. [Roadmap](#roadmap)

---

## Tech Stack

| Layer | Component | Technology |
|---|---|---|
| Telemetry | eBPF probe | Cilium, libbpf, BCC |
| Telemetry | Streaming pipeline | Apache Kafka / Redis Streams |
| Graph ML | Graph construction | NetworkX → PyTorch Geometric |
| Graph ML | Model | GraphSAGE (inductive GNN) |
| Graph ML | Training | PyTorch 2.x + CUDA |
| RAG | Orchestration | LlamaIndex |
| RAG | Vector store | Qdrant |
| RAG | LLM backend | Claude API (claude-sonnet-4-6) / Ollama |
| Agentic | Tool execution | LlamaIndex ReAct Agent |
| Infra | Orchestration | Kubernetes 1.28+, Helm 3 |
| Infra | Service mesh | Cilium CNI |
| API | Backend | FastAPI + WebSockets |
| Monitoring | Metrics | Prometheus + Grafana (baseline) |
| Language | Core | Python 3.11+, Go 1.21 (eBPF loader) |

---

## Project Structure

```
kognos/
├── ebpf/                          # Layer 1 — Kernel telemetry
│   ├── probes/
│   │   ├── network_flow.c         # eBPF C program for network tracing
│   │   ├── syscall_trace.c        # Syscall-level monitoring
│   │   └── pod_lifecycle.c        # Pod start/stop/crash events
│   ├── loader/
│   │   ├── main.go                # Go loader — attaches eBPF probes
│   │   └── exporter.go            # Exports events to Kafka/Redis
│   └── Makefile
│
├── graph/                         # Layer 2 — GNN reasoning
│   ├── builder/
│   │   ├── cluster_graph.py       # Builds PyG graph from k8s API + telemetry
│   │   └── feature_encoder.py     # Node/edge feature engineering
│   ├── model/
│   │   ├── graphsage.py           # GraphSAGE model definition
│   │   ├── anomaly_head.py        # Anomaly scoring head
│   │   └── train.py               # Training loop
│   ├── inference/
│   │   ├── engine.py              # Real-time inference on live graph
│   │   └── scorer.py              # Anomaly score → alert thresholds
│   └── data/
│       ├── synthetic_gen.py       # Synthetic cluster failure data generator
│       └── datasets/              # Saved graph snapshots for training
│
├── rag/                           # Layer 3 — Conversational AI
│   ├── ingestion/
│   │   ├── runbook_loader.py      # Load runbooks into Qdrant
│   │   ├── incident_loader.py     # Load past incident reports
│   │   └── live_context.py        # Stream live anomaly context into index
│   ├── retrieval/
│   │   ├── query_engine.py        # LlamaIndex query engine setup
│   │   └── reranker.py            # Cohere / cross-encoder reranking
│   ├── agent/
│   │   ├── kognos_agent.py        # ReAct agent with kubectl tools
│   │   ├── tools.py               # kubectl exec, describe, rollback tools
│   │   └── prompts.py             # System prompts for agent behavior
│   └── llm/
│       ├── claude_backend.py      # Anthropic Claude API client
│       └── ollama_backend.py      # Local Ollama fallback
│
├── api/                           # FastAPI backend
│   ├── main.py
│   ├── routes/
│   │   ├── query.py               # POST /query — natural language questions
│   │   ├── alerts.py              # GET /alerts — live anomaly feed
│   │   └── graph.py               # GET /graph — cluster graph snapshot
│   └── websocket/
│       └── stream.py              # WS /stream — real-time alert stream
│
├── helm/                          # Helm chart for k8s deployment
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       ├── daemonset.yaml         # eBPF loader runs as DaemonSet
│       └── configmap.yaml
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── docs/
│   ├── architecture.md
│   ├── ebpf-deep-dive.md
│   └── gnn-design.md
│
├── docker-compose.yml             # Local dev stack
├── requirements.txt
├── go.mod
└── README.md
```

---

## Layer 1 — eBPF Telemetry

eBPF programs run inside the Linux kernel with near-zero overhead. KOGNOS attaches probes at the network and syscall level — **no changes to application code required**.

### What gets captured

| Signal | eBPF hook | Data collected |
|---|---|---|
| Pod-to-pod network flows | `kprobe/tcp_sendmsg` | src/dst pod IP, bytes, latency |
| DNS queries | XDP hook | query name, response time, NXDOMAIN |
| Syscall anomalies | `tracepoint/sys_enter` | unusual syscalls per pod |
| Pod lifecycle | cgroupv2 events | start, OOMKill, crash timestamps |

### eBPF probe (network_flow.c)

```c
// ebpf/probes/network_flow.c
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include "bpf_helpers.h"

struct flow_event {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u64 bytes;
    __u64 timestamp_ns;
    __u32 pid;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(int));
    __uint(value_size, sizeof(int));
} flow_events SEC(".maps");

SEC("kprobe/tcp_sendmsg")
int trace_tcp_send(struct pt_regs *ctx) {
    struct flow_event ev = {};
    ev.timestamp_ns = bpf_ktime_get_ns();
    ev.pid = bpf_get_current_pid_tgid() >> 32;
    // populate src/dst from sock struct
    bpf_perf_event_output(ctx, &flow_events, BPF_F_CURRENT_CPU, &ev, sizeof(ev));
    return 0;
}

char _license[] SEC("license") = "GPL";
```

### Go loader

```go
// ebpf/loader/main.go
package main

import (
    "github.com/cilium/ebpf"
    "github.com/cilium/ebpf/link"
    "github.com/cilium/ebpf/perf"
)

func main() {
    // Load compiled eBPF object
    objs := bpfObjects{}
    if err := loadBpfObjects(&objs, nil); err != nil {
        log.Fatalf("loading objects: %v", err)
    }
    defer objs.Close()

    // Attach kprobe to tcp_sendmsg
    kp, err := link.Kprobe("tcp_sendmsg", objs.TraceTcpSend, nil)
    if err != nil {
        log.Fatalf("attach kprobe: %v", err)
    }
    defer kp.Close()

    // Read events from perf ring buffer and export to Kafka
    rd, _ := perf.NewReader(objs.FlowEvents, os.Getpagesize())
    for {
        record, _ := rd.Read()
        exportToKafka(record.RawSample)
    }
}
```

---

## Layer 2 — GNN Reasoning

Plain timeseries ML sees each pod in isolation. Kubernetes failures are **relational** — a crashing database pod cascades to every service that depends on it. GraphSAGE learns these dependency patterns.

### Graph construction

```python
# graph/builder/cluster_graph.py
import torch
from torch_geometric.data import Data
from kubernetes import client, config

def build_cluster_graph(telemetry_window: dict) -> Data:
    """
    Build a PyTorch Geometric graph from live k8s state + eBPF telemetry.
    Nodes = pods. Edges = observed network flows between pods.
    """
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    pods = v1.list_pod_for_all_namespaces().items

    # Node features per pod:
    # [cpu_usage, mem_usage, restart_count, error_rate, latency_p99, is_ready]
    node_features = []
    pod_index = {}

    for i, pod in enumerate(pods):
        pod_index[pod.metadata.name] = i
        features = extract_pod_features(pod, telemetry_window)
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Edges from observed eBPF network flows
    edge_index = []
    edge_attr  = []

    for flow in telemetry_window["flows"]:
        src = pod_index.get(flow["src_pod"])
        dst = pod_index.get(flow["dst_pod"])
        if src is not None and dst is not None:
            edge_index.append([src, dst])
            edge_attr.append([flow["bytes"], flow["latency_ms"], flow["error_rate"]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
```

### GraphSAGE model

```python
# graph/model/graphsage.py
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class KOGNOSGraphSAGE(nn.Module):
    """
    Inductive GNN — can generalize to new pods/nodes not seen during training.
    Outputs an anomaly score per node (pod).
    """
    def __init__(self, in_channels: int, hidden: int = 128, out_channels: int = 64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, out_channels)
        self.anomaly_head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()           # output: anomaly probability [0, 1]
        )

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.anomaly_head(x)   # shape: [num_pods, 1]
```

### Real-time inference

```python
# graph/inference/engine.py
import asyncio
from graph.builder.cluster_graph import build_cluster_graph
from graph.model.graphsage import KOGNOSGraphSAGE

class InferenceEngine:
    def __init__(self, model_path: str):
        self.model = KOGNOSGraphSAGE(in_channels=6)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    async def run(self, telemetry_stream):
        async for window in telemetry_stream:
            graph = build_cluster_graph(window)
            with torch.no_grad():
                scores = self.model(graph.x, graph.edge_index)
            anomalies = self.score_to_alerts(scores, graph)
            yield anomalies

    def score_to_alerts(self, scores, graph) -> list[dict]:
        alerts = []
        for pod_idx, score in enumerate(scores):
            if score.item() > 0.75:          # configurable threshold
                alerts.append({
                    "pod": graph.pod_names[pod_idx],
                    "anomaly_score": round(score.item(), 4),
                    "severity": "critical" if score > 0.9 else "warning",
                })
        return alerts
```

---

## Layer 3 — RAG Conversational AI

When an anomaly fires, an engineer can ask KOGNOS natural language questions. The RAG layer retrieves grounded context from runbooks, past incidents, and live telemetry before answering.

### Qdrant index setup

```python
# rag/ingestion/runbook_loader.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def build_knowledge_base(runbooks_dir: str, incidents_dir: str):
    client = QdrantClient(host="localhost", port=6333)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="kognos_knowledge"
    )

    # Load runbooks + past incident reports
    docs = SimpleDirectoryReader(runbooks_dir).load_data()
    docs += SimpleDirectoryReader(incidents_dir).load_data()

    index = VectorStoreIndex.from_documents(
        docs,
        vector_store=vector_store,
    )
    return index
```

### Query engine

```python
# rag/retrieval/query_engine.py
from llama_index.core import VectorStoreIndex
from llama_index.llms.anthropic import Anthropic
from rag.ingestion.live_context import get_live_context

def build_query_engine(index: VectorStoreIndex, live_alerts: list[dict]):
    llm = Anthropic(model="claude-sonnet-4-6")

    # Inject live anomaly context into every query
    live_ctx = get_live_context(live_alerts)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        system_prompt=f"""
You are KOGNOS, an expert Kubernetes SRE assistant.
You answer questions about cluster health using retrieved runbooks and incident history.

Current live cluster anomalies:
{live_ctx}

Always ground your answers in retrieved context.
If suggesting a remediation action, be explicit about the kubectl command to run.
Be concise and actionable.
        """,
    )
    return query_engine
```

### ReAct agent with kubectl tools

```python
# rag/agent/kognos_agent.py
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
import subprocess

def kubectl_describe(pod_name: str, namespace: str = "default") -> str:
    """Describe a Kubernetes pod to get its current status and events."""
    result = subprocess.run(
        ["kubectl", "describe", "pod", pod_name, "-n", namespace],
        capture_output=True, text=True
    )
    return result.stdout

def kubectl_logs(pod_name: str, namespace: str = "default", tail: int = 50) -> str:
    """Fetch the last N lines of logs from a pod."""
    result = subprocess.run(
        ["kubectl", "logs", pod_name, "-n", namespace, f"--tail={tail}"],
        capture_output=True, text=True
    )
    return result.stdout

def kubectl_rollback(deployment: str, namespace: str = "default") -> str:
    """Roll back a deployment to its previous revision."""
    result = subprocess.run(
        ["kubectl", "rollout", "undo", f"deployment/{deployment}", "-n", namespace],
        capture_output=True, text=True
    )
    return result.stdout

tools = [
    FunctionTool.from_defaults(fn=kubectl_describe),
    FunctionTool.from_defaults(fn=kubectl_logs),
    FunctionTool.from_defaults(fn=kubectl_rollback),
]

def build_agent(llm, query_engine):
    return ReActAgent.from_tools(
        tools,
        llm=llm,
        verbose=True,
        max_iterations=6,
    )
```

---

## Agentic Self-Healing

When anomaly score > 0.9 and a known remediation pattern exists, KOGNOS can act autonomously:

```
Anomaly detected: payment-svc  score=0.94
  → Agent retrieves runbook: "High error rate on payment-svc"
  → Runbook match: OOMKill pattern → scale up replicas
  → Agent executes: kubectl scale deployment/payment-svc --replicas=5
  → Monitors for 60s → score drops to 0.12 → resolved ✅
```

Auto-heal actions are gated by a **confidence threshold** and a **dry-run mode** for safety.

---

## Setup & Installation

### Prerequisites

- Linux kernel ≥ 5.15 (eBPF CO-RE support)
- Kubernetes cluster (minikube, kind, or production)
- Python 3.11+
- Go 1.21+
- Docker + Helm 3
- Qdrant running (Docker or cloud)
- Anthropic API key (or local Ollama)

### 1. Clone and install Python deps

```bash
git clone https://github.com/Andreamercy/KOGNOS.git
cd KOGNOS

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Build eBPF probes

```bash
cd ebpf
make build
# Compiles .c probes → .o objects via clang + bpf target
```

### 3. Build Go loader

```bash
cd ebpf/loader
go build -o kognos-loader .
```

### 4. Start local dev stack

```bash
# Starts Kafka, Qdrant, Redis via Docker Compose
docker compose up -d
```

### 5. Set environment variables

```bash
export ANTHROPIC_API_KEY=your_key_here
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export KAFKA_BOOTSTRAP=localhost:9092
export KUBECONFIG=~/.kube/config
```

### 6. Build the knowledge base

```bash
python -m rag.ingestion.runbook_loader \
  --runbooks docs/runbooks/ \
  --incidents docs/incidents/
```

### 7. Deploy to Kubernetes via Helm

```bash
helm install kognos ./helm \
  --set anthropicApiKey=$ANTHROPIC_API_KEY \
  --set qdrant.host=qdrant-service \
  --namespace kognos --create-namespace
```

---

## Running KOGNOS

### Start the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start the inference engine

```bash
python -m graph.inference.engine \
  --model-path models/graphsage_v1.pt \
  --kafka-topic ebpf-flows
```

### Start the eBPF loader (requires root)

```bash
sudo ./ebpf/loader/kognos-loader
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Natural language question about cluster |
| `GET` | `/alerts` | Current live anomaly list with scores |
| `GET` | `/graph` | Cluster graph snapshot (JSON) |
| `WS` | `/stream` | Real-time alert stream via WebSocket |
| `POST` | `/heal` | Trigger autonomous remediation |

### Example: POST /query

```json
// Request
{
  "question": "Why is the payment service throwing 500 errors?"
}

// Response
{
  "answer": "The payment-svc pod (payment-svc-7d9f8b-xk2pq) has an anomaly score of 0.91. Retrieved runbook 'payment-svc-high-error-rate' indicates this pattern matches an OOMKill loop — the pod is being killed before completing requests. Recommended action: kubectl scale deployment/payment-svc --replicas=5 -n production",
  "sources": ["runbooks/payment-svc.md", "incidents/2024-11-incident-03.md"],
  "anomaly_score": 0.91,
  "suggested_command": "kubectl scale deployment/payment-svc --replicas=5 -n production"
}
```

---

## Example Queries

```
"Why is the payment service degraded?"
"Which pods are most likely to fail in the next 5 minutes?"
"Show me the blast radius if auth-svc goes down."
"What happened last time we saw this error pattern?"
"Roll back the checkout service to the previous version."
```

---

## Roadmap

- [x] Architecture design & proof of concept
- [ ] eBPF probe implementation (network flows)
- [ ] GraphSAGE model training on synthetic data
- [ ] LlamaIndex + Qdrant RAG pipeline
- [ ] FastAPI backend + WebSocket streaming
- [ ] Helm chart for cluster deployment
- [ ] Auto-heal agent with dry-run mode
- [ ] Synthetic failure dataset generator
- [ ] Grafana dashboard integration
- [ ] Multi-cluster support
- [ ] Fine-tuned domain model (replace RAG for common patterns)

---

## Contributing

Pull requests welcome. For major changes, open an issue first.

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check . && mypy .
```

---

## Author

**Andrea Mercy** — AI Engineering, SRM Institute of Science and Technology
GitHub: [@Andreamercy](https://github.com/Andreamercy)

---

*Multi-Agent Kubernetes Observability · eBPF · Graph Neural Networks · RAG*
