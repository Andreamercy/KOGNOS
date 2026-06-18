# GNN Design: KOGNOSGraphSAGE

## Why GraphSAGE?

Plain timeseries anomaly detection (e.g. Isolation Forest on pod metrics)
misses **relational context**: a pod with normal CPU and memory may be
anomalous because it's connected to three failing dependencies.

GraphSAGE (Hamilton et al., 2017) is chosen over GCN because:

1. **Inductive**: Generates embeddings for new nodes not seen during training.
   Critical for Kubernetes where pods are ephemeral — new pods appear every hour.
2. **Scalable**: Samples fixed-size neighbourhoods (mean aggregation) rather than
   requiring the full adjacency matrix.
3. **Edge-aware**: Edge features (flow bytes, latency, error rate) capture the
   *quality* of the dependency relationship, not just its existence.

## Node Feature Engineering

| Feature | Source | Normalisation |
|---------|--------|---------------|
| `cpu_usage` | Prometheus / metrics-server | Already [0, 1] |
| `mem_usage` | Prometheus / metrics-server | Already [0, 1] |
| `restart_count` | Kubernetes API | log1p(n) / log1p(100) |
| `error_rate` | eBPF + Envoy metrics | Already [0, 1] |
| `latency_p99` | eBPF flow measurements | / 10000 ms cap |
| `is_ready` | Kubernetes `Ready` condition | Binary 0/1 |

## Training Strategy

The model is trained on **synthetic cluster failure scenarios** because:
- Real incident data is sparse and expensive to label
- Synthetic data allows controlled failure injection with known ground truth
- The model generalises to real scenarios because the failure *patterns*
  (OOMKill, network partition, cascade) are universal

Six failure types are generated:
1. **Normal** (70% of training data)
2. OOMKill (single pod)
3. Network partition (data tier isolated)
4. CPU throttling storm
5. Rolling restart (multiple pods restarting sequentially)
6. Cascade failure (root cause propagates to dependents)

## Architecture

```
Node features [N, 6]   Edge index [2, E]
       │                     │
  SAGEConv(6 → 128)  ←───────┘
  ReLU + Dropout(0.3)
       │
  SAGEConv(128 → 64)
  ReLU
       │
  Linear(64 → 32)
  ReLU + Dropout(0.2)
       │
  Linear(32 → 1)
  Sigmoid
       │
 Anomaly score [N, 1] ∈ [0, 1]
```

## Loss Function

Binary cross-entropy against synthetic ground-truth labels.
Labels are computed as a weighted sum of observable signals:
```
label = 0.2 * cpu + 0.2 * mem + 0.25 * restart_norm
      + 0.25 * error_rate + 0.10 * latency_norm
```
Pods that are `NotReady` are clamped to label ≥ 0.85.

## Thresholds

| Threshold | Score | Action |
|-----------|-------|--------|
| Info      | ≥ 0.50 | Log to dashboard |
| Warning   | ≥ 0.75 | Slack alert |
| Critical  | ≥ 0.90 | PagerDuty page |
| Auto-heal | ≥ 0.90 | Trigger remediation agent |

All thresholds are configurable via environment variables.
