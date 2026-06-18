# eBPF Deep Dive

## Why eBPF?

Traditional observability requires:
- **Instrumentation**: Adding metrics libraries to application code
- **Sidecars**: Running an Envoy/Istio proxy alongside every pod
- **Polling**: Querying APIs (metrics-server, Prometheus) at fixed intervals

eBPF eliminates all three:
- **Zero instrumentation**: Probes attach to kernel functions
- **No sidecars**: All data collected at the kernel layer
- **Event-driven**: Signals emitted exactly when something happens

## Linux Kernel Requirements

KOGNOS probes use **CO-RE** (Compile Once - Run Everywhere) via `libbpf`.
This requires:
- Linux kernel **≥ 5.8** (BPF Type Format / BTF support)
- Recommended: kernel **≥ 5.15** (full CO-RE + cgroupv2)

## Probe Design

### `network_flow.c` — TCP Flow Tracing

Attaches to `kprobe/tcp_sendmsg`. This fires for every outgoing TCP write,
giving us:
- Source and destination IP (from the `sock` struct in kernel memory)
- Bytes transferred
- Process ID (mapped to pod via cgroup hierarchy)
- Nanosecond-precision timestamp

**Why `tcp_sendmsg` vs XDP?**
XDP operates at the NIC level and is faster, but doesn't have access to
the full socket context needed to correlate traffic to specific pods.
`tcp_sendmsg` has slightly more overhead but provides the pod-level context
KOGNOS needs.

### `syscall_trace.c` — Syscall Anomaly Detection

Attaches to `tracepoint/raw_syscalls/sys_enter`. Tracks syscall frequency
per PID. High-frequency syscall bursts indicate:
- Tight retry loops (application in error state)
- Memory pressure (`mmap`/`brk` calls spike before OOMKill)
- Network saturation (excessive `sendmsg` calls)

Uses a **per-CPU array map** for zero-contention counting.

### `pod_lifecycle.c` — Pod Start/Stop/OOMKill

Attaches to:
- `kprobe/cgroup_attach_task`: Pod start (first task joins cgroup)
- `kprobe/oom_kill_process`: OOMKill event

These events provide the `restart_count` and `is_oomkilled` node features
that are critical predictors of imminent cascade failure.

## Ring Buffer Architecture

All three probes use `BPF_MAP_TYPE_PERF_EVENT_ARRAY` — a per-CPU ring buffer
that allows zero-copy transfer of events from kernel to userspace.

The Go loader reads from this ring buffer continuously and batches events
into 10-second windows before publishing to Kafka.

```
Kernel eBPF prog
    → bpf_perf_event_output()
          → Perf ring buffer (per CPU)
                → Go loader (perf.Reader)
                      → JSON encode
                            → Kafka (ebpf-flows)
                                  → GNN Inference Engine
```

## Security Considerations

- eBPF probes require `CAP_BPF` or `CAP_SYS_ADMIN` (root in practice)
- The DaemonSet runs with `privileged: true` and `hostPID: true`
- All eBPF programs are read-only — they observe kernel state but cannot modify it
- Programs pass the kernel verifier before loading (memory safety guaranteed)
- Use Pod Security Admission (PSA) exceptions to scope the privileged pod
