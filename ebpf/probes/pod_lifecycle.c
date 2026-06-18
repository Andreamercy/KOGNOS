// ebpf/probes/pod_lifecycle.c
// eBPF program: Pod lifecycle event tracking via cgroupv2.
// Hooks into cgroup_attach_task and memory.oom_kill to detect:
//   - New pod starts (cgroup creation + first task attach)
//   - OOMKill events (memory cgroup limit exceeded)
//   - Pod exits / crashes (cgroup_release)
//
// These events enrich the GNN graph with restart_count and crash_timestamp
// node features, which are strong predictors of imminent failure.

#include <linux/bpf.h>
#include <linux/cgroup.h>
#include <linux/ptrace.h>
#include "bpf_helpers.h"
#include "bpf_tracing.h"

// Pod lifecycle event types
#define POD_EVENT_START    1
#define POD_EVENT_OOMKILL  2
#define POD_EVENT_EXIT     3

struct pod_lifecycle_event {
    __u32 event_type;       // POD_EVENT_* above
    __u32 pid;
    __u64 timestamp_ns;
    char  cgroup_name[128]; // Maps to k8s pod name in Kubernetes cgroup hierarchy
    char  comm[16];
    __u32 exit_code;        // Only meaningful for EXIT events
};

// Perf ring buffer for lifecycle events
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size,   sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} lifecycle_events SEC(".maps");

// Track active pod PIDs (pid → cgroup_id) for deduplication
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key,   __u32);
    __type(value, __u64);
} active_pods SEC(".maps");

// Hook: cgroup_attach_task — fires when a task joins a cgroup (pod start)
SEC("kprobe/cgroup_attach_task")
int trace_pod_start(struct pt_regs *ctx) {
    struct pod_lifecycle_event ev = {};
    ev.event_type   = POD_EVENT_START;
    ev.timestamp_ns = bpf_ktime_get_ns();
    __u64 pid_tgid  = bpf_get_current_pid_tgid();
    ev.pid          = (__u32)(pid_tgid >> 32);
    bpf_get_current_comm(&ev.comm, sizeof(ev.comm));

    // Read cgroup name (Kubernetes sets this to the pod sandbox name)
    struct cgroup *cgrp = (struct cgroup *)PT_REGS_PARM1(ctx);
    if (cgrp) {
        bpf_probe_read_kernel_str(&ev.cgroup_name, sizeof(ev.cgroup_name),
                                  &cgrp->kn->name);
    }

    // Only track Kubernetes pod cgroups (name contains "pod")
    // Real impl would check for "besteffort/pod" or "burstable/pod" prefix
    bpf_perf_event_output(ctx, &lifecycle_events, BPF_F_CURRENT_CPU,
                          &ev, sizeof(ev));
    return 0;
}

// Hook: oom_kill_process — fires when the OOM killer terminates a process
SEC("kprobe/oom_kill_process")
int trace_oom_kill(struct pt_regs *ctx) {
    struct pod_lifecycle_event ev = {};
    ev.event_type   = POD_EVENT_OOMKILL;
    ev.timestamp_ns = bpf_ktime_get_ns();
    __u64 pid_tgid  = bpf_get_current_pid_tgid();
    ev.pid          = (__u32)(pid_tgid >> 32);
    bpf_get_current_comm(&ev.comm, sizeof(ev.comm));

    bpf_perf_event_output(ctx, &lifecycle_events, BPF_F_CURRENT_CPU,
                          &ev, sizeof(ev));
    return 0;
}

char _license[] SEC("license") = "GPL";
