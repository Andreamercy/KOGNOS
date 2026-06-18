// ebpf/probes/syscall_trace.c
// eBPF program: per-pod syscall anomaly detection.
// Attaches to the raw_syscalls/sys_enter tracepoint and tracks syscall
// frequency per PID. High syscall rates (e.g. tight loops on EINTR, excessive
// brk/mmap) are a signal of pod misbehaviour before it becomes visible at the
// container metrics level.

#include <linux/bpf.h>
#include <linux/ptrace.h>
#include "bpf_helpers.h"
#include "bpf_tracing.h"

#define MAX_PIDS      65536
#define ALERT_THRESH  10000   // syscalls/second above which we flag anomaly

// Syscall event — one per observed "anomalous burst"
struct syscall_event {
    __u32 pid;
    __u32 syscall_nr;
    __u64 count;            // Burst count since last reset
    __u64 timestamp_ns;
    char  comm[16];
};

// Per-PID syscall counters (reset every inference window)
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, MAX_PIDS);
    __type(key,   __u32);
    __type(value, __u64);
} syscall_count SEC(".maps");

// Perf ring buffer for anomaly events
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size,   sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} syscall_events SEC(".maps");

// Tracepoint format: /sys/kernel/debug/tracing/events/raw_syscalls/sys_enter
struct sys_enter_args {
    __u64 unused;   // common fields
    long  id;       // syscall number
    unsigned long args[6];
};

SEC("tracepoint/raw_syscalls/sys_enter")
int trace_sys_enter(struct sys_enter_args *ctx) {
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid      = (__u32)(pid_tgid >> 32);

    // Increment per-pid counter
    __u64 *cnt = bpf_map_lookup_elem(&syscall_count, &pid);
    if (!cnt) return 0;

    __u64 new_cnt = *cnt + 1;
    bpf_map_update_elem(&syscall_count, &pid, &new_cnt, BPF_ANY);

    // Emit an event on every power-of-2 burst boundary (avoids flooding)
    if ((new_cnt & (new_cnt - 1)) == 0 && new_cnt >= 64) {
        struct syscall_event ev = {};
        ev.pid          = pid;
        ev.syscall_nr   = (__u32)ctx->id;
        ev.count        = new_cnt;
        ev.timestamp_ns = bpf_ktime_get_ns();
        bpf_get_current_comm(&ev.comm, sizeof(ev.comm));
        bpf_perf_event_output(ctx, &syscall_events, BPF_F_CURRENT_CPU,
                              &ev, sizeof(ev));
    }
    return 0;
}

char _license[] SEC("license") = "GPL";
