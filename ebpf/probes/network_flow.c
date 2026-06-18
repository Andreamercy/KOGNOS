// ebpf/probes/network_flow.c
// eBPF program: traces TCP network flows between pods at the kernel level.
// Attached via kprobe on tcp_sendmsg — fires for every TCP send operation.
// Exports flow events to userspace via a perf event array ring buffer.

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/ptrace.h>
#include <linux/socket.h>
#include <net/sock.h>
#include "bpf_helpers.h"
#include "bpf_tracing.h"

// Flow event structure — exported to userspace for each TCP send
struct flow_event {
    __u32 src_ip;           // Source IP (network byte order)
    __u32 dst_ip;           // Destination IP
    __u16 src_port;         // Source port
    __u16 dst_port;         // Destination port
    __u64 bytes;            // Bytes sent in this call
    __u64 timestamp_ns;     // Kernel timestamp (nanoseconds)
    __u32 pid;              // Process ID (pod-level)
    __u32 tid;              // Thread ID
    char  comm[16];         // Process name (e.g., "java", "node")
    __u8  protocol;         // IP protocol (TCP=6)
    __u8  pad[3];           // Alignment padding
};

// Perf event array — one slot per CPU — for zero-copy ring buffer export
struct {
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size,   sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} flow_events SEC(".maps");

// Per-socket in-flight tracking (pid_tgid → bytes so far)
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key,   __u64);
    __type(value, __u64);
} bytes_in_flight SEC(".maps");

// Helper: read IPv4 address from sock struct safely
static __always_inline __u32 read_src_ip(struct sock *sk) {
    __u32 ip = 0;
    bpf_probe_read_kernel(&ip, sizeof(ip), &sk->__sk_common.skc_rcv_saddr);
    return ip;
}

static __always_inline __u32 read_dst_ip(struct sock *sk) {
    __u32 ip = 0;
    bpf_probe_read_kernel(&ip, sizeof(ip), &sk->__sk_common.skc_daddr);
    return ip;
}

SEC("kprobe/tcp_sendmsg")
int trace_tcp_send(struct pt_regs *ctx) {
    struct flow_event ev = {};

    // Capture timing and process identity first — cheap operations
    ev.timestamp_ns = bpf_ktime_get_ns();
    __u64 pid_tgid  = bpf_get_current_pid_tgid();
    ev.pid          = (__u32)(pid_tgid >> 32);
    ev.tid          = (__u32)pid_tgid;
    bpf_get_current_comm(&ev.comm, sizeof(ev.comm));
    ev.protocol     = IPPROTO_TCP;

    // Get sock pointer from first argument register
    struct sock *sk = (struct sock *)PT_REGS_PARM1(ctx);
    if (!sk) return 0;

    // Read connection tuple
    ev.src_ip   = read_src_ip(sk);
    ev.dst_ip   = read_dst_ip(sk);
    bpf_probe_read_kernel(&ev.src_port, sizeof(ev.src_port),
                          &sk->__sk_common.skc_num);
    bpf_probe_read_kernel(&ev.dst_port, sizeof(ev.dst_port),
                          &sk->__sk_common.skc_dport);

    // Skip loopback (127.x.x.x) — not inter-pod traffic
    if ((ev.src_ip & 0xFF) == 127 || (ev.dst_ip & 0xFF) == 127) return 0;

    // Get message size from iov_iter (third argument)
    // Simplified: read size_t from iov_iter->count
    struct iov_iter *iter = (struct iov_iter *)PT_REGS_PARM3(ctx);
    if (iter) {
        bpf_probe_read_kernel(&ev.bytes, sizeof(ev.bytes), &iter->count);
    }

    // Emit event to userspace ring buffer
    bpf_perf_event_output(ctx, &flow_events, BPF_F_CURRENT_CPU,
                          &ev, sizeof(ev));
    return 0;
}

char _license[] SEC("license") = "GPL";
