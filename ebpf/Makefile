# ebpf/Makefile
# Compiles eBPF C probes to BPF ELF objects using clang.
# Requires: clang ≥ 14, libbpf headers, Linux kernel headers.

CLANG        := clang
LLC          := llc
BPFTOOL      := bpftool
KERNEL_VER   := $(shell uname -r)
KERNEL_ARCH  := $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/')

PROBES_DIR   := probes
OUT_DIR      := out
BPF_CFLAGS   := -O2 -g -Wall -Wno-unused-value -Wno-pointer-sign \
                 -Wno-compare-distinct-pointer-types \
                 -target bpf \
                 -D__TARGET_ARCH_$(KERNEL_ARCH) \
                 -I/usr/include/$(shell uname -m)-linux-gnu \
                 -I/usr/include

PROBES := network_flow syscall_trace pod_lifecycle

.PHONY: all build clean check-deps

all: check-deps build

build: $(OUT_DIR) $(PROBES:%=$(OUT_DIR)/%.o)
	@echo "✅ eBPF probes compiled to $(OUT_DIR)/"

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Compile each .c probe to a .o BPF ELF object
$(OUT_DIR)/%.o: $(PROBES_DIR)/%.c
	@echo "  CC $<"
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

# Generate Go bindings from compiled objects (optional, uses bpf2go)
generate:
	go generate ./...

# Verify the compiled objects are valid BPF programs
verify: build
	@for obj in $(OUT_DIR)/*.o; do \
		echo "Verifying $$obj..."; \
		$(BPFTOOL) prog load $$obj /sys/fs/bpf/kognos_verify && \
		rm -f /sys/fs/bpf/kognos_verify; \
	done

clean:
	rm -rf $(OUT_DIR)

check-deps:
	@which $(CLANG) > /dev/null 2>&1 || (echo "❌ clang not found. Install: apt install clang" && exit 1)
	@echo "✅ Dependencies OK (clang=$(shell $(CLANG) --version | head -1))"
