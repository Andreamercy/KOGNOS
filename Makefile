.PHONY: help install build-ebpf build-go run-api run-inference test lint clean

PYTHON := python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF   := $(VENV)/bin/ruff
MYPY   := $(VENV)/bin/mypy

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Python env ────────────────────────────────────────────────────────────────
install: ## Create venv and install Python dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Python environment ready. Activate with: source $(VENV)/bin/activate"

# ── eBPF probes ───────────────────────────────────────────────────────────────
build-ebpf: ## Compile eBPF C probes to .o objects (requires Linux + clang)
	@echo "🔨 Building eBPF probes..."
	$(MAKE) -C ebpf build

# ── Go loader ─────────────────────────────────────────────────────────────────
build-go: ## Build the Go eBPF loader binary
	@echo "🔨 Building Go eBPF loader..."
	cd ebpf/loader && go build -o ../../bin/kognos-loader .
	@echo "✅ Binary at bin/kognos-loader"

# ── Runtime ───────────────────────────────────────────────────────────────────
run-api: ## Start the FastAPI server (demo mode)
	@echo "🚀 Starting KOGNOS API on http://localhost:8000"
	KOGNOS_DEMO_MODE=true $(VENV)/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-inference: ## Run the GNN inference engine (demo mode)
	@echo "🧠 Starting GNN inference engine..."
	KOGNOS_DEMO_MODE=true $(PYTHON) -m graph.inference.engine \
		--model-path models/graphsage_v1.pt \
		--kafka-topic ebpf-flows

run-ebpf-loader: ## Run the eBPF loader (requires root + Linux)
	@echo "⚡ Starting eBPF loader (requires sudo)..."
	sudo ./bin/kognos-loader

# ── Infrastructure ────────────────────────────────────────────────────────────
infra-up: ## Start local dev infrastructure (Kafka, Qdrant, Redis)
	docker compose up -d
	@echo "⏳ Waiting for services to be healthy..."
	@sleep 10
	@echo "✅ Infrastructure running"

infra-down: ## Stop local dev infrastructure
	docker compose down

# ── Knowledge base ────────────────────────────────────────────────────────────
build-kb: ## Build the RAG knowledge base from runbooks and incidents
	@echo "📚 Building knowledge base..."
	$(PYTHON) -m rag.ingestion.runbook_loader \
		--runbooks docs/runbooks/ \
		--incidents docs/incidents/

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short --cov=. --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v

# ── Linting ───────────────────────────────────────────────────────────────────
lint: ## Run ruff + mypy
	$(RUFF) check .
	$(MYPY) graph/ rag/ api/ --ignore-missing-imports

format: ## Auto-format with ruff
	$(RUFF) format .

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and venv
	rm -rf $(VENV) bin/ __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
