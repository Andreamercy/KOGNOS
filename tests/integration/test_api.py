"""
tests/integration/test_api.py

Integration tests for the KOGNOS FastAPI backend.

Uses FastAPI's TestClient which runs the full application in-process,
including the lifespan startup (GNN inference + RAG init).

These tests run in DEMO_MODE with mocked LLM to avoid requiring:
  - A real Kubernetes cluster
  - A real Qdrant instance
  - A real Anthropic API key
"""

import os
import pytest

# Force demo mode for all integration tests
os.environ["KOGNOS_DEMO_MODE"] = "true"
os.environ["LLM_BACKEND"]      = "anthropic"  # mock engine used in demo mode

from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    """Module-scoped test client — starts app once for all tests in this module."""
    with TestClient(app) as c:
        yield c


# ── Health endpoints ──────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "KOGNOS"
        assert data["status"] == "running"
        assert "endpoints" in data

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Alerts endpoint ───────────────────────────────────────────────────────────

class TestAlertsEndpoint:
    def test_get_alerts_returns_200(self, client):
        resp = client.get("/alerts")
        assert resp.status_code == 200

    def test_alerts_response_structure(self, client):
        resp = client.get("/alerts")
        data = resp.json()
        assert "alerts" in data
        assert "total" in data
        assert "critical" in data
        assert "warning" in data
        assert "as_of" in data

    def test_alerts_are_list(self, client):
        resp = client.get("/alerts")
        assert isinstance(resp.json()["alerts"], list)

    def test_severity_filter(self, client):
        resp = client.get("/alerts?severity=critical")
        assert resp.status_code == 200
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["severity"] == "critical"

    def test_invalid_severity_filter(self, client):
        resp = client.get("/alerts?severity=invalid")
        assert resp.status_code == 422  # Pydantic validation error

    def test_min_score_filter(self, client):
        resp = client.get("/alerts?min_score=0.9")
        assert resp.status_code == 200
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["anomaly_score"] >= 0.9


# ── Graph endpoint ────────────────────────────────────────────────────────────

class TestGraphEndpoint:
    def test_get_graph_returns_200(self, client):
        resp = client.get("/graph")
        assert resp.status_code == 200

    def test_graph_has_nodes_and_edges(self, client):
        resp = client.get("/graph")
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert data["num_nodes"] > 0

    def test_graph_nodes_have_required_fields(self, client):
        resp  = client.get("/graph")
        nodes = resp.json()["nodes"]
        assert len(nodes) > 0
        required = {"id", "name", "namespace", "anomaly_score", "severity",
                    "cpu_usage", "mem_usage", "restart_count", "is_ready"}
        for node in nodes:
            missing = required - set(node.keys())
            assert not missing, f"Node missing fields: {missing}"

    def test_anomaly_scores_bounded(self, client):
        resp  = client.get("/graph")
        nodes = resp.json()["nodes"]
        for node in nodes:
            assert 0.0 <= node["anomaly_score"] <= 1.0

    def test_severity_values_valid(self, client):
        resp     = client.get("/graph")
        nodes    = resp.json()["nodes"]
        valid    = {"critical", "warning", "info", "normal"}
        for node in nodes:
            assert node["severity"] in valid, \
                f"Invalid severity: {node['severity']}"


# ── Query endpoint ────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    def test_query_returns_200(self, client):
        resp = client.post("/query", json={"question": "Why is payment-svc down?"})
        assert resp.status_code == 200

    def test_query_response_structure(self, client):
        resp = client.post("/query", json={"question": "What is the cluster status?"})
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "anomaly_score" in data
        assert "latency_ms" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_query_too_short(self, client):
        resp = client.post("/query", json={"question": "hi"})
        assert resp.status_code == 422

    def test_query_payment_svc(self, client):
        resp = client.post("/query", json={
            "question": "Why is the payment service throwing 500 errors?"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "payment" in data["answer"].lower() or "oom" in data["answer"].lower()

    def test_query_blast_radius(self, client):
        resp = client.post("/query", json={
            "question": "Show me the blast radius if auth-svc goes down."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["answer"]) > 10


# ── Heal endpoint ─────────────────────────────────────────────────────────────

class TestHealEndpoint:
    def test_heal_dry_run(self, client):
        resp = client.post("/heal", json={
            "pod": "payment-svc",
            "namespace": "production",
            "dry_run": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["dry_run"] is True
        assert "command" in data
        assert "kubectl" in data["command"]

    def test_heal_response_structure(self, client):
        resp = client.post("/heal", json={
            "pod": "payment-svc",
            "dry_run": True,
        })
        data = resp.json()
        required = {"pod", "namespace", "action_taken", "command", "dry_run",
                    "anomaly_score", "success", "message", "executed_at"}
        missing = required - set(data.keys())
        assert not missing, f"Response missing fields: {missing}"

    def test_heal_auto_action_selects_scale(self, client):
        resp = client.post("/heal", json={
            "pod": "payment-svc",
            "action": "auto",
            "dry_run": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "scale" in data["command"] or "rollback" in data["command"]

    def test_heal_scale_action(self, client):
        resp = client.post("/heal", json={
            "pod": "payment-svc",
            "action": "scale",
            "replicas": 5,
            "dry_run": True,
        })
        assert resp.status_code == 200
        assert "5" in resp.json()["command"]

    def test_heal_invalid_action(self, client):
        resp = client.post("/heal", json={
            "pod": "test-pod",
            "action": "explode",
            "dry_run": True,
        })
        assert resp.status_code == 422
