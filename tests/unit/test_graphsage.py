"""
tests/unit/test_graphsage.py

Unit tests for the KOGNOSGraphSAGE model.

These tests use CPU-only torch and do NOT require a Kubernetes cluster,
Kafka, or Qdrant — they verify the model's forward pass shape, output
range, and anomaly scoring correctness.
"""

import pytest
import torch

# Skip entire module if PyTorch Geometric is not installed
pyg = pytest.importorskip("torch_geometric",
                           reason="torch_geometric required for GNN tests")


from graph.model.graphsage import KOGNOSGraphSAGE
from graph.model.anomaly_head import AnomalyHead, ThresholdConfig, Severity


class TestKOGNOSGraphSAGE:
    """Tests for the GraphSAGE backbone model."""

    @pytest.fixture
    def model(self):
        return KOGNOSGraphSAGE(in_channels=6, hidden=32, out_channels=16, dropout=0.0)

    @pytest.fixture
    def simple_graph(self):
        """A tiny 5-node graph with 4 edges."""
        x          = torch.rand(5, 6)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        return x, edge_index

    def test_output_shape(self, model, simple_graph):
        """Forward pass should return [N, 1] tensor."""
        x, edge_index = simple_graph
        out = model(x, edge_index)
        assert out.shape == (5, 1), f"Expected [5, 1], got {out.shape}"

    def test_output_range(self, model, simple_graph):
        """All anomaly scores should be in [0, 1] (sigmoid output)."""
        x, edge_index = simple_graph
        out = model(x, edge_index)
        assert out.min().item() >= 0.0, "Scores should be >= 0"
        assert out.max().item() <= 1.0, "Scores should be <= 1"

    def test_embed_shape(self, model, simple_graph):
        """embed() should return [N, out_channels]."""
        x, edge_index = simple_graph
        emb = model.embed(x, edge_index)
        assert emb.shape == (5, 16), f"Expected [5, 16], got {emb.shape}"

    def test_score_method(self, model, simple_graph):
        """score() convenience method should return a list of floats."""
        x, edge_index = simple_graph
        scores = model.score(x, edge_index)
        assert isinstance(scores, list)
        assert len(scores) == 5
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_no_gradient_in_eval(self, model, simple_graph):
        """Inference should not require gradients."""
        x, edge_index = simple_graph
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        assert not out.requires_grad

    def test_different_graph_sizes(self, model):
        """Model should handle arbitrary graph sizes inductively."""
        for n in [1, 10, 100]:
            x          = torch.rand(n, 6)
            edge_index = torch.zeros((2, 0), dtype=torch.long)  # no edges
            out = model(x, edge_index)
            assert out.shape == (n, 1)

    def test_isolated_nodes(self, model):
        """Graph with isolated nodes (no edges) should not crash."""
        x          = torch.rand(3, 6)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        out = model(x, edge_index)
        assert out.shape == (3, 1)


class TestAnomalyHead:
    """Tests for the standalone AnomalyHead module."""

    def test_output_range(self):
        head    = AnomalyHead(in_channels=16, hidden=8)
        embeddings = torch.rand(10, 16)
        out = head(embeddings)
        assert out.shape == (10, 1)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0


class TestThresholdConfig:
    """Tests for severity classification thresholds."""

    def setup_method(self):
        self.cfg = ThresholdConfig(critical=0.9, warning=0.75, info=0.5)

    def test_critical(self):
        assert self.cfg.classify(0.95) == Severity.CRITICAL
        assert self.cfg.classify(0.90) == Severity.CRITICAL

    def test_warning(self):
        assert self.cfg.classify(0.80) == Severity.WARNING
        assert self.cfg.classify(0.75) == Severity.WARNING

    def test_info(self):
        assert self.cfg.classify(0.60) == Severity.INFO
        assert self.cfg.classify(0.50) == Severity.INFO

    def test_normal(self):
        assert self.cfg.classify(0.49) == Severity.NORMAL
        assert self.cfg.classify(0.00) == Severity.NORMAL

    def test_score_to_alerts_filters_normal(self):
        scores   = torch.tensor([[0.95], [0.4], [0.8]])
        names    = ["pod-a", "pod-b", "pod-c"]
        alerts   = AnomalyHead.score_to_alerts(scores, names, self.cfg)
        assert len(alerts) == 2
        assert all(a["severity"] != "normal" for a in alerts)

    def test_score_to_alerts_sorted(self):
        scores = torch.tensor([[0.8], [0.95], [0.76]])
        names  = ["a", "b", "c"]
        alerts = AnomalyHead.score_to_alerts(scores, names, self.cfg)
        assert alerts[0]["anomaly_score"] >= alerts[1]["anomaly_score"]

    def test_auto_healable_flag(self):
        scores = torch.tensor([[0.95]])
        alerts = AnomalyHead.score_to_alerts(scores, ["pod-x"], self.cfg)
        assert alerts[0]["auto_healable"] is True

        scores = torch.tensor([[0.76]])
        alerts = AnomalyHead.score_to_alerts(scores, ["pod-y"], self.cfg)
        assert alerts[0]["auto_healable"] is False
