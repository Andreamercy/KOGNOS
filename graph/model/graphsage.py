"""
graph/model/graphsage.py

KOGNOS GraphSAGE model — an inductive Graph Neural Network for Kubernetes
pod anomaly detection.

Key design decisions:
  - Inductive (GraphSAGE) rather than transductive (GCN) so the model
    generalises to new pods not seen during training.
  - Two SAGEConv message-passing layers with ReLU activation.
  - Separate anomaly head outputs a per-node probability in [0, 1].
  - Edge features are not directly consumed by SAGEConv but are used by
    the anomaly head via edge-weighted aggregation (future extension).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class KOGNOSGraphSAGE(nn.Module):
    """
    Inductive GNN for pod-level anomaly scoring.

    Architecture:
        Input → SAGEConv(hidden) → ReLU → Dropout
              → SAGEConv(out)   → ReLU
              → AnomalyHead     → Sigmoid → [0, 1] per node

    Args:
        in_channels:  Number of input node features (default 6).
        hidden:       Hidden dimension of first SAGE layer (default 128).
        out_channels: Output dimension of second SAGE layer (default 64).
        dropout:      Dropout rate applied between layers (default 0.3).
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden: int = 128,
        out_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if not HAS_PYG:
            raise ImportError(
                "torch_geometric is required. Install with:\n"
                "  pip install torch-geometric torch-scatter torch-sparse "
                "-f https://data.pyg.org/whl/torch-2.3.0+cpu.html"
            )

        self.conv1   = SAGEConv(in_channels, hidden, aggr="mean")
        self.conv2   = SAGEConv(hidden, out_channels, aggr="mean")
        self.dropout = nn.Dropout(p=dropout)

        # Anomaly scoring head: [out_channels] → [1] probability
        self.anomaly_head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # output: anomaly probability ∈ [0, 1]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:          Node feature matrix [N, in_channels]
            edge_index: COO edge list       [2, E]
            edge_attr:  Edge features       [E, F] (optional, future use)

        Returns:
            anomaly_scores: Per-node anomaly probability [N, 1]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        return self.anomaly_head(x)  # [N, 1]

    def embed(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return node embeddings (before anomaly head) for analysis."""
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        return F.relu(self.conv2(x, edge_index))  # [N, out_channels]

    @torch.no_grad()
    def score(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> list[float]:
        """Convenience method: run inference and return flat list of scores."""
        self.eval()
        scores = self(x, edge_index)
        return scores.squeeze(-1).tolist()
