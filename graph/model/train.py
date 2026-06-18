"""
graph/model/train.py

Training loop for the KOGNOS GraphSAGE anomaly detection model.

Trains on a synthetic dataset of labelled cluster failure scenarios.
Saves the best checkpoint to models/graphsage_v1.pt.

Usage:
    python -m graph.model.train [--epochs 50] [--lr 1e-3] [--batch 32]
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import time
from typing import Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader

from graph.builder.cluster_graph import build_cluster_graph
from graph.data.synthetic_gen import generate_labelled_window, FailureScenario
from graph.model.graphsage import KOGNOSGraphSAGE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_pyg_dataset(n_samples: int = 1000, seed: int = 42) -> list[Data]:
    """
    Generate n_samples labelled telemetry windows and convert each to a
    PyG Data object with a .y label tensor for training.
    """
    import random
    rng = random.Random(seed)
    dataset: list[Data] = []

    for i in range(n_samples):
        lw = generate_labelled_window(seed=rng.randint(0, 2**31))
        graph = build_cluster_graph(lw.window)
        graph.y = torch.tensor(lw.labels, dtype=torch.float).unsqueeze(-1)  # [N, 1]
        dataset.append(graph)

        if (i + 1) % 100 == 0:
            logger.info("Generated %d / %d training samples", i + 1, n_samples)

    return dataset


def train(
    epochs: int = 50,
    lr: float = 1e-3,
    n_train: int = 800,
    n_val: int = 200,
    hidden: int = 128,
    out_channels: int = 64,
    dropout: float = 0.3,
    model_dir: str = "models",
    seed: int = 42,
) -> KOGNOSGraphSAGE:
    """Full training loop with validation and best-checkpoint saving."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # ── Dataset ─────────────────────────────────────────────────────────────
    logger.info("Generating training dataset (%d samples)...", n_train + n_val)
    all_data  = build_pyg_dataset(n_samples=n_train + n_val, seed=seed)
    train_set = all_data[:n_train]
    val_set   = all_data[n_train:]

    # ── Model ────────────────────────────────────────────────────────────────
    model = KOGNOSGraphSAGE(
        in_channels=6,
        hidden=hidden,
        out_channels=out_channels,
        dropout=dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    model_path    = pathlib.Path(model_dir) / "graphsage_v1.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for graph in train_set:
            graph = graph.to(device)
            optimizer.zero_grad()
            preds = model(graph.x, graph.edge_index)
            loss  = criterion(preds, graph.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_set)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for graph in val_set:
                graph = graph.to(device)
                preds = model(graph.x, graph.edge_index)
                val_loss += criterion(preds, graph.y).item()
        avg_val_loss = val_loss / len(val_set)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e  %.1fs",
            epoch, epochs, avg_train_loss, avg_val_loss,
            scheduler.get_last_lr()[0], elapsed,
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            logger.info("  ✅ Saved best model → %s (val_loss=%.4f)",
                        model_path, best_val_loss)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    # Reload best weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KOGNOS GraphSAGE model")
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--n-train",  type=int,   default=800)
    parser.add_argument("--n-val",    type=int,   default=200)
    parser.add_argument("--hidden",   type=int,   default=128)
    parser.add_argument("--model-dir", type=str,  default="models")
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        n_train=args.n_train,
        n_val=args.n_val,
        hidden=args.hidden,
        model_dir=args.model_dir,
        seed=args.seed,
    )
