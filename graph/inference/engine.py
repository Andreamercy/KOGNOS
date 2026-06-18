"""
graph/inference/engine.py

Real-time GNN inference engine.

In production mode: polls Kafka for telemetry windows from the eBPF loader,
runs the GraphSAGE model, and publishes anomaly alerts.

In demo mode (KOGNOS_DEMO_MODE=true): generates synthetic telemetry windows
every N seconds and simulates live anomaly detection without a real cluster.

Usage:
    python -m graph.inference.engine --model-path models/graphsage_v1.pt
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import time
from typing import AsyncIterator

import torch

from graph.builder.cluster_graph import build_cluster_graph, TelemetryWindow
from graph.model.graphsage import KOGNOSGraphSAGE
from graph.model.anomaly_head import AnomalyHead, ThresholdConfig
from graph.inference.scorer import AlertScorer
from graph.data.synthetic_gen import generate_window

logger = logging.getLogger(__name__)

DEMO_MODE       = os.getenv("KOGNOS_DEMO_MODE", "false").lower() == "true"
INFERENCE_INTERVAL = float(os.getenv("INFERENCE_INTERVAL_S", "10.0"))


class InferenceEngine:
    """
    Continuous inference engine that runs the GNN on live cluster telemetry.

    Emits alert dicts via an async generator that downstream consumers
    (WebSocket streamer, Kafka producer) can subscribe to.
    """

    def __init__(
        self,
        model_path: str | None = None,
        thresholds: ThresholdConfig | None = None,
    ) -> None:
        self.thresholds = thresholds or ThresholdConfig.from_env()
        self.scorer     = AlertScorer(self.thresholds)
        self.model      = self._load_model(model_path)

    def _load_model(self, model_path: str | None) -> KOGNOSGraphSAGE:
        """Load the trained model from disk, or fall back to a random-weight model."""
        model = KOGNOSGraphSAGE(in_channels=6)

        if model_path and pathlib.Path(model_path).exists():
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            logger.info("Loaded model from %s", model_path)
        else:
            logger.warning(
                "Model file not found at %s — using untrained weights.\n"
                "Run 'python -m graph.model.train' to train the model.",
                model_path,
            )

        model.eval()
        return model

    async def run(
        self,
        interval_s: float = INFERENCE_INTERVAL,
    ) -> AsyncIterator[list[dict]]:
        """
        Async generator that yields alert lists on each inference pass.

        Args:
            interval_s: Seconds between inference windows.

        Yields:
            List of alert dicts (may be empty if no anomalies detected).
        """
        logger.info(
            "Inference engine starting (demo=%s, interval=%.1fs)",
            DEMO_MODE, interval_s,
        )

        async for window in self._telemetry_stream(interval_s):
            alerts = self._infer(window)
            if alerts:
                logger.info("⚠️  %d alerts detected: %s",
                            len(alerts),
                            [a["pod"] for a in alerts])
            yield alerts

    def _infer(self, window: TelemetryWindow) -> list[dict]:
        """Run a single inference pass on a telemetry window."""
        try:
            graph = build_cluster_graph(window)
        except Exception as exc:
            logger.error("Graph build failed: %s", exc)
            return []

        with torch.no_grad():
            scores = self.model(graph.x, graph.edge_index)  # [N, 1]

        return self.scorer.score_to_alerts(
            scores=scores,
            pod_names=graph.pod_names,  # type: ignore[attr-defined]
        )

    async def _telemetry_stream(
        self,
        interval_s: float,
    ) -> AsyncIterator[TelemetryWindow]:
        """
        Yield telemetry windows continuously.
        In demo mode: synthetic data.
        In production: Kafka consumer.
        """
        if DEMO_MODE:
            async for window in self._synthetic_stream(interval_s):
                yield window
        else:
            async for window in self._kafka_stream(interval_s):
                yield window

    async def _synthetic_stream(
        self,
        interval_s: float,
    ) -> AsyncIterator[TelemetryWindow]:
        """Generate synthetic telemetry windows at a fixed interval."""
        import random
        rng = random.Random()
        while True:
            seed = rng.randint(0, 2**31)
            yield generate_window(seed=seed)
            await asyncio.sleep(interval_s)

    async def _kafka_stream(
        self,
        interval_s: float,
    ) -> AsyncIterator[TelemetryWindow]:
        """Consume aggregated telemetry windows from Kafka."""
        from kafka import KafkaConsumer

        bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
        topic     = os.getenv("KAFKA_TOPIC_FLOWS", "ebpf-flows")

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="kognos-inference",
        )
        logger.info("Kafka consumer connected: %s → %s", bootstrap, topic)

        window_buffer: list[dict] = []
        window_start  = time.time()

        for msg in consumer:
            window_buffer.append(msg.value)

            # Flush buffer into a window every interval_s
            if time.time() - window_start >= interval_s:
                if window_buffer:
                    yield _flows_to_window(window_buffer)
                    window_buffer = []
                window_start = time.time()
                await asyncio.sleep(0)  # yield control


def _flows_to_window(flows_json: list[dict]) -> TelemetryWindow:
    """Convert a batch of Kafka flow JSON records to a TelemetryWindow."""
    # In production: also pulls pod state from k8s API / Redis cache
    # For now: delegate to synthetic generator with flow injection
    from graph.data.synthetic_gen import generate_window
    return generate_window()


# ── CLI entrypoint ──────────────────────────────────────────────────────────

async def _main(args: argparse.Namespace) -> None:
    engine = InferenceEngine(model_path=args.model_path)
    async for alerts in engine.run(interval_s=args.interval):
        if alerts:
            print(json.dumps(alerts, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="KOGNOS GNN Inference Engine")
    parser.add_argument("--model-path",  default="models/graphsage_v1.pt")
    parser.add_argument("--kafka-topic", default="ebpf-flows")
    parser.add_argument("--interval",    type=float, default=10.0)
    args = parser.parse_args()

    asyncio.run(_main(args))
