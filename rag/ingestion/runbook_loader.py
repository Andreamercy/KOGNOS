"""
rag/ingestion/runbook_loader.py

Loads operational runbooks (Markdown) and past incident reports into a
Qdrant vector store via LlamaIndex. Run once (or when runbooks are updated)
to build / refresh the RAG knowledge base.

Usage:
    python -m rag.ingestion.runbook_loader \\
        --runbooks docs/runbooks/ \\
        --incidents docs/incidents/
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def build_knowledge_base(
    runbooks_dir: str,
    incidents_dir: str,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    collection: str | None = None,
) -> None:
    """
    Load runbooks and incident reports into Qdrant via LlamaIndex.

    Args:
        runbooks_dir:  Path to directory of Markdown runbooks.
        incidents_dir: Path to directory of incident reports.
        qdrant_host:   Qdrant host (defaults to env QDRANT_HOST or localhost).
        qdrant_port:   Qdrant port (defaults to env QDRANT_PORT or 6333).
        collection:    Collection name (defaults to env QDRANT_COLLECTION).
    """
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ImportError as e:
        raise ImportError(
            f"Missing dependency: {e}\n"
            "Run: pip install llama-index-core llama-index-vector-stores-qdrant "
            "llama-index-embeddings-huggingface qdrant-client"
        ) from e

    host       = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
    port       = qdrant_port or int(os.getenv("QDRANT_PORT", "6333"))
    coll_name  = collection  or os.getenv("QDRANT_COLLECTION", "kognos_knowledge")

    # ── Configure embedding model ─────────────────────────────────────────
    # Using a local HuggingFace model avoids requiring an additional API key.
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.embed_model = embed_model
    EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

    # ── Connect to Qdrant ────────────────────────────────────────────────
    client = QdrantClient(host=host, port=port)
    logger.info("Connected to Qdrant at %s:%d", host, port)

    # Create collection if it doesn't exist
    existing = {c.name for c in client.get_collections().collections}
    if coll_name not in existing:
        client.create_collection(
            collection_name=coll_name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", coll_name)
    else:
        logger.info("Using existing Qdrant collection: %s", coll_name)

    vector_store = QdrantVectorStore(client=client, collection_name=coll_name)

    # ── Load documents ───────────────────────────────────────────────────
    docs = []

    for directory, label in [(runbooks_dir, "runbook"), (incidents_dir, "incident")]:
        p = Path(directory)
        if not p.exists():
            logger.warning("Directory not found, skipping: %s", directory)
            continue

        dir_docs = SimpleDirectoryReader(
            str(p),
            recursive=True,
            required_exts=[".md", ".txt", ".json"],
        ).load_data()

        # Tag each document with its source type for retrieval filtering
        for doc in dir_docs:
            doc.metadata["source_type"] = label
            doc.metadata["source_path"] = str(doc.metadata.get("file_path", ""))

        docs.extend(dir_docs)
        logger.info("Loaded %d documents from %s (%s)", len(dir_docs), directory, label)

    if not docs:
        logger.error("No documents found. Check that runbooks/incidents directories exist.")
        return

    # ── Build index ──────────────────────────────────────────────────────
    logger.info("Building vector index for %d documents...", len(docs))
    VectorStoreIndex.from_documents(
        docs,
        vector_store=vector_store,
        show_progress=True,
    )
    logger.info("✅ Knowledge base built successfully (%d docs indexed)", len(docs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build KOGNOS knowledge base")
    parser.add_argument("--runbooks",  required=True, help="Path to runbooks directory")
    parser.add_argument("--incidents", required=True, help="Path to incidents directory")
    parser.add_argument("--qdrant-host", default=None)
    parser.add_argument("--qdrant-port", type=int, default=None)
    parser.add_argument("--collection",  default=None)
    args = parser.parse_args()

    build_knowledge_base(
        runbooks_dir=args.runbooks,
        incidents_dir=args.incidents,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection=args.collection,
    )
