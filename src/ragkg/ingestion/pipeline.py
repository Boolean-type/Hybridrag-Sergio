"""Orquestación de la ingesta: carga → chunking → embedding → upsert a Neo4j."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from ragkg.config.loader import DomainConfig
from ragkg.ingestion.chunker import Chunk, chunk_document
from ragkg.ingestion.loaders import Document, load_document


class GraphClient(Protocol):
    def run(self, query: str, parameters: dict | None = None) -> list: ...


class EmbedderProtocol(Protocol):
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]: ...


def ingest_path(
    path: str | Path,
    config: DomainConfig,
    client: GraphClient,
    embedder: EmbedderProtocol,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> tuple[Document, list[Chunk]]:
    """
    Carga un documento, lo trocea, genera embeddings y lo persiste en Neo4j.

    Devuelve el documento y la lista de chunks. La extracción de entidades se
    hace en una fase separada (ver ragkg.extraction).
    """
    # Import local para evitar ciclos y dependencia obligatoria en tests.
    from ragkg.graph.upsert import (
        link_document_to_chunk,
        upsert_chunk,
        upsert_document,
    )

    document = load_document(path)

    # 1. Upsert del Document
    upsert_document(
        client,
        doc_id=document.doc_id,
        metadata={
            **document.metadata,
            "ingestion_date": datetime.now(timezone.utc).isoformat(),
            "domain": config.domain_name,
        },
    )

    # 2. Chunking
    chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return document, []

    # 3. Embeddings (en lote para eficiencia)
    embeddings = embedder.embed_batch([c.text for c in chunks])

    # 4. Upsert de chunks + relación HAS_CHUNK
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        upsert_chunk(client, chunk, embedding)
        link_document_to_chunk(
            client,
            doc_id=document.doc_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.metadata.get("chunk_index", 0),
        )

    return document, chunks
