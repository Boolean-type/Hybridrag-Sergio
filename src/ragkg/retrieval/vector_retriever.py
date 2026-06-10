"""Búsqueda por similitud semántica en el índice vectorial de Neo4j."""

from __future__ import annotations

from ragkg.embeddings.vector_index import query_vector_index
from ragkg.graph.neo4j_client import Neo4jClient


def vector_search(
    client: Neo4jClient,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict]:
    """Devuelve los `top_k` chunks más similares al embedding de la query."""
    return query_vector_index(client, query_embedding, top_k=top_k)
