"""Helpers para consultar el índice vectorial de Neo4j."""

from __future__ import annotations

from ragkg.graph.neo4j_client import Neo4jClient


def query_vector_index(
    client: Neo4jClient,
    query_embedding: list[float],
    top_k: int = 10,
    index_name: str = "chunk_embedding",
) -> list[dict]:
    """Consulta directa al índice vectorial."""
    query = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
    YIELD node, score
    RETURN node.chunk_id AS chunk_id,
           node.text AS text,
           node.doc_id AS doc_id,
           node.source_file AS source_file,
           node.metadata AS metadata,
           score
    ORDER BY score DESC
    """
    records = client.run(
        query,
        {"index_name": index_name, "top_k": top_k, "query_embedding": query_embedding},
    )
    return [dict(r) for r in records]
