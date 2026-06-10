"""Búsqueda por keywords (BM25) sobre el índice full-text de Neo4j."""

from __future__ import annotations

import re

from ragkg.graph.neo4j_client import Neo4jClient


# Lucene/Neo4j full-text usa una sintaxis con caracteres especiales que pueden
# romper la consulta. Los escapamos.
_LUCENE_SPECIAL_CHARS = r'+-&|!(){}[]^"~*?:\/'


def _sanitize_lucene_query(query: str) -> str:
    """Escapa caracteres reservados de Lucene y construye una consulta tolerante.

    Convierte 'Azure OpenAI con GPT-4' en 'Azure OpenAI con GPT\\-4' y
    además permite que cualquier término case (operador OR implícito).
    """
    if not query.strip():
        return ""
    # Escapar caracteres especiales
    escaped = "".join("\\" + c if c in _LUCENE_SPECIAL_CHARS else c for c in query)
    # Normalizar espacios
    escaped = re.sub(r"\s+", " ", escaped).strip()
    return escaped


def keyword_search(
    client: Neo4jClient,
    query: str,
    top_k: int = 10,
    index_name: str = "chunk_text",
) -> list[dict]:
    """Búsqueda BM25 sobre chunks usando el índice full-text de Neo4j.

    Devuelve la misma forma que `vector_search`: lista de dicts con
    chunk_id, text, doc_id, source_file, metadata y score.
    """
    sanitized = _sanitize_lucene_query(query)
    if not sanitized:
        return []

    cypher = """
    CALL db.index.fulltext.queryNodes($index_name, $query, {limit: $top_k})
    YIELD node, score
    RETURN node.chunk_id AS chunk_id,
           node.text AS text,
           node.doc_id AS doc_id,
           node.source_file AS source_file,
           node.metadata AS metadata,
           score
    ORDER BY score DESC
    """
    try:
        records = client.run(
            cypher,
            {"index_name": index_name, "query": sanitized, "top_k": top_k},
        )
    except Exception:
        # Si el índice no existe todavía o la query es inválida, devolvemos vacío
        # en lugar de romper la búsqueda híbrida.
        return []

    return [dict(r) for r in records]
