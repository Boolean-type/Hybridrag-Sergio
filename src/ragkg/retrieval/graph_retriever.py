"""Expansión en el grafo desde chunks o entidades."""

from __future__ import annotations

from ragkg.graph.neo4j_client import Neo4jClient


def expand_from_chunk(client: Neo4jClient, chunk_id: str, limit: int = 50) -> list[dict]:
    """Para un chunk, recupera las entidades que menciona y sus vecinos directos."""
    query = """
    MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(entity)
    OPTIONAL MATCH (entity)-[r]-(related)
    WHERE related <> c
    RETURN
        entity { .*, _label: head(labels(entity)) } AS entity,
        type(r) AS relation,
        related { .*, _label: head(labels(related)) } AS related
    LIMIT $limit
    """
    return [dict(r) for r in client.run(query, {"chunk_id": chunk_id, "limit": limit})]


def expand_from_entity(
    client: Neo4jClient,
    entity_label: str,
    id_field: str,
    id_value: str,
    max_hops: int = 1,
    limit: int = 50,
) -> list[dict]:
    """Expande desde una entidad concreta hasta `max_hops` saltos."""
    # max_hops limitado a 3 por seguridad/coste
    hops = max(1, min(max_hops, 3))
    query = f"""
    MATCH (e:{entity_label} {{{id_field}: $id_value}})-[r*1..{hops}]-(related)
    WHERE NOT related:Chunk AND NOT related:Document
    RETURN DISTINCT related {{ .*, _label: head(labels(related)) }} AS related
    LIMIT $limit
    """
    return [dict(r) for r in client.run(query, {"id_value": id_value, "limit": limit})]
