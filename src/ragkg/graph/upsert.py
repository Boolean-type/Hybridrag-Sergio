"""Operaciones de upsert agnósticas para nodos y relaciones."""

from __future__ import annotations

from typing import Any

from ragkg.graph.neo4j_client import Neo4jClient


# Labels y relaciones se validan contra una whitelist construida desde la config
# para evitar inyección de Cypher dinámico.
_SAFE_TOKEN = lambda s: bool(s) and s.replace("_", "").isalnum()


def _ensure_safe(label_or_rel: str) -> None:
    if not _SAFE_TOKEN(label_or_rel):
        raise ValueError(f"Identificador no seguro para Cypher: '{label_or_rel}'")


# Neo4j solo acepta primitivos (str, int, float, bool) y arrays homogéneos de
# primitivos como valor de propiedad. Esta función limpia un dict para garantizarlo.
_PRIMITIVE_TYPES = (str, int, float, bool)


def _flatten_to_primitives(d: dict[str, Any]) -> dict[str, Any]:
    """Devuelve solo las entradas del dict cuyo valor es un primitivo
    o un array homogéneo de primitivos. Descarta None y estructuras anidadas."""
    result: dict[str, Any] = {}
    for key, value in (d or {}).items():
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            result[key] = value
        elif isinstance(value, list) and value and all(
            isinstance(x, _PRIMITIVE_TYPES) for x in value
        ):
            result[key] = value
        # Cualquier otro tipo (dict, set, objeto custom, lista heterogénea) se omite.
    return result


# ---------------------------------------------------------------- documentos & chunks


def upsert_document(client: Neo4jClient, doc_id: str, metadata: dict[str, Any]) -> None:
    """Crea o actualiza un nodo Document.

    El metadata se filtra a primitivos antes de pasarlo a Neo4j, porque Neo4j
    no admite dicts anidados como valor de propiedad.
    """
    query = """
    MERGE (d:Document {doc_id: $doc_id})
    SET d += $metadata
    """
    client.run(query, {"doc_id": doc_id, "metadata": _flatten_to_primitives(metadata)})


def upsert_chunk(client: Neo4jClient, chunk, embedding: list[float]) -> None:
    """Crea o actualiza un nodo Chunk con su embedding.

    El metadata del chunk se aplana en propiedades individuales del nodo
    (Neo4j no admite Map como valor de propiedad). Esto además hace cada campo
    consultable desde Cypher (`MATCH (c:Chunk) WHERE c.page = 4 ...`).
    """
    # Empezamos con el metadata aplanado y añadimos/sobrescribimos los campos
    # explícitos. El embedding es list[float], que sí es un array de primitivos.
    properties = _flatten_to_primitives(chunk.metadata)
    properties.update(
        {
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "source_file": chunk.metadata.get("source_file"),
            "embedding": embedding,
        }
    )

    query = """
    MERGE (c:Chunk {chunk_id: $chunk_id})
    SET c += $properties
    """
    client.run(query, {"chunk_id": chunk.chunk_id, "properties": properties})


def link_document_to_chunk(client: Neo4jClient, doc_id: str, chunk_id: str, chunk_index: int) -> None:
    """Crea la relación Document-[:HAS_CHUNK]->Chunk."""
    query = """
    MATCH (d:Document {doc_id: $doc_id})
    MATCH (c:Chunk {chunk_id: $chunk_id})
    MERGE (d)-[r:HAS_CHUNK]->(c)
    SET r.chunk_index = $chunk_index
    """
    client.run(query, {"doc_id": doc_id, "chunk_id": chunk_id, "chunk_index": chunk_index})


# ---------------------------------------------------------------- entidades & relaciones genéricas


def upsert_entity(
    client: Neo4jClient,
    entity_type: str,
    id_field: str,
    id_value: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Crea o actualiza un nodo de entidad de cualquier tipo (label dinámico)."""
    _ensure_safe(entity_type)
    _ensure_safe(id_field)
    props = properties or {}
    query = f"""
    MERGE (e:{entity_type} {{{id_field}: $id_value}})
    SET e += $properties
    """
    client.run(query, {"id_value": id_value, "properties": props})


def upsert_relation(
    client: Neo4jClient,
    source_label: str,
    source_id_field: str,
    source_id: str,
    relation_type: str,
    target_label: str,
    target_id_field: str,
    target_id: str,
    properties: dict[str, Any] | None = None,
) -> None:
    """Crea o actualiza una relación entre dos nodos cualesquiera."""
    for token in (source_label, source_id_field, relation_type, target_label, target_id_field):
        _ensure_safe(token)
    props = properties or {}
    query = f"""
    MERGE (s:{source_label} {{{source_id_field}: $source_id}})
    MERGE (t:{target_label} {{{target_id_field}: $target_id}})
    MERGE (s)-[r:{relation_type}]->(t)
    SET r += $properties
    """
    client.run(query, {"source_id": source_id, "target_id": target_id, "properties": props})


def link_chunk_mentions_entity(
    client: Neo4jClient,
    chunk_id: str,
    entity_label: str,
    entity_id_field: str,
    entity_id: str,
    evidence: str,
    confidence: float,
    char_offset_start: int | None = None,
    char_offset_end: int | None = None,
) -> None:
    """Crea la relación Chunk-[:MENTIONS]->Entidad con evidencia."""
    _ensure_safe(entity_label)
    _ensure_safe(entity_id_field)
    query = f"""
    MATCH (c:Chunk {{chunk_id: $chunk_id}})
    MERGE (e:{entity_label} {{{entity_id_field}: $entity_id}})
    MERGE (c)-[r:MENTIONS]->(e)
    SET r.evidence = $evidence,
        r.confidence = $confidence,
        r.char_offset_start = $char_offset_start,
        r.char_offset_end = $char_offset_end
    """
    client.run(
        query,
        {
            "chunk_id": chunk_id,
            "entity_id": entity_id,
            "evidence": evidence,
            "confidence": confidence,
            "char_offset_start": char_offset_start,
            "char_offset_end": char_offset_end,
        },
    )
