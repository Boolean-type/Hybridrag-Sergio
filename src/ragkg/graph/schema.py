"""Creación de constraints e índice vectorial en Neo4j."""

from __future__ import annotations

import os

from ragkg.config.loader import DomainConfig
from ragkg.graph.neo4j_client import Neo4jClient


def create_constraints(client: Neo4jClient, config: DomainConfig) -> None:
    """Crea constraints de unicidad para todas las entidades del dominio."""
    for entity_name, entity_def in config.ontology["entities"].items():
        label = entity_def.get("label", entity_name)
        for id_field in entity_def.get("id_fields", []):
            constraint_name = f"{label.lower()}_{id_field}_unique"
            query = f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{label})
            REQUIRE n.{id_field} IS UNIQUE
            """
            client.run(query)


def create_vector_index(
    client: Neo4jClient,
    dimensions: int | None = None,
    similarity: str = "cosine",
    index_name: str = "chunk_embedding",
    label: str = "Chunk",
    property_name: str = "embedding",
) -> None:
    """Crea el índice vectorial sobre Chunk.embedding."""
    dim = dimensions or int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
    query = f"""
    CREATE VECTOR INDEX {index_name} IF NOT EXISTS
    FOR (c:{label})
    ON (c.{property_name})
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: '{similarity}'
        }}
    }}
    """
    client.run(query)


def create_fulltext_index(
    client: Neo4jClient,
    index_name: str = "chunk_text",
    label: str = "Chunk",
    property_name: str = "text",
) -> None:
    """Crea el índice de texto completo para búsqueda BM25 sobre Chunk.text.

    Permite consultas tipo `CALL db.index.fulltext.queryNodes(...)`.
    """
    query = f"""
    CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
    FOR (c:{label})
    ON EACH [c.{property_name}]
    """
    client.run(query)


def create_schema(client: Neo4jClient, config: DomainConfig) -> None:
    """Conveniencia: constraints + índice vectorial + índice full-text."""
    create_constraints(client, config)
    create_vector_index(client)
    create_fulltext_index(client)
