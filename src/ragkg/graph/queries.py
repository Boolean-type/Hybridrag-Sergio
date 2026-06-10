"""Consultas Cypher reutilizables para el dominio offers."""

from __future__ import annotations

from ragkg.graph.neo4j_client import Neo4jClient


def filter_by_technologies(client: Neo4jClient, technologies: list[str]) -> list[dict]:
    """Ofertas que requieren TODAS o ALGUNAS de las tecnologías indicadas."""
    query = """
    MATCH (o:Offer)-[:REQUIRES_TECHNOLOGY]->(t:Technology)
    WHERE t.canonical_name IN $technologies
    WITH o, collect(DISTINCT t.canonical_name) AS matched
    RETURN o.offer_id AS offer_id,
           o.title AS title,
           o.client AS client,
           o.sector AS sector,
           matched AS matched_technologies,
           size(matched) AS match_count
    ORDER BY match_count DESC
    """
    return [dict(r) for r in client.run(query, {"technologies": technologies})]


def filter_by_sector_and_tech(
    client: Neo4jClient,
    sector: str,
    technologies: list[str],
) -> list[dict]:
    """Ofertas filtradas por sector + tecnologías."""
    query = """
    MATCH (o:Offer)-[:BELONGS_TO_SECTOR]->(s:Sector {canonical_name: $sector})
    MATCH (o)-[:REQUIRES_TECHNOLOGY]->(t:Technology)
    WHERE t.canonical_name IN $technologies
    WITH o, s, collect(DISTINCT t.canonical_name) AS matched
    RETURN o.offer_id AS offer_id,
           o.title AS title,
           s.canonical_name AS sector,
           matched AS matched_technologies
    ORDER BY size(matched) DESC
    """
    return [dict(r) for r in client.run(query, {"sector": sector, "technologies": technologies})]


def get_technology_frequency(client: Neo4jClient, limit: int = 20) -> list[dict]:
    """Tecnologías más frecuentes en ofertas."""
    query = """
    MATCH (o:Offer)-[:REQUIRES_TECHNOLOGY]->(t:Technology)
    RETURN t.canonical_name AS technology,
           t.category AS category,
           count(DISTINCT o) AS offer_count
    ORDER BY offer_count DESC
    LIMIT $limit
    """
    return [dict(r) for r in client.run(query, {"limit": limit})]


def get_requirements_for_role(client: Neo4jClient, role: str) -> list[dict]:
    """Requisitos asociados a un rol."""
    query = """
    MATCH (r:Role {canonical_name: $role})<-[:HAS_ROLE]-(o:Offer)-[:HAS_REQUIREMENT]->
          (req:TechnicalRequirement)
    RETURN o.offer_id AS offer_id,
           o.title AS offer_title,
           req.canonical_name AS requirement,
           req.raw_text AS raw_text
    ORDER BY o.offer_id
    """
    return [dict(r) for r in client.run(query, {"role": role})]


def get_certifications_by_sector(client: Neo4jClient, sector: str) -> list[dict]:
    """Certificaciones demandadas en un sector."""
    query = """
    MATCH (o:Offer)-[:BELONGS_TO_SECTOR]->(:Sector {canonical_name: $sector})
    MATCH (o)-[:REQUIRES_CERTIFICATION]->(c:Certification)
    RETURN c.canonical_name AS certification,
           c.provider AS provider,
           count(DISTINCT o) AS demand
    ORDER BY demand DESC
    """
    return [dict(r) for r in client.run(query, {"sector": sector})]


def expand_chunk_neighbors(client: Neo4jClient, chunk_id: str, limit: int = 50) -> list[dict]:
    """Entidades mencionadas por un chunk y sus vecinos directos."""
    query = """
    MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(entity)
    OPTIONAL MATCH (entity)-[r]-(related)
    RETURN entity { .*, _label: head(labels(entity)) } AS entity,
           type(r) AS relation,
           related { .*, _label: head(labels(related)) } AS related
    LIMIT $limit
    """
    return [dict(r) for r in client.run(query, {"chunk_id": chunk_id, "limit": limit})]
