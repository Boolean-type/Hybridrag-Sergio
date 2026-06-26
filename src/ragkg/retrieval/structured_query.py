"""Detección de intención y consultas Cypher estructuradas.

Para preguntas tipo 'qué tecnologías usa X', 'qué ofertas cumplen Y', etc.,
es mucho más preciso consultar el grafo directamente que pasar por embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ragkg.graph.neo4j_client import Neo4jClient


# Patrones de intención: regex → (tipo_de_entidad_objetivo, etiqueta_legible).
# El orden importa: el primer match gana.
_INTENT_PATTERNS: list[tuple[str, str, str]] = [
    (r"\b(tecnolog[íi]as?|stack|herramientas?|servicios)\b", "Technology", "tecnologías"),
    (r"\b(certificaciones?|certificados?)\b", "Certification", "certificaciones"),
    (r"\b(normativas?|cumplimiento|complianc?e|ens|gdpr|rgpd|owasp)\b", "ComplianceFramework", "marcos normativos"),
    (r"\b(metodolog[íi]as?|m[ée]todos?\s+de\s+trabajo)\b", "Methodology", "metodologías"),
    (r"\b(roles?|perfiles?|puestos?)\b", "Role", "roles"),
    (r"\b(sectores?|industrias?)\b", "Sector", "sectores"),
    (r"\b(canales?\s+de\s+(?:atenci[óo]n|contacto|comunicaci[óo]n))\b", "ContactChannel", "canales de contacto"),
    (r"\b(SLAs?|ANS|niveles?\s+de\s+servicio|tiempo[s]?\s+de\s+respuesta)\b", "ServiceLevel", "niveles de servicio"),
    (r"\b(entregables?|deliverables?)\b", "Deliverable", "entregables"),
    (r"\b(fases?\s+del?\s+proyecto|etapas)\b", "ProjectPhase", "fases del proyecto"),
    (r"\b(requisitos?\s+funcionales?|RF\s*\d+)\b", "FunctionalRequirement", "requisitos funcionales"),
    (r"\b(ofertantes?|empresas?\s+que\s+presentan|proveedor(?:es)?)\b", "Bidder", "ofertantes"),
    (r"\b(clientes?|adjudicadores?|destinatarios?)\b", "Client", "clientes"),
    (r"\b(conceptos?\s+de\s+IA|RAG|fine.?tuning|embeddings?)\b", "AIConcept", "conceptos de IA"),
]


@dataclass
class StructuredIntent:
    target_label: str
    human_label: str
    matched_pattern: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_label": self.target_label,
            "human_label": self.human_label,
            "matched_pattern": self.matched_pattern,
        }


def detect_intent(question: str) -> StructuredIntent | None:
    """Devuelve la intención estructural detectada, o None si no aplica."""
    q = question.lower()
    for pattern, label, human in _INTENT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return StructuredIntent(target_label=label, human_label=human, matched_pattern=pattern)
    return None


def query_entities_by_label(
    client: Neo4jClient,
    label: str,
    limit: int = 50,
) -> list[dict]:
    """Devuelve todas las entidades de un label dado, ordenadas alfabéticamente.

    Cypher dinámico: el label viene de una whitelist (la ontología), por lo que
    no es inyección.
    """
    # Validación defensiva: solo permitimos labels alfanuméricos.
    if not label.replace("_", "").isalnum():
        return []

    cypher = f"""
    MATCH (n:{label})
    WHERE NOT n:Chunk AND NOT n:Document
    OPTIONAL MATCH (c:Chunk)-[m:MENTIONS]->(n)
    WITH n, collect(DISTINCT c.chunk_id) AS chunk_ids, collect(DISTINCT m.evidence)[..3] AS evidences
    RETURN coalesce(n.canonical_name, n.code, n.name, n.offer_id) AS name,
           labels(n)[0] AS label,
           properties(n) AS properties,
           chunk_ids,
           evidences
    ORDER BY name
    LIMIT $limit
    """
    try:
        records = client.run(cypher, {"limit": limit})
    except Exception:
        return []
    return [dict(r) for r in records]


def get_chunks_mentioning_entities(
    client: Neo4jClient,
    label: str,
    limit: int = 20,
) -> list[dict]:
    """Devuelve los chunks que mencionan entidades del label dado.

    Útil para complementar la búsqueda estructural con el texto evidencia.
    """
    if not label.replace("_", "").isalnum():
        return []

    cypher = f"""
    MATCH (c:Chunk)-[m:MENTIONS]->(n:{label})
    RETURN DISTINCT c.chunk_id AS chunk_id,
           c.text AS text,
           c.doc_id AS doc_id,
           c.source_file AS source_file,
           c.metadata AS metadata,
           m.confidence AS confidence
    ORDER BY m.confidence DESC
    LIMIT $limit
    """
    try:
        records = client.run(cypher, {"limit": limit})
    except Exception:
        return []
    return [dict(r) for r in records]
