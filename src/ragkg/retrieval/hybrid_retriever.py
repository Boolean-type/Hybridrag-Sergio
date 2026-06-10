"""Recuperador híbrido: vector + keyword (BM25) + grafo estructurado.

Ejecuta tres estrategias en paralelo y fusiona sus resultados con RRF.
Una pregunta como "qué tecnologías usa la oferta" se resuelve combinando:

  1. Vector search: chunks semánticamente parecidos a la pregunta.
  2. Keyword (BM25): chunks que contienen las palabras de la pregunta.
  3. Structured query: si detecta intención, consulta directa al grafo
     (Cypher) y trae las entidades y sus chunks de evidencia.

Sobre los chunks ganadores, se expande por el grafo para enriquecer el
contexto con entidades relacionadas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ragkg.embeddings.embedder import Embedder
from ragkg.graph.neo4j_client import Neo4jClient
from ragkg.retrieval.graph_retriever import expand_from_chunk
from ragkg.retrieval.keyword_retriever import keyword_search
from ragkg.retrieval.rrf import reciprocal_rank_fusion
from ragkg.retrieval.structured_query import (
    StructuredIntent,
    detect_intent,
    get_chunks_mentioning_entities,
    query_entities_by_label,
)
from ragkg.retrieval.vector_retriever import vector_search


@dataclass
class HybridResult:
    query: str
    vector_results: list[dict[str, Any]]
    keyword_results: list[dict[str, Any]] = field(default_factory=list)
    fused_results: list[dict[str, Any]] = field(default_factory=list)
    graph_context: list[dict[str, Any]] = field(default_factory=list)
    structured_entities: list[dict[str, Any]] = field(default_factory=list)
    intent: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "fused_results": self.fused_results,
            "vector_results": self.vector_results,
            "keyword_results": self.keyword_results,
            "structured_entities": self.structured_entities,
            "graph_context": self.graph_context,
            "sources": self.sources,
        }


class HybridRetriever:
    """Coordina vector + BM25 + Cypher estructural + expansión por grafo."""

    def __init__(self, client: Neo4jClient, embedder: Embedder):
        self.client = client
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        expand_graph: bool = True,
        expand_top_n: int = 5,
        max_graph_results_per_chunk: int = 30,
        use_keyword: bool = True,
        use_structured: bool = True,
    ) -> HybridResult:
        """Ejecuta las tres búsquedas en paralelo y fusiona.

        Args:
            query: pregunta en lenguaje natural.
            top_k: tope por estrategia (vector y keyword).
            expand_graph: si True, expande por grafo desde los chunks ganadores.
            expand_top_n: cuántos chunks ganadores se usan para expandir.
            max_graph_results_per_chunk: límite de vecinos por chunk en la expansión.
            use_keyword: si True, ejecuta también la búsqueda BM25.
            use_structured: si True, intenta una consulta Cypher estructural cuando
                            detecta intención reconocible en la pregunta.
        """
        # --- 1. Búsqueda vectorial ---
        query_embedding = self.embedder.embed(query)
        vector_results = vector_search(self.client, query_embedding, top_k=top_k)

        # --- 2. Búsqueda por keywords (BM25) ---
        keyword_results: list[dict[str, Any]] = []
        if use_keyword:
            keyword_results = keyword_search(self.client, query, top_k=top_k)

        # --- 3. Consulta estructurada por intención ---
        intent: StructuredIntent | None = None
        structured_entities: list[dict[str, Any]] = []
        structured_chunks: list[dict[str, Any]] = []
        if use_structured:
            intent = detect_intent(query)
            if intent is not None:
                structured_entities = query_entities_by_label(
                    self.client, intent.target_label, limit=50
                )
                structured_chunks = get_chunks_mentioning_entities(
                    self.client, intent.target_label, limit=top_k
                )

        # --- Fusión de rankings con RRF ---
        rankings_to_fuse = [vector_results]
        if keyword_results:
            rankings_to_fuse.append(keyword_results)
        if structured_chunks:
            rankings_to_fuse.append(structured_chunks)

        fused_results = reciprocal_rank_fusion(
            rankings_to_fuse, id_field="chunk_id", top_n=top_k
        )

        # --- Expansión por grafo desde los chunks fusionados ---
        graph_context: list[dict[str, Any]] = []
        if expand_graph:
            seen: set[str] = set()
            for result in fused_results[:expand_top_n]:
                chunk_id = result["chunk_id"]
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                graph_context.extend(
                    expand_from_chunk(self.client, chunk_id, limit=max_graph_results_per_chunk)
                )

        sources = [
            {
                "chunk_id": r["chunk_id"],
                "source_file": r.get("source_file"),
                "rrf_score": r.get("rrf_score"),
                "sources": r.get("sources", []),
            }
            for r in fused_results
        ]

        return HybridResult(
            query=query,
            vector_results=vector_results,
            keyword_results=keyword_results,
            fused_results=fused_results,
            graph_context=graph_context,
            structured_entities=structured_entities,
            intent=intent.to_dict() if intent else None,
            sources=sources,
        )
