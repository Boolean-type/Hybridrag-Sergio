"""Construcción de respuesta con fuentes y evidencia a partir del HybridResult."""

from __future__ import annotations

from typing import Any

from ragkg.retrieval.hybrid_retriever import HybridResult


def _chunk_score_line(r: dict[str, Any], idx: int) -> str:
    """Línea con info de score para un chunk fusionado."""
    rrf = r.get("rrf_score")
    sources = r.get("sources", [])
    source_names = {0: "vec", 1: "key", 2: "struct"}
    src = ",".join(source_names.get(s, f"s{s}") for s in sources)
    src_file = (r.get("source_file") or "").split("/")[-1].split("\\")[-1]
    head = f"[Chunk {idx + 1} | rrf={rrf:.4f} | from={src} | {src_file}]" if rrf else f"[Chunk {idx + 1} | {src_file}]"
    return head


def build_context_prompt(result: HybridResult, max_chunks: int = 5) -> str:
    """Construye un prompt rico con chunks + entidades del grafo para enviarlo a un LLM.

    Prioriza los chunks fusionados (RRF) sobre los solo-vectoriales.
    Si hubo consulta estructural exitosa, añade el listado de entidades a la pregunta.
    """
    # Usamos fused_results si existen; si no, caemos a vector puros (retrocompatibilidad).
    primary = result.fused_results or result.vector_results

    chunks_block = "\n\n".join(
        f"{_chunk_score_line(r, i)}\n{r.get('text', '')}"
        for i, r in enumerate(primary[:max_chunks])
    )

    # --- Bloque de entidades del grafo (expansión) ---
    entities_seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[str]] = {}
    for item in result.graph_context:
        entity = item.get("entity") or {}
        label = entity.get("_label")
        name = entity.get("canonical_name") or entity.get("name") or entity.get("offer_id")
        if not label or not name or (label, name) in entities_seen:
            continue
        entities_seen.add((label, name))
        grouped.setdefault(label, []).append(name)

    graph_lines = [f"- {label}: {', '.join(sorted(names))}" for label, names in sorted(grouped.items())]
    graph_block = "\n".join(graph_lines) if graph_lines else "(sin contexto de grafo)"

    # --- Bloque de respuesta estructural directa (si hubo intent detectada) ---
    structured_block = ""
    if result.intent and result.structured_entities:
        intent_label = result.intent.get("human_label", "elementos")
        names = [e.get("name") for e in result.structured_entities if e.get("name")]
        if names:
            structured_block = (
                f"\n## Respuesta estructural directa del grafo ({intent_label})\n"
                f"El grafo contiene los siguientes {intent_label} extraídos directamente:\n"
                + "\n".join(f"- {n}" for n in names[:30])
                + "\n"
            )

    return (
        f"## Pregunta del usuario\n{result.query}\n"
        f"{structured_block}"
        f"\n## Fragmentos relevantes\n{chunks_block}\n\n"
        f"## Entidades relacionadas en el grafo\n{graph_block}\n\n"
        "## Instrucciones\n"
        "Responde a la pregunta usando SOLO la información anterior. "
        "Si hay 'respuesta estructural directa del grafo', PRIORÍZALA porque viene de "
        "extracción estructurada validada. Cita los chunks usados como evidencia."
    )


def build_answer_summary(result: HybridResult) -> dict[str, Any]:
    """Resumen estructurado del resultado, útil para CLIs o APIs."""
    primary = result.fused_results or result.vector_results

    return {
        "query": result.query,
        "intent": result.intent,
        "top_chunks": [
            {
                "chunk_id": r["chunk_id"],
                "rrf_score": r.get("rrf_score"),
                "sources": r.get("sources", []),
                "source_file": r.get("source_file"),
                "preview": (r.get("text") or "")[:200],
            }
            for r in primary[:5]
        ],
        "structured_entities": [
            {"label": e.get("label"), "name": e.get("name")}
            for e in result.structured_entities[:30]
        ],
        "graph_entities": [
            {
                "label": (item.get("entity") or {}).get("_label"),
                "name": (item.get("entity") or {}).get("canonical_name")
                or (item.get("entity") or {}).get("name"),
            }
            for item in result.graph_context
            if item.get("entity")
        ],
        "sources": result.sources,
    }
