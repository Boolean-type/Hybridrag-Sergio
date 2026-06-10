"""Reciprocal Rank Fusion (RRF): fusión de múltiples rankings.

Es el algoritmo estándar (Cormack, Clarke, Buettcher 2009) para combinar
listas ordenadas de documentos provenientes de fuentes distintas. Funciona
muy bien sin necesidad de normalizar scores: solo usa el rango (posición).

Fórmula: RRF(d) = Σ 1/(k + rank_i(d)) para cada fuente i donde aparece d.
k=60 es el default sugerido en el paper.
"""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    id_field: str = "chunk_id",
    k: int = 60,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Fusiona múltiples rankings en uno solo.

    Args:
        rankings: lista de rankings; cada ranking es una lista de dicts ordenada
                  por relevancia descendente, donde cada dict debe contener `id_field`.
        id_field: clave que identifica cada elemento (chunk_id, canonical_name...).
        k: constante de suavizado (60 es el valor estándar del paper).
        top_n: si se da, limita la salida a los top_n mejores.

    Returns:
        Lista combinada ordenada por puntuación RRF descendente. Cada elemento
        conserva los campos del original (toma el primer dict que vio para cada id)
        y añade `rrf_score`, `sources` (de qué rankings vino) y `ranks` (posición en cada uno).
    """
    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[int]] = {}
    ranks_per_source: dict[str, dict[int, int]] = {}

    for source_idx, ranking in enumerate(rankings):
        for rank, item in enumerate(ranking, start=1):
            item_id = item.get(id_field)
            if item_id is None:
                continue
            score = 1.0 / (k + rank)
            scores[item_id] = scores.get(item_id, 0.0) + score
            if item_id not in items:
                items[item_id] = dict(item)
            sources.setdefault(item_id, []).append(source_idx)
            ranks_per_source.setdefault(item_id, {})[source_idx] = rank

    # Ordenar por puntuación RRF descendente
    ordered_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    result = []
    for item_id in ordered_ids:
        merged = items[item_id]
        merged["rrf_score"] = round(scores[item_id], 6)
        merged["sources"] = sources[item_id]
        merged["ranks"] = ranks_per_source[item_id]
        result.append(merged)

    if top_n is not None:
        result = result[:top_n]
    return result
