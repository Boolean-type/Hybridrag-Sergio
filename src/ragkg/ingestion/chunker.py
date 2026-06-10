"""Fragmentación de documentos en chunks con solapamiento."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ragkg.ingestion.loaders import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Heurísticas para detectar chunks sin valor semántico.
# Si un chunk encaja con estos patrones, se descarta antes de embedding/extracción.
_LOW_VALUE_PATTERNS = [
    # Más del 40% son dígitos, puntos y espacios → probable índice ("3.1.2. Servicio ... 33")
    (lambda t: sum(c.isdigit() or c in ". " for c in t) / max(len(t), 1) > 0.4, "índice o ToC"),
    # Demasiados puntos suspensivos → línea de índice ("Capítulo 1...........3")
    (lambda t: t.count("..") > 5, "puntos de relleno"),
    # Muy corto y sin verbo aparente
    (lambda t: len(t.strip()) < 80, "muy corto"),
]


def _is_low_value(text: str) -> tuple[bool, str | None]:
    """Devuelve (True, razón) si el texto parece no aportar contenido."""
    if not text.strip():
        return True, "vacío"
    for predicate, reason in _LOW_VALUE_PATTERNS:
        try:
            if predicate(text):
                return True, reason
        except ZeroDivisionError:
            return True, "vacío"
    return False, None


def _clean_text(text: str) -> str:
    """Limpieza básica para texto extraído de PDFs con cabeceras/pies repetitivos."""
    # Colapsar múltiples espacios o líneas en blanco
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_document(
    document: Document,
    chunk_size: int = 1200,
    overlap: int = 200,
    filter_low_value: bool = True,
) -> list[Chunk]:
    """
    Fragmenta un documento en chunks con solapamiento.

    El solapamiento garantiza que una entidad que caiga en el límite
    entre dos chunks aparezca completa en al menos uno. Se intenta cortar
    en límites naturales (párrafo, frase, espacio) para no partir palabras.

    Si `filter_low_value=True` (default), descarta chunks que parecen no aportar
    contenido (índices, líneas de relleno, fragmentos muy cortos).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser positivo")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap debe ser >= 0 y < chunk_size")

    chunks: list[Chunk] = []
    text = _clean_text(document.text)
    start = 0
    i = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Intentar cortar en un salto de línea o punto para no partir frases
        if end < len(text):
            window = text[start:end]
            for sep in ["\n\n", "\n", ". ", " "]:
                last_sep = window.rfind(sep)
                if last_sep > chunk_size * 0.5:  # Al menos la mitad del chunk
                    end = start + last_sep + len(sep)
                    break

        chunk_text = text[start:end].strip()

        if chunk_text:
            skip = False
            if filter_low_value:
                low_value, _reason = _is_low_value(chunk_text)
                skip = low_value

            if not skip:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}_chunk_{i:04d}",
                        doc_id=document.doc_id,
                        text=chunk_text,
                        metadata={
                            **document.metadata,
                            "chunk_index": i,
                            "doc_id": document.doc_id,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
                i += 1

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks
