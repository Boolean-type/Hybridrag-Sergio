"""Tests para el chunker."""

from __future__ import annotations

from ragkg.ingestion.chunker import chunk_document
from ragkg.ingestion.loaders import Document


def _doc(text: str) -> Document:
    return Document(doc_id="d1", source_file="x.md", text=text)


def test_chunk_short_text_is_filtered_as_low_value():
    # Por defecto filter_low_value=True descarta chunks <80 caracteres
    document = _doc("Texto corto.")
    chunks = chunk_document(document, chunk_size=200, overlap=20)
    assert len(chunks) == 0


def test_chunk_short_text_without_filter_returns_one_chunk():
    document = _doc("Texto corto.")
    chunks = chunk_document(document, chunk_size=200, overlap=20, filter_low_value=False)
    assert len(chunks) == 1
    assert chunks[0].text == "Texto corto."


def test_chunk_long_text_creates_multiple_chunks():
    document = _doc("Una frase con contenido suficientemente largo. " * 500)
    chunks = chunk_document(document, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["doc_id"] == "d1"
        assert "chunk_index" in c.metadata


def test_chunks_have_overlap():
    document = _doc("Una frase con contenido razonable que pasa el filtro. " * 200)
    chunks = chunk_document(document, chunk_size=200, overlap=40)
    for prev, curr in zip(chunks, chunks[1:], strict=False):
        assert curr.metadata["char_start"] <= prev.metadata["char_end"]


def test_chunk_ids_are_sequential():
    document = _doc("Una frase con contenido razonable que pasa el filtro. " * 300)
    chunks = chunk_document(document, chunk_size=200, overlap=40)
    for i, c in enumerate(chunks):
        assert c.chunk_id.endswith(f"chunk_{i:04d}")


def test_index_like_text_is_filtered():
    # Estilo "Capítulo 1...........3" o "1.1. Algo ........ 5"
    document = _doc("Capítulo 1 ........................ 3\n" * 10)
    chunks = chunk_document(document, chunk_size=200, overlap=20)
    assert len(chunks) == 0


def test_real_prose_passes_filter():
    document = _doc(
        "La solución se desplegará en Azure Kubernetes Service usando Azure OpenAI "
        "con GPT-4, cumpliendo con ENS nivel alto. " * 5
    )
    chunks = chunk_document(document, chunk_size=300, overlap=50)
    assert len(chunks) >= 1


# --- Tests para _flatten_to_primitives (regresión del bug Neo4j Map) ---

from ragkg.graph.upsert import _flatten_to_primitives


def test_flatten_keeps_primitives():
    d = {"a": "x", "b": 1, "c": 1.5, "d": True}
    assert _flatten_to_primitives(d) == d


def test_flatten_drops_nested_dict():
    d = {"a": "x", "nested": {"b": 1}}
    assert _flatten_to_primitives(d) == {"a": "x"}


def test_flatten_drops_none():
    assert _flatten_to_primitives({"a": "x", "b": None}) == {"a": "x"}


def test_flatten_keeps_primitive_arrays():
    d = {"tags": ["java", "azure"], "scores": [0.1, 0.2]}
    assert _flatten_to_primitives(d) == d


def test_flatten_drops_heterogeneous_arrays():
    assert _flatten_to_primitives({"mixed": [1, "a", None]}) == {}


def test_flatten_drops_empty_arrays():
    # Listas vacías no aportan tipo; las descartamos para simplicidad.
    assert _flatten_to_primitives({"empty": []}) == {}


def test_flatten_empty_input():
    assert _flatten_to_primitives({}) == {}
    assert _flatten_to_primitives(None) == {}
