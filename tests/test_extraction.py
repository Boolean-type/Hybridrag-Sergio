"""Tests para la extracción guiada por ontología (sin llamar a LLM real)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragkg.config.loader import load_domain_config
from ragkg.extraction.ontology_extractor import (
    MockLLMClient,
    _extract_json,
    build_extraction_prompt,
    extract_from_chunk,
)
from ragkg.extraction.validators import ExtractionResult

CONFIGS_ROOT = Path(__file__).parent.parent / "configs" / "domains"


@pytest.fixture(scope="module")
def offers_config():
    return load_domain_config("offers", configs_root=CONFIGS_ROOT)


def test_build_extraction_prompt_contains_entities_and_relations(offers_config):
    prompt = build_extraction_prompt("Texto de prueba con Java y Spring Boot.", offers_config)
    assert "Technology" in prompt
    assert "REQUIRES_TECHNOLOGY" in prompt
    assert "Texto de prueba" in prompt


def test_mock_client_returns_empty_extraction(offers_config):
    result = extract_from_chunk("Texto", offers_config, MockLLMClient())
    assert isinstance(result, ExtractionResult)
    assert result.entities == []
    assert result.relations == []


def test_extract_json_handles_markdown_fences():
    raw = """```json
    {"entities": [], "relations": []}
    ```"""
    assert _extract_json(raw) == {"entities": [], "relations": []}


def test_extract_json_handles_extra_text():
    raw = 'Aquí tienes el JSON: {"entities": [], "relations": []} fin'
    assert _extract_json(raw) == {"entities": [], "relations": []}


def test_validation_rejects_low_confidence(offers_config):
    payload = {
        "entities": [
            {
                "temp_id": "e1",
                "type": "Technology",
                "name": "Java",
                "canonical_name": "Java",
                "evidence": "Se pide Java",
                "confidence": 0.3,
            },
            {
                "temp_id": "e2",
                "type": "Technology",
                "name": "Spring",
                "canonical_name": "Spring Boot",
                "evidence": "Con Spring Boot",
                "confidence": 0.9,
            },
        ],
        "relations": [],
    }

    class FixedClient:
        def generate(self, prompt):  # noqa: ARG002
            return json.dumps(payload)

    result = extract_from_chunk("Java y Spring Boot", offers_config, FixedClient(), min_confidence=0.5)
    assert len(result.entities) == 1
    assert result.entities[0].canonical_name == "Spring Boot"


def test_validation_detects_invalid_entity_type(offers_config):
    payload = {
        "entities": [
            {
                "temp_id": "e1",
                "type": "Pizza",  # No existe en la ontología
                "name": "Margarita",
                "canonical_name": "Margarita",
                "evidence": "Pizza Margarita",
                "confidence": 0.9,
            }
        ],
        "relations": [],
    }
    result = ExtractionResult(**payload)
    problems = result.validate_against_config(
        allowed_entity_types=set(offers_config.get_entity_types()),
        allowed_relation_names=offers_config.get_allowed_relation_names(),
    )
    assert any("Pizza" in p for p in problems)


# --- Tests del cliente LLM y la factory ---
# Estos tests requieren el SDK de openai (también usado para Groq).
# Si no está instalado, se saltan en lugar de fallar.
pytest.importorskip("openai")

import os
from unittest.mock import patch

from ragkg.extraction.ontology_extractor import (
    MockLLMClient,
    OpenAICompatibleClient,
    build_llm_client,
)


def test_build_llm_client_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = build_llm_client()
    assert isinstance(client, MockLLMClient)


def test_build_llm_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="no soportado"):
        build_llm_client(provider="cualquier_cosa_rara")


def test_build_llm_client_groq_requires_api_key(monkeypatch):
    # Sin claves, debe fallar limpiamente.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        build_llm_client(provider="groq")


def test_build_llm_client_groq_uses_correct_base_url(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key")
    # No llamamos a la API real: solo verificamos que el cliente queda configurado bien.
    client = build_llm_client(provider="groq", json_mode=True)
    assert isinstance(client, OpenAICompatibleClient)
    assert "groq.com" in str(client.client.base_url)
    assert client.model == "llama-3.3-70b-versatile"
    assert client.json_mode is True


def test_build_llm_client_groq_answer_mode_disables_json(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_fake_key")
    client = build_llm_client(provider="groq", json_mode=False)
    assert client.json_mode is False
    # Modo respuesta usa el system prompt de respuesta, no el de extracción.
    assert "JSON" not in client.system_prompt


def test_build_llm_client_groq_accepts_llm_api_key_fallback(monkeypatch):
    # Si no hay GROQ_API_KEY pero sí LLM_API_KEY, debe funcionar.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "gsk_fallback_fake_key")
    client = build_llm_client(provider="groq")
    assert isinstance(client, OpenAICompatibleClient)


def test_build_llm_client_custom_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    client = build_llm_client(provider="groq", model="llama-3.1-8b-instant")
    assert client.model == "llama-3.1-8b-instant"


# --- Tests de filtros anti-basura (post-extracción) ---

from ragkg.extraction.ontology_extractor import _is_garbage_entity
from ragkg.extraction.validators import ExtractedEntity


def _make_entity(name: str, type_: str = "Technology", canonical: str | None = None) -> ExtractedEntity:
    return ExtractedEntity(
        temp_id="ent_1",
        type=type_,
        name=name,
        canonical_name=canonical or name,
        evidence="dummy evidence",
        confidence=0.9,
    )


def test_garbage_filter_catches_prompt_leakage():
    e = _make_entity("tal cual aparece en el texto")
    is_garbage, _ = _is_garbage_entity(e)
    assert is_garbage


def test_garbage_filter_catches_definition_leakage():
    e = _make_entity("norma, regulación o marco de cumplimiento", type_="ComplianceFramework")
    is_garbage, _ = _is_garbage_entity(e)
    assert is_garbage


def test_garbage_filter_catches_generic_names():
    for bad in ["modelo", "sistema", "solución", "consola", "documento"]:
        e = _make_entity(bad)
        is_garbage, _ = _is_garbage_entity(e)
        assert is_garbage, f"'{bad}' debería ser detectado como basura"


def test_garbage_filter_catches_short_names():
    e = _make_entity("a")
    is_garbage, _ = _is_garbage_entity(e)
    assert is_garbage


def test_garbage_filter_catches_long_descriptions():
    e = _make_entity("Una larga descripción que parece más una frase que una entidad concreta")
    is_garbage, _ = _is_garbage_entity(e)
    assert is_garbage


def test_garbage_filter_catches_ninguno():
    e = _make_entity("Ninguno")
    is_garbage, _ = _is_garbage_entity(e)
    assert is_garbage


def test_real_technologies_pass_filter():
    for good in ["Azure OpenAI", "GPT-4", "Kubernetes", "PostgreSQL", "Java", ".NET"]:
        e = _make_entity(good)
        is_garbage, _ = _is_garbage_entity(e)
        assert not is_garbage, f"'{good}' NO debería ser basura"


def test_real_compliance_passes_filter():
    for good in ["ENS Nivel Alto", "GDPR", "OWASP Top 10", "ISO 27001"]:
        e = _make_entity(good, type_="ComplianceFramework")
        is_garbage, _ = _is_garbage_entity(e)
        assert not is_garbage, f"'{good}' NO debería ser basura"


def test_real_bidder_passes_filter():
    e = _make_entity("Entelgy", type_="Bidder")
    is_garbage, _ = _is_garbage_entity(e)
    assert not is_garbage
