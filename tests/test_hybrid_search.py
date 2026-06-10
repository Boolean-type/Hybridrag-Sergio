"""Tests para los nuevos módulos de búsqueda híbrida."""

from __future__ import annotations

from ragkg.retrieval.rrf import reciprocal_rank_fusion
from ragkg.retrieval.structured_query import detect_intent


# --- Reciprocal Rank Fusion ---

def test_rrf_single_ranking_returns_same_order():
    ranking = [
        {"chunk_id": "a", "text": "hola"},
        {"chunk_id": "b", "text": "mundo"},
    ]
    fused = reciprocal_rank_fusion([ranking])
    assert [r["chunk_id"] for r in fused] == ["a", "b"]


def test_rrf_combines_two_rankings():
    r1 = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    r2 = [{"chunk_id": "c"}, {"chunk_id": "a"}, {"chunk_id": "d"}]
    fused = reciprocal_rank_fusion([r1, r2])
    # 'a' aparece en rango 1 y 2 → suma alta
    # 'c' aparece en rango 3 y 1 → suma alta
    # 'b' solo en uno (r1 rango 2)
    # 'd' solo en uno (r2 rango 3)
    ids = [r["chunk_id"] for r in fused]
    assert ids[0] in {"a", "c"}
    assert "d" in ids
    assert len(ids) == 4


def test_rrf_adds_score_and_sources():
    r1 = [{"chunk_id": "a"}]
    r2 = [{"chunk_id": "a"}]
    fused = reciprocal_rank_fusion([r1, r2])
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_score"] > 0
    assert sorted(fused[0]["sources"]) == [0, 1]


def test_rrf_top_n_limits_output():
    r1 = [{"chunk_id": f"id_{i}"} for i in range(10)]
    fused = reciprocal_rank_fusion([r1], top_n=3)
    assert len(fused) == 3


def test_rrf_handles_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_handles_missing_ids():
    # Elementos sin chunk_id se ignoran sin crashear
    r1 = [{"chunk_id": "a"}, {"text": "no id"}]
    fused = reciprocal_rank_fusion([r1])
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "a"


def test_rrf_preserves_original_fields():
    r1 = [{"chunk_id": "a", "text": "hola", "score": 0.9}]
    fused = reciprocal_rank_fusion([r1])
    assert fused[0]["text"] == "hola"
    assert fused[0]["score"] == 0.9


# --- Detección de intención ---

def test_detect_intent_technologies():
    intent = detect_intent("qué tecnologías propone la oferta")
    assert intent is not None
    assert intent.target_label == "Technology"


def test_detect_intent_compliance():
    intent = detect_intent("qué normativas de cumplimiento cumple")
    assert intent is not None
    assert intent.target_label == "ComplianceFramework"


def test_detect_intent_ens_keyword():
    intent = detect_intent("la oferta cumple ENS?")
    assert intent is not None
    assert intent.target_label == "ComplianceFramework"


def test_detect_intent_roles():
    intent = detect_intent("qué perfiles propone")
    assert intent is not None
    assert intent.target_label == "Role"


def test_detect_intent_methodologies():
    intent = detect_intent("qué metodologías de trabajo aplica")
    assert intent is not None
    assert intent.target_label == "Methodology"


def test_detect_intent_returns_none_for_vague():
    intent = detect_intent("dime más sobre esto")
    assert intent is None


def test_detect_intent_sla():
    intent = detect_intent("qué SLA promete")
    assert intent is not None
    assert intent.target_label == "ServiceLevel"


def test_detect_intent_certifications():
    intent = detect_intent("qué certificados profesionales requiere")
    assert intent is not None
    assert intent.target_label == "Certification"


def test_detect_intent_channels():
    intent = detect_intent("qué canales de atención soporta")
    assert intent is not None
    assert intent.target_label == "ContactChannel"
