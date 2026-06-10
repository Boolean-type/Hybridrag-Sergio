"""Tests de la capa de evaluación (sin Neo4j ni LLM real)."""

from __future__ import annotations

from ragkg.evaluation.dataset import EvalCase, EvalDataset, load_eval_dataset
from ragkg.evaluation.metrics import (
    fact_recall_in_entities,
    fact_recall_in_text,
    grounding_check,
)
from ragkg.evaluation.runner import AnswerBundle, Thresholds, run_evaluation


# --------------------------------------------------------------- capa determinista


def test_fact_recall_in_text_partial():
    cov = fact_recall_in_text(["Java", "Banca"], "Esta oferta de banca usa Java 17.")
    assert cov.recall == 1.0
    assert set(cov.found) == {"Java", "Banca"}


def test_fact_recall_in_text_missing():
    cov = fact_recall_in_text(["Java", "Azure"], "Solo se menciona Java.")
    assert cov.recall == 0.5
    assert cov.missing == ["Azure"]


def test_fact_recall_in_entities_substring_both_ways():
    cov = fact_recall_in_entities(["Java", "RAG"], {"Java 17", "Retrieval (RAG) pipeline"})
    assert cov.recall == 1.0


def test_grounding_detects_invalid_citation():
    g = grounding_check("Según Chunk 1 y Chunk 9, ...", num_chunks_in_context=5)
    assert g.cited == [1, 9]
    assert g.invalid == [9]
    assert g.grounded is False


# --------------------------------------------------------------- dataset


def test_dataset_loads_with_paraphrases():
    ds = load_eval_dataset()
    assert ds.cases
    case = next(c for c in ds.cases if c.id == "q1_entelgy_rf")
    assert len(case.variants) == 5  # original + 4
    assert "Entelgy" in case.expected_facts


def test_alias_and_format_aware_matching():
    from ragkg.evaluation.metrics import build_alias_index, fact_recall_in_text

    idx = build_alias_index(
        {
            "technologies": {".NET": {"aliases": ["dotnet", "asp.net"]}},
            "version": "x",
        }
    )
    # Alias: gold ".NET" debe casar con "dotnet"/"asp.net".
    assert fact_recall_in_text([".NET"], "Backend en dotnet y asp.net core.", idx).recall == 1.0
    # Formato: gold "RF01" debe casar con "RF 01" (sin alias).
    assert fact_recall_in_text(["RF01"], "La oferta cubre RF 01 y RF 09.").recall == 1.0


# --------------------------------------------------------------- runner (sin juez)


def _dataset_one_case():
    return EvalDataset(
        domain="offers",
        version="test",
        cases=[
            EvalCase(
                id="c_ok",
                question="Ofertas con Java y Banca",
                paraphrases=["Propuestas de banca con Java"],
                expected_entities=[
                    {"type": "Technology", "name": "Java"},
                    {"type": "Sector", "name": "Banca"},
                ],
            )
        ],
    )


def test_runner_ok_when_facts_present_no_judge():
    ds = _dataset_one_case()

    def answer_fn(q):
        return AnswerBundle(
            answer_text="La oferta de Banca requiere Java (Chunk 1).",
            retrieved_facts=["Java", "Banca"],
            num_chunks=3,
        )

    report = run_evaluation(ds, answer_fn, judge=None)
    assert report.summary["accuracy"] == 1.0
    assert report.cases[0].verdict == "OK"
    assert report.cases[0].consistent is True


def test_runner_ko_locates_generation_failure():
    """El hecho se recuperó pero no aparece en la respuesta -> fallo de generación."""
    ds = _dataset_one_case()

    def answer_fn(q):
        return AnswerBundle(
            answer_text="No tengo información suficiente.",
            retrieved_facts=["Java", "Banca"],  # estaba recuperado
            num_chunks=3,
        )

    report = run_evaluation(ds, answer_fn, judge=None, thresholds=Thresholds())
    case = report.cases[0]
    assert case.verdict == "KO"
    assert all(v.failure_locus == "generation" for v in case.variants)


def test_runner_ko_locates_retrieval_failure():
    ds = _dataset_one_case()

    def answer_fn(q):
        return AnswerBundle(answer_text="No consta.", retrieved_facts=[], num_chunks=0)

    report = run_evaluation(ds, answer_fn, judge=None)
    case = report.cases[0]
    assert case.verdict == "KO"
    assert all(v.failure_locus == "retrieval" for v in case.variants)


def test_open_case_skipped_without_judge():
    ds = EvalDataset(
        domain="offers",
        version="test",
        cases=[EvalCase(id="open", question="algo abierto", type="similarity")],
    )

    def answer_fn(q):
        return AnswerBundle(answer_text="respuesta libre", num_chunks=2)

    report = run_evaluation(ds, answer_fn, judge=None)
    assert report.cases[0].verdict == "SKIPPED"
    assert report.summary["skipped_cases"] == 1


def test_fake_judge_integration():
    """El runner combina determinista + juez; aquí un juez falso fuerza KO por correctness."""
    ds = _dataset_one_case()

    class FakeJudge:
        def judge(self, **kwargs):
            from ragkg.evaluation.judge import JudgeVerdict

            return JudgeVerdict(
                verdict="KO",
                scores={"correctness": 10, "completeness": 10, "faithfulness": 20},
                confidence=80,
                failure_locus="generation",
                justification="Inventa datos no presentes en el contexto.",
            )

    def answer_fn(q):
        # Hechos presentes (determinista pasaría), pero el juez detecta alucinación.
        return AnswerBundle(
            answer_text="Banca con Java y además promete un 99.99% inventado.",
            retrieved_facts=["Java", "Banca"],
            num_chunks=3,
        )

    report = run_evaluation(ds, answer_fn, judge=FakeJudge())
    assert report.cases[0].verdict == "KO"
    assert report.judge_enabled is True
