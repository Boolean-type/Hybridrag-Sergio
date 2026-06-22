"""Orquestador de la evaluación.

Para cada caso y cada paráfrasis:
  1. Ejecuta el pipeline real (vía un `answer_fn` inyectable -> AnswerBundle).
  2. Capa determinista: recall de hechos gold en la respuesta y en lo recuperado,
     y chequeo de grounding de citas.
  3. Capa juez (opcional): rúbrica + justificación.
  4. Combina ambas en un veredicto OK/KO con confianza anclada y localización
     del fallo (retrieval vs generation).
Agrega por caso (consistencia entre paráfrasis) y por dataset.

`answer_fn` se inyecta para no acoplar la evaluación a Neo4j: el script real
le pasa el pipeline; los tests le pasan un doble.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ragkg.evaluation.dataset import EvalCase, EvalDataset
from ragkg.evaluation.judge import JudgeVerdict, LLMJudge
from ragkg.evaluation.metrics import (
    AliasIndex,
    FactCoverage,
    GroundingCheck,
    fact_recall_in_entities,
    fact_recall_in_text,
    grounding_check,
)


@dataclass
class AnswerBundle:
    """Lo que el pipeline produce para una pregunta, listo para evaluar."""

    answer_text: str
    retrieved_facts: list[str] = field(default_factory=list)  # nombres de entidades recuperadas
    context_excerpt: str = ""  # texto de contexto pasado al LLM (para el juez/grounding)
    num_chunks: int = 0


@dataclass
class Thresholds:
    min_text_recall: float = 0.5  # recall mínimo de hechos en la respuesta
    min_correctness: int = 60  # correctness mínimo del juez (si está activo)
    min_pass_rate: float = 0.6  # fracción de paráfrasis que deben pasar para OK del caso

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> Thresholds:
        cfg = (cfg or {}).get("thresholds", {}) if cfg else {}
        return cls(
            min_text_recall=float(cfg.get("min_text_recall", 0.5)),
            min_correctness=int(cfg.get("min_correctness", 60)),
            min_pass_rate=float(cfg.get("min_pass_rate", 0.6)),
        )


@dataclass
class VariantResult:
    question: str
    is_paraphrase: bool
    answer: str
    verdict: str  # OK | KO | SKIPPED
    confidence: int  # 0..100, anclado
    text_recall: float | None
    graph_recall: float | None
    grounded: bool
    failure_locus: str
    justification: str
    text_coverage: dict[str, Any] | None = None
    judge: dict[str, Any] | None = None


@dataclass
class CaseResult:
    id: str
    type: str
    has_gold: bool
    verdict: str  # OK | KO | SKIPPED
    pass_rate: float
    consistent: bool
    mean_confidence: int
    variants: list[VariantResult]


@dataclass
class EvalReport:
    domain: str
    dataset_version: str
    judge_enabled: bool
    summary: dict[str, Any]
    cases: list[CaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "dataset_version": self.dataset_version,
            "judge_enabled": self.judge_enabled,
            "summary": self.summary,
            "cases": [
                {
                    **{k: v for k, v in asdict(c).items() if k != "variants"},
                    "variants": [asdict(v) for v in c.variants],
                }
                for c in self.cases
            ],
        }


def _anchored_confidence(text_recall: float | None, judge: JudgeVerdict | None) -> int:
    """Confianza combinada: media de recall determinista y correctness del juez."""
    signals: list[float] = []
    if text_recall is not None:
        signals.append(text_recall * 100)
    if judge is not None:
        signals.append(float(judge.correctness))
    return int(round(statistics.mean(signals))) if signals else 0


def _derive_locus(text_recall: float, graph_recall: float, min_recall: float) -> str:
    """Sin juez, localiza el fallo comparando lo recuperado con lo respondido."""
    if text_recall >= min_recall:
        return "none"
    if graph_recall >= min_recall:
        return "generation"  # estaba recuperado pero no se usó/respondió
    return "retrieval"  # ni siquiera se recuperó


def evaluate_variant(
    case: EvalCase,
    question: str,
    bundle: AnswerBundle,
    thresholds: Thresholds,
    judge: LLMJudge | None,
    is_paraphrase: bool,
    alias_index: AliasIndex | None = None,
) -> VariantResult:
    expected = case.expected_facts

    # --- Capa determinista ---
    text_cov: FactCoverage | None = None
    graph_cov: FactCoverage | None = None
    grounding: GroundingCheck = grounding_check(bundle.answer_text, bundle.num_chunks)

    if expected:
        text_cov = fact_recall_in_text(expected, bundle.answer_text, alias_index)
        graph_cov = fact_recall_in_entities(expected, set(bundle.retrieved_facts), alias_index)

    # --- Capa juez (opcional) ---
    jv: JudgeVerdict | None = None
    if judge is not None:
        jv = judge.judge(
            question=question,
            answer=bundle.answer_text,
            expected_facts=expected,
            retrieved_facts=bundle.retrieved_facts,
            context_excerpt=bundle.context_excerpt,
        )

    # --- Combinar ---
    text_recall = text_cov.recall if text_cov else None
    graph_recall = graph_cov.recall if graph_cov else None

    # El juez solo cuenta como señal si respondió un JSON válido (sin error ni abstención).
    judge_usable = jv is not None and jv.error is None and jv.verdict != "ABSTAIN"
    det_pass = None if text_recall is None else text_recall >= thresholds.min_text_recall
    judge_pass = (jv.correctness >= thresholds.min_correctness) if judge_usable else None

    # Sin señal determinista NI juez utilizable -> no evaluable.
    if det_pass is None and judge_pass is None:
        return VariantResult(
            question=question, is_paraphrase=is_paraphrase, answer=bundle.answer_text,
            verdict="SKIPPED", confidence=0, text_recall=None, graph_recall=None,
            grounded=grounding.grounded, failure_locus="none",
            justification="Sin hechos gold y sin juez utilizable: no evaluable.",
            judge=asdict(jv) if jv else None,
        )

    # Política de combinación calibrada para un JUEZ DÉBIL (modelo 8B):
    #   - Con gold determinista, MANDA el determinista. El juez NO puede convertir
    #     un OK determinista en KO (demasiados falsos negativos); solo baja la
    #     confianza y deja una marca de "revisar".
    #   - Sin gold (casos abiertos), el veredicto recae en el juez.
    if det_pass is not None:
        passed = det_pass
        judge_vetoes = (judge_pass is False) and det_pass
    else:
        passed = bool(judge_pass)
        judge_vetoes = False

    # Confianza anclada al determinista cuando existe; el juez solo ajusta.
    if text_recall is not None:
        confidence = int(round(text_recall * 100))
        if judge_vetoes:
            confidence = max(0, confidence - 25)  # sospecha de faithfulness, no veto
    elif judge_usable:
        confidence = int(round((jv.correctness + jv.confidence) / 2))
    else:
        confidence = 0
    # Penaliza citas a chunks inexistentes (señal de fuente alucinada).
    if not grounding.grounded:
        confidence = max(0, confidence - 15)

    # Localización del fallo.
    if passed:
        locus = "none"
    elif det_pass is None and judge_usable:
        locus = jv.failure_locus  # caso abierto: lo decide el juez
    else:
        locus = _derive_locus(text_recall or 0.0, graph_recall or 0.0, thresholds.min_text_recall)

    # Justificación.
    if passed and judge_vetoes:
        justification = (
            f"OK determinista (presentes: {text_cov.found}); el juez discrepa: "
            f"{jv.justification[:160]}"
        )
    elif passed:
        justification = (
            f"Encontrados {text_cov.found} de {expected}." if text_cov
            else (jv.justification if judge_usable else "OK.")
        )
    elif det_pass is None and judge_usable:
        justification = jv.justification
    elif locus == "generation":
        justification = f"Recuperado pero no respondido. Faltan en la respuesta: {text_cov.missing}."
    else:
        justification = f"No recuperado del grafo. Faltan: {text_cov.missing}."

    return VariantResult(
        question=question,
        is_paraphrase=is_paraphrase,
        answer=bundle.answer_text,
        verdict="OK" if passed else "KO",
        confidence=confidence,
        text_recall=text_recall,
        graph_recall=graph_recall,
        grounded=grounding.grounded,
        failure_locus=locus,
        justification=justification,
        text_coverage=asdict(text_cov) if text_cov else None,
        judge=asdict(jv) if jv else None,
    )


def evaluate_case(
    case: EvalCase,
    answer_fn: Callable[[str], AnswerBundle],
    thresholds: Thresholds,
    judge: LLMJudge | None,
    alias_index: AliasIndex | None = None,
) -> CaseResult:
    variants: list[VariantResult] = []
    for i, q in enumerate(case.variants):
        bundle = answer_fn(q)
        variants.append(
            evaluate_variant(
                case, q, bundle, thresholds, judge, is_paraphrase=(i > 0), alias_index=alias_index
            )
        )

    evaluable = [v for v in variants if v.verdict != "SKIPPED"]
    if not evaluable:
        return CaseResult(
            id=case.id, type=case.type, has_gold=case.has_gold, verdict="SKIPPED",
            pass_rate=0.0, consistent=True, mean_confidence=0, variants=variants,
        )

    passed = [v for v in evaluable if v.verdict == "OK"]
    pass_rate = len(passed) / len(evaluable)
    verdicts = {v.verdict for v in evaluable}
    consistent = len(verdicts) == 1
    mean_conf = int(round(statistics.mean(v.confidence for v in evaluable)))
    case_verdict = "OK" if pass_rate >= thresholds.min_pass_rate else "KO"

    return CaseResult(
        id=case.id, type=case.type, has_gold=case.has_gold, verdict=case_verdict,
        pass_rate=round(pass_rate, 3), consistent=consistent,
        mean_confidence=mean_conf, variants=variants,
    )


def run_evaluation(
    dataset: EvalDataset,
    answer_fn: Callable[[str], AnswerBundle],
    judge: LLMJudge | None = None,
    thresholds: Thresholds | None = None,
    alias_index: AliasIndex | None = None,
) -> EvalReport:
    thresholds = thresholds or Thresholds()
    cases = [evaluate_case(c, answer_fn, thresholds, judge, alias_index) for c in dataset.cases]

    scored = [c for c in cases if c.verdict != "SKIPPED"]
    n_ok = sum(1 for c in scored if c.verdict == "OK")
    summary = {
        "total_cases": len(cases),
        "scored_cases": len(scored),
        "skipped_cases": len(cases) - len(scored),
        "ok": n_ok,
        "ko": len(scored) - n_ok,
        "accuracy": round(n_ok / len(scored), 3) if scored else 0.0,
        "consistency_rate": (
            round(sum(1 for c in scored if c.consistent) / len(scored), 3) if scored else 0.0
        ),
        "mean_confidence": (
            int(round(statistics.mean(c.mean_confidence for c in scored))) if scored else 0
        ),
    }
    return EvalReport(
        domain=dataset.domain,
        dataset_version=dataset.version,
        judge_enabled=judge is not None,
        summary=summary,
        cases=cases,
    )
