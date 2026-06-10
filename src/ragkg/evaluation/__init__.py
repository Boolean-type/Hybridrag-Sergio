"""Capa de evaluación del pipeline RAG-KG.

Dos capas combinadas:
  - Determinista: recall de hechos gold + grounding de citas (sin LLM).
  - Juez (LLM-as-judge): rúbrica + justificación (opcional).
"""

from ragkg.evaluation.dataset import EvalCase, EvalDataset, load_eval_dataset
from ragkg.evaluation.judge import JudgeVerdict, LLMJudge
from ragkg.evaluation.metrics import (
    FactCoverage,
    GroundingCheck,
    RetrievalMetrics,
    fact_recall_in_entities,
    fact_recall_in_text,
    grounding_check,
    precision_recall_f1,
)
from ragkg.evaluation.runner import (
    AnswerBundle,
    CaseResult,
    EvalReport,
    Thresholds,
    VariantResult,
    run_evaluation,
)

__all__ = [
    "AnswerBundle",
    "CaseResult",
    "EvalCase",
    "EvalDataset",
    "EvalReport",
    "FactCoverage",
    "GroundingCheck",
    "JudgeVerdict",
    "LLMJudge",
    "RetrievalMetrics",
    "Thresholds",
    "VariantResult",
    "fact_recall_in_entities",
    "fact_recall_in_text",
    "grounding_check",
    "load_eval_dataset",
    "precision_recall_f1",
    "run_evaluation",
]
