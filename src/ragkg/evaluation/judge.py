"""LLM-as-judge para evaluación de respuestas (capa de juicio).

El juez NO es la última palabra: produce puntuaciones por rúbrica y una
justificación, pero el veredicto final (en `runner.py`) las combina con la
capa determinista de hechos. Recomendado usar un modelo distinto/más fuerte
que el respondedor para reducir sesgo de autoevaluación.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ragkg.extraction.ontology_extractor import _extract_json, build_llm_client

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "Eres un evaluador riguroso e imparcial de respuestas de un sistema RAG. "
    "Juzgas únicamente con la evidencia que se te da. Respondes SOLO con JSON válido, "
    "sin markdown ni explicaciones fuera del JSON."
)

DEFAULT_CRITERIA = [
    {"key": "correctness", "desc": "¿La respuesta contiene los hechos gold esperados y no los contradice?"},
    {"key": "completeness", "desc": "¿Cubre todos los hechos gold esperados, sin dejarse ninguno?"},
    {"key": "faithfulness", "desc": "¿Toda afirmación se apoya en el contexto recuperado, sin inventar?"},
]


@dataclass
class JudgeVerdict:
    verdict: str  # "OK" | "KO"
    scores: dict[str, int] = field(default_factory=dict)  # criterio -> 0..100
    confidence: int = 0  # confianza del juez en su propio veredicto (0..100)
    failure_locus: str = "none"  # retrieval | generation | none
    justification: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def correctness(self) -> int:
        return int(self.scores.get("correctness", 0))


class LLMJudge:
    """Evalúa una respuesta contra los hechos gold y el contexto recuperado."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        criteria: list[dict[str, str]] | None = None,
    ):
        # json_mode=True para forzar salida JSON; modelo configurable e independiente.
        self.client = build_llm_client(
            json_mode=True,
            model=model,
            system_prompt=system_prompt or DEFAULT_JUDGE_SYSTEM_PROMPT,
        )
        self.criteria = criteria or DEFAULT_CRITERIA

    def build_prompt(
        self,
        question: str,
        answer: str,
        expected_facts: list[str],
        retrieved_facts: list[str],
        context_excerpt: str = "",
    ) -> str:
        criteria_block = "\n".join(f"- {c['key']}: {c['desc']}" for c in self.criteria)
        scores_template = ", ".join(f'"{c["key"]}": <0-100>' for c in self.criteria)
        facts_block = "\n".join(f"- {f}" for f in expected_facts) or "(sin hechos gold)"
        retrieved_block = ", ".join(retrieved_facts) or "(ninguna)"

        return f"""# Tarea
Evalúa la RESPUESTA de un sistema RAG frente a los HECHOS GOLD que debería contener.

# Pregunta
{question}

# Hechos gold esperados (deben aparecer y no ser contradichos)
{facts_block}

# Entidades que el sistema recuperó del grafo (contexto disponible para responder)
{retrieved_block}

# Fragmento del contexto recuperado (evidencia)
{context_excerpt[:1500]}

# Respuesta del sistema a evaluar
{answer}

# Criterios de puntuación (0 a 100)
{criteria_block}

# Cómo decidir failure_locus cuando el veredicto sea KO
- "retrieval": el/los hechos gold NO estaban entre las entidades recuperadas ni en el contexto.
- "generation": el/los hechos gold SÍ estaban disponibles pero la respuesta no los usó o se equivocó.
- "none": si el veredicto es OK.

# Formato de salida (SOLO este JSON)
{{
  "verdict": "OK" | "KO",
  "scores": {{ {scores_template} }},
  "confidence": <0-100>,
  "failure_locus": "retrieval" | "generation" | "none",
  "justification": "<1-3 frases explicando por qué está bien o mal, citando hechos concretos>"
}}
"""

    def judge(
        self,
        question: str,
        answer: str,
        expected_facts: list[str],
        retrieved_facts: list[str],
        context_excerpt: str = "",
    ) -> JudgeVerdict:
        prompt = self.build_prompt(
            question, answer, expected_facts, retrieved_facts, context_excerpt
        )
        try:
            raw = self.client.generate(prompt)
            parsed = _extract_json(raw)
        except (json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
            return JudgeVerdict(
                verdict="KO",
                confidence=0,
                failure_locus="none",
                justification="El juez no devolvió un JSON válido.",
                error=str(exc),
            )

        scores = parsed.get("scores", {}) or {}
        scores = {k: _clamp_int(v) for k, v in scores.items()}
        verdict = str(parsed.get("verdict", "KO")).upper()
        if verdict not in {"OK", "KO"}:
            verdict = "KO"

        return JudgeVerdict(
            verdict=verdict,
            scores=scores,
            confidence=_clamp_int(parsed.get("confidence", 0)),
            failure_locus=str(parsed.get("failure_locus", "none")),
            justification=str(parsed.get("justification", "")),
            raw=parsed,
        )


def _clamp_int(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0
