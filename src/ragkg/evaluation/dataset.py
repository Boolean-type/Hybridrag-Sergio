"""Carga del conjunto de evaluación (queries gold) desde YAML.

El gold sigue la "opción A": cada caso lleva una lista de `expected_entities`
(hechos verificables por nombre/código/métrica) y, opcionalmente, paráfrasis
congeladas para medir robustez al fraseo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragkg.config.loader import load_yaml
from ragkg.evaluation.metrics import expected_fact_values


@dataclass
class EvalCase:
    """Un caso de evaluación: la pregunta original + sus variantes + el gold."""

    id: str
    question: str
    type: str = "factoid"  # factoid | aggregate | similarity
    paraphrases: list[str] = field(default_factory=list)
    expected_entities: list[dict[str, Any]] = field(default_factory=list)

    @property
    def expected_facts(self) -> list[str]:
        """Valores gold verificables (Java, Banca, RF01, ...)."""
        return expected_fact_values(self.expected_entities)

    @property
    def variants(self) -> list[str]:
        """Pregunta original seguida de las paráfrasis (sin duplicados)."""
        seen: set[str] = set()
        out: list[str] = []
        for q in [self.question, *self.paraphrases]:
            key = q.strip().lower()
            if q and key not in seen:
                seen.add(key)
                out.append(q)
        return out

    @property
    def has_gold(self) -> bool:
        """True si el caso puede evaluarse de forma determinista (tiene hechos gold)."""
        return bool(self.expected_facts)


@dataclass
class EvalDataset:
    domain: str
    version: str
    cases: list[EvalCase]

    def subset(self, ids: list[str] | None = None, limit: int | None = None) -> EvalDataset:
        cases = self.cases
        if ids:
            wanted = set(ids)
            cases = [c for c in cases if c.id in wanted]
        if limit:
            cases = cases[:limit]
        return EvalDataset(domain=self.domain, version=self.version, cases=cases)


def load_eval_dataset(
    path: str | Path = "src/ragkg/evaluation/test_queries.yaml",
) -> EvalDataset:
    """Carga el YAML de queries de evaluación."""
    data = load_yaml(Path(path))
    if not data:
        raise FileNotFoundError(f"No se encontró o está vacío el dataset de evaluación: {path}")

    cases = [
        EvalCase(
            id=q["id"],
            question=q["question"],
            type=q.get("type", "factoid"),
            paraphrases=q.get("paraphrases", []) or [],
            expected_entities=q.get("expected_entities", []) or [],
        )
        for q in data.get("queries", [])
    ]
    return EvalDataset(
        domain=data.get("domain", "generic"),
        version=str(data.get("version", "0")),
        cases=cases,
    )
