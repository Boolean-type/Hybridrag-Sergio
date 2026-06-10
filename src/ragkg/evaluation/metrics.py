"""Métricas básicas de evaluación del recuperador."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    precision: float
    recall: float
    f1: float
    num_expected: int
    num_retrieved: int
    num_hits: int


def precision_recall_f1(expected: set[str], retrieved: set[str]) -> RetrievalMetrics:
    """Calcula precisión, recall y F1 a partir de dos conjuntos comparables (case-insensitive)."""
    exp = {e.lower() for e in expected}
    ret = {r.lower() for r in retrieved}
    hits = exp & ret

    precision = len(hits) / len(ret) if ret else 0.0
    recall = len(hits) / len(exp) if exp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        num_expected=len(exp),
        num_retrieved=len(ret),
        num_hits=len(hits),
    )


def entities_in_result(graph_context: list[dict]) -> set[str]:
    """Extrae el conjunto de nombres canónicos presentes en el contexto del grafo."""
    names: set[str] = set()
    for item in graph_context:
        entity = item.get("entity") or {}
        name = entity.get("canonical_name") or entity.get("name")
        if name:
            names.add(name)
    return names


# --------------------------------------------------------------- capa de hechos
#
# La "opción A" del diseño: el gold de cada pregunta es una lista de hechos
# verificables (entidades por nombre/código/métrica). Comprobamos su presencia
# de forma determinista, sin LLM, sobre dos superficies:
#   1. El TEXTO de la respuesta final (¿el modelo lo dijo?).
#   2. Las entidades RECUPERADAS del grafo (¿el retrieval lo trajo?).
#
# Comparar ambas permite localizar el fallo: si el hecho está en lo recuperado
# pero no en la respuesta -> fallo de generación; si no está en ninguno ->
# fallo de recuperación.


def expected_fact_value(fact: dict) -> str | None:
    """Extrae el valor verificable de un hecho gold (name | code | metric)."""
    for key in ("name", "code", "metric", "value"):
        val = fact.get(key)
        if val:
            return str(val)
    return None


def expected_fact_values(expected: list[dict]) -> list[str]:
    """Lista de valores verificables a partir de los expected_entities del YAML."""
    values = [expected_fact_value(f) for f in (expected or [])]
    return [v for v in values if v]


import re

# Índice de alias = mapa nombre_canónico(lower) -> [alias, ...]. Se construye desde
# el normalization.yaml del dominio, de modo que un hecho gold ".NET" también case
# con "dotnet" o "asp.net", y "RFT" con "reinforcement fine-tuning", etc.
AliasIndex = dict[str, list[str]]

# Claves de normalization.yaml que NO son categorías de entidades.
_NON_CATEGORY_KEYS = {"domain", "version", "_type_to_category"}


def build_alias_index(normalization: dict) -> AliasIndex:
    """Aplana todas las categorías de normalization.yaml en un único índice de alias."""
    index: AliasIndex = {}
    for category, entries in (normalization or {}).items():
        if category in _NON_CATEGORY_KEYS or not isinstance(entries, dict):
            continue
        for canonical, meta in entries.items():
            aliases = (meta or {}).get("aliases", []) if isinstance(meta, dict) else []
            index[_norm(canonical)] = [str(a) for a in aliases]
    return index


def _norm(s: str) -> str:
    """Minúsculas + espacios colapsados."""
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _despace(s: str) -> str:
    """Sin ningún espacio: 'RF 01' -> 'rf01' (casa códigos con/sin separador)."""
    return re.sub(r"\s+", "", str(s).lower())


def surface_forms(fact: str, alias_index: AliasIndex | None) -> list[str]:
    """Todas las formas a buscar para un hecho: el canónico + sus alias."""
    forms = {fact}
    if alias_index:
        forms.update(alias_index.get(_norm(fact), []))
    return [f for f in forms if f]


def _form_in_text(form: str, text: str) -> bool:
    """Casa por substring normalizado y, además, por versión sin espacios (códigos)."""
    n, t = _norm(form), _norm(text)
    if n and n in t:
        return True
    nd, td = _despace(form), _despace(text)
    return bool(nd) and len(nd) >= 3 and nd in td


@dataclass
class FactCoverage:
    """Qué hechos gold aparecen en una superficie de texto/entidades."""

    expected: list[str]
    found: list[str]
    missing: list[str]
    recall: float


def fact_recall_in_text(
    expected: list[str], text: str, alias_index: AliasIndex | None = None
) -> FactCoverage:
    """Recall de hechos gold en un texto, considerando alias y formato (espacios)."""
    text = text or ""
    found = [
        e for e in expected if any(_form_in_text(f, text) for f in surface_forms(e, alias_index))
    ]
    missing = [e for e in expected if e not in found]
    recall = len(found) / len(expected) if expected else 1.0
    return FactCoverage(expected=expected, found=found, missing=missing, recall=recall)


def fact_recall_in_entities(
    expected: list[str], retrieved: set[str], alias_index: AliasIndex | None = None
) -> FactCoverage:
    """Recall de hechos gold contra entidades recuperadas (alias + substring en ambos sentidos)."""
    retrieved = retrieved or set()
    found = []
    for e in expected:
        forms = surface_forms(e, alias_index)
        hit = any(
            _form_in_text(f, r) or _form_in_text(r, f) for f in forms for r in retrieved
        )
        if hit:
            found.append(e)
    missing = [e for e in expected if e not in found]
    recall = len(found) / len(expected) if expected else 1.0
    return FactCoverage(expected=expected, found=found, missing=missing, recall=recall)


@dataclass
class GroundingCheck:
    """¿Las citas 'Chunk N' de la respuesta apuntan a chunks que existen?"""

    cited: list[int]
    valid: list[int]
    invalid: list[int]
    grounded: bool


def grounding_check(answer_text: str, num_chunks_in_context: int) -> GroundingCheck:
    """Detecta citas a chunks inexistentes (señal barata de alucinación de fuentes)."""
    import re

    cited = sorted({int(m) for m in re.findall(r"[Cc]hunk\s+(\d+)", answer_text or "")})
    valid = [c for c in cited if 1 <= c <= num_chunks_in_context]
    invalid = [c for c in cited if c not in valid]
    return GroundingCheck(cited=cited, valid=valid, invalid=invalid, grounded=not invalid)
