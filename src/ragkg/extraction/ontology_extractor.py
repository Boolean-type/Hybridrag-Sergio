"""
Extracción de entidades y relaciones guiada por ontología.

El prompt se construye dinámicamente a partir de la configuración del dominio.
Soporta varios proveedores de LLM (mock, openai, groq, anthropic) detrás de
una interfaz mínima `LLMClient.generate(prompt) -> str`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from ragkg.config.loader import DomainConfig
from ragkg.extraction.validators import ExtractionResult


# --------------------------------------------------------------------- LLM API


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class MockLLMClient:
    """Devuelve siempre una extracción vacía. Útil para validar el pipeline."""

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return json.dumps({"entities": [], "relations": []})


class OpenAICompatibleClient:
    """
    Cliente para cualquier API compatible con OpenAI Chat Completions.

    Soporta directamente OpenAI y Groq (mismo SDK, distinto base_url).
    """

    DEFAULT_EXTRACTION_SYSTEM_PROMPT = (
        "Eres un extractor de entidades estructurado. "
        "Responde SOLO con JSON válido, sin markdown ni explicaciones."
    )
    DEFAULT_ANSWER_SYSTEM_PROMPT = (
        "Eres un asistente que responde preguntas usando SOLO la información "
        "proporcionada en el contexto. Si no hay información suficiente, dilo. "
        "Cita los chunks por número (Chunk 1, Chunk 2...) cuando bases una afirmación en ellos."
    )

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        base_url: str | None = None,
        json_mode: bool = True,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("Instala openai: pip install openai") from e
        self.client = OpenAI(
            api_key=api_key or os.getenv("LLM_API_KEY"),
            base_url=base_url,
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=0,  # nosotros controlamos los reintentos manualmente
        )
        self.model = model
        self.temperature = temperature
        self.json_mode = json_mode
        self.system_prompt = system_prompt or (
            self.DEFAULT_EXTRACTION_SYSTEM_PROMPT if json_mode
            else self.DEFAULT_ANSWER_SYSTEM_PROMPT
        )
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.json_mode:
            params["response_format"] = {"type": "json_object"}
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        return self._call_with_retry(params)

    def _call_with_retry(
        self,
        params: dict[str, Any],
        max_attempts: int = 5,
        base_wait: float = 2.0,
    ) -> str:
        """Llamada con reintento exponencial y respeto del Retry-After / mensaje de rate limit.

        Esto es crítico para tiers gratuitos (Groq free = 12k TPM). Cuando llega un
        429, intentamos:
          1. Leer el tiempo de espera del mensaje de error (Groq lo incluye).
          2. Si no, usar backoff exponencial.
        """
        import re
        import time

        try:
            from openai import APIStatusError, RateLimitError  # type: ignore
        except ImportError:
            RateLimitError = APIStatusError = Exception  # type: ignore

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.chat.completions.create(**params)
                return response.choices[0].message.content or ("{}" if self.json_mode else "")
            except RateLimitError as exc:
                last_exc = exc
                # Extraer 'Please try again in 6.675s' del mensaje de error de Groq.
                wait = base_wait * (2 ** (attempt - 1))
                msg = str(exc)
                match = re.search(r"try again in ([\d.]+)s", msg)
                if match:
                    wait = float(match.group(1)) + 0.5  # margen de seguridad
                if attempt < max_attempts:
                    time.sleep(wait)
                    continue
                raise
            except APIStatusError as exc:
                # Errores transitorios (5xx). Reintentamos con backoff suave.
                last_exc = exc
                status = getattr(exc, "status_code", None)
                if status and 500 <= status < 600 and attempt < max_attempts:
                    time.sleep(base_wait * (2 ** (attempt - 1)))
                    continue
                raise
        if last_exc:
            raise last_exc
        return "{}" if self.json_mode else ""


# Alias por retrocompatibilidad con tests existentes.
OpenAILLMClient = OpenAICompatibleClient


# Configuración de proveedores. Añadir uno nuevo es una sola entrada.
_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "base_url": None,                       # Default de OpenAI
        "default_model": "gpt-4o-mini",
        "api_key_env": "LLM_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",          # Variable estándar de Groq
        "fallback_api_key_env": "LLM_API_KEY",  # Permitimos LLM_API_KEY como alternativa
    },
}


def build_llm_client(
    provider: str | None = None,
    json_mode: bool = True,
    model: str | None = None,
    system_prompt: str | None = None,
) -> LLMClient:
    """Factory que crea el cliente LLM según la variable de entorno o el argumento.

    `json_mode=False` se usa para generar respuestas en lenguaje natural
    (consulta), `json_mode=True` para extracción estructurada.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()

    if provider == "mock":
        return MockLLMClient()

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Proveedor LLM no soportado: '{provider}'. "
            f"Soportados: {['mock'] + list(_PROVIDERS.keys())}"
        )

    cfg = _PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"]) or os.getenv(cfg.get("fallback_api_key_env", ""))
    if not api_key:
        raise ValueError(
            f"No se encontró la API key. Define '{cfg['api_key_env']}' "
            f"(o 'LLM_API_KEY') en tu .env"
        )

    return OpenAICompatibleClient(
        model=model or os.getenv("LLM_MODEL", cfg["default_model"]),
        api_key=api_key,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        base_url=cfg["base_url"],
        json_mode=json_mode,
        system_prompt=system_prompt,
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
    )


# ------------------------------------------------------------------- extractor


# Patrones de basura que el LLM tiende a inventar. Si una entidad coincide
# con uno de estos, se descarta sin pasar al grafo.
_GARBAGE_NAME_PATTERNS = [
    "tal cual",
    "tal como aparece",
    "nombre canónico",
    "fragmento literal",
    "ninguno",
    "ninguna",
    "n/a",
    "no aplica",
    "norma, regulación",
    "marco de cumplimiento",
    "lenguaje, framework",
    "no se menciona",
    "no especificado",
]

# Nombres demasiado genéricos que casi siempre son ruido (no aportan info)
_GENERIC_BAD_NAMES = {
    "modelo",
    "modelo de ia",
    "sistema",
    "solución",
    "solucion",
    "propuesta",
    "herramienta",
    "consola",
    "api",
    "documento",
    "documentación",
    "documentacion",
    "espacios",
    "dependencias",
    "mercado",
    "cliente",
    "ciudadano",
    "usuario",
    "agente",
}


def _is_garbage_entity(entity) -> tuple[bool, str | None]:
    """Detecta entidades obviamente erróneas o demasiado genéricas."""
    name = (entity.canonical_name or entity.name or "").strip()
    name_lower = name.lower()

    if not name or len(name) < 3:
        return True, "nombre demasiado corto"

    for pattern in _GARBAGE_NAME_PATTERNS:
        if pattern in name_lower:
            return True, f"contiene patrón basura '{pattern}'"

    if name_lower in _GENERIC_BAD_NAMES:
        return True, f"nombre genérico ('{name_lower}')"

    # Frases largas no son entidades, son descripciones
    if len(name.split()) > 6:
        return True, "demasiadas palabras (parece descripción, no entidad)"

    return False, None


def build_extraction_prompt(chunk_text: str, config: DomainConfig, compact: bool = True) -> str:
    """Construye el prompt dinámicamente a partir de la configuración del dominio.

    Si `compact=True`, devuelve una versión sin ejemplos extensos, más corta.
    Útil para evitar tope de rate limit.
    """
    definitions = config.semantic_definitions.get("definitions", {})
    allowed_relations = config.get_allowed_relations()
    entity_types = list(definitions.keys())
    relation_names = sorted({r["name"] for r in allowed_relations})

    if compact:
        entity_lines = [
            f"- **{name}**: {(defn.get('description') or '').strip().split('.')[0]}"
            for name, defn in definitions.items()
        ]
        entity_block_str = "\n".join(entity_lines)
        relation_lines = [
            f"- **{r['name']}**: {r['source']} → {r['target']}"
            for r in allowed_relations
        ]
        relation_block_str = "\n".join(relation_lines)
    else:
        entity_block: dict[str, Any] = {}
        for name, defn in definitions.items():
            entity_block[name] = {
                "description": defn.get("description", "").strip(),
                "examples": defn.get("examples", []),
                "counterexamples": defn.get("counterexamples", []),
            }
        relation_block = [
            {"name": r["name"], "source": r["source"], "target": r["target"]}
            for r in allowed_relations
        ]
        entity_block_str = json.dumps(entity_block, indent=2, ensure_ascii=False)
        relation_block_str = json.dumps(relation_block, indent=2, ensure_ascii=False)

    return f"""Eres un extractor preciso de entidades estructuradas para el dominio '{config.domain_name}'. Tu trabajo es leer un fragmento de texto e identificar entidades específicas y bien definidas.

# REGLAS CRÍTICAS (LEE CON ATENCIÓN)

1. **`type` DEBE ser EXACTAMENTE uno** de esta lista (no inventes otros, no uses sinónimos):
   {', '.join(entity_types)}

2. **El nombre de la relación (`type`) DEBE ser EXACTAMENTE uno** de:
   {', '.join(relation_names)}
   ❌ NO inventes USES_TECHNOLOGY, HAS_TECHNOLOGY ni similares.

3. **Solo extrae entidades NOMBRADAS y CONCRETAS**.
   ✅ "Azure OpenAI", "GPT-4", "Java 17", "Banco Santander", "ENS Nivel Alto", "PMP"
   ❌ "modelo", "sistema", "solución", "herramienta", "API", "consola", "documento"
   ❌ "ciudadano", "cliente" (sin nombre propio), "usuario", "agente"

4. **`evidence` debe ser un fragmento LITERAL del texto** que justifica la extracción. NUNCA vacío.

5. **NO copies frases del enunciado del prompt** como entidades.
   ❌ "tal cual aparece en el texto", "nombre canónico", "fragmento literal", "Ninguno"

6. **NO clasifiques abstracciones como Technology**.
   ✅ Technology = producto concreto: Azure OpenAI, Kubernetes, PostgreSQL, GPT-4
   ❌ "documentación técnica", "base de conocimiento", "modelo de IA" no son Technology

7. **Una sola Offer por documento**. Si el documento es una oferta, hay UNA `Offer`,
   no varias por cada mención de "la solución".

8. **`source` y `target` en relaciones son STRINGS con el `temp_id`**, NUNCA objetos ni listas.
   ✅ "target": "ent_3"
   ❌ "target": {{"name": "Java"}}
   ❌ "target": ["ent_3", "ent_4"]

9. **CADA RELACIÓN DEBE INCLUIR TODOS ESTOS CAMPOS OBLIGATORIOS**:
   - `source` (string, temp_id de la entidad origen)
   - `type` (string, nombre de la relación)
   - `target` (string, temp_id de la entidad destino)
   - `evidence` (string, fragmento literal del texto)
   - `confidence` (float entre 0.0 y 1.0)
   Si no puedes proporcionar `evidence` y `confidence`, NO incluyas la relación.

10. **Si no hay entidades claras y concretas, devuelve listas vacías**. Es mejor no extraer nada que inventar.

# Tipos de entidad disponibles

{entity_block_str}

# Relaciones disponibles

{relation_block_str}

# Ejemplo de extracción correcta CON RELACIONES

Texto: "La oferta de Entelgy propone desplegar la solución en Azure Kubernetes Service usando Azure OpenAI con GPT-4, cumpliendo ENS nivel alto."

Respuesta:
{{
  "entities": [
    {{"temp_id": "ent_1", "type": "Offer", "name": "Oferta de Entelgy", "canonical_name": "Oferta de Entelgy", "evidence": "La oferta de Entelgy propone", "confidence": 0.92, "properties": {{}}}},
    {{"temp_id": "ent_2", "type": "Bidder", "name": "Entelgy", "canonical_name": "Entelgy", "evidence": "La oferta de Entelgy", "confidence": 0.98, "properties": {{}}}},
    {{"temp_id": "ent_3", "type": "Technology", "name": "Azure Kubernetes Service", "canonical_name": "Azure Kubernetes Service", "evidence": "desplegar la solución en Azure Kubernetes Service", "confidence": 0.95, "properties": {{}}}},
    {{"temp_id": "ent_4", "type": "Technology", "name": "Azure OpenAI", "canonical_name": "Azure OpenAI", "evidence": "usando Azure OpenAI con GPT-4", "confidence": 0.95, "properties": {{}}}},
    {{"temp_id": "ent_5", "type": "Technology", "name": "GPT-4", "canonical_name": "GPT-4", "evidence": "Azure OpenAI con GPT-4", "confidence": 0.95, "properties": {{}}}},
    {{"temp_id": "ent_6", "type": "ComplianceFramework", "name": "ENS nivel alto", "canonical_name": "ENS Nivel Alto", "evidence": "cumpliendo ENS nivel alto", "confidence": 0.93, "properties": {{}}}}
  ],
  "relations": [
    {{"source": "ent_1", "type": "PRESENTED_BY", "target": "ent_2", "evidence": "La oferta de Entelgy", "confidence": 0.95, "properties": {{}}}},
    {{"source": "ent_1", "type": "REQUIRES_TECHNOLOGY", "target": "ent_3", "evidence": "desplegar la solución en Azure Kubernetes Service", "confidence": 0.92, "properties": {{}}}},
    {{"source": "ent_1", "type": "REQUIRES_TECHNOLOGY", "target": "ent_4", "evidence": "usando Azure OpenAI", "confidence": 0.94, "properties": {{}}}},
    {{"source": "ent_1", "type": "REQUIRES_TECHNOLOGY", "target": "ent_5", "evidence": "Azure OpenAI con GPT-4", "confidence": 0.94, "properties": {{}}}},
    {{"source": "ent_1", "type": "COMPLIES_WITH", "target": "ent_6", "evidence": "cumpliendo ENS nivel alto", "confidence": 0.93, "properties": {{}}}}
  ]
}}

# Ejemplo de qué NO hacer

Texto: "La solución propuesta es una herramienta de IA con consola web."

Respuesta CORRECTA:
{{"entities": [], "relations": []}}

(No hay tecnologías concretas mencionadas; "herramienta de IA" y "consola" son demasiado genéricas.)

# Texto a analizar

\"\"\"
{chunk_text}
\"\"\"

Devuelve SOLO el JSON, sin markdown, sin explicaciones. Recuerda: cada relación debe tener `source`, `type`, `target`, `evidence` y `confidence`."""


def _extract_json(raw: str) -> dict[str, Any]:
    """Parser tolerante: extrae el primer objeto JSON aunque venga con texto alrededor."""
    raw = raw.strip()
    # Quitar fences de markdown si los hay
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_from_chunk(
    chunk_text: str,
    config: DomainConfig,
    llm_client: LLMClient,
    min_confidence: float = 0.5,
) -> ExtractionResult:
    """
    Extrae entidades y relaciones de un chunk usando un LLM.

    Aplica una cascada de filtros:
    1. Parseo JSON tolerante.
    2. Validación Pydantic (formato).
    3. Filtro por confianza mínima.
    4. Filtro de entidades-basura por nombre genérico o sospechoso.
    """
    prompt = build_extraction_prompt(chunk_text, config)
    raw = llm_client.generate(prompt)
    parsed = _extract_json(raw)
    result = ExtractionResult(**parsed)

    # Filtrar entidades basura ANTES del filtro de confianza para reducir ruido.
    filtered_entities = []
    kept_ids: set[str] = set()
    for entity in result.entities:
        is_garbage, _reason = _is_garbage_entity(entity)
        if is_garbage:
            continue
        filtered_entities.append(entity)
        kept_ids.add(entity.temp_id)

    # Quedarnos solo con relaciones cuyos extremos sobrevivieron al filtro.
    filtered_relations = [
        r for r in result.relations
        if r.source in kept_ids and r.target in kept_ids
    ]

    cleaned = ExtractionResult(entities=filtered_entities, relations=filtered_relations)
    return cleaned.filter_by_confidence(min_confidence=min_confidence)
