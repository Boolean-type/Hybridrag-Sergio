# Changelog del dominio offers

## Posterior a v0.2.0 — Recuperación híbrida y capa de evaluación

> Nota: esta sección recoge cambios **del sistema** (no solo del dominio)
> introducidos después de la documentación inicial. Cuando esta documentación y el
> código discrepen, manda el código.

### Recuperación híbrida (RRF)

- La consulta combina cuatro estrategias: **vectorial + keyword (BM25) + Cypher
  estructural por intención + expansión por grafo**, fusionadas con **Reciprocal
  Rank Fusion** (`k=60`). Antes la documentación de fases describía solo
  "vector + expansión"; la implementación actual añade BM25 y consulta estructural.
  Ver [`01-arquitectura.md`](01-arquitectura.md).
- Índices nuevos en el esquema: además del **vectorial**, ahora se crea un índice
  **full-text** `chunk_text` para BM25 (`graph/schema.py`).

### Capa de evaluación (opción A: gold por hechos)

- Nuevo subsistema `src/ragkg/evaluation/` (`dataset`, `metrics`, `judge`,
  `runner`, `report`) y script `scripts/evaluate.py` (`make eval` / `make
  eval-quick`).
- Scoring **determinista-primero**: recall de hechos gold sobre la respuesta y
  sobre lo recuperado, grounding de citas, y un **juez LLM como apoyo (no veto)**.
- Datasets: `test_queries.yaml` (v0.5.0, **calibrado a la única oferta ingerida**,
  11 casos q1–q11) y `test_queries_full.yaml` (corpus amplio multi-oferta).
- Casos q10/q11 dotados de **gold mínimo e innegable** para no depender del juez.
- Cada run se guarda en `data/eval_runs/*.json`. Detalle en
  [`03-evaluacion.md`](03-evaluacion.md).

### Modelos y coste

- Diseño de coste: **modelo potente en ingesta** (extracción, coste único) y
  **modelo pequeño/barato en respuesta y juez**. En el **MVP** todo corrió con
  `gpt-5.4-mini` (ver `meta` en `data/eval_runs/`). El proveedor del proyecto es
  **OpenAI**.

### Correcciones de deriva código↔documentación (documentadas, no resueltas)

- Los entry points `ragkg-ingest`/`ragkg-query` de `pyproject.toml` apuntan a
  `ragkg.cli.*`, que **no existe**: usar `python scripts/...` o `make`.
- El proveedor **Anthropic** se barajó pero **no está implementado** (solo
  `openai`/`groq`/`openrouter`/`mock`).
- `CHUNK_SIZE` del `.env` **no** lo consume `scripts/ingest.py` (usar
  `--chunk-size`).
- Detalle completo en [`04-decisiones-y-limitaciones.md`](04-decisiones-y-limitaciones.md).

---

## v0.2.0 — Refinado contra oferta real (Entelgy/SGTFI)

**Motivación.** La v0.1.0 se diseñó sobre supuestos. Al confrontarla con una
oferta técnica real (servicio de IA para call center público, 50 páginas,
Azure end-to-end) aparecieron cinco huecos:

### Cambios estructurales

1. **`Bidder` ≠ `Client`.** El cliente es a quien va dirigida la oferta;
   el ofertante es quien la presenta. Ahora son entidades distintas con
   propiedades distintas (CIF, empleados, sedes vs sector, acrónimo).
2. **`FunctionalRequirement` con `code`.** Los pliegos reales numeran sus
   requisitos (`RF01`, `RF02`...). Es el nivel al que los evaluadores
   hacen preguntas. Se separa de `TechnicalRequirement` (más granular,
   sin código).
3. **`ComplianceFramework` ≠ `Certification`.** ENS, GDPR, OWASP, CCN-STIC
   no son certificaciones de personas: son marcos que cumple el sistema.
   Mezclarlos rompe consultas tipo "qué ofertas cumplen ENS nivel alto".
4. **`AIConcept`** como entidad propia. RAG, fine-tuning, DPO, STT, TTS no
   son productos: son técnicas. Separarlos permite preguntar
   "qué ofertas usan RAG independientemente de qué proveedor cloud".
5. **`ContactChannel`, `ServiceLevel`, `ProjectPhase`, `Deliverable`.**
   Tipos de pregunta naturales en cualquier oferta de servicios públicos
   ("¿qué SLA promete?", "¿qué canales soporta?", "¿qué entregables hay
   en la fase de transición?") que antes se perdían como texto libre.

### Cambios en relations

- Nuevas: `PRESENTED_BY`, `HAS_FUNCTIONAL_REQUIREMENT`, `USES_AI_CONCEPT`,
  `COMPLIES_WITH`, `SUPPORTS_CHANNEL`, `COMMITS_SLA`, `HAS_PHASE`,
  `HAS_DELIVERABLE`, `FR_USES_TECHNOLOGY`, `FR_PRODUCES_DELIVERABLE`,
  `PHASE_PRODUCES_DELIVERABLE`, `PHASE_FOLLOWS`.

### Cambios en normalización

- De 20 a 60+ tecnologías. Familia Azure completa (OpenAI, Foundry,
  AI Search, Speech, Bot Service, AKS, ACR, APIM, Functions, App Service,
  Service Bus, Logic Apps, Blob Storage, AD, Key Vault, VNet, Monitor,
  App Insights, Log Analytics, DevOps, SQL DB, Cosmos DB, Form Recognizer).
- Modelos LLM concretos (GPT-4, GPT-4o, GPT-5, Claude, Cohere, embeddings).
- Herramientas de calidad/QA (SonarQube, JUnit, Selenium, JMeter, SoapUI,
  Testlink).
- Categoría nueva `ai_concepts` con 12 entradas.
- Categoría nueva `compliance_frameworks` con ENS (alto/medio), GDPR,
  CCN-STIC, OWASP, ISO 27001.
- Categoría nueva `contact_channels`.
- Roles ampliados (Documentalista, Consultor IA, Consultor BI/UX,
  Director de Negocio, Product Owner, Jefe de Proyecto, Analista
  Programador, Técnico de Pruebas).
- Metodologías ampliadas (DevSecOps, UXD, CI/CD, IaC).

### Cambio en código

Solo uno, y minúsculo: `EntityNormalizer` ahora lee el mapeo
`Tipo → categoría` desde una clave reservada `_type_to_category` en
`normalization.yaml`. Antes era un dict hardcoded en Python; ahora añadir
una entidad nueva (con su categoría de normalización) no toca código.
Es el espíritu del proyecto.

### Tests

- De 29 a **88 tests**, todos en verde.
- Se cubre cada categoría nueva con casos representativos de la oferta real.

### Cosas que NO cambiaron

- Ningún cambio en el contrato del extractor LLM (`extraction_schema.yaml`).
- Ningún cambio en upserts ni en el cliente Neo4j.
- Ningún cambio en los retrievers.
- Ningún cambio en los scripts CLI.

Esto valida la decisión arquitectónica: el conocimiento de dominio vive
en YAML, el código Python no se entera.
