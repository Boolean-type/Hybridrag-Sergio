# Decisiones de diseño y limitaciones conocidas

Apartado honesto. Primero las decisiones deliberadas (qué se eligió y por qué);
después las limitaciones reales del estado actual del código. Cuando el código y
otra documentación discrepen, **manda el código**.

---

## Parte 1 — Decisiones de diseño

### 1. Evaluación determinista-primero; el juez apoya, no veta

**Decisión.** Con hechos gold disponibles, el veredicto OK/KO lo decide el recall
determinista. El juez LLM solo puede **bajar la confianza** si discrepa; **nunca**
convierte un OK determinista en KO.

**Por qué.** El juez del proyecto es un modelo pequeño (por coste). Un juez débil
genera **falsos negativos**: si pudiera vetar, descartaría respuestas correctas.
Anclar el veredicto a una comprobación determinista de hechos (presencia de
entidades por nombre/código, con tolerancia a alias y formato) da una señal
estable y auditable; el juez aporta matices (faithfulness) sin poder de
destrucción. En casos **sin** gold (abiertos), el juez sí decide, porque no hay
alternativa determinista. Implementación en `evaluation/runner.py`
(`judge_vetoes`) y detalle en [evaluación](03-evaluacion.md).

### 2. Ingesta con modelo potente; respuesta y juez con modelo barato

**Decisión.** La **extracción** (ingesta) usa un modelo **potente** (clase GPT).
La **respuesta** y el **juez** están pensados para un modelo **pequeño/barato**.

**Por qué.** La extracción estructurada guiada por ontología es donde la calidad
del modelo más rinde: define el grafo, es la base de todo lo demás, y es un
**coste único** por documento (offline, no crítico en tiempo). En consulta, en
cambio, el grafo ya está construido y un modelo barato basta para redactar con el
contexto recuperado; y un juez barato es suficiente porque solo apoya.

**Estado actual / mecanismo.** En el código, ingesta y respuesta leen ambas
`LLM_MODEL`; el juez es separable con `JUDGE_LLM_MODEL`. Como ingesta y consulta
son **invocaciones distintas** que cargan el `.env` en cada ejecución, el split
fuerte/barato se consigue configurando `LLM_MODEL` apropiadamente en cada fase. En
el **MVP** todo corrió con `gpt-5.4-mini` (ver `meta` de los runs en
`data/eval_runs/`). No existe hoy una variable separada tipo `INGEST_LLM_MODEL`
(ver limitaciones).

### 3. Ingesta "en batch" para abaratar (intención de proyecto)

**Decisión.** La ingesta se concibe como un proceso **por lotes, offline**, no
crítico en tiempo; de cara al proyecto puede usar la **Batch API** de OpenAI para
reducir coste.

**Estado actual.** En el MVP la extracción es **síncrona, chunk a chunk**
(`scripts/ingest.py::_extract_and_upsert`), con reintentos exponenciales y una
pausa opcional entre llamadas (`LLM_CHUNK_DELAY_SECONDS`). Lo único que ya va en
lote son los **embeddings** (`embed_batch`). La Batch API es trabajo futuro (ver
limitaciones).

### 4. Las entidades con código se identifican por `code`, no por `canonical_name`

**Decisión.** `FunctionalRequirement` (y cualquier entidad con `id_fields:
[code]`) usa el **código** como identidad (`RF01`, `REQ-12`), no el nombre.

**Por qué.** Los pliegos numeran sus requisitos, y el código es el identificador
**estable**: el título varía ("RF 07" vs "RF 07: Integración con el CRM") pero el
código no. Identificar por código evita duplicados y hace que `RF07` sea siempre
el mismo nodo. El prompt de extracción lo refuerza: `canonical_name` = solo el
código en forma canónica (sin espacios), título en `name`/`properties.title`. La
capa de evaluación lo respeta recogiendo también `code` al medir el recall
(`_collect_retrieved_facts`), porque el valor verificable de estas entidades vive
en `code`, no en `name`.

### 5. q10/q11 con gold mínimo e innegable

**Decisión.** Los casos agregado/similitud (q10/q11) llevan un gold **mínimo**:
solo los hechos que cualquier respuesta correcta debe contener, verificados contra
el grafo.

**Por qué.** Eran "abiertos" y dependían del juez. Exigir exhaustividad en un
agregado sería injusto; no exigir nada los dejaba a merced de un juez débil. El
término medio —2 de 3 hechos innegables con `min_text_recall=0.5`— los hace
deterministas sin fragilizarlos. Detalle en [evaluación](03-evaluacion.md).

### Otras decisiones transversales

- **Conocimiento en YAML, código agnóstico.** El cimiento del proyecto: el código
  no sabe qué es "Java" ni "una oferta". Cambiar de dominio es editar
  `configs/domains/<dominio>/`. Validado por la v0.2.0 del dominio `offers`, que
  reestructuró 14+ entidades **sin** tocar el pipeline (ver [CHANGELOG](CHANGELOG.md)).
- **Trazabilidad obligatoria.** Cada entidad lleva `confidence` y cada relación
  `evidence`; `Chunk-[:MENTIONS]->Entidad` ancla todo al texto. Sistema auditable,
  no caja negra.
- **Whitelist de relaciones con dirección.** No basta con que el tipo exista: se
  valida `source → target` (`is_relation_allowed`) antes de escribir.
- **Cypher dinámico saneado.** Labels y relaciones dinámicos pasan por
  `_ensure_safe` (solo alfanumérico + `_`) para evitar inyección.
- **Filtrado en cascada.** Al descartar una entidad por confianza/basura, se
  descartan sus relaciones (no quedan aristas colgando).
- **RRF sin normalizar scores.** Fusiona rankings de fuentes heterogéneas usando
  solo el rango; robusto y simple.
- **Embeddings locales por defecto.** `all-MiniLM-L6-v2` (384 dim, gratis) como
  punto de partida; el coste de LLM se reserva para extracción y respuesta.

---

## Parte 2 — Limitaciones conocidas

### Configuración y CLI

- **`[project.scripts]` roto.** `pyproject.toml` declara `ragkg-ingest =
  "ragkg.cli.ingest:main"` y `ragkg-query = "ragkg.cli.query:main"`, pero **no
  existe `src/ragkg/cli/`**. Esos entry points fallarían tras `pip install`. La vía
  soportada es `python scripts/...` o `make`. (PHASES.md mencionaba esos comandos;
  se ha corregido.)
- **`CHUNK_SIZE` del `.env` no se consume en la ingesta.** `scripts/ingest.py` usa
  los flags Typer `--chunk-size`/`--overlap` (defaults 1200/200) y **no** lee
  `CHUNK_SIZE`/`CHUNK_OVERLAP` del entorno. Para cambiar el chunking hay que pasar
  los flags; editar el `.env` no tiene efecto sobre la ingesta.
- **`pipeline.yaml` es mayormente declarativo.** Los parámetros de retrieval
  (`top_k_vector`, `expand_top_n_chunks`...) y de chunking de `configs/base/
  pipeline.yaml` no se inyectan en los scripts; los valores efectivos vienen de los
  flags/defaults del CLI. El YAML documenta intención, no siempre configura
  comportamiento.
- **No hay variable separada para el modelo de ingesta.** Para el split
  fuerte/barato (decisión 2) hay que cambiar `LLM_MODEL` entre la fase de ingesta y
  la de consulta manualmente; no existe `INGEST_LLM_MODEL`.

### Proveedores LLM

- **Solo proveedores compatibles OpenAI.** `build_llm_client` soporta `openai`,
  `groq`, `openrouter` y `mock`. **Anthropic se barajó pero no está implementado**
  (aparece en un docstring antiguo; la empresa usa OpenAI). No intentes
  `LLM_PROVIDER=anthropic`: dará error de proveedor no soportado.

### Extracción y grafo

- **Extracción 100% LLM, síncrona.** Cara y lenta a escala; sin cascada
  diccionario/regex → NER → LLM. La arquitectura lo admite (basta otro `LLMClient`),
  pero hoy no existe.
- **Sin Batch API todavía** (decisión 3): la ingesta hace una llamada por chunk.
- **Calidad dependiente del prompt y de los YAML.** El extractor filtra basura y
  tipos inválidos, pero entidades mal definidas en `semantic_definitions.yaml` o
  alias ausentes en `normalization.yaml` degradan el grafo. Afinar el dominio es
  parte del trabajo.
- **Versionado de ontología parcial.** La ontología declara `ontology_version` en
  `Chunk`/`Document`, pero las entidades no la guardan de forma sistemática; filtrar
  por versión cuando la ontología evolucione no está resuelto.

### Recuperación

- **Detección de intención por regex.** `structured_query.py` usa patrones fijos
  (en español, orientados a `offers`). Cubre los tipos del dominio, pero es frágil a
  fraseos no previstos y habría que ampliarla para otros dominios.
- **Chunking por caracteres.** Buen baseline, pero documentos muy estructurados
  (tablas, apartados) se beneficiarían de chunking semántico por secciones.
- **Cambiar el modelo de embeddings obliga a reindexar.** Los espacios vectoriales
  no son comparables; si cambias `EMBEDDING_MODEL`, hay que regenerar **todos** los
  embeddings y recrear el índice con la nueva dimensión.

### Evaluación

- **Juez = respondedor en el MVP.** Los runs guardados usan `gpt-5.4-mini` como
  juez **y** respondedor, lo que introduce sesgo de autoevaluación. El diseño pide
  un juez distinto (y barato); está pendiente de aplicar en la configuración.
- **Dataset calibrado a una sola oferta.** `test_queries.yaml` (v0.5.0) está
  calibrado contra la única oferta ingerida (Entelgy). Las métricas reflejan ese
  corpus mínimo; para conclusiones generales hay que ingerir más ofertas y usar
  `test_queries_full.yaml`.
- **Matching por substring/alias.** Es tolerante por diseño, pero puede producir
  falsos positivos si un hecho gold es un substring de otra cosa presente en el
  texto. El gold se elige para minimizarlo (hechos discriminantes), no se garantiza
  formalmente.

### Operación y seguridad

- **Gestión de secretos.** Las credenciales viven en `.env` (correctamente en
  `.gitignore`, no trackeado en git). Recomendación: tratar la API key como secreto
  rotable, no compartir el `.env` fuera del equipo, y **rotar** cualquier clave que
  haya podido exponerse. Para despliegue, considerar un gestor de secretos en vez de
  un archivo en disco.
- **Neo4j con credenciales por defecto.** `docker-compose.yml` arranca Neo4j con
  `neo4j/password`. Aceptable en local; cambiar antes de cualquier exposición de red.
- **`make reset` es destructivo.** Borra todos los nodos y relaciones (pide
  confirmación, pero no hace backup).

---

## Parte 3 — Trabajo futuro (derivado de lo anterior)

- Arreglar o eliminar los entry points de `pyproject.toml` (o crear
  `src/ragkg/cli/`).
- Variable de modelo separada para ingesta (`INGEST_LLM_MODEL`) y aplicar el split
  fuerte/barato por defecto.
- Integrar la **Batch API** de OpenAI en la ingesta.
- Cascada de extracción (diccionario/regex → NER → LLM) para abaratar y acelerar.
- Configurar un **juez distinto** del respondedor en la evaluación.
- Ampliar el corpus de evaluación más allá de una oferta.
- Inyectar de verdad `pipeline.yaml`/`CHUNK_SIZE` o eliminar la configuración
  muerta para evitar confusión.
- Chunking semántico por secciones y `ontology_version` consistente en entidades.
