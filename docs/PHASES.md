# Fases de implementación y notas de arquitectura

## Revisión arquitectónica (resumen)

La arquitectura propuesta en el documento original es sólida y sigue buenas prácticas
de GraphRAG. Estos son los puntos fuertes y las decisiones que conviene afinar.

### Lo que está bien

- **Separación código / conocimiento de dominio.** Toda la semántica vive en YAML,
  el código Python no sabe qué es "Java" ni "una oferta". Esto es el cimiento
  que permite reutilizar el sistema en otros dominios.
- **Doble capa metadatos + grafo.** Los metadatos enriquecen el RAG clásico y el
  grafo permite razonamiento estructurado. La relación `Chunk-[:MENTIONS]->Entity`
  conecta ambas capas y es lo que hace que el sistema sea explicable.
- **Trazabilidad obligatoria.** Cada entidad y relación tiene `evidence` y
  `confidence`. Esto convierte el sistema en algo auditable, no en una caja negra.
- **Whitelist de relaciones.** Validar contra `relations.yaml` antes de escribir
  evita que el grafo se convierta en un vertedero semántico.

### Mejoras incorporadas en este esqueleto

1. **Validación de la dirección de las relaciones.** No basta con que el
   `type` exista en `relations.yaml`: además se valida que `source -> target`
   coincida con la dirección declarada (`is_relation_allowed`).
2. **Cypher dinámico con whitelist de identificadores.** `upsert_entity` y
   `upsert_relation` admiten labels y relation_types dinámicos pero los
   sanitizan con `_ensure_safe` para evitar inyección de Cypher.
3. **Parser tolerante de la salida del LLM.** `_extract_json` quita fences de
   markdown y se queda con el primer objeto JSON válido. Los LLM pequeños
   meten ruido alrededor del JSON con frecuencia.
4. **Filtrado en cascada al aplicar confianza.** Cuando filtras entidades por
   `min_confidence` también descartas las relaciones cuyos extremos se han ido.
   Si no, el grafo queda con aristas colgando.
5. **Embeddings normalizados.** Se usa `normalize_embeddings=True` para que
   la similitud coseno y el producto escalar coincidan numéricamente.
6. **Cliente LLM detrás de una interfaz mínima.** `LLMClient.generate(prompt) -> str`.
   Hay un `MockLLMClient` para tests y un `OpenAILLMClient` opcional.
   Cambiar de proveedor es añadir 20 líneas, no reescribir el pipeline.

### Decisiones que conviene revisar más adelante

- **Extracción 100 % LLM.** Funciona para el prototipo pero es cara y lenta.
  En producción conviene una cascada: diccionario/regex → NER → LLM solo para
  casos ambiguos. La estructura actual lo soporta sin reescribir nada: basta
  con añadir otro `LLMClient` que sea un `HybridExtractor`.
- **Versionado de ontología y reproducibilidad.** Cada entidad/relación
  guarda `confidence` pero no `ontology_version`. Recomendado añadir
  `ontology_version` al chunk y a la entidad para poder filtrar por versión
  cuando la ontología evolucione.
- **Embeddings con OpenAI vs local.** Empieza con `all-MiniLM-L6-v2` (local,
  384 dim, gratis). Cuando la calidad sea el cuello de botella, salta a
  `text-embedding-3-small` (1536 dim) o a un modelo open multilingüe. **Si
  cambias el modelo, hay que regenerar todos los embeddings** porque los
  espacios vectoriales no son comparables.
- **Estrategia de chunking.** El chunker por caracteres con solapamiento es
  un buen baseline. Si los documentos tienen mucha estructura (apartados,
  tablas), conviene pasar a chunking semántico por secciones, que da mejor
  recall y mejor evidencia.

---

## Roadmap por fases

> **Estado (resumen).** Las fases 1–4 están implementadas en el código: ingesta
> con extracción, grafo en Neo4j, recuperación híbrida (vector + BM25 + Cypher
> estructural + grafo, fusión RRF) y capa de evaluación con gold por hechos. Los
> checklists conservan su valor como guía de puesta en marcha. Donde el texto
> original contradecía al código, se ha corregido. Manda el código.

### Fase 1 — Base agnóstica (semana 1)

**Objetivo.** Que el pipeline pueda leer documentos, trocearlos, embeberlos y
guardarlos en Neo4j sin extracción todavía.

Checklist:

- [ ] `docker compose up -d` levanta Neo4j.
- [ ] `python scripts/check_connection.py` responde "Conexión OK".
- [ ] `python scripts/create_schema.py` crea constraints e índice vectorial.
- [ ] `pytest tests/test_config_loader.py tests/test_normalization.py tests/test_retrieval.py` en verde.
- [ ] `python scripts/ingest.py data/samples/Entelgy_Oferta_tecnica.pdf --no-extract`
      (o `make ingest-sample`) guarda `Document` y `Chunk` con embeddings.
- [ ] Una consulta Cypher en el Browser muestra los chunks:
      `MATCH (c:Chunk) RETURN c LIMIT 10`.

**Decisiones que aterrizar aquí.**

- Tamaño de chunk y overlap (defaults: 1200 / 200).
- Modelo de embeddings (defaults: MiniLM local 384 dim).
- Proveedor LLM: el del proyecto es **OpenAI** (`LLM_PROVIDER=openai`). También
  hay soporte compatible OpenAI para `groq` y `openrouter`, y `mock` para tests.
  Anthropic se barajó pero **no está implementado**. Configura `.env`.

---

### Fase 2 — Ontología y extracción (semana 2)

**Objetivo.** Extraer entidades y relaciones de cada chunk y persistirlas en el grafo.

Checklist:

- [ ] Refinar `configs/domains/offers/semantic_definitions.yaml` con tus ejemplos reales.
- [ ] Ajustar `normalization.yaml` con los alias más frecuentes en tus documentos.
- [ ] Configurar `LLM_PROVIDER=openai` en `.env` (Anthropic no está implementado).
- [ ] `python scripts/ingest.py data/samples/Entelgy_Oferta_tecnica.pdf` (con `--extract`),
      o `make ingest-sample`.
- [ ] En Neo4j Browser: `MATCH (o:Offer)-[:REQUIRES_TECHNOLOGY]->(t) RETURN o, t`.
- [ ] Cada entidad debe tener `confidence` y cada relación `evidence`.

**Trampas frecuentes.**

- El LLM devuelve JSON inválido. El `_extract_json` ayuda, pero hay que loguear
  los casos que fallen y refinar el prompt.
- El LLM inventa tipos de entidad. La validación contra ontología los filtra,
  pero conviene avisar y reentrenar el prompt.
- Tecnologías sin entrada en `normalization.yaml`. Hay dos opciones: añadirlas
  o aceptarlas tal cual y limpiar después.

---

### Fase 3 — GraphRAG (semana 3)

**Objetivo.** Recuperar contexto rico combinando vector search + expansión de grafo.

Checklist:

- [ ] `python scripts/query.py "ofertas de banca con Java"` devuelve chunks y entidades.
- [ ] El recuperador híbrido funciona con `expand_graph=True/False` y los
      resultados cambian.
- [ ] Las consultas Cypher de `graph/queries.py` devuelven resultados útiles:
      tecnologías más frecuentes, requisitos por rol, certificaciones por sector.
- [ ] El `answer_builder` produce un prompt limpio para el LLM final.

**LLM de respuesta (ya integrado).**

`scripts/query.py` **genera respuesta en lenguaje natural por defecto**: arma el
contexto con `build_context_prompt(result)` y llama al `LLMClient`
(`build_llm_client(json_mode=False)`, modelo `LLM_MODEL`). Usa `--raw` para ver
solo el contexto recuperado sin gastar tokens, y `--json` para salida estructurada.

---

### Fase 4 — Demo y evaluación (semana 4)

**Objetivo.** Tener un demo defendible y métricas reales.

Checklist:

- [ ] Ofertas ingeridas en Neo4j. **Estado actual:** el dataset calibrado
      (`test_queries.yaml` v0.5.0, 11 casos q1–q11) está afinado a **una sola**
      oferta (Entelgy). Para un corpus amplio multi-oferta, ingiere más documentos
      y usa `test_queries_full.yaml`.
- [ ] Los 11 casos de `src/ragkg/evaluation/test_queries.yaml` devuelven
      resultados esperables.
- [ ] Métricas calculadas: recall de hechos gold, accuracy, consistencia entre
      paráfrasis y confianza por caso.
- [ ] `make eval` produce accuracy, consistencia y un JSON por run en
      `data/eval_runs/`.
**Cómo evaluar (opción A: gold por hechos).**

Cada query de `test_queries.yaml` lleva 4 paráfrasis congeladas y una lista
`expected_entities` (hechos verificables). El harness ejecuta el pipeline real,
mide recall de esos hechos en la respuesta y en lo recuperado, opcionalmente
juzga con un LLM, y emite OK/KO + confianza anclada + localización del fallo
(retrieval vs generation). Guarda cada run en `data/eval_runs/` para comparar regresiones.

```bash
make eval                 # con juez LLM (necesita LLM_PROVIDER configurado)
make eval-quick           # solo capa determinista, 3 casos, sin gastar tokens
python scripts/evaluate.py --only q1_entelgy_rf --variants          # dataset calibrado
python scripts/evaluate.py --dataset-path src/ragkg/evaluation/test_queries_full.yaml \
    --only q1_java_microservices_banca --variants                      # corpus amplio
```

- [ ] CLI documentada. **Aviso:** los entry points `ragkg-ingest`/`ragkg-query`
      de `pyproject.toml` apuntan a `ragkg.cli.*`, que **no existe**; la vía
      soportada es `python scripts/...` o `make`. Alternativa: una API FastAPI con
      dos endpoints (`/ingest` y `/query`).
- [ ] Una sesión grabada en Neo4j Browser mostrando el grafo.

---

## Cómo cambiar de dominio

1. Copia `configs/domains/generic/` → `configs/domains/mi_dominio/`.
2. Edita los 5 YAML. Solo necesitas saber YAML, no Python.
3. En el `.env`: `DOMAIN=mi_dominio`.
4. `python scripts/create_schema.py` (idempotente).
5. `python scripts/ingest.py mi_documento.pdf`.

No hay que tocar nada en `src/ragkg/`.
