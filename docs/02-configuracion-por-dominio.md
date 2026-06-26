# Configuración por dominio

El principio rector del proyecto es que **el conocimiento de negocio vive en
YAML, no en Python**. Cambiar de dominio (o afinar uno) consiste en editar los
archivos de `configs/`, sin tocar `src/ragkg/`. Esta guía explica cada archivo,
qué controla y cómo añadir un dominio nuevo.

---

## 1. Estructura de `configs/`

```
configs/
├── base/                         # configuración transversal (no por dominio)
│   ├── pipeline.yaml             # tamaños de chunk, retrieval, extracción
│   ├── llm.yaml                  # proveedores LLM y sus defaults
│   └── embeddings.yaml           # modelos de embeddings disponibles
└── domains/
    ├── generic/                  # plantilla mínima (Concept, Person, Org, Location)
    └── offers/                   # dominio de referencia (ofertas técnicas)
        ├── ontology.yaml         # entidades, sus id_fields y propiedades
        ├── relations.yaml        # relaciones permitidas y su dirección
        ├── normalization.yaml    # alias → canónico, por categoría
        ├── semantic_definitions.yaml  # definiciones/ejemplos para el prompt
        ├── extraction_schema.yaml     # contrato de salida del extractor
        └── eval_rubric.yaml      # umbrales y rúbrica de evaluación
```

`config/loader.py` carga estos archivos. Un dominio se compone de **5 YAML** (más
`eval_rubric.yaml` para la evaluación). Si un archivo falta, se carga como vacío;
pero `ontology.yaml` **debe** contener la clave `entities` y `relations.yaml` la
clave `relations` (lo valida `DomainConfig` con Pydantic).

---

## 2. `ontology.yaml` — el catálogo de entidades

Define **qué tipos de entidad existen**, cómo se identifican y qué propiedades
tienen.

```yaml
entities:
  Technology:
    label: Technology          # etiqueta del nodo en Neo4j
    id_fields: [canonical_name] # campo identificador (clave de unicidad)
    properties:
      canonical_name: string
      category: string          # language, framework, database, cloud, ...
      vendor: string
      aliases: list
      confidence: float

  FunctionalRequirement:
    label: FunctionalRequirement
    id_fields: [code]           # ← se identifica por CÓDIGO (RF01, REQ-12...)
    properties:
      code: string
      title: string
      mandatory: boolean
      confidence: float
```

Claves a entender:

- **`id_fields`** define la identidad del nodo. El **primer** `id_field` es el que
  se usa para el `MERGE` (unicidad) y para crear el constraint correspondiente.
- La mayoría de entidades se identifican por **`canonical_name`**. Las entidades
  con código (`FunctionalRequirement`) se identifican por **`code`**: esto es
  intencionado y evita duplicados como `"RF07"` vs `"RF 07: Integración"` (ver
  [decisiones](04-decisiones-y-limitaciones.md)).
- `label` es la etiqueta literal del nodo en Neo4j. Por convención coincide con el
  nombre del tipo.
- Las propiedades son declarativas (documentan el esquema esperado); el extractor
  rellena las que encuentra.

`Document` y `Chunk` también se declaran aquí (son las entidades de la capa
documental). El dominio `offers` define ~18 tipos de entidad; el `generic` solo 4
(Concept, Person, Organization, Location) como punto de partida.

---

## 3. `relations.yaml` — qué conexiones son válidas

Define las relaciones permitidas **con su dirección**. El extractor no puede crear
relaciones fuera de esta whitelist, y la dirección `source → target` se valida
antes de escribir (`is_relation_allowed`).

```yaml
relations:
  - name: REQUIRES_TECHNOLOGY
    source: Offer
    target: Technology
    description: "Una oferta propone o requiere una tecnología."
    properties:
      evidence: string
      mandatory: boolean
      confidence: float

  - name: MENTIONS
    source: Chunk
    target:                     # target puede ser una LISTA de tipos
      - Offer
      - Technology
      - Role
      # ...
```

Notas:

- `target` puede ser un tipo único o una **lista** de tipos (caso de `MENTIONS`,
  que conecta un chunk con cualquier entidad).
- `properties` documenta los atributos esperados de la arista. `evidence` y
  `confidence` son la base de la trazabilidad.
- La whitelist evita que el grafo se convierta en un "vertedero semántico": si el
  LLM inventa `USES_TECHNOLOGY` en vez de `REQUIRES_TECHNOLOGY`, se descarta.

---

## 4. `normalization.yaml` — alias → canónico

Mapea las muchas formas de nombrar algo a un nombre canónico, por **categoría**.

```yaml
technologies:
  ".NET":
    aliases: [dotnet, ".net core", "asp.net", ".net framework"]
    category: framework
    vendor: Microsoft
  Kubernetes:
    aliases: [kubernetes, k8s, kube]
    category: platform

roles:
  Backend Developer:
    aliases: ["backend developer", "desarrollador backend", "programador backend"]

# ... ai_concepts, methodologies, certifications, compliance_frameworks,
#     contact_channels, sectors

# Clave reservada: mapea cada TIPO de la ontología a su categoría aquí.
_type_to_category:
  Technology: technologies
  AIConcept: ai_concepts
  Role: roles
  Methodology: methodologies
  Sector: sectors
  Certification: certifications
  ComplianceFramework: compliance_frameworks
  ContactChannel: contact_channels
```

Cómo funciona:

- `EntityNormalizer` construye un índice invertido `alias(minúsculas) → canónico`
  por categoría. En ingesta, cada entidad de un tipo mapeado se canonicaliza.
- **`_type_to_category`** es la pieza que permite añadir un tipo nuevo y darle
  normalización **sin tocar Python**: basta declarar aquí qué categoría usa. (Es
  el único cambio de código que hubo entre v0.1.0 y v0.2.0 del dominio: antes este
  mapeo estaba hardcoded.)
- Un tipo sin entrada en `_type_to_category` se persiste con su nombre "limpio"
  (sin canonicalizar), lo cual es perfectamente válido.
- El **mismo** `normalization.yaml` lo reutiliza la capa de evaluación para hacer
  el matching tolerante a sinónimos (`.NET` ≈ `dotnet`) — ver
  [evaluación](03-evaluacion.md).

---

## 5. `semantic_definitions.yaml` — lo que ve el LLM

Aporta, por tipo de entidad, la **descripción, ejemplos, contraejemplos y pistas
de detección** que se inyectan en el prompt de extracción.

```yaml
definitions:
  Bidder:
    description: >
      Empresa o consorcio que PRESENTA la oferta. Es el proveedor potencial,
      no el cliente.
    examples: ["Entelgy Consulting, S.A.", "Indra Sistemas"]
    counterexamples: ["AcmeInc", "Banco Santander"]   # son clientes
    detection_hints:
      - "Empresa que firma la oferta o aparece en 'datos de la empresa'"
```

Este archivo es donde más se "afina" la calidad de extracción: buenos
contraejemplos evitan confusiones típicas (Bidder vs Client, Technology vs
AIConcept, Certification vs ComplianceFramework). El prompt se genera
dinámicamente desde aquí (`build_extraction_prompt`), en versión compacta por
defecto (solo descripción) para ahorrar tokens.

---

## 6. `extraction_schema.yaml` — el contrato de salida

Documenta qué campos debe devolver el LLM por entidad y por relación, y las reglas
de validación. Es **descriptivo/documental**: la validación dura la hacen los
modelos Pydantic de `validators.py` y los filtros del extractor.

```yaml
entity_output:
  required_fields: [temp_id, type, name, canonical_name, evidence, confidence]
  optional_fields: [properties]
relation_output:
  required_fields: [source, type, target, evidence, confidence]
  optional_fields: [properties]
validation_rules:
  - "Toda entidad debe tener evidence no vacía"
  - "confidence debe ser >= 0.5 para persistirse"
  # ...
```

---

## 7. `eval_rubric.yaml` — umbrales y juez

Configura la capa de evaluación del dominio (umbrales OK/KO, prompt y criterios
del juez). Se cubre en detalle en [evaluación](03-evaluacion.md):

```yaml
thresholds:
  min_text_recall: 0.5     # fracción de hechos gold en la respuesta para OK
  min_correctness: 60      # correctness mínimo del juez (si está activo)
  min_pass_rate: 0.6       # fracción de paráfrasis que deben pasar
judge:
  system_prompt: "..."
  criteria:
    - { key: correctness,  desc: "..." }
    - { key: completeness, desc: "..." }
    - { key: faithfulness, desc: "..." }
```

---

## 8. `configs/base/` — configuración transversal

- **`pipeline.yaml`** — `chunk_size`/`overlap`, extensiones soportadas, umbral de
  confianza, parámetros de retrieval (`top_k_vector`, `expand_top_n_chunks`...).
  > Aviso de rigor: algunos de estos valores son **declarativos**. Los scripts CLI
  > usan sus propios defaults (flags Typer) y, en el caso del chunking, no leen
  > `pipeline.yaml` ni `CHUNK_SIZE` del `.env`. Ver
  > [limitaciones](04-decisiones-y-limitaciones.md).
- **`llm.yaml`** — catálogo de proveedores (`openai`, `groq`, `mock`) y sus
  modelos por defecto. El proveedor real se inyecta por `LLM_PROVIDER`.
- **`embeddings.yaml`** — modelos de embeddings disponibles y sus dimensiones.

---

## 9. Receta: añadir un dominio nuevo

1. **Copiar la plantilla**

   ```bash
   cp -r configs/domains/generic configs/domains/mi_dominio
   ```

2. **Editar los 5 YAML**
   - `ontology.yaml`: declara tus entidades, su `id_fields` y propiedades.
   - `relations.yaml`: declara las conexiones válidas y su dirección.
   - `normalization.yaml`: añade categorías de alias y el bloque
     `_type_to_category` (mapea cada tipo normalizable a su categoría).
   - `semantic_definitions.yaml`: descripción + ejemplos + contraejemplos por
     tipo (clave para la calidad de extracción).
   - `extraction_schema.yaml`: normalmente se deja igual.
   - `eval_rubric.yaml`: ajusta umbrales y criterios del juez si evalúas.

3. **Seleccionar el dominio**

   ```bash
   # en .env
   DOMAIN=mi_dominio
   ```

4. **Crear esquema e ingerir**

   ```bash
   make schema                       # constraints + índices para el dominio
   make ingest FILE=mi_documento.pdf
   ```

5. **Preparar el gold de evaluación** (opcional pero recomendado): crea tu propio
   dataset de queries con `expected_entities` (hechos verificables). Ver
   [evaluación](03-evaluacion.md).

No hay que tocar nada en `src/ragkg/`. Si necesitas un comportamiento que el YAML
no expresa (un nuevo tipo de identificador, una categoría de normalización con
lógica especial), ese es el límite real de la configuración y sí requeriría
código.

---

## 10. Checklist de validación

- [ ] `ontology.yaml` tiene `entities` y cada entidad un `id_fields` no vacío.
- [ ] `relations.yaml` tiene `relations`; cada relación con `name`, `source`,
      `target`.
- [ ] Cada tipo que quieras normalizar está en `_type_to_category` y su categoría
      existe en `normalization.yaml`.
- [ ] `semantic_definitions.yaml` cubre cada tipo de la ontología con
      descripción y, a ser posible, contraejemplos.
- [ ] `make schema` corre sin error (valida que los labels/`id_fields` son
      tokens seguros).
- [ ] Una ingesta de prueba crea entidades con `confidence` y relaciones con
      `evidence`.
