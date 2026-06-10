# RAG-KG Prototype

Plataforma agnóstica de **RAG enriquecido con grafo de conocimiento** (GraphRAG).
Combina extracción guiada por ontología, metadatos para RAG, grafo en Neo4j y
búsqueda semántica por embeddings.

> **Principio clave:** cambiar de dominio sin tocar código Python. Todo el conocimiento
> de negocio vive en `configs/domains/<dominio>/`.

## Características

- Configuración de dominio en YAML (ontología, taxonomía, relaciones permitidas).
- Extracción guiada por ontología con validación Pydantic.
- Normalización de entidades (alias → nombre canónico).
- Grafo de conocimiento en Neo4j con constraints e índice vectorial.
- Búsqueda híbrida: similitud semántica + expansión por grafo.
- Trazabilidad obligatoria: evidencia textual y `confidence` por entidad/relación.

## Quickstart

### 1. Requisitos

- Python 3.11+
- Docker y Docker Compose
- (Opcional) Una API key de un proveedor LLM si vas a usar el extractor con LLM

### 2. Instalación

```bash
# Clonar y entrar
cd rag-kg-prototype

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -e ".[dev]"

# Copiar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

### 3. Levantar Neo4j

```bash
docker compose up -d
# Browser: http://localhost:7474  (usuario: neo4j / password: password)
```

### 4. Verificar conexión y crear esquema

```bash
python scripts/check_connection.py
python scripts/create_schema.py
```

### 5. Ingesta de un documento de muestra

```bash
python scripts/ingest.py data/samples/oferta_ejemplo.md --domain offers
```

### 6. Consulta

```bash
python scripts/query.py "ofertas de banca con Java y microservicios" --domain offers
```

## Estructura del proyecto

```
rag-kg-prototype/
├── configs/                # Configuración (sin código)
│   ├── base/               # Pipeline, LLM, embeddings
│   └── domains/            # Un subdirectorio por dominio
│       ├── generic/        # Dominio mínimo de ejemplo
│       └── offers/         # Dominio de ofertas técnicas
├── data/                   # Datos (raw, processed, samples)
├── src/ragkg/              # Código Python agnóstico
│   ├── config/             # Carga y validación de YAML
│   ├── ingestion/          # Loaders y chunking
│   ├── extraction/         # Extracción guiada por ontología
│   ├── graph/              # Cliente Neo4j, schema, upserts, queries
│   ├── embeddings/         # Generación de embeddings
│   ├── retrieval/          # Vector + grafo + híbrido
│   ├── generation/         # Construcción de respuesta
│   └── evaluation/         # Métricas y consultas de test
├── scripts/                # Entradas CLI (ingest, query, schema...)
└── tests/                  # Tests unitarios
```

## Roadmap por fases

| Semana | Objetivo | Entregable |
|--------|----------|------------|
| 1 | Base agnóstica | Configs cargables, ingesta TXT/MD, chunks en Neo4j, embeddings, vector search |
| 2 | Ontología y extracción | Extracción LLM con prompt dinámico, normalización, MENTIONS Chunk→Entity |
| 3 | GraphRAG | Hybrid retriever, filtros por tecnología/sector/rol, contexto enriquecido |
| 4 | Demo y evaluación | 20-50 ofertas cargadas, queries de evaluación, CLI/API, demo |

Más detalle en [docs/PHASES.md](docs/PHASES.md) o en la guía original del proyecto.

## Cambiar de dominio

Copia `configs/domains/generic/` como punto de partida, edítalo y lanza:

```bash
DOMAIN=mi_dominio python scripts/ingest.py mi_documento.pdf
```

Sin tocar Python.

## Licencia

MIT (ajustar según necesidad).
