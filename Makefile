.PHONY: help install neo4j-up neo4j-down neo4j-logs check schema reset ingest ingest-sample query eval eval-quick test lint format clean

PYTHON ?= python
DOMAIN ?= offers
# Por defecto, ingest-sample apunta a la oferta de Entelgy. Sobrescribir con SAMPLE_FILE=.
SAMPLE_FILE ?= data/samples/Entelgy_Oferta_tecnica.pdf

help:
	@echo Targets disponibles:
	@echo   make install                    Instala dependencias en modo editable + dev
	@echo   make neo4j-up                   Levanta Neo4j en Docker
	@echo   make neo4j-down                 Detiene Neo4j
	@echo   make neo4j-logs                 Muestra los logs de Neo4j
	@echo   make check                      Verifica la conexion a Neo4j
	@echo   make schema                     Crea constraints e indice vectorial
	@echo   make reset                      Vacia toda la base de datos
	@echo   make ingest-sample              Ingiere SAMPLE_FILE (por defecto el PDF de Entelgy)
	@echo   make ingest FILE=ruta/archivo   Ingiere un archivo concreto
	@echo   make query Q="..."              Lanza una consulta hibrida
	@echo   make eval                       Evaluacion end-to-end (con juez LLM)
	@echo   make eval-quick                 Evaluacion solo determinista, 3 casos
	@echo   make test                       Ejecuta los tests
	@echo   make lint                       Ejecuta ruff
	@echo   make format                     Formatea el codigo con ruff
	@echo   make clean                      Elimina caches

install:
	pip install -e ".[dev]"

neo4j-up:
	docker compose up -d

neo4j-down:
	docker compose down

neo4j-logs:
	docker compose logs -f neo4j

check:
	$(PYTHON) scripts/check_connection.py

schema:
	$(PYTHON) scripts/create_schema.py --domain $(DOMAIN)

reset:
	$(PYTHON) scripts/reset_neo4j.py

# Ingiere el archivo de muestra (por defecto el PDF de Entelgy).
# Sobrescribir con:  make ingest-sample SAMPLE_FILE=otro_archivo.pdf
ingest-sample:
	$(PYTHON) scripts/ingest.py "$(SAMPLE_FILE)" --domain $(DOMAIN)

# Ingesta generica. Uso:  make ingest FILE=data/raw/offers/mi_oferta.pdf
# Si FILE esta vacio, scripts/ingest.py mostrara el help de Typer.
ingest:
	$(PYTHON) scripts/ingest.py "$(FILE)" --domain $(DOMAIN)

# Consulta hibrida. Uso:  make query Q="tu pregunta aqui"
# Si Q esta vacio, scripts/query.py mostrara el help de Typer.
query:
	$(PYTHON) scripts/query.py "$(Q)"


# Evaluacion end-to-end (recall de hechos + juez LLM).
eval:
	$(PYTHON) scripts/evaluate.py --domain $(DOMAIN)

# Evaluacion rapida sin juez, subconjunto, util en tier free.
eval-quick:
	$(PYTHON) scripts/evaluate.py --domain $(DOMAIN) --no-judge --limit 3 --variants

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.ruff_cache', '.mypy_cache', 'build', 'dist'] + list(pathlib.Path('.').rglob('__pycache__')) + list(pathlib.Path('.').rglob('*.egg-info'))]"
