"""
Ingesta un documento (o un directorio) en Neo4j:

  - Carga + chunking
  - Embeddings
  - Extracción guiada por ontología (LLM)
  - Normalización
  - Upserts de Document, Chunk, entidades y relaciones

Uso:
    python scripts/ingest.py data/samples/oferta_ejemplo.md
    python scripts/ingest.py data/raw/offers/                  # ingesta todo el directorio
    python scripts/ingest.py oferta.pdf --domain offers --no-extract
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

from ragkg.config.loader import load_domain_config
from ragkg.embeddings.embedder import Embedder
from ragkg.extraction.normalizer import EntityNormalizer
from ragkg.extraction.ontology_extractor import build_llm_client, extract_from_chunk
from ragkg.graph.neo4j_client import Neo4jClient
from ragkg.graph.upsert import (
    link_chunk_mentions_entity,
    upsert_entity,
    upsert_relation,
)
from ragkg.ingestion.pipeline import ingest_path

app = typer.Typer(add_completion=False)
console = Console()


def _iter_files(target: Path, supported: set[str]) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in supported)


@app.command()
def ingest(
    target: Path = typer.Argument(..., help="Archivo o directorio a ingerir."),
    domain: str = typer.Option(None, help="Dominio. Por defecto la variable DOMAIN del .env."),
    chunk_size: int = typer.Option(1200, help="Tamaño de chunk en caracteres."),
    overlap: int = typer.Option(200, help="Solapamiento entre chunks."),
    extract: bool = typer.Option(True, "--extract/--no-extract", help="Ejecutar extracción LLM."),
    min_confidence: float = typer.Option(0.5, help="Umbral mínimo de confianza para persistir."),
) -> None:
    load_dotenv()
    domain = domain or os.getenv("DOMAIN", "offers")
    config = load_domain_config(domain)
    normalizer = EntityNormalizer(config)
    embedder = Embedder()
    llm = build_llm_client() if extract else None

    supported = {".txt", ".md", ".markdown", ".pdf", ".docx"}
    files = _iter_files(target, supported)
    if not files:
        console.print(f"[red]No se encontraron archivos válidos en {target}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Dominio:[/bold] {domain}")
    console.print(f"[bold]Archivos a procesar:[/bold] {len(files)}")
    console.print(f"[bold]Extracción LLM:[/bold] {'sí' if extract else 'no'}")

    successes = 0
    failures = 0

    with Neo4jClient.from_env() as client:
        for idx, file_path in enumerate(files, 1):
            console.print(f"\n[bold]({idx}/{len(files)}) {file_path.name}[/bold]")
            try:
                document, chunks = ingest_path(
                    path=file_path,
                    config=config,
                    client=client,
                    embedder=embedder,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                console.print(
                    f"  ✓ Cargado y troceado → doc_id={document.doc_id}, chunks={len(chunks)}"
                )

                if extract and llm and chunks:
                    _extract_and_upsert(client, config, normalizer, llm, chunks, min_confidence)

                successes += 1

            except Exception as exc:  # noqa: BLE001
                console.print(f"  ✗ [red]{file_path.name}: {exc}[/red]")
                failures += 1

    if failures == 0:
        console.print(f"[bold green]✅ Ingesta completada: {successes}/{len(files)} archivos.[/bold green]")
    elif successes == 0:
        console.print(f"[bold red]❌ Ingesta fallida: 0/{len(files)} archivos procesados.[/bold red]")
        raise typer.Exit(code=1)
    else:
        console.print(
            f"[bold yellow]⚠ Ingesta parcial: {successes} ok, {failures} con error "
            f"(de {len(files)} archivos).[/bold yellow]"
        )
        raise typer.Exit(code=2)


def _extract_and_upsert(client, config, normalizer, llm, chunks, min_confidence):
    """Para cada chunk: extrae entidades+relaciones y las persiste.

    Respeta LLM_CHUNK_DELAY_SECONDS (default 0) entre llamadas para no saturar
    el rate limit de proveedores con tier gratuito (p.ej. Groq free).

    Muestra una barra de progreso real chunk a chunk con contadores.
    """
    import os
    import time

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    allowed_entities = set(config.get_entity_types())
    allowed_relations = config.get_allowed_relation_names()
    delay = float(os.getenv("LLM_CHUNK_DELAY_SECONDS", "0"))
    entities_total = 0
    relations_total = 0
    chunk_errors = 0

    progress = Progress(
        TextColumn("  [progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("/"),
        TimeRemainingColumn(),
        TextColumn("• ents={task.fields[ents]} rels={task.fields[rels]} err={task.fields[err]}"),
        console=console,
    )

    with progress:
        task = progress.add_task("Extrayendo entidades", total=len(chunks), ents=0, rels=0, err=0)

        for i, chunk in enumerate(chunks, 1):
            if i > 1 and delay > 0:
                time.sleep(delay)

            try:
                result = extract_from_chunk(chunk.text, config, llm, min_confidence=min_confidence)
            except Exception as exc:  # noqa: BLE001
                chunk_errors += 1
                console.print(f"    [yellow]LLM error en chunk {i}/{len(chunks)}: {exc}[/yellow]")
                progress.update(task, advance=1, err=chunk_errors)
                continue

            problems = result.validate_against_config(allowed_entities, allowed_relations)
            for p in problems:
                console.print(f"    [dim yellow]Aviso: {p}[/dim yellow]")

            # Mapa temp_id -> (label, id_field, canonical_value) para resolver relaciones
            temp_to_persisted: dict[str, tuple[str, str, str]] = {}

            # --- Persistir entidades ---
            for entity in result.entities:
                if entity.type not in allowed_entities:
                    continue

                label = config.get_entity_label(entity.type)
                id_field = config.get_entity_id_field(entity.type)
                canonical = normalizer.normalize_entity(entity.type, entity.canonical_name or entity.name)
                properties = {
                    **entity.properties,
                    "confidence": entity.confidence,
                }
                if id_field == "canonical_name":
                    properties["canonical_name"] = canonical

                upsert_entity(client, label, id_field, canonical, properties)
                temp_to_persisted[entity.temp_id] = (label, id_field, canonical)
                entities_total += 1

                # Trazabilidad: chunk → entidad
                link_chunk_mentions_entity(
                    client,
                    chunk_id=chunk.chunk_id,
                    entity_label=label,
                    entity_id_field=id_field,
                    entity_id=canonical,
                    evidence=entity.evidence,
                    confidence=entity.confidence,
                )

            # --- Persistir relaciones ---
            for relation in result.relations:
                if relation.type not in allowed_relations:
                    continue
                if relation.source not in temp_to_persisted or relation.target not in temp_to_persisted:
                    continue

                src_label, src_field, src_id = temp_to_persisted[relation.source]
                tgt_label, tgt_field, tgt_id = temp_to_persisted[relation.target]

                if not config.is_relation_allowed(relation.type, src_label, tgt_label):
                    continue

                upsert_relation(
                    client,
                    source_label=src_label,
                    source_id_field=src_field,
                    source_id=src_id,
                    relation_type=relation.type,
                    target_label=tgt_label,
                    target_id_field=tgt_field,
                    target_id=tgt_id,
                    properties={
                        **relation.properties,
                        "evidence": relation.evidence,
                        "confidence": relation.confidence,
                    },
                )
                relations_total += 1

            progress.update(task, advance=1, ents=entities_total, rels=relations_total)

    console.print(
        f"    [bold]→ Resumen:[/bold] {entities_total} entidades, "
        f"{relations_total} relaciones, {chunk_errors} chunks con error"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
