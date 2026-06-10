"""
Consulta híbrida (vector + BM25 + Cypher estructural + grafo) sobre el grafo.

Por defecto:
  1. Embed de la pregunta
  2. Tres búsquedas en paralelo: vector, BM25, Cypher estructural
  3. Reciprocal Rank Fusion de los rankings
  4. Expansión por grafo desde los chunks fusionados
  5. Construye un prompt con contexto rico
  6. Llama al LLM para generar la respuesta final

Uso:
    python scripts/query.py "qué tecnologías propone esta oferta"
    python scripts/query.py "..." --top-k 20 --no-expand
    python scripts/query.py "..." --raw            # sin LLM
    python scripts/query.py "..." --json           # salida estructurada
    python scripts/query.py "..." --no-keyword     # desactivar BM25
    python scripts/query.py "..." --no-structured  # desactivar Cypher estructural
"""

from __future__ import annotations

import json

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ragkg.embeddings.embedder import Embedder
from ragkg.extraction.ontology_extractor import build_llm_client
from ragkg.generation.answer_builder import (
    build_answer_summary,
    build_context_prompt,
)
from ragkg.graph.neo4j_client import Neo4jClient
from ragkg.retrieval.hybrid_retriever import HybridRetriever

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def query(
    question: str = typer.Argument(..., help="Pregunta en lenguaje natural."),
    top_k: int = typer.Option(15, help="Tope de chunks por estrategia."),
    expand: bool = typer.Option(True, "--expand/--no-expand"),
    use_keyword: bool = typer.Option(True, "--keyword/--no-keyword", help="BM25 sobre full-text."),
    use_structured: bool = typer.Option(True, "--structured/--no-structured", help="Cypher por intención."),
    raw: bool = typer.Option(False, "--raw", help="Sin LLM, solo contexto."),
    json_output: bool = typer.Option(False, "--json", help="Salida JSON estructurada."),
    max_context_chunks: int = typer.Option(5, help="Chunks a meter en el prompt del LLM."),
) -> None:
    load_dotenv()
    embedder = Embedder()

    with Neo4jClient.from_env() as client:
        retriever = HybridRetriever(client, embedder)
        result = retriever.retrieve(
            question,
            top_k=top_k,
            expand_graph=expand,
            use_keyword=use_keyword,
            use_structured=use_structured,
        )

    if json_output:
        summary = build_answer_summary(result)
        if not raw:
            summary["answer"] = _generate_answer(result, max_context_chunks)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return

    console.rule(f"[bold cyan]{question}[/bold cyan]")

    # Mostrar intención detectada (o por qué no se detectó)
    if result.intent:
        console.print(
            f"[bold green]✓ Intención estructural detectada:[/bold green] "
            f"{result.intent['human_label']} → consulta Cypher sobre {result.intent['target_label']}"
        )
        if not result.structured_entities:
            console.print(
                f"  [yellow]Pero el grafo NO contiene entidades de tipo "
                f"'{result.intent['target_label']}'. Verifica con:[/yellow]\n"
                f"  [dim]MATCH (n:{result.intent['target_label']}) RETURN count(n)[/dim]"
            )
    else:
        if use_structured:
            console.print(
                "[dim]· No se detectó intención estructural en la pregunta. "
                "Se usará solo búsqueda vectorial + keyword.[/dim]"
            )

    # Si hubo consulta estructural exitosa, mostrarla primero
    if result.structured_entities:
        struct_table = Table(title=f"Resultado estructural directo ({len(result.structured_entities)})")
        struct_table.add_column("#", width=4)
        struct_table.add_column("Tipo", style="yellow")
        struct_table.add_column("Nombre", style="cyan")
        for i, e in enumerate(result.structured_entities[:20], 1):
            struct_table.add_row(str(i), e.get("label") or "?", e.get("name") or "?")
        console.print(struct_table)

    # Tabla de chunks fusionados
    primary = result.fused_results or result.vector_results
    source_names = {0: "vec", 1: "key", 2: "struct"}
    if primary:
        table = Table(title="Chunks fusionados (RRF)")
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", style="green", width=8)
        table.add_column("Origen", style="magenta", width=10)
        table.add_column("Source", style="cyan")
        table.add_column("Preview", overflow="fold")
        for i, r in enumerate(primary[:5], 1):
            score = f"{r.get('rrf_score', 0):.4f}"
            origins = ",".join(source_names.get(s, f"s{s}") for s in r.get("sources", []))
            table.add_row(
                str(i),
                score,
                origins,
                str(r.get("source_file") or "").split("/")[-1].split("\\")[-1],
                (r.get("text") or "")[:180].replace("\n", " "),
            )
        console.print(table)

    # Entidades del grafo
    if result.graph_context:
        seen: set[tuple[str, str]] = set()
        entities_by_label: dict[str, list[str]] = {}
        for item in result.graph_context:
            entity = item.get("entity") or {}
            label = entity.get("_label")
            name = entity.get("canonical_name") or entity.get("name")
            if not label or not name or (label, name) in seen:
                continue
            seen.add((label, name))
            entities_by_label.setdefault(label, []).append(name)

        if entities_by_label:
            console.print("\n[bold]Entidades relacionadas (grafo):[/bold]")
            for label, names in sorted(entities_by_label.items()):
                console.print(f"  [yellow]{label}[/yellow]: {', '.join(names[:10])}")

    # Respuesta del LLM
    if not raw:
        if not primary and not result.structured_entities:
            console.print("\n[yellow]No hay contexto para generar respuesta.[/yellow]")
            return
        console.print("\n[bold]Generando respuesta con el LLM...[/bold]")
        try:
            answer = _generate_answer(result, max_context_chunks)
            console.print(Panel(answer, title="Respuesta", border_style="green"))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error llamando al LLM: {exc}[/red]")
            console.print("[dim]Sugerencia: revisa tu .env (LLM_PROVIDER y GROQ_API_KEY / LLM_API_KEY).[/dim]")


def _generate_answer(result, max_context_chunks: int) -> str:
    """Llama al LLM con un prompt de contexto rico para generar respuesta natural."""
    llm = build_llm_client(json_mode=False)
    prompt = build_context_prompt(result, max_chunks=max_context_chunks)
    return llm.generate(prompt).strip()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
