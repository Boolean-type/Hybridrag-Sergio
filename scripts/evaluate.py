"""
Evaluación end-to-end del pipeline RAG-KG (opción A: gold por hechos).

Para cada query de `src/ragkg/evaluation/test_queries.yaml` (y sus paráfrasis):
  1. Ejecuta la recuperación híbrida real + genera respuesta con el LLM.
  2. Mide recall de hechos gold (determinista) y, si se activa, juzga con LLM.
  3. Combina en OK/KO + confianza anclada + localización del fallo.
  4. Imprime tabla y guarda el run en data/eval_runs/ para comparar regresiones.

Uso:
    python scripts/evaluate.py                       # con juez (necesita LLM)
    python scripts/evaluate.py --no-judge            # solo capa determinista
    python scripts/evaluate.py --only q1_java_microservices_banca,q2_dotnet_azure
    python scripts/evaluate.py --limit 3 --variants  # subconjunto + detalle por paráfrasis
    python scripts/evaluate.py --judge-model llama-3.3-70b-versatile
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from ragkg.config.loader import load_domain_config, load_yaml
from ragkg.embeddings.embedder import Embedder
from ragkg.evaluation.dataset import load_eval_dataset
from ragkg.evaluation.judge import LLMJudge
from ragkg.evaluation.metrics import build_alias_index
from ragkg.evaluation.report import render_report, save_report
from ragkg.evaluation.runner import AnswerBundle, Thresholds, run_evaluation
from ragkg.generation.answer_builder import build_context_prompt
from ragkg.extraction.ontology_extractor import build_llm_client
from ragkg.graph.neo4j_client import Neo4jClient
from ragkg.retrieval.hybrid_retriever import HybridRetriever

app = typer.Typer(add_completion=False)
console = Console()


def _collect_retrieved_facts(result) -> list[str]:
    """Nombres de entidades recuperadas (estructurales + expansión de grafo).

    Incluimos también `code`: algunas entidades (p. ej. FunctionalRequirement)
    guardan su valor verificable ahí (RF09, ...), no en name/canonical_name. Sin
    esto, el gold por código nunca casa con lo recuperado y graph_recall sale 0.
    """
    facts: set[str] = set()
    for e in result.structured_entities:
        for key in ("name", "code"):
            val = e.get(key)
            if val:
                facts.add(val)
    for item in result.graph_context:
        entity = item.get("entity") or {}
        for key in ("canonical_name", "name", "code"):
            val = entity.get(key)
            if val:
                facts.add(val)
    return sorted(facts)


@app.command()
def evaluate(
    domain: str = typer.Option("offers", help="Dominio (para localizar eval_rubric.yaml)."),
    dataset_path: str = typer.Option(
        "src/ragkg/evaluation/test_queries.yaml", help="Ruta del dataset de queries."
    ),
    use_judge: bool = typer.Option(True, "--judge/--no-judge", help="Activar la capa LLM-juez."),
    judge_model: str | None = typer.Option(None, help="Modelo del juez (default: JUDGE_LLM_MODEL/LLM_MODEL)."),
    only: str | None = typer.Option(None, help="IDs concretos separados por coma."),
    limit: int | None = typer.Option(None, help="Evaluar solo los primeros N casos."),
    top_k: int = typer.Option(15, help="Tope de chunks por estrategia."),
    max_context_chunks: int = typer.Option(5, help="Chunks en el prompt del LLM."),
    variants: bool = typer.Option(False, "--variants", help="Mostrar detalle por paráfrasis."),
    save: bool = typer.Option(True, "--save/--no-save", help="Guardar el run en disco."),
) -> None:
    load_dotenv()

    dataset = load_eval_dataset(dataset_path)
    ids = [s.strip() for s in only.split(",")] if only else None
    dataset = dataset.subset(ids=ids, limit=limit)
    if not dataset.cases:
        console.print("[red]No hay casos que evaluar con esos filtros.[/red]")
        raise typer.Exit(1)

    rubric = load_yaml(Path(f"configs/domains/{domain}/eval_rubric.yaml"))
    thresholds = Thresholds.from_config(rubric)

    # Índice de alias del dominio: hace la capa determinista tolerante a sinónimos
    # (.NET≈dotnet) y formato (RF 01≈RF01), reutilizando normalization.yaml.
    try:
        alias_index = build_alias_index(load_domain_config(domain).normalization)
    except Exception:  # noqa: BLE001
        alias_index = {}

    judge = None
    if use_judge:
        model = judge_model or os.getenv("JUDGE_LLM_MODEL") or os.getenv("LLM_MODEL")
        jcfg = (rubric or {}).get("judge", {})
        judge = LLMJudge(
            model=model,
            system_prompt=jcfg.get("system_prompt"),
            criteria=jcfg.get("criteria"),
        )

    embedder = Embedder()
    answer_llm = build_llm_client(json_mode=False)

    with Neo4jClient.from_env() as client:
        retriever = HybridRetriever(client, embedder)

        def answer_fn(question: str) -> AnswerBundle:
            result = retriever.retrieve(question, top_k=top_k)
            prompt = build_context_prompt(result, max_chunks=max_context_chunks)
            try:
                answer = answer_llm.generate(prompt).strip()
            except Exception as exc:  # noqa: BLE001
                answer = f"[ERROR LLM: {exc}]"
            primary = result.fused_results or result.vector_results
            return AnswerBundle(
                answer_text=answer,
                retrieved_facts=_collect_retrieved_facts(result),
                context_excerpt=prompt,
                num_chunks=len(primary[:max_context_chunks]),
            )

        report = run_evaluation(dataset, answer_fn, judge=judge, thresholds=thresholds, alias_index=alias_index)

    render_report(report, console=console, show_variants=variants)

    if save:
        path = save_report(
            report,
            meta={
                "answer_model": os.getenv("LLM_MODEL"),
                "judge_model": (judge_model or os.getenv("JUDGE_LLM_MODEL") or os.getenv("LLM_MODEL")) if use_judge else None,
                "top_k": top_k,
                "max_context_chunks": max_context_chunks,
                "chunk_size": os.getenv("CHUNK_SIZE"),
            },
        )
        console.print(f"\n[dim]Run guardado en {path}[/dim]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
