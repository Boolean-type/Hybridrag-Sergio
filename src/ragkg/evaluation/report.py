"""Render y persistencia del informe de evaluación."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from ragkg.evaluation.runner import EvalReport


def render_report(report: EvalReport, console: Console | None = None, show_variants: bool = False) -> None:
    console = console or Console()
    s = report.summary

    console.rule("[bold cyan]Resultado de evaluación[/bold cyan]")
    console.print(
        f"Dominio: [bold]{report.domain}[/bold] · dataset v{report.dataset_version} · "
        f"juez: [bold]{'ON' if report.judge_enabled else 'OFF'}[/bold]"
    )
    console.print(
        f"Accuracy: [bold green]{s['accuracy']:.0%}[/bold green] "
        f"({s['ok']} OK / {s['ko']} KO / {s['skipped_cases']} skip) · "
        f"consistencia: {s['consistency_rate']:.0%} · "
        f"confianza media: {s['mean_confidence']}%\n"
    )

    table = Table(title="Casos")
    table.add_column("ID", style="cyan", overflow="fold")
    table.add_column("Tipo", style="dim", width=10)
    table.add_column("Veredicto", width=9)
    table.add_column("Conf.", justify="right", width=6)
    table.add_column("Pass", justify="right", width=6)
    table.add_column("Cons.", justify="center", width=6)
    table.add_column("Fallo", style="yellow", width=11)

    for c in report.cases:
        color = {"OK": "green", "KO": "red", "SKIPPED": "dim"}.get(c.verdict, "white")
        loci = {v.failure_locus for v in c.variants if v.failure_locus != "none"}
        locus_str = ",".join(sorted(loci)) if loci else "-"
        table.add_row(
            c.id, c.type,
            f"[{color}]{c.verdict}[/{color}]",
            f"{c.mean_confidence}%",
            f"{c.pass_rate:.0%}",
            "✓" if c.consistent else "✗",
            locus_str,
        )
    console.print(table)

    if show_variants:
        for c in report.cases:
            console.print(f"\n[bold]{c.id}[/bold] ({c.verdict})")
            for v in c.variants:
                tag = "P" if v.is_paraphrase else "O"
                vc = {"OK": "green", "KO": "red", "SKIPPED": "dim"}.get(v.verdict, "white")
                console.print(
                    f"  [{tag}] [{vc}]{v.verdict}[/{vc}] {v.confidence}% — {v.question}"
                )
                console.print(f"      [dim]{v.justification}[/dim]")


def save_report(report: EvalReport, out_dir: str | Path = "data/eval_runs", meta: dict[str, Any] | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"eval_{report.domain}_{ts}.json"

    payload = {
        "timestamp": ts,
        "meta": meta or {},
        **report.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
