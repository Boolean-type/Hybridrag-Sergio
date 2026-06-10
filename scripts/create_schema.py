"""Crea constraints e índices en Neo4j. Idempotente.

Uso:
    python scripts/create_schema.py                # usa DOMAIN del .env
    python scripts/create_schema.py --domain offers
"""

from __future__ import annotations

import os
import sys

import typer
from dotenv import load_dotenv

from ragkg.config.loader import load_domain_config
from ragkg.graph.neo4j_client import Neo4jClient
from ragkg.graph.schema import create_schema

app = typer.Typer(add_completion=False)


@app.command()
def main(
    domain: str = typer.Option(
        None,
        "--domain",
        help="Dominio a usar. Por defecto la variable DOMAIN del .env.",
    ),
) -> None:
    load_dotenv()
    domain = domain or os.getenv("DOMAIN", "offers")
    print(f"Creando schema para dominio: {domain}")
    config = load_domain_config(domain)

    with Neo4jClient.from_env() as client:
        create_schema(client, config)
    print("✅ Esquema creado/verificado correctamente.")


if __name__ == "__main__":
    app()
