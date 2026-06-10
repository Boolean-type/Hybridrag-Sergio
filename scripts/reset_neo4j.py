"""Limpia toda la base de datos Neo4j. ÚSALO CON CUIDADO."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from ragkg.graph.neo4j_client import Neo4jClient


def main() -> int:
    load_dotenv()
    confirm = input("⚠️  Vas a borrar TODOS los nodos y relaciones. Escribe 'yes' para continuar: ")
    if confirm.strip().lower() != "yes":
        print("Cancelado.")
        return 1

    with Neo4jClient.from_env() as client:
        client.run("MATCH (n) DETACH DELETE n")
        # Borrar índices y constraints (opcional)
        # Comentado por defecto para no reconstruir todo el tiempo.
        # for row in client.run("SHOW CONSTRAINTS YIELD name"):
        #     client.run(f"DROP CONSTRAINT {row['name']}")
        # for row in client.run("SHOW INDEXES YIELD name"):
        #     client.run(f"DROP INDEX {row['name']}")

    print("✅ Base de datos vaciada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
