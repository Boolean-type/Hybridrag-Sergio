"""Verifica que Neo4j está accesible con las credenciales del .env."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from ragkg.graph.neo4j_client import Neo4jClient


def main() -> int:
    load_dotenv()
    try:
        with Neo4jClient.from_env() as client:
            records = client.run("RETURN 'Conexión OK' AS msg")
            print(records[0]["msg"])
            return 0
    except Exception as exc:  # noqa: BLE001
        print(f"❌ No se pudo conectar a Neo4j: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
