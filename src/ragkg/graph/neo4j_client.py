"""Wrapper fino sobre el driver de Neo4j."""

from __future__ import annotations

import os
from typing import Any

from neo4j import GraphDatabase


class Neo4jClient:
    """Cliente reutilizable para Neo4j."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = self._build_driver()

    def _build_driver(self):
        """Crea el driver silenciando los avisos 'UNRECOGNIZED' (p. ej. propiedades
        opcionales como `metadata` que no existen en todos los nodos).

        El nombre del parámetro cambió entre versiones del driver:
          - 5.21+/6.x: notifications_disabled_classifications
          - 5.7–5.20: notifications_disabled_categories
        Probamos en orden y caemos a un driver sin filtro si ninguno aplica.
        """
        auth = (self.user, self.password)
        for kwarg in ("notifications_disabled_classifications", "notifications_disabled_categories"):
            try:
                return GraphDatabase.driver(self.uri, auth=auth, **{kwarg: ["UNRECOGNIZED"]})
            except (TypeError, ValueError):
                continue
        return GraphDatabase.driver(self.uri, auth=auth)

    @classmethod
    def from_env(cls) -> Neo4jClient:
        return cls()

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, parameters: dict[str, Any] | None = None) -> list:
        records, _summary, _keys = self.driver.execute_query(
            query,
            parameters or {},
            database_=self.database,
        )
        return records

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
