"""Carga y validación de archivos YAML de configuración."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DomainConfig(BaseModel):
    """Configuración completa de un dominio (semántica + ontología + relaciones + normalización)."""

    domain_name: str
    semantic_definitions: dict[str, Any] = Field(default_factory=dict)
    ontology: dict[str, Any] = Field(default_factory=dict)
    relations: dict[str, Any] = Field(default_factory=dict)
    extraction_schema: dict[str, Any] = Field(default_factory=dict)
    normalization: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ontology")
    @classmethod
    def _ontology_has_entities(cls, v: dict) -> dict:
        if "entities" not in v:
            raise ValueError("ontology.yaml debe contener la clave 'entities'")
        return v

    @field_validator("relations")
    @classmethod
    def _relations_has_list(cls, v: dict) -> dict:
        if "relations" not in v:
            raise ValueError("relations.yaml debe contener la clave 'relations'")
        return v

    # ------------------------------------------------------------------ helpers

    def get_entity_types(self) -> list[str]:
        """Tipos de entidad definidos en la ontología."""
        return list(self.ontology.get("entities", {}).keys())

    def get_entity_definition(self, entity_type: str) -> dict[str, Any]:
        return self.ontology.get("entities", {}).get(entity_type, {})

    def get_entity_label(self, entity_type: str) -> str:
        return self.get_entity_definition(entity_type).get("label", entity_type)

    def get_entity_id_field(self, entity_type: str) -> str:
        """Primer id_field declarado para una entidad (suele haber uno)."""
        fields = self.get_entity_definition(entity_type).get("id_fields", [])
        if not fields:
            raise ValueError(f"La entidad '{entity_type}' no tiene id_fields.")
        return fields[0]

    def get_allowed_relations(self) -> list[dict[str, Any]]:
        return self.relations.get("relations", [])

    def get_allowed_relation_names(self) -> set[str]:
        return {r["name"] for r in self.get_allowed_relations()}

    def is_relation_allowed(self, relation_name: str, source_type: str, target_type: str) -> bool:
        """Comprueba si una relación está permitida entre dos tipos."""
        for rel in self.get_allowed_relations():
            if rel["name"] != relation_name:
                continue
            if rel["source"] != source_type:
                continue
            target = rel["target"]
            if isinstance(target, list):
                if target_type in target:
                    return True
            elif target == target_type:
                return True
        return False

    def get_normalization_map(self, category: str) -> dict[str, Any]:
        return self.normalization.get(category, {})


# --------------------------------------------------------------------- loading


def load_yaml(path: Path) -> dict[str, Any]:
    """Carga un archivo YAML con manejo de errores. Si no existe, devuelve {}."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_domain_config(
    domain: str,
    configs_root: str | Path = "configs/domains",
) -> DomainConfig:
    """Carga toda la configuración de un dominio."""
    base = Path(configs_root) / domain
    if not base.exists():
        raise FileNotFoundError(f"Dominio no encontrado: {base}")

    return DomainConfig(
        domain_name=domain,
        semantic_definitions=load_yaml(base / "semantic_definitions.yaml"),
        ontology=load_yaml(base / "ontology.yaml"),
        relations=load_yaml(base / "relations.yaml"),
        extraction_schema=load_yaml(base / "extraction_schema.yaml"),
        normalization=load_yaml(base / "normalization.yaml"),
    )


def load_base_config(configs_root: str | Path = "configs/base") -> dict[str, Any]:
    """Carga las configuraciones base (pipeline, llm, embeddings)."""
    base = Path(configs_root)
    return {
        "pipeline": load_yaml(base / "pipeline.yaml"),
        "llm": load_yaml(base / "llm.yaml"),
        "embeddings": load_yaml(base / "embeddings.yaml"),
    }
