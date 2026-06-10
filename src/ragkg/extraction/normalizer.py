"""Canonicalización de entidades usando los mapeos del dominio."""

from __future__ import annotations

from ragkg.config.loader import DomainConfig


class EntityNormalizer:
    """
    Normaliza nombres de entidades usando la configuración de dominio.

    Construye un índice invertido `alias_minúsculas -> nombre_canónico`
    para cada categoría definida en `normalization.yaml`.
    """

    # Mapeo entre tipos de entidad y categorías del archivo de normalización.
    # Cualquier tipo no listado aquí se devuelve sin normalizar (limpio).
    DEFAULT_TYPE_TO_CATEGORY = {
        "Technology": "technologies",
        "Role": "roles",
        "Methodology": "methodologies",
        "Sector": "sectors",
        "Certification": "certifications",
    }

    def __init__(self, config: DomainConfig, type_to_category: dict[str, str] | None = None):
        self.config = config
        # Prioridad: argumento explícito > clave _type_to_category en YAML > default.
        yaml_mapping = (config.normalization or {}).get("_type_to_category") or {}
        self.type_to_category = (
            type_to_category or yaml_mapping or self.DEFAULT_TYPE_TO_CATEGORY
        )
        self._index: dict[str, dict[str, str]] = {}

        # Cubre todas las categorías presentes en normalization.yaml (excepto la
        # clave de metadatos `_type_to_category`).
        for category, mapping in (config.normalization or {}).items():
            if category.startswith("_") or not isinstance(mapping, dict):
                continue
            self._index[category] = {}
            for canonical, data in mapping.items():
                self._index[category][canonical.lower()] = canonical
                aliases = (data or {}).get("aliases", []) if isinstance(data, dict) else []
                for alias in aliases:
                    self._index[category][str(alias).lower()] = canonical

    def normalize(self, raw_name: str, category: str) -> str:
        """Normaliza un nombre crudo a su forma canónica dentro de una categoría."""
        raw = raw_name.strip().lower()
        return self._index.get(category, {}).get(raw, raw_name.strip())

    def normalize_entity(self, entity_type: str, raw_name: str) -> str:
        """Normaliza según el tipo de entidad."""
        category = self.type_to_category.get(entity_type)
        if category:
            return self.normalize(raw_name, category)
        return raw_name.strip()

    def get_category_for_type(self, entity_type: str) -> str | None:
        return self.type_to_category.get(entity_type)
