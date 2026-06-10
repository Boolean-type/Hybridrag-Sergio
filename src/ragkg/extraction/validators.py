"""Modelos Pydantic que validan la salida del extractor."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ExtractedEntity(BaseModel):
    temp_id: str
    type: str
    name: str
    canonical_name: str
    evidence: str
    confidence: float
    properties: dict = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence debe estar entre 0.0 y 1.0, recibido: {v}")
        return v

    @field_validator("evidence")
    @classmethod
    def _evidence_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("evidence no puede estar vacía")
        return v


class ExtractedRelation(BaseModel):
    source: str
    type: str
    target: str
    evidence: str
    confidence: float
    properties: dict = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence debe estar entre 0.0 y 1.0, recibido: {v}")
        return v


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    def filter_by_confidence(self, min_confidence: float = 0.5) -> ExtractionResult:
        """Filtra entidades y relaciones por umbral de confianza."""
        kept_ids = {e.temp_id for e in self.entities if e.confidence >= min_confidence}
        return ExtractionResult(
            entities=[e for e in self.entities if e.confidence >= min_confidence],
            relations=[
                r
                for r in self.relations
                if r.confidence >= min_confidence
                # No mantener relaciones que apuntan a entidades descartadas
                and (r.source in kept_ids or not r.source.startswith(("tech_", "role_", "ent_")))
                and (r.target in kept_ids or not r.target.startswith(("tech_", "role_", "ent_")))
            ],
        )

    def validate_against_config(self, allowed_entity_types: set[str], allowed_relation_names: set[str]) -> list[str]:
        """Devuelve una lista de problemas encontrados (vacía si todo OK)."""
        problems: list[str] = []
        for e in self.entities:
            if e.type not in allowed_entity_types:
                problems.append(f"Entidad con type inválido: '{e.type}' (temp_id={e.temp_id})")
        for r in self.relations:
            if r.type not in allowed_relation_names:
                problems.append(f"Relación con type inválido: '{r.type}' ({r.source}->{r.target})")
        return problems
