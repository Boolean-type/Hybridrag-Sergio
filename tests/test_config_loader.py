"""Tests para el loader de configuración de dominios."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragkg.config.loader import load_domain_config

CONFIGS_ROOT = Path(__file__).parent.parent / "configs" / "domains"


def test_load_offers_domain():
    config = load_domain_config("offers", configs_root=CONFIGS_ROOT)
    assert config.domain_name == "offers"
    types = config.get_entity_types()
    assert "Technology" in types
    assert "Role" in types
    assert "Offer" in types
    assert len(config.get_allowed_relations()) > 0


def test_load_generic_domain():
    config = load_domain_config("generic", configs_root=CONFIGS_ROOT)
    assert config.domain_name == "generic"
    assert "Concept" in config.get_entity_types()


def test_unknown_domain_raises():
    with pytest.raises(FileNotFoundError):
        load_domain_config("nonexistent", configs_root=CONFIGS_ROOT)


def test_is_relation_allowed():
    config = load_domain_config("offers", configs_root=CONFIGS_ROOT)
    assert config.is_relation_allowed("REQUIRES_TECHNOLOGY", "Offer", "Technology") is True
    assert config.is_relation_allowed("HAS_ROLE", "Offer", "Role") is True
    # Una relación que no está permitida en este dominio:
    assert config.is_relation_allowed("REQUIRES_TECHNOLOGY", "Role", "Sector") is False


def test_entity_id_field():
    config = load_domain_config("offers", configs_root=CONFIGS_ROOT)
    assert config.get_entity_id_field("Technology") == "canonical_name"
    assert config.get_entity_id_field("Offer") == "offer_id"
