"""Tests para la normalización de entidades (dominio offers v0.2.0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragkg.config.loader import load_domain_config
from ragkg.extraction.normalizer import EntityNormalizer

CONFIGS_ROOT = Path(__file__).parent.parent / "configs" / "domains"


@pytest.fixture(scope="module")
def normalizer() -> EntityNormalizer:
    config = load_domain_config("offers", configs_root=CONFIGS_ROOT)
    return EntityNormalizer(config)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("java", "Java"),
        ("JDK 17", "Java"),
        ("OpenJDK", "Java"),
        ("dotnet", ".NET"),
        ("net core", ".NET"),
        ("springboot", "Spring Boot"),
        ("postgres", "PostgreSQL"),
        ("K8S", "Kubernetes"),
        ("amazon web services", "AWS"),
        # Servicios Azure
        ("AKS", "Azure Kubernetes Service"),
        ("azure openai", "Azure OpenAI"),
        ("cognitive search", "Azure AI Search"),
        ("azure ai foundry", "Azure AI Foundry"),
        ("blob storage", "Azure Blob Storage"),
        ("azure ad", "Azure Active Directory"),
        ("key vault", "Azure Key Vault"),
        ("app insights", "Azure Application Insights"),
        ("apim", "Azure API Management"),
        ("acr", "Azure Container Registry"),
        # Modelos
        ("gpt-4", "GPT-4"),
        ("text-embedding-3-large", "text-embedding-3-large"),
        # OpenShift
        ("openshift", "OpenShift"),
        ("aro", "OpenShift"),
        # Calidad
        ("sonar", "SonarQube"),
        ("apache jmeter", "JMeter"),
        # BI / CRM / Colaboración
        ("powerbi", "Power BI"),
        ("ms dynamics", "Dynamics 365"),
        ("share point", "SharePoint"),
    ],
)
def test_normalize_technologies(normalizer, raw, expected):
    assert normalizer.normalize_entity("Technology", raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("rag", "RAG"),
        ("retrieval augmented generation", "RAG"),
        ("ajuste fino", "Fine-tuning"),
        ("rft", "RFT"),
        ("dpo", "DPO"),
        ("speech to text", "STT"),
        ("voz a texto", "STT"),
        ("síntesis de voz", "TTS"),
        ("cqa", "Custom Question Answering"),
        ("búsqueda semántica", "Vector Search"),
    ],
)
def test_normalize_ai_concepts(normalizer, raw, expected):
    assert normalizer.normalize_entity("AIConcept", raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("devops engineer", "DevOps Engineer"),
        ("Backend Engineer", "Backend Developer"),
        ("Tech Lead", "Tech Lead"),
        ("PM", "Jefe de Proyecto"),
        ("ai consultant", "Consultor IA"),
        ("documentalist", "Documentalista"),
        ("qa", "Técnico de Pruebas"),
        ("product owner", "Product Owner"),
    ],
)
def test_normalize_roles(normalizer, raw, expected):
    assert normalizer.normalize_entity("Role", raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ágil", "Agile"),
        ("scrum", "Scrum"),
        ("devsecops", "DevSecOps"),
        ("infraestructura como código", "IaC"),
        ("integración continua", "CI/CD"),
        ("user experience design", "UXD"),
    ],
)
def test_normalize_methodologies(normalizer, raw, expected):
    assert normalizer.normalize_entity("Methodology", raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ens", "ENS"),
        ("ens nivel alto", "ENS Nivel Alto"),
        ("rgpd", "GDPR"),
        ("ccn stic", "CCN-STIC"),
        ("owasp top 10", "OWASP Top 10"),
    ],
)
def test_normalize_compliance(normalizer, raw, expected):
    assert normalizer.normalize_entity("ComplianceFramework", raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("teléfono", "Teléfono"),
        ("correo electrónico", "Email"),
        ("whatsapp", "WhatsApp"),
        ("te llamamos", "Callback"),
        ("chat online", "Chat Web"),
    ],
)
def test_normalize_contact_channels(normalizer, raw, expected):
    assert normalizer.normalize_entity("ContactChannel", raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Banking", "Banca"),
        ("telco", "Telecomunicaciones"),
        ("tesoro", "Tesoro Público"),
        ("sector público", "Administración Pública"),
    ],
)
def test_normalize_sectors(normalizer, raw, expected):
    assert normalizer.normalize_entity("Sector", raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("psm i", "PSM I"),
        ("cka", "CKA"),
        ("itil v4 foundation", "ITIL Foundation"),
        ("az-104", "Azure Administrator"),
    ],
)
def test_normalize_certifications(normalizer, raw, expected):
    assert normalizer.normalize_entity("Certification", raw) == expected


def test_unknown_returns_clean_original(normalizer):
    assert normalizer.normalize_entity("Technology", "  RareTech  ") == "RareTech"


def test_unmapped_type_returns_clean_original(normalizer):
    # Entidad sin categoría declarada en _type_to_category → devuelve limpio.
    assert normalizer.normalize_entity("Bidder", "  Entelgy  ") == "Entelgy"


def test_type_to_category_loaded_from_yaml(normalizer):
    # Verifica que el mapeo viene del YAML (no del default hardcoded).
    assert normalizer.get_category_for_type("AIConcept") == "ai_concepts"
    assert normalizer.get_category_for_type("ComplianceFramework") == "compliance_frameworks"
    assert normalizer.get_category_for_type("ContactChannel") == "contact_channels"
