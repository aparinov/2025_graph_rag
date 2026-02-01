# -*- coding: utf-8 -*-
"""Text processing utilities."""

import re
from typing import Dict, Any


_LABEL_SAFE_RE = re.compile(r"[^0-9A-Za-z_\u0400-\u04FF]")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_key(text: str) -> str:
    """Normalize text to lowercase key."""
    return normalize_whitespace(text).lower()


def sanitize_session_name(name: str) -> str:
    """Sanitize session name for safe usage."""
    cleaned = normalize_whitespace(name)
    cleaned = cleaned.replace("`", "").replace('"', "'")
    return cleaned


def sanitize_label(value: str, fallback: str) -> str:
    """Sanitize label value."""
    cleaned = _LABEL_SAFE_RE.sub("_", value or "")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or fallback


def sanitize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize entity/relation properties."""
    cleaned: Dict[str, Any] = {}
    for key, value in (properties or {}).items():
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) for item in value
        ):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```", "", cleaned)
    return cleaned.strip()


def normalize_medical_term(term: str) -> str:
    """Normalize medical terminology for better deduplication."""
    term = term.lower().strip()
    # Remove dosage units from names for better deduplication
    # e.g., "ибупрофен 200мг" -> "ибупрофен"
    term = re.sub(r"\s*\d+\s*(мг|г|мл|таб|капс|мкг|ме|ед)\b", "", term)
    # Remove extra whitespace
    term = re.sub(r"\s+", " ", term).strip()
    return term


def build_entity_id(identifier: str, entity_name: str, entity_type: str) -> str:
    """Build unique entity ID.

    Args:
        identifier: Collection name or session name (used as namespace)
        entity_name: Name of the entity
        entity_type: Type of the entity

    Returns:
        Unique entity ID string
    """
    # Normalize medical terms for better deduplication
    normalized_name = normalize_medical_term(entity_name)
    return f"{normalize_key(identifier)}::{normalize_key(entity_type)}::{normalize_key(normalized_name)}"
