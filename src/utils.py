"""
utils.py — Shared helpers for the HiLabs evaluation framework.

Real data notes (from inspecting drive/ folder):
- Dataset lives in drive/  (30 subfolders, each with .md + .json)
- metadata_from_qa uses a 'relations' list of dicts:
    [ {entity, entity_score, entity_type (e.g. STRENGTH/DOSE/ROUTE), entity_span, entity_source}, ... ]
  plus a 'count' int.  NOT flat key-value pairs.
"""

import json
import os
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """Load and return a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data as a formatted JSON file."""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_text(path: str | Path) -> str:
    """Load and return a plain text / markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def discover_test_data(test_data_dir: str | Path = "drive") -> list[dict]:
    """
    Walk the data directory and return a list of dicts:
        { "name": folder_name, "md_path": ..., "json_path": ... }
    Only includes folders that have BOTH a .md and a .json file.
    Defaults to 'drive/' which is where the real HiLabs dataset lives.
    """
    records = []
    root = Path(test_data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Data directory not found: {root.resolve()}\n"
            f"Expected 'drive/' folder with 30 patient subfolders."
        )

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        md_files   = list(folder.glob("*.md"))
        json_files = list(folder.glob("*.json"))
        if md_files and json_files:
            records.append(
                {
                    "name":      folder.name,
                    "md_path":  str(md_files[0]),
                    "json_path": str(json_files[0]),
                }
            )
    return records


def extract_metadata_flat(metadata_from_qa: dict) -> dict:
    """
    Convert the real metadata_from_qa format (relations list) to a flat
    {entity_type: entity_value} dict for easy comparison.

    Real format:
        {
          "relations": [
            {"entity": "250", "entity_score": 1.0, "entity_type": "STRENGTH", ...},
            {"entity": "mg",  "entity_score": 1.0, "entity_type": "UNIT", ...},
          ],
          "count": 2
        }

    Returns: {"STRENGTH": "250", "UNIT": "mg", ...}
    Empty dict if no relations.
    """
    if not isinstance(metadata_from_qa, dict):
        return {}
    relations = metadata_from_qa.get("relations", [])
    if not isinstance(relations, list):
        return {}
    flat = {}
    for rel in relations:
        if isinstance(rel, dict):
            etype = str(rel.get("entity_type", "")).upper()
            etext = str(rel.get("entity",      "")).strip()
            if etype and etext:
                # Keep first occurrence for each type
                if etype not in flat:
                    flat[etype] = etext
    return flat


def normalise_entity_list(entities: Any) -> list[dict]:
    """
    Accept either:
      - a list of entity dicts  (standard format)
      - a dict with a top-level key like 'entities' / 'data' / 'results'
    Returns a flat list of entity dicts.
    """
    if isinstance(entities, list):
        return entities
    if isinstance(entities, dict):
        for key in ("entities", "data", "results", "items"):
            if key in entities and isinstance(entities[key], list):
                return entities[key]
        # single entity wrapped in a dict?
        if "entity" in entities:
            return [entities]
    return []


ENTITY_TYPES = [
    "IMMUNIZATION",
    "MEDICAL_DEVICE",
    "MEDICINE",
    "MENTAL_STATUS",
    "PROBLEM",
    "PROCEDURE",
    "SDOH",
    "SOCIAL_HISTORY",
    "TEST",
    "VITAL_NAME",
]

ASSERTION_VALUES = ["POSITIVE", "NEGATIVE", "UNCERTAIN"]
TEMPORALITY_VALUES = ["CLINICAL_HISTORY", "CURRENT", "UNCERTAIN", "UPCOMING"]
SUBJECT_VALUES = ["PATIENT", "FAMILY_MEMBER"]

CATEGORICAL_FIELDS = {
    "entity_type": ENTITY_TYPES,
    "assertion": ASSERTION_VALUES,
    "temporality": TEMPORALITY_VALUES,
    "subject": SUBJECT_VALUES,
}

# Metadata relation types seen in the real dataset
METADATA_FIELDS = [
    "STRENGTH", "UNIT", "DOSE", "ROUTE", "FREQUENCY",
    "FORM", "DURATION", "STATUS",
    "exact_date", "derived_date",
    "TEST_VALUE", "TEST_UNIT",
    "VITAL_NAME_UNIT", "VITAL_NAME_VALUE",
]

# All relation entity_type values seen in the real dataset
RELATION_TYPES = [
    "STRENGTH", "UNIT", "DOSE", "ROUTE", "FREQUENCY",
    "FORM", "DURATION", "STATUS",
]
