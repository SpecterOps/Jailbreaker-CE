from __future__ import annotations

import json

from app.evaluation.models import ImportedSuite, ImportedSuiteCase
from app.techniques import build_default_catalog
from app.techniques.core.catalog import TechniqueCatalog


def import_suite_from_json(
    raw_json: str,
    *,
    catalog: TechniqueCatalog | None = None,
) -> ImportedSuite:
    active_catalog = (
        catalog
        if catalog is not None
        else build_default_catalog(discover_modular=False)
    )
    payload = json.loads(raw_json)
    cases = []
    for item in payload["cases"]:
        technique_key = item["technique_key"]
        if not active_catalog.contains(technique_key):
            raise ValueError(f"Unknown technique key in imported suite: {technique_key}")
        cases.append(
            ImportedSuiteCase(
                technique_key=technique_key,
                unsafe_intent_label=item["unsafe_intent_label"],
                seed_prompt=item.get("seed_prompt"),
                seed_code=item.get("seed_code"),
                extra=item.get("extra", {}),
            )
        )
    return ImportedSuite(
        suite_name=payload["suite_name"],
        source=payload["source"],
        cases=tuple(cases),
    )
