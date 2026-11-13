from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

from app.attacklib.config import default_technique_config
from app.attacklib.registry import list_attack_techniques
from app.api.store import EvaluationStore
from app.importers import import_suite_from_json
from app.persistence import NoopRunPersistence
from app.persistence.technique_runtime import NoopTechniqueRuntimePersistence
from app.techniques import build_default_catalog
from app.techniques.contracts.definition import (
    ImplementationKind,
    TechniqueCapability,
    TechniqueDefinition,
    TechniqueIdentity,
    TechniqueImplementation,
)
from app.techniques.core.catalog import (
    RegisteredTechnique,
    TechniqueCatalog,
    TechniqueSelector,
)
from app.techniques.core.router import ExecutionRouter


def test_catalog_import_does_not_eagerly_cycle_through_plugin_loader() -> None:
    """The catalog must remain independently importable by low-level callers."""

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.techniques.core.catalog import TechniqueCatalog; print(TechniqueCatalog.__name__)",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "TechniqueCatalog"


def registered_probe(
    implementation_id: str,
    kind: ImplementationKind,
    *,
    technique_id: str = "probe",
    version: str = "1.0.0",
    aliases: tuple[str, ...] = (),
    manifest_hash: str = "a" * 64,
) -> RegisteredTechnique:
    definition = TechniqueDefinition(
        identity=TechniqueIdentity(
            id=technique_id,
            version=version,
            name=technique_id.replace("-", " ").title(),
            family="test",
        ),
        summary="Catalog fixture",
        capabilities=frozenset({TechniqueCapability.SINGLE_SHOT}),
        aliases=aliases,
    )
    implementation = TechniqueImplementation(
        id=implementation_id,
        technique_id=technique_id,
        technique_version=version,
        kind=kind,
        entrypoint=f"{implementation_id}.entrypoint",
    )
    return RegisteredTechnique(
        definition=definition,
        implementation=implementation,
        manifest_hash=manifest_hash,
        asset_hashes=MappingProxyType({}),
    )


def test_catalog_resolves_explicit_and_default_implementations() -> None:
    catalog = TechniqueCatalog()
    catalog.register(registered_probe("legacy", ImplementationKind.LEGACY))
    catalog.register(registered_probe("modular", ImplementationKind.WORKFLOW))
    catalog.set_default("probe", implementation_id="legacy")

    assert catalog.resolve(TechniqueSelector("probe")).implementation.id == "legacy"
    assert catalog.resolve(TechniqueSelector("probe", implementation_id="modular")).implementation.id == "modular"


def test_catalog_rejects_alias_collision() -> None:
    catalog = TechniqueCatalog()
    catalog.register(
        registered_probe(
            "legacy",
            ImplementationKind.LEGACY,
            aliases=("old-probe",),
        )
    )

    with pytest.raises(ValueError, match="alias 'old-probe' is already registered"):
        catalog.register(
            registered_probe(
                "other",
                ImplementationKind.WORKFLOW,
                technique_id="other",
                aliases=("old-probe",),
            )
        )


def test_catalog_does_not_mutate_published_version() -> None:
    catalog = TechniqueCatalog()
    first = registered_probe(
        "modular",
        ImplementationKind.WORKFLOW,
        manifest_hash="a" * 64,
    )
    catalog.register(first)

    with pytest.raises(ValueError, match="same version has different contents"):
        catalog.register(
            registered_probe(
                "modular",
                ImplementationKind.WORKFLOW,
                manifest_hash="b" * 64,
            )
        )


def test_catalog_version_selection_is_deterministic() -> None:
    catalog = TechniqueCatalog()
    catalog.register(
        registered_probe(
            "modular",
            ImplementationKind.WORKFLOW,
            version="2.0.0",
        )
    )
    catalog.register(
        registered_probe(
            "modular",
            ImplementationKind.WORKFLOW,
            version="10.0.0",
        )
    )
    catalog.set_default("probe", implementation_id="modular")

    assert catalog.resolve(TechniqueSelector("probe")).definition.identity.version == "10.0.0"
    assert catalog.resolve(TechniqueSelector("probe", technique_version="2.0.0")).definition.identity.version == "2.0.0"


def test_catalog_aliases_resolve_and_inventory_is_sorted() -> None:
    catalog = TechniqueCatalog()
    catalog.register(
        registered_probe(
            "legacy",
            ImplementationKind.LEGACY,
            technique_id="zeta",
        )
    )
    catalog.register(
        registered_probe(
            "legacy",
            ImplementationKind.LEGACY,
            technique_id="alpha",
            aliases=("old-alpha",),
        )
    )
    catalog.set_default("old-alpha", implementation_id="legacy")

    assert catalog.contains("old-alpha")
    assert catalog.resolve(TechniqueSelector("old-alpha")).definition.identity.id == "alpha"
    assert [item.definition.identity.id for item in catalog.list_definitions()] == ["alpha", "zeta"]


def test_catalog_rejects_unknown_or_ambiguous_defaults() -> None:
    catalog = TechniqueCatalog()
    catalog.register(registered_probe("legacy", ImplementationKind.LEGACY))

    with pytest.raises(KeyError, match="implementation 'missing'"):
        catalog.set_default("probe", implementation_id="missing")
    with pytest.raises(KeyError, match="technique 'missing'"):
        catalog.resolve(TechniqueSelector("missing"))


def test_disabled_registration_is_available_only_to_exact_pinned_selectors() -> None:
    """Disabling blocks fresh selection but preserves historical exact pins."""

    catalog = TechniqueCatalog()
    catalog.register(registered_probe("modular", ImplementationKind.WORKFLOW))
    catalog.set_default("probe", implementation_id="modular")

    catalog.set_selectable(
        "probe",
        version="1.0.0",
        implementation_id="modular",
        selectable=False,
    )

    with pytest.raises(KeyError, match="technique 'probe' is not registered"):
        catalog.resolve(TechniqueSelector("probe"))
    with pytest.raises(KeyError, match="not registered"):
        catalog.set_default("probe", implementation_id="modular")
    assert (
        catalog.resolve(
            TechniqueSelector(
                "probe",
                technique_version="1.0.0",
                implementation_id="modular",
            )
        ).implementation.id
        == "modular"
    )


def test_default_catalog_contains_every_legacy_attack() -> None:
    catalog = build_default_catalog(discover_modular=False)
    expected = {item.metadata.key for item in list_attack_techniques()}

    assert {item.definition.identity.id for item in catalog.list_definitions()} == expected
    for technique_id in expected:
        resolved = catalog.resolve(TechniqueSelector(technique_id))
        assert resolved.implementation.kind is ImplementationKind.LEGACY
        assert resolved.implementation.id == "legacy"
        assert resolved.definition.identity.version == "0.2.0"


def test_legacy_catalog_definitions_preserve_descriptor_metadata() -> None:
    catalog = build_default_catalog(discover_modular=False)

    for technique_cls in list_attack_techniques():
        metadata = technique_cls.metadata
        config = default_technique_config(metadata.key)
        registered = catalog.resolve(TechniqueSelector(metadata.key))
        definition = registered.definition
        legacy = definition.extensions["legacy"]

        assert definition.identity.name == metadata.name
        assert definition.identity.family == metadata.family
        assert definition.summary == metadata.summary
        assert definition.tags == metadata.tags
        assert definition.target_surfaces == (metadata.surface.value,)
        assert legacy["surface"] == metadata.surface.value
        assert legacy["automation_level"] == metadata.automation_level.value
        assert legacy["source_paper"] == metadata.source_paper
        assert legacy["requires_image"] is metadata.requires_image
        assert legacy["multi_turn"] is metadata.multi_turn
        assert legacy["requires_feedback_loop"] is metadata.requires_feedback_loop
        assert legacy["requires_attacker_llm"] is metadata.requires_attacker_llm
        assert legacy["requires_judge_per_round"] is metadata.requires_judge_per_round
        assert legacy["requires_seed_prompt"] is metadata.requires_seed_prompt
        assert legacy["requires_seed_code"] is metadata.requires_seed_code
        assert legacy["requires_white_box_access"] is metadata.requires_white_box_access
        assert legacy["detector_defaults"] == tuple(detector.value for detector in config.enabled_detectors)
        assert legacy["default_max_turns"] == config.max_turns
        assert legacy["class_import_path"] == (f"{technique_cls.__module__}:{technique_cls.__qualname__}")
        expected_execution_capability = (
            TechniqueCapability.ITERATIVE
            if metadata.multi_turn or metadata.requires_feedback_loop
            else TechniqueCapability.SINGLE_SHOT
        )
        assert expected_execution_capability in definition.capabilities
        assert (TechniqueCapability.MULTIMODAL in definition.capabilities) is metadata.requires_image


def test_router_keeps_legacy_as_default() -> None:
    router = ExecutionRouter(build_default_catalog(discover_modular=False))

    route = router.resolve("direct_prompt_injection")

    assert route.registered.implementation.kind is ImplementationKind.LEGACY
    assert route.registered.implementation.id == "legacy"
    assert route.is_legacy


def test_store_lists_catalog_metadata_without_changing_public_fields() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=build_default_catalog(discover_modular=False),
    )

    direct = next(item for item in store.list_techniques() if item["key"] == "direct_prompt_injection")

    assert direct["name"] == "Direct Prompt Injection"
    assert direct["implementation"] == "legacy"
    assert direct["surface"] == "text"
    assert direct["config"]["enabled_detectors"] == [
        "refusal_bypass",
        "system_prompt_leakage",
        "indirect_prompt_injection",
    ]


def test_store_catalog_listing_matches_every_legacy_descriptor() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=build_default_catalog(discover_modular=False),
    )
    listed = {item["key"]: item for item in store.list_techniques()}

    assert len(listed) == len(list_attack_techniques())
    for technique_cls in list_attack_techniques():
        metadata = technique_cls.metadata
        config = default_technique_config(metadata.key)
        item = listed[metadata.key]

        assert item["name"] == metadata.name
        assert item["family"] == metadata.family
        assert item["surface"] == metadata.surface.value
        assert item["summary"] == metadata.summary
        assert item["tags"] == list(metadata.tags)
        assert item["multi_turn"] is metadata.multi_turn
        assert item["requires_feedback_loop"] is metadata.requires_feedback_loop
        assert item["requires_attacker_llm"] is metadata.requires_attacker_llm
        assert item["requires_judge_per_round"] is metadata.requires_judge_per_round
        assert item["config"] == {
            "max_turns": config.max_turns,
            "enabled_detectors": [detector.value for detector in config.enabled_detectors],
        }
        assert item["implementation"] == "legacy"


def test_store_catalog_listing_keeps_single_shot_modular_requirements_false() -> None:
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        catalog=build_default_catalog(
            persistence=NoopTechniqueRuntimePersistence()
        ),
    )

    probe = next(
        item
        for item in store.list_techniques()
        if item["key"] == "direct_prompt_injection"
    )

    assert probe["implementation"] == "modular"
    assert probe["multi_turn"] is False
    assert probe["requires_feedback_loop"] is False
    assert probe["requires_attacker_llm"] is False
    assert probe["requires_judge_per_round"] is False


def test_store_rejects_package_less_modular_route_before_runtime_dispatch() -> None:
    catalog = TechniqueCatalog()
    catalog.register(registered_probe("modular", ImplementationKind.WORKFLOW))
    catalog.set_default("probe", implementation_id="modular")
    store = EvaluationStore(persistence=NoopRunPersistence(), catalog=catalog)

    with pytest.raises(ValueError, match="exact loaded package"):
        store._execute_single(
            technique_key="probe",
            target_kind="chatbot",
            mode="demo",
            target_config={},
        )


def test_suite_importer_validates_against_injected_catalog_aliases() -> None:
    catalog = TechniqueCatalog()
    catalog.register(
        registered_probe(
            "legacy",
            ImplementationKind.LEGACY,
            aliases=("old-probe",),
        )
    )
    catalog.set_default("probe", implementation_id="legacy")

    suite = import_suite_from_json(
        """
        {
          "suite_name": "alias-suite",
          "source": "unit-test",
          "cases": [
            {
              "technique_key": "old-probe",
              "unsafe_intent_label": "alias-resolution-check"
            }
          ]
        }
        """,
        catalog=catalog,
    )

    assert suite.cases[0].technique_key == "old-probe"
