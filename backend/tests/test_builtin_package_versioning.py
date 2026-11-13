from pathlib import Path

from app.persistence.technique_runtime import (
    NoopTechniqueRuntimePersistence,
    TechniquePackageRecord,
)
from app.techniques import build_default_catalog
from app.techniques.core.catalog import TechniqueSelector
from app.techniques.plugins.loader import _package_id


BUILTIN_ROOT = (
    Path(__file__).resolve().parents[1] / "app" / "techniques" / "builtin"
)


def _archived_registration(
    technique_id: str,
    *,
    version: str = "1.0.0",
    implementation_kind: str = "python",
    package_id: str | None = None,
) -> TechniquePackageRecord:
    package_root = BUILTIN_ROOT / technique_id
    return TechniquePackageRecord(
        package_id=package_id or _package_id(package_root),
        technique_id=technique_id,
        technique_version=version,
        implementation_id="modular",
        implementation_kind=implementation_kind,
        contract_version="1",
        source_kind="source_controlled",
        source_location=str(package_root),
        state="active",
        manifest_hash="0" * 64,
        asset_hashes={"archived": "1" * 64},
    )


def test_changed_builtin_packages_coexist_with_archived_versions() -> None:
    persistence = NoopTechniqueRuntimePersistence()
    archived = [
        _archived_registration("crescendo_liar"),
        _archived_registration(
            "crescendo_liar",
            version="1.0.1",
            package_id="source-conflict-crescendo-liar-1.0.1",
        ),
        _archived_registration("chunked_request"),
        _archived_registration("pair"),
        _archived_registration(
            "pair",
            version="1.0.1",
            package_id="source-conflict-pair-1.0.1",
        ),
        _archived_registration(
            "nested_scenario",
            implementation_kind="workflow",
        ),
        _archived_registration(
            "nested_scenario",
            version="1.0.1",
            implementation_kind="workflow",
            package_id="source-conflict-nested-scenario-1.0.1",
        ),
        _archived_registration(
            "skeleton_key",
            implementation_kind="workflow",
        ),
        _archived_registration(
            "skeleton_key",
            version="1.0.1",
            implementation_kind="workflow",
            package_id="source-conflict-skeleton-key-1.0.1",
        ),
    ]
    for record in archived:
        persistence.save_package(record)

    assert len({record.package_id for record in archived}) == len(archived)

    catalog = build_default_catalog(persistence=persistence)

    for record in archived:
        assert persistence.load_package(record.package_id) == record
        assert persistence.load_package_by_reference(
            technique_id=record.technique_id,
            technique_version=record.technique_version,
            implementation_id=record.implementation_id,
        ) == record

    selected = catalog.resolve(TechniqueSelector("crescendo_liar"))
    assert selected.definition.identity.version == "1.0.2"
    assert selected.implementation.id == "modular"
    current = persistence.load_package_by_reference(
        technique_id="crescendo_liar",
        technique_version="1.0.2",
        implementation_id="modular",
    )
    assert current is not None
    assert current not in archived

    selected = catalog.resolve(TechniqueSelector("chunked_request"))
    assert selected.definition.identity.version == "1.0.1"
    assert selected.implementation.id == "modular"
    assert persistence.load_package_by_reference(
        technique_id="chunked_request",
        technique_version="1.0.1",
        implementation_id="modular",
    ) is not None

    selected = catalog.resolve(TechniqueSelector("pair"))
    assert selected.definition.identity.version == "1.0.2"
    assert selected.implementation.id == "modular"
    current = persistence.load_package_by_reference(
        technique_id="pair",
        technique_version="1.0.2",
        implementation_id="modular",
    )
    assert current is not None
    assert current.implementation_kind == "python"
    assert current not in archived

    selected = catalog.resolve(TechniqueSelector("nested_scenario"))
    assert selected.definition.identity.version == "1.0.2"
    assert selected.implementation.id == "modular"
    current = persistence.load_package_by_reference(
        technique_id="nested_scenario",
        technique_version="1.0.2",
        implementation_id="modular",
    )
    assert current is not None
    assert current.implementation_kind == "workflow"
    assert current not in archived

    selected = catalog.resolve(TechniqueSelector("skeleton_key"))
    assert selected.definition.identity.version == "1.0.2"
    assert selected.implementation.id == "modular"
    current = persistence.load_package_by_reference(
        technique_id="skeleton_key",
        technique_version="1.0.2",
        implementation_id="modular",
    )
    assert current is not None
    assert current.implementation_kind == "workflow"
    assert current not in archived
