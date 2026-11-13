from __future__ import annotations

import shutil
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest
import yaml

from app.persistence.technique_runtime import NoopTechniqueRuntimePersistence
from app.techniques.contracts.definition import TechniqueCapability
from app.techniques.plugins.manifest import (
    ManifestValidationError,
    load_technique_package,
)


FIXTURES = Path(__file__).parent / "fixtures" / "techniques"
SUPPORTED_CAPABILITIES = frozenset(TechniqueCapability)


def _copy_valid_probe(tmp_path: Path) -> Path:
    destination = tmp_path / "valid-probe"
    shutil.copytree(FIXTURES / "valid_probe", destination)
    return destination


def lifecycle_api():
    """Load the Task 20 public boundary inside tests so RED is observable."""

    return import_module("app.techniques.plugins.loader")


def valid_package(*, version: str = "1.0.0", manifest_hash: str | None = None):
    package = load_technique_package(
        FIXTURES / "valid_probe",
        supported_capabilities=SUPPORTED_CAPABILITIES,
    )
    identity = replace(package.definition.identity, version=version)
    definition = replace(package.definition, identity=identity)
    implementation = replace(
        package.implementation,
        technique_version=version,
    )
    return replace(
        package,
        definition=definition,
        implementation=implementation,
        manifest_hash=manifest_hash or package.manifest_hash,
    )


def package_loader(
    api: object,
    *,
    roots: tuple[Path, ...] = (),
    catalog: object | None = None,
    self_test: object | None = None,
    referenced_versions: set[tuple[str, str, str]] | None = None,
    defaults_path: Path | None = None,
    persistence: object | None = None,
):
    catalog_type = import_module("app.techniques.core.catalog").TechniqueCatalog
    return api.TechniquePackageLoader(
        catalog=catalog or catalog_type(),
        roots=roots,
        persistence=persistence
        or NoopTechniqueRuntimePersistence(
            referenced_versions=referenced_versions or set()
        ),
        self_test=self_test,
        defaults_path=defaults_path or api.DEFAULT_OVERRIDES_PATH,
    )


def test_invalid_package_does_not_block_valid_package() -> None:
    """A malformed sibling must not prevent a valid package reaching validated."""

    api = lifecycle_api()
    loader = package_loader(api, roots=(FIXTURES,))

    result = loader.discover()

    assert result.by_id("probe").state is api.PackageState.VALIDATED
    assert result.by_id("invalid-judge").state is api.PackageState.INVALID


def test_activation_is_atomic_when_self_test_fails() -> None:
    """A failed contract self-test must leave the previous catalog default intact."""

    api = lifecycle_api()
    catalog_type = import_module("app.techniques.core.catalog").TechniqueCatalog
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    catalog = catalog_type()
    package_loader(api, catalog=catalog).activate(valid_package(), make_default=True)

    def fail_self_test(*_args: object) -> None:
        raise RuntimeError("contract failure")

    loader = package_loader(api, catalog=catalog, self_test=fail_self_test)
    with pytest.raises(api.PackageActivationError, match="contract failure"):
        loader.activate(valid_package(version="1.1.0"))

    assert (
        loader.state(api.PackageReference("probe", "1.1.0", "modular"))
        is api.PackageState.INVALID
    )
    assert (
        catalog.resolve(selector_type("probe")).definition.identity.version == "1.0.0"
    )


def test_same_version_with_different_hash_is_rejected() -> None:
    """Changing bytes behind an existing version must never replace it."""

    api = lifecycle_api()
    loader = package_loader(api)
    loader.activate(valid_package(manifest_hash="a" * 64))

    with pytest.raises(ValueError, match="same version has different contents"):
        loader.activate(valid_package(manifest_hash="b" * 64))


def test_referenced_package_can_be_soft_uninstalled_but_not_deleted() -> None:
    """Archiving retains a pinned package, while persisted evidence blocks deletion."""

    api = lifecycle_api()
    reference = api.PackageReference("probe", "1.0.0", "modular")
    loader = package_loader(
        api,
        referenced_versions={
            (reference.technique_id, reference.technique_version, reference.implementation_id)
        },
    )
    loader.activate(valid_package())

    loader.soft_uninstall(reference)

    assert loader.state(reference) is api.PackageState.UNINSTALLED
    with pytest.raises(api.PackageInUseError, match="referenced by persisted runs"):
        loader.hard_delete_unreferenced(reference)


def test_discover_only_package_metadata_can_be_hard_deleted() -> None:
    """Unregistered discovery metadata must delete without a late catalog error."""

    api = lifecycle_api()
    persistence = NoopTechniqueRuntimePersistence()
    loader = package_loader(
        api,
        roots=(FIXTURES,),
        persistence=persistence,
    )
    package = loader.discover().by_id("probe")
    package_id = api._package_id(package.root)

    loader.hard_delete_unreferenced(package.reference)

    assert persistence.load_package(package_id) is None
    with pytest.raises(KeyError, match="not known"):
        loader.state(package.reference)


def test_hard_delete_after_rediscovery_removes_durable_registration() -> None:
    """Transient validation must not orphan an exact registered version."""

    api = lifecycle_api()
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    persistence = NoopTechniqueRuntimePersistence()
    loader = package_loader(api, roots=(FIXTURES,), persistence=persistence)
    package = valid_package()
    package = replace(
        package,
        definition=replace(package.definition, aliases=("probe-alias",)),
    )
    reference = api.PackageReference("probe", "1.0.0", "modular")
    loader.activate(package, make_default=True)

    assert loader.discover().by_id("probe").state is api.PackageState.VALIDATED
    loader.hard_delete_unreferenced(reference)

    assert persistence.load_package(api._package_id(package.root)) is None
    assert (
        persistence.load_package_by_reference(
            technique_id="probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )
        is None
    )
    assert not loader.catalog.contains("probe")
    assert not loader.catalog.contains("probe-alias")
    with pytest.raises(KeyError, match="not registered"):
        loader.catalog.resolve(selector_type("probe"))
    with pytest.raises(KeyError, match="not registered"):
        loader.catalog.resolve(selector_type("probe-alias", "1.0.0", "modular"))


def test_hard_delete_catalog_failure_preserves_lifecycle_metadata() -> None:
    """Catalog ineligibility must be detected before persistence is destroyed."""

    api = lifecycle_api()
    catalog_type = import_module("app.techniques.core.catalog").TechniqueCatalog
    persistence = NoopTechniqueRuntimePersistence()
    catalog = catalog_type()
    loader = package_loader(
        api,
        roots=(FIXTURES,),
        catalog=catalog,
        persistence=persistence,
    )
    package = valid_package()
    reference = api.PackageReference("probe", "1.0.0", "modular")
    loader.activate(package)
    loader.discover()
    catalog.unregister("probe", version="1.0.0", implementation_id="modular")

    with pytest.raises(KeyError, match="not registered"):
        loader.hard_delete_unreferenced(reference)

    assert persistence.load_package(api._package_id(package.root)) is not None
    assert (
        persistence.load_package_by_reference(
            technique_id="probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )
        is not None
    )
    assert loader.state(reference) is api.PackageState.VALIDATED


def test_hard_delete_persistence_failure_restores_catalog_registration() -> None:
    """A failed durable delete must not strand the catalog without its package."""

    api = lifecycle_api()
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector

    class FailingDeletePersistence(NoopTechniqueRuntimePersistence):
        def hard_delete_package(self, record) -> None:
            raise RuntimeError("delete failed")

    persistence = FailingDeletePersistence()
    loader = package_loader(
        api,
        roots=(FIXTURES,),
        persistence=persistence,
    )
    package = valid_package()
    reference = api.PackageReference("probe", "1.0.0", "modular")
    loader.activate(package)
    loader.discover()

    with pytest.raises(RuntimeError, match="delete failed"):
        loader.hard_delete_unreferenced(reference)

    assert persistence.load_package(api._package_id(package.root)) is not None
    assert (
        persistence.load_package_by_reference(
            technique_id="probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )
        is not None
    )
    assert loader.catalog.resolve(selector_type("probe", "1.0.0", "modular")).implementation.id == "modular"


def test_disabled_lifecycle_state_does_not_apply_to_changed_package_contents() -> None:
    """A new immutable version at one source path starts selectable again."""

    api = lifecycle_api()
    loader = package_loader(api)
    original = valid_package(manifest_hash="a" * 64)
    original_reference = api.PackageReference("probe", "1.0.0", "modular")
    loader.activate(original)
    loader.disable(original_reference)

    replacement = valid_package(version="1.1.0", manifest_hash="b" * 64)
    loader.activate(replacement, preserve_lifecycle_state=True)

    assert (
        loader.state(api.PackageReference("probe", "1.1.0", "modular"))
        is api.PackageState.ACTIVE
    )


@pytest.mark.parametrize(
    ("transition", "expected_state"),
    [("disable", "disabled"), ("soft_uninstall", "uninstalled")],
)
def test_archived_lifecycle_state_survives_source_relocation(
    tmp_path: Path,
    transition: str,
    expected_state: str,
) -> None:
    """Archived identity follows immutable contents, never its former path."""

    api = lifecycle_api()
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    persistence = NoopTechniqueRuntimePersistence()
    root = tmp_path / "packages"
    original_root = root / "original"
    shutil.copytree(FIXTURES / "valid_probe", original_root)
    first = package_loader(api, roots=(root,), persistence=persistence)
    original = first.discover().by_id("probe")
    assert original.package is not None
    first.activate(original.package)
    getattr(first, transition)(original.reference)

    relocated_root = root / "relocated"
    original_root.rename(relocated_root)
    second = package_loader(api, roots=(root,), persistence=persistence)
    relocated = second.discover().by_id("probe")

    assert relocated.state.value == expected_state
    assert relocated.package is not None
    second.activate(relocated.package, preserve_lifecycle_state=True)
    assert second.state(relocated.reference).value == expected_state
    with pytest.raises(KeyError, match="not registered"):
        second.catalog.resolve(selector_type("probe"))
    assert second.catalog.resolve(selector_type("probe", "1.0.0", "modular")).implementation.id == "modular"


def test_asset_only_change_does_not_restore_archived_state_or_reinstall(
    tmp_path: Path,
) -> None:
    """A manifest match cannot hide changed immutable package assets."""

    api = lifecycle_api()
    persistence = NoopTechniqueRuntimePersistence()
    root = tmp_path / "packages"
    package_root = root / "probe"
    shutil.copytree(FIXTURES / "valid_probe", package_root)
    first = package_loader(api, roots=(root,), persistence=persistence)
    original = first.discover().by_id("probe")
    assert original.package is not None
    first.activate(original.package)
    first.disable(original.reference)
    (package_root / "workflow.yaml").write_text(
        "version: changed-asset-only\nsteps: []\n",
        encoding="utf-8",
    )

    second = package_loader(api, roots=(root,), persistence=persistence)
    changed = second.discover().by_id("probe")

    assert changed.state is api.PackageState.VALIDATED
    assert changed.package is not None
    with pytest.raises(ValueError, match="same version has different contents"):
        second.activate(changed.package, preserve_lifecycle_state=True)


@pytest.mark.parametrize(
    ("transition", "expected_state"),
    [("disable", "disabled"), ("soft_uninstall", "uninstalled")],
)
def test_same_path_asset_conflict_preserves_archived_registration(
    tmp_path: Path,
    transition: str,
    expected_state: str,
) -> None:
    """Conflicting discovery must not overwrite canonical lifecycle evidence."""

    api = lifecycle_api()
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    persistence = NoopTechniqueRuntimePersistence()
    root = tmp_path / "packages"
    package_root = root / "probe"
    shutil.copytree(FIXTURES / "valid_probe", package_root)
    workflow_path = package_root / "workflow.yaml"
    original_workflow = workflow_path.read_bytes()
    first = package_loader(api, roots=(root,), persistence=persistence)
    original = first.discover().by_id("probe")
    assert original.package is not None
    first.activate(original.package)
    getattr(first, transition)(original.reference)
    canonical_id = api._package_id(package_root)
    archived = persistence.load_package(canonical_id)
    assert archived is not None
    assert archived.state == expected_state

    workflow_path.write_text(
        "version: conflicting-asset\nsteps: []\n",
        encoding="utf-8",
    )
    conflicting_loader = package_loader(api, roots=(root,), persistence=persistence)
    conflicting = conflicting_loader.discover().by_id("probe")

    assert persistence.load_package(canonical_id) == archived
    assert conflicting.package is not None
    with pytest.raises(ValueError, match="same version has different contents"):
        conflicting_loader.activate(
            conflicting.package,
            preserve_lifecycle_state=True,
        )
    assert persistence.load_package(canonical_id) == archived
    conflict_id = api._conflict_package_id(conflicting.package)
    assert conflict_id != canonical_id
    conflict_record = persistence.load_package(conflict_id)
    assert conflict_record is not None
    assert conflict_record.state == api.PackageState.INVALID.value
    assert conflict_record.diagnostics[-1]["stage"] == "activation"

    workflow_path.write_bytes(original_workflow)
    restored_loader = package_loader(api, roots=(root,), persistence=persistence)
    restored = restored_loader.discover().by_id("probe")

    assert restored.state.value == expected_state
    assert restored.package is not None
    restored_loader.activate(restored.package, preserve_lifecycle_state=True)
    canonical = persistence.load_package(canonical_id)
    assert canonical is not None
    assert canonical.state == expected_state
    assert dict(canonical.asset_hashes) == dict(archived.asset_hashes)
    with pytest.raises(KeyError, match="not registered"):
        restored_loader.catalog.resolve(selector_type("probe"))
    assert (
        restored_loader.catalog.resolve(
            selector_type("probe", "1.0.0", "modular")
        ).implementation.id
        == "modular"
    )


def test_same_path_invalid_discovery_preserves_archived_registration(
    tmp_path: Path,
) -> None:
    """Invalid package diagnostics must not replace canonical package metadata."""

    api = lifecycle_api()
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    persistence = NoopTechniqueRuntimePersistence()
    root = tmp_path / "packages"
    package_root = root / "probe"
    shutil.copytree(FIXTURES / "valid_probe", package_root)
    workflow_path = package_root / "workflow.yaml"
    original_workflow = workflow_path.read_bytes()
    first = package_loader(api, roots=(root,), persistence=persistence)
    original = first.discover().by_id("probe")
    assert original.package is not None
    first.activate(original.package)
    first.disable(original.reference)
    canonical_id = api._package_id(package_root)
    archived = persistence.load_package(canonical_id)
    assert archived is not None

    workflow_path.unlink()
    invalid_loader = package_loader(api, roots=(root,), persistence=persistence)
    invalid = invalid_loader.discover().by_id("probe")

    assert invalid.state is api.PackageState.INVALID
    assert persistence.load_package(canonical_id) == archived
    invalid_record = persistence.load_package(
        api._invalid_conflict_package_id(invalid)
    )
    assert invalid_record is not None
    assert invalid_record.state == api.PackageState.INVALID.value
    assert invalid_record.diagnostics

    workflow_path.write_bytes(original_workflow)
    restored_loader = package_loader(api, roots=(root,), persistence=persistence)
    restored = restored_loader.discover().by_id("probe")

    assert restored.state is api.PackageState.DISABLED
    assert restored.package is not None
    restored_loader.activate(restored.package, preserve_lifecycle_state=True)
    with pytest.raises(KeyError, match="not registered"):
        restored_loader.catalog.resolve(selector_type("probe"))
    assert (
        restored_loader.catalog.resolve(
            selector_type("probe", "1.0.0", "modular")
        ).implementation.id
        == "modular"
    )


def test_rediscovery_and_successful_activation_preserve_invalid_diagnostic_history() -> None:
    """Runtime activation evidence remains durable across explicit revalidation."""

    api = lifecycle_api()
    persistence = NoopTechniqueRuntimePersistence()

    def fail_self_test(*_args: object) -> None:
        raise RuntimeError("historical activation failure")

    failing = package_loader(api, persistence=persistence, self_test=fail_self_test)
    package = valid_package()
    with pytest.raises(api.PackageActivationError):
        failing.activate(package)
    package_id = api._package_id(package.root)

    revalidating = package_loader(
        api,
        roots=(FIXTURES,),
        persistence=persistence,
    )
    rediscovered = revalidating.discover().by_id("probe")
    after_discovery = persistence.load_package(package_id)

    assert rediscovered.state is api.PackageState.VALIDATED
    assert after_discovery is not None
    assert after_discovery.state == api.PackageState.VALIDATED.value
    assert [item["stage"] for item in after_discovery.diagnostics] == ["activation"]
    assert "historical activation failure" in after_discovery.diagnostics[0]["message"]

    assert rediscovered.package is not None
    revalidating.activate(rediscovered.package, preserve_lifecycle_state=True)
    after_activation = persistence.load_package(package_id)

    assert after_activation is not None
    assert after_activation.state == api.PackageState.ACTIVE.value
    assert [item["stage"] for item in after_activation.diagnostics] == ["activation"]


@pytest.mark.parametrize("failure", ["import", "self-test", "self-test-value"])
def test_startup_isolates_runtime_invalid_python_package_and_applies_valid_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    """One runtime-invalid package cannot abort valid siblings or legacy startup."""

    api = lifecycle_api()
    techniques = import_module("app.techniques")
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    root = tmp_path / "packages"
    valid_root = root / "valid"
    broken_root = root / "broken"
    shutil.copytree(FIXTURES / "valid_probe", valid_root)
    shutil.copytree(FIXTURES / "valid_probe", broken_root)
    manifest_path = broken_root / "technique.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["technique"].update({"id": "broken-probe", "name": "Broken Probe"})
    manifest["implementation"].update(
        {
            "id": "broken-modular",
            "kind": "python",
            "entrypoint": "executor:SourceControlledExecutor",
        }
    )
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    if failure == "import":
        (broken_root / "executor.py").write_text(
            'raise RuntimeError("startup import failure sk-startupsecret123")\n',
            encoding="utf-8",
        )

    defaults_path = tmp_path / "defaults.yaml"
    defaults_path.write_text(
        "- technique_id: probe\n  technique_version: 1.0.0\n  implementation_id: modular\n",
        encoding="utf-8",
    )
    persistence = NoopTechniqueRuntimePersistence()
    loaders = []
    real_loader = api.TechniquePackageLoader

    def self_test(package, _entrypoint) -> None:
        if failure.startswith("self-test") and package.definition.identity.id == "broken-probe":
            if failure == "self-test-value":
                raise ValueError("startup self-test-value failure: same version has different contents sk-startupsecret123")
            raise RuntimeError("startup self-test failure sk-startupsecret123")

    def loader_factory(**kwargs):
        loader = real_loader(
            **{
                **kwargs,
                "roots": (root,),
                "persistence": persistence,
                "self_test": self_test,
                "defaults_path": defaults_path,
            }
        )
        loaders.append(loader)
        return loader

    monkeypatch.setattr(api, "TechniquePackageLoader", loader_factory)

    catalog = techniques.build_default_catalog()

    assert catalog.resolve(selector_type("probe")).implementation.id == "modular"
    assert catalog.resolve(selector_type("direct_prompt_injection")).implementation.id == "legacy"
    with pytest.raises(KeyError, match="broken-probe"):
        catalog.resolve(selector_type("broken-probe", "1.0.0", "broken-modular"))
    reference = api.PackageReference("broken-probe", "1.0.0", "broken-modular")
    assert loaders[0].state(reference) is api.PackageState.INVALID
    persisted = persistence.load_package(api._package_id(broken_root))
    assert persisted is not None
    assert persisted.state == api.PackageState.INVALID.value
    assert persisted.diagnostics[-1]["stage"] == "activation"
    diagnostic = persisted.diagnostics[-1]["message"]
    assert f"startup {failure} failure" in caplog.text
    assert "broken-probe" in caplog.text
    assert str(broken_root) in caplog.text
    assert "Correct the package error" in caplog.text
    assert "Conflicting registrations" not in caplog.text
    assert diagnostic in caplog.text
    assert "sk-startupsecret123" not in caplog.text
    assert "sk-startupsecret123" not in diagnostic
    assert len([record for record in caplog.records if record.name == "techniques"]) == 1


@pytest.mark.parametrize("storage_failure", [None, "diagnostic", "rollback"])
def test_startup_reports_duplicate_package_without_displacing_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    storage_failure: str | None,
) -> None:
    api = lifecycle_api()
    techniques = import_module("app.techniques")
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    root = tmp_path / "packages"
    winner_root = root / "a-winner"
    rejected_root = root / "b-rejected"
    for package_root in (winner_root, rejected_root):
        shutil.copytree(FIXTURES / "valid_probe", package_root)
    manifest_path = rejected_root / "technique.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["technique"]["summary"] = "Conflicting same-version package"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    defaults_path = tmp_path / "defaults.yaml"
    defaults_path.write_text(
        "- technique_id: probe\n  technique_version: 1.0.0\n  implementation_id: modular\n",
        encoding="utf-8",
    )

    class FailingPersistence(NoopTechniqueRuntimePersistence):
        failing_restore = False

        def save_package(self, record):
            if record.source_location == str(rejected_root):
                if storage_failure == "diagnostic" and record.state == "invalid":
                    raise RuntimeError("diagnostic storage failed sk-storagesecret123")
                if storage_failure == "rollback" and self.failing_restore and record.state == "validated":
                    raise RuntimeError("rollback storage failed sk-storagesecret123")
            return super().save_package(record)

    persistence = FailingPersistence()
    real_loader = api.TechniquePackageLoader

    def self_test(package, _entrypoint):
        if package.root == rejected_root:
            persistence.failing_restore = True

    def loader_factory(**kwargs):
        return real_loader(**{
            **kwargs, "roots": (root,), "persistence": persistence,
            "defaults_path": defaults_path, "self_test": self_test,
        })

    monkeypatch.setattr(api, "TechniquePackageLoader", loader_factory)
    catalog = techniques.build_default_catalog()

    winner = catalog.resolve(selector_type("probe"))
    assert winner.package.root == winner_root
    winner_record = persistence.load_package(api._package_id(winner_root))
    assert winner_record.state == "active"
    assert winner_record.diagnostics == ()
    assert catalog.resolve(selector_type("direct_prompt_injection")).implementation.id == "legacy"
    for detail in (
        "probe", "1.0.0", "modular", str(winner_root), str(rejected_root),
        "same version has different contents", "Remove the duplicate", "new version",
    ):
        assert detail in caplog.text
    assert "sk-storagesecret123" not in caplog.text
    assert len([record for record in caplog.records if record.name == "techniques"]) == 1
    if storage_failure:
        assert f"{storage_failure} storage failed" in caplog.text
    if storage_failure != "diagnostic":
        rejected_record = persistence.load_package(api._package_id(rejected_root))
        assert rejected_record.state == "invalid"
        assert rejected_record.diagnostics[-1]["message"] in caplog.text
        assert str(winner_root) in rejected_record.diagnostics[-1]["message"]


@pytest.mark.parametrize(
    "failure",
    ("promoted-activation", "missing-promoted", "missing-defaults", "invalid-defaults"),
)
def test_default_catalog_fails_closed_when_promotion_cannot_be_proven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Promoted defaults must never silently fall back to a legacy registration."""

    api = lifecycle_api()
    techniques = import_module("app.techniques")
    root = tmp_path / "packages"
    root.mkdir()
    if failure == "promoted-activation":
        shutil.copytree(FIXTURES / "valid_probe", root / "probe")
    defaults_path = tmp_path / "defaults.yaml"
    if failure in {"promoted-activation", "missing-promoted"}:
        defaults_path.write_text(
            "- technique_id: probe\n"
            "  technique_version: 1.0.0\n"
            "  implementation_id: modular\n",
            encoding="utf-8",
        )
    elif failure == "invalid-defaults":
        defaults_path.write_text("not: a-list\n", encoding="utf-8")

    real_loader = api.TechniquePackageLoader

    def self_test(package, _entrypoint) -> None:
        if failure == "promoted-activation" and package.definition.identity.id == "probe":
            raise RuntimeError("promoted activation failed")

    def loader_factory(**kwargs):
        return real_loader(
            **{
                **kwargs,
                "roots": (root,),
                "persistence": NoopTechniqueRuntimePersistence(),
                "self_test": self_test,
                "defaults_path": defaults_path,
            }
        )

    monkeypatch.setattr(api, "TechniquePackageLoader", loader_factory)

    with pytest.raises(api.PackageActivationError, match="defaults|promotion"):
        techniques.build_default_catalog()


def test_default_catalog_applies_valid_promoted_default_with_noop_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fail-closed path still permits an ordinary valid promoted startup."""

    api = lifecycle_api()
    techniques = import_module("app.techniques")
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    root = tmp_path / "packages"
    shutil.copytree(FIXTURES / "valid_probe", root / "probe")
    defaults_path = tmp_path / "defaults.yaml"
    defaults_path.write_text(
        "- technique_id: probe\n"
        "  technique_version: 1.0.0\n"
        "  implementation_id: modular\n",
        encoding="utf-8",
    )
    real_loader = api.TechniquePackageLoader

    def loader_factory(**kwargs):
        return real_loader(
            **{
                **kwargs,
                "roots": (root,),
                "persistence": NoopTechniqueRuntimePersistence(),
                "defaults_path": defaults_path,
            }
        )

    monkeypatch.setattr(api, "TechniquePackageLoader", loader_factory)

    catalog = techniques.build_default_catalog()

    assert catalog.resolve(selector_type("probe")).implementation.id == "modular"


def test_invalid_default_override_is_diagnostic_and_keeps_existing_default(
    tmp_path: Path,
) -> None:
    """A bad promotion override must not displace the current default."""

    api = lifecycle_api()
    selector_type = import_module("app.techniques.core.catalog").TechniqueSelector
    defaults_path = tmp_path / "defaults.yaml"
    defaults_path.write_text(
        "- technique_id: probe\n  technique_version: 9.9.9\n  implementation_id: modular\n",
        encoding="utf-8",
    )
    loader = package_loader(api, defaults_path=defaults_path)
    loader.activate(valid_package(), make_default=True)

    diagnostics = loader.apply_default_overrides()

    assert diagnostics[0].stage == "defaults"
    assert (
        loader.catalog.resolve(selector_type("probe")).definition.identity.version
        == "1.0.0"
    )


def test_workflow_entrypoint_is_confined_without_importing_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declarative workflow packages expose only their confined workflow asset."""

    api = lifecycle_api()
    loader = package_loader(api, roots=(FIXTURES,))

    def fail_python_import(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("workflow resolution imported Python")

    monkeypatch.setattr(api.importlib.util, "spec_from_file_location", fail_python_import)
    entrypoint = loader.resolve_entrypoint(valid_package())

    assert entrypoint.asset_path == (FIXTURES / "valid_probe" / "workflow.yaml").resolve()


def test_python_entrypoint_loads_only_from_a_configured_source_root() -> None:
    """Executable packages require an explicit source-controlled discovery root."""

    api = lifecycle_api()
    package = valid_package()
    implementation = replace(
        package.implementation,
        kind=import_module("app.techniques.contracts.definition").ImplementationKind.PYTHON,
        entrypoint="executor:SourceControlledExecutor",
    )
    package = replace(package, implementation=implementation)

    with pytest.raises(api.PackageActivationError, match="source-controlled root"):
        package_loader(api).resolve_entrypoint(package)

    executor = package_loader(api, roots=(FIXTURES,)).resolve_entrypoint(package)

    assert executor.__class__.__name__ == "SourceControlledExecutor"


def test_manifest_loader_hashes_manifest_and_assets() -> None:
    package = load_technique_package(
        FIXTURES / "valid_probe",
        supported_capabilities=SUPPORTED_CAPABILITIES,
    )
    reloaded = load_technique_package(
        FIXTURES / "valid_probe",
        supported_capabilities=SUPPORTED_CAPABILITIES,
    )

    assert package.definition.identity.id == "probe"
    assert package.implementation.entrypoint == "workflow.yaml"
    assert len(package.manifest_hash) == 64
    assert set(package.asset_hashes) == {
        "schemas/parameters.json",
        "workflow.yaml",
    }
    assert all(len(digest) == 64 for digest in package.asset_hashes.values())
    assert package.manifest_hash == reloaded.manifest_hash
    assert package.asset_hashes == reloaded.asset_hashes


def test_manifest_rejects_missing_official_judge_reference() -> None:
    with pytest.raises(ManifestValidationError) as raised:
        load_technique_package(
            FIXTURES / "invalid_judge",
            supported_capabilities=SUPPORTED_CAPABILITIES,
        )

    assert "official judge 'missing_judge'" in str(raised.value)
    assert [item.path for item in raised.value.diagnostics] == [
        ("judges", 0, "role"),
        ("result_policy", "auxiliary_judges", 0),
        ("result_policy", "official_judge"),
    ]


def test_manifest_rejects_assets_outside_package_root() -> None:
    with pytest.raises(ManifestValidationError, match="outside the package root"):
        load_technique_package(
            FIXTURES / "invalid_path",
            supported_capabilities=SUPPORTED_CAPABILITIES,
        )


def test_manifest_rejects_unsupported_capabilities() -> None:
    with pytest.raises(
        ManifestValidationError,
        match="unsupported capability 'single_shot'",
    ):
        load_technique_package(
            FIXTURES / "valid_probe",
            supported_capabilities=frozenset(),
        )


def test_manifest_rejects_unknown_fields() -> None:
    with pytest.raises(ManifestValidationError) as raised:
        load_technique_package(
            FIXTURES / "invalid_shape",
            supported_capabilities=SUPPORTED_CAPABILITIES,
        )

    assert raised.value.diagnostics[0].path == ("technique",)
    assert "Additional properties are not allowed" in str(raised.value)


@pytest.mark.parametrize(
    ("final_artifact", "message"),
    [
        ({"source": "event.payload"}, "source must be final_response"),
        ({"source": "final_response", "label": ""}, "label must be non-empty text"),
        ({"source": "final_response", "media_type": ""}, "media_type must be non-empty text"),
        (
            {"source": "final_response", "include_official_judge": "yes"},
            "include_official_judge must be boolean",
        ),
    ],
)
def test_final_artifact_declaration_is_validated(
    tmp_path: Path,
    final_artifact: dict[str, object],
    message: str,
) -> None:
    root = _copy_valid_probe(tmp_path)
    manifest = yaml.safe_load((root / "technique.yaml").read_text())
    manifest["presentation"] = {
        "conversation_format": "rounds",
        "final_artifact": final_artifact,
    }
    (root / "technique.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ManifestValidationError, match=message):
        load_technique_package(
            root,
            supported_capabilities=SUPPORTED_CAPABILITIES,
        )


def test_final_artifact_declaration_must_be_a_mapping(tmp_path: Path) -> None:
    root = _copy_valid_probe(tmp_path)
    manifest = yaml.safe_load((root / "technique.yaml").read_text())
    manifest["presentation"] = {
        "conversation_format": "rounds",
        "final_artifact": [],
    }
    (root / "technique.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ManifestValidationError, match="must be a mapping") as raised:
        load_technique_package(
            root,
            supported_capabilities=SUPPORTED_CAPABILITIES,
        )

    assert raised.value.diagnostics[0].path == ("presentation", "final_artifact")


def test_final_artifact_declaration_rejects_unknown_keys(tmp_path: Path) -> None:
    root = _copy_valid_probe(tmp_path)
    manifest = yaml.safe_load((root / "technique.yaml").read_text())
    manifest["presentation"] = {
        "conversation_format": "rounds",
        "final_artifact": {
            "source": "final_response",
            "response_label": "misspelled label",
        },
    }
    (root / "technique.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(
        ManifestValidationError,
        match="unknown key 'response_label'",
    ) as raised:
        load_technique_package(
            root,
            supported_capabilities=SUPPORTED_CAPABILITIES,
        )

    assert raised.value.diagnostics[0].path == ("presentation", "final_artifact")


def test_final_artifact_declaration_rejects_heterogeneous_unknown_keys(
    tmp_path: Path,
) -> None:
    root = _copy_valid_probe(tmp_path)
    manifest = yaml.safe_load((root / "technique.yaml").read_text())
    manifest["presentation"] = {
        "conversation_format": "rounds",
        "final_artifact": {
            "source": "final_response",
            "response_label": "misspelled label",
            7: "unexpected numeric key",
        },
    }
    (root / "technique.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ManifestValidationError) as raised:
        load_technique_package(
            root,
            supported_capabilities=SUPPORTED_CAPABILITIES,
        )

    assert {item.message for item in raised.value.diagnostics} == {
        "unknown key 'response_label'",
        "unknown key 7",
    }
    assert {item.path for item in raised.value.diagnostics} == {
        ("presentation", "final_artifact")
    }
