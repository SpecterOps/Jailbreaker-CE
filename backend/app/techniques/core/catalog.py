"""Deterministic, versioned catalog for technique implementations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from app.techniques.contracts.definition import (
    TechniqueDefinition,
    TechniqueImplementation,
)
from app.techniques.plugins.manifest import LoadedTechniquePackage


_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)\."
    r"(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z.-]+))?"
    r"(?:\+[0-9A-Za-z.-]+)?$"
)


@dataclass(frozen=True, slots=True)
class TechniqueSelector:
    technique_id: str
    technique_version: str | None = None
    implementation_id: str | None = None


@dataclass(frozen=True, slots=True)
class RegisteredTechnique:
    definition: TechniqueDefinition
    implementation: TechniqueImplementation
    manifest_hash: str
    asset_hashes: Mapping[str, str] = field(default_factory=dict)
    package: LoadedTechniquePackage | None = None

    def __post_init__(self) -> None:
        identity = self.definition.identity
        implementation = self.implementation
        if implementation.technique_id != identity.id:
            raise ValueError("implementation technique ID does not match definition")
        if implementation.technique_version != identity.version:
            raise ValueError("implementation technique version does not match definition")
        _semantic_version_key(identity.version)
        if not self.manifest_hash:
            raise ValueError("manifest_hash must not be empty")
        if not all(
            isinstance(path, str) and isinstance(digest, str)
            for path, digest in self.asset_hashes.items()
        ):
            raise TypeError("asset_hashes must be a string-to-string mapping")
        object.__setattr__(
            self,
            "asset_hashes",
            MappingProxyType(dict(self.asset_hashes)),
        )
        if self.package is not None:
            if self.package.definition != self.definition:
                raise ValueError("loaded package definition does not match registration")
            if self.package.implementation != self.implementation:
                raise ValueError("loaded package implementation does not match registration")
            if self.package.manifest_hash != self.manifest_hash:
                raise ValueError("loaded package manifest hash does not match registration")
            if self.package.asset_hashes != self.asset_hashes:
                raise ValueError("loaded package asset hashes do not match registration")


CatalogKey = tuple[str, str, str]


class TechniqueRegistrationConflict(ValueError):
    """A rejected registration and the existing registrations it conflicts with."""

    def __init__(
        self,
        message: str,
        conflicting_registrations: tuple[RegisteredTechnique, ...],
    ) -> None:
        super().__init__(message)
        self.conflicting_registrations = conflicting_registrations


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """Generic rollback point for one catalog lifecycle operation."""

    registrations: tuple[RegisteredTechnique, ...]
    selectable: Mapping[CatalogKey, bool]
    defaults: Mapping[str, CatalogKey]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selectable", MappingProxyType(dict(self.selectable)))
        object.__setattr__(self, "defaults", MappingProxyType(dict(self.defaults)))


class TechniqueCatalog:
    def __init__(self) -> None:
        self._registrations: dict[CatalogKey, RegisteredTechnique] = {}
        self._definitions: dict[tuple[str, str], TechniqueDefinition] = {}
        self._aliases: dict[str, str] = {}
        self._canonical_ids: set[str] = set()
        self._defaults: dict[str, CatalogKey] = {}
        self._selectable: dict[CatalogKey, bool] = {}

    def register(self, registered: RegisteredTechnique) -> None:
        identity = registered.definition.identity
        technique_id = identity.id
        version = identity.version
        implementation_id = registered.implementation.id
        key = (technique_id, version, implementation_id)

        alias_owner = self._aliases.get(technique_id)
        if alias_owner is not None and alias_owner != technique_id:
            raise ValueError(
                f"technique id '{technique_id}' is already registered as an alias"
            )
        self._validate_aliases(technique_id, registered.definition.aliases)

        same_version = tuple(
            existing for existing_key, existing in self._registrations.items()
            if existing_key[:2] == (technique_id, version)
        )
        existing_definition = self._definitions.get((technique_id, version))
        if existing_definition is not None and existing_definition != registered.definition:
            raise TechniqueRegistrationConflict(
                f"technique '{technique_id}' same version has different contents",
                same_version,
            )
        if any(
            existing.manifest_hash != registered.manifest_hash
            for existing in same_version
        ):
            raise TechniqueRegistrationConflict(
                f"technique '{technique_id}' same version has different contents",
                same_version,
            )
        existing = self._registrations.get(key)
        if existing is not None:
            if existing != registered:
                raise TechniqueRegistrationConflict(
                    f"technique '{technique_id}' same version has different contents",
                    (existing,),
                )
            return

        self._registrations[key] = registered
        self._selectable[key] = True
        self._definitions[(technique_id, version)] = registered.definition
        self._canonical_ids.add(technique_id)
        for alias in registered.definition.aliases:
            if alias != technique_id:
                self._aliases[alias] = technique_id

    def set_default(
        self,
        technique_id: str,
        *,
        implementation_id: str,
        version: str | None = None,
    ) -> None:
        canonical_id = self._canonical_id(technique_id)
        candidates = [
            key
            for key in self._registrations
            if key[0] == canonical_id
            and key[2] == implementation_id
            and (version is None or key[1] == version)
            and self._selectable.get(key, True)
        ]
        if not candidates:
            version_message = "" if version is None else f" at version '{version}'"
            raise KeyError(
                f"implementation '{implementation_id}' is not registered for "
                f"technique '{canonical_id}'{version_message}"
            )
        selected = max(candidates, key=lambda key: _semantic_version_key(key[1]))
        self._defaults[canonical_id] = selected

    def resolve(self, selector: TechniqueSelector) -> RegisteredTechnique:
        canonical_id = self._canonical_id(selector.technique_id)
        default_key = self._defaults.get(canonical_id)
        exact_pinned_selector = (
            selector.technique_version is not None
            and selector.implementation_id is not None
        )

        if selector.technique_version is None and selector.implementation_id is None:
            if default_key is not None and self._selectable.get(default_key, True):
                return self._registrations[default_key]

        candidates = [
            (key, registered)
            for key, registered in self._registrations.items()
            if key[0] == canonical_id
            and (
                selector.technique_version is None
                or key[1] == selector.technique_version
            )
            and (
                selector.implementation_id is None
                or key[2] == selector.implementation_id
            )
            and (exact_pinned_selector or self._selectable.get(key, True))
        ]
        if not candidates:
            raise KeyError(self._not_found_message(selector, canonical_id))

        if selector.technique_version is not None and selector.implementation_id is None:
            if default_key is not None and default_key[1] == selector.technique_version:
                return self._registrations[default_key]
        if selector.technique_version is None and selector.implementation_id is not None:
            if default_key is not None and default_key[2] == selector.implementation_id:
                return self._registrations[default_key]
            candidates.sort(key=lambda item: _semantic_version_key(item[0][1]))
            return candidates[-1][1]
        if len(candidates) == 1:
            return candidates[0][1]
        raise ValueError(
            f"technique '{canonical_id}' selector is ambiguous; specify a version "
            "and implementation or configure a default"
        )

    def set_selectable(
        self,
        technique_id: str,
        *,
        version: str,
        implementation_id: str,
        selectable: bool,
    ) -> None:
        """Control only new/default selection; exact historical pins still resolve."""

        canonical_id = self._canonical_id(technique_id)
        key = (canonical_id, version, implementation_id)
        if key not in self._registrations:
            raise KeyError(
                f"technique '{canonical_id}' is not registered with version "
                f"'{version}' and implementation '{implementation_id}'"
            )
        self._selectable[key] = selectable
        if not selectable and self._defaults.get(canonical_id) == key:
            del self._defaults[canonical_id]

    def snapshot(self) -> CatalogSnapshot:
        return CatalogSnapshot(
            registrations=tuple(self._registrations.values()),
            selectable=self._selectable,
            defaults=self._defaults,
        )

    def restore(self, snapshot: CatalogSnapshot) -> None:
        """Restore a generic snapshot used by an enclosing lifecycle transaction."""

        self._registrations = {}
        self._definitions = {}
        self._aliases = {}
        self._canonical_ids = set()
        self._defaults = {}
        self._selectable = {}
        for registered in snapshot.registrations:
            self.register(registered)
        for key, selectable in snapshot.selectable.items():
            if key in self._registrations:
                self._selectable[key] = selectable
        self._defaults = {
            technique_id: key
            for technique_id, key in snapshot.defaults.items()
            if key in self._registrations and self._selectable.get(key, True)
        }

    def unregister(
        self,
        technique_id: str,
        *,
        version: str,
        implementation_id: str,
    ) -> None:
        """Remove one registration without touching package assets on disk."""

        canonical_id = self._canonical_id(technique_id)
        key = (canonical_id, version, implementation_id)
        if key not in self._registrations:
            raise KeyError(
                f"technique '{canonical_id}' is not registered with version "
                f"'{version}' and implementation '{implementation_id}'"
            )
        del self._registrations[key]
        self._selectable.pop(key, None)
        if not any(existing_key[:2] == key[:2] for existing_key in self._registrations):
            self._definitions.pop(key[:2], None)
        if self._defaults.get(canonical_id) == key:
            del self._defaults[canonical_id]
        self._rebuild_aliases()

    def contains(self, technique_id: str) -> bool:
        return technique_id in self._canonical_ids or technique_id in self._aliases

    def list_definitions(self) -> tuple[RegisteredTechnique, ...]:
        return tuple(
            registered
            for _, registered in sorted(
                self._registrations.items(),
                key=lambda item: (
                    item[0][0],
                    _semantic_version_key(item[0][1]),
                    item[0][2],
                ),
            )
        )

    def _canonical_id(self, technique_id: str) -> str:
        if technique_id in self._canonical_ids:
            return technique_id
        canonical_id = self._aliases.get(technique_id)
        if canonical_id is None:
            raise KeyError(f"technique '{technique_id}' is not registered")
        return canonical_id

    def _validate_aliases(self, technique_id: str, aliases: tuple[str, ...]) -> None:
        for alias in aliases:
            if alias == technique_id:
                continue
            alias_owner = self._aliases.get(alias)
            if alias_owner is not None and alias_owner != technique_id:
                raise ValueError(f"alias '{alias}' is already registered")
            if alias in self._canonical_ids and alias != technique_id:
                raise ValueError(f"alias '{alias}' is already registered")

    def _rebuild_aliases(self) -> None:
        self._aliases = {}
        self._canonical_ids = {key[0] for key in self._registrations}
        for registered in self._registrations.values():
            technique_id = registered.definition.identity.id
            for alias in registered.definition.aliases:
                if alias != technique_id:
                    self._aliases[alias] = technique_id

    @staticmethod
    def _not_found_message(selector: TechniqueSelector, canonical_id: str) -> str:
        qualifiers: list[str] = []
        if selector.technique_version is not None:
            qualifiers.append(f"version '{selector.technique_version}'")
        if selector.implementation_id is not None:
            qualifiers.append(f"implementation '{selector.implementation_id}'")
        suffix = "" if not qualifiers else " with " + " and ".join(qualifiers)
        return f"technique '{canonical_id}' is not registered{suffix}"


def _semantic_version_key(
    version: str,
) -> tuple[int, int, int, int, tuple[tuple[int, int | str], ...]]:
    match = _SEMVER.fullmatch(version)
    if match is None:
        raise ValueError(f"technique version '{version}' is not semantic versioning")
    major, minor, patch = (int(match.group(index)) for index in range(1, 4))
    prerelease = match.group(4)
    if prerelease is None:
        return major, minor, patch, 1, ()
    identifiers: tuple[tuple[int, int | str], ...] = tuple(
        (0, int(identifier)) if identifier.isdigit() else (1, identifier)
        for identifier in prerelease.split(".")
    )
    return major, minor, patch, 0, identifiers
