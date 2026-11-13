"""Core seed strategies with explicit provenance."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from app.techniques.contracts.definition import SeedDefinition, SeedStrategy


class SeedPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SeedRecord:
    value: str
    strategy: SeedStrategy
    source: str | None
    index: int


class SeedService:
    def __init__(
        self,
        *,
        supplied: tuple[str, ...] = (),
        packaged_assets: Mapping[str, tuple[str, ...]] | None = None,
        sql_libraries: Mapping[str, tuple[str, ...]] | None = None,
        runtime_resolvers: Mapping[str, Callable[[], tuple[str, ...]]] | None = None,
    ) -> None:
        self._supplied = tuple(supplied)
        self._packaged_assets = MappingProxyType(dict(packaged_assets or {}))
        self._sql_libraries = MappingProxyType(dict(sql_libraries or {}))
        self._runtime_resolvers = MappingProxyType(dict(runtime_resolvers or {}))

    def resolve(self, definition: SeedDefinition) -> tuple[SeedRecord, ...]:
        strategy = definition.strategy
        source = definition.source
        if strategy is SeedStrategy.NONE:
            values: tuple[str, ...] = ()
        elif strategy is SeedStrategy.SUPPLIED:
            values = self._supplied
        elif strategy is SeedStrategy.PACKAGED_ASSET:
            values = self._lookup(self._packaged_assets, source, "packaged seed asset")
        elif strategy is SeedStrategy.SQL_LIBRARY:
            values = self._lookup(self._sql_libraries, source, "seed library")
        elif strategy is SeedStrategy.RUNTIME:
            resolver = self._runtime_resolvers.get(source or "")
            if resolver is None:
                raise SeedPolicyError(f"runtime seed source '{source}' is unavailable")
            values = tuple(resolver())
        else:  # pragma: no cover - enum exhaustiveness
            raise SeedPolicyError(f"unsupported seed strategy '{strategy}'")
        if definition.required and not values:
            raise SeedPolicyError("required seed resolution returned no values")
        return tuple(
            SeedRecord(value=value, strategy=strategy, source=source, index=index)
            for index, value in enumerate(values)
        )

    @staticmethod
    def _lookup(
        values: Mapping[str, tuple[str, ...]],
        source: str | None,
        label: str,
    ) -> tuple[str, ...]:
        if source is None or source not in values:
            raise SeedPolicyError(f"{label} '{source}' is unavailable")
        return tuple(values[source])
