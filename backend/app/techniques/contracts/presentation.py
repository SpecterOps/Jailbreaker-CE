"""Declarative presentation and metric projection contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from app.techniques.contracts.execution import _freeze_json_mapping


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class PresentationSection:
    id: str
    kind: str
    title: str
    source: str
    order: int = 0
    visible_when: Mapping[str, object] = field(default_factory=_empty_mapping)
    options: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "visible_when",
            _freeze_json_mapping(self.visible_when),
        )
        object.__setattr__(self, "options", _freeze_json_mapping(self.options))


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    id: str
    name: str
    source: str
    aggregation: str
    description: str = ""
    unit: str | None = None
    group_by: tuple[str, ...] = ()
    filters: Mapping[str, object] = field(default_factory=_empty_mapping)
    options: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_by", tuple(self.group_by))
        object.__setattr__(self, "filters", _freeze_json_mapping(self.filters))
        object.__setattr__(self, "options", _freeze_json_mapping(self.options))
