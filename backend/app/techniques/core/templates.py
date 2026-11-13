"""Strict rendering of loader-approved package templates."""

from __future__ import annotations

import string
from types import MappingProxyType
from typing import Mapping


class TemplatePolicyError(ValueError):
    pass


class TemplateService:
    def __init__(self, *, assets: Mapping[str, str]) -> None:
        self._assets = MappingProxyType(dict(assets))

    def read_asset(self, asset_id: str) -> str:
        """Return an exact loader-approved text asset without template parsing."""

        asset = self._assets.get(asset_id)
        if asset is None:
            raise TemplatePolicyError(f"text asset '{asset_id}' is not approved")
        return asset

    def render_asset(self, asset_id: str, *, fields: Mapping[str, object]) -> str:
        template = self._assets.get(asset_id)
        if template is None:
            raise TemplatePolicyError(f"template asset '{asset_id}' is not approved")
        required: set[str] = set()
        for _, field_name, _, _ in string.Formatter().parse(template):
            if field_name is None:
                continue
            if not field_name.isidentifier():
                raise TemplatePolicyError(
                    f"template field '{field_name}' must be a simple identifier"
                )
            required.add(field_name)
        missing = required - set(fields)
        extra = set(fields) - required
        if missing:
            raise TemplatePolicyError(f"template missing fields: {sorted(missing)!r}")
        if extra:
            raise TemplatePolicyError(f"template has extra fields: {sorted(extra)!r}")
        return template.format_map(dict(fields))
