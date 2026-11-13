"""Pinned parity policies and redacted comparison reports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

from app.techniques.contracts.execution import _freeze_json_mapping
from app.techniques.core.redaction import contains_secret_material

from .values import (
    _empty_mapping,
    _is_number,
    _plain,
    _safe_evidence,
    _sanitize_report_value,
)


@dataclass(frozen=True, slots=True)
class ParityPolicy:
    """Predeclared deterministic/live gates and their pinned fingerprint."""

    policy_id: str = "deterministic-strict"
    version: str = "1"
    max_mean_score_delta: float = 0.0
    max_success_rate_delta: float = 0.0
    success_threshold: float = 7.0
    minimum_sample_count: int = 0
    expected_sample_count: int | None = None
    max_total_attempt_delta: int | None = None
    dataset_id: str | None = None
    repeat_count: int | None = None
    dataset_sample_count: int | None = None
    profile_ids: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=_empty_mapping)
    _source_bytes_hash: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("policy_id", "version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("max_mean_score_delta", "max_success_rate_delta"):
            value = getattr(self, name)
            if not _is_number(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if not _is_number(self.success_threshold):
            raise ValueError("success_threshold must be a finite number")
        for name in ("minimum_sample_count", "expected_sample_count", "max_total_attempt_delta"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be absent or a non-negative integer")
        if self.dataset_id is not None and (
            not isinstance(self.dataset_id, str) or not self.dataset_id
        ):
            raise ValueError("dataset_id must be absent or a non-empty string")
        for name in ("repeat_count", "dataset_sample_count"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be absent or a positive integer")
        profiles = dict(self.profile_ids)
        if not all(
            isinstance(role, str)
            and role
            and isinstance(profile_id, str)
            and profile_id
            for role, profile_id in profiles.items()
        ):
            raise ValueError("profile_ids must map non-empty roles to non-empty profile IDs")
        object.__setattr__(self, "profile_ids", MappingProxyType(dict(sorted(profiles.items()))))
        derived = self._derived_sample_count
        if (
            derived is not None
            and self.expected_sample_count is not None
            and derived != self.expected_sample_count
        ):
            raise ValueError("expected_sample_count must match repeats times dataset_sample_count")
        if contains_secret_material(self.metadata):
            raise ValueError("parity policy metadata must not contain secret material")
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))

    @classmethod
    def strict(cls) -> ParityPolicy:
        return cls()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        require_explicit_success_threshold: bool = False,
    ) -> ParityPolicy:
        if contains_secret_material(value):
            raise ValueError("parity policy must not contain secret material")
        if require_explicit_success_threshold and "success_threshold" not in value:
            raise ValueError(
                "live parity policy must explicitly declare success_threshold"
            )
        allowed = {
            "policy_id",
            "version",
            "max_mean_score_delta",
            "max_success_rate_delta",
            "success_threshold",
            "minimum_sample_count",
            "expected_sample_count",
            "max_total_attempt_delta",
            "dataset_id",
            "repeat_count",
            "dataset_sample_count",
            "profile_ids",
            "metadata",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown parity policy fields: {', '.join(unknown)}")
        return cls(**dict(value))  # type: ignore[arg-type]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        require_explicit_success_threshold: bool = False,
    ) -> ParityPolicy:
        source = Path(path)
        raw = source.read_bytes()
        try:
            loaded = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid parity policy file: {source}") from exc
        if not isinstance(loaded, Mapping):
            raise ValueError("parity policy file must contain a mapping")
        policy = cls.from_mapping(
            {str(key): item for key, item in loaded.items()},
            require_explicit_success_threshold=require_explicit_success_threshold,
        )
        object.__setattr__(policy, "_source_bytes_hash", hashlib.sha256(raw).hexdigest())
        return policy

    @property
    def fingerprint(self) -> str:
        canonical: dict[str, object] = {
            "policy_id": self.policy_id,
            "version": self.version,
            "max_mean_score_delta": self.max_mean_score_delta,
            "max_success_rate_delta": self.max_success_rate_delta,
            "success_threshold": self.success_threshold,
            "minimum_sample_count": self.minimum_sample_count,
            "expected_sample_count": self.expected_sample_count,
            "max_total_attempt_delta": self.max_total_attempt_delta,
            "dataset_id": self.dataset_id,
            "repeat_count": self.repeat_count,
            "dataset_sample_count": self.dataset_sample_count,
            "profile_ids": _plain(self.profile_ids),
            "metadata": _plain(self.metadata),
        }
        if self._source_bytes_hash is not None:
            canonical = {
                "effective_policy": canonical,
                "source_bytes_sha256": self._source_bytes_hash,
            }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        serialized = {
            "policy_id": self.policy_id,
            "version": self.version,
            "fingerprint": self.fingerprint,
            "max_mean_score_delta": self.max_mean_score_delta,
            "max_success_rate_delta": self.max_success_rate_delta,
            "success_threshold": self.success_threshold,
            "minimum_sample_count": self.minimum_sample_count,
            "expected_sample_count": self.expected_sample_count,
            "max_total_attempt_delta": self.max_total_attempt_delta,
            "dataset_id": self.dataset_id,
            "repeat_count": self.repeat_count,
            "dataset_sample_count": self.dataset_sample_count,
            "profile_ids": _plain(self.profile_ids),
            "metadata": _plain(self.metadata),
        }
        sanitized = _sanitize_report_value(serialized)
        if not isinstance(sanitized, dict):  # pragma: no cover - mapping invariant
            raise TypeError("serialized parity policy must be a mapping")
        return sanitized

    @property
    def _derived_sample_count(self) -> int | None:
        if self.repeat_count is None or self.dataset_sample_count is None:
            return None
        return self.repeat_count * self.dataset_sample_count

    @property
    def expected_live_sample_count(self) -> int | None:
        return self._derived_sample_count or self.expected_sample_count

    def validate_live_configuration(self) -> None:
        if self.dataset_id is None:
            raise ValueError("live parity policy must pin dataset_id")
        if self.repeat_count is None:
            raise ValueError("live parity policy must pin repeat_count")
        if not self.profile_ids:
            raise ValueError("live parity policy must pin profile_ids")
        missing_roles = {"target", "target_judge"} - set(self.profile_ids)
        if missing_roles:
            raise ValueError(
                "live parity policy must pin target and target_judge profile IDs"
            )
        expected = self.expected_live_sample_count
        if expected is None:
            raise ValueError(
                "live parity policy must pin expected_sample_count or dataset_sample_count"
            )
        if expected < 1:
            raise ValueError("live parity expected sample count must be positive")
        if self.repeat_count is not None and expected % self.repeat_count:
            raise ValueError(
                "live parity expected sample count must be divisible by repeat_count"
            )


@dataclass(frozen=True, slots=True)
class ParityDifference:
    path: str
    legacy: object
    modular: object
    message: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("parity difference path must not be empty")
        if not self.message:
            raise ValueError("parity difference message must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": _sanitize_report_value(self.path),
            "legacy": _safe_evidence(self.legacy),
            "modular": _safe_evidence(self.modular),
            "message": _sanitize_report_value(self.message),
        }


@dataclass(frozen=True, slots=True)
class ParityReport:
    mode: str
    passed: bool
    policy: ParityPolicy
    differences: tuple[ParityDifference, ...] = ()
    metrics: Mapping[str, object] = field(default_factory=_empty_mapping)
    metadata: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        if self.mode not in {"deterministic", "live"}:
            raise ValueError("parity report mode must be 'deterministic' or 'live'")
        object.__setattr__(self, "differences", tuple(self.differences))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "passed": self.passed,
            "policy": self.policy.to_dict(),
            "differences": [difference.to_dict() for difference in self.differences],
            "metrics": _sanitize_report_value(self.metrics),
            "metadata": _sanitize_report_value(self.metadata),
        }

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        destination.write_text(f"{serialized}\n", encoding="utf-8")

    def render_text(self) -> str:
        """Render a concise, redacted report suitable for assertion failures."""

        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False)


def _difference(path: str, legacy: object, modular: object) -> ParityDifference:
    return ParityDifference(
        path=path,
        legacy=_safe_evidence(legacy),
        modular=_safe_evidence(modular),
        message=f"legacy and modular semantic evidence differ at {path}",
    )
