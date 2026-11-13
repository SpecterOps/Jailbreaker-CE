"""Semantic parity comparison and compatibility imports for parity callers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .canonical import (
    _declares_effective_judge_schedule,
    _deterministic_metrics,
    _routed_configuration_contract,
    _routed_conversation_projection,
    _routed_definition_contract,
    _routed_round_projection,
    _routed_spec_contract,
    _semantic_run,
)
from .canonical import (
    _normalize as _normalize,
)
from .live import (
    compare_live_samples,
)
from .reporting import (
    ParityDifference,
    ParityPolicy,
    ParityReport,
    _difference,
)
from .values import (
    _as_mapping,
    _plain,
)

_DETERMINISTIC_COMPARISONS = (
    ("contract.definition", ("contract", "definition")),
    ("contract.resolved_spec", ("contract", "resolved_spec")),
    ("projections.conversation", ("projections", "conversation")),
    ("projections.rounds", ("projections", "rounds")),
    ("projections.artifacts", ("projections", "artifacts")),
    ("projections.metrics", ("projections", "metrics")),
    ("calls.role_counts", ("calls", "role_counts")),
    ("calls.role_order", ("calls", "role_order")),
    ("calls.semantic", ("calls", "semantic")),
    ("histories.snapshots", ("histories",)),
    ("structure.rounds", ("structure", "rounds")),
    ("structure.generations", ("structure", "generations")),
    ("structure.branches", ("structure", "branches")),
    ("structure.candidates", ("structure", "candidates")),
    ("judges.target_judge.schedule", ("judges", "target_judge", "schedule")),
    ("judges.target_judge.scores", ("judges", "target_judge", "scores")),
    ("transforms.provenance", ("transforms",)),
    ("final_response.selection", ("final_response",)),
    ("outcome.execution_status", ("outcome", "execution_status")),
    ("outcome.result_disposition", ("outcome", "result_disposition")),
    ("outcome.attack_outcome", ("outcome", "attack_outcome")),
    ("outcome.official_score", ("outcome", "official_score")),
    ("outcome.stopping_reason", ("outcome", "stopping_reason")),
    ("experiment.selected_attempts", ("selected_attempts",)),
    ("experiment.plan", ("experiment_plan",)),
    ("execution.provenance", ("execution_provenance",)),
    ("events.semantic", ("events",)),
)


class ParityComparator:
    """Compare captured arms by semantic evidence rather than volatile traces."""

    def compare_deterministic(
        self,
        legacy: object,
        modular: object,
        *,
        policy: ParityPolicy,
    ) -> ParityReport:
        """Strictly compare caller-supplied evidence without routed normalization."""

        if not isinstance(policy, ParityPolicy):
            raise TypeError("policy must be a ParityPolicy")
        from devtools.technique_parity.runtime import _VerifiedDeterministicPair

        if type(legacy) is _VerifiedDeterministicPair or type(modular) is _VerifiedDeterministicPair:
            raise TypeError("runtime-issued deterministic pairs require one-shot consumption")
        legacy_run = _as_mapping(legacy, label="legacy parity run")
        modular_run = _as_mapping(modular, label="modular parity run")
        return self._compare_deterministic_runs(
            legacy_run,
            modular_run,
            policy=policy,
        )

    def _compare_verified_deterministic_pair(
        self,
        pair: object,
        *,
        policy: ParityPolicy,
    ) -> ParityReport:
        """Consume one runtime-issued pair through the internal verification boundary."""

        if not isinstance(policy, ParityPolicy):
            raise TypeError("policy must be a ParityPolicy")
        from devtools.technique_parity.runtime import _consume_verified_deterministic_pair

        legacy, modular = _consume_verified_deterministic_pair(pair)
        legacy_semantic = _semantic_run(
            _as_mapping(legacy, label="verified legacy parity run")
        )
        modular_semantic = _semantic_run(
            _as_mapping(modular, label="verified modular parity run")
        )
        normalize_effective_judge_schedule = any(
            _declares_effective_judge_schedule(source)
            for source in (legacy_semantic, modular_semantic)
        )

        def routed_view(source: Mapping[str, object]) -> Mapping[str, object]:
            view = _plain(source)
            if not isinstance(view, dict):  # pragma: no cover - semantic invariant
                raise TypeError("deterministic semantic evidence must be a mapping")
            contract = view.get("contract")
            if isinstance(contract, dict):
                contract["definition"] = _routed_definition_contract(
                    contract.get("definition"),
                    normalize_effective_judge_schedule=(
                        normalize_effective_judge_schedule
                    ),
                )
                contract["resolved_spec"] = _routed_spec_contract(
                    contract.get("resolved_spec")
                )
            projections = view.get("projections")
            if isinstance(projections, dict):
                projections["conversation"] = _routed_conversation_projection(
                    projections.get("conversation")
                )
                projections["rounds"] = _routed_round_projection(
                    projections.get("rounds")
                )
            selected_attempts = view.get("selected_attempts")
            if isinstance(selected_attempts, list):
                for cell in selected_attempts:
                    if not isinstance(cell, dict):
                        continue
                    configuration = cell.get("configuration")
                    if isinstance(configuration, Mapping):
                        cell["configuration"] = _routed_configuration_contract(
                            configuration
                        )
                    attempts = cell.get("attempts")
                    if isinstance(attempts, list):
                        cell["attempts"] = [
                            routed_attempt(attempt) for attempt in attempts
                        ]
                    selected = cell.get("selected")
                    if isinstance(selected, Mapping):
                        cell["selected"] = routed_attempt(selected)
            plan = view.get("experiment_plan")
            if isinstance(plan, dict):
                for key in ("technique_config_id", "cell_id"):
                    plan.pop(key, None)
            return view

        def routed_attempt(attempt: object) -> object:
            if not isinstance(attempt, Mapping):
                return attempt
            comparable = dict(attempt)
            execution = comparable.get("execution")
            if not isinstance(execution, Mapping):
                return comparable
            routed_execution = dict(execution)
            routed_execution["conversation_projection"] = (
                _routed_conversation_projection(
                    routed_execution.get("conversation_projection")
                )
            )
            resolved_spec = routed_execution.get("resolved_spec")
            if isinstance(resolved_spec, Mapping):
                routed_spec = _routed_configuration_contract(
                    {"resolved_spec": resolved_spec}
                )
                routed_execution["resolved_spec"] = routed_spec.get("resolved_spec")
            comparable["execution"] = routed_execution
            return comparable

        legacy_semantic = routed_view(legacy_semantic)
        modular_semantic = routed_view(modular_semantic)
        differences: list[ParityDifference] = []
        for path, location in _DETERMINISTIC_COMPARISONS:
            left = _at(legacy_semantic, location)
            right = _at(modular_semantic, location)
            if left != right:
                differences.append(_difference(path, left, right))
        for arm, semantic in (("legacy", legacy_semantic), ("modular", modular_semantic)):
            issues = semantic.get("validation", [])
            if not isinstance(issues, list):
                continue
            for issue in issues:
                issue_mapping = issue if isinstance(issue, Mapping) else {"reason": str(issue)}
                issue_path = str(issue_mapping.get("path") or "validation")
                differences.append(
                    ParityDifference(
                        path=issue_path,
                        legacy=issue_mapping if arm == "legacy" else None,
                        modular=issue_mapping if arm == "modular" else None,
                        message=f"{arm} parity evidence has invalid semantic provenance",
                    )
                )
        return ParityReport(
            mode="deterministic",
            passed=not differences,
            policy=policy,
            differences=tuple(differences),
            metrics={
                "legacy": _deterministic_metrics(legacy_semantic),
                "modular": _deterministic_metrics(modular_semantic),
            },
        )

    def _compare_deterministic_runs(
        self,
        legacy_run: Mapping[str, object],
        modular_run: Mapping[str, object],
        *,
        policy: ParityPolicy,
    ) -> ParityReport:
        legacy_semantic = _semantic_run(legacy_run)
        modular_semantic = _semantic_run(modular_run)
        differences: list[ParityDifference] = []
        for path, location in _DETERMINISTIC_COMPARISONS:
            left = _at(legacy_semantic, location)
            right = _at(modular_semantic, location)
            if left != right:
                differences.append(_difference(path, left, right))
        for arm, semantic in (("legacy", legacy_semantic), ("modular", modular_semantic)):
            issues = semantic.get("validation", [])
            if not isinstance(issues, list):
                continue
            for issue in issues:
                issue_mapping = issue if isinstance(issue, Mapping) else {"reason": str(issue)}
                issue_path = str(issue_mapping.get("path") or "validation")
                differences.append(
                    ParityDifference(
                        path=issue_path,
                        legacy=issue_mapping if arm == "legacy" else None,
                        modular=issue_mapping if arm == "modular" else None,
                        message=f"{arm} parity evidence has invalid semantic provenance",
                    )
                )
        return ParityReport(
            mode="deterministic",
            passed=not differences,
            policy=policy,
            differences=tuple(differences),
            metrics={
                "legacy": _deterministic_metrics(legacy_semantic),
                "modular": _deterministic_metrics(modular_semantic),
            },
        )

    def compare_live_samples(
        self,
        legacy: Iterable[object],
        modular: Iterable[object],
        *,
        policy: ParityPolicy,
    ) -> ParityReport:
        return compare_live_samples(legacy, modular, policy=policy)


def _at(value: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = value
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current
