"""Live-sample efficacy, operation metrics, and policy gates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping

from .canonical import (
    _canonical_attempts,
    _normalize,
    _resolve_selected_attempt,
)
from .reporting import (
    ParityDifference,
    ParityPolicy,
    ParityReport,
    _difference,
)
from .values import (
    _as_mapping,
    _is_number,
    _plain,
    _sort_key,
)


def compare_live_samples(
    legacy: Iterable[object],
    modular: Iterable[object],
    *,
    policy: ParityPolicy,
) -> ParityReport:
    """Compare selected-attempt efficacy while retaining all-attempt operations."""

    if not isinstance(policy, ParityPolicy):
        raise TypeError("policy must be a ParityPolicy")
    policy.validate_live_configuration()
    legacy_metrics, legacy_identities = _live_metrics(tuple(legacy), policy)
    modular_metrics, modular_identities = _live_metrics(tuple(modular), policy)
    differences: list[ParityDifference] = []

    _append_live_exact_difference(
        differences,
        "live.sample_count",
        legacy_metrics["sample_count"],
        modular_metrics["sample_count"],
    )
    _append_live_exact_difference(
        differences,
        "live.scored_sample_count",
        legacy_metrics["scored_sample_count"],
        modular_metrics["scored_sample_count"],
    )
    if legacy_identities is not None and modular_identities is not None:
        _append_live_exact_difference(
            differences,
            "live.cell_identities",
            legacy_identities,
            modular_identities,
        )
    _append_live_tolerance_difference(
        differences,
        "live.mean_score",
        legacy_metrics["mean_score"],
        modular_metrics["mean_score"],
        policy.max_mean_score_delta,
    )
    _append_live_tolerance_difference(
        differences,
        "live.success_rate",
        legacy_metrics["success_rate"],
        modular_metrics["success_rate"],
        policy.max_success_rate_delta,
    )
    for arm, metrics in (("legacy", legacy_metrics), ("modular", modular_metrics)):
        sample_count = int(metrics["sample_count"])
        if sample_count < policy.minimum_sample_count:
            differences.append(
                _difference(
                    f"live.{arm}.minimum_sample_count",
                    sample_count,
                    policy.minimum_sample_count,
                )
            )
        expected_sample_count = policy.expected_live_sample_count
        if expected_sample_count is not None and sample_count != expected_sample_count:
            differences.append(
                _difference(
                    f"live.{arm}.expected_sample_count",
                    sample_count,
                    expected_sample_count,
                )
            )
    if policy.max_total_attempt_delta is not None:
        attempt_delta = abs(
            int(legacy_metrics["total_attempts"]) - int(modular_metrics["total_attempts"])
        )
        if attempt_delta > policy.max_total_attempt_delta:
            differences.append(
                _difference(
                    "live.total_attempts",
                    legacy_metrics["total_attempts"],
                    modular_metrics["total_attempts"],
                )
            )
    return ParityReport(
        mode="live",
        passed=not differences,
        policy=policy,
        differences=tuple(differences),
        metrics={"legacy": legacy_metrics, "modular": modular_metrics},
    )


def _live_metrics(
    samples: tuple[object, ...],
    policy: ParityPolicy,
) -> tuple[dict[str, object], list[object] | None]:
    scores: list[float] = []
    sample_count = 0
    total_attempts = 0
    status_counts: Counter[str] = Counter()
    identities: list[object] = []
    seen_identities: set[str] = set()
    cell_repeats: dict[str, set[int]] = {}
    all_identified = True
    for raw_sample in samples:
        sample = _as_mapping(raw_sample, label="live parity sample")
        if sample.get("dataset_id") != policy.dataset_id:
            raise ValueError("live sample dataset evidence does not match the pinned dataset")
        sample_profiles = _plain(sample.get("profile_ids"))
        if sample_profiles != _plain(policy.profile_ids):
            raise ValueError("live sample profile evidence does not match pinned profiles")
        repeat_index = sample.get("repeat_index")
        if (
            type(repeat_index) is not int
            or policy.repeat_count is None
            or repeat_index < 1
            or repeat_index > policy.repeat_count
        ):
            raise ValueError("live sample repeat evidence is outside the pinned repeat count")
        attempts = _plain(sample.get("attempts", ()))
        if not isinstance(attempts, list):
            raise TypeError("live sample attempts must be an explicit sequence")
        attempt_mappings = [
            _as_mapping(attempt, label="live parity attempt") for attempt in attempts
        ]
        _, selected = _resolve_selected_attempt(sample, attempt_mappings)
        _canonical_attempts(attempt_mappings)
        total_attempts += len(attempt_mappings)
        for attempt in attempt_mappings:
            status_counts[
                str(attempt.get("status", attempt.get("execution_status", "unknown")))
            ] += 1
            _validated_target_judge_score(attempt, policy=policy)

        score = _validated_target_judge_score(selected, policy=policy)
        _validate_attempt_profile_evidence(
            selected,
            policy=policy,
            required_roles={"target"},
        )
        sample_count += 1
        if score is not None:
            scores.append(float(score))

        identity = {
            key: sample[key]
            for key in ("technique_key", "prompt_id", "repeat_index")
            if key in sample
        }
        if identity:
            normalized_identity = _normalize(identity)
            identity_key = _sort_key(normalized_identity)
            if identity_key in seen_identities:
                raise ValueError("live efficacy requires one selected attempt per stable cell identity")
            seen_identities.add(identity_key)
            identities.append(normalized_identity)
            base_identity = {
                key: sample[key]
                for key in ("technique_key", "prompt_id")
                if key in sample
            }
            if not isinstance(base_identity.get("prompt_id"), str) or not base_identity[
                "prompt_id"
            ]:
                raise ValueError("live sample requires a non-empty prompt dataset identity")
            base_key = _sort_key(_normalize(base_identity))
            cell_repeats.setdefault(base_key, set()).add(repeat_index)
        else:
            all_identified = False
    if not all_identified:
        raise ValueError("live sample requires a stable prompt and repeat identity")
    expected_sample_count = policy.expected_live_sample_count
    if expected_sample_count is not None and sample_count == expected_sample_count:
        expected_repeats = set(range(1, (policy.repeat_count or 0) + 1))
        if any(repeats != expected_repeats for repeats in cell_repeats.values()):
            raise ValueError("live sample repeat coverage does not match the pinned policy")
        expected_cells = expected_sample_count // (policy.repeat_count or 1)
        if len(cell_repeats) != expected_cells:
            raise ValueError("live dataset cell ownership does not match the pinned policy")
    mean_score = sum(scores) / len(scores) if scores else None
    success_rate = (
        sum(score >= policy.success_threshold for score in scores) / sample_count
        if sample_count
        else None
    )
    return (
        {
            "sample_count": sample_count,
            "scored_sample_count": len(scores),
            "mean_score": mean_score,
            "success_rate": success_rate,
            "total_attempts": total_attempts,
            "attempt_status_counts": dict(sorted(status_counts.items())),
        },
        sorted(identities, key=_sort_key) if all_identified else None,
    )


def _validate_attempt_profile_evidence(
    attempt: Mapping[str, object],
    *,
    policy: ParityPolicy,
    required_roles: set[str],
) -> None:
    raw_profiles = _plain(attempt.get("profile_ids"))
    if not isinstance(raw_profiles, Mapping):
        raise ValueError("selected attempt requires explicit model role profile evidence")
    profiles = {str(role): profile_id for role, profile_id in raw_profiles.items()}
    for role, profile_id in profiles.items():
        pinned_profile = policy.profile_ids.get(role)
        if pinned_profile is None:
            raise ValueError("selected attempt used an unapproved model role profile")
        if profile_id != pinned_profile:
            raise ValueError("selected attempt model role profile mismatches the pinned policy")

    raw_calls = _plain(attempt.get("calls"))
    if not isinstance(raw_calls, list):
        raise ValueError("selected attempt requires explicit model call profile evidence")
    called_roles: set[str] = set()
    for raw_call in raw_calls:
        call = _as_mapping(raw_call, label="live model call profile evidence")
        role = call.get("role")
        profile_id = call.get("profile_id")
        if not isinstance(role, str) or not role or not isinstance(profile_id, str) or not profile_id:
            raise ValueError("live model calls require non-empty role and profile evidence")
        pinned_profile = policy.profile_ids.get(role)
        if pinned_profile is None:
            raise ValueError("selected attempt used an unapproved model role")
        if profile_id != pinned_profile or profiles.get(role) != pinned_profile:
            raise ValueError("model call profile evidence mismatches the selected attempt")
        called_roles.add(role)
    if set(profiles) != called_roles:
        raise ValueError("selected attempt and model call role profile evidence must match exactly")
    if not required_roles.issubset(called_roles):
        missing = ", ".join(sorted(required_roles - called_roles))
        raise ValueError(f"selected attempt is missing required {missing} profile evidence")


def _validated_target_judge_score(
    attempt: Mapping[str, object],
    *,
    policy: ParityPolicy,
) -> float | int | None:
    score = attempt.get("score")
    raw_judge = attempt.get("official_judge")
    if raw_judge is None:
        if score is not None:
            raise ValueError("numeric live score requires explicit succeeded target_judge evidence")
        return None
    judge = _as_mapping(raw_judge, label="live official judge evidence")
    judge_score = judge.get("score")
    _validate_attempt_profile_evidence(
        attempt,
        policy=policy,
        required_roles={"target", "target_judge"},
    )
    if judge.get("profile_id") != policy.profile_ids.get("target_judge"):
        raise ValueError("target_judge record profile evidence mismatches the pinned policy")
    if judge.get("definition_id") != "target_judge" or judge.get("status") != "succeeded":
        if _is_number(score) or _is_number(judge_score):
            raise ValueError("numeric live score requires explicit succeeded target_judge evidence")
        return None
    if not _is_number(judge_score):
        raise ValueError("succeeded target_judge evidence must carry a numeric score")
    if score is not None and (not _is_number(score) or float(score) != float(judge_score)):
        raise ValueError("selected score must match succeeded target_judge evidence")
    return judge_score


def _append_live_exact_difference(
    differences: list[ParityDifference],
    path: str,
    legacy: object,
    modular: object,
) -> None:
    if legacy != modular:
        differences.append(_difference(path, legacy, modular))


def _append_live_tolerance_difference(
    differences: list[ParityDifference],
    path: str,
    legacy: object,
    modular: object,
    tolerance: float,
) -> None:
    if legacy is None and modular is None:
        return
    if not _is_number(legacy) or not _is_number(modular):
        differences.append(_difference(path, legacy, modular))
        return
    if abs(float(legacy) - float(modular)) > tolerance:
        differences.append(_difference(path, legacy, modular))
