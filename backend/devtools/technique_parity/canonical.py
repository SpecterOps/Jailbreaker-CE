"""Canonical semantic evidence for deterministic and live parity."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from app.experiment_plan_integrity import is_legacy_partial_technique_config

from .values import (
    _RESPONSE_KEYS,
    _as_mapping,
    _content_fingerprint,
    _is_number,
    _normalized_text,
    _plain,
    _sort_key,
)

_VOLATILE_KEYS = frozenset(
    {
        "checkpoint_id",
        "event_id",
        "history_snapshot_id",
        "snapshot_id",
        "timestamp",
        "created_at",
        "started_at",
        "completed_at",
        "updated_at",
        "model_call_id",
        "provider_response_id",
        "gateway_call_id",
        "generated_call_id",
        "target_call_id",
        "run_id",
        "attempt_id",
        "experiment_id",
        "cell_id",
        "selected_attempt_id",
        "latest_attempt_id",
        "active_attempt_id",
        "correlation",
        "implementation_id",
        "legacy",
        "sequence",
    }
)


_REFERENCE_KEYS = {
    "caused_by": "event",
    "source_event_id": "event",
    "final_response_event_id": "target",
    "target_response_event_id": "target",
    "original_response_event_id": "target",
    "official_judge_event_id": "judge",
    "original_message_id": "message",
    "subject_message_id": "message",
    "real_response_message_id": "message",
    "audit_user_message_id": "message",
    "attacker_input_message_id": "message",
    "attacker_output_message_id": "message",
    "target_prompt_message_id": "message",
    "transformed_message_id": "message",
    "source_artifact_id": "artifact",
    "final_response_ref": "target",
    "official_judge_ref": "judge",
}


_REFERENCE_COLLECTION_KEYS = {
    "active_message_ids": "message",
    "inactive_message_ids": "message",
    "message_ids": "message",
    "rolled_back_message_ids": "message",
    "artifact_refs": "artifact",
    "auxiliary_judge_refs": "judge",
    "source_response_event_ids": "target",
    "subject_refs": "artifact",
}


_GENERATED_IDENTITY_KEYS = frozenset({"message_id", "artifact_id", "result_id"})


_CANONICAL_CONFIG_IDENTITY_FIELDS = (
    "technique_id",
    "technique_version",
    "implementation_id",
    "implementation_kind",
    "contract_version",
    "manifest_hash",
)


def _is_volatile_key(key: str) -> bool:
    normalized = key.strip().lower()
    return (
        normalized in _VOLATILE_KEYS
        or normalized.endswith("latency_ms")
        or normalized.endswith("_timestamp")
    )


@dataclass(slots=True)
class _ReferenceResolver:
    references: dict[str, dict[str, object]]
    issues: list[dict[str, object]]

    @classmethod
    def from_events(cls, events: Sequence[Mapping[str, object]]) -> _ReferenceResolver:
        resolver = cls(
            references={kind: {} for kind in ("event", "message", "artifact", "judge", "target")},
            issues=[],
        )
        occurrences: Counter[str] = Counter()
        for event in events:
            base = _event_identity_base(event)
            base_key = _sort_key(base)
            occurrence = occurrences[base_key]
            occurrences[base_key] += 1
            event_token = {"semantic": base, "occurrence": occurrence}
            resolver._register("event", event.get("event_id"), event_token)

            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                continue
            message_id = payload.get("message_id")
            if isinstance(message_id, str):
                resolver._register(
                    "message",
                    message_id,
                    {
                        "history_id": payload.get("history_id"),
                        "role": payload.get("role"),
                        "content": _identity_content(payload.get("content")),
                        "round": event.get("round", payload.get("round")),
                        "branch_id": event.get("branch_id", payload.get("branch_id")),
                        "occurrence": occurrence,
                    },
                )
            artifact_id = payload.get("artifact_id")
            if isinstance(artifact_id, str):
                resolver._register(
                    "artifact",
                    artifact_id,
                    {
                        "event_type": event.get("event_type"),
                        "media_type": payload.get("media_type"),
                        "content_hash": payload.get("content_hash"),
                        "size_bytes": payload.get("size_bytes"),
                        "kind": payload.get("kind", payload.get("artifact_kind")),
                        "occurrence": occurrence,
                    },
                )
            is_derived_response = (
                event.get("event_type") == "value.transformed"
                and payload.get("response_kind") == "derived"
                and isinstance(payload.get("response"), str)
            )
            if event.get("event_type") == "target.completed" or is_derived_response:
                target_token = {
                    "response_kind": payload.get("response_kind", "real"),
                    "content": _identity_content(payload.get("response")),
                    "round": event.get("round", payload.get("round")),
                    "branch_id": event.get("branch_id", payload.get("branch_id")),
                    "occurrence": occurrence,
                }
                resolver._register("target", event.get("event_id"), target_token)
                for key in ("response_ref", "value_ref"):
                    resolver._register("target", payload.get(key), target_token)
            if payload.get("definition_id") == "target_judge" and str(
                event.get("event_type") or ""
            ).startswith("judge."):
                judge_token = {
                    "definition_id": "target_judge",
                    "status": _judge_status(event, payload),
                    "score": payload.get("score"),
                    "round": event.get("round", payload.get("round")),
                    "generation": payload.get("generation"),
                    "branch_id": event.get("branch_id", payload.get("branch_id")),
                    "occurrence": occurrence,
                }
                resolver._register("judge", event.get("event_id"), judge_token)
                for key in ("result_id", "result_ref", "value_ref"):
                    resolver._register("judge", payload.get(key), judge_token)
        return resolver

    def _register(self, kind: str, raw: object, semantic: object) -> None:
        if not isinstance(raw, str) or not raw:
            return
        existing = self.references[kind].get(raw)
        if existing is not None and existing != semantic:
            raise ValueError(f"duplicate generated {kind} ID resolves ambiguously")
        self.references[kind][raw] = semantic

    def resolve(self, kind: str, value: object, *, path: str) -> object:
        if isinstance(value, list):
            return [self.resolve(kind, item, path=path) for item in value]
        if value is None:
            return None
        if not isinstance(value, str):
            self.issues.append({"path": path, "reason": f"{kind} reference is not a string"})
            return {"resolution": "invalid", "kind": kind}
        semantic = self.references[kind].get(value)
        if semantic is None:
            self.issues.append({"path": path, "reason": f"unresolved {kind} reference"})
            return {"resolution": "unresolved", "kind": kind}
        return semantic


def _identity_content(value: object) -> object:
    return _content_fingerprint(value) if isinstance(value, str) else value


def _event_identity_base(event: Mapping[str, object]) -> dict[str, object]:
    return {
        "event_type": event.get("event_type"),
        "round": event.get("round"),
        "branch_id": event.get("branch_id"),
        "payload": _identity_value(event.get("payload", {})),
    }


def _identity_value(value: object, *, key: str | None = None) -> object:
    plain = _plain(value)
    if key is not None and (
        _is_volatile_key(key)
        or key in _GENERATED_IDENTITY_KEYS
        or key in _REFERENCE_KEYS
        or key in _REFERENCE_COLLECTION_KEYS
    ):
        return None
    if key is not None and key.strip().lower() in _RESPONSE_KEYS and isinstance(plain, str):
        return _content_fingerprint(plain)
    if isinstance(plain, Mapping):
        return {
            str(child_key): _identity_value(item, key=str(child_key))
            for child_key, item in sorted(plain.items(), key=lambda pair: str(pair[0]))
            if _identity_value(item, key=str(child_key)) is not None
        }
    if isinstance(plain, list):
        return [_identity_value(item) for item in plain]
    if isinstance(plain, str):
        return _normalized_text(plain)
    return plain


def _normalize(
    value: object,
    *,
    key: str | None = None,
    resolver: _ReferenceResolver | None = None,
    path: str = "$",
) -> object:
    plain = _plain(value)
    if key is not None and key.strip().lower() in _RESPONSE_KEYS and isinstance(plain, str):
        return _content_fingerprint(plain)
    normalized_key = key.strip().lower() if key is not None else None
    if normalized_key in _GENERATED_IDENTITY_KEYS:
        return None
    if resolver is not None and normalized_key in _REFERENCE_KEYS:
        return resolver.resolve(_REFERENCE_KEYS[normalized_key], plain, path=path)
    if resolver is not None and normalized_key in _REFERENCE_COLLECTION_KEYS:
        return resolver.resolve(_REFERENCE_COLLECTION_KEYS[normalized_key], plain, path=path)
    if isinstance(plain, Mapping):
        normalized: dict[str, object] = {}
        for child_key, child in sorted(plain.items(), key=lambda pair: str(pair[0])):
            name = str(child_key)
            if _is_volatile_key(name):
                continue
            child_value = _normalize(
                child,
                key=name,
                resolver=resolver,
                path=f"{path}.{name}",
            )
            if child_value is not None or child is None:
                normalized[name] = child_value
        return normalized
    if isinstance(plain, list):
        return [
            _normalize(item, resolver=resolver, path=f"{path}[{index}]")
            for index, item in enumerate(plain)
        ]
    if isinstance(plain, str):
        return _normalized_text(plain)
    return plain


def _normalize_parallel_collection(
    value: object,
    *,
    resolver: _ReferenceResolver | None = None,
) -> object:
    normalized = _normalize(value, resolver=resolver)
    if isinstance(normalized, list):
        return sorted(normalized, key=_sort_key)
    return normalized


def _semantic_run(run: Mapping[str, object]) -> dict[str, object]:
    raw_events = run.get("events", ())
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes, bytearray)):
        raise TypeError("parity run events must be a sequence")
    events = [_as_mapping(item, label="parity event") for item in raw_events]
    _validate_judge_scores(events)
    resolver = _ReferenceResolver.from_events(events)
    canonical_events = _canonical_events(events, resolver=resolver)
    canonical_calls = _canonical_calls(run.get("calls", ()), resolver=resolver)
    judges, judge_records = _judge_evidence(events)
    outcome = _outcome_evidence(run, judge_records, issues=resolver.issues)
    return {
        "contract": _contract_evidence(
            run,
            resolver=resolver,
        ),
        "projections": _projection_evidence(
            run,
            resolver=resolver,
        ),
        "calls": canonical_calls,
        "events": canonical_events,
        "histories": _history_evidence(run, canonical_events, resolver=resolver),
        "structure": _structure_evidence(run, canonical_events, resolver=resolver),
        "judges": {"target_judge": judges},
        "transforms": _transform_evidence(run, canonical_events, resolver=resolver),
        "final_response": _final_response_evidence(run, events, issues=resolver.issues),
        "outcome": outcome,
        "selected_attempts": _selected_attempt_evidence(run),
        "experiment_plan": _experiment_plan_evidence(run),
        "execution_provenance": _execution_provenance_evidence(
            run,
            events=events,
            resolver=resolver,
        ),
        "validation": resolver.issues,
    }


def _contract_evidence(
    run: Mapping[str, object],
    *,
    resolver: _ReferenceResolver,
) -> dict[str, object]:
    raw_contract = run.get("contract", {})
    contract = _as_mapping(raw_contract, label="parity contract") if raw_contract is not None else {}
    definition = contract.get("definition", run.get("definition"))
    resolved_spec = contract.get(
        "resolved_spec",
        run.get("resolved_spec", run.get("spec")),
    )
    for path, value in (
        ("contract.definition", definition),
        ("contract.resolved_spec", resolved_spec),
    ):
        if not isinstance(_plain(value), Mapping) or not _plain(value):
            resolver.issues.append(
                {"path": path, "reason": "required deterministic contract evidence is missing"}
            )
    normalized_definition = _normalize(
        definition,
        resolver=resolver,
        path="contract.definition",
    )
    normalized_spec = _normalize(
        resolved_spec,
        resolver=resolver,
        path="contract.resolved_spec",
    )
    return {
        "definition": normalized_definition,
        "resolved_spec": normalized_spec,
    }


def _declares_effective_judge_schedule(source: Mapping[str, object]) -> bool:
    contract = source.get("contract")
    if not isinstance(contract, Mapping):
        return False
    definition = contract.get("definition")
    if not isinstance(definition, Mapping):
        return False
    extensions = definition.get("extensions")
    return (
        isinstance(extensions, Mapping)
        and extensions.get("parity_effective_judge_schedule")
        == "judge_per_round"
    )


def _routed_definition_contract(
    value: object,
    *,
    normalize_effective_judge_schedule: bool = False,
) -> object:
    """Compare behavior while exact per-arm package identity stays route-verified."""

    if not isinstance(value, Mapping):
        return value
    comparable = dict(value)
    identity = comparable.get("identity")
    if isinstance(identity, Mapping):
        comparable["identity"] = {
            key: item for key, item in identity.items() if key != "version"
        }
    for field_name in (
        "assets",
        "capabilities",
        "extensions",
        "histories",
        "model_roles",
        "parameters_schema",
        "presets",
        "presentation",
        "recommended_limits",
        "seed",
        "source",
    ):
        comparable.pop(field_name, None)
    inputs = comparable.get("inputs")
    if isinstance(inputs, list):
        # The legacy metadata adapter describes seed inputs only when they are
        # required. Exact route identity still binds the modular definition;
        # deterministic behavior comparison therefore ignores additive,
        # optional inputs that the legacy contract cannot represent.
        comparable["inputs"] = [
            item
            for item in inputs
            if not isinstance(item, Mapping) or item.get("required") is not False
        ]
    judges = comparable.get("judges")
    if isinstance(judges, list):
        ignored_judge_fields = {
            "evidence",
            "policy",
            "purposes",
            "required",
            "threshold",
        }
        if normalize_effective_judge_schedule:
            ignored_judge_fields.add("schedule")
        comparable["judges"] = [
            {
                key: item
                for key, item in judge.items()
                if key not in ignored_judge_fields
            }
            if isinstance(judge, Mapping)
            else judge
            for judge in judges
        ]
    result_policy = comparable.get("result_policy")
    if isinstance(result_policy, Mapping):
        normalized_policy = dict(result_policy)
        modular_final_response = _normalize(
            {"final_response": "target.real_response"}
        )["final_response"]
        if normalized_policy.get("final_response") == modular_final_response:
            normalized_policy["final_response"] = _normalize(
                {"final_response": "target_response"}
            )["final_response"]
        comparable["result_policy"] = normalized_policy
    return comparable


def _routed_spec_contract(value: object) -> object:
    """Retain contract compatibility after exact route provenance validation."""

    if not isinstance(value, Mapping):
        return value
    return {
        key: item
        for key, item in value.items()
        if key not in {"asset_hashes", "manifest_hash", "spec_hash"}
    }


def _projection_evidence(
    run: Mapping[str, object],
    *,
    resolver: _ReferenceResolver,
) -> dict[str, object]:
    raw = run.get("projections", {})
    projections = _as_mapping(raw, label="parity projections") if raw is not None else {}
    aliases = {
        "conversation": "conversation_projection",
        "rounds": "round_projection",
        "artifacts": "artifact_projection",
        "metrics": "metric_projection",
    }
    evidence: dict[str, object] = {}
    for key, alias in aliases.items():
        if key in projections:
            value = projections[key]
        elif alias in run:
            value = run[alias]
        else:
            value = None
        if value is None:
            resolver.issues.append(
                {
                    "path": f"projections.{key}",
                    "reason": "required deterministic projection evidence is missing",
                }
            )
        normalized = _normalize(
            value,
            resolver=resolver,
            path=f"projections.{key}",
        )
        evidence[key] = normalized
    return evidence


def _routed_conversation_projection(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    if value.get("format") == _normalize("single_shot"):
        return {
            key: item
            for key, item in value.items()
            if key not in {"projected_event_count", "sections"}
        }
    return {
        key: item
        for key, item in value.items()
        if key != "projected_event_count"
    }


def _routed_round_projection(value: object) -> object:
    """Remove route identity already proven by the exact arm binding."""

    if not isinstance(value, list):
        return value
    return [
        {
            key: item
            for key, item in event.items()
            if key != "technique_version"
        }
        if isinstance(event, Mapping)
        else event
        for event in value
    ]


def _validate_judge_scores(events: Sequence[Mapping[str, object]]) -> None:
    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        definition_id = payload.get("definition_id")
        if not definition_id and not event_type.startswith("judge."):
            continue
        score = payload.get("score")
        if event_type == "judge.completed":
            status = "succeeded"
        elif event_type == "judge.failed":
            status = "failed"
        else:
            status = str(payload.get("status") or event_type.removeprefix("judge.") or "unknown")
        if _is_number(score) and status != "succeeded":
            raise ValueError("only succeeded judges may carry numeric scores")
        if status == "succeeded" and not _is_number(score):
            raise ValueError("succeeded judges must carry numeric scores")


def _canonical_event(
    event: Mapping[str, object],
    *,
    resolver: _ReferenceResolver,
) -> dict[str, object]:
    normalized = _normalize(event, resolver=resolver, path="events")
    if not isinstance(normalized, dict):  # pragma: no cover - mapping input invariant
        raise TypeError("normalized parity event must be a mapping")
    return normalized


def _canonical_events(
    events: Sequence[Mapping[str, object]],
    *,
    resolver: _ReferenceResolver,
) -> dict[str, object]:
    main: list[dict[str, object]] = []
    branches: dict[str, list[dict[str, object]]] = {}
    for event in events:
        canonical = _canonical_event(event, resolver=resolver)
        branch_id = event.get("branch_id")
        if not isinstance(branch_id, str):
            payload = event.get("payload")
            branch_id = payload.get("branch_id") if isinstance(payload, Mapping) else None
        if isinstance(branch_id, str) and branch_id:
            branches.setdefault(branch_id, []).append(canonical)
        else:
            main.append(canonical)
    return {
        "main": main,
        "branches": {branch_id: branches[branch_id] for branch_id in sorted(branches)},
    }


def _canonical_calls(
    raw_calls: object,
    *,
    resolver: _ReferenceResolver,
) -> dict[str, object]:
    plain = _plain(raw_calls)
    if plain is None:
        calls: list[object] = []
    elif isinstance(plain, list):
        calls = plain
    else:
        raise TypeError("parity run calls must be a sequence")
    main: list[dict[str, object]] = []
    branches: dict[str, list[dict[str, object]]] = {}
    for raw_call in calls:
        call = _as_mapping(raw_call, label="parity model call")
        if call.get("role") == "target_judge":
            call = _canonical_judge_transport_call(call)
        canonical = _normalize(call, resolver=resolver, path="calls")
        if not isinstance(canonical, dict):  # pragma: no cover - mapping input invariant
            raise TypeError("normalized parity call must be a mapping")
        branch_id = call.get("branch_id")
        if isinstance(branch_id, str) and branch_id:
            branches.setdefault(branch_id, []).append(canonical)
        else:
            main.append(canonical)
    ordered = [*main, *(call for branch in sorted(branches) for call in branches[branch])]
    roles = [str(call.get("role") or "") for call in ordered]
    return {
        "role_counts": dict(sorted(Counter(roles).items())),
        "role_order": roles,
        "semantic": {
            "main": main,
            "branches": {branch: branches[branch] for branch in sorted(branches)},
        },
    }


def _canonical_judge_transport_call(
    call: Mapping[str, object],
) -> dict[str, object]:
    """Compare judge evidence while allowing legacy and core prompt transports."""

    canonical = dict(call)
    messages = _plain(call.get("messages"))
    evidence: Mapping[str, object] | None = None
    if isinstance(messages, list) and len(messages) == 1:
        message = messages[0]
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, Mapping):
            evidence = content
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                evidence = parsed
    elif isinstance(messages, list) and len(messages) >= 2:
        message = messages[-1]
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str):
            topic_prefix = "HARMFUL REQUEST TOPIC: "
            response_marker = "\n\nTARGET AI'S RESPONSE:\n"
            procedure_marker = "\n\nFollow these steps IN ORDER:"
            if content.startswith(topic_prefix) and response_marker in content:
                intent, separator, remainder = content.removeprefix(topic_prefix).partition(
                    response_marker
                )
                response, procedure_separator, _ = remainder.partition(procedure_marker)
                if separator and procedure_separator:
                    evidence = {"intent": intent, "response": response}
    if evidence is None:
        return canonical
    canonical["messages"] = [
        {
            "role": "judge_evidence",
            "content": {
                key: evidence[key]
                for key in ("intent", "response")
                if key in evidence
            },
        }
    ]
    canonical["options"] = {"transport_contract": "official-target-judge-v1"}
    return canonical


def _flatten_events(canonical: Mapping[str, object]) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    main = canonical.get("main", [])
    if isinstance(main, list):
        flattened.extend(item for item in main if isinstance(item, dict))
    branches = canonical.get("branches", {})
    if isinstance(branches, Mapping):
        for branch_id in sorted(branches):
            branch = branches[branch_id]
            if isinstance(branch, list):
                flattened.extend(item for item in branch if isinstance(item, dict))
    return flattened


def _history_evidence(
    run: Mapping[str, object],
    canonical_events: Mapping[str, object],
    *,
    resolver: _ReferenceResolver,
) -> object:
    explicit = run.get("histories")
    event_evidence = [
        event
        for event in _flatten_events(canonical_events)
        if str(event.get("event_type") or "").startswith("history.")
    ]
    return {
        "snapshots": (
            _normalize(explicit, resolver=resolver, path="histories")
            if explicit is not None
            else None
        ),
        "events": event_evidence,
    }


def _structure_evidence(
    run: Mapping[str, object],
    canonical_events: Mapping[str, object],
    *,
    resolver: _ReferenceResolver,
) -> dict[str, object]:
    events = _flatten_events(canonical_events)
    rounds = [
        event
        for event in events
        if str(event.get("event_type") or "").startswith("round.")
        or event.get("round") is not None
    ]
    generations = [
        event
        for event in events
        if "generation" in str(event.get("event_type") or "")
        or _mapping_value(event, "payload", "generation") is not None
        or _mapping_value(event, "payload", "generation_id") is not None
    ]
    candidates = [
        event
        for event in events
        if "candidate" in str(event.get("event_type") or "")
        or _mapping_value(event, "payload", "candidate_id") is not None
    ]
    return {
        "rounds": {
            "explicit": _normalize(run.get("rounds"), resolver=resolver, path="rounds"),
            "events": rounds,
        },
        "generations": {
            "explicit": _normalize(
                run.get("generations"),
                resolver=resolver,
                path="generations",
            ),
            "events": generations,
        },
        "branches": {
            "explicit": _normalize_parallel_collection(run.get("branches"), resolver=resolver),
            "events": canonical_events.get("branches", {}),
        },
        "candidates": {
            "explicit": _normalize_parallel_collection(run.get("candidates"), resolver=resolver),
            "events": candidates,
        },
    }


def _mapping_value(value: Mapping[str, object], parent: str, child: str) -> object:
    nested = value.get(parent)
    return nested.get(child) if isinstance(nested, Mapping) else None


def _judge_status(event: Mapping[str, object], payload: Mapping[str, object]) -> str:
    event_type = str(event.get("event_type") or "")
    if event_type == "judge.completed":
        return "succeeded"
    if event_type == "judge.failed":
        return "failed"
    return str(payload.get("status") or event_type.removeprefix("judge.") or "unknown")


def _judge_evidence(
    events: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    evidence: list[dict[str, object]] = []
    local_orders: Counter[tuple[str, str, str]] = Counter()
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or payload.get("definition_id") != "target_judge":
            continue
        event_type = str(event.get("event_type") or "")
        if not event_type.startswith("judge."):
            continue
        status = _judge_status(event, payload)
        round_number = event.get("round", payload.get("round"))
        identity = (
            _sort_key(round_number),
            _sort_key(payload.get("generation")),
            _sort_key(event.get("branch_id", payload.get("branch_id"))),
        )
        local_order = local_orders[identity]
        local_orders[identity] += 1
        item = {
            "round": round_number,
            "generation": payload.get("generation"),
            "branch_id": event.get("branch_id", payload.get("branch_id")),
            "status": status,
            "score": payload.get("score") if status == "succeeded" else None,
            "local_order": local_order,
            "event_id": event.get("event_id"),
            "result_id": payload.get("result_id"),
            "result_ref": payload.get("result_ref"),
            "value_ref": payload.get("value_ref"),
        }
        evidence.append(item)
    evidence.sort(
        key=lambda item: (
            _sort_key(item.get("round")),
            _sort_key(item.get("generation")),
            _sort_key(item.get("branch_id")),
            _sort_key(item.get("local_order")),
        )
    )
    return (
        {
            "schedule": [
                {
                    "round": item["round"],
                    "generation": item["generation"],
                    "branch_id": item["branch_id"],
                    "status": item["status"],
                }
                for item in evidence
            ],
            "scores": [item["score"] for item in evidence if item["status"] == "succeeded"],
        },
        evidence,
    )


def _transform_evidence(
    run: Mapping[str, object],
    canonical_events: Mapping[str, object],
    *,
    resolver: _ReferenceResolver,
) -> object:
    events = [
        event
        for event in _flatten_events(canonical_events)
        if "transform" in str(event.get("event_type") or "")
        or _mapping_value(event, "payload", "transform_provenance") is not None
        or _mapping_value(event, "payload", "provenance") is not None
    ]
    return {
        "explicit": _normalize(run.get("transforms"), resolver=resolver, path="transforms"),
        "events": events,
    }


def _outcome_evidence(
    run: Mapping[str, object],
    judge_records: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    raw_outcome = run.get("outcome", {})
    outcome = _as_mapping(raw_outcome, label="parity outcome") if raw_outcome is not None else {}
    official_score = run.get("official_score", outcome.get("official_score"))
    official_ref = outcome.get("official_judge_ref", run.get("official_judge_ref"))
    official_resolution = "absent"
    matches: list[Mapping[str, object]] = []
    if isinstance(official_ref, str) and official_ref:
        matches = [
            record
            for record in judge_records
            if official_ref
            in {
                record.get("event_id"),
                record.get("result_id"),
                record.get("result_ref"),
                record.get("value_ref"),
            }
        ]
        if len(matches) == 1:
            official_resolution = "resolved"
        else:
            official_resolution = "unresolved" if not matches else "ambiguous"
            issues.append(
                {
                    "path": "outcome.official_judge.provenance",
                    "reason": f"official target_judge reference is {official_resolution}",
                }
            )
    elif official_score is not None:
        issues.append(
            {
                "path": "outcome.official_judge.provenance",
                "reason": "numeric official score has no target_judge reference",
            }
        )

    if len(matches) == 1:
        official = matches[0]
        failed_inconclusive = (
            official.get("status") == "failed"
            and official.get("score") is None
            and official_score is None
            and _enum_value(outcome.get("result_disposition")) == "inconclusive"
            and _enum_value(outcome.get("attack_outcome")) == "inconclusive"
        )
        if failed_inconclusive:
            pass
        elif official.get("status") != "succeeded" or not _is_number(official.get("score")):
            issues.append(
                {
                    "path": "outcome.official_judge.provenance",
                    "reason": "official reference does not identify a succeeded target_judge",
                }
            )
        elif official_score is None:
            official_score = official.get("score")
        elif not _is_number(official_score) or float(official_score) != float(official["score"]):
            issues.append(
                {
                    "path": "outcome.official_judge.provenance",
                    "reason": "official score does not match its succeeded target_judge",
                }
            )
    return {
        "execution_status": _enum_value(outcome.get("execution_status")),
        "result_disposition": _enum_value(outcome.get("result_disposition")),
        "attack_outcome": _enum_value(outcome.get("attack_outcome")),
        "official_score": official_score,
        "official_judge_resolution": official_resolution,
        "stopping_reason": outcome.get("stopping_reason", run.get("stopping_reason")),
    }


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _final_response_evidence(
    run: Mapping[str, object],
    events: Sequence[Mapping[str, object]],
    *,
    issues: list[dict[str, object]],
) -> object:
    outcome = run.get("outcome", {})
    outcome_mapping = _as_mapping(outcome, label="parity outcome") if outcome is not None else {}
    selected_ref = outcome_mapping.get("final_response_ref")
    targets: list[Mapping[str, object]] = []
    for event in events:
        payload = event.get("payload")
        is_derived_response = (
            event.get("event_type") == "value.transformed"
            and isinstance(payload, Mapping)
            and payload.get("response_kind") == "derived"
        )
        if (
            event.get("event_type") in {"target.completed", "value.transformed"}
            and (event.get("event_type") == "target.completed" or is_derived_response)
            and isinstance(payload, Mapping)
            and isinstance(payload.get("response"), str)
        ):
            targets.append(event)
    selected: Mapping[str, object] | None = None
    if isinstance(selected_ref, str):
        matches = [
            candidate
            for candidate in targets
            if candidate.get("event_id") == selected_ref
            or isinstance(candidate.get("payload"), Mapping)
            and selected_ref
            in {
                candidate["payload"].get("response_ref"),  # type: ignore[index]
                candidate["payload"].get("value_ref"),  # type: ignore[index]
            }
        ]
        if len(matches) == 1:
            selected = matches[0]
        else:
            resolution = "unresolved" if not matches else "ambiguous"
            issues.append(
                {
                    "path": "final_response.provenance",
                    "reason": f"final-response reference is {resolution}",
                }
            )
            return {"resolution": resolution}
    elif targets or isinstance(run.get("final_response"), str):
        issues.append(
            {
                "path": "final_response.provenance",
                "reason": "final response has no explicit provenance reference",
            }
        )
        return {"resolution": "missing"}
    if selected is None:
        return None
    payload = selected.get("payload")
    if not isinstance(payload, Mapping):  # pragma: no cover - target filter invariant
        return None
    response = payload.get("response")
    return {
        "resolution": "resolved",
        "response_kind": payload.get("response_kind", "real"),
        "content": _content_fingerprint(response) if isinstance(response, str) else None,
        "round": selected.get("round"),
        "branch_id": selected.get("branch_id"),
    }


def _execution_provenance_evidence(
    run: Mapping[str, object],
    *,
    events: Sequence[Mapping[str, object]],
    resolver: _ReferenceResolver,
) -> object:
    raw = run.get("execution_provenance")
    if raw is None:
        return None
    provenance = _as_mapping(raw, label="execution provenance")
    correlation = _as_mapping(
        provenance.get("correlation"),
        label="execution provenance correlation",
    )
    correlation_owned = True
    for event in events:
        if _plain(event.get("correlation")) != correlation:
            correlation_owned = False
            resolver.issues.append(
                {
                    "path": "execution.provenance",
                    "reason": "selected event correlation mismatches execution provenance",
                }
            )
            break

    final_response = resolver.resolve(
        "target",
        provenance.get("final_response_event_id"),
        path="execution.provenance",
    )
    official_judge = resolver.resolve(
        "judge",
        provenance.get("official_judge_event_id"),
        path="execution.provenance",
    )
    outcome = _as_mapping(run.get("outcome", {}), label="parity outcome")
    outcome_final = resolver.resolve(
        "target",
        outcome.get("final_response_ref"),
        path="execution.provenance",
    )
    outcome_judge = resolver.resolve(
        "judge",
        outcome.get("official_judge_ref"),
        path="execution.provenance",
    )
    if final_response != outcome_final:
        resolver.issues.append(
            {
                "path": "execution.provenance",
                "reason": "selected response provenance disagrees with the projected outcome",
            }
        )
    if official_judge != outcome_judge:
        resolver.issues.append(
            {
                "path": "execution.provenance",
                "reason": "selected target_judge provenance disagrees with the projected outcome",
            }
        )
    return {
        "correlation_owned": correlation_owned,
        "final_response": final_response,
        "official_judge": official_judge,
    }


@dataclass(frozen=True, slots=True)
class _TechniqueConfigurationResolver:
    configurations: Mapping[str, Mapping[str, object]]

    @classmethod
    def from_run(
        cls,
        run: Mapping[str, object],
    ) -> _TechniqueConfigurationResolver:
        raw_configurations = run.get("technique_configurations")
        if raw_configurations is None:
            raw_plan = run.get("experiment_plan", run.get("plan"))
            if isinstance(raw_plan, Mapping):
                raw_configurations = raw_plan.get("technique_configurations")
        plain_configurations = _plain(raw_configurations)
        if not isinstance(plain_configurations, list):
            raise ValueError("parity cells require an explicit technique configuration sequence")
        configurations: dict[str, Mapping[str, object]] = {}
        for raw_configuration in plain_configurations:
            configuration = _as_mapping(
                raw_configuration,
                label="parity technique configuration",
            )
            configuration_id = configuration.get("technique_config_id")
            if not isinstance(configuration_id, str) or not configuration_id:
                raise ValueError("every technique configuration requires a non-empty generated ID")
            if configuration_id in configurations:
                raise ValueError("duplicate technique configuration IDs are ambiguous")
            technique_key = configuration.get("technique_key")
            resolved_spec = configuration.get("resolved_spec")
            if not isinstance(technique_key, str) or not technique_key:
                raise ValueError("technique configuration requires a non-empty technique key")
            if not isinstance(resolved_spec, Mapping) or not resolved_spec:
                raise ValueError("technique configuration requires a pinned resolved spec")
            _validate_configuration_spec_identity(
                configuration,
                technique_key=technique_key,
                resolved_spec=resolved_spec,
            )
            semantic = {
                key: value
                for key, value in configuration.items()
                if key != "technique_config_id"
            }
            normalized = _normalize(semantic)
            if not isinstance(normalized, Mapping):  # pragma: no cover - mapping invariant
                raise TypeError("normalized technique configuration must be a mapping")
            configurations[configuration_id] = normalized
        return cls(configurations=MappingProxyType(configurations))

    def resolve(self, cell: Mapping[str, object]) -> Mapping[str, object]:
        configuration_id = cell.get("technique_config_id")
        if not isinstance(configuration_id, str) or not configuration_id:
            raise ValueError("every parity cell requires a technique configuration reference")
        configuration = self.configurations.get(configuration_id)
        if configuration is None:
            raise ValueError("parity cell technique configuration reference is dangling")
        technique_key = cell.get("technique_key")
        if not isinstance(technique_key, str) or not technique_key:
            raise ValueError("every parity cell requires a non-empty technique key")
        if configuration.get("technique_key") != technique_key:
            raise ValueError("parity cell technique key mismatches its pinned configuration")
        if "resolved_spec" in cell:
            cell_spec = _normalize(cell["resolved_spec"])
            if cell_spec != configuration.get("resolved_spec"):
                raise ValueError("parity cell resolved spec mismatches its pinned configuration")
        return configuration


def _validate_configuration_spec_identity(
    configuration: Mapping[str, object],
    *,
    technique_key: str,
    resolved_spec: Mapping[str, object],
) -> None:
    if is_legacy_partial_technique_config(
        technique_version=configuration.get("technique_version"),
        implementation_id=configuration.get("implementation_id"),
        resolved_spec=resolved_spec,
    ):
        embedded_technique_id = resolved_spec.get("technique_id")
        if embedded_technique_id is not None and embedded_technique_id != technique_key:
            raise ValueError("legacy resolved specification technique identity mismatches config")
        return

    for identity_field in _CANONICAL_CONFIG_IDENTITY_FIELDS:
        embedded = resolved_spec.get(identity_field)
        if not isinstance(embedded, str) or not embedded:
            raise ValueError(
                f"canonical resolved specification requires non-empty {identity_field} identity"
            )
        if identity_field in configuration and configuration[identity_field] != embedded:
            raise ValueError(
                f"technique configuration {identity_field} identity mismatches resolved specification"
            )
    if resolved_spec["technique_id"] != technique_key:
        raise ValueError("resolved specification technique identity mismatches config and cell")


def _routed_configuration_contract(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    comparable = {
        key: item
        for key, item in value.items()
        if key not in {"implementation_id", "spec_hash", "technique_version"}
    }
    resolved_spec = comparable.get("resolved_spec")
    if isinstance(resolved_spec, Mapping):
        comparable["resolved_spec"] = {
            key: item
            for key, item in resolved_spec.items()
            if key
            not in {
                "asset_hashes",
                "implementation_id",
                "implementation_kind",
                "manifest_hash",
                "technique_version",
            }
        }
    return comparable


def _selected_attempt_evidence(run: Mapping[str, object]) -> list[object]:
    raw_cells = run.get("cells")
    if raw_cells is None:
        raw_cells = ()
    cells = _plain(raw_cells)
    if not isinstance(cells, list):
        raise TypeError("parity cells must be a sequence")
    configuration_resolver = (
        _TechniqueConfigurationResolver.from_run(run)
        if cells
        else None
    )
    evidence: list[object] = []
    for ordinal, raw_cell in enumerate(cells):
        cell = _as_mapping(raw_cell, label="parity cell")
        raw_attempts = cell.get("attempts", ())
        attempts = _plain(raw_attempts)
        if not isinstance(attempts, list):
            raise TypeError("cell attempts must be an explicit sequence")
        attempt_mappings = [
            _as_mapping(attempt, label="parity attempt") for attempt in attempts
        ]
        selected_index, _ = _resolve_selected_attempt(cell, attempt_mappings)
        canonical_attempts = _canonical_attempts(attempt_mappings)
        if configuration_resolver is None:  # pragma: no cover - non-empty cells invariant
            raise AssertionError("configuration resolver must exist for parity cells")
        configuration = configuration_resolver.resolve(cell)
        identity = {
            key: cell[key]
            for key in ("technique_key", "prompt_id", "repeat_index")
            if key in cell
        }
        if not identity:
            identity = {"ordinal": ordinal}
        evidence.append(
            {
                "identity": _normalize(identity),
                "configuration": configuration,
                "total_attempts": len(attempt_mappings),
                "attempts": canonical_attempts,
                "selected_index": selected_index,
                "selected": canonical_attempts[selected_index],
            }
        )
    return sorted(evidence, key=_sort_key)


def _resolve_selected_attempt(
    owner: Mapping[str, object],
    attempts: Sequence[Mapping[str, object]],
) -> tuple[int, Mapping[str, object]]:
    if not attempts:
        raise ValueError("selected attempt requires at least one explicit attempt")
    attempt_ids: list[str] = []
    for attempt in attempts:
        attempt_id = attempt.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("every attempt requires a non-empty generated attempt ID")
        if attempt_id in attempt_ids:
            raise ValueError("duplicate attempt IDs are not valid parity evidence")
        attempt_ids.append(attempt_id)

    selected_id = owner.get("selected_attempt_id")
    selected_value = owner.get("selected_attempt")
    if selected_value is not None:
        selected_mapping = _as_mapping(selected_value, label="selected parity attempt")
        embedded_id = selected_mapping.get("attempt_id")
        if not isinstance(embedded_id, str) or not embedded_id:
            raise ValueError("selected attempt must identify membership by attempt_id")
        if selected_id is not None and selected_id != embedded_id:
            raise ValueError("selected attempt evidence is ambiguous")
        selected_id = embedded_id
    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError("selected attempt is required for every parity cell")
    matches = [index for index, attempt_id in enumerate(attempt_ids) if attempt_id == selected_id]
    if len(matches) != 1:
        raise ValueError("selected attempt does not resolve exactly once within its cell")
    index = matches[0]
    if selected_value is not None and _plain(selected_value) != _plain(attempts[index]):
        raise ValueError("selected attempt must be an exact member of the attempts sequence")
    return index, attempts[index]


def _canonical_attempts(
    attempts: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    attempt_ids = [attempt.get("attempt_id") for attempt in attempts]
    if not all(isinstance(attempt_id, str) and attempt_id for attempt_id in attempt_ids):
        raise ValueError("retry topology requires non-empty attempt IDs")
    id_to_index = {attempt_id: index for index, attempt_id in enumerate(attempt_ids)}
    if len(id_to_index) != len(attempt_ids):
        raise ValueError("retry topology requires unique attempt IDs")

    parent_indices: list[int | None] = []
    for index, attempt in enumerate(attempts):
        parent_id = attempt.get("parent_attempt_id")
        if parent_id is None:
            parent_indices.append(None)
            continue
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError("retry parent attempt ID must be a non-empty string")
        parent_index = id_to_index.get(parent_id)
        if parent_index is None:
            raise ValueError("retry parent attempt reference is dangling")
        if parent_index == index:
            raise ValueError("retry attempt cannot be its own parent")
        parent_indices.append(parent_index)

    states = [0] * len(attempts)

    def visit(index: int) -> None:
        if states[index] == 1:
            raise ValueError("retry topology contains a cycle")
        if states[index] == 2:
            return
        states[index] = 1
        parent_index = parent_indices[index]
        if parent_index is not None:
            visit(parent_index)
        states[index] = 2

    for index in range(len(attempts)):
        visit(index)

    canonical: list[Mapping[str, object]] = []
    for index, attempt in enumerate(attempts):
        semantic = {
            key: value
            for key, value in attempt.items()
            if key not in {"attempt_id", "parent_attempt_id", "execution"}
        }
        execution = attempt.get("execution")
        if execution is not None:
            semantic["execution"] = _attempt_execution_semantic(
                attempt,
                _as_mapping(execution, label="attempt execution"),
            )
        normalized = _normalize(semantic)
        if not isinstance(normalized, Mapping):  # pragma: no cover - mapping invariant
            raise TypeError("normalized attempt must be a mapping")
        canonical.append(
            {
                **normalized,
                "parent_attempt_index": parent_indices[index],
            }
        )
    return canonical


def _attempt_execution_semantic(
    attempt: Mapping[str, object],
    execution: Mapping[str, object],
) -> Mapping[str, object]:
    correlation = _as_mapping(
        execution.get("correlation"),
        label="attempt execution correlation",
    )
    for field_name in ("run_id", "attempt_id"):
        expected = attempt.get(field_name)
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"executed parity attempt requires a non-empty {field_name}")
        if correlation.get(field_name) != expected:
            raise ValueError(
                f"attempt execution correlation {field_name} mismatches its persisted attempt"
            )

    raw_events = _plain(execution.get("events"))
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("executed parity attempt requires its actual event sequence")
    events = [_as_mapping(event, label="attempt execution event") for event in raw_events]
    _validate_judge_scores(events)
    event_ids: dict[str, Mapping[str, object]] = {}
    operation_ids: set[str] = set()
    for event in events:
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("attempt execution events require generated event IDs")
        if event_id in event_ids:
            raise ValueError("attempt execution event IDs must be unique")
        event_ids[event_id] = event
        if _plain(event.get("correlation")) != correlation:
            raise ValueError("attempt execution event correlation mismatches its owner")
        operation_id = event.get("operation_id")
        if isinstance(operation_id, str) and operation_id:
            operation_ids.add(operation_id)

    event_types = [str(event.get("event_type") or "") for event in events]
    if "run.started" not in event_types:
        raise ValueError("attempt execution lifecycle is missing run.started")
    status = attempt.get("execution_status", attempt.get("status"))
    expected_terminal = "run.failed" if status == "failed" else "run.completed"
    if status not in {"failed", "completed"} or expected_terminal not in event_types:
        raise ValueError("attempt execution lifecycle disagrees with persisted status")
    unexpected_terminal = (
        "run.completed" if expected_terminal == "run.failed" else "run.failed"
    )
    if unexpected_terminal in event_types:
        raise ValueError("attempt execution has conflicting terminal lifecycle events")

    calls = _plain(execution.get("calls"))
    if not isinstance(calls, list):
        raise TypeError("attempt execution calls must be an explicit sequence")
    call_mappings = [_as_mapping(call, label="attempt execution call") for call in calls]
    for call in call_mappings:
        operation_id = call.get("operation_id")
        if not isinstance(operation_id, str) or operation_id not in operation_ids:
            raise ValueError("attempt execution call has no owned operation event")
        call_correlation = call.get("correlation")
        if call_correlation is not None and _plain(call_correlation) != correlation:
            raise ValueError("attempt execution call correlation mismatches its owner")

    semantic_events = _plain(execution.get("semantic_events"))
    if not isinstance(semantic_events, list):
        raise TypeError("attempt semantic events must be an explicit sequence")
    semantic_mappings = [
        _as_mapping(event, label="attempt semantic event") for event in semantic_events
    ]
    for event in semantic_mappings:
        event_id = event.get("event_id")
        source = event_ids.get(event_id) if isinstance(event_id, str) else None
        if source is None:
            raise ValueError("attempt semantic event does not belong to its execution")
        for field_name in ("event_type", "operation_id", "round", "branch_id", "caused_by"):
            if event.get(field_name) != source.get(field_name):
                raise ValueError(
                    f"attempt semantic event {field_name} disagrees with its source event"
                )
        if _plain(event.get("correlation")) != correlation:
            raise ValueError("attempt semantic event correlation mismatches its owner")
        payload = event.get("payload")
        source_payload = source.get("payload")
        if not isinstance(payload, Mapping) or not isinstance(source_payload, Mapping):
            raise TypeError("attempt semantic event payloads must be mappings")
        if any(_plain(source_payload.get(key)) != _plain(value) for key, value in payload.items()):
            raise ValueError("attempt semantic event payload is not derived from its source event")

    histories = _plain(execution.get("histories"))
    if not isinstance(histories, list):
        raise TypeError("attempt execution histories must be an explicit sequence")
    history_mappings = [
        _as_mapping(history, label="attempt execution history") for history in histories
    ]
    message_events: dict[str, Mapping[str, object]] = {}
    for event in events:
        if event.get("event_type") != "message.appended":
            continue
        payload = event.get("payload")
        message_id = payload.get("message_id") if isinstance(payload, Mapping) else None
        if isinstance(message_id, str) and message_id:
            message_events[message_id] = event
    for history in history_mappings:
        messages = _plain(history.get("messages"))
        message_ids = _plain(history.get("message_ids"))
        if not isinstance(messages, list) or not isinstance(message_ids, list):
            raise TypeError("attempt history snapshots require messages and message IDs")
        if [
            message.get("message_id")
            for message in messages
            if isinstance(message, Mapping)
        ] != message_ids:
            raise ValueError("attempt history snapshot message order is not owned")
        for raw_message in messages:
            message = _as_mapping(raw_message, label="attempt history message")
            source_event = message_events.get(str(message.get("message_id") or ""))
            source_payload = source_event.get("payload") if source_event is not None else None
            if not isinstance(source_payload, Mapping):
                raise ValueError("attempt history message has no appended source event")
            comparable_fields = {
                "history_id",
                "message_id",
                "role",
                "content",
                "source_kind",
                "original_message_id",
                "visibility_roles",
            }
            if any(
                _plain(message.get(field_name)) != _plain(source_payload.get(field_name))
                for field_name in comparable_fields
            ):
                raise ValueError("attempt history message disagrees with its appended event")
            if message.get("source_event_id") != source_event.get("caused_by"):
                raise ValueError("attempt history source provenance is not owned")

    outcome = _as_mapping(execution.get("outcome"), label="attempt execution outcome")
    if outcome.get("execution_status") != status:
        raise ValueError("attempt execution outcome status disagrees with persisted status")
    if outcome.get("result_disposition") != attempt.get("result_disposition"):
        raise ValueError("attempt execution disposition disagrees with persisted result")
    attempt_score = attempt.get("score")
    outcome_score = outcome.get("official_score")
    if attempt_score != outcome_score:
        raise ValueError("attempt execution score disagrees with persisted result")
    if status != "completed" and (attempt_score is not None or outcome_score is not None):
        raise ValueError("failed attempt execution cannot carry a numeric score")

    resolver = _ReferenceResolver.from_events(events)
    final_ref = outcome.get("final_response_ref")
    final_resolution = resolver.resolve(
        "target",
        final_ref,
        path="experiment.selected_attempts.execution.final_response",
    )
    official_ref = outcome.get("official_judge_ref")
    official_resolution = (
        resolver.resolve(
            "judge",
            official_ref,
            path="experiment.selected_attempts.execution.official_judge",
        )
        if official_ref is not None
        else None
    )
    failed_target_judge = any(
        event.get("event_type") == "judge.failed"
        and isinstance(event.get("payload"), Mapping)
        and event["payload"].get("definition_id") == "target_judge"  # type: ignore[index]
        and event["payload"].get("score") is None  # type: ignore[index]
        for event in events
    )
    unscored_inconclusive = (
        failed_target_judge
        and (
            official_ref is None
            or (
                isinstance(official_resolution, Mapping)
                and official_resolution.get("status") == "failed"
                and official_resolution.get("score") is None
            )
        )
        and outcome_score is None
        and outcome.get("result_disposition") == "inconclusive"
        and outcome.get("attack_outcome") == "inconclusive"
    )
    raw_resolved_spec = _plain(execution.get("resolved_spec"))
    resolved_parameters = (
        raw_resolved_spec.get("parameters")
        if isinstance(raw_resolved_spec, Mapping)
        else None
    )
    explicitly_unjudged = (
        isinstance(resolved_parameters, Mapping)
        and resolved_parameters.get("judge_per_round") is False
        and not any(
            str(event.get("event_type") or "").startswith("judge.")
            for event in events
        )
        and official_ref is None
        and outcome_score is None
        and outcome.get("result_disposition") == "inconclusive"
        and outcome.get("attack_outcome") == "inconclusive"
    )
    if status == "completed" and not (
        unscored_inconclusive or explicitly_unjudged
    ):
        if official_resolution is None:
            raise ValueError("completed attempt has no owned official target_judge")
        if (
            not isinstance(official_resolution, Mapping)
            or official_resolution.get("status") != "succeeded"
            or not _is_number(official_resolution.get("score"))
            or not _is_number(outcome_score)
            or float(official_resolution["score"]) != float(outcome_score)
        ):
            raise ValueError("completed attempt has no succeeded numeric target_judge result")
    if resolver.issues:
        raise ValueError("attempt execution contains unresolved provenance")

    normalized_spec = _normalize(execution.get("resolved_spec"))
    normalized_projection = _normalize(
        execution.get("conversation_projection"),
        resolver=resolver,
        path="experiment.selected_attempts.execution.conversation_projection",
    )
    return {
        "lifecycle": [
            event_type
            for event_type in event_types
            if event_type in {"run.started", "run.completed", "run.failed"}
        ],
        "calls": _canonical_calls(call_mappings, resolver=resolver),
        "events": _normalize(
            semantic_mappings,
            resolver=resolver,
            path="experiment.selected_attempts.execution.events",
        ),
        "histories": _normalize(
            history_mappings,
            resolver=resolver,
            path="experiment.selected_attempts.execution.histories",
        ),
        "conversation_projection": normalized_projection,
        "resolved_spec": normalized_spec,
        "outcome": {
            "execution_status": outcome.get("execution_status"),
            "result_disposition": outcome.get("result_disposition"),
            "attack_outcome": outcome.get("attack_outcome"),
            "official_score": outcome_score,
            "final_response": final_resolution,
            "official_judge": official_resolution,
            "stopping_reason": outcome.get("stopping_reason"),
        },
    }


def _experiment_plan_evidence(run: Mapping[str, object]) -> object:
    plan = run.get("experiment_plan", run.get("plan"))
    return _normalize(plan) if plan is not None else None


def _deterministic_metrics(semantic: Mapping[str, object]) -> dict[str, object]:
    calls = semantic.get("calls")
    role_counts = calls.get("role_counts", {}) if isinstance(calls, Mapping) else {}
    events = semantic.get("events")
    event_count = len(_flatten_events(events)) if isinstance(events, Mapping) else 0
    selected = semantic.get("selected_attempts", [])
    return {
        "role_call_counts": role_counts,
        "event_count": event_count,
        "cell_count": len(selected) if isinstance(selected, list) else 0,
    }
