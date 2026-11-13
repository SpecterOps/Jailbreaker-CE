from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.techniques.contracts.events import ExecutionEvent
from app.techniques.contracts.definition import (
    ImplementationKind,
    ResultPolicy,
    TechniqueCapability,
    TechniqueDefinition,
    TechniqueIdentity,
    TechniqueImplementation,
)
from app.techniques.contracts.execution import (
    AttackOutcome,
    CorrelationIds,
    ExecutionStatus,
    ModelTransport,
    ResultDisposition,
    TechniqueOutcome,
)
from app.techniques.core.catalog import RegisteredTechnique, TechniqueCatalog, TechniqueSelector
from app.techniques.core.models import ProviderRequestContext
from app.techniques.core.router import ExecutionRouter
from devtools.technique_parity import (
    ParityArmRequest,
    ParityComparator,
    ParityDifference,
    ParityPolicy,
    ParityReport,
    ParityRuntime,
    ParityScenario,
    ScriptedResponse,
    ScriptedRoleClient,
    compare_live_samples,
    parity_route_identity,
)
from devtools.technique_parity.__main__ import main as parity_main


CORRELATION = CorrelationIds(
    experiment_id="experiment-generated",
    cell_id="cell-generated",
    run_id="run-generated",
    attempt_id="attempt-generated",
)


def parity_runtime(
    *,
    runner: object | None = None,
    modular_contract_version: str = "1",
    legacy_threshold: float | None = 7,
    modular_threshold: float | None = 7,
    modular_version: str = "1.0.0",
) -> ParityRuntime:
    catalog = TechniqueCatalog()
    for implementation_id, kind in (
        ("legacy", ImplementationKind.LEGACY),
        ("modular", ImplementationKind.PYTHON),
    ):
        technique_version = (
            "1.0.0" if implementation_id == "legacy" else modular_version
        )
        definition = TechniqueDefinition(
            identity=TechniqueIdentity(
                id="direct_prompt_injection",
                version=technique_version,
                name="Direct Prompt Injection",
                family="test",
            ),
            summary="Parity CLI fixture",
            capabilities=frozenset({TechniqueCapability.SINGLE_SHOT}),
            result_policy=ResultPolicy(
                success_threshold=(
                    legacy_threshold
                    if implementation_id == "legacy"
                    else modular_threshold
                )
            ),
        )
        catalog.register(
            RegisteredTechnique(
                definition=definition,
                implementation=TechniqueImplementation(
                    id=implementation_id,
                    technique_id="direct_prompt_injection",
                    technique_version=technique_version,
                    kind=kind,
                    entrypoint=f"tests.{implementation_id}",
                    contract_version=(
                        modular_contract_version if implementation_id == "modular" else "1"
                    ),
                ),
                manifest_hash="a" * 64,
                asset_hashes={"prompt.j2": "b" * 64},
            )
        )
    return ParityRuntime(router=ExecutionRouter(catalog), arm_runner=runner)  # type: ignore[arg-type]


def strict_policy() -> ParityPolicy:
    return ParityPolicy.strict()


def live_policy(*, max_mean_score_delta: float) -> ParityPolicy:
    return ParityPolicy(
        policy_id="live-test",
        version="1",
        max_mean_score_delta=max_mean_score_delta,
        max_success_rate_delta=1.0,
        success_threshold=70.0,
        dataset_id="dataset-a",
        repeat_count=1,
        dataset_sample_count=1,
        profile_ids={"target": "target-profile", "target_judge": "judge-profile"},
    )


def test_parity_policy_deeply_pins_nested_metadata() -> None:
    metadata = {"sampling": {"temperature": 0}}
    policy = ParityPolicy(metadata=metadata)
    fingerprint = policy.fingerprint

    metadata["sampling"]["temperature"] = 1

    assert policy.metadata["sampling"] == {"temperature": 0}
    assert policy.fingerprint == fingerprint


def test_parity_serialization_redacts_credentials_in_every_string_field() -> None:
    credentials = {
        "policy": "sk-policycredential123",
        "profile": "sk-profilecredential123",
        "difference": "sk-differencecredential123",
        "message": "sk-messagecredential123",
    }
    policy = ParityPolicy(
        policy_id=credentials["policy"],
        metadata={"nested": {"profile_id": credentials["profile"]}},
    )
    report = ParityReport(
        mode="deterministic",
        passed=False,
        policy=policy,
        differences=(
            ParityDifference(
                path=f"identity.{credentials['difference']}",
                legacy={"profile_id": credentials["profile"]},
                modular=None,
                message=f"difference contains {credentials['message']}",
            ),
        ),
        metadata={"nested": {"profile_id": credentials["profile"]}},
    )

    serialized = json.dumps(report.to_dict())

    assert all(value not in serialized for value in credentials.values())
    assert "[REDACTED]" in serialized


def test_parity_policy_fingerprint_cannot_be_supplied_by_caller() -> None:
    with pytest.raises(TypeError):
        ParityPolicy(source_hash="0" * 64)  # type: ignore[call-arg]


def test_file_policy_fingerprint_binds_source_bytes_and_effective_policy(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_bytes(b"policy_id: pinned\nversion: '1'\n")
    second.write_bytes(b"policy_id: pinned\nversion: '1'\n\n")

    first_policy = ParityPolicy.from_file(first)
    second_policy = ParityPolicy.from_file(second)
    programmatic = ParityPolicy(policy_id="pinned", version="1")

    assert first_policy.fingerprint != hashlib.sha256(first.read_bytes()).hexdigest()
    assert first_policy.fingerprint != second_policy.fingerprint
    assert first_policy.fingerprint != programmatic.fingerprint


def event(
    event_id: str,
    sequence: int,
    event_type: str,
    *,
    implementation_id: str,
    round_number: int | None = None,
    branch_id: str | None = None,
    timestamp_delta: int = 0,
    payload: dict[str, object] | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        event_id=event_id,
        correlation=CORRELATION,
        sequence=sequence,
        event_type=event_type,
        schema_version=1,
        timestamp=datetime(2026, 8, 18, tzinfo=UTC) + timedelta(seconds=timestamp_delta),
        technique_id="direct_prompt_injection",
        technique_version="1.0.0",
        implementation_id=implementation_id,
        round=round_number,
        branch_id=branch_id,
        payload=payload or {},
    )


def parity_run(
    *,
    event_ids: tuple[str, ...] = ("target-event",),
    target_scores: tuple[int, ...] = (2, 8),
    implementation_id: str = "legacy",
    timestamp_delta: int = 0,
    response: str = "real response",
) -> dict[str, object]:
    target_event_id = event_ids[0]
    events: list[ExecutionEvent] = [
        event(
            target_event_id,
            1,
            "target.completed",
            implementation_id=implementation_id,
            timestamp_delta=timestamp_delta,
            payload={
                "response": response,
                "response_kind": "real",
                "model_call_id": f"{implementation_id}-generated-call",
                "latency_ms": 15 + timestamp_delta,
            },
        )
    ]
    for round_number, score in enumerate(target_scores, start=1):
        events.append(
            event(
                f"{implementation_id}-judge-{round_number}",
                round_number + 1,
                "judge.completed",
                implementation_id=implementation_id,
                round_number=round_number,
                timestamp_delta=timestamp_delta + round_number,
                payload={
                    "definition_id": "target_judge",
                    "status": "succeeded",
                    "score": score,
                    "result_id": f"{implementation_id}-result-{round_number}",
                    "provider_latency_ms": 25 + timestamp_delta,
                },
            )
        )
    return {
        "contract": {
            "definition": {
                "contract_version": "1",
                "capabilities": ["single_shot"],
                "parameters_schema": {"type": "object"},
            },
            "resolved_spec": {
                "spec_hash": "shared-spec-hash",
                "manifest_hash": "shared-manifest-hash",
                "asset_hashes": {"prompt.j2": "shared-asset-hash"},
            },
        },
        "projections": {
            "conversation": [{"role": "user", "content_hash": "projected-prompt"}],
            "rounds": [{"round": 1, "status": "completed"}],
            "artifacts": [{"kind": "response", "content_hash": "response-hash"}],
            "metrics": {"official_score": target_scores[-1], "target_calls": 1},
        },
        "calls": (
            {
                "role": "target",
                "messages": ({"role": "user", "content": "rendered\r\nprompt"},),
                "options": {"temperature": 0, "generated_call_id": f"call-{implementation_id}"},
                "operation_id": "target-round",
                "call_index": 1,
            },
        ),
        "events": tuple(events),
        "outcome": TechniqueOutcome(
            execution_status=ExecutionStatus.COMPLETED,
            result_disposition=ResultDisposition.TESTED,
            attack_outcome=AttackOutcome.SUCCEEDED,
            final_response_ref=target_event_id,
            official_judge_ref=f"{implementation_id}-result-{len(target_scores)}",
            stopping_reason="threshold_reached",
        ),
        "official_score": target_scores[-1],
    }


def relational_parity_run(*, implementation_id: str, prefix: str) -> dict[str, object]:
    run = parity_run(
        event_ids=(f"{prefix}-target",),
        implementation_id=implementation_id,
    )
    run["events"] = (
        *run["events"],
        {
            "event_id": f"{prefix}-message-event-a",
            "event_type": "message.appended",
            "branch_id": "target",
            "payload": {
                "history_id": "target",
                "message_id": f"{prefix}-message-a",
                "content": "first semantic message",
                "role": "user",
            },
        },
        {
            "event_id": f"{prefix}-message-event-b",
            "event_type": "message.appended",
            "branch_id": "target",
            "caused_by": f"{prefix}-message-event-a",
            "payload": {
                "history_id": "target",
                "message_id": f"{prefix}-message-b",
                "content": "second semantic message",
                "role": "assistant",
                "source_event_id": f"{prefix}-message-event-a",
            },
        },
        {
            "event_id": f"{prefix}-rollback",
            "event_type": "history.rolled_back",
            "branch_id": "target",
            "payload": {
                "history_id": "target",
                "active_message_ids": (f"{prefix}-message-a",),
                "rolled_back_message_ids": (f"{prefix}-message-b",),
            },
        },
        {
            "event_id": f"{prefix}-artifact-a-event",
            "event_type": "artifact.created",
            "payload": {
                "artifact_id": f"{prefix}-artifact-a",
                "media_type": "application/json",
                "content_hash": "artifact-content-a",
            },
        },
        {
            "event_id": f"{prefix}-artifact-b-event",
            "event_type": "artifact.created",
            "payload": {
                "artifact_id": f"{prefix}-artifact-b",
                "media_type": "application/json",
                "content_hash": "artifact-content-b",
            },
        },
        {
            "event_id": f"{prefix}-transform",
            "event_type": "response.transformed",
            "payload": {
                "source_artifact_id": f"{prefix}-artifact-a",
                "transform_provenance": {"converter_id": "poison-response"},
            },
        },
    )
    return run


def sample(*, score: int, attempts: int) -> dict[str, object]:
    return explicit_live_sample(score=score, attempts=attempts)


def explicit_live_sample(
    *,
    score: int,
    attempts: int = 1,
    prompt_id: str = "prompt-a",
    repeat_index: int = 1,
    dataset_id: str = "dataset-a",
    profile_ids: dict[str, str] | None = None,
    prefix: str = "sample",
) -> dict[str, object]:
    pinned_profiles = profile_ids or {
        "target": "target-profile",
        "target_judge": "judge-profile",
    }
    attempt_records: list[dict[str, object]] = [
        {
            "attempt_id": f"{prefix}-attempt-{index}",
            "attempt_number": index,
            "status": "failed",
            "retry_kind": "restart",
        }
        for index in range(1, attempts)
    ]
    selected_id = f"{prefix}-attempt-{attempts}"
    attempt_records.append(
        {
            "attempt_id": selected_id,
            "attempt_number": attempts,
            "status": "completed",
            "score": score,
            "profile_ids": {
                "target": pinned_profiles["target"],
                "target_judge": pinned_profiles["target_judge"],
            },
            "calls": (
                {"role": "target", "profile_id": pinned_profiles["target"]},
                {
                    "role": "target_judge",
                    "profile_id": pinned_profiles["target_judge"],
                },
            ),
            "official_judge": {
                "definition_id": "target_judge",
                "status": "succeeded",
                "score": score,
                "profile_id": pinned_profiles["target_judge"],
            },
        }
    )
    return {
        "dataset_id": dataset_id,
        "profile_ids": pinned_profiles,
        "prompt_id": prompt_id,
        "repeat_index": repeat_index,
        "selected_attempt_id": selected_id,
        "attempts": attempt_records,
    }


def json_parity_run(*, implementation_id: str, prefix: str) -> dict[str, object]:
    return {
        "projections": {
            "conversation": [{"role": "user", "content_hash": "projected-prompt"}],
            "rounds": [{"round": 1, "status": "completed"}],
            "artifacts": [{"kind": "response", "content_hash": "response-hash"}],
            "metrics": {"official_score": 8, "target_calls": 1},
        },
        "events": [
            {
                "event_id": f"{prefix}-target",
                "event_type": "target.completed",
                "implementation_id": implementation_id,
                "payload": {"response": "same response", "response_kind": "real"},
            },
            {
                "event_id": f"{prefix}-judge",
                "event_type": "judge.completed",
                "implementation_id": implementation_id,
                "round": 1,
                "payload": {
                    "definition_id": "target_judge",
                    "status": "succeeded",
                    "score": 8,
                    "result_id": f"{prefix}-result",
                },
            },
        ],
        "outcome": {
            "execution_status": "completed",
            "result_disposition": "tested",
            "attack_outcome": "succeeded",
            "final_response_ref": f"{prefix}-target",
            "official_judge_ref": f"{prefix}-result",
            "stopping_reason": "threshold_reached",
        },
        "official_score": 8,
    }


def technique_configuration(
    config_id: str,
    *,
    technique_key: str = "direct_prompt_injection",
    spec_hash: str = "config-spec-a",
) -> dict[str, object]:
    resolved_spec = {
        "technique_id": technique_key,
        "technique_version": "1.0.0",
        "implementation_id": "parity-implementation",
        "implementation_kind": "python",
        "contract_version": "1",
        "manifest_hash": "manifest-a",
        "asset_hashes": {"prompt.j2": "asset-a"},
        "preset": None,
        "parameters": {"temperature": 0},
        "model_roles": {},
        "judge_policies": {},
        "effective_limits": {},
        "runtime_config": {"configuration_spec_hash": spec_hash},
    }
    return {
        "technique_config_id": config_id,
        "technique_key": technique_key,
        "technique_id": technique_key,
        "technique_version": "1.0.0",
        "implementation_id": "parity-implementation",
        "implementation_kind": "python",
        "contract_version": "1",
        "manifest_hash": "manifest-a",
        "parameters": {"temperature": 0},
        "spec_hash": spec_hash,
        "resolved_spec": resolved_spec,
    }


def deterministic_cell(
    config_id: str,
    *,
    prefix: str,
    technique_key: str = "direct_prompt_injection",
) -> dict[str, object]:
    return {
        "technique_config_id": config_id,
        "technique_key": technique_key,
        "prompt_id": "prompt-a",
        "repeat_index": 1,
        "selected_attempt_id": f"{prefix}-selected",
        "attempts": (
            {
                "attempt_id": f"{prefix}-selected",
                "attempt_number": 1,
                "status": "completed",
            },
        ),
    }


@pytest.mark.parametrize("location", ["intent", "target_response", "planner_request", "model_output", "history"])
def test_parity_preserves_literal_judge_score_text(location: str) -> None:
    runs = [parity_run(), parity_run(implementation_id="modular")]
    for run, text in zip(
        runs, ["Judge score: 0/10", "Judge score: not scheduled/10"], strict=True,
    ):
        if location == "intent":
            run["contract"]["resolved_spec"]["inputs"] = {"intent": text}
        elif location == "target_response":
            run["events"] = (
                replace(run["events"][0], payload={**run["events"][0].payload, "response": text}),
                *run["events"][1:],
            )
        elif location == "planner_request":
            run["calls"] = ({
                "role": "planner", "messages": [{"role": "user", "content": text}],
            },)
        else:
            run["events"] = (*run["events"], {
                "event_id": "literal-text",
                "event_type": "model.completed" if location == "model_output" else "message.appended",
                "payload": (
                    {"role": "planner", "response": text}
                    if location == "model_output"
                    else {"history_id": "audit", "role": "assistant", "content": text}
                ),
            })

    report = ParityComparator().compare_deterministic(*runs, policy=strict_policy())

    assert not report.passed
    assert any(difference.path != "validation" for difference in report.differences)


def test_parity_ignores_ids_and_timestamps_but_compares_judge_schedule() -> None:
    legacy = parity_run(event_ids=("legacy-1",), target_scores=(2, 8), implementation_id="legacy")
    modular = parity_run(
        event_ids=("modular-9",),
        target_scores=(2, 8),
        implementation_id="modular",
        timestamp_delta=90,
    )

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert report.passed


def test_parity_contract_and_projection_evidence_are_hard_gates() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    shared_contract = {
        "definition": {
            "contract_version": "1",
            "capabilities": ["single_turn"],
            "parameters_schema": {"type": "object"},
        },
        "resolved_spec": {
            "spec_hash": "spec-pinned-a",
            "manifest_hash": "manifest-pinned-a",
            "asset_hashes": {"prompt.j2": "asset-pinned-a"},
            "technique_config_id": "config-pinned-a",
        },
    }
    shared_projections = {
        "conversation": [{"role": "user", "content": "projected prompt"}],
        "rounds": [{"round": 1, "status": "completed"}],
        "artifacts": [{"kind": "response", "content_hash": "response-hash"}],
        "metrics": {"official_score": 8, "target_calls": 1},
    }
    legacy["contract"] = deepcopy(shared_contract)
    modular["contract"] = deepcopy(shared_contract)
    legacy["projections"] = deepcopy(shared_projections)
    modular["projections"] = deepcopy(shared_projections)

    assert ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy()).passed

    modular["contract"]["definition"]["capabilities"] = ["multi_turn"]
    modular["projections"]["artifacts"][0]["content_hash"] = "different-response-hash"
    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    paths = {item.path for item in report.differences}
    assert "contract.definition" in paths
    assert "projections.artifacts" in paths


def test_routed_contract_preserves_declared_judge_schedule_without_opt_in() -> None:
    from devtools.technique_parity.compare import _routed_definition_contract

    run_end = {
        "judges": [{"id": "target_judge", "schedule": "run_end"}],
    }
    every_round = {
        "judges": [{"id": "target_judge", "schedule": "every_round"}],
    }

    assert _routed_definition_contract(run_end) != _routed_definition_contract(
        every_round
    )
    assert _routed_definition_contract(
        run_end,
        normalize_effective_judge_schedule=True,
    ) == _routed_definition_contract(
        every_round,
        normalize_effective_judge_schedule=True,
    )


def test_parity_rejects_missing_contract_and_projection_evidence_on_both_arms() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    legacy.pop("contract")
    modular.pop("contract")
    legacy.pop("projections")
    modular.pop("projections")

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert not report.passed
    paths = {item.path for item in report.differences}
    assert "contract.definition" in paths
    assert "contract.resolved_spec" in paths
    assert "projections.conversation" in paths
    assert "projections.rounds" in paths
    assert "projections.artifacts" in paths
    assert "projections.metrics" in paths


def test_parity_rejects_contract_or_projection_evidence_missing_from_one_arm() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    modular["contract"].pop("resolved_spec")
    modular["projections"].pop("metrics")

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert not report.passed
    paths = {item.path for item in report.differences}
    assert "contract.resolved_spec" in paths
    assert "projections.metrics" in paths


@pytest.mark.parametrize(
    ("field", "legacy_value", "modular_value"),
    (
        ("plan_hash", "plan-a", "plan-b"),
        ("spec_hash", "spec-a", "spec-b"),
        ("manifest_hash", "manifest-a", "manifest-b"),
        ("asset_hashes", {"prompt.j2": "asset-a"}, {"prompt.j2": "asset-b"}),
        ("technique_config_id", "config-a", "config-b"),
    ),
)
def test_parity_does_not_normalize_pinned_semantic_identity(
    field: str,
    legacy_value: object,
    modular_value: object,
) -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    legacy["contract"] = {"resolved_spec": {field: legacy_value}}
    modular["contract"] = {"resolved_spec": {field: modular_value}}

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert not report.passed
    assert any(item.path == "contract.resolved_spec" for item in report.differences)


def test_parity_resolves_history_and_transform_references_by_semantic_identity() -> None:
    legacy = relational_parity_run(implementation_id="legacy", prefix="legacy")
    modular = relational_parity_run(implementation_id="modular", prefix="modular")

    assert ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy()).passed

    modular_history = relational_parity_run(implementation_id="modular", prefix="modular")
    rollback = modular_history["events"][5]["payload"]
    rollback["active_message_ids"], rollback["rolled_back_message_ids"] = (
        rollback["rolled_back_message_ids"],
        rollback["active_message_ids"],
    )
    history_report = ParityComparator().compare_deterministic(
        legacy,
        modular_history,
        policy=strict_policy(),
    )
    assert any(item.path == "histories.snapshots" for item in history_report.differences)

    modular_transform = relational_parity_run(implementation_id="modular", prefix="modular")
    modular_transform["events"][-1]["payload"]["source_artifact_id"] = "modular-artifact-b"
    transform_report = ParityComparator().compare_deterministic(
        legacy,
        modular_transform,
        policy=strict_policy(),
    )
    assert any(item.path == "transforms.provenance" for item in transform_report.differences)


def test_parity_records_unresolved_final_and_official_provenance_as_failures() -> None:
    legacy = parity_run()
    unresolved_final = parity_run(implementation_id="modular")
    unresolved_final["outcome"] = TechniqueOutcome(
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        attack_outcome=AttackOutcome.SUCCEEDED,
        final_response_ref="missing-target-response",
        official_judge_ref="modular-result-2",
        stopping_reason="threshold_reached",
    )
    final_report = ParityComparator().compare_deterministic(
        legacy,
        unresolved_final,
        policy=strict_policy(),
    )
    assert not final_report.passed
    assert any(item.path == "final_response.provenance" for item in final_report.differences)

    missing_official = parity_run(implementation_id="modular")
    missing_official["outcome"] = TechniqueOutcome(
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        attack_outcome=AttackOutcome.SUCCEEDED,
        final_response_ref="target-event",
        official_judge_ref=None,
        stopping_reason="threshold_reached",
    )
    official_report = ParityComparator().compare_deterministic(
        legacy,
        missing_official,
        policy=strict_policy(),
    )
    assert not official_report.passed
    assert any(item.path == "outcome.official_judge.provenance" for item in official_report.differences)


def test_parity_fails_when_modular_skips_target_judge_round() -> None:
    legacy = parity_run(target_scores=(1, 4, 8))
    modular = parity_run(target_scores=(1, 8), implementation_id="modular")

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert not report.passed
    assert any(item.path == "judges.target_judge.schedule" for item in report.differences)


def test_parity_normalizes_prompt_newlines_but_keeps_role_order() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    modular["calls"] = (
        {
            "role": "target",
            "messages": ({"role": "user", "content": "rendered\nprompt"},),
            "options": {"temperature": 0},
            "operation_id": "target-round",
            "call_index": 1,
        },
    )
    assert ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy()).passed

    modular["calls"] = (
        {
            "role": "attacker",
            "messages": ({"role": "user", "content": "rendered\nprompt"},),
            "options": {"temperature": 0},
            "operation_id": "target-round",
            "call_index": 1,
        },
    )
    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())
    assert any(item.path == "calls.role_order" for item in report.differences)


def test_parallel_branch_events_compare_by_identity_not_completion_order() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    branch_a = event(
        "legacy-a",
        10,
        "branch.completed",
        implementation_id="legacy",
        branch_id="branch-a",
        payload={"candidate_id": "candidate-a", "score": 4},
    )
    branch_b = event(
        "legacy-b",
        11,
        "branch.completed",
        implementation_id="legacy",
        branch_id="branch-b",
        payload={"candidate_id": "candidate-b", "score": 7},
    )
    modular_b = event(
        "modular-b",
        10,
        "branch.completed",
        implementation_id="modular",
        branch_id="branch-b",
        payload={"candidate_id": "candidate-b", "score": 7},
    )
    modular_a = event(
        "modular-a",
        11,
        "branch.completed",
        implementation_id="modular",
        branch_id="branch-a",
        payload={"candidate_id": "candidate-a", "score": 4},
    )
    legacy["events"] = (*legacy["events"], branch_a, branch_b)
    modular["events"] = (*modular["events"], modular_b, modular_a)

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert report.passed


def test_explicit_parallel_structures_compare_by_stable_identity_and_content() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    legacy["branches"] = (
        {"branch_id": "branch-a", "status": "completed"},
        {"branch_id": "branch-b", "status": "pruned"},
    )
    modular["branches"] = tuple(reversed(legacy["branches"]))
    legacy["candidates"] = (
        {"candidate_id": "candidate-a", "score": 4},
        {"candidate_id": "candidate-b", "score": 8},
    )
    modular["candidates"] = tuple(reversed(legacy["candidates"]))

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert report.passed


def test_parity_normalizes_generated_history_artifact_and_causal_references() -> None:
    legacy = parity_run(event_ids=("legacy-1",))
    modular = parity_run(event_ids=("modular-9",), implementation_id="modular")
    legacy["events"] = (
        *legacy["events"],
        {
            "event_id": "legacy-active-event",
            "event_type": "message.appended",
            "branch_id": "target",
            "payload": {
                "history_id": "target",
                "message_id": "legacy-active",
                "content": "active content",
                "role": "user",
            },
        },
        {
            "event_id": "legacy-message-event",
            "event_type": "message.appended",
            "branch_id": "target",
            "caused_by": "legacy-1",
            "payload": {
                "history_id": "target",
                "message_id": "legacy-message",
                "content": "same content",
                "role": "assistant",
                "source_event_id": "legacy-1",
            },
        },
        {
            "event_id": "legacy-rollback",
            "event_type": "history.rolled_back",
            "branch_id": "target",
            "payload": {
                "history_id": "target",
                "snapshot_id": "legacy-snapshot",
                "active_message_ids": ("legacy-active",),
                "rolled_back_message_ids": ("legacy-message",),
            },
        },
        {
            "event_id": "legacy-artifact",
            "event_type": "artifact.created",
            "payload": {
                "artifact_id": "legacy-artifact-id",
                "media_type": "application/json",
                "content_hash": "stable-content-hash",
                "size_bytes": 12,
            },
        },
    )
    modular["events"] = (
        *modular["events"],
        {
            "event_id": "modular-active-event",
            "event_type": "message.appended",
            "branch_id": "target",
            "payload": {
                "history_id": "target",
                "message_id": "modular-active",
                "content": "active content",
                "role": "user",
            },
        },
        {
            "event_id": "modular-message-event",
            "event_type": "message.appended",
            "branch_id": "target",
            "caused_by": "modular-9",
            "payload": {
                "history_id": "target",
                "message_id": "modular-message",
                "content": "same content",
                "role": "assistant",
                "source_event_id": "modular-9",
            },
        },
        {
            "event_id": "modular-rollback",
            "event_type": "history.rolled_back",
            "branch_id": "target",
            "payload": {
                "history_id": "target",
                "snapshot_id": "modular-snapshot",
                "active_message_ids": ("modular-active",),
                "rolled_back_message_ids": ("modular-message",),
            },
        },
        {
            "event_id": "modular-artifact",
            "event_type": "artifact.created",
            "payload": {
                "artifact_id": "modular-artifact-id",
                "media_type": "application/json",
                "content_hash": "stable-content-hash",
                "size_bytes": 12,
            },
        },
    )

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert report.passed

    modular["events"][5]["payload"]["rolled_back_message_ids"] = (
        "modular-message",
        "modular-extra",
    )
    changed = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())
    assert any(item.path == "histories.snapshots" for item in changed.differences)


def test_parity_keeps_target_judge_lifecycle_order_within_one_branch() -> None:
    scheduled = {
        "event_type": "judge.scheduled",
        "round": 1,
        "branch_id": "branch-a",
        "payload": {"definition_id": "target_judge", "status": "scheduled"},
    }
    completed = {
        "event_type": "judge.completed",
        "round": 1,
        "branch_id": "branch-a",
        "payload": {"definition_id": "target_judge", "status": "succeeded", "score": 8},
    }

    report = ParityComparator().compare_deterministic(
        {"events": (scheduled, completed)},
        {"events": (completed, scheduled)},
        policy=strict_policy(),
    )

    assert not report.passed
    assert any(item.path == "judges.target_judge.schedule" for item in report.differences)


def test_parity_compares_outcome_and_final_real_response() -> None:
    legacy = parity_run(response="selected legacy response")
    modular = parity_run(implementation_id="modular", response="different modular response")
    modular["outcome"] = TechniqueOutcome(
        execution_status=ExecutionStatus.FAILED,
        result_disposition=ResultDisposition.INCONCLUSIVE,
        attack_outcome=AttackOutcome.INCONCLUSIVE,
        stopping_reason="provider_failure",
    )

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    paths = {item.path for item in report.differences}
    assert "final_response.selection" in paths
    assert "outcome.execution_status" in paths
    assert "outcome.result_disposition" in paths
    assert "outcome.attack_outcome" in paths
    assert "outcome.stopping_reason" in paths


def test_parity_compares_selected_attempt_semantics_not_generated_ids() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    legacy["technique_configurations"] = (technique_configuration("legacy-config"),)
    modular["technique_configurations"] = (technique_configuration("modular-config"),)
    legacy["cells"] = (
        {
            "cell_id": "legacy-cell",
            "technique_config_id": "legacy-config",
            "technique_key": "direct_prompt_injection",
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "legacy-second",
            "attempts": (
                {"attempt_id": "legacy-first", "attempt_number": 1, "status": "failed"},
                {"attempt_id": "legacy-second", "attempt_number": 2, "status": "completed", "score": 8},
            ),
        },
    )
    modular["cells"] = (
        {
            "cell_id": "modular-cell",
            "technique_config_id": "modular-config",
            "technique_key": "direct_prompt_injection",
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "modular-second",
            "attempts": (
                {"attempt_id": "modular-first", "attempt_number": 1, "status": "failed"},
                {"attempt_id": "modular-second", "attempt_number": 2, "status": "completed", "score": 8},
            ),
        },
    )

    assert ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy()).passed

    modular["cells"][0]["selected_attempt_id"] = "modular-first"
    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())
    assert any(item.path == "experiment.selected_attempts" for item in report.differences)


def test_parity_resolves_generated_config_ids_with_matching_embedded_identity() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    legacy["technique_configurations"] = (technique_configuration("legacy-config"),)
    modular["technique_configurations"] = (technique_configuration("modular-config"),)
    legacy["cells"] = (deterministic_cell("legacy-config", prefix="legacy"),)
    modular["cells"] = (deterministic_cell("modular-config", prefix="modular"),)

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert report.passed


def test_parity_rejects_cross_technique_identity_inside_resolved_config_spec() -> None:
    invalid = parity_run()
    configuration = technique_configuration("config-a")
    configuration["resolved_spec"]["technique_id"] = "other-technique"
    invalid["technique_configurations"] = (configuration,)
    invalid["cells"] = (deterministic_cell("config-a", prefix="invalid"),)

    with pytest.raises(ValueError, match="technique identity|technique_id"):
        ParityComparator().compare_deterministic(invalid, invalid, policy=strict_policy())


def test_parity_rejects_missing_required_embedded_config_identity() -> None:
    invalid = parity_run()
    configuration = technique_configuration("config-a")
    configuration["resolved_spec"].pop("technique_id")
    invalid["technique_configurations"] = (configuration,)
    invalid["cells"] = (deterministic_cell("config-a", prefix="invalid"),)

    with pytest.raises(ValueError, match="identity|technique_id|canonical"):
        ParityComparator().compare_deterministic(invalid, invalid, policy=strict_policy())


def test_parity_fails_when_resolved_cell_configuration_spec_differs() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    legacy["technique_configurations"] = (technique_configuration("same-config"),)
    modular["technique_configurations"] = (
        technique_configuration("same-config", spec_hash="config-spec-b"),
    )
    legacy["cells"] = (deterministic_cell("same-config", prefix="legacy"),)
    modular["cells"] = (deterministic_cell("same-config", prefix="modular"),)

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert not report.passed
    assert any(item.path == "experiment.selected_attempts" for item in report.differences)


@pytest.mark.parametrize("failure", ("dangling", "duplicate", "mismatched"))
def test_parity_rejects_invalid_cell_configuration_membership(failure: str) -> None:
    invalid = parity_run()
    invalid["cells"] = (deterministic_cell("config-a", prefix="invalid"),)
    if failure == "dangling":
        invalid["technique_configurations"] = (technique_configuration("config-b"),)
    elif failure == "duplicate":
        invalid["technique_configurations"] = (
            technique_configuration("config-a"),
            technique_configuration("config-a"),
        )
    else:
        invalid["technique_configurations"] = (
            technique_configuration("config-a", technique_key="other-technique"),
        )

    with pytest.raises(ValueError, match="configuration|technique"):
        ParityComparator().compare_deterministic(invalid, invalid, policy=strict_policy())


@pytest.mark.parametrize(
    "cell",
    (
        {
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "missing",
            "attempts": ({"attempt_id": "attempt-a", "status": "completed"},),
        },
        {
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "attempts": ({"attempt_id": "attempt-a", "status": "completed"},),
        },
        {
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "duplicate",
            "attempts": (
                {"attempt_id": "duplicate", "status": "failed"},
                {"attempt_id": "duplicate", "status": "completed"},
            ),
        },
    ),
)
def test_deterministic_parity_rejects_ambiguous_or_missing_attempt_selection(
    cell: dict[str, object],
) -> None:
    invalid = parity_run()
    cell["technique_config_id"] = "config-a"
    cell["technique_key"] = "direct_prompt_injection"
    invalid["technique_configurations"] = (technique_configuration("config-a"),)
    invalid["cells"] = (cell,)

    with pytest.raises(ValueError, match="selected attempt|duplicate attempt"):
        ParityComparator().compare_deterministic(invalid, invalid, policy=strict_policy())


def test_parity_compares_non_selected_retry_lifecycle_semantics() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    legacy["technique_configurations"] = (technique_configuration("legacy-config"),)
    modular["technique_configurations"] = (technique_configuration("modular-config"),)
    legacy["cells"] = (
        {
            "technique_config_id": "legacy-config",
            "technique_key": "direct_prompt_injection",
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "legacy-selected",
            "attempts": (
                {
                    "attempt_id": "legacy-first",
                    "attempt_number": 1,
                    "status": "failed",
                    "retry_kind": "restart",
                },
                {
                    "attempt_id": "legacy-selected",
                    "attempt_number": 2,
                    "status": "completed",
                },
            ),
        },
    )
    modular["cells"] = (
        {
            "technique_config_id": "modular-config",
            "technique_key": "direct_prompt_injection",
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "modular-selected",
            "attempts": (
                {
                    "attempt_id": "modular-first",
                    "attempt_number": 1,
                    "status": "cancelled",
                    "retry_kind": "resume",
                },
                {
                    "attempt_id": "modular-selected",
                    "attempt_number": 2,
                    "status": "completed",
                },
            ),
        },
    )

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert not report.passed
    assert any(item.path == "experiment.selected_attempts" for item in report.differences)


def test_parity_canonicalizes_equivalent_retry_lineage_with_generated_ids() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    legacy["technique_configurations"] = (technique_configuration("legacy-config"),)
    modular["technique_configurations"] = (technique_configuration("modular-config"),)
    legacy_cell = deterministic_cell("legacy-config", prefix="legacy")
    modular_cell = deterministic_cell("modular-config", prefix="modular")
    legacy_cell["selected_attempt_id"] = "legacy-child"
    modular_cell["selected_attempt_id"] = "modular-child"
    legacy_cell["attempts"] = (
        {
            "attempt_id": "legacy-root",
            "attempt_number": 1,
            "status": "failed",
            "retry_kind": "restart",
        },
        {
            "attempt_id": "legacy-child",
            "parent_attempt_id": "legacy-root",
            "attempt_number": 2,
            "status": "completed",
            "retry_kind": "resume",
        },
    )
    modular_cell["attempts"] = (
        {
            "attempt_id": "modular-root",
            "attempt_number": 1,
            "status": "failed",
            "retry_kind": "restart",
        },
        {
            "attempt_id": "modular-child",
            "parent_attempt_id": "modular-root",
            "attempt_number": 2,
            "status": "completed",
            "retry_kind": "resume",
        },
    )
    legacy["cells"] = (legacy_cell,)
    modular["cells"] = (modular_cell,)

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert report.passed


def test_parity_fails_when_retry_lineage_differs() -> None:
    legacy = parity_run()
    modular = parity_run(implementation_id="modular")
    legacy["technique_configurations"] = (technique_configuration("legacy-config"),)
    modular["technique_configurations"] = (technique_configuration("modular-config"),)
    legacy_cell = deterministic_cell("legacy-config", prefix="legacy")
    modular_cell = deterministic_cell("modular-config", prefix="modular")
    legacy_cell["selected_attempt_id"] = "legacy-selected"
    modular_cell["selected_attempt_id"] = "modular-selected"
    legacy_cell["attempts"] = (
        {"attempt_id": "legacy-root", "attempt_number": 1, "status": "failed"},
        {
            "attempt_id": "legacy-middle",
            "parent_attempt_id": "legacy-root",
            "attempt_number": 2,
            "status": "failed",
        },
        {
            "attempt_id": "legacy-selected",
            "parent_attempt_id": "legacy-middle",
            "attempt_number": 3,
            "status": "completed",
        },
    )
    modular_cell["attempts"] = (
        {"attempt_id": "modular-root", "attempt_number": 1, "status": "failed"},
        {
            "attempt_id": "modular-middle",
            "parent_attempt_id": "modular-root",
            "attempt_number": 2,
            "status": "failed",
        },
        {
            "attempt_id": "modular-selected",
            "parent_attempt_id": "modular-root",
            "attempt_number": 3,
            "status": "completed",
        },
    )
    legacy["cells"] = (legacy_cell,)
    modular["cells"] = (modular_cell,)

    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())

    assert not report.passed
    assert any(item.path == "experiment.selected_attempts" for item in report.differences)


@pytest.mark.parametrize("invalid_topology", ("dangling", "self", "cycle"))
def test_parity_rejects_invalid_retry_topology(invalid_topology: str) -> None:
    invalid = parity_run()
    invalid["technique_configurations"] = (technique_configuration("config-a"),)
    cell = deterministic_cell("config-a", prefix="invalid")
    cell["selected_attempt_id"] = "attempt-b"
    attempts = [
        {"attempt_id": "attempt-a", "attempt_number": 1, "status": "failed"},
        {
            "attempt_id": "attempt-b",
            "parent_attempt_id": "attempt-a",
            "attempt_number": 2,
            "status": "completed",
        },
    ]
    if invalid_topology == "dangling":
        attempts[1]["parent_attempt_id"] = "missing"
    elif invalid_topology == "self":
        attempts[1]["parent_attempt_id"] = "attempt-b"
    else:
        attempts[0]["parent_attempt_id"] = "attempt-b"
    cell["attempts"] = tuple(attempts)
    invalid["cells"] = (cell,)

    with pytest.raises(ValueError, match="parent|cycle|retry topology"):
        ParityComparator().compare_deterministic(invalid, invalid, policy=strict_policy())


def test_live_score_requires_explicit_succeeded_target_judge_and_preserves_zero() -> None:
    missing_evidence = (
        {
            "dataset_id": "dataset-a",
            "profile_ids": {"target": "target-profile", "target_judge": "judge-profile"},
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "attempt-a",
            "attempts": (
                {
                    "attempt_id": "attempt-a",
                    "status": "completed",
                    "score": 0,
                    "profile_ids": {
                        "target": "target-profile",
                        "target_judge": "judge-profile",
                    },
                    "calls": (
                        {"role": "target", "profile_id": "target-profile"},
                        {"role": "target_judge", "profile_id": "judge-profile"},
                    ),
                },
            ),
        },
    )
    with pytest.raises(ValueError, match="succeeded target_judge"):
        compare_live_samples(
            missing_evidence,
            missing_evidence,
            policy=live_policy(max_mean_score_delta=0),
        )

    valid_zero = deepcopy(missing_evidence)
    valid_zero[0]["attempts"][0]["official_judge"] = {
        "definition_id": "target_judge",
        "status": "succeeded",
        "score": 0,
        "profile_id": "judge-profile",
    }
    report = compare_live_samples(
        valid_zero,
        valid_zero,
        policy=live_policy(max_mean_score_delta=0),
    )

    assert report.passed
    assert report.metrics["legacy"]["mean_score"] == 0.0


def test_live_judge_scale_preserves_zero_and_treats_seven_as_success() -> None:
    policy = ParityPolicy(
        policy_id="judge-scale",
        success_threshold=7,
        dataset_id="dataset-a",
        repeat_count=1,
        dataset_sample_count=2,
        profile_ids={"target": "target-profile", "target_judge": "judge-profile"},
    )
    samples = (
        explicit_live_sample(score=0, prompt_id="prompt-zero", prefix="zero"),
        explicit_live_sample(score=7, prompt_id="prompt-seven", prefix="seven"),
    )

    report = compare_live_samples(samples, samples, policy=policy)

    assert report.passed
    assert report.metrics["legacy"]["mean_score"] == 3.5
    assert report.metrics["legacy"]["success_rate"] == 0.5


def test_live_missing_judge_evidence_counts_as_non_success() -> None:
    policy = ParityPolicy(
        policy_id="live-missing-judge",
        dataset_id="dataset-a",
        repeat_count=1,
        dataset_sample_count=2,
        profile_ids={"target": "target-profile", "target_judge": "judge-profile"},
    )
    succeeded = explicit_live_sample(
        score=80,
        prompt_id="prompt-a",
        prefix="succeeded",
    )
    missing = explicit_live_sample(
        score=80,
        prompt_id="prompt-b",
        prefix="missing",
    )
    missing["attempts"][0].pop("score")
    missing["attempts"][0].pop("official_judge")
    missing["attempts"][0]["profile_ids"] = {"target": "target-profile"}
    missing["attempts"][0]["calls"] = (
        {"role": "target", "profile_id": "target-profile"},
    )

    report = compare_live_samples((succeeded, missing), (succeeded, missing), policy=policy)

    assert report.passed
    assert report.metrics["legacy"]["sample_count"] == 2
    assert report.metrics["legacy"]["scored_sample_count"] == 1
    assert report.metrics["legacy"]["success_rate"] == 0.5


def test_live_policy_requires_target_and_target_judge_profile_pins() -> None:
    policy = ParityPolicy(
        dataset_id="dataset-a",
        repeat_count=1,
        dataset_sample_count=1,
        profile_ids={"target": "target-profile"},
    )

    with pytest.raises(ValueError, match="target_judge"):
        policy.validate_live_configuration()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_attempt_judge",
        "missing_judge_call",
        "missing_judge_record_profile",
        "mismatched_judge_profile",
        "unapproved_role",
    ),
)
def test_live_numeric_score_requires_owned_attempt_call_and_judge_profiles(
    mutation: str,
) -> None:
    sample_value = explicit_live_sample(score=0)
    attempt = sample_value["attempts"][0]
    if mutation == "missing_attempt_judge":
        attempt["profile_ids"].pop("target_judge")
    elif mutation == "missing_judge_call":
        attempt["calls"] = (attempt["calls"][0],)
    elif mutation == "missing_judge_record_profile":
        attempt["official_judge"].pop("profile_id")
    elif mutation == "mismatched_judge_profile":
        attempt["official_judge"]["profile_id"] = "other-judge-profile"
    else:
        attempt["profile_ids"]["unapproved"] = "unapproved-profile"
        attempt["calls"] = (
            *attempt["calls"],
            {"role": "unapproved", "profile_id": "unapproved-profile"},
        )

    with pytest.raises(ValueError, match="profile|role|target_judge"):
        compare_live_samples(
            (sample_value,),
            (sample_value,),
            policy=live_policy(max_mean_score_delta=0),
        )


def test_live_policy_derives_and_enforces_dataset_repeat_and_profile_evidence() -> None:
    policy = ParityPolicy(
        policy_id="live-owned",
        max_mean_score_delta=0,
        max_success_rate_delta=0,
        dataset_id="dataset-a",
        repeat_count=2,
        dataset_sample_count=2,
        profile_ids={"target": "target-profile", "target_judge": "judge-profile"},
    )
    profiles = {"target": "target-profile", "target_judge": "judge-profile"}
    samples = tuple(
        explicit_live_sample(
            score=80,
            prompt_id=prompt_id,
            repeat_index=repeat_index,
            profile_ids=profiles,
            prefix=f"{prompt_id}-{repeat_index}",
        )
        for prompt_id in ("prompt-a", "prompt-b")
        for repeat_index in (1, 2)
    )

    report = compare_live_samples(samples, samples, policy=policy)

    assert report.passed
    assert report.metrics["legacy"]["sample_count"] == 4
    assert report.policy.expected_live_sample_count == 4

    missing = samples[:-1]
    missing_report = compare_live_samples(missing, missing, policy=policy)
    assert not missing_report.passed
    assert any(
        item.path == "live.legacy.expected_sample_count"
        for item in missing_report.differences
    )


@pytest.mark.parametrize(
    "sample_override",
    (
        {"dataset_id": "dataset-b"},
        {"profile_ids": {"target": "different-profile", "target_judge": "judge-profile"}},
        {"repeat_index": 3},
    ),
)
def test_live_policy_rejects_sample_evidence_outside_predeclared_ownership(
    sample_override: dict[str, object],
) -> None:
    policy = ParityPolicy(
        policy_id="live-owned",
        max_mean_score_delta=0,
        max_success_rate_delta=0,
        dataset_id="dataset-a",
        repeat_count=2,
        expected_sample_count=2,
        profile_ids={"target": "target-profile", "target_judge": "judge-profile"},
    )
    sample_value = explicit_live_sample(
        score=80,
        profile_ids={"target": "target-profile", "target_judge": "judge-profile"},
    )
    sample_value.update(sample_override)

    with pytest.raises(ValueError, match="dataset|profile|repeat"):
        compare_live_samples((sample_value,), (sample_value,), policy=policy)


def test_live_policy_rejects_conflicting_derived_and_explicit_sample_counts() -> None:
    with pytest.raises(ValueError, match="expected_sample_count"):
        ParityPolicy(
            dataset_id="dataset-a",
            repeat_count=2,
            dataset_sample_count=2,
            expected_sample_count=3,
            profile_ids={"target": "target-profile", "target_judge": "judge-profile"},
        )


def test_live_policy_rejects_count_correct_but_incomplete_repeat_coverage() -> None:
    policy = ParityPolicy(
        policy_id="live-owned",
        dataset_id="dataset-a",
        repeat_count=2,
        dataset_sample_count=2,
        profile_ids={"target": "target-profile", "target_judge": "judge-profile"},
    )
    samples = tuple(
        explicit_live_sample(
            score=80,
            prompt_id=f"prompt-{index}",
            repeat_index=1,
            prefix=f"sample-{index}",
        )
        for index in range(4)
    )

    with pytest.raises(ValueError, match="repeat coverage"):
        compare_live_samples(samples, samples, policy=policy)


def test_live_policy_requires_positive_integral_dataset_cell_ownership() -> None:
    policy = ParityPolicy(
        dataset_id="dataset-a",
        repeat_count=2,
        expected_sample_count=3,
        profile_ids={"target": "target-profile", "target_judge": "judge-profile"},
    )

    with pytest.raises(ValueError, match="divisible"):
        policy.validate_live_configuration()


def test_live_selected_attempt_mapping_must_be_an_exact_member() -> None:
    sample_value = explicit_live_sample(score=80)
    sample_value["selected_attempt"] = {
        **sample_value["attempts"][0],
        "score": 79,
        "official_judge": {
            "definition_id": "target_judge",
            "status": "succeeded",
            "score": 79,
        },
    }

    with pytest.raises(ValueError, match="exact member"):
        compare_live_samples(
            (sample_value,),
            (sample_value,),
            policy=live_policy(max_mean_score_delta=0),
        )


def test_parity_rejects_numeric_score_on_failed_judge() -> None:
    legacy = {"events": ()}
    modular = {
        "events": (
            {
                "event_type": "judge.failed",
                "round": 1,
                "payload": {"definition_id": "target_judge", "status": "failed", "score": 4},
            },
        )
    }

    with pytest.raises(ValueError, match="only succeeded judges may carry numeric scores"):
        ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())


def test_live_comparison_counts_one_selected_attempt_per_cell() -> None:
    report = compare_live_samples(
        legacy=(sample(score=80, attempts=2),),
        modular=(sample(score=70, attempts=1),),
        policy=live_policy(max_mean_score_delta=10.0),
    )

    assert report.metrics["legacy"]["sample_count"] == 1
    assert report.metrics["legacy"]["total_attempts"] == 2
    assert report.metrics["legacy"]["mean_score"] == 80.0
    assert report.passed


def test_live_comparison_uses_selected_attempt_score_from_attempt_records() -> None:
    legacy = (
        {
            "cell_id": "generated-a",
            "dataset_id": "dataset-a",
            "profile_ids": {"target": "target-profile", "target_judge": "judge-profile"},
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "selected-a",
            "attempts": (
                {"attempt_id": "failed-a", "status": "failed", "score": None},
                {
                    "attempt_id": "selected-a",
                    "status": "completed",
                    "score": 80,
                    "profile_ids": {
                        "target": "target-profile",
                        "target_judge": "judge-profile",
                    },
                    "calls": (
                        {"role": "target", "profile_id": "target-profile"},
                        {"role": "target_judge", "profile_id": "judge-profile"},
                    ),
                    "official_judge": {
                        "definition_id": "target_judge",
                        "status": "succeeded",
                        "score": 80,
                        "profile_id": "judge-profile",
                    },
                },
            ),
        },
    )
    modular = (
        {
            "cell_id": "generated-b",
            "dataset_id": "dataset-a",
            "profile_ids": {"target": "target-profile", "target_judge": "judge-profile"},
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "selected-b",
            "attempts": (
                {
                    "attempt_id": "selected-b",
                    "status": "completed",
                    "score": 80,
                    "profile_ids": {
                        "target": "target-profile",
                        "target_judge": "judge-profile",
                    },
                    "calls": (
                        {"role": "target", "profile_id": "target-profile"},
                        {"role": "target_judge", "profile_id": "judge-profile"},
                    ),
                    "official_judge": {
                        "definition_id": "target_judge",
                        "status": "succeeded",
                        "score": 80,
                        "profile_id": "judge-profile",
                    },
                },
            ),
        },
    )

    report = compare_live_samples(legacy, modular, policy=live_policy(max_mean_score_delta=0))

    assert report.passed
    assert report.metrics["legacy"]["sample_count"] == 1
    assert report.metrics["legacy"]["scored_sample_count"] == 1
    assert report.metrics["legacy"]["total_attempts"] == 2


def test_live_comparison_fails_outside_pinned_mean_tolerance() -> None:
    report = compare_live_samples(
        legacy=(sample(score=80, attempts=1),),
        modular=(sample(score=69, attempts=1),),
        policy=live_policy(max_mean_score_delta=10.0),
    )

    assert not report.passed
    assert any(item.path == "live.mean_score" for item in report.differences)


def test_live_comparison_rejects_failed_selected_judge_with_score() -> None:
    invalid_sample = explicit_live_sample(score=40)
    invalid_sample["attempts"][0]["official_judge"]["status"] = "failed"
    invalid = (invalid_sample,)
    with pytest.raises(ValueError, match="succeeded target_judge"):
        compare_live_samples(invalid, invalid, policy=live_policy(max_mean_score_delta=0))


def test_live_comparison_rejects_duplicate_stable_cell_identity() -> None:
    duplicate = explicit_live_sample(score=80)
    with pytest.raises(ValueError, match="one selected attempt per stable cell identity"):
        compare_live_samples(
            (duplicate, duplicate),
            (duplicate, duplicate),
            policy=live_policy(max_mean_score_delta=0),
        )


def test_live_comparison_rejects_unresolved_selected_attempt_id() -> None:
    invalid = (
        {
            "dataset_id": "dataset-a",
            "profile_ids": {"target": "target-profile", "target_judge": "judge-profile"},
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "missing",
            "attempts": ({"attempt_id": "attempt-a", "score": 80},),
        },
    )
    with pytest.raises(ValueError, match="selected attempt does not resolve"):
        compare_live_samples(invalid, invalid, policy=live_policy(max_mean_score_delta=0))


def test_live_comparison_rejects_selected_sample_with_zero_attempts() -> None:
    invalid = (
        {
            "dataset_id": "dataset-a",
            "profile_ids": {"target": "target-profile", "target_judge": "judge-profile"},
            "prompt_id": "prompt-a",
            "repeat_index": 1,
            "selected_attempt_id": "missing",
            "attempts": (),
        },
    )
    with pytest.raises(ValueError, match="selected attempt requires at least one"):
        compare_live_samples(invalid, invalid, policy=live_policy(max_mean_score_delta=0))


def test_live_comparison_validates_attempt_count_per_cell_not_running_total() -> None:
    invalid = (
        explicit_live_sample(score=80, prompt_id="prompt-a", prefix="first"),
        {
            "dataset_id": "dataset-a",
            "profile_ids": {"target": "target-profile", "target_judge": "judge-profile"},
            "prompt_id": "prompt-b",
            "repeat_index": 1,
            "selected_attempt_id": "missing",
            "attempts": (),
        },
    )
    with pytest.raises(ValueError, match="selected attempt requires at least one"):
        compare_live_samples(invalid, invalid, policy=live_policy(max_mean_score_delta=0))


def request_context(*, role: str = "target", operation_id: str = "operation-a") -> ProviderRequestContext:
    return ProviderRequestContext(
        correlation=CORRELATION,
        model_call_id="generated-model-call",
        role=role,
        operation_id=operation_id,
        technique_id="direct_prompt_injection",
        technique_version="1.0.0",
        implementation_id="modular",
        transport=ModelTransport.OPENAI_COMPATIBLE,
    )


def test_scripted_role_client_consumes_queue_and_records_semantic_call() -> None:
    client = ScriptedRoleClient(
        role="target",
        responses=(
            ScriptedResponse(
                text="scripted response",
                expected_operation_id="operation-a",
                usage={"input_tokens": 4, "output_tokens": 2},
            ),
        ),
    )

    result = client.chat_correlated_once(
        [{"role": "user", "content": "hello"}],
        request_context=request_context(),
        temperature=0,
    )

    assert result.text == "scripted response"
    assert result.usage == {"input_tokens": 4, "output_tokens": 2}
    assert client.remaining == 0
    assert client.calls[0].role == "target"
    assert client.calls[0].messages == ({"role": "user", "content": "hello"},)
    assert client.calls[0].options == {"temperature": 0}
    assert client.calls[0].operation_id == "operation-a"
    assert client.calls[0].call_index == 1
    client.assert_exhausted()


def test_scripted_call_and_scenario_deeply_snapshot_nested_inputs() -> None:
    response_format = {"schema": {"type": "object"}}
    scenario_inputs = {"parameters": {"temperature": 0}}
    client = ScriptedRoleClient(role="target", responses=("response",))
    scenario = ParityScenario(
        technique_id="direct_prompt_injection",
        legacy_implementation="legacy",
        modular_implementation="modular",
        seed=1,
        inputs=scenario_inputs,
    )

    client.chat_correlated_once(
        [],
        request_context=request_context(),
        response_format=response_format,
    )
    response_format["schema"]["type"] = "array"
    scenario_inputs["parameters"]["temperature"] = 1

    assert client.calls[0].options["response_format"] == {"schema": {"type": "object"}}
    assert scenario.inputs["parameters"] == {"temperature": 0}


def test_scripted_role_client_fails_immediately_on_unexpected_or_exhausted_call() -> None:
    client = ScriptedRoleClient(role="target", responses=("only",))
    with pytest.raises(AssertionError, match="expected role 'target'"):
        client.chat_correlated_once([], request_context=request_context(role="attacker"))

    client.chat_correlated_once([], request_context=request_context())
    with pytest.raises(AssertionError, match="response queue exhausted"):
        client.chat_correlated_once([], request_context=request_context())

    unconsumed = ScriptedRoleClient(role="target", responses=("unused",))
    with pytest.raises(AssertionError, match="1 scripted response"):
        unconsumed.assert_exhausted()


def test_scripted_role_client_rejects_invalid_queue_without_recording_call() -> None:
    client = ScriptedRoleClient(role="target", responses=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scripted responses must be"):
        client.chat_correlated_once([], request_context=request_context())

    assert client.calls == ()
    assert client.remaining == 1


def test_parity_scenario_resets_seed_before_each_arm_and_builds_pinned_selectors() -> None:
    scenario = ParityScenario(
        technique_id="direct_prompt_injection",
        technique_version="1.0.0",
        legacy_implementation="legacy",
        modular_implementation="modular",
        seed=417,
    )

    legacy_value, modular_value = scenario.run(lambda: random.random(), lambda: random.random())

    assert legacy_value == modular_value
    assert scenario.legacy_selector == TechniqueSelector(
        "direct_prompt_injection",
        technique_version="1.0.0",
        implementation_id="legacy",
    )
    assert scenario.modular_selector == TechniqueSelector(
        "direct_prompt_injection",
        technique_version="1.0.0",
        implementation_id="modular",
    )


@pytest.mark.parametrize("mode", ["deterministic", "live"])
def test_parity_only_restores_rng_for_seeded_legacy_arms(mode: str) -> None:
    before = random.getstate()
    expected = random.Random()
    expected.setstate(before)
    expected_draws = (
        [random.Random(417).random(), expected.random()]
        if mode == "deterministic"
        else [expected.random(), expected.random()]
    )
    draws = []

    def runner(request):
        draws.append(random.random())
        evidence = parity_run(implementation_id=request.route.registered.implementation.id)
        evidence.pop("contract")
        if mode == "live":
            evidence = [explicit_live_sample(score=80, attempts=1)]
        return {"identity": parity_route_identity(request.route), "evidence": evidence}

    runtime = parity_runtime(runner=runner)
    session = runtime.issue_deterministic_arms(
        "direct_prompt_injection",
        legacy_implementation_id="legacy",
        modular_implementation_id="modular",
    )
    execute = (
        runtime.execute_and_compare_deterministic
        if mode == "deterministic"
        else runtime.execute_and_compare_live
    )
    try:
        execute(
            session,
            policy=strict_policy() if mode == "deterministic" else live_policy(max_mean_score_delta=0),
            dataset_id="dataset-a", repeat_count=1,
            profile_ids={"target": "target-profile", "target_judge": "judge-profile"}, seed=417,
        )
        assert draws == expected_draws
        assert random.getstate() == expected.getstate()
    finally:
        random.setstate(before)


def test_report_path_redacts_secrets_and_never_contains_full_raw_responses(tmp_path: Path) -> None:
    legacy = parity_run(response="legacy raw model response sk-supersecret123")
    modular = parity_run(implementation_id="modular", response="modular full raw model response")
    report = ParityComparator().compare_deterministic(legacy, modular, policy=strict_policy())
    output = tmp_path / "reports" / "parity.json"

    report.write_json(output)

    text = output.read_text()
    assert "legacy raw model response" not in text
    assert "modular full raw model response" not in text
    assert "sk-supersecret123" not in text
    payload = json.loads(text)
    assert payload["passed"] is False
    assert payload["policy"]["fingerprint"] == strict_policy().fingerprint


def test_cli_without_captures_resolves_routes_and_runs_generic_arms(tmp_path: Path) -> None:
    requests: list[ParityArmRequest] = []

    def runner(request: ParityArmRequest) -> dict[str, object]:
        requests.append(request)
        implementation_id = request.route.registered.implementation.id
        evidence = parity_run(implementation_id=implementation_id)
        evidence.pop("contract")
        return {
            "identity": parity_route_identity(request.route),
            "evidence": evidence,
        }

    output = tmp_path / "parity.json"
    exit_code = parity_main(
        [
            "--technique",
            "direct_prompt_injection",
            "--legacy-implementation",
            "legacy",
            "--modular-implementation",
            "modular",
            "--mode",
            "deterministic",
            "--output",
            str(output),
        ],
        runtime=parity_runtime(runner=runner),
    )

    assert exit_code == 0
    assert json.loads(output.read_text())["passed"] is True
    assert [request.route.registered.implementation.id for request in requests] == [
        "legacy",
        "modular",
    ]


def test_routed_capture_injects_exact_resolved_definition_evidence() -> None:
    runtime = parity_runtime()
    session = runtime.issue_deterministic_arms(
        "direct_prompt_injection",
        legacy_implementation_id="legacy",
        modular_implementation_id="modular",
    )
    routes = runtime.deterministic_arm_routes(session)
    captures = []
    for route in routes:
        implementation_id = route.registered.implementation.id
        evidence = parity_run(implementation_id=implementation_id)
        evidence.pop("contract")
        captures.append(
            {"identity": parity_route_identity(route), "evidence": evidence}
        )

    pair = runtime._prepare_deterministic_pair(  # noqa: SLF001
        (captures[0], captures[1]),
        session,
    )

    assert pair.legacy_evidence["contract"]["route_identity"] == parity_route_identity(
        routes[0]
    )
    assert pair.legacy_evidence["contract"]["definition"]["identity"]["id"] == (
        "direct_prompt_injection"
    )
    assert pair.legacy_evidence["contract"]["resolved_spec"]["manifest_hash"] == (
        "a" * 64
    )
    assert ParityComparator()._compare_verified_deterministic_pair(  # noqa: SLF001
        pair,
        policy=strict_policy(),
    ).passed


def test_cli_rejects_forged_contract_that_does_not_match_resolved_route(
    tmp_path: Path,
) -> None:
    def runner(request: ParityArmRequest) -> dict[str, object]:
        evidence = parity_run(
            implementation_id=request.route.registered.implementation.id,
        )
        evidence["contract"] = {
            "definition": {"identity": {"id": "forged-technique"}},
            "resolved_spec": {
                "spec_hash": "forged-spec",
                "manifest_hash": "forged-manifest",
                "asset_hashes": {},
            },
        }
        return {
            "identity": parity_route_identity(request.route),
            "evidence": evidence,
        }

    output = tmp_path / "parity.json"
    exit_code = parity_main(
        [
            "--technique",
            "direct_prompt_injection",
            "--legacy-implementation",
            "legacy",
            "--modular-implementation",
            "modular",
            "--mode",
            "deterministic",
            "--output",
            str(output),
        ],
        runtime=parity_runtime(runner=runner),
    )

    assert exit_code == 1
    payload = json.loads(output.read_text())
    assert payload["passed"] is False
    assert payload["differences"][0]["path"] == "comparison.capture_validation"


def test_routed_contract_compares_implementation_contract_version(tmp_path: Path) -> None:
    def runner(request: ParityArmRequest) -> dict[str, object]:
        evidence = parity_run(
            implementation_id=request.route.registered.implementation.id,
        )
        evidence.pop("contract")
        return {
            "identity": parity_route_identity(request.route),
            "evidence": evidence,
        }

    output = tmp_path / "parity.json"
    exit_code = parity_main(
        [
            "--technique",
            "direct_prompt_injection",
            "--legacy-implementation",
            "legacy",
            "--modular-implementation",
            "modular",
            "--mode",
            "deterministic",
            "--output",
            str(output),
        ],
        runtime=parity_runtime(
            runner=runner,
            modular_contract_version="2",
        ),
    )

    assert exit_code == 1
    payload = json.loads(output.read_text())
    assert any(
        difference["path"] == "contract.resolved_spec"
        for difference in payload["differences"]
    )


def test_cli_without_registered_arm_runner_fails_without_passed_report(tmp_path: Path) -> None:
    output = tmp_path / "parity.json"

    exit_code = parity_main(
        [
            "--technique",
            "direct_prompt_injection",
            "--legacy-implementation",
            "legacy",
            "--modular-implementation",
            "modular",
            "--mode",
            "deterministic",
            "--output",
            str(output),
        ],
        runtime=parity_runtime(),
    )

    assert exit_code == 1
    payload = json.loads(output.read_text())
    assert payload["passed"] is False
    assert payload["metadata"]["status"] == "not_run"


def test_cli_capture_loading_failure_consumes_issued_route_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = parity_runtime()
    issued: list[object] = []
    original_issue = ParityRuntime.issue_deterministic_arms

    def record_issue(self, *args, **kwargs):
        session = original_issue(self, *args, **kwargs)
        issued.append(session)
        return session

    monkeypatch.setattr(ParityRuntime, "issue_deterministic_arms", record_issue)

    with pytest.raises(SystemExit) as raised:
        parity_main(
            [
                "--technique",
                "direct_prompt_injection",
                "--legacy-implementation",
                "legacy",
                "--modular-implementation",
                "modular",
                "--mode",
                "deterministic",
                "--legacy-input",
                str(tmp_path / "missing-legacy.json"),
                "--modular-input",
                str(tmp_path / "missing-modular.json"),
                "--output",
                str(tmp_path / "parity.json"),
            ],
            runtime=runtime,
        )

    assert raised.value.code == 2
    assert len(issued) == 1
    with pytest.raises(ValueError, match="consumed|stale|issued"):
        runtime.deterministic_arm_routes(issued[0])


def test_cli_rejects_unverified_offline_capture_identity_and_redacts_it(tmp_path: Path) -> None:
    runtime = parity_runtime()
    legacy_route = runtime.router.resolve(
        "direct_prompt_injection",
        implementation_id="legacy",
    )
    modular_route = runtime.router.resolve(
        "direct_prompt_injection",
        implementation_id="modular",
    )
    credential = "sk-capturecredential123"
    legacy_capture = {
        "identity": {
            **parity_route_identity(legacy_route),
            "profile_id": credential,
        },
        "evidence": parity_run(),
    }
    modular_capture = {
        "identity": parity_route_identity(modular_route),
        "evidence": parity_run(implementation_id="modular"),
    }
    legacy_path = tmp_path / "legacy.json"
    modular_path = tmp_path / "modular.json"
    legacy_path.write_text(json.dumps(legacy_capture, default=str))
    modular_path.write_text(json.dumps(modular_capture, default=str))
    output = tmp_path / "parity.json"

    exit_code = parity_main(
        [
            "--technique",
            "direct_prompt_injection",
            "--legacy-implementation",
            "legacy",
            "--modular-implementation",
            "modular",
            "--mode",
            "deterministic",
            "--legacy-input",
            str(legacy_path),
            "--modular-input",
            str(modular_path),
            "--output",
            str(output),
        ],
        runtime=runtime,
    )

    assert exit_code == 1
    serialized = output.read_text()
    assert credential not in serialized
    assert json.loads(serialized)["passed"] is False


def test_live_cli_rejects_policy_cli_ownership_mismatch_before_arm_runner(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "\n".join(
            (
                "policy_id: live-owned",
                "dataset_id: dataset-a",
                "repeat_count: 2",
                "dataset_sample_count: 1",
                "profile_ids:",
                "  target: target-profile",
                "  target_judge: judge-profile",
            )
        )
    )
    requests: list[ParityArmRequest] = []

    def runner(request: ParityArmRequest) -> object:
        requests.append(request)
        raise AssertionError("runner must not be called for mismatched live ownership")

    with pytest.raises(SystemExit) as raised:
        parity_main(
            [
                "--technique",
                "direct_prompt_injection",
                "--legacy-implementation",
                "legacy",
                "--modular-implementation",
                "modular",
                "--mode",
                "live",
                "--dataset-id",
                "dataset-b",
                "--repeat-count",
                "2",
                "--profile-id",
                "target=target-profile",
                "--profile-id",
                "target_judge=judge-profile",
                "--policy",
                str(policy),
                "--output",
                str(tmp_path / "parity.json"),
            ],
            runtime=parity_runtime(runner=runner),
        )

    assert raised.value.code == 2
    assert requests == []


def test_live_cli_rejects_threshold_mismatch_before_arm_runner(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "\n".join(
            (
                "policy_id: wrong-scale",
                "success_threshold: 70",
                "dataset_id: dataset-a",
                "repeat_count: 1",
                "dataset_sample_count: 1",
                "profile_ids:",
                "  target: target-profile",
                "  target_judge: judge-profile",
            )
        )
    )
    requests: list[ParityArmRequest] = []

    def runner(request: ParityArmRequest) -> object:
        requests.append(request)
        raise AssertionError("runner must not be called for a mismatched score scale")

    output = tmp_path / "parity.json"
    exit_code = parity_main(
        [
            "--technique",
            "direct_prompt_injection",
            "--legacy-implementation",
            "legacy",
            "--modular-implementation",
            "modular",
            "--mode",
            "live",
            "--dataset-id",
            "dataset-a",
            "--repeat-count",
            "1",
            "--target-profile-id",
            "target-profile",
            "--target-judge-profile-id",
            "judge-profile",
            "--policy",
            str(policy),
            "--output",
            str(output),
        ],
        runtime=parity_runtime(runner=runner),
    )

    assert exit_code == 1
    assert requests == []
    payload = json.loads(output.read_text())
    assert payload["differences"][0]["path"] == "comparison.route_threshold"


@pytest.mark.parametrize("captured_inputs", (False, True))
def test_live_cli_requires_explicit_threshold_before_runner_or_capture_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    captured_inputs: bool,
) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "\n".join(
            (
                "policy_id: omitted-threshold",
                "dataset_id: dataset-a",
                "repeat_count: 1",
                "dataset_sample_count: 1",
                "profile_ids:",
                "  target: target-profile",
                "  target_judge: judge-profile",
            )
        )
    )
    runtime = parity_runtime()
    requests: list[ParityArmRequest] = []
    loaded: list[Path] = []

    def capture_for(implementation_id: str) -> dict[str, object]:
        route = runtime.router.resolve(
            "direct_prompt_injection",
            implementation_id=implementation_id,
        )
        evidence = parity_run(implementation_id=implementation_id)
        evidence.pop("contract")
        return {"identity": parity_route_identity(route), "evidence": evidence}

    def runner(request: ParityArmRequest) -> dict[str, object]:
        requests.append(request)
        return capture_for(request.route.registered.implementation.id)

    captures = iter((capture_for("legacy"), capture_for("modular")))

    def load_capture(path: Path, _parser: object) -> object:
        loaded.append(path)
        return next(captures)

    monkeypatch.setattr("devtools.technique_parity.__main__._load_json", load_capture)
    arguments = [
        "--technique",
        "direct_prompt_injection",
        "--legacy-implementation",
        "legacy",
        "--modular-implementation",
        "modular",
        "--mode",
        "live",
        "--dataset-id",
        "dataset-a",
        "--repeat-count",
        "1",
        "--target-profile-id",
        "target-profile",
        "--target-judge-profile-id",
        "judge-profile",
        "--policy",
        str(policy),
        "--output",
        str(tmp_path / "parity.json"),
    ]
    if captured_inputs:
        arguments.extend(
            (
                "--legacy-input",
                str(tmp_path / "legacy.json"),
                "--modular-input",
                str(tmp_path / "modular.json"),
            )
        )

    with pytest.raises(SystemExit) as raised:
        parity_main(arguments, runtime=ParityRuntime(runtime.router, runner))

    assert raised.value.code == 2
    assert "success_threshold" in capsys.readouterr().err
    assert requests == []
    assert loaded == []


def test_live_cli_rejects_route_threshold_disagreement_before_arm_runner(
    tmp_path: Path,
) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "\n".join(
            (
                "policy_id: route-disagreement",
                "success_threshold: 7",
                "dataset_id: dataset-a",
                "repeat_count: 1",
                "dataset_sample_count: 1",
                "profile_ids:",
                "  target: target-profile",
                "  target_judge: judge-profile",
            )
        )
    )
    requests: list[ParityArmRequest] = []

    def runner(request: ParityArmRequest) -> object:
        requests.append(request)
        raise AssertionError("runner must not be called for route threshold disagreement")

    output = tmp_path / "parity.json"
    exit_code = parity_main(
        [
            "--technique",
            "direct_prompt_injection",
            "--legacy-implementation",
            "legacy",
            "--modular-implementation",
            "modular",
            "--mode",
            "live",
            "--dataset-id",
            "dataset-a",
            "--repeat-count",
            "1",
            "--target-profile-id",
            "target-profile",
            "--target-judge-profile-id",
            "judge-profile",
            "--policy",
            str(policy),
            "--output",
            str(output),
        ],
        runtime=parity_runtime(
            runner=runner,
            legacy_threshold=7,
            modular_threshold=8,
            modular_version="1.1.0",
        ),
    )

    assert exit_code == 1
    assert requests == []
    payload = json.loads(output.read_text())
    assert payload["differences"][0]["path"] == "comparison.route_threshold"


def test_cli_compares_captured_evidence_and_writes_redacted_report(tmp_path: Path) -> None:
    runtime = parity_runtime()
    legacy_route = runtime.router.resolve(
        "direct_prompt_injection",
        implementation_id="legacy",
    )
    modular_route = runtime.router.resolve(
        "direct_prompt_injection",
        implementation_id="modular",
    )
    legacy = tmp_path / "legacy.json"
    modular = tmp_path / "modular.json"
    output = tmp_path / "out" / "parity.json"
    legacy.write_text(
        json.dumps(
            {
                "identity": parity_route_identity(legacy_route),
                "evidence": json_parity_run(implementation_id="legacy", prefix="legacy"),
            }
        )
    )
    modular.write_text(
        json.dumps(
            {
                "identity": parity_route_identity(modular_route),
                "evidence": json_parity_run(implementation_id="modular", prefix="modular"),
            }
        )
    )

    exit_code = parity_main(
        [
            "--technique",
            "direct_prompt_injection",
            "--legacy-implementation",
            "legacy",
            "--modular-implementation",
            "modular",
            "--mode",
            "deterministic",
            "--legacy-input",
            str(legacy),
            "--modular-input",
            str(modular),
            "--output",
            str(output),
        ],
        runtime=runtime,
    )

    assert exit_code == 0
    assert json.loads(output.read_text())["passed"] is True


def test_live_cli_requires_policy_and_pins_it_before_reading_samples(tmp_path: Path) -> None:
    missing_input = tmp_path / "does-not-exist.json"
    output = tmp_path / "parity.json"
    with pytest.raises(SystemExit) as raised:
        parity_main(
            [
                "--technique",
                "direct_prompt_injection",
                "--legacy-implementation",
                "legacy",
                "--modular-implementation",
                "modular",
                "--mode",
                "live",
                "--dataset-id",
                "dataset-a",
                "--repeat-count",
                "3",
                "--profile-id",
                "target=target-profile",
                "--profile-id",
                "target_judge=judge-profile",
                "--legacy-input",
                str(missing_input),
                "--modular-input",
                str(missing_input),
                "--output",
                str(output),
            ]
        )

    assert raised.value.code == 2
    assert not output.exists()


def test_live_cli_writes_loaded_policy_fingerprint(tmp_path: Path) -> None:
    runtime = parity_runtime()
    legacy_route = runtime.router.resolve(
        "direct_prompt_injection",
        implementation_id="legacy",
    )
    modular_route = runtime.router.resolve(
        "direct_prompt_injection",
        implementation_id="modular",
    )
    policy = tmp_path / "policy.yaml"
    policy_bytes = (
        b"policy_id: migration-live\n"
        b"version: '2026-08-26'\n"
        b"max_mean_score_delta: 5.0\n"
        b"max_success_rate_delta: 0.1\n"
        b"success_threshold: 7\n"
        b"dataset_id: dataset-a\n"
        b"repeat_count: 3\n"
        b"dataset_sample_count: 1\n"
        b"profile_ids:\n"
        b"  target: target-profile\n"
        b"  target_judge: judge-profile\n"
    )
    policy.write_bytes(policy_bytes)
    legacy = tmp_path / "legacy.json"
    modular = tmp_path / "modular.json"
    legacy_samples = [
        explicit_live_sample(
            score=80,
            attempts=2,
            repeat_index=index,
            prefix=f"legacy-{index}",
        )
        for index in (1, 2, 3)
    ]
    modular_samples = [
        explicit_live_sample(
            score=80,
            attempts=1,
            repeat_index=index,
            prefix=f"modular-{index}",
        )
        for index in (1, 2, 3)
    ]
    legacy.write_text(
        json.dumps(
            {
                "identity": parity_route_identity(legacy_route),
                "evidence": legacy_samples,
            }
        )
    )
    modular.write_text(
        json.dumps(
            {
                "identity": parity_route_identity(modular_route),
                "evidence": modular_samples,
            }
        )
    )
    output = tmp_path / "parity.json"

    exit_code = parity_main(
        [
            "--technique",
            "direct_prompt_injection",
            "--legacy-implementation",
            "legacy",
            "--modular-implementation",
            "modular",
            "--mode",
            "live",
            "--dataset-id",
            "dataset-a",
            "--repeat-count",
            "3",
            "--profile-id",
            "target=target-profile",
            "--profile-id",
            "target_judge=judge-profile",
            "--policy",
            str(policy),
            "--legacy-input",
            str(legacy),
            "--modular-input",
            str(modular),
            "--output",
            str(output),
        ],
        runtime=runtime,
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["policy"]["fingerprint"] == ParityPolicy.from_file(policy).fingerprint
    assert payload["metadata"]["dataset_id"] == "dataset-a"
    assert payload["metadata"]["repeat_count"] == 3
