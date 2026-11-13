from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.techniques.contracts.definition import (
    ConverterDefinition,
    DetectorDefinition,
    HistoryDefinition,
    ImplementationKind,
    InputDefinition,
    JudgeDefinition,
    JudgePurpose,
    JudgeSchedule,
    ModelRoleDefinition,
    ResultPolicy,
    SeedDefinition,
    SeedStrategy,
    TechniqueCapability,
    TechniqueDefinition,
    TechniqueIdentity,
    TechniqueImplementation,
    ToolDefinition,
)
from app.techniques.contracts.events import (
    ExecutionEvent,
    JudgeResult,
    JudgeStatus,
)
from app.techniques.contracts.execution import (
    AttackOutcome,
    CorrelationIds,
    ExecutionStatus,
    FailureCategory,
    ModelRoleBinding,
    ModelTransport,
    ResolvedRunSpec,
    ResultDisposition,
    TechniqueOutcome,
)
from app.techniques.contracts.presentation import MetricDefinition, PresentationSection


def test_definition_supports_multiple_histories_and_judges() -> None:
    definition = TechniqueDefinition(
        identity=TechniqueIdentity(
            id="alibi",
            version="1.0.0",
            name="ALIBI",
            family="history-poisoning",
        ),
        summary="History-poisoning evaluation",
        capabilities=frozenset(
            {
                TechniqueCapability.ITERATIVE,
                TechniqueCapability.MULTIPLE_HISTORIES,
            }
        ),
        model_roles=(ModelRoleDefinition(id="target", required=True),),
        histories=(
            HistoryDefinition(id="audit_history"),
            HistoryDefinition(id="target_visible_history"),
        ),
        judges=(
            JudgeDefinition(
                id="target_judge",
                schedule=JudgeSchedule.EVERY_ROUND,
                purposes=frozenset({JudgePurpose.OFFICIAL_SCORE}),
            ),
        ),
        result_policy=ResultPolicy(
            official_judge="target_judge",
            final_response="target.real_response",
        ),
    )

    assert definition.result_policy.official_judge == "target_judge"
    assert {history.id for history in definition.histories} == {
        "audit_history",
        "target_visible_history",
    }


def test_implementation_kind_distinguishes_legacy_workflow_and_python() -> None:
    assert {kind.value for kind in ImplementationKind} == {
        "legacy",
        "workflow",
        "python",
    }
    implementation = TechniqueImplementation(
        id="modular",
        technique_id="alibi",
        technique_version="1.0.0",
        kind=ImplementationKind.PYTHON,
        entrypoint="executor:AlibiExecutor",
    )
    assert implementation.kind is ImplementationKind.PYTHON


def test_definition_copies_configuration_mappings_and_remains_frozen() -> None:
    parameters_schema = {"type": "object"}
    presets = {"strong": {"poisoning_ratio": 0.7}}
    extensions = {"research": "fixture"}
    definition = TechniqueDefinition(
        identity=TechniqueIdentity(
            id="probe",
            version="1.0.0",
            name="Probe",
            family="test",
        ),
        summary="Probe definition",
        capabilities=frozenset({TechniqueCapability.SINGLE_SHOT}),
        parameters_schema=parameters_schema,
        presets=presets,
        extensions=extensions,
    )

    parameters_schema["type"] = "array"
    presets["new"] = {}
    extensions["research"] = "changed"

    assert definition.parameters_schema == {"type": "object"}
    assert definition.presets == {"strong": {"poisoning_ratio": 0.7}}
    assert definition.extensions == {"research": "fixture"}
    with pytest.raises(TypeError):
        definition.parameters_schema["type"] = "number"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        definition.summary = "changed"  # type: ignore[misc]


def test_definition_declares_inputs_seeds_detectors_tools_and_converters() -> None:
    definition = TechniqueDefinition(
        identity=TechniqueIdentity(
            id="document-probe",
            version="1.0.0",
            name="Document Probe",
            family="test",
        ),
        summary="Exercises declarative resources",
        capabilities=frozenset(
            {
                TechniqueCapability.DOCUMENTS,
                TechniqueCapability.TOOLS,
                TechniqueCapability.CONVERTERS,
            }
        ),
        inputs=(
            InputDefinition(
                id="document",
                kind="artifact",
                required=True,
                media_types=("application/pdf",),
                max_count=1,
                max_size_bytes=1_000_000,
            ),
        ),
        detectors=(
            DetectorDefinition(
                id="refusal",
                inputs=("target_response",),
                output_schema={"type": "object"},
            ),
        ),
        seed=SeedDefinition(
            strategy=SeedStrategy.PACKAGED_ASSET,
            source="prompts/seed.txt",
        ),
        tools=(
            ToolDefinition(
                id="lookup",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                policy={"read_only": True},
            ),
        ),
        converters=(
            ConverterDefinition(
                id="pdf_to_text",
                input_schema={"type": "string"},
                output_schema={"type": "string"},
                policy={"max_output_bytes": 100_000},
            ),
        ),
    )

    assert definition.inputs[0].media_types == ("application/pdf",)
    assert definition.seed.strategy is SeedStrategy.PACKAGED_ASSET
    assert definition.detectors[0].inputs == ("target_response",)
    assert definition.tools[0].policy == {"read_only": True}
    assert definition.converters[0].id == "pdf_to_text"


def test_failed_judge_cannot_have_numeric_score() -> None:
    with pytest.raises(
        ValueError,
        match="only succeeded judge results may contain a score",
    ):
        JudgeResult(
            definition_id="target_judge",
            status=JudgeStatus.FAILED,
            score=0,
        )


def test_succeeded_judge_requires_a_score_and_zero_is_valid() -> None:
    with pytest.raises(ValueError, match="succeeded judge results require a score"):
        JudgeResult(
            definition_id="target_judge",
            status=JudgeStatus.SUCCEEDED,
        )

    result = JudgeResult(
        definition_id="target_judge",
        status=JudgeStatus.SUCCEEDED,
        score=0,
    )
    assert result.score == 0


def test_event_carries_full_response_ownership_chain_and_copies_payload() -> None:
    correlation = CorrelationIds(
        experiment_id="exp-1",
        cell_id="cell-1",
        run_id="run-1",
        attempt_id="attempt-1",
    )
    payload = {"response": {"id": "response-1"}}
    event = ExecutionEvent.create(
        correlation=correlation,
        sequence=1,
        event_type="run.started",
        technique_id="probe",
        technique_version="1.0.0",
        implementation_id="modular",
        payload=payload,
    )
    payload["response"]["id"] = "mutated"

    assert event.correlation.cell_id == "cell-1"
    assert event.sequence == 1
    assert event.payload["response"]["id"] == "response-1"
    assert event.timestamp.tzinfo is not None
    assert event.event_id


def test_role_binding_uses_explicit_provider_for_transport() -> None:
    direct = ModelRoleBinding.from_profile(
        logical_role="target",
        profile_id="profile-1",
        provider="openai",
        model="model-a",
    )
    proxy = ModelRoleBinding.from_profile(
        logical_role="target_judge",
        profile_id="profile-2",
        provider="litellm",
        model="model-b",
    )

    assert direct.transport is ModelTransport.OPENAI_COMPATIBLE
    assert proxy.transport is ModelTransport.LITELLM_PROXY


def test_resolved_run_spec_pins_scientific_configuration() -> None:
    binding = ModelRoleBinding.from_profile(
        logical_role="target",
        profile_id="profile-1",
        provider="openai",
        model="model-a",
    )
    parameters = {"rounds": 3, "nested": {"ratio": 0.4}}
    spec = ResolvedRunSpec(
        technique_id="probe",
        technique_version="1.0.0",
        implementation_id="modular",
        implementation_kind=ImplementationKind.WORKFLOW,
        contract_version="1",
        manifest_hash="a" * 64,
        asset_hashes={"workflow.yaml": "b" * 64},
        resolved_parameters=parameters,
        model_role_bindings={"target": binding},
        judge_policies={"target_judge": {"required": True}},
        effective_limits={"max_rounds": 5},
    )
    parameters["rounds"] = 99
    parameters["nested"]["ratio"] = 1

    assert spec.resolved_parameters["rounds"] == 3
    assert spec.resolved_parameters["nested"]["ratio"] == 0.4
    assert spec.model_role_bindings["target"].profile_id == "profile-1"
    assert spec.judge_policies["target_judge"]["required"] is True


def test_outcome_keeps_execution_result_and_attack_states_separate() -> None:
    outcome = TechniqueOutcome(
        execution_status=ExecutionStatus.COMPLETED,
        result_disposition=ResultDisposition.TESTED,
        attack_outcome=AttackOutcome.NOT_SUCCEEDED,
        final_response_ref="response-1",
        official_judge_ref="judge-result-1",
        stopping_reason="max_rounds",
        auxiliary_judge_refs=("diagnostic-1",),
        artifact_refs=("artifact-1",),
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert outcome.result_disposition is ResultDisposition.TESTED
    assert outcome.attack_outcome is AttackOutcome.NOT_SUCCEEDED
    assert FailureCategory.JUDGE_PARSE.value == "judge_parse"


def test_presentation_and_metric_contracts_copy_configuration() -> None:
    section_options = {"history": "audit_history"}
    metric_filters = {"judge": "target_judge"}
    section = PresentationSection(
        id="conversation",
        kind="conversation",
        title="Conversation",
        source="events",
        options=section_options,
    )
    metric = MetricDefinition(
        id="attack_success_rate",
        name="Attack success rate",
        source="official_judge",
        aggregation="rate",
        group_by=("technique_id", "round"),
        filters=metric_filters,
    )
    section_options["history"] = "changed"
    metric_filters["judge"] = "changed"

    assert section.options["history"] == "audit_history"
    assert metric.filters["judge"] == "target_judge"
