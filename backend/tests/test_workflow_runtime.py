from __future__ import annotations

import asyncio
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from types import MappingProxyType
from typing import Mapping

import pytest

from app.persistence.technique_runtime import NoopTechniqueRuntimePersistence
from app.techniques.contracts import (
    AttackOutcome,
    CorrelationIds,
    ConverterDefinition,
    DetectorDefinition,
    ExecutionStatus,
    ImplementationKind,
    InputDefinition,
    JudgeDefinition,
    JudgePurpose,
    HistoryDefinition,
    ModelRoleBinding,
    ModelRoleDefinition,
    ResolvedRunSpec,
    ResultDisposition,
    ResultPolicy,
    SeedDefinition,
    SeedStrategy,
    TechniqueCapability,
    TechniqueDefinition,
    TechniqueIdentity,
    TechniqueImplementation,
    TechniqueRequest,
    ToolDefinition,
)
from app.techniques.core.budgets import BudgetExhausted
from app.techniques.core.cancellation import CancellationError
from app.techniques.core.context import ExecutionContextFactory
from app.techniques.core.models import ProviderCallResult
from app.techniques.workflow import (
    BranchDefinition,
    Condition,
    ConditionEvaluator,
    MAX_CONDITION_DEPTH,
    MAX_JSON_COLLECTION_ITEMS,
    MAX_JSON_DEPTH,
    MAX_JSON_STRING_LENGTH,
    MAX_WORKFLOW_STEPS,
    StepDefinition,
    WorkflowDefinition,
    WorkflowState,
    WorkflowValidationError,
    load_workflow,
)


class WorkflowModelClient:
    max_tokens = 32

    def __init__(self, *responses: str, timeline: list[str] | None = None) -> None:
        self.responses = list(responses)
        self.calls: list[object] = []
        self.requests: list[tuple[list[dict[str, str]], object, Mapping[str, object]]] = []
        self.timeline = timeline

    def chat_correlated_once(
        self,
        messages: list[dict[str, str]],
        *,
        request_context: object,
        **options: object,
    ) -> ProviderCallResult:
        self.calls.append(request_context)
        self.requests.append((messages, request_context, options))
        if self.timeline is not None:
            self.timeline.append("provider.dispatched")
        return ProviderCallResult(
            text=self.responses.pop(0),
            usage={"total_tokens": 1},
        )


class RaisingWorkflowModelClient:
    max_tokens = 32

    def __init__(self, error_type: type[Exception]) -> None:
        self.error_type = error_type
        self.requests: list[object] = []

    def chat_correlated_once(
        self,
        messages: list[dict[str, str]],
        *,
        request_context: object,
        **options: object,
    ) -> ProviderCallResult:
        self.requests.append(request_context)
        raise self.error_type("provider completion ownership is unknown")


class WorkflowTargetAdapter:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[object] = []

    def execute(self, request: object) -> SimpleNamespace:
        self.requests.append(request)
        return SimpleNamespace(final_output=self.response, metadata={})


def workflow_execution_fixture() -> tuple[
    TechniqueRequest,
    object,
    TechniqueDefinition,
    WorkflowTargetAdapter,
]:
    definition = TechniqueDefinition(
        identity=TechniqueIdentity("probe", "1.0.0", "Probe", "test"),
        summary="workflow fixture",
        capabilities=frozenset({TechniqueCapability.SINGLE_SHOT}),
        inputs=(InputDefinition("intent", "text", required=True),),
        model_roles=(ModelRoleDefinition("target_judge"),),
        judges=(
            JudgeDefinition(
                "target_judge",
                purposes=frozenset({JudgePurpose.OFFICIAL_SCORE}),
                role="target_judge",
                evidence=("intent", "response"),
            ),
        ),
        result_policy=ResultPolicy(
            official_judge="target_judge",
            final_response="target.real_response",
            success_threshold=7,
        ),
        assets=("workflow.yaml", "prompt.txt"),
    )
    binding = ModelRoleBinding.from_profile(
        logical_role="target_judge",
        profile_id="judge-profile",
        provider="openai",
        model="judge-model",
    )
    request = TechniqueRequest(
        correlation=CorrelationIds("run-workflow", "attempt-workflow"),
        spec=ResolvedRunSpec(
            technique_id="probe",
            technique_version="1.0.0",
            implementation_id="modular",
            implementation_kind=ImplementationKind.WORKFLOW,
            contract_version="1",
            manifest_hash="a" * 64,
            asset_hashes={
                "workflow.yaml": "b" * 64,
                "prompt.txt": "e" * 64,
            },
            model_role_bindings={"target_judge": binding},
            effective_limits={
                "max_model_calls": 2,
                "max_target_calls": 1,
                "max_judge_calls": 1,
            },
        ),
        inputs={"intent": "test intent"},
    )
    package = SimpleNamespace(
        definition=definition,
        implementation=TechniqueImplementation(
            id="modular",
            technique_id="probe",
            technique_version="1.0.0",
            kind=ImplementationKind.WORKFLOW,
            entrypoint="workflow.yaml",
        ),
        manifest_hash=request.spec.manifest_hash,
        asset_hashes=request.spec.asset_hashes,
    )
    target = WorkflowTargetAdapter("target response")
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target_judge": WorkflowModelClient('{"score": 8, "rationale": "complied"}')},
        target_adapter=target,
        template_assets={
            "prompt.txt": "Intent: {intent}",
            "rogue.txt": "not pinned: {intent}",
        },
    ).create(request=request, definition=definition, loaded_package=package)
    return request, context, definition, target


def workflow_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "start": "render",
        "steps": [
            {"id": "render", "type": "render", "next": "finish"},
            {"id": "finish", "type": "finish"},
        ],
    }
    payload.update(overrides)
    return payload


def packaged_workflow_execution_fixture(
    tmp_path: Path,
    *,
    workflow_bytes: bytes | None = None,
    prompt_bytes: bytes = b"Package intent: {intent}",
    context_prompt: str = "SUBSTITUTED intent: {intent}",
):
    from app.techniques.core.catalog import RegisteredTechnique, TechniqueCatalog
    from app.techniques.core.router import ExecutionRouter
    from app.techniques.plugins.manifest import load_technique_package

    source = Path(__file__).parent / "fixtures" / "techniques" / "valid_probe"
    package_root = tmp_path / "valid_probe"
    shutil.copytree(source, package_root)
    manifest_path = package_root / "technique.yaml"
    manifest = manifest_path.read_text(encoding="utf-8")
    manifest = (
        manifest.replace(
            "parameters:\n  schema: schemas/parameters.json\n  presets:\n    default:\n      temperature: 0\n",
            "",
        )
        .replace(
            "result_policy:\n  official_judge: target_judge\n  final_response: target_response\n  success_threshold: 7\n",
            "result_policy:\n  final_response: target_response\n",
        )
        .replace(
            "assets:\n  - workflow.yaml\n",
            "assets:\n  - workflow.yaml\n  - prompt.txt\n",
        )
    )
    manifest_path.write_text(manifest, encoding="utf-8")
    (package_root / "prompt.txt").write_bytes(prompt_bytes)
    (package_root / "workflow.yaml").write_bytes(
        workflow_bytes
        or b"""version: 1
start: render
steps:
  - id: render
    type: render
    options:
      template_asset: prompt.txt
      output: rendered_prompt
    next: target
  - id: target
    type: target_call
    options:
      messages:
        - role: user
          content_from: values.rendered_prompt
      output: target_response
    next: finish
  - id: finish
    type: finish
    options:
      final_response_from: values.target_response
"""
    )
    package = load_technique_package(
        package_root,
        supported_capabilities=frozenset(TechniqueCapability),
    )
    registered = RegisteredTechnique(
        definition=package.definition,
        implementation=package.implementation,
        manifest_hash=package.manifest_hash,
        asset_hashes=package.asset_hashes,
        package=package,
    )
    catalog = TechniqueCatalog()
    catalog.register(registered)
    router = ExecutionRouter(catalog)
    target_binding = ModelRoleBinding.from_profile(
        logical_role="target",
        profile_id="target-profile",
        provider="openai",
        model="target-model",
    )
    judge_binding = ModelRoleBinding.from_profile(
        logical_role="target_judge",
        profile_id="judge-profile",
        provider="openai",
        model="judge-model",
    )
    request = TechniqueRequest(
        correlation=CorrelationIds("run-package", "attempt-package"),
        spec=ResolvedRunSpec(
            technique_id="probe",
            technique_version="1.0.0",
            implementation_id="modular",
            implementation_kind=ImplementationKind.WORKFLOW,
            contract_version="1",
            manifest_hash=package.manifest_hash,
            asset_hashes=package.asset_hashes,
            model_role_bindings={
                "target": target_binding,
                "target_judge": judge_binding,
            },
            effective_limits={"max_target_calls": 1},
        ),
        inputs={"intent": "trusted"},
    )
    target = WorkflowTargetAdapter("package response")
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={
            "target": WorkflowModelClient("unused"),
            "target_judge": WorkflowModelClient("unused"),
        },
        target_adapter=target,
        template_assets={"prompt.txt": context_prompt},
        detector_registry={"refusal": lambda inputs: {"matched": False}},
    ).create(
        request=request,
        definition=package.definition,
        loaded_package=package,
    )
    return router, request, context, target, registered, package_root


def multiple_judge_execution_fixture(
    *,
    official_output: str = '{"score": 1, "rationale": "minimum score"}',
    diagnostic_output: str = '{"score": 9, "rationale": "diagnostic"}',
):
    definition = TechniqueDefinition(
        identity=TechniqueIdentity("judge-probe", "1.0.0", "Judge Probe", "test"),
        summary="finish evidence fixture",
        capabilities=frozenset(
            {
                TechniqueCapability.SINGLE_SHOT,
                TechniqueCapability.MULTIPLE_JUDGES,
            }
        ),
        inputs=(InputDefinition("intent", "text", required=True),),
        model_roles=(
            ModelRoleDefinition("target_judge"),
            ModelRoleDefinition("diagnostic_judge"),
        ),
        judges=(
            JudgeDefinition(
                "target_judge",
                purposes=frozenset({JudgePurpose.OFFICIAL_SCORE}),
                role="target_judge",
                evidence=("intent", "response"),
            ),
            JudgeDefinition(
                "diagnostic_judge",
                purposes=frozenset({JudgePurpose.DIAGNOSTICS}),
                role="diagnostic_judge",
                evidence=("intent", "response"),
            ),
        ),
        result_policy=ResultPolicy(
            official_judge="target_judge",
            final_response="target.real_response",
            success_threshold=7,
            auxiliary_judges=("diagnostic_judge",),
        ),
        assets=("workflow.yaml",),
    )
    bindings = {
        role: ModelRoleBinding.from_profile(
            logical_role=role,
            profile_id=f"{role}-profile",
            provider="openai",
            model=f"{role}-model",
        )
        for role in ("target_judge", "diagnostic_judge")
    }
    request = TechniqueRequest(
        correlation=CorrelationIds("run-judge-evidence", "attempt-judge-evidence"),
        spec=ResolvedRunSpec(
            technique_id="judge-probe",
            technique_version="1.0.0",
            implementation_id="modular",
            implementation_kind=ImplementationKind.WORKFLOW,
            contract_version="1",
            manifest_hash="9" * 64,
            asset_hashes={"workflow.yaml": "8" * 64},
            model_role_bindings=bindings,
            effective_limits={
                "max_model_calls": 2,
                "max_target_calls": 2,
                "max_judge_calls": 2,
            },
        ),
        inputs={"intent": "test"},
    )
    package = SimpleNamespace(
        definition=definition,
        implementation=TechniqueImplementation(
            id="modular",
            technique_id="judge-probe",
            technique_version="1.0.0",
            kind=ImplementationKind.WORKFLOW,
            entrypoint="workflow.yaml",
        ),
        manifest_hash=request.spec.manifest_hash,
        asset_hashes=request.spec.asset_hashes,
    )
    target = WorkflowTargetAdapter("real response")
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={
            "target_judge": WorkflowModelClient(official_output),
            "diagnostic_judge": WorkflowModelClient(diagnostic_output),
        },
        target_adapter=target,
    ).create(request=request, definition=definition, loaded_package=package)
    return request, context, definition


def test_workflow_executor_supports_the_brief_keyword_call_contract() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="finish",
            steps=[{"id": "finish", "type": "finish"}],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor().execute(
            request=request,
            context=context,
            workflow=workflow,
            result_policy=ResultPolicy(),
        )
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert context.events.events()[-1].event_type == "run.completed"


def test_workflow_executor_constructor_and_keyword_calls_are_equivalent() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    constructor_request, constructor_context, _, _ = workflow_execution_fixture()
    keyword_request, keyword_context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="finish",
            steps=[{"id": "finish", "type": "finish"}],
        )
    )

    constructor_outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
        ).execute(constructor_request, constructor_context)
    )
    keyword_outcome = asyncio.run(
        WorkflowExecutor().execute(
            request=keyword_request,
            context=keyword_context,
            workflow=workflow,
            result_policy=ResultPolicy(),
        )
    )

    assert constructor_outcome == keyword_outcome
    assert [event.event_type for event in constructor_context.events.events()] == [
        event.event_type for event in keyword_context.events.events()
    ]


def test_workflow_executor_direct_calls_require_validated_contract_types() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="finish",
            steps=[{"id": "finish", "type": "finish"}],
        )
    )

    with pytest.raises(TypeError, match="validated WorkflowDefinition"):
        asyncio.run(
            WorkflowExecutor().execute(
                request=request,
                context=context,
                workflow=workflow_payload(),  # type: ignore[arg-type]
                result_policy=ResultPolicy(),
            )
        )
    with pytest.raises(TypeError, match="ResultPolicy"):
        asyncio.run(
            WorkflowExecutor().execute(
                request=request,
                context=context,
                workflow=workflow,
                result_policy={},  # type: ignore[arg-type]
            )
        )


def test_workflow_executor_reuses_no_state_across_direct_runs() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    first_request, first_context, _, _ = workflow_execution_fixture()
    second_request, second_context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="render",
            steps=[
                {
                    "id": "render",
                    "type": "render",
                    "options": {"text": "fresh", "fields": {}},
                    "next": "finish",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )
    executor = WorkflowExecutor()

    async def run_twice():
        first = await executor.execute(
            request=first_request,
            context=first_context,
            workflow=workflow,
            result_policy=ResultPolicy(),
        )
        second = await executor.execute(
            request=second_request,
            context=second_context,
            workflow=workflow,
            result_policy=ResultPolicy(),
        )
        return first, second

    first, second = asyncio.run(run_twice())

    assert first.execution_status is ExecutionStatus.COMPLETED
    assert second.execution_status is ExecutionStatus.COMPLETED
    for context in (first_context, second_context):
        started = [event for event in context.events.events() if event.event_type == "step.started"]
        assert [event.operation_id for event in started] == [
            "workflow:render:visit:1",
            "workflow:finish:visit:1",
        ]


def test_workflow_executes_render_target_judge_finish() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, definition, target = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="render",
            steps=[
                {
                    "id": "render",
                    "type": "render",
                    "options": {
                        "asset": "prompt.txt",
                        "fields": {"intent": "inputs.intent"},
                        "output": "rendered_prompt",
                    },
                    "next": "target",
                },
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content_ref": "rendered_prompt"}],
                        "output": "target_response",
                    },
                    "next": "judge",
                },
                {
                    "id": "judge",
                    "type": "judge",
                    "options": {
                        "definition": "target_judge",
                        "evidence": {
                            "intent": "inputs.intent",
                            "response": "target_response.text",
                        },
                        "provenance": {
                            "target_response_event_id_from": (
                                "values.target_response.event_id"
                            ),
                            "workflow_step_id": {"value": "forged-step"},
                        },
                    },
                    "next": "finish",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=definition.result_policy,
            trusted_template_assets={"prompt.txt": "Intent: {intent}"},
        ).execute(request, context)
    )

    assert outcome.outputs["official_score"] == 8
    assert outcome.final_response_ref
    assert len(target.requests) == 1
    target_request = target.requests[0]
    assert target_request.correlation == request.correlation
    assert target_request.operation_id == "workflow:target:visit:1"
    assert target_request.messages == ({"role": "user", "content": "Intent: test intent"},)
    assert target_request.history_snapshot_id
    assert context.budget.usage.target_calls == 1
    assert context.budget.usage.judge_calls == 1
    assert context.budget.usage.model_calls == 1
    finish_evidence = outcome.outputs["finish_evidence"]
    assert finish_evidence["final_response_event_id"] == outcome.final_response_ref
    assert finish_evidence["official_judge_result_id"] == outcome.official_judge_ref
    judge_completed = next(event for event in context.events.events() if event.event_type == "judge.completed")
    assert judge_completed.operation_id == "workflow:judge:visit:1"
    target_completed = next(
        event
        for event in context.events.events()
        if event.event_type == "target.completed"
    )
    assert judge_completed.payload["target_response_event_id"] == target_completed.event_id
    assert judge_completed.payload["workflow_step_id"] == "judge"
    assert [event.event_type for event in context.events.events() if event.event_type.startswith("step.")] == [
        "step.started",
        "step.completed",
        "step.started",
        "step.completed",
        "step.started",
        "step.completed",
        "step.started",
        "step.completed",
    ]


def test_workflow_accepts_declared_from_references_and_values_namespace() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, definition, _ = workflow_execution_fixture()
    definition = replace(
        definition,
        histories=(HistoryDefinition("audit_history"),),
    )
    context.loaded_package.definition = definition
    workflow = load_workflow(
        workflow_payload(
            start="render",
            steps=[
                {
                    "id": "render",
                    "type": "render",
                    "options": {
                        "template_asset": "prompt.txt",
                        "output": "rendered_prompt",
                    },
                    "next": "target",
                },
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [
                            {
                                "role": "user",
                                "content_from": "values.rendered_prompt",
                            }
                        ],
                        "output": "target_response",
                        "append_to_history": "audit_history",
                    },
                    "next": "judge",
                },
                {
                    "id": "judge",
                    "type": "judge",
                    "options": {
                        "definition_id": "target_judge",
                        "evidence": {
                            "intent_from": "inputs.intent",
                            "response_from": "values.target_response",
                        },
                        "output": "target_judge",
                    },
                    "next": "finish",
                },
                {
                    "id": "finish",
                    "type": "finish",
                    "options": {
                        "final_response_from": "values.target_response",
                        "official_judge_from": "values.target_judge",
                    },
                },
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=definition.result_policy,
            trusted_template_assets={"prompt.txt": "Intent: {intent}"},
        ).execute(request, context)
    )

    assert outcome.outputs["official_score"] == 8
    audit = context.histories.snapshot("audit_history")
    assert [message.role for message in audit.messages] == ["user", "assistant"]
    assert audit.messages[-1].content == "target response"


def test_workflow_core_nodes_delegate_to_execution_context_services() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    class RecordingPersistence(NoopTechniqueRuntimePersistence):
        def __init__(self, timeline: list[str]) -> None:
            super().__init__()
            self.timeline = timeline
            self.model_records: list[object] = []
            self.artifact_records: list[object] = []
            self.checkpoint_records: list[object] = []

        def record_model_call_requested(self, call) -> None:
            self.timeline.append("model.persisted")
            self.model_records.append(call)
            super().record_model_call_requested(call)

        def append_event(self, event) -> None:
            super().append_event(event)
            if event.event_type in {"model.requested", "tool.requested"}:
                self.timeline.append(f"event.persisted:{event.event_type}")

        def save_artifact(self, artifact) -> None:
            self.artifact_records.append(artifact)
            super().save_artifact(artifact)

        def save_checkpoint(self, checkpoint) -> None:
            self.checkpoint_records.append(checkpoint)
            super().save_checkpoint(checkpoint)

    definition = TechniqueDefinition(
        identity=TechniqueIdentity("core-probe", "1.0.0", "Core Probe", "test"),
        summary="all core node services",
        capabilities=frozenset(
            {
                TechniqueCapability.SINGLE_SHOT,
                TechniqueCapability.MULTIPLE_HISTORIES,
                TechniqueCapability.TOOLS,
                TechniqueCapability.CONVERTERS,
            }
        ),
        inputs=(InputDefinition("intent", "text", required=True),),
        model_roles=(ModelRoleDefinition("planner"),),
        histories=(HistoryDefinition("audit"),),
        detectors=(DetectorDefinition("contains_plan", inputs=("text",)),),
        seed=SeedDefinition(strategy=SeedStrategy.SUPPLIED, required=True),
        tools=(ToolDefinition("echo"),),
        converters=(ConverterDefinition("artifact_converter"),),
        result_policy=ResultPolicy(),
        assets=("workflow.yaml",),
    )
    binding = ModelRoleBinding.from_profile(
        logical_role="planner",
        profile_id="planner-profile",
        provider="openai",
        model="planner-model",
    )
    request = TechniqueRequest(
        correlation=CorrelationIds("run-core-nodes", "attempt-core-nodes"),
        spec=ResolvedRunSpec(
            technique_id="core-probe",
            technique_version="1.0.0",
            implementation_id="modular",
            implementation_kind=ImplementationKind.WORKFLOW,
            contract_version="1",
            manifest_hash="c" * 64,
            asset_hashes={"workflow.yaml": "d" * 64},
            model_role_bindings={"planner": binding},
            effective_limits={
                "max_model_calls": 1,
                "max_tool_calls": 1,
                "max_artifact_bytes": 1_000,
            },
        ),
        inputs={"intent": "make a plan"},
    )
    package = SimpleNamespace(
        definition=definition,
        implementation=TechniqueImplementation(
            id="modular",
            technique_id="core-probe",
            technique_version="1.0.0",
            kind=ImplementationKind.WORKFLOW,
            entrypoint="workflow.yaml",
        ),
        manifest_hash=request.spec.manifest_hash,
        asset_hashes=request.spec.asset_hashes,
    )
    timeline: list[str] = []
    persistence = RecordingPersistence(timeline)
    model_client = WorkflowModelClient("planned response", timeline=timeline)
    tool_arguments: list[Mapping[str, object]] = []
    detector_inputs: list[Mapping[str, object]] = []
    converted_artifact_ids: list[str] = []

    def echo(arguments: Mapping[str, object]) -> Mapping[str, object]:
        timeline.append("tool.dispatched")
        tool_arguments.append(arguments)
        return {"echo": arguments["text"]}

    def convert(artifact_id: str) -> Mapping[str, object]:
        converted_artifact_ids.append(artifact_id)
        return {"artifact_id": artifact_id}

    def detect(inputs: Mapping[str, object]) -> Mapping[str, object]:
        detector_inputs.append(inputs)
        return {
            "matched": "plan" in str(inputs["text"]),
            "rationale": "literal check",
        }

    context = ExecutionContextFactory(
        persistence=persistence,
        model_clients={"planner": model_client},
        target_adapter=WorkflowTargetAdapter("unused"),
        tool_registry={"echo": echo},
        converter_registry={"artifact_converter": convert},
        detector_registry={"contains_plan": detect},
        supplied_seeds=("seed one",),
    ).create(request=request, definition=definition, loaded_package=package)
    workflow = load_workflow(
        workflow_payload(
            start="inputs",
            steps=[
                {"id": "inputs", "type": "input_materialize", "next": "seed"},
                {
                    "id": "seed",
                    "type": "seed_resolve",
                    "options": {"output": "seeds"},
                    "next": "render",
                },
                {
                    "id": "render",
                    "type": "render",
                    "options": {
                        "text": "Intent: {intent}",
                        "fields": {"intent": "inputs.intent"},
                        "output": "rendered",
                    },
                    "next": "model",
                },
                {
                    "id": "model",
                    "type": "model_call",
                    "options": {
                        "role": "planner",
                        "messages": [{"role": "user", "content_ref": "rendered"}],
                        "output": "planner_response",
                    },
                    "next": "history",
                },
                {
                    "id": "history",
                    "type": "history_create",
                    "options": {"history": "audit"},
                    "next": "append",
                },
                {
                    "id": "append",
                    "type": "history_append",
                    "options": {
                        "history": "audit",
                        "role": "assistant",
                        "content_ref": "planner_response.text",
                    },
                    "next": "snapshot",
                },
                {
                    "id": "snapshot",
                    "type": "history_snapshot",
                    "options": {"history": "audit", "output": "audit_snapshot"},
                    "next": "tool",
                },
                {
                    "id": "tool",
                    "type": "tool_call",
                    "options": {
                        "tool": "echo",
                        "arguments": {"text": "planner_response.text"},
                        "output": "tool_result",
                    },
                    "next": "detector",
                },
                {
                    "id": "detector",
                    "type": "detector",
                    "options": {
                        "detector": "contains_plan",
                        "inputs": {"text": "planner_response.text"},
                    },
                    "next": "artifact",
                },
                {
                    "id": "artifact",
                    "type": "artifact_create",
                    "options": {
                        "media_type": "text/plain",
                        "content_ref": "planner_response.text",
                        "output": "artifact",
                    },
                    "next": "convert",
                },
                {
                    "id": "convert",
                    "type": "convert",
                    "options": {
                        "converter": "artifact_converter",
                        "artifact_ref": "artifact",
                        "output": "converted",
                    },
                    "next": "checkpoint",
                },
                {
                    "id": "checkpoint",
                    "type": "checkpoint",
                    "options": {"output": "checkpoint"},
                    "next": "finish",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=definition.result_policy,
        ).execute(request, context)
    )

    event_types = {event.event_type for event in context.events.events()}
    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert {
        "artifact.created",
        "checkpoint.saved",
        "converter.completed",
        "detector.completed",
        "history.created",
        "message.appended",
        "model.completed",
        "tool.completed",
    } <= event_types
    assert (
        timeline.index("model.persisted")
        < timeline.index("event.persisted:model.requested")
        < timeline.index("provider.dispatched")
    )
    assert timeline.index("event.persisted:tool.requested") < timeline.index("tool.dispatched")
    assert model_client.requests[0][0] == [{"role": "user", "content": "Intent: make a plan"}]
    model_context = model_client.requests[0][1]
    assert model_context.operation_id == "workflow:model:visit:1"
    assert model_context.correlation == request.correlation
    assert persistence.model_records[0].operation_attempt == 1
    assert tool_arguments == [{"text": "planned response"}]
    assert detector_inputs == [{"text": "planned response"}]
    assert converted_artifact_ids == [persistence.artifact_records[0].artifact_id]
    assert persistence.artifact_records[0].correlation == request.correlation
    assert persistence.artifact_records[0].media_type == "text/plain"
    assert persistence.checkpoint_records[0].correlation == request.correlation
    assert persistence.checkpoint_records[0].spec_hash
    assert persistence.checkpoint_records[0].payload["history_heads"]["audit"]
    assert len(context.histories.heads()["audit"]) == 1
    assert context.budget.usage.model_calls == 1
    assert context.budget.usage.tool_calls == 1
    assert context.budget.usage.tokens == 1
    assert context.budget.usage.artifact_bytes == len(b"planned response")
    started = [event for event in context.events.events() if event.event_type == "step.started"]
    assert all(event.attempt == 1 for event in started)
    assert [event.operation_id for event in started] == [
        f"workflow:{step_id}:visit:1"
        for step_id in (
            "inputs",
            "seed",
            "render",
            "model",
            "history",
            "append",
            "snapshot",
            "tool",
            "detector",
            "artifact",
            "convert",
            "checkpoint",
            "finish",
        )
    ]


def test_target_call_persists_request_event_before_dispatch() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    timeline: list[str] = []

    class OrderingPersistence(NoopTechniqueRuntimePersistence):
        def append_event(self, event) -> None:
            super().append_event(event)
            if event.event_type == "target.requested":
                timeline.append("target.requested.persisted")

    class OrderingTarget(WorkflowTargetAdapter):
        def execute(self, request: object) -> SimpleNamespace:
            timeline.append("target.dispatched")
            return super().execute(request)

    request, original, definition, _ = workflow_execution_fixture()
    target = OrderingTarget("owned response")
    context = ExecutionContextFactory(
        persistence=OrderingPersistence(),
        model_clients={"target_judge": WorkflowModelClient("unused")},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original.loaded_package,
    )
    workflow = load_workflow(
        workflow_payload(
            start="target",
            steps=[
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "owned"}],
                        "output": "target_response",
                        "round": 2,
                    },
                    "next": "finish",
                },
                {
                    "id": "finish",
                    "type": "finish",
                    "options": {"final_response_from": "values.target_response"},
                },
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(final_response="target.real_response"),
        ).execute(request, context)
    )

    assert timeline == ["target.requested.persisted", "target.dispatched"]
    assert context.budget.usage.target_calls == 1
    assert target.requests[0].operation_id == "workflow:target:visit:1"
    assert {
        event.round
        for event in context.events.events()
        if event.event_type in {"target.requested", "target.completed"}
    } == {2}
    assert outcome.final_response_ref == outcome.outputs["finish_evidence"]["final_response_event_id"]


def test_workflow_cancellation_is_terminal_before_the_next_step() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    context.cancellation.cancel("do not dispatch")
    workflow = load_workflow(workflow_payload())

    with pytest.raises(CancellationError, match="do not dispatch"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    assert not any(event.event_type == "step.started" for event in context.events.events())
    assert context.events.events()[-1].event_type == "run.cancelled"


def test_render_rejects_an_approved_but_unpinned_template_asset() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            steps=[
                {
                    "id": "render",
                    "type": "render",
                    "options": {
                        "asset": "rogue.txt",
                        "fields": {"intent": "inputs.intent"},
                    },
                }
            ]
        )
    )

    with pytest.raises(WorkflowNodeError, match="verified or explicitly trusted"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )


def test_history_nodes_reject_undeclared_history_names() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="history",
            steps=[
                {
                    "id": "history",
                    "type": "history_create",
                    "options": {"history": "rogue"},
                }
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="not declared"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )


def test_finish_rejects_a_missing_declared_final_response() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, definition, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="finish",
            steps=[{"id": "finish", "type": "finish"}],
        )
    )

    with pytest.raises(WorkflowNodeError, match="matched no completed target evidence"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=definition.result_policy,
            ).execute(request, context)
        )

    event_types = [event.event_type for event in context.events.events()]
    assert event_types.count("run.failed") == 1
    assert "run.completed" not in event_types
    run_failed = next(event for event in context.events.events() if event.event_type == "run.failed")
    assert run_failed.payload == {"category": "finish"}


def test_workflow_cannot_complete_without_a_finish_node() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="render",
            steps=[
                {
                    "id": "render",
                    "type": "render",
                    "options": {"text": "done", "fields": {}},
                }
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="terminated without finish"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    event_types = [event.event_type for event in context.events.events()]
    assert event_types.count("run.failed") == 1
    assert "run.completed" not in event_types


def test_projection_failure_precedes_run_completed_and_emits_one_run_failed() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="target",
            steps=[
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {"messages": [{"role": "user", "content": "hello"}]},
                }
            ],
        )
    )

    with pytest.raises(ValueError, match="matched no event"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(final_response="missing.response"),
            ).execute(request, context)
        )

    event_types = [event.event_type for event in context.events.events()]
    assert event_types.count("run.failed") == 1
    assert "run.completed" not in event_types


def test_run_completed_is_last_only_after_finish_projection_succeeds() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="finish",
            steps=[{"id": "finish", "type": "finish"}],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    event_types = [event.event_type for event in context.events.events()]
    assert event_types[-1] == "run.completed"
    assert "run.failed" not in event_types


def test_finish_terminalization_failure_cannot_follow_an_ordinary_error_edge() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="invalid_finish",
            steps=[
                {
                    "id": "invalid_finish",
                    "type": "finish",
                    "options": {
                        "official_judge_from": "values.inputs",
                        "error_types": ["finish"],
                    },
                    "error": "rescue",
                },
                {
                    "id": "rescue",
                    "type": "render",
                    "options": {"text": "must not execute", "fields": {}},
                    "next": "fallback_finish",
                },
                {"id": "fallback_finish", "type": "finish"},
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="not declared by the result policy"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    assert not any(
        event.payload.get("step_id") == "rescue"
        for event in context.events.events()
        if event.event_type == "step.started"
    )
    terminal_events = [
        event.event_type for event in context.events.events() if event.event_type in {"run.completed", "run.failed"}
    ]
    assert terminal_events == ["run.failed"]


def test_budget_exhaustion_cannot_follow_an_ordinary_error_edge() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    context.budget.reserve(target_calls=1)
    workflow = load_workflow(
        workflow_payload(
            start="target",
            steps=[
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "hello"}],
                        "error_types": ["target_call"],
                    },
                    "error": "finish",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    with pytest.raises(BudgetExhausted, match="target_calls"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    assert context.events.events()[-1].event_type == "budget.exhausted"
    assert not any(
        event.event_type == "step.completed" and event.payload.get("step_id") == "finish"
        for event in context.events.events()
    )


def test_typed_ordinary_error_edge_is_fail_closed() -> None:
    from app.techniques.workflow.executor import WorkflowErrorPolicyError, WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            steps=[
                {
                    "id": "render",
                    "type": "render",
                    "options": {
                        "text": "{missing}",
                        "fields": {"missing": "inputs.absent"},
                    },
                    "error": "finish",
                },
                {"id": "finish", "type": "finish"},
            ]
        )
    )

    with pytest.raises(WorkflowErrorPolicyError, match="requires non-empty error_types"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    failed = [event for event in context.events.events() if event.event_type == "step.failed"]
    assert failed[-1].payload == {
        "step_id": "render",
        "step_type": "render",
        "category": "render",
        "error_type": "WorkflowNodeError",
    }


def test_declared_error_category_can_transition_to_finish() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            steps=[
                {
                    "id": "render",
                    "type": "render",
                    "options": {
                        "text": "{missing}",
                        "fields": {"missing": "inputs.absent"},
                        "error_types": ["render"],
                    },
                    "error": "finish",
                },
                {"id": "finish", "type": "finish"},
            ]
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert outcome.result_disposition is ResultDisposition.NOT_TESTED
    assert outcome.attack_outcome is AttackOutcome.INCONCLUSIVE


def test_branch_uses_first_matching_declared_condition() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="branch",
            steps=[
                {
                    "id": "branch",
                    "type": "branch",
                    "branches": [
                        {
                            "condition": {
                                "path": "inputs.intent",
                                "operator": "eq",
                                "value": "test intent",
                            },
                            "next": "first",
                        },
                        {"next": "second"},
                    ],
                },
                {"id": "first", "type": "finish", "options": {"reason": "first"}},
                {"id": "second", "type": "finish", "options": {"reason": "second"}},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
        ).execute(request, context)
    )

    assert outcome.stopping_reason == "first"
    selected = next(event for event in context.events.events() if event.event_type == "branch.selected")
    assert selected.payload["branch_index"] == 0


def test_transform_cannot_replace_the_real_final_response() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, definition, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="target",
            steps=[
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "hello"}],
                        "output": "target_response",
                    },
                    "next": "transform",
                },
                {
                    "id": "transform",
                    "type": "transform",
                    "options": {
                        "source": "target_response.text",
                        "output": "poisoned_response",
                        "kind": "prefix",
                        "value": "poisoned::",
                    },
                    "next": "finish",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(final_response=definition.result_policy.final_response),
        ).execute(request, context)
    )

    assert outcome.outputs["final_response"] == "target response"
    transformed = next(event for event in context.events.events() if event.event_type == "value.transformed")
    assert transformed.payload["source_ref"] == "target_response.text"
    assert "poisoned::" not in repr(transformed.payload)


def test_transform_base64_encodes_exact_utf8_bytes_and_emits_provenance() -> None:
    from app.techniques.workflow.nodes import TransformHandler

    _, context, _, _ = workflow_execution_fixture()
    state = WorkflowState(
        current_step_id="transform",
        values={"inputs": {"intent": "café 🔒"}},
    )
    state.increment_visit("transform")
    step = StepDefinition(
        id="transform",
        type="transform",
        next_step_id="finish",
        options={
            "source": "inputs.intent",
            "output": "encoded_payload",
            "kind": "base64",
        },
    )

    result = asyncio.run(
        TransformHandler().execute(
            step=step,
            state=state,
            context=context,
        )
    )

    assert result.values["encoded_payload"] == {
        "value": "Y2Fmw6kg8J+Ukg==",
        "source_ref": "inputs.intent",
        "transform": "base64",
    }
    transformed = next(
        event
        for event in context.events.events()
        if event.event_type == "value.transformed"
    )
    assert transformed.operation_id == "workflow:transform:visit:1"
    assert dict(transformed.payload) == {
        "source_ref": "inputs.intent",
        "value_ref": "encoded_payload",
        "transform": "base64",
    }
    assert "café" not in repr(transformed.payload)


def test_transform_reverse_uses_legacy_unicode_codepoint_order() -> None:
    from app.techniques.workflow.nodes import TransformHandler

    _, context, _, _ = workflow_execution_fixture()
    state = WorkflowState(
        current_step_id="transform",
        values={"inputs": {"intent": "A👩‍💻e\u0301"}},
    )
    state.increment_visit("transform")
    step = StepDefinition(
        id="transform",
        type="transform",
        next_step_id="finish",
        options={
            "source": "inputs.intent",
            "output": "reversed_intent",
            "kind": "reverse",
        },
    )

    result = asyncio.run(
        TransformHandler().execute(
            step=step,
            state=state,
            context=context,
        )
    )

    assert result.values["reversed_intent"] == {
        "value": "\u0301e💻‍👩A",
        "source_ref": "inputs.intent",
        "transform": "reverse",
    }
    transformed = next(
        event
        for event in context.events.events()
        if event.event_type == "value.transformed"
    )
    assert dict(transformed.payload) == {
        "source_ref": "inputs.intent",
        "value_ref": "reversed_intent",
        "transform": "reverse",
    }
    assert "A👩" not in repr(transformed.payload)


def test_finish_rejects_a_transformed_final_response_surrogate() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, definition = multiple_judge_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="target",
            steps=[
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "hello"}],
                        "output": "target_response",
                    },
                    "next": "transform",
                },
                {
                    "id": "transform",
                    "type": "transform",
                    "options": {
                        "source_from": "values.target_response.text",
                        "output": "poisoned_response",
                        "kind": "prefix",
                        "value": "poisoned::",
                    },
                    "next": "judge",
                },
                {
                    "id": "judge",
                    "type": "judge",
                    "options": {
                        "definition_id": "target_judge",
                        "evidence": {
                            "intent_from": "inputs.intent",
                            "response_from": "values.target_response",
                        },
                        "output": "target_judge",
                    },
                    "next": "finish",
                },
                {
                    "id": "finish",
                    "type": "finish",
                    "options": {
                        "final_response_from": "values.poisoned_response",
                        "official_judge_from": "values.target_judge",
                    },
                },
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="final response evidence"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=definition.result_policy,
            ).execute(request, context)
        )


def test_finish_rejects_an_unrelated_auxiliary_judge_result() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, definition = multiple_judge_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="target",
            steps=[
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "hello"}],
                        "output": "target_response",
                    },
                    "next": "official",
                },
                {
                    "id": "official",
                    "type": "judge",
                    "options": {
                        "definition_id": "target_judge",
                        "evidence": {
                            "intent_from": "inputs.intent",
                            "response_from": "values.target_response",
                        },
                        "output": "target_judge",
                    },
                    "next": "diagnostic",
                },
                {
                    "id": "diagnostic",
                    "type": "judge",
                    "options": {
                        "definition_id": "diagnostic_judge",
                        "evidence": {
                            "intent_from": "inputs.intent",
                            "response_from": "values.target_response",
                        },
                        "output": "diagnostic_judge",
                    },
                    "next": "finish",
                },
                {
                    "id": "finish",
                    "type": "finish",
                    "options": {
                        "final_response_from": "values.target_response",
                        "official_judge_from": "values.diagnostic_judge",
                    },
                },
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="official judge evidence"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=definition.result_policy,
            ).execute(request, context)
        )


def test_finish_accepts_minimum_score_from_the_exact_official_judge() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, definition = multiple_judge_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="target",
            steps=[
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "hello"}],
                        "output": "target_response",
                    },
                    "next": "judge",
                },
                {
                    "id": "judge",
                    "type": "judge",
                    "options": {
                        "definition_id": "target_judge",
                        "evidence": {
                            "intent_from": "inputs.intent",
                            "response_from": "values.target_response",
                        },
                        "output": "target_judge",
                    },
                    "next": "finish",
                },
                {
                    "id": "finish",
                    "type": "finish",
                    "options": {
                        "final_response_from": "values.target_response",
                        "official_judge_from": "values.target_judge",
                    },
                },
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=definition.result_policy,
        ).execute(request, context)
    )

    assert outcome.outputs["official_score"] == 1
    assert outcome.attack_outcome is AttackOutcome.NOT_SUCCEEDED
    evidence = outcome.outputs["finish_evidence"]
    assert evidence["final_response_event_id"] == outcome.final_response_ref
    assert evidence["official_judge_result_id"] == outcome.official_judge_ref


def test_finish_rejects_failed_official_judge_evidence() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, definition = multiple_judge_execution_fixture(
        official_output="not-json",
    )
    workflow = load_workflow(
        workflow_payload(
            start="target",
            steps=[
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "hello"}],
                        "output": "target_response",
                    },
                    "next": "judge",
                },
                {
                    "id": "judge",
                    "type": "judge",
                    "options": {
                        "definition_id": "target_judge",
                        "evidence": {
                            "intent_from": "inputs.intent",
                            "response_from": "values.target_response",
                        },
                        "output": "target_judge",
                    },
                    "next": "finish",
                },
                {
                    "id": "finish",
                    "type": "finish",
                    "options": {
                        "final_response_from": "values.target_response",
                        "official_judge_from": "values.target_judge",
                    },
                },
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="official judge evidence"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=definition.result_policy,
            ).execute(request, context)
        )


def test_finish_rejects_missing_official_judge_evidence() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, definition = multiple_judge_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="target",
            steps=[
                {
                    "id": "target",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "hello"}],
                        "output": "target_response",
                    },
                    "next": "finish",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="official judge evidence"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=definition.result_policy,
            ).execute(request, context)
        )


def test_finish_rejects_a_target_response_not_selected_by_policy() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, _ = multiple_judge_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="first",
            steps=[
                {
                    "id": "first",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "first"}],
                        "output": "first_response",
                    },
                    "next": "second",
                },
                {
                    "id": "second",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "second"}],
                        "output": "second_response",
                    },
                    "next": "finish",
                },
                {
                    "id": "finish",
                    "type": "finish",
                    "options": {"final_response_from": "values.first_response"},
                },
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="final response evidence"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(final_response="target.real_response"),
            ).execute(request, context)
        )


def test_transform_values_namespace_cannot_overwrite_its_source_root() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="transform",
            steps=[
                {
                    "id": "transform",
                    "type": "transform",
                    "options": {
                        "source_from": "values.inputs.intent",
                        "output": "inputs",
                    },
                }
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="failed with ValueError"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )


@pytest.mark.parametrize(
    "delays",
    [
        {"branch-a": 0.02, "branch-b": 0},
        {"branch-a": 0, "branch-b": 0.02},
    ],
)
def test_parallel_results_and_events_are_keyed_by_declared_branch_id(
    delays: Mapping[str, float],
) -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult

    class DelayedBranch:
        async def execute(self, *, step, state, context) -> NodeResult:
            branch = state.values["branch"]
            branch_id = branch["id"]
            await asyncio.sleep(delays[branch_id])
            return NodeResult({"owner": branch_id})

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="parallel",
            max_step_visits=2,
            steps=[
                {
                    "id": "parallel",
                    "type": "parallel",
                    "options": {
                        "branches": [
                            {"id": "branch-b", "step": "work"},
                            {"id": "branch-a", "step": "work"},
                        ],
                        "output": "fanout",
                    },
                    "next": "finish",
                },
                {"id": "work", "type": "render"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
            handlers={"render": DelayedBranch()},
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    parallel_event = next(event for event in context.events.events() if event.event_type == "parallel.completed")
    branches = parallel_event.payload["branches"]
    assert [item["branch_id"] for item in branches] == ["branch-a", "branch-b"]
    assert [item["values"]["owner"] for item in branches] == ["branch-a", "branch-b"]
    assert parallel_event.payload["selected_branch_id"] == "branch-a"
    branch_events = [
        event
        for event in context.events.events()
        if event.event_type == "step.completed" and event.payload["step_id"] == "work"
    ]
    assert [event.branch_id for event in branch_events] == ["branch-a", "branch-b"]


def test_parallel_children_share_run_services_but_isolate_values_and_history_heads() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult

    request, context, definition, _ = workflow_execution_fixture()
    context.loaded_package.definition = replace(
        definition,
        histories=(HistoryDefinition("target"),),
    )
    context.histories.create("target")
    shared = context.histories.append(
        "target",
        role="user",
        content="shared",
        source_kind="fixture",
    )

    class BranchHistory:
        async def execute(self, *, step, state, context) -> NodeResult:
            branch_id = state.values["branch"]["id"]
            assert context.models is parent_context.models
            assert context.events is parent_context.events
            assert context.budget is parent_context.budget
            assert context.checkpoints is parent_context.checkpoints
            assert context.cancellation is parent_context.cancellation
            assert context.histories is not parent_context.histories
            context.histories.append(
                "target",
                role="assistant",
                content=branch_id,
                source_kind="branch",
            )
            return NodeResult(
                {
                    "owner": branch_id,
                    "message_ids": context.histories.snapshot("target").message_ids,
                }
            )

    parent_context = context
    workflow = load_workflow(
        workflow_payload(
            start="parallel",
            max_step_visits=1,
            steps=[
                {
                    "id": "parallel",
                    "type": "parallel",
                    "options": {
                        "branches": [
                            {"id": "branch-b", "step": "work"},
                            {"id": "branch-a", "step": "work"},
                        ]
                    },
                    "next": "finish",
                },
                {"id": "work", "type": "render"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
            handlers={"render": BranchHistory()},
        ).execute(request, context)
    )

    assert context.histories.snapshot("target").message_ids == (shared.message_id,)
    assert len(context.histories.all_message_ids()) == 3
    completed = next(event for event in context.events.events() if event.event_type == "parallel.completed")
    branches = completed.payload["branches"]
    assert [item["values"]["message_ids"][0] for item in branches] == [
        shared.message_id,
        shared.message_id,
    ]
    assert branches[0]["values"]["message_ids"][1] != branches[1]["values"]["message_ids"][1]


def test_parallel_upfront_reservation_failure_dispatches_zero_branches() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    original_request, original_context, definition, target = workflow_execution_fixture()
    request = replace(
        original_request,
        spec=replace(
            original_request.spec,
            effective_limits={
                **original_request.spec.effective_limits,
                "max_model_calls": 1,
                "max_tokens": 128,
                "max_parallel_branches": 2,
            },
        ),
    )
    client = WorkflowModelClient("first", "second")
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target_judge": client},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original_context.loaded_package,
    )
    workflow = load_workflow(
        workflow_payload(
            start="parallel",
            max_step_visits=2,
            steps=[
                {
                    "id": "parallel",
                    "type": "parallel",
                    "options": {
                        "branches": [
                            {"id": "branch-a", "step": "work"},
                            {"id": "branch-b", "step": "work"},
                        ]
                    },
                    "next": "finish",
                },
                {
                    "id": "work",
                    "type": "model_call",
                    "options": {
                        "role": "target_judge",
                        "messages": [{"role": "user", "content": "evaluate"}],
                        "call_options": {"max_tokens": 8},
                    },
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    with pytest.raises(BudgetExhausted, match="model_calls"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    assert client.calls == []


def test_parallel_reservation_uses_bound_client_output_ceiling() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    original_request, original_context, definition, target = workflow_execution_fixture()
    request = replace(
        original_request,
        spec=replace(
            original_request.spec,
            effective_limits={
                **original_request.spec.effective_limits,
                "max_model_calls": 2,
                    "max_tokens": 96,
            },
        ),
    )
    client = WorkflowModelClient("first", "second")
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target_judge": client},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original_context.loaded_package,
    )
    workflow = load_workflow(
        workflow_payload(
            start="parallel",
            max_step_visits=2,
            steps=[
                {
                    "id": "parallel",
                    "type": "parallel",
                    "options": {
                        "branches": [
                            {"id": "branch-a", "step": "work"},
                            {"id": "branch-b", "step": "work"},
                        ]
                    },
                    "next": "finish",
                },
                {
                    "id": "work",
                    "type": "model_call",
                    "options": {
                        "role": "target_judge",
                        "messages": [{"role": "user", "content": "evaluate"}],
                    },
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert len(client.calls) == 2
    assert all(call[2]["max_tokens"] == client.max_tokens for call in client.requests)


def test_backtrack_restores_checkpoint_heads_and_retains_exact_message_evidence() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, definition, _ = workflow_execution_fixture()
    context.loaded_package.definition = replace(
        definition,
        histories=(HistoryDefinition("target"),),
    )
    context.histories.create("target")
    active = context.histories.append(
        "target",
        role="user",
        content="keep",
        source_kind="fixture",
    )
    workflow = load_workflow(
        workflow_payload(
            start="checkpoint",
            max_step_visits=1,
            steps=[
                {
                    "id": "checkpoint",
                    "type": "checkpoint",
                    "options": {"output": "saved"},
                    "next": "append",
                },
                {
                    "id": "append",
                    "type": "history_append",
                    "options": {
                        "history": "target",
                        "role": "assistant",
                        "content": "abandon",
                        "output": "abandoned",
                    },
                    "next": "backtrack",
                },
                {
                    "id": "backtrack",
                    "type": "backtrack",
                    "options": {"checkpoint_from": "values.saved.checkpoint_id"},
                    "next": "finish",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert context.histories.snapshot("target").message_ids == (active.message_id,)
    appended = [
        event.payload["message_id"]
        for event in context.events.events()
        if event.event_type == "message.appended" and event.payload.get("content") == "abandon"
    ]
    assert len(appended) == 1
    assert appended[0] in context.histories.all_message_ids()
    events = context.events.events()
    backtrack = next(event for event in events if event.event_type == "backtrack.started")
    assert backtrack.payload["inactive_message_ids"] == {"target": (appended[0],)}
    rollback = next(event for event in events if event.event_type == "history.rolled_back")
    assert backtrack.sequence < rollback.sequence


def test_backtrack_replays_external_work_with_monotonic_unique_operation_ids() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult

    class CaptureFinish:
        def __init__(self) -> None:
            self.visit_counts: Mapping[str, int] = {}
            self.control_counts: Mapping[str, int] = {}

        async def execute(self, *, step, state, context) -> NodeResult:
            self.visit_counts = state.visit_counts
            self.control_counts = state.control_counts
            return NodeResult(finish_evidence={})

    request, original_context, definition, target = workflow_execution_fixture()
    persistence = NoopTechniqueRuntimePersistence()
    client = WorkflowModelClient("first", "second")
    context = ExecutionContextFactory(
        persistence=persistence,
        model_clients={"target_judge": client},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original_context.loaded_package,
    )
    finish = CaptureFinish()
    workflow = load_workflow(
        workflow_payload(
            start="checkpoint",
            max_step_visits=2,
            steps=[
                {
                    "id": "checkpoint",
                    "type": "checkpoint",
                    "options": {"output": "saved"},
                    "next": "model",
                },
                {
                    "id": "model",
                    "type": "model_call",
                    "options": {
                        "role": "target_judge",
                        "messages": [{"role": "user", "content": "repeat"}],
                        "output": "response",
                    },
                    "next": "decide",
                },
                {
                    "id": "decide",
                    "type": "branch",
                    "branches": [
                        {
                            "condition": {
                                "path": "response.text",
                                "operator": "eq",
                                "value": "first",
                            },
                            "next": "backtrack",
                        },
                        {"next": "finish"},
                    ],
                },
                {
                    "id": "backtrack",
                    "type": "backtrack",
                    "options": {
                        "checkpoint_from": "values.saved.checkpoint_id",
                        "max_backtracks": 1,
                    },
                    "next": "model",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
            handlers={"finish": finish},
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert [call.operation_id for call in client.calls] == [
        "workflow:model:visit:1",
        "workflow:model:visit:2",
    ]
    requested = [event for event in context.events.events() if event.event_type == "model.requested"]
    assert [event.attempt for event in requested] == [1, 1]
    assert finish.visit_counts["model"] == 2
    assert finish.visit_counts["decide"] == 2
    assert finish.visit_counts["backtrack"] == 1
    assert finish.control_counts["backtracks"] == 1
    assert len(persistence._model_calls) == 2


def test_workflow_state_checkpoint_reconciliation_is_atomic_and_monotonic() -> None:
    workflow = load_workflow(
        workflow_payload(
            start="checkpoint",
            max_step_visits=2,
            steps=[
                {"id": "checkpoint", "type": "checkpoint", "next": "loop"},
                {
                    "id": "loop",
                    "type": "loop",
                    "branches": [{"next": "finish"}],
                },
                {"id": "backtrack", "type": "backtrack", "next": "finish"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )
    state = WorkflowState(
        current_step_id="backtrack",
        values={"current": True},
        visit_counts={"checkpoint": 1, "loop": 2, "backtrack": 1},
        max_step_visits=2,
        control_counts={"rounds:loop": 1, "backtracks": 1},
    )

    state.reconcile_checkpoint(
        position={
            "step_id": "checkpoint",
            "next_step_id": "loop",
            "scope_id": None,
            "visit_counts": {"checkpoint": 1},
            "control_counts": {"rounds:loop": 2},
        },
        values={"checkpointed": True},
        workflow=workflow,
    )

    assert state.values == {"checkpointed": True}
    assert state.visit_counts == {"checkpoint": 1, "loop": 2, "backtrack": 1}
    assert state.control_counts == {"rounds:loop": 2, "backtracks": 1}
    assert state._visit_ledger.snapshot() == state.visit_counts
    assert state._control_ledger.snapshot() == state.control_counts
    with pytest.raises(RuntimeError, match="visit limit exceeded for step 'loop'"):
        state.increment_visit("loop")
    with pytest.raises(RuntimeError, match="rounds:loop"):
        state.increment_control("rounds:loop", maximum=2)
    with pytest.raises(RuntimeError, match="backtracks"):
        state.increment_control("backtracks", maximum=1)


@pytest.mark.parametrize(
    "position",
    [
        {
            "step_id": "checkpoint",
            "next_step_id": "finish",
            "scope_id": "other-branch",
            "visit_counts": {"checkpoint": 1},
            "control_counts": {},
        },
        {
            "step_id": "checkpoint",
            "next_step_id": "finish",
            "scope_id": None,
            "visit_counts": {"unknown": 1},
            "control_counts": {},
        },
        {
            "step_id": "checkpoint",
            "next_step_id": "finish",
            "scope_id": None,
            "visit_counts": {"checkpoint": 1},
            "control_counts": {"rounds:checkpoint": 1},
        },
        {
            "step_id": "checkpoint",
            "next_step_id": "finish",
            "scope_id": None,
            "visit_counts": {"checkpoint": 1},
            "control_counts": {"backtracks": True},
        },
    ],
)
def test_workflow_state_checkpoint_reconciliation_rejects_malformed_evidence_atomically(
    position: Mapping[str, object],
) -> None:
    workflow = load_workflow(
        workflow_payload(
            start="checkpoint",
            max_step_visits=3,
            steps=[
                {"id": "checkpoint", "type": "checkpoint", "next": "finish"},
                {"id": "backtrack", "type": "backtrack", "next": "finish"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )
    state = WorkflowState(
        current_step_id="backtrack",
        values={"before": "value"},
        visit_counts={"checkpoint": 1, "backtrack": 1},
        max_step_visits=3,
        control_counts={"backtracks": 1},
    )
    before = (
        state.current_step_id,
        state.values,
        state.visit_counts,
        state.control_counts,
    )

    with pytest.raises((TypeError, ValueError)):
        state.reconcile_checkpoint(
            position=position,
            values={"after": "value"},
            workflow=workflow,
        )

    assert (
        state.current_step_id,
        state.values,
        state.visit_counts,
        state.control_counts,
    ) == before


@pytest.mark.parametrize(
    ("ledger_name", "key", "message"),
    [
        ("_visit_ledger", "checkpoint", "private visit ledger"),
        ("_control_ledger", "backtracks", "private control ledger"),
    ],
)
def test_workflow_state_checkpoint_reconciliation_rejects_private_ledger_divergence(
    ledger_name: str,
    key: str,
    message: str,
) -> None:
    workflow = load_workflow(
        workflow_payload(
            start="checkpoint",
            max_step_visits=3,
            steps=[
                {"id": "checkpoint", "type": "checkpoint", "next": "finish"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )
    state = WorkflowState(
        current_step_id="checkpoint",
        values={"before": True},
        visit_counts={"checkpoint": 1},
        max_step_visits=3,
    )
    getattr(state, ledger_name)._counts[key] = 2

    with pytest.raises(ValueError, match=message):
        state.reconcile_checkpoint(
            position={
                "step_id": "checkpoint",
                "next_step_id": "finish",
                "scope_id": None,
                "visit_counts": {"checkpoint": 1},
                "control_counts": {},
            },
            values={"after": True},
            workflow=workflow,
        )

    assert state.values == {"before": True}
    assert state.visit_counts == {"checkpoint": 1}
    assert state.control_counts == {}


def test_workflow_state_checkpoint_reconciliation_rolls_back_on_history_restore_failure() -> None:
    class FailRestoreBatch(NoopTechniqueRuntimePersistence):
        def __init__(self) -> None:
            super().__init__()
            self.fail = False

        def append_events(self, events):
            if self.fail:
                raise RuntimeError("injected restore failure")
            super().append_events(events)

    request, original_context, definition, target = workflow_execution_fixture()
    persistence = FailRestoreBatch()
    context = ExecutionContextFactory(
        persistence=persistence,
        model_clients={"target_judge": WorkflowModelClient("unused")},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original_context.loaded_package,
    )
    context.histories.create("target")
    keep = context.histories.append("target", role="user", content="keep", source_kind="fixture")
    checkpoint_heads = context.histories.heads()
    abandon = context.histories.append("target", role="assistant", content="abandon", source_kind="fixture")
    expected_inactive = context.histories.validate_restore_heads(checkpoint_heads)
    workflow = load_workflow(
        workflow_payload(
            start="checkpoint",
            max_step_visits=3,
            steps=[
                {"id": "checkpoint", "type": "checkpoint", "next": "finish"},
                {"id": "backtrack", "type": "backtrack", "next": "finish"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )
    state = WorkflowState(
        current_step_id="backtrack",
        values={"before": True},
        visit_counts={"checkpoint": 1, "backtrack": 1},
        max_step_visits=3,
        control_counts={"backtracks": 1},
    )

    persistence.fail = True
    with pytest.raises(RuntimeError, match="restore failure"):
        state.reconcile_checkpoint(
            position={
                "step_id": "checkpoint",
                "next_step_id": "finish",
                "scope_id": None,
                "visit_counts": {"checkpoint": 1},
                "control_counts": {},
            },
            values={"after": True},
            workflow=workflow,
            histories=context.histories,
            history_heads=checkpoint_heads,
            checkpoint_id="checkpoint-atomic",
            expected_inactive=expected_inactive,
        )

    assert state.values == {"before": True}
    assert state.visit_counts == {"checkpoint": 1, "backtrack": 1}
    assert state.control_counts == {"backtracks": 1}
    assert context.histories.heads() == {"target": (keep.message_id, abandon.message_id)}


def test_workflow_state_checkpoint_reconciliation_rejects_history_head_drift_atomically() -> None:
    request, context, definition, _ = workflow_execution_fixture()
    context.histories.create("target")
    keep = context.histories.append("target", role="user", content="keep", source_kind="fixture")
    checkpoint_heads = context.histories.heads()
    first_inactive = context.histories.append("target", role="assistant", content="first", source_kind="fixture")
    expected_inactive = context.histories.validate_restore_heads(checkpoint_heads)
    drift = context.histories.append("target", role="assistant", content="drift", source_kind="fixture")
    workflow = load_workflow(
        workflow_payload(
            start="checkpoint",
            max_step_visits=3,
            steps=[
                {"id": "checkpoint", "type": "checkpoint", "next": "finish"},
                {"id": "backtrack", "type": "backtrack", "next": "finish"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )
    state = WorkflowState(
        current_step_id="backtrack",
        values={"before": True},
        visit_counts={"checkpoint": 1, "backtrack": 1},
        max_step_visits=3,
    )

    with pytest.raises(ValueError, match="changed after checkpoint validation"):
        state.reconcile_checkpoint(
            position={
                "step_id": "checkpoint",
                "next_step_id": "finish",
                "scope_id": None,
                "visit_counts": {"checkpoint": 1},
                "control_counts": {},
            },
            values={"after": True},
            workflow=workflow,
            histories=context.histories,
            history_heads=checkpoint_heads,
            checkpoint_id="checkpoint-drift",
            expected_inactive=expected_inactive,
        )

    assert state.values == {"before": True}
    assert context.histories.heads() == {"target": (keep.message_id, first_inactive.message_id, drift.message_id)}


def test_backtrack_rejects_a_checkpoint_owned_by_another_branch_scope() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, _, _ = workflow_execution_fixture()
    checkpoint = context.checkpoints.save(
        position={
            "step_id": "backtrack",
            "next_step_id": "finish",
            "visit_counts": {},
            "scope_id": "branch-a",
        },
        values={},
        history_heads={},
        budget_usage={},
    )
    request = replace(
        request,
        metadata={"checkpoint_id": checkpoint.checkpoint_id},
    )
    workflow = load_workflow(
        workflow_payload(
            start="backtrack",
            steps=[
                {
                    "id": "backtrack",
                    "type": "backtrack",
                    "options": {"checkpoint_from": "values.metadata.checkpoint_id"},
                    "next": "finish",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="failed with ValueError") as error:
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )
    assert str(error.value.__cause__) == "checkpoint scope does not belong to this workflow context"


def test_loop_round_limit_is_checked_before_external_body_work() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult

    class RoundBody:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, *, step, state, context) -> NodeResult:
            self.calls += 1
            return NodeResult(next_step_id="loop")

    request, context, _, _ = workflow_execution_fixture()
    body = RoundBody()
    workflow = load_workflow(
        workflow_payload(
            start="loop",
            max_step_visits=3,
            steps=[
                {
                    "id": "loop",
                    "type": "loop",
                    "options": {"max_rounds": 2},
                    "branches": [
                        {
                            "condition": {
                                "path": "inputs.intent",
                                "operator": "exists",
                            },
                            "next": "body",
                        },
                        {"next": "finish"},
                    ],
                },
                {"id": "body", "type": "render"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    with pytest.raises(RuntimeError, match="rounds:loop"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
                handlers={"render": body},
            ).execute(request, context)
        )

    assert body.calls == 2
    assert [event.round for event in context.events.events() if event.event_type == "round.started"] == [1, 2]


def test_population_candidates_preserve_provenance_without_new_cell_or_attempt_ids() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult

    class CandidateWork:
        async def execute(self, *, step, state, context) -> NodeResult:
            candidate = state.values["candidate"]
            return NodeResult({"rendered": f"mapped:{candidate['value']}"})

    request, context, _, _ = workflow_execution_fixture()
    workflow = load_workflow(
        workflow_payload(
            start="population",
            max_step_visits=4,
            steps=[
                {
                    "id": "population",
                    "type": "population_map",
                    "options": {
                        "generation": 3,
                        "candidates": [
                            {
                                "id": "candidate-b",
                                "value": "beta",
                                "provenance": {"parent_id": "seed-b"},
                            },
                            {
                                "id": "candidate-a",
                                "value": "alpha",
                                "provenance": {"parent_id": "seed-a"},
                            },
                        ],
                        "step": "work",
                        "output": "population",
                    },
                    "next": "finish",
                },
                {"id": "work", "type": "render"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
            handlers={"render": CandidateWork()},
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    generation = next(event for event in context.events.events() if event.event_type == "generation.completed")
    assert [item["candidate_id"] for item in generation.payload["candidates"]] == [
        "candidate-a",
        "candidate-b",
    ]
    assert generation.payload["candidates"][0]["provenance"] == {"parent_id": "seed-a"}
    assert {(event.correlation.cell_id, event.correlation.attempt_id) for event in context.events.events()} == {
        (request.correlation.cell_id, request.correlation.attempt_id)
    }
    assert len([event for event in context.events.events() if event.event_type == "run.started"]) == 1


def test_population_judges_reserve_each_bound_model_ceiling_upfront() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    original_request, original_context, definition, target = workflow_execution_fixture()
    request = replace(
        original_request,
        spec=replace(
            original_request.spec,
            effective_limits={
                **original_request.spec.effective_limits,
                "max_judge_calls": 2,
                "max_model_calls": 2,
                "max_tokens": 6_000,
            },
        ),
    )
    client = WorkflowModelClient(
        '{"score": 1, "rationale": "a"}',
        '{"score": 2, "rationale": "b"}',
    )
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target_judge": client},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original_context.loaded_package,
    )
    workflow = load_workflow(
        workflow_payload(
            start="population",
            max_step_visits=1,
            steps=[
                {
                    "id": "population",
                    "type": "population_map",
                    "options": {
                        "candidates": [
                            {"id": "candidate-a", "value": "a"},
                            {"id": "candidate-b", "value": "b"},
                        ],
                        "step": "judge",
                    },
                    "next": "finish",
                },
                {
                    "id": "judge",
                    "type": "judge",
                    "options": {
                        "definition": "target_judge",
                        "evidence": {
                            "intent_from": "inputs.intent",
                            "response_from": "values.candidate.value",
                        },
                    },
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert len(client.calls) == 2
    assert context.budget.usage.judge_calls == 2


def test_candidate_selection_ranks_only_successful_judges_with_stable_zero_tie() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, _, _ = workflow_execution_fixture()
    successful_events = {}
    for candidate_id in ("candidate-b", "candidate-a"):
        result_id = f"result-{candidate_id}"
        successful_events[candidate_id] = context.events.emit(
            "judge.completed",
            branch_id=candidate_id,
            payload={
                "definition_id": "target_judge",
                "result_id": result_id,
                "status": "succeeded",
                "score": 0,
            },
        )
    failed_event = context.events.emit(
        "judge.failed",
        branch_id="candidate-failed",
        payload={
            "definition_id": "target_judge",
            "result_id": "result-failed",
            "status": "failed",
            "score": None,
        },
    )
    candidates = []
    for candidate_id in ("candidate-b", "candidate-a"):
        event = successful_events[candidate_id]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "status": "succeeded",
                "provenance": {"parent_id": "seed"},
                "values": {
                    "judge": {
                        "definition_id": "target_judge",
                        "result_id": event.payload["result_id"],
                        "event_id": event.event_id,
                        "status": "succeeded",
                        "score": 0,
                    }
                },
            }
        )
    candidates.extend(
        [
            {
                "candidate_id": "candidate-failed",
                "status": "succeeded",
                "provenance": {},
                "values": {
                    "judge": {
                        "definition_id": "target_judge",
                        "result_id": "result-failed",
                        "event_id": failed_event.event_id,
                        "status": "failed",
                        "score": None,
                    }
                },
            },
            {
                "candidate_id": "candidate-missing",
                "status": "succeeded",
                "provenance": {},
                "values": {},
            },
            {
                "candidate_id": "candidate-work-failed",
                "status": "failed",
                "provenance": {},
                "values": {},
            },
        ]
    )
    request = replace(
        request,
        metadata={"population": {"generation": 1, "candidates": candidates}},
    )
    workflow = load_workflow(
        workflow_payload(
            start="select",
            steps=[
                {
                    "id": "select",
                    "type": "candidate_select",
                    "options": {
                        "candidates_from": "values.metadata.population",
                        "judge_from": "judge",
                        "judge_definition": "target_judge",
                        "output": "selection",
                    },
                    "next": "finish",
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    selected = next(event for event in context.events.events() if event.event_type == "candidate.selected")
    assert selected.payload["candidate_id"] == "candidate-a"
    assert selected.payload["score"] == 0
    selection = next(event for event in context.events.events() if event.event_type == "candidate.selection_completed")
    assert {item["candidate_id"]: item["rank_status"] for item in selection.payload["ranking"]} == {
        "candidate-a": "eligible",
        "candidate-b": "eligible",
        "candidate-failed": "judge_failed",
        "candidate-missing": "missing_judge",
        "candidate-work-failed": "candidate_failed",
    }


def test_parallel_failure_cancels_and_joins_sibling_tasks() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult, WorkflowNodeError

    class FailingBranches:
        def __init__(self) -> None:
            self.sibling_started = asyncio.Event()
            self.sibling_cleaned = False

        async def execute(self, *, step, state, context) -> NodeResult:
            branch_id = state.values["branch"]["id"]
            if branch_id == "branch-a":
                await self.sibling_started.wait()
                raise ValueError("declared branch failure")
            self.sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.sibling_cleaned = True

    request, context, _, _ = workflow_execution_fixture()
    handler = FailingBranches()
    workflow = load_workflow(
        workflow_payload(
            start="parallel",
            max_step_visits=2,
            steps=[
                {
                    "id": "parallel",
                    "type": "parallel",
                    "options": {
                        "branches": [
                            {"id": "branch-a", "step": "work"},
                            {"id": "branch-b", "step": "work"},
                        ],
                        "failure_policy": "fail_fast",
                    },
                },
                {"id": "work", "type": "render"},
            ],
        )
    )

    async def execute() -> None:
        with pytest.raises(WorkflowNodeError, match="failed with ValueError"):
            await WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
                handlers={"render": handler},
            ).execute(request, context)
        assert [task for task in asyncio.all_tasks() if task is not asyncio.current_task()] == []

    asyncio.run(execute())

    assert handler.sibling_cleaned is True
    assert {event.branch_id for event in context.events.events() if event.event_type == "branch.failed"} == {
        "branch-a",
        "branch-b",
    }


@pytest.mark.parametrize("node_type", ["parallel", "population_map"])
def test_bounded_fanout_lifecycle_has_cleanup_merge_and_reservation_parity(
    node_type: str,
) -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult, WorkflowNodeError

    class FanoutWork:
        def __init__(self) -> None:
            self.started: set[str] = set()
            self.cancelled: set[str] = set()

        async def execute(self, *, step, state, context) -> NodeResult:
            namespace = "branch" if node_type == "parallel" else "candidate"
            item_id = state.values[namespace]["id"]
            assert type(item_id) is str
            self.started.add(item_id)
            context.budget.reserve(model_calls=1)
            if item_id == "item-a":
                await asyncio.sleep(0)
                raise WorkflowNodeError("fixture", "declared fan-out failure")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.add(item_id)
                raise

    request, context, _, _ = workflow_execution_fixture()
    worker = FanoutWork()
    if node_type == "parallel":
        options = {
            "branches": [
                {"id": "item-b", "step": "work"},
                {"id": "item-a", "step": "work"},
            ],
            "per_branch_budget": {"model_calls": 1},
        }
        started_type = "branch.started"
        failed_type = "branch.failed"
    else:
        options = {
            "candidates": [
                {"id": "item-b", "value": "slow"},
                {"id": "item-a", "value": "fail"},
            ],
            "step": "work",
            "per_candidate_budget": {"model_calls": 1},
        }
        started_type = "candidate.created"
        failed_type = "candidate.failed"
    workflow = load_workflow(
        workflow_payload(
            start="fanout",
            max_step_visits=2,
            steps=[
                {
                    "id": "fanout",
                    "type": node_type,
                    "options": options,
                    "next": "finish",
                },
                {"id": "work", "type": "render"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="declared fan-out failure"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
                handlers={"render": worker},
            ).execute(request, context)
        )

    assert worker.started == {"item-a", "item-b"}
    assert worker.cancelled == {"item-b"}
    assert context.budget.usage.model_calls == 2
    events = context.events.events()
    assert [event.branch_id for event in events if event.event_type == started_type] == ["item-a", "item-b"]
    assert {event.branch_id for event in events if event.event_type == failed_type} == {"item-a", "item-b"}
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))


def test_nested_parallel_reservations_compose_without_double_charging() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    original_request, original_context, definition, target = workflow_execution_fixture()
    request = replace(
        original_request,
        spec=replace(
            original_request.spec,
            effective_limits={
                **original_request.spec.effective_limits,
                "max_model_calls": 2,
                "max_tokens": 128,
                "max_parallel_branches": 3,
            },
        ),
    )
    client = WorkflowModelClient("first", "second")
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target_judge": client},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original_context.loaded_package,
    )
    workflow = load_workflow(
        workflow_payload(
            start="outer",
            max_step_visits=4,
            steps=[
                {
                    "id": "outer",
                    "type": "parallel",
                    "options": {
                        "branches": [{"id": "outer-a", "step": "inner"}],
                    },
                    "next": "finish",
                },
                {
                    "id": "inner",
                    "type": "parallel",
                    "options": {
                        "branches": [
                            {"id": "inner-b", "step": "work"},
                            {"id": "inner-a", "step": "work"},
                        ],
                    },
                },
                {
                    "id": "work",
                    "type": "model_call",
                    "options": {
                        "role": "target_judge",
                        "messages": [{"role": "user", "content": "evaluate"}],
                        "call_options": {"max_tokens": 8},
                    },
                },
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
        ).execute(request, context)
    )

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert len(client.calls) == 2
    assert context.budget.usage.model_calls == 2
    completed = [event for event in context.events.events() if event.event_type == "model.completed"]
    assert [event.branch_id for event in completed] == ["inner-a", "inner-b"]
    assert len({event.operation_id for event in completed}) == 2


@pytest.mark.parametrize("invalid_limit", [True, -1, 1.0, "2"])
def test_parallel_rejects_non_exact_limits_before_branch_work(
    invalid_limit: object,
) -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult, WorkflowNodeError

    class Work:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, *, step, state, context) -> NodeResult:
            self.calls += 1
            return NodeResult()

    request, context, _, _ = workflow_execution_fixture()
    work = Work()
    workflow = load_workflow(
        workflow_payload(
            start="parallel",
            max_step_visits=2,
            steps=[
                {
                    "id": "parallel",
                    "type": "parallel",
                    "options": {
                        "max_branches": invalid_limit,
                        "branches": [{"id": "branch-a", "step": "work"}],
                    },
                },
                {"id": "work", "type": "render"},
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="failed with ValueError"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
                handlers={"render": work},
            ).execute(request, context)
        )

    assert work.calls == 0


def test_population_limit_rejects_off_by_one_before_candidate_work() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult, WorkflowNodeError

    class Work:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, *, step, state, context) -> NodeResult:
            self.calls += 1
            return NodeResult()

    request, context, _, _ = workflow_execution_fixture()
    context = replace(
        context,
        spec=replace(
            context.spec,
            effective_limits={
                **context.spec.effective_limits,
                "max_population_size": 1,
            },
        ),
    )
    work = Work()
    workflow = load_workflow(
        workflow_payload(
            start="population",
            max_step_visits=2,
            steps=[
                {
                    "id": "population",
                    "type": "population_map",
                    "options": {
                        "candidates": [
                            {"id": "candidate-a", "value": "a"},
                            {"id": "candidate-b", "value": "b"},
                        ],
                        "step": "work",
                    },
                },
                {"id": "work", "type": "render"},
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="population size 2 exceeds limit 1"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
                handlers={"render": work},
            ).execute(request, context)
        )

    assert work.calls == 0


def test_candidate_selection_cannot_bypass_the_pinned_population_limit() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, context, _, _ = workflow_execution_fixture()
    context = replace(
        context,
        spec=replace(
            context.spec,
            effective_limits={
                **context.spec.effective_limits,
                "max_population_size": 1,
            },
        ),
    )
    request = replace(
        request,
        metadata={
            "population": {
                "generation": 1,
                "candidates": [
                    {"candidate_id": "candidate-a", "status": "failed", "values": {}},
                    {"candidate_id": "candidate-b", "status": "failed", "values": {}},
                ],
            }
        },
    )
    workflow = load_workflow(
        workflow_payload(
            start="select",
            steps=[
                {
                    "id": "select",
                    "type": "candidate_select",
                    "options": {
                        "candidates_from": "values.metadata.population",
                        "judge_from": "judge",
                        "judge_definition": "target_judge",
                    },
                },
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="population size 2 exceeds limit 1"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )


def test_retry_replays_only_declared_bounded_work() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult, WorkflowNodeError

    class FlakyRender:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, *, step, state, context) -> NodeResult:
            self.calls += 1
            if self.calls < 3:
                raise WorkflowNodeError("render", "declared transient failure")
            return NodeResult({"retried_value": "ok"})

    request, context, _, _ = workflow_execution_fixture()
    handler = FlakyRender()
    workflow = load_workflow(
        workflow_payload(
            start="retry",
            max_step_visits=3,
            steps=[
                {
                    "id": "retry",
                    "type": "retry",
                    "options": {
                        "step": "work",
                        "max_attempts": 3,
                        "retry_on": ["render"],
                    },
                    "next": "finish",
                },
                {"id": "work", "type": "render"},
                {"id": "finish", "type": "finish"},
            ],
        )
    )

    outcome = asyncio.run(
        WorkflowExecutor(
            workflow=workflow,
            result_policy=ResultPolicy(),
            handlers={"render": handler},
        ).execute(request, context)
    )

    assert handler.calls == 3
    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert [
        event.attempt
        for event in context.events.events()
        if event.event_type == "step.started" and event.payload.get("step_id") == "work"
    ] == [1, 2, 3]
    assert [
        event.operation_id
        for event in context.events.events()
        if event.event_type == "step.started" and event.payload.get("step_id") == "work"
    ] == [
        "workflow:work:visit:1",
        "workflow:work:visit:2",
        "workflow:work:visit:3",
    ]


def test_retry_does_not_reissue_indeterminate_work_without_pinned_permission() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult, WorkflowNodeError

    class IndeterminateRender:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, *, step, state, context) -> NodeResult:
            self.calls += 1
            raise WorkflowNodeError(
                "render",
                "completion ownership is unknown",
                indeterminate=True,
            )

    request, context, _, _ = workflow_execution_fixture()
    handler = IndeterminateRender()
    workflow = load_workflow(
        workflow_payload(
            start="retry",
            max_step_visits=3,
            steps=[
                {
                    "id": "retry",
                    "type": "retry",
                    "options": {
                        "step": "work",
                        "max_attempts": 3,
                        "retry_on": ["render"],
                    },
                },
                {"id": "work", "type": "render"},
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="completion ownership is unknown"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
                handlers={"render": handler},
            ).execute(request, context)
        )

    assert handler.calls == 1


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError, RuntimeError])
@pytest.mark.parametrize(
    ("retry_indeterminate", "expected_dispatches"),
    [(False, 1), (True, 3)],
)
def test_model_retry_requires_pinned_permission_for_unknown_completion(
    error_type: type[Exception],
    retry_indeterminate: bool,
    expected_dispatches: int,
) -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    original_request, original_context, definition, target = workflow_execution_fixture()
    request = replace(
        original_request,
        spec=replace(
            original_request.spec,
            effective_limits={
                **original_request.spec.effective_limits,
                "max_model_calls": 3,
                "max_operation_attempts": 1,
                "retry_indeterminate": retry_indeterminate,
            },
        ),
    )
    client = RaisingWorkflowModelClient(error_type)
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target_judge": client},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original_context.loaded_package,
    )
    workflow = load_workflow(
        workflow_payload(
            start="retry",
            max_step_visits=3,
            steps=[
                {
                    "id": "retry",
                    "type": "retry",
                    "options": {
                        "step": "work",
                        "max_attempts": 3,
                        "retry_on": ["model_call"],
                    },
                },
                {
                    "id": "work",
                    "type": "model_call",
                    "options": {
                        "role": "target_judge",
                        "messages": [{"role": "user", "content": "evaluate"}],
                    },
                },
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match=f"failed with {error_type.__name__}"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    assert len(client.requests) == expected_dispatches
    failed = [event for event in context.events.events() if event.event_type == "model.failed"]
    assert len(failed) == expected_dispatches
    assert all(event.payload["indeterminate"] is True for event in failed)


def test_model_retry_keeps_known_preflight_failure_determinate_without_dispatch() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    request, original_context, definition, target = workflow_execution_fixture()
    client = WorkflowModelClient("must not dispatch")
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target_judge": client},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original_context.loaded_package,
    )
    workflow = load_workflow(
        workflow_payload(
            start="retry",
            max_step_visits=2,
            steps=[
                {
                    "id": "retry",
                    "type": "retry",
                    "options": {
                        "step": "work",
                        "max_attempts": 2,
                        "retry_on": ["model_call"],
                    },
                },
                {
                    "id": "work",
                    "type": "model_call",
                    "options": {
                        "role": "undeclared_role",
                        "messages": [{"role": "user", "content": "evaluate"}],
                    },
                },
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="failed with KeyError"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    assert client.requests == []
    assert not any(event.event_type == "model.requested" for event in context.events.events())
    assert [
        event.attempt
        for event in context.events.events()
        if event.event_type == "step.started" and event.payload.get("step_id") == "work"
    ] == [1, 2]


@pytest.mark.parametrize(
    "node_type",
    ["convert", "detector", "judge", "seed_resolve", "tool_call"],
)
def test_retry_treats_unknown_service_failures_as_indeterminate(node_type: str) -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult, WorkflowNodeError

    class UnknownCompletionHandler:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, *, step, state, context) -> NodeResult:
            self.calls += 1
            raise TimeoutError("service completion ownership is unknown")

    request, context, _, _ = workflow_execution_fixture()
    handler = UnknownCompletionHandler()
    workflow = load_workflow(
        workflow_payload(
            start="retry",
            max_step_visits=2,
            steps=[
                {
                    "id": "retry",
                    "type": "retry",
                    "options": {
                        "step": "work",
                        "max_attempts": 2,
                        "retry_on": [node_type],
                    },
                },
                {"id": "work", "type": node_type},
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="failed with TimeoutError"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
                handlers={node_type: handler},
            ).execute(request, context)
        )

    assert handler.calls == 1


def test_retry_preflights_target_history_before_any_external_dispatch() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    original_request, _, definition, target = workflow_execution_fixture()
    request = replace(
        original_request,
        spec=replace(
            original_request.spec,
            effective_limits={
                **original_request.spec.effective_limits,
                "max_target_calls": 2,
                "retry_indeterminate": True,
            },
        ),
    )
    package = SimpleNamespace(
        definition=definition,
        implementation=TechniqueImplementation(
            id="modular",
            technique_id="probe",
            technique_version="1.0.0",
            kind=ImplementationKind.WORKFLOW,
            entrypoint="workflow.yaml",
        ),
        manifest_hash=request.spec.manifest_hash,
        asset_hashes=request.spec.asset_hashes,
    )
    context = ExecutionContextFactory(
        persistence=NoopTechniqueRuntimePersistence(),
        model_clients={"target_judge": WorkflowModelClient("unused")},
        target_adapter=target,
    ).create(request=request, definition=definition, loaded_package=package)
    workflow = load_workflow(
        workflow_payload(
            start="retry",
            steps=[
                {
                    "id": "retry",
                    "type": "retry",
                    "options": {
                        "step": "work",
                        "max_attempts": 2,
                        "retry_on": ["target_call"],
                    },
                },
                {
                    "id": "work",
                    "type": "target_call",
                    "options": {
                        "messages": [{"role": "user", "content": "request"}],
                        "append_to_history": "undeclared_history",
                    },
                },
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="is not declared by the pinned technique"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    assert target.requests == []


def test_retry_never_replays_a_checkpoint_that_persisted_before_timeout() -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import WorkflowNodeError

    class SaveThenTimeoutPersistence(NoopTechniqueRuntimePersistence):
        def __init__(self) -> None:
            super().__init__()
            self.saved_checkpoint_ids: list[str] = []

        def save_checkpoint(self, checkpoint) -> None:
            super().save_checkpoint(checkpoint)
            self.saved_checkpoint_ids.append(checkpoint.checkpoint_id)
            raise TimeoutError("storage acknowledgement lost")

    request, original, definition, target = workflow_execution_fixture()
    persistence = SaveThenTimeoutPersistence()
    context = ExecutionContextFactory(
        persistence=persistence,
        model_clients={"target_judge": WorkflowModelClient("unused")},
        target_adapter=target,
    ).create(
        request=request,
        definition=definition,
        loaded_package=original.loaded_package,
    )
    workflow = load_workflow(
        workflow_payload(
            start="checkpoint",
            max_step_visits=3,
            steps=[
                {
                    "id": "checkpoint",
                    "type": "checkpoint",
                    "options": {"error_types": ["checkpoint"]},
                    "error": "retry",
                },
                {
                    "id": "retry",
                    "type": "retry",
                    "options": {
                        "step": "checkpoint",
                        "max_attempts": 3,
                        "retry_on": ["checkpoint"],
                    },
                },
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="failed with ValueError"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
            ).execute(request, context)
        )

    assert len(persistence.saved_checkpoint_ids) == 1


@pytest.mark.parametrize(
    "node_type",
    [
        "artifact_create",
        "checkpoint",
        "history_append",
        "history_create",
        "history_snapshot",
        "transform",
    ],
)
def test_retry_rejects_state_mutating_targets_before_dispatch(node_type: str) -> None:
    from app.techniques.workflow.executor import WorkflowExecutor
    from app.techniques.workflow.nodes import NodeResult, WorkflowNodeError

    class SideEffectingHandler:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, *, step, state, context) -> NodeResult:
            self.calls += 1
            return NodeResult()

    request, context, _, _ = workflow_execution_fixture()
    handler = SideEffectingHandler()
    workflow = load_workflow(
        workflow_payload(
            start="retry",
            steps=[
                {
                    "id": "retry",
                    "type": "retry",
                    "options": {
                        "step": "work",
                        "max_attempts": 2,
                        "retry_on": [node_type],
                    },
                },
                {"id": "work", "type": node_type},
            ],
        )
    )

    with pytest.raises(WorkflowNodeError, match="failed with ValueError"):
        asyncio.run(
            WorkflowExecutor(
                workflow=workflow,
                result_policy=ResultPolicy(),
                handlers={node_type: handler},
            ).execute(request, context)
        )

    assert handler.calls == 0


def test_workflow_route_rejects_a_registration_without_a_loaded_package() -> None:
    from app.techniques.core.catalog import RegisteredTechnique, TechniqueCatalog
    from app.techniques.core.router import ExecutionRouter

    request, _, definition, _ = workflow_execution_fixture()
    catalog = TechniqueCatalog()
    catalog.register(
        RegisteredTechnique(
            definition=definition,
            implementation=TechniqueImplementation(
                id="modular",
                technique_id="probe",
                technique_version="1.0.0",
                kind=ImplementationKind.WORKFLOW,
                entrypoint="workflow.yaml",
            ),
            manifest_hash=request.spec.manifest_hash,
            asset_hashes=request.spec.asset_hashes,
        )
    )

    with pytest.raises(ValueError, match="loaded package"):
        ExecutionRouter(catalog).resolve(
            "probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )


def test_execution_route_rejects_a_caller_injected_workflow() -> None:
    from app.techniques.core.catalog import RegisteredTechnique
    from app.techniques.core.router import ExecutionRoute

    request, _, definition, _ = workflow_execution_fixture()
    registered = RegisteredTechnique(
        definition=definition,
        implementation=TechniqueImplementation(
            id="modular",
            technique_id="probe",
            technique_version="1.0.0",
            kind=ImplementationKind.WORKFLOW,
            entrypoint="workflow.yaml",
        ),
        manifest_hash=request.spec.manifest_hash,
        asset_hashes=request.spec.asset_hashes,
    )
    injected = load_workflow(workflow_payload(start="finish", steps=[{"id": "finish", "type": "finish"}]))

    with pytest.raises(TypeError, match="workflow"):
        ExecutionRoute(registered=registered, workflow=injected)


def test_workflow_router_cannot_fall_back_to_a_preconfigured_executor() -> None:
    from app.techniques.core.catalog import RegisteredTechnique, TechniqueCatalog
    from app.techniques.core.router import ExecutionRouter
    from app.techniques.workflow.executor import WorkflowExecutor

    request, context, definition, _ = workflow_execution_fixture()
    catalog = TechniqueCatalog()
    catalog.register(
        RegisteredTechnique(
            definition=definition,
            implementation=TechniqueImplementation(
                id="modular",
                technique_id="probe",
                technique_version="1.0.0",
                kind=ImplementationKind.WORKFLOW,
                entrypoint="workflow.yaml",
            ),
            manifest_hash=request.spec.manifest_hash,
            asset_hashes=request.spec.asset_hashes,
        )
    )
    fallback = load_workflow(workflow_payload(start="finish", steps=[{"id": "finish", "type": "finish"}]))

    with pytest.raises(ValueError, match="loaded package"):
        route = ExecutionRouter(catalog).resolve(
            "probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )
        asyncio.run(
            ExecutionRouter(catalog).execute(
                route,
                request,
                context,
                executors={
                    ImplementationKind.WORKFLOW: WorkflowExecutor(
                        workflow=fallback,
                        result_policy=ResultPolicy(),
                    )
                },
            )
        )


def test_workflow_router_rejects_an_executor_that_overrides_verified_binding(
    tmp_path: Path,
) -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    injected = load_workflow(
        workflow_payload(
            start="finish",
            steps=[{"id": "finish", "type": "finish"}],
        )
    )

    class OverridingVerifiedExecutor(WorkflowExecutor):
        async def _execute_verified(self, request, context, binding):
            return await self._execute_selected(
                request,
                context,
                workflow=injected,
                result_policy=ResultPolicy(),
                template_assets=None,
            )

    router, request, context, target, _, _ = packaged_workflow_execution_fixture(tmp_path)
    route = router.resolve(
        "probe",
        technique_version="1.0.0",
        implementation_id="modular",
    )

    with pytest.raises(TypeError, match="exact WorkflowExecutor"):
        asyncio.run(
            router.execute(
                route,
                request,
                context,
                executors={ImplementationKind.WORKFLOW: OverridingVerifiedExecutor()},
            )
        )

    assert target.requests == []


def test_router_resolves_only_the_exact_pinned_workflow_asset(tmp_path: Path) -> None:
    from app.techniques.core.catalog import (
        RegisteredTechnique,
        TechniqueCatalog,
    )
    from app.techniques.core.router import ExecutionRouter
    from app.techniques.plugins.manifest import load_technique_package

    source = Path(__file__).parent / "fixtures" / "techniques" / "valid_probe"
    package_root = tmp_path / "valid_probe"
    shutil.copytree(source, package_root)
    package = load_technique_package(
        package_root,
        supported_capabilities=frozenset(TechniqueCapability),
    )
    catalog = TechniqueCatalog()
    catalog.register(
        RegisteredTechnique(
            definition=package.definition,
            implementation=package.implementation,
            manifest_hash=package.manifest_hash,
            asset_hashes=package.asset_hashes,
            package=package,
        )
    )
    router = ExecutionRouter(catalog)

    route = router.resolve(
        "probe",
        technique_version="1.0.0",
        implementation_id="modular",
    )

    assert route.workflow is not None
    assert route.workflow.start == "finish"

    (package_root / "workflow.yaml").write_text(
        "version: 1\nstart: changed\nsteps: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash does not match"):
        router.resolve(
            "probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )


def test_router_rejects_tampered_package_template_content(tmp_path: Path) -> None:
    router, _, _, _, _, package_root = packaged_workflow_execution_fixture(tmp_path)
    (package_root / "prompt.txt").write_text("tampered {intent}", encoding="utf-8")

    with pytest.raises(ValueError, match="template asset hash"):
        router.resolve(
            "probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )


def test_router_rejects_an_unpinned_swapped_template_asset_id(tmp_path: Path) -> None:
    workflow = b"""version: 1
start: render
steps:
  - id: render
    type: render
    options:
      template_asset: substituted.txt
    next: finish
  - id: finish
    type: finish
"""
    router, _, _, _, _, _ = packaged_workflow_execution_fixture(
        tmp_path,
        workflow_bytes=workflow,
    )

    with pytest.raises(ValueError, match="pinned package template"):
        router.resolve(
            "probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )


def test_router_rejects_a_package_template_with_invalid_utf8(tmp_path: Path) -> None:
    router, _, _, _, _, _ = packaged_workflow_execution_fixture(
        tmp_path,
        prompt_bytes=b"\xff\xfe",
    )

    with pytest.raises(ValueError, match="UTF-8"):
        router.resolve(
            "probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )


def test_router_uses_one_immutable_package_asset_snapshot(tmp_path: Path) -> None:
    from app.techniques.workflow.executor import WorkflowExecutor

    router, request, context, target, _, package_root = packaged_workflow_execution_fixture(tmp_path)
    route = router.resolve(
        "probe",
        technique_version="1.0.0",
        implementation_id="modular",
    )
    (package_root / "prompt.txt").write_text("mutated after resolve: {intent}", encoding="utf-8")

    outcome = asyncio.run(
        router.execute(
            route,
            request,
            context,
            executors={
                ImplementationKind.WORKFLOW: WorkflowExecutor(
                    workflow=load_workflow(
                        workflow_payload(
                            start="finish",
                            steps=[{"id": "finish", "type": "finish"}],
                        )
                    ),
                    result_policy=ResultPolicy(),
                )
            },
        )
    )

    assert outcome.outputs["final_response"] == "package response"
    assert target.requests[0].messages == ({"role": "user", "content": "Package intent: trusted"},)


def test_router_rejects_a_symlinked_package_asset_escape(tmp_path: Path) -> None:
    router, _, _, _, _, package_root = packaged_workflow_execution_fixture(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("Package intent: {intent}", encoding="utf-8")
    (package_root / "prompt.txt").unlink()
    (package_root / "prompt.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="confined package asset"):
        router.resolve(
            "probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )


def test_router_fails_closed_without_secure_descriptor_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.techniques.core.router as router_module

    router, _, _, _, _, _ = packaged_workflow_execution_fixture(tmp_path)
    monkeypatch.delattr(router_module.os, "O_NOFOLLOW")

    with pytest.raises(ValueError, match="secure confined package reads"):
        router.resolve(
            "probe",
            technique_version="1.0.0",
            implementation_id="modular",
        )


@pytest.mark.parametrize("field", ["options", "config"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("source", ["mapping", "yaml"])
def test_workflow_rejects_explicit_null_nested_options(
    tmp_path: Path,
    field: str,
    flat: bool,
    source: str,
) -> None:
    import yaml

    step = {"id": "finish", "type": "finish", field: None}
    if flat:
        step["final_response_from"] = "target.real_response"
    payload = workflow_payload(start="finish", steps=[step])
    document = payload
    if source == "yaml":
        document = tmp_path / "workflow.yaml"
        document.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(WorkflowValidationError) as caught:
        load_workflow(document)

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.stage == "schema"
    assert diagnostic.path == (("steps", 0) if flat else ("steps", 0, field))
    assert diagnostic.message == (
        "a step may declare nested or flat options, not both"
        if flat else "step options must be a mapping"
    )


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, {}),
        ({"options": {}}, {}),
        ({"config": {}}, {}),
        ({"final_response_from": "target.real_response"}, {"final_response_from": "target.real_response"}),
    ],
)
def test_workflow_preserves_empty_and_flat_options(options, expected) -> None:
    workflow = load_workflow(
        workflow_payload(start="finish", steps=[{"id": "finish", "type": "finish", **options}])
    )

    assert workflow.steps["finish"].options == expected


def test_workflow_rejects_missing_next_step() -> None:
    with pytest.raises(WorkflowValidationError, match="unknown next step 'missing'"):
        load_workflow(
            {
                "version": 1,
                "start": "render",
                "steps": [{"id": "render", "type": "render", "next": "missing"}],
            }
        )


def test_structured_condition_reads_values_without_eval() -> None:
    evaluator = ConditionEvaluator()
    condition = Condition(path="judges.target_judge.score", operator="gte", value=7)

    assert evaluator.evaluate(condition, {"judges": {"target_judge": {"score": 8}}})


def test_condition_canonical_mapping_omits_internal_missing_value() -> None:
    condition = Condition(path="inputs.seed_code", operator="truthy")

    assert condition.canonical_mapping() == {
        "path": "inputs.seed_code",
        "operator": "truthy",
    }


def test_condition_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError, match="unsupported condition operator"):
        ConditionEvaluator().evaluate(Condition(path="value", operator="exec", value="code"), {"value": 1})


def test_workflow_copies_nested_options_and_remains_immutable() -> None:
    options = {"template": {"parts": ["initial"]}}
    workflow = load_workflow(
        workflow_payload(
            steps=[
                {
                    "id": "render",
                    "type": "render",
                    "options": options,
                    "next": "finish",
                },
                {"id": "finish", "type": "finish"},
            ]
        )
    )

    options["template"]["parts"].append("changed")  # type: ignore[index]

    assert workflow.steps["render"].options == {"template": {"parts": ("initial",)}}
    with pytest.raises(TypeError):
        workflow.steps["render"].options["template"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        workflow.steps["render"].options["template"]["parts"] += ("nope",)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        workflow.start = "finish"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("start", "missing", "unknown start step 'missing'"),
        (
            "steps",
            [
                {"id": "render", "type": "render"},
                {"id": "render", "type": "finish"},
            ],
            "duplicate step ID 'render'",
        ),
        (
            "steps",
            [
                {"id": "render", "type": "render", "error": "missing"},
                {"id": "finish", "type": "finish"},
            ],
            "unknown error step 'missing'",
        ),
        (
            "steps",
            [
                {
                    "id": "render",
                    "type": "branch",
                    "branches": [
                        {
                            "condition": {"path": "score", "operator": "gte", "value": 7},
                            "next": "missing",
                        }
                    ],
                },
                {"id": "finish", "type": "finish"},
            ],
            "unknown branch step 'missing'",
        ),
    ],
)
def test_workflow_validates_graph_references(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(WorkflowValidationError, match=message):
        load_workflow(workflow_payload(**{field: value}))


def test_workflow_rejects_unregistered_node_type() -> None:
    with pytest.raises(WorkflowValidationError, match="unsupported workflow node type 'shell'"):
        load_workflow(
            workflow_payload(
                steps=[
                    {"id": "render", "type": "shell", "next": "finish"},
                    {"id": "finish", "type": "finish"},
                ]
            )
        )


@pytest.mark.parametrize("node_type", ["retry", "backtrack", "parallel", "population_map", "candidate_select"])
def test_workflow_recognizes_future_core_and_advanced_node_types(node_type: str) -> None:
    workflow = load_workflow(
        workflow_payload(
            steps=[
                {"id": "first", "type": node_type, "next": "finish"},
                {"id": "finish", "type": "finish"},
            ],
            start="first",
        )
    )

    assert workflow.steps["first"].type == node_type


def test_unreachable_steps_are_reported_in_source_order() -> None:
    workflow = load_workflow(
        workflow_payload(
            steps=[
                {"id": "render", "type": "render", "next": "finish"},
                {"id": "finish", "type": "finish"},
                {"id": "orphan-z", "type": "finish"},
                {"id": "orphan-a", "type": "finish"},
            ]
        )
    )

    assert [item.message for item in workflow.diagnostics] == [
        "unreachable step 'orphan-z'",
        "unreachable step 'orphan-a'",
    ]


def test_cycle_requires_an_explicit_visit_bound() -> None:
    with pytest.raises(WorkflowValidationError, match="cycle requires an explicit max_step_visits"):
        load_workflow(
            workflow_payload(
                steps=[
                    {"id": "render", "type": "render", "next": "finish"},
                    {"id": "finish", "type": "finish", "next": "render"},
                ]
            )
        )


@pytest.mark.parametrize("bound", [0, -1, True, 10_001])
def test_workflow_rejects_invalid_visit_bounds(bound: int) -> None:
    with pytest.raises(WorkflowValidationError, match="max_step_visits"):
        load_workflow(workflow_payload(max_step_visits=bound))


def test_cycle_with_a_positive_explicit_visit_bound_is_valid() -> None:
    workflow = load_workflow(
        workflow_payload(
            max_step_visits=2,
            steps=[
                {"id": "render", "type": "render", "next": "finish"},
                {"id": "finish", "type": "finish", "next": "render"},
            ],
        )
    )

    assert workflow.max_step_visits == 2


def test_large_bounded_cycle_is_validated_without_recursion() -> None:
    step_count = 1_100
    workflow = load_workflow(
        {
            "version": 1,
            "start": "step-0",
            "max_step_visits": 1,
            "steps": [
                {
                    "id": f"step-{index}",
                    "type": "render",
                    "next": f"step-{(index + 1) % step_count}",
                }
                for index in range(step_count)
            ],
        }
    )

    assert len(workflow.steps) == step_count


def test_workflow_state_enforces_the_validated_visit_bound() -> None:
    state = WorkflowState(current_step_id="render")
    state.increment_visit("render", maximum=1)

    with pytest.raises(RuntimeError, match="visit limit exceeded for step 'render'"):
        state.increment_visit("render", maximum=1)


@pytest.mark.parametrize(
    ("operator", "actual", "expected"),
    [
        ("eq", 7, True),
        ("ne", 7, True),
        ("gt", 8, True),
        ("gte", 7, True),
        ("lt", 6, True),
        ("lte", 7, True),
        ("in", "judge", True),
        ("contains", ["judge", "target"], True),
        ("exists", None, True),
        ("truthy", "yes", True),
    ],
)
def test_condition_dispatches_supported_leaf_operators(
    operator: str,
    actual: object,
    expected: object,
) -> None:
    values = {"value": actual}
    value = {
        "eq": 7,
        "ne": 6,
        "gt": 7,
        "gte": 7,
        "lt": 7,
        "lte": 7,
        "in": ["target", "judge"],
        "contains": "judge",
    }.get(operator)

    assert ConditionEvaluator().evaluate(Condition(path="value", operator=operator, value=value), values) is expected


def test_conditions_combine_only_nested_structured_conditions() -> None:
    at_least_seven = Condition(path="score", operator="gte", value=7)
    present = Condition(path="response", operator="exists")
    combined = Condition(operator="all", conditions=(at_least_seven, present))
    negated = Condition(operator="not", conditions=(Condition(path="blocked", operator="truthy"),))

    evaluator = ConditionEvaluator()
    assert evaluator.evaluate(combined, {"score": 8, "response": "text"})
    assert evaluator.evaluate(negated, {"blocked": False})


def test_missing_or_incompatible_values_never_satisfy_a_leaf_condition() -> None:
    evaluator = ConditionEvaluator()

    assert not evaluator.evaluate(Condition(path="missing", operator="ne", value=1), {})
    assert not evaluator.evaluate(Condition(path="score", operator="gt", value="7"), {"score": 8})
    assert not evaluator.evaluate(Condition(path="score", operator="eq", value=True), {"score": 1})


def test_condition_path_reads_mapping_keys_and_non_negative_list_indices_only() -> None:
    evaluator = ConditionEvaluator()
    condition = Condition(path="rounds.0.scores.1", operator="eq", value=8)

    assert evaluator.evaluate(condition, {"rounds": [{"scores": [7, 8]}]})
    assert not evaluator.evaluate(Condition(path="rounds.-1", operator="exists"), {"rounds": [1]})
    assert not evaluator.evaluate(Condition(path="rounds.01", operator="exists"), {"rounds": [1, 2]})


def test_condition_does_not_follow_python_attributes() -> None:
    class Trap:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"unexpected attribute access: {name}")

    assert not ConditionEvaluator().evaluate(Condition(path="payload.secret", operator="exists"), {"payload": Trap()})
    assert not ConditionEvaluator().evaluate(  # type: ignore[arg-type]
        Condition(path="payload.secret", operator="exists"), Trap()
    )


def test_workflow_rejects_implicit_or_malformed_conditions() -> None:
    with pytest.raises(WorkflowValidationError, match="condition must be a mapping"):
        load_workflow(
            workflow_payload(
                steps=[
                    {
                        "id": "branch",
                        "type": "branch",
                        "branches": [{"condition": "score >= 7", "next": "finish"}],
                    },
                    {"id": "finish", "type": "finish"},
                ],
                start="branch",
            )
        )

    with pytest.raises(WorkflowValidationError, match="operator 'all' requires a non-empty conditions list"):
        load_workflow(
            workflow_payload(
                steps=[
                    {
                        "id": "branch",
                        "type": "branch",
                        "branches": [
                            {
                                "condition": {"operator": "all", "conditions": []},
                                "next": "finish",
                            }
                        ],
                    },
                    {"id": "finish", "type": "finish"},
                ],
                start="branch",
            )
        )


def test_workflow_rejects_an_explicitly_empty_branch_list() -> None:
    with pytest.raises(WorkflowValidationError, match="branches must not be empty"):
        load_workflow(
            workflow_payload(
                start="branch",
                steps=[
                    {"id": "branch", "type": "branch", "branches": []},
                    {"id": "finish", "type": "finish"},
                ],
            )
        )


def test_yaml_loader_rejects_duplicate_keys_non_mappings_and_unsafe_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "version: 1\nstart: render\nstart: finish\nsteps: []\n",
        encoding="utf-8",
    )
    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- version: 1\n", encoding="utf-8")
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(
        "version: 1\nstart: render\nsteps:\n  - id: render\n    type: render\n    options:\n      value: .nan\n",
        encoding="utf-8",
    )
    cyclic = tmp_path / "cyclic.yaml"
    cyclic.write_text(
        "version: 1\nstart: render\nsteps:\n  - id: render\n    type: render\n    options: &options\n      self: *options\n",
        encoding="utf-8",
    )

    with pytest.raises(WorkflowValidationError, match="duplicate YAML key 'start'"):
        load_workflow(duplicate)
    with pytest.raises(WorkflowValidationError, match="workflow root must be a mapping"):
        load_workflow(non_mapping)
    with pytest.raises(WorkflowValidationError, match="JSON-safe"):
        load_workflow(unsafe)
    with pytest.raises(WorkflowValidationError, match="YAML aliases are not allowed"):
        load_workflow(cyclic)


def test_yaml_loader_rejects_non_cyclic_and_cyclic_aliases_before_construction(tmp_path: Path) -> None:
    shared = tmp_path / "shared-alias.yaml"
    shared.write_text(
        "version: 1\nstart: render\nsteps:\n"
        "  - id: render\n    type: render\n    options: &shared\n      role: target\n    next: finish\n"
        "  - id: finish\n    type: finish\n    options: *shared\n",
        encoding="utf-8",
    )
    cyclic = tmp_path / "cyclic-alias.yaml"
    cyclic.write_text(
        "version: 1\nstart: render\nsteps:\n  - id: render\n    type: render\n    options: &loop\n      self: *loop\n",
        encoding="utf-8",
    )

    for source in (shared, cyclic):
        with pytest.raises(WorkflowValidationError, match="YAML aliases are not allowed"):
            load_workflow(source)


def test_public_workflow_strings_reject_hostile_subclasses_without_hooks() -> None:
    class HostileString(str):
        hook_calls = 0

        def __getattribute__(self, name: str) -> object:
            type(self).hook_calls += 1
            raise AssertionError(f"unexpected string attribute access: {name}")

        def __hash__(self) -> int:
            type(self).hook_calls += 1
            raise AssertionError("unexpected string hash")

        def __eq__(self, other: object) -> bool:
            type(self).hook_calls += 1
            raise AssertionError("unexpected string equality")

    hostile = HostileString("render")
    with pytest.raises(TypeError, match="built-in str"):
        Condition(path=hostile)
    with pytest.raises(TypeError, match="built-in str"):
        StepDefinition(id=hostile, type="render")
    with pytest.raises(TypeError, match="built-in str"):
        StepDefinition(id="render", type=hostile)
    with pytest.raises(ValueError, match="built-in str"):
        WorkflowState(current_step_id=hostile)
    with pytest.raises(ValueError, match="built-in str"):
        WorkflowDefinition(version=1, start=hostile, steps={})

    state = WorkflowState(current_step_id="render")
    with pytest.raises(ValueError, match="built-in str"):
        state.increment_visit(hostile, maximum=1)
    with pytest.raises(ValueError, match="built-in str"):
        state.current_step_id = hostile
    assert HostileString.hook_calls == 0


def test_json_keys_reject_hostile_string_subclasses_without_runtime_hooks() -> None:
    class HostileString(str):
        hook_calls = 0

        def __getattribute__(self, name: str) -> object:
            type(self).hook_calls += 1
            raise AssertionError(f"unexpected string attribute access: {name}")

        def __hash__(self) -> int:
            type(self).hook_calls += 1
            return str.__hash__(self)

        def __eq__(self, other: object) -> bool:
            type(self).hook_calls += 1
            return str.__eq__(self, other)

    key = HostileString("key")
    hostile_mapping = {key: "value"}
    HostileString.hook_calls = 0

    with pytest.raises(TypeError, match="built-in str"):
        Condition(value=hostile_mapping)
    assert HostileString.hook_calls == 0


def test_public_workflow_scalars_reject_bool_and_int_subclasses() -> None:
    class IntegerSubclass(int):
        pass

    with pytest.raises(ValueError, match="version"):
        WorkflowDefinition(version=True, start="render", steps={})
    with pytest.raises(ValueError, match="version"):
        WorkflowDefinition(version=IntegerSubclass(1), start="render", steps={})
    with pytest.raises(ValueError, match="max_step_visits"):
        WorkflowState(current_step_id="render", max_step_visits=True)
    with pytest.raises(ValueError, match="max_step_visits"):
        WorkflowState(current_step_id="render", max_step_visits=IntegerSubclass(1))
    with pytest.raises(TypeError, match="JSON-safe"):
        Condition(value=IntegerSubclass(1))


def test_workflow_rejects_deep_json_values_without_recursion() -> None:
    nested: object = "leaf"
    for _ in range(600):
        nested = [nested]

    with pytest.raises(WorkflowValidationError, match="maximum JSON depth"):
        load_workflow(
            workflow_payload(
                steps=[
                    {"id": "render", "type": "render", "options": {"value": nested}},
                ]
            )
        )


def test_workflow_rejects_deep_logical_conditions_without_recursion() -> None:
    condition: object = {"path": "score", "operator": "gte", "value": 7}
    for _ in range(128):
        condition = {"operator": "all", "conditions": [condition]}

    with pytest.raises(WorkflowValidationError, match="maximum condition depth"):
        load_workflow(
            workflow_payload(
                start="branch",
                steps=[
                    {
                        "id": "branch",
                        "type": "branch",
                        "branches": [{"condition": condition, "next": "finish"}],
                    },
                    {"id": "finish", "type": "finish"},
                ],
            )
        )


def test_workflow_rejects_excessive_collections_and_step_counts() -> None:
    with pytest.raises(WorkflowValidationError, match="maximum JSON collection size"):
        load_workflow(
            workflow_payload(
                steps=[
                    {
                        "id": "render",
                        "type": "render",
                        "options": {"values": list(range(MAX_JSON_COLLECTION_ITEMS + 1))},
                    }
                ]
            )
        )
    with pytest.raises(WorkflowValidationError, match="maximum workflow step count"):
        load_workflow(
            {
                "version": 1,
                "start": "step-0",
                "steps": [{"id": f"step-{index}", "type": "finish"} for index in range(MAX_WORKFLOW_STEPS + 1)],
            }
        )


def test_workflow_rejects_strings_beyond_the_declared_input_limit() -> None:
    with pytest.raises(WorkflowValidationError, match="maximum JSON string length"):
        load_workflow(
            workflow_payload(
                steps=[
                    {
                        "id": "render",
                        "type": "render",
                        "options": {"text": "x" * (MAX_JSON_STRING_LENGTH + 1)},
                    }
                ]
            )
        )


def test_workflow_state_cannot_raise_its_validated_visit_bound() -> None:
    workflow = load_workflow(
        workflow_payload(
            max_step_visits=2,
            steps=[
                {"id": "render", "type": "render", "next": "render"},
            ],
        )
    )
    state = WorkflowState.initial(workflow)

    assert state.max_step_visits == 2
    state.increment_visit("render", maximum=100)
    state.increment_visit("render", maximum=100)
    with pytest.raises(RuntimeError, match="visit limit exceeded"):
        state.increment_visit("render", maximum=100)


def test_workflow_state_allows_only_stricter_optional_visit_bounds() -> None:
    state = WorkflowState(current_step_id="render", max_step_visits=3)

    state.increment_visit("render", maximum=1)
    with pytest.raises(RuntimeError, match="visit limit exceeded"):
        state.increment_visit("render", maximum=1)
    with pytest.raises(ValueError, match="positive integer"):
        state.increment_visit("other", maximum=True)


def test_workflow_input_limits_accept_declared_boundary_values() -> None:
    nested_json: object = "leaf"
    for _ in range(MAX_JSON_DEPTH):
        nested_json = [nested_json]
    condition = Condition(path="value", operator="eq", value=nested_json)
    nested_condition = Condition(path="value", operator="exists")
    for _ in range(MAX_CONDITION_DEPTH - 1):
        nested_condition = Condition(operator="not", conditions=(nested_condition,))
    boundary_string = "x" * MAX_JSON_STRING_LENGTH

    workflow = load_workflow(
        workflow_payload(
            steps=[
                {
                    "id": "render",
                    "type": "render",
                    "options": {
                        "values": list(range(MAX_JSON_COLLECTION_ITEMS)),
                        "text": boundary_string,
                    },
                }
            ]
        )
    )

    assert condition.value is not None
    assert nested_condition.operator == "not"
    assert len(workflow.steps["render"].options["values"]) == MAX_JSON_COLLECTION_ITEMS  # type: ignore[arg-type]
    assert workflow.steps["render"].options["text"] == boundary_string


def test_workflow_rejects_excessive_json_node_count() -> None:
    values = [[0] * MAX_JSON_COLLECTION_ITEMS for _ in range(5)]

    with pytest.raises(WorkflowValidationError, match="maximum JSON node count"):
        load_workflow(
            workflow_payload(
                steps=[
                    {
                        "id": "render",
                        "type": "render",
                        "options": {"values": values},
                    }
                ]
            )
        )


def test_workflow_state_keeps_its_initial_visit_cap_immutable() -> None:
    workflow = load_workflow(workflow_payload(max_step_visits=2))
    state = WorkflowState.initial(workflow)

    with pytest.raises(AttributeError, match="immutable"):
        state._max_step_visits = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        state.max_step_visits = 3  # type: ignore[misc]
    assert state.max_step_visits == 2


def test_on_error_reference_preserves_its_source_field_in_diagnostics() -> None:
    with pytest.raises(WorkflowValidationError) as raised:
        load_workflow(
            workflow_payload(
                steps=[
                    {"id": "render", "type": "render", "on_error": "missing"},
                ]
            )
        )

    diagnostic = raised.value.diagnostics[0]
    assert diagnostic.path == ("steps", 0, "on_error")
    assert diagnostic.message == "unknown on_error step 'missing'"


def test_workflow_state_rejects_hostile_mapping_inputs_without_hooks() -> None:
    class HostileMapping(Mapping[str, int]):
        hook_calls = 0

        def __getitem__(self, key: str) -> int:
            type(self).hook_calls += 1
            raise AssertionError("unexpected mapping lookup")

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).hook_calls += 1
            raise AssertionError("unexpected mapping iteration")

        def __len__(self) -> int:
            type(self).hook_calls += 1
            raise AssertionError("unexpected mapping length")

        def items(self):  # type: ignore[no-untyped-def]
            type(self).hook_calls += 1
            raise AssertionError("unexpected mapping items")

    hostile = HostileMapping()

    with pytest.raises(TypeError, match="built-in dict"):
        WorkflowState(current_step_id="render", visit_counts=hostile)
    assert HostileMapping.hook_calls == 0

    state = WorkflowState(current_step_id="render")
    with pytest.raises(TypeError, match="built-in dict"):
        state.values = hostile
    assert HostileMapping.hook_calls == 0


def test_workflow_state_rejects_hostile_dicts_and_proxies_without_hooks() -> None:
    class HostileDict(dict[str, int]):
        hook_calls = 0

        def items(self):  # type: ignore[no-untyped-def]
            type(self).hook_calls += 1
            raise AssertionError("unexpected dict items")

        def __iter__(self):  # type: ignore[no-untyped-def]
            type(self).hook_calls += 1
            raise AssertionError("unexpected dict iteration")

        def __hash__(self) -> int:
            type(self).hook_calls += 1
            raise AssertionError("unexpected dict hash")

        def __eq__(self, other: object) -> bool:
            type(self).hook_calls += 1
            raise AssertionError("unexpected dict equality")

        def keys(self):  # type: ignore[no-untyped-def]
            type(self).hook_calls += 1
            raise AssertionError("unexpected dict keys")

    hostile = HostileDict({"render": 1})
    for value in (hostile, MappingProxyType(hostile)):
        HostileDict.hook_calls = 0
        with pytest.raises(TypeError, match="built-in dict"):
            WorkflowState(current_step_id="render", visit_counts=value)
        assert HostileDict.hook_calls == 0

        state = WorkflowState(current_step_id="render")
        with pytest.raises(TypeError, match="built-in dict"):
            state.values = value
        assert HostileDict.hook_calls == 0


def test_step_definition_preserves_the_original_positional_field_order() -> None:
    branch = BranchDefinition(next_step_id="finish")
    step = StepDefinition(
        "render",
        "render",
        "finish",
        "recover",
        (branch,),
        {"template": "safe"},
    )

    assert step.id == "render"
    assert step.type == "render"
    assert step.next_step_id == "finish"
    assert step.error_step_id == "recover"
    assert step.branches == (branch,)
    assert step.options == {"template": "safe"}
    assert step.error_source_field == "error"


def _nested_flow_sequence(depth: int) -> str:
    return "[" * depth + "null" + "]" * depth


_YAML_STRUCTURAL_DEPTH_LIMIT = 128


@pytest.mark.parametrize(
    "depth",
    [_YAML_STRUCTURAL_DEPTH_LIMIT - 1, _YAML_STRUCTURAL_DEPTH_LIMIT],
)
def test_yaml_structures_at_or_below_the_depth_limit_reach_schema_validation(
    depth: int,
) -> None:
    with pytest.raises(WorkflowValidationError, match="workflow root must be a mapping"):
        load_workflow(_nested_flow_sequence(depth))


@pytest.mark.parametrize("depth", [_YAML_STRUCTURAL_DEPTH_LIMIT + 1, 1_500])
def test_yaml_structures_over_the_depth_limit_fail_before_construction(depth: int) -> None:
    with pytest.raises(WorkflowValidationError, match="maximum YAML depth"):
        load_workflow(_nested_flow_sequence(depth))


def test_yaml_scalar_length_is_limited_before_construction() -> None:
    with pytest.raises(WorkflowValidationError, match="maximum YAML scalar length"):
        load_workflow("x" * (MAX_JSON_STRING_LENGTH + 1))


def test_yaml_timestamp_construction_failure_is_normalized() -> None:
    with pytest.raises(WorkflowValidationError) as raised:
        load_workflow("2001-99-99")

    assert "invalid workflow YAML" in str(raised.value)
    assert "month must be in 1..12" in str(raised.value)


def test_yaml_integer_conversion_limit_failure_is_normalized() -> None:
    with pytest.raises(WorkflowValidationError) as raised:
        load_workflow("9" * 5_000)

    assert "invalid workflow YAML" in str(raised.value)
    assert "value has 5000 digits" in str(raised.value)
