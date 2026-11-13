from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping

import pytest

from app.api.store import EvaluationStore
from app.llm_client import ChatClient, ChatCompletionResult
from app.persistence import NoopRunPersistence
from app.persistence.technique_runtime import (
    ArtifactRecord,
    CheckpointRecord,
    ModelCallRecord,
    TechniqueRuntimePersistence,
)
from app.techniques.contracts.definition import ImplementationKind
from app.techniques.contracts.events import ExecutionEvent
from app.techniques.contracts.execution import (
    CorrelationIds,
    ModelRoleBinding,
    ResolvedRunSpec,
)
from app.techniques.core.budgets import BudgetExhausted, BudgetLimits, BudgetService
from app.techniques.core.correlation import CorrelationScope, current_correlation
from app.techniques.core.events import EventJournal, TechniqueEventIdentity
from app.techniques.core.models import (
    CallRetryPolicy,
    ModelCallResult,
    ModelService,
    ProviderCallResult,
)
from app.techniques.core.redaction import RedactionService
from app.techniques.legacy.adapter import LegacyCorrelatedChatClient


class RecordingRuntimePersistence(TechniqueRuntimePersistence):
    def __init__(self) -> None:
        self.events: list[ExecutionEvent] = []
        self.calls: list[ModelCallRecord] = []
        self.completions: dict[str, dict[str, object]] = {}
        self.failures: dict[str, dict[str, object]] = {}

    def append_event(self, event: ExecutionEvent) -> None:
        self.events.append(event)

    def append_events(self, events: tuple[ExecutionEvent, ...]) -> None:
        self.events.extend(events)

    def record_model_call_requested(self, call: ModelCallRecord) -> None:
        self.calls.append(call)

    def complete_model_call(
        self,
        model_call_id: str,
        *,
        response: str,
        usage: Mapping[str, object],
        provider_response_id: str | None,
        gateway_call_id: str | None,
        gateway_attempts: Mapping[str, int],
    ) -> None:
        self.completions[model_call_id] = {
            "response": response,
            "usage": dict(usage),
            "provider_response_id": provider_response_id,
            "gateway_call_id": gateway_call_id,
            "gateway_attempts": dict(gateway_attempts),
        }

    def fail_model_call(
        self,
        model_call_id: str,
        *,
        category: str,
        error: str,
        indeterminate: bool,
    ) -> None:
        self.failures[model_call_id] = {
            "category": category,
            "error": error,
            "indeterminate": indeterminate,
        }

    def save_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        return

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        return None

    def latest_checkpoint(
        self,
        *,
        experiment_id: str,
        cell_id: str,
        attempt_ids: tuple[str, ...],
    ) -> CheckpointRecord | None:
        return None

    def attempt_has_indeterminate_call(self, attempt_id: str) -> bool:
        failed_call_ids = {
            model_call_id for model_call_id, failure in self.failures.items() if failure.get("indeterminate") is True
        }
        return any(
            call.model_call_id in failed_call_ids and call.correlation.attempt_id == attempt_id for call in self.calls
        )

    def save_artifact(self, artifact: ArtifactRecord) -> None:
        return

    def serialized_records(self) -> str:
        return repr((self.events, self.calls, self.completions, self.failures))


class DelayedClient:
    def __init__(
        self,
        text: str,
        *,
        delay: float = 0,
        failures: int = 0,
        usage: Mapping[str, object] | None = None,
    ) -> None:
        self.text = text
        self.delay = delay
        self.failures = failures
        self.usage = usage or {"total_tokens": 1}
        self.request_contexts: list[object] = []

    def chat_correlated_once(
        self,
        messages: list[dict[str, str]],
        *,
        request_context: object,
        **options: object,
    ) -> ProviderCallResult:
        self.request_contexts.append(request_context)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("HTTP 429 transient provider failure")
        if self.delay:
            time.sleep(self.delay)
        return ProviderCallResult(
            text=self.text,
            usage=self.usage,
            provider_response_id=f"provider-{self.text}",
        )


class TimeoutClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_correlated_once(
        self,
        messages: list[dict[str, str]],
        *,
        request_context: object,
        **options: object,
    ) -> ProviderCallResult:
        self.calls += 1
        raise TimeoutError("provider response state is unknown")


class FakeHTTPResponse:
    def __init__(self, *, headers: Mapping[str, str]) -> None:
        self.headers = dict(headers)

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "id": "provider-response-1",
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 2},
            }
        ).encode("utf-8")


def correlation(cell_id: str) -> CorrelationIds:
    return CorrelationIds(
        experiment_id="exp-1",
        cell_id=cell_id,
        run_id=f"run-{cell_id}",
        attempt_id="attempt-1",
    )


def resolved_spec(*, provider: str = "openai") -> ResolvedRunSpec:
    binding = ModelRoleBinding.from_profile(
        logical_role="target",
        profile_id="profile-1",
        provider=provider,
        model="model-a",
    )
    return ResolvedRunSpec(
        technique_id="probe",
        technique_version="1.0.0",
        implementation_id="modular",
        implementation_kind=ImplementationKind.WORKFLOW,
        contract_version="1",
        manifest_hash="a" * 64,
        model_role_bindings={"target": binding},
    )


def model_service(
    *,
    cell_id: str,
    client: object,
    provider: str = "openai",
    max_attempts: int = 1,
    retry_indeterminate: bool = False,
    budget: BudgetService | None = None,
) -> tuple[ModelService, RecordingRuntimePersistence]:
    call_correlation = correlation(cell_id)
    persistence = RecordingRuntimePersistence()
    events = EventJournal(
        correlation=call_correlation,
        persistence=persistence,
        technique=TechniqueEventIdentity("probe", "1.0.0", "modular"),
    )
    return (
        ModelService(
            correlation=call_correlation,
            spec=resolved_spec(provider=provider),
            clients={"target": client},
            persistence=persistence,
            events=events,
            redaction=RedactionService(),
            retry_policy=CallRetryPolicy(
                max_attempts=max_attempts,
                retry_indeterminate=retry_indeterminate,
            ),
            budget=budget,
        ),
        persistence,
    )


def test_parallel_model_results_keep_their_originating_cell() -> None:
    async def scenario() -> tuple[ModelCallResult, ModelCallResult]:
        service_a, _ = model_service(
            cell_id="cell-a",
            client=DelayedClient("response-a", delay=0.02),
        )
        service_b, _ = model_service(
            cell_id="cell-b",
            client=DelayedClient("response-b"),
        )
        results = await asyncio.gather(
            service_a.call(
                role="target",
                messages=({"role": "user", "content": "a"},),
                operation_id="target-r1",
            ),
            service_b.call(
                role="target",
                messages=({"role": "user", "content": "b"},),
                operation_id="target-r1",
            ),
        )
        return results[0], results[1]

    result_a, result_b = asyncio.run(scenario())

    assert result_a.correlation.cell_id == "cell-a"
    assert result_a.text == "response-a"
    assert result_b.correlation.cell_id == "cell-b"
    assert result_b.text == "response-b"


def test_correlation_scope_resets_nested_context() -> None:
    outer = correlation("cell-a")
    inner = correlation("cell-b")

    assert current_correlation() is None
    with CorrelationScope(outer):
        assert current_correlation() == outer
        with CorrelationScope(inner):
            assert current_correlation() == inner
        assert current_correlation() == outer
    assert current_correlation() is None


def test_model_gateway_redacts_secrets_before_persistence() -> None:
    service, persistence = model_service(
        cell_id="cell-1",
        client=DelayedClient("ok"),
    )

    asyncio.run(
        service.call(
            role="target",
            messages=({"role": "user", "content": "hello"},),
            operation_id="target-r1",
            metadata={"api_key": "sk-test-secret"},
        )
    )

    assert "sk-test-secret" not in persistence.serialized_records()


def test_provider_call_result_preserves_existing_positional_constructor() -> None:
    result = ProviderCallResult(
        "ok",
        {"total_tokens": 1},
        "provider-response-1",
        "gateway-call-1",
        {"attempted_retries": 0},
    )

    assert result.text == "ok"
    assert result.usage == {"total_tokens": 1}
    assert result.provider_response_id == "provider-response-1"
    assert result.gateway_call_id == "gateway-call-1"
    assert result.gateway_attempts == {"attempted_retries": 0}
    assert result.tool_calls == ()


def test_litellm_proxy_receives_metadata_and_returns_gateway_id(
    monkeypatch,
) -> None:
    requests = []

    def fake_urlopen(request, timeout, context):
        requests.append(request)
        return FakeHTTPResponse(
            headers={
                "x-litellm-call-id": "gateway-call-1",
                "x-litellm-attempted-retries": "1",
                "x-litellm-attempted-fallbacks": "0",
            }
        )

    monkeypatch.setattr("app.llm_client.urllib.request.urlopen", fake_urlopen)
    service, persistence = model_service(
        cell_id="cell-1",
        provider="litellm_proxy",
        client=ChatClient(api_base_url="http://gateway.test/v1", api_key="secret"),
    )

    result = asyncio.run(
        service.call(
            role="target",
            messages=({"role": "user", "content": "hello"},),
            operation_id="target-r1",
        )
    )

    headers = {key.lower(): value for key, value in requests[0].header_items()}
    assert headers["x-litellm-trace-id"] == "attempt-1"
    metadata = json.loads(headers["x-litellm-spend-logs-metadata"])
    assert metadata["cell_id"] == "cell-1"
    assert metadata["model_call_id"] == result.model_call_id
    assert result.gateway_call_id == "gateway-call-1"
    assert persistence.completions[result.model_call_id]["gateway_attempts"] == {
        "attempted_retries": 1,
        "attempted_fallbacks": 0,
    }


def test_direct_provider_does_not_receive_litellm_headers(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout, context):
        requests.append(request)
        return FakeHTTPResponse(headers={})

    monkeypatch.setattr("app.llm_client.urllib.request.urlopen", fake_urlopen)
    service, _ = model_service(
        cell_id="cell-1",
        provider="openai",
        client=ChatClient(api_base_url="http://litellm-named-host.test/v1"),
    )

    asyncio.run(
        service.call(
            role="target",
            messages=({"role": "user", "content": "hello"},),
            operation_id="target-r1",
        )
    )

    headers = {key.lower(): value for key, value in requests[0].header_items()}
    assert all(not key.startswith("x-litellm-") for key in headers)


def test_operation_retries_have_distinct_local_call_ids() -> None:
    service, persistence = model_service(
        cell_id="cell-1",
        client=DelayedClient("ok", failures=1),
        max_attempts=2,
        retry_indeterminate=True,
    )

    result = asyncio.run(
        service.call(
            role="target",
            messages=({"role": "user", "content": "hello"},),
            operation_id="target-r1",
        )
    )

    assert [record.operation_attempt for record in persistence.calls] == [1, 2]
    assert len({record.model_call_id for record in persistence.calls}) == 2
    assert result.model_call_id == persistence.calls[-1].model_call_id


def test_indeterminate_failure_is_not_retried_without_explicit_policy() -> None:
    client = TimeoutClient()
    service, persistence = model_service(
        cell_id="cell-1",
        client=client,
        max_attempts=2,
    )

    try:
        asyncio.run(
            service.call(
                role="target",
                messages=({"role": "user", "content": "hello"},),
                operation_id="target-r1",
            )
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected the indeterminate timeout to propagate")

    assert client.calls == 1
    assert len(persistence.calls) == 1
    assert next(iter(persistence.failures.values()))["indeterminate"] is True


def test_model_budget_rejects_before_client_dispatch() -> None:
    client = DelayedClient("must not be called")
    service, persistence = model_service(
        cell_id="cell-1",
        client=client,
        budget=BudgetService(BudgetLimits(max_model_calls=0)),
    )

    with pytest.raises(BudgetExhausted, match="model_calls"):
        asyncio.run(
            service.call(
                role="target",
                messages=({"role": "user", "content": "hello"},),
                operation_id="target-r1",
            )
        )

    assert client.request_contexts == []
    assert persistence.calls == []


class LegacyRawClient:
    model = "legacy-model"
    api_base_url = "http://provider.test/v1"

    def __init__(self, persistence: RecordingRuntimePersistence) -> None:
        self.persistence = persistence
        self.contexts = []

    def prepare_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        return [dict(message) for message in messages]

    def chat_raw(self, messages, *, _request_context, **options):
        assert self.persistence.calls
        self.contexts.append(_request_context)
        return ChatCompletionResult(
            content="legacy response",
            usage={"total_tokens": "4"},
            raw_response={"id": "provider-legacy"},
            choice={},
            message={"content": "legacy response"},
            finish_reason="stop",
            refusal=None,
            provider_response_id="provider-legacy",
        )


class FailingLegacyRawClient(LegacyRawClient):
    def __init__(
        self,
        persistence: RecordingRuntimePersistence,
        secret: str,
    ) -> None:
        super().__init__(persistence)
        self.secret = secret

    def chat_raw(self, messages, *, _request_context, **options):
        assert self.persistence.calls
        self.contexts.append(_request_context)
        raise RuntimeError(f"provider rejected credential {self.secret}")


class SequencedLegacyRawClient(LegacyRawClient):
    def __init__(
        self,
        persistence: RecordingRuntimePersistence,
        failures: tuple[Exception, ...],
    ) -> None:
        super().__init__(persistence)
        self.failures = list(failures)

    def chat_raw(self, messages, *, _request_context, **options):
        assert self.persistence.calls
        self.contexts.append(_request_context)
        if self.failures:
            raise self.failures.pop(0)
        return ChatCompletionResult(
            content="legacy response",
            usage={"total_tokens": "4"},
            raw_response={"id": "provider-legacy"},
            choice={},
            message={"content": "legacy response"},
            finish_reason="stop",
            refusal=None,
            provider_response_id="provider-legacy",
        )


def legacy_retry_fixture(
    failures: tuple[Exception, ...],
) -> tuple[
    LegacyCorrelatedChatClient,
    SequencedLegacyRawClient,
    RecordingRuntimePersistence,
    CorrelationIds,
]:
    call_correlation = correlation("cell-legacy-retry")
    persistence = RecordingRuntimePersistence()
    journal = EventJournal(
        correlation=call_correlation,
        persistence=persistence,
        technique=TechniqueEventIdentity("probe", "1.0.0", "legacy"),
    )
    raw = SequencedLegacyRawClient(persistence, failures)
    wrapper = LegacyCorrelatedChatClient(
        client=raw,
        correlation=call_correlation,
        binding=ModelRoleBinding.from_profile(
            logical_role="target",
            profile_id="profile-legacy",
            provider="openai",
            model=raw.model,
        ),
        technique_id="probe",
        technique_version="1.0.0",
        implementation_id="legacy",
        persistence=persistence,
        events=journal,
    )
    return wrapper, raw, persistence, call_correlation


def test_legacy_chat_wrapper_persists_ownership_before_dispatch() -> None:
    call_correlation = correlation("cell-legacy")
    persistence = RecordingRuntimePersistence()
    journal = EventJournal(
        correlation=call_correlation,
        persistence=persistence,
        technique=TechniqueEventIdentity("probe", "1.0.0", "legacy"),
    )
    raw = LegacyRawClient(persistence)
    wrapper = LegacyCorrelatedChatClient(
        client=raw,
        correlation=call_correlation,
        binding=ModelRoleBinding.from_profile(
            logical_role="target",
            profile_id="profile-legacy",
            provider="openai",
            model=raw.model,
        ),
        technique_id="probe",
        technique_version="1.0.0",
        implementation_id="legacy",
        persistence=persistence,
        events=journal,
    )

    text, usage = wrapper.chat_with_retry([{"role": "user", "content": "hello"}])

    assert text == "legacy response"
    assert usage["total_tokens"] == "4"
    assert persistence.calls[0].correlation.cell_id == "cell-legacy"
    assert raw.contexts[0].model_call_id == persistence.calls[0].model_call_id


@pytest.mark.parametrize("status_code", ["429", "500", "502", "503", "504"])
def test_legacy_retry_preserves_recognized_status_compatibility(
    status_code: str,
) -> None:
    wrapper, raw, persistence, call_correlation = legacy_retry_fixture(
        (RuntimeError(f"HTTP {status_code} transient provider failure"),)
    )

    text, usage = wrapper.chat_with_retry(
        [{"role": "user", "content": "hello"}],
        max_retries=1,
    )

    assert text == "legacy response"
    assert usage == {"total_tokens": "4"}
    assert len(raw.contexts) == 2
    assert [call.operation_attempt for call in persistence.calls] == [1, 2]
    assert len({call.operation_id for call in persistence.calls}) == 1
    assert all(call.correlation == call_correlation for call in persistence.calls)
    assert [context.operation_id for context in raw.contexts] == [
        persistence.calls[0].operation_id,
        persistence.calls[0].operation_id,
    ]
    first_failure = persistence.failures[persistence.calls[0].model_call_id]
    assert first_failure["category"] == "provider"
    assert first_failure["indeterminate"] is False
    assert [event.event_type for event in persistence.events] == [
        "model.requested",
        "model.failed",
        "model.requested",
        "model.completed",
    ]
    assert [event.attempt for event in persistence.events] == [1, 1, 2, 2]
    assert all(event.correlation == call_correlation for event in persistence.events)
    assert {event.operation_id for event in persistence.events} == {persistence.calls[0].operation_id}


def test_legacy_retry_stops_at_the_existing_declared_bound() -> None:
    wrapper, raw, persistence, _ = legacy_retry_fixture(
        tuple(RuntimeError("HTTP 503 transient provider failure") for _ in range(3))
    )

    with pytest.raises(RuntimeError, match="HTTP 503"):
        wrapper.chat_with_retry(
            [{"role": "user", "content": "hello"}],
            max_retries=2,
        )

    assert len(raw.contexts) == 3
    assert [call.operation_attempt for call in persistence.calls] == [1, 2, 3]
    assert len({call.operation_id for call in persistence.calls}) == 1
    assert all(persistence.failures[call.model_call_id]["indeterminate"] is False for call in persistence.calls)
    assert [event.attempt for event in persistence.events] == [1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize(
    ("error", "expected_category", "expected_indeterminate"),
    [
        (TimeoutError("provider response state is unknown"), "timeout", True),
        (RuntimeError("provider rejected request"), "provider", False),
    ],
)
def test_legacy_retry_preserves_timeout_and_unknown_failure_behavior(
    error: Exception,
    expected_category: str,
    expected_indeterminate: bool,
) -> None:
    wrapper, raw, persistence, _ = legacy_retry_fixture((error,))

    with pytest.raises(type(error), match=str(error)):
        wrapper.chat_with_retry(
            [{"role": "user", "content": "hello"}],
            max_retries=2,
        )

    assert len(raw.contexts) == 1
    assert [call.operation_attempt for call in persistence.calls] == [1]
    failure = persistence.failures[persistence.calls[0].model_call_id]
    assert failure["category"] == expected_category
    assert failure["indeterminate"] is expected_indeterminate


def test_model_budget_reconciles_string_provider_token_usage() -> None:
    budget = BudgetService(BudgetLimits(max_tokens=10))
    service, _ = model_service(
        cell_id="cell-usage",
        client=DelayedClient("ok", usage={"total_tokens": "7"}),
        budget=budget,
    )

    asyncio.run(
        service.call(
            role="target",
            messages=({"role": "user", "content": "hello"},),
            operation_id="target-r1",
        )
    )

    assert budget.usage.tokens == 7


def test_store_model_calls_keep_public_run_and_experiment_cell_ownership() -> None:
    persistence = RecordingRuntimePersistence()
    raw = LegacyRawClient(persistence)
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=persistence,
    )
    store.catalog.set_default(
        "system_prompt_extraction",
        implementation_id="legacy",
    )
    store._build_chat_client = lambda config: raw

    summary, _ = store._execute_single(
        technique_key="system_prompt_extraction",
        target_kind="chatbot",
        mode="live",
        target_config={"api_key": "opaque-key", "unsafe_intent": "probe"},
        run_id="run-public",
        experiment_id="exp-1",
        cell_id="cell-1",
    )

    assert summary.run_id == "run-public"
    assert persistence.calls
    assert {
        (
            call.correlation.experiment_id,
            call.correlation.cell_id,
            call.correlation.run_id,
        )
        for call in persistence.calls
    } == {("exp-1", "cell-1", "run-public")}


def test_independent_stores_allocate_distinct_run_ids() -> None:
    first = EvaluationStore(persistence=NoopRunPersistence())
    second = EvaluationStore(persistence=NoopRunPersistence())

    first_run = first.create_run("system_prompt_extraction", mode="demo")
    second_run = second.create_run("system_prompt_extraction", mode="demo")

    assert first_run["run_id"].startswith("run-")
    assert second_run["run_id"].startswith("run-")
    assert first_run["run_id"] != second_run["run_id"]


def test_store_redacts_configured_opaque_secret_from_runtime_evidence() -> None:
    persistence = RecordingRuntimePersistence()
    secret = "opaque-test-credential"
    raw = FailingLegacyRawClient(persistence, secret)
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=persistence,
    )
    store.catalog.set_default(
        "system_prompt_extraction",
        implementation_id="legacy",
    )
    store._build_chat_client = lambda config: raw

    summary, _ = store._execute_single(
        technique_key="system_prompt_extraction",
        target_kind="chatbot",
        mode="live",
        target_config={"api_key": secret, "unsafe_intent": "probe"},
    )

    assert secret not in persistence.serialized_records()
    assert secret not in repr(tuple(store.normalized_events_by_run.values()))
    assert secret not in repr(summary)
    assert summary.status == "error"
    assert summary.error == "provider rejected credential [REDACTED]"
    event_types = [event.event_type for event in persistence.events]
    assert "run.failed" in event_types
    assert "run.completed" not in event_types
    assert "target.failed" in event_types
    assert "target.completed" not in event_types


def test_legacy_judge_failure_terminates_run_as_failed() -> None:
    persistence = RecordingRuntimePersistence()
    target = LegacyRawClient(persistence)
    judge = FailingLegacyRawClient(persistence, "judge failed")
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=persistence,
    )
    store.catalog.set_default(
        "system_prompt_extraction",
        implementation_id="legacy",
    )
    store._build_chat_client = lambda config: target
    store._build_attacker_client = lambda config: None
    store._build_judge_client = lambda config, *, target_client, attacker_client: judge

    summary, _ = store._execute_single(
        technique_key="system_prompt_extraction",
        target_kind="chatbot",
        mode="live",
        target_config={"api_key": "opaque-key", "unsafe_intent": "probe"},
    )

    event_types = [event.event_type for event in persistence.events]
    assert summary.status == "error"
    assert "judge.failed" in event_types
    assert "run.failed" in event_types
    assert "run.completed" not in event_types


def test_custom_technique_model_calls_keep_experiment_cell_ownership() -> None:
    persistence = RecordingRuntimePersistence()
    raw = LegacyRawClient(persistence)
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=persistence,
    )
    store._build_chat_client = lambda config: raw
    store._build_attacker_client = lambda config: None

    summary, _ = store._execute_single(
        technique_key="custom:probe",
        target_kind="chatbot",
        mode="live",
        target_config={"api_key": "opaque-key", "unsafe_intent": "probe"},
        custom_definition={"mode": "single_shot", "prompt_template": "{intent}"},
        run_id="run-custom",
        experiment_id="exp-custom",
        cell_id="cell-custom",
    )

    assert summary.run_id == "run-custom"
    assert persistence.calls
    assert {
        (
            call.correlation.experiment_id,
            call.correlation.cell_id,
            call.correlation.run_id,
        )
        for call in persistence.calls
    } == {("exp-custom", "cell-custom", "run-custom")}


def test_baseline_judge_failure_is_correlated_and_null_scored() -> None:
    persistence = RecordingRuntimePersistence()
    target = LegacyRawClient(persistence)
    judge = FailingLegacyRawClient(persistence, "opaque-judge-credential")
    store = EvaluationStore(
        persistence=NoopRunPersistence(),
        technique_runtime_persistence=persistence,
    )
    store._build_chat_client = lambda config: target
    store._build_attacker_client = lambda config: None
    store._build_judge_client = lambda config, *, target_client, attacker_client: judge

    result = store.run_baseline(
        target_config={"api_key": "opaque-key", "unsafe_intent": "probe"},
        evaluation_id="exp-baseline",
        experiment_id="exp-baseline",
        cell_id="cell-baseline",
    )

    assert result["judge_score"] is None
    assert result["run_id"].startswith("run-")
    assert {
        (
            call.correlation.experiment_id,
            call.correlation.cell_id,
            call.correlation.run_id,
        )
        for call in persistence.calls
    } == {("exp-baseline", "cell-baseline", result["run_id"])}
    event_types = [event.event_type for event in persistence.events]
    assert "judge.failed" in event_types
    assert "run.failed" in event_types
    assert "run.completed" not in event_types
