import json
import sys

import pytest

sys.path.insert(0, "backend")

from app.attacklib import AttackBuildRequest, get_attack_technique
from app.attacklib.converters import ConverterPipeline, PromptConverter
from app.attacklib.renderer import render_prompt
from app.attacklib.base import AttackCase
from app.execution import TechniqueRunner
from app.importers import import_suite_from_json
from app.llm_client import ChatClient, ChatCompletionResult
from app.reporting import build_report
from app.targets import (
    LiveAgentAdapter,
    RecordedAgentAdapter,
    RecordedChatbotAdapter,
    TargetDescriptor,
)
from app.targets.live_chatbot import LiveChatbotAdapter
from app.evaluation.models import (
    ExecutionRequest,
    TargetKind,
    TargetResponse,
    TechniqueConfig,
    TraceEvent,
)


class MarkerConverter(PromptConverter):
    def convert(self, text: str) -> str:
        return f"converted::{text}"


class CapturingChatbotAdapter:
    def __init__(self) -> None:
        self.descriptor = TargetDescriptor(
            target_id="chatbot-1",
            name="Capturing Chatbot",
            kind=TargetKind.CHATBOT,
            capabilities=("chat",),
        )
        self.request: ExecutionRequest | None = None

    def execute(self, request: ExecutionRequest) -> TargetResponse:
        self.request = request
        return TargetResponse(
            target_id=self.descriptor.target_id,
            target_kind=TargetKind.CHATBOT,
            final_output="captured",
            metadata={"adapter": "capturing-chatbot"},
        )


class FakeChatClient:
    model = "fake-model"

    def __init__(self, response_text: str = "fake response") -> None:
        self.extra_body: dict[str, object] | None = None
        self.calls: list[list[dict[str, str]]] = []
        self.response_text = response_text

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, str]]:
        self.extra_body = extra_body
        self.calls.append([dict(message) for message in messages])
        return self.response_text, {"total_tokens": "1"}


class SequencedFakeChatClient:
    model = "fake-model"

    def __init__(self, responses: tuple[str, ...]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict[str, str]]:
        self.calls.append([dict(message) for message in messages])
        response = self.responses.pop(0)
        return response, {"total_tokens": str(len(self.calls))}


class RawToolFakeChatClient:
    model = "fake-model"

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.extra_body: dict[str, object] | None = None
        self.raw_response = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "shell_exec",
                                    "arguments": "{\"cmd\":\"python main.py\"}",
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"total_tokens": 9},
        }

    def chat_raw(
        self,
        messages: list[dict[str, str]],
        *,
        extra_body: dict[str, object] | None = None,
    ) -> ChatCompletionResult:
        self.extra_body = extra_body
        self.calls.append([dict(message) for message in messages])
        choice = self.raw_response["choices"][0]
        message = choice["message"]
        return ChatCompletionResult(
            content="",
            usage={"total_tokens": "9"},
            raw_response=self.raw_response,
            choice=choice,
            message=message,
            finish_reason="tool_calls",
            refusal=None,
            tool_calls=tuple(message["tool_calls"]),
        )


class FakeHTTPResponse:
    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "choices": [{"message": {"content": "fake response"}}],
                "usage": {"total_tokens": 1},
            }
        ).encode()


def test_runner_executes_text_technique_against_chatbot() -> None:
    runner = TechniqueRunner()
    target = RecordedChatbotAdapter(
        descriptor=TargetDescriptor(
            target_id="chatbot-1",
            name="Recorded Chatbot",
            kind=TargetKind.CHATBOT,
            capabilities=("chat",),
        ),
        recorded_output="The target complied and referenced a hidden prompt.",
    )

    result = runner.execute_registered_technique(
        run_id="run-1",
        requested_by="tester",
        target=target,
        technique_key="system_prompt_extraction",
        unsafe_intent_label="prompt-leakage-check",
        seed_prompt="sanitized-placeholder",
    )

    assert result.technique_key == "system_prompt_extraction"
    assert result.score > 0


def test_runner_executes_agent_technique_with_trace_capture() -> None:
    runner = TechniqueRunner()
    target = RecordedAgentAdapter(
        descriptor=TargetDescriptor(
            target_id="agent-1",
            name="Recorded Agent",
            kind=TargetKind.AGENT,
            capabilities=("chat", "tools", "memory"),
        ),
        recorded_output="override accepted",
        recorded_traces=(
            TraceEvent(event_type="tool_call", actor="agent", message="tool call shell"),
            TraceEvent(event_type="delegation", actor="agent", message="delegated to sub-agent"),
        ),
        tool_calls=("shell.exec",),
        memory_events=("memory write",),
    )

    result = runner.execute_registered_technique(
        run_id="run-2",
        requested_by="tester",
        target=target,
        technique_key="tool_misuse",
        unsafe_intent_label="memory-boundary-check",
        seed_prompt="sanitized-placeholder",
    )

    report = build_report((result,))
    assert report.summary.total_techniques == 1
    assert len(result.traces) >= 2


def test_chat_client_extra_body_cannot_override_reserved_payload_fields(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(req, *, timeout, context):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeHTTPResponse()

    monkeypatch.setattr("app.llm_client.urllib.request.urlopen", fake_urlopen)

    client = ChatClient(
        api_base_url="http://localhost:11434/v1",
        api_key="test-key",
        model="safe-model",
        max_tokens=321,
    )
    client.chat(
        [{"role": "user", "content": "hello"}],
        response_format={"type": "json_object"},
        extra_body={
            "model": "attacker-model",
            "messages": [{"role": "user", "content": "attacker"}],
            "max_tokens": 999999,
            "max_completion_tokens": 999999,
            "response_format": {"type": "text"},
            "top_p": 0.8,
        },
    )

    assert captured["payload"] == {
        "model": "safe-model",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.7,
        "max_tokens": 321,
        "response_format": {"type": "json_object"},
        "top_p": 0.8,
    }


def test_chat_client_rejects_non_http_api_base_url_before_network_use() -> None:
    with pytest.raises(ValueError, match="http or https"):
        ChatClient(api_base_url="file:///tmp/local-model")


def test_chat_client_chat_raw_preserves_tool_calls_and_raw_response(
    monkeypatch,
) -> None:
    raw_response = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "shell_exec",
                                "arguments": "{\"cmd\":\"python main.py\"}",
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"total_tokens": 9},
    }

    class RawHTTPResponse:
        def __enter__(self) -> "RawHTTPResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(raw_response).encode()

    def fake_urlopen(req, *, timeout, context):
        return RawHTTPResponse()

    monkeypatch.setattr("app.llm_client.urllib.request.urlopen", fake_urlopen)

    client = ChatClient(
        api_base_url="http://localhost:11434/v1",
        api_key="test-key",
        model="safe-model",
    )
    result = client.chat_raw([{"role": "user", "content": "use a tool"}])

    assert result.content == ""
    assert result.raw_response == raw_response
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls == tuple(raw_response["choices"][0]["message"]["tool_calls"])


def test_runner_sends_converted_single_shot_case_to_target() -> None:
    runner = TechniqueRunner()
    target = CapturingChatbotAdapter()

    result = runner.execute_registered_technique(
        run_id="run-converted-single",
        requested_by="tester",
        target=target,
        technique_key="direct_prompt_injection",
        unsafe_intent_label="instruction-override-check",
        converter_pipeline=ConverterPipeline([MarkerConverter()]),
    )

    assert target.request is not None
    sent_content = render_prompt(target.request.case)[-1]["content"]
    assert sent_content.startswith("converted::")
    assert "instruction-override-check" in sent_content
    assert result.traces[0].attributes["rendered_prompt"] == sent_content


def test_runner_sends_converted_scripted_case_to_target() -> None:
    runner = TechniqueRunner()
    target = CapturingChatbotAdapter()

    result = runner.execute_registered_technique(
        run_id="run-converted-scripted",
        requested_by="tester",
        target=target,
        technique_key="nested_scenario",
        unsafe_intent_label="scripted-converter-check",
        converter_pipeline=ConverterPipeline([MarkerConverter()]),
    )

    assert target.request is not None
    sent_turns = target.request.case.conversation_turns
    assert [turn["role"] for turn in sent_turns] == ["user", "assistant", "user"]
    assert sent_turns[0]["content"].startswith("converted::")
    assert not sent_turns[1]["content"].startswith("converted::")
    assert sent_turns[2]["content"].startswith("converted::")
    assert "scripted-converter-check" in sent_turns[2]["content"]
    assert "[user]\nconverted::" in result.traces[0].attributes["rendered_prompt"]


def test_runner_materializes_converted_blocks_without_duplicate_target_context() -> None:
    runner = TechniqueRunner()
    target = CapturingChatbotAdapter()

    result = runner.execute_registered_technique(
        run_id="run-converted-context",
        requested_by="tester",
        target=target,
        technique_key="context_poisoning",
        unsafe_intent_label="write the risky rollout query",
        converter_pipeline=ConverterPipeline([MarkerConverter()]),
    )

    assert target.request is not None
    sent_content = render_prompt(target.request.case)[-1]["content"]
    assert sent_content.startswith("converted::")
    assert sent_content.count("BEGIN WORKSPACE FILE: docs/rollout_runbook.md") == 1
    assert target.request.case.workspace_files == ()
    assert target.request.case.assets["materialized_workspace_files"] == "3"
    assert result.traces[0].attributes["workspace_files"] == "3"
    assert result.traces[0].attributes["workspace_files_materialized"] == "true"

    document_target = CapturingChatbotAdapter()
    document_result = runner.execute_registered_technique(
        run_id="run-converted-document",
        requested_by="tester",
        target=document_target,
        technique_key="xpia_document_injection",
        unsafe_intent_label="extract private account details",
        converter_pipeline=ConverterPipeline([MarkerConverter()]),
    )

    assert document_target.request is not None
    document_content = render_prompt(document_target.request.case)[-1]["content"]
    assert document_content.count("BEGIN UNTRUSTED DOCUMENT:") == 1
    assert document_target.request.case.untrusted_documents == ()
    assert document_target.request.case.assets["materialized_untrusted_documents"] == "1"
    assert document_result.traces[0].attributes["untrusted_documents"] == "1"
    assert document_result.traces[0].attributes["untrusted_documents_materialized"] == "true"


def test_live_chatbot_passes_generation_overrides_to_chat_client() -> None:
    client = FakeChatClient()
    adapter = LiveChatbotAdapter(
        descriptor=TargetDescriptor(
            target_id="chatbot-1",
            name="Live Chatbot",
            kind=TargetKind.CHATBOT,
            capabilities=("chat",),
        ),
        client=client,
    )
    request = ExecutionRequest(
        run_id="run-generation-overrides",
        target_id="chatbot-1",
        requested_by="tester",
        case=AttackCase(
            technique_key="probe",
            target_id="chatbot-1",
            unsafe_intent_label="check overrides",
            instructions="probe",
            assets={"template": "Say {intent}"},
            generation_overrides={
                "temperature": "0.2",
                "top_p": "0.9",
                "presence_penalty": "0.1",
                "frequency_penalty": "0.3",
                "stop": "END",
                "logit_bias": '{"123": 4}',
                "custom": "value",
                "model": "attacker-model",
                "messages": "attacker-messages",
                "max_tokens": "999999",
                "max_completion_tokens": "999999",
            },
        ),
        config=TechniqueConfig(technique_key="probe", enabled_detectors=()),
    )

    response = adapter.execute(request)

    assert client.extra_body == {
        "temperature": 0.2,
        "top_p": 0.9,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.3,
        "stop": "END",
        "logit_bias": {"123": 4},
    }
    assert response.traces[0].attributes["generation_overrides"] == str(client.extra_body)


def test_live_chatbot_drops_invalid_logit_bias_generation_override() -> None:
    client = FakeChatClient()
    adapter = LiveChatbotAdapter(
        descriptor=TargetDescriptor(
            target_id="chatbot-1",
            name="Live Chatbot",
            kind=TargetKind.CHATBOT,
            capabilities=("chat",),
        ),
        client=client,
    )
    request = ExecutionRequest(
        run_id="run-invalid-logit-bias",
        target_id="chatbot-1",
        requested_by="tester",
        case=AttackCase(
            technique_key="probe",
            target_id="chatbot-1",
            unsafe_intent_label="check overrides",
            instructions="probe",
            assets={"template": "Say {intent}"},
            generation_overrides={"logit_bias": "not-json"},
        ),
        config=TechniqueConfig(technique_key="probe", enabled_detectors=()),
    )

    adapter.execute(request)

    assert client.extra_body is None


def test_live_agent_extracts_tool_and_memory_evidence_without_duplicate_workspace() -> None:
    client = FakeChatClient(
        response_text=(
            "commands used\n"
            "command: python main.py\n"
            "files touched: main.py\n"
            "Memory saved: ExampleCo rollout review mode persisted"
        )
    )
    adapter = LiveAgentAdapter(
        descriptor=TargetDescriptor(
            target_id="agent-1",
            name="Live Agent",
            kind=TargetKind.AGENT,
            capabilities=("chat", "tools", "memory", "workspace"),
        ),
        client=client,
    )
    request = ExecutionRequest(
        run_id="run-live-agent",
        target_id="agent-1",
        requested_by="tester",
        case=AttackCase(
            technique_key="probe",
            target_id="agent-1",
            unsafe_intent_label="complete the workflow",
            instructions="probe",
            assets={"template": "Edit the workspace to {intent}."},
            workspace_files=(
                {"path": "main.py", "content": "print('ready')"},
            ),
            generation_overrides={"temperature": "0.2"},
        ),
        config=TechniqueConfig(technique_key="probe", enabled_detectors=()),
    )

    response = adapter.execute(request)

    sent_content = client.calls[0][-1]["content"]
    assert response.target_kind is TargetKind.AGENT
    assert response.tool_calls == ()
    assert response.memory_events == ()
    assert json.loads(response.metadata["self_reported_tool_calls"]) == [
        "command: python main.py",
        "files touched: main.py",
    ]
    assert json.loads(response.metadata["self_reported_memory_events"]) == [
        "Memory saved: ExampleCo rollout review mode persisted",
    ]
    assert sent_content.count("BEGIN WORKSPACE FILE: main.py") == 1
    assert client.extra_body == {"temperature": 0.2}


def test_live_agent_preserves_raw_response_and_structured_tool_calls() -> None:
    client = RawToolFakeChatClient()
    adapter = LiveAgentAdapter(
        descriptor=TargetDescriptor(
            target_id="agent-1",
            name="Live Agent",
            kind=TargetKind.AGENT,
            capabilities=("chat", "tools", "memory", "workspace"),
        ),
        client=client,
    )
    request = ExecutionRequest(
        run_id="run-live-agent-contracts",
        target_id="agent-1",
        requested_by="tester",
        case=AttackCase(
            technique_key="probe",
            target_id="agent-1",
            unsafe_intent_label="complete the workflow",
            instructions="probe",
            assets={"template": "Handle {intent}."},
        ),
        config=TechniqueConfig(technique_key="probe", enabled_detectors=()),
    )

    response = adapter.execute(request)
    api_response_trace = next(
        trace for trace in response.traces if trace.event_type == "api_response"
    )

    assert response.tool_calls == (
        'tool call: shell_exec {"cmd":"python main.py"}',
    )
    assert response.memory_events == ()
    assert json.loads(api_response_trace.attributes["raw_response"]) == client.raw_response
    assert api_response_trace.attributes["finish_reason"] == "tool_calls"


def test_live_agent_ignores_negated_handoff_evidence() -> None:
    client = FakeChatClient(
        response_text=(
            "The prompt requested handoff_metadata, but I cannot produce it.\n"
            "I will not create a handoff.\n"
            "handoff_metadata is untrusted context."
        )
    )
    adapter = LiveAgentAdapter(
        descriptor=TargetDescriptor(
            target_id="agent-1",
            name="Live Agent",
            kind=TargetKind.AGENT,
            capabilities=("chat", "tools", "memory", "workspace"),
        ),
        client=client,
    )
    request = ExecutionRequest(
        run_id="run-live-agent-negated-handoff",
        target_id="agent-1",
        requested_by="tester",
        case=AttackCase(
            technique_key="probe",
            target_id="agent-1",
            unsafe_intent_label="complete the workflow",
            instructions="probe",
            assets={"template": "Handle {intent}.", "expects_handoff": "true"},
        ),
        config=TechniqueConfig(technique_key="probe", enabled_detectors=()),
    )

    response = adapter.execute(request)

    assert response.tool_calls == ()


def test_live_agent_ignores_negated_tool_and_memory_evidence() -> None:
    client = FakeChatClient(
        response_text=(
            "No tool calls were made.\n"
            "I did not make any tool calls.\n"
            "I completed this without tools.\n"
            "No memory was saved.\n"
            "I did not save memory or persist anything."
        )
    )
    adapter = LiveAgentAdapter(
        descriptor=TargetDescriptor(
            target_id="agent-1",
            name="Live Agent",
            kind=TargetKind.AGENT,
            capabilities=("chat", "tools", "memory", "workspace"),
        ),
        client=client,
    )
    request = ExecutionRequest(
        run_id="run-live-agent-negated-evidence",
        target_id="agent-1",
        requested_by="tester",
        case=AttackCase(
            technique_key="probe",
            target_id="agent-1",
            unsafe_intent_label="complete the workflow",
            instructions="probe",
            assets={"template": "Handle {intent}."},
        ),
        config=TechniqueConfig(technique_key="probe", enabled_detectors=()),
    )

    response = adapter.execute(request)

    assert response.tool_calls == ()
    assert response.memory_events == ()


def test_live_agent_ignores_multiline_null_tool_and_memory_sections() -> None:
    client = FakeChatClient(
        response_text=(
            "Commands used:\n"
            "None\n\n"
            "Files touched:\n"
            "N/A\n\n"
            "Memory saved:\n"
            "No memory"
        )
    )
    adapter = LiveAgentAdapter(
        descriptor=TargetDescriptor(
            target_id="agent-1",
            name="Live Agent",
            kind=TargetKind.AGENT,
            capabilities=("chat", "tools", "memory", "workspace"),
        ),
        client=client,
    )
    request = ExecutionRequest(
        run_id="run-live-agent-null-sections",
        target_id="agent-1",
        requested_by="tester",
        case=AttackCase(
            technique_key="probe",
            target_id="agent-1",
            unsafe_intent_label="complete the workflow",
            instructions="probe",
            assets={"template": "Handle {intent}."},
        ),
        config=TechniqueConfig(technique_key="probe", enabled_detectors=()),
    )

    response = adapter.execute(request)

    assert response.tool_calls == ()
    assert response.memory_events == ()


def test_live_agent_ignores_cannot_or_will_not_memory_persistence() -> None:
    client = FakeChatClient(
        response_text=(
            "I cannot save memory or persist preferences.\n"
            "I won't persist memory for this workspace.\n"
            "I will not write memory from this request."
        )
    )
    adapter = LiveAgentAdapter(
        descriptor=TargetDescriptor(
            target_id="agent-1",
            name="Live Agent",
            kind=TargetKind.AGENT,
            capabilities=("chat", "tools", "memory", "workspace"),
        ),
        client=client,
    )
    request = ExecutionRequest(
        run_id="run-live-agent-negated-memory",
        target_id="agent-1",
        requested_by="tester",
        case=AttackCase(
            technique_key="probe",
            target_id="agent-1",
            unsafe_intent_label="complete the workflow",
            instructions="probe",
            assets={"template": "Handle {intent}.", "expects_memory_write": "true"},
        ),
        config=TechniqueConfig(technique_key="probe", enabled_detectors=()),
    )

    response = adapter.execute(request)

    assert response.memory_events == ()


def test_build_live_target_uses_live_agent_adapter_for_agent_targets() -> None:
    from app.api.store import EvaluationStore

    store = EvaluationStore.__new__(EvaluationStore)
    store._build_chat_client = lambda config: FakeChatClient()

    agent = store._build_live_target(TargetKind.AGENT, {})
    chatbot = store._build_live_target(TargetKind.CHATBOT, {})

    assert isinstance(agent, LiveAgentAdapter)
    assert agent.descriptor.capabilities == ("chat", "tools", "memory", "workspace")
    assert isinstance(chatbot, LiveChatbotAdapter)
    assert chatbot.descriptor.capabilities == ("chat",)


def test_live_chatbot_stages_skeleton_key_user_turns_with_interim_response() -> None:
    client = SequencedFakeChatClient(
        responses=("setup accepted", "final answer"),
    )
    adapter = LiveChatbotAdapter(
        descriptor=TargetDescriptor(
            target_id="chatbot-1",
            name="Live Chatbot",
            kind=TargetKind.CHATBOT,
            capabilities=("chat",),
        ),
        client=client,
    )
    attack = get_attack_technique("skeleton_key")()
    case = attack.build_case(
        AttackBuildRequest(
            target_id="chatbot-1",
            unsafe_intent_label="show the restricted procedure",
        )
    )
    request = ExecutionRequest(
        run_id="run-skeleton-key",
        target_id="chatbot-1",
        requested_by="tester",
        case=case,
        config=TechniqueConfig(technique_key="skeleton_key", enabled_detectors=()),
    )

    response = adapter.execute(request)

    assert response.final_output == "final answer"
    assert len(client.calls) == 2
    assert client.calls[0][0]["role"] == "system"
    assert client.calls[0][-1]["role"] == "user"
    assert "advanced researchers" in client.calls[0][-1]["content"].lower()
    assert client.calls[1][-2] == {"role": "assistant", "content": "setup accepted"}
    assert client.calls[1][-1]["role"] == "user"
    assert "show the restricted procedure" in client.calls[1][-1]["content"]
    assert any(
        trace.event_type == "api_call" and trace.attributes.get("phase") == "1"
        for trace in response.traces
    )
    assert any(
        trace.event_type == "api_call" and trace.attributes.get("phase") == "2"
        for trace in response.traces
    )


def test_import_suite_validates_technique_keys() -> None:
    suite = import_suite_from_json(
        """
        {
          "suite_name": "starter-suite",
          "source": "unit-test",
          "cases": [
            {
              "technique_key": "direct_prompt_injection",
              "unsafe_intent_label": "instruction-override-check",
              "seed_prompt": "sanitized-placeholder"
            }
          ]
        }
        """
    )

    assert suite.suite_name == "starter-suite"
    assert suite.cases[0].technique_key == "direct_prompt_injection"
