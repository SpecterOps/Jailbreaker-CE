import sys

sys.path.insert(0, "backend")

import app.execution.runner as runner_module
from app.api.store import EvaluationStore, ExperimentState
from app.attacklib.base import (
    AttackSurface,
    AttackTechniqueMetadata,
    AutomationLevel,
    IterationRound,
    IterativeAttackResult,
)
from app.evaluation.models import TargetKind, TargetResponse
from app.execution import TechniqueRunner
from app.llm_client import ChatClient
from app.persistence import NoopRunPersistence
from app.targets import TargetDescriptor


class RecordingChatClient:
    def __init__(self, responses: list[str] | None = None, *, model: str) -> None:
        self.responses = list(responses or [])
        self.model = model
        self.calls: list[list[dict[str, str]]] = []

    def chat_with_retry(
        self,
        messages,
        temperature=None,
        max_tokens=None,
        response_format=None,
    ):
        self.calls.append([dict(message) for message in messages])
        response = self.responses.pop(0) if self.responses else '{"score": 1, "rationale": "default"}'
        return response, {"total_tokens": "1"}


class FixedTargetAdapter:
    descriptor = TargetDescriptor(
        target_id="live-chatbot",
        name="Live Chatbot",
        kind=TargetKind.CHATBOT,
        capabilities=("chat",),
    )

    def execute(self, request):
        return TargetResponse(
            target_id=self.descriptor.target_id,
            target_kind=TargetKind.CHATBOT,
            final_output="The target provided detailed procedural content.",
            metadata={"adapter": "fixed"},
        )


def test_build_judge_client_uses_custom_config() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    target = ChatClient(model="target-model")
    attacker = ChatClient(model="attacker-model")

    judge = store._build_judge_client(
        {
            "judge_mode": "custom",
            "judge_api_key": "judge-key",
            "judge_api_base_url": "http://judge.test/v1",
            "judge_model": "judge-model",
            "judge_system_prompt": "Use the strict review profile.",
        },
        target_client=target,
        attacker_client=attacker,
    )

    assert judge is not None
    assert judge.model == "judge-model"
    assert judge.api_base_url == "http://judge.test/v1"
    assert judge.api_key == "judge-key"
    assert judge.system_prompt == "Use the strict review profile."


def test_build_attacker_client_uses_run_system_prompt() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())

    attacker = store._build_attacker_client(
        {
            "attacker_api_key": "attacker-key",
            "attacker_api_base_url": "http://attacker.test/v1",
            "attacker_model": "attacker-model",
            "attacker_system_prompt": "Generate concise candidates.",
        }
    )

    assert attacker is not None
    assert attacker.system_prompt == "Generate concise candidates."


def test_target_selected_as_judge_uses_independent_run_system_prompt() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    target = ChatClient(model="target-model", system_prompt="Target instructions.")

    judge = store._build_judge_client(
        {
            "judge_mode": "target",
            "judge_system_prompt": "Judge instructions.",
        },
        target_client=target,
        attacker_client=None,
    )

    assert judge is not target
    assert judge.system_prompt == "Judge instructions."
    assert target.system_prompt == "Target instructions."


def test_attacker_selected_as_judge_does_not_inherit_attacker_system_prompt() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    target = ChatClient(model="target-model", system_prompt="Target instructions.")
    attacker = ChatClient(model="attacker-model", system_prompt="Attacker instructions.")

    judge = store._build_judge_client(
        {"judge_mode": "attacker"},
        target_client=target,
        attacker_client=attacker,
    )

    assert judge is not attacker
    assert judge.system_prompt == ""
    assert attacker.system_prompt == "Attacker instructions."


def test_localhost_urls_are_rewritten_for_container_clients() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())

    target = store._build_chat_client(
        {
            "api_base_url": "http://localhost:1234/v1",
            "api_key": "not-needed",
            "model": "target-model",
        }
    )
    attacker = store._build_attacker_client(
        {
            "attacker_api_base_url": "http://127.0.0.1:1234/v1",
            "attacker_api_key": "lm-studio",
            "attacker_model": "attacker-model",
        }
    )
    judge = store._build_judge_client(
        {
            "judge_mode": "custom",
            "judge_local": True,
            "judge_api_base_url": "http://localhost:1234/v1",
            "judge_api_key": "lm-studio",
            "judge_model": "judge-model",
        },
        target_client=target,
        attacker_client=attacker,
    )

    assert target.api_base_url == "http://host.docker.internal:1234/v1"
    assert attacker is not None
    assert attacker.api_base_url == "http://host.docker.internal:1234/v1"
    assert judge.api_base_url == "http://host.docker.internal:1234/v1"


def test_list_experiments_uses_display_judge_model_helper() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    store._experiments["exp-1"] = ExperimentState(
        experiment_id="exp-1",
        name="Judge model display",
        description="",
        hypothesis="",
        status="completed",
        mode="live",
        dataset={"name": "dataset", "prompts": []},
        technique_keys=["dan_style_roleplay"],
        target_config={
            "model": "target-model",
            "judge_mode": "custom",
            "judge_model": "judge-model",
        },
        total_cells=0,
    )

    experiments = store.list_experiments()

    assert experiments[0]["judge_model"] == "judge-model"


def test_build_judge_client_supports_target_attacker_and_legacy_modes() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    target = ChatClient(model="target-model")
    attacker = ChatClient(model="attacker-model")

    target_judge = store._build_judge_client(
        {"judge_mode": "target"},
        target_client=target,
        attacker_client=attacker,
    )
    attacker_judge = store._build_judge_client(
        {"judge_mode": "attacker"},
        target_client=target,
        attacker_client=attacker,
    )
    legacy_attacker_judge = store._build_judge_client(
        {},
        target_client=target,
        attacker_client=attacker,
    )
    legacy_target_judge = store._build_judge_client(
        {},
        target_client=target,
        attacker_client=None,
    )

    assert target_judge is not target
    assert target_judge.model == "target-model"
    assert attacker_judge is not attacker
    assert attacker_judge.model == "attacker-model"
    assert legacy_attacker_judge is not attacker
    assert legacy_attacker_judge.model == "attacker-model"
    assert legacy_target_judge is not target
    assert legacy_target_judge.model == "target-model"


def test_baseline_uses_selected_custom_judge() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    target = RecordingChatClient(["Target response."], model="target-model")
    attacker = RecordingChatClient(
        ['{"score": 2, "rationale": "attacker judge"}'],
        model="attacker-model",
    )
    judge = RecordingChatClient(
        ['{"score": 8, "rationale": "custom judge"}'],
        model="judge-model",
    )
    store._build_chat_client = lambda config: target
    store._build_attacker_client = lambda config: attacker
    store._build_judge_client = lambda config, *, target_client, attacker_client: judge

    result = store.run_baseline(
        target_config={
            "api_key": "target-key",
            "model": "target-model",
            "unsafe_intent": "exfiltrate secrets",
            "judge_mode": "custom",
        }
    )

    assert result["judge_score"] == 8
    assert len(target.calls) == 1
    assert len(attacker.calls) == 0
    assert len(judge.calls) == 1


def test_single_shot_live_uses_selected_custom_judge() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    target_client = RecordingChatClient(model="target-model")
    attacker = RecordingChatClient(
        ['{"score": 2, "rationale": "attacker judge"}'],
        model="attacker-model",
    )
    judge = RecordingChatClient(
        ['{"score": 8, "rationale": "custom judge"}'],
        model="judge-model",
    )
    store._build_chat_client = lambda config: target_client
    store._build_attacker_client = lambda config: attacker
    store._build_live_target = lambda *, target_kind, config: FixedTargetAdapter()
    store._build_judge_client = lambda config, *, target_client, attacker_client: judge

    summary, _ = store._execute_single(
        technique_key="dan_style_roleplay",
        target_kind="chatbot",
        mode="live",
        target_config={
            "api_key": "target-key",
            "model": "target-model",
            "unsafe_intent": "exfiltrate secrets",
            "judge_mode": "custom",
        },
    )

    assert summary.score == 80
    assert len(attacker.calls) == 0
    assert len(judge.calls) == 1


def test_iterative_runner_passes_distinct_judge_client(monkeypatch) -> None:
    captured = {}

    class FakeIterativeTechnique:
        metadata = AttackTechniqueMetadata(
            key="pair",
            name="PAIR",
            surface=AttackSurface.TEXT,
            family="feedback",
            summary="test",
            source_paper="test",
            automation_level=AutomationLevel.AUTOMATIC,
            requires_feedback_loop=True,
            requires_attacker_llm=True,
            requires_judge_per_round=True,
        )

        def run_iterative(self, **kwargs):
            captured.update(kwargs)
            return IterativeAttackResult(
                rounds=(
                    IterationRound(
                        round_number=1,
                        target_response="response",
                        judge_score=8,
                        judge_rationale="custom judge",
                        succeeded=True,
                    ),
                ),
                best_round_number=1,
                final_prompt="prompt",
                final_response="response",
                succeeded=True,
                total_api_calls=3,
            )

    monkeypatch.setattr(runner_module, "get_attack_technique", lambda key: FakeIterativeTechnique)

    target = object()
    attacker = object()
    judge = object()

    TechniqueRunner().execute_iterative_technique(
        run_id="run-judge",
        requested_by="tester",
        target_client=target,
        attacker_client=attacker,
        judge_client=judge,
        technique_key="pair",
        unsafe_intent_label="exfiltrate secrets",
    )

    assert captured["target_client"] is target
    assert captured["attacker_client"] is attacker
    assert captured["judge_client"] is judge
