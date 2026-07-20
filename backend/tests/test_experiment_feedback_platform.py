import sys

sys.path.insert(0, "backend")

from app.api.models import RunSummary
from app.api.store import EvaluationStore, ExperimentState
from app.persistence import NoopRunPersistence


class RecordingJudgeClient:
    model = "third-judge"
    api_base_url = "http://judge.test/v1"

    def __init__(self, scores: list[int]) -> None:
        self.scores = list(scores)
        self.calls: list[list[dict[str, str]]] = []

    def chat_with_retry(self, messages, temperature=None, max_tokens=None, response_format=None):
        self.calls.append([dict(message) for message in messages])
        score = self.scores.pop(0)
        return f'{{"score": {score}, "rationale": "score {score}"}}', {"total_tokens": "1"}


def _experiment_state(*, repeat_count: int = 3) -> ExperimentState:
    return ExperimentState(
        experiment_id="exp-repeat",
        name="Repeated Crescendo-Liar",
        description="",
        hypothesis="",
        status="running",
        mode="live",
        dataset={
            "dataset_id": "ds-1",
            "name": "dataset",
            "prompts": [
                {"id": "p1", "text": "intent one", "category": "cat-a"},
                {"id": "p2", "text": "intent two", "category": "cat-b"},
            ],
        },
        technique_keys=["crescendo_liar"],
        target_config={"model": "target-model"},
        total_cells=2 * repeat_count,
        repeat_count=repeat_count,
        max_rounds=5,
        experiment_group="crescendo-liar-main",
    )


def test_expected_experiment_cells_include_repeat_index() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=3)

    cells = list(store._iter_expected_experiment_cells(state))

    assert [(tech, prompt["id"], repeat) for tech, prompt, repeat in cells] == [
        ("crescendo_liar", "p1", 1),
        ("crescendo_liar", "p1", 2),
        ("crescendo_liar", "p1", 3),
        ("crescendo_liar", "p2", 1),
        ("crescendo_liar", "p2", 2),
        ("crescendo_liar", "p2", 3),
    ]


def test_experiment_report_includes_wilson_ci_and_score_stddev() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())

    report = store._build_experiment_report(
        [
            {"technique_key": "crescendo_liar", "prompt_category": "cat", "status": "completed", "score": 100, "asr": 100},
            {"technique_key": "crescendo_liar", "prompt_category": "cat", "status": "completed", "score": 80, "asr": 100},
            {"technique_key": "crescendo_liar", "prompt_category": "cat", "status": "completed", "score": 70, "asr": 100},
            {"technique_key": "crescendo_liar", "prompt_category": "cat", "status": "completed", "score": 20, "asr": 0},
        ]
    )

    assert report["tested"] == 4
    assert report["overall_asr"] == 75
    assert report["asr_ci_low"] == 30.1
    assert report["asr_ci_high"] == 95.4
    assert report["score_stddev"] == 29.5
    assert report["technique_stats"]["crescendo_liar"]["asr_ci_low"] == 30.1


def test_experiment_dataset_prompt_does_not_replace_technique_template() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=1)
    state.technique_keys = ["aim"]
    state.dataset["prompts"] = [{"id": "p1", "text": "dataset intent", "category": "cat-a"}]
    state.total_cells = 1
    store._experiments[state.experiment_id] = state
    captured_configs = []

    def fake_execute_single(**kwargs):
        captured_configs.append(kwargs["target_config"])
        return RunSummary(
            run_id="run-aim",
            technique_key="aim",
            target_kind="chatbot",
            status="completed",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    store._execute_single = fake_execute_single
    store._run_experiment(
        state.experiment_id,
        {"model": "target-model", "seed_prompt": "must not replace AIM"},
        "target-key",
    )

    assert captured_configs[0]["unsafe_intent"] == "dataset intent"
    assert "seed_prompt" not in captured_configs[0]


def test_experiment_retry_does_not_restore_dataset_prompt_as_seed() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=1)
    state.technique_keys = ["aim"]
    state.dataset["prompts"] = [{"id": "p1", "text": "dataset intent", "category": "cat-a"}]
    state.total_cells = 1
    store._experiments[state.experiment_id] = state
    captured_configs = []

    def fake_execute_single(**kwargs):
        captured_configs.append(kwargs["target_config"])
        return RunSummary(
            run_id="run-aim-retry",
            technique_key="aim",
            target_kind="chatbot",
            status="completed",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[],
        ), object()

    store._execute_single = fake_execute_single
    store._run_experiment_retry(
        state.experiment_id,
        [("aim", state.dataset["prompts"][0], 1)],
        {"model": "target-model", "seed_prompt": "must not replace AIM"},
        "target-key",
    )

    assert captured_configs[0]["unsafe_intent"] == "dataset intent"
    assert "seed_prompt" not in captured_configs[0]


def test_experiment_keeps_judge_failure_cells_retryable_with_conversation() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=1)
    state.technique_keys = ["aim"]
    state.dataset["prompts"] = [{"id": "p1", "text": "dataset intent", "category": "cat-a"}]
    state.total_cells = 1
    store._experiments[state.experiment_id] = state

    def fake_execute_single(**kwargs):
        return RunSummary(
            run_id="run-judge-error",
            technique_key="aim",
            target_kind="chatbot",
            status="error",
            score=0,
            asr=0,
            matched_detections=0,
            detections=(),
            highest_severity="info",
            conversation=[
                {
                    "kind": "single_shot",
                    "response": "Preserved target response.",
                    "judge": {"error": "judge provider rejected the request"},
                }
            ],
            error="judge provider rejected the request",
        ), object()

    store._execute_single = fake_execute_single
    store._run_experiment(state.experiment_id, {"model": "target-model"}, "target-key")

    cell = state.cell_results[0]
    assert cell["status"] == "error"
    assert cell["error"] == "judge provider rejected the request"
    assert cell["conversation"][0]["response"] == "Preserved target response."


def test_experiment_metrics_do_not_emit_removed_ablation() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=1)
    state.cell_results = [
        {
            "technique_key": "crescendo_liar",
            "prompt_id": "p1",
            "prompt_text": "intent one",
            "prompt_category": "cat-a",
            "repeat_index": 1,
            "status": "completed",
            "score": 80,
            "asr": 100,
            "conversation": [],
        }
    ]
    store._experiments[state.experiment_id] = state

    metrics = store.compute_experiment_metrics(state.experiment_id)

    assert metrics is not None
    assert metrics["ablation"] == {}


def test_rejudge_scores_saved_target_and_fabricator_outputs() -> None:
    store = EvaluationStore(persistence=NoopRunPersistence())
    state = _experiment_state(repeat_count=1)
    state.cell_results = [
        {
            "technique_key": "crescendo_liar",
            "prompt_id": "p1",
            "prompt_text": "intent one",
            "prompt_category": "cat-a",
            "repeat_index": 1,
            "status": "completed",
            "score": 0,
            "asr": 0,
            "conversation": [
                {
                    "kind": "round",
                    "round_number": 1,
                    "target": {
                        "response": "target refusal",
                        "poisoned": "fabricated harmful response",
                    },
                }
            ],
        }
    ]
    store._experiments[state.experiment_id] = state
    judge = RecordingJudgeClient([2, 8])
    store._build_judge_client = lambda config, *, target_client, attacker_client: judge

    store._run_experiment_rejudge(
        state.experiment_id,
        {
            "judge_mode": "custom",
            "judge_api_key": "judge-key",
            "judge_api_base_url": "http://judge.test/v1",
            "judge_model": "third-judge",
        },
    )

    result = state.cell_results[0]["rejudge_results"][0]
    assert len(judge.calls) == 2
    assert result["judge_model"] == "third-judge"
    assert result["target_score"] == 2
    assert result["fabricator_score"] == 8
    assert result["target_asr"] == 0
    assert result["fabricator_asr"] == 100
    assert result["pipeline_unsafe_asr"] == 100
