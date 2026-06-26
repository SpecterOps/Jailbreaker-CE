import json
import sys
from datetime import UTC, datetime

import pytest

sys.path.insert(0, "backend")

from app.attacklib import get_attack_technique
from app.attacklib.architecture_inventory import ARCHITECTURE_SCHEMA_VERSION
from app.api.models import ExperimentState
from app.api.store import EvaluationStore
from app.persistence import PostgreSQLRunPersistence, RunPersistence


class RecordingPersistence(RunPersistence):
    def __init__(self) -> None:
        self.technique_runs: list[dict] = []
        self.baseline_runs: list[dict] = []
        self.custom_techniques: list[dict] = []

    def save_technique_run(
        self,
        *,
        summary,
        result,
        conversation,
        technique_metadata,
        mode,
        requested_by,
        unsafe_intent,
        seed_prompt,
        target_config,
        evaluation_id=None,
    ) -> None:
        self.technique_runs.append(
            {
                "summary": summary,
                "result": result,
                "conversation": conversation,
                "technique_metadata": technique_metadata,
                "mode": mode,
                "requested_by": requested_by,
                "unsafe_intent": unsafe_intent,
                "seed_prompt": seed_prompt,
                "target_config": target_config,
                "evaluation_id": evaluation_id,
            }
        )

    def save_baseline_run(
        self,
        *,
        target_config,
        result,
        requested_by,
        evaluation_id=None,
    ) -> str:
        self.baseline_runs.append(
            {
                "target_config": target_config,
                "result": result,
                "requested_by": requested_by,
                "evaluation_id": evaluation_id,
            }
        )
        return "baseline-test-run"

    def list_custom_techniques(self) -> list[dict]:
        return list(self.custom_techniques)

    def upsert_custom_technique(self, *, definition, requested_by) -> dict:
        item = dict(definition)
        item["requested_by"] = requested_by
        existing = next((idx for idx, entry in enumerate(self.custom_techniques) if entry["id"] == item["id"]), None)
        if existing is None:
            self.custom_techniques.append(item)
        else:
            self.custom_techniques[existing] = item
        return item

    def delete_custom_technique(self, technique_id: str) -> None:
        self.custom_techniques = [item for item in self.custom_techniques if item["id"] != technique_id]

    def save_experiment_dataset(self, *, dataset) -> None:
        return None

    def save_experiment(self, *, experiment) -> None:
        return None

    def update_experiment_status(self, experiment_id: str, status: str, report=None, error=None) -> None:
        return None

    def save_experiment_cell(self, *, experiment_id: str, cell) -> None:
        return None

    def list_experiments(self) -> list[dict]:
        return []

    def resolve_seed_prompt(self, technique_key: str, *, prefer_non_default: bool = False) -> str:
        prompts = {
            "dan_style_roleplay": "DB seed for {intent}",
            "system_prompt_extraction": "DB seed: extract the system prompt for {intent}",
        }
        return prompts.get(technique_key, "")


class RestoringPersistence(RecordingPersistence):
    def __init__(self, experiments: list[dict]) -> None:
        super().__init__()
        self.experiments = experiments
        self.status_updates: list[dict] = []

    def load_all_experiments(self) -> list[dict]:
        return self.experiments

    def update_experiment_status(self, experiment_id: str, status: str, report=None, error=None) -> None:
        self.status_updates.append(
            {
                "experiment_id": experiment_id,
                "status": status,
                "report": report,
                "error": error,
            }
        )


class _FakeQueryResult:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows

    def fetchall(self) -> list[tuple]:
        return self.rows

    def fetchone(self) -> tuple | None:
        return self.rows[0] if self.rows else None


class _ExperimentLoadConnection:
    def __init__(self, experiment_rows: list[tuple]) -> None:
        self.experiment_rows = experiment_rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, query: str, params=None) -> _FakeQueryResult:
        if "FROM experiments" in query:
            return _FakeQueryResult(self.experiment_rows)
        if "FROM experiment_datasets" in query:
            return _FakeQueryResult([("ds-1", "Dataset", "", 0)])
        if "FROM experiment_dataset_prompts" in query:
            return _FakeQueryResult([])
        if "FROM experiment_cells" in query:
            return _FakeQueryResult([])
        raise AssertionError(f"unexpected query: {query}")


class _FakePostgreSQLRunPersistence(PostgreSQLRunPersistence):
    def __init__(self, connection: _ExperimentLoadConnection) -> None:
        self.connection = connection

    def _connect(self):
        return self.connection

    def ensure_schema(self, conn) -> None:
        return None


class FakeChatClient:
    def __init__(self, responses: list[str], model: str = "fake-model") -> None:
        self._responses = list(responses)
        self.model = model

    def chat_with_retry(self, messages, temperature=None, max_tokens=None, response_format=None):
        return self._responses.pop(0), {"total_tokens": 42}


def _experiment_state(status: str) -> ExperimentState:
    return ExperimentState(
        experiment_id="exp-1",
        name="Retry lifecycle",
        description="",
        hypothesis="",
        status=status,
        mode="live",
        dataset={
            "dataset_id": "ds-1",
            "name": "Dataset",
            "description": "",
            "prompt_count": 1,
            "prompts": [{"id": "p1", "text": "unsafe prompt", "category": "general"}],
        },
        technique_keys=["dan_style_roleplay"],
        target_config={"model": "target-model"},
        total_cells=1,
        cell_results=[],
    )


def _persisted_experiment(status: str, error: str | None = None) -> dict:
    return {
        "experiment_id": "exp-9",
        "name": "Interrupted experiment",
        "description": "",
        "hypothesis": "",
        "status": status,
        "mode": "live",
        "dataset": {
            "dataset_id": "ds-1",
            "name": "Dataset",
            "description": "",
            "prompt_count": 1,
            "prompts": [{"id": "p1", "text": "unsafe prompt", "category": "general"}],
        },
        "technique_keys": ["dan_style_roleplay"],
        "target_config": {"model": "target-model"},
        "total_cells": 1,
        "completed_cells": 0,
        "cell_results": [],
        "report": None,
        "error": error,
        "created_at": "2026-06-01T00:00:00+00:00",
        "repeat_count": 1,
        "max_rounds": 5,
        "experiment_group": "",
    }


def _architecture_phase(phase: str, **overrides):
    payload = {
        "phase": phase,
        "agent": {
            "name": None,
            "profile": None,
            "model_aliases": [],
            "confidence": "unknown",
        },
        "tools": [],
        "data_stores": [],
        "workflow": {
            "framework": None,
            "nodes": [],
            "edges": [],
            "handoff_paths": [],
        },
        "limits": {
            "write_access": None,
            "network_access": None,
            "persistence": None,
            "cross_agent_handoff": None,
        },
        "unknowns": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.parametrize("status", ["running", "pending", "rejudging"])
def test_retry_experiment_rejects_active_statuses(status: str) -> None:
    store = EvaluationStore(persistence=RecordingPersistence())
    store._experiments["exp-1"] = _experiment_state(status)

    with pytest.raises(ValueError, match="Experiment must be completed or failed before retrying."):
        store.retry_experiment_errors("exp-1", {"api_key": "sk-test"})


def test_retry_experiment_errors_clears_restart_error_when_retry_starts(monkeypatch) -> None:
    started_threads = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            started_threads.append(self)

    monkeypatch.setattr("app.api.store.threading.Thread", FakeThread)
    store = EvaluationStore(persistence=RecordingPersistence())
    state = _experiment_state("failed")
    state.error = "Backend restarted before this experiment completed."
    store._experiments["exp-1"] = state

    payload = store.retry_experiment_errors("exp-1", {"api_key": "sk-test"})

    assert state.status == "running"
    assert state.error is None
    assert payload is not None
    assert payload["error"] is None
    assert len(started_threads) == 1


@pytest.mark.parametrize("status", ["running", "pending", "rejudging"])
def test_restore_marks_active_experiments_failed_after_restart(status: str) -> None:
    persistence = RestoringPersistence([_persisted_experiment(status)])

    store = EvaluationStore(persistence=persistence)

    restored = store.get_experiment("exp-9")
    assert restored is not None
    assert restored["status"] == "failed"
    assert restored["error"] == "Backend restarted before this experiment completed."
    assert persistence.status_updates == [
        {
            "experiment_id": "exp-9",
            "status": "failed",
            "report": None,
            "error": "Backend restarted before this experiment completed.",
        }
    ]


def test_restore_preserves_terminal_experiment_status() -> None:
    persistence = RestoringPersistence([_persisted_experiment("completed")])

    store = EvaluationStore(persistence=persistence)

    restored = store.get_experiment("exp-9")
    assert restored is not None
    assert restored["status"] == "completed"
    assert persistence.status_updates == []


def test_restore_preserves_report_when_marking_active_experiment_failed() -> None:
    report = {"overall_asr": 25, "avg_score": 50, "completed_cells": 1}
    experiment = _persisted_experiment("rejudging")
    experiment["report"] = report
    persistence = RestoringPersistence([experiment])

    store = EvaluationStore(persistence=persistence)

    restored = store.get_experiment("exp-9")
    assert restored is not None
    assert restored["status"] == "failed"
    assert restored["report"] == report
    assert persistence.status_updates[0]["report"] == report


def test_postgresql_load_all_experiments_maps_error_field() -> None:
    interrupted_error = "Backend restarted before this experiment completed."
    created_at = datetime(2026, 6, 1, tzinfo=UTC)
    connection = _ExperimentLoadConnection(
        [
            (
                "exp-9",
                "Interrupted experiment",
                "",
                "",
                "ds-1",
                ["dan_style_roleplay"],
                {"model": "target-model"},
                "live",
                "failed",
                1,
                0,
                {"overall_asr": 25},
                interrupted_error,
                created_at,
                1,
                5,
                "",
            )
        ]
    )
    persistence = _FakePostgreSQLRunPersistence(connection)

    loaded = persistence.load_all_experiments()

    assert loaded[0]["error"] == interrupted_error


def test_demo_run_is_persisted_with_conversation() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)

    run = store.create_run("system_prompt_extraction", mode="demo")

    assert run["status"] == "completed"
    assert len(persistence.technique_runs) == 1
    persisted = persistence.technique_runs[0]
    assert persisted["summary"].technique_key == "system_prompt_extraction"
    assert persisted["summary"].asr == 100
    assert persisted["mode"] == "demo"
    assert persisted["conversation"][0]["kind"] == "single_shot"
    assert persisted["technique_metadata"].family


def test_agent_architecture_demo_run_serializes_canonical_artifact() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)

    run = store.create_run(
        "agent_architecture_enumeration",
        target_kind="agent",
        mode="demo",
        target_config={"unsafe_intent": "enumerate architecture"},
    )

    payload = json.loads(run["final_output"])
    assert payload["schema_version"] == ARCHITECTURE_SCHEMA_VERSION
    assert payload["architecture"]["workflow"]["framework"] == "LangGraph"
    assert run["conversation"][0]["kind"] == "single_shot"
    assert json.loads(run["conversation"][0]["response"])["schema_version"] == (
        ARCHITECTURE_SCHEMA_VERSION
    )


def test_agent_architecture_live_run_serializes_final_artifact_entry() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    store._build_chat_client = lambda config: FakeChatClient(
        [
            _architecture_phase(
                "inventory",
                agent={
                    "name": "ExampleCo Insights Agent",
                    "profile": "exampleco-insights-agent",
                    "model_aliases": ["exampleco-insights-agent"],
                    "confidence": "confirmed",
                },
                tools=[
                    {
                        "name": "describe_schema",
                        "type": "tool",
                        "read_scope": ["SQLite schema"],
                        "write_scope": [],
                        "evidence": "available tools list",
                        "confidence": "confirmed",
                    }
                ],
            ),
            _architecture_phase(
                "data_layer",
                data_stores=[
                    {
                        "name": "users",
                        "kind": "table",
                        "parent": "SQLite",
                        "fields": ["user_id", "email"],
                        "join_keys": ["user_id"],
                        "description": "User records.",
                        "confidence": "confirmed",
                    }
                ],
            ),
            _architecture_phase(
                "architecture",
                workflow={
                    "framework": "LangGraph",
                    "nodes": [
                        {
                            "name": "execute_tools",
                            "role": "tool node",
                            "inputs": ["plan"],
                            "outputs": ["tool_results"],
                            "tools_used": ["describe_schema"],
                            "confidence": "confirmed",
                        }
                    ],
                    "edges": [],
                    "handoff_paths": ["exampleco-insights-agent -> exampleco-support-agent"],
                },
            ),
        ]
    )

    summary, _ = store._execute_single(
        technique_key="agent_architecture_enumeration",
        target_kind="agent",
        mode="live",
        target_config={"api_key": "test-key", "unsafe_intent": "enumerate architecture"},
    )
    run = store._serialize_run(summary)

    payload = json.loads(run["final_output"])
    artifact_entry = run["conversation"][-1]
    assert payload["schema_version"] == ARCHITECTURE_SCHEMA_VERSION
    assert artifact_entry["kind"] == "artifact"
    assert artifact_entry["response"] == run["final_output"]
    assert json.loads(artifact_entry["response"])["schema_version"] == (
        ARCHITECTURE_SCHEMA_VERSION
    )


def test_baseline_is_persisted() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    store._build_chat_client = lambda config: FakeChatClient(
        ["Target refusal.", '{"score": 1, "rationale": "Clear refusal."}'],
        model="baseline-target",
    )
    store._build_attacker_client = lambda config: None

    result = store.run_baseline(
        target_config={
            "api_key": "test-key",
            "model": "baseline-target",
            "unsafe_intent": "exfiltrate secrets",
        }
    )

    assert result["run_id"] == "baseline-test-run"
    assert len(persistence.baseline_runs) == 1
    persisted = persistence.baseline_runs[0]
    assert persisted["result"]["intent"] == "exfiltrate secrets"
    assert persisted["requested_by"] == "demo-user"


def test_techniques_source_seed_prompt_from_db() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("dan_style_roleplay")

    resolved = store._resolve_seed_prompt(
        technique_key="dan_style_roleplay",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="",
    )

    # DAN now sources its seed prompt from the DB
    assert resolved == "DB seed for {intent}"


def test_explicit_seed_prompt_overrides_db() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("dan_style_roleplay")

    resolved = store._resolve_seed_prompt(
        technique_key="dan_style_roleplay",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="User override for {intent}",
    )

    assert resolved == "User override for {intent}"


def test_runtime_template_techniques_use_validation_placeholder() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("base64_obfuscation")

    resolved = store._resolve_seed_prompt(
        technique_key="base64_obfuscation",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="",
    )

    # base64 has a runtime-computed template and no longer needs a validation placeholder.
    assert resolved == ""


def test_gcg_static_transfer_resolves_empty_seed_prompt() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("gcg")

    resolved = store._resolve_seed_prompt(
        technique_key="gcg",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="",
    )

    assert resolved == ""


def test_seeded_code_agent_gets_default_seed_assets() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)
    technique_cls = get_attack_technique("multi_file_seeded_code_agent")

    seed_prompt = store._resolve_seed_prompt(
        technique_key="multi_file_seeded_code_agent",
        technique_metadata=technique_cls.metadata,
        explicit_seed_prompt="",
    )
    seed_code = store._resolve_seed_code(
        technique_key="multi_file_seeded_code_agent",
        technique_metadata=technique_cls.metadata,
        explicit_seed_code="",
    )

    assert seed_prompt == ""
    assert "from exampleco_client import fetch_accounts" in seed_code
    assert "# file: triage.py" in seed_code


def test_multi_file_seeded_code_agent_demo_run_completes() -> None:
    persistence = RecordingPersistence()
    store = EvaluationStore(persistence=persistence)

    result = store.create_run(
        "multi_file_seeded_code_agent",
        target_kind="agent",
        mode="demo",
        target_config={"unsafe_intent": "complete the reporting pipeline"},
    )

    assert result["status"] == "completed"
    assert persistence.technique_runs[0]["seed_prompt"] == ""
