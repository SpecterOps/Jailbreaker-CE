from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.attacklib.base import AttackTechniqueMetadata
from app.evaluation.models import TechniqueRunResult

if TYPE_CHECKING:
    from app.api.models import RunSummary

log = logging.getLogger("persistence.runs")


class RunPersistence(ABC):
    @abstractmethod
    def save_technique_run(
        self,
        *,
        summary: Any,
        result: TechniqueRunResult,
        conversation: list[dict[str, Any]],
        technique_metadata: AttackTechniqueMetadata,
        mode: str,
        requested_by: str,
        unsafe_intent: str,
        seed_prompt: str,
        target_config: dict[str, str],
        evaluation_id: str | None = None,
    ) -> None:
        """Persist one technique run."""

    @abstractmethod
    def save_baseline_run(
        self,
        *,
        target_config: dict[str, str],
        result: dict[str, Any],
        requested_by: str,
        evaluation_id: str | None = None,
    ) -> str:
        """Persist one baseline control run and return its run ID."""

    @abstractmethod
    def list_custom_techniques(self) -> list[dict[str, Any]]:
        """Return persisted custom technique definitions."""

    @abstractmethod
    def upsert_custom_technique(
        self,
        *,
        definition: dict[str, Any],
        requested_by: str,
    ) -> dict[str, Any]:
        """Create or update one custom technique definition."""

    @abstractmethod
    def delete_custom_technique(self, technique_id: str) -> None:
        """Delete one custom technique definition."""

    @abstractmethod
    def save_experiment_dataset(
        self,
        *,
        dataset: dict[str, Any],
    ) -> None:
        """Persist a dataset with its prompts."""

    @abstractmethod
    def save_experiment(
        self,
        *,
        experiment: dict[str, Any],
    ) -> None:
        """Persist experiment metadata."""

    @abstractmethod
    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update experiment status and optionally its report."""

    @abstractmethod
    def save_experiment_cell(
        self,
        *,
        experiment_id: str,
        cell: dict[str, Any],
    ) -> None:
        """Persist one experiment cell result."""

    @abstractmethod
    def list_experiments(self) -> list[dict[str, Any]]:
        """Return persisted experiment summaries."""

    def load_all_experiments(self) -> list[dict[str, Any]]:
        """Load full experiment state from DB (cells + conversations). Override in DB backends."""
        return []

    def list_run_history(
        self,
        *,
        filters: dict[str, str],
        limit: int = 50,
        offset: int = 0,
        sort: str = "completed_desc",
    ) -> dict[str, Any]:
        """Return persisted one-off technique runs for operator history."""
        return {"items": [], "total": 0, "limit": limit, "offset": offset, "facets": {}}

    def get_run_history_item(self, run_id: str) -> dict[str, Any] | None:
        """Return one persisted one-off technique run with child evidence."""
        return None

    @abstractmethod
    def resolve_seed_prompt(self, technique_key: str, *, prefer_non_default: bool = False) -> str:
        """Return a DB-backed seed prompt for a technique.

        When *prefer_non_default* is True (seed-agnostic techniques), the
        query favours non-default seeds so the prompt varies across runs.
        Otherwise it returns the canonical default seed first.
        """


class NoopRunPersistence(RunPersistence):
    def save_technique_run(
        self,
        *,
        summary: Any,
        result: TechniqueRunResult,
        conversation: list[dict[str, Any]],
        technique_metadata: AttackTechniqueMetadata,
        mode: str,
        requested_by: str,
        unsafe_intent: str,
        seed_prompt: str,
        target_config: dict[str, str],
        evaluation_id: str | None = None,
    ) -> None:
        return

    def save_baseline_run(
        self,
        *,
        target_config: dict[str, str],
        result: dict[str, Any],
        requested_by: str,
        evaluation_id: str | None = None,
    ) -> str:
        return ""

    def list_custom_techniques(self) -> list[dict[str, Any]]:
        return []

    def upsert_custom_technique(
        self,
        *,
        definition: dict[str, Any],
        requested_by: str,
    ) -> dict[str, Any]:
        return dict(definition)

    def delete_custom_technique(self, technique_id: str) -> None:
        return

    def save_experiment_dataset(self, *, dataset: dict[str, Any]) -> None:
        return

    def save_experiment(self, *, experiment: dict[str, Any]) -> None:
        return

    def update_experiment_status(
        self, experiment_id: str, status: str,
        report: dict[str, Any] | None = None, error: str | None = None,
    ) -> None:
        return

    def save_experiment_cell(self, *, experiment_id: str, cell: dict[str, Any]) -> None:
        return

    def list_experiments(self) -> list[dict[str, Any]]:
        return []

    def resolve_seed_prompt(self, technique_key: str, *, prefer_non_default: bool = False) -> str:
        return ""


class PostgreSQLRunPersistence(RunPersistence):
    def __init__(self, dsn: str | None = None) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise ImportError(
                "PostgreSQLRunPersistence requires the 'psycopg' package. "
                "Install it with: uv sync --extra postgres"
            ) from None
        self._dsn = dsn or os.environ.get(
            "DATABASE_URL",
            "postgresql://jailbreaker:jailbreaker@localhost:5432/jailbreaker",
        )

    def save_technique_run(
        self,
        *,
        summary: RunSummary,
        result: TechniqueRunResult,
        conversation: list[dict[str, Any]],
        technique_metadata: AttackTechniqueMetadata,
        mode: str,
        requested_by: str,
        unsafe_intent: str,
        seed_prompt: str,
        target_config: dict[str, str],
        evaluation_id: str | None = None,
    ) -> None:
        target_model = target_config.get("model", result.response.metadata.get("model", ""))
        attacker_model = _resolve_attacker_model(target_config, mode)
        final_prompt = _extract_final_prompt(conversation)
        final_output = result.response.final_output
        run_metadata = {
            "config": _sanitize_config(target_config),
            "response_metadata": dict(result.response.metadata),
            "tool_calls": list(result.response.tool_calls),
            "memory_events": list(result.response.memory_events),
            "audit_log": [_serialize_audit_log_event(item) for item in result.audit_log],
            "requires_feedback_loop": technique_metadata.requires_feedback_loop,
            "requires_attacker_llm": technique_metadata.requires_attacker_llm,
            "requires_judge_per_round": technique_metadata.requires_judge_per_round,
            "multi_turn": technique_metadata.multi_turn,
            "tags": list(technique_metadata.tags),
        }

        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, run_kind, evaluation_id, status, mode, requested_by,
                    technique_key, technique_name, technique_family, technique_surface,
                    target_kind, target_id, target_model, target_base_url,
                    attacker_model, attacker_base_url,
                    unsafe_intent, system_prompt, seed_prompt,
                    success, score, asr, matched_detections, highest_severity,
                    final_prompt, final_output, response_metadata, run_metadata,
                    created_at, completed_at
                ) VALUES (
                    %s, 'technique', %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s::jsonb, %s::jsonb,
                    %s, %s
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    evaluation_id = EXCLUDED.evaluation_id,
                    status = EXCLUDED.status,
                    mode = EXCLUDED.mode,
                    requested_by = EXCLUDED.requested_by,
                    technique_key = EXCLUDED.technique_key,
                    technique_name = EXCLUDED.technique_name,
                    technique_family = EXCLUDED.technique_family,
                    technique_surface = EXCLUDED.technique_surface,
                    target_kind = EXCLUDED.target_kind,
                    target_id = EXCLUDED.target_id,
                    target_model = EXCLUDED.target_model,
                    target_base_url = EXCLUDED.target_base_url,
                    attacker_model = EXCLUDED.attacker_model,
                    attacker_base_url = EXCLUDED.attacker_base_url,
                    unsafe_intent = EXCLUDED.unsafe_intent,
                    system_prompt = EXCLUDED.system_prompt,
                    seed_prompt = EXCLUDED.seed_prompt,
                    success = EXCLUDED.success,
                    score = EXCLUDED.score,
                    asr = EXCLUDED.asr,
                    matched_detections = EXCLUDED.matched_detections,
                    highest_severity = EXCLUDED.highest_severity,
                    final_prompt = EXCLUDED.final_prompt,
                    final_output = EXCLUDED.final_output,
                    response_metadata = EXCLUDED.response_metadata,
                    run_metadata = EXCLUDED.run_metadata,
                    completed_at = EXCLUDED.completed_at
                """,
                (
                    summary.run_id,
                    evaluation_id,
                    summary.status,
                    mode,
                    requested_by,
                    summary.technique_key,
                    technique_metadata.name,
                    technique_metadata.family,
                    technique_metadata.surface.value,
                    summary.target_kind,
                    result.target_id,
                    target_model,
                    target_config.get("api_base_url", ""),
                    attacker_model,
                    target_config.get("attacker_api_base_url", ""),
                    unsafe_intent,
                    target_config.get("system_prompt", "You are a helpful assistant."),
                    seed_prompt,
                    summary.asr == 100,
                    summary.score,
                    summary.asr,
                    summary.matched_detections,
                    summary.highest_severity,
                    final_prompt,
                    final_output,
                    json.dumps(dict(result.response.metadata)),
                    json.dumps(run_metadata),
                    _trace_created_at(result),
                    result.completed_at.isoformat(),
                ),
            )
            self._replace_detections(conn, summary.run_id, result)
            self._replace_conversation(conn, summary.run_id, conversation)
            self._replace_traces(conn, summary.run_id, result)

    def save_baseline_run(
        self,
        *,
        target_config: dict[str, str],
        result: dict[str, Any],
        requested_by: str,
        evaluation_id: str | None = None,
    ) -> str:
        run_id = f"baseline-{uuid.uuid4().hex[:12]}"
        conversation = [{
            "kind": "single_shot",
            "prompt": result.get("prompt", ""),
            "response": result.get("response", ""),
            "judge": {
                "score": str(result.get("judge_score", "")),
                "rationale": result.get("judge_rationale", ""),
                "input": "",
                "output": "",
            },
        }]
        run_metadata = {
            "config": _sanitize_config(target_config),
            "succeeded": bool(result.get("succeeded", False)),
            "judge_score": int(result.get("judge_score", 0)),
            "judge_rationale": result.get("judge_rationale", ""),
        }
        score = int(round((int(result.get("judge_score", 0)) / 10) * 100))
        asr = 100 if result.get("succeeded") else 0
        now = datetime.now(UTC).isoformat()

        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    run_id, run_kind, evaluation_id, status, mode, requested_by,
                    technique_key, technique_name, technique_family, technique_surface,
                    target_kind, target_id, target_model, target_base_url,
                    attacker_model, attacker_base_url,
                    unsafe_intent, system_prompt, seed_prompt,
                    success, score, asr, matched_detections, highest_severity,
                    final_prompt, final_output, response_metadata, run_metadata,
                    created_at, completed_at
                ) VALUES (
                    %s, 'baseline', %s, 'completed', 'live', %s,
                    'baseline', 'Baseline', 'control', 'control',
                    'chatbot', 'baseline', %s, %s,
                    %s, %s,
                    %s, %s, '',
                    %s, %s, %s, 0, 'info',
                    %s, %s, '{}'::jsonb, %s::jsonb,
                    %s, %s
                )
                """,
                (
                    run_id,
                    evaluation_id,
                    requested_by,
                    target_config.get("model", ""),
                    target_config.get("api_base_url", ""),
                    _resolve_attacker_model(target_config, "live"),
                    target_config.get("attacker_api_base_url", ""),
                    result.get("intent", ""),
                    target_config.get("system_prompt", "You are a helpful assistant."),
                    bool(result.get("succeeded", False)),
                    score,
                    asr,
                    result.get("prompt", ""),
                    result.get("response", ""),
                    json.dumps(run_metadata),
                    now,
                    now,
                ),
            )
            self._replace_conversation(conn, run_id, conversation)

        return run_id

    def list_custom_techniques(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT
                    technique_id,
                    name,
                    summary,
                    target_system_prompt,
                    mode,
                    prompt_template,
                    attacker_system_prompt,
                    refinement_template,
                    max_rounds,
                    created_at,
                    updated_at,
                    requested_by
                FROM custom_techniques
                ORDER BY updated_at DESC, technique_id ASC
                """
            ).fetchall()
        return [
            {
                "id": row[0],
                "name": row[1],
                "summary": row[2],
                "target_system_prompt": row[3],
                "mode": row[4],
                "prompt_template": row[5],
                "attacker_system_prompt": row[6],
                "refinement_template": row[7],
                "max_rounds": row[8],
                "created_at": row[9].isoformat() if row[9] else "",
                "updated_at": row[10].isoformat() if row[10] else "",
                "requested_by": row[11],
            }
            for row in rows
        ]

    def upsert_custom_technique(
        self,
        *,
        definition: dict[str, Any],
        requested_by: str,
    ) -> dict[str, Any]:
        technique_id = str(definition.get("id", "")).strip()
        if not technique_id:
            raise ValueError("Custom technique `id` is required.")

        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                INSERT INTO custom_techniques (
                    technique_id,
                    name,
                    summary,
                    target_system_prompt,
                    mode,
                    prompt_template,
                    attacker_system_prompt,
                    refinement_template,
                    max_rounds,
                    requested_by
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (technique_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    summary = EXCLUDED.summary,
                    target_system_prompt = EXCLUDED.target_system_prompt,
                    mode = EXCLUDED.mode,
                    prompt_template = EXCLUDED.prompt_template,
                    attacker_system_prompt = EXCLUDED.attacker_system_prompt,
                    refinement_template = EXCLUDED.refinement_template,
                    max_rounds = EXCLUDED.max_rounds,
                    requested_by = EXCLUDED.requested_by,
                    updated_at = now()
                RETURNING
                    technique_id,
                    name,
                    summary,
                    target_system_prompt,
                    mode,
                    prompt_template,
                    attacker_system_prompt,
                    refinement_template,
                    max_rounds,
                    created_at,
                    updated_at,
                    requested_by
                """,
                (
                    technique_id,
                    str(definition.get("name", "")).strip(),
                    str(definition.get("summary", "")).strip(),
                    str(definition.get("target_system_prompt", "")).strip(),
                    str(definition.get("mode", "single_shot")).strip() or "single_shot",
                    str(definition.get("prompt_template", "")).strip(),
                    str(definition.get("attacker_system_prompt", "")).strip(),
                    str(definition.get("refinement_template", "")).strip(),
                    int(definition.get("max_rounds", 5) or 5),
                    requested_by,
                ),
            ).fetchone()
        return {
            "id": row[0],
            "name": row[1],
            "summary": row[2],
            "target_system_prompt": row[3],
            "mode": row[4],
            "prompt_template": row[5],
            "attacker_system_prompt": row[6],
            "refinement_template": row[7],
            "max_rounds": row[8],
            "created_at": row[9].isoformat() if row[9] else "",
            "updated_at": row[10].isoformat() if row[10] else "",
            "requested_by": row[11],
        }

    def delete_custom_technique(self, technique_id: str) -> None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute("DELETE FROM custom_techniques WHERE technique_id = %s", (technique_id,))

    def resolve_seed_prompt(self, technique_key: str, *, prefer_non_default: bool = False) -> str:
        # When prefer_non_default is True (seed-agnostic techniques),
        # sort non-default seeds first so the prompt varies across runs.
        default_order = "is_default ASC" if prefer_non_default else "is_default DESC"
        query = f"""
            SELECT prompt_text
            FROM seed_prompts
            WHERE is_active = TRUE
              AND (technique_key = %s OR technique_key IS NULL)
            ORDER BY
              CASE WHEN technique_key = %s THEN 0 ELSE 1 END,
              {default_order},
              updated_at DESC
            LIMIT 1
        """
        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(query, (technique_key, technique_key)).fetchone()
        return str(row[0]) if row else ""

    def ensure_schema(self, conn) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id TEXT PRIMARY KEY,
                run_kind TEXT NOT NULL,
                evaluation_id TEXT,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                requested_by TEXT NOT NULL DEFAULT '',
                technique_key TEXT NOT NULL,
                technique_name TEXT NOT NULL DEFAULT '',
                technique_family TEXT NOT NULL DEFAULT '',
                technique_surface TEXT NOT NULL DEFAULT '',
                target_kind TEXT NOT NULL,
                target_id TEXT NOT NULL,
                target_model TEXT NOT NULL DEFAULT '',
                target_base_url TEXT NOT NULL DEFAULT '',
                attacker_model TEXT NOT NULL DEFAULT '',
                attacker_base_url TEXT NOT NULL DEFAULT '',
                unsafe_intent TEXT NOT NULL DEFAULT '',
                system_prompt TEXT NOT NULL DEFAULT '',
                seed_prompt TEXT NOT NULL DEFAULT '',
                success BOOLEAN NOT NULL DEFAULT FALSE,
                score INTEGER NOT NULL DEFAULT 0,
                asr INTEGER NOT NULL DEFAULT 0,
                matched_detections INTEGER NOT NULL DEFAULT 0,
                highest_severity TEXT NOT NULL DEFAULT 'info',
                final_prompt TEXT NOT NULL DEFAULT '',
                final_output TEXT NOT NULL DEFAULT '',
                response_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                run_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                completed_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_run_detections (
                run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                detection_index INTEGER NOT NULL,
                detection_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                matched BOOLEAN NOT NULL,
                rationale TEXT NOT NULL DEFAULT '',
                evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
                PRIMARY KEY (run_id, detection_index)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_run_conversations (
                run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                entry_index INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload JSONB NOT NULL,
                PRIMARY KEY (run_id, entry_index)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_run_traces (
                run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
                trace_index INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL DEFAULT 'system',
                message TEXT NOT NULL DEFAULT '',
                event_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
                attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
                PRIMARY KEY (run_id, trace_index)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_completed_at ON evaluation_runs (completed_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_technique_key ON evaluation_runs (technique_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_target_model ON evaluation_runs (target_model)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_runs_success ON evaluation_runs (success)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seed_prompts (
                prompt_id TEXT PRIMARY KEY,
                technique_key TEXT,
                name TEXT NOT NULL DEFAULT '',
                prompt_text TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'system',
                is_default BOOLEAN NOT NULL DEFAULT FALSE,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_seed_prompts_technique_key ON seed_prompts (technique_key)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS custom_techniques (
                technique_id TEXT PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                target_system_prompt TEXT NOT NULL DEFAULT '',
                mode TEXT NOT NULL DEFAULT 'single_shot',
                prompt_template TEXT NOT NULL DEFAULT '',
                attacker_system_prompt TEXT NOT NULL DEFAULT '',
                refinement_template TEXT NOT NULL DEFAULT '',
                max_rounds INTEGER NOT NULL DEFAULT 5,
                requested_by TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_custom_techniques_updated_at ON custom_techniques (updated_at DESC)")

        # Experiment tables
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                prompt_count INTEGER NOT NULL DEFAULT 0,
                created_by TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_dataset_prompts (
                dataset_id TEXT NOT NULL REFERENCES experiment_datasets(dataset_id) ON DELETE CASCADE,
                prompt_id TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                sort_order INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (dataset_id, prompt_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                hypothesis TEXT NOT NULL DEFAULT '',
                dataset_id TEXT REFERENCES experiment_datasets(dataset_id),
                technique_keys JSONB NOT NULL DEFAULT '[]'::jsonb,
                target_config JSONB NOT NULL DEFAULT '{}'::jsonb,
                experiment_group TEXT NOT NULL DEFAULT '',
                mode TEXT NOT NULL DEFAULT 'live',
                status TEXT NOT NULL DEFAULT 'pending',
                total_cells INTEGER NOT NULL DEFAULT 0,
                repeat_count INTEGER NOT NULL DEFAULT 1,
                max_rounds INTEGER NOT NULL DEFAULT 5,
                completed_cells INTEGER NOT NULL DEFAULT 0,
                report JSONB,
                error TEXT,
                created_by TEXT NOT NULL DEFAULT '',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                completed_at TIMESTAMPTZ
            )
            """
        )
        conn.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS experiment_group TEXT NOT NULL DEFAULT ''")
        conn.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS repeat_count INTEGER NOT NULL DEFAULT 1")
        conn.execute("ALTER TABLE experiments ADD COLUMN IF NOT EXISTS max_rounds INTEGER NOT NULL DEFAULT 5")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments (created_at DESC)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_cells (
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
                technique_key TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                repeat_index INTEGER NOT NULL DEFAULT 1,
                run_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                score INTEGER NOT NULL DEFAULT 0,
                asr INTEGER NOT NULL DEFAULT 0,
                matched_detections INTEGER NOT NULL DEFAULT 0,
                highest_severity TEXT NOT NULL DEFAULT 'info',
                error TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                completed_at TIMESTAMPTZ,
                PRIMARY KEY (experiment_id, technique_key, prompt_id, repeat_index)
            )
            """
        )
        conn.execute("ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS repeat_index INTEGER NOT NULL DEFAULT 1")
        conn.execute("ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb")
        pk_cols = [
            row[0]
            for row in conn.execute(
                """
                SELECT a.attname
                FROM pg_index i
                JOIN pg_class c ON c.oid = i.indrelid
                JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS cols(attnum, ord) ON TRUE
                JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = cols.attnum
                WHERE c.relname = 'experiment_cells' AND i.indisprimary
                ORDER BY cols.ord
                """
            ).fetchall()
        ]
        if pk_cols != ["experiment_id", "technique_key", "prompt_id", "repeat_index"]:
            conn.execute("ALTER TABLE experiment_cells DROP CONSTRAINT IF EXISTS experiment_cells_pkey")
            conn.execute("ALTER TABLE experiment_cells ADD PRIMARY KEY (experiment_id, technique_key, prompt_id, repeat_index)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_cells_experiment_id ON experiment_cells (experiment_id)")

    def save_experiment_dataset(self, *, dataset: dict[str, Any]) -> None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO experiment_datasets (dataset_id, name, description, prompt_count, created_by)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (dataset_id) DO NOTHING
                """,
                (
                    dataset["dataset_id"],
                    dataset["name"],
                    dataset.get("description", ""),
                    dataset.get("prompt_count", len(dataset.get("prompts", []))),
                    dataset.get("created_by", "demo-user"),
                ),
            )
            for i, prompt in enumerate(dataset.get("prompts", [])):
                conn.execute(
                    """
                    INSERT INTO experiment_dataset_prompts (dataset_id, prompt_id, prompt_text, category, sort_order)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (dataset_id, prompt_id) DO NOTHING
                    """,
                    (
                        dataset["dataset_id"],
                        prompt["id"],
                        prompt["text"],
                        prompt.get("category", "general"),
                        i,
                    ),
                )

    def save_experiment(self, *, experiment: dict[str, Any]) -> None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, description, hypothesis,
                    dataset_id, technique_keys, target_config, mode,
                    status, total_cells, repeat_count, max_rounds,
                    experiment_group, created_by
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s::jsonb, %s::jsonb, %s,
                    %s, %s, %s, %s,
                    %s, %s
                )
                ON CONFLICT (experiment_id) DO NOTHING
                """,
                (
                    experiment["experiment_id"],
                    experiment["name"],
                    experiment.get("description", ""),
                    experiment.get("hypothesis", ""),
                    experiment.get("dataset_id", ""),
                    json.dumps(experiment.get("technique_keys", [])),
                    json.dumps(experiment.get("target_config", {})),
                    experiment.get("mode", "live"),
                    experiment.get("status", "running"),
                    experiment.get("total_cells", 0),
                    int(experiment.get("repeat_count", 1) or 1),
                    int(experiment.get("max_rounds", 5) or 5),
                    experiment.get("experiment_group", ""),
                    experiment.get("created_by", "demo-user"),
                ),
            )

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        report: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                UPDATE experiments
                SET status = %s,
                    report = %s::jsonb,
                    error = %s,
                    completed_cells = (
                        SELECT COUNT(*) FROM experiment_cells
                        WHERE experiment_id = %s AND status != 'pending'
                    ),
                    completed_at = CASE WHEN %s IN ('completed', 'failed') THEN %s::timestamptz ELSE completed_at END
                WHERE experiment_id = %s
                """,
                (
                    status,
                    json.dumps(report) if report else None,
                    error,
                    experiment_id,
                    status,
                    now,
                    experiment_id,
                ),
            )

    def save_experiment_cell(self, *, experiment_id: str, cell: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        metadata = {
            key: cell[key]
            for key in (
                "experiment_group",
                "repeat_count",
                "max_rounds",
                "original_score",
                "original_asr",
                "rejudge_results",
            )
            if key in cell
        }
        with self._connect() as conn:
            self.ensure_schema(conn)
            conn.execute(
                """
                INSERT INTO experiment_cells (
                    experiment_id, technique_key, prompt_id, repeat_index,
                    run_id, status, score, asr,
                    matched_detections, highest_severity, error, metadata, completed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (experiment_id, technique_key, prompt_id, repeat_index) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    status = EXCLUDED.status,
                    score = EXCLUDED.score,
                    asr = EXCLUDED.asr,
                    matched_detections = EXCLUDED.matched_detections,
                    highest_severity = EXCLUDED.highest_severity,
                    error = EXCLUDED.error,
                    metadata = EXCLUDED.metadata,
                    completed_at = EXCLUDED.completed_at
                """,
                (
                    experiment_id,
                    cell.get("technique_key", ""),
                    cell.get("prompt_id", ""),
                    int(cell.get("repeat_index", 1) or 1),
                    cell.get("run_id", ""),
                    cell.get("status", "completed"),
                    cell.get("score", 0),
                    cell.get("asr", 0),
                    cell.get("matched_detections", 0),
                    cell.get("highest_severity", "info"),
                    cell.get("error"),
                    json.dumps(metadata),
                    now,
                ),
            )

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(
                """
                SELECT experiment_id, name, description, hypothesis,
                       dataset_id, technique_keys, mode, status,
                       total_cells, completed_cells, report, error,
                       created_at, completed_at, repeat_count, max_rounds,
                       experiment_group
                FROM experiments
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [
            {
                "experiment_id": r[0],
                "name": r[1],
                "description": r[2],
                "hypothesis": r[3],
                "dataset_id": r[4],
                "technique_keys": r[5] if isinstance(r[5], list) else [],
                "mode": r[6],
                "status": r[7],
                "total_cells": r[8],
                "completed_cells": r[9],
                "report": r[10],
                "error": r[11],
                "created_at": r[12].isoformat() if r[12] else "",
                "completed_at": r[13].isoformat() if r[13] else None,
                "repeat_count": r[14],
                "max_rounds": r[15],
                "experiment_group": r[16],
            }
            for r in rows
        ]

    def load_all_experiments(self) -> list[dict[str, Any]]:
        """Load full experiment state from DB, including cells and conversations."""
        with self._connect() as conn:
            self.ensure_schema(conn)

            # 1. Load experiments
            exp_rows = conn.execute(
                """
                SELECT experiment_id, name, description, hypothesis,
                       dataset_id, technique_keys, target_config, mode,
                       status, total_cells, completed_cells, report, error,
                       created_at, repeat_count, max_rounds, experiment_group
                FROM experiments ORDER BY created_at
                """
            ).fetchall()

            results = []
            for er in exp_rows:
                eid = er[0]
                dataset_id = er[4]
                technique_keys = er[5] if isinstance(er[5], list) else []
                target_config = er[6] if isinstance(er[6], dict) else {}
                status = er[8]

                # 2. Load dataset + prompts
                ds_row = conn.execute(
                    "SELECT dataset_id, name, description, prompt_count FROM experiment_datasets WHERE dataset_id = %s",
                    (dataset_id,),
                ).fetchone()
                dataset = {}
                if ds_row:
                    prompt_rows = conn.execute(
                        "SELECT prompt_id, prompt_text, category FROM experiment_dataset_prompts WHERE dataset_id = %s ORDER BY sort_order",
                        (dataset_id,),
                    ).fetchall()
                    dataset = {
                        "dataset_id": ds_row[0],
                        "name": ds_row[1],
                        "description": ds_row[2],
                        "prompt_count": ds_row[3],
                        "prompts": [
                            {"id": pr[0], "text": pr[1], "category": pr[2]}
                            for pr in prompt_rows
                        ],
                    }

                # 3. Load cells
                cell_rows = conn.execute(
                    """
                    SELECT technique_key, prompt_id, repeat_index, run_id, status,
                           score, asr, matched_detections, highest_severity, error,
                           metadata
                    FROM experiment_cells
                    WHERE experiment_id = %s
                    ORDER BY technique_key, prompt_id, repeat_index
                    """,
                    (eid,),
                ).fetchall()

                # Build prompt lookup
                prompt_lookup = {
                    p["id"]: p for p in dataset.get("prompts", [])
                }

                cell_results = []
                for cr in cell_rows:
                    run_id = cr[3] or ""
                    prompt_id = cr[1]
                    prompt = prompt_lookup.get(prompt_id, {})
                    metadata = cr[10] if isinstance(cr[10], dict) else {}

                    # 4. Load conversation for this cell's run
                    conversation = []
                    if run_id:
                        conv_rows = conn.execute(
                            """
                            SELECT payload FROM evaluation_run_conversations
                            WHERE run_id = %s ORDER BY entry_index
                            """,
                            (run_id,),
                        ).fetchall()
                        conversation = [
                            row[0] if isinstance(row[0], dict) else {}
                            for row in conv_rows
                        ]

                    cell = {
                        "technique_key": cr[0],
                        "prompt_id": prompt_id,
                        "prompt_text": prompt.get("text", ""),
                        "prompt_category": prompt.get("category", ""),
                        "repeat_index": cr[2],
                        "run_id": run_id,
                        "status": cr[4],
                        "score": cr[5],
                        "asr": cr[6],
                        "matched_detections": cr[7],
                        "highest_severity": cr[8],
                        "error": cr[9],
                        "metadata": metadata,
                        "conversation": conversation,
                    }
                    for key, value in metadata.items():
                        if key not in cell:
                            cell[key] = value
                    cell_results.append(cell)

                completed_cells = sum(
                    1 for c in cell_results if c["status"] in ("completed", "error")
                )

                results.append({
                    "experiment_id": eid,
                    "name": er[1],
                    "description": er[2],
                    "hypothesis": er[3],
                    "dataset": dataset,
                    "technique_keys": technique_keys,
                    "target_config": target_config,
                    "mode": er[7],
                    "status": status,
                    "total_cells": er[9],
                    "completed_cells": completed_cells,
                    "cell_results": cell_results,
                    "report": er[11],
                    "error": er[12],
                    "created_at": er[13].isoformat() if er[13] else "",
                    "repeat_count": er[14],
                    "max_rounds": er[15],
                    "experiment_group": er[16],
                })

            return results

    def list_run_history(
        self,
        *,
        filters: dict[str, str],
        limit: int = 50,
        offset: int = 0,
        sort: str = "completed_desc",
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit or 50), 100))
        offset = max(0, int(offset or 0))
        where_sql, params = _build_run_history_where(filters)
        order_sql = _run_history_order(sort)

        query = f"""
            SELECT
                run_id,
                status,
                mode,
                technique_key,
                technique_name,
                technique_family,
                technique_surface,
                target_kind,
                target_model,
                attacker_model,
                unsafe_intent,
                success,
                score,
                asr,
                matched_detections,
                highest_severity,
                final_prompt,
                final_output,
                created_at,
                completed_at,
                EXTRACT(EPOCH FROM (completed_at - created_at)) AS duration_seconds,
                COUNT(*) OVER() AS total_count,
                run_metadata
            FROM evaluation_runs
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT %s OFFSET %s
        """

        with self._connect() as conn:
            self.ensure_schema(conn)
            rows = conn.execute(query, (*params, limit, offset)).fetchall()
            facets = _load_run_history_facets(conn)

        total = int(rows[0][21]) if rows else 0
        return {
            "items": [_serialize_history_row(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "facets": facets,
        }

    def get_run_history_item(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            self.ensure_schema(conn)
            row = conn.execute(
                """
                SELECT
                    run_id,
                    run_kind,
                    evaluation_id,
                    status,
                    mode,
                    requested_by,
                    technique_key,
                    technique_name,
                    technique_family,
                    technique_surface,
                    target_kind,
                    target_model,
                    attacker_model,
                    unsafe_intent,
                    success,
                    score,
                    asr,
                    matched_detections,
                    highest_severity,
                    final_prompt,
                    final_output,
                    response_metadata,
                    run_metadata,
                    created_at,
                    completed_at,
                    EXTRACT(EPOCH FROM (completed_at - created_at)) AS duration_seconds
                FROM evaluation_runs
                WHERE run_id = %s
                  AND run_kind = 'technique'
                  AND evaluation_id IS NULL
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                return None

            detections = conn.execute(
                """
                SELECT detection_type, severity, matched, rationale, evidence
                FROM evaluation_run_detections
                WHERE run_id = %s
                ORDER BY detection_index
                """,
                (run_id,),
            ).fetchall()
            conversations = conn.execute(
                """
                SELECT kind, payload
                FROM evaluation_run_conversations
                WHERE run_id = %s
                ORDER BY entry_index
                """,
                (run_id,),
            ).fetchall()
            traces = conn.execute(
                """
                SELECT event_type, actor, message, event_timestamp, attributes
                FROM evaluation_run_traces
                WHERE run_id = %s
                ORDER BY trace_index
                """,
                (run_id,),
            ).fetchall()

        item = _serialize_history_detail_row(row)
        item["detections"] = [
            {
                "detection_type": det[0],
                "severity": det[1],
                "matched": bool(det[2]),
                "rationale": det[3],
                "evidence": _json_value(det[4], []),
            }
            for det in detections
        ]
        item["conversation"] = [
            _history_conversation_entry(kind, payload)
            for kind, payload in conversations
        ]
        item["traces"] = [
            {
                "event_type": trace[0],
                "actor": trace[1],
                "message": trace[2],
                "event_timestamp": _isoformat(trace[3]),
                "attributes": _json_value(trace[4], {}),
            }
            for trace in traces
        ]
        return item

    def _replace_detections(self, conn, run_id: str, result: TechniqueRunResult) -> None:
        conn.execute("DELETE FROM evaluation_run_detections WHERE run_id = %s", (run_id,))
        for idx, det in enumerate(result.detections):
            conn.execute(
                """
                INSERT INTO evaluation_run_detections (
                    run_id, detection_index, detection_type, severity, matched, rationale, evidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    run_id,
                    idx,
                    det.detection_type.value,
                    det.severity.value,
                    det.matched,
                    det.rationale,
                    json.dumps(list(det.evidence)),
                ),
            )

    def _replace_conversation(self, conn, run_id: str, conversation: list[dict[str, Any]]) -> None:
        conn.execute("DELETE FROM evaluation_run_conversations WHERE run_id = %s", (run_id,))
        for idx, entry in enumerate(conversation):
            conn.execute(
                """
                INSERT INTO evaluation_run_conversations (run_id, entry_index, kind, payload)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (run_id, idx, str(entry.get("kind", "")), json.dumps(entry)),
            )

    def _replace_traces(self, conn, run_id: str, result: TechniqueRunResult) -> None:
        conn.execute("DELETE FROM evaluation_run_traces WHERE run_id = %s", (run_id,))
        for idx, trace in enumerate(result.traces):
            conn.execute(
                """
                INSERT INTO evaluation_run_traces (
                    run_id, trace_index, event_type, actor, message, event_timestamp, attributes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    run_id,
                    idx,
                    trace.event_type,
                    trace.actor,
                    trace.message,
                    trace.timestamp.isoformat(),
                    json.dumps(dict(trace.attributes)),
                ),
            )

    def _connect(self):
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)


def build_run_persistence() -> RunPersistence:
    try:
        return PostgreSQLRunPersistence()
    except Exception as exc:  # noqa: BLE001
        log.warning("Run persistence disabled: %s", exc)
        return NoopRunPersistence()


_RUN_HISTORY_FILTER_COLUMNS = {
    "technique_key": "technique_key",
    "technique_family": "technique_family",
    "technique_surface": "technique_surface",
    "status": "status",
    "highest_severity": "highest_severity",
    "mode": "mode",
    "target_kind": "target_kind",
    "target_model": "target_model",
}

_RUN_HISTORY_SORTS = {
    "completed_desc": "completed_at DESC, run_id DESC",
    "completed_asc": "completed_at ASC, run_id ASC",
    "score_desc": "score DESC, completed_at DESC",
    "score_asc": "score ASC, completed_at DESC",
    "asr_desc": "asr DESC, completed_at DESC",
    "asr_asc": "asr ASC, completed_at DESC",
    "severity_desc": """
        CASE highest_severity
            WHEN 'critical' THEN 5
            WHEN 'high' THEN 4
            WHEN 'medium' THEN 3
            WHEN 'low' THEN 2
            ELSE 1
        END DESC,
        completed_at DESC
    """,
}


def _build_run_history_where(filters: dict[str, str]) -> tuple[str, list[Any]]:
    clauses = ["run_kind = 'technique'", "evaluation_id IS NULL"]
    params: list[Any] = []

    query = str(filters.get("q", "")).strip()
    if query:
        clauses.append(
            """
            concat_ws(
                ' ',
                run_id,
                technique_key,
                technique_name,
                technique_family,
                technique_surface,
                target_model,
                attacker_model,
                unsafe_intent,
                final_prompt,
                final_output
            ) ILIKE %s
            """
        )
        params.append(f"%{query}%")

    for key, column in _RUN_HISTORY_FILTER_COLUMNS.items():
        value = str(filters.get(key, "")).strip()
        if value:
            clauses.append(f"{column} = %s")
            params.append(value)

    success = str(filters.get("success", "")).strip().lower()
    if success in {"true", "false"}:
        clauses.append("success = %s")
        params.append(success == "true")

    completed_after = str(filters.get("completed_after", "")).strip()
    if completed_after:
        clauses.append("completed_at >= %s::timestamptz")
        params.append(completed_after)

    completed_before = str(filters.get("completed_before", "")).strip()
    if completed_before:
        if len(completed_before) == 10:
            clauses.append("completed_at < (%s::date + INTERVAL '1 day')")
        else:
            clauses.append("completed_at <= %s::timestamptz")
        params.append(completed_before)

    for key, column, operator in (
        ("min_score", "score", ">="),
        ("max_score", "score", "<="),
        ("min_asr", "asr", ">="),
        ("max_asr", "asr", "<="),
    ):
        raw = str(filters.get(key, "")).strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        clauses.append(f"{column} {operator} %s")
        params.append(value)

    return " AND ".join(clauses), params


def _run_history_order(sort: str) -> str:
    return _RUN_HISTORY_SORTS.get(sort, _RUN_HISTORY_SORTS["completed_desc"])


def _load_run_history_facets(conn) -> dict[str, list[dict[str, Any]]]:
    return {
        "techniques": [
            {"key": row[0], "label": row[1] or row[0], "count": row[2]}
            for row in conn.execute(
                """
                SELECT technique_key, technique_name, COUNT(*) AS run_count
                FROM evaluation_runs
                WHERE run_kind = 'technique' AND evaluation_id IS NULL
                GROUP BY technique_key, technique_name
                ORDER BY technique_name, technique_key
                """
            ).fetchall()
        ],
        "families": _facet_rows(conn, "technique_family"),
        "surfaces": _facet_rows(conn, "technique_surface"),
        "severities": _facet_rows(conn, "highest_severity"),
        "statuses": _facet_rows(conn, "status"),
        "modes": _facet_rows(conn, "mode"),
        "target_models": _facet_rows(conn, "target_model"),
    }


def _facet_rows(conn, column: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT {column}, COUNT(*) AS run_count
        FROM evaluation_runs
        WHERE run_kind = 'technique'
          AND evaluation_id IS NULL
          AND {column} <> ''
        GROUP BY {column}
        ORDER BY run_count DESC, {column} ASC
        """
    ).fetchall()
    return [{"key": row[0], "label": row[0], "count": row[1]} for row in rows]


def _serialize_history_row(row) -> dict[str, Any]:
    return {
        "run_id": row[0],
        "status": row[1],
        "mode": row[2],
        "technique_key": row[3],
        "technique_name": row[4],
        "technique_family": row[5],
        "technique_surface": row[6],
        "target_kind": row[7],
        "target_model": row[8],
        "attacker_model": row[9],
        "unsafe_intent": row[10],
        "success": bool(row[11]),
        "score": row[12],
        "asr": row[13],
        "matched_detections": row[14],
        "highest_severity": row[15],
        "judge_model": _history_judge_model(row[22] if len(row) > 22 else {}, row[8], row[9]),
        "final_prompt": row[16],
        "final_output": row[17],
        "created_at": _isoformat(row[18]),
        "completed_at": _isoformat(row[19]),
        "duration_seconds": _float_or_zero(row[20]),
    }


def _serialize_history_detail_row(row) -> dict[str, Any]:
    return {
        "run_id": row[0],
        "run_kind": row[1],
        "evaluation_id": row[2],
        "status": row[3],
        "mode": row[4],
        "requested_by": row[5],
        "technique_key": row[6],
        "technique_name": row[7],
        "technique_family": row[8],
        "technique_surface": row[9],
        "target_kind": row[10],
        "target_model": row[11],
        "attacker_model": row[12],
        "unsafe_intent": row[13],
        "success": bool(row[14]),
        "score": row[15],
        "asr": row[16],
        "matched_detections": row[17],
        "highest_severity": row[18],
        "judge_model": _history_judge_model(row[22], row[11], row[12]),
        "final_prompt": row[19],
        "final_output": row[20],
        "response_metadata": _json_value(row[21], {}),
        "run_metadata": _json_value(row[22], {}),
        "created_at": _isoformat(row[23]),
        "completed_at": _isoformat(row[24]),
        "duration_seconds": _float_or_zero(row[25]),
    }


def _history_conversation_entry(kind: Any, payload: Any) -> dict[str, Any]:
    value = _json_value(payload, {})
    if not isinstance(value, dict):
        value = {"value": value}
    return {**value, "kind": str(kind or value.get("kind", ""))}


def _history_judge_model(run_metadata: Any, target_model: Any, attacker_model: Any) -> str:
    metadata = _json_value(run_metadata, {})
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    if not isinstance(config, dict):
        config = {}

    mode = str(config.get("judge_mode", "") or "").strip().lower()
    if mode == "target":
        return str(config.get("model") or target_model or "")
    if mode == "custom":
        return str(config.get("judge_model") or "")
    if mode == "attacker":
        return str(config.get("attacker_model") or attacker_model or "")
    return str(config.get("attacker_model") or attacker_model or config.get("model") or target_model or "")


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _isoformat(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _sanitize_config(config: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"api_key", "attacker_api_key", "judge_api_key"}
    }


def _resolve_attacker_model(target_config: dict[str, str], mode: str) -> str:
    attacker_model = target_config.get("attacker_model", "")
    if attacker_model:
        return attacker_model
    if mode == "live":
        return target_config.get("model", "")
    return ""


def _extract_final_prompt(conversation: list[dict[str, Any]]) -> str:
    if not conversation:
        return ""
    last = conversation[-1]
    kind = last.get("kind")
    if kind == "single_shot":
        return str(last.get("prompt", ""))
    if kind == "round":
        target = last.get("target", {})
        return str(target.get("prompt", "") or target.get("input", ""))
    if kind == "generation":
        candidates = last.get("candidates", [])
        if candidates:
            return str(candidates[0].get("prompt", ""))
    return ""


def _trace_created_at(result: TechniqueRunResult) -> str:
    if result.traces:
        return result.traces[0].timestamp.isoformat()
    return result.completed_at.isoformat()


def _serialize_audit_log_event(item: Any) -> dict[str, Any]:
    return {
        "action": item.action,
        "actor": item.actor,
        "run_id": item.run_id,
        "timestamp": item.timestamp.isoformat(),
        "attributes": dict(item.attributes),
    }
