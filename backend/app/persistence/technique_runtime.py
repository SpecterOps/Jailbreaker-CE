"""Persistence boundary for normalized technique runtime evidence."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from app.techniques.contracts.events import ExecutionEvent
from app.techniques.contracts.execution import CorrelationIds, _freeze_json_mapping


log = logging.getLogger("persistence.technique_runtime")


_RUNTIME_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS technique_packages (
        package_id TEXT PRIMARY KEY, technique_id TEXT NOT NULL,
        technique_version TEXT NOT NULL, source_kind TEXT NOT NULL,
        source_location TEXT NOT NULL DEFAULT '', state TEXT NOT NULL DEFAULT 'validated',
        manifest_hash TEXT NOT NULL, diagnostics JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS technique_versions (
        technique_id TEXT NOT NULL, technique_version TEXT NOT NULL,
        implementation_id TEXT NOT NULL, implementation_kind TEXT NOT NULL,
        contract_version TEXT NOT NULL, manifest_hash TEXT NOT NULL,
        asset_hashes JSONB NOT NULL DEFAULT '{}'::jsonb,
        package_id TEXT REFERENCES technique_packages(package_id),
        enabled BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (technique_id, technique_version, implementation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS execution_events (
        event_id TEXT PRIMARY KEY, experiment_id TEXT, cell_id TEXT,
        run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, sequence INTEGER NOT NULL,
        event_type TEXT NOT NULL, schema_version INTEGER NOT NULL,
        event_timestamp TIMESTAMPTZ NOT NULL, technique_id TEXT NOT NULL,
        technique_version TEXT NOT NULL, implementation_id TEXT NOT NULL,
        round_number INTEGER, branch_id TEXT, caused_by TEXT, operation_id TEXT,
        operation_attempt INTEGER NOT NULL DEFAULT 1,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb, event_hash TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE (run_id, sequence)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_execution_events_attempt ON execution_events (attempt_id, sequence)",
    """
    CREATE TABLE IF NOT EXISTS execution_checkpoints (
        checkpoint_id TEXT PRIMARY KEY, experiment_id TEXT, cell_id TEXT,
        run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, spec_hash TEXT NOT NULL,
        state_schema_version INTEGER NOT NULL, payload JSONB NOT NULL,
        checkpoint_hash TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_execution_checkpoints_attempt ON execution_checkpoints (attempt_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS model_calls (
        model_call_id TEXT PRIMARY KEY, experiment_id TEXT, cell_id TEXT,
        run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, role TEXT NOT NULL,
        operation_id TEXT NOT NULL, operation_attempt INTEGER NOT NULL,
        request_payload JSONB NOT NULL, request_hash TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'requested', response_text TEXT,
        usage JSONB NOT NULL DEFAULT '{}'::jsonb, provider_response_id TEXT,
        gateway_call_id TEXT, gateway_attempted_retries INTEGER,
        gateway_attempted_fallbacks INTEGER, failure_category TEXT,
        failure_error TEXT, failure_indeterminate BOOLEAN,
        requested_at TIMESTAMPTZ NOT NULL DEFAULT now(), completed_at TIMESTAMPTZ,
        UNIQUE (run_id, operation_id, operation_attempt)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_model_calls_attempt ON model_calls (attempt_id, requested_at)",
    """
    CREATE TABLE IF NOT EXISTS run_artifacts (
        artifact_id TEXT PRIMARY KEY, experiment_id TEXT, cell_id TEXT,
        run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, media_type TEXT NOT NULL,
        content_hash TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_run_artifacts_attempt ON run_artifacts (attempt_id, created_at)",
)


_EXPERIMENT_ATTEMPT_SCHEMA_STATEMENTS = (
    "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS execution_generation BIGINT NOT NULL DEFAULT 0",
    "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS execution_owner_id TEXT NOT NULL DEFAULT ''",
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'experiments'::regclass
              AND conname = 'experiments_execution_generation_check'
        ) THEN
            ALTER TABLE experiments
                ADD CONSTRAINT experiments_execution_generation_check
                CHECK (execution_generation >= 0);
        END IF;
    END $$
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_execution_operations (
        operation_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL
            REFERENCES experiments(experiment_id) ON DELETE CASCADE,
        execution_generation BIGINT NOT NULL CHECK (execution_generation >= 1),
        execution_owner_id TEXT NOT NULL,
        operation_kind TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('running', 'finished')),
        outcome TEXT CHECK (
            outcome IS NULL OR outcome IN ('succeeded', 'failed', 'cancelled')
        ),
        lease_expires_at TIMESTAMPTZ,
        started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        completed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        CHECK (
            (status = 'running' AND lease_expires_at IS NOT NULL
                AND outcome IS NULL AND completed_at IS NULL)
            OR (status = 'finished' AND lease_expires_at IS NULL
                AND outcome IS NOT NULL AND completed_at IS NOT NULL)
        )
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_experiment_execution_operations_owner "
    "ON experiment_execution_operations "
    "(experiment_id, execution_generation, execution_owner_id) "
    "WHERE status = 'running'",
    """
    CREATE TABLE IF NOT EXISTS experiment_technique_configs (
        technique_config_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
        technique_id TEXT NOT NULL, technique_version TEXT NOT NULL,
        implementation_id TEXT NOT NULL, preset TEXT,
        parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
        resolved_spec JSONB NOT NULL DEFAULT '{}'::jsonb,
        spec_hash TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (experiment_id, technique_config_id)
    )
    """,
    "ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS cell_id TEXT",
    "ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS technique_config_id TEXT",
    """
    INSERT INTO experiment_technique_configs (
        technique_config_id, experiment_id, technique_id, technique_version,
        implementation_id, parameters, resolved_spec, spec_hash
    )
    SELECT DISTINCT
        COALESCE(
            cell.technique_config_id,
            'etc-' || md5(cell.experiment_id || E'\\x1f' || cell.technique_key)
        ),
        cell.experiment_id, cell.technique_key, '0.2.0', 'legacy', '{}'::jsonb,
        jsonb_build_object(
            'technique_key', cell.technique_key,
            'implementation_id', 'legacy'
        ),
        md5(cell.experiment_id || E'\\x1f' || cell.technique_key || E'\\x1flegacy')
    FROM experiment_cells AS cell
    WHERE cell.technique_config_id IS NULL
       OR NOT EXISTS (
            SELECT 1
            FROM experiment_technique_configs AS existing
            WHERE existing.experiment_id = cell.experiment_id
              AND existing.technique_config_id = cell.technique_config_id
       )
    ON CONFLICT DO NOTHING
    """,
    """
    UPDATE experiment_cells
    SET cell_id = COALESCE(cell_id, 'cell-' || md5(
            experiment_id || E'\\x1f' || technique_key || E'\\x1f' || prompt_id
            || E'\\x1f' || repeat_index::text
        )),
        technique_config_id = COALESCE(
            technique_config_id, 'etc-' || md5(experiment_id || E'\\x1f' || technique_key)
        )
    """,
    """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM experiment_cells
            WHERE cell_id IS NULL OR technique_config_id IS NULL
        ) THEN
            RAISE EXCEPTION 'experiment cell identity backfill left null values';
        END IF;
    END $$
    """,
    "ALTER TABLE experiment_cells ALTER COLUMN cell_id SET NOT NULL",
    "ALTER TABLE experiment_cells ALTER COLUMN technique_config_id SET NOT NULL",
    "ALTER TABLE experiment_cells ALTER COLUMN score DROP NOT NULL",
    "ALTER TABLE experiment_cells ALTER COLUMN asr DROP NOT NULL",
    """
    DO $$
    DECLARE primary_key_columns TEXT[]; primary_key_name TEXT;
    BEGIN
        SELECT array_agg(attribute.attname ORDER BY key_column.ordinality)
        INTO primary_key_columns
        FROM pg_index index_definition
        JOIN pg_class relation ON relation.oid = index_definition.indrelid
        JOIN LATERAL unnest(index_definition.indkey) WITH ORDINALITY AS key_column(attnum, ordinality)
            ON TRUE
        JOIN pg_attribute attribute
            ON attribute.attrelid = relation.oid AND attribute.attnum = key_column.attnum
        WHERE relation.oid = 'experiment_cells'::regclass AND index_definition.indisprimary;
        SELECT conname INTO primary_key_name
        FROM pg_constraint
        WHERE conrelid = 'experiment_cells'::regclass AND contype = 'p';
        IF primary_key_columns IS DISTINCT FROM ARRAY['cell_id'] THEN
            IF primary_key_name IS NOT NULL THEN
                EXECUTE format('ALTER TABLE experiment_cells DROP CONSTRAINT %I', primary_key_name);
            END IF;
            ALTER TABLE experiment_cells ADD PRIMARY KEY (cell_id);
        END IF;
    END $$
    """,
    """
    DO $$
    DECLARE item RECORD; actual_name TEXT; actual_definition TEXT;
    BEGIN
        FOR item IN SELECT * FROM (VALUES
            ('experiment_cells_technique_config_fkey', 'FOREIGN KEY (experiment_id, technique_config_id) REFERENCES experiment_technique_configs(experiment_id, technique_config_id) ON DELETE RESTRICT'),
            ('experiment_cells_selected_attempt_fkey', 'FOREIGN KEY (cell_id, selected_attempt_id) REFERENCES experiment_cell_attempts(cell_id, attempt_id) ON DELETE RESTRICT'),
            ('experiment_cells_latest_attempt_fkey', 'FOREIGN KEY (cell_id, latest_attempt_id) REFERENCES experiment_cell_attempts(cell_id, attempt_id) ON DELETE RESTRICT'),
            ('experiment_cells_active_attempt_fkey', 'FOREIGN KEY (cell_id, active_attempt_id) REFERENCES experiment_cell_attempts(cell_id, attempt_id) ON DELETE RESTRICT')
        ) AS expected(name, definition) LOOP
            SELECT conname, regexp_replace(pg_get_constraintdef(oid), '[[:space:]]+', '', 'g')
            INTO actual_name, actual_definition FROM pg_constraint
            WHERE conrelid = 'experiment_cells'::regclass AND conname = item.name
              AND (regexp_replace(pg_get_constraintdef(oid), '[[:space:]]+', '', 'g')
                   <> regexp_replace(item.definition, '[[:space:]]+', '', 'g')
                   OR NOT convalidated OR condeferrable OR condeferred);
            IF actual_name IS NOT NULL AND actual_definition IS NOT NULL THEN
                EXECUTE format('ALTER TABLE experiment_cells DROP CONSTRAINT %I', actual_name);
            END IF;
        END LOOP;
    END $$
    """,
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'experiment_cells'::regclass
              AND conname = 'experiment_cells_technique_config_fkey'
        ) THEN
            ALTER TABLE experiment_cells
                ADD CONSTRAINT experiment_cells_technique_config_fkey
                FOREIGN KEY (experiment_id, technique_config_id)
                REFERENCES experiment_technique_configs (experiment_id, technique_config_id)
                ON DELETE RESTRICT;
        END IF;
    END $$
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_cell_attempts (
        attempt_id TEXT PRIMARY KEY,
        cell_id TEXT NOT NULL REFERENCES experiment_cells(cell_id) ON DELETE CASCADE,
        attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
        parent_attempt_id TEXT REFERENCES experiment_cell_attempts(attempt_id),
        run_id TEXT, retry_kind TEXT NOT NULL DEFAULT 'initial', spec_hash TEXT NOT NULL,
        execution_status TEXT NOT NULL, result_disposition TEXT NOT NULL,
        execution_generation BIGINT NOT NULL DEFAULT 0,
        official_judge_status TEXT NOT NULL DEFAULT 'not_scheduled', score DOUBLE PRECISION, asr INTEGER,
        failure_category TEXT, checkpoint_id TEXT,
        promotion_policy TEXT NOT NULL DEFAULT 'replace_on_valid_completion',
        selection_reason TEXT, late_completion BOOLEAN NOT NULL DEFAULT FALSE,
        recoverable_cancellation BOOLEAN NOT NULL DEFAULT FALSE,
        worker_id TEXT, lease_expires_at TIMESTAMPTZ,
        started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (cell_id, attempt_number), UNIQUE (cell_id, attempt_id),
        CONSTRAINT experiment_cell_attempts_score_judge_status_check CHECK (
            (official_judge_status = 'succeeded' AND score IS NOT NULL)
            OR (official_judge_status <> 'succeeded' AND score IS NULL)
        ),
        CONSTRAINT experiment_cell_attempts_execution_status_check CHECK (execution_status IN ('queued', 'running', 'completed', 'failed', 'timed_out', 'cancelled', 'budget_exhausted')),
        CONSTRAINT experiment_cell_attempts_result_disposition_check CHECK (result_disposition IN ('tested', 'inconclusive', 'not_tested')),
        CONSTRAINT experiment_cell_attempts_judge_status_check CHECK (official_judge_status IN ('not_scheduled', 'scheduled', 'running', 'succeeded', 'failed', 'skipped')),
        CONSTRAINT experiment_cell_attempts_failure_category_check CHECK (failure_category IS NULL OR failure_category IN ('provider', 'timeout', 'judge_parse', 'validation', 'cancelled', 'budget', 'worker_lost', 'persistence', 'internal'))
    )
    """,
    "ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS worker_id TEXT",
    "ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ",
    "ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS checkpoint_id TEXT",
    "ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS promotion_policy TEXT NOT NULL DEFAULT 'replace_on_valid_completion'",
    "ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS selection_reason TEXT",
    "ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS late_completion BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS recoverable_cancellation BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS execution_generation BIGINT NOT NULL DEFAULT 0",
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'experiment_cell_attempts'::regclass
              AND conname = 'experiment_cell_attempts_execution_generation_check'
        ) THEN
            ALTER TABLE experiment_cell_attempts
                ADD CONSTRAINT experiment_cell_attempts_execution_generation_check
                CHECK (execution_generation >= 0);
        END IF;
    END $$
    """,
    """
    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1 FROM pg_attribute
            WHERE attrelid = 'experiment_cell_attempts'::regclass
              AND attname = 'score'
              AND atttypid <> 'double precision'::regtype
              AND NOT attisdropped
        ) THEN
            ALTER TABLE experiment_cell_attempts
            ALTER COLUMN score TYPE DOUBLE PRECISION
            USING score::double precision;
        END IF;
    END $$
    """,
    """
    DO $$
    DECLARE item RECORD; actual_name TEXT; actual_definition TEXT;
    BEGIN
        FOR item IN SELECT * FROM (VALUES
            ('experiment_cell_attempts_score_judge_status_check', 'CHECK ((((official_judge_status = ''succeeded''::text) AND (score IS NOT NULL)) OR ((official_judge_status <> ''succeeded''::text) AND (score IS NULL))))'),
            ('experiment_cell_attempts_execution_status_check', 'CHECK ((execution_status = ANY (ARRAY[''queued''::text, ''running''::text, ''completed''::text, ''failed''::text, ''timed_out''::text, ''cancelled''::text, ''budget_exhausted''::text])))'),
            ('experiment_cell_attempts_result_disposition_check', 'CHECK ((result_disposition = ANY (ARRAY[''tested''::text, ''inconclusive''::text, ''not_tested''::text])))'),
            ('experiment_cell_attempts_judge_status_check', 'CHECK ((official_judge_status = ANY (ARRAY[''not_scheduled''::text, ''scheduled''::text, ''running''::text, ''succeeded''::text, ''failed''::text, ''skipped''::text])))'),
            ('experiment_cell_attempts_failure_category_check', 'CHECK (((failure_category IS NULL) OR (failure_category = ANY (ARRAY[''provider''::text, ''timeout''::text, ''judge_parse''::text, ''validation''::text, ''cancelled''::text, ''budget''::text, ''worker_lost''::text, ''persistence''::text, ''internal''::text)))))')
        ) AS expected(name, definition) LOOP
            SELECT conname, regexp_replace(pg_get_constraintdef(oid), '[[:space:]]+', '', 'g')
            INTO actual_name, actual_definition
            FROM pg_constraint WHERE conrelid = 'experiment_cell_attempts'::regclass AND conname = item.name;
            IF actual_definition IS NOT NULL
               AND (actual_definition <> regexp_replace(item.definition, '[[:space:]]+', '', 'g')
                    OR EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'experiment_cell_attempts'::regclass
                               AND conname = item.name AND (NOT convalidated OR condeferrable OR condeferred))) THEN
                EXECUTE format('ALTER TABLE experiment_cell_attempts DROP CONSTRAINT %I', actual_name);
            END IF;
        END LOOP;
        UPDATE experiment_cell_attempts
        SET official_judge_status = 'not_scheduled'
        WHERE official_judge_status = 'unscheduled';
        UPDATE experiment_cell_attempts
        SET execution_status = 'queued'
        WHERE execution_status = 'pending';
        UPDATE experiment_cell_attempts
        SET failure_category = 'internal'
        WHERE failure_category IS NOT NULL
          AND failure_category NOT IN ('provider', 'timeout', 'judge_parse', 'validation', 'cancelled', 'budget', 'worker_lost', 'persistence', 'internal');
        ALTER TABLE experiment_cell_attempts ALTER COLUMN official_judge_status SET DEFAULT 'not_scheduled';
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'experiment_cell_attempts'::regclass AND conname = 'experiment_cell_attempts_score_judge_status_check') THEN
            ALTER TABLE experiment_cell_attempts ADD CONSTRAINT experiment_cell_attempts_score_judge_status_check CHECK ((official_judge_status = 'succeeded' AND score IS NOT NULL) OR (official_judge_status <> 'succeeded' AND score IS NULL));
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'experiment_cell_attempts'::regclass AND conname = 'experiment_cell_attempts_execution_status_check') THEN
            ALTER TABLE experiment_cell_attempts ADD CONSTRAINT experiment_cell_attempts_execution_status_check CHECK (execution_status IN ('queued', 'running', 'completed', 'failed', 'timed_out', 'cancelled', 'budget_exhausted'));
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'experiment_cell_attempts'::regclass AND conname = 'experiment_cell_attempts_result_disposition_check') THEN
            ALTER TABLE experiment_cell_attempts ADD CONSTRAINT experiment_cell_attempts_result_disposition_check CHECK (result_disposition IN ('tested', 'inconclusive', 'not_tested'));
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'experiment_cell_attempts'::regclass AND conname = 'experiment_cell_attempts_judge_status_check') THEN
            ALTER TABLE experiment_cell_attempts ADD CONSTRAINT experiment_cell_attempts_judge_status_check CHECK (official_judge_status IN ('not_scheduled', 'scheduled', 'running', 'succeeded', 'failed', 'skipped'));
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'experiment_cell_attempts'::regclass AND conname = 'experiment_cell_attempts_failure_category_check') THEN
            ALTER TABLE experiment_cell_attempts ADD CONSTRAINT experiment_cell_attempts_failure_category_check CHECK (failure_category IS NULL OR failure_category IN ('provider', 'timeout', 'judge_parse', 'validation', 'cancelled', 'budget', 'worker_lost', 'persistence', 'internal'));
        END IF;
    END $$
    """,
    """
    INSERT INTO experiment_cell_attempts (
        attempt_id, cell_id, attempt_number, run_id, retry_kind, spec_hash,
        execution_status, result_disposition, official_judge_status, score, asr,
        failure_category, selection_reason, started_at, completed_at
    )
    SELECT
        'attempt-' || md5(cell.cell_id || E'\\x1f1'), cell.cell_id, 1, cell.run_id,
        'initial', config.spec_hash,
        CASE
            WHEN COALESCE(BTRIM(cell.error), '') <> '' THEN 'failed'
            WHEN cell.status IN ('failed', 'error') THEN 'failed'
            WHEN cell.status IN ('cancelled', 'canceled') THEN 'cancelled'
            WHEN cell.status IN ('timed_out', 'timeout') THEN 'timed_out'
            WHEN cell.status IN ('pending', 'queued') THEN 'queued'
            WHEN cell.status = 'running' THEN 'running'
            WHEN cell.status IN ('completed', 'succeeded') THEN 'completed'
            ELSE 'failed'
        END,
        CASE
            WHEN COALESCE(BTRIM(cell.error), '') <> '' THEN 'inconclusive'
            WHEN cell.status IN ('failed', 'error', 'cancelled', 'canceled', 'timed_out', 'timeout') THEN 'inconclusive'
            WHEN cell.status IN ('pending', 'queued', 'running') THEN 'not_tested'
            WHEN cell.status IN ('completed', 'succeeded') AND cell.score IS NOT NULL THEN 'tested'
            ELSE 'inconclusive'
        END,
        CASE
            WHEN COALESCE(BTRIM(cell.error), '') <> '' THEN 'failed'
            WHEN cell.status IN ('failed', 'error', 'cancelled', 'canceled', 'timed_out', 'timeout') THEN 'failed'
            WHEN cell.status IN ('pending', 'queued', 'running') THEN 'not_scheduled'
            WHEN cell.status IN ('completed', 'succeeded') AND cell.score IS NOT NULL THEN 'succeeded'
            ELSE 'failed'
        END,
        CASE
            WHEN COALESCE(BTRIM(cell.error), '') = ''
                 AND cell.status IN ('completed', 'succeeded') AND cell.score IS NOT NULL
                THEN cell.score
            ELSE NULL
        END,
        CASE
            WHEN COALESCE(BTRIM(cell.error), '') = ''
                 AND cell.status IN ('completed', 'succeeded') AND cell.score IS NOT NULL
                THEN cell.asr
            ELSE NULL
        END,
        CASE
            WHEN COALESCE(BTRIM(cell.error), '') <> '' OR cell.status IN ('failed', 'error') THEN 'internal'
            WHEN cell.status IN ('cancelled', 'canceled') THEN 'cancelled'
            WHEN cell.status IN ('timed_out', 'timeout') THEN 'timeout'
            ELSE NULL
        END,
        'legacy_backfill', cell.completed_at, cell.completed_at
    FROM experiment_cells AS cell
    JOIN experiment_technique_configs AS config
        ON config.experiment_id = cell.experiment_id
        AND config.technique_config_id = cell.technique_config_id
    WHERE NOT (
        COALESCE(cell.status, 'pending') IN ('pending', 'queued')
        AND COALESCE(BTRIM(cell.run_id), '') = ''
        AND COALESCE(BTRIM(cell.error), '') = ''
        AND cell.completed_at IS NULL
    )
    ON CONFLICT DO NOTHING
    """,
    """
    DO $$
    DECLARE
        item RECORD;
        index_oid OID;
        actual_keys TEXT[];
        actual_predicate TEXT;
        actual_access_method TEXT;
        actual_operator_classes OID[];
        expected_operator_classes OID[];
        actual_collations OID[];
        expected_collations OID[];
        actual_options SMALLINT[];
        actual_nulls_not_distinct BOOLEAN;
    BEGIN
        FOR item IN SELECT * FROM (VALUES
            ('uq_experiment_cells_technique_config_prompt_repeat', 'experiment_cells', ARRAY['experiment_id', 'technique_config_id', 'prompt_id', 'repeat_index']::TEXT[], NULL::TEXT),
            ('uq_active_attempt_per_cell', 'experiment_cell_attempts', ARRAY['cell_id']::TEXT[], '(execution_status = ''running''::text)')
        ) AS expected(index_name, table_name, key_columns, predicate) LOOP
            SELECT index_class.oid,
                   array_agg(attribute.attname ORDER BY key_column.ordinality),
                   regexp_replace(COALESCE(pg_get_expr(index_definition.indpred, index_definition.indrelid), ''), '[[:space:]]+', '', 'g'),
                   access_method.amname,
                   array_agg(index_definition.indclass[(key_column.ordinality - 1)::integer] ORDER BY key_column.ordinality),
                   array_agg(default_operator_class.oid ORDER BY key_column.ordinality),
                   array_agg(index_definition.indcollation[(key_column.ordinality - 1)::integer] ORDER BY key_column.ordinality),
                   array_agg(attribute.attcollation ORDER BY key_column.ordinality),
                   array_agg(index_definition.indoption[(key_column.ordinality - 1)::integer] ORDER BY key_column.ordinality),
                   index_definition.indnullsnotdistinct
            INTO index_oid, actual_keys, actual_predicate, actual_access_method,
                 actual_operator_classes, expected_operator_classes,
                 actual_collations, expected_collations, actual_options,
                 actual_nulls_not_distinct
            FROM pg_class index_class
            JOIN pg_namespace index_schema ON index_schema.oid = index_class.relnamespace
            JOIN pg_index index_definition ON index_definition.indexrelid = index_class.oid
            JOIN pg_am access_method ON access_method.oid = index_class.relam
            LEFT JOIN LATERAL unnest(index_definition.indkey) WITH ORDINALITY AS key_column(attnum, ordinality)
                ON key_column.ordinality <= index_definition.indnkeyatts
            LEFT JOIN pg_attribute attribute
                ON attribute.attrelid = index_definition.indrelid AND attribute.attnum = key_column.attnum
            LEFT JOIN pg_opclass default_operator_class
                ON default_operator_class.opcmethod = access_method.oid
                AND default_operator_class.opcdefault
                AND default_operator_class.opcintype = attribute.atttypid
            WHERE index_schema.nspname = current_schema() AND index_class.relname = item.index_name
            GROUP BY index_class.oid, access_method.amname, index_definition.indrelid, index_definition.indisunique,
                     index_definition.indisvalid, index_definition.indisready, index_definition.indimmediate,
                     index_definition.indnkeyatts, index_definition.indnatts, index_definition.indpred,
                     index_definition.indnullsnotdistinct;
            IF index_oid IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM pg_index index_definition
                WHERE index_definition.indexrelid = index_oid
                  AND index_definition.indrelid = item.table_name::regclass
                  AND index_definition.indisunique AND index_definition.indisvalid
                  AND index_definition.indisready AND index_definition.indimmediate
                  AND actual_access_method = 'btree'
                  AND index_definition.indnkeyatts = cardinality(item.key_columns)
                  AND index_definition.indnatts = cardinality(item.key_columns)
                  AND actual_keys = item.key_columns
                  AND actual_predicate = regexp_replace(COALESCE(item.predicate, ''), '[[:space:]]+', '', 'g')
                  AND actual_operator_classes = expected_operator_classes
                  AND actual_collations = expected_collations
                  AND actual_options = array_fill(0::SMALLINT, ARRAY[cardinality(item.key_columns)])
                  AND NOT actual_nulls_not_distinct
            ) THEN
                EXECUTE format('DROP INDEX %I', item.index_name);
            END IF;
        END LOOP;
    END $$
    """,
    """
    DO $$
    DECLARE legacy_constraint RECORD;
    BEGIN
        FOR legacy_constraint IN
            SELECT constraint_definition.conname
            FROM pg_constraint AS constraint_definition
            JOIN LATERAL unnest(constraint_definition.conkey) WITH ORDINALITY
                AS key_column(attnum, ordinality) ON TRUE
            JOIN pg_attribute AS attribute
                ON attribute.attrelid = constraint_definition.conrelid
                AND attribute.attnum = key_column.attnum
            WHERE constraint_definition.conrelid = 'experiment_cells'::regclass
              AND constraint_definition.contype = 'u'
            GROUP BY constraint_definition.oid, constraint_definition.conname
            HAVING array_agg(attribute.attname::TEXT ORDER BY key_column.ordinality)
                = ARRAY['experiment_id', 'technique_key', 'prompt_id', 'repeat_index']::TEXT[]
        LOOP
            EXECUTE format(
                'ALTER TABLE experiment_cells DROP CONSTRAINT %I',
                legacy_constraint.conname
            );
        END LOOP;
    END $$
    """,
    "DROP INDEX IF EXISTS uq_experiment_cells_legacy_technique_prompt_repeat",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_cells_technique_config_prompt_repeat ON experiment_cells (experiment_id, technique_config_id, prompt_id, repeat_index)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_attempt_per_cell ON experiment_cell_attempts (cell_id) WHERE execution_status = 'running'",
    "ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS selected_attempt_id TEXT",
    "ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS latest_attempt_id TEXT",
    "ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS active_attempt_id TEXT",
    """
    UPDATE experiment_cells AS cell
    SET selected_attempt_id = COALESCE(cell.selected_attempt_id, attempt.attempt_id),
        latest_attempt_id = COALESCE(cell.latest_attempt_id, attempt.attempt_id),
        active_attempt_id = CASE
            WHEN attempt.execution_status = 'running' THEN COALESCE(cell.active_attempt_id, attempt.attempt_id)
            ELSE cell.active_attempt_id
        END
    FROM experiment_cell_attempts AS attempt
    WHERE attempt.cell_id = cell.cell_id AND attempt.attempt_number = 1
    """,
    """
    DO $$
    DECLARE item RECORD; actual_name TEXT;
    BEGIN
        FOR item IN SELECT * FROM (VALUES
            ('experiment_cells_selected_attempt_fkey', 'FOREIGN KEY (cell_id, selected_attempt_id) REFERENCES experiment_cell_attempts(cell_id, attempt_id) ON DELETE RESTRICT'),
            ('experiment_cells_latest_attempt_fkey', 'FOREIGN KEY (cell_id, latest_attempt_id) REFERENCES experiment_cell_attempts(cell_id, attempt_id) ON DELETE RESTRICT'),
            ('experiment_cells_active_attempt_fkey', 'FOREIGN KEY (cell_id, active_attempt_id) REFERENCES experiment_cell_attempts(cell_id, attempt_id) ON DELETE RESTRICT')
        ) AS expected(name, definition) LOOP
            SELECT conname INTO actual_name FROM pg_constraint
            WHERE conrelid = 'experiment_cells'::regclass AND conname = item.name
              AND (regexp_replace(pg_get_constraintdef(oid), '[[:space:]]+', '', 'g')
                   <> regexp_replace(item.definition, '[[:space:]]+', '', 'g')
                   OR NOT convalidated OR condeferrable OR condeferred);
            IF actual_name IS NOT NULL THEN
                EXECUTE format('ALTER TABLE experiment_cells DROP CONSTRAINT %I', actual_name);
            END IF;
        END LOOP;
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'experiment_cells'::regclass AND conname = 'experiment_cells_selected_attempt_fkey') THEN
            ALTER TABLE experiment_cells ADD CONSTRAINT experiment_cells_selected_attempt_fkey FOREIGN KEY (cell_id, selected_attempt_id) REFERENCES experiment_cell_attempts (cell_id, attempt_id) ON DELETE RESTRICT;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'experiment_cells'::regclass AND conname = 'experiment_cells_latest_attempt_fkey') THEN
            ALTER TABLE experiment_cells ADD CONSTRAINT experiment_cells_latest_attempt_fkey FOREIGN KEY (cell_id, latest_attempt_id) REFERENCES experiment_cell_attempts (cell_id, attempt_id) ON DELETE RESTRICT;
        END IF;
        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'experiment_cells'::regclass AND conname = 'experiment_cells_active_attempt_fkey') THEN
            ALTER TABLE experiment_cells ADD CONSTRAINT experiment_cells_active_attempt_fkey FOREIGN KEY (cell_id, active_attempt_id) REFERENCES experiment_cell_attempts (cell_id, attempt_id) ON DELETE RESTRICT;
        END IF;
    END $$
    """,
    """
    CREATE TABLE IF NOT EXISTS retry_batches (
        retry_batch_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
        request_id TEXT, action TEXT NOT NULL,
        completed_policy TEXT NOT NULL DEFAULT 'exclude',
        status TEXT NOT NULL DEFAULT 'requested',
        requested_by TEXT NOT NULL DEFAULT '', selection JSONB NOT NULL DEFAULT '{}'::jsonb,
        execution_generation BIGINT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(), started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retry_batch_cells (
        retry_batch_id TEXT NOT NULL REFERENCES retry_batches(retry_batch_id) ON DELETE CASCADE,
        cell_id TEXT NOT NULL REFERENCES experiment_cells(cell_id) ON DELETE CASCADE,
        member_ordinal INTEGER NOT NULL CHECK (member_ordinal >= 0),
        status TEXT NOT NULL DEFAULT 'requested',
        attempt_id TEXT REFERENCES experiment_cell_attempts(attempt_id),
        error_category TEXT, completed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (retry_batch_id, cell_id)
    )
    """,
    "ALTER TABLE retry_batches ADD COLUMN IF NOT EXISTS request_id TEXT",
    "ALTER TABLE retry_batches ADD COLUMN IF NOT EXISTS completed_policy TEXT NOT NULL DEFAULT 'exclude'",
    "ALTER TABLE retry_batches ADD COLUMN IF NOT EXISTS execution_generation BIGINT",
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'retry_batches'::regclass
              AND conname = 'retry_batches_execution_generation_check'
        ) THEN
            ALTER TABLE retry_batches
                ADD CONSTRAINT retry_batches_execution_generation_check
                CHECK (execution_generation IS NULL OR execution_generation >= 0);
        END IF;
    END $$
    """,
    "ALTER TABLE retry_batch_cells ADD COLUMN IF NOT EXISTS member_ordinal INTEGER",
    """
    WITH ranked AS (
        SELECT retry_batch_id, cell_id,
               row_number() OVER (
                   PARTITION BY retry_batch_id ORDER BY created_at, cell_id
               ) - 1 AS member_ordinal
        FROM retry_batch_cells
        WHERE member_ordinal IS NULL
    )
    UPDATE retry_batch_cells AS member
    SET member_ordinal = ranked.member_ordinal
    FROM ranked
    WHERE member.retry_batch_id = ranked.retry_batch_id
      AND member.cell_id = ranked.cell_id
    """,
    "ALTER TABLE retry_batch_cells ALTER COLUMN member_ordinal SET NOT NULL",
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid = 'retry_batch_cells'::regclass
              AND conname = 'retry_batch_cells_member_ordinal_check'
        ) THEN
            ALTER TABLE retry_batch_cells
                ADD CONSTRAINT retry_batch_cells_member_ordinal_check
                CHECK (member_ordinal >= 0);
        END IF;
    END $$
    """,
    "ALTER TABLE retry_batch_cells ADD COLUMN IF NOT EXISTS error_category TEXT",
    "ALTER TABLE retry_batch_cells ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_retry_batches_experiment_request ON retry_batches (experiment_id, request_id) WHERE request_id IS NOT NULL",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_retry_batch_cells_member_ordinal ON retry_batch_cells (retry_batch_id, member_ordinal)",
    """
    WITH phantom AS MATERIALIZED (
        SELECT cell.cell_id, attempt.attempt_id
        FROM experiment_cells AS cell
        JOIN experiment_cell_attempts AS attempt
          ON attempt.cell_id = cell.cell_id
        JOIN experiment_technique_configs AS config
          ON config.experiment_id = cell.experiment_id
         AND config.technique_config_id = cell.technique_config_id
        WHERE cell.selected_attempt_id = attempt.attempt_id
          AND cell.latest_attempt_id = attempt.attempt_id
          AND cell.active_attempt_id IS NULL
          AND COALESCE(cell.status, 'pending') IN ('pending', 'queued')
          AND COALESCE(BTRIM(cell.run_id), '') = ''
          AND COALESCE(BTRIM(cell.error), '') = ''
          AND cell.completed_at IS NULL
          AND attempt.attempt_id = 'attempt-' || md5(cell.cell_id || E'\\x1f1')
          AND attempt.attempt_number = 1
          AND attempt.parent_attempt_id IS NULL
          AND attempt.retry_kind = 'initial'
          AND attempt.spec_hash = config.spec_hash
          AND attempt.execution_status = 'queued'
          AND attempt.result_disposition = 'not_tested'
          AND attempt.official_judge_status = 'not_scheduled'
          AND attempt.score IS NULL
          AND attempt.asr IS NULL
          AND COALESCE(BTRIM(attempt.run_id), '') = ''
          AND attempt.failure_category IS NULL
          AND attempt.checkpoint_id IS NULL
          AND attempt.promotion_policy = 'replace_on_valid_completion'
          AND attempt.selection_reason = 'legacy_backfill'
          AND attempt.late_completion IS FALSE
          AND attempt.recoverable_cancellation IS FALSE
          AND attempt.worker_id IS NULL
          AND attempt.lease_expires_at IS NULL
          AND attempt.started_at IS NULL
          AND attempt.completed_at IS NULL
          AND NOT EXISTS (
              SELECT 1
              FROM experiment_cell_attempts AS sibling
              WHERE sibling.cell_id = attempt.cell_id
                AND sibling.attempt_id <> attempt.attempt_id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM retry_batch_cells AS member
              WHERE member.attempt_id = attempt.attempt_id
          )
        FOR UPDATE OF cell, attempt
    ),
    cleared AS (
        UPDATE experiment_cells AS cell
        SET selected_attempt_id = NULL,
            latest_attempt_id = NULL,
            active_attempt_id = NULL
        FROM phantom
        WHERE cell.cell_id = phantom.cell_id
          AND cell.selected_attempt_id = phantom.attempt_id
          AND cell.latest_attempt_id = phantom.attempt_id
          AND cell.active_attempt_id IS NULL
        RETURNING cell.cell_id
    )
    DELETE FROM experiment_cell_attempts AS attempt
    USING phantom, cleared
    WHERE attempt.attempt_id = phantom.attempt_id
      AND phantom.cell_id = cleared.cell_id
    """,
)


def _empty_mapping() -> Mapping[str, object]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class ModelCallRecord:
    model_call_id: str
    correlation: CorrelationIds
    role: str
    operation_id: str
    operation_attempt: int
    request: Mapping[str, object]
    request_hash: str

    def __post_init__(self) -> None:
        if self.operation_attempt < 1:
            raise ValueError("operation_attempt must be at least 1")
        object.__setattr__(self, "request", _freeze_json_mapping(self.request))


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    checkpoint_id: str
    correlation: CorrelationIds
    spec_hash: str
    state_schema_version: int
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.state_schema_version < 1:
            raise ValueError("state_schema_version must be at least 1")
        object.__setattr__(self, "payload", _freeze_json_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    correlation: CorrelationIds
    media_type: str
    content_hash: str
    metadata: Mapping[str, object] = field(default_factory=_empty_mapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class TechniquePackageRecord:
    """Durable package lifecycle facts, independent of loader process memory."""

    package_id: str
    technique_id: str
    technique_version: str
    implementation_id: str
    implementation_kind: str
    contract_version: str
    source_kind: str
    source_location: str
    state: str
    manifest_hash: str
    asset_hashes: Mapping[str, str] = field(default_factory=_empty_mapping)
    diagnostics: tuple[Mapping[str, str], ...] = ()
    is_registration: bool = True

    def __post_init__(self) -> None:
        if not self.package_id:
            raise ValueError("package_id must not be empty")
        if not self.source_kind or not self.source_location or not self.state:
            raise ValueError("package source and state must not be empty")
        if not all(isinstance(path, str) and isinstance(digest, str) for path, digest in self.asset_hashes.items()):
            raise TypeError("asset_hashes must be a string-to-string mapping")
        if not all(
            isinstance(diagnostic, Mapping)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in diagnostic.items())
            for diagnostic in self.diagnostics
        ):
            raise TypeError("diagnostics must contain string-to-string mappings")
        object.__setattr__(
            self,
            "asset_hashes",
            MappingProxyType(dict(self.asset_hashes)),
        )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(MappingProxyType(dict(item)) for item in self.diagnostics),
        )


class TechniqueRuntimePersistence(ABC):
    @abstractmethod
    def append_event(self, event: ExecutionEvent) -> None: ...

    @abstractmethod
    def append_events(self, events: tuple[ExecutionEvent, ...]) -> None:
        """Atomically append an ordered event batch."""

    @abstractmethod
    def record_model_call_requested(self, call: ModelCallRecord) -> None: ...

    @abstractmethod
    def complete_model_call(
        self,
        model_call_id: str,
        *,
        response: str,
        usage: Mapping[str, object],
        provider_response_id: str | None,
        gateway_call_id: str | None,
        gateway_attempts: Mapping[str, int],
    ) -> None: ...

    @abstractmethod
    def fail_model_call(
        self,
        model_call_id: str,
        *,
        category: str,
        error: str,
        indeterminate: bool,
    ) -> None: ...

    @abstractmethod
    def save_checkpoint(self, checkpoint: CheckpointRecord) -> None: ...

    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None: ...

    @abstractmethod
    def latest_checkpoint(
        self,
        *,
        experiment_id: str,
        cell_id: str,
        attempt_ids: tuple[str, ...],
    ) -> CheckpointRecord | None: ...

    @abstractmethod
    def attempt_has_indeterminate_call(self, attempt_id: str) -> bool: ...

    def attempt_has_unreconciled_call_events(self, attempt_id: str) -> bool:
        """Check durable call/event completeness before replaying an inactive lineage.

        Implementations must fail closed when recovery checks are unavailable.
        An in-flight call is also incomplete; callers must fence execution first.
        """
        raise NotImplementedError("model-call event recovery checks are not configured")

    @abstractmethod
    def save_artifact(self, artifact: ArtifactRecord) -> None: ...

    def save_package(self, record: TechniquePackageRecord) -> None:
        """Persist a package lifecycle transition where the backend supports it."""

        raise NotImplementedError("package lifecycle persistence is not configured")

    def load_package(self, package_id: str) -> TechniquePackageRecord | None:
        raise NotImplementedError("package lifecycle persistence is not configured")

    def load_package_by_reference(
        self,
        *,
        technique_id: str,
        technique_version: str,
        implementation_id: str,
    ) -> TechniquePackageRecord | None:
        """Load one immutable registration independently of its install path."""

        raise NotImplementedError("package lifecycle persistence is not configured")

    def has_package_references(
        self,
        *,
        technique_id: str,
        technique_version: str,
        implementation_id: str,
    ) -> bool:
        raise NotImplementedError("package lifecycle persistence is not configured")

    def hard_delete_package(self, record: TechniquePackageRecord) -> None:
        """Delete registration metadata only; package files are never touched here."""

        raise NotImplementedError("package lifecycle persistence is not configured")


class NoopTechniqueRuntimePersistence(TechniqueRuntimePersistence):
    def __init__(
        self,
        *,
        referenced_versions: set[tuple[str, str, str]] | None = None,
    ) -> None:
        self._event_hashes: dict[str, str] = {}
        self._event_sequences: dict[tuple[str, int], str] = {}
        self._model_call_event_records: dict[str, ExecutionEvent] = {}
        self._checkpoints: dict[str, CheckpointRecord] = {}
        self._model_calls: dict[str, dict[str, object]] = {}
        self._model_operations: dict[tuple[str, str, int], str] = {}
        self._artifacts: dict[str, ArtifactRecord] = {}
        self._packages: dict[str, TechniquePackageRecord] = {}
        self._package_versions: dict[tuple[str, str, str], TechniquePackageRecord] = {}
        self._referenced_versions = set(referenced_versions or ())
        self._event_lock = threading.Lock()

    def append_event(self, event: ExecutionEvent) -> None:
        self.append_events((event,))

    def append_events(self, events: tuple[ExecutionEvent, ...]) -> None:
        batch = _event_batch(events)
        with self._event_lock:
            event_hashes = dict(self._event_hashes)
            event_sequences = dict(self._event_sequences)
            for event in batch:
                event_hash = execution_event_hash(event)
                existing = event_hashes.get(event.event_id)
                if existing is not None and existing != event_hash:
                    raise ValueError(f"event ID '{event.event_id}' already exists with different content")
                sequence_key = (event.correlation.run_id, event.sequence)
                sequence_owner = event_sequences.get(sequence_key)
                if sequence_owner is not None and sequence_owner != event.event_id:
                    raise ValueError(f"run sequence {sequence_key!r} is already assigned to event '{sequence_owner}'")
                event_hashes[event.event_id] = event_hash
                event_sequences[sequence_key] = event.event_id
            self._event_hashes = event_hashes
            self._event_sequences = event_sequences
            self._model_call_event_records.update({
                event.event_id: event for event in batch
                if event.event_type in {"model.requested", "model.completed", "model.failed"}
            })

    def record_model_call_requested(self, call: ModelCallRecord) -> None:
        if call.model_call_id in self._model_calls:
            raise ValueError(f"model call '{call.model_call_id}' already exists")
        operation_key = (
            call.correlation.run_id,
            call.operation_id,
            call.operation_attempt,
        )
        operation_owner = self._model_operations.get(operation_key)
        if operation_owner is not None:
            raise ValueError(
                f"model operation attempt {operation_key!r} is already assigned to call '{operation_owner}'"
            )
        self._model_calls[call.model_call_id] = {
            "status": "requested",
            "call": call,
        }
        self._model_operations[operation_key] = call.model_call_id

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
        terminal = {
            "status": "completed",
            "response": response,
            "usage": _freeze_json_mapping(usage),
            "provider_response_id": provider_response_id,
            "gateway_call_id": gateway_call_id,
            "gateway_attempts": _freeze_json_mapping(gateway_attempts),
        }
        self._transition_model_call(model_call_id, terminal)

    def fail_model_call(
        self,
        model_call_id: str,
        *,
        category: str,
        error: str,
        indeterminate: bool,
    ) -> None:
        self._transition_model_call(
            model_call_id,
            {
                "status": "failed",
                "category": category,
                "error": error,
                "indeterminate": indeterminate,
            },
        )

    def save_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        existing = self._checkpoints.get(checkpoint.checkpoint_id)
        if existing is not None and existing != checkpoint:
            raise ValueError(f"checkpoint ID '{checkpoint.checkpoint_id}' already exists with different content")
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        return self._checkpoints.get(checkpoint_id)

    def latest_checkpoint(
        self,
        *,
        experiment_id: str,
        cell_id: str,
        attempt_ids: tuple[str, ...],
    ) -> CheckpointRecord | None:
        allowed_attempts = set(attempt_ids)
        candidates = [
            checkpoint
            for checkpoint in self._checkpoints.values()
            if checkpoint.correlation.experiment_id == experiment_id
            and checkpoint.correlation.cell_id == cell_id
            and checkpoint.correlation.attempt_id in allowed_attempts
        ]
        return candidates[-1] if candidates else None

    def attempt_has_indeterminate_call(self, attempt_id: str) -> bool:
        return any(
            item.get("status") == "failed"
            and item.get("indeterminate") is True
            and isinstance(item.get("call"), ModelCallRecord)
            and item["call"].correlation.attempt_id == attempt_id
            for item in self._model_calls.values()
        )

    def attempt_has_unreconciled_call_events(self, attempt_id: str) -> bool:
        with self._event_lock:
            events = tuple(self._model_call_event_records.values())
            return any(
                not _model_call_lifecycle_complete(item["call"], item["status"], events)
                for item in tuple(self._model_calls.values())
                if item["call"].correlation.attempt_id == attempt_id
            )

    def save_artifact(self, artifact: ArtifactRecord) -> None:
        existing = self._artifacts.get(artifact.artifact_id)
        if existing is not None and existing != artifact:
            raise ValueError(f"artifact ID '{artifact.artifact_id}' already exists with different content")
        self._artifacts[artifact.artifact_id] = artifact

    def save_package(self, record: TechniquePackageRecord) -> None:
        if record.is_registration:
            key = (
                record.technique_id,
                record.technique_version,
                record.implementation_id,
            )
            existing = self._package_versions.get(key)
            if existing is not None and _package_version_content(existing) != _package_version_content(record):
                raise ValueError(f"technique '{record.technique_id}' same version has different contents")
            self._package_versions[key] = record
        self._packages[record.package_id] = record

    def load_package(self, package_id: str) -> TechniquePackageRecord | None:
        return self._packages.get(package_id)

    def load_package_by_reference(
        self,
        *,
        technique_id: str,
        technique_version: str,
        implementation_id: str,
    ) -> TechniquePackageRecord | None:
        return self._package_versions.get((technique_id, technique_version, implementation_id))

    def has_package_references(
        self,
        *,
        technique_id: str,
        technique_version: str,
        implementation_id: str,
    ) -> bool:
        return (technique_id, technique_version, implementation_id) in self._referenced_versions

    def hard_delete_package(self, record: TechniquePackageRecord) -> None:
        if self.has_package_references(
            technique_id=record.technique_id,
            technique_version=record.technique_version,
            implementation_id=record.implementation_id,
        ):
            raise ValueError("package is referenced by persisted runs")
        self._packages.pop(record.package_id, None)
        if record.is_registration:
            self._package_versions.pop(
                (
                    record.technique_id,
                    record.technique_version,
                    record.implementation_id,
                ),
                None,
            )

    def _transition_model_call(
        self,
        model_call_id: str,
        terminal: dict[str, object],
    ) -> None:
        existing = self._model_calls.get(model_call_id)
        if existing is None:
            raise KeyError(f"model call '{model_call_id}' does not exist")
        status = existing["status"]
        if status == "requested":
            self._model_calls[model_call_id] = {
                "call": existing["call"],
                **terminal,
            }
            return
        comparable = {key: value for key, value in existing.items() if key != "call"}
        if comparable != terminal:
            raise ValueError(f"model call '{model_call_id}' is already terminal with different evidence")


class PostgreSQLTechniqueRuntimePersistence(TechniqueRuntimePersistence):
    def __init__(self, dsn: str | None = None) -> None:
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise ImportError(
                "PostgreSQLTechniqueRuntimePersistence requires the 'psycopg' "
                "package. Install it with: uv sync --extra postgres"
            ) from None
        self._dsn = dsn or os.environ.get(
            "DATABASE_URL",
            "postgresql://jailbreaker:jailbreaker@localhost:5432/jailbreaker",
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def append_event(self, event: ExecutionEvent) -> None:
        self.append_events((event,))

    def append_events(self, events: tuple[ExecutionEvent, ...]) -> None:
        batch = _event_batch(events)
        if not batch:
            return
        with self._connect() as connection, connection.transaction():
            for event in batch:
                self._append_event(connection, event)

    def _append_event(self, connection: object, event: ExecutionEvent) -> None:
        record = _event_record(event)
        event_hash = _canonical_hash(record)
        correlation = event.correlation
        inserted = connection.execute(
            """
            INSERT INTO execution_events (
                event_id, experiment_id, cell_id, run_id, attempt_id,
                sequence, event_type, schema_version, event_timestamp,
                technique_id, technique_version, implementation_id,
                round_number, branch_id, caused_by, operation_id,
                operation_attempt, payload, event_hash
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (event_id) DO NOTHING
            RETURNING event_hash
            """,
            (
                event.event_id,
                correlation.experiment_id,
                correlation.cell_id,
                correlation.run_id,
                correlation.attempt_id,
                event.sequence,
                event.event_type,
                event.schema_version,
                event.timestamp,
                event.technique_id,
                event.technique_version,
                event.implementation_id,
                event.round,
                event.branch_id,
                event.caused_by,
                event.operation_id,
                event.attempt,
                self._jsonb(event.payload),
                event_hash,
            ),
        ).fetchone()
        if inserted is not None:
            return
        persisted = connection.execute(
            "SELECT event_hash FROM execution_events WHERE event_id = %s",
            (event.event_id,),
        ).fetchone()
        if persisted is None or persisted[0] != event_hash:
            raise ValueError(f"event ID '{event.event_id}' already exists with different content")

    def record_model_call_requested(self, call: ModelCallRecord) -> None:
        correlation = call.correlation
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_calls (
                    model_call_id, experiment_id, cell_id, run_id, attempt_id,
                    role, operation_id, operation_attempt, request_payload,
                    request_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    call.model_call_id,
                    correlation.experiment_id,
                    correlation.cell_id,
                    correlation.run_id,
                    correlation.attempt_id,
                    call.role,
                    call.operation_id,
                    call.operation_attempt,
                    self._jsonb(call.request),
                    call.request_hash,
                ),
            )

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
        attempted_retries = gateway_attempts.get(
            "attempted_retries",
            gateway_attempts.get("retries", 0),
        )
        attempted_fallbacks = gateway_attempts.get(
            "attempted_fallbacks",
            gateway_attempts.get("fallbacks", 0),
        )
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE model_calls
                SET status = 'completed', response_text = %s, usage = %s,
                    provider_response_id = %s, gateway_call_id = %s,
                    gateway_attempted_retries = %s,
                    gateway_attempted_fallbacks = %s,
                    completed_at = now()
                WHERE model_call_id = %s AND status = 'requested'
                RETURNING model_call_id
                """,
                (
                    response,
                    self._jsonb(usage),
                    provider_response_id,
                    gateway_call_id,
                    attempted_retries,
                    attempted_fallbacks,
                    model_call_id,
                ),
            ).fetchone()
            if updated is None:
                persisted = connection.execute(
                    """
                    SELECT status, response_text, usage, provider_response_id,
                           gateway_call_id, gateway_attempted_retries,
                           gateway_attempted_fallbacks
                    FROM model_calls WHERE model_call_id = %s
                    """,
                    (model_call_id,),
                ).fetchone()
                expected = (
                    "completed",
                    response,
                    _to_json_value(usage),
                    provider_response_id,
                    gateway_call_id,
                    attempted_retries,
                    attempted_fallbacks,
                )
                if persisted != expected:
                    raise ValueError(f"model call '{model_call_id}' is already terminal with different evidence")

    def fail_model_call(
        self,
        model_call_id: str,
        *,
        category: str,
        error: str,
        indeterminate: bool,
    ) -> None:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE model_calls
                SET status = 'failed', failure_category = %s,
                    failure_error = %s, failure_indeterminate = %s,
                    completed_at = now()
                WHERE model_call_id = %s AND status = 'requested'
                RETURNING model_call_id
                """,
                (category, error, indeterminate, model_call_id),
            ).fetchone()
            if updated is None:
                persisted = connection.execute(
                    """
                    SELECT status, failure_category, failure_error,
                           failure_indeterminate
                    FROM model_calls WHERE model_call_id = %s
                    """,
                    (model_call_id,),
                ).fetchone()
                if persisted != ("failed", category, error, indeterminate):
                    raise ValueError(f"model call '{model_call_id}' is already terminal with different evidence")

    def save_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        correlation = checkpoint.correlation
        checkpoint_hash = _canonical_hash(_checkpoint_record(checkpoint))
        with self._connect() as connection:
            inserted = connection.execute(
                """
                INSERT INTO execution_checkpoints (
                    checkpoint_id, experiment_id, cell_id, run_id, attempt_id,
                    spec_hash, state_schema_version, payload, checkpoint_hash
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (checkpoint_id) DO NOTHING
                RETURNING checkpoint_hash
                """,
                (
                    checkpoint.checkpoint_id,
                    correlation.experiment_id,
                    correlation.cell_id,
                    correlation.run_id,
                    correlation.attempt_id,
                    checkpoint.spec_hash,
                    checkpoint.state_schema_version,
                    self._jsonb(checkpoint.payload),
                    checkpoint_hash,
                ),
            ).fetchone()
            if inserted is not None:
                return
            persisted = connection.execute(
                """
                SELECT checkpoint_hash FROM execution_checkpoints
                WHERE checkpoint_id = %s
                """,
                (checkpoint.checkpoint_id,),
            ).fetchone()
            if persisted is None or persisted[0] != checkpoint_hash:
                raise ValueError(f"checkpoint ID '{checkpoint.checkpoint_id}' already exists with different content")

    def load_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT checkpoint_id, experiment_id, cell_id, run_id, attempt_id,
                       spec_hash, state_schema_version, payload
                FROM execution_checkpoints
                WHERE checkpoint_id = %s
                """,
                (checkpoint_id,),
            ).fetchone()
        if row is None:
            return None
        return CheckpointRecord(
            checkpoint_id=row[0],
            correlation=CorrelationIds(
                experiment_id=row[1],
                cell_id=row[2],
                run_id=row[3],
                attempt_id=row[4],
            ),
            spec_hash=row[5],
            state_schema_version=row[6],
            payload=row[7],
        )

    def latest_checkpoint(
        self,
        *,
        experiment_id: str,
        cell_id: str,
        attempt_ids: tuple[str, ...],
    ) -> CheckpointRecord | None:
        if not attempt_ids:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT checkpoint_id, experiment_id, cell_id, run_id, attempt_id,
                       spec_hash, state_schema_version, payload
                FROM execution_checkpoints
                WHERE experiment_id = %s AND cell_id = %s
                  AND attempt_id = ANY(%s)
                ORDER BY created_at DESC, checkpoint_id DESC
                LIMIT 1
                """,
                (experiment_id, cell_id, list(attempt_ids)),
            ).fetchone()
        if row is None:
            return None
        return CheckpointRecord(
            checkpoint_id=row[0],
            correlation=CorrelationIds(
                experiment_id=row[1],
                cell_id=row[2],
                run_id=row[3],
                attempt_id=row[4],
            ),
            spec_hash=row[5],
            state_schema_version=row[6],
            payload=row[7],
        )

    def attempt_has_indeterminate_call(self, attempt_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM model_calls
                WHERE attempt_id = %s AND status = 'failed'
                  AND failure_indeterminate IS TRUE
                LIMIT 1
                """,
                (attempt_id,),
            ).fetchone()
        return row is not None

    def attempt_has_unreconciled_call_events(self, attempt_id: str) -> bool:
        # One database snapshot observes the call and the atomic root merge.
        # Local buffers and a reconstructed EventJournal cannot clear this gate.
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM model_calls AS call
                WHERE call.attempt_id = %s AND (
                    call.status NOT IN ('completed', 'failed') OR NOT EXISTS (
                        SELECT 1 FROM execution_events AS requested
                        JOIN execution_events AS terminal
                          ON terminal.run_id = requested.run_id
                         AND terminal.attempt_id = requested.attempt_id
                         AND terminal.experiment_id IS NOT DISTINCT FROM requested.experiment_id
                         AND terminal.cell_id IS NOT DISTINCT FROM requested.cell_id
                         AND terminal.operation_id = requested.operation_id
                         AND terminal.operation_attempt = requested.operation_attempt
                         AND terminal.payload ->> 'model_call_id' = call.model_call_id
                         AND terminal.event_type = 'model.' || call.status
                         AND terminal.sequence > requested.sequence
                        WHERE requested.run_id = call.run_id
                          AND requested.attempt_id = call.attempt_id
                          AND requested.experiment_id IS NOT DISTINCT FROM call.experiment_id
                          AND requested.cell_id IS NOT DISTINCT FROM call.cell_id
                          AND requested.operation_id = call.operation_id
                          AND requested.operation_attempt = call.operation_attempt
                          AND requested.payload ->> 'model_call_id' = call.model_call_id
                          AND requested.payload ->> 'role' = call.role
                          AND requested.event_type = 'model.requested'
                    )
                ) LIMIT 1
                """,
                (attempt_id,),
            ).fetchone()
        return row is not None

    def save_artifact(self, artifact: ArtifactRecord) -> None:
        correlation = artifact.correlation
        with self._connect() as connection:
            inserted = connection.execute(
                """
                INSERT INTO run_artifacts (
                    artifact_id, experiment_id, cell_id, run_id, attempt_id,
                    media_type, content_hash, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (artifact_id) DO NOTHING
                RETURNING artifact_id
                """,
                (
                    artifact.artifact_id,
                    correlation.experiment_id,
                    correlation.cell_id,
                    correlation.run_id,
                    correlation.attempt_id,
                    artifact.media_type,
                    artifact.content_hash,
                    self._jsonb(artifact.metadata),
                ),
            ).fetchone()
            if inserted is None:
                persisted = connection.execute(
                    """
                    SELECT experiment_id, cell_id, run_id, attempt_id,
                           media_type, content_hash, metadata
                    FROM run_artifacts WHERE artifact_id = %s
                    """,
                    (artifact.artifact_id,),
                ).fetchone()
                expected = (
                    correlation.experiment_id,
                    correlation.cell_id,
                    correlation.run_id,
                    correlation.attempt_id,
                    artifact.media_type,
                    artifact.content_hash,
                    _to_json_value(artifact.metadata),
                )
                if persisted != expected:
                    raise ValueError(f"artifact ID '{artifact.artifact_id}' already exists with different content")

    def save_package(self, record: TechniquePackageRecord) -> None:
        with self._connect() as connection, connection.transaction():
            connection.execute(
                """
                INSERT INTO technique_packages (
                    package_id, technique_id, technique_version, source_kind,
                    source_location, state, manifest_hash, diagnostics
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (package_id) DO UPDATE
                SET technique_id = EXCLUDED.technique_id,
                    technique_version = EXCLUDED.technique_version,
                    source_kind = EXCLUDED.source_kind,
                    source_location = EXCLUDED.source_location,
                    state = EXCLUDED.state,
                    manifest_hash = EXCLUDED.manifest_hash,
                    diagnostics = EXCLUDED.diagnostics,
                    updated_at = now()
                """,
                (
                    record.package_id,
                    record.technique_id,
                    record.technique_version,
                    record.source_kind,
                    record.source_location,
                    record.state,
                    record.manifest_hash,
                    self._jsonb([dict(item) for item in record.diagnostics]),
                ),
            )
            if not record.is_registration:
                return
            inserted = connection.execute(
                """
                INSERT INTO technique_versions (
                    technique_id, technique_version, implementation_id,
                    implementation_kind, contract_version, manifest_hash,
                    asset_hashes, package_id, enabled
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (technique_id, technique_version, implementation_id)
                DO NOTHING
                RETURNING technique_id
                """,
                (
                    record.technique_id,
                    record.technique_version,
                    record.implementation_id,
                    record.implementation_kind,
                    record.contract_version,
                    record.manifest_hash,
                    self._jsonb(record.asset_hashes),
                    record.package_id,
                    _package_enabled(record.state),
                ),
            ).fetchone()
            if inserted is not None:
                return
            existing = connection.execute(
                """
                SELECT implementation_kind, contract_version, manifest_hash,
                       asset_hashes
                FROM technique_versions
                WHERE technique_id = %s AND technique_version = %s
                  AND implementation_id = %s
                """,
                (
                    record.technique_id,
                    record.technique_version,
                    record.implementation_id,
                ),
            ).fetchone()
            if existing != (
                record.implementation_kind,
                record.contract_version,
                record.manifest_hash,
                _to_json_value(record.asset_hashes),
            ):
                raise ValueError(f"technique '{record.technique_id}' same version has different contents")
            connection.execute(
                """
                UPDATE technique_versions
                SET enabled = %s
                WHERE technique_id = %s AND technique_version = %s
                  AND implementation_id = %s
                """,
                (
                    _package_enabled(record.state),
                    record.technique_id,
                    record.technique_version,
                    record.implementation_id,
                ),
            )

    def load_package(self, package_id: str) -> TechniquePackageRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT package.package_id, package.technique_id,
                       package.technique_version, package.source_kind,
                       package.source_location, package.state,
                       package.manifest_hash, package.diagnostics,
                       version.implementation_id, version.implementation_kind,
                       version.contract_version, version.asset_hashes
                FROM technique_packages AS package
                LEFT JOIN technique_versions AS version
                  ON version.package_id = package.package_id
                WHERE package.package_id = %s
                ORDER BY version.created_at DESC NULLS LAST
                LIMIT 1
                """,
                (package_id,),
            ).fetchone()
        if row is None:
            return None
        diagnostics = row[7] if isinstance(row[7], list) else []
        return TechniquePackageRecord(
            package_id=row[0],
            technique_id=row[1],
            technique_version=row[2],
            implementation_id=row[8] or "",
            implementation_kind=row[9] or "",
            contract_version=row[10] or "",
            source_kind=row[3],
            source_location=row[4],
            state=row[5],
            manifest_hash=row[6],
            asset_hashes=row[11] if isinstance(row[11], Mapping) else {},
            diagnostics=tuple(item for item in diagnostics if isinstance(item, Mapping)),
            is_registration=row[8] is not None,
        )

    def load_package_by_reference(
        self,
        *,
        technique_id: str,
        technique_version: str,
        implementation_id: str,
    ) -> TechniquePackageRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT package.package_id, package.technique_id,
                       package.technique_version, package.source_kind,
                       package.source_location, package.state,
                       package.manifest_hash, package.diagnostics,
                       version.implementation_id, version.implementation_kind,
                       version.contract_version, version.asset_hashes
                FROM technique_versions AS version
                JOIN technique_packages AS package
                  ON package.package_id = version.package_id
                WHERE version.technique_id = %s
                  AND version.technique_version = %s
                  AND version.implementation_id = %s
                """,
                (technique_id, technique_version, implementation_id),
            ).fetchone()
        if row is None:
            return None
        diagnostics = row[7] if isinstance(row[7], list) else []
        return TechniquePackageRecord(
            package_id=row[0],
            technique_id=row[1],
            technique_version=row[2],
            implementation_id=row[8],
            implementation_kind=row[9],
            contract_version=row[10],
            source_kind=row[3],
            source_location=row[4],
            state=row[5],
            manifest_hash=row[6],
            asset_hashes=row[11] if isinstance(row[11], Mapping) else {},
            diagnostics=tuple(item for item in diagnostics if isinstance(item, Mapping)),
            is_registration=True,
        )

    def has_package_references(
        self,
        *,
        technique_id: str,
        technique_version: str,
        implementation_id: str,
    ) -> bool:
        with self._connect() as connection:
            return self._has_package_references(
                connection,
                technique_id=technique_id,
                technique_version=technique_version,
                implementation_id=implementation_id,
            )

    def hard_delete_package(self, record: TechniquePackageRecord) -> None:
        """Atomically remove lifecycle rows after locking every reference writer."""

        with self._connect() as connection:
            with connection.transaction():
                connection.execute(
                    """
                    LOCK TABLE experiments, experiment_technique_configs,
                               experiment_cells, experiment_cell_attempts,
                               execution_checkpoints, execution_events,
                               model_calls, run_artifacts, evaluation_runs,
                               technique_versions, technique_packages IN SHARE MODE
                    """
                )
                if self._has_package_references(
                    connection,
                    technique_id=record.technique_id,
                    technique_version=record.technique_version,
                    implementation_id=record.implementation_id,
                ):
                    raise ValueError("package is referenced by persisted runs")
                if record.is_registration:
                    connection.execute(
                        """
                        DELETE FROM technique_versions
                        WHERE technique_id = %s AND technique_version = %s
                          AND implementation_id = %s
                        """,
                        (
                            record.technique_id,
                            record.technique_version,
                            record.implementation_id,
                        ),
                    )
                connection.execute(
                    "DELETE FROM technique_packages WHERE package_id = %s",
                    (record.package_id,),
                )

    def _has_package_references(
        self,
        connection: object,
        *,
        technique_id: str,
        technique_version: str,
        implementation_id: str,
    ) -> bool:
        reference = {
            "technique_id": technique_id,
            "technique_version": technique_version,
            "implementation_id": implementation_id,
        }
        row = connection.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM experiments
                WHERE plan @> %s
                UNION ALL
                SELECT 1 FROM experiment_technique_configs
                WHERE technique_id = %s AND technique_version = %s
                  AND implementation_id = %s
                UNION ALL
                SELECT 1
                FROM experiment_cell_attempts AS attempt
                JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id
                JOIN experiment_technique_configs AS config
                  ON config.experiment_id = cell.experiment_id
                 AND config.technique_config_id = cell.technique_config_id
                WHERE config.technique_id = %s AND config.technique_version = %s
                  AND config.implementation_id = %s
                UNION ALL
                SELECT 1 FROM execution_events
                WHERE technique_id = %s AND technique_version = %s
                  AND implementation_id = %s
                UNION ALL
                SELECT 1
                FROM execution_checkpoints AS checkpoint
                JOIN experiment_cell_attempts AS attempt
                  ON attempt.attempt_id = checkpoint.attempt_id
                JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id
                JOIN experiment_technique_configs AS config
                  ON config.experiment_id = cell.experiment_id
                 AND config.technique_config_id = cell.technique_config_id
                WHERE config.technique_id = %s AND config.technique_version = %s
                  AND config.implementation_id = %s
                UNION ALL
                SELECT 1
                FROM run_artifacts AS artifact
                JOIN experiment_cell_attempts AS attempt
                  ON attempt.attempt_id = artifact.attempt_id
                JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id
                JOIN experiment_technique_configs AS config
                  ON config.experiment_id = cell.experiment_id
                 AND config.technique_config_id = cell.technique_config_id
                WHERE config.technique_id = %s AND config.technique_version = %s
                  AND config.implementation_id = %s
                UNION ALL
                SELECT 1
                FROM model_calls AS call
                JOIN experiment_cell_attempts AS attempt
                  ON attempt.attempt_id = call.attempt_id
                JOIN experiment_cells AS cell ON cell.cell_id = attempt.cell_id
                JOIN experiment_technique_configs AS config
                  ON config.experiment_id = cell.experiment_id
                 AND config.technique_config_id = cell.technique_config_id
                WHERE config.technique_id = %s AND config.technique_version = %s
                  AND config.implementation_id = %s
                UNION ALL
                SELECT 1 FROM evaluation_runs
                WHERE technique_key = %s
                  AND (
                      run_metadata @> %s
                      OR response_metadata @> %s
                  )
            )
            """,
            (
                self._jsonb({"technique_configs": [reference]}),
                technique_id,
                technique_version,
                implementation_id,
                technique_id,
                technique_version,
                implementation_id,
                technique_id,
                technique_version,
                implementation_id,
                technique_id,
                technique_version,
                implementation_id,
                technique_id,
                technique_version,
                implementation_id,
                technique_id,
                technique_version,
                implementation_id,
                technique_id,
                self._jsonb(reference),
                self._jsonb(reference),
            ),
        ).fetchone()
        return row is not None and bool(row[0])

    def _connect(self):
        import psycopg

        connection = psycopg.connect(self._dsn, autocommit=True)
        if not self._schema_ready:
            with self._schema_lock:
                if not self._schema_ready:
                    try:
                        ensure_technique_runtime_schema(connection)
                    except Exception:
                        connection.close()
                        raise
                    self._schema_ready = True
        return connection

    @staticmethod
    def _jsonb(value: object):
        from psycopg.types.json import Jsonb

        return Jsonb(_to_json_value(value))


def build_technique_runtime_persistence() -> TechniqueRuntimePersistence:
    try:
        return PostgreSQLTechniqueRuntimePersistence()
    except Exception as error:  # noqa: BLE001
        log.warning("Technique runtime persistence disabled: %s", error)
        return NoopTechniqueRuntimePersistence()


def ensure_technique_runtime_schema(connection: object) -> None:
    for statement in _RUNTIME_SCHEMA_STATEMENTS:
        connection.execute(statement)


def ensure_experiment_attempt_schema(connection: object) -> None:
    for statement in _EXPERIMENT_ATTEMPT_SCHEMA_STATEMENTS:
        connection.execute(statement)


def _model_call_lifecycle_complete(
    call: ModelCallRecord,
    status: str,
    events: tuple[ExecutionEvent, ...],
) -> bool:
    if status not in {"completed", "failed"}:
        return False
    matching = tuple(
        event for event in events
        if event.correlation == call.correlation
        and event.operation_id == call.operation_id
        and event.attempt == call.operation_attempt
        and event.payload.get("model_call_id") == call.model_call_id
    )
    return any(
        requested.sequence < terminal.sequence
        for requested in matching
        if requested.event_type == "model.requested" and requested.payload.get("role") == call.role
        for terminal in matching
        if terminal.event_type == f"model.{status}"
    )


def _event_record(event: ExecutionEvent) -> Mapping[str, object]:
    correlation = event.correlation
    return {
        "event_id": event.event_id,
        "correlation": {
            "experiment_id": correlation.experiment_id,
            "cell_id": correlation.cell_id,
            "run_id": correlation.run_id,
            "attempt_id": correlation.attempt_id,
        },
        "sequence": event.sequence,
        "event_type": event.event_type,
        "schema_version": event.schema_version,
        "timestamp": event.timestamp.isoformat(),
        "technique_id": event.technique_id,
        "technique_version": event.technique_version,
        "implementation_id": event.implementation_id,
        "round": event.round,
        "branch_id": event.branch_id,
        "caused_by": event.caused_by,
        "operation_id": event.operation_id,
        "attempt": event.attempt,
        "payload": event.payload,
    }


def execution_event_hash(event: ExecutionEvent) -> str:
    if type(event) is not ExecutionEvent:
        raise TypeError("execution events must be exact ExecutionEvent values")
    return _canonical_hash(_event_record(event))


def _event_batch(events: tuple[ExecutionEvent, ...]) -> tuple[ExecutionEvent, ...]:
    if type(events) is not tuple:
        raise TypeError("execution event batch must be a tuple")
    if any(type(event) is not ExecutionEvent for event in events):
        raise TypeError("execution event batches must contain exact ExecutionEvent values")
    return events


def _checkpoint_record(checkpoint: CheckpointRecord) -> Mapping[str, object]:
    correlation = checkpoint.correlation
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "correlation": {
            "experiment_id": correlation.experiment_id,
            "cell_id": correlation.cell_id,
            "run_id": correlation.run_id,
            "attempt_id": correlation.attempt_id,
        },
        "spec_hash": checkpoint.spec_hash,
        "state_schema_version": checkpoint.state_schema_version,
        "payload": checkpoint.payload,
    }


def _package_version_content(
    record: TechniquePackageRecord,
) -> tuple[object, ...]:
    return (
        record.implementation_kind,
        record.contract_version,
        record.manifest_hash,
        dict(record.asset_hashes),
    )


def _package_enabled(state: str) -> bool:
    return state not in {"disabled", "uninstalled", "invalid", "incompatible"}


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _to_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _to_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json_value(item) for item in value]
    return value
