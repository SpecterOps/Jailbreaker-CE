-- Versioned technique packages and normalized modular runtime evidence.

CREATE TABLE IF NOT EXISTS technique_packages (
    package_id TEXT PRIMARY KEY,
    technique_id TEXT NOT NULL,
    technique_version TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_location TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'validated',
    manifest_hash TEXT NOT NULL,
    diagnostics JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS technique_versions (
    technique_id TEXT NOT NULL,
    technique_version TEXT NOT NULL,
    implementation_id TEXT NOT NULL,
    implementation_kind TEXT NOT NULL,
    contract_version TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    asset_hashes JSONB NOT NULL DEFAULT '{}'::jsonb,
    package_id TEXT REFERENCES technique_packages(package_id),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (technique_id, technique_version, implementation_id)
);

CREATE TABLE IF NOT EXISTS execution_events (
    event_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    cell_id TEXT,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    event_timestamp TIMESTAMPTZ NOT NULL,
    technique_id TEXT NOT NULL,
    technique_version TEXT NOT NULL,
    implementation_id TEXT NOT NULL,
    round_number INTEGER,
    branch_id TEXT,
    caused_by TEXT,
    operation_id TEXT,
    operation_attempt INTEGER NOT NULL DEFAULT 1,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    event_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_execution_events_attempt
    ON execution_events (attempt_id, sequence);

CREATE TABLE IF NOT EXISTS execution_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    cell_id TEXT,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    spec_hash TEXT NOT NULL,
    state_schema_version INTEGER NOT NULL,
    payload JSONB NOT NULL,
    checkpoint_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_execution_checkpoints_attempt
    ON execution_checkpoints (attempt_id, created_at DESC);

CREATE TABLE IF NOT EXISTS model_calls (
    model_call_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    cell_id TEXT,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    role TEXT NOT NULL,
    operation_id TEXT NOT NULL,
    operation_attempt INTEGER NOT NULL,
    request_payload JSONB NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'requested',
    response_text TEXT,
    usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    provider_response_id TEXT,
    gateway_call_id TEXT,
    gateway_attempted_retries INTEGER,
    gateway_attempted_fallbacks INTEGER,
    failure_category TEXT,
    failure_error TEXT,
    failure_indeterminate BOOLEAN,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE (run_id, operation_id, operation_attempt)
);

CREATE INDEX IF NOT EXISTS idx_model_calls_attempt
    ON model_calls (attempt_id, requested_at);

CREATE TABLE IF NOT EXISTS run_artifacts (
    artifact_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    cell_id TEXT,
    run_id TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    media_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_run_artifacts_attempt
    ON run_artifacts (attempt_id, created_at);

-- Attempt-aware experiment persistence. The deterministic legacy IDs below
-- make this migration safe to reapply to an existing database volume.

ALTER TABLE experiments ADD COLUMN IF NOT EXISTS plan JSONB;

CREATE TABLE IF NOT EXISTS experiment_technique_configs (
    technique_config_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    technique_id TEXT NOT NULL,
    technique_version TEXT NOT NULL,
    implementation_id TEXT NOT NULL,
    preset TEXT,
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    resolved_spec JSONB NOT NULL DEFAULT '{}'::jsonb,
    spec_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (experiment_id, technique_config_id)
);

ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS cell_id TEXT;
ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS technique_config_id TEXT;

INSERT INTO experiment_technique_configs (
    technique_config_id, experiment_id, technique_id, technique_version,
    implementation_id, parameters, resolved_spec, spec_hash
)
SELECT DISTINCT
    COALESCE(
        cell.technique_config_id,
        'etc-' || md5(cell.experiment_id || E'\x1f' || cell.technique_key)
    ),
    cell.experiment_id,
    cell.technique_key,
    '0.2.0',
    'legacy',
    '{}'::jsonb,
    jsonb_build_object(
        'technique_key', cell.technique_key,
        'implementation_id', 'legacy'
    ),
    md5(cell.experiment_id || E'\x1f' || cell.technique_key || E'\x1flegacy')
FROM experiment_cells AS cell
WHERE cell.technique_config_id IS NULL
   OR NOT EXISTS (
        SELECT 1
        FROM experiment_technique_configs AS existing
        WHERE existing.experiment_id = cell.experiment_id
          AND existing.technique_config_id = cell.technique_config_id
   )
ON CONFLICT DO NOTHING;

UPDATE experiment_cells
SET cell_id = COALESCE(
        cell_id,
        'cell-' || md5(
            experiment_id || E'\x1f' || technique_key || E'\x1f' || prompt_id
            || E'\x1f' || repeat_index::text
        )
    ),
    technique_config_id = COALESCE(
        technique_config_id,
        'etc-' || md5(experiment_id || E'\x1f' || technique_key)
    );

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM experiment_cells
        WHERE cell_id IS NULL OR technique_config_id IS NULL
    ) THEN
        RAISE EXCEPTION 'experiment cell identity backfill left null values';
    END IF;
END $$;

ALTER TABLE experiment_cells ALTER COLUMN cell_id SET NOT NULL;
ALTER TABLE experiment_cells ALTER COLUMN technique_config_id SET NOT NULL;
ALTER TABLE experiment_cells ALTER COLUMN score DROP NOT NULL;
ALTER TABLE experiment_cells ALTER COLUMN asr DROP NOT NULL;

DO $$
DECLARE
    primary_key_columns TEXT[];
    primary_key_name TEXT;
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
END $$;

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
END $$;

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
END $$;

CREATE TABLE IF NOT EXISTS experiment_cell_attempts (
    attempt_id TEXT PRIMARY KEY,
    cell_id TEXT NOT NULL REFERENCES experiment_cells(cell_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    parent_attempt_id TEXT REFERENCES experiment_cell_attempts(attempt_id),
    run_id TEXT,
    retry_kind TEXT NOT NULL DEFAULT 'initial',
    spec_hash TEXT NOT NULL,
    execution_status TEXT NOT NULL,
    result_disposition TEXT NOT NULL,
    execution_generation BIGINT NOT NULL DEFAULT 0,
    official_judge_status TEXT NOT NULL DEFAULT 'not_scheduled',
    score DOUBLE PRECISION,
    asr INTEGER,
    failure_category TEXT,
    checkpoint_id TEXT,
    promotion_policy TEXT NOT NULL DEFAULT 'replace_on_valid_completion',
    selection_reason TEXT,
    late_completion BOOLEAN NOT NULL DEFAULT FALSE,
    recoverable_cancellation BOOLEAN NOT NULL DEFAULT FALSE,
    worker_id TEXT,
    lease_expires_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (cell_id, attempt_number),
    UNIQUE (cell_id, attempt_id),
    CONSTRAINT experiment_cell_attempts_score_judge_status_check CHECK (
        (official_judge_status = 'succeeded' AND score IS NOT NULL)
        OR (official_judge_status <> 'succeeded' AND score IS NULL)
    ),
    CONSTRAINT experiment_cell_attempts_execution_status_check CHECK (execution_status IN ('queued', 'running', 'completed', 'failed', 'timed_out', 'cancelled', 'budget_exhausted')),
    CONSTRAINT experiment_cell_attempts_result_disposition_check CHECK (result_disposition IN ('tested', 'inconclusive', 'not_tested')),
    CONSTRAINT experiment_cell_attempts_judge_status_check CHECK (official_judge_status IN ('not_scheduled', 'scheduled', 'running', 'succeeded', 'failed', 'skipped')),
    CONSTRAINT experiment_cell_attempts_failure_category_check CHECK (failure_category IS NULL OR failure_category IN ('provider', 'timeout', 'judge_parse', 'validation', 'cancelled', 'budget', 'worker_lost', 'persistence', 'internal'))
);

ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS worker_id TEXT;
ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;
ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS checkpoint_id TEXT;
ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS promotion_policy TEXT NOT NULL DEFAULT 'replace_on_valid_completion';
ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS selection_reason TEXT;
ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS late_completion BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS recoverable_cancellation BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE experiment_cell_attempts ADD COLUMN IF NOT EXISTS execution_generation BIGINT NOT NULL DEFAULT 0;
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
END $$;
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
END $$;
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
    ALTER TABLE experiment_cell_attempts
        ALTER COLUMN official_judge_status SET DEFAULT 'not_scheduled';
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
END $$;

INSERT INTO experiment_cell_attempts (
    attempt_id, cell_id, attempt_number, run_id, retry_kind, spec_hash,
    execution_status, result_disposition, official_judge_status, score, asr,
    failure_category, selection_reason, started_at, completed_at
)
SELECT
    'attempt-' || md5(cell.cell_id || E'\x1f1'),
    cell.cell_id,
    1,
    cell.run_id,
    'initial',
    config.spec_hash,
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
        WHEN cell.status IN ('failed', 'error', 'cancelled', 'canceled', 'timed_out', 'timeout')
            THEN 'inconclusive'
        WHEN cell.status IN ('pending', 'queued', 'running') THEN 'not_tested'
        WHEN cell.status IN ('completed', 'succeeded') AND cell.score IS NOT NULL THEN 'tested'
        ELSE 'inconclusive'
    END,
    CASE
        WHEN COALESCE(BTRIM(cell.error), '') <> '' THEN 'failed'
        WHEN cell.status IN ('failed', 'error', 'cancelled', 'canceled', 'timed_out', 'timeout')
            THEN 'failed'
        WHEN cell.status IN ('pending', 'queued', 'running') THEN 'not_scheduled'
        WHEN cell.status IN ('completed', 'succeeded') AND cell.score IS NOT NULL THEN 'succeeded'
        ELSE 'failed'
    END,
    CASE
        WHEN COALESCE(BTRIM(cell.error), '') = ''
             AND cell.status IN ('completed', 'succeeded')
             AND cell.score IS NOT NULL
            THEN cell.score
        ELSE NULL
    END,
    CASE
        WHEN COALESCE(BTRIM(cell.error), '') = ''
             AND cell.status IN ('completed', 'succeeded')
             AND cell.score IS NOT NULL
            THEN cell.asr
        ELSE NULL
    END,
    CASE
        WHEN COALESCE(BTRIM(cell.error), '') <> '' OR cell.status IN ('failed', 'error') THEN 'internal'
        WHEN cell.status IN ('cancelled', 'canceled') THEN 'cancelled'
        WHEN cell.status IN ('timed_out', 'timeout') THEN 'timeout'
        ELSE NULL
    END,
    'legacy_backfill',
    cell.completed_at,
    cell.completed_at
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
ON CONFLICT DO NOTHING;

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
END $$;

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
END $$;

DROP INDEX IF EXISTS uq_experiment_cells_legacy_technique_prompt_repeat;

CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_cells_technique_config_prompt_repeat
    ON experiment_cells (experiment_id, technique_config_id, prompt_id, repeat_index);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_attempt_per_cell
    ON experiment_cell_attempts (cell_id) WHERE execution_status = 'running';

ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS selected_attempt_id TEXT;
ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS latest_attempt_id TEXT;
ALTER TABLE experiment_cells ADD COLUMN IF NOT EXISTS active_attempt_id TEXT;

UPDATE experiment_cells AS cell
SET selected_attempt_id = COALESCE(cell.selected_attempt_id, attempt.attempt_id),
    latest_attempt_id = COALESCE(cell.latest_attempt_id, attempt.attempt_id),
    active_attempt_id = CASE
        WHEN attempt.execution_status = 'running' THEN COALESCE(cell.active_attempt_id, attempt.attempt_id)
        ELSE cell.active_attempt_id
    END
FROM experiment_cell_attempts AS attempt
WHERE attempt.cell_id = cell.cell_id AND attempt.attempt_number = 1;

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
END $$;

CREATE TABLE IF NOT EXISTS retry_batches (
    retry_batch_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    request_id TEXT,
    action TEXT NOT NULL,
    completed_policy TEXT NOT NULL DEFAULT 'exclude',
    status TEXT NOT NULL DEFAULT 'requested',
    requested_by TEXT NOT NULL DEFAULT '',
    selection JSONB NOT NULL DEFAULT '{}'::jsonb,
    execution_generation BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS retry_batch_cells (
    retry_batch_id TEXT NOT NULL REFERENCES retry_batches(retry_batch_id) ON DELETE CASCADE,
    cell_id TEXT NOT NULL REFERENCES experiment_cells(cell_id) ON DELETE CASCADE,
    member_ordinal INTEGER NOT NULL CHECK (member_ordinal >= 0),
    status TEXT NOT NULL DEFAULT 'requested',
    attempt_id TEXT REFERENCES experiment_cell_attempts(attempt_id),
    error_category TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE (retry_batch_id, cell_id)
);

ALTER TABLE retry_batches ADD COLUMN IF NOT EXISTS request_id TEXT;
ALTER TABLE retry_batches ADD COLUMN IF NOT EXISTS completed_policy TEXT NOT NULL DEFAULT 'exclude';
ALTER TABLE retry_batches ADD COLUMN IF NOT EXISTS execution_generation BIGINT;
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
END $$;
ALTER TABLE retry_batch_cells ADD COLUMN IF NOT EXISTS member_ordinal INTEGER;
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
  AND member.cell_id = ranked.cell_id;
ALTER TABLE retry_batch_cells ALTER COLUMN member_ordinal SET NOT NULL;
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
END $$;
ALTER TABLE retry_batch_cells ADD COLUMN IF NOT EXISTS error_category TEXT;
ALTER TABLE retry_batch_cells ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ;
CREATE UNIQUE INDEX IF NOT EXISTS uq_retry_batches_experiment_request
    ON retry_batches (experiment_id, request_id) WHERE request_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_retry_batch_cells_member_ordinal
    ON retry_batch_cells (retry_batch_id, member_ordinal);

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
      AND attempt.attempt_id = 'attempt-' || md5(cell.cell_id || E'\x1f1')
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
  AND phantom.cell_id = cleared.cell_id;
