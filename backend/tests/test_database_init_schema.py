from pathlib import Path


def _database_init_sql() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((repo_root / "database" / "init").glob("*.sql"))
    )


def test_database_init_declares_experiment_tables() -> None:
    init_sql = _database_init_sql()

    required_tables = [
        "CREATE TABLE IF NOT EXISTS experiment_datasets",
        "CREATE TABLE IF NOT EXISTS experiment_dataset_prompts",
        "CREATE TABLE IF NOT EXISTS experiments",
        "CREATE TABLE IF NOT EXISTS experiment_cells",
    ]
    for fragment in required_tables:
        assert fragment in init_sql


def test_database_init_experiment_cells_support_repeats_and_metadata() -> None:
    init_sql = _database_init_sql()

    required_fragments = [
        "repeat_index INTEGER NOT NULL DEFAULT 1",
        "metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
        "PRIMARY KEY (experiment_id, technique_key, prompt_id, repeat_index)",
        "CREATE INDEX IF NOT EXISTS idx_experiment_cells_experiment_id",
    ]
    for fragment in required_fragments:
        assert fragment in init_sql


def test_database_init_dataset_prompts_support_optional_gold_answers() -> None:
    assert "gold_answer TEXT" in _database_init_sql()


def test_database_init_declares_modular_runtime_tables() -> None:
    init_sql = _database_init_sql()

    for table in (
        "technique_packages",
        "technique_versions",
        "execution_events",
        "execution_checkpoints",
        "model_calls",
        "run_artifacts",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in init_sql


def test_database_init_declares_attempt_aware_experiment_schema() -> None:
    init_sql = _database_init_sql()

    required = (
        "CREATE TABLE IF NOT EXISTS experiment_technique_configs",
        "ADD COLUMN IF NOT EXISTS cell_id TEXT",
        "CREATE TABLE IF NOT EXISTS experiment_cell_attempts",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_attempt_per_cell",
        "CREATE TABLE IF NOT EXISTS retry_batches",
        "CREATE TABLE IF NOT EXISTS retry_batch_cells",
        "selected_attempt_id",
        "latest_attempt_id",
        "active_attempt_id",
        "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS plan JSONB",
    )
    for fragment in required:
        assert fragment in init_sql


def test_retry_schema_persists_idempotency_lifecycle_and_late_evidence() -> None:
    from app.persistence.technique_runtime import _EXPERIMENT_ATTEMPT_SCHEMA_STATEMENTS

    for sql in (
        _database_init_sql(),
        "\n".join(_EXPERIMENT_ATTEMPT_SCHEMA_STATEMENTS),
    ):
        for fragment in (
            "promotion_policy TEXT",
            "late_completion BOOLEAN",
            "recoverable_cancellation BOOLEAN",
            "request_id TEXT",
            "completed_policy TEXT",
            "error_category TEXT",
            "completed_at TIMESTAMPTZ",
            "uq_retry_batches_experiment_request",
            "member_ordinal INTEGER",
            "retry_batch_cells_member_ordinal_check",
            "uq_retry_batch_cells_member_ordinal",
        ):
            assert fragment in sql


def test_owned_operation_fence_schema_has_fresh_runtime_parity() -> None:
    from app.persistence.technique_runtime import _EXPERIMENT_ATTEMPT_SCHEMA_STATEMENTS

    for sql in (
        _database_init_sql(),
        "\n".join(_EXPERIMENT_ATTEMPT_SCHEMA_STATEMENTS),
    ):
        for fragment in (
            "CREATE TABLE IF NOT EXISTS experiment_execution_operations",
            "execution_generation BIGINT NOT NULL CHECK (execution_generation >= 1)",
            "execution_owner_id TEXT NOT NULL",
            "status TEXT NOT NULL CHECK (status IN ('running', 'finished'))",
            "lease_expires_at TIMESTAMPTZ",
            "idx_experiment_execution_operations_owner",
        ):
            assert fragment in sql


def test_attempt_schema_repairs_away_legacy_technique_cell_uniqueness() -> None:
    from app.persistence.technique_runtime import _EXPERIMENT_ATTEMPT_SCHEMA_STATEMENTS

    init_sql = _database_init_sql()
    runtime_sql = "\n".join(_EXPERIMENT_ATTEMPT_SCHEMA_STATEMENTS)
    legacy_create = (
        "CREATE UNIQUE INDEX IF NOT EXISTS "
        "uq_experiment_cells_legacy_technique_prompt_repeat"
    )

    assert legacy_create not in init_sql
    assert legacy_create not in runtime_sql
    assert "DROP INDEX IF EXISTS uq_experiment_cells_legacy_technique_prompt_repeat" in init_sql
    assert "DROP INDEX IF EXISTS uq_experiment_cells_legacy_technique_prompt_repeat" in runtime_sql
    assert "array_agg(attribute.attname::TEXT ORDER BY key_column.ordinality)" in init_sql
    assert "array_agg(attribute.attname::TEXT ORDER BY key_column.ordinality)" in runtime_sql


def test_modular_runtime_schema_enforces_identity_and_ordering() -> None:
    init_sql = _database_init_sql()

    for constraint in (
        "UNIQUE (run_id, sequence)",
        "UNIQUE (run_id, operation_id, operation_attempt)",
        "event_hash TEXT NOT NULL",
        "provider_response_id TEXT",
        "gateway_call_id TEXT",
        "gateway_attempted_retries INTEGER",
        "gateway_attempted_fallbacks INTEGER",
    ):
        assert constraint in init_sql
