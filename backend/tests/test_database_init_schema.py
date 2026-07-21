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
