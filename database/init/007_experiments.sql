-- Experiment datasets, experiment metadata, and per-cell results.
-- Runtime ensure_schema() keeps these definitions compatible for existing
-- volumes; this file makes fresh Compose databases complete on first boot.

CREATE TABLE IF NOT EXISTS experiment_datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    prompt_count INTEGER NOT NULL DEFAULT 0,
    created_by TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment_dataset_prompts (
    dataset_id TEXT NOT NULL REFERENCES experiment_datasets(dataset_id) ON DELETE CASCADE,
    prompt_id TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    sort_order INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (dataset_id, prompt_id)
);

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
);

CREATE INDEX IF NOT EXISTS idx_experiments_created_at
    ON experiments (created_at DESC);

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
);

CREATE INDEX IF NOT EXISTS idx_experiment_cells_experiment_id
    ON experiment_cells (experiment_id);
