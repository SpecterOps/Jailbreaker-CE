CREATE TABLE IF NOT EXISTS custom_techniques (
    technique_id            TEXT PRIMARY KEY,
    name                    TEXT NOT NULL DEFAULT '',
    summary                 TEXT NOT NULL DEFAULT '',
    target_system_prompt    TEXT NOT NULL DEFAULT '',
    mode                    TEXT NOT NULL DEFAULT 'single_shot',
    prompt_template         TEXT NOT NULL DEFAULT '',
    attacker_system_prompt  TEXT NOT NULL DEFAULT '',
    refinement_template     TEXT NOT NULL DEFAULT '',
    max_rounds              INTEGER NOT NULL DEFAULT 5,
    requested_by            TEXT NOT NULL DEFAULT '',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_custom_techniques_updated_at
    ON custom_techniques (updated_at DESC);
