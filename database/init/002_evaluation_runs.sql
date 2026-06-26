-- Run-level persistence for evaluation metadata, conversation entries,
-- detections, and trace events.

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id              TEXT            PRIMARY KEY,
    run_kind            TEXT            NOT NULL,
    evaluation_id       TEXT,
    status              TEXT            NOT NULL,
    mode                TEXT            NOT NULL,
    requested_by        TEXT            NOT NULL DEFAULT '',
    technique_key       TEXT            NOT NULL,
    technique_name      TEXT            NOT NULL DEFAULT '',
    technique_family    TEXT            NOT NULL DEFAULT '',
    technique_surface   TEXT            NOT NULL DEFAULT '',
    target_kind         TEXT            NOT NULL,
    target_id           TEXT            NOT NULL,
    target_model        TEXT            NOT NULL DEFAULT '',
    target_base_url     TEXT            NOT NULL DEFAULT '',
    attacker_model      TEXT            NOT NULL DEFAULT '',
    attacker_base_url   TEXT            NOT NULL DEFAULT '',
    unsafe_intent       TEXT            NOT NULL DEFAULT '',
    system_prompt       TEXT            NOT NULL DEFAULT '',
    seed_prompt         TEXT            NOT NULL DEFAULT '',
    success             BOOLEAN         NOT NULL DEFAULT FALSE,
    score               INTEGER         NOT NULL DEFAULT 0,
    asr                 INTEGER         NOT NULL DEFAULT 0,
    matched_detections  INTEGER         NOT NULL DEFAULT 0,
    highest_severity    TEXT            NOT NULL DEFAULT 'info',
    final_prompt        TEXT            NOT NULL DEFAULT '',
    final_output        TEXT            NOT NULL DEFAULT '',
    response_metadata   JSONB           NOT NULL DEFAULT '{}'::jsonb,
    run_metadata        JSONB           NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ     NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluation_run_detections (
    run_id              TEXT            NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
    detection_index     INTEGER         NOT NULL,
    detection_type      TEXT            NOT NULL,
    severity            TEXT            NOT NULL,
    matched             BOOLEAN         NOT NULL,
    rationale           TEXT            NOT NULL DEFAULT '',
    evidence            JSONB           NOT NULL DEFAULT '[]'::jsonb,
    PRIMARY KEY (run_id, detection_index)
);

CREATE TABLE IF NOT EXISTS evaluation_run_conversations (
    run_id              TEXT            NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
    entry_index         INTEGER         NOT NULL,
    kind                TEXT            NOT NULL,
    payload             JSONB           NOT NULL,
    PRIMARY KEY (run_id, entry_index)
);

CREATE TABLE IF NOT EXISTS evaluation_run_traces (
    run_id              TEXT            NOT NULL REFERENCES evaluation_runs(run_id) ON DELETE CASCADE,
    trace_index         INTEGER         NOT NULL,
    event_type          TEXT            NOT NULL,
    actor               TEXT            NOT NULL DEFAULT 'system',
    message             TEXT            NOT NULL DEFAULT '',
    event_timestamp     TIMESTAMPTZ     NOT NULL DEFAULT now(),
    attributes          JSONB           NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (run_id, trace_index)
);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_completed_at
    ON evaluation_runs (completed_at);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_technique_key
    ON evaluation_runs (technique_key);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_target_model
    ON evaluation_runs (target_model);

CREATE INDEX IF NOT EXISTS idx_evaluation_runs_success
    ON evaluation_runs (success);
