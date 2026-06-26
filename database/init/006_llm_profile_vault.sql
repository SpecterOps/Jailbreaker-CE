CREATE TABLE IF NOT EXISTS llm_vault (
    vault_id          TEXT        PRIMARY KEY,
    salt_b64          TEXT        NOT NULL,
    kdf_name          TEXT        NOT NULL,
    kdf_iterations    INTEGER     NOT NULL,
    encrypted_canary  TEXT        NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_target_profiles (
    profile_id        TEXT        PRIMARY KEY,
    name              TEXT        NOT NULL,
    provider          TEXT        NOT NULL DEFAULT '',
    base_url          TEXT        NOT NULL DEFAULT '',
    model             TEXT        NOT NULL DEFAULT '',
    scope             TEXT        NOT NULL DEFAULT 'all',
    encrypted_api_key TEXT        NOT NULL DEFAULT '',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_target_profiles_scope
    ON llm_target_profiles (scope);

CREATE INDEX IF NOT EXISTS idx_llm_target_profiles_name
    ON llm_target_profiles (lower(name));
