-- Conversation memory storage for iterative attack state.
-- Auto-executed by PostgreSQL on first docker-compose up.

CREATE TABLE IF NOT EXISTS conversation_messages (
    id              BIGSERIAL       PRIMARY KEY,
    conversation_id TEXT            NOT NULL,
    sequence        INTEGER         NOT NULL,
    role            TEXT            NOT NULL,
    content         TEXT            NOT NULL DEFAULT '',
    timestamp       TIMESTAMPTZ     NOT NULL DEFAULT now(),
    labels          JSONB           NOT NULL DEFAULT '{}'::jsonb,

    UNIQUE (conversation_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_conversation_messages_conv_id
    ON conversation_messages (conversation_id);

COMMENT ON TABLE conversation_messages IS
    'Stores every message across all attack conversation threads. '
    'Each conversation_id is a UUID grouping messages for one attack session. '
    'The sequence column orders messages within a conversation.';
