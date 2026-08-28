PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS app_user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_session (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES app_user(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_message (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('USER', 'ASSISTANT', 'SYSTEM')),
    content TEXT NOT NULL,
    token_count INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_session(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS intent_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    user_message_id INTEGER NOT NULL UNIQUE,
    assistant_message_id INTEGER UNIQUE,
    schema_version TEXT NOT NULL,
    primary_intent TEXT NOT NULL,
    task_spec_json TEXT NOT NULL,
    context_snapshot_json TEXT,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_session(id) ON DELETE CASCADE,
    FOREIGN KEY (user_message_id) REFERENCES chat_message(id),
    FOREIGN KEY (assistant_message_id) REFERENCES chat_message(id)
);

CREATE INDEX IF NOT EXISTS idx_chat_session_user_updated
    ON chat_session(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_chat_message_session_created
    ON chat_message(session_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_intent_analysis_session_created
    ON intent_analysis(session_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_intent_analysis_primary_intent
    ON intent_analysis(primary_intent);

CREATE UNIQUE INDEX IF NOT EXISTS idx_intent_analysis_assistant_message
    ON intent_analysis(assistant_message_id);

PRAGMA optimize;
