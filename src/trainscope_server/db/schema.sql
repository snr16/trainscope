-- Portable schema (DuckDB + PostgreSQL). Timescale-specific statements live in timescale.sql.
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    name        TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    config_json TEXT,
    env_json    TEXT,
    started_at  DOUBLE PRECISION,
    ended_at    DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    step   BIGINT NOT NULL,
    ts     DOUBLE PRECISION NOT NULL,
    name   TEXT NOT NULL,
    layer  TEXT NOT NULL DEFAULT '',
    value  DOUBLE PRECISION,
    PRIMARY KEY (run_id, step, name, layer)
);

CREATE TABLE IF NOT EXISTS events (
    run_id       TEXT NOT NULL,
    step         BIGINT NOT NULL,
    ts           DOUBLE PRECISION NOT NULL,
    type         TEXT NOT NULL,
    payload_json TEXT,
    dedup_key    TEXT NOT NULL,
    PRIMARY KEY (run_id, dedup_key)
);

CREATE TABLE IF NOT EXISTS findings (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    detector      TEXT NOT NULL,
    severity      TEXT NOT NULL,
    step_start    BIGINT,
    step_end      BIGINT,
    summary       TEXT,
    evidence_json TEXT,
    created_at    DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS commands (
    id           TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    action       TEXT NOT NULL,
    args_json    TEXT,
    status       TEXT NOT NULL,
    requested_by TEXT,
    approved_by  TEXT,
    created_at   DOUBLE PRECISION,
    executed_at  DOUBLE PRECISION,
    result_json  TEXT
);
