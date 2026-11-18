-- Applied after schema.sql when the timescaledb extension is available.
-- Statements are separated by marker lines (two dashes, a space, two at-signs) and run one at a time
-- (continuous aggregates cannot be created inside a transaction; one failure must not roll back the rest).
CREATE EXTENSION IF NOT EXISTS timescaledb;
-- @@
-- metrics is keyed by step, so step is the (integer) time dimension
SELECT create_hypertable('metrics', 'step', chunk_time_interval => 100000, if_not_exists => TRUE,
                         migrate_data => TRUE);
-- @@
-- integer-time hypertables need a "now" for continuous aggregates: the latest recorded step
CREATE OR REPLACE FUNCTION trainscope_max_step() RETURNS BIGINT LANGUAGE SQL STABLE AS
$$ SELECT COALESCE(max(step), 0) FROM metrics $$;
-- @@
SELECT set_integer_now_func('metrics', 'trainscope_max_step', replace_if_exists => TRUE);
-- @@
-- downsampled view for long runs: one row per 100-step bucket
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_100
WITH (timescaledb.continuous) AS
SELECT run_id, name, layer, time_bucket(100, step) AS bucket,
       avg(value) AS mean, min(value) AS min, max(value) AS max, last(value, step) AS last
FROM metrics
GROUP BY run_id, name, layer, bucket
WITH NO DATA;
-- @@
SELECT add_continuous_aggregate_policy('metrics_100', start_offset => NULL, end_offset => 100,
                                       schedule_interval => INTERVAL '1 minute', if_not_exists => TRUE);
