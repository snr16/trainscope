-- Applied after schema.sql when the timescaledb extension is available.
CREATE EXTENSION IF NOT EXISTS timescaledb;
-- metrics is keyed by step, so use step as the (integer) time dimension.
SELECT create_hypertable('metrics', 'step', chunk_time_interval => 100000, if_not_exists => TRUE,
                         migrate_data => TRUE);

-- Downsampled view for long runs: one row per 100-step bucket.
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics_100
WITH (timescaledb.continuous) AS
SELECT run_id, name, layer, time_bucket(100, step) AS bucket,
       avg(value) AS mean, min(value) AS min, max(value) AS max, last(value, step) AS last
FROM metrics
GROUP BY run_id, name, layer, bucket
WITH NO DATA;
