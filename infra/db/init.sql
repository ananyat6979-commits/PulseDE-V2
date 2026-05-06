-- Runs once on first container start via docker-entrypoint-initdb.d
-- Enables the TimescaleDB extension before Python creates any tables.
-- If this file is missing, create_hypertable() will fail silently.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable pg_stat_statements for query performance monitoring (Grafana)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create the application user role if using a separate superuser for init
-- (In docker-compose, POSTGRES_USER=pulsede already has superuser rights)
GRANT ALL PRIVILEGES ON DATABASE pulsede TO pulsede;
