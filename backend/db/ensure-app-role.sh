#!/bin/sh
set -eu

# PostgreSQL only runs initdb scripts when the data directory is first created.
# Reconcile the runtime role on every production Compose start so an existing
# volume cannot retain an obsolete password after deployment credentials rotate.
export PGPASSWORD="$POSTGRES_PASSWORD"

psql --no-psqlrc --set=ON_ERROR_STOP=1 \
  --username "$POSTGRES_USER" \
  --dbname "$POSTGRES_DB" <<'SQL'
\getenv app_user POSTGRES_APP_USER
\getenv app_password POSTGRES_APP_PASSWORD

SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'app_user', :'app_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user')
\gexec

SELECT format('ALTER ROLE %I LOGIN PASSWORD %L', :'app_user', :'app_password')
\gexec
SQL
