#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  CREATE ROLE metabase WITH LOGIN PASSWORD '${METABASE_DB_PASSWORD}';
  CREATE DATABASE metabaseappdb WITH OWNER metabase;
EOSQL