# db/Dockerfile — image for running Alembic migrations.
#
# This image has exactly one job: run `alembic upgrade head` against
# DATABASE_URL and exit. Same image is used for:
#   - Local dev (via docker-compose, one-shot service)
#   - CI pipelines (migration smoke tests)
#   - Portainer / on-premise (triggered manually or by a deploy step)
#   - Azure (triggered as a Container Instance job)
#
# It is NOT the backend app image — the backend will have its own Dockerfile
# in a later task. Keeping them separate means:
#   - Migrations can run before the app image is even built
#   - A broken migration can't take down the backend image
#   - Prod migration jobs pull a much smaller image

FROM python:3.11-slim

# Don't write .pyc files, don't buffer stdout (logs show up immediately)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only what migrations need. No SQLAlchemy ORM models are executed
# here — Alembic imports them to read Base.metadata, but nothing hits the
# wider app.
RUN pip install --no-cache-dir \
        "sqlalchemy>=2.0" \
        "alembic>=1.13" \
        "psycopg[binary]>=3.1"

# Copy only the db package. No backend, no frontend, no CV code.
COPY db/ /app/db/

# Default command: run all pending migrations, then exit.
# Override for other alembic commands, e.g.:
#   docker run --rm -e DATABASE_URL=... <image> alembic -c db/alembic.ini current
CMD ["alembic", "-c", "db/alembic.ini", "upgrade", "head"]
