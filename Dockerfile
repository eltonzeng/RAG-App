# Multi-stage build: install deps into a venv, then copy into a slim runtime.
FROM python:3.11-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1

# psycopg2-binary + asyncpg wheels are prebuilt; no compiler toolchain needed.
COPY requirements.txt .
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip \
    && /venv/bin/pip install -r requirements.txt


FROM python:3.11-slim AS runtime

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json

# Copy the application (see .dockerignore for exclusions).
COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default command runs the API; the UI service overrides this in compose.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
