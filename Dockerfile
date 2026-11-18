FROM ghcr.io/astral-sh/uv:0.13.8 AS uv
FROM python:3.13-slim-bookworm

COPY --from=uv /uv /uvx /bin/
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# GLiNER is a required dependency. Model weights are cached on first use, not baked in.
COPY pyproject.toml uv.lock README.md ./
# Keep the large dependency layer cached when application code changes.
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

COPY data/taxonomy ./data/taxonomy
COPY data/ontologies ./data/ontologies

# Local review server only; Compose exposes it exclusively on localhost.
CMD ["flask", "--app", "career_coach.app:create_app", "run", "--host=0.0.0.0", "--port=5000"]
