FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=0 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Keep the environment outside /app because Compose bind-mounts backend source there in development.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --extra postgres

RUN addgroup --system app && adduser --system --ingroup app app

COPY --chown=app:app backend/app ./app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["python", "-m", "app.api"]
