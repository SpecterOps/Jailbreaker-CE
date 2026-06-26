# Local Development

## Python Environment

```bash
uv sync --locked --extra dev --extra docs --extra postgres
```

## Backend Checks

```bash
uv run --locked --extra dev --extra docs --extra postgres python -m pytest backend/tests
uv run --locked --extra dev --extra docs --extra postgres python -m compileall -q backend/app
```

## Documentation Check

```bash
uv run --locked --extra dev --extra docs --extra postgres mkdocs build --strict
```

## Frontend Checks

```bash
node --test frontend/tests/*.test.mjs
node --check frontend/server.mjs
node --check frontend/public/app.js
node --check frontend/public/analysis.js
node --check frontend/public/experiments.js
```

## Compose Validation

```bash
docker compose -f docker-compose.yml config --quiet
```

## Optional Scanners

```bash
uv run --locked --extra security scripts/run_optional_scanners.sh
```
