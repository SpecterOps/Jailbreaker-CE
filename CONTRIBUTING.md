# Contributing

## Getting Help

Use GitHub issues for bugs and feature requests. Use `SECURITY.md` for vulnerability reporting.

## Bug Reports

Use the bug report template and include reproduction steps, expected behavior, actual behavior, and local environment details. Do not include secrets, private target URLs, or sensitive transcripts.

## Feature Requests

Use the feature request template and explain the problem, the proposed change, and alternatives you considered.

## Development Setup

```bash
cp .env.example .env
docker compose up -d --build frontend backend database
uv sync --locked --extra dev --extra docs --extra postgres
uv run --locked --extra dev --extra docs --extra postgres python -m pytest backend/tests
node --test frontend/tests/*.test.mjs
```

## Pull Requests

Create a branch, keep changes focused, add or update tests, run the local checks, and open a pull request against `main`.

## Security Issues

Do not open public issues for vulnerabilities. Follow `SECURITY.md` and use GitHub private vulnerability reporting.
