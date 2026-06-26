# Backend

The backend is a dependency-light Python API and execution engine. Runtime code
uses the Python standard library plus `cryptography` for the encrypted LLM
profile vault. `psycopg[binary]` is used when PostgreSQL persistence, analysis
queries, or the Postgres-backed profile vault are enabled.

## Responsibilities

- Serve the REST API from `app/api/server.py`.
- Maintain process-local run, evaluation, dataset, and experiment state in
  `app/api/store.py`.
- Execute built-in and custom techniques through `app/execution/runner.py`.
- Adapt demo and OpenAI-compatible live targets from `app/targets/`.
- Persist runs, detections, conversations, traces, datasets, experiments, and
  custom techniques through `app/persistence/runs.py`.
- Encrypt and resolve reusable target/attacker/judge LLM profiles through
  `app/profile_vault.py`.
- Score outputs with deterministic detectors and refusal/judge helpers under
  `app/scoring/`.

## Design Map

```text
backend/app/
|-- api/          # HTTP routing and EvaluationStore orchestration
|-- attacklib/    # Technique classes, registry, rendering, iterative attacks
|-- evaluation/   # Dataclasses and enums shared across execution/scoring/API
|-- execution/    # TechniqueRunner and execution flows
|-- importers/    # Suite/prompt import helpers
|-- llm_client.py # Small OpenAI-compatible chat-completions client
|-- memory/       # Conversation memory backends
|-- persistence/  # No-op and PostgreSQL run persistence
|-- profile_vault.py # Encrypted reusable LLM target profiles
|-- reporting/    # Aggregate report builder
|-- scoring/      # Detectors, refusal scoring, aggregate scoring
`-- targets/      # Demo, live chatbot, and live agent adapters
```

## Run

Normal development should use the Compose stack from the repository root:

```bash
docker compose up -d --build frontend backend database
```

Backend-only fallback:

```bash
cd backend
uv run --locked --extra postgres python -m app.api
```

The direct backend fallback defaults to:

```text
postgresql://jailbreaker:jailbreaker@localhost:5432/jailbreaker
```

Set `DATABASE_URL` to override it.

## Test

```bash
cd backend
uv run --locked --extra dev --extra postgres python -m pytest tests
uv run --locked --extra dev --extra postgres python -m compileall -q app
```

## Adding Built-In Techniques

1. Add or update an `AttackTechnique` subclass under `app/attacklib/`.
2. Register it in `app/attacklib/registry.py`.
3. Add detector defaults in `app/attacklib/config.py` when surface defaults are
   not enough.
4. Add focused tests in `backend/tests/`.
5. Restart the backend container because the registry is imported at process
   start.

Custom one-off techniques should usually go through the Dashboard custom
technique builder instead of the built-in registry.

## Notes

- Reusable provider API keys should be stored through the encrypted profile
  vault. Manual one-off keys can still exist in request payloads and backend
  process memory during live runs.
- The API is intended for local/offline use. CORS is allowlist-based, but there
  is still no authentication.
