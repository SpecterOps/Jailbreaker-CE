# Database

This folder owns PostgreSQL schema initialization for local Compose runs.

## Layout

- `init/`: scripts mounted into `/docker-entrypoint-initdb.d`; executed once
  when a fresh Postgres volume is created.
- `migrations/`: placeholder for future schema history. There is no migration
  runner yet.
- `seeds/`: placeholder for future seed packaging.

## Bootstrap Order

The current init scripts create:

1. conversation memory tables
2. evaluation run, detection, conversation, and trace tables
3. analysis views
4. custom technique tables
5. seed prompts
6. encrypted LLM profile vault tables
7. experiment datasets, experiments, and experiment cells

## Reset

From the repository root:

```bash
docker compose down -v
docker compose up -d --build frontend backend database
```

The backend also has `ensure_schema()` logic in `backend/app/persistence/runs.py`
and `backend/app/profile_vault.py` for compatibility with existing volumes and
direct local runs, but the Compose init scripts are the fresh-clone source of
truth.
