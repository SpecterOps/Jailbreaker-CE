# Architecture

## Components

- `frontend/`: static operator console
- `backend/app/api/`: HTTP API and evaluation orchestration
- `backend/app/attacklib/`: built-in technique registry and prompt builders
- `backend/app/profile_vault.py`: encrypted LLM profile storage
- `database/init/`: fresh database schema and seed prompts

## Request Flow

The frontend calls the backend API, the backend resolves the selected target and optional attacker/judge profiles, the runner executes the chosen technique, and PostgreSQL stores completed results when available.

## Persistence

PostgreSQL stores runs, experiments, analysis views, custom techniques, and encrypted profile records.

## Technique Registry

Built-in techniques are registered in `backend/app/attacklib/registry.py`; custom techniques are stored through the API.
