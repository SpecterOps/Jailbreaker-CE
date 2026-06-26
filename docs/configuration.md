# Configuration

## Environment File

Copy `.env.example` to `.env` and change only local ports or endpoint defaults that you need.

## Target, Attacker, And Judge Profiles

Reusable API keys belong in the encrypted vault. Browser storage keeps profile IDs and non-secret UI state only.

## Local Endpoints

Local OpenAI-compatible endpoints can use the local/no-key profile option.

## Docker Compose Ports

Frontend, backend, and PostgreSQL bind to `127.0.0.1` by default.
