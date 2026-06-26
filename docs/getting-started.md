# Getting Started

## Prerequisites

- Docker Desktop or Docker Engine with Compose
- Python 3.11+ for backend checks
- Node 22+ for frontend checks

## Start The Stack

```bash
cp .env.example .env
docker compose up -d --build frontend backend database
```

Compose publishes the frontend, backend, and database ports on `127.0.0.1` only.

Open `http://localhost:3000`.

## Create The Vault

Open the sidebar vault control, create a new passphrase, then save target, attacker, or judge profiles from the Dashboard.

## Run A First Evaluation

Select a target profile, enter a system prompt and intent, then run a single technique or a full evaluation from the Dashboard.

## Reset Local State

```bash
docker compose down -v
docker compose up -d --build frontend backend database
```
