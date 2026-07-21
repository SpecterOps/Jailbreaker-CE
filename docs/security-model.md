# Security Model

## Local-Only Default

Jailbreaker is designed for local offline use. Compose publishes all default ports on `127.0.0.1` only.

## Vault And Secrets

Reusable provider keys are encrypted in the backend vault. The passphrase is not stored and must be re-entered after backend restart or manual lock.

## Database And Logs

Do not commit `.env`, logs, exported datasets, or transcripts that contain sensitive prompts or model output.

## Dataset Trust

Experiment dataset `gold_answer` values are treated as trusted Judge configuration and are placed in the Judge's system instructions. Import datasets only from trusted sources, keep instructions and secrets out of `gold_answer`, and review third-party datasets before running them. Gold answers are not sent to Target or Attacker models.

## Unsupported Deployment Modes

The backend has no authentication or authorization layer. Widening the default Compose bindings beyond `127.0.0.1` is an explicit operator change, not a supported default; do not expose it to the public internet without adding appropriate controls.
