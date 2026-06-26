# Release Process

## Release Worktree

Create a dedicated release worktree from `main` and keep release-only changes isolated there.

## Validation Gate

Run `uv run --locked --extra dev --extra docs --extra postgres python scripts/verify_public_release.py`, `bash scripts/smoke_public_fresh_start.sh`, and `uv run --locked --extra security scripts/run_optional_scanners.sh` before tagging a release.

## Release Preparation

Confirm that the branch is clean, review the generated archive contents, and create the release tag only after the validation gate passes.

## Repository Settings

Before announcing a release, verify that the GitHub repository has branch protection, Dependabot alerts, secret scanning, push protection, and private vulnerability reporting enabled.
