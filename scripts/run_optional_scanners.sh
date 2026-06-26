#!/usr/bin/env bash
set -u

usage() {
  cat <<'USAGE'
Usage: scripts/run_optional_scanners.sh [--strict]

Runs optional shareability/security scanners that are installed on the local
machine. Missing tools are reported. With --strict, missing tools exit 2.
Scanner failures exit 1. Bandit findings (exit 1) are reported as advisory.
USAGE
}

strict=0
case "${1:-}" in
  "") ;;
  --strict) strict=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 64 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root" || exit 1

missing=0
failed=0
ran=0
advisory=0

run_tool() {
  name="$1"
  install_hint="$2"
  shift 2
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "MISSING: $name ($install_hint)"
    missing=$((missing + 1))
    return 0
  fi

  echo
  echo "RUN ($name): $*"
  ran=$((ran + 1))
  "$@"
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "FAIL: $name exited $status"
    failed=$((failed + 1))
  fi
}

run_advisory_tool() {
  name="$1"
  install_hint="$2"
  shift 2
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "MISSING: $name ($install_hint)"
    missing=$((missing + 1))
    return 0
  fi

  echo
  echo "RUN ($name): $*"
  ran=$((ran + 1))
  "$@"
  status=$?
  if [ "$status" -eq 1 ]; then
    echo "ADVISORY: $name exited $status (review findings above)"
    advisory=$((advisory + 1))
  elif [ "$status" -ne 0 ]; then
    echo "FAIL: $name exited $status"
    failed=$((failed + 1))
  fi
}

python_sources=(backend)
bandit_sources=(backend/app)
semgrep_sources=(backend frontend database)

run_tool ruff "uv run --locked --extra security scripts/run_optional_scanners.sh" ruff check "${python_sources[@]}"
run_advisory_tool bandit "uv run --locked --extra security scripts/run_optional_scanners.sh" bandit -c pyproject.toml -r "${bandit_sources[@]}"
run_tool semgrep "uv run --locked --extra security scripts/run_optional_scanners.sh" semgrep scan --config auto "${semgrep_sources[@]}"
trivy_args=(
  trivy fs
  --skip-dirs .git
  --skip-dirs .worktrees
  --skip-dirs site
  .
)
run_tool trivy "brew install trivy" "${trivy_args[@]}"
run_tool osv-scanner "brew install osv-scanner" osv-scanner --recursive .

gitleaks_args=(gitleaks detect --source . --no-banner)
if [ -f .gitleaks.toml ]; then
  gitleaks_args+=(--config .gitleaks.toml)
fi
run_tool gitleaks "brew install gitleaks" "${gitleaks_args[@]}"
run_tool shellcheck "brew install shellcheck" shellcheck scripts/run_optional_scanners.sh scripts/smoke_public_fresh_start.sh

dockerfiles=(backend/Dockerfile)
run_tool hadolint "brew install hadolint" hadolint "${dockerfiles[@]}"

echo
echo "Summary: ran=$ran missing=$missing failed=$failed advisory=$advisory strict=$strict"

if [ "$failed" -ne 0 ]; then
  exit 1
fi

if [ "$missing" -ne 0 ] && [ "$strict" -ne 0 ]; then
  exit 2
fi

exit 0
