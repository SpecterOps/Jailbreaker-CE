from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root: Path, relative_path: str, content: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _compose_service_names() -> set[str]:
    service_names: set[str] = set()
    in_services = False
    for line in (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8").splitlines():
        if line == "services:":
            in_services = True
            continue
        if in_services and line and not line.startswith(" "):
            break
        if in_services and line.startswith("  ") and not line.startswith("    "):
            service_names.add(line.strip().removesuffix(":"))
    return service_names


def _env_example_keys() -> set[str]:
    return {
        line.split("=", 1)[0]
        for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
    }


def _smoke_script_source() -> str:
    return (REPO_ROOT / "scripts" / "smoke_public_fresh_start.sh").read_text(encoding="utf-8")


def _optional_scanner_script_source() -> str:
    return (REPO_ROOT / "scripts" / "run_optional_scanners.sh").read_text(encoding="utf-8")


def test_public_release_version_is_synchronized() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    major_minor = ".".join(version.split(".")[:2])
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_project = next(package for package in lock["package"] if package["name"] == "jailbreaker")

    assert locked_project["version"] == version
    assert f'JailbreakerAPI/{major_minor}' in (
        REPO_ROOT / "backend" / "app" / "api" / "server.py"
    ).read_text(encoding="utf-8")
    for page in ("index.html", "experiments.html", "analysis.html", "history.html"):
        assert f'<span class="brand-version">v{major_minor}</span>' in (
            REPO_ROOT / "frontend" / "public" / page
        ).read_text(encoding="utf-8")
    assert f"## [{version}] - " in (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def _run_optional_scanners_with_fake_tools(
    tmp_path: Path,
    *,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    tool_source = r"""#!/usr/bin/env bash
set -eu
tool_name="$(basename "$0")"
status_var="FAKE_${tool_name//-/_}_STATUS"
eval "status=\${${status_var}:-0}"
printf '%s\n' "$tool_name $*" >> "${FAKE_SCANNER_LOG}"
exit "$status"
"""
    for name in ("ruff", "bandit", "semgrep", "trivy", "osv-scanner", "gitleaks", "shellcheck", "hadolint"):
        _write(bin_dir, name, tool_source)
        (bin_dir / name).chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_SCANNER_LOG"] = str(tmp_path / "scanner.log")
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", "scripts/run_optional_scanners.sh"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _run_smoke_with_fake_compose(
    tmp_path: Path,
    *,
    compose_up_status: int = 0,
    compose_down_status: int = 0,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _write(
        bin_dir,
        "docker",
        """#!/usr/bin/env bash
set -eu
for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "down" ]]; then
    exit "${FAKE_COMPOSE_DOWN_STATUS:-0}"
  fi
  if [[ "$arg" == "up" ]]; then
    {
      printf 'FRONTEND_PORT=%s\n' "${FRONTEND_PORT:-}"
      printf 'BACKEND_PORT=%s\n' "${BACKEND_PORT:-}"
      printf 'DATABASE_PORT=%s\n' "${DATABASE_PORT:-}"
    } > "${FAKE_COMPOSE_ENV_LOG}"
    exit "${FAKE_COMPOSE_UP_STATUS:-0}"
  fi
  if [[ "$arg" == "port" ]]; then
    service_index=$((i + 1))
    target_port_index=$((i + 2))
    service="${!service_index}"
    target_port="${!target_port_index}"
    case "$service:$target_port" in
      frontend:3000)
        printf '%s\n' "${FAKE_FRONTEND_PORT_MAPPING:-127.0.0.1:43000}"
        ;;
      backend:8000)
        printf '%s\n' "${FAKE_BACKEND_PORT_MAPPING:-127.0.0.1:48000}"
        ;;
      database:5432)
        printf '%s\n' "${FAKE_DATABASE_PORT_MAPPING:-127.0.0.1:45432}"
        ;;
      *)
        exit 1
        ;;
    esac
    exit 0
  fi
done
exit "${FAKE_COMPOSE_UP_STATUS:-0}"
""",
    )
    _write(
        bin_dir,
        "python",
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${FAKE_PYTHON_ARGS_LOG}"
exit 0
""",
    )
    for name in ("docker", "python"):
        (bin_dir / name).chmod(0o755)

    env = os.environ.copy()
    for key in (
        "SMOKE_FRONTEND_PORT",
        "SMOKE_BACKEND_PORT",
        "SMOKE_DATABASE_PORT",
    ):
        env.pop(key, None)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_COMPOSE_UP_STATUS"] = str(compose_up_status)
    env["FAKE_COMPOSE_DOWN_STATUS"] = str(compose_down_status)
    env["FAKE_COMPOSE_ENV_LOG"] = str(tmp_path / "compose-env.log")
    env["FAKE_PYTHON_ARGS_LOG"] = str(tmp_path / "python-args.log")
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", "scripts/smoke_public_fresh_start.sh"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_public_compose_only_exposes_core_services() -> None:
    assert _compose_service_names() == {"frontend", "backend", "database"}


def test_public_compose_publishes_core_services_on_loopback_only() -> None:
    verifier = _load_script("verify_public_release.py")
    violations = verifier.collect_violations(
        REPO_ROOT,
        tracked_files=[],
        compose_config=verifier.compose_config(REPO_ROOT),
    )
    loopback_violations = [
        violation
        for violation in violations
        if violation.startswith("service ") and "instead of 127.0.0.1" in violation
    ]

    assert loopback_violations == []


def test_public_env_example_omits_agent_target_keys() -> None:
    assert not {key for key in _env_example_keys() if key.startswith("AGENT_")}


def test_public_env_example_documents_intentional_compose_binding_changes() -> None:
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    loopback_note = "# Docker Compose publishes frontend, backend, and database ports on 127.0.0.1 only."
    override_note = "# Change the Compose bindings deliberately if a different network posture is required."
    loopback_note_index = lines.index(loopback_note)

    assert lines[loopback_note_index : loopback_note_index + 2] == [loopback_note, override_note]


def test_public_fresh_start_smoke_script_covers_clean_vault_lifecycle() -> None:
    script_path = REPO_ROOT / "scripts" / "smoke_public_fresh_start.sh"

    assert script_path.exists()
    assert script_path.stat().st_mode & stat.S_IXUSR

    script = _smoke_script_source()
    for expected in (
        "docker compose -p",
        "PROJECT_NAME",
        "SMOKE_FRONTEND_PORT",
        "SMOKE_BACKEND_PORT",
        "SMOKE_DATABASE_PORT",
        "frontend backend database",
        "127.0.0.1",
        "/health",
        "down -v",
        "GET /api/llm-vault",
        "GET /api/llm-profiles",
        "POST /api/llm-vault/create",
        '"exists"',
        '"unlocked"',
        '"items"',
        "SMOKE_VAULT_PASSPHRASE",
    ):
        assert expected in script


def test_optional_scanner_wrapper_checks_all_public_shell_scripts() -> None:
    script = _optional_scanner_script_source()

    assert "shellcheck scripts/run_optional_scanners.sh scripts/smoke_public_fresh_start.sh" in script


def test_optional_scanner_wrapper_reports_bandit_findings_without_failing(tmp_path: Path) -> None:
    result = _run_optional_scanners_with_fake_tools(
        tmp_path,
        env_overrides={"FAKE_bandit_STATUS": "1"},
    )

    assert result.returncode == 0
    assert "ADVISORY: bandit exited 1" in result.stdout
    assert "failed=0" in result.stdout


def test_public_fresh_start_smoke_script_uses_hermetic_compose_values() -> None:
    script = _smoke_script_source()

    for expected in (
        "export APP_ENV=development",
        "export POSTGRES_DB=jailbreaker",
        "export POSTGRES_USER=jailbreaker",
        "export POSTGRES_PASSWORD=jailbreaker",
        "export DATABASE_URL=postgresql://jailbreaker:jailbreaker@database:5432/jailbreaker",
        "export API_BASE_URL=http://backend:8000",
        'export FRONTEND_PORT="${SMOKE_FRONTEND_PORT:-0}"',
        'export BACKEND_PORT="${SMOKE_BACKEND_PORT:-0}"',
        'export DATABASE_PORT="${SMOKE_DATABASE_PORT:-0}"',
        "SMOKE_FRONTEND_PORT",
        "SMOKE_BACKEND_PORT",
        "SMOKE_DATABASE_PORT",
        "export NEXT_PUBLIC_API_BASE_URL=",
        "export API_CORS_ORIGINS=",
    ):
        assert expected in script

    assert "SMOKE_PORT_OFFSET" not in script
    assert "import socket" not in script


def test_public_fresh_start_smoke_script_defaults_to_ephemeral_ports_and_honors_overrides(tmp_path: Path) -> None:
    default_result = _run_smoke_with_fake_compose(tmp_path / "default")

    assert default_result.returncode == 0
    assert (tmp_path / "default" / "compose-env.log").read_text(encoding="utf-8").splitlines() == [
        "FRONTEND_PORT=0",
        "BACKEND_PORT=0",
        "DATABASE_PORT=0",
    ]

    override_result = _run_smoke_with_fake_compose(
        tmp_path / "override",
        env_overrides={
            "SMOKE_FRONTEND_PORT": "49122",
            "SMOKE_BACKEND_PORT": "49123",
            "SMOKE_DATABASE_PORT": "49124",
        },
    )

    assert override_result.returncode == 0
    assert (tmp_path / "override" / "compose-env.log").read_text(encoding="utf-8").splitlines() == [
        "FRONTEND_PORT=49122",
        "BACKEND_PORT=49123",
        "DATABASE_PORT=49124",
    ]


def test_public_fresh_start_smoke_script_resolves_compose_port_mappings(tmp_path: Path) -> None:
    result = _run_smoke_with_fake_compose(
        tmp_path,
        env_overrides={
            "FAKE_FRONTEND_PORT_MAPPING": "127.0.0.1:43111",
            "FAKE_BACKEND_PORT_MAPPING": "127.0.0.1:48111",
            "FAKE_DATABASE_PORT_MAPPING": "127.0.0.1:45543",
        },
    )

    assert result.returncode == 0
    python_args = (tmp_path / "python-args.log").read_text(encoding="utf-8").splitlines()
    assert "- http://127.0.0.1:48111/health" in python_args
    assert "- http://127.0.0.1:43111/" in python_args
    assert "- http://127.0.0.1:48111 http://127.0.0.1:43111 local-smoke-passphrase" in python_args


def test_public_fresh_start_smoke_script_fails_on_malformed_compose_port_mapping(tmp_path: Path) -> None:
    result = _run_smoke_with_fake_compose(
        tmp_path,
        env_overrides={"FAKE_BACKEND_PORT_MAPPING": "not-a-port-mapping"},
    )

    assert result.returncode != 0
    assert "Malformed Compose port mapping for backend:8000" in result.stderr


def test_public_fresh_start_smoke_script_checks_frontend_and_reports_cleanup_failures() -> None:
    script = _smoke_script_source()

    for expected in (
        'FRONTEND_BASE_URL="http://127.0.0.1:${resolved_frontend_port}"',
        'BACKEND_BASE_URL="http://127.0.0.1:${resolved_backend_port}"',
        'wait_for_http_200 "frontend" "$FRONTEND_BASE_URL/"',
        "# GET / via frontend",
        "# GET /api/llm-vault via frontend proxy",
        "proxied_vault == vault",
        "# POST /api/llm-vault/create via frontend proxy",
        'request_json(frontend_base_url, "POST", "/api/llm-vault/create"',
        "# GET /api/llm-vault after frontend proxy create",
        'compose port "$service" "$target_port"',
        "resolve_host_port frontend 3000",
        "resolve_host_port backend 8000",
        "Warning: failed to tear down smoke Compose project",
    ):
        assert expected in script

    assert 'request_json(backend_base_url, "POST", "/api/llm-vault/create"' not in script
    assert ">/dev/null 2>&1 || true" not in script


def test_public_fresh_start_smoke_script_fails_if_cleanup_fails_after_success(tmp_path: Path) -> None:
    result = _run_smoke_with_fake_compose(tmp_path, compose_down_status=19)

    assert result.returncode == 19
    assert "Warning: failed to tear down smoke Compose project" in result.stderr


def test_public_fresh_start_smoke_script_preserves_primary_failure_if_cleanup_also_fails(tmp_path: Path) -> None:
    result = _run_smoke_with_fake_compose(tmp_path, compose_up_status=7, compose_down_status=19)

    assert result.returncode == 7
    assert "Warning: failed to tear down smoke Compose project" in result.stderr


def test_generic_local_target_urls_remain_container_reachable() -> None:
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.api.store import EvaluationStore

    store = object.__new__(EvaluationStore)

    assert store._container_reachable_url("http://localhost:1234/v1") == "http://host.docker.internal:1234/v1"
    assert store._container_reachable_url("https://127.0.0.1:8443/v1") == "https://host.docker.internal:8443/v1"
    assert store._container_reachable_url("https://api.openai.com/v1") == "https://api.openai.com/v1"


def test_public_verifier_accepts_external_denylist_files(tmp_path: Path) -> None:
    verifier = _load_script("verify_public_release.py")
    _write(tmp_path, "README.md", "# Public repo\n")
    _write(tmp_path, "internal/notes/private.md", "private-marker details\n")
    _write(tmp_path, "backend/app/example.py", 'MARKER = "private-marker"\n')
    _write(tmp_path, "deny-terms.txt", "private-marker\n")
    _write(tmp_path, "deny-paths.txt", "internal/notes/\n")
    tracked = [
        "README.md",
        "internal/notes/private.md",
        "backend/app/example.py",
    ]

    violations = verifier.collect_violations(
        tmp_path,
        tracked_files=tracked,
        compose_config={"services": {}},
        deny_terms=verifier.read_list_file(tmp_path / "deny-terms.txt"),
        deny_paths=verifier.read_list_file(tmp_path / "deny-paths.txt"),
    )

    assert "denied path: internal/notes/private.md" in violations
    assert "denied term 'private-marker' in internal/notes/private.md" in violations
    assert "denied term 'private-marker' in backend/app/example.py" in violations


def test_public_verifier_requires_loopback_compose_bindings(tmp_path: Path) -> None:
    verifier = _load_script("verify_public_release.py")
    compose_config = {
        "services": {
            "frontend": {"ports": [{"host_ip": "0.0.0.0", "published": "3000", "target": 3000}]},
            "backend": {"ports": [{"host_ip": "127.0.0.1", "published": "8000", "target": 8000}]},
            "database": {"ports": [{"host_ip": "127.0.0.1", "published": "5432", "target": 5432}]},
        }
    }

    violations = verifier.collect_violations(
        tmp_path,
        tracked_files=[],
        compose_config=compose_config,
    )

    assert "service frontend publishes 3000 on 0.0.0.0 instead of 127.0.0.1" in violations


def test_public_verifier_requires_public_root_files(tmp_path: Path) -> None:
    verifier = _load_script("verify_public_release.py")
    violations = verifier.collect_violations(
        tmp_path,
        tracked_files=[],
        compose_config={"services": {}},
    )

    assert "missing required file: README.md" in violations
    assert "missing required file: LICENSE.md" in violations
    assert "missing required file: SECURITY.md" in violations


def test_public_verifier_rejects_tracked_env_files_without_external_denylist(tmp_path: Path) -> None:
    verifier = _load_script("verify_public_release.py")
    _write(tmp_path, ".env.local", "TARGET_API_KEY=secret\n")

    violations = verifier.collect_violations(
        tmp_path,
        tracked_files=[".env.local"],
        compose_config={"services": {}},
    )

    assert "denied path: .env.local" in violations


def test_public_verifier_rejects_nested_env_files_but_allows_examples(tmp_path: Path) -> None:
    verifier = _load_script("verify_public_release.py")

    violations = verifier.collect_violations(
        tmp_path,
        tracked_files=[
            "backend/.env",
            "backend/.env.local",
            "frontend/.env.production",
            "frontend/.env.example",
        ],
        compose_config={"services": {}},
    )

    assert "denied path: backend/.env" in violations
    assert "denied path: backend/.env.local" in violations
    assert "denied path: frontend/.env.production" in violations
    assert "denied path: frontend/.env.example" not in violations


def test_public_verifier_requires_public_files_to_be_tracked(tmp_path: Path) -> None:
    verifier = _load_script("verify_public_release.py")
    for required in (
        *verifier.REQUIRED_ROOT_FILES,
        *verifier.REQUIRED_DOC_FILES,
        *verifier.REQUIRED_TEMPLATE_FILES,
    ):
        _write(tmp_path, required, "placeholder\n")

    violations = verifier.collect_violations(
        tmp_path,
        tracked_files=[],
        compose_config={"services": {}},
    )

    assert "missing required file: README.md" in violations
    assert "missing required file: docs/index.md" in violations
    assert "missing required file: .github/pull_request_template.md" in violations


def test_public_verifier_scans_extensionless_text_files_but_skips_binary_files(tmp_path: Path) -> None:
    verifier = _load_script("verify_public_release.py")
    _write(tmp_path, "backend/Dockerfile", "RUN echo private-marker\n")
    _write(tmp_path, "LICENSE", "private-marker terms\n")
    binary_path = tmp_path / "assets" / "logo"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00private-marker")

    violations = verifier.collect_violations(
        tmp_path,
        tracked_files=["backend/Dockerfile", "LICENSE", "assets/logo"],
        compose_config={"services": {}},
        deny_terms=("private-marker",),
    )

    assert "denied term 'private-marker' in backend/Dockerfile" in violations
    assert "denied term 'private-marker' in LICENSE" in violations
    assert "denied term 'private-marker' in assets/logo" not in violations


def test_public_verifier_scans_unknown_suffix_text_files_but_skips_binary_files(tmp_path: Path) -> None:
    verifier = _load_script("verify_public_release.py")
    _write(tmp_path, "backend/settings.ini", "private-marker=true\n")
    _write(tmp_path, "assets/logo.svg", "<svg>private-marker</svg>\n")
    binary_path = tmp_path / "config.cfg"
    binary_path.write_bytes(b"\x00private-marker")

    violations = verifier.collect_violations(
        tmp_path,
        tracked_files=["backend/settings.ini", "assets/logo.svg", "config.cfg"],
        compose_config={"services": {}},
        deny_terms=("private-marker",),
    )

    assert "denied term 'private-marker' in backend/settings.ini" in violations
    assert "denied term 'private-marker' in assets/logo.svg" in violations
    assert "denied term 'private-marker' in config.cfg" not in violations


def test_public_fixture_namespace_is_exampleco_only() -> None:
    def joined(*parts: str) -> str:
        return "".join(parts)

    test_source = (REPO_ROOT / "backend/tests/test_public_release_tooling.py").read_text(encoding="utf-8")
    runtime_fixture_paths = (
        "backend/app/attacklib/code_agent_attacks.py",
        "backend/app/attacklib/customer_service_attacks.py",
        "backend/app/api/conversation.py",
        "backend/app/api/store.py",
        "backend/app/scoring/detectors.py",
        "database/init/005_seed_prompts.sql",
        "frontend/public/app.js",
    )
    runtime_fixtures = "\n".join(
        (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in runtime_fixture_paths
    )

    for expected_token in (
        "ExampleCo",
        "ExampleCo Electronics",
        "ExampleCo Insights",
        "exampleco-insights-agent",
        "exampleco-support-agent",
        "EXAMPLECO-AUTO-1842",
        "EXAMPLECO-ROLLFORWARD",
        "exampleco_client",
        "1-800-555-0100",
    ):
        assert expected_token in runtime_fixtures

    for retired_token in (
        joined("A", "cme"),
        joined("AC", "ME"),
        joined("a", "cme"),
        joined("a", "cme", "_"),
        joined("a", "cme", "-"),
        joined("a", "cme", ".com"),
        joined("analytics", ".internal", ".example"),
        joined("analytics", "-", "agent"),
        joined("agentic", "-", "analytics"),
        joined("customer", "-", "success", "-", "agent"),
        joined("1-800-", "AC", "ME", "-HELP"),
    ):
        assert retired_token not in runtime_fixtures
        assert retired_token not in test_source


def test_public_docs_and_community_surface_exists() -> None:
    expected = [
        "README.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "mkdocs.yml",
        "docs/index.md",
        "docs/getting-started.md",
        "docs/configuration.md",
        "docs/architecture.md",
        "docs/security-model.md",
        "docs/local-development.md",
        "docs/release-process.md",
        "docs/changelog.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/pull_request_template.md",
    ]
    for relative_path in expected:
        assert (REPO_ROOT / relative_path).exists(), relative_path


def test_public_python_dependencies_use_one_committed_uv_lockfile() -> None:
    lockfile = REPO_ROOT / "uv.lock"
    ci = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "backend/Dockerfile").read_text(encoding="utf-8")
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert lockfile.exists()
    assert "uv sync --locked --extra dev --extra docs --extra postgres" in ci
    assert "uv pip install" not in ci
    assert "COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /uvx /bin/" in dockerfile
    assert "uv sync --locked --no-install-project --extra postgres" in dockerfile
    assert "pip install" not in dockerfile
    assert "UV_PROJECT_ENVIRONMENT=/opt/venv" in dockerfile
    assert "context: ." in compose
    assert "dockerfile: backend/Dockerfile" in compose


def test_public_repo_omits_internal_release_automation() -> None:
    for relative_path in (
        "scripts/export_public_release.py",
        "scripts/bootstrap_github_repo.sh",
    ):
        assert not (REPO_ROOT / relative_path).exists(), relative_path


def test_public_readme_is_brief_and_links_to_docs() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert len(readme.splitlines()) < 180
    assert "docs/getting-started.md" in readme
    assert "docs/security-model.md" in readme
    assert "Contributing" in readme
    assert "Supporters" in readme


def test_public_license_and_security_policy_are_public_ready() -> None:
    license_text = (REPO_ROOT / "LICENSE.md").read_text(encoding="utf-8")
    security = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "BSD 3-Clause License" in license_text
    assert "Copyright (c) 2026, Neeraj Gupta" in license_text
    assert "GitHub private vulnerability reporting" in security
    assert "0.1.x" in security
    assert "private issue tracker" not in security
