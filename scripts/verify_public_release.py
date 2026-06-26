from __future__ import annotations

import argparse
import codecs
import json
import subprocess
from pathlib import Path, PurePosixPath


REQUIRED_ROOT_FILES = (
    "README.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "mkdocs.yml",
)
REQUIRED_DOC_FILES = (
    "docs/index.md",
    "docs/getting-started.md",
    "docs/configuration.md",
    "docs/architecture.md",
    "docs/security-model.md",
    "docs/local-development.md",
    "docs/release-process.md",
    "docs/changelog.md",
)
REQUIRED_TEMPLATE_FILES = (
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/pull_request_template.md",
)
DEFAULT_DENIED_EXACT_PATHS = (".env",)
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yml",
    ".yaml",
}


def _run(
    command: list[str],
    *,
    cwd: Path,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=capture_output,
        text=True,
    )


def tracked_files(repo_root: Path) -> list[str]:
    result = _run(["git", "ls-files", "-z"], cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return [item for item in result.stdout.split("\0") if item]


def compose_config(repo_root: Path) -> dict:
    result = _run(
        ["docker", "compose", "-f", "docker-compose.yml", "config", "--format", "json"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "docker compose config failed")
    return json.loads(result.stdout)


def read_list_file(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def required_commands(repo_root: Path) -> list[list[str]]:
    frontend_tests = [
        str(path.relative_to(repo_root))
        for path in sorted((repo_root / "frontend" / "tests").glob("*.test.mjs"))
    ]
    return [
        ["python", "-m", "pytest", "backend/tests"],
        ["python", "-m", "compileall", "-q", "backend/app"],
        ["mkdocs", "build", "--strict"],
        ["node", "--test", *frontend_tests],
        ["node", "--check", "frontend/server.mjs"],
        ["node", "--check", "frontend/public/app.js"],
        ["node", "--check", "frontend/public/analysis.js"],
        ["node", "--check", "frontend/public/experiments.js"],
        ["docker", "compose", "-f", "docker-compose.yml", "config", "--quiet"],
        ["bash", "scripts/run_optional_scanners.sh"],
    ]


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", ".env.example"}:
        return True
    return _is_utf8_text_file(path)


def _is_utf8_text_file(path: Path) -> bool:
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8192):
                if b"\0" in chunk:
                    return False
                decoder.decode(chunk)
        decoder.decode(b"", final=True)
    except (OSError, UnicodeDecodeError):
        return False
    return True


def _matches_denied_path(relative_path: str, denied_path: str) -> bool:
    return relative_path == denied_path or (
        denied_path.endswith("/") and relative_path.startswith(denied_path)
    )


def _is_default_denied_path(relative_path: str) -> bool:
    name = PurePosixPath(relative_path).name
    return relative_path in DEFAULT_DENIED_EXACT_PATHS or name == ".env" or (
        name.startswith(".env.") and name != ".env.example"
    )


def collect_violations(
    repo_root: Path,
    *,
    tracked_files: list[str] | None = None,
    compose_config: dict | None = None,
    deny_terms: tuple[str, ...] = (),
    deny_paths: tuple[str, ...] = (),
) -> list[str]:
    tracked = tracked_files if tracked_files is not None else globals()["tracked_files"](repo_root)
    tracked_set = set(tracked)
    resolved_compose = compose_config if compose_config is not None else globals()["compose_config"](repo_root)
    violations: list[str] = []

    for required in (*REQUIRED_ROOT_FILES, *REQUIRED_DOC_FILES, *REQUIRED_TEMPLATE_FILES):
        if required not in tracked_set or not (repo_root / required).exists():
            violations.append(f"missing required file: {required}")

    for relative_path in tracked:
        if _is_default_denied_path(relative_path):
            violations.append(f"denied path: {relative_path}")
        if any(_matches_denied_path(relative_path, denied_path) for denied_path in deny_paths):
            violations.append(f"denied path: {relative_path}")
        path = repo_root / relative_path
        if not path.exists() or not _is_text_file(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        folded_text = text.casefold()
        for term in deny_terms:
            if term.casefold() in folded_text:
                violations.append(f"denied term '{term}' in {relative_path}")

    for service_name in ("frontend", "backend", "database"):
        service = resolved_compose.get("services", {}).get(service_name, {})
        for port in service.get("ports", []):
            host_ip = str(port.get("host_ip", ""))
            published = str(port.get("published", ""))
            if host_ip != "127.0.0.1":
                violations.append(
                    f"service {service_name} publishes {published} on {host_ip or '<empty>'} instead of 127.0.0.1"
                )
    return violations


def run_required_commands(repo_root: Path) -> list[str]:
    failures: list[str] = []
    for command in required_commands(repo_root):
        shell_command = " ".join(command)
        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            failures.append(f"command failed ({result.returncode}): {shell_command}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--denylist-file", type=Path)
    parser.add_argument("--deny-path-file", type=Path)
    parser.add_argument("--skip-command-checks", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()

    violations = collect_violations(
        repo_root,
        deny_terms=read_list_file(args.denylist_file),
        deny_paths=read_list_file(args.deny_path_file),
    )
    if not args.skip_command_checks:
        violations.extend(run_required_commands(repo_root))
    if violations:
        for violation in violations:
            print(f"FAIL: {violation}")
        return 1
    print("Public release verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
