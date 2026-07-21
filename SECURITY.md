# Security Policy

## Supported Versions

| Version | Supported |
| --- | --- |
| 0.2.x | Yes |
| 0.1.x | No |

## Reporting A Vulnerability

Use GitHub private vulnerability reporting for suspected vulnerabilities. Do not open a public issue for unpatched security findings, secrets, private target URLs, or prompt/response transcripts that may contain sensitive content.

## What To Expect

The maintainer will acknowledge the report through the private advisory, validate the issue against supported versions, coordinate a fix when needed, and publish a disclosure after a remediation path is available.

## Security Model

Jailbreaker is designed for local offline use. The backend has no authentication layer and should not be exposed beyond the local machine without additional controls. See [docs/security-model.md](docs/security-model.md) for the full operating model.
