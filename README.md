# Jailbreaker

[![Sponsored by SpecterOps](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fspecterops%2F.github%2Fmain%2Fconfig%2Fshield.json)](https://github.com/specterops)

Jailbreaker is a local offline evaluation harness for testing chatbot and agent systems against jailbreak, prompt-injection, and related safety failures.

![Jailbreaker dashboard](assets/Jailbreaker-screenshot.png)

## Blog

Launch write-up: [Jailbreaker: LLM Jailbreak Testing You Can Actually Repeat](https://specterops.io/blog/2026/06/29/llm-jailbreak-testing-with-jailbreaker/)

## Quick Start

```bash
cp .env.example .env
docker compose up -d --build frontend backend database
```

Open `http://localhost:3000`.

Read the full setup guide in [docs/getting-started.md](docs/getting-started.md) and the security model in [docs/security-model.md](docs/security-model.md).

## Documentation

- [Getting started](docs/getting-started.md)
- [Configuration](docs/configuration.md)
- [Architecture](docs/architecture.md)
- [Security model](docs/security-model.md)
- [Local development](docs/local-development.md)
- [Release process](docs/release-process.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for issue, pull request, and development guidance.

## Contributions

Contributors are acknowledged in release notes and pull requests.

## Supporters

This project is currently maintained by Neeraj Gupta.
