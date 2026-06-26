# Jailbreaker

Jailbreaker is a local offline evaluation harness for testing chatbot and agent systems against jailbreak, prompt-injection, and related safety failures.

![Jailbreaker dashboard](assets/Jailbreaker-screenshot.png)

## Blog

Launch write-up: [Blog post coming soon](BLOG_POST_URL)

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
