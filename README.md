<p align="center">
  <img src="assets/Jailbreaker_Vertical_AllWhite.png" alt="Jailbreaker" width="300">
</p>

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

## Run Prompts and Evaluation Criteria

Target, Attacker, and Judge system prompts are independent run configuration and
are not stored in reusable LLM vault profiles. Target prompts are used directly;
Attacker and Judge prompts are prepended to Jailbreaker's required technique and
scoring instructions.

Judge evaluation criteria can use the existing standard behavior, explicit
success and failure outcomes, or a complete custom rubric. Custom criteria keep
the platform's `1-10` score contract, with `7` as the minimum successful score,
and are shared by Dashboard runs, Experiment Design, and experiment retries.

Experiment datasets may optionally include a `gold_answer` on individual prompt
objects. When present, Jailbreaker supplies it to the Judge as a trusted factual
reference; prompts without one retain the standard Judge behavior. Gold answers
are never added to Target or Attacker messages. Because `gold_answer` is treated
as trusted Judge configuration, import experiment datasets only from sources you
trust and do not place instructions or secrets in that field.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for issue, pull request, and development guidance.

## Contributions

Contributors are acknowledged in release notes and pull requests.

## Supporters

This project is currently maintained by Neeraj Gupta.
