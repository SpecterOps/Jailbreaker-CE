# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- Optional per-datapoint `gold_answer` references for experiment Judges, including mixed datasets, retries, rejudging, and PostgreSQL persistence without exposing answers to Target or Attacker messages.

### Changed

- Ordered standard and configurable Judge outputs so evidence, component scores, and criteria decisions are generated before the final score.

### Fixed

- Rejected malformed or internally contradictory configurable Judge verdicts before they can affect experiment scores.
- Preserved scalar `gold_answer` values such as `0` and `false` instead of treating them as missing.

## [0.2.0] - 2026-07-20

### Added

- Independent run-scoped system prompts for Target, Attacker, and Judge roles without storing those prompts in reusable vault profiles.
- Versioned Judge evaluation policies supporting standard criteria, explicit success and failure outcomes, and complete custom rubrics.
- Shared Judge-policy controls across Dashboard runs, Experiment Design, and experiment retries.
- Exact effective Judge inputs and explicit Judge-error details in run and experiment traces.

### Changed

- Made the standard Judge introduction describe a general rubric evaluator instead of a genetic-algorithm fitness function.
- Gave session API keys precedence over saved profile keys while continuing to use saved profile endpoints and models.
- Kept the standard Judge's additional run prompt blank by default while preserving its built-in scoring rubric.

### Fixed

- Prevented Target or Attacker system prompts from leaking into Judge requests when those clients are reused for evaluation.
- Preserved failed Judge evaluations as retryable error results instead of silently presenting them as completed scores.
- Used experiment unsafe intents independently from technique seed prompts.

## [0.1.0] - 2026-06-24

### Added

- Initial public release of the local offline evaluation harness.
- Encrypted LLM profile vault, experiment workflows, and SQL-backed analysis views.
- Public documentation, community templates, and release verification tooling.
