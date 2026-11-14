# Architecture

## Components

- `frontend/`: static operator console
- `backend/app/api/`: HTTP API and evaluation orchestration
- `backend/app/techniques/contracts/`: immutable definitions, requests,
  outcomes, events, and executor protocol
- `backend/app/techniques/plugins/`: strict package manifest loading and
  provenance hashing
- `backend/app/techniques/core/`: catalog, selector/router dependencies, model,
  target, judge, history, seed, template, budget, cancellation, and persistence
  services
- `backend/app/techniques/workflow/`: bounded declarative workflow parser and
  executor
- `backend/app/techniques/builtin/`: reviewed source-controlled workflow and
  Python packages
- `backend/app/techniques/projection/`: generic result and manifest-selected
  conversation views
- `backend/app/attacklib/`: built-in technique registry and prompt builders
- `backend/app/profile_vault.py`: encrypted LLM profile storage
- `database/init/`: fresh database schema and seed prompts

## Request Flow

For a modular run, strict contracts feed a discovered catalog registration.
The selector/router chooses the requested pinned implementation; experiment
planning resolves parameter defaults/presets/overrides, logical model roles,
judge policy, limits, and content hashes. Preflight validates the whole route.
A bounded workflow or package-local Python executor can then use only controlled
core services. Normalized event/call evidence is projected by result policy and
manifest presentation before the application and experiment layers serialize
or persist it.

```text
contracts -> package loader/preflight -> catalog/selector/router
          -> pinned spec -> controlled core services
          -> workflow | Python executor
          -> events/calls -> outcome/conversation/parity projection
          -> application/experiments -> persistence/frontend
```

Legacy runs follow the retained attack registry and runner behind clearly
labeled compatibility boundaries. Both routes produce established public
fields. Explicit `implementation_id: legacy` is the per-technique rollback;
changing the default never deletes immutable historical evidence.

## Persistence

PostgreSQL stores runs, experiments, pinned technique configurations, cells,
immutable attempts/retries, execution events, model calls, checkpoints,
artifacts, analysis views, custom techniques, and encrypted profile records.
New judge results must contain an integer score from 1 through 10. Invalid or
failed required judges remain null-scored/inconclusive. Historical successful
zero scores remain readable as tested results. Rejudge adds evaluation evidence
without another target call or replacing the selected execution attempt.

### Recovery of incomplete model-call evidence

Model-call records are saved before provider dispatch and updated when the call
returns. Parallel child events stay buffered until the root journal commits
them as one ordered batch. A process crash between those writes can leave a
saved call without its lifecycle events. A merge into another child buffer
does not make those events durable.

Recovery restarts only the interrupted case within the same experiment, using
a new attempt and fresh histories. Default retry selection excludes completed,
tested cases. For example, if cases 1–39 are complete and case 40 crashes,
retrying failed cases runs only case 40. Its partial attempt and call evidence
remain in history as inconclusive; the new valid result becomes selected when
it completes. Completed cases keep their results, attempt IDs, and scores.
Existing execution ownership and retry budgets still apply.

A fresh restart may repeat a call already made inside the interrupted case.
That is the recovery choice when its saved lifecycle is incomplete, including
a call still marked `requested`. It does not require a new experiment revision
or the `retry_indeterminate` flag. Uncertain provider outcomes with complete
lifecycle evidence retain the existing pinned `retry_indeterminate` policy.

Checkpoint resume is different: it would reuse partial state. Before allowing
that path, the store checks every attempt in the case lineage. Each terminal
call requires a persisted `model.requested` event followed by the matching
`model.completed` or `model.failed` event, with exact call ownership, operation,
operation-attempt, role, and sequence evidence. Missing events block checkpoint
resume and direct the user to restart that case. They are not reconstructed,
and saved response text alone cannot turn an interrupted case into a tested
result. An existing in-process journal can still retry its identical atomic
merge after a transient persistence failure.

## Technique Registry

Built-in techniques are registered in `backend/app/attacklib/registry.py`; custom techniques are stored through the API.

Source-controlled technique packages live in `backend/app/techniques/builtin/`;
`backend/app/techniques/defaults.yaml` selects their default versions and
implementations. The static registry, loops, renderer, and custom-technique
builder remain for coexistence and rollback. Authenticated workspace package
upload and general third-party package installation are not implemented.

See [Authoring Complex Technique Architectures](technique-authoring.md)
(`docs/technique-authoring.md`) for package fields, enums, runtime services,
workflow/Python protocols, validation, parity, promotion, and rollback.

## Provider and package limitations

Deterministic parity uses recorded responses to compare legacy and modular
behavior. Passing those scenarios does not establish live-provider reliability
or jailbreak success. A provider can return truncated JSON or a custom judge
verdict whose criteria, verdict, and score contradict each other. Invalid
results remain unscored; if every AutoDAN candidate is unscored, the search
stops with `all_candidates_unscored`.

Package versions identify immutable manifests and assets. Changes to a recorded
package require a new version and a corresponding default-selection update;
editing an existing version can prevent activation against a database that
already holds that version. Preserve archived registrations and historical
evidence when upgrading or rolling back.
