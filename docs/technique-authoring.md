# Source-Controlled Technique Package Authoring

This guide describes the package API implemented in `backend/app/techniques`.
It applies to reviewed, source-controlled built-ins. The dashboard's custom
technique builder remains a separate legacy-compatible feature; authenticated
workspace package upload and general third-party package authoring are future
work and are not claimed here.

## Package layout

A package is a directory discovered by the built-in package loader. The
manifest and its implementation entry point are required; all referenced files
must remain under the package root.

```text
technique-id/
├── technique.yaml                 # required manifest
├── workflow.yaml                  # required by a workflow entrypoint
├── executor.py                    # required by a Python entrypoint
├── schemas/
│   └── parameters.json            # optional JSON Schema 2020-12 parameters
├── prompts/                       # optional declared template assets
├── seeds/                         # optional declared packaged seed assets
├── presentation.yaml              # optional presentation data
├── metrics.yaml                   # optional metric data
└── tests/
    └── scenarios.yaml             # deterministic parity scenarios
```

<!-- package-layout-entries:start -->
| Contract path | Use |
|---|---|
| `technique.yaml` | Required manifest. |
| `workflow.yaml` | Declarative workflow entrypoint. |
| `executor.py` | Python executor entrypoint. |
| `schemas/parameters.json` | Parameter JSON Schema. |
| `presentation.yaml` | Optional presentation asset. |
| `metrics.yaml` | Optional metrics asset. |
| `tests/scenarios.yaml` | Deterministic scenario asset. |
<!-- package-layout-entries:end -->

`technique.yaml` is loaded with safe YAML, validated with JSON Schema, checked
for references and supported capabilities, and hashed byte-for-byte. Declared
assets are also hashed. Absolute paths, traversal outside the root, symlinks,
missing assets, duplicate identities, and unsupported entry points are rejected
before registration.

## Manifest field reference

The four required top-level fields are `api_version`, `contract_version`,
`technique`, and `implementation`. Unknown top-level or nested schema fields
are rejected unless they live in an explicitly declared `extensions` mapping.

<!-- manifest-required-fields:start -->
| Required field | Reason |
|---|---|
| `api_version` | Selects the package API. |
| `contract_version` | Pins execution contracts. |
| `technique` | Declares the technique identity. |
| `implementation` | Declares the executable route. |
<!-- manifest-required-fields:end -->

<!-- manifest-top-level-fields:start -->
| Field | Required | Runtime meaning |
|---|---:|---|
| `api_version` | yes | Non-empty package API version. |
| `contract_version` | yes | Non-empty execution contract version pinned in each run spec. |
| `technique` | yes | Identity, description, capabilities, aliases, tags, and target surfaces. |
| `implementation` | yes | Implementation ID, kind, entry point, and implementation extensions. |
| `parameters` | no | Parameter-schema path and named preset mappings. |
| `inputs` | no | Typed runtime inputs and optional media/count/size/schema constraints. |
| `model_roles` | no | Logical model roles required by the package. |
| `histories` | no | Named histories, visibility, and message/token bounds. |
| `judges` | no | Judge definitions, schedules, purposes, evidence, scales, and policies. |
| `detectors` | no | Detector IDs, inputs, output schemas, and policies. |
| `seed` | no | Seed strategy, source, requirement, and options. |
| `tools` | no | Tool IDs with input/output schemas and policy. |
| `converters` | no | Converter IDs using the same declared tool-shaped contract. |
| `result_policy` | no | Official response/judge selection, threshold, auxiliaries, and stopping. |
| `presentation` | no | Generic conversation format and package presentation metadata. |
| `metrics` | no | Declarative metric records consumed by projection/reporting. |
| `recommended_limits` | no | Package defaults for bounded runtime resources. |
| `assets` | no | Unique package-relative files included in provenance hashing. |
| `state_schema_version` | no | Positive checkpoint-state schema version. |
| `result_schema_version` | no | Positive result schema version. |
| `source` | no | Source attribution metadata. |
| `extensions` | no | Namespaced package extensions; `provider_free_demo` has the bounded generic behavior documented below. |
<!-- manifest-top-level-fields:end -->

Nested fields are exact:

- `technique` requires `id`, `version`, `name`, `family`, `summary`, and
  `capabilities`; it may add `tags`, `aliases`, and `target_surfaces`.
- `implementation` requires `id`, `kind`, and `entrypoint`; it may add
  `extensions`.
- `parameters` accepts only `schema` and `presets`.
- Each `input` requires `id` and `kind`; optional fields are `required`,
  `media_types`, `max_count`, `max_size_bytes`, and inline `schema`.
- Each `model_role` requires `id`; optional fields are `required`,
  `description`, `capabilities`, and `options`.
- Each `history` requires `id`; optional fields are `visible_to`,
  `max_messages`, `max_tokens`, and `extensions`.
- Each `judge` requires `id`; optional fields are `schedule`, `purposes`,
  `role`, `required`, `evidence`, `threshold`, `scale_min`, `scale_max`, and
  `policy`.
- Each `detector` requires `id`; optional fields are `inputs`,
  `output_schema`, and `policy`.
- `seed` accepts `strategy`, `source`, `required`, and `options`.
- Each tool/converter requires `id`; optional fields are `input_schema`,
  `output_schema`, and `policy`.
- `result_policy` accepts `official_judge`, `final_response`,
  `success_threshold`, `auxiliary_judges`, and `stopping`.

### Provider-free demo evidence

A package may declare `extensions.provider_free_demo` to own the recorded
target evidence used by application `demo` mode:

```yaml
extensions:
  provider_free_demo:
    response: A bounded recorded target response.
    tool_calls:
      - 'workspace.write: src/task.py'
```

This extension never executes a tool or calls a provider. It supplies only the
recorded response and optional tool-call evidence to the generic provider-free
target adapter. `response` must be non-empty text of at most 20,000 characters.
`tool_calls` must be an array of at most 32 non-empty strings, each at most
2,000 characters. Unsupported fields and malformed or oversized values fail
before target execution. Packages that omit the extension retain the existing
compatibility fallback while their legacy path is preserved.

All IDs and referenced files must be non-empty. Technique IDs match
`^[a-z0-9][a-z0-9_-]*$`. String arrays are unique. Counts and schema versions
are positive. A result-policy judge reference must name a declared judge;
judge roles and history visibility must name declared logical roles.

## Runtime enum reference

<!-- manifest-enum-values:start -->
| Enum member | Meaning |
|---|---|
| `ImplementationKind.legacy` | Registered compatibility implementation. |
| `ImplementationKind.workflow` | Declarative bounded workflow. |
| `ImplementationKind.python` | Package-local async executor. |
| `TechniqueCapability.single_shot` | One-shot execution. |
| `TechniqueCapability.iterative` | Bounded multi-step execution. |
| `TechniqueCapability.multiple_histories` | More than one named history. |
| `TechniqueCapability.multiple_judges` | More than one judge definition. |
| `TechniqueCapability.population_search` | Candidate population search. |
| `TechniqueCapability.parallel_branches` | Bounded parallel branch execution. |
| `TechniqueCapability.backtracking` | Explicit bounded backtracking. |
| `TechniqueCapability.response_transformation` | Response transformation. |
| `TechniqueCapability.structured_output` | Structured model output. |
| `TechniqueCapability.documents` | Document inputs. |
| `TechniqueCapability.workspace_files` | Workspace-file inputs. |
| `TechniqueCapability.tools` | Tool execution. |
| `TechniqueCapability.multimodal` | Multimodal inputs. |
| `TechniqueCapability.memory` | Memory service use. |
| `TechniqueCapability.converters` | Converter service use. |
| `JudgeSchedule.after_target` | Evaluate after target evidence. |
| `JudgeSchedule.every_round` | Evaluate every round. |
| `JudgeSchedule.final_round` | Evaluate the final round. |
| `JudgeSchedule.run_end` | Evaluate at run completion. |
| `JudgeSchedule.on_demand` | Evaluate only when requested. |
| `JudgePurpose.official_score` | Authoritative efficacy score. |
| `JudgePurpose.success_control` | Success decision. |
| `JudgePurpose.stopping_control` | Stop decision. |
| `JudgePurpose.backtracking_control` | Backtracking decision. |
| `JudgePurpose.candidate_ranking` | Candidate ordering. |
| `JudgePurpose.diagnostics` | Non-authoritative diagnostic evidence. |
| `SeedStrategy.none` | No seed. |
| `SeedStrategy.supplied` | Run-supplied seed. |
| `SeedStrategy.packaged_asset` | Source-controlled package asset. |
| `SeedStrategy.sql_library` | Legacy SQL seed library. |
| `SeedStrategy.runtime` | Runtime-computed seed. |
<!-- manifest-enum-values:end -->

Application-backed `sql_library` packages may use `seed.options` for
`fallback_asset`, `prefer_non_default`, `placeholder_uses_fallback`, and
`strip_trailing_newline`. `placeholder_uses_fallback: true` makes an explicit
trimmed `sanitized-placeholder` select the package fallback without querying
the SQL library. When a legacy template consumes additional named placeholders
before rendering, the package declares their text replacements under
`template_fields`; `intent` is reserved for runtime rendering and cannot be
replaced there. Explicit and database seed values are trimmed, declared
template fields are replaced, and all other literal braces are preserved
rather than interpreted as workflow format fields.

## Parameters, presets, and validation

Parameter schemas are JSON Schema Draft 2020-12 documents referenced by
`parameters.schema`. The loader validates the schema itself and validates every
preset against it. At experiment planning time the resolver deep-merges, from
lowest to highest precedence:

1. defaults declared in the parameter schema;
2. the selected named preset;
3. explicit request parameters.

The merged value is validated again, rejected if it resembles credential
material, and frozen in `ResolvedRunSpec.resolved_parameters`. The spec also
pins the manifest hash, asset hashes, implementation identity, logical role
bindings, judge policies, and effective limits. Retries reuse that behavioral
spec rather than resolving the current default again.

## Logical model roles and bindings

Packages declare logical roles such as `target`, `attacker`, `fabricator`, or
`target_judge`; they never embed provider credentials. Experiment planning
maps each required logical role to a profile-backed `ModelRoleBinding`, which
pins profile ID, provider, model, transport, and options. Preflight aggregates
all missing role/client/capability/asset/tool/converter/surface diagnostics
before provider work.

Every model request owns a generated `model_call_id`, role, operation ID,
operation attempt, and the full correlation tuple. Target requests analogously
own a `target_call_id`. Provider responses complete the same persisted record;
they are not matched by queue position.

## Histories and provenance

Create only manifest-declared histories. Each append records a message ID,
history ID, branch, role, source kind, and optional source event. Snapshots are
immutable ordered message sets with their own IDs. Pass the exact snapshot ID
to a target call so persisted response evidence can be traced back to its
input. Forked branches have isolated history heads; never share a mutable list
between cells or branches.

## Judges and result semantics

`target_judge` is the conventional authoritative judge role, but authority is
declared by `result_policy.official_judge`, not inferred from a technique ID.
A judge's schedule and purposes control when and why it runs. Evidence and
provenance should include the exact `target_response_event_id` and candidate or
round identity.

A required judge failure emits `judge.failed`, has `score: null`, and makes the
attempt inconclusive/retryable. A successful literal score `0` emits
`judge.completed`, remains tested numeric zero, and is not a parse failure.
Never coerce missing/failed scores to zero. Official outcome projection uses
the exact selected response event and the official judge event linked to it.

## Workflow implementations

A workflow entrypoint is package-relative YAML parsed into a bounded graph.
Registered node types are:

`artifact_create`, `backtrack`, `branch`, `candidate_select`, `checkpoint`,
`convert`, `detector`, `finish`, `history_append`, `history_create`,
`history_snapshot`, `input_materialize`, `judge`, `loop`, `model_call`,
`parallel`, `population_map`, `render`, `retry`, `seed_resolve`, `target_call`,
`tool_call`, and `transform`.

Declarative `transform` steps support `identity`, `prefix`, `suffix`,
`replace`, `base64`, and `reverse`. The `base64` transform encodes the source
string as UTF-8 bytes and emits standard Base64 ASCII; `reverse` uses Python
string-slice order, reversing Unicode code points rather than grapheme
clusters. Render steps may set
`strip_trailing_newline: true` when a text asset's conventional terminal
newline is not part of the provider-visible prompt.

`target_call` steps may declare `call_options` inside the step's nested
`options` mapping. Those values are captured on the canonical target request
and forwarded by the OpenAI-compatible adapter. Provider-specific sampling
settings belong under `call_options.extra_body`; deterministic fixtures must
assert the exact option mapping without contacting a provider.

The parser rejects unsupported types, dangling edges, malformed conditions,
unbounded graphs, excessive YAML/condition size, and illegal replay. `retry`
may target only explicitly replay-safe service nodes, requires one or more
failure categories, is capped at 100 and by `max_workflow_retry_attempts`, and
does not replay indeterminate work unless the pinned limit explicitly permits
it. `parallel`, `population_map`, loops, and visits are bounded by validated
limits; arbitrary expressions, Python, SQL, and frontend JavaScript are not a
workflow API.

## Python executor protocol

A Python entrypoint resolves to an object satisfying this protocol:

```python
class TechniqueExecutor(Protocol):
    async def execute(
        self,
        request: TechniqueRequest,
        context: ExecutionContext,
    ) -> TechniqueOutcome: ...
```

Use only controlled context services for models, targets, judges, histories,
events, templates, seeds, tools, converters, artifacts, checkpoints, budgets,
and cancellation. Emit normalized evidence with stable operation ownership.
Return references to emitted evidence in `TechniqueOutcome`; do not construct
application/database/frontend objects or call providers directly.

Use Python only for bounded algorithms not expressible as the declarative
graph—population evolution or parallel tree search are current examples.

## Assets, templates, and seeds

Declare every consumed package file in `assets`. Render templates through the
template service using explicit fields. Packaged seeds are immutable source
assets. Supplied seeds are run inputs. SQL-library and runtime seeds are
legacy-coexistence strategies and must be called out in migration design.
Generated artifacts pass through the artifact service and budget accounting.

## Presentation and metrics

Set `presentation.conversation_format` (or `format`) to a generic adapter:

- `single_shot` preserves prompt, response, and judge fields;
- `rounds` produces ordered `kind="round"` entries;
- `generation` produces stable candidate identity/origin/parent/prompt/
  response/judge/selection fields;
- `tree` produces ordered depth entries and branch identity, parent, origin,
  prompt, response, judge, retain/prune/select state;
- `diff`, `judge`, `artifact`, `table`, and unknown declarations use safe
  raw evidence where no richer application adapter exists.

The captured manifest-selected projection is authoritative. Application and
frontend code must not identify a package to choose a format. Metrics are
declarative mappings over normalized evidence; efficacy uses only the selected
completed tested attempt, while operational metrics include failed/retried
attempts.

## Budgets, retries, cancellation, and failure policy

Declare bounded `recommended_limits`; planning pins effective limits. Reserve
capacity before parallel work and use child reservations for branches. Check
cancellation before external work and at bounded iteration points. Emit
`budget.exhausted` or `run.cancelled` rather than turning either into a score.

Use the normalized failure categories `provider`, `timeout`, `judge_parse`,
`validation`, `cancelled`, `budget`, `worker_lost`, `persistence`, and
`internal`. A retry creates a new immutable attempt with parent/retry metadata;
old attempts, events, and calls stay intact. Only the selected completed
attempt contributes efficacy. Rejudge reads stored target evidence, creates
new evaluation evidence, makes no target call, and does not replace the
selected execution attempt.

## Deterministic scenarios

`tests/scenarios.yaml` has exactly `version: 1` and named `scenarios`. Each
scenario declares `intent`, optional validated `parameters`, ordered
`attempts`, and `selected_attempt_number`. Each attempt has a `retry_kind` and
role-keyed recorded `responses`; a role's responses are either an ordered
string sequence or a mapping from exact operation ID to string response.
Packages using `SeedStrategy.sql_library` also declare `sql_seed_libraries` as
a mapping from the manifest's exact seed source to a non-empty ordered string
sequence. The runner injects those local values into both arms; it never reads
the database or calls a provider during deterministic parity.
Optional fixture data must remain deterministic and provider-free. Concurrent
fixtures use operation mappings—never a shared FIFO queue. The deterministic
runner derives separate stable ordered queues only for an actual sequential
legacy arm.

## Local validation and parity

Comparison tooling lives in `backend/devtools/technique_parity/`, separate from
the application runtime, and is excluded from application images.

From the repository root, run focused package and contract tests first:

```bash
cd backend
uv run --locked --extra dev python -m pytest tests/test_technique_manifest.py tests/test_technique_runtime.py -v
uv run --locked --extra dev python -m pytest tests/test_modular_<package>.py -v
```

Generate deterministic parity with the package's checked-in scenario. Reports
are local generated output and are not committed. Check that each report has
`passed: true` and no differences; compare repeated runs when checking
determinism. For example, from `backend/`:

```bash
uv run --locked --extra dev python -m devtools.technique_parity --technique direct_prompt_injection --legacy-implementation legacy --modular-implementation modular --mode deterministic --output /tmp/jailbreaker-direct-prompt-injection-parity.json
uv run --locked --extra dev python -m devtools.technique_parity --technique crescendo_liar --legacy-implementation legacy --modular-implementation modular --mode deterministic --output /tmp/jailbreaker-crescendo-liar-parity.json
uv run --locked --extra dev python -m devtools.technique_parity --technique autodan --legacy-implementation legacy --modular-implementation modular --mode deterministic --random-seed 1234 --output /tmp/jailbreaker-autodan-parity.json
uv run --locked --extra dev python -m devtools.technique_parity --technique tap --legacy-implementation legacy --modular-implementation modular --mode deterministic --output /tmp/jailbreaker-tap-parity.json
```

The deterministic runner loads each package's checked-in scenario. Live parity
requires separate explicit authorization and pinned policy/profile/dataset
metadata; deterministic success is never reported as live evidence.

Before promotion also run Ruff, compileall, the full no-database backend,
frontend Node tests/checks, public-release verification, fresh isolated
PostgreSQL, existing-volume migration, and the disposable fresh-stack smoke
script described by the repository release documentation.

## Promotion, coexistence, and rollback

Promotion is a catalog default-selection change after deterministic parity and
review. Keep both the modular package and registered `legacy` implementation.
Experiments pin the selected technique version, implementation ID, spec/hash,
roles, parameters, and limits. Explicit `implementation_id: legacy` is the
application rollback proof.

Rollback is data-only default selection: remove/disable the modular default
override and retain historical packages, specs, attempts, calls, events, and
compatibility columns. Do not delete evidence or remove the legacy registry,
loops, renderer, aliases, or custom-technique builder during migration.
