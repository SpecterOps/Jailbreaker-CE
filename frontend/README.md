# Frontend

The frontend is a dependency-free static operator console served by
`server.mjs`. It uses vanilla JavaScript, static HTML, and one shared stylesheet.

## Files

```text
frontend/
|-- public/
|   |-- index.html       # Dashboard
|   |-- app.js           # Dashboard state, technique runs, custom techniques
|   |-- experiments.html # Experiment builder and results workbench
|   |-- experiments.js
|   |-- analysis.html    # SQL/reporting views
|   |-- analysis.js
|   |-- components/
|   |   `-- sidebar-nav.js
|   `-- styles.css
|-- server.mjs           # Static server and /api proxy
`-- tests/               # Node built-in test runner checks
```

## Run

Use the Compose stack from the repository root:

```bash
docker compose up -d --build frontend backend database
```

The frontend container serves on port `3000` inside Docker. Override the host
port with `FRONTEND_PORT`.

The frontend server proxies `/api/*` to `API_BASE_URL`, defaulting to
`http://backend:8000` inside Compose.

## Test

From the repository root:

```bash
node --test frontend/tests/*.test.mjs
node --check frontend/server.mjs
node --check frontend/public/app.js
node --check frontend/public/analysis.js
node --check frontend/public/experiments.js
```

The tests are lightweight layout and regression checks. They do not replace a
browser-based smoke test for full live runs.

## Maintenance Notes

- The UI stores selected profile IDs and non-secret target settings in
  localStorage. Raw API keys are either one-off session form values or encrypted
  backend vault entries.
- The scripts prefer DOM construction through shared helpers, `replaceChildren()`,
  and `textContent` for dynamic content.
- `app.js` and `experiments.js` are large page controllers. Split by workflow
  before adding another major page state machine.
