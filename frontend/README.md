# FedPDA-IDS — Web Interface

Presentation, monitoring and demonstration layer for the FedPDA-IDS research system.

This is a **read-mostly** surface over the existing research codebase. It does not train models,
change any completed experiment's numbers, or replace the Prometheus/Grafana stack.

```
React (Vite + TS + Tailwind)   :5173
        │  HTTP/JSON
        ▼
FastAPI  api/                  :8001
        │  direct imports, read-only
        ▼
src/fedpda_ids/  ·  experiments/results/  ·  experiments/checkpoints/

        ╰─ untouched: monitoring/ → Prometheus :9090 · Grafana :3000 (linked, not replaced)
```

The API uses **8001** because the project's Prometheus exporter already owns 8000 during training runs.

---

## Running it

### 1. Backend

From the repository root, with the project venv active:

```bash
pip install -r requirements.txt          # includes fastapi, uvicorn
uvicorn api.main:app --reload --port 8001
```

Check it: <http://localhost:8001/api/health> · interactive docs at <http://localhost:8001/api/docs>

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                              # http://localhost:5173
```

Configuration lives in `frontend/.env` (copy from `.env.example`):

```
VITE_API_BASE=http://localhost:8001/api
VITE_GRAFANA_URL=http://localhost:3000
```

### 3. Grafana (optional, unchanged)

```bash
cd monitoring && docker compose up -d    # Prometheus :9090, Grafana :3000 (admin/admin)
```

The sidebar's **Open Monitoring** link and the Settings screen both point at it. Grafana remains the
technical observability layer; this app does not duplicate its low-level metrics.

---

## Screens

| Screen | Status | What it shows |
|---|---|---|
| **Dashboard** | ✅ built | KPI tiles, per-round charts, privacy–utility curve, comms cost, recent events |
| **Research Results** | ✅ built | E1–E6 + T7 explorer: tables, charts, interpretation, per-scope switching |
| **Settings** | ✅ built | Connection state, active scope, Grafana/Prometheus/API-docs links |
| Intrusion Detection | stage 2 | Prepared-sample browser + CSV upload → real inference |
| Federated Learning | stage 2 | Run selector, round timeline, per-client panel |
| Privacy & Security | stage 2 | DP config, SecAgg+, privacy–utility, attack evaluation |
| Drift Monitor | stage 2 | State machine, ADWIN/PSI, sandboxed retrain trigger |

Every stage-2 screen's **backend is already built and tested** — only the interface is pending. The
placeholder pages list the endpoints you can exercise now via `/api/docs`.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Inventory + connection indicator |
| GET | `/api/datasets` | The two federations and their scopes |
| GET | `/api/dashboard` | KPIs, chart series, events |
| GET | `/api/results` | E1–E6/T7 index |
| GET | `/api/results/{id}` | One experiment's table + interpretation |
| GET | `/api/fl/runs` | Completed FL runs for a scope |
| GET | `/api/fl/runs/{run_name}` | Round history + per-client panel |
| GET | `/api/privacy` | DP config, SecAgg+, attack evaluation |
| GET | `/api/privacy/utility` | ε vs macro-F1 / rare-class recall |
| GET | `/api/drift/status` | Detector state + recovery metrics |
| GET | `/api/drift/events` | Drift timeline |
| POST | `/api/drift/retrain` | **Real** retrain, sandboxed run name |
| GET | `/api/drift/retrain/{job_id}` | Poll a retrain job |
| GET | `/api/samples` | Prepared real samples (metadata only) |
| POST | `/api/predict` | Real inference on a prepared sample |
| POST | `/api/predict/upload` | Real inference on an uploaded CSV |
| GET | `/api/predictions` | Recent prediction history |

All query endpoints take `?dataset=cicids2017|nbaiot&scope=<scope>`.

---

## Research-integrity rules enforced in the code

These are not stylistic preferences — they are the reason several things look the way they do.

1. **The two datasets are never merged.** There is no "all datasets" option anywhere, and every
   screen labels which federation it is showing.
2. **Every metric carries provenance** — `stored` (completed experiment), `new run`, `derived`
   (computed from a saved confusion matrix/checkpoint), `live` (computed just now), `demo`.
3. **A missing metric is never a zero.** Unmeasured cells render as "not measured" with the reason.
   A measured `0.000` (for example rare-class recall under DP) is shown as a real zero, because it is.
4. **E1–E6/T7 numbers come only from `experiments/results/e1_e6_t7_tables.json`** — the same file the
   published results artifact is written from. The API never rounds or reshapes them.
5. **Retraining is real but sandboxed.** `POST /api/drift/retrain` runs the actual pipeline, always
   under a `*_uidemo_<timestamp>` run name, so an official checkpoint can never be overwritten.
6. **Inference is real.** Predictions run a genuine forward pass through a real checkpoint, and the
   sample's true label is returned alongside the prediction so a demo can be checked, not trusted.
7. **No tracebacks reach the user.** The API logs them server-side and returns a friendly message.

---

## Tests

```bash
pytest tests/api_layer -q     # 24 API contract tests
pytest -q                     # full suite: 241 tests
npm run build                 # typecheck + production bundle
```

The API tests run against the **real** results directory on purpose: the contract that matters is
"the API serves the same numbers the thesis tables do", which a mocked repository could not verify.

---

## Design system

Tokens are ported from the project's published results artifact (teal accent, Fraunces + IBM Plex
pairing, the finding/caveat/mechanism/resolved callout vocabulary) into `tailwind.config.ts` and
`src/styles/tokens.css`, with full light and dark themes.

A matching preview bundle for Claude Design lives in `design-system/` — regenerate with
`python design-system/build.py`, then push with the DesignSync tool (requires `/design-login` once).
