# AQEM Web UI

A FastAPI + React (Vite/TypeScript) console for the Agentic QEM Lab, in the
spirit of the NVIDIA Quantum-Calibration-Agent-Blueprint UI. It drives a run,
streams **live per-node DAG progress** over Server-Sent Events, and renders the
probe histograms, ZNE extrapolation, accuracy-vs-shots comparison, and the
Policy audit trail.

```
ui/
  src/
    App.tsx       run form + live DAG progress + results
    Chart.tsx     Plotly wrapper (dark theme)
    api.ts        SSE client for /api/run
    types.ts      shared types + the 8 node ids
  vite.config.ts  dev server proxies /api -> :8000
```

The backend lives in `src/aqem/web/server.py` (`aqem-web` entry point).

## Develop

Two terminals — backend, then frontend with hot reload:

```bash
# 1. backend (FastAPI on :8000)
pip install -e ".[web]"
aqem-web

# 2. frontend (Vite on :3099, proxies /api -> :8000)
cd ui
npm install
npm run dev
# open http://localhost:3099
```

## Production (single process)

Build the static frontend; the backend serves it from `ui/dist` at `/`:

```bash
cd ui && npm run build && cd ..
aqem-web          # http://localhost:8000 serves UI + API
```

## What you see

- **Run panel** — qubits, target accuracy, device (noise model), seed, VLM toggle
  (managed Bedrock Claude), and a compare-vs-baseline toggle.
- **DAG progress** — the eight stages light up live as the loop runs; retries
  re-run the invalidated nodes; decisions stream into a log.
- **Results** — estimate vs ideal, shots used, techniques chosen; the efficiency
  comparison vs the blind full-stack baseline (shot ratio, gain badge); the
  Plotly charts; and the full Policy audit table.
