# 🚀 NeuroScope — Deployment Runbook

> Two supported paths: **Streamlit Community Cloud** (free public demo, 5 minutes) and
> **Docker** (portable self-hosting). Both serve the same app; pick per audience.
> Prereq: the code lives in a GitHub repo (see [IMPLEMENTATION_CHRONICLE.md](IMPLEMENTATION_CHRONICLE.md) Phase 6).

---

## Path A — Streamlit Community Cloud (recommended demo)

Free, zero-config for Streamlit apps, and it redeploys on every push to the tracked
branch.

### A1. Prerequisites
- The repo on GitHub (e.g., `SidSadhu/NeuroScope`), with the `main` branch containing
  `app.py`, `requirements.txt`, and `pages/`.
- (Optional) A free Groq key from [console.groq.com](https://console.groq.com/) for
  live AI insights — the app runs in mock mode without it.

### A2. Deploy (one-time, ~5 minutes)
1. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app** → paste the repo (`SidSadhu/NeuroScope`).
3. Settings:
   - **Main file path:** `app.py`
   - **Branch:** `main`
   - **Python version:** 3.11 (matches the Dockerfile and CI matrix)
4. Before first deploy, open **Settings → Secrets** and paste (toml):
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
   Secrets are encrypted at rest and injected as env vars at runtime — never committed.
5. **Deploy.** First build installs `requirements.txt` (~3–5 min); subsequent pushes
   to `main` rebuild automatically.

### A3. Post-deploy checklist
- [ ] App loads at `https://<app-name>.streamlit.app` with no error banner
- [ ] Overview page renders donut + correlation heatmap
- [ ] Churn page: **Train All Models** completes; all four tabs populate; the SHAP tab renders a ranking (not the fallback ones-vector)
- [ ] Segmentation page: **Run All Algorithms** completes; personas render; CSV download works
- [ ] AI Insights page: key detected (sidebar ✅) and streaming works (or mock mode if intentionally keyless)
- [ ] Boot/deploy logs show no deprecation or exception spam

### A4. Ops notes
- Apps sleep after ~12 days of inactivity (community tier) — a visit wakes them.
- Resource limit ~1 GB RAM: fine for this workload (training peaks well below that).
- Re deploying the PR branch first: Streamlit Cloud can track any branch — point a
  second "staging" app at the feature branch if you want to preview pre-merge.

---

## Path B — Docker (portable / self-hosted)

### B1. Compose (simplest)
```bash
export GROQ_API_KEY="gsk_your_key_here"   # optional; mock mode without it
docker compose up --build -d
curl -f http://localhost:8501/_stcore/health   # → "ok"
```
Open `http://localhost:8501`.

### B2. Manual build
```bash
docker build -t neuroscope:latest .
docker run -d -p 8501:8501 --name neuroscope \
  -e GROQ_API_KEY="${GROQ_API_KEY:-}" \
  neuroscope:latest
```

### B3. What the container guarantees
- `python:3.11-slim` base; `requirements.txt` installed before source copy (layer-cache
  friendly — dependency changes rebuild the small layer only).
- Built-in `HEALTHCHECK` hitting `/_stcore/health` every 30s; compose mirrors it.
- Server binds `0.0.0.0:8501`, headless — safe behind a reverse proxy.

### B4. Cloud targets (same image)
| Target | Command sketch |
|---|---|
| **Google Cloud Run** | `gcloud run deploy neuroscope --image <img> --port 8501 --set-env-vars GROQ_API_KEY=...` |
| **Fly.io** | `fly launch --image <img>` then set the secret via `fly secrets set GROQ_API_KEY=...` |
| **AWS ECS/Fargate** | Push to ECR; task def: container port 8501, env `GROQ_API_KEY`, health check `/_stcore/health` |

### B5. Verify before sharing the URL
1. `curl -f http://<host>:8501/_stcore/health` → `ok`
2. Full manual pass from Path A's checklist (A3)
3. Logs clean: `docker logs neuroscope` shows no deprecation warnings

---

## Secrets policy (both paths)

| Secret | Where it lives | Never |
|---|---|---|
| `GROQ_API_KEY` | Streamlit Cloud Secrets / container env / local `.streamlit/secrets.toml` | committed, logged, echoed in screenshots |

`.gitignore` already excludes `.streamlit/secrets.toml`; the repo ships only
`.streamlit/secrets.toml.example`. If a key ever leaks, revoke at console.groq.com.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cloud build fails on `shap` install | Keep Python 3.11 and `shap>=0.45` (prebuilt wheels exist); avoid pinning an old shap |
| App sleeps / cold start | Community tier sleeps after ~12 idle days; any visit (or redeploy) wakes it |
| AI page shows "sample output" | No key resolved: check Cloud Secrets (or env var) spelling: `GROQ_API_KEY` |
| Port conflicts locally | `streamlit run app.py --server.port 8502` |
| `healthcheck` fails in compose | Wait past `start_period`; check `docker logs neuroscope` for a boot error |
| Container OOM on big CSV | Limit uploaded CSV size; Cloud Run: bump memory to 1–2 GiB |
