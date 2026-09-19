# 📖 Implementation Chronicle — NeuroScope End-to-End

> The complete, honest build log of this project: every phase from empty folder to
> deployed app, every significant decision, every bug that taught something, and the
> verification evidence at each step.
> **Audience:** developers who want to understand how this project came to be, and
> reviewers who want evidence the engineering is real.
> Companion docs: [Playbook](PLAYBOOK.md) · [Model Card](MODEL_CARD.md) ·
> [ADRs](ARCHITECTURE_DECISIONS.md) · [One-Pager](ONE_PAGER.md)

---

## Phase 0 — The Idea & Scope

**Where it started.** The goal: a platform where a non-technical stakeholder uploads
(or loads) customer data and walks away with churn predictions they can *understand*,
segments they can *act on*, and strategies they can *defend*.

**Scope decisions made on day one:**
- **Interactive app over API service.** The user is an analyst at a dashboard, not
  another service. → Streamlit multi-page app, not FastAPI (see ADR-001).
- **Modular core regardless of UI choice.** Whatever the UI, models/charts/AI live in
  importable packages — because that's what makes testing possible (ADR-001).
- **Demo data must be honest demo data.** A seeded synthetic generator, clearly
  labeled — not a scraped dataset with murky provenance (ADR-006).

**Stack chosen:** Python 3.10–3.12 · scikit-learn · SHAP · Plotly · Streamlit ·
Groq (LLaMA-3.3-70B) · pytest · ruff · GitHub Actions · Docker.

---

## Phase 1 — Data Layer

**Built:** `src/data/generator.py` (seeded synthetic customers) and
`src/data/loader.py` (caching, CSV validation, X/y split).

**Decision — seeded reproducibility (ADR-006).** Every random operation takes
`random_state`. Consequence: `pytest` is deterministic, demo metrics are byte-identical
across machines, and the Model Card's numbers mean something.

**Decision — graceful degradation on upload.** `load_csv()` never hard-fails: missing
feature columns → warning + use what exists; missing `churned` → classification
disabled, clustering keeps working. The demo can't be killed by a bad CSV.

**Evidence:** `test_data.py` — schema, feature ranges, seeded reproducibility
(`assert_frame_equal` on two runs), seed-sensitivity.

---

## Phase 2 — ML Core

**Built:** classifiers arena, clustering suite, SHAP explainability.

**Decision — model registry + one shared split (ADR-002).** `CLASSIFIER_REGISTRY`
dict; every model trains on the identical stratified 80/20 split. Adding a model =
one dict entry. Rejected cross-validation per model as too slow for an interactive
button (documented as the natural v2 upgrade).

**Decision — AUC-ROC as the selection metric.** With ~31% churn, accuracy misleads;
AUC is threshold-free. Every `ModelResult` carries all five metrics anyway.

**Decision — scaling inside the Pipeline.** Logistic Regression gets `StandardScaler`
as an in-pipeline step, fit on train folds only. Leakage-safe by construction, and the
deployed artifact is one object.

**The bug that shaped the architecture (SHAP):** during verification, shap 0.49 +
sklearn 1.7 returned tree-explainer output as a 3D `(n_samples, n_features, n_classes)`
array; the code assumed the legacy `[class0, class1]` list → `ValueError:
Per-column arrays must each be 1-dimensional`. 3 tests failed.

**Fix (now ADR-003):** central `_select_positive_class()` normalizer accepting all
three known SHAP layouts (legacy list / 2D single-output / 3D per-class) + scalar base
value; every SHAP path wrapped with fallbacks (`feature_importances_` → `|coef_|` →
ones). The UI degrades; it never crashes.

**Evidence:** `test_models.py` — 39+ tests incl. all three SHAP layouts as synthetic
inputs, all three fallback branches, metrics-in-range and AUC>0.55 assertions.

---

## Phase 3 — Visualization & UI

**Built:** `src/visualization/charts.py` (Plotly dark-theme suite), four Streamlit
pages, the design system in `app.py`.

**Decision — one CSS token file (ADR-007).** All colors/radii are custom properties
in `app.py`; pages use named classes (`stat-card`, `persona-card`, `winner-banner`,
`page-head`, `section-title`). Rebranding = editing one `:root` block.

**Decision — session state as the inter-page API (ADR-008).** A documented key table
(`churn_results`, `seg_profiles`, …) is the only cross-page handoff mechanism. Training
happens once on button click; reruns read from state.

**Decision — streaming generators for AI (ADR-004).** `stream_*_insights()` yield
chunks consumed by `st.write_stream()`. No key → clearly-labeled offline mock. The AI
page works with zero network.

**UI upgrades in the polish phase:** churn donut with center KPI, feature correlation
heatmap, semi-circular risk gauge for single predictions, algorithm explorer on
Segmentation (view/export any algorithm, winner pre-selected), sidebar nav pills,
section-title accent bars, `prefers-reduced-motion` support.

**The bug the E2E layer caught:** the first donut implementation passed `legend=` to
`update_layout()` while `_layout()` already set it → `TypeError: got multiple values
for keyword argument 'legend'`. Unit tests passed; **the new AppTest E2E suite caught
it** on the Overview page. Same class of bug fixed in the gauge (`margin=`). Lesson,
now codified in the Playbook: `_layout()` owns `legend` and `margin` — merge overrides
into the dict.

**Second E2E catch:** Streamlit 1.55 logs `use_container_width` deprecation
(removal after 2025-12-31). Migrated all 24 occurrences to `width="stretch"`
(ADR-009). Boot log verified clean.

**Evidence:** `test_charts.py` (100% charts coverage), `test_app_smoke.py` (boots
every page, clicks Train/Run, asserts session state).

---

## Phase 4 — Testing & CI Hardening

**Built:** 4-layer suite, 54 tests, coverage gate.

**Decision — 4 layers, including UI E2E (ADR-005).** Data unit → ML unit (incl. all
SHAP layouts) → chart builders → **AppTest E2E**. Coverage gate `fail_under=70` in
`pyproject.toml`; actual 92%. Rejected browser-based Playwright as overkill for a
Streamlit app (AppTest covers script-level behavior).

**The migration:** coverage was 55% with `charts.py` at 0% and the explainability
module at 45%. Added chart tests + SHAP fallback tests to reach 92%.

**The de-dearring of mocks:** an early "unknown model" fallback test passed a Random
Forest pipeline (which has `feature_importances_`) and asserted the ones-vector branch
— it could never reach that branch. Rebuilt it around a `DummyClassifier` so the test
actually tests the branch.

**Evidence:** `ruff check .` + `ruff format --check .` clean (matching CI), `pytest`
87 passed, coverage 92.3%.

---

## Phase 5 — Documentation Layer

**Built:** five documents, each with a distinct audience:

| Doc | Audience | Content |
|---|---|---|
| `PLAYBOOK.md` | contributors | architecture, ML/AI internals, design system, testing, extension recipes, troubleshooting |
| `MODEL_CARD.md` | ML reviewers | data provenance, full metrics, calibration, limitations, ethics |
| `ARCHITECTURE_DECISIONS.md` | senior engineers | 9 ADRs with alternatives considered and rejected |
| `ONE_PAGER.md` | recruiters / reviewers | 60-second pitch with engineering highlights |
| `RESUME_BULLETS.md` | the author | Google XYZ / STAR / CAR narratives with reproducible numbers |

**Principle: real numbers only.** Every metric quoted anywhere (0.8045 AUC, 92.3%
coverage, silhouettes, SHAP rankings) was computed by running the code, and the
commands to reproduce are in the docs. **Principle: limitations stated out loud** —
recall ~0.42 at default threshold, uncalibrated probabilities, synthetic-data caveat.

---

## Phase 6 — Deployment

**Built:** deployment config + `DEPLOYMENT.md` runbook.

**Decision — two deployment targets.**
1. **Streamlit Community Cloud** — the right default for an interactive demo: free,
   zero-config for Streamlit apps, pushes from GitHub deploy automatically.
2. **Docker** (unchanged from earlier phases) — the portable/self-host answer for
   VPS/Cloud Run/ECS.

**Verified locally before writing the guide:** Streamlit 1.55 boots the app with a
zero-warning log (the same entrypoint the container uses); a `.dockerignore` was added
(excluding secrets, docs, tests, and VCS files from the image context);
`.streamlit/secrets.toml.example` was scrubbed to a placeholder after a real-looking
key was discovered in it; port/health checks documented. The Docker daemon is not
installed on the primary dev machine, so the image build is validated via the
Dockerfile's documented `docker build`/`docker-compose up` steps in
[DEPLOYMENT.md](DEPLOYMENT.md) — run them once before publishing the demo URL.

**Decision — secrets never in git.** `GROQ_API_KEY` goes to Streamlit Cloud secrets
or container env; `.streamlit/secrets.toml` stays git-ignored; only
`secrets.toml.example` ships.

---

## Phase 7 — Release Engineering

**Built:** git hygiene, the PR, and this chronicle.

**Process:**
1. Full verification pass (lint + format + 87 tests + clean boot) before any commit.
2. Feature branch (`feature/v2.1-platform-upgrade`), one clean commit, pushed to the fork.
3. PR opened **from `SiddharthSadhu:feature/v2.1-platform-upgrade` to
   `SidSadhu:main`** with the full change narrative.
4. Merge → Streamlit Cloud picks up `main` automatically → live demo.

---

## The Decision Log (one table to rule them all)

| # | Decision | Why | Rejected alternative |
|---|---|---|---|
| 1 | Streamlit app, modular core | interactive analyst tool; testability | single-file app; FastAPI+SPA |
| 2 | Model registry + shared split | one-line model additions; fair comparison | per-model CV (slow UI); AutoML (opaque) |
| 3 | AUC-ROC selection | threshold-free under 31% churn | accuracy (misleads) |
| 4 | Scaler inside pipeline | leakage-safe; single artifact | manual scaling outside |
| 5 | SHAP normalizer + fallbacks | survives shap/sklearn version drift | pin shap; own explainer |
| 6 | Streaming generator LLM contract | progressive UX; offline mock | blocking call; LangChain |
| 7 | 4-layer tests incl. AppTest E2E | UI bugs live where unit tests don't look | Playwright; no gate |
| 8 | Seeded synthetic data | deterministic CI; honest demo | bundled public dataset |
| 9 | One CSS token file | rebrand in one place; consistency | per-page styles; Tailwind |
| 10 | Session state as page API | cheap reruns; explicit handoffs | st.cache_data intent; query params |
| 11 | `width="stretch"` API | forward-compatible; clean logs | stay on deprecated flag |
| 12 | Streamlit Cloud + Docker deploys | free demo + portable self-host | single-target lock-in |

---

## Bug Museum (every bug, and what it taught)

1. **SHAP 3D arrays** (3 failing tests) → version-tolerance layer, now unit-tested
   across all layouts. *Lesson: dependencies change contracts; normalize at the boundary.*
2. **Coverage 55% → gate failing** → built chart + fallback tests; 92%. *Lesson: a
   coverage gate only helps if the untested modules are the risky ones.*
3. **Tautological test** (fallback ones-vector via RF pipeline) → rebuilt with
   DummyClassifier. *Lesson: a test that can't fail proves nothing.*
4. **Plotly `update_layout` conflict** (donut `legend=`, gauge `margin=`) → caught by
   AppTest E2E, not unit tests; fixed by merging into the layout dict. *Lesson: the
   E2E layer pays for itself; helpers own their keys.*
5. **Deprecated `use_container_width`** (24 call sites) → migrated to
   `width="stretch"`; boot log clean. *Lesson: deprecation warnings are debt with a
   deadline.*
6. **AppTest missing `download_button`** attribute → asserted via session state
   instead. *Lesson: test against the framework's actual API, not the imagined one.*
7. **Stale server on port 8501** → killed by PID via `netstat`. *Lesson: verify the
   process you started is the process serving.*

---

## Verification Trail (what was proven, and how)

| Claim | Proof |
|---|---|
| Tests pass | `pytest` → 87 passed (unit + charts + E2E smoke) |
| Coverage ≥ gate | `pytest --cov` → 92.31% vs 70% gate |
| Lint clean | `ruff check .` + `ruff format --check .` → all pass (CI parity) |
| App boots clean | `streamlit run` → HTTP 200, zero warnings in log |
| Pages work E2E | AppTest: all 4 pages boot, buttons click, session state populates |
| SHAP robust | unit tests pin all 3 output layouts + 3 fallback branches |
| Docker deployable | Dockerfile + compose validated step-by-step in DEPLOYMENT.md; local daemon unavailable — run the two documented commands to confirm on your machine |
| Metrics reproducible | seeded pipeline reproduces Model Card tables exactly |
| Docs accurate | every number cross-checked against a fresh run |

---

## Where It Goes Next (honest roadmap)

- Threshold tuning / calibration for recall-sensitive operational use
- Per-model cross-validation mode (flag-gated; UI already supports it)
- Drift monitoring + a real model registry if wired to production data
- Subgroup fairness evaluation once real (non-synthetic) data lands
- Multi-LLM support behind the existing generator contract

---

*This chronicle is maintained as part of the repo — new phases append, history
doesn't rewrite.*
