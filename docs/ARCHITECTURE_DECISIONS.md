# 🏗️ Architecture Decision Records — NeuroScope

> The *why* behind every significant choice, in lightweight ADR format.
> **Audience:** senior engineers who want to know what was considered, rejected, and why.
> Status legend: ✅ accepted · 🔁 superseded (noted inline)

---

## ADR-001 — Modular core (`src/`) decoupled from Streamlit UI

**Status:** ✅ accepted

**Context:** Streamlit encourages script-style code where models, charts, and UI are
one file. That's fast for demos and hostile to testing, reuse, and review.

**Decision:** All business logic lives in importable `src/` packages (`data`, `models`,
`ai`, `visualization`). Pages are thin orchestration layers. Only `loader.py` imports
Streamlit (for `@st.cache_data`).

**Alternatives considered:**
- Single-file app — fastest to write, untestable, not reviewable. Rejected.
- Full FastAPI backend + separate frontend — correct for a real product, massive
  overkill for a self-serve analytics tool and this repo's scope. Rejected.

**Consequences:** pytest runs the whole ML core with no browser; charts and models are
reusable outside Streamlit; slightly more boilerplate per feature.

---

## ADR-002 — Model registry + identical splits for the classifier arena

**Status:** ✅ accepted

**Context:** Comparing models fairly requires identical data windows; adding a model
should be a one-line change, not a refactor.

**Decision:** `CLASSIFIER_REGISTRY` dict (name → estimator). `train_all_models()` fits
every entry on one stratified 80/20 split (`random_state=42`) and returns uniform
`ModelResult` objects. Model selection = max AUC-ROC.

**Alternatives considered:**
- Cross-validation per model — statistically stronger, ~5× slower for an interactive
  button; the UI is the product here. Rejected for v1; noted as a natural extension.
- AutoML (e.g., FLAML/optuna) — impressive but opaque; the pedagogical point is the
  transparent comparison. Rejected.

**Consequences:** Adding a model touches exactly one dict (plus a color + a test);
leaderboard comparisons are apples-to-apples; repeated runs are deterministic.

---

## ADR-003 — Version-tolerant SHAP layer with graceful fallbacks

**Status:** ✅ accepted (superseded the naive direct-call implementation after shap 0.49 broke it)

**Context:** SHAP changed its output contract (legacy list → 2D → 3D per-class arrays
as sklearn ≥1.4 + shap ≥0.45 landed). Direct calls crashed at DataFrame construction.
Any real deployment hits this within months.

**Decision:** Central `_select_positive_class()` normalizer accepts all three layouts
and returns `(n_samples, n_features)` + scalar base value. Every SHAP call is wrapped
with fallbacks: `feature_importances_` → `|coef_|` → ones vector. The UI degrades,
never crashes.

**Alternatives considered:**
- Pin shap to an old version — hides the problem, breaks with sklearn updates. Rejected.
- Write our own explainer — weeks of work, worse than the real thing. Rejected.

**Consequences:** Works across shap 0.45–0.49+ and sklearn 1.4–1.7; unit tests
explicitly cover all three layouts; the fallback path is itself tested.

---

## ADR-004 — Streaming generators as the LLM contract

**Status:** ✅ accepted

**Context:** LLM responses take seconds; a blocking call would freeze the UI, and a
network failure would break the whole page.

**Decision:** `stream_*_insights()` functions are **generators yielding string
chunks**. The UI consumes them with `st.write_stream()`. No API key or API error →
the generator yields a deterministic, clearly-labeled mock narrative.

**Alternatives considered:**
- One-shot `st.write(full_text)` — no progressive rendering. Rejected.
- LangChain agent stack — abstraction the use case doesn't need; one vendor call is
  30 lines. Rejected (revisit if multi-tool agents are added).

**Consequences:** Token-by-token UX; the AI page works 100% offline; swapping LLM
vendors touches one file.

---

## ADR-005 — 4-layer test strategy with a hard coverage gate

**Status:** ✅ accepted

**Context:** ML apps usually ship "tested" with a couple of metric smoke checks and
then break in the UI, where no test looks.

**Decision:** Four layers: (1) data unit tests, (2) ML unit tests including SHAP's
three output layouts and all fallback branches, (3) chart-builder tests, (4) **AppTest
end-to-end tests that boot every page and click the train/run buttons**, asserting
session state. `pyproject.toml` enforces `fail_under = 70` coverage (actual: 92%).

**Alternatives considered:**
- Playwright/browser tests — heaviest possible setup for a Streamlit app; AppTest covers
  the script-level behavior natively. Rejected for now.
- No coverage gate — gates are how "temporarily skipped tests" become permanent. Rejected.

**Consequences:** The E2E layer has already caught two real bugs (a Plotly
`update_layout` keyword conflict and a deprecated Streamlit API) that unit tests
missed; test suite takes ~2 minutes, which keeps CI fast.

---

## ADR-006 — Synthetic data generator as the default dataset

**Status:** ✅ accepted

**Context:** A demo platform needs data that is (a) always available, (b) realistic
enough that models learn non-trivial patterns, (c) seed-stable for reproducibility.

**Decision:** `generator.py` produces a seeded 1,500-row dataset where churn
probability responds to satisfaction, support calls, tenure, and spend. Uploads are
first-class: `load_csv()` degrades gracefully (warnings, classification optional).

**Alternatives considered:**
- Bundling a public dataset (e.g., Telco churn) — licensing and provenance questions,
  less control over feature semantics. Rejected.
- No default data (upload-only) — kills the zero-friction first-run experience. Rejected.

**Consequences:** `pytest` is deterministic; demo metrics are reproducible forever;
documented honestly in the Model Card as synthetic (metrics not real-world claims).

---

## ADR-007 — One CSS token file for the whole design system

**Status:** ✅ accepted

**Context:** Rebranding a Streamlit app usually means hunting inline styles across
every page.

**Decision:** All colors/radii live as CSS custom properties in `app.py`
(`--bg`, `--cyan`, `--purple`…) with named component classes (`stat-card`,
`persona-card`, `winner-banner`, `page-head`, `section-title`). Pages reference classes;
they never hardcode palettes. `.streamlit/config.toml` mirrors base colors for native
widgets. `prefers-reduced-motion` disables animations.

**Alternatives considered:**
- Tailwind/CSS framework — build pipeline for marginal benefit here. Rejected.
- Per-page styles — that's the problem being fixed. Rejected.

**Consequences:** Rebranding = editing one `:root` block; consistency is enforced by
convention; minor risk that Plotly chart colors (defined in `charts.py`) can drift
from CSS tokens — kept in sync manually.

---

## ADR-008 — Session state as the inter-page contract

**Status:** ✅ accepted

**Context:** Streamlit reruns scripts constantly; cross-page data (metrics → AI page,
clusters → personas) needs explicit handoff.

**Decision:** A documented set of session-state keys (table in `PLAYBOOK.md` §7) is
the official API between pages. Training results are stored once on button click and
read thereafter — never recomputed on rerun.

**Alternatives considered:**
- `st.cache_data` on training functions — caches by input hash but doesn't model the
  "user chose to train" intent, and pollutes the only cache-allowed module. Rejected.
- Query params — wrong tool for large objects. Rejected.

**Consequences:** Reruns are cheap; pages stay independent; the key table is normative
— undocumented keys are a code smell.

---

## ADR-009 — Modern Streamlit widget API

**Status:** ✅ accepted (supersedes `use_container_width` usage from v2.0)

**Context:** Streamlit deprecated `use_container_width` (removal announced for after
2025-12-31); current version 1.55 supports `width="stretch"`.

**Decision:** All layout-affecting widget calls use `width="stretch"`. Multi-page
navigation uses `st.navigation`/`st.Page` (the modern API) rather than the legacy
`pages/` auto-discovery — the directory name is convention, not mechanism.

**Consequences:** No deprecation warnings in logs (boot log is verified clean);
forward-compatible with Streamlit's removal timeline.
