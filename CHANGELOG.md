# Changelog

## 2.1.0 — 2026-09-19

### Fixed
- SHAP integration for shap ≥ 0.45 / sklearn ≥ 1.4: added `_select_positive_class()` normalizer handling legacy list, 2D, and 3D per-class explainer outputs in both the importance and waterfall paths
- Plotly `update_layout()` keyword conflicts (`legend`, `margin`) in the new donut and gauge charts
- Removed conflicting `server.enableCORS` config (eliminates boot warning)

### Changed
- Migrated all widget layout calls from deprecated `use_container_width` to `width="stretch"` (Streamlit 1.55+)
- Reformatted codebase with ruff; lint clean (`ruff check .` + `ruff format --check .`)
- Removed dead code in `charts.py` flagged by ruff (F841)

### Added
- UI: churn donut + center KPI and feature correlation heatmap on the Overview page; semi-circular risk gauge on the single-prediction view; algorithm explorer on Segmentation (visualize/export any algorithm, winner pre-selected); sidebar nav pill styling, section-title accent bars, page-head component, hover micro-interactions, `prefers-reduced-motion` support
- Tests: 4-layer suite (54 tests, 92% coverage) — new `test_charts.py` (100% charts coverage) and `test_app_smoke.py` (AppTest E2E boots every page and clicks train/run)
- Docs: `docs/PLAYBOOK.md` (developer handbook), `docs/MODEL_CARD.md`, `docs/ARCHITECTURE_DECISIONS.md` (9 ADRs), `docs/ONE_PAGER.md`, `docs/RESUME_BULLETS.md`

## 2.0.0 — 2026-09-19

- Rebuilt the Streamlit interface
- Added sidebar navigation and dashboard overview
- Added standardized Logistic Regression pipeline
- Added classification metrics and visual diagnostics
- Added prediction form validation
- Added K-Means elbow and silhouette analysis
- Added clustered CSV download
- Split application logic into reusable modules
- Added automated tests
- Added GitHub Actions CI
- Rewrote project documentation
- Added MIT license and contribution guide
