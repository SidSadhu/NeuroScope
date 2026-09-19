"""Plotly-based dark-theme visualization suite for NeuroScope."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve

# ── Palette ───────────────────────────────────────────────────────────────────

C = {
    "cyan": "#00d4ff",
    "purple": "#7c3aed",
    "coral": "#ff6b6b",
    "green": "#10b981",
    "amber": "#f59e0b",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "bg": "rgba(0,0,0,0)",
    "card": "rgba(13,22,38,0.45)",
    "grid": "rgba(255,255,255,0.06)",
    "zero": "rgba(255,255,255,0.10)",
}

MODEL_COLORS: dict[str, str] = {
    "Logistic Regression": C["cyan"],
    "Random Forest": C["purple"],
    "Gradient Boosting": C["coral"],
    "Extra Trees": C["green"],
}

CLUSTER_PALETTE = [C["cyan"], C["purple"], C["coral"], C["green"], C["amber"], "#06b6d4", "#f97316"]


def _hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Shared layout helper ──────────────────────────────────────────────────────


def _layout(title: str = "", height: int = 420, **extra) -> dict:
    base = dict(
        title=dict(text=title, font=dict(size=16, color=C["text"]), x=0, xanchor="left"),
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["card"],
        font=dict(family="Inter, sans-serif", color=C["text"], size=12),
        legend=dict(
            bgcolor="rgba(13,22,38,0.92)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color=C["text"]),
        ),
        margin=dict(l=20, r=20, t=52 if title else 20, b=20),
        height=height,
        hoverlabel=dict(
            bgcolor="rgba(10,15,30,0.97)",
            bordercolor="rgba(0,212,255,0.4)",
            font=dict(family="Inter, sans-serif", color=C["text"]),
        ),
    )
    base.update(extra)
    return base


def _axes_style(title: str = "", color: str = "") -> dict:
    d: dict = dict(
        gridcolor=C["grid"],
        zerolinecolor=C["zero"],
        tickfont=dict(color=C["muted"]),
        linecolor=C["grid"],
    )
    if title:
        d["title"] = dict(text=title, font=dict(color=C["text"]))
    return d


# ── ROC curves ────────────────────────────────────────────────────────────────


def plot_roc_curves(results: dict) -> go.Figure:
    """Overlay ROC curves for all models on one chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=C["muted"], width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result.y_test, result.y_prob)
        auc = result.metrics["AUC-ROC"]
        color = MODEL_COLORS.get(name, C["cyan"])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name}  (AUC={auc:.3f})",
                line=dict(color=color, width=2.5),
                fill="tozeroy",
                fillcolor=_hex_to_rgba(color, 0.06),
                hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_layout("ROC Curves — All Models"),
        xaxis=dict(**_axes_style("False Positive Rate")),
        yaxis=dict(**_axes_style("True Positive Rate"), range=[0, 1.05]),
    )
    return fig


# ── Confusion matrix ─────────────────────────────────────────────────────────


def plot_confusion_matrix(y_true, y_pred, model_name: str = "") -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Retained", "Churned"]
    total = cm.sum()
    annotation = [
        [
            f"{cm[i, j]}<br><span style='font-size:11px'>{cm[i, j] / total * 100:.1f}%</span>"
            for j in range(2)
        ]
        for i in range(2)
    ]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            text=annotation,
            texttemplate="%{text}",
            textfont=dict(size=15, color="white"),
            colorscale=[[0, "#070e1e"], [0.5, "#2d1b69"], [1, "#7c3aed"]],
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    title = f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix"
    fig.update_layout(
        **_layout(title, height=380),
        xaxis=dict(**_axes_style("Predicted")),
        yaxis=dict(**_axes_style("Actual")),
    )
    return fig


# ── SHAP importance ───────────────────────────────────────────────────────────


def plot_shap_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal gradient bar chart of mean |SHAP| values."""
    df = importance_df.sort_values("importance", ascending=True)
    labels = df["feature"].str.replace("_", " ").str.title()
    vals = df["importance"].values

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker=dict(
                color=vals,
                colorscale=[[0, "#0d1626"], [0.4, "#2d1b69"], [0.75, "#7c3aed"], [1.0, "#00d4ff"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout("Feature Importance — Mean |SHAP| Value", height=400),
        xaxis=dict(**_axes_style("Mean |SHAP value|")),
        yaxis=dict(**_axes_style()),
    )
    return fig


# ── SHAP waterfall ────────────────────────────────────────────────────────────


def plot_shap_waterfall(waterfall_data: dict, predicted_prob: float) -> go.Figure:
    """SHAP waterfall for a single prediction."""
    if not waterfall_data:
        return go.Figure()

    fnames = [f.replace("_", " ").title() for f in waterfall_data["feature_names"]]
    shap_vals = np.array(waterfall_data["shap_values"])
    fvals = waterfall_data["feature_values"]
    order = np.argsort(np.abs(shap_vals))

    colors = [C["coral"] if s > 0 else C["green"] for s in shap_vals[order]]
    y_labels = [f"{fnames[i]}  ({fvals[i]:.2f})" for i in order]

    fig = go.Figure(
        go.Bar(
            x=shap_vals[order],
            y=y_labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=C["muted"], line_width=1.5)
    fig.update_layout(
        **_layout(f"Prediction Breakdown  (churn probability: {predicted_prob:.1%})", height=450),
        xaxis=dict(**_axes_style("SHAP value (positive = increases churn risk)")),
        yaxis=dict(**_axes_style()),
    )
    return fig


# ── Model radar ──────────────────────────────────────────────────────────────


def plot_model_radar(metrics_df: pd.DataFrame) -> go.Figure:
    """Spider/radar chart comparing all models across 5 metrics."""
    metric_cols = [
        c for c in ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"] if c in metrics_df.columns
    ]
    fig = go.Figure()
    for model_name, row in metrics_df.iterrows():
        vals = [row[m] for m in metric_cols]
        vals_closed = vals + [vals[0]]
        theta_closed = metric_cols + [metric_cols[0]]
        color = MODEL_COLORS.get(str(model_name), C["cyan"])
        fig.add_trace(
            go.Scatterpolar(
                r=vals_closed,
                theta=theta_closed,
                name=str(model_name),
                line=dict(color=color, width=2.2),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.09),
                hovertemplate=f"<b>{model_name}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_layout("Model Performance Radar", height=420),
        polar=dict(
            bgcolor=C["card"],
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=C["grid"],
                tickfont=dict(color=C["muted"]),
                tickangle=0,
            ),
            angularaxis=dict(gridcolor=C["grid"], tickfont=dict(color=C["text"])),
        ),
    )
    return fig


# ── 3-D cluster scatter ───────────────────────────────────────────────────────


def plot_cluster_3d(
    X_3d: np.ndarray, labels: np.ndarray, title: str = "3D Cluster View"
) -> go.Figure:
    """3-D PCA scatter coloured by cluster assignment."""
    fig = go.Figure()
    for i, label in enumerate(sorted(set(labels))):
        mask = labels == label
        name = "Noise" if label == -1 else f"Cluster {label}"
        color = C["muted"] if label == -1 else CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        n_comp = X_3d.shape[1]
        fig.add_trace(
            go.Scatter3d(
                x=X_3d[mask, 0],
                y=X_3d[mask, 1],
                z=X_3d[mask, 2] if n_comp > 2 else np.zeros(mask.sum()),
                mode="markers",
                name=name,
                marker=dict(size=3.5, color=color, opacity=0.75, line=dict(width=0)),
                hovertemplate=(
                    f"<b>{name}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        **_layout(title, height=540),
        scene=dict(
            bgcolor=C["card"],
            xaxis=dict(gridcolor=C["grid"], title="PC 1", tickfont=dict(color=C["muted"])),
            yaxis=dict(gridcolor=C["grid"], title="PC 2", tickfont=dict(color=C["muted"])),
            zaxis=dict(gridcolor=C["grid"], title="PC 3", tickfont=dict(color=C["muted"])),
        ),
    )
    return fig


# ── 2-D cluster scatter ───────────────────────────────────────────────────────


def plot_cluster_2d(x, y, labels: np.ndarray, xlabel: str, ylabel: str) -> go.Figure:
    fig = go.Figure()
    for i, label in enumerate(sorted(set(labels))):
        mask = labels == label
        name = "Noise" if label == -1 else f"Cluster {label}"
        color = C["muted"] if label == -1 else CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=np.array(x)[mask],
                y=np.array(y)[mask],
                mode="markers",
                name=name,
                marker=dict(size=6, color=color, opacity=0.72, line=dict(width=0)),
                hovertemplate=f"<b>{name}</b><br>{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_layout("Cluster Scatter (2D)"),
        xaxis=dict(**_axes_style(xlabel.replace("_", " ").title())),
        yaxis=dict(**_axes_style(ylabel.replace("_", " ").title())),
    )
    return fig


# ── Elbow + silhouette ────────────────────────────────────────────────────────


def plot_elbow_silhouette(elbow_df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: inertia (left) + silhouette (right)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=elbow_df["k"],
            y=elbow_df["inertia"],
            name="Inertia (Elbow)",
            mode="lines+markers",
            line=dict(color=C["cyan"], width=2.5),
            marker=dict(size=8, color=C["cyan"]),
            hovertemplate="K=%{x}<br>Inertia: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=elbow_df["k"],
            y=elbow_df["silhouette"],
            name="Silhouette Score",
            mode="lines+markers",
            line=dict(color=C["coral"], width=2.5, dash="dash"),
            marker=dict(size=8, color=C["coral"]),
            hovertemplate="K=%{x}<br>Silhouette: %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        **_layout("Optimal K — Elbow & Silhouette Analysis"),
        xaxis=dict(**_axes_style("Number of Clusters (K)")),
    )
    fig.update_yaxes(
        title_text="Inertia", secondary_y=False, gridcolor=C["grid"], tickfont=dict(color=C["cyan"])
    )
    fig.update_yaxes(
        title_text="Silhouette Score",
        secondary_y=True,
        tickfont=dict(color=C["coral"]),
        showgrid=False,
    )
    return fig


# ── Algorithm comparison bar ──────────────────────────────────────────────────


def plot_algorithm_comparison(results: dict) -> go.Figure:
    names = list(results.keys())
    scores = [results[n].silhouette for n in names]
    n_k = [results[n].n_clusters for n in names]
    colors = [C["cyan"], C["purple"], C["coral"]][: len(names)]

    fig = go.Figure(
        go.Bar(
            x=names,
            y=scores,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{s:.3f}  ({k} clusters)" for s, k in zip(scores, n_k)],
            textposition="outside",
            textfont=dict(color=C["text"]),
            hovertemplate="<b>%{x}</b><br>Silhouette: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout("Algorithm Comparison — Silhouette Score"),
        xaxis=dict(**_axes_style()),
        yaxis=dict(**_axes_style("Silhouette Score (higher = better)"), range=[0, 0.9]),
    )
    return fig


# ── Churn donut ─────────────────────────────────────────────────────────────


def plot_churn_donut(y: pd.Series) -> go.Figure:
    """Donut of retained vs churned customers with a center KPI."""
    counts = y.value_counts()
    retained = int(counts.get(0, 0))
    churned = int(counts.get(1, 0))
    total = max(retained + churned, 1)
    churn_rate = churned / total

    fig = go.Figure(
        go.Pie(
            values=[retained, churned],
            labels=["Retained", "Churned"],
            hole=0.62,
            marker=dict(colors=[C["green"], C["coral"]], line=dict(width=0)),
            textinfo="percent",
            textfont=dict(color=C["text"], size=12),
            hovertemplate="<b>%{label}</b><br>%{value:,} customers<extra></extra>",
        )
    )
    layout = _layout("Churn Distribution", height=340)
    layout["legend"] = dict(orientation="h", y=-0.08, x=0.5, xanchor="center")
    layout["annotations"] = [
        dict(
            text=f"<b>{churn_rate:.1%}</b><br><span style='font-size:11px;color:{C['muted']}'>churn rate</span>",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=24, color=C["cyan"], family="Space Grotesk, sans-serif"),
        )
    ]
    fig.update_layout(**layout)
    return fig


# ── Correlation heatmap ──────────────────────────────────────────────────────


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Dark-theme correlation heatmap of numeric feature interrelationships."""
    corr = df.select_dtypes(include=[np.number]).corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.columns),
            zmin=-1,
            zmax=1,
            colorscale=[[0, C["cyan"]], [0.5, "#0d1626"], [1, C["coral"]]],
            xgap=2,
            ygap=2,
            colorbar=dict(
                tickfont=dict(color=C["muted"], size=10),
                len=0.85,
                thickness=12,
                outlinewidth=0,
            ),
            hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout("Feature Correlation Matrix", height=520),
        xaxis=dict(tickfont=dict(color=C["muted"], size=10), tickangle=-35),
        yaxis=dict(tickfont=dict(color=C["muted"], size=10)),
    )
    return fig


# ── Risk gauge ───────────────────────────────────────────────────────────────


def plot_risk_gauge(probability: float) -> go.Figure:
    """Semi-circular gauge showing single-prediction churn probability."""
    prob = float(np.clip(probability, 0.0, 1.0))
    if prob > 0.65:
        bar, label = C["coral"], "HIGH RISK"
    elif prob > 0.35:
        bar, label = C["amber"], "MEDIUM RISK"
    else:
        bar, label = C["green"], "LOW RISK"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number=dict(
                suffix="%", font=dict(size=44, color=bar, family="Space Grotesk, sans-serif")
            ),
            gauge={
                "axis": dict(
                    range=[0, 100],
                    tickvals=[0, 35, 65, 100],
                    ticktext=["0", "35", "65", "100"],
                    tickfont=dict(color=C["muted"], size=10),
                ),
                "bar": dict(color=bar, thickness=0.28, line=dict(width=0)),
                "bgcolor": "rgba(13,22,38,0.6)",
                "borderwidth": 0,
                "steps": [
                    dict(range=[0, 35], color="rgba(16,185,129,0.07)"),
                    dict(range=[35, 65], color="rgba(245,158,11,0.07)"),
                    dict(range=[65, 100], color="rgba(255,107,107,0.07)"),
                ],
                "threshold": dict(line=dict(color=bar, width=2), thickness=0.8, value=prob * 100),
            },
        )
    )
    layout = _layout(f"Churn Probability — {label}", height=280)
    layout["margin"] = dict(l=30, r=30, t=52, b=20)
    fig.update_layout(**layout)
    return fig


# ── Cluster feature profiles radar ───────────────────────────────────────────


def plot_cluster_profiles(profiles_df: pd.DataFrame) -> go.Figure:
    """Normalised radar chart of per-cluster feature means."""
    norm = (profiles_df - profiles_df.min()) / (profiles_df.max() - profiles_df.min() + 1e-9)
    features = [f.replace("_", " ").title() for f in profiles_df.columns]
    fig = go.Figure()
    for i, (cluster_id, row) in enumerate(norm.iterrows()):
        vals = row.values.tolist()
        vals_c = vals + [vals[0]]
        feats_c = features + [features[0]]
        color = CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
        fig.add_trace(
            go.Scatterpolar(
                r=vals_c,
                theta=feats_c,
                name=f"Cluster {cluster_id}",
                line=dict(color=color, width=2.2),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.1),
                hovertemplate=f"<b>Cluster {cluster_id}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_layout("Cluster Feature Profiles (Normalised)", height=480),
        polar=dict(
            bgcolor=C["card"],
            radialaxis=dict(
                visible=True, range=[0, 1], gridcolor=C["grid"], tickfont=dict(color=C["muted"])
            ),
            angularaxis=dict(gridcolor=C["grid"], tickfont=dict(color=C["text"], size=11)),
        ),
    )
    return fig
