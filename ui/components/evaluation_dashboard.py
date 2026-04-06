"""
Evaluation Dashboard Component
────────────────────────────────
Renders interactive Plotly charts for feature evaluation results:
  – IV score bar chart (colored by WOE bin quality)
  – SHAP importance chart
  – MI score heatmap
  – PSI stability treemap
  – Correlation matrix heatmap
  – Feature rank scatter (SHAP vs IV)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def render_evaluation_dashboard(
    evaluations: List[Any],
    candidates: Optional[List[Any]] = None,
    selected_ids: Optional[List[str]] = None,
) -> None:
    """
    Render the full evaluation dashboard.

    Parameters
    ----------
    evaluations   List of FeatureEvaluation objects.
    candidates    Optional list of FeatureDefinition objects (for category colouring).
    selected_ids  List of selected feature_ids (highlighted in charts).
    """
    if not evaluations:
        st.info("No evaluation results yet. Run the pipeline first.")
        return

    if not PLOTLY_AVAILABLE:
        st.error("Plotly is required for the evaluation dashboard. Run `pip install plotly`.")
        return

    selected_ids = selected_ids or []
    df = _build_eval_df(evaluations, candidates or [], selected_ids)

    st.subheader("📊 Feature Evaluation Dashboard")

    # Top-level KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total features", len(df))
    c2.metric("Selected", int(df["is_selected"].sum()))
    c3.metric("Stable (PSI<0.1)", int((df["stability_label"] == "stable").sum()) if "stability_label" in df.columns else "—")
    c4.metric("High IV (≥0.1)", int((df["iv_score"] >= 0.1).sum()) if "iv_score" in df.columns else "—")
    c5.metric("High leakage risk", int((df.get("leakage_risk", pd.Series(dtype=str)) == "high").sum()))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["IV Score", "SHAP Importance", "Stability (PSI)", "Correlation Matrix", "Score Scatter"]
    )

    with tab1:
        _render_iv_chart(df)

    with tab2:
        _render_shap_chart(df)

    with tab3:
        _render_stability_chart(df)

    with tab4:
        _render_correlation_matrix(df)

    with tab5:
        _render_scatter(df)


# ── Individual charts ─────────────────────────────────────────────────────────

def _render_iv_chart(df: pd.DataFrame) -> None:
    st.markdown("**Information Value (IV) — higher is better**")
    st.caption(
        "IV < 0.02: unusable · 0.02–0.1: weak · 0.1–0.3: medium · 0.3–0.5: strong · >0.5: suspect leakage"
    )
    d = df.nlargest(40, "iv_score").copy() if "iv_score" in df.columns else df
    if d.empty:
        st.info("No IV scores to display.")
        return
    fig = px.bar(
        d.sort_values("iv_score"),
        x="iv_score",
        y="name",
        orientation="h",
        color="category",
        text="iv_score",
        height=max(400, 20 * len(d)),
        title="Top 40 Features by Information Value",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(yaxis_title="", xaxis_title="IV", showlegend=True, margin=dict(l=0))
    st.plotly_chart(fig, use_container_width=True)


def _render_shap_chart(df: pd.DataFrame) -> None:
    st.markdown("**SHAP Mean Absolute Importance**")
    if "shap_importance" not in df.columns:
        st.info("No SHAP scores available.")
        return
    d = df.nlargest(40, "shap_importance").copy()
    if d.empty:
        st.info("No SHAP scores to display.")
        return
    d["color"] = d["is_selected"].map({True: "Selected", False: "Not selected"})
    fig = px.bar(
        d.sort_values("shap_importance"),
        x="shap_importance",
        y="name",
        orientation="h",
        color="color",
        color_discrete_map={"Selected": "#10B981", "Not selected": "#9CA3AF"},
        text="shap_importance",
        height=max(400, 20 * len(d)),
        title="Top 40 Features by SHAP Importance",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(yaxis_title="", xaxis_title="Mean |SHAP|", margin=dict(l=0))
    st.plotly_chart(fig, use_container_width=True)


def _render_stability_chart(df: pd.DataFrame) -> None:
    st.markdown("**Population Stability Index (PSI)**")
    st.caption("Stable < 0.10 · Monitor 0.10–0.25 · Unstable > 0.25")
    if "psi_score" not in df.columns:
        st.info("No PSI scores available.")
        return
    d = df.sort_values("psi_score", ascending=False).head(30)
    color_map = {"stable": "#10B981", "monitor": "#F59E0B", "unstable": "#EF4444", "—": "#9CA3AF"}
    fig = px.bar(
        d,
        x="name",
        y="psi_score",
        color="stability_label",
        color_discrete_map=color_map,
        title="PSI by Feature (top 30)",
        height=400,
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="#F59E0B", annotation_text="Monitor threshold")
    fig.add_hline(y=0.25, line_dash="dash", line_color="#EF4444", annotation_text="Unstable threshold")
    fig.update_layout(xaxis_tickangle=-40, yaxis_title="PSI", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def _render_correlation_matrix(df: pd.DataFrame) -> None:
    st.markdown("**Feature–Feature Spearman Correlation**")
    numeric_cols = [c for c in ["iv_score", "shap_importance", "mi_score",
                                "psi_score", "composite_score", "pearson_r"]
                   if c in df.columns]
    if len(numeric_cols) < 2:
        st.info("Not enough numeric metrics to build a correlation matrix.")
        return
    corr = df[numeric_cols].corr(method="spearman").round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Metric–Metric Correlation", height=400)
    st.plotly_chart(fig, use_container_width=True)


def _render_scatter(df: pd.DataFrame) -> None:
    st.markdown("**IV vs SHAP Importance (bubble = MI score)**")
    if "iv_score" not in df.columns or "shap_importance" not in df.columns:
        st.info("Requires both IV and SHAP scores.")
        return
    size_col = "mi_score" if "mi_score" in df.columns else None
    d = df.dropna(subset=["iv_score", "shap_importance"])
    fig = px.scatter(
        d,
        x="iv_score",
        y="shap_importance",
        color="category" if "category" in d.columns else None,
        size=size_col,
        size_max=20,
        hover_name="name",
        hover_data=["leakage_risk", "stability_label", "composite_score"],
        symbol="is_selected",
        symbol_map={True: "circle", False: "x"},
        title="Feature Quality Landscape (○ = selected)",
        height=500,
    )
    fig.add_vline(x=0.1, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Information Value (IV)", yaxis_title="Mean |SHAP|")
    st.plotly_chart(fig, use_container_width=True)


# ── Helper ────────────────────────────────────────────────────────────────────

def _build_eval_df(
    evaluations: list, candidates: list, selected_ids: list
) -> pd.DataFrame:
    cat_map: Dict[str, str] = {}
    risk_map: Dict[str, str] = {}
    for c in candidates:
        d = c.model_dump() if hasattr(c, "model_dump") else vars(c)
        fid = d.get("feature_id", "")
        cat_map[fid] = str(d.get("category", "other"))
        risk_map[fid] = str(d.get("leakage_risk", "unknown"))

    rows = []
    for ev in evaluations:
        d = ev.model_dump() if hasattr(ev, "model_dump") else vars(ev)
        fid = d.get("feature_id", "")
        rows.append({
            "feature_id": fid,
            "name": d.get("feature_name", d.get("name", fid)),
            "iv_score": _safe_float(d.get("iv_score")),
            "shap_importance": _safe_float(d.get("shap_mean_abs", d.get("shap_importance"))),
            "mi_score": _safe_float(d.get("mutual_information", d.get("mi_score"))),
            "psi_score": _safe_float(d.get("psi_score")),
            "pearson_r": _safe_float(d.get("pearson_corr_with_target", d.get("pearson_r"))),
            "composite_score": _safe_float(d.get("composite_score")),
            "stability_label": d.get("stability_label", "—"),
            "rank": d.get("rank"),
            "category": cat_map.get(fid, "other"),
            "leakage_risk": risk_map.get(fid, "unknown"),
            "is_selected": fid in selected_ids,
        })
    return pd.DataFrame(rows)


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
