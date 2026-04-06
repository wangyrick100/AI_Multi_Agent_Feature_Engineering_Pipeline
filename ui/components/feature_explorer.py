"""
Feature Explorer Component
───────────────────────────
Interactive table + detail panel for browsing feature candidates.

Features:
  - Filter by category, leakage risk, predictive value
  - Full-text search across name / business_intuition
  - Sort by any evaluation metric
  - Inline Python/SQL code viewer
  - Side-by-side comparison of similar features
"""
from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd
import streamlit as st


def render_feature_explorer(
    candidates: List[Any],
    evaluations: Optional[List[Any]] = None,
    selected_ids: Optional[List[str]] = None,
) -> None:
    """
    Render the feature exploration panel.

    Parameters
    ----------
    candidates    List of FeatureDefinition objects.
    evaluations   List of FeatureEvaluation objects (optional).
    selected_ids  List of selected feature IDs (used for highlighting).
    """
    if not candidates:
        st.info("No feature candidates available yet. Run the pipeline first.")
        return

    st.subheader("🧩 Feature Explorer")

    # Build a merged DataFrame for display
    df = _build_display_df(candidates, evaluations or [], selected_ids or [])

    # ── Filters ──────────────────────────────────────────────────────────────
    with st.expander("🔎 Filters & Search", expanded=True):
        col_search, col_cat, col_risk, col_val = st.columns([3, 2, 2, 2])

        with col_search:
            query = st.text_input(
                "Search name / description",
                placeholder="e.g. rolling, entropy, ratio…",
            )
        with col_cat:
            all_cats = sorted(df["category"].dropna().unique().tolist())
            selected_cats = st.multiselect(
                "Category", all_cats, default=all_cats
            )
        with col_risk:
            all_risks = sorted(df["leakage_risk"].dropna().unique().tolist())
            selected_risks = st.multiselect(
                "Leakage risk", all_risks, default=all_risks
            )
        with col_val:
            show_selected_only = st.checkbox("Show selected only", value=False)

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if query:
        mask &= (
            df["name"].str.contains(query, case=False, na=False)
            | df["business_intuition"].str.contains(query, case=False, na=False)
        )
    if selected_cats:
        mask &= df["category"].isin(selected_cats)
    if selected_risks:
        mask &= df["leakage_risk"].isin(selected_risks)
    if show_selected_only:
        mask &= df["is_selected"]

    filtered = df[mask].reset_index(drop=True)
    st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** features")

    # ── Sort controls ────────────────────────────────────────────────────────
    sort_col, sort_dir = st.columns([3, 1])
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            options=["composite_score", "iv_score", "shap_importance",
                     "mi_score", "rank", "name"],
            index=0,
        )
    with sort_dir:
        ascending = st.radio("Direction", ["↑ Asc", "↓ Desc"], index=1, horizontal=True) == "↑ Asc"

    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=ascending, na_position="last")

    # ── Table ────────────────────────────────────────────────────────────────
    display_cols = [
        c for c in [
            "is_selected", "rank", "name", "category", "leakage_risk",
            "predictive_value", "iv_score", "shap_importance",
            "mi_score", "composite_score", "stability_label",
        ]
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[display_cols].style.applymap(
            _color_risk, subset=["leakage_risk"] if "leakage_risk" in display_cols else []
        ).applymap(
            _color_selected, subset=["is_selected"] if "is_selected" in display_cols else []
        ),
        use_container_width=True,
        height=min(600, 50 + 35 * len(filtered)),
    )

    # ── Detail view ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Feature Detail")

    feature_names = filtered["name"].tolist()
    if not feature_names:
        return

    selected_name = st.selectbox("Select a feature to inspect:", feature_names)
    row = filtered[filtered["name"] == selected_name].iloc[0]

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"**Category:** `{row.get('category', '—')}`")
        st.markdown(f"**Leakage Risk:** `{row.get('leakage_risk', '—')}`")
        st.markdown(f"**Predictive Value:** `{row.get('predictive_value', '—')}`")
        st.markdown(f"**IV Score:** `{row.get('iv_score', '—')}`")
        st.markdown(f"**SHAP Importance:** `{row.get('shap_importance', '—')}`")
        st.markdown(f"**MI Score:** `{row.get('mi_score', '—')}`")
        st.markdown(f"**Composite Score:** `{row.get('composite_score', '—')}`")
    with col_right:
        st.markdown("**Business Intuition:**")
        st.info(row.get("business_intuition", "—") or "—")
        st.markdown("**Mathematical Definition:**")
        st.latex(row.get("mathematical_definition", "").strip() or r"\text{N/A}")

    tab_py, tab_sql = st.tabs(["🐍 Python Code", "🗄️ SQL Code"])
    with tab_py:
        py_code = row.get("python_code", "") or "# No code generated."
        st.code(py_code, language="python")
    with tab_sql:
        sql_code = row.get("sql_code", "") or "-- No SQL generated."
        st.code(sql_code, language="sql")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_display_df(
    candidates: list, evaluations: list, selected_ids: list
) -> pd.DataFrame:
    """Merge definitions and evaluations into a flat DataFrame."""
    def_records = []
    for c in candidates:
        d = c.model_dump() if hasattr(c, "model_dump") else vars(c)
        def_records.append(d)

    eval_map: dict = {}
    for ev in evaluations:
        ed = ev.model_dump() if hasattr(ev, "model_dump") else vars(ev)
        eval_map[ed.get("feature_id", "")] = ed

    rows = []
    for d in def_records:
        fid = d.get("feature_id", "")
        ev = eval_map.get(fid, {})
        rows.append({
            "feature_id": fid,
            "name": d.get("name", fid),
            "category": str(d.get("category", "unknown")),
            "leakage_risk": str(d.get("leakage_risk", "unknown")),
            "predictive_value": str(
                d.get("expected_predictive_value", d.get("predictive_value", "unknown"))
            ),
            "business_intuition": d.get("business_intuition", ""),
            "mathematical_definition": d.get("mathematical_definition", ""),
            "python_code": d.get("python_code", ""),
            "sql_code": d.get("sql_code", ""),
            "iv_score": _round(ev.get("iv_score")),
            "shap_importance": _round(ev.get("shap_mean_abs", ev.get("shap_importance"))),
            "mi_score": _round(ev.get("mutual_information", ev.get("mi_score"))),
            "composite_score": _round(ev.get("composite_score")),
            "stability_label": ev.get("stability_label", "—"),
            "rank": ev.get("rank"),
            "is_selected": fid in selected_ids,
        })

    return pd.DataFrame(rows)


def _round(v: Any, n: int = 4) -> Any:
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return None


def _color_risk(val: str) -> str:
    colors = {"none": "#DCFCE7", "low": "#FEF9C3", "medium": "#FED7AA", "high": "#FEE2E2"}
    return f"background-color: {colors.get(str(val).lower(), 'white')}"


def _color_selected(val: bool) -> str:
    return "background-color: #D1FAE5; font-weight: bold" if val else ""
