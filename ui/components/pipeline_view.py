"""
Pipeline View Component
────────────────────────
Renders a live, step-by-step visualisation of the multi-agent pipeline:
  – Stage completion badge (pending / running / done / error)
  – Per-stage metrics (feature count, elapsed time, etc.)
  – Expandable log output per stage
  – Feedback loop progress indicator
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import streamlit as st


# ── Stage definitions ─────────────────────────────────────────────────────────

STAGES: List[Dict[str, str]] = [
    {"key": "schema",       "label": "🔍 Schema Understanding",  "icon": "🔍"},
    {"key": "ideation",     "label": "💡 Feature Ideation",      "icon": "💡"},
    {"key": "construction", "label": "🔨 Feature Construction",  "icon": "🔨"},
    {"key": "evaluation",   "label": "📊 Feature Evaluation",    "icon": "📊"},
    {"key": "selection",    "label": "🎯 Feature Selection",     "icon": "🎯"},
    {"key": "governance",   "label": "🛡️ Governance & Lineage",  "icon": "🛡️"},
    {"key": "feedback",     "label": "🔄 Feedback Optimisation", "icon": "🔄"},
]

STATUS_COLORS = {
    "pending":  "#888888",
    "running":  "#F59E0B",
    "done":     "#10B981",
    "error":    "#EF4444",
    "skipped":  "#6B7280",
}

STATUS_ICONS = {
    "pending":  "⏳",
    "running":  "⚙️",
    "done":     "✅",
    "error":    "❌",
    "skipped":  "⏭️",
}


def render_pipeline_view(
    stage_statuses: Optional[Dict[str, str]] = None,
    stage_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
    feedback_iteration: int = 0,
    max_iterations: int = 5,
) -> None:
    """
    Render the pipeline progress panel.

    Parameters
    ----------
    stage_statuses    Dict mapping stage key → status string.
    stage_metrics     Dict mapping stage key → dict of KV metrics to display.
    feedback_iteration  Current feedback loop iteration.
    max_iterations    Maximum allowed feedback iterations.
    """
    if stage_statuses is None:
        stage_statuses = {}
    if stage_metrics is None:
        stage_metrics = {}

    st.subheader("🚀 Pipeline Progress")

    # Feedback loop indicator
    if max_iterations > 1:
        pct = min(feedback_iteration / max(max_iterations, 1), 1.0)
        st.markdown(
            f"**Feedback loop:** iteration {feedback_iteration} / {max_iterations}"
        )
        st.progress(pct)
        st.markdown("")

    # Stage cards
    for stage in STAGES:
        key = stage["key"]
        status = stage_statuses.get(key, "pending")
        icon = STATUS_ICONS.get(status, "⏳")
        color = STATUS_COLORS.get(status, "#888888")
        metrics = stage_metrics.get(key, {})

        with st.container():
            col_icon, col_label, col_status, col_meta = st.columns([0.6, 3, 1.2, 3])

            with col_icon:
                st.markdown(
                    f"<div style='font-size:1.6rem;text-align:center'>{stage['icon']}</div>",
                    unsafe_allow_html=True,
                )
            with col_label:
                st.markdown(
                    f"<div style='font-weight:600;padding-top:6px'>{stage['label']}</div>",
                    unsafe_allow_html=True,
                )
            with col_status:
                st.markdown(
                    f"<span style='background:{color};color:white;border-radius:6px;"
                    f"padding:2px 8px;font-size:0.78rem'>{icon} {status.upper()}</span>",
                    unsafe_allow_html=True,
                )
            with col_meta:
                if metrics:
                    meta_str = "  ·  ".join(
                        f"**{k}:** {v}" for k, v in metrics.items()
                    )
                    st.markdown(
                        f"<div style='font-size:0.82rem;padding-top:6px'>{meta_str}</div>",
                        unsafe_allow_html=True,
                    )

        # Separator line
        st.markdown(
            "<hr style='margin:4px 0; border:none; border-top: 1px solid #E5E7EB'>",
            unsafe_allow_html=True,
        )


def render_stage_logs(logs: Dict[str, List[str]]) -> None:
    """Render collapsible log panels for each stage."""
    st.subheader("📋 Stage Logs")
    for stage in STAGES:
        key = stage["key"]
        stage_logs = logs.get(key, [])
        with st.expander(stage["label"], expanded=False):
            if stage_logs:
                st.code("\n".join(stage_logs), language="text")
            else:
                st.caption("No logs yet.")


def render_pipeline_summary(state: Any) -> None:
    """
    Render a high-level metrics summary once the pipeline is complete.

    Parameters
    ----------
    state  PipelineState object.
    """
    st.subheader("📈 Pipeline Summary")

    candidates = getattr(state, "feature_candidates", []) or []
    selected = getattr(state, "selection_result", None)
    selected_ids = (selected.selected_feature_ids if selected else []) or []
    evaluations = getattr(state, "feature_evaluations", []) or []
    feedback_reports = getattr(state, "feedback_reports", []) or []
    iterations = len(feedback_reports)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💡 Candidates generated", len(candidates))
    c2.metric("🎯 Features selected", len(selected_ids))
    top_iv = max((e.iv_score for e in evaluations if e.iv_score), default=0.0) if evaluations else 0.0
    c3.metric("🏆 Best IV score", f"{top_iv:.3f}")
    c4.metric("🔄 Feedback iterations", iterations)
