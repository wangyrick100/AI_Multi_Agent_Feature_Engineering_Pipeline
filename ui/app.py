"""
Feature Engineering System — Streamlit UI
──────────────────────────────────────────
Multi-page, single-file Streamlit application.

Pages (sidebar navigation):
  1. 🏠 Home          — architecture overview + quick-start
  2. 📂 Data Upload   — upload dataset / load demo
  3. 🚀 Run Pipeline  — configure + execute the full agent pipeline
  4. 🧩 Features      — explore generated feature candidates
  5. 📊 Evaluation    — charts: IV, SHAP, PSI, correlation
  6. 🎯 Selection     — selected feature set + rejection reasons
  7. 🛡️ Governance   — lineage graph + bias report
  8. ⬇️ Export        — download Python module / SQL / JSON report

Usage:
    streamlit run ui/app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── UI Components ─────────────────────────────────────────────────────────────
from ui.components.data_upload import render_data_upload
from ui.components.pipeline_view import render_pipeline_view, render_pipeline_summary, render_stage_logs
from ui.components.feature_explorer import render_feature_explorer
from ui.components.evaluation_dashboard import render_evaluation_dashboard

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Feature Engineering System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1E40AF; margin-bottom: 0.25rem; }
    .sub-header  { font-size: 1rem; color: #6B7280; margin-bottom: 1.5rem; }
    .agent-card  {
        background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 1rem; margin-bottom: 0.75rem;
    }
    .stProgress > div > div { background-color: #3B82F6 !important; }
    div[data-testid="metric-container"] { background:#F1F5F9; border-radius:8px; padding:0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state() -> None:
    defaults: Dict[str, Any] = {
        "page": "🏠 Home",
        "dataframe": None,
        "target_col": None,
        "task_type": "classification",
        "pipeline_state": None,
        "stage_statuses": {},
        "stage_metrics": {},
        "stage_logs": {},
        "pipeline_running": False,
        "pipeline_done": False,
        "demo_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = [
    "🏠 Home",
    "📂 Data Upload",
    "🚀 Run Pipeline",
    "🧩 Features",
    "📊 Evaluation",
    "🎯 Selection",
    "🛡️ Governance",
    "⬇️ Export",
]

with st.sidebar:
    st.markdown(
        "<div class='main-header' style='font-size:1.4rem'>🧠 FE System</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sub-header'>Multi-Agent Feature Engineering</div>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state["page"]), label_visibility="collapsed")
    st.session_state["page"] = page

    st.markdown("---")

    # Quick status
    state = st.session_state.get("pipeline_state")
    if state:
        candidates = getattr(state, "feature_candidates", []) or []
        sel = getattr(state, "selected_features", None)
        sel_ids = (sel.selected_feature_ids if sel else []) or []
        st.metric("Candidates", len(candidates))
        st.metric("Selected", len(sel_ids))
    else:
        st.caption("Pipeline not yet run.")

    st.markdown("---")
    st.caption("GitHub Copilot · Feature Engineering System")


# ── Page renderer ─────────────────────────────────────────────────────────────

def page_home() -> None:
    st.markdown("<div class='main-header'>🧠 Multi-Agent Feature Engineering System</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Autonomously generates, evaluates, and selects 100+ features "
        "for any tabular ML dataset.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    cols = st.columns(3)
    agents = [
        ("🔍 Schema Agent", "Profiles columns, infers semantic types, detects entity relationships and temporal structure."),
        ("💡 Ideation Agent", "Generates 100–200+ diverse feature candidates across 12 categories with business intuition + math definitions."),
        ("🔨 Construction Agent", "Converts definitions to executable pandas Python code and BigQuery SQL using 50+ code templates."),
        ("📊 Evaluation Agent", "Computes IV/WOE, SHAP, Mutual Information, Pearson/Spearman, and PSI stability for every feature."),
        ("🎯 Selection Agent", "Selects an optimal, non-redundant subset via Pareto ranking, greedy search, or LASSO."),
        ("🛡️ Governance Agent", "Tracks lineage DAGs, runs fairness checks, computes reproducibility hashes, and exports JSON reports."),
        ("🔄 Feedback Agent", "Analyses model performance, detects weak features, proposes improvements, and drives the feedback loop."),
    ]

    for i, (name, desc) in enumerate(agents):
        with cols[i % 3]:
            st.markdown(
                f"<div class='agent-card'><b>{name}</b><br/>"
                f"<span style='font-size:0.85rem;color:#6B7280'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("🚀 Quick Start")
    st.markdown("""
1. **Upload Data** — go to *📂 Data Upload* or load the built-in demo dataset.
2. **Configure** — choose the target column and task type.
3. **Run Pipeline** — click *▶ Run Full Pipeline* on the *🚀 Run Pipeline* page.
4. **Explore** — browse 100+ generated features on the *🧩 Features* page.
5. **Evaluate** — analyse IV, SHAP, PSI charts on the *📊 Evaluation* page.
6. **Export** — download the Python transform module or SQL script.
    """)

    st.markdown("---")
    st.subheader("📐 Architecture")
    st.code("""
raw_df  →  SchemaAgent  →  IdeationAgent  →  ConstructionAgent
                                                       │
                                         FeatureMatrix (pandas DataFrame)
                                                       │
                                           EvaluationAgent  →  SelectionAgent
                                                       │
                                           GovernanceAgent  →  FeedbackAgent
                                                       │
                                               (Orchestrator loop)
                                                       │
                                      selected_features + Python + SQL exports
    """, language="text")


def page_data_upload() -> None:
    df, target_col, task_type = render_data_upload()
    if df is not None:
        st.session_state["dataframe"] = df
        st.session_state["target_col"] = target_col
        st.session_state["task_type"] = task_type

        if st.button("➡️ Proceed to Pipeline", type="primary"):
            st.session_state["page"] = "🚀 Run Pipeline"
            st.rerun()


def page_run_pipeline() -> None:
    st.markdown("<div class='main-header'>🚀 Run Pipeline</div>", unsafe_allow_html=True)

    df = st.session_state.get("dataframe")
    if df is None:
        st.warning("No data loaded. Please go to **📂 Data Upload** first.")
        return

    target_col = st.session_state.get("target_col")
    task_type = st.session_state.get("task_type", "classification")

    with st.expander("⚙️ Advanced Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_features = st.slider("Max selected features", 10, 200, 50)
            max_iter = st.slider("Max feedback iterations", 1, 5, 3)
        with col2:
            selection_method = st.selectbox("Selection method", ["pareto", "greedy", "lasso"])
            min_iv = st.slider("Minimum IV threshold", 0.0, 0.5, 0.02, step=0.01)
        with col3:
            shap_samples = st.slider("SHAP sample limit", 500, 10_000, 5_000, step=500)
            corr_thresh = st.slider("Redundancy correlation threshold", 0.7, 0.99, 0.90, step=0.01)

    st.markdown("---")

    col_run, col_step = st.columns([2, 1])
    with col_run:
        run_full = st.button("▶ Run Full Pipeline", type="primary", use_container_width=True,
                             disabled=st.session_state.get("pipeline_running", False))
    with col_step:
        step_mode = st.toggle("Step-by-step mode", value=False)

    # Pipeline execution
    if run_full:
        _run_pipeline(df, target_col, task_type, {
            "max_selected_features": max_features,
            "max_feedback_iterations": max_iter,
            "selection_method": selection_method,
            "min_iv_threshold": min_iv,
            "shap_max_samples": shap_samples,
            "correlation_redundancy_threshold": corr_thresh,
        })

    # Display current stage statuses
    render_pipeline_view(
        stage_statuses=st.session_state.get("stage_statuses", {}),
        stage_metrics=st.session_state.get("stage_metrics", {}),
    )

    if st.session_state.get("stage_logs"):
        render_stage_logs(st.session_state["stage_logs"])

    if st.session_state.get("pipeline_done") and st.session_state.get("pipeline_state"):
        st.markdown("---")
        render_pipeline_summary(st.session_state["pipeline_state"])
        if st.button("➡️ Explore Features", type="primary"):
            st.session_state["page"] = "🧩 Features"
            st.rerun()


def _run_pipeline(
    df: Any,
    target_col: Optional[str],
    task_type: str,
    overrides: Dict,
) -> None:
    """Execute the full agent pipeline with live Streamlit updates."""
    from core.config import Config
    from core.orchestrator import FeatureEngineeringOrchestrator

    st.session_state["pipeline_running"] = True
    st.session_state["pipeline_done"] = False
    st.session_state["stage_statuses"] = {}
    st.session_state["stage_metrics"] = {}
    st.session_state["stage_logs"] = {}

    cfg = Config()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            object.__setattr__(cfg, k, v)

    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    def progress_cb(stage: str, status: str, metrics: Dict) -> None:
        st.session_state["stage_statuses"][stage] = status
        st.session_state["stage_metrics"][stage] = metrics

    try:
        orch = FeatureEngineeringOrchestrator(config=cfg)
        with st.spinner("🔄 Pipeline running…"):
            state = orch.run(
                df=df,
                target_column=target_col or df.columns[-1],
                task_type=task_type,
                progress_callback=progress_cb,
            )
        st.session_state["pipeline_state"] = state
        st.session_state["pipeline_done"] = True
        st.session_state["pipeline_running"] = False
        st.success("✅ Pipeline completed successfully!")
        st.rerun()
    except Exception as e:
        st.session_state["pipeline_running"] = False
        st.error(f"Pipeline error: {e}")
        import traceback
        st.code(traceback.format_exc(), language="text")


def page_features() -> None:
    state = st.session_state.get("pipeline_state")
    if state is None:
        st.warning("Pipeline has not been run yet.")
        return
    candidates = getattr(state, "feature_candidates", []) or []
    evaluations = getattr(state, "evaluations", []) or []
    sel = getattr(state, "selected_features", None)
    selected_ids = (sel.selected_feature_ids if sel else []) or []
    render_feature_explorer(candidates, evaluations, selected_ids)


def page_evaluation() -> None:
    state = st.session_state.get("pipeline_state")
    if state is None:
        st.warning("Pipeline has not been run yet.")
        return
    evaluations = getattr(state, "evaluations", []) or []
    candidates = getattr(state, "feature_candidates", []) or []
    sel = getattr(state, "selected_features", None)
    selected_ids = (sel.selected_feature_ids if sel else []) or []
    render_evaluation_dashboard(evaluations, candidates, selected_ids)


def page_selection() -> None:
    st.markdown("<div class='main-header'>🎯 Feature Selection</div>", unsafe_allow_html=True)
    state = st.session_state.get("pipeline_state")
    if state is None:
        st.warning("Pipeline has not been run yet.")
        return

    sel = getattr(state, "selected_features", None)
    if sel is None:
        st.info("Selection results not available.")
        return

    selected_ids = sel.selected_feature_ids or []
    rejected_ids = sel.rejected_feature_ids or []
    rejection_reasons = sel.rejection_reasons or {}

    c1, c2 = st.columns(2)
    c1.metric("Selected features", len(selected_ids))
    c2.metric("Rejected features", len(rejected_ids))

    tab_sel, tab_rej = st.tabs(["✅ Selected", "❌ Rejected"])

    with tab_sel:
        if selected_ids:
            st.dataframe(pd.DataFrame({"feature_id": selected_ids}), use_container_width=True)
        else:
            st.info("No features selected.")

    with tab_rej:
        if rejected_ids:
            rej_data = [
                {"feature_id": fid, "reason": rejection_reasons.get(fid, "—")}
                for fid in rejected_ids
            ]
            st.dataframe(pd.DataFrame(rej_data), use_container_width=True)
        else:
            st.info("No rejections recorded.")


def page_governance() -> None:
    st.markdown("<div class='main-header'>🛡️ Governance & Lineage</div>", unsafe_allow_html=True)
    state = st.session_state.get("pipeline_state")
    if state is None:
        st.warning("Pipeline has not been run yet.")
        return

    governance = getattr(state, "governance_records", []) or []
    if not governance:
        st.info("No governance records available.")
        return

    st.metric("Governed features", len(governance))

    for rec in governance[:5]:
        d = rec.model_dump() if hasattr(rec, "model_dump") else vars(rec)
        with st.expander(f"📋 {d.get('feature_id', 'unknown')}", expanded=False):
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(f"**Reproducibility hash:** `{d.get('reproducibility_hash', '—')[:16]}…`")
                st.markdown(f"**PII adjacent:** `{d.get('pii_adjacent', False)}`")
                st.markdown(f"**Bias flags:** `{d.get('bias_flags', [])}`")
            with col_r:
                bias_report = d.get("bias_report", {})
                if bias_report:
                    try:
                        import plotly.express as px
                        bias_df = pd.DataFrame(
                            [(k, v) for k, v in bias_report.items()],
                            columns=["protected_group", "demographic_parity_diff"],
                        )
                        if not bias_df.empty:
                            fig = px.bar(
                                bias_df, x="protected_group",
                                y="demographic_parity_diff",
                                title="Demographic Parity Difference",
                                color_discrete_sequence=["#F59E0B"],
                                height=250,
                            )
                            fig.add_hline(y=0.1, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.json(bias_report)
                else:
                    st.caption("No bias data recorded.")


def page_export() -> None:
    st.markdown("<div class='main-header'>⬇️ Export</div>", unsafe_allow_html=True)
    state = st.session_state.get("pipeline_state")
    if state is None:
        st.warning("Pipeline has not been run yet.")
        return

    sel = getattr(state, "selected_features", None)
    selected_ids = (sel.selected_feature_ids if sel else []) or []
    candidates = getattr(state, "feature_candidates", []) or []
    evaluations = getattr(state, "evaluations", []) or []

    st.info(
        f"Export **{len(selected_ids)} selected features** in your preferred format below."
    )

    tab_py, tab_sql, tab_json, tab_report = st.tabs(
        ["🐍 Python Module", "🗄️ SQL Script", "📋 JSON Definitions", "📄 Governance Report"]
    )

    selected_defs = [c for c in candidates
                     if (c.feature_id if hasattr(c, "feature_id") else c.get("feature_id")) in selected_ids]

    with tab_py:
        st.markdown("**Standalone Python transform class** — copy into any project.")
        if selected_defs:
            from pipelines.python_pipeline import PythonPipeline
            pipe = PythonPipeline(selected_defs)
            py_source = pipe.generate_module_code()
            st.code(py_source[:3000] + ("\n…(truncated)" if len(py_source) > 3000 else ""), language="python")
            st.download_button(
                "⬇️ Download features_transform.py",
                data=py_source.encode(),
                file_name="features_transform.py",
                mime="text/x-python",
            )
        else:
            st.info("No features to export.")

    with tab_sql:
        st.markdown("**BigQuery-compatible SQL SELECT** with all feature expressions.")
        src_table = st.text_input("Source table name", value="raw_events")
        if selected_defs and src_table:
            from pipelines.sql_pipeline import SQLPipeline
            sql_pipe = SQLPipeline(selected_defs, source_table=src_table, dialect="bigquery")
            sql_text = sql_pipe.generate()
            st.code(sql_text[:5000] + ("\n-- …(truncated)" if len(sql_text) > 5000 else ""), language="sql")
            st.download_button(
                "⬇️ Download features.sql",
                data=sql_text.encode(),
                file_name="features.sql",
                mime="text/plain",
            )

    with tab_json:
        st.markdown("**Feature definitions as JSON** — importable into other systems.")
        import json
        defs_json = json.dumps(
            [c.model_dump() if hasattr(c, "model_dump") else vars(c) for c in selected_defs],
            indent=2,
            default=str,
        )
        st.json(json.loads(defs_json))
        st.download_button(
            "⬇️ Download feature_definitions.json",
            data=defs_json.encode(),
            file_name="feature_definitions.json",
            mime="application/json",
        )

    with tab_report:
        st.markdown("**Full pipeline report** (state summary as JSON).")
        report: Dict = {
            "n_candidates": len(candidates),
            "n_selected": len(selected_ids),
            "selected_feature_ids": selected_ids,
            "evaluation_summary": [
                {k: getattr(ev, k, None) for k in ["feature_id", "iv_score", "shap_importance", "composite_score", "rank"]}
                for ev in (evaluations or [])
            ],
        }
        import json
        report_json = json.dumps(report, indent=2, default=str)
        st.json(json.loads(report_json))
        st.download_button(
            "⬇️ Download pipeline_report.json",
            data=report_json.encode(),
            file_name="pipeline_report.json",
            mime="application/json",
        )


# ── Import pandas here for page_selection use ─────────────────────────────────
import pandas as pd

# ── Route to the active page ──────────────────────────────────────────────────
PAGE_MAP = {
    "🏠 Home": page_home,
    "📂 Data Upload": page_data_upload,
    "🚀 Run Pipeline": page_run_pipeline,
    "🧩 Features": page_features,
    "📊 Evaluation": page_evaluation,
    "🎯 Selection": page_selection,
    "🛡️ Governance": page_governance,
    "⬇️ Export": page_export,
}

active_fn = PAGE_MAP.get(st.session_state["page"], page_home)
active_fn()
