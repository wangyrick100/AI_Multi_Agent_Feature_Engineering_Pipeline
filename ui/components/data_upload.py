"""
Data Upload Component
──────────────────────
Provides a Streamlit widget that lets users:
  1. Drag-and-drop CSV / Parquet / Excel files
  2. Configure the target column and task type
  3. Load a built-in synthetic demo dataset

Returns a (df, target_col, task_type) tuple stored in session state.
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import streamlit as st


def render_data_upload() -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
    """
    Render the full data-upload panel.

    Returns
    -------
    df          Loaded DataFrame (None if nothing uploaded yet).
    target_col  Name of the target column chosen by the user.
    task_type   'classification' or 'regression'.
    """
    st.header("📂 Data Upload")
    st.markdown(
        "Upload your dataset (CSV, Parquet, or Excel) **or** load the built-in "
        "demo dataset (synthetic e-commerce transactions)."
    )

    col_upload, col_demo = st.columns([3, 1])

    df: Optional[pd.DataFrame] = None

    with col_upload:
        uploaded = st.file_uploader(
            "Drop a file here or click to browse",
            type=["csv", "parquet", "xlsx", "xls"],
            help="Max 200 MB. Ensure the file has a header row.",
        )
        if uploaded is not None:
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            try:
                if ext == "csv":
                    df = pd.read_csv(uploaded)
                elif ext == "parquet":
                    df = pd.read_parquet(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                st.success(f"Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    with col_demo:
        st.markdown("**Or use demo data**")
        if st.button("🔄 Load Demo Dataset", use_container_width=True):
            df = _load_demo_dataset()
            st.session_state["demo_loaded"] = True
            st.success(f"Demo loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")

    # If a demo was loaded in a previous run, restore it
    if df is None and st.session_state.get("demo_loaded") and "dataframe" in st.session_state:
        df = st.session_state["dataframe"]

    if df is not None:
        st.session_state["dataframe"] = df

        with st.expander("👁 Data Preview", expanded=True):
            st.dataframe(df.head(20), use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{df.shape[0]:,}")
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing cells", f"{df.isna().sum().sum():,}")
            c4.metric(
                "Missing %",
                f"{df.isna().mean().mean() * 100:.1f}%",
            )

        st.markdown("---")
        st.subheader("⚙️ Configuration")

        col_a, col_b = st.columns(2)

        with col_a:
            target_col = st.selectbox(
                "Target column",
                options=list(df.columns),
                index=_guess_target_index(df),
                help="The column you want to predict. It will be excluded from feature candidates.",
            )

        with col_b:
            task_type = st.radio(
                "Task type",
                options=["classification", "regression"],
                index=0,
                horizontal=True,
            )

        return df, target_col, task_type

    return None, None, "classification"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_demo_dataset() -> pd.DataFrame:
    """Import and call the sample data generator."""
    try:
        import importlib.util, sys
        from pathlib import Path
        gen_path = Path(__file__).parent.parent.parent / "data" / "sample_data_generator.py"
        spec = importlib.util.spec_from_file_location("sample_data_generator", gen_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.generate_transactions(n_users=500, n_rows=15_000, seed=42)
    except Exception:
        # Fallback: tiny synthetic dataframe so the UI doesn't break
        import numpy as np
        rng = np.random.default_rng(42)
        n = 5_000
        return pd.DataFrame({
            "user_id": rng.choice(200, n).astype(str),
            "event_timestamp": pd.date_range("2023-01-01", periods=n, freq="1h"),
            "amount": rng.exponential(50, n).round(2),
            "category": rng.choice(["food", "travel", "retail", "tech"], n),
            "merchant": rng.choice([f"merchant_{i}" for i in range(50)], n),
            "channel": rng.choice(["web", "mobile", "in-store"], n),
            "is_fraud": (rng.random(n) < 0.035).astype(int),
        })


def _guess_target_index(df: pd.DataFrame) -> int:
    """Heuristic: pick the first column whose name looks like a target."""
    target_hints = ["target", "label", "y", "fraud", "churn", "default", "outcome"]
    for i, col in enumerate(df.columns):
        if any(h in col.lower() for h in target_hints):
            return i
    return len(df.columns) - 1  # default: last column
