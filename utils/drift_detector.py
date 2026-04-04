"""
Drift Detector Utility
───────────────────────
Provides Population Stability Index (PSI) computation, covariate shift
detection, and time-based feature drift monitoring.

PSI Interpretation:
  < 0.10  — no significant shift (stable)
  0.10–0.25 — moderate shift (monitor)
  > 0.25  — significant shift (retrain / flag)
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class DriftDetector:
    """
    Standalone drift detector for numeric and categorical features.

    Parameters
    ----------
    n_bins            Number of bins for PSI computation.
    psi_warn_threshold   PSI threshold for 'monitor' label.
    psi_fail_threshold   PSI threshold for 'unstable' label.
    """

    STABLE_LABEL = "stable"
    MONITOR_LABEL = "monitor"
    UNSTABLE_LABEL = "unstable"

    def __init__(
        self,
        n_bins: int = 10,
        psi_warn_threshold: float = 0.10,
        psi_fail_threshold: float = 0.25,
    ) -> None:
        self.n_bins = n_bins
        self.psi_warn_threshold = psi_warn_threshold
        self.psi_fail_threshold = psi_fail_threshold

    # ── Public API ───────────────────────────────────────────────────────────

    def compute_psi(
        self,
        expected: pd.Series,
        actual: pd.Series,
    ) -> float:
        """Return PSI scalar for a single feature."""
        if expected.dtype == object or str(expected.dtype) == "category":
            return self._psi_categorical(expected, actual)
        return self._psi_numeric(expected, actual)

    def label(self, psi: float) -> str:
        if psi >= self.psi_fail_threshold:
            return self.UNSTABLE_LABEL
        if psi >= self.psi_warn_threshold:
            return self.MONITOR_LABEL
        return self.STABLE_LABEL

    def scan_dataframe(
        self,
        expected_df: pd.DataFrame,
        actual_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute PSI for all common numeric/categorical columns.

        Returns a DataFrame with columns [feature, psi, label, ks_stat, ks_p].
        """
        records = []
        common = [c for c in expected_df.columns if c in actual_df.columns]
        for col in common:
            exp_s = expected_df[col].dropna()
            act_s = actual_df[col].dropna()
            if len(exp_s) < 2 or len(act_s) < 2:
                continue
            psi = self.compute_psi(exp_s, act_s)
            lbl = self.label(psi)
            ks_stat, ks_p = (np.nan, np.nan)
            if exp_s.dtype != object:
                try:
                    ks_stat, ks_p = ks_2samp(exp_s, act_s)
                except Exception:
                    pass
            records.append(
                dict(feature=col, psi=round(psi, 4), label=lbl,
                     ks_stat=round(float(ks_stat), 4) if not np.isnan(ks_stat) else None,
                     ks_p=round(float(ks_p), 4) if not np.isnan(ks_p) else None)
            )
        return pd.DataFrame(records)

    def monitor_feature_drift(
        self,
        df: pd.DataFrame,
        ts_col: str,
        feature_cols: List[str],
        window_days: int = 30,
    ) -> pd.DataFrame:
        """
        Split data at the midpoint of the time range, then compute PSI
        for each numeric feature. Returns drift report DataFrame.
        """
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        midpoint = ts.min() + (ts.max() - ts.min()) / 2
        base = df[ts <= midpoint]
        current = df[ts > midpoint]
        if len(base) < 10 or len(current) < 10:
            warnings.warn("Insufficient data for drift detection after time split.")
            return pd.DataFrame()
        return self.scan_dataframe(base[feature_cols], current[feature_cols])

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _psi_numeric(self, expected: pd.Series, actual: pd.Series) -> float:
        """PSI for a continuous numeric series."""
        exp_clean = expected.replace([np.inf, -np.inf], np.nan).dropna()
        act_clean = actual.replace([np.inf, -np.inf], np.nan).dropna()
        if len(exp_clean) < self.n_bins:
            return 0.0
        breakpoints = np.nanpercentile(
            exp_clean, np.linspace(0, 100, self.n_bins + 1)
        )
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        exp_counts = np.histogram(exp_clean, bins=breakpoints)[0]
        act_counts = np.histogram(act_clean, bins=breakpoints)[0]

        exp_pct = (exp_counts / exp_counts.sum()).clip(min=1e-6)
        act_pct = (act_counts / act_counts.sum()).clip(min=1e-6)
        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi)

    def _psi_categorical(self, expected: pd.Series, actual: pd.Series) -> float:
        """PSI for a categorical / string series."""
        all_cats = set(expected.astype(str)) | set(actual.astype(str))
        exp_counts = expected.astype(str).value_counts()
        act_counts = actual.astype(str).value_counts()
        total_exp = len(expected)
        total_act = len(actual)
        psi = 0.0
        for cat in all_cats:
            p_exp = max(exp_counts.get(cat, 0) / total_exp, 1e-6)
            p_act = max(act_counts.get(cat, 0) / total_act, 1e-6)
            psi += (p_act - p_exp) * np.log(p_act / p_exp)
        return float(psi)
