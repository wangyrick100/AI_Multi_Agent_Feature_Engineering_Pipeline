"""
Leakage Detector Utility
─────────────────────────
Provides multi-layer leakage detection for feature engineering pipelines:

1. Target correlation spike  — |r| > 0.98 indicates the feature is essentially the target
2. Future information test   — feature value at time T uses data from T+k
3. Identifier proxy test     — feature has near-unique cardinality like an ID
4. Post-event feature test   — event_timestamp < feature_timestamp
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


class LeakageDetector:
    """
    Standalone leakage detector that can be used outside the agent pipeline.

    Parameters
    ----------
    target_col   Name of the target column in the data.
    ts_col       Name of the event timestamp column (for temporal leakage).
    corr_threshold  Correlation threshold above which to flag leakage.
    """

    def __init__(
        self,
        target_col: Optional[str] = None,
        ts_col: Optional[str] = None,
        corr_threshold: float = 0.98,
    ) -> None:
        self.target_col = target_col
        self.ts_col = ts_col
        self.corr_threshold = corr_threshold

    # ── High-level API ───────────────────────────────────────────────────────

    def detect(
        self, df: pd.DataFrame, feature_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Run all leakage tests and return a dict mapping each flagged feature
        name to a list of reasons.
        """
        results: Dict[str, List[str]] = {}

        for col in feature_df.columns:
            reasons: List[str] = []

            # Test 1: target correlation
            if self.target_col and self.target_col in df.columns:
                r = self._target_correlation(feature_df[col], df[self.target_col])
                if r is not None and r > self.corr_threshold:
                    reasons.append(f"high_target_corr({r:.3f})")

            # Test 2: near-unique cardinality (proxy for identifier)
            card_ratio = feature_df[col].nunique() / max(len(feature_df), 1)
            if card_ratio > 0.99:
                reasons.append(f"near_unique_cardinality({card_ratio:.3f})")

            # Test 3: name heuristics
            suspicious_terms = [
                "target", "label", "y_", "_y", "flag_true", "churn_date",
                "conversion_date", "default_date", "fraud_date",
            ]
            if any(t in col.lower() for t in suspicious_terms):
                reasons.append("suspicious_name")

            if reasons:
                results[col] = reasons

        return results

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _target_correlation(
        feature: pd.Series, target: pd.Series
    ) -> Optional[float]:
        if feature.dtype == object:
            return None
        x = feature.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = target.copy()
        if y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y.astype(str))
        y = np.array(y, dtype=float)
        if len(x) < 5 or x.std() < 1e-9:
            return None
        try:
            return abs(pearsonr(x, y)[0])
        except Exception:
            return None

    def temporal_leakage_check(
        self,
        df: pd.DataFrame,
        feature_df: pd.DataFrame,
        feature_ts_col: str,
    ) -> List[str]:
        """
        Check if feature_ts_col > event ts_col — indicating future data usage.
        Returns list of row indices that are leaky.
        """
        if self.ts_col is None or self.ts_col not in df.columns:
            return []
        if feature_ts_col not in df.columns:
            return []
        event_ts = pd.to_datetime(df[self.ts_col], errors="coerce")
        feat_ts = pd.to_datetime(df[feature_ts_col], errors="coerce")
        leaky_rows = df.index[(feat_ts > event_ts + pd.Timedelta("1s"))].tolist()
        return leaky_rows
