"""
Schema Understanding Agent
──────────────────────────
Profiles every column of the input DataFrame:
  • infers raw dtype + semantic type (user_id, timestamp, amount, …)
  • computes per-column statistics (mean, std, cardinality, missingness, …)
  • estimates distribution shape
  • detects temporal granularity for date/time columns
  • identifies entity relationships between columns
  • emits a SchemaAnalysis object that downstream agents use
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from agents.base_agent import BaseAgent
from core.config import Config
from core.data_models import (
    ColumnProfile,
    DistributionType,
    EntityRelationship,
    PipelineState,
    SchemaAnalysis,
    SemanticType,
)


class SchemaUnderstandingAgent(BaseAgent):
    """
    Agent 1 — Schema Understanding

    Fully characterises the input dataset and enriches PipelineState with a
    `SchemaAnalysis` containing per-column profiles, detected entity columns,
    temporal columns, and inter-column relationships.
    """

    name = "SchemaUnderstandingAgent"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._temporal_kw = [k.lower() for k in config.temporal_column_keywords]
        self._id_kw = [k.lower() for k in config.id_column_keywords]

    # ── Main logic ──────────────────────────────────────────────────────────

    def _run(self, df: pd.DataFrame, state: PipelineState) -> PipelineState:
        sample = df.head(self.config.sample_rows_for_profiling).copy()

        profiles: List[ColumnProfile] = []
        for col in sample.columns:
            self.logger.debug("  profiling column: %s", col)
            profile = self._profile_column(col, sample, df)
            profiles.append(profile)

        entity_cols = [p.name for p in profiles if p.semantic_type in (
            SemanticType.USER_ID, SemanticType.SESSION_ID,
            SemanticType.ITEM_ID, SemanticType.TRANSACTION_ID
        )]
        temporal_cols = [p.name for p in profiles if p.semantic_type in (
            SemanticType.TIMESTAMP, SemanticType.DATE
        )]
        numeric_cols = [p.name for p in profiles if p.semantic_type in (
            SemanticType.AMOUNT, SemanticType.CONTINUOUS
        ) or (p.raw_dtype.startswith(("int", "float")) and
              p.semantic_type not in (SemanticType.USER_ID, SemanticType.BINARY_FLAG))]
        categorical_cols = [p.name for p in profiles if p.semantic_type in (
            SemanticType.CATEGORICAL, SemanticType.BINARY_FLAG
        )]
        text_cols = [p.name for p in profiles if p.semantic_type == SemanticType.TEXT]

        # Detect target automatically if not provided
        target = state.target_column
        if not target:
            target = self._guess_target(profiles)
            if target:
                state.target_column = target
                self.logger.info("  Auto-detected target column: %s", target)

        relationships = self._detect_relationships(df, profiles, entity_cols, temporal_cols)

        analysis = SchemaAnalysis(
            dataset_id=state.dataset_id,
            n_rows=len(df),
            n_cols=len(df.columns),
            column_profiles=profiles,
            entity_columns=entity_cols,
            temporal_columns=temporal_cols,
            target_column=target,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            text_columns=text_cols,
            relationships=relationships,
            domain_hints=state.domain_hints,
        )

        state.schema_analysis = analysis
        self._log_summary(analysis)
        return state

    # ── Column profiler ─────────────────────────────────────────────────────

    def _profile_column(
        self, col: str, sample: pd.DataFrame, full_df: pd.DataFrame
    ) -> ColumnProfile:
        series = sample[col]
        n = len(series)
        missing_rate = series.isna().mean()
        unique_count = series.nunique(dropna=True)
        cardinality_ratio = unique_count / max(n, 1)

        raw_dtype = str(series.dtype)
        semantic_type = self._infer_semantic_type(col, series, cardinality_ratio, full_df)

        profile = ColumnProfile(
            name=col,
            raw_dtype=raw_dtype,
            semantic_type=semantic_type,
            missing_rate=float(missing_rate),
            unique_count=int(unique_count),
            cardinality_ratio=float(cardinality_ratio),
            sample_values=series.dropna().head(5).tolist(),
        )

        # Numeric stats
        if pd.api.types.is_numeric_dtype(series) and semantic_type not in (
            SemanticType.USER_ID, SemanticType.IDENTIFIER
        ):
            clean = series.dropna()
            if len(clean) > 0:
                profile.mean = float(clean.mean())
                profile.std = float(clean.std())
                profile.min_val = float(clean.min())
                profile.max_val = float(clean.max())
                profile.p25 = float(clean.quantile(0.25))
                profile.p50 = float(clean.median())
                profile.p75 = float(clean.quantile(0.75))
                if len(clean) >= 4:
                    profile.skewness = float(stats.skew(clean))
                    profile.kurtosis = float(stats.kurtosis(clean))
                profile.distribution = self._estimate_distribution(clean)
                profile.is_monotonic = bool(clean.is_monotonic_increasing or
                                             clean.is_monotonic_decreasing)

        # Top values for categoricals
        if semantic_type in (SemanticType.CATEGORICAL, SemanticType.BINARY_FLAG):
            top = series.value_counts().head(10)
            profile.top_values = top.index.tolist()

        # Temporal granularity
        if semantic_type in (SemanticType.TIMESTAMP, SemanticType.DATE):
            profile.temporal_granularity = self._detect_temporal_granularity(series)

        # Auto-description
        profile.description = self._describe_column(profile)
        return profile

    # ── Type inference ──────────────────────────────────────────────────────

    def _infer_semantic_type(
        self,
        col: str,
        series: pd.Series,
        cardinality_ratio: float,
        full_df: pd.DataFrame,
    ) -> SemanticType:
        col_lower = col.lower()

        # ── Datetime first (most reliable) ──────────────────────────────────
        if pd.api.types.is_datetime64_any_dtype(series):
            if series.dt.time.notna().any() and (series.dt.second != 0).any():
                return SemanticType.TIMESTAMP
            return SemanticType.DATE

        # Try parsing as datetime if dtype ~object
        if series.dtype == object:
            if any(kw in col_lower for kw in self._temporal_kw):
                try:
                    parsed = pd.to_datetime(series.dropna().head(20), infer_datetime_format=True, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        return SemanticType.TIMESTAMP
                except Exception:
                    pass

        # ── Binary flag ─────────────────────────────────────────────────────
        uniq = series.dropna().unique()
        if set(map(str, uniq)).issubset({"0", "1", "True", "False", "true", "false", "yes", "no", "Y", "N"}):
            return SemanticType.BINARY_FLAG

        # ── Target / label ──────────────────────────────────────────────────
        if col_lower in ("target", "label", "y", "churn", "fraud", "default",
                         "is_fraud", "is_churn", "converted", "clicked"):
            return SemanticType.LABEL

        # ── Identifier columns ──────────────────────────────────────────────
        if any(kw in col_lower for kw in self._id_kw) and cardinality_ratio > 0.5:
            if "user" in col_lower:
                return SemanticType.USER_ID
            if "session" in col_lower:
                return SemanticType.SESSION_ID
            if "item" in col_lower or "product" in col_lower or "sku" in col_lower:
                return SemanticType.ITEM_ID
            if "txn" in col_lower or "transaction" in col_lower or "order" in col_lower:
                return SemanticType.TRANSACTION_ID
            return SemanticType.IDENTIFIER

        # ── Numeric: amount vs continuous ───────────────────────────────────
        if pd.api.types.is_numeric_dtype(series):
            if cardinality_ratio > 0.8 and any(kw in col_lower for kw in
                    ("amount", "price", "revenue", "spend", "cost", "value", "fee")):
                return SemanticType.AMOUNT
            if cardinality_ratio > 0.8:
                return SemanticType.CONTINUOUS
            if series.nunique() == 2:
                return SemanticType.BINARY_FLAG
            # Moderate cardinality numeric → treat as categorical-like continuous
            return SemanticType.CONTINUOUS

        # ── Geographic ──────────────────────────────────────────────────────
        if any(kw in col_lower for kw in ("country", "state", "city", "zip",
                                           "postal", "region", "geo", "lat", "lon")):
            return SemanticType.GEOGRAPHIC

        # ── Text (long strings) ─────────────────────────────────────────────
        if series.dtype == object:
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 40:
                return SemanticType.TEXT
            if cardinality_ratio < 0.05:
                return SemanticType.CATEGORICAL
            if cardinality_ratio > 0.5:
                return SemanticType.IDENTIFIER

        return SemanticType.CATEGORICAL

    # ── Distribution estimation ─────────────────────────────────────────────

    def _estimate_distribution(self, s: pd.Series) -> DistributionType:
        if len(s) < 20:
            return DistributionType.UNKNOWN
        skew = float(stats.skew(s))
        kurt = float(stats.kurtosis(s))

        if abs(skew) < 0.5 and abs(kurt) < 1:
            return DistributionType.NORMAL
        if skew > 1.5 and (s > 0).all():
            return DistributionType.LOG_NORMAL
        if skew > 1.0 and (s >= 0).all():
            return DistributionType.EXPONENTIAL
        if kurt > 3:
            return DistributionType.HEAVY_TAIL
        if abs(skew) < 0.3 and s.std() / (s.max() - s.min() + 1e-9) > 0.25:
            return DistributionType.UNIFORM
        if abs(skew) > 0.5:
            return DistributionType.SKEWED
        return DistributionType.UNKNOWN

    # ── Temporal granularity ────────────────────────────────────────────────

    def _detect_temporal_granularity(self, series: pd.Series) -> str:
        try:
            dt = pd.to_datetime(series.dropna().head(500), errors="coerce").dropna()
            if len(dt) < 2:
                return "unknown"
            diffs = dt.sort_values().diff().dropna()
            median_diff = diffs.median()
            secs = median_diff.total_seconds()
            if secs < 60:
                return "second"
            if secs < 3600:
                return "minute"
            if secs < 86400:
                return "hour"
            if secs < 7 * 86400:
                return "day"
            if secs < 32 * 86400:
                return "week"
            return "month"
        except Exception:
            return "unknown"

    # ── Relationship detection ──────────────────────────────────────────────

    def _detect_relationships(
        self,
        df: pd.DataFrame,
        profiles: List[ColumnProfile],
        entity_cols: List[str],
        temporal_cols: List[str],
    ) -> List[EntityRelationship]:
        relationships: List[EntityRelationship] = []

        # 1:N between high-cardinality ID and lower-cardinality categorical
        for ent_col in entity_cols:
            for profile in profiles:
                other = profile.name
                if other == ent_col or other not in df.columns:
                    continue
                if profile.semantic_type in (SemanticType.CATEGORICAL,
                                              SemanticType.GEOGRAPHIC):
                    try:
                        ratio = df.groupby(ent_col)[other].nunique().mean()
                        if ratio > 1.5:
                            relationships.append(EntityRelationship(
                                from_col=ent_col,
                                to_col=other,
                                relationship_type="one_to_many",
                                confidence=min(1.0, ratio / 10),
                            ))
                    except Exception:
                        pass
        return relationships

    # ── Target guessing ─────────────────────────────────────────────────────

    def _guess_target(self, profiles: List[ColumnProfile]) -> Optional[str]:
        target_kw = ["target", "label", "y", "churn", "fraud", "default",
                     "is_fraud", "is_churn", "converted", "clicked", "outcome"]
        for p in profiles:
            if p.semantic_type == SemanticType.LABEL:
                return p.name
            if any(kw in p.name.lower() for kw in target_kw):
                return p.name
        return None

    # ── Description generator ───────────────────────────────────────────────

    def _describe_column(self, p: ColumnProfile) -> str:
        desc = f"Column '{p.name}' — {p.semantic_type.value}. "
        desc += f"Missing: {p.missing_rate:.1%}. Cardinality: {p.unique_count:,}. "
        if p.mean is not None:
            desc += f"Mean={p.mean:.2f}, Std={p.std:.2f}. "
        if p.distribution != DistributionType.UNKNOWN:
            desc += f"Distribution: {p.distribution.value}."
        return desc.strip()

    # ── Logging ─────────────────────────────────────────────────────────────

    def _log_summary(self, analysis: SchemaAnalysis) -> None:
        self.logger.info(
            "  Schema: %d rows × %d cols | entities=%s | temporals=%s | target=%s",
            analysis.n_rows, analysis.n_cols,
            analysis.entity_columns,
            analysis.temporal_columns,
            analysis.target_column,
        )
        for p in analysis.column_profiles:
            self.logger.debug(
                "    %-30s %-15s miss=%.1f%% card=%d",
                p.name, p.semantic_type.value, p.missing_rate * 100, p.unique_count,
            )
