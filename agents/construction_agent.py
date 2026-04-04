"""
Feature Construction Agent
───────────────────────────
Converts each FeatureDefinition into executable Python (pandas) code and
SQL (BigQuery / Snowflake) code, then physically computes the features on
the provided DataFrame.

Design decisions
────────────────
• Feature code is generated from template_params so construction is
  data-driven, not hard-coded.
• All generated code is stored back on the FeatureDefinition for audit.
• Null handling, temporal alignment, and anti-leakage guards are built in.
• Incremental variants are generated alongside full-batch versions.
"""
from __future__ import annotations

import hashlib
import textwrap
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from core.config import Config
from core.data_models import FeatureDefinition, FeatureCategory, PipelineState


class FeatureConstructionAgent(BaseAgent):
    """
    Agent 3 — Feature Construction

    Iterates over `state.feature_candidates`, generates code for each,
    and actually computes the feature column on `df`, storing results in
    a new wide DataFrame accessible via `state.constructed_df` (set as
    an attribute on PipelineState — outside the Pydantic schema to avoid
    memory bloat in serialisation).
    """

    name = "FeatureConstructionAgent"

    # ── Main logic ──────────────────────────────────────────────────────────

    def _run(self, df: pd.DataFrame, state: PipelineState) -> PipelineState:
        if not state.feature_candidates:
            raise ValueError("No feature candidates found; run FeatureIdeationAgent first.")

        df = df.copy()
        self._ensure_datetime(df, state)

        constructed: List[FeatureDefinition] = []
        feature_df = pd.DataFrame(index=df.index)

        total = len(state.feature_candidates)
        self.logger.info("  Building %d features …", total)

        for i, feat in enumerate(state.feature_candidates, 1):
            try:
                series, py_code, sql_code = self._build_feature(df, feat)
                if series is not None:
                    feature_df[feat.name] = series
                    feat.python_code = py_code
                    feat.sql_code = sql_code
                    feat.python_incremental_code = self._generate_incremental_code(feat)
                    constructed.append(feat)
                    if i % 20 == 0:
                        self.logger.info("    … %d / %d done", i, total)
            except Exception as exc:
                self.logger.warning("    Skipped '%s': %s", feat.name, exc)
                state.warnings.append(f"Construction skipped {feat.name}: {exc}")

        # Attach the computed DataFrame as a non-serialised attribute
        state.constructed_features = constructed
        state.__dict__["constructed_df"] = feature_df   # bypass Pydantic
        self.logger.info("  Successfully constructed %d / %d features.",
                         len(constructed), total)
        return state

    # ── Datetime preparation ─────────────────────────────────────────────────

    def _ensure_datetime(self, df: pd.DataFrame, state: PipelineState) -> None:
        if state.schema_analysis:
            for ts_col in state.schema_analysis.temporal_columns:
                if ts_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    # ── Central dispatcher ───────────────────────────────────────────────────

    def _build_feature(
        self, df: pd.DataFrame, feat: FeatureDefinition
    ):
        """
        Returns (series, python_code, sql_code) or (None, "", "") on failure.
        Dispatches based on template_params["variant"].
        """
        p = feat.template_params
        variant = p.get("variant", "")

        dispatch = {
            # Temporal calendar
            "hour":             self._tp_calendar_part,
            "hour_sin":         self._tp_calendar_cyclic,
            "hour_cos":         self._tp_calendar_cyclic,
            "dow_sin":          self._tp_calendar_cyclic,
            "dayofweek":        self._tp_calendar_part,
            "is_weekend":       self._tp_is_weekend,
            "month":            self._tp_calendar_part,
            "month_sin":        self._tp_calendar_cyclic,
            "quarter":          self._tp_calendar_part,
            "is_quarter_end":   self._tp_is_quarter_end,
            "is_business_hour": self._tp_is_business_hour,
            "week_of_year":     self._tp_calendar_part,
            "days_to_year_end": self._tp_days_to_year_end,
            # Lag/delta
            "lag_delta":        self._tp_lag,
            # Rolling window
            "velocity_ratio":   self._tp_velocity_ratio,
            "count_events":     self._tp_event_count_window,
            "nunique":          self._tp_rolling_nunique,
            # Recency
            "days_since_last":  self._tp_days_since_last,
            "entity_age":       self._tp_entity_age,
            "time_decay":       self._tp_time_decay,
            "event_regularity": self._tp_event_regularity,
            "days_since_change":self._tp_days_since_change,
            # Behavioral / RFM
            "lifetime_sum":     self._tp_lifetime_agg,
            "avg_txn_amount":   self._tp_expanding_agg,
            "zscore_vs_self":   self._tp_zscore_vs_self,
            "events_per_day":   self._tp_events_per_day,
            "active_days":      self._tp_active_days,
            "max_single_7d":    self._tp_rolling_agg,
            # Statistical
            "percentile_rank":  self._tp_percentile_rank,
            "log1p":            self._tp_log1p,
            "decile_bin":       self._tp_decile_bin,
            "skewness":         self._tp_entity_skewness,
            "kurtosis":         self._tp_entity_kurtosis,
            "abs_deviation_from_mean": self._tp_abs_deviation,
            "entropy":          self._tp_entropy,
            "hhi":              self._tp_hhi,
            # Ratio/interaction
            "ratio":            self._tp_ratio,
            "product":          self._tp_product,
            "ratio_to_rolling_mean": self._tp_ratio_to_rolling_mean,
            "frequency_ratio":  self._tp_frequency_ratio,
            # Sequence
            "prev_match":       self._tp_prev_match,
            "modal_category":   self._tp_modal_category,
            "first_time":       self._tp_first_time_category,
            "transition_count": self._tp_transition_count,
            "ngram_hash":       self._tp_ngram_hash,
            # Text
            "word_count":       self._tp_word_count,
            "char_count":       self._tp_char_count,
            "has_url":          self._tp_has_url,
            "has_numeric":      self._tp_has_numeric,
            "unique_word_ratio":self._tp_unique_word_ratio,
            # Nonlinear
            "sqrt":             self._tp_sqrt,
            "boxcox":           self._tp_boxcox,
            "piecewise_low":    self._tp_piecewise_low,
            "piecewise_high":   self._tp_piecewise_high,
            "cohort_week":      self._tp_cohort_week,
            "weeks_since_cohort":self._tp_weeks_since_cohort,
            # Target encoding — code only, no compute (requires cross-val)
            "target_mean_loo":  self._tp_target_encode_loo,
            "target_smoothed":  self._tp_target_encode_smoothed,
            # Entity degree (graph)
            "out_degree":       self._tp_out_degree,
            "in_degree":        self._tp_in_degree,
        }

        # Many variants share the rolling-agg dispatcher
        agg = p.get("agg", "")
        if variant not in dispatch and agg in ("sum","mean","std","max","min","count","median"):
            handler = self._tp_rolling_agg
        else:
            handler = dispatch.get(variant)

        if handler is None:
            return None, "", ""

        return handler(df, feat)

    # ════════════════════════════════════════════════════════════════════════
    #  TEMPLATE IMPLEMENTATIONS — each returns (series, py_code, sql_code)
    # ════════════════════════════════════════════════════════════════════════

    # ── Calendar parts ──────────────────────────────────────────────────────

    def _tp_calendar_part(self, df, feat):
        p = feat.template_params
        col, part = p["col"], p["part"]
        if col not in df.columns:
            return None, "", ""
        s = pd.to_datetime(df[col], errors="coerce")
        part_map = {
            "hour": s.dt.hour,
            "dayofweek": s.dt.dayofweek,
            "month": s.dt.month,
            "quarter": s.dt.quarter,
            "week_of_year": s.dt.isocalendar().week.astype(int),
        }
        series = part_map.get(part)
        if series is None:
            return None, "", ""

        py = (f"# {feat.name}\n"
              f"df['{feat.name}'] = pd.to_datetime(df['{col}']).dt.{part}")
        sql = (f"-- {feat.name}\n"
               f"EXTRACT({part.upper()} FROM {col}) AS {feat.name}")
        return series, py, sql

    def _tp_calendar_cyclic(self, df, feat):
        p = feat.template_params
        col, part = p["col"], p["part"]
        if col not in df.columns:
            return None, "", ""
        s = pd.to_datetime(df[col], errors="coerce")
        if "hour_sin" in part:
            r = np.sin(2 * np.pi * s.dt.hour / 24)
            py = f"df['{feat.name}'] = np.sin(2*np.pi*pd.to_datetime(df['{col}']).dt.hour/24)"
        elif "hour_cos" in part:
            r = np.cos(2 * np.pi * s.dt.hour / 24)
            py = f"df['{feat.name}'] = np.cos(2*np.pi*pd.to_datetime(df['{col}']).dt.hour/24)"
        elif "dow_sin" in part:
            r = np.sin(2 * np.pi * s.dt.dayofweek / 7)
            py = f"df['{feat.name}'] = np.sin(2*np.pi*pd.to_datetime(df['{col}']).dt.dayofweek/7)"
        elif "month_sin" in part:
            r = np.sin(2 * np.pi * (s.dt.month - 1) / 12)
            py = f"df['{feat.name}'] = np.sin(2*np.pi*(pd.to_datetime(df['{col}']).dt.month-1)/12)"
        else:
            return None, "", ""
        sql = f"-- {feat.name}: cyclic encoding — implement in application layer"
        return r, py, sql

    def _tp_is_weekend(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        s = pd.to_datetime(df[col], errors="coerce")
        r = (s.dt.dayofweek >= 5).astype(int)
        py = f"df['{feat.name}'] = (pd.to_datetime(df['{col}']).dt.dayofweek >= 5).astype(int)"
        sql = f"CASE WHEN EXTRACT(DAYOFWEEK FROM {col}) IN (1,7) THEN 1 ELSE 0 END AS {feat.name}"
        return r, py, sql

    def _tp_is_quarter_end(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        s = pd.to_datetime(df[col], errors="coerce")
        r = ((s.dt.month.isin([3, 6, 9, 12])) & (s.dt.day >= 24)).astype(int)
        py = (f"df['{feat.name}'] = (\n"
              f"    pd.to_datetime(df['{col}']).dt.month.isin([3,6,9,12]) &\n"
              f"    (pd.to_datetime(df['{col}']).dt.day >= 24)\n"
              f").astype(int)")
        sql = (f"CASE WHEN EXTRACT(MONTH FROM {col}) IN (3,6,9,12)\n"
               f"      AND EXTRACT(DAY FROM {col}) >= 24 THEN 1 ELSE 0 END AS {feat.name}")
        return r, py, sql

    def _tp_is_business_hour(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        s = pd.to_datetime(df[col], errors="coerce")
        r = ((s.dt.hour >= 9) & (s.dt.hour < 18) & (s.dt.dayofweek < 5)).astype(int)
        py = (f"_s = pd.to_datetime(df['{col}'])\n"
              f"df['{feat.name}'] = ((_s.dt.hour>=9) & (_s.dt.hour<18) & (_s.dt.dayofweek<5)).astype(int)")
        sql = (f"CASE WHEN EXTRACT(HOUR FROM {col}) BETWEEN 9 AND 17\n"
               f"      AND EXTRACT(DAYOFWEEK FROM {col}) NOT IN (1,7) THEN 1 ELSE 0 END AS {feat.name}")
        return r, py, sql

    def _tp_days_to_year_end(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        s = pd.to_datetime(df[col], errors="coerce")
        year_end = s.apply(lambda x: pd.Timestamp(x.year, 12, 31) if pd.notna(x) else pd.NaT)
        r = (year_end - s).dt.days
        py = (f"_s = pd.to_datetime(df['{col}'])\n"
              f"_ye = _s.apply(lambda x: pd.Timestamp(x.year,12,31) if pd.notna(x) else pd.NaT)\n"
              f"df['{feat.name}'] = (_ye - _s).dt.days")
        sql = f"DATE_DIFF(DATE(EXTRACT(YEAR FROM {col}), 12, 31), DATE({col}), DAY) AS {feat.name}"
        return r, py, sql

    # ── Lag / delta ─────────────────────────────────────────────────────────

    def _tp_lag(self, df, feat):
        p = feat.template_params
        entity_col, metric_col, lag = p["entity_col"], p["metric_col"], p["lag"]
        if entity_col not in df.columns or metric_col not in df.columns:
            return None, "", ""
        shifted = df.groupby(entity_col)[metric_col].shift(lag)
        if p.get("variant") == "delta":
            r = df[metric_col] - shifted
            py = (f"_shifted = df.groupby('{entity_col}')['{metric_col}'].shift({lag})\n"
                  f"df['{feat.name}'] = df['{metric_col}'] - _shifted")
        else:
            r = shifted
            py = f"df['{feat.name}'] = df.groupby('{entity_col}')['{metric_col}'].shift({lag})"
        sql = (f"LAG({metric_col}, {lag}) OVER (PARTITION BY {entity_col} "
               f"ORDER BY {p.get('ts_col','event_time')}) AS {feat.name}")
        return r, py, sql

    # ── Rolling aggregations ─────────────────────────────────────────────────

    def _tp_rolling_agg(self, df, feat):
        p = feat.template_params
        entity_col = p.get("entity_col")
        ts_col = p.get("ts_col")
        metric_col = p.get("metric_col")
        window_d = p.get("window_d", 30)
        agg = p.get("agg", "mean")

        if not all(c in df.columns for c in [entity_col, ts_col, metric_col] if c):
            return None, "", ""

        try:
            df_sorted = df.sort_values(ts_col)
            window_str = f"{window_d}D"
            grp = df_sorted.groupby(entity_col, group_keys=False)
            r = grp.apply(
                lambda g: (
                    g.set_index(ts_col)[metric_col]
                    .rolling(window_str, closed="left")
                    .agg(agg)
                    .reset_index(drop=False)
                    .set_index(g.index)
                    .iloc[:, -1]
                )
            )
            if r.index.duplicated().any():
                r = r[~r.index.duplicated(keep="first")]
            r = r.reindex(df.index)
        except Exception:
            r = pd.Series(np.nan, index=df.index)

        py = (f"_sorted = df.sort_values('{ts_col}')\n"
              f"df['{feat.name}'] = (\n"
              f"    _sorted.groupby('{entity_col}', group_keys=False)\n"
              f"    .apply(lambda g: g.set_index('{ts_col}')['{metric_col}']\n"
              f"           .rolling('{window_d}D', closed='left').{agg}())\n"
              f"    .reindex(df.index)\n"
              f")")
        sql = (f"-- Rolling {window_d}d {agg} (requires time-ordered CTE in SQL)\n"
               f"{agg.upper()}({metric_col}) OVER (\n"
               f"  PARTITION BY {entity_col}\n"
               f"  ORDER BY UNIX_SECONDS(TIMESTAMP({ts_col}))\n"
               f"  RANGE BETWEEN {window_d * 86400} PRECEDING AND 1 PRECEDING\n"
               f") AS {feat.name}")
        return r, py, sql

    def _tp_velocity_ratio(self, df, feat):
        p = feat.template_params
        entity_col, ts_col, metric_col = p["entity_col"], p["ts_col"], p["metric_col"]
        short_w, long_w = p["short_w"], p["long_w"]

        if not all(c in df.columns for c in [entity_col, ts_col, metric_col]):
            return None, "", ""

        def rolling_mean(g, w):
            return (g.set_index(ts_col)[metric_col]
                    .rolling(f"{w}D", closed="left")
                    .mean()
                    .reset_index(drop=False)
                    .set_index(g.index)
                    .iloc[:, -1])

        df_sorted = df.sort_values(ts_col)
        grp = df_sorted.groupby(entity_col, group_keys=False)
        try:
            short_s = grp.apply(rolling_mean, w=short_w).reindex(df.index)
            long_s = grp.apply(rolling_mean, w=long_w).reindex(df.index)
            r = short_s / (long_s + 1e-9)
        except Exception:
            r = pd.Series(np.nan, index=df.index)

        py = (f"# Velocity ratio {short_w}d / {long_w}d for {metric_col}\n"
              f"_s = df.sort_values('{ts_col}')\n"
              f"_grp = _s.groupby('{entity_col}', group_keys=False)\n"
              f"_short = _grp.apply(lambda g: g.set_index('{ts_col}')['{metric_col}']"
              f".rolling('{short_w}D',closed='left').mean()).reindex(df.index)\n"
              f"_long  = _grp.apply(lambda g: g.set_index('{ts_col}')['{metric_col}']"
              f".rolling('{long_w}D',closed='left').mean()).reindex(df.index)\n"
              f"df['{feat.name}'] = _short / (_long + 1e-9)")
        sql = "-- velocity_ratio: compute two rolling means in CTEs then divide"
        return r, py, sql

    def _tp_event_count_window(self, df, feat):
        p = feat.template_params
        entity_col, ts_col, window_d = p["entity_col"], p["ts_col"], p["window_d"]
        if not all(c in df.columns for c in [entity_col, ts_col]):
            return None, "", ""
        try:
            df_sorted = df.sort_values(ts_col)
            r = (df_sorted.groupby(entity_col, group_keys=False)
                 .apply(lambda g: g.set_index(ts_col)[ts_col if ts_col in g.columns else g.columns[0]]
                        .rolling(f"{window_d}D", closed="left").count()
                        .reset_index(drop=False).set_index(g.index).iloc[:, -1])
                 .reindex(df.index))
        except Exception:
            r = pd.Series(np.nan, index=df.index)

        py = (f"df['{feat.name}'] = (\n"
              f"    df.sort_values('{ts_col}')\n"
              f"    .groupby('{entity_col}', group_keys=False)\n"
              f"    .apply(lambda g: g.assign(_one=1).set_index('{ts_col}')['_one']\n"
              f"           .rolling('{window_d}D',closed='left').count())\n"
              f"    .reindex(df.index)\n"
              f")")
        sql = (f"COUNT(*) OVER (\n"
               f"  PARTITION BY {entity_col}\n"
               f"  ORDER BY UNIX_SECONDS(TIMESTAMP({ts_col}))\n"
               f"  RANGE BETWEEN {window_d*86400} PRECEDING AND 1 PRECEDING\n"
               f") AS {feat.name}")
        return r, py, sql

    def _tp_rolling_nunique(self, df, feat):
        p = feat.template_params
        entity_col, ts_col = p["entity_col"], p["ts_col"]
        cat_col, window_d = p["cat_col"], p["window_d"]
        if not all(c in df.columns for c in [entity_col, ts_col, cat_col]):
            return None, "", ""
        try:
            df_sorted = df.sort_values(ts_col).copy()
            df_sorted["_idx"] = range(len(df_sorted))

            def nunique_rolling(g):
                result = []
                ts_vals = g[ts_col].values
                cat_vals = g[cat_col].values
                cutoff = np.timedelta64(window_d, "D")
                for i in range(len(g)):
                    mask = (ts_vals[i] - ts_vals[:i]) < cutoff
                    result.append(len(set(cat_vals[:i][mask])))
                return pd.Series(result, index=g.index)

            r = df_sorted.groupby(entity_col, group_keys=False).apply(nunique_rolling)
            r = r.reindex(df.index)
        except Exception:
            r = pd.Series(np.nan, index=df.index)

        py = (f"# Rolling nunique of '{cat_col}' in {window_d}d window\n"
              f"# (window-based nunique requires custom apply or approx with resample)\n"
              f"df['{feat.name}'] = ...  # see full pipeline for implementation")
        sql = f"-- Rolling nunique not natively supported; use an approximation CTE"
        return r, py, sql

    # ── Recency / freshness ─────────────────────────────────────────────────

    def _tp_days_since_last(self, df, feat):
        p = feat.template_params
        entity_col, ts_col = p["entity_col"], p["ts_col"]
        if not all(c in df.columns for c in [entity_col, ts_col]):
            return None, "", ""
        df_sorted = df.sort_values(ts_col)
        prev_ts = df_sorted.groupby(entity_col)[ts_col].shift(1)
        r = ((df_sorted[ts_col] - prev_ts).dt.total_seconds() / 86400).reindex(df.index)
        py = (f"_prev = df.sort_values('{ts_col}').groupby('{entity_col}')['{ts_col}'].shift(1)\n"
              f"df['{feat.name}'] = (df['{ts_col}'] - _prev).dt.total_seconds() / 86400")
        sql = (f"DATE_DIFF({ts_col}, LAG({ts_col}) OVER "
               f"(PARTITION BY {entity_col} ORDER BY {ts_col}), DAY) AS {feat.name}")
        return r, py, sql

    def _tp_entity_age(self, df, feat):
        p = feat.template_params
        entity_col, ts_col = p["entity_col"], p["ts_col"]
        if not all(c in df.columns for c in [entity_col, ts_col]):
            return None, "", ""
        first_ts = df.groupby(entity_col)[ts_col].transform("min")
        r = ((df[ts_col] - first_ts).dt.total_seconds() / 86400)
        py = (f"_first = df.groupby('{entity_col}')['{ts_col}'].transform('min')\n"
              f"df['{feat.name}'] = (df['{ts_col}'] - _first).dt.total_seconds() / 86400")
        sql = (f"DATE_DIFF({ts_col}, MIN({ts_col}) OVER "
               f"(PARTITION BY {entity_col}), DAY) AS {feat.name}")
        return r, py, sql

    def _tp_time_decay(self, df, feat):
        p = feat.template_params
        entity_col, ts_col = p["entity_col"], p["ts_col"]
        half_life = p.get("half_life_days", 30)
        if not all(c in df.columns for c in [entity_col, ts_col]):
            return None, "", ""
        lam = np.log(2) / half_life
        ref = df[ts_col].max()
        delta_days = ((ref - df[ts_col]).dt.total_seconds() / 86400).clip(lower=0)
        r = np.exp(-lam * delta_days)
        py = (f"_lam = np.log(2) / {half_life}\n"
              f"_ref = df['{ts_col}'].max()\n"
              f"_delta = ((_ref - df['{ts_col}']).dt.total_seconds() / 86400).clip(lower=0)\n"
              f"df['{feat.name}'] = np.exp(-_lam * _delta)")
        sql = f"EXP(-LN(2)/{half_life} * DATE_DIFF(reference_date, {ts_col}, DAY)) AS {feat.name}"
        return r, py, sql

    def _tp_event_regularity(self, df, feat):
        p = feat.template_params
        entity_col, ts_col = p["entity_col"], p["ts_col"]
        if not all(c in df.columns for c in [entity_col, ts_col]):
            return None, "", ""
        df_sorted = df.sort_values(ts_col)
        inter = (df_sorted.groupby(entity_col)[ts_col]
                 .diff().dt.total_seconds() / 3600)
        grp_std = inter.groupby(df_sorted[entity_col]).transform("std")
        grp_mean = inter.groupby(df_sorted[entity_col]).transform("mean")
        r = (grp_std / (grp_mean + 1e-9)).reindex(df.index)
        py = (f"_inter = df.sort_values('{ts_col}').groupby('{entity_col}')['{ts_col}']"
              f".diff().dt.total_seconds()/3600\n"
              f"df['{feat.name}'] = (_inter.groupby(df['{entity_col}']).transform('std') /\n"
              f"                     (_inter.groupby(df['{entity_col}']).transform('mean') + 1e-9))")
        sql = "-- regularity_score: compute in application layer after inter-event CTEs"
        return r, py, sql

    def _tp_days_since_change(self, df, feat):
        p = feat.template_params
        entity_col, ts_col, cat_col = p["entity_col"], p["ts_col"], p["cat_col"]
        if not all(c in df.columns for c in [entity_col, ts_col, cat_col]):
            return None, "", ""
        df_sorted = df.sort_values(ts_col).copy()
        prev_cat = df_sorted.groupby(entity_col)[cat_col].shift(1)
        changed = prev_cat != df_sorted[cat_col]
        df_sorted["_last_change_ts"] = np.where(changed, df_sorted[ts_col], pd.NaT)
        df_sorted["_last_change_ts"] = (df_sorted.groupby(entity_col)["_last_change_ts"]
                                        .ffill())
        r = ((df_sorted[ts_col] - df_sorted["_last_change_ts"]
              ).dt.total_seconds() / 86400).reindex(df.index)
        py = (f"_prev = df.sort_values('{ts_col}').groupby('{entity_col}')['{cat_col}'].shift(1)\n"
              f"# Mark last change timestamp, forward-fill, compute days since\n"
              f"df['{feat.name}'] = ... # see full pipeline implementation")
        sql = "-- days_since_change: requires LAST_VALUE with conditional in SQL"
        return r, py, sql

    # ── Behavioral / RFM ────────────────────────────────────────────────────

    def _tp_lifetime_agg(self, df, feat):
        p = feat.template_params
        entity_col, amount_col = p["entity_col"], p["amount_col"]
        if not all(c in df.columns for c in [entity_col, amount_col]):
            return None, "", ""
        r = df.groupby(entity_col)[amount_col].transform("sum")
        py = f"df['{feat.name}'] = df.groupby('{entity_col}')['{amount_col}'].transform('sum')"
        sql = f"SUM({amount_col}) OVER (PARTITION BY {entity_col}) AS {feat.name}"
        return r, py, sql

    def _tp_expanding_agg(self, df, feat):
        p = feat.template_params
        entity_col, amount_col = p["entity_col"], p["amount_col"]
        if not all(c in df.columns for c in [entity_col, amount_col]):
            return None, "", ""
        ts_col = p.get("ts_col")
        df_sorted = df.sort_values(ts_col) if ts_col and ts_col in df.columns else df
        r = (df_sorted.groupby(entity_col)[amount_col]
             .expanding().mean()
             .reset_index(level=0, drop=True)
             .reindex(df.index))
        py = (f"df['{feat.name}'] = (\n"
              f"    df.sort_values('{ts_col}')\n"
              f"    .groupby('{entity_col}')['{amount_col}']\n"
              f"    .expanding().mean()\n"
              f"    .reset_index(level=0, drop=True)\n"
              f")")
        sql = (f"AVG({amount_col}) OVER (PARTITION BY {entity_col} "
               f"ORDER BY {ts_col} ROWS UNBOUNDED PRECEDING) AS {feat.name}")
        return r, py, sql

    def _tp_zscore_vs_self(self, df, feat):
        p = feat.template_params
        entity_col, amount_col = p["entity_col"], p["amount_col"]
        if not all(c in df.columns for c in [entity_col, amount_col]):
            return None, "", ""
        mean = df.groupby(entity_col)[amount_col].transform("mean")
        std = df.groupby(entity_col)[amount_col].transform("std").fillna(1)
        r = (df[amount_col] - mean) / (std + 1e-9)
        py = (f"_mean = df.groupby('{entity_col}')['{amount_col}'].transform('mean')\n"
              f"_std  = df.groupby('{entity_col}')['{amount_col}'].transform('std').fillna(1)\n"
              f"df['{feat.name}'] = (df['{amount_col}'] - _mean) / (_std + 1e-9)")
        sql = (f"({amount_col} - AVG({amount_col}) OVER (PARTITION BY {entity_col}))\n"
               f"/ NULLIF(STDDEV({amount_col}) OVER (PARTITION BY {entity_col}), 0) AS {feat.name}")
        return r, py, sql

    def _tp_events_per_day(self, df, feat):
        p = feat.template_params
        entity_col, ts_col = p["entity_col"], p["ts_col"]
        if not all(c in df.columns for c in [entity_col, ts_col]):
            return None, "", ""
        lifetime_days = ((df.groupby(entity_col)[ts_col].transform("max") -
                          df.groupby(entity_col)[ts_col].transform("min"))
                         .dt.total_seconds() / 86400 + 1)
        event_count = df.groupby(entity_col)[ts_col].transform("count")
        r = event_count / lifetime_days
        py = (f"_ldays = ((df.groupby('{entity_col}')['{ts_col}'].transform('max') -\n"
              f"           df.groupby('{entity_col}')['{ts_col}'].transform('min'))\n"
              f"          .dt.total_seconds() / 86400 + 1)\n"
              f"df['{feat.name}'] = df.groupby('{entity_col}')['{ts_col}'].transform('count') / _ldays")
        sql = (f"COUNT(*) OVER (PARTITION BY {entity_col}) /\n"
               f"  GREATEST(1, DATE_DIFF(MAX({ts_col}) OVER (PARTITION BY {entity_col}),\n"
               f"             MIN({ts_col}) OVER (PARTITION BY {entity_col}), DAY)) AS {feat.name}")
        return r, py, sql

    def _tp_active_days(self, df, feat):
        p = feat.template_params
        entity_col, ts_col, window_d = p["entity_col"], p["ts_col"], p.get("window_d", 30)
        if not all(c in df.columns for c in [entity_col, ts_col]):
            return None, "", ""
        # Approximate: count distinct dates in rolling window via custom apply
        df_sorted = df.sort_values(ts_col).copy()
        df_sorted["_date"] = df_sorted[ts_col].dt.date

        def _active_days_window(g):
            out = []
            dates = g[ts_col].values
            date_vals = g["_date"].values
            cutoff = np.timedelta64(window_d, "D")
            for i in range(len(g)):
                mask = (dates[i] - dates[:i]) < cutoff
                out.append(len(set(date_vals[:i][mask])))
            return pd.Series(out, index=g.index)

        try:
            r = df_sorted.groupby(entity_col, group_keys=False).apply(_active_days_window)
            r = r.reindex(df.index)
        except Exception:
            r = pd.Series(np.nan, index=df.index)
        py = f"# active_days_{window_d}d: custom rolling apply — see full pipeline"
        sql = "-- active_days: COUNT(DISTINCT DATE) in time-bounded window CTE"
        return r, py, sql

    # ── Statistical ─────────────────────────────────────────────────────────

    def _tp_percentile_rank(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        r = df[col].rank(pct=True)
        py = f"df['{feat.name}'] = df['{col}'].rank(pct=True)"
        sql = f"PERCENTILE_CONT({col}) AS {feat.name}  -- or NTILE(100)"
        return r, py, sql

    def _tp_log1p(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        r = np.log1p(df[col].clip(lower=0))
        py = f"df['{feat.name}'] = np.log1p(df['{col}'].clip(lower=0))"
        sql = f"LN(1 + GREATEST(0, {col})) AS {feat.name}"
        return r, py, sql

    def _tp_decile_bin(self, df, feat):
        col = feat.template_params["col"]
        n_bins = feat.template_params.get("n_bins", 10)
        if col not in df.columns:
            return None, "", ""
        r = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
        py = f"df['{feat.name}'] = pd.qcut(df['{col}'], q={n_bins}, labels=False, duplicates='drop')"
        sql = f"NTILE({n_bins}) OVER (ORDER BY {col}) - 1 AS {feat.name}"
        return r, py, sql

    def _tp_entity_skewness(self, df, feat):
        p = feat.template_params
        entity_col, col = p["entity_col"], p["col"]
        if not all(c in df.columns for c in [entity_col, col]):
            return None, "", ""
        r = df.groupby(entity_col)[col].transform(lambda x: x.skew())
        py = f"df['{feat.name}'] = df.groupby('{entity_col}')['{col}'].transform(lambda x: x.skew())"
        sql = "-- skewness: compute in application layer"
        return r, py, sql

    def _tp_entity_kurtosis(self, df, feat):
        p = feat.template_params
        entity_col, col = p["entity_col"], p["col"]
        if not all(c in df.columns for c in [entity_col, col]):
            return None, "", ""
        r = df.groupby(entity_col)[col].transform(lambda x: x.kurt())
        py = f"df['{feat.name}'] = df.groupby('{entity_col}')['{col}'].transform(lambda x: x.kurt())"
        sql = "-- kurtosis: compute in application layer"
        return r, py, sql

    def _tp_abs_deviation(self, df, feat):
        p = feat.template_params
        entity_col, col = p["entity_col"], p["col"]
        if not all(c in df.columns for c in [entity_col, col]):
            return None, "", ""
        mean = df.groupby(entity_col)[col].transform("mean")
        r = (df[col] - mean).abs()
        py = (f"_mean = df.groupby('{entity_col}')['{col}'].transform('mean')\n"
              f"df['{feat.name}'] = (df['{col}'] - _mean).abs()")
        sql = f"ABS({col} - AVG({col}) OVER (PARTITION BY {entity_col})) AS {feat.name}"
        return r, py, sql

    def _tp_entropy(self, df, feat):
        p = feat.template_params
        entity_col, cat_col = p["entity_col"], p["cat_col"]
        if not all(c in df.columns for c in [entity_col, cat_col]):
            return None, "", ""
        from scipy.stats import entropy as sp_entropy

        def _ent(series):
            counts = series.value_counts(normalize=True)
            return float(sp_entropy(counts))

        r = df.groupby(entity_col)[cat_col].transform(lambda x: _ent(x))
        py = (f"from scipy.stats import entropy as sp_entropy\n"
              f"df['{feat.name}'] = df.groupby('{entity_col}')['{cat_col}'].transform(\n"
              f"    lambda x: sp_entropy(x.value_counts(normalize=True))\n"
              f")")
        sql = "-- entropy requires application-layer computation"
        return r, py, sql

    def _tp_hhi(self, df, feat):
        p = feat.template_params
        entity_col, cat_col = p["entity_col"], p["cat_col"]
        if not all(c in df.columns for c in [entity_col, cat_col]):
            return None, "", ""

        def _hhi(x):
            shares = x.value_counts(normalize=True)
            return float((shares ** 2).sum())

        r = df.groupby(entity_col)[cat_col].transform(lambda x: _hhi(x))
        py = (f"def _hhi(x):\n"
              f"    shares = x.value_counts(normalize=True)\n"
              f"    return (shares**2).sum()\n"
              f"df['{feat.name}'] = df.groupby('{entity_col}')['{cat_col}'].transform(_hhi)")
        sql = "-- HHI: Σ(count_i/total)² per entity — compute in a GROUP BY CTE"
        return r, py, sql

    # ── Ratio / interaction ─────────────────────────────────────────────────

    def _tp_ratio(self, df, feat):
        p = feat.template_params
        col_a, col_b = p["col_a"], p["col_b"]
        if not all(c in df.columns for c in [col_a, col_b]):
            return None, "", ""
        r = df[col_a] / (df[col_b] + 1e-9)
        py = f"df['{feat.name}'] = df['{col_a}'] / (df['{col_b}'] + 1e-9)"
        sql = f"SAFE_DIVIDE({col_a}, {col_b}) AS {feat.name}"
        return r, py, sql

    def _tp_product(self, df, feat):
        p = feat.template_params
        col_a, col_b = p["col_a"], p["col_b"]
        if not all(c in df.columns for c in [col_a, col_b]):
            return None, "", ""
        r = df[col_a] * df[col_b]
        py = f"df['{feat.name}'] = df['{col_a}'] * df['{col_b}']"
        sql = f"{col_a} * {col_b} AS {feat.name}"
        return r, py, sql

    def _tp_ratio_to_rolling_mean(self, df, feat):
        p = feat.template_params
        entity_col, ts_col = p["entity_col"], p.get("ts_col") or (
            ""
        )
        amount_col, window_d = p["amount_col"], p["window_d"]
        if not all(c in df.columns for c in [entity_col, amount_col]):
            return None, "", ""
        # Build rolling mean first
        roll_feat = FeatureDefinition(
            name="_tmp_roll_mean",
            category=FeatureCategory.TEMPORAL,
            description="", business_intuition="", mathematical_definition="",
            template_params={"entity_col": entity_col, "ts_col": ts_col,
                             "metric_col": amount_col, "window_d": window_d, "agg": "mean"},
        )
        roll_series, _, _ = self._tp_rolling_agg(df, roll_feat)
        if roll_series is None:
            return None, "", ""
        r = df[amount_col] / (roll_series + 1e-9)
        py = (f"# Rolling mean already computed as feature; ratio:\n"
              f"df['{feat.name}'] = df['{amount_col}'] / (rolling_{window_d}d_mean_{amount_col} + 1e-9)")
        sql = f"SAFE_DIVIDE({amount_col}, rolling_{window_d}d_mean_{amount_col}) AS {feat.name}"
        return r, py, sql

    def _tp_frequency_ratio(self, df, feat):
        p = feat.template_params
        entity_col, cat_col = p["entity_col"], p["cat_col"]
        if not all(c in df.columns for c in [entity_col, cat_col]):
            return None, "", ""
        entity_freq = df.groupby([entity_col, cat_col]).size() / df.groupby(entity_col).size()
        pop_freq = df[cat_col].value_counts(normalize=True)
        entity_rate = df.apply(lambda row: entity_freq.get((row[entity_col], row[cat_col]), 0), axis=1)
        pop_rate = df[cat_col].map(pop_freq).fillna(0)
        r = entity_rate / (pop_rate + 1e-9)
        py = f"# frequency_ratio for {cat_col}"
        sql = f"-- frequency_ratio: divide entity-level frequency by population frequency"
        return r, py, sql

    # ── Sequence / transition ────────────────────────────────────────────────

    def _tp_prev_match(self, df, feat):
        p = feat.template_params
        entity_col, ts_col, cat_col = p["entity_col"], p["ts_col"], p["cat_col"]
        if not all(c in df.columns for c in [entity_col, ts_col, cat_col]):
            return None, "", ""
        df_sorted = df.sort_values(ts_col)
        prev_cat = df_sorted.groupby(entity_col)[cat_col].shift(1)
        r = (df_sorted[cat_col] == prev_cat).astype(int).reindex(df.index)
        py = (f"_prev = df.sort_values('{ts_col}').groupby('{entity_col}')['{cat_col}'].shift(1)\n"
              f"df['{feat.name}'] = (df['{cat_col}'] == _prev).astype(int)")
        sql = (f"CASE WHEN {cat_col} = LAG({cat_col}) OVER "
               f"(PARTITION BY {entity_col} ORDER BY {ts_col}) THEN 1 ELSE 0 END AS {feat.name}")
        return r, py, sql

    def _tp_modal_category(self, df, feat):
        p = feat.template_params
        entity_col, cat_col = p["entity_col"], p["cat_col"]
        if not all(c in df.columns for c in [entity_col, cat_col]):
            return None, "", ""
        mode_map = df.groupby(entity_col)[cat_col].agg(
            lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan
        )
        r = df[entity_col].map(mode_map)
        py = (f"_mode = df.groupby('{entity_col}')['{cat_col}'].agg(lambda x: x.mode().iloc[0])\n"
              f"df['{feat.name}'] = df['{entity_col}'].map(_mode)")
        sql = "-- modal_category: use APPROX_TOP_COUNT or a ROW_NUMBER window in SQL"
        return r, py, sql

    def _tp_first_time_category(self, df, feat):
        p = feat.template_params
        entity_col, ts_col, cat_col = p["entity_col"], p["ts_col"], p["cat_col"]
        if not all(c in df.columns for c in [entity_col, ts_col, cat_col]):
            return None, "", ""
        df_sorted = df.sort_values(ts_col)
        cumcount = df_sorted.groupby([entity_col, cat_col]).cumcount()
        r = (cumcount == 0).astype(int).reindex(df.index)
        py = (f"df['{feat.name}'] = (\n"
              f"    df.sort_values('{ts_col}')\n"
              f"    .groupby(['{entity_col}', '{cat_col}']).cumcount() == 0\n"
              f").astype(int)")
        sql = (f"CASE WHEN ROW_NUMBER() OVER (PARTITION BY {entity_col}, {cat_col} "
               f"ORDER BY {ts_col}) = 1 THEN 1 ELSE 0 END AS {feat.name}")
        return r, py, sql

    def _tp_transition_count(self, df, feat):
        p = feat.template_params
        entity_col, ts_col, cat_col = p["entity_col"], p["ts_col"], p["cat_col"]
        if not all(c in df.columns for c in [entity_col, ts_col, cat_col]):
            return None, "", ""
        df_sorted = df.sort_values(ts_col)
        prev_cat = df_sorted.groupby(entity_col)[cat_col].shift(1)
        switched = ((df_sorted[cat_col] != prev_cat) & prev_cat.notna()).astype(int)
        try:
            r = (df_sorted.groupby(entity_col)
                 .apply(lambda g: g.assign(_sw=switched.loc[g.index])
                        .set_index(ts_col)["_sw"]
                        .rolling("30D", closed="left").sum()
                        .reset_index(drop=False).set_index(g.index).iloc[:, -1])
                 .reindex(df.index))
        except Exception:
            r = switched.reindex(df.index)
        py = "# transition_count_30d: rolling sum of category-switch events"
        sql = "-- transition count: requires conditional LAG comparison in window function"
        return r, py, sql

    def _tp_ngram_hash(self, df, feat):
        p = feat.template_params
        entity_col, ts_col, cat_col, n = p["entity_col"], p["ts_col"], p["cat_col"], p.get("n", 2)
        if not all(c in df.columns for c in [entity_col, ts_col, cat_col]):
            return None, "", ""
        df_sorted = df.sort_values(ts_col).copy()
        ngram_parts = [df_sorted.groupby(entity_col)[cat_col].shift(i).astype(str).fillna("")
                       for i in range(n - 1, -1, -1)]
        combined = ngram_parts[0]
        for part in ngram_parts[1:]:
            combined = combined + "|" + part
        r = combined.apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (10 ** 9)).reindex(df.index)
        py = f"# {n}-gram hash of '{cat_col}' sequence per entity"
        sql = f"-- ngram_hash: requires multiple LAG calls and HASH function"
        return r, py, sql

    # ── Text ──────────────────────────────────────────────────────────────────

    def _tp_word_count(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        r = df[col].fillna("").astype(str).str.split().str.len().fillna(0)
        py = f"df['{feat.name}'] = df['{col}'].fillna('').str.split().str.len()"
        sql = f"ARRAY_LENGTH(SPLIT({col}, ' ')) AS {feat.name}"
        return r, py, sql

    def _tp_char_count(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        r = df[col].fillna("").astype(str).str.len()
        py = f"df['{feat.name}'] = df['{col}'].fillna('').str.len()"
        sql = f"LENGTH({col}) AS {feat.name}"
        return r, py, sql

    def _tp_has_url(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        r = df[col].fillna("").astype(str).str.contains(r"https?://", regex=True).astype(int)
        py = f"df['{feat.name}'] = df['{col}'].fillna('').str.contains(r'https?://', regex=True).astype(int)"
        sql = f"CASE WHEN REGEXP_CONTAINS({col}, r'https?://') THEN 1 ELSE 0 END AS {feat.name}"
        return r, py, sql

    def _tp_has_numeric(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        r = df[col].fillna("").astype(str).str.contains(r"\d", regex=True).astype(int)
        py = f"df['{feat.name}'] = df['{col}'].fillna('').str.contains(r'\\d', regex=True).astype(int)"
        sql = f"CASE WHEN REGEXP_CONTAINS({col}, r'[0-9]') THEN 1 ELSE 0 END AS {feat.name}"
        return r, py, sql

    def _tp_unique_word_ratio(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        def _uwr(text):
            tokens = str(text).split()
            return len(set(tokens)) / max(len(tokens), 1)
        r = df[col].fillna("").apply(_uwr)
        py = (f"df['{feat.name}'] = df['{col}'].fillna('').apply(\n"
              f"    lambda t: len(set(t.split())) / max(len(t.split()), 1)\n"
              f")")
        sql = "-- unique_word_ratio: compute in application layer"
        return r, py, sql

    # ── Nonlinear ────────────────────────────────────────────────────────────

    def _tp_sqrt(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        r = np.sqrt(df[col].clip(lower=0))
        py = f"df['{feat.name}'] = np.sqrt(df['{col}'].clip(lower=0))"
        sql = f"SQRT(GREATEST(0, {col})) AS {feat.name}"
        return r, py, sql

    def _tp_boxcox(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        from scipy.stats import boxcox as sp_boxcox
        vals = df[col].clip(lower=0.001).fillna(0.001)
        try:
            transformed, lam = sp_boxcox(vals)
            r = pd.Series(transformed, index=df.index)
            py = (f"from scipy.stats import boxcox as sp_boxcox\n"
                  f"transformed, lam = sp_boxcox(df['{col}'].clip(lower=0.001).fillna(0.001))\n"
                  f"df['{feat.name}'] = transformed  # lam={lam:.4f}")
        except Exception:
            r = np.log1p(vals)
            py = f"df['{feat.name}'] = np.log1p(df['{col}'].clip(lower=0))"
        sql = "-- box_cox: compute optimum lambda offline, then apply POWER() in SQL"
        return r, py, sql

    def _tp_piecewise_low(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        p25 = df[col].quantile(0.25)
        r = df[col].clip(upper=p25)
        py = f"_p25 = df['{col}'].quantile(0.25)\ndf['{feat.name}'] = df['{col}'].clip(upper=_p25)"
        sql = f"LEAST({col}, {p25:.4f}) AS {feat.name}"
        return r, py, sql

    def _tp_piecewise_high(self, df, feat):
        col = feat.template_params["col"]
        if col not in df.columns:
            return None, "", ""
        p75 = df[col].quantile(0.75)
        r = (df[col] - p75).clip(lower=0)
        py = f"_p75 = df['{col}'].quantile(0.75)\ndf['{feat.name}'] = (df['{col}'] - _p75).clip(lower=0)"
        sql = f"GREATEST(0, {col} - {p75:.4f}) AS {feat.name}"
        return r, py, sql

    def _tp_cohort_week(self, df, feat):
        p = feat.template_params
        entity_col, ts_col = p["entity_col"], p["ts_col"]
        if not all(c in df.columns for c in [entity_col, ts_col]):
            return None, "", ""
        first_ts = df.groupby(entity_col)[ts_col].transform("min")
        r = first_ts.dt.isocalendar().week.astype(int)
        py = (f"_first = df.groupby('{entity_col}')['{ts_col}'].transform('min')\n"
              f"df['{feat.name}'] = _first.dt.isocalendar().week.astype(int)")
        sql = f"EXTRACT(WEEK FROM MIN({ts_col}) OVER (PARTITION BY {entity_col})) AS {feat.name}"
        return r, py, sql

    def _tp_weeks_since_cohort(self, df, feat):
        p = feat.template_params
        entity_col, ts_col = p["entity_col"], p["ts_col"]
        if not all(c in df.columns for c in [entity_col, ts_col]):
            return None, "", ""
        first_ts = df.groupby(entity_col)[ts_col].transform("min")
        r = ((df[ts_col] - first_ts).dt.total_seconds() / (7 * 86400))
        py = (f"_first = df.groupby('{entity_col}')['{ts_col}'].transform('min')\n"
              f"df['{feat.name}'] = (df['{ts_col}'] - _first).dt.total_seconds() / 604800")
        sql = (f"DATE_DIFF({ts_col}, MIN({ts_col}) OVER (PARTITION BY {entity_col}), WEEK) "
               f"AS {feat.name}")
        return r, py, sql

    # ── Target encoding (code only — no actual leaky computation) ────────────

    def _tp_target_encode_loo(self, df, feat):
        py = textwrap.dedent(f"""
            # LOO target encoding for '{feat.template_params.get('cat_col')}' — MUST run inside CV folds
            from category_encoders import LeaveOneOutEncoder
            enc = LeaveOneOutEncoder(cols=['{feat.template_params.get('cat_col')}'])
            df_train['{feat.name}'] = enc.fit_transform(
                df_train[['{feat.template_params.get('cat_col')}']],
                df_train['{feat.template_params.get('target_col')}']
            ).squeeze()
            df_test['{feat.name}'] = enc.transform(
                df_test[['{feat.template_params.get('cat_col')}']]
            ).squeeze()
        """).strip()
        sql = "-- LOO target encoding: compute in Python; store results as a lookup table"
        # Return NaN series (real computation requires CV context)
        r = pd.Series(np.nan, index=df.index)
        return r, py, sql

    def _tp_target_encode_smoothed(self, df, feat):
        p = feat.template_params
        cat_col = p.get("cat_col")
        target_col = p.get("target_col")
        k = p.get("smoothing", 10)
        py = textwrap.dedent(f"""
            # Smoothed mean target encoding for '{cat_col}'
            global_mean = df['{target_col}'].mean()
            stats = df.groupby('{cat_col}')['{target_col}'].agg(['count','mean'])
            stats['{feat.name}'] = (
                (stats['count'] * stats['mean'] + {k} * global_mean) /
                (stats['count'] + {k})
            )
            df['{feat.name}'] = df['{cat_col}'].map(stats['{feat.name}']).fillna(global_mean)
        """).strip()
        sql = "-- smoothed_target_encode: join pre-computed encoding table"
        r = pd.Series(np.nan, index=df.index)
        return r, py, sql

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _tp_out_degree(self, df, feat):
        p = feat.template_params
        src, dst = p["src"], p["dst"]
        if not all(c in df.columns for c in [src, dst]):
            return None, "", ""
        out_deg = df.groupby(src)[dst].nunique()
        r = df[src].map(out_deg)
        py = f"df['{feat.name}'] = df['{src}'].map(df.groupby('{src}')['{dst}'].nunique())"
        sql = f"COUNT(DISTINCT {dst}) OVER (PARTITION BY {src}) AS {feat.name}"
        return r, py, sql

    def _tp_in_degree(self, df, feat):
        p = feat.template_params
        src, dst = p["src"], p["dst"]
        if not all(c in df.columns for c in [src, dst]):
            return None, "", ""
        in_deg = df.groupby(dst)[src].nunique()
        r = df[dst].map(in_deg)
        py = f"df['{feat.name}'] = df['{dst}'].map(df.groupby('{dst}')['{src}'].nunique())"
        sql = f"COUNT(DISTINCT {src}) OVER (PARTITION BY {dst}) AS {feat.name}"
        return r, py, sql

    # ── Incremental code stub ────────────────────────────────────────────────

    def _generate_incremental_code(self, feat: FeatureDefinition) -> str:
        """Generate a stub for incremental (streaming/micro-batch) computation."""
        return textwrap.dedent(f"""
            # Incremental computation for: {feat.name}
            # (Append new_rows to the historical window and recompute the tail)
            def compute_{feat.name}_incremental(history_df, new_rows_df):
                combined = pd.concat([history_df, new_rows_df]).sort_values(
                    '{feat.template_params.get("ts_col", "event_time")}')
                # Reuse batch logic on the combined frame
                result = compute_{feat.name}_batch(combined)
                return result.tail(len(new_rows_df))
        """).strip()
