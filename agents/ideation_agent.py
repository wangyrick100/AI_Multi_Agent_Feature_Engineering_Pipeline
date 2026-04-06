"""
Feature Ideation Agent  ★ CRITICAL COMPONENT ★
────────────────────────────────────────────────
Generates a large, diverse, and high-quality catalogue of candidate features
organised across 10 feature categories.  For a typical transactional dataset
this produces 120-200+ distinct feature definitions before any construction.

Design principle:  each logical *feature template* is parameterised by the
actual columns detected in the SchemaAnalysis, so the output is always
dataset-specific, not generic.

Categories generated
────────────────────
1.  Temporal / Calendar                 (cyclic encodings, seasonality, proximity)
2.  Lag / Lead                          (autoregressive memory)
3.  Rolling Window Aggregations         (statistical summaries over time windows)
4.  Recency / Freshness                 (time-since-last, time-to-next)
5.  Behavioral / RFM                    (frequency, intensity, monetary patterns)
6.  Statistical / Distributional        (entropy, z-score, percentile, skew)
7.  Ratio & Interaction                 (cross-feature signals)
8.  Sequence / Transition               (pattern memory, HHI, diversity)
9.  Graph / Network                     (entity centrality, shared neighbours)
10. Semantic / Text (LLM-powered)       (sentiment, topics, keyword indicators)
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.base_agent import BaseAgent
from core.config import Config
from core.data_models import (
    FeatureCategory,
    FeatureDefinition,
    LeakageRisk,
    PipelineState,
    PredictiveValue,
    SchemaAnalysis,
    SemanticType,
)


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def _fdef(
    name: str,
    category: FeatureCategory,
    description: str,
    business_intuition: str,
    math_def: str,
    predictive_value: PredictiveValue = PredictiveValue.MEDIUM,
    leakage_risk: LeakageRisk = LeakageRisk.NONE,
    source_columns: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    requires_entity_grouping: bool = False,
    requires_time_ordering: bool = False,
    requires_target: bool = False,
    template_params: Optional[Dict[str, Any]] = None,
    display_name: str = "",
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        display_name=display_name or name,
        category=category,
        description=description,
        business_intuition=business_intuition,
        mathematical_definition=math_def,
        expected_predictive_value=predictive_value,
        leakage_risk=leakage_risk,
        source_columns=source_columns or [],
        tags=tags or [],
        requires_entity_grouping=requires_entity_grouping,
        requires_time_ordering=requires_time_ordering,
        requires_target=requires_target,
        template_params=template_params or {},
    )


# ────────────────────────────────────────────────────────────────────────────
#  Agent
# ────────────────────────────────────────────────────────────────────────────

class FeatureIdeationAgent(BaseAgent):
    """
    Agent 2 — Feature Ideation

    Produces the full candidate feature catalogue given the SchemaAnalysis
    from Agent 1.  Code generation is intentionally *not* performed here;
    each FeatureDefinition carries its template_params which the Construction
    Agent uses to emit actual Python / SQL.
    """

    name = "FeatureIdeationAgent"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.windows = config.rolling_windows_days
        self.lags = config.lag_periods

    # ── Main logic ──────────────────────────────────────────────────────────

    def _run(self, df: pd.DataFrame, state: PipelineState) -> PipelineState:
        schema = state.schema_analysis
        if schema is None:
            raise ValueError("SchemaUnderstandingAgent must run before FeatureIdeationAgent.")

        self.logger.info("  Generating feature candidates …")

        all_features: List[FeatureDefinition] = []

        # Apply each template group; generators return lists to avoid dupes
        all_features += self._gen_temporal_calendar(schema)
        all_features += self._gen_lag_lead(schema)
        all_features += self._gen_rolling_window_aggregations(schema)
        all_features += self._gen_recency_freshness(schema)
        all_features += self._gen_behavioral_rfm(schema)
        all_features += self._gen_statistical_distributional(schema)
        all_features += self._gen_ratio_interaction(schema)
        all_features += self._gen_sequence_transition(schema)
        all_features += self._gen_graph_network(schema)
        all_features += self._gen_semantic_text(schema)
        all_features += self._gen_advanced_nonlinear(schema)
        all_features += self._gen_target_encoding_proxies(schema)

        # De-duplicate by name
        seen: set = set()
        unique_features: List[FeatureDefinition] = []
        for f in all_features:
            if f.name not in seen:
                seen.add(f.name)
                unique_features.append(f)

        state.feature_candidates = unique_features
        self.logger.info("  Total candidate features generated: %d", len(unique_features))

        # Breakdown by category
        from collections import Counter
        counts = Counter(f.category.value for f in unique_features)
        for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
            self.logger.info("    %-30s %d", cat, n)

        return state

    # ────────────────────────────────────────────────────────────────────────
    #  1. TEMPORAL / CALENDAR
    # ────────────────────────────────────────────────────────────────────────

    def _gen_temporal_calendar(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        for ts_col in schema.temporal_columns:

            features.append(_fdef(
                name=f"{ts_col}__hour_of_day",
                category=FeatureCategory.TEMPORAL,
                description=f"Hour extracted from {ts_col} (0-23).",
                business_intuition="Purchase/event times cluster by hour; morning vs. evening behaviour differs dramatically.",
                math_def=f"hour({ts_col})",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[ts_col],
                tags=["calendar", "cyclic"],
                template_params={"col": ts_col, "part": "hour"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__hour_sin",
                category=FeatureCategory.TEMPORAL,
                description=f"Sine encoding of hour-of-day from {ts_col} (cyclic).",
                business_intuition="Captures circular daily seasonality without ordinal bias.",
                math_def=f"sin(2π * hour({ts_col}) / 24)",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[ts_col],
                tags=["calendar", "cyclic", "sine"],
                template_params={"col": ts_col, "part": "hour_sin"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__hour_cos",
                category=FeatureCategory.TEMPORAL,
                description=f"Cosine encoding of hour-of-day from {ts_col} (cyclic).",
                business_intuition="Pairs with sin to preserve cyclic distance between midnight and 23:00.",
                math_def=f"cos(2π * hour({ts_col}) / 24)",
                source_columns=[ts_col],
                tags=["calendar", "cyclic", "cosine"],
                template_params={"col": ts_col, "part": "hour_cos"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__day_of_week",
                category=FeatureCategory.TEMPORAL,
                description=f"Day-of-week (0=Monday … 6=Sunday) from {ts_col}.",
                business_intuition="Weekly rhythm dominates many consumer behaviours; weekends vs weekdays flip risk profiles.",
                math_def=f"dayofweek({ts_col})",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[ts_col],
                tags=["calendar", "weekly"],
                template_params={"col": ts_col, "part": "dayofweek"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__dow_sin",
                category=FeatureCategory.TEMPORAL,
                description=f"Sine encoding of day-of-week from {ts_col}.",
                math_def=f"sin(2π * dayofweek({ts_col}) / 7)",
                business_intuition="Preserves cyclic distance: Sunday is adjacent to Monday.",
                source_columns=[ts_col], tags=["calendar", "cyclic"],
                template_params={"col": ts_col, "part": "dow_sin"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__is_weekend",
                category=FeatureCategory.TEMPORAL,
                description=f"Binary flag: 1 if {ts_col} falls on Sat or Sun.",
                business_intuition="Weekend events have distinct risk/engagement profiles across nearly every domain.",
                math_def=f"int(dayofweek({ts_col}) >= 5)",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[ts_col], tags=["calendar", "binary"],
                template_params={"col": ts_col, "part": "is_weekend"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__month",
                category=FeatureCategory.TEMPORAL,
                description=f"Month (1-12) from {ts_col}.",
                business_intuition="Annual seasonality: tax season, holidays, fiscal quarters all affect behaviour.",
                math_def=f"month({ts_col})",
                source_columns=[ts_col], tags=["calendar", "seasonal"],
                template_params={"col": ts_col, "part": "month"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__month_sin",
                category=FeatureCategory.TEMPORAL,
                description=f"Sine encoding of month from {ts_col}.",
                math_def=f"sin(2π * (month({ts_col})-1) / 12)",
                business_intuition="Cyclic monthly seasonality without ordinal artefacts.",
                source_columns=[ts_col], tags=["calendar", "cyclic"],
                template_params={"col": ts_col, "part": "month_sin"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__quarter",
                category=FeatureCategory.TEMPORAL,
                description=f"Fiscal quarter (1-4) from {ts_col}.",
                business_intuition="Quarterly reporting periods cause systematic behaviour shifts in B2B and retail.",
                math_def=f"quarter({ts_col})",
                source_columns=[ts_col], tags=["calendar", "seasonal"],
                template_params={"col": ts_col, "part": "quarter"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__is_quarter_end",
                category=FeatureCategory.TEMPORAL,
                description=f"Flag: last 7 days of a fiscal quarter for {ts_col}.",
                business_intuition="B2B spend spikes at quarter-end (budget use-it-or-lose-it); strong fraud signal.",
                math_def="int(month in {3,6,9,12} and day >= 24)",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[ts_col], tags=["calendar", "business"],
                template_params={"col": ts_col, "part": "is_quarter_end"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__is_business_hour",
                category=FeatureCategory.TEMPORAL,
                description=f"Flag: event at {ts_col} falls within 9am-6pm on weekday.",
                business_intuition="Off-hours activity (late night, weekend) is anomalous for many services.",
                math_def="int(9 <= hour < 18 and dayofweek < 5)",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[ts_col], tags=["calendar", "binary"],
                template_params={"col": ts_col, "part": "is_business_hour"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__days_to_year_end",
                category=FeatureCategory.TEMPORAL,
                description=f"Days remaining until December 31 of the same year.",
                business_intuition="Year-end proximity drives urgency; strong seasonal signal in insurance and finance.",
                math_def="(date(year,12,31) - date({ts_col})).days",
                source_columns=[ts_col], tags=["calendar", "proximity"],
                template_params={"col": ts_col, "part": "days_to_year_end"},
            ))

            features.append(_fdef(
                name=f"{ts_col}__week_of_year",
                category=FeatureCategory.TEMPORAL,
                description=f"ISO week-of-year (1-53) from {ts_col}.",
                math_def=f"isocalendar({ts_col}).week",
                business_intuition="Captures bi-weekly pay cycles, promotional cadence, and seasonal peaks.",
                source_columns=[ts_col], tags=["calendar"],
                template_params={"col": ts_col, "part": "week_of_year"},
            ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  2. LAG / LEAD (per entity, time-ordered)
    # ────────────────────────────────────────────────────────────────────────

    def _gen_lag_lead(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        if not schema.entity_columns or not schema.temporal_columns:
            return features

        entity_col = schema.entity_columns[0]
        ts_col = schema.temporal_columns[0]

        for metric_col in schema.numeric_columns[:4]:   # top 4 numeric columns
            for lag in self.lags:
                features.append(_fdef(
                    name=f"{metric_col}__lag_{lag}",
                    category=FeatureCategory.TEMPORAL,
                    description=f"Value of {metric_col} at t-{lag} (previous {lag} events per {entity_col}).",
                    business_intuition=f"Autoregressive signal: prior {metric_col} is highly predictive of next.",
                    math_def=f"groupby({entity_col})[{metric_col}].shift({lag})",
                    predictive_value=PredictiveValue.HIGH,
                    leakage_risk=LeakageRisk.LOW,
                    source_columns=[entity_col, ts_col, metric_col],
                    tags=["lag", "autoregressive"],
                    requires_entity_grouping=True,
                    requires_time_ordering=True,
                    template_params={"entity_col": entity_col, "ts_col": ts_col,
                                     "metric_col": metric_col, "lag": lag},
                ))

                features.append(_fdef(
                    name=f"{metric_col}__lag_{lag}_delta",
                    category=FeatureCategory.TEMPORAL,
                    description=f"Change in {metric_col} from t-{lag} to t.",
                    business_intuition="Momentum / acceleration signal; rate-of-change is often more predictive than level.",
                    math_def=f"{metric_col} - groupby({entity_col})[{metric_col}].shift({lag})",
                    predictive_value=PredictiveValue.HIGH,
                    source_columns=[entity_col, ts_col, metric_col],
                    tags=["lag", "delta", "momentum"],
                    requires_entity_grouping=True,
                    requires_time_ordering=True,
                    template_params={"entity_col": entity_col, "ts_col": ts_col,
                                     "metric_col": metric_col, "lag": lag, "variant": "delta"},
                ))

            # Inter-event time
            features.append(_fdef(
                name=f"{entity_col}__inter_event_time_seconds",
                category=FeatureCategory.TEMPORAL,
                description=f"Seconds elapsed since the previous event for this {entity_col}.",
                business_intuition="Velocity of behaviour; unusually short gaps signal bots/fraud; long gaps indicate dormancy.",
                math_def=f"({ts_col} - groupby({entity_col})[{ts_col}].shift(1)).dt.total_seconds()",
                predictive_value=PredictiveValue.HIGH,
                leakage_risk=LeakageRisk.LOW,
                source_columns=[entity_col, ts_col],
                tags=["inter_event", "velocity"],
                requires_entity_grouping=True,
                requires_time_ordering=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "variant": "inter_event_time"},
            ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  3. ROLLING WINDOW AGGREGATIONS
    # ────────────────────────────────────────────────────────────────────────

    def _gen_rolling_window_aggregations(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        if not schema.entity_columns or not schema.temporal_columns:
            return features

        entity_col = schema.entity_columns[0]
        ts_col = schema.temporal_columns[0]
        agg_funcs = {
            "sum":    ("Sum", PredictiveValue.HIGH),
            "mean":   ("Mean (average)", PredictiveValue.HIGH),
            "std":    ("Standard deviation (volatility)", PredictiveValue.MEDIUM),
            "max":    ("Maximum", PredictiveValue.MEDIUM),
            "min":    ("Minimum", PredictiveValue.LOW),
            "count":  ("Event count", PredictiveValue.HIGH),
            "median": ("Median", PredictiveValue.MEDIUM),
            "nunique":("Distinct-value count (diversity)", PredictiveValue.HIGH),
        }

        _numeric_cols = schema.numeric_columns[:3]
        _cat_cols = schema.categorical_columns[:2]

        for window_d in self.windows:
            for metric_col in _numeric_cols:
                for agg_name, (agg_desc, pv) in agg_funcs.items():
                    features.append(_fdef(
                        name=f"{metric_col}__rolling_{window_d}d_{agg_name}",
                        category=FeatureCategory.TEMPORAL,
                        description=f"{agg_desc} of {metric_col} over trailing {window_d}-day window per {entity_col}.",
                        business_intuition=(
                            f"Trailing {window_d}d {agg_name} of {metric_col} captures "
                            f"{'recent accumulation' if agg_name=='sum' else 'recent variability' if agg_name=='std' else 'recent behaviour'}."
                        ),
                        math_def=(
                            f"groupby({entity_col}).rolling('{window_d}D', on='{ts_col}')['{metric_col}']"
                            f".{agg_name}()"
                        ),
                        predictive_value=pv,
                        leakage_risk=LeakageRisk.LOW,
                        source_columns=[entity_col, ts_col, metric_col],
                        tags=["rolling", f"window_{window_d}d", agg_name],
                        requires_entity_grouping=True,
                        requires_time_ordering=True,
                        template_params={
                            "entity_col": entity_col, "ts_col": ts_col,
                            "metric_col": metric_col, "window_d": window_d,
                            "agg": agg_name,
                        },
                    ))

            # Count of events per entity in window (no specific metric needed)
            features.append(_fdef(
                name=f"{entity_col}__event_count_{window_d}d",
                category=FeatureCategory.BEHAVIORAL,
                description=f"Number of events in trailing {window_d} days for each {entity_col}.",
                business_intuition="Activity frequency is one of the strongest behavioural predictors.",
                math_def=f"groupby({entity_col}).rolling('{window_d}D', on='{ts_col}').count()",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[entity_col, ts_col],
                tags=["rolling", "frequency", f"window_{window_d}d"],
                requires_entity_grouping=True,
                requires_time_ordering=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "window_d": window_d, "agg": "count_events"},
            ))

            # Velocity ratio: short window / long window
            if window_d in (7, 14):
                long_window = window_d * 4
                for metric_col in _numeric_cols[:2]:
                    features.append(_fdef(
                        name=f"{metric_col}__velocity_ratio_{window_d}d_vs_{long_window}d",
                        category=FeatureCategory.BEHAVIORAL,
                        description=(
                            f"Ratio of {window_d}d rolling mean to {long_window}d rolling mean for {metric_col}. "
                            f"Values >1 indicate acceleration."
                        ),
                        business_intuition=(
                            "Velocity changes are leading indicators: rising spend vs. baseline "
                            "signals engagement; falling signals churn risk."
                        ),
                        math_def=(
                            f"rolling_{window_d}d_mean({metric_col}) / "
                            f"(rolling_{long_window}d_mean({metric_col}) + ε)"
                        ),
                        predictive_value=PredictiveValue.VERY_HIGH,
                        source_columns=[entity_col, ts_col, metric_col],
                        tags=["velocity", "ratio", "acceleration"],
                        requires_entity_grouping=True,
                        requires_time_ordering=True,
                        template_params={
                            "entity_col": entity_col, "ts_col": ts_col,
                            "metric_col": metric_col,
                            "short_w": window_d, "long_w": long_window,
                            "variant": "velocity_ratio",
                        },
                    ))

            # Category-count diversity in window
            for cat_col in _cat_cols:
                features.append(_fdef(
                    name=f"{cat_col}__distinct_in_{window_d}d",
                    category=FeatureCategory.BEHAVIORAL,
                    description=f"Number of distinct {cat_col} values per {entity_col} in trailing {window_d} days.",
                    business_intuition="Breadth of category usage is a strong loyalty/churn indicator.",
                    math_def=f"groupby({entity_col}).rolling('{window_d}D')['{cat_col}'].nunique()",
                    predictive_value=PredictiveValue.MEDIUM,
                    source_columns=[entity_col, ts_col, cat_col],
                    tags=["diversity", "rolling", f"window_{window_d}d"],
                    requires_entity_grouping=True,
                    requires_time_ordering=True,
                    template_params={"entity_col": entity_col, "ts_col": ts_col,
                                     "cat_col": cat_col, "window_d": window_d,
                                     "agg": "nunique"},
                ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  4. RECENCY / FRESHNESS
    # ────────────────────────────────────────────────────────────────────────

    def _gen_recency_freshness(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        if not schema.entity_columns or not schema.temporal_columns:
            return features

        entity_col = schema.entity_columns[0]
        ts_col = schema.temporal_columns[0]

        features.append(_fdef(
            name=f"{entity_col}__days_since_last_event",
            category=FeatureCategory.RFM,
            description=f"Days elapsed since the most recent prior event for each {entity_col}.",
            business_intuition="Recency is one of the most powerful churn/engagement predictors (RFM's 'R').",
            math_def=f"(reference_date - max({ts_col}_up_to_now)).days, grouped by {entity_col}",
            predictive_value=PredictiveValue.VERY_HIGH,
            leakage_risk=LeakageRisk.LOW,
            source_columns=[entity_col, ts_col],
            tags=["recency", "rfm"],
            requires_entity_grouping=True,
            requires_time_ordering=True,
            template_params={"entity_col": entity_col, "ts_col": ts_col, "variant": "days_since_last"},
        ))

        features.append(_fdef(
            name=f"{entity_col}__days_since_first_event",
            category=FeatureCategory.RFM,
            description=f"Days between earliest and current event for each {entity_col} (entity age).",
            business_intuition="Older accounts/users are typically lower risk / higher LTV.",
            math_def=f"({ts_col} - min({ts_col})).dt.days, grouped by {entity_col}",
            predictive_value=PredictiveValue.MEDIUM,
            source_columns=[entity_col, ts_col],
            tags=["tenure", "age", "rfm"],
            requires_entity_grouping=True,
            template_params={"entity_col": entity_col, "ts_col": ts_col, "variant": "entity_age"},
        ))

        features.append(_fdef(
            name=f"{entity_col}__time_decay_score_30d",
            category=FeatureCategory.RFM,
            description="Exponentially time-decayed event score with half-life 30d.",
            business_intuition="Recent interactions are more predictive than old ones; decay weights naturally.",
            math_def="Σ exp(-λ · Δt_i),  λ = ln(2)/30, Δt_i = days since event i",
            predictive_value=PredictiveValue.VERY_HIGH,
            leakage_risk=LeakageRisk.LOW,
            source_columns=[entity_col, ts_col],
            tags=["time_decay", "recency"],
            requires_entity_grouping=True,
            requires_time_ordering=True,
            template_params={"entity_col": entity_col, "ts_col": ts_col,
                             "half_life_days": 30, "variant": "time_decay"},
        ))

        features.append(_fdef(
            name=f"{entity_col}__event_regularity_score",
            category=FeatureCategory.BEHAVIORAL,
            description="Coefficient of variation (std/mean) of inter-event times; lower = more regular.",
            business_intuition="Regular users are more predictable; erratic behaviour signals stress or fraud.",
            math_def="std(inter_event_times) / (mean(inter_event_times) + ε), per entity",
            predictive_value=PredictiveValue.HIGH,
            source_columns=[entity_col, ts_col],
            tags=["regularity", "consistency", "behavioral"],
            requires_entity_grouping=True,
            template_params={"entity_col": entity_col, "ts_col": ts_col,
                             "variant": "event_regularity"},
        ))

        for cat_col in schema.categorical_columns[:3]:
            features.append(_fdef(
                name=f"{entity_col}__days_since_last_{cat_col}_change",
                category=FeatureCategory.RFM,
                description=f"Days since the last change in {cat_col} for this {entity_col}.",
                business_intuition=f"State changes in {cat_col} are actionable events; freshness matters.",
                math_def=f"days since ({ts_col} where {cat_col} != prev_{cat_col}), per {entity_col}",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[entity_col, ts_col, cat_col],
                tags=["recency", "state_change"],
                requires_entity_grouping=True,
                requires_time_ordering=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "cat_col": cat_col, "variant": "days_since_change"},
            ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  5. BEHAVIORAL / RFM
    # ────────────────────────────────────────────────────────────────────────

    def _gen_behavioral_rfm(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        if not schema.entity_columns:
            return features

        entity_col = schema.entity_columns[0]
        ts_col = schema.temporal_columns[0] if schema.temporal_columns else None
        amount_cols = [c for c in schema.numeric_columns if "amount" in c.lower()
                       or "price" in c.lower() or "revenue" in c.lower()
                       or "spend" in c.lower() or "value" in c.lower()]
        amount_col = amount_cols[0] if amount_cols else (schema.numeric_columns[0] if schema.numeric_columns else None)

        if amount_col:
            features.append(_fdef(
                name=f"{entity_col}__total_lifetime_amount",
                category=FeatureCategory.RFM,
                description=f"Total cumulative {amount_col} per {entity_col} over all history.",
                business_intuition="Lifetime value (LTV) proxy; high-value customers behave differently.",
                math_def=f"groupby({entity_col})[{amount_col}].transform('sum') up to current t",
                predictive_value=PredictiveValue.VERY_HIGH,
                leakage_risk=LeakageRisk.LOW,
                source_columns=[entity_col, amount_col],
                tags=["ltv", "rfm", "monetary"],
                requires_entity_grouping=True,
                template_params={"entity_col": entity_col, "amount_col": amount_col,
                                 "variant": "lifetime_sum"},
            ))

            features.append(_fdef(
                name=f"{entity_col}__avg_transaction_amount",
                category=FeatureCategory.RFM,
                description=f"Historical average {amount_col} per event for each {entity_col}.",
                business_intuition="Ticket size is a stable behaviour signature; large deviation signals anomaly.",
                math_def=f"groupby({entity_col})[{amount_col}].expanding().mean()",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[entity_col, amount_col],
                tags=["rfm", "monetary", "average"],
                requires_entity_grouping=True,
                template_params={"entity_col": entity_col, "amount_col": amount_col,
                                 "variant": "avg_txn_amount"},
            ))

            features.append(_fdef(
                name=f"{entity_col}__amount_zscore_vs_self",
                category=FeatureCategory.BEHAVIORAL,
                description=f"Z-score of current {amount_col} vs this entity's own historical mean/std.",
                business_intuition="Detects anomalously large/small transactions for this specific entity.",
                math_def=f"({amount_col} - entity_avg) / (entity_std + ε)",
                predictive_value=PredictiveValue.VERY_HIGH,
                leakage_risk=LeakageRisk.LOW,
                source_columns=[entity_col, amount_col],
                tags=["zscore", "anomaly", "behavioral"],
                requires_entity_grouping=True,
                template_params={"entity_col": entity_col, "amount_col": amount_col,
                                 "variant": "zscore_vs_self"},
            ))

            features.append(_fdef(
                name=f"{entity_col}__max_single_amount_7d",
                category=FeatureCategory.BEHAVIORAL,
                description=f"Maximum single {amount_col} in trailing 7 days.",
                business_intuition="One large transaction dwarfs typical behaviour; strong fraud/risk signal.",
                math_def=f"groupby({entity_col}).rolling('7D')[{amount_col}].max()",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[entity_col, amount_col],
                tags=["max", "rolling", "fraud"],
                requires_entity_grouping=True,
                requires_time_ordering=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "amount_col": amount_col, "window_d": 7, "agg": "max",
                                 "variant": "max_single_7d"},
            ))

        if ts_col:
            features.append(_fdef(
                name=f"{entity_col}__events_per_day_lifetime",
                category=FeatureCategory.RFM,
                description=f"Average daily event rate over entire observation window (RFM Frequency).",
                business_intuition="Frequency is the second pillar of RFM; high frequency = high engagement.",
                math_def=f"lifetime_event_count / entity_lifetime_days",
                predictive_value=PredictiveValue.VERY_HIGH,
                source_columns=[entity_col, ts_col],
                tags=["frequency", "rfm"],
                requires_entity_grouping=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "variant": "events_per_day"},
            ))

            features.append(_fdef(
                name=f"{entity_col}__active_days_30d",
                category=FeatureCategory.BEHAVIORAL,
                description="Count of distinct calendar days with at least one event in last 30 days.",
                business_intuition="True activity days (vs. burst events on one day) is a habit indicator.",
                math_def=f"groupby({entity_col}).rolling('30D')[{ts_col}].apply(lambda x: x.dt.date.nunique())",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[entity_col, ts_col],
                tags=["habit", "activity", "rolling"],
                requires_entity_grouping=True,
                requires_time_ordering=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "variant": "active_days", "window_d": 30},
            ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  6. STATISTICAL / DISTRIBUTIONAL
    # ────────────────────────────────────────────────────────────────────────

    def _gen_statistical_distributional(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        entity_col = schema.entity_columns[0] if schema.entity_columns else None
        ts_col = schema.temporal_columns[0] if schema.temporal_columns else None

        for metric_col in schema.numeric_columns[:4]:
            features.append(_fdef(
                name=f"{metric_col}__global_percentile_rank",
                category=FeatureCategory.STATISTICAL,
                description=f"Percentile rank of {metric_col} across the full dataset.",
                business_intuition="Absolute value matters less than relative standing in the population.",
                math_def=f"rank({metric_col}) / n",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[metric_col],
                tags=["percentile", "rank", "normalization"],
                template_params={"col": metric_col, "variant": "percentile_rank"},
            ))

            features.append(_fdef(
                name=f"{metric_col}__log_transform",
                category=FeatureCategory.STATISTICAL,
                description=f"log(1 + {metric_col}) — regularises heavy-tailed monetary distributions.",
                business_intuition="Log-transform compresses outliers; most amount/value cols are log-normal.",
                math_def=f"log(1 + {metric_col})",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[metric_col],
                tags=["transform", "log"],
                template_params={"col": metric_col, "variant": "log1p"},
            ))

            features.append(_fdef(
                name=f"{metric_col}__binned_decile",
                category=FeatureCategory.STATISTICAL,
                description=f"Decile bin (0-9) of {metric_col} using quantile-based binning.",
                business_intuition="Non-linear binning captures threshold effects that linear models miss.",
                math_def=f"pd.qcut({metric_col}, q=10, labels=False, duplicates='drop')",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[metric_col],
                tags=["binning", "discretization"],
                template_params={"col": metric_col, "variant": "decile_bin", "n_bins": 10},
            ))

            if entity_col:
                features.append(_fdef(
                    name=f"{metric_col}__entity_skewness",
                    category=FeatureCategory.STATISTICAL,
                    description=f"Skewness of {metric_col} values per {entity_col} over all history.",
                    business_intuition="Skewed distributions reveal one-off large spikes vs. uniform behaviour.",
                    math_def=f"groupby({entity_col})[{metric_col}].skew()",
                    source_columns=[entity_col, metric_col],
                    tags=["distribution", "skewness"],
                    requires_entity_grouping=True,
                    template_params={"entity_col": entity_col, "col": metric_col,
                                     "variant": "skewness"},
                ))

                features.append(_fdef(
                    name=f"{metric_col}__entity_kurtosis",
                    category=FeatureCategory.STATISTICAL,
                    description=f"Excess kurtosis of {metric_col} per {entity_col}.",
                    business_intuition="High kurtosis = heavy tails = occasional extreme events (risky profile).",
                    math_def=f"groupby({entity_col})[{metric_col}].apply(pd.Series.kurt)",
                    source_columns=[entity_col, metric_col],
                    tags=["distribution", "kurtosis"],
                    requires_entity_grouping=True,
                    template_params={"entity_col": entity_col, "col": metric_col,
                                     "variant": "kurtosis"},
                ))

                features.append(_fdef(
                    name=f"{metric_col}__deviation_from_entity_mean",
                    category=FeatureCategory.STATISTICAL,
                    description=(
                        f"Difference between the current {metric_col} and the entity's "
                        f"own historical mean (absolute deviation)."
                    ),
                    business_intuition="Captures anomalous single events relative to personal baseline.",
                    math_def=f"|{metric_col} - entity_mean({metric_col})|",
                    predictive_value=PredictiveValue.HIGH,
                    leakage_risk=LeakageRisk.LOW,
                    source_columns=[entity_col, metric_col],
                    tags=["anomaly", "deviation"],
                    requires_entity_grouping=True,
                    template_params={"entity_col": entity_col, "col": metric_col,
                                     "variant": "abs_deviation_from_mean"},
                ))

        # Categorical entropy per entity
        for cat_col in schema.categorical_columns[:3]:
            if entity_col:
                features.append(_fdef(
                    name=f"{cat_col}__entropy_per_{entity_col}",
                    category=FeatureCategory.STATISTICAL,
                    description=f"Shannon entropy of {cat_col} distribution for each {entity_col}.",
                    business_intuition=(
                        "Low entropy = specialised behaviour (one dominant category); "
                        "high entropy = diverse/exploratory. Both extremes are meaningful."
                    ),
                    math_def="-Σ p_i · log(p_i)  where p_i = P(cat=i | entity)",
                    predictive_value=PredictiveValue.HIGH,
                    source_columns=[entity_col, cat_col],
                    tags=["entropy", "diversity", "information"],
                    requires_entity_grouping=True,
                    template_params={"entity_col": entity_col, "cat_col": cat_col,
                                     "variant": "entropy"},
                ))

                features.append(_fdef(
                    name=f"{cat_col}__hhi_per_{entity_col}",
                    category=FeatureCategory.STATISTICAL,
                    description=f"Herfindahl-Hirschman Index of {cat_col} concentration per {entity_col}.",
                    business_intuition="HHI measures monopoly-like concentration; 1.0 = uses only one category.",
                    math_def="Σ p_i²  where p_i = share of category i",
                    source_columns=[entity_col, cat_col],
                    tags=["hhi", "concentration", "diversity"],
                    requires_entity_grouping=True,
                    template_params={"entity_col": entity_col, "cat_col": cat_col,
                                     "variant": "hhi"},
                ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  7. RATIO & INTERACTION FEATURES
    # ────────────────────────────────────────────────────────────────────────

    def _gen_ratio_interaction(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        numeric_cols = schema.numeric_columns[:6]
        entity_col = schema.entity_columns[0] if schema.entity_columns else None

        # Pairwise ratios (avoid division by zero — denominator shifted by ε)
        pairs = list(itertools.combinations(numeric_cols, 2))[:self.config.max_interaction_pairs]
        for col_a, col_b in pairs:
            features.append(_fdef(
                name=f"ratio__{col_a}_div_{col_b}",
                category=FeatureCategory.RATIO_INTERACTION,
                description=f"Ratio {col_a} / ({col_b} + ε).",
                business_intuition=f"Relative scale between {col_a} and {col_b}; captures efficiency or balance.",
                math_def=f"{col_a} / ({col_b} + 1e-9)",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[col_a, col_b],
                tags=["ratio", "interaction"],
                template_params={"col_a": col_a, "col_b": col_b, "variant": "ratio"},
            ))

            features.append(_fdef(
                name=f"product__{col_a}_x_{col_b}",
                category=FeatureCategory.RATIO_INTERACTION,
                description=f"Product {col_a} × {col_b}.",
                business_intuition="Interaction term to capture joint effects invisible to additive models.",
                math_def=f"{col_a} * {col_b}",
                source_columns=[col_a, col_b],
                tags=["product", "interaction", "nonlinear"],
                template_params={"col_a": col_a, "col_b": col_b, "variant": "product"},
            ))

        # Amount vs rolling mean (deviation ratio)
        if entity_col:
            amount_col = numeric_cols[0] if numeric_cols else None
            if amount_col:
                for w in [7, 30]:
                    features.append(_fdef(
                        name=f"{amount_col}__ratio_to_rolling_{w}d_mean",
                        category=FeatureCategory.RATIO_INTERACTION,
                        description=(
                            f"Current {amount_col} divided by trailing {w}d mean — "
                            f"measures how this event compares to recent typical size."
                        ),
                        business_intuition="A 3x spike vs. baseline is far more informative than the raw value.",
                        math_def=f"{amount_col} / (rolling_{w}d_mean({amount_col}) + ε)",
                        predictive_value=PredictiveValue.VERY_HIGH,
                        leakage_risk=LeakageRisk.LOW,
                        source_columns=[entity_col, amount_col],
                        tags=["ratio", "spike_detection", "anomaly"],
                        requires_entity_grouping=True,
                        requires_time_ordering=True,
                        template_params={"entity_col": entity_col, "amount_col": amount_col,
                                         "window_d": w, "variant": "ratio_to_rolling_mean"},
                    ))

        # Category frequency ratio
        for cat_col in schema.categorical_columns[:2]:
            if entity_col:
                features.append(_fdef(
                    name=f"{cat_col}__frequency_ratio_vs_population",
                    category=FeatureCategory.RATIO_INTERACTION,
                    description=(
                        f"Entity's usage rate of the current {cat_col} value vs. "
                        f"population-wide base rate."
                    ),
                    business_intuition="Lift score: does this entity over-index or under-index on this category?",
                    math_def="P(cat | entity) / P(cat | population)",
                    predictive_value=PredictiveValue.HIGH,
                    source_columns=[entity_col, cat_col],
                    tags=["lift", "targeting"],
                    requires_entity_grouping=True,
                    template_params={"entity_col": entity_col, "cat_col": cat_col,
                                     "variant": "frequency_ratio"},
                ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  8. SEQUENCE / TRANSITION FEATURES
    # ────────────────────────────────────────────────────────────────────────

    def _gen_sequence_transition(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        if not schema.entity_columns or not schema.temporal_columns:
            return features

        entity_col = schema.entity_columns[0]
        ts_col = schema.temporal_columns[0]

        for cat_col in schema.categorical_columns[:3]:
            features.append(_fdef(
                name=f"{cat_col}__prev_event_match",
                category=FeatureCategory.SEQUENCE,
                description=f"Flag: the previous event's {cat_col} matches the current event's {cat_col}.",
                business_intuition="Repeated category visits indicate focused intent vs. exploratory behaviour.",
                math_def=f"int({cat_col} == prev({cat_col}), per {entity_col})",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[entity_col, ts_col, cat_col],
                tags=["sequence", "repetition"],
                requires_entity_grouping=True,
                requires_time_ordering=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "cat_col": cat_col, "variant": "prev_match"},
            ))

            features.append(_fdef(
                name=f"{cat_col}__modal_category",
                category=FeatureCategory.SEQUENCE,
                description=f"Most frequently occurring {cat_col} value per {entity_col} over history.",
                business_intuition="Primary domain/category is a stable identity signal.",
                math_def=f"mode({cat_col}) grouped by {entity_col}",
                source_columns=[entity_col, cat_col],
                tags=["mode", "preference"],
                requires_entity_grouping=True,
                template_params={"entity_col": entity_col, "cat_col": cat_col,
                                 "variant": "modal_category"},
            ))

            features.append(_fdef(
                name=f"{cat_col}__is_first_time_category",
                category=FeatureCategory.SEQUENCE,
                description=f"Flag: this is the first time {entity_col} uses this {cat_col} value.",
                business_intuition="First-time category events signal new behaviour or expansion — key in attribution.",
                math_def=f"cumcount grouped by ({entity_col}, {cat_col}) == 0",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[entity_col, ts_col, cat_col],
                tags=["first_time", "novelty"],
                requires_entity_grouping=True,
                requires_time_ordering=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "cat_col": cat_col, "variant": "first_time"},
            ))

            features.append(_fdef(
                name=f"{cat_col}__transition_same_to_different",
                category=FeatureCategory.SEQUENCE,
                description=f"Count of times {entity_col} switched {cat_col} value in last 30 days.",
                business_intuition="Switching frequency is a strong churn/exploration signal.",
                math_def=f"rolling_30d count of (prev_{cat_col} != {cat_col}) per {entity_col}",
                source_columns=[entity_col, ts_col, cat_col],
                tags=["transition", "switching", "churn"],
                requires_entity_grouping=True,
                requires_time_ordering=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "cat_col": cat_col, "variant": "transition_count"},
            ))

        # Event-sequence encoding (last N actions as tuple)
        if schema.categorical_columns:
            cat_col = schema.categorical_columns[0]
            for n in [2, 3]:
                features.append(_fdef(
                    name=f"{cat_col}__last_{n}_gram_hash",
                    category=FeatureCategory.SEQUENCE,
                    description=f"Hash of the last-{n} {cat_col} values sequence per {entity_col}.",
                    business_intuition=f"N-gram sequences capture intent patterns invisible to bag-of-events models.",
                    math_def=f"hash(tuple(prev_{n}_{cat_col})) per {entity_col}",
                    source_columns=[entity_col, ts_col, cat_col],
                    tags=["ngram", "sequence", "hash"],
                    requires_entity_grouping=True,
                    requires_time_ordering=True,
                    template_params={"entity_col": entity_col, "ts_col": ts_col,
                                     "cat_col": cat_col, "n": n, "variant": "ngram_hash"},
                ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  9. GRAPH / NETWORK FEATURES
    # ────────────────────────────────────────────────────────────────────────

    def _gen_graph_network(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        entity_cols = schema.entity_columns
        if len(entity_cols) < 2:
            return features   # Need ≥2 entity types to build a bipartite graph

        src, dst = entity_cols[0], entity_cols[1]

        features.append(_fdef(
            name=f"{src}__out_degree",
            category=FeatureCategory.GRAPH,
            description=f"Number of distinct {dst} nodes linked to each {src}.",
            business_intuition="Hub users/accounts with many connections have distinct risk/LTV profiles.",
            math_def=f"degree in bipartite({src} → {dst})",
            predictive_value=PredictiveValue.HIGH,
            source_columns=[src, dst],
            tags=["graph", "degree", "network"],
            template_params={"src": src, "dst": dst, "variant": "out_degree"},
        ))

        features.append(_fdef(
            name=f"{dst}__in_degree",
            category=FeatureCategory.GRAPH,
            description=f"Number of distinct {src} entities that link to each {dst} (popularity).",
            business_intuition="Highly popular destinations are low-risk / high-engagement nodes.",
            math_def=f"in_degree in bipartite({src} → {dst})",
            source_columns=[src, dst],
            tags=["graph", "popularity", "network"],
            template_params={"src": src, "dst": dst, "variant": "in_degree"},
        ))

        features.append(_fdef(
            name=f"{src}__{dst}__shared_neighbours_count",
            category=FeatureCategory.GRAPH,
            description=f"How many {dst} entities this {src} shares with similar {src} nodes.",
            business_intuition="Shared-neighbour overlap is the basis of collaborative filtering and risk rings.",
            math_def="| N(src) ∩ N(src') | for nearest src' neighbours",
            source_columns=[src, dst],
            tags=["graph", "collaborative", "overlap"],
            template_params={"src": src, "dst": dst, "variant": "shared_neighbours"},
        ))

        features.append(_fdef(
            name=f"{src}__jaccard_similarity_to_cluster",
            category=FeatureCategory.GRAPH,
            description=f"Jaccard similarity of {src}'s {dst} set vs. its graph community centroid.",
            business_intuition="Low similarity to cluster = outlier / emerging pattern; high = conforming.",
            math_def="| N(src) ∩ N_cluster | / | N(src) ∪ N_cluster |",
            source_columns=[src, dst],
            tags=["graph", "jaccard", "community"],
            template_params={"src": src, "dst": dst, "variant": "jaccard_to_cluster"},
        ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  10. SEMANTIC / TEXT FEATURES
    # ────────────────────────────────────────────────────────────────────────

    def _gen_semantic_text(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        features: List[FeatureDefinition] = []
        for text_col in schema.text_columns[:3]:
            features += [
                _fdef(
                    name=f"{text_col}__word_count",
                    category=FeatureCategory.SEMANTIC_TEXT,
                    description=f"Number of whitespace-separated tokens in {text_col}.",
                    business_intuition="Message length correlates with specificity of request or complaint.",
                    math_def=f"{text_col}.str.split().str.len()",
                    source_columns=[text_col],
                    tags=["text", "length"],
                    template_params={"col": text_col, "variant": "word_count"},
                ),
                _fdef(
                    name=f"{text_col}__char_count",
                    category=FeatureCategory.SEMANTIC_TEXT,
                    description=f"Character length of {text_col}.",
                    math_def=f"{text_col}.str.len()",
                    business_intuition="Short texts are often templated; very long texts signal specific issues.",
                    source_columns=[text_col],
                    tags=["text", "length"],
                    template_params={"col": text_col, "variant": "char_count"},
                ),
                _fdef(
                    name=f"{text_col}__sentiment_score",
                    category=FeatureCategory.SEMANTIC_TEXT,
                    description=f"Sentiment polarity score of {text_col} (-1 to +1).",
                    business_intuition="Negative sentiment in feedback/notes is a leading churn/dissatisfaction signal.",
                    math_def="TextBlob/VADER sentiment_polarity(text)",
                    predictive_value=PredictiveValue.HIGH,
                    source_columns=[text_col],
                    tags=["text", "sentiment", "nlp"],
                    template_params={"col": text_col, "variant": "sentiment"},
                ),
                _fdef(
                    name=f"{text_col}__has_url_flag",
                    category=FeatureCategory.SEMANTIC_TEXT,
                    description=f"Binary: 1 if {text_col} contains a URL.",
                    business_intuition="URLs in messages signal phishing, spam, or promotional content.",
                    math_def=r"int(re.search(r'https?://', text) is not None)",
                    predictive_value=PredictiveValue.HIGH,
                    leakage_risk=LeakageRisk.NONE,
                    source_columns=[text_col],
                    tags=["text", "regex", "flag"],
                    template_params={"col": text_col, "variant": "has_url"},
                ),
                _fdef(
                    name=f"{text_col}__has_numeric_flag",
                    category=FeatureCategory.SEMANTIC_TEXT,
                    description=f"Binary: 1 if {text_col} contains numeric digits.",
                    math_def=r"int(re.search(r'\d', text) is not None)",
                    business_intuition="Numeric digits in free-text often indicate amounts, codes, or identifiers.",
                    source_columns=[text_col],
                    tags=["text", "regex", "flag"],
                    template_params={"col": text_col, "variant": "has_numeric"},
                ),
                _fdef(
                    name=f"{text_col}__unique_word_ratio",
                    category=FeatureCategory.SEMANTIC_TEXT,
                    description=f"Ratio of unique words to total words in {text_col} (lexical diversity).",
                    math_def="len(set(tokens)) / max(len(tokens), 1)",
                    business_intuition="Low ratio = repetitive/template text; high ratio = authentic, specific.",
                    source_columns=[text_col],
                    tags=["text", "diversity", "nlp"],
                    template_params={"col": text_col, "variant": "unique_word_ratio"},
                ),
                _fdef(
                    name=f"{text_col}__embedding_cluster_id",
                    category=FeatureCategory.SEMANTIC_TEXT,
                    description=f"Cluster ID from K-Means on {text_col} sentence embeddings.",
                    business_intuition="Semantic clusters discretise topic space; cluster membership is a stable categorical.",
                    math_def="KMeans(n_clusters=20).fit(sentence_transformer.encode(texts))",
                    predictive_value=PredictiveValue.VERY_HIGH,
                    source_columns=[text_col],
                    tags=["text", "embedding", "cluster", "llm"],
                    template_params={"col": text_col, "variant": "embedding_cluster",
                                     "n_clusters": 20},
                ),
            ]
        return features

    # ────────────────────────────────────────────────────────────────────────
    #  11. ADVANCED NON-LINEAR FEATURES
    # ────────────────────────────────────────────────────────────────────────

    def _gen_advanced_nonlinear(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        """Piecewise, spline-like, and complex derived features."""
        features: List[FeatureDefinition] = []
        numeric_cols = schema.numeric_columns[:4]
        entity_col = schema.entity_columns[0] if schema.entity_columns else None

        for col in numeric_cols:
            features.append(_fdef(
                name=f"{col}__sqrt_transform",
                category=FeatureCategory.STATISTICAL,
                description=f"Square-root transform of {col} — variance stabiliser for count data.",
                math_def=f"sqrt({col})",
                business_intuition="Variance-stabilising transform for Poisson-distributed counts.",
                source_columns=[col],
                tags=["transform", "nonlinear"],
                template_params={"col": col, "variant": "sqrt"},
            ))

            features.append(_fdef(
                name=f"{col}__box_cox",
                category=FeatureCategory.STATISTICAL,
                description=f"Box-Cox power transform of {col} (optimal λ estimated from data).",
                math_def="(x^λ - 1) / λ  for λ≠0;  log(x) for λ=0",
                business_intuition="Optimal power transform makes distribution closer to Gaussian — boosts GLMs.",
                source_columns=[col],
                tags=["transform", "boxcox", "normalization"],
                template_params={"col": col, "variant": "boxcox"},
            ))

            features.append(_fdef(
                name=f"{col}__piecewise_low",
                category=FeatureCategory.STATISTICAL,
                description=f"Value of {col} clipped at its 25th percentile (captures low-end effect).",
                math_def=f"min({col}, p25({col}))",
                business_intuition="Piecewise encodings let linear models fit non-linear segments of the distribution.",
                source_columns=[col],
                tags=["piecewise", "nonlinear"],
                template_params={"col": col, "variant": "piecewise_low"},
            ))

            features.append(_fdef(
                name=f"{col}__piecewise_high",
                category=FeatureCategory.STATISTICAL,
                description=f"Value of {col} above its 75th percentile (captures high-end effect).",
                math_def=f"max(0, {col} - p75({col}))",
                business_intuition="Isolates the high-value tail of the distribution — critical for risk models.",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[col],
                tags=["piecewise", "nonlinear", "tail"],
                template_params={"col": col, "variant": "piecewise_high"},
            ))

        # Cross-time features (if entity + temporal)
        if entity_col and schema.temporal_columns:
            ts_col = schema.temporal_columns[0]
            features.append(_fdef(
                name=f"{entity_col}__cohort_week",
                category=FeatureCategory.TEMPORAL,
                description=f"ISO week of {entity_col}'s very first event (cohort assignment).",
                business_intuition="Cohort membership explains systematic vintage effects across acquisition channels.",
                math_def=f"min({ts_col}).dt.isocalendar().week, per {entity_col}",
                predictive_value=PredictiveValue.MEDIUM,
                source_columns=[entity_col, ts_col],
                tags=["cohort", "vintage"],
                requires_entity_grouping=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "variant": "cohort_week"},
            ))

            features.append(_fdef(
                name=f"{entity_col}__weeks_since_cohort",
                category=FeatureCategory.TEMPORAL,
                description=f"Weeks elapsed since {entity_col}'s acquisition cohort week.",
                business_intuition="Lifecycle stage (maturity) is a powerful predictor independent of calendar date.",
                math_def=f"(current_{ts_col} - cohort_{ts_col}).dt.days / 7",
                predictive_value=PredictiveValue.HIGH,
                source_columns=[entity_col, ts_col],
                tags=["cohort", "lifecycle"],
                requires_entity_grouping=True,
                requires_time_ordering=True,
                template_params={"entity_col": entity_col, "ts_col": ts_col,
                                 "variant": "weeks_since_cohort"},
            ))

        return features

    # ────────────────────────────────────────────────────────────────────────
    #  12. LEAKAGE-SAFE TARGET ENCODING PROXIES
    # ────────────────────────────────────────────────────────────────────────

    def _gen_target_encoding_proxies(self, schema: SchemaAnalysis) -> List[FeatureDefinition]:
        """
        These are leave-one-out target-encoded features.  They carry a HIGH
        leakage risk if computed naively; the Construction Agent will emit
        code with proper cross-validation folds.
        """
        features: List[FeatureDefinition] = []
        if not schema.target_column:
            return features

        target = schema.target_column
        for cat_col in schema.categorical_columns[:4]:
            features.append(_fdef(
                name=f"{cat_col}__target_mean_encode_loo",
                category=FeatureCategory.LEAKAGE_SAFE_PROXY,
                description=(
                    f"Leave-one-out mean target ({target}) encoding of {cat_col}. "
                    f"Computed inside cross-validation folds to prevent leakage."
                ),
                business_intuition="Target encoding transforms high-cardinality categories into dense predictive scalars.",
                math_def="E[target | cat=c, i ≠ row] (LOO average)",
                predictive_value=PredictiveValue.VERY_HIGH,
                leakage_risk=LeakageRisk.HIGH,
                source_columns=[cat_col, target],
                tags=["target_encoding", "loo", "careful"],
                requires_target=True,
                template_params={"cat_col": cat_col, "target_col": target,
                                 "variant": "target_mean_loo"},
            ))

            features.append(_fdef(
                name=f"{cat_col}__smoothed_target_encode",
                category=FeatureCategory.LEAKAGE_SAFE_PROXY,
                description=(
                    f"Smoothed target-mean encoding of {cat_col} with global-prior blending. "
                    f"Reduces variance for rare categories."
                ),
                business_intuition="Blending with global mean regularises rare-category estimates.",
                math_def="(n_c · mean_c + k · global_mean) / (n_c + k), k=smoothing_factor",
                predictive_value=PredictiveValue.VERY_HIGH,
                leakage_risk=LeakageRisk.HIGH,
                source_columns=[cat_col, target],
                tags=["target_encoding", "smoothed", "careful"],
                requires_target=True,
                template_params={"cat_col": cat_col, "target_col": target,
                                 "smoothing": 10, "variant": "target_smoothed"},
            ))

        return features
