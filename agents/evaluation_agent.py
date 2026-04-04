"""
Feature Evaluation Agent
─────────────────────────
Evaluates every constructed feature using a battery of statistical and
ML-based methods.

Computed metrics
────────────────
  • IV / WOE                — binary classification signal strength
  • Mutual Information      — non-parametric target dependency
  • Pearson & Spearman r   — linear / monotone correlation with target
  • SHAP (XGBoost)          — attribution-based importance
  • XGB Gain importance     — tree-based importance
  • PSI                     — population stability across time splits
  • Redundancy              — pairwise feature correlation heatmap

Outputs
───────
  • List[FeatureEvaluation] attached to state.feature_evaluations
  • Ranked feature list (composite score)
  • iv_label per feature
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, spearmanr

from agents.base_agent import BaseAgent
from core.config import Config
from core.data_models import (
    FeatureDefinition,
    FeatureEvaluation,
    PipelineState,
    WOEBin,
)

warnings.filterwarnings("ignore", category=UserWarning)


class FeatureEvaluationAgent(BaseAgent):
    """
    Agent 4 — Feature Evaluation

    Requires:
      •  state.constructed_df    (set by FeatureConstructionAgent)
      •  state.target_column     (set by SchemaAgent or user)
      •  state.task_type         ("classification" | "regression")
    """

    name = "FeatureEvaluationAgent"

    # ── Main entry ──────────────────────────────────────────────────────────

    def _run(self, df: pd.DataFrame, state: PipelineState) -> PipelineState:
        feature_df: Optional[pd.DataFrame] = state.__dict__.get("constructed_df")
        if feature_df is None or feature_df.empty:
            raise ValueError("No constructed DataFrame found. Run FeatureConstructionAgent first.")

        target_col = state.target_column
        if target_col is None or target_col not in df.columns:
            self.logger.warning("  No target column available; skipping ML-based evaluations.")
            target = None
        else:
            target = df[target_col].copy()

        feature_cols = [c for c in feature_df.columns]
        self.logger.info("  Evaluating %d features …", len(feature_cols))

        # ── Leakage pre-check ────────────────────────────────────────────────
        leakage_suspects = self._detect_leakage(feature_df, target, df)

        # ── Pairwise correlation (redundancy) ────────────────────────────────
        corr_matrix = self._compute_correlation_matrix(feature_df)

        # ── SHAP + XGB ───────────────────────────────────────────────────────
        shap_scores: Dict[str, float] = {}
        xgb_gain: Dict[str, float] = {}
        if target is not None and len(feature_cols) > 0:
            shap_scores, xgb_gain = self._compute_shap_importance(
                feature_df, target, state.task_type
            )

        # ── Mutual Information ───────────────────────────────────────────────
        mi_scores: Dict[str, float] = {}
        if target is not None:
            mi_scores = self._compute_mutual_information(feature_df, target, state.task_type)

        # ── Correlation with target ──────────────────────────────────────────
        corr_scores: Dict[str, Tuple[float, float]] = {}
        if target is not None:
            corr_scores = self._compute_target_correlations(feature_df, target)

        # ── IV / WOE ─────────────────────────────────────────────────────────
        iv_results: Dict[str, Tuple[float, str, List[WOEBin]]] = {}
        if target is not None and state.task_type == "classification":
            iv_results = self._compute_iv_woe(feature_df, target)

        # ── PSI (time-based stability) ────────────────────────────────────────
        psi_scores: Dict[str, float] = {}
        ts_col = (state.schema_analysis.temporal_columns[0]
                  if state.schema_analysis and state.schema_analysis.temporal_columns
                  else None)
        if ts_col and ts_col in df.columns:
            psi_scores = self._compute_psi(feature_df, df[ts_col])

        # ── Assemble FeatureEvaluation objects ────────────────────────────────
        evaluations: List[FeatureEvaluation] = []
        for feat_def in state.constructed_features:
            name = feat_def.name
            if name not in feature_df.columns:
                continue

            iv_val, iv_label, woe_bins = iv_results.get(name, (None, "unknown", []))
            pearson_r, spearman_r = corr_scores.get(name, (None, None))
            psi = psi_scores.get(name, None)
            stability_score, stability_label = self._psi_to_stability(psi)
            max_corr, redundant_with = self._redundancy_check(
                name, feature_cols, corr_matrix, self.config.correlation_redundancy_threshold
            )

            ev = FeatureEvaluation(
                feature_id=feat_def.feature_id,
                feature_name=name,
                iv_score=iv_val,
                woe_bins=woe_bins,
                iv_label=iv_label or "unknown",
                mutual_information=mi_scores.get(name),
                pearson_corr_with_target=pearson_r,
                spearman_corr_with_target=spearman_r,
                shap_mean_abs=shap_scores.get(name),
                xgb_gain_importance=xgb_gain.get(name),
                psi_score=psi,
                stability_score=stability_score,
                stability_label=stability_label,
                max_correlation_with_others=max_corr,
                redundant_with=redundant_with,
                leakage_detected=name in leakage_suspects,
                leakage_notes="High correlation with target (>0.98)" if name in leakage_suspects else "",
            )

            ev = self._apply_quality_gates(ev)
            ev.composite_score = self._composite_score(ev)
            evaluations.append(ev)

        # Rank by composite score (descending)
        evaluations.sort(key=lambda e: e.composite_score, reverse=True)
        for rank, ev in enumerate(evaluations, 1):
            ev.rank = rank

        state.feature_evaluations = evaluations
        self._log_top_features(evaluations, n=15)
        return state

    # ════════════════════════════════════════════════════════════════════════
    #  Leakage Detection
    # ════════════════════════════════════════════════════════════════════════

    def _detect_leakage(
        self, feature_df: pd.DataFrame, target: Optional[pd.Series], raw_df: pd.DataFrame
    ) -> set:
        suspects = set()
        if target is None:
            return suspects
        numeric_features = feature_df.select_dtypes(include=[np.number])
        for col in numeric_features.columns:
            series = numeric_features[col].fillna(0)
            try:
                corr = abs(pearsonr(series, target.fillna(0))[0])
                if corr > 0.98:
                    suspects.add(col)
            except Exception:
                pass
        return suspects

    # ════════════════════════════════════════════════════════════════════════
    #  SHAP / XGBoost Importance
    # ════════════════════════════════════════════════════════════════════════

    def _compute_shap_importance(
        self, feature_df: pd.DataFrame, target: pd.Series, task_type: str
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        try:
            import xgboost as xgb
            import shap
        except ImportError:
            self.logger.warning("  xgboost/shap not installed — skipping SHAP computation.")
            return {}, {}

        # Prepare data
        X = feature_df.select_dtypes(include=[np.number]).copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = target.copy()

        # Encode target if needed
        if y.dtype == object or str(y.dtype) == "category":
            y = LabelEncoder().fit_transform(y)

        # Cap samples for speed
        n_samples = min(len(X), self.config.shap_max_samples)
        if n_samples < len(X):
            idx = np.random.RandomState(self.config.random_seed).choice(
                len(X), n_samples, replace=False
            )
            X, y = X.iloc[idx], y[idx] if hasattr(y, "iloc") else y[idx]

        if X.empty or len(X) < 50:
            return {}, {}

        try:
            if task_type == "classification":
                model = xgb.XGBClassifier(
                    n_estimators=self.config.shap_n_estimators,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=self.config.random_seed,
                    verbosity=0,
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=self.config.shap_n_estimators,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=self.config.random_seed,
                    verbosity=0,
                )

            model.fit(X, y)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class for binary

            shap_mean = dict(zip(X.columns, np.abs(shap_values).mean(axis=0)))
            gain_map = model.get_booster().get_score(importance_type="gain")
            # Normalise gain
            total = sum(gain_map.values()) + 1e-9
            gain_norm = {k: v / total for k, v in gain_map.items()}

            return shap_mean, gain_norm
        except Exception as exc:
            self.logger.warning("  SHAP computation failed: %s", exc)
            return {}, {}

    # ════════════════════════════════════════════════════════════════════════
    #  Mutual Information
    # ════════════════════════════════════════════════════════════════════════

    def _compute_mutual_information(
        self, feature_df: pd.DataFrame, target: pd.Series, task_type: str
    ) -> Dict[str, float]:
        X = feature_df.select_dtypes(include=[np.number]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
        if X.empty:
            return {}
        y = target.copy()
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
        try:
            func = mutual_info_classif if task_type == "classification" else mutual_info_regression
            mi = func(X, y, random_state=self.config.random_seed)
            return dict(zip(X.columns, mi))
        except Exception as exc:
            self.logger.warning("  MI computation failed: %s", exc)
            return {}

    # ════════════════════════════════════════════════════════════════════════
    #  Target Correlations
    # ════════════════════════════════════════════════════════════════════════

    def _compute_target_correlations(
        self, feature_df: pd.DataFrame, target: pd.Series
    ) -> Dict[str, Tuple[float, float]]:
        results: Dict[str, Tuple[float, float]] = {}
        y = target.copy()
        if y.dtype == object:
            y = LabelEncoder().fit_transform(y.astype(str))
        y = np.array(y, dtype=float)

        for col in feature_df.select_dtypes(include=[np.number]).columns:
            x = feature_df[col].replace([np.inf, -np.inf], np.nan).fillna(0).values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                continue
            try:
                pr = pearsonr(x[mask], y[mask])[0]
                sr = spearmanr(x[mask], y[mask])[0]
                results[col] = (float(pr), float(sr))
            except Exception:
                pass
        return results

    # ════════════════════════════════════════════════════════════════════════
    #  IV / WOE
    # ════════════════════════════════════════════════════════════════════════

    def _compute_iv_woe(
        self, feature_df: pd.DataFrame, target: pd.Series
    ) -> Dict[str, Tuple[float, str, List[WOEBin]]]:
        results: Dict[str, Tuple[float, str, List[WOEBin]]] = {}
        y = target.copy()
        if y.dtype != int and y.dtype != float:
            y = LabelEncoder().fit_transform(y.astype(str))
        y = np.array(y)
        total_events = y.sum()
        total_non_events = len(y) - total_events

        if total_events == 0 or total_non_events == 0:
            return results

        for col in feature_df.select_dtypes(include=[np.number]).columns:
            try:
                x = feature_df[col].replace([np.inf, -np.inf], np.nan)
                # Bin via quantile cut
                try:
                    bins = pd.qcut(x, q=10, duplicates="drop", retbins=False)
                except Exception:
                    bins = pd.cut(x, bins=10, duplicates="drop")

                df_temp = pd.DataFrame({"x_bin": bins, "y": y})
                iv_total = 0.0
                woe_bins: List[WOEBin] = []

                for bin_label, group in df_temp.groupby("x_bin", observed=True):
                    n = len(group)
                    events = group["y"].sum()
                    non_events = n - events
                    rate_events = events / (total_events + 1e-9)
                    rate_non = non_events / (total_non_events + 1e-9)
                    if rate_events <= 0 or rate_non <= 0:
                        continue
                    woe = np.log(rate_events / rate_non)
                    iv_contrib = (rate_events - rate_non) * woe
                    iv_total += iv_contrib
                    woe_bins.append(WOEBin(
                        bin_label=str(bin_label),
                        count=n,
                        event_rate=float(events / max(n, 1)),
                        woe=float(woe),
                        iv_contribution=float(iv_contrib),
                    ))

                iv_label = self._iv_to_label(abs(iv_total))
                results[col] = (float(abs(iv_total)), iv_label, woe_bins)
            except Exception:
                pass
        return results

    def _iv_to_label(self, iv: float) -> str:
        t = self.config.iv_thresholds
        if iv < t["useless"]:
            return "useless"
        if iv < t["weak"]:
            return "weak"
        if iv < t["medium"]:
            return "medium"
        if iv < t["strong"]:
            return "strong"
        return "suspect"

    # ════════════════════════════════════════════════════════════════════════
    #  PSI — Population Stability Index
    # ════════════════════════════════════════════════════════════════════════

    def _compute_psi(
        self, feature_df: pd.DataFrame, ts_col: pd.Series
    ) -> Dict[str, float]:
        psi_results: Dict[str, float] = {}
        try:
            split_point = ts_col.median()
            mask_old = ts_col <= split_point
            mask_new = ts_col > split_point
            if mask_old.sum() < 50 or mask_new.sum() < 50:
                return psi_results
        except Exception:
            return psi_results

        for col in feature_df.select_dtypes(include=[np.number]).columns:
            try:
                old = feature_df.loc[mask_old, col].replace([np.inf, -np.inf], np.nan).dropna()
                new = feature_df.loc[mask_new, col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(old) < 10 or len(new) < 10:
                    continue
                psi_results[col] = float(self._psi_score(old.values, new.values))
            except Exception:
                pass
        return psi_results

    @staticmethod
    def _psi_score(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        breaks = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breaks[0] -= 1e-9
        breaks[-1] += 1e-9
        exp_bins = np.histogram(expected, bins=breaks)[0] / len(expected)
        act_bins = np.histogram(actual, bins=breaks)[0] / len(actual)
        exp_bins = np.where(exp_bins == 0, 1e-4, exp_bins)
        act_bins = np.where(act_bins == 0, 1e-4, act_bins)
        return float(np.sum((act_bins - exp_bins) * np.log(act_bins / exp_bins)))

    def _psi_to_stability(self, psi: Optional[float]) -> Tuple[float, str]:
        if psi is None:
            return 1.0, "unknown"
        if psi < self.config.psi_threshold_monitor:
            return 1.0, "stable"
        if psi < self.config.psi_threshold_unstable:
            return 0.5, "monitor"
        return 0.0, "unstable"

    # ════════════════════════════════════════════════════════════════════════
    #  Correlation Matrix & Redundancy
    # ════════════════════════════════════════════════════════════════════════

    def _compute_correlation_matrix(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        X = feature_df.select_dtypes(include=[np.number]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0)
        if X.shape[1] < 2:
            return pd.DataFrame()
        try:
            return X.corr(method="spearman")
        except Exception:
            return pd.DataFrame()

    def _redundancy_check(
        self,
        col: str,
        all_cols: List[str],
        corr_matrix: pd.DataFrame,
        threshold: float,
    ) -> Tuple[Optional[float], List[str]]:
        if corr_matrix.empty or col not in corr_matrix.columns:
            return None, []
        row = corr_matrix[col].drop(col, errors="ignore").abs()
        redundant = [c for c in row.index if row[c] >= threshold]
        max_corr = float(row.max()) if len(row) > 0 else None
        return max_corr, redundant

    # ════════════════════════════════════════════════════════════════════════
    #  Quality Gates
    # ════════════════════════════════════════════════════════════════════════

    def _apply_quality_gates(self, ev: FeatureEvaluation) -> FeatureEvaluation:
        notes = []
        if ev.iv_label in ("useless",) and ev.mutual_information is not None and ev.mutual_information < 0.001:
            ev.passes_quality_gate = False
            notes.append("IV + MI both below threshold — likely noise.")
        if ev.leakage_detected:
            ev.passes_quality_gate = False
            notes.append("Potential data leakage detected.")
        if ev.stability_label == "unstable":
            notes.append("Feature unstable across time (PSI > threshold).")
        if ev.max_correlation_with_others is not None and ev.max_correlation_with_others >= self.config.correlation_redundancy_threshold:
            notes.append(f"Highly correlated with: {ev.redundant_with[:3]}")
        ev.quality_notes = notes
        return ev

    # ════════════════════════════════════════════════════════════════════════
    #  Composite Score
    # ════════════════════════════════════════════════════════════════════════

    def _composite_score(self, ev: FeatureEvaluation) -> float:
        """
        Weighted composite score combining multiple signals.
        Higher = better.  Range ≈ 0–1.
        """
        score = 0.0
        weights_sum = 0.0

        def _add(val, weight, normalize_fn=None):
            nonlocal score, weights_sum
            if val is not None and np.isfinite(val):
                v = normalize_fn(val) if normalize_fn else val
                score += weight * np.clip(v, 0, 1)
                weights_sum += weight

        # SHAP (0–1 after softmax-like normalisation done later)
        _add(ev.shap_mean_abs, 0.35, lambda x: min(x, 1.0))

        # IV (0–0.5+ → clip at 0.5 for "strong")
        _add(ev.iv_score, 0.25, lambda x: min(x / 0.5, 1.0))

        # MI (0–1 natural range)
        _add(ev.mutual_information, 0.15)

        # |Spearman corr| with target
        _add(abs(ev.spearman_corr_with_target) if ev.spearman_corr_with_target else None, 0.15)

        # Stability bonus
        _add(ev.stability_score, 0.10)

        # Leakage penalty
        if ev.leakage_detected:
            score *= 0.1

        return round(score / max(weights_sum, 1e-9), 4)

    # ════════════════════════════════════════════════════════════════════════
    #  Logging
    # ════════════════════════════════════════════════════════════════════════

    def _log_top_features(self, evaluations: List[FeatureEvaluation], n: int = 15) -> None:
        self.logger.info("  Top %d features by composite score:", n)
        for ev in evaluations[:n]:
            self.logger.info(
                "    #%-3d %-45s score=%.4f  iv=%-8s  shap=%-8s  psi=%s",
                ev.rank, ev.feature_name, ev.composite_score,
                f"{ev.iv_score:.3f}" if ev.iv_score else "n/a",
                f"{ev.shap_mean_abs:.4f}" if ev.shap_mean_abs else "n/a",
                ev.stability_label,
            )
