"""
Feature Selection Agent
────────────────────────
Selects the optimal subset of features balancing:
  • Predictive power (composite score, IV, SHAP)
  • Stability  (PSI-based stability score)
  • Redundancy elimination (correlation clustering)
  • Interpretability / complexity budget

Selection methods
─────────────────
  pareto  — keep non-dominated features on (score, stability)
  greedy  — sequentially add highest-scoring non-redundant features
  lasso   — L1 regularised logistic/linear regression coefficient > 0
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from core.config import Config
from core.data_models import (
    FeatureEvaluation,
    FeatureSelectionResult,
    PipelineState,
)


class FeatureSelectionAgent(BaseAgent):
    """Agent 5 — Feature Selection."""

    name = "FeatureSelectionAgent"

    def _run(self, df: pd.DataFrame, state: PipelineState) -> PipelineState:
        evaluations = state.feature_evaluations
        if not evaluations:
            raise ValueError("No feature evaluations found. Run FeatureEvaluationAgent first.")

        method = self.config.selection_method
        self.logger.info("  Selecting features using method='%s' …", method)

        if method == "pareto":
            result = self._pareto_selection(evaluations)
        elif method == "lasso":
            result = self._lasso_selection(df, evaluations, state)
        else:
            result = self._greedy_selection(evaluations)

        state.selection_result = result
        self.logger.info(
            "  Selected %d / %d features.",
            len(result.selected_feature_ids),
            len(evaluations),
        )
        for fid in result.selected_feature_ids[:20]:
            ev = next((e for e in evaluations if e.feature_id == fid), None)
            if ev:
                self.logger.info(
                    "    ✔  %-45s score=%.4f", ev.feature_name, ev.composite_score
                )
        return state

    # ── Pareto selection ─────────────────────────────────────────────────────

    def _pareto_selection(self, evaluations: List[FeatureEvaluation]) -> FeatureSelectionResult:
        """
        Keep features that are Pareto-non-dominated on
        (composite_score, stability_score) — then apply quality gates
        and redundancy filter.
        """
        candidates = [e for e in evaluations if e.passes_quality_gate
                      and not e.leakage_detected
                      and (e.iv_score is None or e.iv_score >= self.config.min_iv_for_selection)
                      and e.stability_score >= self.config.min_stability_score]

        # Pareto front on (score, stability)
        pareto_front: List[FeatureEvaluation] = []
        for cand in candidates:
            dominated = False
            for other in candidates:
                if (other.composite_score >= cand.composite_score and
                        other.stability_score >= cand.stability_score and
                        (other.composite_score > cand.composite_score or
                         other.stability_score > cand.stability_score)):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(cand)

        # If too many, take top N by composite score
        pareto_front.sort(key=lambda e: e.composite_score, reverse=True)
        pareto_front = pareto_front[:self.config.max_selected_features]

        # Redundancy elimination (greedy within Pareto front)
        selected = self._remove_redundant(pareto_front)
        rejected_ids = {e.feature_id for e in evaluations} - {e.feature_id for e in selected}
        reasons = self._build_rejection_reasons(evaluations, selected)

        return FeatureSelectionResult(
            selected_feature_ids=[e.feature_id for e in selected],
            rejected_feature_ids=list(rejected_ids),
            rejection_reasons=reasons,
            selection_method="pareto",
            expected_model_complexity=f"{len(selected)} features",
        )

    # ── Greedy selection ─────────────────────────────────────────────────────

    def _greedy_selection(self, evaluations: List[FeatureEvaluation]) -> FeatureSelectionResult:
        """Sequentially add best non-redundant feature until budget exhausted."""
        candidates = sorted(
            [e for e in evaluations if e.passes_quality_gate and not e.leakage_detected],
            key=lambda e: e.composite_score,
            reverse=True,
        )

        selected: List[FeatureEvaluation] = []
        selected_names: Set[str] = set()

        for cand in candidates:
            if len(selected) >= self.config.max_selected_features:
                break
            # Skip if redundant with an already-selected feature
            if any(n in cand.redundant_with for n in selected_names):
                continue
            if cand.stability_score < self.config.min_stability_score:
                continue
            selected.append(cand)
            selected_names.add(cand.feature_name)

        rejected_ids = {e.feature_id for e in evaluations} - {e.feature_id for e in selected}
        rejected_reasons = self._build_rejection_reasons(evaluations, selected)

        return FeatureSelectionResult(
            selected_feature_ids=[e.feature_id for e in selected],
            rejected_feature_ids=list(rejected_ids),
            rejection_reasons=rejected_reasons,
            selection_method="greedy",
            expected_model_complexity=f"{len(selected)} features",
        )

    # ── Lasso / L1 selection ─────────────────────────────────────────────────

    def _lasso_selection(
        self, df: pd.DataFrame, evaluations: List[FeatureEvaluation], state: PipelineState
    ) -> FeatureSelectionResult:
        """Use L1-penalised model coefficients to select features."""
        feature_df: Optional[pd.DataFrame] = state.__dict__.get("constructed_df")
        if feature_df is None or state.target_column is None:
            self.logger.warning("  Lasso selection requires constructed_df + target; falling back to greedy.")
            return self._greedy_selection(evaluations)

        try:
            from sklearn.linear_model import LogisticRegressionCV, LassoCV
            from sklearn.preprocessing import StandardScaler

            X = feature_df.select_dtypes(include=[np.number]).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0)
            y = df[state.target_column].copy()
            if y.dtype == object:
                from sklearn.preprocessing import LabelEncoder
                y = LabelEncoder().fit_transform(y.astype(str))

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if state.task_type == "classification":
                model = LogisticRegressionCV(
                    Cs=10, cv=3, penalty="l1", solver="saga",
                    max_iter=1000, random_state=self.config.random_seed
                )
                model.fit(X_scaled, y)
                coef = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                model = LassoCV(cv=3, max_iter=2000, random_state=self.config.random_seed)
                model.fit(X_scaled, y)
                coef = np.abs(model.coef_)

            selected_names = set(X.columns[coef > 0])
            selected = [e for e in evaluations
                        if e.feature_name in selected_names and not e.leakage_detected][:self.config.max_selected_features]

        except Exception as exc:
            self.logger.warning("  Lasso failed (%s); falling back to greedy.", exc)
            return self._greedy_selection(evaluations)

        rejected_ids = {e.feature_id for e in evaluations} - {e.feature_id for e in selected}
        reasons = self._build_rejection_reasons(evaluations, selected)

        return FeatureSelectionResult(
            selected_feature_ids=[e.feature_id for e in selected],
            rejected_feature_ids=list(rejected_ids),
            rejection_reasons=reasons,
            selection_method="lasso",
            expected_model_complexity=f"{len(selected)} features",
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _remove_redundant(
        self, features: List[FeatureEvaluation]
    ) -> List[FeatureEvaluation]:
        """Iterative: keep highest-scoring within each redundancy cluster."""
        kept: List[FeatureEvaluation] = []
        eliminated: Set[str] = set()
        for feat in sorted(features, key=lambda e: e.composite_score, reverse=True):
            if feat.feature_name in eliminated:
                continue
            kept.append(feat)
            # Eliminate what this feature is redundant with
            for red in feat.redundant_with:
                eliminated.add(red)
        return kept

    def _build_rejection_reasons(
        self, all_evals: List[FeatureEvaluation], selected: List[FeatureEvaluation]
    ) -> Dict[str, str]:
        selected_ids = {e.feature_id for e in selected}
        reasons: Dict[str, str] = {}
        for ev in all_evals:
            if ev.feature_id in selected_ids:
                continue
            if ev.leakage_detected:
                reasons[ev.feature_id] = "Leakage detected"
            elif not ev.passes_quality_gate:
                reasons[ev.feature_id] = "; ".join(ev.quality_notes) or "Failed quality gate"
            elif ev.stability_score < self.config.min_stability_score:
                reasons[ev.feature_id] = f"Unstable (PSI={ev.psi_score:.3f})"
            elif ev.redundant_with:
                reasons[ev.feature_id] = f"Redundant with {ev.redundant_with[0]}"
            else:
                reasons[ev.feature_id] = "Below selection threshold"
        return reasons
