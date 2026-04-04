"""
Feedback & Optimization Agent
───────────────────────────────
Completes the feedback loop: analyses model performance + SHAP explanations,
identifies underperforming features, proposes new feature ideas, and decides
whether another ideation → construction → evaluation cycle is worthwhile.

Inputs
──────
  • state.performance_history  — list of ModelPerformanceSnapshot
  • state.feature_evaluations  — current evaluation results
  • state.selection_result     — which features are in the model

Outputs
───────
  • state.feedback_reports     — cumulative list (one per iteration)
  • Returns continue_loop=True if another iteration is recommended
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from core.config import Config
from core.data_models import (
    FeedbackReport,
    ModelPerformanceSnapshot,
    PipelineState,
)


class FeedbackOptimizationAgent(BaseAgent):
    """Agent 7 — Feedback & Optimization Loop."""

    name = "FeedbackOptimizationAgent"

    def _run(self, df: pd.DataFrame, state: PipelineState) -> PipelineState:
        iteration = state.iteration
        perf_history = state.performance_history

        # ── Baseline check ───────────────────────────────────────────────────
        if len(perf_history) == 0:
            self.logger.info("  No model performance provided; simulating baseline metrics …")
            simulated = self._simulate_baseline_metrics(state)
            state.performance_history.append(simulated)
            perf_history = state.performance_history

        current_perf = perf_history[-1]
        prev_perf = perf_history[-2] if len(perf_history) >= 2 else None

        # ── Compute improvement ──────────────────────────────────────────────
        performance_delta = self._compute_delta(current_perf, prev_perf)

        # ── Identify weak features ────────────────────────────────────────────
        weak_features = self._identify_weak_features(state, current_perf)

        # ── Propose improvements ───────────────────────────────────────────────
        improvements, hints = self._propose_improvements(state, current_perf, weak_features)

        # ── Decide whether to loop ────────────────────────────────────────────
        continue_loop = self._should_continue(
            iteration, performance_delta, len(weak_features), state
        )

        report = FeedbackReport(
            iteration=iteration,
            weak_features=weak_features,
            candidate_improvements=improvements,
            new_ideation_hints=hints,
            recommended_actions=self._build_actions(weak_features, improvements, continue_loop),
            continue_loop=continue_loop,
            performance_delta=performance_delta,
        )

        state.feedback_reports.append(report)
        state.iteration += 1
        self._log_feedback(report, current_perf)
        return state

    # ── Baseline simulator ────────────────────────────────────────────────────

    def _simulate_baseline_metrics(self, state: PipelineState) -> ModelPerformanceSnapshot:
        """
        When no real model metrics are supplied, estimate them from the
        evaluation scores to allow the loop to reason about feature quality.
        """
        evaluations = state.feature_evaluations
        if not evaluations:
            return ModelPerformanceSnapshot(
                iteration=state.iteration, auc_roc=0.5
            )
        avg_score = np.mean([e.composite_score for e in evaluations])
        estimated_auc = 0.5 + 0.45 * avg_score   # rough linear approximation
        importance = {e.feature_name: e.composite_score for e in evaluations[:20]}
        return ModelPerformanceSnapshot(
            iteration=state.iteration,
            auc_roc=float(np.clip(estimated_auc, 0.5, 0.99)),
            feature_importances=importance,
            shap_values_summary=importance,
        )

    # ── Weak feature identification ───────────────────────────────────────────

    def _identify_weak_features(
        self, state: PipelineState, perf: ModelPerformanceSnapshot
    ) -> List[str]:
        """
        A feature is 'weak' if:
          * SHAP importance < 1% of max SHAP, OR
          * Both IV < 0.02 AND MI < 0.001, OR
          * PSI unstable (stability_label == "unstable")
        """
        weak: List[str] = []
        evaluations = state.feature_evaluations
        if not evaluations:
            return weak

        max_shap = max(
            (e.shap_mean_abs or 0.0) for e in evaluations
        ) + 1e-9

        for ev in evaluations:
            reasons = []
            shap = ev.shap_mean_abs or 0.0
            if shap < 0.01 * max_shap:
                reasons.append("low_shap")
            if (ev.iv_score is not None and ev.iv_score < 0.02 and
                    ev.mutual_information is not None and ev.mutual_information < 0.001):
                reasons.append("low_iv_mi")
            if ev.stability_label == "unstable":
                reasons.append("unstable")
            if reasons:
                weak.append(ev.feature_name)

        if len(weak) > 5:
            # Sort by composite score ascending; return bottom 10
            scored_weak = sorted(
                [e for e in evaluations if e.feature_name in weak],
                key=lambda e: e.composite_score,
            )
            weak = [e.feature_name for e in scored_weak[:10]]

        return weak

    # ── Improvement proposals ─────────────────────────────────────────────────

    def _propose_improvements(
        self,
        state: PipelineState,
        perf: ModelPerformanceSnapshot,
        weak_features: List[str],
    ):
        improvements: List[str] = []
        hints: Dict[str, Any] = {}

        evaluations = state.feature_evaluations
        top_features = sorted(evaluations, key=lambda e: e.composite_score, reverse=True)[:5]

        # 1. Suggest polynomial terms of top features
        for ev in top_features[:3]:
            improvements.append(
                f"Polynomial degree-2 term: {ev.feature_name}^2 "
                f"(top SHAP={ev.shap_mean_abs:.4f})"
            )

        # 2. Suggest longer rolling windows if short windows are top features
        for ev in top_features:
            for tag in ev.feature_name.split("__"):
                if "rolling" in tag and "7d" in ev.feature_name and "7d" not in str(hints):
                    improvements.append(
                        f"Extend rolling window: 7d → 90d or 180d for features like {ev.feature_name}"
                    )
                    hints["extend_windows"] = True

        # 3. Cross-feature interactions among top features
        if len(top_features) >= 2:
            for i in range(min(3, len(top_features))):
                for j in range(i + 1, min(4, len(top_features))):
                    improvements.append(
                        f"Interaction: {top_features[i].feature_name} × {top_features[j].feature_name}"
                    )

        # 4. Flag weak feature types for replacement
        if any("rolling_7d" in w for w in weak_features):
            improvements.append("Weak short-window rolling features — try 14d or 21d windows instead.")
            hints["try_14d_21d_windows"] = True

        if any("lag_1" in w for w in weak_features):
            improvements.append("Lag-1 is weak — try lag-3 or lag-7 which may capture slower dynamics.")
            hints["try_longer_lags"] = True

        # 5. Bias-correction hints if equity issues found
        bias_features = [
            r.feature_name for rec in state.governance_records
            for r in rec.bias_reports if r.flag_for_review
        ]
        if bias_features:
            improvements.append(
                f"Bias detected in {bias_features[:3]}. "
                f"Consider fairness-aware encoding or exclusion."
            )
            hints["bias_review"] = bias_features[:3]

        return improvements[:15], hints

    # ── Loop control ──────────────────────────────────────────────────────────

    def _compute_delta(
        self,
        current: ModelPerformanceSnapshot,
        prev: Optional[ModelPerformanceSnapshot],
    ) -> Optional[float]:
        if prev is None:
            return None
        if current.auc_roc is not None and prev.auc_roc is not None:
            return current.auc_roc - prev.auc_roc
        if current.rmse is not None and prev.rmse is not None:
            return prev.rmse - current.rmse   # improvement = decrease in RMSE
        return None

    def _should_continue(
        self,
        iteration: int,
        delta: Optional[float],
        n_weak: int,
        state: PipelineState,
    ) -> bool:
        if iteration >= self.config.max_feedback_iterations:
            self.logger.info(
                "  Max iterations (%d) reached — stopping loop.",
                self.config.max_feedback_iterations,
            )
            return False
        if delta is not None and delta < self.config.improvement_threshold and iteration > 0:
            self.logger.info(
                "  Performance delta %.4f < threshold %.4f — stopping loop.",
                delta, self.config.improvement_threshold,
            )
            return False
        if n_weak == 0:
            self.logger.info("  No weak features detected — stopping loop.")
            return False
        return True

    # ── Actions list ──────────────────────────────────────────────────────────

    def _build_actions(
        self, weak: List[str], improvements: List[str], continue_loop: bool
    ) -> List[str]:
        actions = []
        if weak:
            actions.append(f"DROP or REPLACE: {weak[:5]}")
        if improvements:
            actions.append(f"GENERATE NEW: {improvements[:3]}")
        if continue_loop:
            actions.append("TRIGGER next ideation → construction → evaluation cycle.")
        else:
            actions.append("FINALISE feature set — no further iterations recommended.")
        return actions

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_feedback(self, report: FeedbackReport, perf: ModelPerformanceSnapshot) -> None:
        self.logger.info(
            "  Iteration %d | AUC=%.4f | Δ=%s | Weak=%d | Continue=%s",
            report.iteration,
            perf.auc_roc or 0.0,
            f"{report.performance_delta:+.4f}" if report.performance_delta is not None else "n/a",
            len(report.weak_features),
            report.continue_loop,
        )
        if report.candidate_improvements:
            self.logger.info("  Proposed improvements:")
            for imp in report.candidate_improvements[:5]:
                self.logger.info("    • %s", imp)
