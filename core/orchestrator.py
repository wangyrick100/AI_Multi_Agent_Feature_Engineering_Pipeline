"""
Central Orchestrator
─────────────────────
Manages the multi-agent pipeline as a directed graph with:
  • Sequential stage execution (Schema → Ideation → Construction → Evaluation
    → Selection → Governance → Feedback)
  • Configurable feedback loop (max N iterations)
  • Progress callbacks for UI integration
  • Graceful error recovery (agent failure → log + continue)

Usage
─────
    from core.orchestrator import FeatureEngineeringOrchestrator
    from core.config import Config

    orch = FeatureEngineeringOrchestrator(Config())
    state = orch.run(df, target_column="is_fraud")
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.config import Config
from core.data_models import PipelineState
from agents.schema_agent import SchemaUnderstandingAgent
from agents.ideation_agent import FeatureIdeationAgent
from agents.construction_agent import FeatureConstructionAgent
from agents.evaluation_agent import FeatureEvaluationAgent
from agents.selection_agent import FeatureSelectionAgent
from agents.governance_agent import FeatureGovernanceAgent
from agents.feedback_agent import FeedbackOptimizationAgent


ProgressCallback = Callable[[str, int, int], None]   # (stage_name, current, total)


class FeatureEngineeringOrchestrator:
    """
    Central controller for the Feature Engineering multi-agent system.

    Parameters
    ----------
    config
        System-wide configuration object.
    progress_callback
        Optional callable invoked after each agent completes.
        Signature: callback(stage_name: str, step: int, total_steps: int)
    """

    PIPELINE_STAGES = [
        "SchemaUnderstanding",
        "FeatureIdeation",
        "FeatureConstruction",
        "FeatureEvaluation",
        "FeatureSelection",
        "FeatureGovernance",
        "FeedbackOptimization",
    ]

    def __init__(
        self,
        config: Optional[Config] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config or Config()
        self.config.ensure_dirs()
        self._setup_logging()
        self.logger = logging.getLogger("Orchestrator")
        self.progress_callback = progress_callback

        # Instantiate all agents
        self.schema_agent      = SchemaUnderstandingAgent(self.config)
        self.ideation_agent    = FeatureIdeationAgent(self.config)
        self.construction_agent= FeatureConstructionAgent(self.config)
        self.evaluation_agent  = FeatureEvaluationAgent(self.config)
        self.selection_agent   = FeatureSelectionAgent(self.config)
        self.governance_agent  = FeatureGovernanceAgent(self.config)
        self.feedback_agent    = FeedbackOptimizationAgent(self.config)

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: str = "classification",
        domain_hints: Optional[Dict[str, Any]] = None,
        run_feedback_loop: bool = True,
    ) -> PipelineState:
        """
        Execute the complete pipeline and return the final PipelineState.

        Parameters
        ----------
        df              Raw data DataFrame.
        target_column   Column name for the supervised target.
        task_type       "classification" | "regression".
        domain_hints    Optional dict with domain knowledge hints.
        run_feedback_loop  Whether to execute the feedback iteration loop.
        """
        self.logger.info("=" * 70)
        self.logger.info("  Feature Engineering System — Pipeline Starting")
        self.logger.info("  Dataset: %d rows × %d cols", len(df), len(df.columns))
        self.logger.info("=" * 70)

        state = PipelineState(
            dataset_id=getattr(df, "name", "dataset"),
            target_column=target_column,
            task_type=task_type,
            domain_hints=domain_hints or {},
        )

        total_stages = len(self.PIPELINE_STAGES)
        t_start = time.perf_counter()

        # ── Stage 1: Schema Understanding ────────────────────────────────────
        state = self._execute_stage(
            self.schema_agent, df, state, "SchemaUnderstanding", 1, total_stages
        )
        if state.status.startswith("error"):
            return state

        # ── Stage 2: Feature Ideation ─────────────────────────────────────────
        state = self._execute_stage(
            self.ideation_agent, df, state, "FeatureIdeation", 2, total_stages
        )
        if state.status.startswith("error"):
            return state

        # ── Stage 3: Feature Construction ─────────────────────────────────────
        state = self._execute_stage(
            self.construction_agent, df, state, "FeatureConstruction", 3, total_stages
        )
        if state.status.startswith("error"):
            return state

        # ── Stage 4: Feature Evaluation ───────────────────────────────────────
        state = self._execute_stage(
            self.evaluation_agent, df, state, "FeatureEvaluation", 4, total_stages
        )
        # Evaluation failure is non-fatal

        # ── Stage 5: Feature Selection ────────────────────────────────────────
        state = self._execute_stage(
            self.selection_agent, df, state, "FeatureSelection", 5, total_stages
        )

        # ── Stage 6: Governance ───────────────────────────────────────────────
        state = self._execute_stage(
            self.governance_agent, df, state, "FeatureGovernance", 6, total_stages
        )

        # ── Stage 7: Feedback Loop ────────────────────────────────────────────
        if run_feedback_loop:
            state = self._run_feedback_loop(df, state, total_stages)

        elapsed = time.perf_counter() - t_start
        state.status = "completed"
        self.logger.info("=" * 70)
        self.logger.info("  Pipeline complete in %.1fs", elapsed)
        self.logger.info(
            "  Features: %d candidates → %d constructed → %d selected",
            len(state.feature_candidates),
            len(state.constructed_features),
            len(state.selection_result.selected_feature_ids)
            if state.selection_result else 0,
        )
        self.logger.info("=" * 70)

        # Persist pipeline timings
        self._save_summary(state)
        return state

    def run_single_stage(
        self, stage_name: str, df: pd.DataFrame, state: PipelineState
    ) -> PipelineState:
        """Run a single named stage against an existing state (for UI step-by-step mode)."""
        agent_map = {
            "SchemaUnderstanding": self.schema_agent,
            "FeatureIdeation":     self.ideation_agent,
            "FeatureConstruction": self.construction_agent,
            "FeatureEvaluation":   self.evaluation_agent,
            "FeatureSelection":    self.selection_agent,
            "FeatureGovernance":   self.governance_agent,
            "FeedbackOptimization":self.feedback_agent,
        }
        agent = agent_map.get(stage_name)
        if agent is None:
            raise ValueError(f"Unknown stage: {stage_name}")
        return agent.run(df, state)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _execute_stage(
        self,
        agent,
        df: pd.DataFrame,
        state: PipelineState,
        name: str,
        step: int,
        total: int,
    ) -> PipelineState:
        self.logger.info("── Stage %d/%d: %s ──", step, total, name)
        state = agent.run(df, state)
        if self.progress_callback:
            try:
                self.progress_callback(name, step, total)
            except Exception:
                pass
        return state

    def _run_feedback_loop(
        self, df: pd.DataFrame, state: PipelineState, total_stages: int
    ) -> PipelineState:
        max_iters = self.config.max_feedback_iterations
        self.logger.info("── Feedback Loop (max %d iterations) ──", max_iters)

        for i in range(max_iters):
            state = self._execute_stage(
                self.feedback_agent, df, state,
                f"FeedbackOptimization[{i+1}]", 7, total_stages
            )
            if not state.feedback_reports:
                break
            report = state.feedback_reports[-1]
            if not report.continue_loop:
                self.logger.info("  Feedback loop terminated at iteration %d.", i + 1)
                break

            # Re-ideate with new hints if loop continues
            if report.new_ideation_hints and i < max_iters - 1:
                self.logger.info("  Re-running ideation with new hints …")
                # Merge hints into domain_hints
                state.domain_hints.update(report.new_ideation_hints)
                # Clear previous candidates so we get new ones
                old_candidates = state.feature_candidates
                state = self._execute_stage(
                    self.ideation_agent, df, state,
                    f"FeatureIdeation[re-{i+1}]", 2, total_stages
                )
                # Merge: keep old + new unique
                old_names = {f.name for f in old_candidates}
                new_only = [f for f in state.feature_candidates
                            if f.name not in old_names]
                if new_only:
                    state.feature_candidates = old_candidates + new_only
                    self.logger.info("  Added %d new feature candidates.", len(new_only))
                    # Re-construct only the new features
                    new_state_tmp = state.model_copy()
                    new_state_tmp.feature_candidates = new_only
                    new_state_tmp = self.construction_agent.run(df, new_state_tmp)
                    new_constructed = new_state_tmp.constructed_features
                    state.constructed_features.extend(new_constructed)
                    # Merge new feature columns into constructed_df
                    new_df = new_state_tmp.__dict__.get("constructed_df")
                    old_cdf = state.__dict__.get("constructed_df", pd.DataFrame())
                    if new_df is not None and not new_df.empty:
                        merged = pd.concat([old_cdf, new_df], axis=1)
                        state.__dict__["constructed_df"] = merged

                    # Re-evaluate
                    state = self._execute_stage(
                        self.evaluation_agent, df, state,
                        f"FeatureEvaluation[re-{i+1}]", 4, total_stages
                    )
                    state = self._execute_stage(
                        self.selection_agent, df, state,
                        f"FeatureSelection[re-{i+1}]", 5, total_stages
                    )
        return state

    # ── Logging & persistence ────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        log_level = getattr(logging, self.config.log_level, logging.INFO)
        handlers: List[logging.Handler] = [logging.StreamHandler()]
        if self.config.log_to_file:
            self.config.ensure_dirs()
            handlers.append(logging.FileHandler(self.config.log_file, encoding="utf-8"))
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s  %(name)-30s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
            force=True,
        )

    def _save_summary(self, state: PipelineState) -> None:
        import json
        from datetime import datetime

        summary = {
            "run_id": state.run_id,
            "completed_at": datetime.utcnow().isoformat(),
            "n_candidates": len(state.feature_candidates),
            "n_constructed": len(state.constructed_features),
            "n_selected": (len(state.selection_result.selected_feature_ids)
                           if state.selection_result else 0),
            "iterations": state.iteration,
            "agent_timings": state.agent_timings,
            "errors": state.errors,
            "warnings": state.warnings[:10],
        }
        path = self.config.output_dir / f"run_summary_{state.run_id}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass
