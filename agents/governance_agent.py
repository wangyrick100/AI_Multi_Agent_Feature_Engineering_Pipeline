"""
Feature Governance Agent
─────────────────────────
Tracks lineage, auto-generates documentation, detects bias, and ensures
reproducibility for every selected feature.

Outputs
───────
  • GovernanceRecord per feature — stored in state.governance_records
  • A governance JSON report written to config.governance_db_path
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from core.config import Config
from core.data_models import (
    BiasReport,
    FeatureDefinition,
    GovernanceRecord,
    LineageEdge,
    LineageNode,
    PipelineState,
)


class FeatureGovernanceAgent(BaseAgent):
    """Agent 6 — Feature Governance."""

    name = "FeatureGovernanceAgent"

    def _run(self, df: pd.DataFrame, state: PipelineState) -> PipelineState:
        selected_defs = state.get_selected_definitions()
        if not selected_defs:
            selected_defs = state.constructed_features

        self.logger.info("  Generating governance records for %d features …", len(selected_defs))
        records: List[GovernanceRecord] = []

        for feat_def in selected_defs:
            record = self._build_record(feat_def, df, state)
            records.append(record)

        state.governance_records = records
        self._write_governance_report(records, state)
        self.logger.info("  Governance records written to %s", self.config.governance_db_path)
        return state

    # ── Record builder ───────────────────────────────────────────────────────

    def _build_record(
        self, feat_def: FeatureDefinition, df: pd.DataFrame, state: PipelineState
    ) -> GovernanceRecord:
        # Lineage nodes
        nodes: List[LineageNode] = []
        edges: List[LineageEdge] = []

        # Source column nodes
        for src_col in feat_def.source_columns:
            node_id = f"src__{src_col}"
            nodes.append(LineageNode(
                node_id=node_id,
                node_type="source_column",
                label=src_col,
                metadata={"dtype": str(df[src_col].dtype) if src_col in df.columns else "unknown"},
            ))
            edges.append(LineageEdge(from_id=node_id, to_id=feat_def.feature_id))

        # Feature node
        nodes.append(LineageNode(
            node_id=feat_def.feature_id,
            node_type="feature",
            label=feat_def.name,
            metadata={
                "category": feat_def.category.value,
                "predictive_value": feat_def.expected_predictive_value.value,
                "leakage_risk": feat_def.leakage_risk.value,
            },
        ))

        # Bias checks
        bias_reports = self._check_bias(feat_def, df)

        # Reproducibility hash (hash of python_code + source_columns)
        hash_input = feat_def.python_code + str(sorted(feat_def.source_columns))
        rep_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # Check if adjacent to PII
        pii_indicators = ["name", "email", "phone", "ssn", "dob",
                          "birth", "address", "ip", "device_id"]
        is_pii = any(p in c.lower() for c in feat_def.source_columns for p in pii_indicators)

        return GovernanceRecord(
            feature_id=feat_def.feature_id,
            feature_name=feat_def.name,
            description=feat_def.description,
            business_definitions={
                "business_intuition": feat_def.business_intuition,
                "mathematical_definition": feat_def.mathematical_definition,
                "category": feat_def.category.value,
                "expected_predictive_value": feat_def.expected_predictive_value.value,
                "leakage_risk": feat_def.leakage_risk.value,
            },
            lineage_nodes=nodes,
            lineage_edges=edges,
            bias_reports=bias_reports,
            tags=feat_def.tags,
            is_pii_adjacent=is_pii,
            reproducibility_hash=rep_hash,
        )

    # ── Bias checking ────────────────────────────────────────────────────────

    def _check_bias(
        self, feat_def: FeatureDefinition, df: pd.DataFrame
    ) -> List[BiasReport]:
        reports: List[BiasReport] = []
        if not self.config.track_lineage:
            return reports

        feature_df: Optional[pd.DataFrame] = None
        # Try to get constructed feature value
        constructed_df = getattr(df, "_feature_df", None)
        if constructed_df is None:
            return reports

        feature_col = feat_def.name
        if feature_col not in constructed_df.columns:
            return reports

        for protected_col in self.config.bias_protected_columns:
            if protected_col not in df.columns:
                continue
            try:
                feature_vals = constructed_df[feature_col].fillna(0)
                groups = df[protected_col]

                # Compute group means (demographic parity difference)
                group_means = feature_vals.groupby(groups).mean()
                if len(group_means) < 2:
                    continue
                dp_diff = float(group_means.max() - group_means.min())
                flag = dp_diff > 0.1  # flag large demographic disparities

                reports.append(BiasReport(
                    protected_column=protected_col,
                    feature_name=feature_col,
                    demographic_parity_diff=dp_diff,
                    flag_for_review=flag,
                    notes=(
                        f"Max group mean difference = {dp_diff:.4f}. "
                        f"{'REVIEW REQUIRED' if flag else 'Acceptable.'}"
                    ),
                ))
            except Exception:
                pass
        return reports

    # ── Governance report ────────────────────────────────────────────────────

    def _write_governance_report(
        self, records: List[GovernanceRecord], state: PipelineState
    ) -> None:
        self.config.ensure_dirs()
        report = {
            "run_id": state.run_id,
            "generated_at": datetime.utcnow().isoformat(),
            "n_features": len(records),
            "features": [r.model_dump() for r in records],
        }

        def _json_default(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Not serialisable: {type(obj)}")

        try:
            with open(self.config.governance_db_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=_json_default)
        except Exception as exc:
            self.logger.warning("  Could not write governance report: %s", exc)
