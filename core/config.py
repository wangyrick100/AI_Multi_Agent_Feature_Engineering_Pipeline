"""
Configuration module for the Feature Engineering System.
Centralises all tunable parameters so agents consume Config rather than
hard-coding values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────────
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    output_dir: Path = ROOT_DIR / "outputs"
    feature_store_path: Path = ROOT_DIR / "outputs" / "feature_store.parquet"
    governance_db_path: Path = ROOT_DIR / "outputs" / "governance.json"
    pipeline_cache_dir: Path = ROOT_DIR / "outputs" / "cache"

    # ── Schema agent ──────────────────────────────────────────────────────────
    sample_rows_for_profiling: int = 10_000
    high_cardinality_threshold: int = 100          # > this → treat as identifier
    temporal_column_keywords: List[str] = field(
        default_factory=lambda: [
            "date", "time", "ts", "timestamp", "created", "updated",
            "at", "on", "dt", "period", "event",
        ]
    )
    id_column_keywords: List[str] = field(
        default_factory=lambda: ["id", "key", "uuid", "code", "ref", "no", "num"]
    )

    # ── Ideation agent ─────────────────────────────────────────────────────────
    rolling_windows_days: List[int] = field(
        default_factory=lambda: [3, 7, 14, 30, 60, 90, 180]
    )
    lag_periods: List[int] = field(
        default_factory=lambda: [1, 2, 3, 7, 14, 30]
    )
    top_n_categorical_values: int = 20   # one-hot threshold for low-card cats
    max_interaction_pairs: int = 30      # cap on ratio/product feature pairs

    # ── Construction agent ─────────────────────────────────────────────────────
    null_fill_strategy: str = "median"   # "median" | "mean" | "zero" | "ffill"
    output_python: bool = True
    output_sql: bool = True
    sql_dialect: str = "bigquery"        # "bigquery" | "snowflake" | "standard"

    # ── Evaluation agent ──────────────────────────────────────────────────────
    iv_thresholds: dict = field(default_factory=lambda: {
        "useless": 0.02, "weak": 0.1, "medium": 0.3, "strong": 0.5
    })
    psi_threshold_monitor: float = 0.1
    psi_threshold_unstable: float = 0.2
    correlation_redundancy_threshold: float = 0.90
    shap_n_estimators: int = 200
    shap_max_samples: int = 5_000        # cap for SHAP speed
    eval_cv_folds: int = 3

    # ── Selection agent ────────────────────────────────────────────────────────
    max_selected_features: int = 50
    min_iv_for_selection: float = 0.02
    min_stability_score: float = 0.7    # 0-1 normalised PSI-based score
    selection_method: str = "pareto"    # "pareto" | "greedy" | "lasso"

    # ── Governance agent ──────────────────────────────────────────────────────
    track_lineage: bool = True
    bias_protected_columns: List[str] = field(
        default_factory=lambda: ["gender", "age", "race", "religion", "ethnicity"]
    )

    # ── Feedback agent ─────────────────────────────────────────────────────────
    max_feedback_iterations: int = 5
    improvement_threshold: float = 0.01  # min AUC gain to continue iterations

    # ── LLM / Semantic features (optional) ────────────────────────────────────
    use_llm_semantic_features: bool = False
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: Path = ROOT_DIR / "outputs" / "pipeline.log"

    # ── Runtime ────────────────────────────────────────────────────────────────
    n_jobs: int = -1   # parallelism (-1 = all cores)
    random_seed: int = 42

    def ensure_dirs(self) -> None:
        """Create output/cache directories if they don't exist."""
        for d in [self.output_dir, self.pipeline_cache_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)
