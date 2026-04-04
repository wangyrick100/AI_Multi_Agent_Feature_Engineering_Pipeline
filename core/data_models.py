"""
Pydantic data-models that flow through the multi-agent pipeline.

Every agent reads from and writes to a PipelineState object, ensuring
a single, type-safe source of truth throughout the system.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────────────
#  Enumerations
# ────────────────────────────────────────────────────────────────────────────

class SemanticType(str, Enum):
    USER_ID       = "user_id"
    ITEM_ID       = "item_id"
    SESSION_ID    = "session_id"
    TRANSACTION_ID= "transaction_id"
    TIMESTAMP     = "timestamp"
    DATE          = "date"
    AMOUNT        = "amount"
    CATEGORICAL   = "categorical"
    BINARY_FLAG   = "binary_flag"
    CONTINUOUS    = "continuous"
    TEXT          = "text"
    IDENTIFIER    = "identifier"
    GEOGRAPHIC    = "geographic"
    LABEL         = "label"
    UNKNOWN       = "unknown"


class FeatureCategory(str, Enum):
    TEMPORAL           = "temporal"
    BEHAVIORAL         = "behavioral"
    RFM                = "rfm"
    STATISTICAL        = "statistical"
    RATIO_INTERACTION  = "ratio_interaction"
    SEQUENCE           = "sequence"
    GRAPH              = "graph"
    SEMANTIC_TEXT      = "semantic_text"
    IDENTITY_ENCODING  = "identity_encoding"
    LEAKAGE_SAFE_PROXY = "leakage_safe_proxy"
    CUSTOM             = "custom"


class LeakageRisk(str, Enum):
    NONE   = "none"
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class PredictiveValue(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    VERY_HIGH = "very_high"


class DistributionType(str, Enum):
    NORMAL     = "normal"
    LOG_NORMAL = "log_normal"
    EXPONENTIAL= "exponential"
    BIMODAL    = "bimodal"
    UNIFORM    = "uniform"
    SKEWED     = "skewed"
    HEAVY_TAIL = "heavy_tail"
    UNKNOWN    = "unknown"


# ────────────────────────────────────────────────────────────────────────────
#  Schema Analysis
# ────────────────────────────────────────────────────────────────────────────

class ColumnProfile(BaseModel):
    """Statistical and semantic profile of a single DataFrame column."""
    name: str
    raw_dtype: str
    semantic_type: SemanticType = SemanticType.UNKNOWN
    missing_rate: float = 0.0
    unique_count: int = 0
    cardinality_ratio: float = 0.0   # unique / total
    distribution: DistributionType = DistributionType.UNKNOWN
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    top_values: List[Any] = Field(default_factory=list)
    sample_values: List[Any] = Field(default_factory=list)
    is_monotonic: bool = False
    temporal_granularity: Optional[str] = None   # "second","minute","hour","day"…
    description: str = ""
    entity_role: Optional[str] = None    # "entity_key", "event_time", "target"…


class EntityRelationship(BaseModel):
    """Detected relationship between two columns."""
    from_col: str
    to_col: str
    relationship_type: str   # "one_to_many", "many_to_one", "many_to_many"
    confidence: float = 1.0


class SchemaAnalysis(BaseModel):
    """Full output of the Schema Understanding Agent."""
    dataset_id: str
    n_rows: int
    n_cols: int
    column_profiles: List[ColumnProfile] = Field(default_factory=list)
    entity_columns: List[str] = Field(default_factory=list)
    temporal_columns: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    text_columns: List[str] = Field(default_factory=list)
    relationships: List[EntityRelationship] = Field(default_factory=list)
    domain_hints: Dict[str, Any] = Field(default_factory=dict)
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)

    def get_profile(self, col: str) -> Optional[ColumnProfile]:
        for p in self.column_profiles:
            if p.name == col:
                return p
        return None


# ────────────────────────────────────────────────────────────────────────────
#  Feature Definitions (Ideation → Construction)
# ────────────────────────────────────────────────────────────────────────────

class FeatureDefinition(BaseModel):
    """A single, fully-described derived feature candidate."""
    feature_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    display_name: str = ""
    category: FeatureCategory
    tags: List[str] = Field(default_factory=list)

    # Human-readable documentation
    description: str
    business_intuition: str
    mathematical_definition: str

    # Quality metadata
    expected_predictive_value: PredictiveValue = PredictiveValue.MEDIUM
    leakage_risk: LeakageRisk = LeakageRisk.NONE
    requires_entity_grouping: bool = False
    requires_time_ordering: bool = False
    requires_target: bool = False    # True → evaluation-only, not training

    # Lineage
    source_columns: List[str] = Field(default_factory=list)
    derived_from: List[str] = Field(default_factory=list)  # feature_ids

    # Generated code (filled by Construction Agent)
    python_code: str = ""
    sql_code: str = ""
    python_incremental_code: str = ""

    # Parameters used to instantiate this feature from a template
    template_params: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    def __hash__(self):
        return hash(self.feature_id)

    def __eq__(self, other):
        if isinstance(other, FeatureDefinition):
            return self.feature_id == other.feature_id
        return False


# ────────────────────────────────────────────────────────────────────────────
#  Feature Evaluation
# ────────────────────────────────────────────────────────────────────────────

class WOEBin(BaseModel):
    """A single WOE/IV bin."""
    bin_label: str
    count: int
    event_rate: float
    woe: float
    iv_contribution: float


class FeatureEvaluation(BaseModel):
    """Statistical and ML-based quality assessment of a constructed feature."""
    feature_id: str
    feature_name: str

    # Information theoretic
    iv_score: Optional[float] = None          # Information Value
    woe_bins: List[WOEBin] = Field(default_factory=list)
    mutual_information: Optional[float] = None

    # Correlation
    pearson_corr_with_target: Optional[float] = None
    spearman_corr_with_target: Optional[float] = None

    # ML importance
    shap_mean_abs: Optional[float] = None
    xgb_gain_importance: Optional[float] = None
    permutation_importance: Optional[float] = None

    # Stability
    psi_score: Optional[float] = None         # Population Stability Index
    stability_label: str = "unknown"          # stable / monitor / unstable
    stability_score: float = 1.0              # 1 = perfectly stable

    # Redundancy
    max_correlation_with_others: Optional[float] = None
    redundant_with: List[str] = Field(default_factory=list)

    # Leakage
    leakage_detected: bool = False
    leakage_notes: str = ""

    # Quality gates
    iv_label: str = "unknown"   # useless / weak / medium / strong / suspect
    passes_quality_gate: bool = True
    quality_notes: List[str] = Field(default_factory=list)

    # Overall rank (lower = better)
    composite_score: float = 0.0
    rank: int = 9999

    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


# ────────────────────────────────────────────────────────────────────────────
#  Feature Selection
# ────────────────────────────────────────────────────────────────────────────

class FeatureSelectionResult(BaseModel):
    """Output of the Feature Selection Agent."""
    selected_feature_ids: List[str] = Field(default_factory=list)
    rejected_feature_ids: List[str] = Field(default_factory=list)
    rejection_reasons: Dict[str, str] = Field(default_factory=dict)
    selection_method: str = ""
    expected_model_complexity: str = ""
    selection_timestamp: datetime = Field(default_factory=datetime.utcnow)


# ────────────────────────────────────────────────────────────────────────────
#  Governance
# ────────────────────────────────────────────────────────────────────────────

class LineageNode(BaseModel):
    node_id: str
    node_type: str   # "source_column" | "feature" | "model"
    label: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LineageEdge(BaseModel):
    from_id: str
    to_id: str
    edge_type: str = "derived_from"


class BiasReport(BaseModel):
    protected_column: str
    feature_name: str
    demographic_parity_diff: float
    flag_for_review: bool
    notes: str = ""


class GovernanceRecord(BaseModel):
    """Lineage, documentation, and compliance record for a feature."""
    feature_id: str
    feature_name: str
    version: str = "1.0.0"
    owner: str = "feature-engineering-system"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    description: str = ""
    business_definitions: Dict[str, str] = Field(default_factory=dict)
    lineage_nodes: List[LineageNode] = Field(default_factory=list)
    lineage_edges: List[LineageEdge] = Field(default_factory=list)
    bias_reports: List[BiasReport] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    is_pii_adjacent: bool = False
    reproducibility_hash: str = ""
    sign_off: Optional[str] = None


# ────────────────────────────────────────────────────────────────────────────
#  Feedback Loop
# ────────────────────────────────────────────────────────────────────────────

class ModelPerformanceSnapshot(BaseModel):
    """Model metrics snapshot passed into the Feedback Agent."""
    iteration: int
    auc_roc: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    log_loss: Optional[float] = None
    rmse: Optional[float] = None
    feature_importances: Dict[str, float] = Field(default_factory=dict)
    shap_values_summary: Dict[str, float] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


class FeedbackReport(BaseModel):
    """What the Feedback Agent recommends for the next iteration."""
    iteration: int
    weak_features: List[str] = Field(default_factory=list)       # to drop
    candidate_improvements: List[str] = Field(default_factory=list)  # descriptions
    new_ideation_hints: Dict[str, Any] = Field(default_factory=dict)
    recommended_actions: List[str] = Field(default_factory=list)
    continue_loop: bool = True
    performance_delta: Optional[float] = None  # AUC gain vs previous iteration


# ────────────────────────────────────────────────────────────────────────────
#  Master Pipeline State
# ────────────────────────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """
    The single source of truth passed between agents during a pipeline run.
    Each agent receives this object, enriches it, and returns it.
    """
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    dataset_id: str = "dataset"
    iteration: int = 0

    # Agent outputs accumulated across stages
    schema_analysis: Optional[SchemaAnalysis] = None
    feature_candidates: List[FeatureDefinition] = Field(default_factory=list)
    constructed_features: List[FeatureDefinition] = Field(default_factory=list)
    feature_evaluations: List[FeatureEvaluation] = Field(default_factory=list)
    selection_result: Optional[FeatureSelectionResult] = None
    governance_records: List[GovernanceRecord] = Field(default_factory=list)
    feedback_reports: List[FeedbackReport] = Field(default_factory=list)
    performance_history: List[ModelPerformanceSnapshot] = Field(default_factory=list)

    # Pipeline control
    target_column: Optional[str] = None
    task_type: str = "classification"   # "classification" | "regression"
    domain_hints: Dict[str, Any] = Field(default_factory=dict)

    # Status flags
    status: str = "initialized"
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    agent_timings: Dict[str, float] = Field(default_factory=dict)

    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True

    def touch(self, agent_name: str = "") -> None:
        self.updated_at = datetime.utcnow()
        if agent_name:
            self.status = f"completed:{agent_name}"

    def get_selected_definitions(self) -> List[FeatureDefinition]:
        if not self.selection_result:
            return self.constructed_features
        ids = set(self.selection_result.selected_feature_ids)
        return [f for f in self.constructed_features if f.feature_id in ids]

    def get_evaluation(self, feature_id: str) -> Optional[FeatureEvaluation]:
        for ev in self.feature_evaluations:
            if ev.feature_id == feature_id:
                return ev
        return None
