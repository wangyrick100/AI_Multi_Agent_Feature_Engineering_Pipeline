# 🧠 Multi-Agent Feature Engineering System

A production-grade, autonomous feature engineering pipeline that orchestrates **7 specialized AI agents** to generate, evaluate, and select 100+ high-quality features for any tabular ML dataset.

---

## Architecture

```
raw_df
  │
  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   FeatureEngineeringOrchestrator                      │
│                                                                        │
│  ① SchemaUnderstandingAgent  →  column profiles, semantic types       │
│  ② FeatureIdeationAgent      →  100–200+ feature candidates           │
│  ③ FeatureConstructionAgent  →  pandas Python + BigQuery SQL code     │
│  ④ FeatureEvaluationAgent    →  IV/WOE, SHAP, MI, PSI, correlation    │
│  ⑤ FeatureSelectionAgent     →  Pareto / greedy / LASSO subset        │
│  ⑥ FeatureGovernanceAgent    →  lineage DAG, bias checks, hash        │
│  ⑦ FeedbackOptimizationAgent →  model performance → next iteration   │
│                                                                        │
│          ◄──────── feedback loop (up to N iterations) ──────────►     │
└──────────────────────────────────────────────────────────────────────┘
  │
  ▼
selected_features  +  Python module  +  SQL script  +  JSON report
```

---

## Project Structure

```
code_agent_2_feature_engineering/
├── agents/
│   ├── base_agent.py             # Abstract BaseAgent with timing + logging
│   ├── schema_agent.py           # Schema profiling, semantic type inference
│   ├── ideation_agent.py         # 12-category feature template generator (100–200+)
│   ├── construction_agent.py     # Python + SQL code generation (50+ templates)
│   ├── evaluation_agent.py       # IV/WOE, SHAP, MI, PSI, correlation
│   ├── selection_agent.py        # Pareto / greedy / LASSO selection
│   ├── governance_agent.py       # Lineage DAG, bias, reproducibility hash
│   └── feedback_agent.py         # Feedback loop controller
│
├── core/
│   ├── config.py                 # Config dataclass (all tunable parameters)
│   ├── data_models.py            # Pydantic v2 models (lingua franca between agents)
│   └── orchestrator.py           # Central pipeline controller
│
├── pipelines/
│   ├── python_pipeline.py        # Compiles feature defs → pandas transform class
│   └── sql_pipeline.py           # Assembles BigQuery/Snowflake/ANSI SQL
│
├── utils/
│   ├── leakage_detector.py       # Target correlation, future info, name heuristics
│   ├── drift_detector.py         # PSI, KS test, covariate shift monitoring
│   └── feature_store.py          # Parquet persistence, registry, module export
│
├── ui/
│   ├── app.py                    # Streamlit multi-page application
│   └── components/
│       ├── data_upload.py        # File upload + demo data loader
│       ├── pipeline_view.py      # Live stage progress + metrics
│       ├── feature_explorer.py   # Interactive feature browser (filter/sort/code)
│       └── evaluation_dashboard.py  # Plotly charts: IV, SHAP, PSI, scatter
│
├── data/
│   └── sample_data_generator.py  # Synthetic e-commerce transactions (~100k rows)
│
├── main.py                        # CLI entry point (typer)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo (synthetic data, end-to-end)

```bash
python main.py demo
```

### 3. Run on your own dataset

```bash
python main.py run --data transactions.csv --target is_fraud --task classification
```

Outputs (in `output/`):
- `features_transform.py` — standalone Python class, copy into any project
- `features.sql` — BigQuery-compatible SELECT with all feature expressions
- `feature_store/` — Parquet cache + feature registry JSON

### 4. Launch the Streamlit UI

```bash
python main.py ui
# or directly:
streamlit run ui/app.py
```

---

## The 7 Agents

### ① Schema Understanding Agent
Profiles every column and infers:
- **Semantic type**: `user_id`, `timestamp`, `amount`, `categorical`, `text`, `label`, etc.
- **Distribution**: normal, log-normal, power-law, uniform, bimodal
- **Temporal granularity**: seconds / hours / days / weeks
- **Entity relationships**: 1:N between columns (e.g. user → transaction)

### ② Feature Ideation Agent — *the core engine*
Generates **100–200+ candidates** across 12 template categories:

| Category | Examples |
|----------|---------|
| Temporal calendar | `hour_of_day_sin`, `is_weekend`, `month_cos`, `is_business_hour` |
| Lag & lead | `amount_lag_7d`, `amount_delta_1d`, `inter_event_time_seconds` |
| Rolling window aggs | `amount_sum_30d`, `amount_std_7d`, `event_count_14d` |
| Recency & freshness | `days_since_last_event`, `time_decay_score_30d`, `event_regularity` |
| Behavioral / RFM | `lifetime_value`, `amount_zscore_vs_self`, `active_days_30d` |
| Statistical | `percentile_rank`, `entity_entropy`, `herfindahl_index` |
| Ratio & interaction | `amount_ratio_to_rolling_mean`, `col_a_div_col_b`, `frequency_ratio` |
| Sequence & transition | `modal_category`, `is_first_time_category`, `category_ngram_hash` |
| Graph / network | `out_degree`, `shared_neighbours`, `jaccard_similarity` |
| Semantic text | `word_count`, `sentiment_score`, `has_url_flag` |
| Advanced nonlinear | `sqrt_amount`, `box_cox`, `piecewise_high`, `cohort_week` |
| Target encoding proxies | `loo_target_encode`, `smoothed_target_encode` (leakage-guarded) |

Each feature definition includes:
- **Business intuition** — why this feature should be predictive
- **Mathematical definition** — LaTeX-renderable formula
- **Leakage risk** — `none / low / medium / high`
- **Predictive value prior** — `low / medium / high / very_high`
- **Python code** — executable pandas expression
- **SQL code** — BigQuery window function expression

### ③ Feature Construction Agent
Executes all 50+ code templates against the real DataFrame:
- Anti-leakage rolling windows: `df.rolling("7D", closed="left")`
- GroupBy + temporal aggregations
- SQL: `RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW`

### ④ Feature Evaluation Agent
Computes quality metrics for every feature:

| Metric | Range | Interpretation |
|--------|-------|---------------|
| IV (Information Value) | 0 → ∞ | < 0.02 useless · > 0.3 strong |
| SHAP importance | 0 → ∞ | Mean absolute SHAP contribution |
| Mutual Information | 0 → ∞ | Non-linear association with target |
| Pearson / Spearman r | −1 → 1 | Linear / monotonic correlation |
| PSI | 0 → ∞ | < 0.1 stable · > 0.25 unstable |
| Composite score | 0 → 1 | Weighted: SHAP(35%) + IV(25%) + MI(15%) + corr(15%) + stability(10%) |

### ⑤ Feature Selection Agent
Three selection strategies:
- **Pareto** — non-dominated front on (score × stability)
- **Greedy** — sequential addition, rejecting Spearman r > 0.9
- **LASSO** — L1 regularised logistic/linear regression, keep non-zero coefficients

### ⑥ Governance Agent
For each selected feature:
- Builds a lineage DAG (source columns → derived feature)
- Computes SHA-256 reproducibility hash from code + sources
- Detects PII adjacency (proximity to user_id, email, name columns)
- Measures demographic parity difference for protected attributes
- Exports JSON governance report

### ⑦ Feedback Optimization Agent
After each pipeline pass:
- Identifies weak features (SHAP < 1% of max, IV + MI both below threshold, PSI unstable)
- Proposes improvements: wider windows, polynomial terms, cross-feature interactions
- Controls the feedback loop (stops at convergence or max iterations)

---

## Configuration

All parameters live in `core/config.py`:

```python
@dataclass
class Config:
    rolling_windows_days: List[int] = (3, 7, 14, 30, 60, 90, 180)
    lag_periods: List[int]          = (1, 2, 3, 7, 14, 30)
    shap_max_samples: int           = 5_000
    iv_thresholds: dict             = {"useless": 0.02, "weak": 0.1, "moderate": 0.3}
    psi_threshold_monitor: float    = 0.10
    psi_threshold_unstable: float   = 0.25
    correlation_redundancy_threshold: float = 0.90
    max_selected_features: int      = 50
    selection_method: str           = "pareto"
    max_feedback_iterations: int    = 5
    min_iv_threshold: float         = 0.02
    output_dir: str                 = "output"
```

---

## Streamlit UI Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Architecture diagram, agent overview, quick-start guide |
| 📂 Data Upload | Drag-and-drop CSV/Parquet/Excel or load demo dataset |
| 🚀 Run Pipeline | Configure and execute with live stage-by-stage progress |
| 🧩 Features | Searchable, filterable feature explorer with code viewer |
| 📊 Evaluation | IV bar chart, SHAP importance, PSI treemap, correlation heatmap |
| 🎯 Selection | Selected vs rejected features with rejection reasons |
| 🛡️ Governance | Per-feature lineage, bias plots, reproducibility hashes |
| ⬇️ Export | Download Python module, SQL script, JSON definitions |

---

## Data Model (key Pydantic models)

```python
class FeatureDefinition(BaseModel):
    feature_id: str
    name: str
    category: FeatureCategory
    business_intuition: str
    mathematical_definition: str
    python_code: str
    sql_code: str
    leakage_risk: LeakageRisk
    predictive_value: PredictiveValue
    source_columns: List[str]

class FeatureEvaluation(BaseModel):
    feature_id: str
    iv_score: Optional[float]
    shap_importance: Optional[float]
    mi_score: Optional[float]
    pearson_r: Optional[float]
    spearman_r: Optional[float]
    psi_score: Optional[float]
    stability_label: str          # stable / monitor / unstable
    composite_score: Optional[float]
    rank: Optional[int]

class PipelineState(BaseModel):
    schema_analysis: Optional[SchemaAnalysis]
    feature_candidates: List[FeatureDefinition]
    evaluations: List[FeatureEvaluation]
    selected_features: Optional[FeatureSelectionResult]
    governance_records: List[GovernanceRecord]
    feedback_report: Optional[FeedbackReport]
    iteration: int
```

---

## Security Notes

- Rolling window computations use `closed="left"` to prevent data leakage across time
- Target encoding is flagged as `leakage_risk=HIGH` and requires CV context for production use
- Governance agent hashes all feature code for auditability
- Leakage detector runs correlation thresholding (|r| > 0.98) and name-based heuristics

---

## License

MIT
